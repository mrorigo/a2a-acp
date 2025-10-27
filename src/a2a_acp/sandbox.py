"""
Sandboxing Framework for Tool Execution

Provides secure, user-configured execution environments for bash-based tools.
Handles working directory management, environment variable injection, and execution isolation.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import uuid
import re
import sys
import resource
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, AsyncGenerator
from datetime import datetime

from .tool_config import ToolConfig, BashTool

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of a tool execution."""
    success: bool
    return_code: int
    stdout: str
    stderr: str
    execution_time: float
    output_files: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.output_files is None:
            self.output_files = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ExecutionContext:
    """Context information for tool execution."""
    tool_id: str
    session_id: str
    task_id: str
    user_id: str = "anonymous"
    timestamp: Optional[datetime] = None
    environment: Optional[Dict[str, str]] = None
    working_directory: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.environment is None:
            self.environment = {}
        if self.metadata is None:
            self.metadata = {}


class SandboxError(Exception):
    """Raised when sandbox operations fail."""
    pass


class SandboxSecurityError(SandboxError):
    """Raised when security violations are detected."""
    pass


class ToolSandbox:
    """Manages secure execution environments for bash tools."""

    def __init__(self, base_temp_dir: Optional[str] = None):
        """Initialize the sandbox.

        Args:
            base_temp_dir: Base directory for temporary files. If None, uses system temp.
        """
        self.base_temp_dir = base_temp_dir or tempfile.gettempdir()
        self._active_sandboxes: Dict[str, Dict[str, Any]] = {}
        self._cleanup_lock = asyncio.Lock()

        # Security settings
        self.max_execution_time = 300  # 5 minutes
        self.max_output_size = 10 * 1024 * 1024  # 10MB
        self.max_memory_mb = 512  # 512MB per tool
        self.allowed_paths: List[str] = []  # Paths tools are allowed to access

    async def prepare_environment(self, tool: BashTool, context: ExecutionContext) -> Dict[str, str]:
        """Prepare the execution environment for a tool.

        Args:
            tool: The tool to prepare environment for
            context: Execution context information

        Returns:
            Dictionary of environment variables for the tool
        """
        # Start with current environment
        env = os.environ.copy()

        # Apply tool-specific configuration
        config = tool.config

        # Set working directory
        if config.working_directory:
            working_dir = await self._prepare_working_directory(config.working_directory, context)
            env["TOOL_WORKING_DIR"] = working_dir
            env["PWD"] = working_dir

        # Inject tool-specific environment variables
        if config.environment_variables:
            env.update(config.environment_variables)

        # Add execution context metadata
        env.update({
            "TOOL_ID": tool.id,
            "TOOL_NAME": tool.name,
            "SESSION_ID": context.session_id,
            "TASK_ID": context.task_id,
            "USER_ID": context.user_id,
            "EXECUTION_TIMESTAMP": context.timestamp.isoformat() if context.timestamp else datetime.now().isoformat(),
            "TEMP": env.get("TOOL_WORKING_DIR", tempfile.gettempdir()),
            "TMPDIR": env.get("TOOL_WORKING_DIR", tempfile.gettempdir()),
        })

        # Security: Restrict dangerous environment variables
        await self._sanitize_environment(env)

        # Apply filesystem access controls
        await self._apply_filesystem_controls(env, config)

        # Apply network access controls
        await self._apply_network_controls(env, config)

        return env

    async def _prepare_working_directory(self, base_dir: str, context: ExecutionContext) -> str:
        """Prepare and return the working directory for tool execution."""
        # Create a unique subdirectory for this execution
        execution_id = str(uuid.uuid4())[:8]
        working_dir = Path(base_dir) / f"tool_{context.tool_id}_{execution_id}"

        try:
            # Create the directory
            working_dir.mkdir(parents=True, exist_ok=True)

            # Set restrictive permissions (readable/writable by owner only)
            working_dir.chmod(0o700)

            # Track for cleanup
            async with self._cleanup_lock:
                self._active_sandboxes[str(working_dir)] = {
                    "created": datetime.now(),
                    "tool_id": context.tool_id,
                    "context": context
                }

            logger.debug(f"Prepared working directory: {working_dir}")
            return str(working_dir)

        except Exception as e:
            logger.error(f"Failed to prepare working directory {working_dir}", extra={"error": str(e)})
            raise SandboxError(f"Failed to prepare working directory: {e}")

    async def _sanitize_environment(self, env: Dict[str, Any]) -> None:
        """Sanitize environment variables for security."""
        # Remove or restrict dangerous variables
        dangerous_vars = [
            "LD_PRELOAD", "LD_LIBRARY_PATH",  # Library injection
            "PATH",  # We'll set our own controlled PATH
            "HOME", "USER", "SHELL",  # User identity
            "SSH_AUTH_SOCK", "SSH_AGENT_PID",  # SSH access
            "GPG_AGENT_INFO",  # GPG access
            "DBUS_SESSION_BUS_ADDRESS",  # D-Bus access
        ]

        for var in dangerous_vars:
            if var in env:
                if var == "PATH":
                    # Set a controlled PATH
                    env[var] = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
                else:
                    # Remove dangerous variables
                    del env[var]

        # Set controlled PATH if not already set
        if "PATH" not in env:
            env["PATH"] = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

    async def _apply_filesystem_controls(self, env: Dict[str, str], config: ToolConfig) -> None:
        """Apply filesystem access controls for the tool.

        Args:
            env: Environment variables dictionary to modify
            config: Tool configuration with security settings
        """
        # Set restrictive umask for file creation
        env["UMASK"] = "0077"  # Files created will be readable/writable by owner only

        # Set restrictive file permissions for new files
        env["FORCE_FILE_PERMISSIONS"] = "600"  # -rw-------
        env["FORCE_DIR_PERMISSIONS"] = "700"   # drwx------

        # If tool has allowed_commands, validate file operations against allowlist
        if config.allowed_commands:
            allowed_patterns = [cmd.strip() for cmd in config.allowed_commands]

            # Map common file operations to allowed commands
            file_operation_mapping = {
                "read": ["cat", "head", "tail", "more", "less"],
                "write": ["echo", "printf", "tee"],
                "list": ["ls", "find", "dir"],
                "copy": ["cp", "scp"],
                "move": ["mv"],
                "delete": ["rm", "rmdir"],
                "permissions": ["chmod", "chown", "chgrp"],
            }

            # Validate that requested file operations are in allowed commands
            # This is a basic check - more sophisticated validation would be needed for production
            for operation, required_commands in file_operation_mapping.items():
                operation_allowed = any(cmd in allowed_patterns for cmd in required_commands)
                env[f"FILE_OP_{operation.upper()}_ALLOWED"] = str(operation_allowed).lower()

    async def _apply_network_controls(self, env: Dict[str, str], config: ToolConfig) -> None:
        """Apply network access controls for the tool.

        Args:
            env: Environment variables dictionary to modify
            config: Tool configuration with security settings
        """
        # Default network restrictions
        env["NETWORK_ACCESS_ALLOWED"] = "true"  # Basic network access is generally allowed

        # If tool has specific allowed commands, check for network operations
        if config.allowed_commands:
            allowed_patterns = [cmd.strip() for cmd in config.allowed_commands]

            # Check for allowed network commands
            network_commands = ["curl", "wget", "ping", "nslookup", "dig", "nc", "netcat", "ssh", "scp"]
            network_allowed = any(cmd in allowed_patterns for cmd in network_commands)

            if not network_allowed:
                # If no network commands are explicitly allowed, restrict network access
                env["NETWORK_ACCESS_ALLOWED"] = "false"
                env["CURL_ALLOWED"] = "false"
                env["WGET_ALLOWED"] = "false"
            else:
                # Check specific network command permissions
                env["CURL_ALLOWED"] = str("curl" in allowed_patterns).lower()
                env["WGET_ALLOWED"] = str("wget" in allowed_patterns).lower()

        # Set DNS restrictions (basic)
        env["DNS_RESTRICTED"] = "true"

    async def validate_script_security(self, script: str, allowed_commands: Optional[List[str]] = None) -> tuple[bool, List[str]]:
        """Validate script for security violations.

        Args:
            script: The bash script to validate
            allowed_commands: List of allowed command patterns

        Returns:
            Tuple of (is_valid, list_of_warnings)
        """
        if not allowed_commands:
            # If no allowlist, only allow basic safe commands
            allowed_commands = [
                "curl", "wget", "grep", "awk", "sed", "cut", "sort", "uniq",
                "head", "tail", "cat", "echo", "printf", "date", "basename",
                "dirname", "wc", "tr", "tee", "jq", "base64"
            ]

        warnings = []
        violations = []

        # Enhanced dangerous patterns with command injection prevention
        dangerous_patterns = [
            (r"\brm\b", "File deletion command detected"),
            (r"\brmdir\b", "Directory deletion command detected"),
            (r"\bmv\b.*\s*/\s*", "File moving to root filesystem"),
            (r"\bchmod\b", "File permission changes"),
            (r"\bchown\b", "File ownership changes"),
            (r"\bchgrp\b", "File group changes"),
            (r"\bsu\b", "Privilege escalation attempt"),
            (r"\bsudo\b", "Privilege escalation attempt"),
            (r"\bpasswd\b", "Password modification attempt"),
            (r"\bkill\b", "Process termination command"),
            (r"\bkillall\b", "Process termination command"),
            (r"\breboot\b", "System reboot command"),
            (r"\bshutdown\b", "System shutdown command"),
            (r"\bmount\b", "Filesystem mounting"),
            (r"\bumount\b", "Filesystem unmounting"),
            (r"\bapt\b", "Package management command"),
            (r"\byum\b", "Package management command"),
            (r"\bpacman\b", "Package management command"),
            (r"\bservice\b", "Service management command"),
            (r"\ssystemctl\b", "Service management command"),
            (r"\bcrontab\b", "Scheduled task management"),
            (r"\bat\b", "Scheduled task creation"),
            (r"\bssh\b", "SSH access command"),
            (r"\bscp\b", "SCP file transfer"),
            (r"\bsftp\b", "SFTP access"),
        ]

        # Command injection patterns
        injection_patterns = [
            (r';\s*rm\s+', "Command chaining with rm (potential injection)"),
            (r'\|\s*rm\s+', "Pipe to rm (potential injection)"),
            (r'&&\s*rm\s+', "Conditional execution with rm (potential injection)"),
            (r';\s*wget\s+', "Command chaining with wget (potential injection)"),
            (r';\s*curl\s+', "Command chaining with curl (potential injection)"),
            (r'>\s*/dev/null', "Output redirection hiding (potential injection)"),
            (r'2>\s*/dev/null', "Error redirection hiding (potential injection)"),
            (r'\$\((?!\()', "Command substitution (potential injection)"),
            (r'`.*`', "Backtick command execution (potential injection)"),
            (r';\s*cat\s+', "Command chaining with cat (potential injection)"),
            (r'\|\s*cat\s+', "Pipe to cat (potential injection)"),
        ]

        script_lower = script.lower()

        # Check for dangerous commands
        for pattern, description in dangerous_patterns:
            if re.search(pattern, script_lower):
                command_name = pattern.split(r"\b")[1] if r"\b" in pattern else pattern.replace('\\b', '').replace('\\s+', ' ')
                if not any(cmd in script for cmd in allowed_commands if cmd.lower() in command_name.lower()):
                    violations.append(f"Blocked dangerous command: {description}")
                else:
                    warnings.append(f"Warning: {description} (allowed by configuration)")

        # Check for command injection patterns
        for pattern, description in injection_patterns:
            if re.search(pattern, script, re.IGNORECASE):
                violations.append(f"Blocked command injection attempt: {description}")

        # Check for filesystem path traversal
        if re.search(r'\.\.\s*/', script) or re.search(r'/\.\.', script):
            violations.append("Blocked path traversal attempt")

        # Check for suspicious file operations
        if re.search(r'[<>]\s*/', script):
            violations.append("Blocked suspicious file redirection")

        # Validate file paths in common commands
        file_operation_violations = self._validate_file_paths(script, allowed_commands)
        violations.extend(file_operation_violations)

        is_valid = len(violations) == 0

        # Log security events
        if violations or warnings:
            self._log_security_events(script, violations, warnings)

        return is_valid, warnings + violations

    def _validate_file_paths(self, script: str, allowed_commands: List[str]) -> List[str]:
        """Validate file paths in script for security."""
        violations = []

        # Extract file paths from common commands
        lines = script.split('\n')
        for line in lines:
            line = line.strip()

            # Check file reading operations
            if any(cmd in line for cmd in ['cat ', 'head ', 'tail ', 'less ', 'more ']):
                parts = line.split()
                if len(parts) >= 2:
                    file_path = parts[-1]
                    if not self._is_path_allowed(file_path, allowed_commands):
                        violations.append(f"Blocked file access: {file_path}")

            # Check file writing operations
            elif any(cmd in line for cmd in ['echo ', 'printf ', 'tee ']) and '>' in line:
                parts = line.split('>')
                if len(parts) >= 2:
                    file_path = parts[1].strip()
                    if not self._is_path_allowed(file_path, allowed_commands):
                        violations.append(f"Blocked file write: {file_path}")

            # Check directory operations
            elif 'cd ' in line:
                dir_path = line.replace('cd ', '').strip()
                if not self._is_path_allowed(dir_path, allowed_commands):
                    violations.append(f"Blocked directory access: {dir_path}")

        return violations

    def _is_path_allowed(self, path: str, allowed_commands: List[str]) -> bool:
        """Check if a file path is allowed based on security configuration."""
        # Basic path validation - in production this would be more sophisticated
        if not path or path.startswith('/dev/') or path.startswith('/proc/'):
            return False

        if '..' in path:
            return False

        # If no specific restrictions, allow common safe paths
        if not self.allowed_paths:
            return True

        # Check against allowed paths
        for allowed_path in self.allowed_paths:
            if path.startswith(allowed_path):
                return True

        return False

    def _log_security_events(self, script: str, violations: List[str], warnings: List[str]) -> None:
        """Log security events for monitoring and audit."""
        if violations:
            logger.warning("Security violations detected in script", extra={
                "violations": violations,
                "script_length": len(script),
                "script_preview": script[:200] + "..." if len(script) > 200 else script
            })

        if warnings:
            logger.info("Security warnings in script", extra={
                "warnings": warnings,
                "script_length": len(script)
            })

    def _set_process_limits(self, tool_config: Optional[ToolConfig] = None):
        """Set process resource limits for security.

        Args:
            tool_config: Tool configuration with optional resource limits. If None, uses defaults.
        """
        try:
            # Get limits from tool config or use defaults
            max_memory_mb = tool_config.max_memory_mb if tool_config else None
            max_cpu_seconds = tool_config.max_cpu_time_seconds if tool_config else None
            max_file_mb = tool_config.max_file_size_mb if tool_config else None
            max_files = tool_config.max_open_files if tool_config else None
            max_processes = tool_config.max_processes if tool_config else None

            # Apply memory limit (default: 512MB)
            memory_limit_mb = max_memory_mb or 512
            memory_bytes = memory_limit_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))

            # Apply CPU time limit (default: 30 seconds)
            cpu_limit_seconds = max_cpu_seconds or 30
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit_seconds, cpu_limit_seconds))

            # Apply file size limit (default: 10MB)
            file_limit_mb = max_file_mb or 10
            file_bytes = file_limit_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_FSIZE, (file_bytes, file_bytes))

            # Apply open files limit (default: 100)
            files_limit = max_files or 100
            resource.setrlimit(resource.RLIMIT_NOFILE, (files_limit, files_limit))

            # Apply process limit (default: 10)
            process_limit = max_processes or 10
            resource.setrlimit(resource.RLIMIT_NPROC, (process_limit, process_limit))

            logger.debug(f"Set process limits for tool", extra={
                "memory_mb": memory_limit_mb,
                "cpu_seconds": cpu_limit_seconds,
                "file_mb": file_limit_mb,
                "files": files_limit,
                "processes": process_limit
            })

        except (ImportError, AttributeError):
            # Resource module not available on all platforms
            logger.debug("Resource limits not available on this platform")
        except Exception as e:
            logger.warning(f"Could not set process limits: {e}")

    async def execute_in_sandbox(
        self,
        script: str,
        env: Dict[str, str],
        context: ExecutionContext,
        timeout: Optional[int] = None,
        tool_config: Optional[ToolConfig] = None
    ) -> ExecutionResult:
        """Execute a script in the sandboxed environment.

        Args:
            script: Bash script to execute
            env: Environment variables
            context: Execution context
            timeout: Execution timeout in seconds

        Returns:
            Execution result with output and metadata
        """
        start_time = datetime.now()
        working_dir = env.get("TOOL_WORKING_DIR", tempfile.gettempdir())

        try:
            # Enhanced script security validation
            allowed_commands = context.metadata.get("allowed_commands") if context.metadata else None
            is_valid, security_issues = await self.validate_script_security(script, allowed_commands)

            if not is_valid:
                raise SandboxSecurityError(f"Script failed security validation: {'; '.join(security_issues)}")

            # Log security warnings if any
            if security_issues:
                logger.warning("Script security warnings", extra={
                    "tool_id": context.tool_id,
                    "warnings": security_issues
                })

            # Set up execution environment
            execution_timeout = timeout or 30

            # Execute the script with enhanced security
            process = await asyncio.create_subprocess_shell(
                script,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=working_dir,
                preexec_fn=lambda: self._set_process_limits(tool_config)  # Set resource limits
            )

            try:
                # Wait for completion with timeout
                stdout_data, stderr_data = await asyncio.wait_for(
                    process.communicate(),
                    timeout=execution_timeout
                )

                execution_time = (datetime.now() - start_time).total_seconds()

                # Decode output
                stdout = stdout_data.decode('utf-8', errors='replace') if stdout_data else ""
                stderr = stderr_data.decode('utf-8', errors='replace') if stderr_data else ""

                # Check output size limits
                if len(stdout.encode('utf-8')) > self.max_output_size:
                    raise SandboxError(f"Output size exceeds limit of {self.max_output_size} bytes")

                if len(stderr.encode('utf-8')) > self.max_output_size:
                    raise SandboxError(f"Error output size exceeds limit of {self.max_output_size} bytes")

                # Collect any output files created
                output_files = await self._collect_output_files(working_dir)

                logger.info(f"Script executed", extra={
                    "success": process.returncode == 0,
                    "stdout": stdout,
                    "stderr": stderr,
                    "tool_id": context.tool_id,
                    "return_code": process.returncode,
                    "execution_time": execution_time,
                    "output_files": output_files
                })

                return ExecutionResult(
                    success=process.returncode == 0,
                    return_code=process.returncode or -1,
                    stdout=stdout,
                    stderr=stderr,
                    execution_time=execution_time,
                    output_files=output_files,
                    metadata={
                        "working_directory": working_dir,
                        "timeout": execution_timeout,
                        "process_id": process.pid,
                        "tool_id": context.tool_id,
                    }
                )

            except asyncio.TimeoutError:
                # Kill the process group if it exists
                if hasattr(os, 'killpg'):
                    try:
                        os.killpg(os.getpgid(process.pid), 9)  # SIGKILL
                    except (ProcessLookupError, OSError):
                        pass

                raise SandboxError(f"Script execution timed out after {execution_timeout} seconds")

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Sandbox execution failed", extra={
                "tool_id": context.tool_id,
                "error": str(e),
                "execution_time": execution_time
            })

            return ExecutionResult(
                success=False,
                return_code=-1,
                stdout="",
                stderr=str(e),
                execution_time=execution_time,
                metadata={
                    "working_directory": working_dir,
                    "error_type": type(e).__name__,
                    "tool_id": context.tool_id,
                }
            )

    async def _collect_output_files(self, working_dir: str) -> List[str]:
        """Collect any files created in the working directory."""
        output_files = []

        try:
            working_path = Path(working_dir)
            if not working_path.exists():
                return output_files

            # Look for common output file patterns
            output_patterns = [
                "*.txt", "*.json", "*.csv", "*.xml", "*.yaml", "*.yml",
                "output.*", "result.*", "*.out", "*.result"
            ]

            for pattern in output_patterns:
                for file_path in working_path.glob(pattern):
                    if file_path.is_file():
                        output_files.append(str(file_path.relative_to(working_path)))

        except Exception as e:
            logger.warning(f"Failed to collect output files from {working_dir}", extra={"error": str(e)})

        return output_files

    async def cleanup_sandbox(self, working_dir: str, context: ExecutionContext) -> None:
        """Clean up a sandboxed execution environment.

        Args:
            working_dir: Working directory to clean up
            context: Execution context
        """
        async with self._cleanup_lock:
            if working_dir not in self._active_sandboxes:
                return

            try:
                working_path = Path(working_dir)

                # Remove the working directory and all contents
                if working_path.exists():
                    import shutil

                    # Set permissions to allow deletion
                    for file_path in working_path.rglob('*'):
                        if file_path.is_file():
                            file_path.chmod(0o600)
                        elif file_path.is_dir():
                            file_path.chmod(0o700)

                    shutil.rmtree(working_path, ignore_errors=True)
                    logger.debug(f"Cleaned up sandbox: {working_dir}")

                # Remove from tracking
                self._active_sandboxes.pop(working_dir, None)

            except Exception as e:
                logger.error(f"Failed to cleanup sandbox {working_dir}", extra={"error": str(e)})

    async def cleanup_expired_sandboxes(self, max_age_seconds: int = 3600) -> int:
        """Clean up sandboxes older than the specified age.

        Args:
            max_age_seconds: Maximum age in seconds before cleanup

        Returns:
            Number of sandboxes cleaned up
        """
        async with self._cleanup_lock:
            current_time = datetime.now()
            expired_sandboxes = []

            for sandbox_dir, info in self._active_sandboxes.items():
                age = (current_time - info["created"]).total_seconds()
                if age > max_age_seconds:
                    expired_sandboxes.append(sandbox_dir)

            # Clean up expired sandboxes
            for sandbox_dir in expired_sandboxes:
                info = self._active_sandboxes[sandbox_dir]
                await self.cleanup_sandbox(sandbox_dir, info["context"])

            return len(expired_sandboxes)

    async def get_sandbox_stats(self) -> Dict[str, Any]:
        """Get statistics about active sandboxes."""
        async with self._cleanup_lock:
            current_time = datetime.now()

            stats = {
                "active_sandboxes": len(self._active_sandboxes),
                "sandboxes_by_tool": {},
                "oldest_sandbox_age": 0,
                "total_sandboxes_created": 0  # Would need persistent tracking for this
            }

            if self._active_sandboxes:
                oldest_time = min(
                    info["created"] for info in self._active_sandboxes.values()
                )
                stats["oldest_sandbox_age"] = (current_time - oldest_time).total_seconds()

                # Count by tool
                tool_counts = {}
                for info in self._active_sandboxes.values():
                    tool_id = info["tool_id"]
                    tool_counts[tool_id] = tool_counts.get(tool_id, 0) + 1
                stats["sandboxes_by_tool"] = tool_counts

            return stats


# Global sandbox manager instance
_sandbox_manager: Optional[ToolSandbox] = None


def get_sandbox_manager() -> ToolSandbox:
    """Get the global sandbox manager instance."""
    global _sandbox_manager
    if _sandbox_manager is None:
        _sandbox_manager = ToolSandbox()
    return _sandbox_manager


@asynccontextmanager
async def managed_sandbox(
    tool: BashTool,
    context: ExecutionContext
) -> AsyncGenerator[tuple[Dict[str, str], str], None]:
    """Context manager for managed sandbox execution.

    Args:
        tool: The tool to execute
        context: Execution context

    Yields:
        Tuple of (environment_dict, working_directory)
    """
    sandbox = get_sandbox_manager()
    env = await sandbox.prepare_environment(tool, context)
    working_dir = env.get("TOOL_WORKING_DIR", tempfile.gettempdir())

    try:
        yield env, working_dir
    finally:
        await sandbox.cleanup_sandbox(working_dir, context)
