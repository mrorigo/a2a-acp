"""
Bash Tool Executor

Core execution engine for bash-based tool scripts with comprehensive error handling,
parameter processing, template rendering, and ZedACP integration.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from .tool_config import BashTool
from .sandbox import (
    ToolSandbox,
    ExecutionContext,
    ExecutionResult,
    SandboxError,
    SandboxSecurityError,
    get_sandbox_manager,
    managed_sandbox,
)
from .push_notification_manager import PushNotificationManager
from .audit import get_audit_logger, AuditEventType
from .error_profiles import ErrorProfile, ErrorContract, build_acp_error
from a2a_acp.a2a.models import InputRequiredNotification
from .models import (
    ToolOutput,
    ErrorDetails,
    ExecuteDetails,
    FileDiff,
    DevelopmentToolEvent,
    DevelopmentToolEventKind,
    McpDetails,
)

logger = logging.getLogger(__name__)

ACP_ERROR_CODES = {
    "PARSE_ERROR": -32700,
    "INVALID_REQUEST": -32600,
    "METHOD_NOT_FOUND": -32601,
    "INVALID_PARAMS": -32602,
    "INTERNAL_ERROR": -32603,
    "AUTH_REQUIRED": -32000,
    "RESOURCE_NOT_FOUND": -32002,
}

DEFAULT_ERROR_MESSAGES = {
    ACP_ERROR_CODES["PARSE_ERROR"]: "Parse error",
    ACP_ERROR_CODES["INVALID_REQUEST"]: "Invalid request",
    ACP_ERROR_CODES["METHOD_NOT_FOUND"]: "Command not found",
    ACP_ERROR_CODES["INVALID_PARAMS"]: "Invalid parameters",
    ACP_ERROR_CODES["INTERNAL_ERROR"]: "Internal error",
    ACP_ERROR_CODES["AUTH_REQUIRED"]: "Authentication required",
    ACP_ERROR_CODES["RESOURCE_NOT_FOUND"]: "Resource not found",
}

RETRYABLE_ERROR_CODES = {
    ACP_ERROR_CODES["INTERNAL_ERROR"],
}

@dataclass
class ToolExecutionResult:
    """Structured result of tool execution."""
    tool_id: str
    success: bool
    output: str
    error: str
    execution_time: float
    return_code: int
    metadata: Dict[str, Any]
    output_files: List[str]
    mcp_error: Optional[Dict[str, Any]] = None

    @classmethod
    def from_execution_result(
        cls,
        tool_id: str,
        execution_result: ExecutionResult
    ) -> ToolExecutionResult:
        """Create ToolExecutionResult from ExecutionResult."""
        return cls(
            tool_id=tool_id,
            success=execution_result.success,
            output=execution_result.stdout,
            error=execution_result.stderr,
            execution_time=execution_result.execution_time,
            return_code=execution_result.return_code,
            metadata=execution_result.metadata or {},
            output_files=execution_result.output_files or [],
            mcp_error=execution_result.mcp_error if execution_result.mcp_error else None
        )


class ParameterError(Exception):
    """Raised when parameter processing fails."""
    pass


class TemplateRenderError(Exception):
    """Raised when script template rendering fails."""
    pass


class ToolError(Exception):
    """Base class for tool execution errors."""
    pass


class ToolRetryableError(ToolError):
    """Errors that are retryable (network issues, temporary failures)."""
    pass


class ToolPermanentError(ToolError):
    """Errors that are permanent (invalid parameters, tool not found)."""
    pass


class ToolCircuitBreaker:
    """Circuit breaker for failing tools."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 300):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state: str = "closed"  # closed, open, half-open

    def is_call_allowed(self) -> bool:
        """Check if call is allowed through circuit breaker."""
        if self.state == "closed":
            return True
        elif self.state == "open":
            # Check if recovery timeout has elapsed
            if (self.last_failure_time and
                (datetime.now() - self.last_failure_time).total_seconds() > self.recovery_timeout):
                self.state = "half-open"
                return True
            return False
        else:  # half-open
            return True

    def record_success(self) -> None:
        """Record successful execution."""
        if self.state == "half-open":
            # Success in half-open state, close the circuit
            self.state = "closed"
            self.failure_count = 0
        elif self.state == "closed":
            # Reset failure count on success
            self.failure_count = 0

    def record_failure(self) -> None:
        """Record failed execution."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.failure_count >= self.failure_threshold:
            self.state = "open"


class BashToolExecutor:
    """Executes bash-based tools with parameter processing and template rendering."""

    def __init__(
        self,
        sandbox: Optional[ToolSandbox] = None,
        push_notification_manager: Optional[PushNotificationManager] = None,
        task_manager: Optional[Any] = None,
        error_profile: ErrorProfile = ErrorProfile.ACP_BASIC,
    ):
        """Initialize the bash tool executor.

        Args:
            sandbox: Sandbox manager instance. If None, uses global instance.
            push_notification_manager: Push notification manager for event emission.
            task_manager: A2A task manager for INPUT_REQUIRED integration.
            error_profile: Selected MCP error contract profile.
        """
        self.sandbox = sandbox or get_sandbox_manager()
        self.push_notification_manager = push_notification_manager
        self.task_manager = task_manager
        self.error_profile = error_profile
        self._execution_cache: Dict[str, Dict[str, Any]] = {}

        # Circuit breakers for failing tools
        self._circuit_breakers: Dict[str, ToolCircuitBreaker] = {}
        self._cache_lock: Optional[asyncio.Lock] = None

        # Template patterns for parameter substitution
        self.template_patterns = {
            "variable": r"\{\{(\w+)(\|default\('([^']+)'\))?\}\}",
            "conditional": r"\{\{#(\w+)\}\}(.*?)\{\{\/\1\}\}",
            "loop": r"\{\{#(\w+)\}\}(.*?)\{\{\/\1\}\}",
        }

        # Retry configuration
        self.max_retry_attempts = 3
        self.retry_delay = 1.0  # seconds
        self.retry_backoff_factor = 2.0

    def _get_cache_lock(self) -> asyncio.Lock:
        """Lazily instantiate the cache lock when event loop is available."""
        if self._cache_lock is None:
            self._cache_lock = asyncio.Lock()
        return self._cache_lock

    def _metadata_error_profile(self) -> str:
        """Return a string-safe error profile identifier for metadata."""
        return self.error_profile.value.replace("-", "_")

    def _error_code_label(self, code: int) -> str:
        """Map an MCP code to its textual label when available."""
        for label, value in ACP_ERROR_CODES.items():
            if value == code:
                return label
        return str(code)

    async def execute_tool(
        self,
        tool: BashTool,
        parameters: Dict[str, Any],
        context: ExecutionContext
    ) -> ToolExecutionResult:
        """Execute a bash tool with the provided parameters.

        Args:
            tool: The tool to execute
            parameters: Parameters to pass to the tool
            context: Execution context

        Returns:
            Structured execution result
        """
        logger.info(f"Executing tool: {tool.id}", extra={
            "tool_id": tool.id,
            "tool_name": tool.name,
            "session_id": context.session_id,
            "task_id": context.task_id,
            "param_count": len(parameters)
        })

        start_time = datetime.now()

        # Emit tool execution started event with EXECUTING status
        await self._emit_tool_event("started", context, parameters=parameters)

        try:
            # Validate parameters
            is_valid, errors = tool.validate_parameters(parameters)
            if not is_valid:
                raise ParameterError(f"Parameter validation failed: {'; '.join(errors)}")

            # Handle tool confirmation if required
            if tool.config.requires_confirmation:
                confirmed, reason = await self._handle_tool_confirmation_a2a(tool, parameters, context)
                if not confirmed:
                    # Tool execution was cancelled or failed confirmation
                    await self._emit_tool_event("cancelled", context, reason=reason)
                    return ToolExecutionResult(
                        tool_id=tool.id,
                        success=False,
                        output="",
                        error=f"Tool execution cancelled: {reason}",
                        execution_time=0.0,
                        return_code=-1,
                        metadata={
                            "cancelled": True,
                            "reason": reason,
                            "parameters": parameters,
                            "tool_version": tool.version,
                            "cached": False,
                            "error_profile": self._metadata_error_profile(),
                        },
                        output_files=[]
                    )

            # Use managed sandbox for automatic cleanup
            async with managed_sandbox(tool, context) as (environment, working_dir):
                # Render the bash script with parameters
                rendered_script = await self.render_script(tool.script, parameters, context)

                # Execute the script
                execution_result = await self.sandbox.execute_in_sandbox(
                    script=rendered_script,
                    env=environment,
                    context=context,
                    timeout=tool.config.timeout,
                    tool_config=tool.config
                )

            # Convert to tool result
            tool_result = ToolExecutionResult.from_execution_result(tool.id, execution_result)

            # Add execution metadata
            metadata_profile = self._metadata_error_profile()
            tool_result.metadata.update({
                "execution_start": start_time.isoformat(),
                "execution_end": datetime.now().isoformat(),
                "parameters": parameters,
                "tool_version": tool.version,
                "cached": False,
                "error_profile": metadata_profile,
            })
            tool_result.metadata.setdefault("error_profile", metadata_profile)

            # Map to extension schemas
            if tool_result.success:
                if tool_result.metadata.get("mcp"):
                    details = McpDetails(tool_name=tool.name)
                else:
                    details = ExecuteDetails(
                        stdout=tool_result.output,
                        stderr=tool_result.error or None,
                        exit_code=tool_result.return_code
                    )
                tool_output = ToolOutput(content=tool_result.output, details=details)
                tool_result.metadata["tool_output"] = tool_output.to_dict()

                # For output_files, create FileDiff if applicable
                if tool_result.output_files:
                    file_diffs = []
                    for file_path in tool_result.output_files:
                        # Assume new file for now; in real impl, read content
                        file_diff = FileDiff(
                            path=file_path,
                            old_content=None,
                            new_content="File created during tool execution"  # Placeholder
                        )
                        file_diffs.append(file_diff.to_dict())
                    tool_result.metadata["file_diffs"] = file_diffs

                await self._emit_tool_event("completed", context,
                    success=True,
                    execution_time=tool_result.execution_time,
                    return_code=tool_result.return_code,
                    output_length=len(tool_result.output),
                    tool_output=tool_output.to_dict()
                )

                logger.info(f"Tool execution completed: {tool.id}", extra={
                    "tool_id": tool.id,
                    "success": tool_result.success,
                    "execution_time": tool_result.execution_time,
                    "return_code": tool_result.return_code,
                    "output_length": len(tool_result.output)
                })
            else:
                sanitized_error = self._sanitize_error_output(tool_result.error)
                error_message = sanitized_error or tool_result.error or "Tool execution failed"
                if tool_result.metadata.get("timeout") or tool_result.return_code == 124:
                    error_message = "Tool execution timeout"
                error_details = ErrorDetails(
                    message=error_message,
                    code=str(tool_result.return_code) if tool_result.return_code else "INTERNAL_ERROR",
                    details={"sanitized_stderr": sanitized_error or tool_result.error or ""}
                )
                tool_result.metadata["error_details"] = error_details.to_dict()

                error_contract = self._resolve_mcp_error(tool, tool_result)
                mcp_error = self._apply_error_contract(tool_result, error_contract)

                await self._emit_tool_event("failed", context,
                    success=False,
                    error=error_message,
                    execution_time=tool_result.execution_time,
                    return_code=tool_result.return_code,
                    mcp_error=mcp_error,
                    diagnostics=error_contract.diagnostics,
                    error_details=error_details.to_dict()
                )

                logger.error(f"Tool execution failed with non-zero exit: {tool.id}", extra={
                    "tool_id": tool.id,
                    "success": tool_result.success,
                    "execution_time": tool_result.execution_time,
                    "return_code": tool_result.return_code,
                    "error": tool_result.error,
                    "mcp_error": mcp_error,
                    "diagnostics": error_contract.diagnostics,
                })

            return tool_result

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()

            logger.error(f"Tool execution failed: {tool.id}", extra={
                "tool_id": tool.id,
                "error": str(e),
                "error_type": type(e).__name__,
                "execution_time": execution_time
            })

            error_contract = self._build_mcp_error_from_exception(e)
            mcp_error = error_contract.to_mcp_error()
            sanitized_error = self._sanitize_error_output(str(e))
            error_message = sanitized_error or str(e)
            code_value = self._error_code_label(error_contract.code)
            if error_contract.code != ACP_ERROR_CODES["INTERNAL_ERROR"]:
                code_value = str(error_contract.code)
            error_details = ErrorDetails(
                message=error_message,
                code=code_value,
                details={"sanitized_stderr": sanitized_error or str(e)}
            )

            # Emit tool execution failed event
            await self._emit_tool_event("failed", context,
                success=False,
                error=error_message,
                error_type=type(e).__name__,
                execution_time=execution_time,
                mcp_error=mcp_error,
                diagnostics=error_contract.diagnostics,
                error_details=error_details.to_dict(),
            )

            # Return error result
            result = ToolExecutionResult(
                tool_id=tool.id,
                success=False,
                output="",
                error=error_message,
                execution_time=execution_time,
                return_code=-1,
                metadata={
                    "execution_start": start_time.isoformat(),
                    "execution_end": datetime.now().isoformat(),
                    "error_type": type(e).__name__,
                    "parameters": parameters,
                    "tool_version": tool.version,
                    "cached": False,
                    "error_profile": self._metadata_error_profile(),
                    "error_details": error_details.to_dict(),
                },
                output_files=[],
                mcp_error=mcp_error
            )
            self._apply_error_contract(result, error_contract)
            return result

    async def _emit_tool_event(self, event_type: str, context: ExecutionContext, **event_data) -> None:
        """Emit a tool execution event via the push notification system.
 
        Args:
            event_type: Type of event (started, completed, failed)
            context: Execution context
            **event_data: Additional event data
        """
        if not self.push_notification_manager:
            return

        # Add development-tool extension metadata
        dev_tool_event = None
        if "tool_output" in event_data:
            dev_tool_event = DevelopmentToolEvent(
                kind=DevelopmentToolEventKind.TOOL_CALL_UPDATE,
                data={"status": "succeeded", "live_content": event_data.get("tool_output", {}).get("content", "")}
            )
            event_data["development-tool"] = dev_tool_event.to_dict()
        elif "error_details" in event_data:
            dev_tool_event = DevelopmentToolEvent(
                kind=DevelopmentToolEventKind.TOOL_CALL_UPDATE,
                data={"status": "failed"}
            )
            event_data["development-tool"] = dev_tool_event.to_dict()
        elif event_type == "started":
            dev_tool_event = DevelopmentToolEvent(
                kind=DevelopmentToolEventKind.TOOL_CALL_UPDATE,
                data={"status": "executing"}
            )
            event_data["development-tool"] = dev_tool_event.to_dict()

        event = {
            "event": f"tool_{event_type}",
            "task_id": context.task_id,
            "tool_id": context.tool_id,
            "session_id": context.session_id,
            "timestamp": datetime.now().isoformat(),
            **event_data
        }
 
        try:
            await self.push_notification_manager.send_notification(context.task_id, event)
 
            # Also log to audit system
            audit_logger = get_audit_logger()
            await audit_logger.log_tool_execution(
                self._map_event_type_to_audit(event_type),
                context,
                tool_id=context.tool_id,
                **event_data
            )
 
            logger.debug(f"Emitted tool {event_type} event", extra={
                "tool_id": context.tool_id,
                "task_id": context.task_id,
                "event_type": event_type
            })
        except Exception as e:
            logger.error(f"Failed to emit tool {event_type} event", extra={
                "tool_id": context.tool_id,
                "task_id": context.task_id,
                "error": str(e)
            })

    def _map_event_type_to_audit(self, event_type: str) -> AuditEventType:
        """Map internal event types to audit event types."""
        mapping = {
            "started": AuditEventType.TOOL_EXECUTION_STARTED,
            "completed": AuditEventType.TOOL_EXECUTION_COMPLETED,
            "failed": AuditEventType.TOOL_EXECUTION_FAILED,
            "cancelled": AuditEventType.TOOL_EXECUTION_FAILED,
            "confirmation_required": AuditEventType.TOOL_CONFIRMATION_REQUESTED,
            "permanent_failure": AuditEventType.TOOL_EXECUTION_FAILED,
            "max_retries_exceeded": AuditEventType.TOOL_EXECUTION_FAILED,
            "retry_attempt": AuditEventType.TOOL_EXECUTION_STARTED,
        }
        return mapping.get(event_type, AuditEventType.TOOL_EXECUTION_STARTED)

    async def _handle_tool_confirmation_a2a(
        self,
        tool: BashTool,
        parameters: Dict[str, Any],
        context: ExecutionContext
    ) -> tuple[bool, str]:
        """Handle tool confirmation using A2A INPUT_REQUIRED protocol.

        Args:
            tool: The tool requiring confirmation
            parameters: Tool parameters
            context: Execution context

        Returns:
            Tuple of (confirmed, reason). Reason is empty if confirmed, or cancellation reason.
        """
        if not self.task_manager:
            # Fallback to simple confirmation if no task manager available
            logger.warning("No task manager available for A2A INPUT_REQUIRED confirmation")
            return True, ""

        try:
            # Get the task from the task manager
            task = await self.task_manager.get_task(context.task_id)
            if not task:
                logger.error(f"Task not found for confirmation: {context.task_id}")
                return False, "Task not found"

            # Create INPUT_REQUIRED notification
            input_notification = InputRequiredNotification(
                taskId=context.task_id,
                contextId=task.contextId,
                message=tool.config.confirmation_message or f"Execute tool '{tool.name}'?",
                inputTypes=["text/plain"],  # Simple confirmation
                timeout=300,  # 5 minute timeout
                metadata={
                    "tool_id": tool.id,
                    "tool_name": tool.name,
                    "parameters": parameters,
                    "confirmation_required": True
                }
            )

            # Emit the INPUT_REQUIRED event via push notification system
            await self._emit_tool_event("confirmation_required", context,
                confirmation_message=tool.config.confirmation_message,
                tool_name=tool.name,
                input_notification=input_notification.model_dump()
            )

            logger.info(f"Tool confirmation required via A2A INPUT_REQUIRED: {tool.id}", extra={
                "tool_id": tool.id,
                "task_id": context.task_id,
                "confirmation_message": tool.config.confirmation_message
            })

            # For now, return True as the confirmation flow will be handled
            # by the existing A2A INPUT_REQUIRED detection in task manager
            # The actual confirmation response will come through the normal
            # task continuation flow

            return True, ""

        except Exception as e:
            logger.error(f"Failed to handle A2A tool confirmation: {tool.id}", extra={
                "tool_id": tool.id,
                "task_id": context.task_id,
                "error": str(e)
            })
            return False, f"Confirmation failed: {str(e)}"

    def _get_circuit_breaker(self, tool_id: str) -> ToolCircuitBreaker:
        """Get or create circuit breaker for a tool."""
        if tool_id not in self._circuit_breakers:
            self._circuit_breakers[tool_id] = ToolCircuitBreaker()
        return self._circuit_breakers[tool_id]

    def _classify_error(self, error: Exception, return_code: int, stderr: str) -> ToolError:
        """Classify an error as retryable or permanent.

        Args:
            error: The exception that occurred
            return_code: Process return code
            stderr: Standard error output

        Returns:
            Classified error type
        """
        error_msg = str(error).lower()
        stderr_lower = stderr.lower()

        # Permanent errors (don't retry)
        permanent_patterns = [
            "command not found",
            "permission denied",
            "no such file or directory",
            "invalid parameter",
            "parameter validation failed",
            "template render error",
            "syntax error",
        ]

        for pattern in permanent_patterns:
            if pattern in error_msg or pattern in stderr_lower:
                return ToolPermanentError(f"Permanent error: {error}")

        # Retryable errors (network, temporary issues)
        retryable_patterns = [
            "connection refused",
            "connection timeout",
            "temporary failure",
            "network unreachable",
            "dns resolution failed",
            "http 5",  # 5xx server errors
            "timeout",
            "temporary",
        ]

        for pattern in retryable_patterns:
            if pattern in error_msg or pattern in stderr_lower:
                return ToolRetryableError(f"Retryable error: {error}")

        # Check return codes
        if return_code in [137, 143, 130]:  # SIGKILL, SIGTERM, SIGINT
            return ToolRetryableError(f"Process interrupted (code {return_code}): {error}")
        elif return_code >= 1 and return_code <= 125:  # General errors
                return ToolRetryableError(f"Retryable process error (code {return_code}): {error}")
        else:
            # Default to retryable for unknown errors
            return ToolRetryableError(f"Unknown error: {error}")

    def _apply_error_contract(
        self,
        result: ToolExecutionResult,
        contract: ErrorContract,
    ) -> Dict[str, Any]:
        """Attach a normalised error contract to the tool result metadata."""
        payload = contract.to_mcp_error()
        final_mcp_error = result.mcp_error or payload
        final_mcp_error_copy = dict(final_mcp_error)
        result.mcp_error = final_mcp_error_copy
        result.metadata["mcp_error"] = final_mcp_error_copy
        result.metadata.setdefault("error_profile", self._metadata_error_profile())
        if contract.diagnostics:
            diagnostics = result.metadata.setdefault("a2a_diagnostics", {})
            diagnostics.update(contract.diagnostics)
            result.metadata.setdefault("diagnostics", {}).update(contract.diagnostics)
        return final_mcp_error_copy

    @staticmethod
    def _sanitize_error_output(output: Optional[str]) -> Optional[str]:
        """Filter internal warning noise from tool stderr for cleaner diagnostics."""
        if not output:
            return output

        filtered = []
        for line in output.splitlines():
            if line.startswith("WARNING:a2a_acp.sandbox:Could not set process limits"):
                continue
            filtered.append(line)

        cleaned = "\n".join(filtered).strip()
        return cleaned or None

    def _resolve_mcp_error(self, tool: BashTool, result: ToolExecutionResult) -> ErrorContract:
        """Resolve the MCP error payload for a failed tool result."""
        mapping_entry = tool.error_mapping.get(result.return_code) if tool.error_mapping else None

        if mapping_entry:
            return self._build_error_contract(
                code=mapping_entry.code,
                configured_message=mapping_entry.message,
                result=result,
                retryable_override=mapping_entry.retryable,
                configured_detail=getattr(mapping_entry, "detail", None),
            )

        fallback_code = self._default_error_code_for_return_code(result.return_code)
        return self._build_error_contract(
            code=fallback_code,
            configured_message=None,
            result=result,
            retryable_override=None,
        )

    def _build_mcp_error_from_exception(self, error: Exception) -> ErrorContract:
        """Construct an MCP error payload from an exception."""
        message = str(error)

        if isinstance(error, ParameterError):
            return self._build_error_contract(ACP_ERROR_CODES["INVALID_PARAMS"], None, error_message=message, retryable_override=False)
        if isinstance(error, TemplateRenderError):
            return self._build_error_contract(ACP_ERROR_CODES["INVALID_PARAMS"], None, error_message=message, retryable_override=False)
        if isinstance(error, ToolPermanentError):
            return self._build_error_contract(ACP_ERROR_CODES["INVALID_PARAMS"], None, error_message=message, retryable_override=False)
        if isinstance(error, ToolRetryableError):
            return self._build_error_contract(ACP_ERROR_CODES["INTERNAL_ERROR"], None, error_message=message, retryable_override=True)
        if isinstance(error, SandboxSecurityError):
            return self._build_error_contract(ACP_ERROR_CODES["INVALID_REQUEST"], None, error_message=message, retryable_override=False)
        if isinstance(error, SandboxError):
            return self._build_error_contract(ACP_ERROR_CODES["INTERNAL_ERROR"], None, error_message=message, retryable_override=False)

        return self._build_error_contract(ACP_ERROR_CODES["INTERNAL_ERROR"], None, error_message=message, retryable_override=False)

    def _build_error_contract(
        self,
        code: int,
        configured_message: Optional[str],
        result: Optional[ToolExecutionResult] = None,
        retryable_override: Optional[bool] = None,
        error_message: Optional[str] = None,
        configured_detail: Optional[str] = None,
    ) -> ErrorContract:
        """Forms a consistent MCP error structure."""
        resolved_message = (
            (configured_message.strip() if configured_message else None)
            or DEFAULT_ERROR_MESSAGES.get(code)
            or "Tool execution failed"
        )

        sanitized_error: Optional[str] = None
        if result:
            sanitized_error = self._sanitize_error_output(result.error)

        detail_candidates: List[str] = []
        if configured_detail:
            detail_candidates.append(configured_detail.strip())
        if error_message:
            detail_candidates.append(error_message.strip())
        if sanitized_error:
            detail_candidates.append(sanitized_error)
        elif result and result.output:
            detail_candidates.append(result.output.strip())

        if result and isinstance(result.return_code, int) and result.return_code < 0:
            signal_number = abs(result.return_code)
            detail_candidates.append(f"Process terminated by signal {signal_number}")

        detail_value = next((m for m in detail_candidates if m), None)
        retryable = retryable_override if retryable_override is not None else self._is_retryable_error_code(code)

        diagnostics: Dict[str, Any] = {"error_profile": self._metadata_error_profile()}
        if result:
            diagnostics["return_code"] = result.return_code
            if isinstance(result.return_code, int) and result.return_code < 0:
                diagnostics["signal"] = abs(result.return_code)
        elif error_message:
            diagnostics["error_message"] = error_message

        return build_acp_error(
            profile=self.error_profile,
            code=code,
            message=resolved_message,
            detail=detail_value,
            retryable=retryable,
            diagnostics=diagnostics,
        )

    def _default_error_code_for_return_code(self, return_code: int) -> int:
        """Map a process return code to a reasonable default MCP error."""
        if return_code == 0:
            return ACP_ERROR_CODES["INTERNAL_ERROR"]
        if return_code == 2:
            return ACP_ERROR_CODES["INVALID_PARAMS"]
        if return_code in (126, 127):
            return ACP_ERROR_CODES["METHOD_NOT_FOUND"]
        if return_code in (130, 137, 143):
            return ACP_ERROR_CODES["INTERNAL_ERROR"]
        return ACP_ERROR_CODES["INTERNAL_ERROR"]

    def _is_retryable_error_code(self, code: int) -> bool:
        """Determine whether an MCP error code should be treated as retryable."""
        return code in RETRYABLE_ERROR_CODES

    async def render_script(
        self,
        template: str,
        parameters: Dict[str, Any],
        context: ExecutionContext
    ) -> str:
        """Render a script template with parameter values.

        Supports:
        - {{variable}} or {{variable|default('value')}} for variable substitution
        - {{#array}}...{{/array}} for array iteration
        - {{#object}}...{{/object}} for object property access

        Args:
            template: Script template with placeholders
            parameters: Parameter values for substitution
            context: Execution context

        Returns:
            Rendered script ready for execution
        """
        try:
            script = template

            # Handle variable substitution with defaults
            def replace_variable(match):
                var_name = match.group(1)
                default_value = match.group(3)  # From |default('value')

                if var_name in parameters:
                    value = parameters[var_name]
                    # Convert value to appropriate string representation
                    if isinstance(value, (dict, list)):
                        return str(value)  # JSON-like representation for complex types
                    return str(value)
                elif default_value is not None:
                    return default_value
                else:
                    raise TemplateRenderError(f"Missing required parameter: {var_name}")

            # Variable substitution pattern
            var_pattern = r"\{\{(\w+)(\|default\('([^']+)'\))?\}\}"
            script = re.sub(var_pattern, replace_variable, script)

            # Handle conditional blocks {{#condition}}...{{/condition}}
            def replace_conditional(match):
                var_name = match.group(1)
                content = match.group(2)

                if var_name in parameters and parameters[var_name]:
                    # Render nested variables in the content
                    return re.sub(var_pattern, replace_variable, content)
                return ""

            # Conditional pattern
            cond_pattern = r"\{\{#(\w+)\}\}(.*?)\{\{\/\1\}\}"
            script = re.sub(cond_pattern, replace_conditional, script)

            # Handle array/object iteration {{#items}}...{{/items}}
            def replace_iteration(match):
                var_name = match.group(1)
                content = match.group(2)

                if var_name not in parameters:
                    return ""

                value = parameters[var_name]

                if isinstance(value, list):
                    # Array iteration
                    result_parts = []
                    for item in value:
                        # Create temporary parameters for this iteration
                        iter_params = parameters.copy()
                        iter_params[f"{var_name}_item"] = item
                        iter_params[f"{var_name}_index"] = len(result_parts)

                        # Render content with item available
                        item_content = re.sub(var_pattern, lambda m: replace_variable_with_params(m, iter_params), content)
                        result_parts.append(item_content)

                    return "".join(result_parts)

                elif isinstance(value, dict):
                    # Object property access
                    result_parts = []
                    for key, val in value.items():
                        # Create temporary parameters for this property
                        iter_params = parameters.copy()
                        iter_params[f"{var_name}_key"] = key
                        iter_params[f"{var_name}_value"] = val

                        # Render content with property available
                        prop_content = re.sub(var_pattern, lambda m: replace_variable_with_params(m, iter_params), content)
                        result_parts.append(prop_content)

                    return "".join(result_parts)

                return ""

            def replace_variable_with_params(match, iter_params):
                var_name = match.group(1)
                default_value = match.group(3)

                if var_name in iter_params:
                    value = iter_params[var_name]
                    if isinstance(value, (dict, list)):
                        return str(value)
                    return str(value)
                elif default_value is not None:
                    return default_value
                else:
                    return ""

            # Array/object iteration pattern
            iter_pattern = r"\{\{#(\w+)\}\}(.*?)\{\{\/\1\}\}"
            script = re.sub(iter_pattern, replace_iteration, script)

            # Final cleanup - remove any remaining template syntax
            script = re.sub(r'\{\{[^}]+\}\}', '', script)

            return script.strip()

        except Exception as e:
            logger.error("Script template rendering failed", extra={
                "error": str(e),
                "template_length": len(template),
                "parameters": list(parameters.keys())
            })
            raise TemplateRenderError(f"Failed to render script template: {e}")

    async def execute_tool_with_caching(
        self,
        tool: BashTool,
        parameters: Dict[str, Any],
        context: ExecutionContext,
        use_cache: bool = True
    ) -> ToolExecutionResult:
        """Execute tool with optional response caching.

        Args:
            tool: The tool to execute
            parameters: Parameters for execution
            context: Execution context
            use_cache: Whether to use cached results when available

        Returns:
            Execution result (potentially from cache)
        """
        # Create cache key from tool ID, version, and parameters
        cache_key = self._create_cache_key(tool.id, tool.version, parameters)

        # Check cache first (only if tool caching is enabled)
        if use_cache and tool.config.caching_enabled:
            async with self._get_cache_lock():
                if cache_key in self._execution_cache:
                    cached_entry = self._execution_cache[cache_key]
                    cache_age = (datetime.now() - cached_entry["timestamp"]).total_seconds()

                    # Check if cache entry has expired
                    if cache_age < tool.config.cache_ttl_seconds:
                        cached_result = cached_entry["result"]
                        cached_result.metadata["cached"] = True
                        logger.debug(f"Using cached result for tool: {tool.id}", extra={
                            "cache_age_seconds": cache_age,
                            "ttl_seconds": tool.config.cache_ttl_seconds
                        })
                        return cached_result
                    else:
                        # Cache entry expired, remove it
                        logger.debug(f"Cache entry expired for tool: {tool.id}", extra={
                            "cache_age_seconds": cache_age,
                            "ttl_seconds": tool.config.cache_ttl_seconds
                        })
                        del self._execution_cache[cache_key]

        # Execute the tool
        result = await self.execute_tool(tool, parameters, context)

        # Cache the result (only if tool caching is enabled and execution succeeded)
        if use_cache and tool.config.caching_enabled and result.success:
            async with self._get_cache_lock():
                # Check cache size limit before adding
                cache_size_mb = len(str(self._execution_cache)) / (1024 * 1024)
                if cache_size_mb < tool.config.cache_max_size_mb:
                    self._execution_cache[cache_key] = {
                        "result": result,
                        "timestamp": datetime.now(),
                        "tool_id": tool.id,
                        "tool_version": tool.version
                    }
                    logger.debug(f"Cached result for tool: {tool.id}", extra={
                        "cache_size_mb": cache_size_mb,
                        "max_size_mb": tool.config.cache_max_size_mb
                    })
                else:
                    logger.warning(f"Cache size limit exceeded for tool: {tool.id}", extra={
                        "cache_size_mb": cache_size_mb,
                        "max_size_mb": tool.config.cache_max_size_mb
                    })

        return result

    def _create_cache_key(self, tool_id: str, tool_version: str, parameters: Dict[str, Any]) -> str:
        """Create a cache key from tool ID, version, and parameters."""
        # Sort parameters for consistent hashing
        sorted_params = sorted(parameters.items())
        param_str = str(sorted_params)

        # Include tool version in cache key for cache invalidation
        cache_content = f"{tool_id}:{tool_version}:{param_str}"

        # Create hash for cache key
        import hashlib
        content_hash = hashlib.md5(cache_content.encode()).hexdigest()[:8]

        return f"{tool_id}:{tool_version}:{content_hash}"

    async def clear_cache(self, tool_id: Optional[str] = None) -> int:
        """Clear execution cache.

        Args:
            tool_id: Specific tool ID to clear cache for. If None, clears all.

        Returns:
            Number of cache entries removed
        """
        async with self._get_cache_lock():
            if tool_id is None:
                count = len(self._execution_cache)
                self._execution_cache.clear()
                logger.debug(f"Cleared all cache entries: {count}")
            else:
                keys_to_remove = [key for key in self._execution_cache.keys() if key.startswith(f"{tool_id}:")]
                count = len(keys_to_remove)
                for key in keys_to_remove:
                    self._execution_cache.pop(key, None)
                logger.debug(f"Cleared {count} cache entries for tool: {tool_id}")

            return count

    async def cleanup_expired_cache(self) -> int:
        """Clean up expired cache entries based on tool-specific TTL settings.

        Returns:
            Number of expired entries removed
        """
        from .tool_config import get_tool

        async with self._get_cache_lock():
            expired_keys = []
            current_time = datetime.now()

            for cache_key, cache_entry in self._execution_cache.items():
                try:
                    # Extract tool_id from cache key (format: "tool_id:version:hash")
                    tool_id = cache_key.split(":")[0]

                    # Get tool to check its TTL setting
                    tool = await get_tool(tool_id)
                    if tool and tool.config.caching_enabled:
                        cache_age = (current_time - cache_entry["timestamp"]).total_seconds()
                        if cache_age > tool.config.cache_ttl_seconds:
                            expired_keys.append(cache_key)
                    else:
                        # Tool not found or caching disabled, remove entry
                        expired_keys.append(cache_key)

                except Exception as e:
                    logger.warning(f"Error checking cache expiry for key {cache_key}", extra={"error": str(e)})
                    # Remove problematic entries
                    expired_keys.append(cache_key)

            # Remove expired entries
            for key in expired_keys:
                self._execution_cache.pop(key, None)

            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

            return len(expired_keys)

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get detailed cache statistics."""
        async with self._get_cache_lock():
            datetime.now()

            # Basic stats
            total_entries = len(self._execution_cache)
            tools_cached = set()
            cache_size_mb = len(str(self._execution_cache)) / (1024 * 1024)  # Rough estimate

            # Tool-specific stats
            tool_stats = {}

            for cache_key, cache_entry in self._execution_cache.items():
                tool_id = cache_entry.get("tool_id", "unknown")
                tools_cached.add(tool_id)

                # Initialize tool stats if not exists
                if tool_id not in tool_stats:
                    tool_stats[tool_id] = {
                        "entries": 0,
                        "oldest_entry": None,
                        "newest_entry": None,
                        "total_size_mb": 0.0
                    }

                # Update tool stats
                tool_stats[tool_id]["entries"] += 1

                entry_timestamp = cache_entry.get("timestamp")
                if entry_timestamp:
                    if (tool_stats[tool_id]["oldest_entry"] is None or
                        entry_timestamp < tool_stats[tool_id]["oldest_entry"]):
                        tool_stats[tool_id]["oldest_entry"] = entry_timestamp

                    if (tool_stats[tool_id]["newest_entry"] is None or
                        entry_timestamp > tool_stats[tool_id]["newest_entry"]):
                        tool_stats[tool_id]["newest_entry"] = entry_timestamp

                # Estimate entry size
                entry_size = len(str(cache_entry))
                tool_stats[tool_id]["total_size_mb"] += entry_size / (1024 * 1024)

            return {
                "total_entries": total_entries,
                "cache_size_mb": cache_size_mb,
                "tools_cached": len(tools_cached),
                "tools_with_cache": list(tools_cached),
                "tool_stats": tool_stats,
                "cache_efficiency": {
                    "hit_ratio_estimate": "N/A",  # Would need hit/miss tracking
                    "average_entries_per_tool": total_entries / len(tools_cached) if tools_cached else 0,
                    "cache_utilization_percent": min(100.0, (cache_size_mb / 100.0) * 100)  # Assuming 100MB max
                }
            }

    async def execute_batch(
        self,
        executions: List[tuple[BashTool, Dict[str, Any], ExecutionContext]]
    ) -> List[ToolExecutionResult]:
        """Execute multiple tools in parallel.

        Args:
            executions: List of (tool, parameters, context) tuples

        Returns:
            List of execution results in same order as input
        """
        if not executions:
            return []

        logger.info(f"Executing batch of {len(executions)} tools")

        # Create tasks for parallel execution
        tasks = []
        for tool, parameters, context in executions:
            task = asyncio.create_task(
                self.execute_tool(tool, parameters, context)
            )
            tasks.append((task, tool.id))

        # Wait for all executions to complete
        results = []
        for task, tool_id in tasks:
            try:
                result = await task
                results.append(result)
            except Exception as e:
                logger.error(f"Batch execution failed for tool: {tool_id}", extra={"error": str(e)})
                # Create error result
                error_contract = self._build_mcp_error_from_exception(e)
                error_result = ToolExecutionResult(
                    tool_id=tool_id,
                    success=False,
                    output="",
                    error=str(e),
                    execution_time=0.0,
                    return_code=-1,
                    metadata={
                        "error_type": type(e).__name__,
                        "error_profile": self._metadata_error_profile(),
                    },
                    output_files=[],
                    mcp_error=None
                )
                self._apply_error_contract(error_result, error_contract)
                results.append(error_result)

        logger.info(f"Batch execution completed: {len(results)} results")
        return results

    async def validate_tool_script(self, tool: BashTool) -> tuple[bool, List[str]]:
        """Validate a tool's script for common issues.

        Args:
            tool: The tool to validate

        Returns:
            Tuple of (is_valid, list_of_warnings)
        """
        warnings = []

        # Check for missing script
        if not tool.script.strip():
            return False, ["Script template is empty"]

        # Check for dangerous patterns
        dangerous_patterns = [
            r"\brm\s+-rf\s+/",  # rm -rf /
            r"sudo\s+rm",       # sudo rm
            r"dd\s+if=/dev/zero",  # Disk wiping
            r">\s*/dev/sda",    # Overwriting disk
        ]

        script_lower = tool.script.lower()
        for pattern in dangerous_patterns:
            if re.search(pattern, script_lower):
                warnings.append(f"Potentially dangerous pattern found: {pattern}")

        # Check for template syntax issues
        open_braces = tool.script.count("{{")
        close_braces = tool.script.count("}}")

        if open_braces != close_braces:
            warnings.append(f"Mismatched template braces: {open_braces} open, {close_braces} close")

        # Check for common bash issues
        if "#!/bin/bash" not in tool.script and "#!/bin/sh" not in tool.script:
            warnings.append("Script should include shebang (#!/bin/bash or #!/bin/sh)")

        # Validate parameter references in template
        param_names = {param.name for param in tool.parameters}
        referenced_params = set(re.findall(r"\{\{(\w+)", tool.script))

        for ref_param in referenced_params:
            if ref_param not in param_names:
                warnings.append(f"Template references undefined parameter: {ref_param}")

        return len(warnings) == 0, warnings


# Global bash executor instance
_bash_executor: Optional[BashToolExecutor] = None


def get_bash_executor() -> BashToolExecutor:
    """Get the global bash executor instance."""
    global _bash_executor
    if _bash_executor is None:
        from .settings import get_settings

        settings = get_settings()
        _bash_executor = BashToolExecutor(error_profile=settings.error_profile)
    return _bash_executor


def set_bash_executor(executor: BashToolExecutor) -> None:
    """Set the global bash executor instance."""
    global _bash_executor
    _bash_executor = executor


async def execute_tool(
    tool_id: str,
    parameters: Dict[str, Any],
    session_id: str,
    task_id: str,
    user_id: str = "anonymous"
) -> ToolExecutionResult:
    """Convenience function to execute a tool by ID."""
    from .tool_config import get_tool

    tool = await get_tool(tool_id)
    if not tool:
        raise ValueError(f"Tool not found: {tool_id}")

    context = ExecutionContext(
        tool_id=tool_id,
        session_id=session_id,
        task_id=task_id,
        user_id=user_id
    )

    executor = get_bash_executor()
    return await executor.execute_tool(tool, parameters, context)


async def execute_tool_batch(
    executions: List[tuple[str, Dict[str, Any], str, str, str]]
) -> List[ToolExecutionResult]:
    """Convenience function to execute multiple tools by ID.

    Args:
        executions: List of (tool_id, parameters, session_id, task_id, user_id) tuples

    Returns:
        List of execution results
    """
    from .tool_config import get_tool

    # Resolve tool IDs to tool objects
    tools_and_contexts = []
    for tool_id, parameters, session_id, task_id, user_id in executions:
        tool = await get_tool(tool_id)
        if not tool:
            raise ValueError(f"Tool not found: {tool_id}")

        context = ExecutionContext(
            tool_id=tool_id,
            session_id=session_id,
            task_id=task_id,
            user_id=user_id
        )

        tools_and_contexts.append((tool, parameters, context))

    executor = get_bash_executor()
    return await executor.execute_batch(tools_and_contexts)
