"""
Tool Configuration System

YAML-based tool configuration loader and management for bash-based tool execution.
Supports dynamic tool discovery, parameter validation, and sandboxing configuration.
"""

from __future__ import annotations

import asyncio
import logging
import os
import yaml
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from .error_profiles import ErrorProfile

logger = logging.getLogger(__name__)


@dataclass
class ToolParameter:
    """Definition of a tool parameter with validation rules."""

    name: str
    type: str  # string, number, boolean, object, array
    required: bool = False
    description: str = ""
    default: Any = None
    enum: Optional[List[str]] = None
    minimum: Optional[Union[int, float]] = None
    maximum: Optional[Union[int, float]] = None
    pattern: Optional[str] = None  # regex pattern for string validation

    def validate(self, value: Any) -> tuple[bool, str]:
        """Validate a parameter value against the defined constraints."""
        # Type validation
        if self.type == "string" and not isinstance(value, str):
            return False, f"Parameter '{self.name}' must be a string"
        elif self.type == "number" and not isinstance(value, (int, float)):
            return False, f"Parameter '{self.name}' must be a number"
        elif self.type == "boolean" and not isinstance(value, bool):
            return False, f"Parameter '{self.name}' must be a boolean"
        elif self.type == "object" and not isinstance(value, dict):
            return False, f"Parameter '{self.name}' must be an object"
        elif self.type == "array" and not isinstance(value, list):
            return False, f"Parameter '{self.name}' must be an array"

        # String validations
        if self.type == "string" and isinstance(value, str):
            if self.pattern and not __import__("re").match(self.pattern, value):
                return False, f"Parameter '{self.name}' does not match required pattern"
            if self.enum and value not in self.enum:
                return False, f"Parameter '{self.name}' must be one of: {self.enum}"

        # Number validations
        if self.type == "number" and isinstance(value, (int, float)):
            if self.minimum is not None and value < self.minimum:
                return False, f"Parameter '{self.name}' must be >= {self.minimum}"
            if self.maximum is not None and value > self.maximum:
                return False, f"Parameter '{self.name}' must be <= {self.maximum}"

        return True, ""


@dataclass
class ToolConfig:
    """Sandboxing and execution configuration for a tool."""

    requires_confirmation: bool = False
    confirmation_message: str = ""
    timeout: int = 30
    working_directory: str = "/tmp"
    use_temp_isolation: bool = False  # New: Whether to create isolated temp directory
    environment_variables: Optional[Dict[str, str]] = None
    allowed_commands: Optional[List[str]] = None  # Command allowlist for security
    allowed_paths: List[str] = field(
        default_factory=list
    )  # Paths allowed when not using temp isolation
    resource_limits: Optional[Dict[str, Any]] = None
    caching_enabled: bool = True
    cache_ttl_seconds: int = 3600  # 1 hour default
    cache_max_size_mb: float = 100.0

    # Process-level resource limits (overrides sandbox defaults)
    max_memory_mb: Optional[int] = None  # Max memory in MB (default: 512)
    max_cpu_time_seconds: Optional[int] = None  # Max CPU time in seconds (default: 30)
    max_file_size_mb: Optional[int] = None  # Max file size in MB (default: 10)
    max_open_files: Optional[int] = None  # Max open files (default: 100)
    max_processes: Optional[int] = None  # Max child processes (default: 10)


@dataclass
class ToolErrorMappingEntry:
    """Maps a tool exit code to a Zed MCP error response."""

    code: int
    message: Optional[str] = None
    retryable: Optional[bool] = None
    detail: Optional[Any] = None


@dataclass
class BashTool:
    """Complete definition of a bash-based tool."""

    id: str
    name: str
    description: str
    script: str  # Bash script template with parameter placeholders
    parameters: List[ToolParameter] = field(default_factory=list)
    config: ToolConfig = field(default_factory=ToolConfig)
    tags: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    author: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    error_mapping: Dict[int, ToolErrorMappingEntry] = field(default_factory=dict)

    def __post_init__(self):
        """Normalize nested configuration structures for consistent typing."""
        self.parameters = [
            self._normalize_parameter(p) for p in (self.parameters or [])
        ]
        self.config = self._normalize_config(self.config)

    def _normalize_parameter(self, param: Any) -> ToolParameter:
        """Ensure tool parameters are stored as ToolParameter instances."""
        if isinstance(param, ToolParameter):
            return param
        if isinstance(param, dict):
            return ToolParameter(**param)
        if hasattr(param, "__dict__"):
            return ToolParameter(**vars(param))
        raise TypeError(f"Unsupported parameter definition type: {type(param)}")

    def _normalize_config(self, config_value: Any) -> ToolConfig:
        """Ensure tool configuration is a ToolConfig dataclass."""
        if isinstance(config_value, ToolConfig):
            return config_value
        if isinstance(config_value, dict):
            allowed = {f.name for f in fields(ToolConfig)}
            filtered = {k: v for k, v in config_value.items() if k in allowed}
            return ToolConfig(**filtered)
        if config_value is None:
            return ToolConfig()
        raise TypeError(f"Unsupported tool config type: {type(config_value)}")

    def validate_parameters(self, params: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate provided parameters against tool definition."""
        errors = []

        # Check required parameters
        for param in self.parameters:
            if param.required and param.name not in params:
                errors.append(f"Missing required parameter: '{param.name}'")
            elif param.name in params:
                value = params[param.name]
                is_valid, error_msg = param.validate(value)
                if not is_valid:
                    errors.append(error_msg)

        # Check for unexpected parameters
        param_names = {p.name for p in self.parameters}
        for param_name in params:
            if param_name not in param_names:
                errors.append(f"Unexpected parameter: '{param_name}'")

        return len(errors) == 0, errors

    def render_script(self, parameters: Dict[str, Any]) -> str:
        """Render the bash script template with provided parameters."""

        # Simple template rendering - replace {{param}} with values
        script = self.script
        for key, value in parameters.items():
            # Convert value to string if not already
            str_value = str(value) if value is not None else ""
            script = script.replace(f"{{{{{key}}}}}", str_value)

        return script

    def get_metadata(self) -> Dict[str, Any]:
        """Get tool metadata for API responses (excludes sensitive script content)."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "examples": self.examples,
            "version": self.version,
            "author": self.author,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "parameter_count": len(self.parameters),
            "requires_confirmation": self.config.requires_confirmation,
            "timeout": self.config.timeout,
            "working_directory": self.config.working_directory,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize tool to dictionary for storage or API responses."""
        data = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "script": self.script,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "required": p.required,
                    "description": p.description,
                    "default": p.default,
                    "enum": p.enum,
                    "minimum": p.minimum,
                    "maximum": p.maximum,
                    "pattern": p.pattern,
                }
                for p in self.parameters
            ],
            "config": {
                "requires_confirmation": self.config.requires_confirmation,
                "confirmation_message": self.config.confirmation_message,
                "timeout": self.config.timeout,
                "working_directory": self.config.working_directory,
                "use_temp_isolation": self.config.use_temp_isolation,
                "allowed_paths": self.config.allowed_paths,
                "environment_variables": self.config.environment_variables,
                "allowed_commands": self.config.allowed_commands,
                "caching_enabled": self.config.caching_enabled,
                "cache_ttl_seconds": self.config.cache_ttl_seconds,
                "cache_max_size_mb": self.config.cache_max_size_mb,
                "max_memory_mb": self.config.max_memory_mb,
                "max_cpu_time_seconds": self.config.max_cpu_time_seconds,
                "max_file_size_mb": self.config.max_file_size_mb,
                "max_open_files": self.config.max_open_files,
                "max_processes": self.config.max_processes,
            },
            "tags": self.tags,
            "examples": self.examples,
            "version": self.version,
            "author": self.author,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
        if self.error_mapping:
            data["error_mapping"] = {
                str(code): {
                    "code": entry.code,
                    **({"message": entry.message} if entry.message else {}),
                    **(
                        {"retryable": entry.retryable}
                        if entry.retryable is not None
                        else {}
                    ),
                    **({"detail": entry.detail} if entry.detail is not None else {}),
                }
                for code, entry in self.error_mapping.items()
            }
        return data


class ToolConfigurationError(Exception):
    """Raised when tool configuration is invalid."""

    pass


class ToolConfigurationManager:
    """Manages loading, validation, and hot-reloading of tool configurations."""

    def __init__(
        self,
        config_paths: Optional[List[str]] = None,
        error_profile: ErrorProfile = ErrorProfile.ACP_BASIC,
    ):
        """Initialize the tool configuration manager.

        Args:
            config_paths: List of paths to search for tool configuration files.
                         Defaults to standard locations.
        """
        default_paths = [
            "tools.yaml",
            "tools.yml",
            "config/tools.yaml",
            "config/tools.yml",
            "/etc/a2a-acp/tools.yaml",
        ]
        base_paths = config_paths if config_paths is not None else default_paths

        env_value = os.environ.get("A2A_TOOLS_CONFIG", "")
        env_paths = (
            [p.strip() for p in env_value.split(os.pathsep) if p.strip()]
            if env_value
            else []
        )

        combined_paths: List[str] = []
        for path in env_paths + base_paths:
            if path not in combined_paths:
                combined_paths.append(path)

        self.config_paths = combined_paths or default_paths
        self._tools: Dict[str, BashTool] = {}
        self._config_mtimes: Dict[str, float] = {}
        self._hot_reload_enabled = True
        self._lock = asyncio.Lock()
        self.error_profile = error_profile

    async def load_tools(self, force_reload: bool = False) -> Dict[str, BashTool]:
        """Load and validate all tool configurations.

        Args:
            force_reload: Force reload even if files haven't changed.

        Returns:
            Dictionary mapping tool IDs to BashTool instances.
        """
        # DEBUG: load_tools called with force_reload={force_reload}
        async with self._lock:
            await self._check_for_updates(force_reload)
            return self._tools.copy()

    async def get_tool(self, tool_id: str) -> Optional[BashTool]:
        """Get a specific tool by ID."""
        async with self._lock:
            await self._check_for_updates()
            return self._tools.get(tool_id)

    async def list_tools(self) -> List[BashTool]:
        """List all available tools."""
        async with self._lock:
            await self._check_for_updates()
            return list(self._tools.values())

    async def _check_for_updates(self, force_reload: bool = False) -> None:
        """Check for configuration file updates and reload if necessary."""
        if not self._hot_reload_enabled and not force_reload:
            return

        updated_tools = {}

        for config_path in self.config_paths:
            path = Path(config_path)
            if not path.exists():
                continue

            # Check if file has been modified
            current_mtime = path.stat().st_mtime
            if not force_reload and config_path in self._config_mtimes:
                if current_mtime <= self._config_mtimes[config_path]:
                    # File hasn't changed, use cached tools
                    updated_tools.update(self._tools)
                    continue

            try:
                # Load and parse configuration file
                tools_config = await self._load_config_file(path)
                if tools_config:
                    # Validate and create tool objects
                    tools = await self._parse_tools_config(tools_config, str(path))
                    updated_tools.update(tools)
                    self._config_mtimes[config_path] = current_mtime
                #    logger.info(f"Loaded {len(tools)} tools from {config_path}")

            except Exception as e:
                logger.error(
                    f"Failed to load tools from {config_path}", extra={"error": str(e)}
                )
                if config_path not in self._config_mtimes:
                    # Only raise error for new files, not existing ones
                    raise ToolConfigurationError(
                        f"Failed to load tools from {config_path}: {e}"
                    )

        self._tools = updated_tools

    async def _load_config_file(self, path: Path) -> Optional[Dict[str, Any]]:
        """Load a YAML configuration file."""
        try:
            # Use asyncio.to_thread for file I/O operations
            def read_file():
                with open(path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f)

            return await asyncio.to_thread(read_file)
        except Exception as e:
            logger.error(f"Error reading config file {path}", extra={"error": str(e)})
            return None

    async def _parse_tools_config(
        self, config: Dict[str, Any], source_file: str
    ) -> Dict[str, BashTool]:
        """Parse tools configuration and create BashTool objects."""
        tools = {}

        if not isinstance(config, dict) or "tools" not in config:
            logger.warning(
                f"Invalid config format in {source_file}: missing 'tools' key"
            )
            return tools

        tools_config = config["tools"]
        if not isinstance(tools_config, dict):
            logger.warning(
                f"Invalid config format in {source_file}: 'tools' must be a dictionary"
            )
            return tools

        for tool_id, tool_config in tools_config.items():
            try:
                tool = await self._parse_tool_config(tool_id, tool_config, source_file)
                if tool:
                    tools[tool_id] = tool
            except ToolConfigurationError:
                raise
            except Exception as e:
                logger.error(
                    f"Failed to parse tool '{tool_id}' from {source_file}",
                    extra={"error": str(e)},
                )

        return tools

    async def _parse_tool_config(
        self, tool_id: str, config: Dict[str, Any], source_file: str
    ) -> Optional[BashTool]:
        """Parse a single tool configuration."""
        try:
            # DEBUG: Parsing tool '{tool_id}' from {source_file}
            # Required fields
            name = config.get("name", tool_id)
            description = config.get("description", "")
            script = config.get("script", "")

            if not script:
                logger.error(f"Tool '{tool_id}' missing script in {source_file}")
                return None

            # Parse and validate parameters
            parameters = []
            for param_config in config.get("parameters", []):
                # Validate parameter configuration
                param_name = param_config.get("name", "")
                if not param_name or not param_name.strip():
                    logger.error(
                        f"Tool '{tool_id}' has parameter with empty name in {source_file}"
                    )
                    return None

                param_type = param_config.get("type", "string")
                valid_types = ["string", "number", "boolean", "object", "array"]
                if param_type not in valid_types:
                    logger.error(
                        f"Tool '{tool_id}' has parameter '{param_name}' with invalid type '{param_type}' in {source_file}"
                    )
                    return None

                param = ToolParameter(
                    name=param_name,
                    type=param_type,
                    required=param_config.get("required", False),
                    description=param_config.get("description", ""),
                    default=param_config.get("default"),
                    enum=param_config.get("enum"),
                    minimum=param_config.get("minimum"),
                    maximum=param_config.get("maximum"),
                    pattern=param_config.get("pattern"),
                )
                parameters.append(param)

            # Parse sandbox configuration
            sandbox_config = config.get("sandbox", {})
            tool_config = ToolConfig(
                requires_confirmation=sandbox_config.get(
                    "requires_confirmation", False
                ),
                confirmation_message=sandbox_config.get("confirmation_message", ""),
                timeout=sandbox_config.get("timeout", 30),
                working_directory=sandbox_config.get("working_directory", "/tmp"),
                use_temp_isolation=sandbox_config.get("use_temp_isolation", False),
                allowed_paths=sandbox_config.get("allowed_paths", []),
                environment_variables=sandbox_config.get("environment_variables"),
                allowed_commands=sandbox_config.get("allowed_commands"),
                caching_enabled=sandbox_config.get("caching_enabled", True),
                cache_ttl_seconds=sandbox_config.get("cache_ttl_seconds", 3600),
                cache_max_size_mb=sandbox_config.get("cache_max_size_mb", 100.0),
                max_memory_mb=sandbox_config.get("max_memory_mb"),
                max_cpu_time_seconds=sandbox_config.get("max_cpu_time_seconds"),
                max_file_size_mb=sandbox_config.get("max_file_size_mb"),
                max_open_files=sandbox_config.get("max_open_files"),
                max_processes=sandbox_config.get("max_processes"),
            )

            # Error mapping configuration (maps process exit codes to MCP errors)
            error_mapping_config = config.get("error_mapping", {})
            parsed_error_mapping: Dict[int, ToolErrorMappingEntry] = {}
            if isinstance(error_mapping_config, dict):
                for exit_code_key, entry in error_mapping_config.items():
                    try:
                        exit_code = int(exit_code_key)
                    except (TypeError, ValueError):
                        logger.error(
                            f"Tool '{tool_id}' has invalid exit code '{exit_code_key}' in error_mapping ({source_file})"
                        )
                        continue

                    mcp_code: Optional[int] = None
                    message: Optional[str] = None
                    retryable: Optional[bool] = None

                    detail: Optional[str] = None

                    if isinstance(entry, dict):
                        mcp_code = entry.get("code")
                        message = entry.get("message")
                        retryable_value = entry.get("retryable")
                        if retryable_value is not None and not isinstance(
                            retryable_value, bool
                        ):
                            logger.error(
                                f"Tool '{tool_id}' has non-boolean retryable for exit code {exit_code} in {source_file}"
                            )
                        else:
                            retryable = retryable_value

                        detail_value = entry.get("detail")
                        if detail_value is not None and not isinstance(
                            detail_value, str
                        ):
                            if self.error_profile is ErrorProfile.ACP_BASIC:
                                raise ToolConfigurationError(
                                    f"Tool '{tool_id}' uses non-string detail for exit code {exit_code} in {source_file} "
                                    "while the acp-basic profile requires detail to be a plain string."
                                )
                        detail = detail_value
                    elif isinstance(entry, (int, float)):
                        mcp_code = int(entry)
                    else:
                        logger.error(
                            f"Tool '{tool_id}' has invalid mapping entry for exit code {exit_code} in {source_file}"
                        )
                        continue

                    if mcp_code is None:
                        logger.error(
                            f"Tool '{tool_id}' missing MCP error code for exit code {exit_code} in {source_file}"
                        )
                        continue

                    parsed_error_mapping[exit_code] = ToolErrorMappingEntry(
                        code=int(mcp_code),
                        message=message,
                        retryable=retryable,
                        detail=detail,
                    )
            elif error_mapping_config:
                logger.error(
                    f"Tool '{tool_id}' error_mapping must be a mapping object in {source_file}"
                )

            # Optional fields
            tags = config.get("tags", [])
            examples = config.get("examples", [])
            version = config.get("version", "1.0.0")
            author = config.get("author", "")

            return BashTool(
                id=tool_id,
                name=name,
                description=description,
                script=script,
                parameters=parameters,
                config=tool_config,
                tags=tags,
                examples=examples,
                version=version,
                author=author,
                error_mapping=parsed_error_mapping,
            )

        except ToolConfigurationError:
            raise
        except Exception as e:
            logger.error(
                f"Error parsing tool '{tool_id}' config", extra={"error": str(e)}
            )
            return None

    def enable_hot_reload(self, enabled: bool = True) -> None:
        """Enable or disable hot reloading of configuration files."""
        self._hot_reload_enabled = enabled

    def add_config_path(self, path: str) -> None:
        """Add a new configuration file path to search."""
        if path not in self.config_paths:
            self.config_paths.append(path)
            self._config_mtimes.pop(path, None)  # Clear cached mtime


# Global tool configuration manager instance
_tool_manager: Optional[ToolConfigurationManager] = None


def get_tool_configuration_manager() -> ToolConfigurationManager:
    """Get the global tool configuration manager instance."""
    global _tool_manager
    if _tool_manager is None:
        from .settings import get_settings

        settings = get_settings()
        _tool_manager = ToolConfigurationManager(error_profile=settings.error_profile)
    return _tool_manager


async def load_available_tools() -> Dict[str, BashTool]:
    """Convenience function to load all available tools."""
    manager = get_tool_configuration_manager()
    return await manager.load_tools()


async def get_tool(tool_id: str) -> Optional[BashTool]:
    """Convenience function to get a specific tool."""
    manager = get_tool_configuration_manager()
    return await manager.get_tool(tool_id)


async def list_tools() -> List[BashTool]:
    """Convenience function to list all available tools."""
    manager = get_tool_configuration_manager()
    return await manager.list_tools()
