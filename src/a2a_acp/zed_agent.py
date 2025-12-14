from __future__ import annotations

import asyncio
import logging
import json
from asyncio import StreamReader, StreamWriter
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Coroutine, Dict, Optional, Sequence
from uuid import uuid4

from .bash_executor import get_bash_executor, ToolExecutionResult
from .sandbox import ExecutionContext
from .tool_config import get_tool
from .error_profiles import ErrorProfile

logger = logging.getLogger(__name__)


class AgentProcessError(RuntimeError):
    """Errors raised when interacting with the agent subprocess."""


class PromptCancelled(AgentProcessError):
    """Raised when the agent acknowledges cancellation."""


NotificationHandler = Callable[[dict], Awaitable[None]]


@dataclass
class ToolPermissionRequest:
    """Information about a tool permission prompt."""

    session_id: str
    tool_call: Dict[str, Any]
    options: Sequence[Dict[str, Any]]


@dataclass
class ToolPermissionDecision:
    """Decision returned by a permission handler."""

    option_id: Optional[str] = None
    future: Optional[asyncio.Future[str]] = None


PermissionHandler = Callable[[ToolPermissionRequest], Awaitable[ToolPermissionDecision]]


class ZedAgentConnection:
    """Manage a single agent subprocess lifecycle."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        api_key: str | None = None,
        log: logging.Logger | None = None,
        permission_handler: PermissionHandler | None = None,
        error_profile: ErrorProfile | None = None,
    ) -> None:
        if not command:
            raise ValueError("Agent command cannot be empty")
        self._command = list(command)
        self._api_key = api_key
        self._logger = log or logger.getChild("ZedAgentConnection")
        self._process: Optional[asyncio.subprocess.Process] = None
        self._stdout: Optional[StreamReader] = None
        self._stdin: Optional[StreamWriter] = None
        self._stderr_buffer: list[str] = []
        self._id_counter = 0
        # Create locks lazily to avoid event loop issues in tests
        self._read_lock: Optional[asyncio.Lock] = None
        self._write_lock: Optional[asyncio.Lock] = None
        self._stderr_task: Optional[asyncio.Task[None]] = None
        self._permission_handler = permission_handler
        self._error_profile = self._resolve_error_profile(error_profile)

    def _ensure_locks(self) -> None:
        """Lazily create locks to avoid event loop issues in tests."""
        if self._read_lock is None:
            self._read_lock = asyncio.Lock()
        if self._write_lock is None:
            self._write_lock = asyncio.Lock()

    async def __aenter__(self) -> "ZedAgentConnection":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    def _resolve_error_profile(self, override: Optional[ErrorProfile]) -> ErrorProfile:
        if override is not None:
            return override

        try:
            executor = get_bash_executor()
            profile = getattr(executor, "error_profile", None)
            if isinstance(profile, ErrorProfile):
                return profile
        except Exception:  # pragma: no cover - defensive guard
            self._logger.debug("Failed to derive error profile from executor", exc_info=True)

        from .settings import get_settings

        try:
            return get_settings().error_profile
        except ValueError:
            self._logger.debug(
                "Settings validation failed while resolving error profile; falling back to acp-basic",
                exc_info=True,
            )
            return ErrorProfile.ACP_BASIC

    async def start(self) -> None:
        """Launch the agent subprocess."""
        if self._process:
            return
        self._logger.debug("Starting agent process", extra={"command": self._command})

        # Set up environment variables
        import os
        env = None
        if self._api_key:
            env = os.environ.copy()
            # Set environment variable based on agent type - this will be updated after authentication
            # For now, set both to be compatible with different agent types
            env["OPENAI_API_KEY"] = self._api_key
            env["GEMINI_API_KEY"] = self._api_key
            self._logger.debug("Setting API key environment variables", extra={"key_length": len(self._api_key)})
        else:
            self._logger.debug("No API key provided for agent authentication")

        # Always copy environment to ensure proper inheritance
        if env is None:
            env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"  # Ensure immediate stdout/stderr flushing

        process = await asyncio.create_subprocess_exec(
            *self._command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        self._process = process
        assert process.stdin and process.stdout
        self._stdin = process.stdin
        self._stdout = process.stdout
        if process.stderr:
            self._stderr_task = asyncio.create_task(self._collect_stderr(process.stderr))

    async def _collect_stderr(self, stream: StreamReader) -> None:
        try:
            while True:
                data = await stream.readline()
                if not data:
                    break
                decoded_line = data.decode().rstrip()
                self._stderr_buffer.append(decoded_line)
                # Log stderr output for debugging (but avoid flooding logs)
                if decoded_line.strip():
                    self._logger.info("Agent stderr output", extra={"stderr_line": decoded_line})
        except asyncio.CancelledError:
            # Handle cancellation gracefully - this is expected during cleanup
            self._logger.debug("Stderr collection cancelled")
            raise
        except Exception:  # pragma: no cover - best effort logging
            self._logger.exception("Error collecting agent stderr")

    async def close(self) -> None:
        """Terminate the subprocess and cleanup resources."""
        if not self._process:
            return
        self._logger.debug("Closing agent process")
        if self._stdin:
            try:
                self._stdin.write_eof()
            except (AttributeError, OSError, RuntimeError):
                pass
            self._stdin.close()
            self._logger.debug("stdin closed")
        if self._stderr_task:
            self._logger.debug("cancelling stderr task")
            self._stderr_task.cancel()
            await asyncio.gather(self._stderr_task, return_exceptions=True)
            self._logger.debug("stderr task cancelled")
        try:
            await asyncio.wait_for(self._process.wait(), timeout=1)
            self._logger.debug("Agent process exited", extra={"returncode": self._process.returncode})
        except asyncio.TimeoutError:
            self._logger.debug("Agent process still running, terminating")
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=2)
                self._logger.debug("Agent process terminated gracefully", extra={"returncode": self._process.returncode})
            except asyncio.TimeoutError:
                self._logger.warning("Agent process did not terminate, killing")
                self._process.kill()
                await self._process.wait()
                self._logger.debug("Agent process killed", extra={"returncode": self._process.returncode})
        self._process = None
        self._stdin = None
        self._stdout = None
        self._stderr_task = None

    async def _write_json(self, payload: dict[str, Any]) -> None:
        if not self._stdin:
            raise AgentProcessError("Agent stdin unavailable")
        self._ensure_locks()
        data = json.dumps(payload)
        async with self._write_lock:
            self._stdin.write(data.encode() + b"\n")
            await self._stdin.drain()
        self._logger.info("Sent JSON-RPC message to agent", extra={
            "method": payload.get("method"),
            "id": payload.get("id"),
            "payload": payload,
            "raw_message": data
        })

    async def _read_json(self) -> dict[str, Any]:
        if not self._stdout:
            raise AgentProcessError("Agent stdout unavailable")

        self._ensure_locks()

        # Read lines until we find valid JSON
        while True:
            async with self._read_lock:
                raw = await self._stdout.readline()
            if not raw:
                stderr = self.stderr()
                message = "Agent process closed stdout unexpectedly"
                if stderr:
                    message = f"{message}. stderr: {stderr}"
                else:
                    message = f"{message}. No stderr output available."
                raise AgentProcessError(message)

            decoded = raw.decode().strip()
            # self._logger.debug("Raw agent output", extra={"raw_output": repr(decoded), "length": len(decoded)})

            if not decoded:
                continue  # Skip empty lines

            # Skip log lines (they contain ANSI color codes and don't start with '{')
            if not decoded.startswith('{'):
                # self._logger.debug("Skipping non-JSON line", extra={"line": repr(decoded)})
                continue

            try:
                payload = json.loads(decoded)
                self._logger.info("Received JSON-RPC message from agent", extra={
                    "method": payload.get("method"),
                    "id": payload.get("id"),
                    "has_result": "result" in payload,
                    "has_error": "error" in payload,
                    "result": payload.get("result"),
                    "error": payload.get("error"),
                    "full_payload": payload
                })
                return payload
            except json.JSONDecodeError as e:
                # If it's not valid JSON, skip it and continue reading
                self._logger.debug("Skipping invalid JSON", extra={"line": repr(decoded), "error": str(e)})
                continue

    def _next_id(self) -> int:
        self._id_counter += 1
        return self._id_counter

    async def request(
        self,
        method: str,
        params: Optional[dict[str, Any]] = None,
        *,
        handler: NotificationHandler | None = None,
    ) -> dict[str, Any] | None:
        """Send a JSON-RPC request and wait for its response."""
        request_id = self._next_id()
        message = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            message["params"] = params
        await self._write_json(message)

        while True:
            payload = await self._read_json()
            if payload.get("id") == request_id:
                if "error" in payload:
                    raise AgentProcessError(payload["error"])
                return payload.get("result")
            if handler:
                await handler(payload)

    async def notify(self, method: str, params: Optional[dict[str, Any]] = None) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        self._logger.debug("Sending notification", extra={"method": method, "params": params})
        await self._write_json(message)

    async def initialize(self) -> dict[str, Any] | None:
        """Send initialize request."""
        params = {"protocolVersion": 1, "clientName": "cli", "clientCapabilities": {      
            "fs": {
                "readTextFile": True,
                "writeTextFile": True
            },
            # "terminal": True.    # Future capability flags can be added here
            }}
        self._logger.info("=== INITIALIZE PHASE ===")
        self._logger.debug("Sending initialize request", extra={"params": params})
        try:
            result = await self.request("initialize", params)
            self._logger.info("Initialize response received", extra={"result": result})

            # Check if authentication is required
            auth_methods = result.get("authMethods", []) if result else []
            if auth_methods:
                self._logger.info("Authentication required", extra={"auth_methods": [m.get("id") for m in auth_methods]})

                # Look for API key authentication method (support multiple agent types)
                api_key_method = None
                api_key_method_id = None
                supported_api_key_methods = ["apikey", "gemini-api-key", "codex-api-key", "openai-api-key"]  # Support codex, gemini and openai

                for method in auth_methods:
                    method_id = method.get("id")
                    if method_id in supported_api_key_methods:
                        api_key_method = method
                        api_key_method_id = method_id
                        break

                if api_key_method and self._api_key and api_key_method_id:
                    self._logger.info("Using API key authentication", extra={"method_id": api_key_method_id})
                    await self.authenticate(api_key_method_id, self._api_key)
                elif api_key_method:
                    # Try chatgpt method as fallback when no API key is provided
                    chatgpt_method = next((m for m in auth_methods if m.get("id") == "chatgpt"), None)
                    if chatgpt_method:
                        self._logger.info("Using ChatGPT authentication with existing auth.json", extra={"method_id": "chatgpt"})
                        await self.authenticate("chatgpt")
                    else:
                        raise AgentProcessError("Agent requires API key authentication but no API key provided and no chatgpt method available")
                else:
                    # No API key method available, try chatgpt method if available
                    chatgpt_method = next((m for m in auth_methods if m.get("id") == "chatgpt"), None)
                    if chatgpt_method:
                        self._logger.info("Using ChatGPT authentication with existing auth.json", extra={"method_id": "chatgpt"})
                        await self.authenticate("chatgpt")
                    else:
                        raise AgentProcessError(f"Agent requires authentication but no supported method found. Available: {[m.get('id') for m in auth_methods]}")
            else:
                self._logger.debug("No authentication required")

            return result
        except Exception as e:
            self._logger.error("Initialize request failed", extra={"error": str(e)})
            raise

    async def authenticate(self, method_id: str, api_key: str | None = None) -> dict[str, Any] | None:
        """Send authenticate request."""
        params = {"methodId": method_id}
        # self._logger.info("=== AUTHENTICATION PHASE ===")
        self._logger.debug("Sending authenticate request", extra={"method_id": method_id, "has_api_key": api_key is not None})
        try:
            result = await self.request("authenticate", params)
            self._logger.info("Authentication completed successfully", extra={"method_id": method_id})
            return result
        except Exception as e:
            self._logger.error("Authentication failed", extra={"error": str(e), "method_id": method_id})
            raise

    async def start_session(self, cwd: str, mcp_servers: list[dict[str, Any]] | None = None) -> str:
        """Create a new session and return its identifier."""
        params = {
            "cwd": cwd,
            "mcpServers": mcp_servers or []
        }
        # self._logger.info("=== SESSION CREATION PHASE ===")
        self._logger.debug("Sending session/new request", extra={"params": params})
        try:
            result = await self.request("session/new", params)
            if not result or "sessionId" not in result:
                raise AgentProcessError("session/new missing sessionId")
            session_id = str(result["sessionId"])
            self._logger.info("Session created successfully", extra={"session_id": session_id, "result": result})
            return session_id
        except Exception as e:
            self._logger.error("Session creation failed", extra={"error": str(e)})
            raise

    async def load_session(self, session_id: str, cwd: str, mcp_servers: list[dict[str, Any]] | None = None) -> None:
        """Load an existing session by ID."""
        params = {
            "sessionId": session_id,
            "cwd": cwd,
            "mcpServers": mcp_servers or []
        }
        # self._logger.info("=== SESSION LOADING PHASE ===")
        self._logger.debug("Sending session/load request", extra={"session_id": session_id, "params": params})

        # Handle the session/load response
        # The agent will stream conversation history via session/update notifications
        async def load_handler(payload: dict[str, Any]) -> None:
            self._logger.debug("Received notification during session load", extra={
                "method": payload.get("method"),
                "payload_keys": list(payload.keys())
            })

            if payload.get("method") == "session/update":
                params = payload.get("params", {})
                update_data = params.get("update", {})
                event = update_data.get("sessionUpdate")
                self._logger.debug("Session load update", extra={
                    "event": event,
                    "update_keys": list(update_data.keys())
                })
                # TODO: History replay notifications will be sent here

        try:
            result = await self.request("session/load", params, handler=load_handler)
            self._logger.info("Session loaded successfully", extra={
                "session_id": session_id,
                "result": result
            })
        except Exception as e:
            self._logger.error("Session loading failed", extra={
                "session_id": session_id,
                "error": str(e)
            })
            raise

    async def prompt(
        self,
        session_id: str,
        prompt: list[dict[str, Any]],
        *,
        on_chunk: Callable[[str], Coroutine[Any, Any, None]] | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> dict[str, Any]:
        """Send a session/prompt request and return the final result."""
        # self._logger.info("=== PROMPT PHASE ===")
        self._logger.debug("Sending session/prompt request", extra={
            "session_id": session_id,
            "prompt_length": len(prompt),
            "has_cancel_event": cancel_event is not None
        })

        # Accumulate chunks for the final result
        accumulated_chunks: list[str] = []
        executed_tool_calls: list[dict[str, Any]] = []

        def _record_tool_call_update(update: dict[str, Any]) -> None:
            """Normalize and store tool_call_update payloads for the final response."""
            cleaned_update = {
                key: value for key, value in update.items()
                if key != "sessionUpdate"
            }
            executed_tool_calls.append(cleaned_update)
            self._logger.info(
                "Recorded tool_call_update notification",
                extra={
                    "tool_call_id": cleaned_update.get("toolCallId"),
                    "status": cleaned_update.get("status"),
                    "title": cleaned_update.get("title"),
                },
            )

        def _merge_tool_call_results(response: dict[str, Any] | None) -> dict[str, Any]:
            """Merge collected tool call updates into the response."""
            if response is None:
                response = {}

            if not executed_tool_calls:
                return response

            existing_calls = response.get("toolCalls")
            merged_calls: list[dict[str, Any]] = []
            if isinstance(existing_calls, list):
                for call in existing_calls:
                    if isinstance(call, dict):
                        merged_calls.append(call)
                    else:
                        merged_calls.append({"raw": call})
            merged_calls.extend(executed_tool_calls)
            response["toolCalls"] = merged_calls
            return response

        async def handler(payload: dict[str, Any]) -> None:
            self._logger.debug("Handling notification during prompt", extra={
                "method": payload.get("method"),
                "payload_keys": list(payload.keys())
            })

            # Handle agent-initiated requests (e.g., fs/read_text_file)
            if payload.get("id") is not None and payload.get("method"):
                handled = await self._handle_agent_request(payload, session_id)
                if handled:
                    return

            if payload.get("method") == "session/update":
                params = payload.get("params", {})
                update_data = params.get("update", {})
                event = update_data.get("sessionUpdate")
                self._logger.debug("Received session/update notification", extra={
                    "event": event,
                    "params_keys": list(params.keys()),
                    "update_keys": list(update_data.keys())
                })

                if event == "agent_message_chunk":
                    # ZedACP protocol: agent_message_chunk is in params.update.content
                    update_data = params.get("update", {})
                    content = update_data.get("content", {})
                    text = content.get("text")
                    self._logger.debug("Received agent_message_chunk", extra={
                        "has_text": text is not None,
                        "text_length": len(text) if text else 0,
                        "text_preview": text[:100] + "..." if text and len(text) > 100 else (text or "None"),
                        "content_keys": list(content.keys()),
                        "update_keys": list(update_data.keys())
                    })
                    if text:
                        # Accumulate chunk for final result
                        accumulated_chunks.append(text)
                        self._logger.debug("Accumulated chunk", extra={
                            "chunk_length": len(text),
                            "total_chunks": len(accumulated_chunks),
                            "text_preview": text[:50] + "..." if len(text) > 50 else text
                        })

                        # Also call external chunk handler if provided
                        if on_chunk:
                            self._logger.debug("Processing agent message chunk", extra={
                                "text_length": len(text),
                                "text_preview": text[:100] + "..." if len(text) > 100 else text
                            })
                            await on_chunk(text)
                    elif on_chunk:
                        self._logger.warning("Received agent_message_chunk but no text content or no on_chunk handler", extra={
                            "has_text": text is not None,
                            "has_on_chunk": on_chunk is not None
                        })

                elif event == "input_required":
                    # ZedACP input_required notification - agent needs user input
                    update_data = params.get("update", {})
                    self._logger.info("Agent requested input", extra={
                        "event": event,
                        "update_data": update_data
                    })

                    # Extract input requirement details
                    input_required_info = update_data.get("inputRequired", {})
                    text = input_required_info.get("text", "Additional input required")
                    input_types = input_required_info.get("inputTypes", ["text/plain"])

                    # Create enhanced input required message with structured data
                    input_required_message = f"INPUT_REQUIRED: {text}"
                    if input_types:
                        input_required_message += f"\nINPUT_TYPES: {','.join(input_types)}"

                    if on_chunk:
                        await on_chunk(input_required_message)

                    # Also accumulate input required message for consistency
                    accumulated_chunks.append(input_required_message)

                elif event == "tool_call_update":
                    self._logger.debug("Received tool_call_update session/update", extra={
                        "update_keys": list(update_data.keys())
                    })
                    if isinstance(update_data, dict):
                        _record_tool_call_update(update_data)

            # Handle tool call updates (ZedACP tool execution results)
            elif payload.get("method") == "tool_call_update":
                self._logger.debug("Received tool_call_update notification", extra={
                    "payload_keys": list(payload.keys())
                })

                # Tool call updates indicate tool execution results from the agent
                # These are typically sent when the agent processes tool calls
                update_data = payload.get("params", {})
                if isinstance(update_data, dict):
                    _record_tool_call_update(update_data)
            elif payload.get("method") == "session/cancelled":
                self._logger.warning("Agent reported direct cancellation")
                raise PromptCancelled("Agent reported cancellation")

        # If a cancel_event is provided, also check for external cancellation
        if cancel_event:
            # logger.info("Setting up external cancellation check", extra={"session_id": session_id, "cancel_event_is_set": cancel_event.is_set()})
            async def check_external_cancellation():
                # logger.info("Waiting for external cancellation", extra={"session_id": session_id})
                await cancel_event.wait()
                logger.info("External cancellation detected", extra={"session_id": session_id})
                # Send cancellation to the agent when external cancellation is requested
                try:
                    await self.cancel(session_id)
                except Exception as e:
                    logger.warning("Failed to send cancellation to agent", extra={"error": str(e)})
                raise PromptCancelled("External cancellation requested")

            cancel_task = asyncio.create_task(check_external_cancellation())
            prompt_task = asyncio.create_task(
                self.request(
                    "session/prompt",
                    {"sessionId": session_id, "prompt": prompt},
                    handler=handler,
                )
            )

            done, pending = await asyncio.wait(
                {cancel_task, prompt_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Cancel the other task
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

            if cancel_task in done:
                # External cancellation was requested
                logger.info("External cancellation completed first", extra={"session_id": session_id})
                self._logger.info("External cancellation detected during prompt processing", extra={"session_id": session_id})
                raise PromptCancelled("External cancellation requested")
            else:
                # Prompt completed normally
                logger.info("Prompt completed normally", extra={"session_id": session_id})
                result = prompt_task.result()

                # Process any tool calls in the result
                if result:
                    result = await self._process_zedacp_tool_calls(result, session_id)
                result = _merge_tool_call_results(result)

                # Add accumulated chunks to the result
                if accumulated_chunks:
                    full_text = "".join(accumulated_chunks)
                    if not result:
                        result = {}
                    if "result" not in result:
                        result["result"] = {}
                    result["result"]["text"] = full_text
                    self._logger.info("Added accumulated chunks to result", extra={
                        "chunks": len(accumulated_chunks),
                        "total_length": len(full_text),
                        "result_keys": list(result.keys()),
                        "text_preview": full_text[:100] + "..." if len(full_text) > 100 else full_text
                    })
                else:
                    self._logger.info("No chunks accumulated, final result", extra={
                        "result": result,
                        "result_keys": list(result.keys()) if result else "empty"
                    })

                self._logger.info("Prompt processing completed successfully", extra={
                    "session_id": session_id,
                    "has_result": result is not None,
                    "result_keys": list(result.keys()) if result else [],
                    "chunks_accumulated": len(accumulated_chunks)
                })
                return result or {}
        else:
            result = await self.request(
                "session/prompt",
                {"sessionId": session_id, "prompt": prompt},
                handler=handler,
            )

            # Process any tool calls in the result
            if result:
                result = await self._process_zedacp_tool_calls(result, session_id)
            result = _merge_tool_call_results(result)

            # Add accumulated chunks to the result
            if accumulated_chunks:
                full_text = "".join(accumulated_chunks)
                if not result:
                    result = {}
                if "result" not in result:
                    result["result"] = {}
                result["result"]["text"] = full_text
                self._logger.info("Added accumulated chunks to result", extra={
                    "chunks": len(accumulated_chunks),
                    "total_length": len(full_text),
                    "result_keys": list(result.keys()),
                    "text_preview": full_text[:100] + "..." if len(full_text) > 100 else full_text
                })
            else:
                self._logger.info("No chunks accumulated, final result", extra={
                    "result": result,
                    "result_keys": list(result.keys()) if result else "empty"
                })

            return result or {}

    async def cancel(self, session_id: str | None = None) -> None:
        """Send cancellation request to the agent."""
        params: dict[str, Any] | None = None
        if session_id:
            params = {"sessionId": session_id}
        await self.notify("session/cancel", params)

    def stderr(self) -> str:
        """Return aggregated stderr output."""
        return "\n".join(self._stderr_buffer)

    async def _handle_agent_request(self, payload: dict[str, Any], session_id: str) -> bool:
        """Handle JSON-RPC requests initiated by the agent."""
        method = payload.get("method")

        if method == "fs/read_text_file":
            await self._handle_fs_read_text_file(payload, session_id)
            return True

        elif method == "fs/write_text_file":
            await self._handle_fs_write_text_file(payload, session_id)
            return True

        elif method == "session/request_permission":
            await self._handle_session_request_permission(payload, session_id)
            return True

        return False

    async def _handle_fs_read_text_file(self, payload: dict[str, Any], session_id: str) -> None:
        """Execute the filesystem read request via the bash tool."""
        request_id = payload.get("id")
        params = payload.get("params") or {}

        if request_id is None:
            self._logger.warning("fs/read_text_file request missing id", extra={"payload": payload})
            return

        self._logger.info("Handling fs/read_text_file request", extra={
            "request_id": request_id,
            "path": params.get("path"),
            "line": params.get("line"),
            "limit": params.get("limit")
        })

        path = params.get("path")
        if not path:
            await self._write_json({
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32602,
                    "message": "Missing required parameter: path"
                }
            })
            return

        # Resolve tool configuration
        tool = await get_tool("functions.acp_fs__read_text_file")
        if not tool:
            await self._write_json({
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32001,
                    "message": "Tool functions.acp_fs__read_text_file is not available"
                }
            })
            self._logger.warning("fs/read_text_file tool not available, check configuration", extra={"request_id": request_id})
            return

        tool_params: dict[str, Any] = {}
        tool_params["path"] = path

        # Optional parameters with validation
        if "line" in params and params["line"] is not None:
            try:
                tool_params["line"] = int(params["line"])
            except (TypeError, ValueError):
                await self._write_json({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32602,
                        "message": "Invalid line parameter; must be an integer"
                    }
                })
                self._logger.warning("fs/read_text_file invalid line parameter", extra={"request_id": request_id, "line": params["line"]})
                return

        if "limit" in params and params["limit"] is not None:
            try:
                tool_params["limit"] = int(params["limit"])
            except (TypeError, ValueError):
                await self._write_json({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32602,
                        "message": "Invalid limit parameter; must be an integer"
                    }
                })
                self._logger.warning("fs/read_text_file invalid limit parameter", extra={"request_id": request_id, "limit": params["limit"]})
                return

        tool_call = {
            "id": f"fs_read_text_file_{uuid4().hex}",
            "toolId": tool.id,
            "parameters": tool_params,
            "metadata": {"origin": "fs/read_text_file"}
        }

        permission_options = [
            {"optionId": "allow", "name": "Allow", "kind": "allow_once"},
            {"optionId": "deny", "name": "Deny", "kind": "reject_once"},
        ]

        option_id = await self._resolve_permission_option(
            session_id,
            tool_call,
            permission_options,
            fallback_to_agent=False,
        )

        if not self._is_option_allowed(option_id, permission_options):
            await self._write_json({
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32003,
                    "message": "fs/read_text_file denied by governance policy"
                }
            })
            self._logger.warning("fs/read_text_file permission denied", extra={
                "request_id": request_id,
                "path": path,
                "option_id": option_id
            })
            return

        executor = get_bash_executor()
        exec_session_id = params.get("sessionId") or session_id
        context = ExecutionContext(
            tool_id=tool.id,
            session_id=exec_session_id,
            task_id=f"fs_read_text_file_{uuid4().hex}",
            user_id="zedacp_user"
        )

        result = await executor.execute_tool(tool, tool_params, context)

        if result.success:
            await self._write_json({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": result.output
                }
            })
            self._logger.info("fs/read_text_file completed", extra={
                "request_id": request_id,
                "path": path,
                "line": tool_params.get("line"),
                "limit": tool_params.get("limit"),
                "output_length": len(result.output) if result.output else 0
            })
        else:
            error_payload = self._jsonrpc_error_from_tool_result(result)
            await self._write_json({
                "jsonrpc": "2.0",
                "id": request_id,
                "error": error_payload
            })
            self._logger.error("fs/read_text_file failed", extra={
                "request_id": request_id,
                "path": path,
                "error": result.error,
                "return_code": result.return_code,
                "mcp_error": error_payload
            })

    async def _handle_fs_write_text_file(self, payload: dict[str, Any], session_id: str) -> None:
        """Execute the Codex filesystem write request via the bash tool."""
        request_id = payload.get("id")
        params = payload.get("params") or {}

        if request_id is None:
            self._logger.warning("fs/write_text_file request missing id", extra={"payload": payload})
            return

        self._logger.info("Handling fs/write_text_file request", extra={
            "request_id": request_id,
            "path": params.get("path"),
            "content_length": len(params.get("content", "")) if params.get("content") else 0
        })

        tool = await get_tool("functions.acp_fs__write_text_file")
        if not tool:
            await self._write_json({
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32001,
                    "message": "Tool functions.acp_fs__write_text_file is not available"
                }
            })
            self._logger.warning("fs/write_text_file tool not available, check configuration", extra={"request_id": request_id})
            return

        path = params.get("path")
        content = params.get("content")
        if not path or content is None:
            await self._write_json({
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32602,
                    "message": "Missing required parameters: path and content"
                }
            })
            return

        tool_params: dict[str, Any] = {"path": path, "content": content}

        tool_call = {
            "id": f"fs_write_text_file_{uuid4().hex}",
            "toolId": tool.id,
            "parameters": tool_params,
            "metadata": {"origin": "fs/write_text_file"}
        }

        permission_options = [
            {"optionId": "allow", "name": "Allow", "kind": "allow_once"},
            {"optionId": "deny", "name": "Deny", "kind": "reject_once"},
        ]

        option_id = await self._resolve_permission_option(
            session_id,
            tool_call,
            permission_options,
            fallback_to_agent=False,
        )

        if not self._is_option_allowed(option_id, permission_options):
            await self._write_json({
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32003,
                    "message": "fs/write_text_file denied by governance policy"
                }
            })
            self._logger.warning("fs/write_text_file permission denied", extra={
                "request_id": request_id,
                "path": path,
                "option_id": option_id
            })
            return

        executor = get_bash_executor()
        exec_session_id = params.get("sessionId") or session_id
        context = ExecutionContext(
            tool_id=tool.id,
            session_id=exec_session_id,
            task_id=f"fs_write_text_file_{uuid4().hex}",
            user_id="zedacp_user"
        )

        result = await executor.execute_tool(tool, tool_params, context)

        if result.success:
            await self._write_json({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": result.output
                }
            })
            self._logger.info("fs/write_text_file completed", extra={
                "request_id": request_id,
                "path": path,
                "output_length": len(result.output) if result.output else 0
            })
        else:
            error_payload = self._jsonrpc_error_from_tool_result(result)
            await self._write_json({
                "jsonrpc": "2.0",
                "id": request_id,
                "error": error_payload
            })
            self._logger.error("fs/write_text_file failed", extra={
                "request_id": request_id,
                "path": path,
                "error": result.error,
                "return_code": result.return_code,
                "mcp_error": error_payload
            })

    async def _handle_session_request_permission(self, payload: dict[str, Any], session_id: str) -> None:
        """Handle a permission request initiated by the agent."""
        request_id = payload.get("id")
        params = payload.get("params") or {}
        options: list[dict[str, Any]] = params.get("options") or []
        tool_call = params.get("toolCall") or {}
        session_ref = params.get("sessionId") or session_id

        if request_id is None:
            self._logger.warning("session/request_permission request missing id", extra={"payload": payload})
            return

        self._logger.info(
            "Handling session/request_permission",
            extra={
                "session_id": session_ref,
                "tool_call_id": tool_call.get("toolCallId"),
                "tool_id": tool_call.get("toolId"),
                "option_count": len(options),
            },
        )

        option_id = await self._resolve_permission_option(
            session_ref,
            tool_call,
            options,
            fallback_to_agent=False,
        )

        if not self._is_option_allowed(option_id, options):
            option_id = next(
                (
                    opt.get("optionId")
                    for opt in options
                    if (opt.get("kind") or "").startswith("allow")
                ),
                options[0].get("optionId") if options else None,
            )

        outcome = {"optionId": option_id}

        await self._write_json({
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"outcome": outcome},
        })

        self._logger.info(
            "Responded to session/request_permission",
            extra={
                "session_id": session_ref,
                "tool_call_id": tool_call.get("toolCallId"),
                "selected_option": option_id,
                "options_returned": len(options),
            },
        )

    async def _resolve_permission_option(
        self,
        session_id: str,
        tool_call: Dict[str, Any],
        options: list[dict[str, Any]],
        *,
        fallback_to_agent: bool,
    ) -> Optional[str]:
        """Resolve a permission decision using the configured handler."""
        option_id: Optional[str] = None

        if self._permission_handler:
            try:
                decision = await self._permission_handler(
                    ToolPermissionRequest(
                        session_id=session_id,
                        tool_call=tool_call,
                        options=options,
                    )
                )
            except Exception as exc:
                self._logger.error(
                    "Permission handler failed",
                    extra={
                        "tool_call_id": tool_call.get("id"),
                        "tool_id": tool_call.get("toolId"),
                        "error": str(exc),
                    },
                )
                decision = ToolPermissionDecision(option_id=None)

            if decision.future:
                option_id = await decision.future
            else:
                option_id = decision.option_id
        else:
            option_id = next(
                (
                    opt.get("optionId")
                    for opt in options
                    if (opt.get("kind") or "").startswith("allow")
                ),
                options[0].get("optionId") if options else None,
            )

        if option_id is None and fallback_to_agent:
            response = await self.request(
                "session/request_permission",
                {
                    "sessionId": session_id,
                    "toolCall": tool_call,
                    "options": options,
                },
            )
            outcome = response.get("outcome", {})
            option_id = outcome.get("optionId")

        return option_id

    @staticmethod
    def _is_option_allowed(option_id: Optional[str], options: Sequence[Dict[str, Any]]) -> bool:
        if not option_id:
            return False
        selected_option = next((opt for opt in options if opt.get("optionId") == option_id), None)
        if not selected_option:
            return False
        kind = selected_option.get("kind", "") or ""
        if kind:
            return kind.startswith("allow")
        return option_id in {"allow", "approved"}

    @staticmethod
    def _extract_mcp_error(result: ToolExecutionResult) -> Optional[Dict[str, Any]]:
        """Extract the MCP error payload from a tool result, if present."""
        if result.mcp_error:
            return result.mcp_error
        metadata = result.metadata or {}
        mcp_error = metadata.get("mcp_error")
        if isinstance(mcp_error, dict) and mcp_error:
            return mcp_error
        return None

    def _jsonrpc_error_from_tool_result(
        self,
        result: ToolExecutionResult,
        *,
        default_code: int = -32603,
        default_message: str = "Tool execution failed"
    ) -> Dict[str, Any]:
        """Convert a failed tool execution into a JSON-RPC error payload."""
        mcp_error = ZedAgentConnection._extract_mcp_error(result)
        base_message = result.error or result.output or default_message
        profile = self._error_profile
        if mcp_error:
            error_payload: Dict[str, Any] = {
                "code": mcp_error.get("code", default_code),
                "message": mcp_error.get("message") or base_message,
            }
            retryable = mcp_error.get("retryable")
            detail_value = mcp_error.get("detail")
            diagnostics = (result.metadata or {}).get("a2a_diagnostics")

            if profile is ErrorProfile.ACP_BASIC:
                if detail_value is not None:
                    error_payload["data"] = detail_value if isinstance(detail_value, str) else str(detail_value)
            else:
                data_block: Dict[str, Any] = {"return_code": result.return_code}
                if detail_value is not None:
                    data_block["detail"] = detail_value
                if retryable is not None:
                    data_block["retryable"] = retryable
                if diagnostics:
                    data_block["diagnostics"] = diagnostics
                error_payload["data"] = data_block
            return error_payload

        if profile is ErrorProfile.ACP_BASIC:
            return {
                "code": default_code,
                "message": base_message,
            }

        return {
            "code": default_code,
            "message": base_message,
            "data": {"return_code": result.return_code}
        }

    async def _process_zedacp_tool_calls(self, response: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Process ZedACP tool calls in agent response.

        ZedACP tool calls come from agent responses in session/prompt.
        Agent sends: session/update with tool_call updates
        We need to: execute tools and return results in ZedACP format

        Args:
            response: ZedACP response that may contain tool calls
            session_id: Current session ID

        Returns:
            Modified response with tool call results
        """
        if not response:
            return response

        # Check if response contains tool calls
        if "toolCalls" not in response:
            return response

        tool_calls = response["toolCalls"]
        if not tool_calls or not isinstance(tool_calls, list):
            return response

        logger.info(f"Processing {len(tool_calls)} ZedACP tool calls", extra={
            "session_id": session_id,
            "tool_call_ids": [call.get("id") for call in tool_calls]
        })

        # Execute each tool call
        executed_calls = []
        for tool_call in tool_calls:
            try:
                result = await self._execute_zedacp_tool_call(tool_call, session_id)
                executed_calls.append(result)
            except Exception as e:
                logger.error(f"Failed to execute tool call: {tool_call.get('id')}", extra={
                    "error": str(e),
                    "tool_call": tool_call
                })

                # Return error result in ZedACP format
                executed_calls.append({
                    "toolCallId": tool_call.get("id"),
                    "status": "failed",
                    "content": [{"type": "text", "text": f"Tool execution failed: {str(e)}"}],
                    "rawOutput": str(e)
                })

        # Replace toolCalls with executed results
        response["toolCalls"] = executed_calls

        logger.info(f"Processed {len(executed_calls)} tool call results", extra={
            "session_id": session_id,
            "successful": len([r for r in executed_calls if r.get("status") == "completed"])
        })

        return response

    async def _execute_zedacp_tool_call(self, tool_call: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Execute a single ZedACP tool call.

        Args:
            tool_call: ZedACP tool call object
            session_id: Current session ID

        Returns:
            ZedACP-compatible tool call result
        """
        tool_call_id = tool_call.get("id")
        tool_id = tool_call.get("toolId")

        if not tool_id:
            raise ValueError("Tool call missing toolId")

        logger.debug(f"Executing ZedACP tool call: {tool_call_id}", extra={
            "tool_call_id": tool_call_id,
            "tool_id": tool_id,
            "session_id": session_id
        })

        # Get tool configuration
        tool = await get_tool(tool_id)
        if not tool:
            raise ValueError(f"Unknown tool: {tool_id}")

        # Extract parameters from tool call
        parameters = tool_call.get("parameters", {})

        # Check if tool requires confirmation
        if tool.config.requires_confirmation:
            confirmed = await self._request_tool_permission(tool_call, session_id)
            if not confirmed:
                return {
                    "toolCallId": tool_call_id,
                    "status": "cancelled",
                    "content": [{"type": "text", "text": "Tool execution cancelled by user"}],
                    "rawOutput": "Tool execution cancelled by user"
                }

        # Create execution context
        context = ExecutionContext(
            tool_id=tool_id,
            session_id=session_id,
            task_id=f"zedacp_{tool_call_id}",
            user_id="zedacp_user"
        )

        # Execute the tool
        executor = get_bash_executor()
        result = await executor.execute_tool(tool, parameters, context)

        # Convert result to ZedACP format
        status = "completed" if result.success else "failed"
        mcp_error = self._extract_mcp_error(result)
        failure_text = result.error or (mcp_error.get("message") if mcp_error else "Tool execution failed")
        content_text = result.output if result.success else failure_text
        raw_output = result.output if result.output else (result.error or failure_text)

        zedacp_result: Dict[str, Any] = {
            "toolCallId": tool_call_id,
            "status": status,
            "content": [{"type": "text", "text": content_text}],
            "rawOutput": raw_output
        }

        if mcp_error:
            zedacp_result["error"] = mcp_error

        diagnostics = (result.metadata or {}).get("a2a_diagnostics")
        metadata_payload: Dict[str, Any] = {
            "execution_time": result.execution_time,
            "return_code": result.return_code,
        }
        retryable_flag = mcp_error.get("retryable") if mcp_error else None
        if isinstance(retryable_flag, bool):
            metadata_payload["retryable"] = retryable_flag
        if result.output_files:
            metadata_payload["output_files"] = result.output_files
        if mcp_error and self._error_profile is ErrorProfile.EXTENDED_JSON:
            metadata_payload["mcp_error"] = mcp_error

        if result.metadata:
            for key, value in result.metadata.items():
                if key in {"a2a_diagnostics", "mcp_error"}:
                    continue
                if key in metadata_payload:
                    continue
                if self._error_profile is ErrorProfile.ACP_BASIC and isinstance(value, (dict, list)):
                    continue
                metadata_payload[key] = value

        if diagnostics:
            meta_payload = zedacp_result.setdefault("meta", {})
            meta_payload["a2a_diagnostics"] = diagnostics
            if self._error_profile is ErrorProfile.ACP_BASIC:
                diagnostics_text = json.dumps(diagnostics, separators=(",", ":"), default=str)
                zedacp_result["content"].append({"type": "text", "text": f"Diagnostics: {diagnostics_text}"})

        if metadata_payload:
            zedacp_result["metadata"] = metadata_payload

        logger.debug(f"ZedACP tool call completed: {tool_call_id}", extra={
            "tool_call_id": tool_call_id,
            "status": status,
            "execution_time": result.execution_time
        })

        return zedacp_result

    async def _request_tool_permission(self, tool_call: Dict[str, Any], session_id: str) -> bool:
        """Request tool execution permission from ZedACP agent.

        Args:
            tool_call: The tool call requiring permission
            session_id: Current session ID

        Returns:
            True if permission granted, False otherwise
        """
        tool_id = tool_call.get("toolId")
        tool_call_id = tool_call.get("id")

        logger.info(f"Requesting tool permission: {tool_id}", extra={
            "tool_id": tool_id,
            "tool_call_id": tool_call_id,
            "session_id": session_id
        })

        try:
            options: list[dict[str, Any]] = [
                {"optionId": "proceed_always", "name": "Allow All Edits", "kind": "allow_always"},
                {"optionId": "proceed_once", "name": "Allow", "kind": "allow_once"},
                {"optionId": "cancel", "name": "Reject", "kind": "reject_once"},
            ]

            option_id = await self._resolve_permission_option(
                session_id,
                tool_call,
                options,
                fallback_to_agent=True,
            )

            permitted = self._is_option_allowed(option_id, options)

            logger.info(
                "Tool permission decision",
                extra={
                    "tool_id": tool_id,
                    "tool_call_id": tool_call_id,
                    "selected_option": option_id,
                    "permitted": permitted,
                    "session_id": session_id,
                },
            )

            return permitted

        except Exception as e:
            logger.error(f"Permission request failed for tool: {tool_id}", extra={
                "tool_id": tool_id,
                "tool_call_id": tool_call_id,
                "error": str(e),
                "session_id": session_id
            })
            # Default to deny on error
            return False

    async def execute_bash_tool(
        self,
        tool_call: Dict[str, Any],
        session_id: str
    ) -> ToolExecutionResult:
        """Legacy method for executing bash tools (for backward compatibility).

        Args:
            tool_call: Tool call information
            session_id: Current session ID

        Returns:
            Tool execution result
        """
        tool_id = tool_call.get("toolId")
        if not tool_id:
            raise ValueError("Tool call missing toolId")

        # Get tool configuration
        tool = await get_tool(tool_id)
        if not tool:
            raise ValueError(f"Unknown tool: {tool_id}")

        # Extract parameters
        parameters = tool_call.get("parameters", {})

        # Create execution context
        context = ExecutionContext(
            tool_id=tool_id,
            session_id=session_id,
            task_id=f"legacy_{tool_call.get('id', 'unknown')}",
            user_id="legacy_user"
        )

        # Execute using bash executor
        executor = get_bash_executor()
        return await executor.execute_tool(tool, parameters, context)
