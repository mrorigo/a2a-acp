from __future__ import annotations

import asyncio
import json
import logging
from asyncio import StreamReader, StreamWriter
from typing import Any, Awaitable, Callable, Coroutine, Optional, Sequence


logger = logging.getLogger(__name__)


class AgentProcessError(RuntimeError):
    """Errors raised when interacting with the agent subprocess."""


class PromptCancelled(AgentProcessError):
    """Raised when the agent acknowledges cancellation."""


NotificationHandler = Callable[[dict], Awaitable[None]]


class ZedAgentConnection:
    """Manage a single agent subprocess lifecycle."""

    def __init__(self, command: Sequence[str], *, api_key: str | None = None, log: logging.Logger | None = None) -> None:
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
        self._read_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        self._stderr_task: Optional[asyncio.Task[None]] = None

    async def __aenter__(self) -> "ZedAgentConnection":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def start(self) -> None:
        """Launch the agent subprocess."""
        if self._process:
            return
        self._logger.debug("Starting agent process", extra={"command": self._command})

        # Set up environment variables
        env = None
        if self._api_key:
            import os
            env = os.environ.copy()
            env["OPENAI_API_KEY"] = self._api_key
            self._logger.debug("Setting OPENAI_API_KEY environment variable", extra={"key_length": len(self._api_key)})
        else:
            self._logger.debug("No API key provided for agent authentication")

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
                    self._logger.debug("Agent stderr output", extra={"stderr_line": decoded_line})
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
        data = json.dumps(payload)
        async with self._write_lock:
            self._stdin.write(data.encode() + b"\n")
            await self._stdin.drain()
        self._logger.debug("Sent JSON-RPC message to agent", extra={
            "method": payload.get("method"),
            "id": payload.get("id"),
            "params": payload.get("params")
        })

    async def _read_json(self) -> dict[str, Any]:
        if not self._stdout:
            raise AgentProcessError("Agent stdout unavailable")

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
                self._logger.debug("Received JSON-RPC message from agent", extra={
                    "method": payload.get("method"),
                    "id": payload.get("id"),
                    "has_result": "result" in payload,
                    "has_error": "error" in payload,
                    "result": payload.get("result"),
                    "error": payload.get("error")
                })
                self._logger.debug("Parsed JSON payload", extra={"payload": payload})
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
        # Try the exact format from the user's example first
        params = {"protocolVersion": "v1", "clientName": "cli", "capabilities": {}}
        self._logger.info("=== INITIALIZE PHASE ===")
        self._logger.debug("Sending initialize request", extra={"params": params})
        try:
            result = await self.request("initialize", params)
            self._logger.info("Initialize response received", extra={"result": result})

            # Check if authentication is required
            auth_methods = result.get("authMethods", []) if result else []
            if auth_methods:
                self._logger.info("Authentication required", extra={"auth_methods": [m.get("id") for m in auth_methods]})

                # Look for API key authentication method
                api_key_method = None
                for method in auth_methods:
                    if method.get("id") == "apikey":
                        api_key_method = method
                        break

                if api_key_method and self._api_key:
                    self._logger.info("Using API key authentication", extra={"method_id": "apikey"})
                    await self.authenticate("apikey", self._api_key)
                elif api_key_method:
                    raise AgentProcessError("Agent requires API key authentication but no API key provided")
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
        self._logger.info("=== AUTHENTICATION PHASE ===")
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
        self._logger.info("=== SESSION CREATION PHASE ===")
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
        self._logger.info("=== SESSION LOADING PHASE ===")
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
                # History replay notifications will be sent here

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
        self._logger.info("=== PROMPT PHASE ===")
        self._logger.debug("Sending session/prompt request", extra={
            "session_id": session_id,
            "prompt_length": len(prompt),
            "has_cancel_event": cancel_event is not None
        })
        """Send a session/prompt request and return the final result."""

        async def handler(payload: dict[str, Any]) -> None:
            self._logger.debug("Handling notification during prompt", extra={
                "method": payload.get("method"),
                "payload_keys": list(payload.keys())
            })

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
                    if text and on_chunk:
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
                if event == "session/cancelled":
                    self._logger.warning("Agent reported cancellation via session/update")
                    raise PromptCancelled("Agent reported cancellation")
            elif payload.get("method") == "session/cancelled":
                self._logger.warning("Agent reported direct cancellation")
                raise PromptCancelled("Agent reported cancellation")

        # If a cancel_event is provided, also check for external cancellation
        if cancel_event:
            logger.info("Setting up external cancellation check", extra={"session_id": session_id, "cancel_event_is_set": cancel_event.is_set()})
            async def check_external_cancellation():
                logger.info("Waiting for external cancellation", extra={"session_id": session_id})
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
                self._logger.info("Prompt processing completed successfully", extra={
                    "session_id": session_id,
                    "has_result": result is not None,
                    "result_keys": list(result.keys()) if result else []
                })
                return result or {}
        else:
            result = await self.request(
                "session/prompt",
                {"sessionId": session_id, "prompt": prompt},
                handler=handler,
            )
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
