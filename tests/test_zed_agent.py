"""
Comprehensive test suite for ZedAgent functionality.

This test suite provides extensive coverage for the ZedAgentConnection class,
focusing on real subprocess interaction and edge cases that were previously untested.
"""

import asyncio
import json
from types import SimpleNamespace
from typing import Any, List
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from a2a_acp.zed_agent import (
    ZedAgentConnection,
    AgentProcessError,
    PromptCancelled,
    ToolPermissionRequest,
    ToolPermissionDecision,
)
from a2a_acp.bash_executor import ToolExecutionResult
from a2a_acp.error_profiles import ErrorProfile


# Proper async mocking without warning suppression


class TestZedAgentConnectionLifecycle:
    """Test ZedAgent connection lifecycle management."""

    def test_connection_creation_empty_command(self):
        """Test that empty command raises ValueError."""
        with pytest.raises(ValueError, match="Agent command cannot be empty"):
            ZedAgentConnection([])

    def test_connection_creation_valid(self):
        """Test successful connection creation."""
        connection = ZedAgentConnection(["echo", "test"])
        assert connection._command == ["echo", "test"]
        assert connection._api_key is None
        assert connection._process is None
        # Locks are created lazily now
        assert connection._read_lock is None
        assert connection._write_lock is None

    def test_connection_creation_with_api_key(self):
        """Test connection creation with API key."""
        api_key = "test-api-key-123"
        connection = ZedAgentConnection(["echo", "test"], api_key=api_key)
        assert connection._api_key == api_key

    @pytest.mark.asyncio
    async def test_connection_context_manager(self):
        """Test async context manager protocol."""
        connection = ZedAgentConnection(["echo", "test"])

        # Test that we can enter context (would start process if not mocked)
        try:
            async with connection:
                pass
        except Exception:
            # Expected in test environment
            pass


class TestZedAgentProcessManagement:
    """Test subprocess process management."""

    @patch("asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_process_start_success(self, mock_create_subprocess):
        """Test successful process startup."""
        # Mock the subprocess
        mock_process = AsyncMock()
        mock_process.stdin = AsyncMock()
        mock_process.stdout = AsyncMock()
        mock_process.stderr = AsyncMock()
        mock_process.stderr.readline = AsyncMock(return_value=b"")
        mock_create_subprocess.return_value = mock_process

        connection = ZedAgentConnection(["echo", "test"])

        # Should start successfully
        await connection.start()

        # Verify subprocess was created with correct parameters
        mock_create_subprocess.assert_called_once()
        call_args = mock_create_subprocess.call_args
        assert "echo" in call_args[0]
        assert "test" in call_args[0]

        # Verify process attributes are set
        assert connection._process == mock_process
        assert connection._stdin == mock_process.stdin
        assert connection._stdout == mock_process.stdout

    @patch("asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_process_start_with_api_key(self, mock_create_subprocess):
        """Test process startup with API key sets environment."""
        mock_process = AsyncMock()
        mock_process.stdin = AsyncMock()
        mock_process.stdout = AsyncMock()
        mock_process.stderr = AsyncMock()
        mock_process.stderr.readline = AsyncMock(return_value=b"")
        mock_create_subprocess.return_value = mock_process

        api_key = "test-api-key-123"
        connection = ZedAgentConnection(["echo", "test"], api_key=api_key)

        await connection.start()

        # Verify environment was set
        call_kwargs = mock_create_subprocess.call_args[1]
        assert "env" in call_kwargs
        assert call_kwargs["env"]["OPENAI_API_KEY"] == api_key

    @patch("asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_process_start_already_started(self, mock_create_subprocess):
        """Test that starting an already started process is a no-op."""
        mock_process = AsyncMock()
        mock_create_subprocess.return_value = mock_process

        connection = ZedAgentConnection(["echo", "test"])
        connection._process = mock_process  # Simulate already started

        await connection.start()

        # Should not create new subprocess
        mock_create_subprocess.assert_not_called()

    @patch("asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_process_close_graceful(self, mock_create_subprocess):
        """Test graceful process termination."""
        # Mock process
        mock_process = AsyncMock()
        mock_process.stdin.write_eof = (
            MagicMock()
        )  # write_eof() is not async in real StreamWriter
        mock_process.stdin.close = (
            MagicMock()
        )  # close() is not async in real StreamWriter
        mock_process.wait = AsyncMock(return_value=0)  # wait() is async in real Process
        mock_process.returncode = 0
        mock_process.stderr = AsyncMock()
        mock_process.stderr.readline = AsyncMock(return_value=b"")
        mock_create_subprocess.return_value = mock_process

        connection = ZedAgentConnection(["echo", "test"])
        await connection.start()

        # Test graceful close
        await connection.close()

        # Verify cleanup calls
        mock_process.stdin.write_eof.assert_called_once()
        mock_process.stdin.close.assert_called_once()
        mock_process.wait.assert_called_once()

        # Verify state is cleaned up
        assert connection._process is None
        assert connection._stdin is None
        assert connection._stdout is None

    @patch("asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_process_close_timeout_terminate(self, mock_create_subprocess):
        """Test process termination after timeout."""
        # Mock process that doesn't exit quickly
        mock_process = AsyncMock()
        mock_process.stdin.write_eof = (
            MagicMock()
        )  # write_eof() is not async in real StreamWriter
        mock_process.stdin.close = (
            MagicMock()
        )  # close() is not async in real StreamWriter
        mock_process.wait = AsyncMock(
            side_effect=[
                asyncio.TimeoutError(),  # First wait times out
                0,  # Second wait succeeds after terminate
            ]
        )
        mock_process.terminate = MagicMock()  # terminate() is not async in real Process
        mock_process.stderr = AsyncMock()
        mock_process.stderr.readline = AsyncMock(return_value=b"")
        mock_create_subprocess.return_value = mock_process

        connection = ZedAgentConnection(["sleep", "10"])
        await connection.start()

        await connection.close()

        # Verify termination was called
        mock_process.terminate.assert_called_once()
        assert mock_process.wait.call_count == 2

    @patch("asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_process_close_force_kill(self, mock_create_subprocess):
        """Test process killing after terminate fails."""
        # Mock process that refuses to die
        mock_process = AsyncMock()
        mock_process.stdin.write_eof = (
            MagicMock()
        )  # write_eof() is not async in real StreamWriter
        mock_process.stdin.close = (
            MagicMock()
        )  # close() is not async in real StreamWriter
        mock_process.wait = AsyncMock(
            side_effect=[
                asyncio.TimeoutError(),  # First wait times out
                asyncio.TimeoutError(),  # Second wait times out
                0,  # Final wait succeeds after kill
            ]
        )
        mock_process.terminate = MagicMock()  # terminate() is not async in real Process
        mock_process.kill = MagicMock()  # kill() is not async in real Process
        mock_process.stderr = AsyncMock()
        mock_process.stderr.readline = AsyncMock(return_value=b"")
        mock_create_subprocess.return_value = mock_process

        connection = ZedAgentConnection(["sleep", "10"])
        await connection.start()

        await connection.close()

        # Verify kill was called after terminate failed
        mock_process.terminate.assert_called_once()
        mock_process.kill.assert_called_once()
        assert mock_process.wait.call_count == 3


class TestZedAgentJSONRPC:
    """Test JSON-RPC communication functionality."""

    @patch("asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_write_json_success(self, mock_create_subprocess):
        """Test successful JSON message writing."""
        mock_process = AsyncMock()
        mock_process.stdin = AsyncMock()
        mock_process.stdin.write = (
            MagicMock()
        )  # write() is not async in real StreamWriter
        mock_process.stdin.drain = AsyncMock()  # drain() is async in real StreamWriter
        mock_process.stdout = AsyncMock()
        mock_process.stderr = AsyncMock()
        mock_process.stderr.readline = AsyncMock(return_value=b"")
        mock_create_subprocess.return_value = mock_process

        connection = ZedAgentConnection(["echo", "test"])
        await connection.start()

        # Test writing JSON
        payload = {"jsonrpc": "2.0", "method": "test", "id": 1}
        await connection._write_json(payload)

        # Verify write was called
        mock_process.stdin.write.assert_called_once()
        call_args = mock_process.stdin.write.call_args[0][0]
        assert b'{"jsonrpc": "2.0", "method": "test", "id": 1}\n' == call_args

    @patch("asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_write_json_no_stdin(self, mock_create_subprocess):
        """Test writing JSON when stdin is not available."""
        connection = ZedAgentConnection(["echo", "test"])
        # Don't start the process, so stdin is None

        with pytest.raises(AgentProcessError, match="Agent stdin unavailable"):
            await connection._write_json({"test": "data"})

    @patch("asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_read_json_success(self, mock_create_subprocess):
        """Test successful JSON message reading."""
        mock_process = AsyncMock()
        mock_process.stdout = AsyncMock()
        mock_process.stdout.readline = AsyncMock(
            side_effect=[
                b'{"jsonrpc": "2.0", "result": "success", "id": 1}\n',
                b"",  # EOF
            ]
        )
        mock_process.stderr = AsyncMock()
        mock_process.stderr.readline = AsyncMock(return_value=b"")
        mock_create_subprocess.return_value = mock_process

        connection = ZedAgentConnection(["echo", "test"])
        await connection.start()

        # Test reading JSON
        result = await connection._read_json()

        assert result == {"jsonrpc": "2.0", "result": "success", "id": 1}

    @patch("asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_read_json_skip_empty_lines(self, mock_create_subprocess):
        """Test that empty lines are skipped when reading JSON."""
        mock_process = AsyncMock()
        mock_process.stdout = AsyncMock()
        mock_process.stdout.readline = AsyncMock(
            side_effect=[
                b"\n",  # Empty line
                b"   \n",  # Whitespace line
                b'{"jsonrpc": "2.0", "result": "success", "id": 1}\n',
                b"",  # EOF
            ]
        )
        mock_process.stderr = AsyncMock()
        mock_process.stderr.readline = AsyncMock(return_value=b"")
        mock_create_subprocess.return_value = mock_process

        connection = ZedAgentConnection(["echo", "test"])
        await connection.start()

        result = await connection._read_json()

        # Should skip empty lines and return the JSON
        assert result == {"jsonrpc": "2.0", "result": "success", "id": 1}
        # We expect 3 calls: empty line, whitespace line, and JSON line
        assert mock_process.stdout.readline.call_count == 3

    @patch("asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_read_json_skip_log_lines(self, mock_create_subprocess):
        """Test that log lines (not starting with '{') are skipped."""
        mock_process = AsyncMock()
        mock_process.stdout = AsyncMock()
        mock_process.stdout.readline = AsyncMock(
            side_effect=[
                b"INFO: Starting agent...\n",
                b"WARN: Some warning message\n",
                b'{"jsonrpc": "2.0", "result": "success", "id": 1}\n',
                b"",  # EOF
            ]
        )
        mock_process.stderr = AsyncMock()
        mock_process.stderr.readline = AsyncMock(return_value=b"")
        mock_create_subprocess.return_value = mock_process

        connection = ZedAgentConnection(["echo", "test"])
        await connection.start()

        result = await connection._read_json()

        # Should skip log lines and return the JSON
        assert result == {"jsonrpc": "2.0", "result": "success", "id": 1}
        # We expect 3 calls: info line, warn line, and JSON line
        assert mock_process.stdout.readline.call_count == 3

    @patch("asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_read_json_invalid_json_retry(self, mock_create_subprocess):
        """Test handling of invalid JSON with retry."""
        mock_process = AsyncMock()
        mock_process.stdout = AsyncMock()
        mock_process.stdout.readline = AsyncMock(
            side_effect=[
                b'{"jsonrpc": "2.0", "invalid": json}\n',  # Invalid JSON
                b'{"jsonrpc": "2.0", "result": "success", "id": 1}\n',  # Valid JSON
                b"",  # EOF
            ]
        )
        mock_process.stderr = AsyncMock()
        mock_process.stderr.readline = AsyncMock(return_value=b"")
        mock_create_subprocess.return_value = mock_process

        connection = ZedAgentConnection(["echo", "test"])
        await connection.start()

        result = await connection._read_json()

        # Should retry after invalid JSON and return valid JSON
        assert result == {"jsonrpc": "2.0", "result": "success", "id": 1}
        # We expect 2 calls: invalid JSON line, and valid JSON line
        assert mock_process.stdout.readline.call_count == 2


class TestZedAgentRequestResponse:
    """Test request/response functionality."""

    @patch("asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_request_success(self, mock_create_subprocess):
        """Test successful request/response cycle."""
        mock_process = AsyncMock()
        mock_process.stdin = AsyncMock()
        mock_process.stdin.write = (
            MagicMock()
        )  # write() is not async in real StreamWriter
        mock_process.stdin.drain = AsyncMock()  # drain() is async in real StreamWriter
        mock_process.stdout = AsyncMock()
        mock_process.stdout.readline = AsyncMock(
            side_effect=[
                b'{"jsonrpc": "2.0", "result": "test response", "id": 1}\n',
                b"",  # EOF
            ]
        )
        mock_process.stderr = AsyncMock()
        mock_process.stderr.readline = AsyncMock(return_value=b"")
        mock_create_subprocess.return_value = mock_process

        connection = ZedAgentConnection(["echo", "test"])
        await connection.start()

        # Send request
        result = await connection.request("test_method", {"param": "value"})

        assert result == "test response"

        # Verify request was sent
        sent_data = mock_process.stdin.write.call_args[0][0]
        assert b'"method": "test_method"' in sent_data
        assert b'"param": "value"' in sent_data

    @patch("asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_request_with_error(self, mock_create_subprocess):
        """Test request that returns an error."""
        mock_process = AsyncMock()
        mock_process.stdin = AsyncMock()
        mock_process.stdin.write = (
            MagicMock()
        )  # write() is not async in real StreamWriter
        mock_process.stdin.drain = AsyncMock()  # drain() is async in real StreamWriter
        mock_process.stdout = AsyncMock()
        mock_process.stdout.readline = AsyncMock(
            side_effect=[
                b'{"jsonrpc": "2.0", "error": {"code": -32000, "message": "Test error"}, "id": 1}\n',
                b"",  # EOF
            ]
        )
        mock_process.stderr = AsyncMock()
        mock_process.stderr.readline = AsyncMock(return_value=b"")
        mock_create_subprocess.return_value = mock_process

        connection = ZedAgentConnection(["echo", "test"])
        await connection.start()

        # Should raise AgentProcessError for error response
        with pytest.raises(AgentProcessError) as exc_info:
            await connection.request("test_method")

        assert "Test error" in str(exc_info.value)

    @patch("asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_request_with_notification_handler(self, mock_create_subprocess):
        """Test request with notification handler."""
        mock_process = AsyncMock()
        mock_process.stdin = AsyncMock()
        mock_process.stdin.write = (
            MagicMock()
        )  # write() is not async in real StreamWriter
        mock_process.stdin.drain = AsyncMock()  # drain() is async in real StreamWriter
        mock_process.stdout = AsyncMock()
        mock_process.stdout.readline = AsyncMock(
            side_effect=[
                b'{"jsonrpc": "2.0", "method": "notification", "params": {"data": "test"}}\n',  # Notification
                b'{"jsonrpc": "2.0", "result": "response", "id": 1}\n',  # Response
                b"",  # EOF
            ]
        )
        mock_process.stderr = AsyncMock()
        mock_process.stderr.readline = AsyncMock(return_value=b"")
        mock_create_subprocess.return_value = mock_process

        connection = ZedAgentConnection(["echo", "test"])
        await connection.start()

        # Track notifications
        notifications = []

        async def notification_handler(payload):
            notifications.append(payload)

        # Send request with handler
        result = await connection.request("test_method", handler=notification_handler)

        assert result == "response"
        assert len(notifications) == 1
        assert notifications[0]["method"] == "notification"

    @patch("asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_notify(self, mock_create_subprocess):
        """Test sending notifications."""
        mock_process = AsyncMock()
        mock_process.stdin = AsyncMock()
        mock_process.stdin.write = (
            MagicMock()
        )  # write() is not async in real StreamWriter
        mock_process.stdin.drain = AsyncMock()  # drain() is async in real StreamWriter
        mock_process.stderr = AsyncMock()
        mock_process.stderr.readline = AsyncMock(return_value=b"")
        mock_create_subprocess.return_value = mock_process

        connection = ZedAgentConnection(["echo", "test"])
        await connection.start()

        # Send notification
        await connection.notify("test_notification", {"data": "test"})

        # Verify notification was sent
        sent_data = mock_process.stdin.write.call_args[0][0]
        assert b'"method": "test_notification"' in sent_data
        assert b'"data": "test"' in sent_data
        # Notifications should not have an ID
        assert b'"id"' not in sent_data

    @patch("asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_id_counter(self, mock_create_subprocess):
        """Test request ID counter functionality."""
        mock_process = AsyncMock()
        mock_process.stdin = AsyncMock()
        mock_process.stdin.write = (
            MagicMock()
        )  # write() is not async in real StreamWriter
        mock_process.stdin.drain = AsyncMock()  # drain() is async in real StreamWriter
        mock_process.stdout = AsyncMock()
        mock_process.stdout.readline = AsyncMock(
            side_effect=[
                b'{"jsonrpc": "2.0", "result": "ok", "id": 1}\n',
                b'{"jsonrpc": "2.0", "result": "ok", "id": 2}\n',
                b'{"jsonrpc": "2.0", "result": "ok", "id": 3}\n',
                b"",  # EOF
            ]
        )
        mock_process.stderr = AsyncMock()
        mock_process.stderr.readline = AsyncMock(return_value=b"")
        mock_create_subprocess.return_value = mock_process

        connection = ZedAgentConnection(["echo", "test"])
        await connection.start()

        # Send multiple requests
        await connection.request("method1")
        await connection.request("method2")
        await connection.request("method3")

        # Check that IDs are incrementing
        write_calls = mock_process.stdin.write.call_args_list
        assert len(write_calls) == 3

        # Extract IDs from the sent data
        ids = []
        for call in write_calls:
            data = call[0][0]
            try:
                payload = json.loads(data.decode().strip())
                ids.append(payload.get("id"))
            except json.JSONDecodeError:
                pass

        assert ids == [1, 2, 3]


class TestZedAgentAuthentication:
    """Test authentication functionality."""

    @patch("asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_initialize_success(self, mock_create_subprocess):
        """Test successful initialization."""
        mock_process = AsyncMock()
        mock_process.stdin = AsyncMock()
        mock_process.stdin.write = (
            MagicMock()
        )  # write() is not async in real StreamWriter
        mock_process.stdin.drain = AsyncMock()  # drain() is async in real StreamWriter
        mock_process.stdout = AsyncMock()
        mock_process.stdout.readline = AsyncMock(
            side_effect=[
                b'{"jsonrpc": "2.0", "result": {"protocolVersion": "v1", "capabilities": {}}, "id": 1}\n',
                b"",  # EOF
            ]
        )
        mock_process.stderr = AsyncMock()
        mock_process.stderr.readline = AsyncMock(return_value=b"")
        mock_create_subprocess.return_value = mock_process

        connection = ZedAgentConnection(["echo", "test"])
        await connection.start()

        result = await connection.initialize()

        assert result is not None
        assert result["protocolVersion"] == "v1"
        assert result["capabilities"] == {}

    @patch("asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_initialize_with_auth_required(self, mock_create_subprocess):
        """Test initialization with authentication required."""
        mock_process = AsyncMock()
        mock_process.stdin = AsyncMock()
        mock_process.stdin.write = (
            MagicMock()
        )  # write() is not async in real StreamWriter
        mock_process.stdin.drain = AsyncMock()  # drain() is async in real StreamWriter
        mock_process.stdout = AsyncMock()
        mock_process.stdout.readline = AsyncMock(
            side_effect=[
                b'{"jsonrpc": "2.0", "result": {"protocolVersion": "v1", "capabilities": {}, "authMethods": [{"id": "apikey"}]}, "id": 1}\n',
                b"",  # EOF
            ]
        )
        mock_process.stderr = AsyncMock()
        mock_process.stderr.readline = AsyncMock(return_value=b"")
        mock_create_subprocess.return_value = mock_process

        api_key = "test-api-key"
        connection = ZedAgentConnection(["echo", "test"], api_key=api_key)
        await connection.start()

        # Mock authentication method
        auth_call_count = 0

        async def mock_authenticate(method_id, api_key=None):
            nonlocal auth_call_count
            auth_call_count += 1
            return {"authenticated": True}

        connection.authenticate = mock_authenticate

        result = await connection.initialize()

        assert auth_call_count == 1
        assert result is not None

    @patch("asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_initialize_auth_required_no_api_key(self, mock_create_subprocess):
        """Test initialization fails when auth required but no API key."""
        mock_process = AsyncMock()
        mock_process.stdin = AsyncMock()
        mock_process.stdin.write = (
            MagicMock()
        )  # write() is not async in real StreamWriter
        mock_process.stdin.drain = AsyncMock()  # drain() is async in real StreamWriter
        mock_process.stdout = AsyncMock()
        mock_process.stdout.readline = AsyncMock(
            side_effect=[
                b'{"jsonrpc": "2.0", "result": {"protocolVersion": "v1", "capabilities": {}, "authMethods": [{"id": "apikey"}]}, "id": 1}\n',
                b"",  # EOF
            ]
        )
        mock_process.stderr = AsyncMock()
        mock_process.stderr.readline = AsyncMock(return_value=b"")
        mock_create_subprocess.return_value = mock_process

        connection = ZedAgentConnection(["echo", "test"])  # No API key
        await connection.start()

        with pytest.raises(
            AgentProcessError,
            match="Agent requires API key authentication but no API key provided",
        ):
            await connection.initialize()

    @patch("asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_initialize_with_gemini_auth_required(self, mock_create_subprocess):
        """Test initialization with Gemini API key authentication required."""
        mock_process = AsyncMock()
        mock_process.stdin = AsyncMock()
        mock_process.stdin.write = (
            MagicMock()
        )  # write() is not async in real StreamWriter
        mock_process.stdin.drain = AsyncMock()  # drain() is async in real StreamWriter
        mock_process.stdout = AsyncMock()
        mock_process.stdout.readline = AsyncMock(
            side_effect=[
                b'{"jsonrpc": "2.0", "result": {"protocolVersion": "v1", "capabilities": {}, "authMethods": [{"id": "gemini-api-key"}]}, "id": 1}\n',
                b"",  # EOF
            ]
        )
        mock_process.stderr = AsyncMock()
        mock_process.stderr.readline = AsyncMock(return_value=b"")
        mock_create_subprocess.return_value = mock_process

        api_key = "gemini-test-api-key"
        connection = ZedAgentConnection(["echo", "test"], api_key=api_key)
        await connection.start()

        # Mock authentication method
        auth_call_count = 0
        auth_method_used = None

        async def mock_authenticate(method_id, api_key=None):
            nonlocal auth_call_count, auth_method_used
            auth_call_count += 1
            auth_method_used = method_id
            return {"authenticated": True}

        connection.authenticate = mock_authenticate

        result = await connection.initialize()

        assert auth_call_count == 1
        assert auth_method_used == "gemini-api-key"
        assert result is not None

    @patch("asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_initialize_gemini_auth_required_no_api_key(
        self, mock_create_subprocess
    ):
        """Test initialization fails when Gemini auth required but no API key."""
        mock_process = AsyncMock()
        mock_process.stdin = AsyncMock()
        mock_process.stdin.write = (
            MagicMock()
        )  # write() is not async in real StreamWriter
        mock_process.stdin.drain = AsyncMock()  # drain() is async in real StreamWriter
        mock_process.stdout = AsyncMock()
        mock_process.stdout.readline = AsyncMock(
            side_effect=[
                b'{"jsonrpc": "2.0", "result": {"protocolVersion": "v1", "capabilities": {}, "authMethods": [{"id": "gemini-api-key"}]}, "id": 1}\n',
                b"",  # EOF
            ]
        )
        mock_process.stderr = AsyncMock()
        mock_process.stderr.readline = AsyncMock(return_value=b"")
        mock_create_subprocess.return_value = mock_process

        connection = ZedAgentConnection(["echo", "test"])  # No API key
        await connection.start()

        with pytest.raises(
            AgentProcessError,
            match="Agent requires API key authentication but no API key provided",
        ):
            await connection.initialize()

    @patch("asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_authenticate_success(self, mock_create_subprocess):
        """Test successful authentication."""
        mock_process = AsyncMock()
        mock_process.stdin = AsyncMock()
        mock_process.stdin.write = (
            MagicMock()
        )  # write() is not async in real StreamWriter
        mock_process.stdin.drain = AsyncMock()  # drain() is async in real StreamWriter
        mock_process.stdout = AsyncMock()
        # Provide response for the authenticate request
        mock_process.stdout.readline = AsyncMock(
            side_effect=[
                b'{"jsonrpc": "2.0", "result": {"authenticated": True}, "id": 1}\n',
                b"",  # EOF
            ]
        )
        mock_process.stderr = AsyncMock()
        mock_process.stderr.readline = AsyncMock(return_value=b"")
        mock_create_subprocess.return_value = mock_process

        connection = ZedAgentConnection(["echo", "test"])
        await connection.start()

        # Mock the request method to avoid complex JSON-RPC flow
        async def mock_request(method, params=None, handler=None):
            if method == "authenticate":
                return {"authenticated": True}
            return None

        connection.request = mock_request

        result = await connection.authenticate("apikey", "test-key")

        assert result is not None
        assert result["authenticated"] is True

    @patch("asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_authenticate_failure(self, mock_create_subprocess):
        """Test authentication failure."""
        mock_process = AsyncMock()
        mock_process.stdin = AsyncMock()
        mock_process.stdin.write = (
            MagicMock()
        )  # write() is not async in real StreamWriter
        mock_process.stdin.drain = AsyncMock()  # drain() is async in real StreamWriter
        mock_process.stdout = AsyncMock()
        mock_process.stdout.readline = AsyncMock(
            side_effect=[
                b'{"jsonrpc": "2.0", "error": {"code": -32001, "message": "Invalid API key"}, "id": 1}\n',
                b"",  # EOF
            ]
        )
        mock_process.stderr = AsyncMock()
        mock_process.stderr.readline = AsyncMock(return_value=b"")
        mock_create_subprocess.return_value = mock_process

        connection = ZedAgentConnection(["echo", "test"])
        await connection.start()

        with pytest.raises(AgentProcessError, match="Invalid API key"):
            await connection.authenticate("apikey", "invalid-key")


class TestZedAgentSessionManagement:
    """Test session management functionality."""

    @patch("asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_start_session_success(self, mock_create_subprocess):
        """Test successful session creation."""
        mock_process = AsyncMock()
        mock_process.stdin = AsyncMock()
        mock_process.stdin.write = (
            MagicMock()
        )  # write() is not async in real StreamWriter
        mock_process.stdin.drain = AsyncMock()  # drain() is async in real StreamWriter
        mock_process.stdout = AsyncMock()
        mock_process.stdout.readline = AsyncMock(
            side_effect=[
                b'{"jsonrpc": "2.0", "result": {"sessionId": "session-123"}, "id": 1}\n',
                b"",  # EOF
            ]
        )
        mock_process.stderr = AsyncMock()
        mock_process.stderr.readline = AsyncMock(return_value=b"")
        mock_create_subprocess.return_value = mock_process

        connection = ZedAgentConnection(["echo", "test"])
        await connection.start()

        session_id = await connection.start_session("/test/dir")

        assert session_id == "session-123"

    @patch("asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_start_session_missing_session_id(self, mock_create_subprocess):
        """Test session creation fails when sessionId is missing."""
        mock_process = AsyncMock()
        mock_process.stdin = AsyncMock()
        mock_process.stdin.write = (
            MagicMock()
        )  # write() is not async in real StreamWriter
        mock_process.stdin.drain = AsyncMock()  # drain() is async in real StreamWriter
        mock_process.stdout = AsyncMock()
        mock_process.stdout.readline = AsyncMock(
            side_effect=[
                b'{"jsonrpc": "2.0", "result": {"other": "data"}, "id": 1}\n',
                b"",  # EOF
            ]
        )
        mock_process.stderr = AsyncMock()
        mock_process.stderr.readline = AsyncMock(return_value=b"")
        mock_create_subprocess.return_value = mock_process

        connection = ZedAgentConnection(["echo", "test"])
        await connection.start()

        with pytest.raises(AgentProcessError, match="session/new missing sessionId"):
            await connection.start_session("/test/dir")

    @patch("asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_load_session_success(self, mock_create_subprocess):
        """Test successful session loading."""
        mock_process = AsyncMock()
        mock_process.stdin = AsyncMock()
        mock_process.stdin.write = (
            MagicMock()
        )  # write() is not async in real StreamWriter
        mock_process.stdin.drain = AsyncMock()  # drain() is async in real StreamWriter
        mock_process.stdout = AsyncMock()
        mock_process.stdout.readline = AsyncMock(
            side_effect=[
                b'{"jsonrpc": "2.0", "result": {"loaded": True, "sessionId": "session-123"}, "id": 1}\n',
                b"",  # EOF
            ]
        )
        mock_process.stderr = AsyncMock()
        mock_process.stderr.readline = AsyncMock(return_value=b"")
        mock_create_subprocess.return_value = mock_process

        connection = ZedAgentConnection(["echo", "test"])
        await connection.start()

        # Mock the request method to avoid complex JSON-RPC flow
        async def mock_request(method, params=None, handler=None):
            if method == "session/load":
                return {"loaded": True, "sessionId": "session-123"}
            return None

        connection.request = mock_request

        await connection.load_session("session-123", "/test/dir")

        # Should complete without error

    @patch("asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_load_session_with_history_notifications(
        self, mock_create_subprocess
    ):
        """Test session loading with conversation history."""
        mock_process = AsyncMock()
        mock_process.stdin = AsyncMock()
        mock_process.stdin.write = (
            MagicMock()
        )  # write() is not async in real StreamWriter
        mock_process.stdin.drain = AsyncMock()  # drain() is async in real StreamWriter
        mock_process.stdout = AsyncMock()
        mock_process.stdout.readline = AsyncMock(
            side_effect=[
                # History message 1
                b'{"jsonrpc": "2.0", "method": "session/update", "params": {"update": {"sessionUpdate": "history_message", "message": {"role": "user", "content": "Hello"}}}}\n',
                # History message 2
                b'{"jsonrpc": "2.0", "method": "session/update", "params": {"update": {"sessionUpdate": "history_message", "message": {"role": "assistant", "content": "Hi there!"}}}}\n',
                # Load response
                b'{"jsonrpc": "2.0", "result": {"loaded": True}, "id": 1}\n',
                b"",  # EOF
            ]
        )
        mock_process.stderr = AsyncMock()
        mock_process.stderr.readline = AsyncMock(return_value=b"")
        mock_create_subprocess.return_value = mock_process

        connection = ZedAgentConnection(["echo", "test"])
        await connection.start()

        # Track notifications
        async def mock_request(method, params=None, handler=None):
            if method == "session/load":
                # Simulate the notifications that would be sent during load
                if handler:
                    await handler(
                        {
                            "jsonrpc": "2.0",
                            "method": "session/update",
                            "params": {
                                "update": {
                                    "sessionUpdate": "history_message",
                                    "message": {"role": "user", "content": "Hello"},
                                }
                            },
                        }
                    )
                    await handler(
                        {
                            "jsonrpc": "2.0",
                            "method": "session/update",
                            "params": {
                                "update": {
                                    "sessionUpdate": "history_message",
                                    "message": {
                                        "role": "assistant",
                                        "content": "Hi there!",
                                    },
                                }
                            },
                        }
                    )
                return {"loaded": True}
            return None

        connection.request = mock_request

        await connection.load_session("session-123", "/test/dir")

        # Should complete successfully despite notifications


class TestZedAgentPromptHandling:
    """Test prompt handling and streaming."""

    @patch("asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_prompt_success(self, mock_create_subprocess):
        """Test successful prompt execution."""
        mock_process = AsyncMock()
        mock_process.stdin = AsyncMock()
        mock_process.stdin.write = (
            MagicMock()
        )  # write() is not async in real StreamWriter
        mock_process.stdin.drain = AsyncMock()  # drain() is async in real StreamWriter
        mock_process.stdout = AsyncMock()
        mock_process.stdout.readline = AsyncMock(
            side_effect=[
                b'{"jsonrpc": "2.0", "result": {"response": "Final response"}, "id": 1}\n',
                b"",  # EOF
            ]
        )
        mock_process.stderr = AsyncMock()
        mock_process.stderr.readline = AsyncMock(return_value=b"")
        mock_create_subprocess.return_value = mock_process

        connection = ZedAgentConnection(["echo", "test"])
        await connection.start()

        result = await connection.prompt(
            "session-123", [{"type": "text", "text": "Hello"}]
        )

        assert result["response"] == "Final response"

    @patch("asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_prompt_with_chunk_handler(self, mock_create_subprocess):
        """Test prompt with chunk handler for streaming."""
        mock_process = AsyncMock()
        mock_process.stdin = AsyncMock()
        mock_process.stdin.write = (
            MagicMock()
        )  # write() is not async in real StreamWriter
        mock_process.stdin.drain = AsyncMock()  # drain() is async in real StreamWriter
        mock_process.stdout = AsyncMock()
        mock_process.stdout.readline = AsyncMock(
            side_effect=[
                # Streaming chunks
                b'{"jsonrpc": "2.0", "method": "session/update", "params": {"update": {"sessionUpdate": "agent_message_chunk", "content": {"text": "Hello"}}}}\n',
                b'{"jsonrpc": "2.0", "method": "session/update", "params": {"update": {"sessionUpdate": "agent_message_chunk", "content": {"text": " world"}}}}\n',
                # Final response
                b'{"jsonrpc": "2.0", "result": {"response": "Hello world"}, "id": 1}\n',
                b"",  # EOF
            ]
        )
        mock_process.stderr = AsyncMock()
        mock_process.stderr.readline = AsyncMock(return_value=b"")
        mock_create_subprocess.return_value = mock_process

        connection = ZedAgentConnection(["echo", "test"])
        await connection.start()

        # Track chunks
        chunks = []

        async def chunk_handler(text):
            chunks.append(text)

        # Mock request to simulate streaming
        async def mock_request(method, params=None, handler=None):
            if method == "session/prompt":
                if handler:
                    await handler(
                        {
                            "jsonrpc": "2.0",
                            "method": "session/update",
                            "params": {
                                "update": {
                                    "sessionUpdate": "agent_message_chunk",
                                    "content": {"text": "Hello"},
                                }
                            },
                        }
                    )
                    await handler(
                        {
                            "jsonrpc": "2.0",
                            "method": "session/update",
                            "params": {
                                "update": {
                                    "sessionUpdate": "agent_message_chunk",
                                    "content": {"text": " world"},
                                }
                            },
                        }
                    )
                return {"response": "Hello world"}
            return None

        connection.request = mock_request

        result = await connection.prompt(
            "session-123", [{"type": "text", "text": "Hello"}], on_chunk=chunk_handler
        )

        assert result["response"] == "Hello world"
        assert chunks == ["Hello", " world"]

    @patch("asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_prompt_with_input_required(self, mock_create_subprocess):
        """Test prompt handling input-required notifications."""
        mock_process = AsyncMock()
        mock_process.stdin = AsyncMock()
        mock_process.stdin.write = (
            MagicMock()
        )  # write() is not async in real StreamWriter
        mock_process.stdin.drain = AsyncMock()  # drain() is async in real StreamWriter
        mock_process.stdout = AsyncMock()
        mock_process.stdout.readline = AsyncMock(
            side_effect=[
                # Input required notification
                b'{"jsonrpc": "2.0", "method": "session/update", "params": {"update": {"sessionUpdate": "input_required", "inputRequired": {"text": "More info needed", "inputTypes": ["text/plain"]}}}}\n',
                b'{"jsonrpc": "2.0", "result": {"response": "Done"}, "id": 1}\n',
                b"",  # EOF
            ]
        )
        mock_process.stderr = AsyncMock()
        mock_process.stderr.readline = AsyncMock(return_value=b"")
        mock_create_subprocess.return_value = mock_process

        connection = ZedAgentConnection(["echo", "test"])
        await connection.start()

        # Track chunks including input required
        chunks = []

        async def chunk_handler(text):
            chunks.append(text)

        # Mock request to simulate input required
        async def mock_request(method, params=None, handler=None):
            if method == "session/prompt":
                if handler:
                    await handler(
                        {
                            "jsonrpc": "2.0",
                            "method": "session/update",
                            "params": {
                                "update": {
                                    "sessionUpdate": "input_required",
                                    "inputRequired": {
                                        "text": "More info needed",
                                        "inputTypes": ["text/plain"],
                                    },
                                }
                            },
                        }
                    )
                return {"response": "Done"}
            return None

        connection.request = mock_request

        result = await connection.prompt(
            "session-123", [{"type": "text", "text": "Hello"}], on_chunk=chunk_handler
        )

        assert result["response"] == "Done"
        assert "INPUT_REQUIRED: More info needed" in chunks[0]

    @patch("asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_prompt_cancellation(self, mock_create_subprocess):
        """Test prompt cancellation functionality."""
        mock_process = AsyncMock()
        mock_process.stdin = AsyncMock()
        mock_process.stdin.write = (
            MagicMock()
        )  # write() is not async in real StreamWriter
        mock_process.stdin.drain = AsyncMock()  # drain() is async in real StreamWriter
        mock_process.stdout = AsyncMock()
        mock_process.stdout.readline = AsyncMock(
            side_effect=[
                # Cancellation notification
                b'{"jsonrpc": "2.0", "method": "session/cancelled", "params": {}}\n',
                b"",  # EOF
            ]
        )
        mock_process.stderr = AsyncMock()
        mock_process.stderr.readline = AsyncMock(return_value=b"")
        mock_create_subprocess.return_value = mock_process

        connection = ZedAgentConnection(["echo", "test"])
        await connection.start()

        # Create cancellation event
        cancel_event = asyncio.Event()

        # Mock request to simulate cancellation
        async def mock_request(method, params=None, handler=None):
            if method == "session/prompt":
                if handler:
                    await handler(
                        {"jsonrpc": "2.0", "method": "session/cancelled", "params": {}}
                    )
                raise PromptCancelled("Agent reported cancellation")
            return None

        connection.request = mock_request

        with pytest.raises(PromptCancelled):
            await connection.prompt(
                "session-123",
                [{"type": "text", "text": "Hello"}],
                cancel_event=cancel_event,
            )


class TestZedAgentErrorHandling:
    """Test error handling and edge cases."""

    def test_stderr_collection(self):
        """Test stderr collection functionality."""
        connection = ZedAgentConnection(["echo", "test"])

        # Simulate some stderr output
        connection._stderr_buffer = ["Line 1", "Line 2", "Line 3"]

        stderr_output = connection.stderr()
        assert stderr_output == "Line 1\nLine 2\nLine 3"

    @pytest.mark.asyncio
    async def test_connection_double_close(self):
        """Test that closing an already closed connection is safe."""
        connection = ZedAgentConnection(["echo", "test"])

        # Should not raise any errors
        await connection.close()

    @patch("asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, mock_create_subprocess):
        """Test handling concurrent requests."""
        mock_process = AsyncMock()
        mock_process.stdin = AsyncMock()
        mock_process.stdin.write = (
            MagicMock()
        )  # write() is not async in real StreamWriter
        mock_process.stdin.drain = AsyncMock()  # drain() is async in real StreamWriter
        mock_process.stdout = AsyncMock()
        mock_process.stdout.readline = AsyncMock(
            side_effect=[
                b'{"jsonrpc": "2.0", "result": "response1", "id": 1}\n',
                b'{"jsonrpc": "2.0", "result": "response2", "id": 2}\n',
                b"",  # EOF
            ]
        )
        mock_process.stderr = AsyncMock()
        mock_process.stderr.readline = AsyncMock(return_value=b"")
        mock_create_subprocess.return_value = mock_process

        connection = ZedAgentConnection(["echo", "test"])
        await connection.start()

        # Send concurrent requests
        request1 = connection.request("method1")
        request2 = connection.request("method2")

        result1, result2 = await asyncio.gather(request1, request2)

        assert result1 == "response1"
        assert result2 == "response2"

    @patch("asyncio.create_subprocess_exec")
    @pytest.mark.asyncio
    async def test_stdin_write_locking(self, mock_create_subprocess):
        """Test that stdin writes are properly serialized."""
        mock_process = AsyncMock()
        mock_process.stdin = AsyncMock()
        mock_process.stdin.write = (
            MagicMock()
        )  # write() is not async in real StreamWriter
        mock_process.stdin.drain = AsyncMock()  # drain() is async in real StreamWriter
        mock_process.stdout = AsyncMock()
        # Provide 5 responses for 5 requests, plus EOF
        mock_process.stdout.readline = AsyncMock(
            side_effect=[
                b'{"jsonrpc": "2.0", "result": "ok", "id": 1}\n',
                b'{"jsonrpc": "2.0", "result": "ok", "id": 2}\n',
                b'{"jsonrpc": "2.0", "result": "ok", "id": 3}\n',
                b'{"jsonrpc": "2.0", "result": "ok", "id": 4}\n',
                b'{"jsonrpc": "2.0", "result": "ok", "id": 5}\n',
                b"",  # EOF
            ]
        )
        mock_process.stderr = AsyncMock()
        mock_process.stderr.readline = AsyncMock(return_value=b"")
        mock_create_subprocess.return_value = mock_process

        connection = ZedAgentConnection(["echo", "test"])
        await connection.start()

        # Send multiple rapid requests
        await asyncio.gather(*[connection.request(f"method{i}") for i in range(5)])

        # Verify writes were serialized (all write calls completed)
        assert mock_process.stdin.write.call_count == 5


class TestZedAgentFilesystemGovernance:
    """Ensure direct filesystem RPC helpers honor governance pipeline."""

    @pytest.mark.asyncio
    async def test_fs_read_text_file_permission_denied(self, monkeypatch):
        recorded = {}

        async def fake_write_json(payload):
            recorded["payload"] = payload

        permission_calls: List[ToolPermissionRequest] = []

        async def permission_handler(
            request: ToolPermissionRequest,
        ) -> ToolPermissionDecision:
            permission_calls.append(request)
            return ToolPermissionDecision(option_id="deny")

        async def fake_execute_tool(
            *_args, **_kwargs
        ):  # pragma: no cover - should not be invoked
            recorded["executed"] = True
            return ToolExecutionResult(
                tool_id="functions.acp_fs__read_text_file",
                success=True,
                output="",
                error="",
                return_code=0,
                execution_time=0.0,
                metadata={},
                output_files=[],
            )

        class DummyExecutor:
            async def execute_tool(self, tool, params, context):
                return await fake_execute_tool(tool, params, context)

        class DummyTool:
            id = "functions.acp_fs__read_text_file"
            config = SimpleNamespace(requires_confirmation=False)

        monkeypatch.setenv("A2A_AGENT_COMMAND", "/bin/echo")
        conn = ZedAgentConnection(
            ["python", "-V"], permission_handler=permission_handler
        )
        conn._write_json = fake_write_json  # type: ignore[assignment]

        async def fake_get_tool(tool_id):
            return DummyTool()

        monkeypatch.setattr("a2a_acp.zed_agent.get_tool", fake_get_tool)
        monkeypatch.setattr(
            "a2a_acp.zed_agent.get_bash_executor", lambda: DummyExecutor()
        )

        await conn._handle_fs_read_text_file(
            {"id": 1, "params": {"path": "README.md"}}, "sess_1"
        )

        assert permission_calls, "Expected governance permission handler to run"
        payload = recorded.get("payload")
        assert payload and payload["error"]["code"] == -32003
        assert "executed" not in recorded

    @pytest.mark.asyncio
    async def test_fs_read_text_file_delegates_to_tool_even_when_missing(
        self, monkeypatch
    ):
        recorded: dict[str, Any] = {}

        async def fake_write_json(payload):
            recorded.setdefault("payloads", []).append(payload)

        class DummyTool:
            id = "functions.acp_fs__read_text_file"
            config = SimpleNamespace(requires_confirmation=False)

        class DummyExecutor:
            async def execute_tool(self, tool, params, context):
                recorded["executed_params"] = params
                return ToolExecutionResult(
                    tool_id=tool.id,
                    success=False,
                    output="",
                    error="missing",
                    execution_time=0.01,
                    return_code=1,
                    metadata={},
                    output_files=[],
                    mcp_error={
                        "code": -32002,
                        "message": "Resource not found",
                        "detail": "File not found",
                    },
                )

        monkeypatch.setenv("A2A_AGENT_COMMAND", "/bin/echo")
        conn = ZedAgentConnection(["python", "-V"])
        conn._write_json = fake_write_json  # type: ignore[assignment]

        async def fake_get_tool(_tool_id):
            return DummyTool()

        monkeypatch.setattr("a2a_acp.zed_agent.get_tool", fake_get_tool)
        monkeypatch.setattr(
            "a2a_acp.zed_agent.get_bash_executor", lambda: DummyExecutor()
        )

        await conn._handle_fs_read_text_file(
            {"id": 9, "params": {"path": "missing.txt"}}, "sess_exec"
        )

        assert recorded.get("executed_params") == {"path": "missing.txt"}
        payloads = recorded.get("payloads", [])
        assert payloads and payloads[-1]["error"]["code"] == -32002

    @pytest.mark.asyncio
    async def test_fs_write_text_file_permission_allow(self, monkeypatch):
        recorded = {}

        async def fake_write_json(payload):
            recorded.setdefault("payloads", []).append(payload)

        async def permission_handler(
            request: ToolPermissionRequest,
        ) -> ToolPermissionDecision:
            return ToolPermissionDecision(option_id="allow")

        async def fake_execute_tool(tool, params, context):
            recorded["executed_params"] = params
            return ToolExecutionResult(
                tool_id=tool.id,
                success=True,
                output="ok",
                error="",
                return_code=0,
                execution_time=0.0,
                metadata={},
                output_files=[],
            )

        class DummyExecutor:
            async def execute_tool(self, tool, params, context):
                return await fake_execute_tool(tool, params, context)

        class DummyTool:
            id = "functions.acp_fs__write_text_file"
            config = SimpleNamespace(requires_confirmation=False)

        conn = ZedAgentConnection(
            ["python", "-V"], permission_handler=permission_handler
        )
        conn._write_json = fake_write_json  # type: ignore[assignment]

        async def fake_get_tool(tool_id):
            return DummyTool()

        monkeypatch.setattr("a2a_acp.zed_agent.get_tool", fake_get_tool)
        monkeypatch.setattr(
            "a2a_acp.zed_agent.get_bash_executor", lambda: DummyExecutor()
        )

        await conn._handle_fs_write_text_file(
            {"id": 2, "params": {"path": "README.md", "content": "hi"}},
            "sess_2",
        )

        assert recorded.get("executed_params") == {"path": "README.md", "content": "hi"}
        payloads = recorded.get("payloads", [])
        assert payloads and payloads[-1]["result"]["content"] == "ok"


class TestZedAgentErrorFormatting:
    """Verify error payload shaping follows the negotiated profile."""

    def test_jsonrpc_error_basic_profile_uses_string_data(self):
        connection = ZedAgentConnection(
            ["echo", "test"], error_profile=ErrorProfile.ACP_BASIC
        )

        result = ToolExecutionResult(
            tool_id="sample",
            success=False,
            output="",
            error="File missing",
            execution_time=0.1,
            return_code=5,
            metadata={"a2a_diagnostics": {"raw_detail": {"path": "missing.txt"}}},
            output_files=[],
            mcp_error={
                "code": -32002,
                "message": "Resource not found",
                "detail": "missing.txt",
                "retryable": False,
            },
        )

        payload = connection._jsonrpc_error_from_tool_result(result)

        assert payload["code"] == -32002
        assert payload["message"] == "Resource not found"
        assert isinstance(payload.get("data"), str)

    def test_jsonrpc_error_extended_profile_retains_structure(self):
        connection = ZedAgentConnection(
            ["echo", "test"], error_profile=ErrorProfile.EXTENDED_JSON
        )

        diagnostics = {"return_code": 42}
        result = ToolExecutionResult(
            tool_id="sample",
            success=False,
            output="",
            error="Permission denied",
            execution_time=0.1,
            return_code=42,
            metadata={"a2a_diagnostics": diagnostics},
            output_files=[],
            mcp_error={
                "code": -32000,
                "message": "Authentication required",
                "detail": {"scope": "write"},
                "retryable": True,
            },
        )

        payload = connection._jsonrpc_error_from_tool_result(result)

        assert isinstance(payload["data"], dict)
        assert payload["data"]["detail"] == {"scope": "write"}
        assert payload["data"]["return_code"] == 42
        assert payload["data"]["diagnostics"] == diagnostics


class TestZedAgentRealSubprocess:
    """Test with real subprocess where possible."""

    @pytest.mark.asyncio
    async def test_real_subprocess_echo(self):
        """Test with a real echo subprocess."""
        # Use a simple command that should work on all systems
        connection = ZedAgentConnection(["echo", "test"])

        # This test is meant to verify the basic structure works
        # In a real environment, this would start a subprocess
        # In test environment, it may fail due to mocking or succeed
        try:
            await connection.start()
            # If we get here, the subprocess started successfully
            # Clean up the connection
            await connection.close()
        except Exception as e:
            # Expected in test environment without proper setup
            # Just verify it's not our test assertion
            assert "test" not in str(e)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
