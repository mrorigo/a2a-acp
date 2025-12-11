
from unittest.mock import MagicMock, patch, AsyncMock
import pytest
from a2a_acp.zed_agent import ZedAgentConnection

class TestZedAgentAuthenticationExtended:
    """Test extended authentication functionality for codex and openai keys."""

    @patch('asyncio.create_subprocess_exec')
    @pytest.mark.asyncio
    async def test_initialize_with_codex_auth_required(self, mock_create_subprocess):
        """Test initialization with Codex API key authentication required."""
        mock_process = AsyncMock()
        mock_process.stdin = AsyncMock()
        mock_process.stdin.write = MagicMock()
        mock_process.stdin.drain = AsyncMock()
        mock_process.stdout = AsyncMock()
        mock_process.stdout.readline = AsyncMock(side_effect=[
            b'{"jsonrpc": "2.0", "result": {"protocolVersion": "v1", "capabilities": {}, "authMethods": [{"id": "codex-api-key"}]}, "id": 1}\n',
            b'',  # EOF
        ])
        mock_process.stderr = AsyncMock()
        mock_process.stderr.readline = AsyncMock(return_value=b'')
        mock_create_subprocess.return_value = mock_process

        api_key = "codex-test-api-key"
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
        assert auth_method_used == "codex-api-key"
        assert result is not None

    @patch('asyncio.create_subprocess_exec')
    @pytest.mark.asyncio
    async def test_initialize_with_openai_auth_required(self, mock_create_subprocess):
        """Test initialization with OpenAI API key authentication required."""
        mock_process = AsyncMock()
        mock_process.stdin = AsyncMock()
        mock_process.stdin.write = MagicMock()
        mock_process.stdin.drain = AsyncMock()
        mock_process.stdout = AsyncMock()
        mock_process.stdout.readline = AsyncMock(side_effect=[
            b'{"jsonrpc": "2.0", "result": {"protocolVersion": "v1", "capabilities": {}, "authMethods": [{"id": "openai-api-key"}]}, "id": 1}\n',
            b'',  # EOF
        ])
        mock_process.stderr = AsyncMock()
        mock_process.stderr.readline = AsyncMock(return_value=b'')
        mock_create_subprocess.return_value = mock_process

        api_key = "openai-test-api-key"
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
        assert auth_method_used == "openai-api-key"
        assert result is not None

    @patch('asyncio.create_subprocess_exec')
    @pytest.mark.asyncio
    async def test_initialize_with_mixed_auth_methods(self, mock_create_subprocess):
        """Test initialization with multiple auth methods including supported ones."""
        mock_process = AsyncMock()
        mock_process.stdin = AsyncMock()
        mock_process.stdin.write = MagicMock()
        mock_process.stdin.drain = AsyncMock()
        mock_process.stdout = AsyncMock()
        mock_process.stdout.readline = AsyncMock(side_effect=[
            b'{"jsonrpc": "2.0", "result": {"protocolVersion": "v1", "capabilities": {}, "authMethods": [{"id": "chatgpt"}, {"id": "codex-api-key"}, {"id": "openai-api-key"}]}, "id": 1}\n',
            b'',  # EOF
        ])
        mock_process.stderr = AsyncMock()
        mock_process.stderr.readline = AsyncMock(return_value=b'')
        mock_create_subprocess.return_value = mock_process

        api_key = "test-api-key"
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
        # It should pick the first supported one found in the loop or list
        # In the code:
        # supported_api_key_methods = ["apikey", "gemini-api-key", "codex-api-key", "openai-api-key"]
        # for method in auth_methods: ...
        # auth_methods order: chatgpt, codex-api-key, openai-api-key
        # chatgpt is not supported.
        # codex-api-key is supported.
        assert auth_method_used == "codex-api-key"
        assert result is not None
