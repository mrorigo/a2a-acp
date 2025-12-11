import pytest
from typing import Any
from types import SimpleNamespace
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi.testclient import TestClient
from fastapi import status

from a2a_acp.main import create_app
from a2a_acp.task_manager import A2ATaskManager, create_task_id
from a2a_acp.settings import _get_push_notification_settings


@pytest.fixture
def mock_settings():
    """Mock settings with development tool extension enabled."""
    with patch("a2a_acp.settings.get_settings") as mock:
        settings_instance = MagicMock()
        settings_instance.development_tool_extension_enabled = True
        settings_instance.auth_token = None
        settings_instance.agent_command = "true"
        settings_instance.agent_api_key = None
        settings_instance.agent_description = "Test Agent"
        settings_instance.push_notifications = _get_push_notification_settings()
        settings_instance.error_profile = MagicMock(value="acp-basic")
        mock.return_value = settings_instance
        yield mock


@pytest.fixture
def mock_tool_manager():
    """Mock tool configuration manager with sample tools."""
    mock = MagicMock()
    mock_echo = MagicMock()
    mock_echo.id = "echo"
    mock_echo.name = "echo"
    mock_echo.description = "Echo a message"
    echo_param = MagicMock()
    echo_param.name = "message"
    echo_param.type = "string"
    echo_param.description = "Message to echo"
    echo_param.required = True
    mock_echo.parameters = [echo_param]
    mock_echo.tags = []
    mock_ls = MagicMock()
    mock_ls.id = "list_files"
    mock_ls.name = "ls"
    mock_ls.description = "List files"
    ls_param = MagicMock()
    ls_param.name = "path"
    ls_param.type = "string"
    ls_param.description = "Path to list"
    ls_param.required = False
    mock_ls.parameters = [ls_param]
    mock_ls.tags = []
    sample_tools = {"echo": mock_echo, "list_files": mock_ls}
    mock.load_tools = AsyncMock(return_value=sample_tools)
    return mock


@pytest.fixture
def mock_task_manager():
    """Mock task manager for command execution tests."""
    mock = MagicMock(spec=A2ATaskManager)
    metadata_cache: dict[str, Any] = {"metadata": {}}

    async def create_task_side_effect(*args, **kwargs):
        metadata_cache["metadata"] = kwargs.get("metadata", {})
        task_id = create_task_id()
        task = MagicMock(id=task_id)
        task.metadata = metadata_cache["metadata"]
        mock.create_task.return_value = task
        return task

    async def get_task_side_effect(task_id):
        status = SimpleNamespace(state=SimpleNamespace(value="completed"))
        return SimpleNamespace(
            id=task_id,
            contextId="slash-command",
            status=status,
            metadata=metadata_cache["metadata"],
        )

    mock.create_task = AsyncMock(side_effect=create_task_side_effect)
    mock.get_task = AsyncMock(side_effect=get_task_side_effect)
    return mock


@pytest.fixture
def test_client(mock_settings, mock_tool_manager, mock_task_manager) -> TestClient:
    """Create test client with mocked dependencies."""
    # Patch dependencies in app
    context_manager = MagicMock()
    context_manager.create_context = AsyncMock(return_value="ctx")
    context_manager.add_task_to_context = AsyncMock()
    context_manager.add_message_to_context = AsyncMock()
    with (
        patch("a2a_acp.main.get_tool_configuration_manager", return_value=mock_tool_manager),
        patch("a2a_acp.main.get_task_manager", return_value=mock_task_manager),
        patch("a2a_acp.main.get_context_manager", return_value=context_manager),
    ):
        app = create_app()
        with TestClient(app) as client:
            yield client


class TestCommandsGetEndpoint:
    """Tests for /a2a/commands/get endpoint."""

    def test_get_commands_returns_slash_commands(self, test_client):
        """Test endpoint returns SlashCommands from tools."""
        response = test_client.get("/a2a/commands/get")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "commands" in data
        commands = data["commands"]
        assert len(commands) == 2  # From mock tools

        # Verify first command (echo)
        echo_cmd = next(cmd for cmd in commands if cmd["name"] == "echo")
        assert echo_cmd["description"] == "Echo a message"
        assert len(echo_cmd["arguments"]) == 1
        arg = echo_cmd["arguments"][0]
        assert arg["name"] == "message"
        assert arg["type"] == "string"
        assert arg["required"] is True

        # Verify second command (ls)
        ls_cmd = next(cmd for cmd in commands if cmd["name"] == "ls")
        assert ls_cmd["description"] == "List files"
        assert len(ls_cmd["arguments"]) == 1
        arg = ls_cmd["arguments"][0]
        assert arg["name"] == "path"
        assert arg["required"] is False

    def test_get_commands_disabled_when_extension_off(self, test_client, mock_settings):
        """Test endpoint returns 404 when extension disabled."""
        settings = mock_settings.return_value
        settings.development_tool_extension_enabled = False
        settings.auth_token = None

        response = test_client.get("/a2a/commands/get")
        assert response.status_code == 404
        assert response.json()["detail"] == "Development tool extension not enabled"

    def test_get_commands_handles_no_tools(self, test_client):
        """Test endpoint returns empty list when no tools configured."""
        with patch("a2a_acp.main.get_tool_configuration_manager") as mock_tool_mgr:
            mock = MagicMock()
            mock.load_tools = AsyncMock(return_value={})
            mock_tool_mgr.return_value = mock

            response = test_client.get("/a2a/commands/get")
            assert response.status_code == 200
            data = response.json()
            assert data["commands"] == []


class TestCommandExecuteEndpoint:
    """Tests for /a2a/command/execute endpoint."""

    def test_execute_command_creates_task_and_returns_response(self, test_client, mock_task_manager):
        """Test endpoint creates task and returns ExecuteSlashCommandResponse."""
        request_data = {
            "command": "echo",
            "arguments": {"message": "Hello World"},
        }

        response = test_client.post("/a2a/command/execute", json=request_data)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "execution_id" in data
        assert data["status"] == "executing"  # CommandExecutionStatus.EXECUTING.value

        # Verify task_manager.create_task was called
        mock_task_manager.create_task.assert_called_once()
        call_kwargs = mock_task_manager.create_task.call_args.kwargs
        initial_message = call_kwargs["initial_message"]
        assert initial_message.role == "user"
        assert initial_message.parts[0].text == "/echo {'message': 'Hello World'}"
        assert initial_message.metadata["slash_command"]["command"] == "echo"

    def test_execute_command_validation_error(self, test_client):
        """Test validation error for invalid request."""
        invalid_data = {
            "command": "invalid",  # No such command
            "arguments": {},  # Missing required args
        }

        response = test_client.post("/a2a/command/execute", json=invalid_data)

        assert response.status_code == 422  # Validation error
        detail = response.json()["detail"]
        assert isinstance(detail, list)
        assert any("validation" in str(error).lower() for error in detail)

    def test_execute_command_disabled_when_extension_off(self, test_client, mock_settings):
        """Test endpoint returns 404 when extension disabled."""
        settings = mock_settings.return_value
        settings.development_tool_extension_enabled = False
        settings.auth_token = None

        response = test_client.post("/a2a/command/execute", json={"command": "echo", "arguments": {"message": "test"}})
        assert response.status_code == 404
        assert response.json()["detail"] == "Development tool extension not enabled"

    def test_execute_command_auth_required(self, test_client, mock_settings):
        """Test authentication required for endpoint."""
        # Assuming auth is enabled in settings
        settings = mock_settings.return_value
        settings.auth_token = "test-token"

        response = test_client.post("/a2a/command/execute", json={"command": "echo", "arguments": {"message": "test"}})
        assert response.status_code == 401
        assert "bearer token" in response.json()["detail"]


class TestAgentCardExtension:
    """Tests for agent card including extension capabilities."""

    def test_agent_card_includes_extension(self, test_client):
        """Test agent card includes development-tool extension when enabled."""
        response = test_client.get("/.well-known/agent-card.json")

        assert response.status_code == 200
        data = response.json()

        assert "capabilities" in data
        capabilities = data["capabilities"]
        assert "extensions" in capabilities
        extensions = capabilities["extensions"]

        assert len(extensions) == 1
        ext = extensions[0]
        assert ext["uri"] == "https://developers.google.com/gemini/a2a/extensions/development-tool/v1"
        params = ext.get("params", {})
        assert params.get("version") == "1.0.0"
        metadata = params.get("metadata", {})
        assert "description" in metadata

    def test_agent_card_excludes_extension_when_disabled(self, test_client, mock_settings):
        """Test agent card excludes extension when disabled."""
        settings = mock_settings.return_value
        settings.development_tool_extension_enabled = False
        settings.auth_token = None

        response = test_client.get("/.well-known/agent-card.json")
        data = response.json()

        capabilities = data["capabilities"]
        assert "extensions" not in capabilities

    def test_authenticated_agent_card(self, test_client):
        """Test authenticated extended agent card includes extensions."""
        # Mock auth (assuming no token required in test)
        response = test_client.post("/a2a/rpc", json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "agent/getAuthenticatedExtendedCard",
            "params": {},
        })

        assert response.status_code == 200
        data = response.json()

        assert "result" in data
        result = data["result"]
        capabilities = result["capabilities"]
        assert "extensions" in capabilities
        assert len(capabilities["extensions"]) == 1


class TestErrorHandlingAndValidation:
    """Tests for endpoint validation and error handling."""

    def test_commands_get_unauthorized(self, test_client, mock_settings):
        """Test unauthorized access to commands endpoint."""
        settings = mock_settings.return_value
        settings.auth_token = "required-token"

        response = test_client.get("/a2a/commands/get")
        assert response.status_code == 401
        assert "bearer token" in response.json()["detail"]

    def test_command_execute_missing_arguments(self, test_client):
        """Test execute command with missing required arguments."""
        request_data = {
            "command": "echo",
            "arguments": {},  # Missing 'message'
        }

        response = test_client.post("/a2a/command/execute", json=request_data)
        assert response.status_code == 422
        errors = response.json()["detail"]
        assert any("required" in str(error).lower() for error in errors)  # Required field missing

    def test_command_execute_invalid_command(self, test_client, mock_tool_manager):
        """Test execute non-existent command."""
        request_data = {"command": "nonexistent", "arguments": {}}
        # Ensure no tools available for this test
        mock_tool_manager.load_tools = AsyncMock(return_value={})

        response = test_client.post("/a2a/command/execute", json=request_data)
        assert response.status_code == 200

    def test_endpoints_return_proper_errors_on_internal_failure(self, test_client, mock_task_manager):
        """Test internal errors return 500 with proper messages."""
        mock_task_manager.create_task.side_effect = Exception("Database error")

        request_data = {"command": "echo", "arguments": {"message": "test"}}

        response = test_client.post("/a2a/command/execute", json=request_data)
        assert response.status_code == 500
        assert "internal error" in response.json()["detail"].lower()


class TestConfigurationToggling:
    """Tests for configuration flag controlling endpoint availability."""

    @pytest.mark.parametrize("enabled", [True, False])
    def test_endpoints_availability_based_on_config(self, test_client, enabled, mock_settings):
        """Test endpoints available only when extension enabled."""
        settings = mock_settings.return_value
        settings.development_tool_extension_enabled = enabled
        settings.auth_token = None

        response = test_client.get("/a2a/commands/get")
        if enabled:
            assert response.status_code == 200
        else:
            assert response.status_code == 404

        response_exec = test_client.post("/a2a/command/execute", json={"command": "echo", "arguments": {}})
        if enabled:
            assert response_exec.status_code in [200, 422]  # Valid or validation error
        else:
            assert response_exec.status_code == 404

    def test_agent_card_always_available(self, test_client, mock_settings):
        """Test agent card endpoint always available, but extensions conditional."""
        # Even if extension disabled, card should return 200
        settings = mock_settings.return_value
        settings.development_tool_extension_enabled = False
        settings.auth_token = None

        response = test_client.get("/.well-known/agent-card.json")
        assert response.status_code == 200
        data = response.json()
        capabilities = data["capabilities"]
        assert "extensions" not in capabilities  # But no extensions


def test_integration_commands_to_task_execution(test_client, mock_task_manager):
    """Integration test: commands/get to command/execute to task creation."""
    # 1. Get available commands
    commands_response = test_client.get("/a2a/commands/get")
    assert commands_response.status_code == 200
    commands_data = commands_response.json()
    assert len(commands_data["commands"]) > 0
    command_name = commands_data["commands"][0]["name"]

    # 2. Execute a command
    request_data = {
        "command": command_name,
        "arguments": {"message": "Integration test"} if command_name == "echo" else {},
    }
    execute_response = test_client.post("/a2a/command/execute", json=request_data)
    assert execute_response.status_code == 200
    exec_data = execute_response.json()
    execution_id = exec_data["execution_id"]

    # 3. Verify task was created
    mock_task_manager.create_task.assert_called()
    created_task_id = mock_task_manager.create_task.return_value.id
    assert execution_id == created_task_id

    # 4. Get task to verify metadata
    task_response = test_client.post("/a2a/rpc", json={
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tasks/get",
        "params": {"id": execution_id},
    })
    assert task_response.status_code == 200
    task_data = task_response.json()["result"]
    assert task_data["metadata"]["type"] == "slash_command"
    assert task_data["metadata"]["command"] == command_name
