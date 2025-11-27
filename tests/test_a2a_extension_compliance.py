import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import status
import json

from a2a_acp.main import create_app
from a2a_acp.task_manager import A2ATaskManager
from a2a_acp.models import (
    ToolCallStatus,
    DevelopmentToolEventKind,
    ToolCall,
    ToolOutput,
    ErrorDetails,
    ExecuteDetails,
    ToolCallConfirmation,
    AgentSettings,
    CommandExecutionStatus,
    ExecuteSlashCommandResponse,
)
from a2a_acp.settings import get_settings
from a2a.models import (
    Message,
    TextPart,
    Task,
    TaskStatus,
    TaskState,
    create_message_id,
    create_task_id,
)


@pytest.fixture
def mock_settings():
    """Mock settings with extension enabled."""
    with patch("a2a_acp.settings.get_settings") as mock:
        settings_instance = MagicMock()
        settings_instance.development_tool_extension_enabled = True
        mock.return_value = settings_instance
        yield mock


@pytest.fixture
def mock_tool_manager():
    """Mock tool config with sample tools."""
    mock = MagicMock()
    mock.load_tools.return_value = {
        "echo": MagicMock(id="echo", name="echo", description="Echo", script="echo {{msg}}"),
    }
    return mock


@pytest.fixture
def mock_task_manager_full():
    """Full mock for task manager with extension support."""
    mock = AsyncMock(spec=A2ATaskManager)
    # Mock create_task to return task with extension metadata
    def create_task_mock(context_id, agent_name, initial_message=None, metadata=None):
        task_id = create_task_id()
        task = Task(
            id=task_id,
            contextId=context_id,
            status=TaskStatus(state=TaskState.SUBMITTED, timestamp=...),
            history=[initial_message] if initial_message else [],
            metadata=metadata or {},
        )
        # Add extension metadata
        if "development-tool" not in task.metadata:
            task.metadata["development-tool"] = {}
        return task

    mock.create_task = AsyncMock(side_effect=create_task_mock)
    mock.execute_task.return_value = MagicMock(
        status=MagicMock(state=TaskState.COMPLETED),
        metadata={"development-tool": {"tool_calls": {}}},
    )
    mock.get_task.return_value = MagicMock(
        status=MagicMock(state=TaskState.COMPLETED),
        metadata={"development-tool": {"tool_calls": {}}},
    )
    mock.provide_input_and_continue.return_value = MagicMock(
        status=MagicMock(state=TaskState.COMPLETED),
        metadata={"development-tool": {"tool_calls": {}}},
    )
    return mock


@pytest.fixture
def mock_push_manager():
    """Mock push manager to verify metadata."""
    mock = AsyncMock()
    return mock


@pytest.fixture
def test_client_full(mock_settings, mock_tool_manager, mock_task_manager_full, mock_push_manager):
    """Test client with full mocks for E2E simulation."""
    with (
        patch("a2a_acp.main.get_tool_configuration_manager", return_value=mock_tool_manager),
        patch("a2a_acp.main.get_task_manager", return_value=mock_task_manager_full),
        patch("a2a_acp.main.PushNotificationManager", return_value=mock_push_manager),
        patch("a2a_acp.main.get_context_manager", return_value=MagicMock()),
    ):
        app = create_app()
        with TestClient(app) as client:
            yield client


class TestRFCFlowSimulation:
    """Tests simulating complete RFC flows (tool call lifecycle, etc.)."""

    @pytest.mark.asyncio
    def test_complete_tool_call_lifecycle(self, test_client_full, mock_task_manager_full):
        """Simulate full tool call flow: pending -> confirmation -> executing -> succeeded."""
        # 1. Create task that triggers tool call (via message)
        message_data = {
            "role": "user",
            "parts": [{"kind": "text", "text": "Run echo tool"}],
            "messageId": str(create_message_id()),
            "metadata": {},
        }
        response = test_client_full.post("/a2a/message/send", json={"message": message_data})

        assert response.status_code == 200
        task_data = response.json()["task"]
        task_id = task_data["id"]

        # Verify initial ToolCall PENDING in metadata (mocked in create_task)
        metadata = task_data["metadata"]
        dev_tool = metadata["development-tool"]
        assert "tool_calls" in dev_tool
        tool_call = ToolCall.from_dict(list(dev_tool["tool_calls"].values())[0])
        assert tool_call.status == ToolCallStatus.PENDING

        # 2. Simulate confirmation: provide input with ToolCallConfirmation
        confirmation_data = {
            "tool_call_id": list(dev_tool["tool_calls"].keys())[0],
            "selected_option_id": "approve",
        }
        input_message = {
            "role": "user",
            "parts": [{"kind": "text", "text": ""}],  # Empty for permission
            "messageId": str(create_message_id()),
            "metadata": {
                "development-tool": {"tool_call_confirmation": confirmation_data},
            },
        }
        continue_response = test_client_full.post(
            "/a2a/message/send",  # Reuse for continuation, or mock provide_input
            json={"message": input_message},
        )

        assert continue_response.status_code == 200
        continued_task = continue_response.json()["task"]
        continued_metadata = continued_task["metadata"]["development-tool"]
        updated_tool_call = ToolCall.from_dict(list(continued_metadata["tool_calls"].values())[0])
        assert updated_tool_call.status == ToolCallStatus.EXECUTING

        # 3. Verify completion: get task shows SUCCEEDED
        get_response = test_client_full.post("/a2a/rpc", json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tasks/get",
            "params": {"id": task_id},
        })
        final_task = get_response.json()["result"]
        final_dev_tool = final_task["metadata"]["development-tool"]
        final_tool_call = ToolCall.from_dict(list(final_dev_tool["tool_calls"].values())[0])
        assert final_tool_call.status == ToolCallStatus.SUCCEEDED
        assert isinstance(final_tool_call.result, ToolOutput)
        assert "echo" in final_tool_call.result.content  # From mock

        # Verify mock calls
        mock_task_manager_full.provide_input_and_continue.assert_called()

    @pytest.mark.asyncio
    def test_tool_confirmation_roundtrip(self, test_client_full):
        """Test tool confirmation roundtrip flow."""
        # 1. Trigger tool call requiring confirmation
        message_data = {
            "role": "user",
            "parts": [{"kind": "text", "text": "Confirm sensitive tool"}],
            "messageId": str(create_message_id()),
            "metadata": {},
        }
        response = test_client_full.post("/a2a/message/send", json={"message": message_data})
        task_data = response.json()["task"]
        task_id = task_data["id"]

        # Verify PENDING ToolCall with confirmation_request
        dev_tool = task_data["metadata"]["development-tool"]
        tool_call = ToolCall.from_dict(list(dev_tool["tool_calls"].values())[0])
        assert tool_call.status == ToolCallStatus.PENDING
        assert tool_call.confirmation_request is not None
        assert len(tool_call.confirmation_request.options) >= 2

        # 2. Deny confirmation
        deny_confirmation = {
            "tool_call_id": list(dev_tool["tool_calls"].keys())[0],
            "selected_option_id": "deny",
        }
        deny_input = {
            "role": "user",
            "parts": [{"kind": "text", "text": ""}],
            "messageId": str(create_message_id()),
            "metadata": {
                "development-tool": {"tool_call_confirmation": deny_confirmation},
            },
        }
        deny_response = test_client_full.post("/a2a/message/send", json={"message": deny_input})
        assert deny_response.status_code == 200
        denied_task = deny_response.json()["task"]
        denied_tool_call = ToolCall.from_dict(
            list(denied_task["metadata"]["development-tool"]["tool_calls"].values())[0]
        )
        assert denied_tool_call.status == ToolCallStatus.FAILED
        assert isinstance(denied_tool_call.result, ErrorDetails)
        assert "denied" in denied_tool_call.result.message.lower()

        # 3. Approve in separate flow (new task for simplicity)
        approve_message = {
            "role": "user",
            "parts": [{"kind": "text", "text": "Approve tool"}],
            "messageId": str(create_message_id()),
        }
        approve_response = test_client_full.post("/a2a/message/send", json={"message": approve_message})
        approve_task = approve_response.json()["task"]
        approve_tool_call = ToolCall.from_dict(
            list(approve_task["metadata"]["development-tool"]["tool_calls"].values())[0]
        )
        assert approve_tool_call.status == ToolCallStatus.SUCCEEDED  # Mock completion


class TestSlashCommandExecutionLifecycle:
    """Tests for slash command execution lifecycle."""

    def test_slash_command_full_lifecycle(self, test_client_full):
        """Test get commands -> execute -> task creation -> completion."""
        # 1. Get commands
        commands_resp = test_client_full.get("/a2a/commands/get")
        assert commands_resp.status_code == 200
        commands = commands_resp.json()["commands"]
        assert len(commands) > 0
        cmd_name = commands[0]["name"]
        cmd_args = {arg["name"]: "test" for arg in commands[0]["arguments"] if arg["required"]}

        # 2. Execute command
        exec_request = {"command": cmd_name, "arguments": cmd_args}
        exec_resp = test_client_full.post("/a2a/command/execute", json=exec_request)
        assert exec_resp.status_code == 200
        exec_data = exec_resp.json()
        exec_id = exec_data["execution_id"]
        assert exec_data["status"] == "executing"

        # 3. Get task to verify creation and metadata
        get_rpc = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tasks/get",
            "params": {"id": exec_id},
        }
        task_resp = test_client_full.post("/a2a/rpc", json=get_rpc)
        assert task_resp.status_code == 200
        task_data = task_resp.json()["result"]
        assert task_data["id"] == exec_id
        metadata = task_data["metadata"]
        assert metadata["type"] == "slash_command"
        assert metadata["command"] == cmd_name
        assert "development-tool" in metadata

        # 4. Simulate completion (mocked)
        assert task_data["status"]["state"] == "completed"  # From mock

    def test_slash_command_push_notification_with_metadata(self, test_client_full, mock_push_manager):
        """Test slash command execution emits push notification with extension metadata."""
        exec_request = {"command": "echo", "arguments": {"message": "Notify"}}
        response = test_client_full.post("/a2a/command/execute", json=exec_request)

        # Verify push notification called with metadata
        mock_push_manager.send_notification.assert_called()
        call = mock_push_manager.send_notification.call_args[0][1]
        assert "development-tool" in call
        dev_event = call["development-tool"]
        assert dev_event["kind"] == "tool_call_update"  # Or appropriate


class TestPushNotificationsWithExtensionMetadata:
    """Tests for push notifications including extension metadata."""

    @pytest.mark.asyncio
    def test_task_events_include_development_tool_metadata(self, test_client_full, mock_push_manager):
        """Test task status changes include DevelopmentToolEvent in notifications."""
        # Create task
        message_data = {"role": "user", "parts": [{"kind": "text", "text": "Test event"}], "messageId": str(create_message_id())}
        response = test_client_full.post("/a2a/message/send", json={"message": message_data})
        task_id = response.json()["task"]["id"]

        # Verify notifications emitted with metadata
        assert mock_push_manager.send_notification.call_count > 0
        for call in mock_push_manager.send_notification.call_args_list:
            payload = call[0][1]
            if "development_tool_metadata" in payload:
                dev_meta = payload["development_tool_metadata"]
                assert isinstance(dev_meta, dict)
                # Verify structure
                if "tool_calls" in dev_meta:
                    tool_call = ToolCall.from_dict(list(dev_meta["tool_calls"].values())[0])
                    assert tool_call.status in [ToolCallStatus.PENDING, ToolCallStatus.SUCCEEDED]

    @pytest.mark.asyncio
    def test_error_notifications_include_error_details(self, test_client_full, mock_push_manager):
        """Test error events include ErrorDetails in metadata."""
        # Mock task failure
        with patch.object(mock_push_manager, "send_notification"):
            # Simulate failed task (via RPC or send)
            fail_message = {"role": "user", "parts": [{"kind": "text", "text": "Fail task"}], "messageId": str(create_message_id())}
            response = test_client_full.post("/a2a/message/send", json={"message": fail_message})

        # Verify failure notification
        calls = mock_push_manager.send_notification.call_args_list
        failure_call = next((c for c in calls if c[0][1].get("event") == "task_failed"), None)
        if failure_call:
            payload = failure_call[0][1]
            dev_meta = payload.get("development_tool_metadata", {})
            if "tool_calls" in dev_meta:
                tool_call = ToolCall.from_dict(list(dev_meta["tool_calls"].values())[0])
                assert tool_call.status == ToolCallStatus.FAILED
                assert isinstance(tool_call.result, ErrorDetails)


class TestBackwardCompatibility:
    """Basic backward compatibility checks in E2E context."""

    def test_existing_message_flow_unchanged(self, test_client_full):
        """Test existing message/send flow works without extension interference."""
        message_data = {
            "role": "user",
            "parts": [{"kind": "text", "text": "Simple message"}],
            "messageId": str(create_message_id()),
        }
        response = test_client_full.post("/a2a/message/send", json={"message": message_data})

        assert response.status_code == 200
        task_data = response.json()["task"]
        # Verify no breaking changes: standard fields present
        assert "id" in task_data
        assert "status" in task_data
        assert task_data["status"]["state"] == "completed"  # Mock
        # Extension metadata optional, not required
        metadata = task_data.get("metadata", {})
        assert "development-tool" not in metadata or isinstance(metadata.get("development-tool"), dict)

    def test_extension_optional_in_agent_card(self, test_client_full):
        """Test agent card works without extension when disabled."""
        with patch("a2a_acp.settings.get_settings") as mock_settings:
            settings = MagicMock()
            settings.development_tool_extension_enabled = False
            mock_settings.return_value = settings

            response = test_client_full.get("/.well-known/agent-card.json")
            assert response.status_code == 200
            data = response.json()
            capabilities = data["capabilities"]
            # Core capabilities present
            assert "streaming" in capabilities
            assert capabilities["streaming"] is True
            # No extensions
            assert "extensions" not in capabilities

    def test_push_notifications_backward_compatible(self, test_client_full, mock_push_manager):
        """Test push notifications work without extension metadata for legacy clients."""
        # Send message
        message_data = {"role": "user", "parts": [{"kind": "text", "text": "Legacy test"}], "messageId": str(create_message_id())}
        test_client_full.post("/a2a/message/send", json={"message": message_data})

        # Verify notifications emitted with optional metadata
        calls = mock_push_manager.send_notification.call_args_list
        for call in calls:
            payload = call[0][1]
            # Legacy fields present
            assert "event" in payload
            assert "task_id" in payload
            # Extension metadata optional
            if "development_tool_metadata" in payload:
                dev_meta = payload["development_tool_metadata"]
                assert isinstance(dev_meta, dict)  # Safe for legacy to ignore


@pytest.mark.asyncio
async def test_full_rfc_example_flow(test_client_full):
    """Simulate complete RFC example flow (tool confirmation roundtrip + slash command)."""
    # Step 1: AgentSettings in initial message
    settings_message = {
        "role": "user",
        "parts": [{"kind": "text", "text": "Initialize with workspace"}],
        "messageId": str(create_message_id()),
        "metadata": {
            "agent_settings": {"workspace_path": "/rfc/workspace"},
        },
    }
    init_response = test_client_full.post("/a2a/message/send", json={"message": settings_message})
    init_task = init_response.json()["task"]
    assert init_task["metadata"].get("development-tool", {}).get("agent_settings", {}).get("workspace_path") == "/rfc/workspace"

    # Step 2: Get slash commands
    commands_resp = test_client_full.get("/a2a/commands/get")
    commands = commands_resp.json()["commands"][0]  # First command
    cmd_name = commands["name"]

    # Step 3: Execute slash command
    slash_args = {arg["name"]: "RFC test" for arg in commands["arguments"]}
    exec_request = {"command": cmd_name, "arguments": slash_args}
    exec_resp = test_client_full.post("/a2a/command/execute", json=exec_request)
    exec_id = exec_resp.json()["execution_id"]

    # Step 4: Get task and verify slash metadata
    get_rpc = {"jsonrpc": "2.0", "id": 4, "method": "tasks/get", "params": {"id": exec_id}}
    task_resp = test_client_full.post("/a2a/rpc", json=get_rpc)
    slash_task = task_resp.json()["result"]
    assert slash_task["metadata"]["type"] == "slash_command"

    # Step 5: Simulate tool call in task requiring confirmation
    confirm_input = {
        "role": "user",
        "parts": [{"kind": "text", "text": "Confirm tool in slash"}],
        "messageId": str(create_message_id()),
        "metadata": {
            "development-tool": {
                "tool_call_confirmation": {
                    "tool_call_id": "tc_rfc_1",  # Assume from task
                    "selected_option_id": "approve",
                }
            }
        },
    }
    confirm_resp = test_client_full.post("/a2a/message/send", json={"message": confirm_input})
    assert confirm_resp.status_code == 200

    # Step 6: Verify completion with push notification metadata
    # Mock push would have been called with extension events
    # Here, verify task final state
    final_get = test_client_full.post("/a2a/rpc", json=get_rpc)
    final_task = final_get.json()["result"]
    final_tool_call = ToolCall.from_dict(
        list(final_task["metadata"]["development-tool"]["tool_calls"].values())[0]
    )
    assert final_tool_call.status == ToolCallStatus.SUCCEEDED

    # Verify backward compat: standard task fields intact
    assert "id" in final_task
    assert "status" in final_task
    assert final_task["status"]["state"] == "completed"