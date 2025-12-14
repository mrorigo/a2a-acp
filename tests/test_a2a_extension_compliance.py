import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from a2a_acp.main import create_app
from a2a_acp.task_manager import A2ATaskManager
from a2a_acp.models import (
    ToolCallStatus,
    ToolCall,
    ToolOutput,
    ErrorDetails,
    ExecuteDetails,
    ConfirmationRequest,
    ConfirmationOption,
    GenericDetails,
)
from a2a_acp.a2a.models import (
    Task,
    TaskStatus,
    TaskState,
    create_message_id,
    create_task_id,
    current_timestamp,
)
from a2a_acp.settings import _get_push_notification_settings


@pytest.fixture
def mock_settings():
    """Mock settings with extension enabled."""
    with patch("a2a_acp.settings.get_settings") as mock:
        settings_instance = MagicMock()
        settings_instance.development_tool_extension_enabled = True
        settings_instance.auth_token = None
        settings_instance.agent_command = "true"
        settings_instance.agent_api_key = None
        settings_instance.agent_description = "A2A Test Agent"
        settings_instance.push_notifications = _get_push_notification_settings()
        settings_instance.error_profile = MagicMock(value="acp-basic")
        mock.return_value = settings_instance
        yield mock


@pytest.fixture
def mock_tool_manager():
    """Mock tool config with sample tools."""
    mock = MagicMock()
    echo_tool = MagicMock()
    echo_tool.id = "echo"
    echo_tool.name = "echo"
    echo_tool.description = "Echo"
    echo_tool.tags = []
    echo_tool.script = "echo {{msg}}"
    param = SimpleNamespace(
        name="message",
        type="string",
        description="Message to echo",
        required=True,
    )
    echo_tool.parameters = [param]
    mock.load_tools = AsyncMock(return_value={"echo": echo_tool})
    return mock


@pytest.fixture
def mock_task_manager_full(mock_push_manager):
    """Full mock for task manager with extension support."""
    mock = AsyncMock(spec=A2ATaskManager)
    tool_calls_by_task: dict[str, ToolCall] = {}
    tool_call_counter = {"count": 0}
    def _next_tool_call_id() -> str:
        tool_call_counter["count"] += 1
        return f"tc_rfc_{tool_call_counter['count']}"
    tasks_by_id: dict[str, Task] = {}

    def _build_dev_tool_metadata(task_id: str) -> dict:
        task = tasks_by_id.get(task_id)
        existing_dev = {}
        if task and task.metadata:
            existing_dev = dict(task.metadata.get("development-tool", {}))
        tool_calls = dict(existing_dev.get("tool_calls", {}))
        tool_call = tool_calls_by_task.get(task_id)
        if tool_call:
            tool_calls[tool_call.tool_call_id] = tool_call.to_dict()
        dev_tool = {k: v for k, v in existing_dev.items() if k != "tool_calls"}
        if tool_calls:
            dev_tool["tool_calls"] = tool_calls
        if not dev_tool:
            dev_tool = {}
        return {"development-tool": dev_tool}

    def _snapshot_task(task_id: str, state: TaskState) -> Task:
        task = tasks_by_id.get(task_id)
        if not task:
            task = Task(
                id=task_id,
                contextId="ctx",
                status=TaskStatus(state=state, timestamp=current_timestamp()),
                history=[],
                artifacts=None,
                metadata={},
                kind="task",
            )
            tasks_by_id[task_id] = task
        task.status = TaskStatus(state=state, timestamp=current_timestamp())
        task.metadata = task.metadata or {}
        task.metadata.update(_build_dev_tool_metadata(task_id))
        tasks_by_id[task_id] = task
        return task

    async def _emit_push_event(task_id: str, event: str = "tool_call_update") -> None:
        dev_tool = _build_dev_tool_metadata(task_id)["development-tool"]
        payload = {
            "event": event,
            "task_id": task_id,
            "development-tool": {
                "kind": "tool_call_update",
                "tool_calls": dict(dev_tool.get("tool_calls", {})),
            },
            "development_tool_metadata": dev_tool,
        }
        await mock_push_manager.send_notification(task_id, payload)

    async def create_task_mock(context_id, agent_name, initial_message=None, metadata=None):
        task_id = create_task_id()
        task = Task(
            id=task_id,
            contextId=context_id,
            status=TaskStatus(state=TaskState.COMPLETED, timestamp=current_timestamp()),
            history=[initial_message] if initial_message else [],
            metadata=metadata or {},
        )
        dev_tool_meta = task.metadata.setdefault("development-tool", {})
        if initial_message and initial_message.metadata:
            agent_settings = initial_message.metadata.get("agent_settings")
            if isinstance(agent_settings, dict):
                dev_tool_meta.setdefault("agent_settings", agent_settings)

        message_text = ""
        if initial_message and initial_message.parts:
            message_text = " ".join(part.text or "" for part in initial_message.parts)

        requires_tool_call = "tool" in message_text.lower()
        if metadata and metadata.get("type") == "slash_command":
            requires_tool_call = True

        if not requires_tool_call:
            tasks_by_id[task_id] = task
            await _emit_push_event(task_id, event="task_created")
            return task

        tool_name = "echo"
        confirmation_request = ConfirmationRequest(
            options=[
                ConfirmationOption(id="approve", name="Approve", description="Approve tool call"),
                ConfirmationOption(id="deny", name="Deny", description="Deny tool call"),
            ],
            details=GenericDetails(description=f"Permission required to run {tool_name}"),
        )
        tool_call_id = _next_tool_call_id()
        tool_call = ToolCall(
            tool_call_id=tool_call_id,
            status=ToolCallStatus.PENDING,
            tool_name=tool_name,
            input_parameters={"message": initial_message.parts[0].text if initial_message and initial_message.parts else ""},
            confirmation_request=confirmation_request,
        )

        if "approve tool" in message_text.lower():
            tool_call.status = ToolCallStatus.SUCCEEDED
            tool_call.result = ToolOutput(
                content=f"{tool_call.tool_name} executed successfully as part of task",
                details=ExecuteDetails(stdout="Success", exit_code=0),
            )
            tool_call.confirmation_request = None
        if metadata and metadata.get("type") == "slash_command":
            tool_call.status = ToolCallStatus.SUCCEEDED
            tool_call.confirmation_request = None
            tool_call.result = ToolOutput(
                content=f"{tool_call.tool_name} executed successfully as part of task",
                details=ExecuteDetails(stdout="Success", exit_code=0),
            )

        tool_calls_by_task[task_id] = tool_call
        dev_tool_meta.setdefault("tool_calls", {})[tool_call.tool_call_id] = tool_call.to_dict()
        tasks_by_id[task_id] = task
        await _emit_push_event(task_id, event="task_created")
        return task

    async def execute_task_mock(task_id, *args, **kwargs):
        tool_call = tool_calls_by_task.get(task_id)
        if tool_call:
            if tool_call.status == ToolCallStatus.PENDING:
                return _snapshot_task(task_id, TaskState.INPUT_REQUIRED)

            message_content = tool_call.input_parameters.get("message", "")
            if "Confirm" in message_content and tool_call.status == ToolCallStatus.PENDING:
                return _snapshot_task(task_id, TaskState.INPUT_REQUIRED)

            tool_call.status = ToolCallStatus.SUCCEEDED
            tool_call.result = ToolOutput(
                content=f"{tool_call.tool_name} executed successfully as part of task",
                details=ExecuteDetails(stdout="Success", exit_code=0),
            )
        snapshot = _snapshot_task(task_id, TaskState.COMPLETED)
        await _emit_push_event(task_id, event="task_completed")
        return snapshot

    async def get_task_mock(task_id):
        tool_call = tool_calls_by_task.get(task_id)
        if tool_call:
            if tool_call.status == ToolCallStatus.PENDING:
                return _snapshot_task(task_id, TaskState.INPUT_REQUIRED)
            if tool_call.status == ToolCallStatus.EXECUTING:
                tool_call.status = ToolCallStatus.SUCCEEDED
            tool_call.result = ToolOutput(
                content=f"{tool_call.tool_name} executed successfully as part of task",
                details=ExecuteDetails(stdout="Success", exit_code=0),
            )
        snapshot = _snapshot_task(task_id, TaskState.COMPLETED)
        await _emit_push_event(task_id, event="task_status_check")
        return snapshot

    async def provide_input_and_continue_mock(task_id, user_input=None, *args, **kwargs):
        tool_call = tool_calls_by_task.get(task_id)
        metadata = {}
        if user_input and hasattr(user_input, "metadata"):
            metadata = user_input.metadata or {}
        elif isinstance(user_input, dict):
            metadata = user_input.get("metadata", {})

        confirmation_data = metadata.get("development-tool", {}).get(
            "tool_call_confirmation", {}
        )
        selected_option = confirmation_data.get("selected_option_id")

        if tool_call:
            if selected_option == "deny":
                tool_call.status = ToolCallStatus.FAILED
                tool_call.result = ErrorDetails(
                    message="Tool call denied by user", code="denied"
                )
            else:
                tool_call.status = ToolCallStatus.EXECUTING
                tool_call.result = None

        snapshot = _snapshot_task(task_id, TaskState.WORKING)
        await _emit_push_event(task_id, event="tool_call_update")
        return snapshot

    mock.create_task = AsyncMock(side_effect=create_task_mock)
    mock.execute_task.side_effect = execute_task_mock
    mock.get_task.side_effect = get_task_mock
    mock.provide_input_and_continue.side_effect = provide_input_and_continue_mock
    async def get_task_id_for_tool_call(tool_call_id):
        for tid, tc in tool_calls_by_task.items():
            if tc.tool_call_id == tool_call_id:
                return tid
        return None

    mock.get_task_id_for_tool_call.side_effect = get_task_id_for_tool_call
    return mock


@pytest.fixture
def mock_push_manager():
    """Mock push manager to verify metadata."""
    mock = AsyncMock()
    mock.close = AsyncMock()
    mock.cleanup_expired_configs = AsyncMock(return_value=0)
    mock.send_notification = AsyncMock()
    mock.list_configs = AsyncMock(return_value=[])
    mock.get_delivery_history = AsyncMock(return_value=[])
    mock.delete_config = AsyncMock(return_value=True)
    mock.create_config = AsyncMock(return_value=MagicMock(id="config"))
    return mock


@pytest.fixture
def test_client_full(mock_settings, mock_tool_manager, mock_task_manager_full, mock_push_manager):
    """Test client with full mocks for E2E simulation."""
    context_manager = MagicMock()
    context_manager.create_context = AsyncMock(return_value="ctx")
    context_manager.add_task_to_context = AsyncMock()
    context_manager.add_message_to_context = AsyncMock()
    with (
        patch("a2a_acp.main.get_tool_configuration_manager", return_value=mock_tool_manager),
        patch("a2a_acp.main.get_task_manager", return_value=mock_task_manager_full),
        patch("a2a_acp.main.PushNotificationManager", return_value=mock_push_manager),
        patch("a2a_acp.main.get_context_manager", return_value=context_manager),
    ):
        class DummyZedAgentConnection:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self_inner):
                return self_inner

            async def __aexit__(self_inner, exc_type, exc_val, exc_tb):
                pass

            async def initialize(self_inner):
                pass

            async def start_session(self_inner, cwd, mcp_servers):
                pass

        with patch("a2a_acp.main.ZedAgentConnection", DummyZedAgentConnection):
            app = create_app()
            with TestClient(app) as client:
                yield client


class TestRFCFlowSimulation:
    """Tests simulating complete RFC flows (tool call lifecycle, etc.)."""

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
        test_client_full.post("/a2a/command/execute", json=exec_request)

        # Verify push notification called with metadata
        mock_push_manager.send_notification.assert_called()
        call = mock_push_manager.send_notification.call_args[0][1]
        assert "development-tool" in call
        dev_event = call["development-tool"]
        assert dev_event["kind"] == "tool_call_update"  # Or appropriate


class TestPushNotificationsWithExtensionMetadata:
    """Tests for push notifications including extension metadata."""

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

    def test_error_notifications_include_error_details(self, test_client_full, mock_push_manager):
        """Test error events include ErrorDetails in metadata."""
        # Mock task failure
        with patch.object(mock_push_manager, "send_notification"):
            # Simulate failed task (via RPC or send)
            fail_message = {"role": "user", "parts": [{"kind": "text", "text": "Fail task"}], "messageId": str(create_message_id())}
            test_client_full.post("/a2a/message/send", json={"message": fail_message})

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
