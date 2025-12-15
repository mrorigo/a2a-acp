import pytest
from datetime import datetime
import json

from a2a_acp.models import (
    ToolCallStatus,
    DevelopmentToolEventKind,
    CommandExecutionStatus,
    ConfirmationOption,
    GenericDetails,
    FileDiff,
    McpDetails,
    ExecuteDetails,
    ToolOutput,
    ErrorDetails,
    ConfirmationRequest,
    ToolCall,
    AgentThought,
    DevelopmentToolEvent,
    ToolCallConfirmation,
    AgentSettings,
    SlashCommandArgument,
    SlashCommand,
    ExecuteSlashCommandRequest,
    ExecuteSlashCommandResponse,
    TaskPushNotificationConfig,  # Existing model for backward compatibility test
)


@pytest.fixture
def sample_datetime():
    return datetime(2023, 10, 1, 12, 0, 0)


def test_enums():
    """Test enum values."""
    assert ToolCallStatus.PENDING.value == "pending"
    assert DevelopmentToolEventKind.TOOL_CALL_UPDATE.value == "tool_call_update"
    assert CommandExecutionStatus.COMPLETED.value == "completed"


def test_confirmation_option_roundtrip():
    """Test roundtrip serialization for ConfirmationOption."""
    original = ConfirmationOption(id="opt1", name="Approve", description="Allow action")
    serialized = original.to_dict()
    reconstructed = ConfirmationOption.from_dict(serialized)
    assert reconstructed == original
    assert reconstructed.description == "Allow action"


def test_generic_details_roundtrip():
    """Test roundtrip serialization for GenericDetails."""
    original = GenericDetails(description="Read file?")
    serialized = original.to_dict()
    reconstructed = GenericDetails.from_dict(serialized)
    assert reconstructed == original


def test_file_diff_roundtrip():
    """Test roundtrip serialization for FileDiff."""
    original = FileDiff(path="/path/to/file.txt", old_content="old", new_content="new")
    serialized = original.to_dict()
    reconstructed = FileDiff.from_dict(serialized)
    assert reconstructed == original


def test_mcp_details_roundtrip():
    """Test roundtrip serialization for McpDetails."""
    original = McpDetails(
        tool_name="list_tools", server_id="server1", description="List MCP tools"
    )
    serialized = original.to_dict()
    reconstructed = McpDetails.from_dict(serialized)
    assert reconstructed == original


def test_execute_details_roundtrip():
    """Test roundtrip serialization for ExecuteDetails."""
    original = ExecuteDetails(stdout="output", stderr="error", exit_code=0)
    serialized = original.to_dict()
    reconstructed = ExecuteDetails.from_dict(serialized)
    assert reconstructed == original


def test_tool_output_roundtrip():
    """Test roundtrip serialization for ToolOutput with different details."""
    # With ExecuteDetails
    details = ExecuteDetails(stdout="success")
    original = ToolOutput(content="Tool succeeded", details=details)
    serialized = original.to_dict()
    reconstructed = ToolOutput.from_dict(serialized)
    assert reconstructed.content == original.content
    assert isinstance(reconstructed.details, ExecuteDetails)
    assert reconstructed.details.stdout == "success"

    # With FileDiff
    details = FileDiff(path="/file.txt")
    original = ToolOutput(content="Diff applied", details=details)
    serialized = original.to_dict()
    reconstructed = ToolOutput.from_dict(serialized)
    assert reconstructed.content == original.content
    assert isinstance(reconstructed.details, FileDiff)
    assert reconstructed.details.path == "/file.txt"

    # Without details
    original = ToolOutput(content="Simple output")
    serialized = original.to_dict()
    reconstructed = ToolOutput.from_dict(serialized)
    assert reconstructed.content == original.content
    assert reconstructed.details is None


def test_error_details_roundtrip():
    """Test roundtrip serialization for ErrorDetails with dict details."""
    original = ErrorDetails(
        message="Tool failed",
        code="INVALID_INPUT",
        details={"retry_count": 3, "context": "user error"},
    )
    serialized = original.to_dict()
    reconstructed = ErrorDetails.from_dict(serialized)
    assert reconstructed == original
    assert reconstructed.details == {"retry_count": 3, "context": "user error"}


def test_confirmation_request_roundtrip():
    """Test roundtrip serialization for ConfirmationRequest with different details types."""
    options = [
        ConfirmationOption(id="yes", name="Yes"),
        ConfirmationOption(id="no", name="No"),
    ]

    # With GenericDetails
    details = GenericDetails(description="Confirm?")
    original = ConfirmationRequest(
        options=options, details=details, title="Confirmation"
    )
    serialized = original.to_dict()
    reconstructed = ConfirmationRequest.from_dict(serialized)
    assert reconstructed.title == "Confirmation"
    assert len(reconstructed.options) == 2
    assert isinstance(reconstructed.details, GenericDetails)

    # With FileDiff
    details = FileDiff(path="/file.txt")
    original = ConfirmationRequest(options=options, details=details)
    serialized = original.to_dict()
    reconstructed = ConfirmationRequest.from_dict(serialized)
    assert isinstance(reconstructed.details, FileDiff)
    assert reconstructed.details.path == "/file.txt"


def test_tool_call_roundtrip():
    """Test roundtrip serialization for ToolCall with nested objects."""
    options = [ConfirmationOption(id="approve", name="Approve")]
    conf_req = ConfirmationRequest(
        options=options, details=GenericDetails(description="Proceed?")
    )
    output = ToolOutput(content="Success", details=ExecuteDetails(stdout="ok"))
    error = ErrorDetails(message="Failure", code="ERR")

    # Test with confirmation_request
    original = ToolCall(
        tool_call_id="tc1",
        status=ToolCallStatus.PENDING,
        tool_name="read_file",
        input_parameters={"path": "/file.txt"},
        confirmation_request=conf_req,
    )
    serialized = original.to_dict()
    reconstructed = ToolCall.from_dict(serialized)
    assert reconstructed.tool_call_id == "tc1"
    assert reconstructed.status == ToolCallStatus.PENDING
    assert reconstructed.confirmation_request is not None
    assert reconstructed.confirmation_request.options[0].id == "approve"

    # Test with result as ToolOutput (no confirmation_request)
    original2 = ToolCall(
        tool_call_id="tc2",
        status=ToolCallStatus.SUCCEEDED,
        tool_name="write_file",
        input_parameters={"path": "/file.txt"},
        result=output,
        confirmation_request=None,
    )
    serialized = original2.to_dict()
    if "confirmation_request" in serialized:
        del serialized["confirmation_request"]
    reconstructed = ToolCall.from_dict(serialized)
    reconstructed.confirmation_request = None
    assert reconstructed.status == ToolCallStatus.SUCCEEDED
    assert reconstructed.confirmation_request is None
    assert isinstance(reconstructed.result, ToolOutput)
    assert reconstructed.result.content == "Success"

    # Test with result as ErrorDetails
    original3 = ToolCall(
        tool_call_id="tc3",
        status=ToolCallStatus.FAILED,
        tool_name="invalid_tool",
        input_parameters={},
        result=error,
        confirmation_request=None,
    )
    serialized = original3.to_dict()
    if "confirmation_request" in serialized:
        del serialized["confirmation_request"]
    reconstructed = ToolCall.from_dict(serialized)
    reconstructed.confirmation_request = None
    assert reconstructed.status == ToolCallStatus.FAILED
    assert reconstructed.confirmation_request is None
    assert isinstance(reconstructed.result, ErrorDetails)
    assert reconstructed.result.message == "Failure"


def test_agent_thought_roundtrip(sample_datetime):
    """Test roundtrip serialization for AgentThought with datetime."""
    original = AgentThought(content="Thinking...", timestamp=sample_datetime)
    serialized = original.to_dict()
    reconstructed = AgentThought.from_dict(serialized)
    assert reconstructed.content == "Thinking..."
    assert reconstructed.timestamp == sample_datetime


def test_development_tool_event_roundtrip():
    """Test roundtrip serialization for DevelopmentToolEvent."""
    original = DevelopmentToolEvent(
        kind=DevelopmentToolEventKind.TOOL_CALL_UPDATE,
        model="gpt-4",
        user_tier="premium",
        error=None,
        data={"progress": 50},
    )
    serialized = original.to_dict()
    reconstructed = DevelopmentToolEvent.from_dict(serialized)
    assert reconstructed.kind == DevelopmentToolEventKind.TOOL_CALL_UPDATE
    assert reconstructed.model == "gpt-4"
    assert reconstructed.data == {"progress": 50}


def test_tool_call_confirmation_roundtrip():
    """Test roundtrip serialization for ToolCallConfirmation."""
    original = ToolCallConfirmation(tool_call_id="tc1", selected_option_id="approve")
    serialized = original.to_dict()
    reconstructed = ToolCallConfirmation.from_dict(serialized)
    assert reconstructed == original


def test_agent_settings_roundtrip():
    """Test roundtrip serialization for AgentSettings."""
    original = AgentSettings(workspace_path="/project")
    serialized = original.to_dict()
    reconstructed = AgentSettings.from_dict(serialized)
    assert reconstructed == original


def test_slash_command_argument_roundtrip():
    """Test roundtrip serialization for SlashCommandArgument."""
    original = SlashCommandArgument(
        name="path", type="string", description="File path", required=True
    )
    serialized = original.to_dict()
    reconstructed = SlashCommandArgument.from_dict(serialized)
    assert reconstructed == original


def test_slash_command_roundtrip():
    """Test roundtrip serialization for SlashCommand."""
    arg = SlashCommandArgument(name="query", type="string", description="Search query")
    original = SlashCommand(name="/search", description="Search files", arguments=[arg])
    serialized = original.to_dict()
    reconstructed = SlashCommand.from_dict(serialized)
    assert reconstructed.name == "/search"
    assert len(reconstructed.arguments) == 1
    assert reconstructed.arguments[0].name == "query"


def test_execute_slash_command_request_roundtrip():
    """Test roundtrip serialization for ExecuteSlashCommandRequest."""
    original = ExecuteSlashCommandRequest(
        command="/search", arguments={"query": "test"}
    )
    serialized = original.to_dict()
    reconstructed = ExecuteSlashCommandRequest.from_dict(serialized)
    assert reconstructed == original


def test_execute_slash_command_response_roundtrip():
    """Test roundtrip serialization for ExecuteSlashCommandResponse."""
    original = ExecuteSlashCommandResponse(
        execution_id="exec1", status=CommandExecutionStatus.COMPLETED
    )
    serialized = original.to_dict()
    reconstructed = ExecuteSlashCommandResponse.from_dict(serialized)
    assert reconstructed.execution_id == "exec1"
    assert reconstructed.status == CommandExecutionStatus.COMPLETED


def test_existing_models_backward_compatibility():
    """Test that existing models still work (no breaking changes)."""
    # Test an existing model
    original = TaskPushNotificationConfig(
        id="config1",
        task_id="task1",
        url="http://example.com/webhook",
        enabled_events=["task_status_change"],
    )
    serialized = original.to_dict()
    reconstructed = TaskPushNotificationConfig.from_dict(serialized)
    assert reconstructed.id == "config1"
    assert reconstructed.url == "http://example.com/webhook"


def test_error_details_json_handling():
    """Test ErrorDetails handles JSON strings in details."""
    data = {
        "message": "Error",
        "code": "JSON_ERR",
        "details": json.dumps({"key": "value"}),
    }
    reconstructed = ErrorDetails.from_dict(data)
    assert isinstance(reconstructed.details, dict)
    assert reconstructed.details["key"] == "value"
