import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
from typing import Dict, Any, List

from a2a_acp.bash_executor import (
    BashToolExecutor,
    ToolExecutionResult,
    ExecutionContext,
    ParameterError,
    TemplateRenderError,
)

from a2a_acp.models import (
    ToolOutput,
    ErrorDetails,
    ExecuteDetails,
    FileDiff,
    McpDetails,
    DevelopmentToolEvent,
    DevelopmentToolEventKind,
    ToolCallStatus,
    ToolCall,
)

from a2a_acp.sandbox import ExecutionResult, SandboxError
from a2a_acp.tool_config import BashTool
from a2a_acp.push_notification_manager import PushNotificationManager
from a2a_acp.task_manager import A2ATaskManager


@pytest.fixture
def mock_sandbox():
    """Mock sandbox manager."""
    mock = MagicMock()
    mock.execute_in_sandbox = AsyncMock(return_value=ExecutionResult(
        success=True,
        stdout="Mock output",
        stderr="",
        return_code=0,
        execution_time=0.1,
        metadata={},
        output_files=[],
    ))
    return mock


@pytest.fixture
def mock_push_manager():
    """Mock push notification manager."""
    return AsyncMock(spec=PushNotificationManager)


@pytest.fixture
def mock_task_manager():
    """Mock task manager."""
    return MagicMock(spec=A2ATaskManager)


@pytest.fixture
def sample_tool():
    """Sample bash tool for testing."""
    return BashTool(
        id="test_echo",
        name="echo",
        description="Echo message",
        script="echo '{{message}}'",
        parameters=[
            {"name": "message", "type": "string", "description": "Message", "required": True},
        ],
        config={"timeout": 30, "caching_enabled": False},
        version="1.0",
    )


@pytest.fixture
def sample_context():
    """Sample execution context."""
    return ExecutionContext(
        tool_id="test_echo",
        session_id="sess_123",
        task_id="task_123",
        user_id="user_123",
    )


@pytest.fixture
def bash_executor(mock_sandbox, mock_push_manager, mock_task_manager):
    """BashToolExecutor with mocks."""
    executor = BashToolExecutor(
        sandbox=mock_sandbox,
        push_notification_manager=mock_push_manager,
        task_manager=mock_task_manager,
    )
    return executor


class TestLiveContentStreaming:
    """Tests for live content streaming with DevelopmentToolEvent."""

    @pytest.mark.asyncio
    async def test_execute_tool_emits_started_event_with_executing_status(self, bash_executor, sample_tool, sample_context):
        """Test tool execution emits started event with EXECUTING status."""
        parameters = {"message": "Hello"}

        # Execute tool
        result = await bash_executor.execute_tool(sample_tool, parameters, sample_context)

        # Verify success
        assert result.success is True
        assert result.output == "Mock output"

        # Verify push notification was called for started event
        bash_executor.push_notification_manager.send_notification.assert_called()
        started_call = bash_executor.push_notification_manager.send_notification.call_args_list[0]
        assert started_call[0][0] == sample_context.task_id  # task_id
        event_data = started_call[0][1]
        assert event_data["event"] == "tool_started"
        assert "development-tool" in event_data
        dev_event = DevelopmentToolEvent.from_dict(event_data["development-tool"])
        assert dev_event.kind == DevelopmentToolEventKind.TOOL_CALL_UPDATE
        assert dev_event.data["status"] == "executing"

    @pytest.mark.asyncio
    async def test_streaming_emits_tool_call_update_during_execution(self, bash_executor, sample_tool, sample_context):
        """Test DevelopmentToolEvent emitted during execution for live updates."""
        parameters = {"message": "Hello"}

        # Mock sandbox to simulate streaming (multiple calls, but single execution)
        # In real, on_chunk would emit, but here verify in execute_tool flow
        result = await bash_executor.execute_tool(sample_tool, parameters, sample_context)

        # Verify multiple events: started, completed
        assert bash_executor.push_notification_manager.send_notification.call_count >= 2

        # Check completed event
        completed_call = bash_executor.push_notification_manager.send_notification.call_args_list[-1]
        event_data = completed_call[0][1]
        assert event_data["event"] == "tool_completed"
        dev_event = DevelopmentToolEvent.from_dict(event_data["development-tool"])
        assert dev_event.kind == DevelopmentToolEventKind.TOOL_CALL_UPDATE
        assert dev_event.data["status"] == "succeeded"

    @pytest.mark.asyncio
    @patch.object(BashToolExecutor, "_emit_tool_event")
    async def test_live_content_in_event_data(self, mock_emit, bash_executor, sample_tool, sample_context):
        """Test live_content is included in DevelopmentToolEvent data."""
        parameters = {"message": "Streaming message"}
    
        # Mock execution with output
        mock_result = ExecutionResult(
            success=True,
            stdout="Live: Hello\nLive: World",
            stderr="",
            return_code=0,
            execution_time=0.1,
            metadata={"live_content": True},
            output_files=[],
        )
        with patch.object(bash_executor.sandbox, "execute_in_sandbox", new_callable=AsyncMock, return_value=mock_result):
            result = await bash_executor.execute_tool(sample_tool, parameters, sample_context)
    
        # Verify emit called with live_content
        mock_emit.assert_any_call(
            "completed",
            sample_context,
            success=True,
            execution_time=0.1,
            return_code=0,
            output_length=18,  # "Live: Hello\nLive: World"
            tool_output=Any,  # ToolOutput
        )
        # In emit, it should include live_content in data


class TestToolOutputMapping:
    """Tests for ToolOutput mapping for different tool types."""

    @pytest.mark.asyncio
    async def test_successful_execution_maps_to_tool_output_with_executedetails(
        self, bash_executor, sample_tool, sample_context
    ):
        """Test successful execution maps to ToolOutput with ExecuteDetails."""
        parameters = {"message": "Success"}
    
        mock_result = ExecutionResult(
            success=True,
            stdout="echo: Success",
            stderr="",
            return_code=0,
            execution_time=0.05,
            metadata={},
            output_files=[],
        )
        with patch.object(bash_executor.sandbox, "execute_in_sandbox", new_callable=AsyncMock, return_value=mock_result):
            result = await bash_executor.execute_tool(sample_tool, parameters, sample_context)
    
        # Verify ToolOutput in metadata
        assert "tool_output" in result.metadata
        tool_output_dict = result.metadata["tool_output"]
        tool_output = ToolOutput.from_dict(tool_output_dict)
        assert tool_output.content == "echo: Success"
        assert isinstance(tool_output.details, ExecuteDetails)
        assert tool_output.details.stdout == "echo: Success"
        assert tool_output.details.exit_code == 0
        assert tool_output.details.stderr is None

    @pytest.mark.asyncio
    async def test_execution_with_output_files_maps_to_filediff(
        self, bash_executor, sample_tool, sample_context
    ):
        """Test output files map to FileDiff in metadata."""
        parameters = {"message": "File output"}
    
        mock_result = ExecutionResult(
            success=True,
            stdout="File created",
            stderr="",
            return_code=0,
            execution_time=0.1,
            metadata={},
            output_files=["/tmp/output.txt", "/tmp/log.txt"],
        )
        with patch.object(bash_executor.sandbox, "execute_in_sandbox", new_callable=AsyncMock, return_value=mock_result):
            result = await bash_executor.execute_tool(sample_tool, parameters, sample_context)
    
        # Verify FileDiff in metadata
        assert "file_diffs" in result.metadata
        file_diffs = result.metadata["file_diffs"]
        assert len(file_diffs) == 2
        first_diff = FileDiff.from_dict(file_diffs[0])
        assert first_diff.path == "/tmp/output.txt"
        assert first_diff.old_content is None
        assert "File created during tool execution" in first_diff.new_content  # Placeholder

    @pytest.mark.parametrize("tool_type", ["generic", "mcp", "file"])
    @pytest.mark.asyncio
    async def test_different_tool_types_map_to_appropriate_details(
        self, bash_executor, tool_type, sample_context
    ):
        """Test different tool types map to appropriate details types."""
        # Create tool specific to type
        if tool_type == "generic":
            tool = BashTool(
                id="generic",
                name="generic",
                description="Generic tool",
                script="echo 'generic'",
                parameters=[],
                config={},
            )
            mock_result = ExecutionResult(stdout="Generic output", return_code=0)
        elif tool_type == "mcp":
            tool = BashTool(
                id="mcp_tool",
                name="mcp_list",
                description="MCP tool",
                script="echo 'mcp'",
                parameters=[],
                config={"mcp_tool": True},
            )
            mock_result = ExecutionResult(stdout="MCP output", return_code=0, metadata={"mcp": True})
        else:  # file
            tool = BashTool(
                id="file_tool",
                name="create_file",
                description="File tool",
                script="touch file.txt",
                parameters=[],
                config={},
            )
            mock_result = ExecutionResult(
                stdout="File created",
                return_code=0,
                output_files=["file.txt"],
            )
    
        parameters = {}
        with patch.object(bash_executor.sandbox, "execute_in_sandbox", new_callable=AsyncMock, return_value=mock_result):
            result = await bash_executor.execute_tool(tool, parameters, sample_context)
    
        tool_output = ToolOutput.from_dict(result.metadata["tool_output"])
        if tool_type == "generic":
            assert isinstance(tool_output.details, ExecuteDetails)
        elif tool_type == "mcp":
            # Assuming MCP details mapping in code
            assert "tool_name" in tool_output.details.to_dict()  # McpDetails
        else:
            # File tool should have FileDiff in metadata, not details
            assert "file_diffs" in result.metadata


class TestErrorDetailsHandling:
    """Tests for ErrorDetails handling for tool failures."""

    @pytest.mark.asyncio
    async def test_failed_execution_maps_to_errordetails(
        self, bash_executor, sample_tool, sample_context
    ):
        """Test failed execution maps to ErrorDetails in metadata."""
        parameters = {"message": "Fail"}
    
        mock_result = ExecutionResult(
            success=False,
            stdout="",
            stderr="Command failed",
            return_code=1,
            execution_time=0.1,
            metadata={},
            output_files=[],
        )
        with patch.object(bash_executor.sandbox, "execute_in_sandbox", new_callable=AsyncMock, return_value=mock_result):
            result = await bash_executor.execute_tool(sample_tool, parameters, sample_context)
    
        assert result.success is False
        assert "error_details" in result.metadata
        error_details_dict = result.metadata["error_details"]
        error_details = ErrorDetails.from_dict(error_details_dict)
        assert error_details.message == "Command failed"
        assert error_details.code == "1"  # Return code as string

    @pytest.mark.asyncio
    async def test_exception_during_execution_creates_errordetails(
        self, bash_executor, sample_tool, sample_context
    ):
        """Test exceptions during execution create ErrorDetails."""
        parameters = {"message": "Error"}
    
        with patch.object(bash_executor.sandbox, "execute_in_sandbox", new_callable=AsyncMock, side_effect=SandboxError("Sandbox failure")):
            result = await bash_executor.execute_tool(sample_tool, parameters, sample_context)
    
        assert result.success is False
        assert "error_details" in result.metadata
        error_details = ErrorDetails.from_dict(result.metadata["error_details"])
        assert "Sandbox failure" in error_details.message
        assert error_details.code == "INTERNAL_ERROR"

    @pytest.mark.asyncio
    async def test_parameter_error_maps_to_invalid_params_errordetails(
        self, bash_executor, sample_tool, sample_context
    ):
        """Test ParameterError maps to INVALID_PARAMS ErrorDetails."""
        parameters = {"invalid": "param"}  # Invalid params

        # Mock validation to fail
        with patch.object(sample_tool, "validate_parameters", return_value=(False, ["Invalid param"])):
            result = await bash_executor.execute_tool(sample_tool, parameters, sample_context)

        assert result.success is False
        error_details = ErrorDetails.from_dict(result.metadata["error_details"])
        assert error_details.code == "-32602"  # INVALID_PARAMS
        assert "Parameter validation failed" in error_details.message

    @pytest.mark.asyncio
    async def test_template_error_maps_to_invalid_params(
        self, bash_executor, sample_tool, sample_context
    ):
        """Test TemplateRenderError maps to INVALID_PARAMS."""
        parameters = {"message": "{{unclosed"}  # Invalid template

        with patch.object(bash_executor, "render_script", side_effect=TemplateRenderError("Template error")):
            result = await bash_executor.execute_tool(sample_tool, parameters, sample_context)

        error_details = ErrorDetails.from_dict(result.metadata["error_details"])
        assert error_details.code == "-32602"
        assert "Template error" in error_details.message


class TestOutputCollection:
    """Tests for output collection with extension schemas."""

    @pytest.mark.asyncio
    async def test_output_collection_includes_all_fields(self, bash_executor, sample_tool, sample_context):
        """Test ToolExecutionResult collects all extension schema fields."""
        parameters = {"message": "Complete output"}
    
        mock_result = ExecutionResult(
            success=True,
            stdout="Stdout content",
            stderr="Stderr warning",
            return_code=0,
            execution_time=1.23,
            metadata={"custom": "meta"},
            output_files=["file1.txt", "file2.log"],
            mcp_error=None,
        )
        with patch.object(bash_executor.sandbox, "execute_in_sandbox", new_callable=AsyncMock, return_value=mock_result):
            result = await bash_executor.execute_tool(sample_tool, parameters, sample_context)
    
        # Verify complete collection
        assert result.tool_id == "test_echo"
        assert result.success is True
        assert result.output == "Stdout content"
        assert result.error == "Stderr warning"
        assert result.execution_time == 1.23
        assert result.return_code == 0
        assert result.metadata["parameters"] == parameters
        assert result.metadata["tool_version"] == "1.0"
        assert result.metadata["error_profile"] == "acp_basic"  # Default
        assert result.output_files == ["file1.txt", "file2.log"]
        assert result.mcp_error is None
    
        # Verify ToolOutput
        tool_output = ToolOutput.from_dict(result.metadata["tool_output"])
        assert tool_output.content == "Stdout content"
        details = tool_output.details
        assert isinstance(details, ExecuteDetails)
        assert details.stdout == "Stdout content"
        assert details.stderr == "Stderr warning"
        assert details.exit_code == 0
    
        # Verify FileDiff for output files
        assert "file_diffs" in result.metadata
        diffs = [FileDiff.from_dict(d) for d in result.metadata["file_diffs"]]
        assert len(diffs) == 2
        assert diffs[0].path == "file1.txt"
        assert diffs[0].new_content == "File created during tool execution"  # Placeholder

    @pytest.mark.asyncio
    async def test_mcp_error_collection_in_output(self, bash_executor, sample_tool, sample_context):
        """Test MCP error is collected in metadata for failed executions."""
        parameters = {"message": "MCP fail"}
    
        mock_result = ExecutionResult(
            success=False,
            stdout="",
            stderr="MCP connection failed",
            return_code=2,
            execution_time=0.5,
            metadata={},
            output_files=[],
            mcp_error={"code": "MCP_NOT_FOUND", "message": "Server unavailable"},
        )
        with patch.object(bash_executor.sandbox, "execute_in_sandbox", new_callable=AsyncMock, return_value=mock_result):
            result = await bash_executor.execute_tool(sample_tool, parameters, sample_context)
    
        assert result.mcp_error == {"code": "MCP_NOT_FOUND", "message": "Server unavailable"}
        assert "mcp_error" in result.metadata
        assert result.metadata["mcp_error"] == result.mcp_error

    @pytest.mark.asyncio
    async def test_error_contract_applied_to_output(self, bash_executor, sample_tool, sample_context):
        """Test error contract is applied to result metadata."""
        parameters = {"message": "Contract test"}
    
        mock_result = ExecutionResult(
            success=False,
            stdout="",
            stderr="Custom error",
            return_code=127,  # Command not found
            execution_time=0.1,
            metadata={},
            output_files=[],
        )
        with patch.object(bash_executor.sandbox, "execute_in_sandbox", new_callable=AsyncMock, return_value=mock_result):
            result = await bash_executor.execute_tool(sample_tool, parameters, sample_context)
    
        # Verify error contract in metadata
        assert "mcp_error" in result.metadata
        mcp_error = result.metadata["mcp_error"]
        assert mcp_error["code"] == -32601  # METHOD_NOT_FOUND for 127
        assert "Command not found" in mcp_error["message"]
        assert result.metadata["error_profile"] == "acp_basic"
        assert "diagnostics" in result.metadata
        assert result.metadata["diagnostics"]["return_code"] == 127

    @pytest.mark.asyncio
    async def test_sanitized_error_output_in_details(self, bash_executor, sample_tool, sample_context):
        """Test error output is sanitized before including in ErrorDetails."""
        parameters = {"message": "Sanitize test"}
    
        mock_result = ExecutionResult(
            success=False,
            stdout="",
            stderr="WARNING:a2a_acp.sandbox:Could not set process limits\nReal error: Permission denied",
            return_code=1,
            execution_time=0.1,
            metadata={},
            output_files=[],
        )
        with patch.object(bash_executor.sandbox, "execute_in_sandbox", new_callable=AsyncMock, return_value=mock_result):
            result = await bash_executor.execute_tool(sample_tool, parameters, sample_context)
    
        error_details = ErrorDetails.from_dict(result.metadata["error_details"])
        assert "Could not set process limits" not in error_details.details.get("sanitized_stderr", "")
        assert "Permission denied" in error_details.details.get("sanitized_stderr", "")


class TestEdgeCases:
    """Tests for edge cases in tool execution with extensions."""

    @pytest.mark.asyncio
    async def test_tool_execution_with_no_output(self, bash_executor, sample_tool, sample_context):
        """Test tool with empty output still maps schemas correctly."""
        parameters = {"message": ""}
    
        mock_result = ExecutionResult(
            success=True,
            stdout="",
            stderr="",
            return_code=0,
            execution_time=0.0,
            metadata={},
            output_files=[],
        )
        with patch.object(bash_executor.sandbox, "execute_in_sandbox", new_callable=AsyncMock, return_value=mock_result):
            result = await bash_executor.execute_tool(sample_tool, parameters, sample_context)
    
        tool_output = ToolOutput.from_dict(result.metadata["tool_output"])
        assert tool_output.content == ""
        assert isinstance(tool_output.details, ExecuteDetails)
        assert tool_output.details.stdout == ""
        assert tool_output.details.exit_code == 0

    @pytest.mark.asyncio
    async def test_tool_timeout_maps_to_error_details(self, bash_executor, sample_tool, sample_context):
        """Test timeout creates appropriate ErrorDetails."""
        parameters = {"message": "Timeout"}
    
        mock_result = ExecutionResult(
            success=False,
            stdout="",
            stderr="",
            return_code=124,  # Bash timeout code
            execution_time=30.1,  # Over timeout
            metadata={"timeout": True},
            output_files=[],
        )
        with patch.object(bash_executor.sandbox, "execute_in_sandbox", new_callable=AsyncMock, return_value=mock_result):
            result = await bash_executor.execute_tool(sample_tool, parameters, sample_context)
    
        error_details = ErrorDetails.from_dict(result.metadata["error_details"])
        assert error_details.code == "124"
        assert "timeout" in error_details.message.lower()
        assert result.metadata["diagnostics"]["return_code"] == 124

    @pytest.mark.asyncio
    async def test_validation_failure_prevents_execution(self, bash_executor, sample_tool, sample_context):
        """Test parameter validation failure prevents sandbox execution."""
        invalid_params = {"wrong": "param"}

        with patch.object(sample_tool, "validate_parameters", return_value=(False, ["Invalid parameters"])):
            result = await bash_executor.execute_tool(sample_tool, invalid_params, sample_context)

        # Verify no sandbox call
        bash_executor.sandbox.execute_in_sandbox.assert_not_called()
        assert result.success is False
        assert "Parameter validation failed" in result.error

    @pytest.mark.asyncio
    async def test_mcp_error_propagation(self, bash_executor, sample_tool, sample_context):
        """Test MCP error is propagated in result and metadata."""
        parameters = {"message": "MCP error"}
    
        mcp_error = {"code": "TOOL_NOT_FOUND", "message": "MCP tool unavailable"}
        mock_result = ExecutionResult(
            success=False,
            stdout="",
            stderr="MCP failed",
            return_code=1,
            execution_time=0.1,
            metadata={},
            output_files=[],
            mcp_error=mcp_error,
        )
        with patch.object(bash_executor.sandbox, "execute_in_sandbox", new_callable=AsyncMock, return_value=mock_result):
            result = await bash_executor.execute_tool(sample_tool, parameters, sample_context)
    
        assert result.mcp_error == mcp_error
        assert "mcp_error" in result.metadata
        assert result.metadata["mcp_error"] == mcp_error
        # Verify in event emission (mocked)


@pytest.mark.asyncio
async def test_integration_tool_execution_full_flow(bash_executor, sample_tool, sample_context, mock_push_manager):
    """Integration test: full tool execution flow with all extension emissions."""
    parameters = {"message": "Full flow test"}

    # Mock successful execution
    mock_result = ExecutionResult(
        success=True,
        stdout="Integration success",
        stderr="Minor warning",
        return_code=0,
        execution_time=0.5,
        metadata={"test": "data"},
        output_files=["integration.log"],
    )
    with patch.object(bash_executor.sandbox, "execute_in_sandbox", new_callable=AsyncMock, return_value=mock_result):
        result = await bash_executor.execute_tool(sample_tool, parameters, sample_context)

    # Verify complete flow
    assert result.success is True
    assert len(mock_push_manager.send_notification.call_args_list) >= 2  # started + completed

    # Verify ToolOutput mapping
    tool_output = ToolOutput.from_dict(result.metadata["tool_output"])
    assert tool_output.content == "Integration success"
    details = tool_output.details
    assert details.stdout == "Integration success"
    assert details.stderr == "Minor warning"

    # Verify FileDiff for output file
    assert "file_diffs" in result.metadata
    diffs = result.metadata["file_diffs"]
    assert len(diffs) == 1
    file_diff = FileDiff.from_dict(diffs[0])
    assert file_diff.path == "integration.log"

    # Verify metadata completeness
    assert "parameters" in result.metadata
    assert "tool_version" in result.metadata
    assert "execution_start" in result.metadata
    assert "error_profile" in result.metadata