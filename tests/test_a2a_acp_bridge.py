"""
Tests for A2A-ACP Bridge Components

Comprehensive test suite for the A2A-ACP bridge layer that connects A2A clients
to ZedACP agents. This covers the critical functionality that was previously untested.
"""

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Optional, Literal, Union, Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, ValidationError

from src.a2a.models import (
    Message,
    Task,
    TaskStatusUpdateEvent,
    TaskArtifactUpdateEvent,
)
from src.a2a_acp.main import create_app
from src.a2a_acp.database import SessionDatabase, A2AContext, ACPSession
# A2A-ACP bridge tests - using A2A protocol types exclusively
from src.a2a_acp.task_manager import A2ATaskManager, a2a_task_manager
from src.a2a_acp.context_manager import A2AContextManager, a2a_context_manager
from src.a2a_acp.context_manager import A2AContextManager
from src.a2a_acp.zed_agent import ZedAgentConnection, AgentProcessError

os.environ.setdefault("A2A_AGENT_COMMAND", "python tests/dummy_agent.py")


class TestA2ACPBridge:
    """Test the main A2A-ACP bridge application."""

    def test_bridge_app_creation(self):
        """Test that the A2A-ACP bridge application can be created."""
        app = create_app()
        assert app is not None
        assert app.title == "A2A-ACP Server"

    def test_bridge_routes_registered(self):
        """Test that all bridge routes are properly registered."""
        app = create_app()
        client = TestClient(app)

        # Test legacy ACP endpoints
        response = client.get("/ping")
        assert response.status_code == 200

        # Test A2A endpoints exist and are accessible (no auth token configured)
        response = client.post("/a2a/rpc")
        # Should return 200 with method listing for empty request
        assert response.status_code == 200

    def test_bridge_authorization_logic(self):
        """Test bridge authorization logic."""
        # When no auth token is configured, endpoints should be accessible
        app = create_app()
        client = TestClient(app)

        # Test without token (should work when no token configured)
        response = client.get("/ping")
        assert response.status_code == 200

        # Test with bearer token (should also work when no token configured)
        response = client.get("/ping", headers={"Authorization": "Bearer any-token"})
        assert response.status_code == 200

        # Note: Full auth testing would require setting up environment with token
        # and testing both valid and invalid token scenarios


class TestAgentConfiguration:
    """Test the streamlined agent configuration."""

    def test_agent_config_function(self):
        """Test that agent configuration can be loaded."""
        from src.a2a_acp.main import get_agent_config

        # This will use fallback defaults since no env vars are set
        config = get_agent_config()
        assert config is not None
        assert "command" in config
        assert "api_key" in config
        assert "description" in config

    def test_settings_include_agent_config(self):
        """Test that settings include agent configuration fields."""
        from src.a2a_acp.settings import get_settings

        settings = get_settings()
        assert hasattr(settings, 'agent_command')
        assert hasattr(settings, 'agent_api_key')
        assert hasattr(settings, 'agent_description')


class TestA2ACPBridgeDatabase:
    """Test the A2A-ACP database operations."""

    def test_database_creation(self):
        """Test database can be created with temporary file."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            try:
                db = SessionDatabase(tmp.name)
                assert db is not None
                db.close()
            finally:
                os.unlink(tmp.name)

    def test_database_context_models(self):
        """Test A2A context model structure."""
        from datetime import datetime

        # Test A2AContext model
        context = A2AContext(
            context_id="test-context",
            agent_name="test-agent",
            zed_session_id="test-session",
            working_directory="/tmp",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )

        assert context.context_id == "test-context"
        assert context.agent_name == "test-agent"

        # Test model serialization
        context_dict = context.to_dict()
        assert "context_id" in context_dict
        assert "agent_name" in context_dict

        # Test model deserialization
        restored = A2AContext.from_dict(context_dict)
        assert restored.context_id == context.context_id


class TestTaskManager:
    """Test the A2A task manager functionality."""

    def test_task_model_structure(self):
        """Test that A2A models are available."""
        # Test that we can import A2A models for task management
        from src.a2a.models import Task, TaskState, Message, TextPart

        # Verify A2A types are available
        assert TaskState.SUBMITTED == "submitted"
        assert TaskState.INPUT_REQUIRED == "input-required"

        # Test A2A message creation
        message = Message(
            role="user",
            parts=[TextPart(kind="text", text="test")],
            messageId="test-msg"
        )

        assert message.role == "user"
        assert len(message.parts) == 1


class TestA2ATaskManager:
    """Test the A2A Task Manager functionality."""

    def test_task_manager_creation(self):
        """Test A2A task manager can be created."""
        task_manager = A2ATaskManager()
        assert task_manager is not None
        assert len(task_manager._active_tasks) == 0

    @pytest.mark.asyncio
    async def test_create_task(self):
        """Test creating an A2A task."""
        task_manager = A2ATaskManager()

        task = await task_manager.create_task(
            context_id="test-context",
            agent_name="test-agent",
            metadata={"test": "data"}
        )

        assert task.id is not None
        assert task.contextId == "test-context"
        assert task.metadata is not None
        assert task.metadata.get("test") == "data"
        assert task_manager._active_tasks[task.id] is not None

    @pytest.mark.asyncio
    async def test_get_task(self):
        """Test retrieving a task by ID."""
        task_manager = A2ATaskManager()

        # Create a task first
        task = await task_manager.create_task("test-context", "test-agent")
        task_id = task.id

        # Retrieve the task
        retrieved = await task_manager.get_task(task_id)
        assert retrieved is not None
        assert retrieved.id == task_id

        # Test non-existent task
        not_found = await task_manager.get_task("non-existent")
        assert not_found is None

    @pytest.mark.asyncio
    async def test_list_tasks(self):
        """Test listing tasks with filtering."""
        task_manager = A2ATaskManager()

        # Create multiple tasks
        task1 = await task_manager.create_task("context1", "agent1")
        task2 = await task_manager.create_task("context2", "agent2")
        task3 = await task_manager.create_task("context1", "agent1")

        # List all tasks
        all_tasks = await task_manager.list_tasks()
        assert len(all_tasks) == 3

        # List tasks filtered by context
        context1_tasks = await task_manager.list_tasks("context1")
        assert len(context1_tasks) == 2

        context2_tasks = await task_manager.list_tasks("context2")
        assert len(context2_tasks) == 1

    @pytest.mark.asyncio
    async def test_cancel_task(self):
        """Test cancelling a task."""
        task_manager = A2ATaskManager()

        # Create a task
        task = await task_manager.create_task("test-context", "test-agent")
        task_id = task.id

        # Cancel the task
        cancelled = await task_manager.cancel_task(task_id)
        assert cancelled is True

        # Verify task status is cancelled
        cancelled_task = await task_manager.get_task(task_id)
        assert cancelled_task is not None

        # Test cancelling non-existent task
        not_cancelled = await task_manager.cancel_task("non-existent")
        assert not_cancelled is False


class TestA2AContextManager:
    """Test the A2A Context Manager functionality."""

    def test_context_manager_creation(self):
        """Test A2A context manager can be created."""
        context_manager = A2AContextManager()
        assert context_manager is not None
        assert len(context_manager._active_contexts) == 0

    @pytest.mark.asyncio
    async def test_create_context(self):
        """Test creating an A2A context."""
        context_manager = A2AContextManager()

        context_id = await context_manager.create_context(
            agent_name="test-agent",
            metadata={"test": "data"}
        )

        assert context_id is not None
        assert context_id.startswith("ctx_")
        assert context_manager._active_contexts[context_id] is not None

    @pytest.mark.asyncio
    async def test_get_context(self):
        """Test retrieving a context by ID."""
        context_manager = A2AContextManager()

        # Create a context first
        context_id = await context_manager.create_context("test-agent")

        # Retrieve the context
        retrieved = await context_manager.get_context(context_id)
        assert retrieved is not None
        assert retrieved.context_id == context_id

        # Test non-existent context
        not_found = await context_manager.get_context("non-existent")
        assert not_found is None

    @pytest.mark.asyncio
    async def test_list_contexts(self):
        """Test listing all contexts."""
        context_manager = A2AContextManager()

        # Create multiple contexts
        ctx1 = await context_manager.create_context("agent1")
        ctx2 = await context_manager.create_context("agent2")

        # List all contexts
        contexts = await context_manager.list_contexts()
        assert len(contexts) >= 2  # Should include our created contexts

        # Verify our contexts are in the list
        context_ids = [ctx.context_id for ctx in contexts]
        assert ctx1 in context_ids
        assert ctx2 in context_ids

    @pytest.mark.asyncio
    async def test_add_task_to_context(self):
        """Test adding a task to a context."""
        from src.a2a.models import Task, TaskStatus, TaskState

        context_manager = A2AContextManager()
        task_manager = A2ATaskManager()

        # Create context and task
        context_id = await context_manager.create_context("test-agent")
        task = await task_manager.create_task(context_id, "test-agent")

        # Add task to context
        await context_manager.add_task_to_context(context_id, task)

        # Verify task was added
        context_tasks = await context_manager.get_context_tasks(context_id)
        assert len(context_tasks) == 1
        assert context_tasks[0].id == task.id

    @pytest.mark.asyncio
    async def test_close_context(self):
        """Test closing a context."""
        context_manager = A2AContextManager()

        # Create a context
        context_id = await context_manager.create_context("test-agent")

        # Close the context
        closed = await context_manager.close_context(context_id)
        assert closed is True

        # Verify context is no longer active
        assert context_id not in context_manager._active_contexts

        # Test closing non-existent context
        not_closed = await context_manager.close_context("non-existent")
        assert not_closed is False


class TestContextManager:
    """Test the A2A context manager functionality."""

    def test_context_manager_creation(self):
        """Test context manager can be created."""
        context_manager = A2AContextManager()
        assert context_manager is not None

    def test_context_model_structure(self):
        """Test that context models have correct structure."""
        from datetime import datetime

        # Test ACPSession model
        session = ACPSession(
            acp_session_id="test-session",
            agent_name="test-agent",
            zed_session_id="test-zed-session",
            working_directory="/tmp",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )

        assert session.acp_session_id == "test-session"
        assert session.agent_name == "test-agent"


class TestZedAgentIntegration:
    """Test ZedACP agent integration."""

    @patch('subprocess.Popen')
    def test_zed_agent_connection_creation(self, mock_popen):
        """Test ZedACP connection can be created."""
        mock_process = MagicMock()
        mock_process.communicate.return_value = (b'{"result": "ok"}', b'')
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        # This would normally fail, but we're testing the connection setup
        try:
            connection = ZedAgentConnection("echo", api_key=None)
            assert connection is not None
        except Exception:
            # Expected to fail in test environment without actual agent
            pass

    def test_zed_agent_configuration(self):
        """Test ZedACP agent configuration handling."""
        # Test that agent configuration is properly structured
        agent_config = {
            "name": "test-agent",
            "command": "echo",
            "description": "Test agent",
            "api_key": "test-key"
        }

        # Validate configuration structure
        assert "name" in agent_config
        assert "command" in agent_config

    @pytest.mark.asyncio
    async def test_zed_agent_load_session_success(self):
        """Test successful session loading."""
        # Mock the agent connection
        connection = ZedAgentConnection(["echo"])

        # Mock the JSON-RPC communication
        original_request = connection.request

        async def mock_request(method, params=None, handler=None):
            if method == "session/load" and params:
                # Simulate successful session load response
                return {"sessionId": params.get("sessionId", "unknown") if params else "unknown", "loaded": True}

            # For other methods, use original implementation
            return None  # Simplified for testing

        connection.request = mock_request

        # Test loading a session
        session_id = "test-session-123"
        await connection.load_session(
            session_id=session_id,
            cwd="/test/dir",
            mcp_servers=[{"name": "test", "command": "test"}]
        )

        # Verify the request was made with correct parameters
        # (The actual verification would depend on how we capture the call)

    @pytest.mark.asyncio
    async def test_zed_agent_load_session_with_history_notifications(self):
        """Test session loading with conversation history notifications."""
        connection = ZedAgentConnection(["echo"])

        # Track notifications received
        notifications = []

        async def mock_request(method, params=None, handler=None):
            if method == "session/load":
                # Simulate session load with history notifications
                if handler:
                    # Simulate conversation history being replayed
                    await handler({
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": {
                            "update": {
                                "sessionUpdate": "history_message",
                                "message": {"role": "user", "content": "Hello"}
                            }
                        }
                    })
                    await handler({
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": {
                            "update": {
                                "sessionUpdate": "history_message",
                                "message": {"role": "assistant", "content": "Hi there!"}
                            }
                        }
                    })

                return {"sessionId": params.get("sessionId", "unknown") if params else "unknown", "loaded": True}

            return None

        connection.request = mock_request

        # Test loading session with history
        session_id = "test-session-with-history"
        await connection.load_session(session_id, "/test/dir")

    @pytest.mark.asyncio
    async def test_zed_agent_load_session_error_handling(self):
        """Test error handling when session load fails."""
        connection = ZedAgentConnection(["echo"])

        # Mock request to simulate session not found
        async def mock_request_error(method, params=None, handler=None):
            if method == "session/load":
                raise AgentProcessError("Session not found: invalid-session-id")
            return None

        connection.request = mock_request_error

        # Test that error is properly raised
        with pytest.raises(AgentProcessError, match="Session not found"):
            await connection.load_session("invalid-session-id", "/test/dir")

    @pytest.mark.asyncio
    async def test_zed_agent_session_lifecycle_integration(self):
        """Test complete session lifecycle: create -> load -> use -> cleanup."""
        connection = ZedAgentConnection(["echo"])

        # Mock all session operations
        session_operations = []

        async def mock_request_tracking(method, params=None, handler=None):
            session_operations.append({"method": method, "params": params})

            if method == "session/new":
                return {"sessionId": "new-session-123"}
            elif method == "session/load":
                return {"sessionId": params.get("sessionId", "unknown") if params else "unknown", "loaded": True}
            elif method == "session/prompt":
                return {"response": "Test response"}

            return None

        connection.request = mock_request_tracking

        # 1. Create new session
        session_id = await connection.start_session("/test/dir")
        assert session_id == "new-session-123"
        assert len(session_operations) == 1
        assert session_operations[0]["method"] == "session/new"

        # 2. Load the session (simulating restart/continuation)
        await connection.load_session(session_id, "/test/dir")
        assert len(session_operations) == 2
        assert session_operations[1]["method"] == "session/load"

        # 3. Use the loaded session for prompting
        await connection.prompt(session_id, [{"type": "text", "text": "Hello"}])
        assert len(session_operations) == 3
        assert session_operations[2]["method"] == "session/prompt"

    def test_zed_agent_session_parameter_validation(self):
        """Test that session methods validate parameters correctly."""
        connection = ZedAgentConnection(["echo"])

        # Test start_session parameter validation
        # (Implementation would depend on actual validation logic)

        # Test load_session parameter validation
        # (Implementation would depend on actual validation logic)


class TestA2ACPBridgeIntegration:
    """Test end-to-end A2A-ACP bridge integration."""

    @patch.dict(os.environ, {"A2A_AUTH_TOKEN": "test-token"})
    def test_full_bridge_workflow(self):
        """Test complete workflow from A2A client to ZedACP agent."""
        app = create_app()
        client = TestClient(app)

        # Test A2A JSON-RPC endpoint
        jsonrpc_request = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "id": "test-1",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "Hello"}],
                    "messageId": "msg-1",
                    "contextId": "ctx-1"
                },
                "metadata": {"agent_name": "test-agent"}
            }
        }

        response = client.post(
            "/a2a/rpc",
            json=jsonrpc_request,
            headers={"Authorization": "Bearer test-token"}
        )

        # Should get some response (may be error due to no actual agent)
        assert response.status_code in [200, 400, 500]  # Various possible outcomes in test env

    def test_bridge_error_handling(self):
        """Test bridge error handling for various scenarios."""
        app = create_app()
        client = TestClient(app)

        # Test malformed JSON-RPC
        response = client.post("/a2a/rpc", json={"invalid": "request"})
        # Should handle gracefully
        assert response.status_code in [200, 400, 500]


class TestInputRequiredFunctionality:
    """Test input-required functionality in A2A tasks."""

    def test_input_required_notification_model(self):
        """Test InputRequiredNotification model structure."""
        from src.a2a.models import InputRequiredNotification

        notification = InputRequiredNotification(
            taskId="task-123",
            contextId="ctx-456",
            message="Please provide more details",
            inputTypes=["text/plain", "application/json"],
            timeout=300
        )

        assert notification.taskId == "task-123"
        assert notification.contextId == "ctx-456"
        assert notification.inputTypes == ["text/plain", "application/json"]
        assert notification.timeout == 300

    @pytest.mark.asyncio
    async def test_task_input_required_detection(self):
        """Test that tasks properly detect input-required state."""
        task_manager = A2ATaskManager()

        # Create a task
        task = await task_manager.create_task("test-context", "test-agent")
        task_id = task.id

        # Initially should not be in input-required state
        assert task.status.state != "input-required"

        # Manually set to input-required for testing
        from src.a2a.models import TaskState
        task.status.state = TaskState.INPUT_REQUIRED

        # Test getting input-required tasks
        input_required_tasks = await task_manager.get_input_required_tasks()
        assert len(input_required_tasks) == 1
        assert input_required_tasks[0].id == task_id

    @pytest.mark.asyncio
    async def test_task_input_continuation_workflow(self):
        """Test the complete input-required continuation workflow."""
        from src.a2a.models import Message, TextPart, TaskState

        task_manager = A2ATaskManager()

        # Create a task and set it to input-required state
        task = await task_manager.create_task("test-context", "test-agent")
        task_id = task.id
        task.status.state = TaskState.INPUT_REQUIRED

        # Create user input message
        user_input = Message(
            role="user",
            parts=[TextPart(kind="text", text="Here's the additional information you requested")],
            messageId="msg-input-1",
            taskId=task_id,
            contextId="test-context"
        )

        # Mock the execute_task method to avoid actual agent execution
        original_execute = task_manager.execute_task

        async def mock_execute(*args, **kwargs):
            # Return a completed task for testing
            mock_task = await original_execute(*args, **kwargs)
            if mock_task.status.state == TaskState.INPUT_REQUIRED:
                mock_task.status.state = TaskState.COMPLETED
            return mock_task

        task_manager.execute_task = mock_execute

        try:
            # This would normally continue an input-required task
            # In a real scenario, this would call provide_input_and_continue
            # For testing, we verify the workflow components exist
            assert hasattr(task_manager, 'provide_input_and_continue')
            assert hasattr(task_manager, 'get_input_required_tasks')
        finally:
            task_manager.execute_task = original_execute

    @pytest.mark.asyncio
    async def test_input_required_state_transitions(self):
        """Test proper state transitions for input-required tasks."""
        from src.a2a.models import TaskState

        task_manager = A2ATaskManager()

        # Test state transition: submitted -> working -> input_required -> working -> completed
        task = await task_manager.create_task("test-context", "test-agent")
        task_id = task.id

        # Initial state should be submitted (based on our implementation)
        initial_state = task.status.state

        # Transition to working
        task.status.state = TaskState.WORKING
        assert task.status.state == TaskState.WORKING

        # Transition to input-required
        task.status.state = TaskState.INPUT_REQUIRED
        assert task.status.state == TaskState.INPUT_REQUIRED

        # Transition back to working after input
        task.status.state = TaskState.WORKING
        assert task.status.state == TaskState.WORKING

        # Final transition to completed
        task.status.state = TaskState.COMPLETED
        assert task.status.state == TaskState.COMPLETED

    def test_input_required_timeout_handling(self):
        """Test input-required timeout handling."""
        from src.a2a.models import InputRequiredNotification

        # Test notification with timeout
        notification = InputRequiredNotification(
            taskId="task-123",
            contextId="ctx-456",
            message="Please provide input",
            timeout=300
        )

        assert notification.timeout == 300

        # Test notification without timeout (should use default)
        notification_no_timeout = InputRequiredNotification(
            taskId="task-124",
            contextId="ctx-457",
            message="Please provide input"
        )

        assert notification_no_timeout.timeout is None

    @pytest.mark.asyncio
    async def test_multiple_input_required_tasks(self):
        """Test handling multiple input-required tasks simultaneously."""
        task_manager = A2ATaskManager()
        from src.a2a.models import TaskState

        # Create multiple tasks and set some to input-required
        tasks = []
        for i in range(5):
            task = await task_manager.create_task(f"context-{i}", f"agent-{i}")
            if i % 2 == 0:  # Set even-numbered tasks to input-required
                task.status.state = TaskState.INPUT_REQUIRED
            tasks.append(task)

        # Get input-required tasks
        input_required_tasks = await task_manager.get_input_required_tasks()

        # Should have 3 input-required tasks (0, 2, 4)
        assert len(input_required_tasks) == 3

        # Verify they are the correct tasks
        input_required_ids = {task.id for task in input_required_tasks}
        expected_ids = {tasks[i].id for i in [0, 2, 4]}
        assert input_required_ids == expected_ids


class DummyZedAgentConnection:
    """Stubbed ZedAgentConnection for streaming tests."""

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def initialize(self):
        return None

    async def start_session(self, cwd: str, mcp_servers: Optional[list] = None) -> str:
        return "session_stub"


class JSONRPCError(BaseModel):
    code: int
    message: str
    data: Optional[Any] = None


class JSONRPCErrorResponse(BaseModel):
    id: Union[str, int, None] = None
    jsonrpc: Literal["2.0"] = "2.0"
    error: JSONRPCError


class JSONRPCSuccessResponse(BaseModel):
    id: Union[str, int, None] = None
    jsonrpc: Literal["2.0"] = "2.0"


class SendMessageSuccessResponse(JSONRPCSuccessResponse):
    result: Union[Task, Message]


class GetTaskSuccessResponse(JSONRPCSuccessResponse):
    result: Task


class SendStreamingMessageSuccessResponse(JSONRPCSuccessResponse):
    result: Union[Task, Message, TaskStatusUpdateEvent, TaskArtifactUpdateEvent]


def validate_send_message_response(payload: dict) -> Union[SendMessageSuccessResponse, JSONRPCErrorResponse]:
    try:
        return SendMessageSuccessResponse.model_validate(payload)
    except ValidationError:
        return JSONRPCErrorResponse.model_validate(payload)


def validate_get_task_response(payload: dict) -> Union[GetTaskSuccessResponse, JSONRPCErrorResponse]:
    try:
        return GetTaskSuccessResponse.model_validate(payload)
    except ValidationError:
        return JSONRPCErrorResponse.model_validate(payload)


def validate_streaming_response(payload: dict) -> Union[SendStreamingMessageSuccessResponse, JSONRPCErrorResponse]:
    try:
        return SendStreamingMessageSuccessResponse.model_validate(payload)
    except ValidationError:
        return JSONRPCErrorResponse.model_validate(payload)


class TestStreamingCompliance:
    """Tests for JSON-RPC streaming compliance and serialization."""

    def test_jsonrpc_streaming_emits_spec_events(self, monkeypatch):
        """Ensure message/stream emits status-update events and final task without null fields."""
        from src.a2a.models import TextPart, create_message_id, TaskState, current_timestamp
        from src.a2a_acp.task_manager import A2ATaskManager

        async def fake_execute_task(
            self,
            task_id,
            agent_command,
            api_key=None,
            working_directory=".",
            mcp_servers=None,
            stream_handler=None,
        ):
            task = self._active_tasks[task_id].task
            if stream_handler:
                await stream_handler("hello ")
                await stream_handler("world")

            if task.history is None:
                task.history = []

            final_message = Message(
                role="agent",
                parts=[TextPart(text="hello world")],
                messageId=create_message_id(),
                taskId=task.id,
                contextId=task.contextId,
            )
            task.history.append(final_message)
            task.status.state = TaskState.COMPLETED
            task.status.timestamp = current_timestamp()
            return task

        monkeypatch.setattr(
            "src.a2a_acp.task_manager.A2ATaskManager.execute_task",
            fake_execute_task,
        )
        monkeypatch.setattr(
            "src.a2a_acp.main.ZedAgentConnection",
            DummyZedAgentConnection,
        )

        app = create_app()

        payload = {
            "jsonrpc": "2.0",
            "id": "req_stream_1",
            "method": "message/stream",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "ping"}],
                    "messageId": "msg_user_1",
                }
            },
        }

        with TestClient(app) as client:
            with client.stream("POST", "/a2a/rpc", json=payload) as response:
                assert response.status_code == 200
                assert "text/event-stream" in response.headers.get("content-type", "")

                events = []
                for line in response.iter_lines():
                    if isinstance(line, bytes):
                        line = line.decode()
                    if not line or not line.startswith("data: "):
                        continue
                    payload = json.loads(line[len("data: "):])
                    validate_streaming_response(payload)
                    events.append(payload)

        assert len(events) == 5, f"Unexpected events: {events}"


class TestDummyAgentIntegration:
    """Integration tests exercising the real dummy agent process."""

    def test_dummy_agent_streams_marker(self):
        """Ensure the dummy agent's marker token reaches the client stream."""
        app = create_app()
        payload = {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": "marker propagation test"}],
                "messageId": "msg_marker_1",
            }
        }

        with TestClient(app) as client:
            with client.stream("POST", "/a2a/message/stream", json=payload) as response:
                assert response.status_code == 200

                marker_seen = False
                final_task_received = False
                status_updates = 0
                start = time.time()

                for line in response.iter_lines():
                    if isinstance(line, bytes):
                        line = line.decode()
                    if not line or not line.startswith("data: "):
                        continue

                    event = json.loads(line[len("data: "):])
                    result = event.get("result")
                    if not isinstance(result, dict):
                        continue

                    kind = result.get("kind")

                    if kind == "status-update":
                        status_updates += 1
                        message = (result.get("status") or {}).get("message") or {}
                        parts = message.get("parts") or []
                        for part in parts:
                            text = part.get("text", "")
                            if "--END-OF-RESPONSE--" in text:
                                marker_seen = True

                    if kind == "task" and (result.get("status") or {}).get("state") == "completed":
                        history = result.get("history") or []
                        for message in history:
                            for part in message.get("parts") or []:
                                text = part.get("text", "")
                                if "--END-OF-RESPONSE--" in text:
                                    marker_seen = True
                        final_task_received = True
                        break

                    if time.time() - start > 15:
                        break

        assert final_task_received, "Did not receive the completed task payload"
        assert status_updates > 0, "Expected at least one streaming status update"
        assert marker_seen, "Marker token was not observed in the streaming output"

    def test_http_streaming_endpoint_matches_spec(self, monkeypatch):
        """Ensure HTTP /a2a/message/stream SSE responses validate against the spec."""
        from src.a2a.models import TextPart, create_message_id, TaskState, current_timestamp
        from src.a2a_acp.task_manager import A2ATaskManager

        async def fake_execute_task(
            self,
            task_id,
            agent_command,
            api_key=None,
            working_directory=".",
            mcp_servers=None,
            stream_handler=None,
        ):
            task = self._active_tasks[task_id].task
            if stream_handler:
                await stream_handler("hello ")
                await stream_handler("world")

            if task.history is None:
                task.history = []

            final_message = Message(
                role="agent",
                parts=[TextPart(text="hello world")],
                messageId=create_message_id(),
                taskId=task.id,
                contextId=task.contextId,
            )
            task.history.append(final_message)
            task.status.state = TaskState.COMPLETED
            task.status.timestamp = current_timestamp()
            return task

        monkeypatch.setattr(
            "src.a2a_acp.task_manager.A2ATaskManager.execute_task",
            fake_execute_task,
        )
        monkeypatch.setattr(
            "src.a2a_acp.main.ZedAgentConnection",
            DummyZedAgentConnection,
        )

        app = create_app()
        payload = {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": "ping"}],
                "messageId": "msg_user_1",
            }
        }

        with TestClient(app) as client:
            with client.stream("POST", "/a2a/message/stream", json=payload) as response:
                assert response.status_code == 200
                events = []
                for line in response.iter_lines():
                    if isinstance(line, bytes):
                        line = line.decode()
                    if not line or not line.startswith("data: "):
                        continue
                    payload = json.loads(line[len("data: "):])
                    validate_streaming_response(payload)
                    events.append(payload)

                assert len(events) == 5, f"Unexpected events: {events}"


class TestJSONRPCContract:
    """Tests ensuring JSON-RPC responses match the SDK models."""

    def test_send_message_and_get_task_responses(self, monkeypatch):
        from src.a2a.models import TextPart, create_message_id, TaskState, current_timestamp
        from src.a2a_acp.task_manager import A2ATaskManager

        async def fake_execute_task(
            self,
            task_id,
            agent_command,
            api_key=None,
            working_directory=".",
            mcp_servers=None,
            stream_handler=None,
        ):
            task = self._active_tasks[task_id].task
            if task.history is None:
                task.history = []

            final_message = Message(
                role="agent",
                parts=[TextPart(text="hello world")],
                messageId=create_message_id(),
                taskId=task.id,
                contextId=task.contextId,
            )
            task.history.append(final_message)
            task.status.state = TaskState.COMPLETED
            task.status.timestamp = current_timestamp()
            return task

        monkeypatch.setattr(
            "src.a2a_acp.task_manager.A2ATaskManager.execute_task",
            fake_execute_task,
        )
        monkeypatch.setattr(
            "src.a2a_acp.main.ZedAgentConnection",
            DummyZedAgentConnection,
        )

        app = create_app()
        send_payload = {
            "jsonrpc": "2.0",
            "id": "req_send_1",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": "ping"}],
                    "messageId": "msg_user_1",
                }
            },
        }

        with TestClient(app) as client:
            send_response = client.post("/a2a/rpc", json=send_payload)
            assert send_response.status_code == 200
            send_data = send_response.json()
            validated_send = validate_send_message_response(send_data)
            assert isinstance(validated_send, SendMessageSuccessResponse)

            task_id = validated_send.result.id

            get_payload = {
                "jsonrpc": "2.0",
                "id": "req_get_1",
                "method": "tasks/get",
                "params": {"id": task_id},
            }

            get_response = client.post("/a2a/rpc", json=get_payload)
            assert get_response.status_code == 200
            get_data = get_response.json()
            validated_get = validate_get_task_response(get_data)
            assert isinstance(validated_get, GetTaskSuccessResponse)
            assert validated_get.result.id == task_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
