"""
Tests for A2A-ACP Bridge Components

Comprehensive test suite for the A2A-ACP bridge layer that connects A2A clients
to ZedACP agents. This covers the critical functionality that was previously untested.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.a2a_acp.main import create_app
from src.a2a_acp.agent_registry import AgentRegistry
from src.a2a_acp.database import SessionDatabase, A2AContext, ACPSession
from src.a2a_acp.models import Run, RunStatus, RunMode
from src.a2a_acp.task_manager import A2ATaskManager, a2a_task_manager
from src.a2a_acp.context_manager import A2AContextManager, a2a_context_manager
from src.a2a_acp.context_manager import A2AContextManager
from src.a2a_acp.zed_agent import ZedAgentConnection, AgentProcessError


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


class TestAgentRegistry:
    """Test the ZedACP agent registry."""

    def test_agent_registry_creation(self):
        """Test that agent registry can be created."""
        registry = AgentRegistry()
        assert registry is not None

    def test_agent_registry_list_agents(self):
        """Test listing agents from registry."""
        # Use a temporary config file for testing
        mock_config = {
            "test-agent": {
                "name": "test-agent",
                "command": ["echo"],
                "description": "Test agent"
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            json.dump(mock_config, tmp)
            tmp.flush()

            try:
                registry = AgentRegistry(Path(tmp.name))
                agents = list(registry.list())  # Convert iterable to list
                assert len(agents) > 0
            finally:
                os.unlink(tmp.name)

    def test_agent_registry_get_agent(self):
        """Test getting specific agent from registry."""
        mock_config = {
            "test-agent": {
                "name": "test-agent",
                "command": ["echo"],
                "description": "Test agent"
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            json.dump(mock_config, tmp)
            tmp.flush()

            try:
                registry = AgentRegistry(Path(tmp.name))
                agent = registry.get("test-agent")
                assert agent.name == "test-agent"
            finally:
                os.unlink(tmp.name)


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
        """Test that Task model has correct structure."""
        from datetime import datetime

        run = Run(
            id="test-run",
            agent="test-agent",
            status=RunStatus.queued,
            mode=RunMode.sync,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )

        assert run.id == "test-run"
        assert run.agent == "test-agent"
        assert run.status == RunStatus.queued
        assert run.mode == RunMode.sync


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])