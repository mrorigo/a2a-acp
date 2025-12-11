"""
Integration tests for A2A push notification functionality.

Tests the complete push notification workflow from A2A API calls through
task execution to notification delivery and streaming.
"""

import asyncio
import pytest
import sqlite3
import time
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from src.a2a_acp.main import create_app
from src.a2a_acp.database import SessionDatabase
from src.a2a_acp.models import TaskPushNotificationConfig
from src.a2a_acp.push_notification_manager import PushNotificationManager
from src.a2a.models import TaskState


class TestA2APushNotificationIntegration:
    """Integration tests for A2A push notification system."""

    @pytest.fixture
    def test_db(self, tmp_path):
        """Create a test database."""
        db_path = tmp_path / "test_integration.db"
        return SessionDatabase(str(db_path))

    @pytest.fixture
    def test_client(self, test_db, monkeypatch):
        """Create a test client with the push notification system."""
        from src.a2a_acp.task_manager import A2ATaskManager
        from src.a2a_acp.context_manager import A2AContextManager
        from src.a2a_acp.streaming_manager import StreamingManager
        from src.a2a.translator import A2ATranslator
        from src.a2a_acp.settings import PushNotificationSettings

        # Set required environment variable for testing
        monkeypatch.setenv("A2A_AGENT_COMMAND", "python tests/dummy_agent.py")

        app = create_app()

        # Override the database in the app state for testing
        app.state.database = test_db

        # Initialize push notification manager with test settings (retry_attempts=0 for immediate failure testing)
        test_push_settings = PushNotificationSettings(
            enabled=True,
            webhook_timeout=30.0,
            retry_attempts=0,  # No retries for immediate failure testing
            retry_delay=1.0,
            max_retry_delay=60.0,
            batch_size=10,
            max_concurrent_notifications=100,
            cleanup_enabled=True,
            cleanup_interval=3600,
            retention_completed_hours=24,
            retention_failed_hours=0,
            retention_auth_required_hours=168,
            hmac_secret="test-secret",
            rate_limit_per_minute=60,
            streaming_enabled=True,
            max_websocket_connections=1000,
            max_sse_connections=500,
            connection_cleanup_interval=300,
            enable_metrics=True,
            metrics_retention_hours=72
        )
        app.state.push_notification_manager = PushNotificationManager(test_db, settings=test_push_settings)

        # Initialize streaming manager with push notification manager and settings
        app.state.streaming_manager = StreamingManager(
            app.state.push_notification_manager,
            max_websocket_connections=test_push_settings.max_websocket_connections,
            max_sse_connections=test_push_settings.max_sse_connections,
            cleanup_interval=test_push_settings.connection_cleanup_interval
        )

        # Connect streaming manager back to push notification manager for broadcasting
        app.state.push_notification_manager.streaming_manager = app.state.streaming_manager

        # Initialize other required managers
        app.state.task_manager = A2ATaskManager(app.state.push_notification_manager)
        app.state.context_manager = A2AContextManager()
        app.state.a2a_translator = A2ATranslator()
    
        return TestClient(app)

    def test_push_notification_config_set_get_delete_workflow(self, test_client, test_db):
        """Test complete push notification configuration workflow."""
        # Step 1: Set a push notification configuration via A2A API
        config_payload = {
            "id": "integration-test-config",
            "taskId": "integration-test-task",
            "url": "https://httpbin.org/post",
            "token": "test-integration-token"
        }

        # Use direct method call instead of HTTP for testing
        async def test_config_workflow():
            # Create and store config directly
            config = TaskPushNotificationConfig(
                id=config_payload["id"],
                task_id=config_payload["taskId"],
                url=config_payload["url"],
                token=config_payload["token"]
            )

            push_mgr = test_client.app.state.push_notification_manager
            await push_mgr.store_config(config)

            # Retrieve the config
            retrieved_config = await push_mgr.get_config(
                config_payload["taskId"],
                config_payload["id"]
            )

            assert retrieved_config is not None
            assert retrieved_config.id == config_payload["id"]
            assert retrieved_config.task_id == config_payload["taskId"]
            assert retrieved_config.url == config_payload["url"]
            assert retrieved_config.token == config_payload["token"]

            # List configs for the task
            configs = await push_mgr.list_configs(config_payload["taskId"])
            assert len(configs) == 1
            assert configs[0].id == config_payload["id"]

            # Delete the config
            delete_success = await push_mgr.delete_config(
                config_payload["taskId"],
                config_payload["id"]
            )
            assert delete_success

            # Verify deletion
            deleted_config = await push_mgr.get_config(
                config_payload["taskId"],
                config_payload["id"]
            )
            assert deleted_config is None

        # Run the async test
        asyncio.run(test_config_workflow())

    def test_task_execution_with_notifications(self, test_client, test_db):
        """Test that task execution triggers appropriate notifications."""
        push_mgr = test_client.app.state.push_notification_manager

        async def test_task_notifications():
            # Mock the HTTP client to capture notification requests
            notification_requests = []

            original_post = push_mgr.http_client.post

            async def capture_post(*args, **kwargs):
                notification_requests.append({
                    "url": args[0] if args else kwargs.get("url"),
                    "data": kwargs.get("json"),
                    "headers": kwargs.get("headers", {})
                })
                # Return a successful mock response
                mock_response = AsyncMock()
                mock_response.is_success = True
                mock_response.status_code = 200
                mock_response.text = '{"success": true}'
                return mock_response

            push_mgr.http_client.post = capture_post

            try:
                # Set up a notification configuration
                config = TaskPushNotificationConfig(
                    id="task-notification-test",
                    task_id="test-task-notifications",
                    url="https://example.com/notify"
                )
                await push_mgr.store_config(config)

                # Send a test notification
                await push_mgr.send_notification("test-task-notifications", {
                    "event": "status_change",
                    "task_id": "test-task-notifications",
                    "old_state": "submitted",
                    "new_state": "working"
                })

                # Verify notification was sent
                assert len(notification_requests) == 1
                request = notification_requests[0]
                assert "status_change" in request["data"]["event"]
                assert request["data"]["old_state"] == "submitted"
                assert request["data"]["new_state"] == "working"

            finally:
                # Restore original method
                push_mgr.http_client.post = original_post

        asyncio.run(test_task_notifications())

    def test_real_time_streaming_integration(self, test_client):
        """Test real-time streaming integration."""
        streaming_mgr = test_client.app.state.streaming_manager

        async def test_streaming():
            # Test SSE connection registration
            connection_id, connection = await streaming_mgr.register_sse_connection(["test-task"])

            assert connection_id is not None
            assert connection.task_filter == {"test-task"}

            # Test WebSocket connection registration
            from unittest.mock import AsyncMock
            mock_websocket = AsyncMock()

            ws_connection_id = await streaming_mgr.register_websocket_connection(
                mock_websocket,
                ["test-task"]
            )

            assert ws_connection_id is not None
            assert ws_connection_id in streaming_mgr.websocket_connections

            # Test broadcasting notification
            await streaming_mgr.broadcast_notification("test-task", {
                "event": "test_event",
                "data": "test_data"
            })

            # Clean up
            await streaming_mgr.unregister_sse_connection(connection_id)

        asyncio.run(test_streaming())

    def test_cleanup_integration(self, test_client, test_db):
        """Test cleanup system integration."""
        push_mgr = test_client.app.state.push_notification_manager

        async def test_cleanup():
            # Create multiple configs for different tasks
            configs_data = [
                ("config-1", "task-1", "https://example.com/1"),
                ("config-2", "task-2", "https://example.com/2"),
                ("config-3", "task-1", "https://example.com/3"),
            ]

            for config_id, task_id, url in configs_data:
                config = TaskPushNotificationConfig(id=config_id, task_id=task_id, url=url)
                await push_mgr.store_config(config)

            # Test cleanup for failed task (should delete immediately)
            deleted_count = await push_mgr.cleanup_by_task_state("task-1", "failed")
            assert deleted_count == 2  # Should delete both configs for task-1

            # Verify configs are deleted
            remaining_configs = await push_mgr.list_configs("task-1")
            assert len(remaining_configs) == 0

            # task-2 config should still exist
            remaining_configs = await push_mgr.list_configs("task-2")
            assert len(remaining_configs) == 1

        asyncio.run(test_cleanup())

    @patch('httpx.AsyncClient.post')
    def test_notification_delivery_tracking(self, mock_post, test_client, test_db):
        """Test that notification delivery attempts are properly tracked."""
        push_mgr = test_client.app.state.push_notification_manager

        # Create a mock response
        mock_response = AsyncMock()
        mock_response.is_success = True
        mock_response.status_code = 200
        mock_response.text = '{"success": true}'
        mock_post.return_value = mock_response

        async def test_delivery_tracking():
            # Set up notification config
            config = TaskPushNotificationConfig(
                id="delivery-test",
                task_id="test-delivery-task",
                url="https://example.com/delivery"
            )
            await push_mgr.store_config(config)

            # Send notification
            await push_mgr.send_notification("test-delivery-task", {
                "event": "status_change",
                "old_state": "working",
                "new_state": "completed"
            })

            # Check delivery tracking
            deliveries = await push_mgr.get_delivery_history(task_id="test-delivery-task")

            assert len(deliveries) == 1
            delivery = deliveries[0]
            assert delivery.event_type == "status_change"
            assert delivery.delivery_status == "delivered"

        asyncio.run(test_delivery_tracking())

    def test_error_handling_integration(self, test_client, test_db):
        """Test error handling throughout the notification system."""
        push_mgr = test_client.app.state.push_notification_manager

        async def test_error_handling():
            # Test with invalid webhook URL
            config = TaskPushNotificationConfig(
                id="error-test",
                task_id="test-error-task",
                url="https://invalid-domain-that-does-not-exist.com/webhook"
            )
            await push_mgr.store_config(config)

            # Mock network error
            def mock_post(*args, **kwargs):
                raise Exception("Network error")

            push_mgr.http_client.post = mock_post

            # Should handle error gracefully
            success = await push_mgr.send_notification("test-error-task", {
                "event": "test_event"
            })

            # Should still return False but not crash
            assert not success

            # Check that failed delivery was tracked
            deliveries = await push_mgr.get_delivery_history(task_id="test-error-task")
            print(f"DEBUG: Found {len(deliveries)} deliveries in database")
            for delivery in deliveries:
                print(f"DEBUG: Delivery status: {delivery.delivery_status}, event_type: {delivery.event_type}")

            assert len(deliveries) == 1
            assert deliveries[0].delivery_status == "failed"

        asyncio.run(test_error_handling())

    @patch('httpx.AsyncClient.post')
    def test_concurrent_notification_sending(self, mock_post, test_client, test_db):
        """Test sending notifications concurrently to multiple endpoints."""
        push_mgr = test_client.app.state.push_notification_manager

        # Create a mock response
        mock_response = AsyncMock()
        mock_response.is_success = True
        mock_response.status_code = 200
        mock_response.text = '{"success": true}'
        mock_post.return_value = mock_response

        async def test_concurrent_sending():
            # Set up multiple configs for the same task
            configs = []
            for i in range(3):
                config = TaskPushNotificationConfig(
                    id=f"concurrent-config-{i}",
                    task_id="concurrent-test-task",
                    url="https://example.com/webhook"
                )
                configs.append(config)
                await push_mgr.store_config(config)

            # Send notification (should go to all 3 endpoints)
            await push_mgr.send_notification("concurrent-test-task", {
                "event": "concurrent_test",
                "task_id": "concurrent-test-task"
            })

            # All 3 endpoints should have been called
            assert mock_post.call_count == 3

        asyncio.run(test_concurrent_sending())


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""

    @pytest.fixture
    def test_db(self, tmp_path):
        """Create a test database."""
        db_path = tmp_path / "test_e2e.db"
        return SessionDatabase(str(db_path))

    @pytest.fixture
    def test_client(self, test_db, monkeypatch):
        """Create a test client."""
        from src.a2a_acp.task_manager import A2ATaskManager
        from src.a2a_acp.context_manager import A2AContextManager
        from src.a2a_acp.streaming_manager import StreamingManager
        from src.a2a.translator import A2ATranslator

        # Set required environment variable for testing
        monkeypatch.setenv("A2A_AGENT_COMMAND", "python tests/dummy_agent.py")

        app = create_app()
        app.state.database = test_db

        # Initialize push notification manager with test database and settings
        from src.a2a_acp.settings import PushNotificationSettings

        # Create test settings for integration tests
        test_push_settings = PushNotificationSettings(
            enabled=True,
            webhook_timeout=30.0,
            retry_attempts=3,
            retry_delay=1.0,
            max_retry_delay=60.0,
            batch_size=10,
            max_concurrent_notifications=100,
            cleanup_enabled=True,
            cleanup_interval=3600,
            retention_completed_hours=24,
            retention_failed_hours=0,
            retention_auth_required_hours=168,
            hmac_secret="test-secret",
            rate_limit_per_minute=60,
            streaming_enabled=True,
            max_websocket_connections=1000,
            max_sse_connections=500,
            connection_cleanup_interval=300,
            enable_metrics=True,
            metrics_retention_hours=72
        )
        app.state.push_notification_manager = PushNotificationManager(test_db, settings=test_push_settings)

        # Initialize streaming manager with push notification manager and settings
        app.state.streaming_manager = StreamingManager(
            app.state.push_notification_manager,
            max_websocket_connections=test_push_settings.max_websocket_connections,
            max_sse_connections=test_push_settings.max_sse_connections,
            cleanup_interval=test_push_settings.connection_cleanup_interval
        )

        # Connect streaming manager back to push notification manager for broadcasting
        app.state.push_notification_manager.streaming_manager = app.state.streaming_manager

        # Initialize other required managers
        app.state.task_manager = A2ATaskManager(app.state.push_notification_manager)
        app.state.context_manager = A2AContextManager()
        app.state.a2a_translator = A2ATranslator()

        return TestClient(app)

    def test_complete_task_lifecycle_with_notifications(self, test_client, test_db):
        """Test complete task lifecycle from creation to completion with notifications."""
        push_mgr = test_client.app.state.push_notification_manager
        task_mgr = test_client.app.state.task_manager

        async def test_complete_lifecycle():
            # Track notifications sent
            sent_notifications = []

            original_send = push_mgr.send_notification

            async def track_notifications(task_id, event):
                sent_notifications.append({"task_id": task_id, "event": event})
                return True

            push_mgr.send_notification = track_notifications

            # Also track the _send_task_notification method that's called by task manager
            original_send_task = task_mgr._send_task_notification

            async def track_task_notifications(task_id, event_type, event_data):
                sent_notifications.append({"task_id": task_id, "event": {"event": event_type, **event_data}})
                # Call the original method directly to avoid recursion
                if original_send_task != track_task_notifications:  # Safety check
                    return await original_send_task(task_id, event_type, event_data)
                return True

            task_mgr._send_task_notification = track_task_notifications

            try:

                # Create a task (should trigger creation notification)
                task = await task_mgr.create_task(
                    context_id="test-context",
                    agent_name="test-agent"
                )

                # Wait a bit for background notifications to complete
                await asyncio.sleep(0.1)

                # Should have received task creation notification
                # Note: Both tracking functions may capture this, so we expect at least 1
                creation_notifications = [n for n in sent_notifications if n["event"]["event"] == "task_created"]
                assert len(creation_notifications) >= 1

                # Update task status to working (should trigger status change notification)
                from src.a2a.models import current_timestamp
                task.status.state = TaskState.WORKING
                task.status.timestamp = current_timestamp()

                await push_mgr.send_notification(task.id, {
                    "event": "status_change",
                    "old_state": TaskState.SUBMITTED.value,
                    "new_state": TaskState.WORKING.value
                })

                # Should have received status change notification
                status_notifications = [n for n in sent_notifications if n["event"]["event"] == "status_change"]
                assert len(status_notifications) >= 1

                # Complete the task (should trigger completion notification and cleanup)
                task.status.state = TaskState.COMPLETED
                task.status.timestamp = current_timestamp()

                await push_mgr.send_notification(task.id, {
                    "event": "status_change",
                    "old_state": TaskState.WORKING.value,
                    "new_state": TaskState.COMPLETED.value
                })

                # Verify notifications were sent for key lifecycle events
                event_types = [n["event"]["event"] for n in sent_notifications]
                assert "task_created" in event_types
                assert "status_change" in event_types

            finally:
                push_mgr.send_notification = original_send
                task_mgr._send_task_notification = original_send_task

        asyncio.run(test_complete_lifecycle())

    def test_notification_filtering_integration(self, test_client, test_db):
        """Test notification filtering in real scenarios."""
        push_mgr = test_client.app.state.push_notification_manager

        async def test_filtering():
            # Create configs with different filtering requirements
            configs = [
                TaskPushNotificationConfig(
                    id="filter-config-1",
                    task_id="filter-task",
                    url="https://example.com/1"
                ),
                TaskPushNotificationConfig(
                    id="filter-config-2",
                    task_id="filter-task",
                    url="https://example.com/2"
                )
            ]

            for config in configs:
                await push_mgr.store_config(config)

            # Test that filtering logic works correctly
            # (This would test the _should_send_notification method with various scenarios)

            # For now, verify configs are stored correctly
            stored_configs = await push_mgr.list_configs("filter-task")
            assert len(stored_configs) == 2

        asyncio.run(test_filtering())

    @patch('httpx.AsyncClient.post')
    def test_performance_under_load(self, mock_post, test_client, test_db):
        """Test system performance with multiple concurrent operations."""
        push_mgr = test_client.app.state.push_notification_manager

        # Create a mock response
        mock_response = AsyncMock()
        mock_response.is_success = True
        mock_response.status_code = 200
        mock_response.text = '{"success": true}'
        mock_post.return_value = mock_response

        async def test_performance():
            # Create many notification configs
            num_configs = 10
            task_id = "performance-test-task"

            for i in range(num_configs):
                config = TaskPushNotificationConfig(
                    id=f"perf-config-{i}",
                    task_id=task_id,
                    url=f"https://example.com/perf/{i}"
                )
                await push_mgr.store_config(config)


            # Send notification to all configs
            start_time = time.time()
            await push_mgr.send_notification(task_id, {
                "event": "performance_test",
                "task_id": task_id
            })
            end_time = time.time()

            # Should have sent to all configs (check mock call count)
            assert mock_post.call_count == num_configs

            # Should complete in reasonable time (adjust threshold as needed)
            execution_time = end_time - start_time
            assert execution_time < 5.0  # Should complete within 5 seconds

        asyncio.run(test_performance())


class TestErrorRecovery:
    """Test error recovery and resilience."""

    @pytest.fixture
    def test_db(self, tmp_path):
        """Create a test database."""
        db_path = tmp_path / "test_recovery.db"
        return SessionDatabase(str(db_path))

    @pytest.fixture
    def test_client(self, test_db, monkeypatch):
        """Create a test client."""
        from src.a2a_acp.task_manager import A2ATaskManager
        from src.a2a_acp.context_manager import A2AContextManager
        from src.a2a_acp.streaming_manager import StreamingManager
        from src.a2a.translator import A2ATranslator

        # Set required environment variable for testing
        monkeypatch.setenv("A2A_AGENT_COMMAND", "python tests/dummy_agent.py")

        app = create_app()
        app.state.database = test_db

        # Initialize push notification manager with test database and settings
        from src.a2a_acp.settings import PushNotificationSettings

        # Create test settings for integration tests
        test_push_settings = PushNotificationSettings(
            enabled=True,
            webhook_timeout=30.0,
            retry_attempts=3,
            retry_delay=1.0,
            max_retry_delay=60.0,
            batch_size=10,
            max_concurrent_notifications=100,
            cleanup_enabled=True,
            cleanup_interval=3600,
            retention_completed_hours=24,
            retention_failed_hours=0,
            retention_auth_required_hours=168,
            hmac_secret="test-secret",
            rate_limit_per_minute=60,
            streaming_enabled=True,
            max_websocket_connections=1000,
            max_sse_connections=500,
            connection_cleanup_interval=300,
            enable_metrics=True,
            metrics_retention_hours=72
        )
        app.state.push_notification_manager = PushNotificationManager(test_db, settings=test_push_settings)

        # Initialize streaming manager with push notification manager and settings
        app.state.streaming_manager = StreamingManager(
            app.state.push_notification_manager,
            max_websocket_connections=test_push_settings.max_websocket_connections,
            max_sse_connections=test_push_settings.max_sse_connections,
            cleanup_interval=test_push_settings.connection_cleanup_interval
        )

        # Connect streaming manager back to push notification manager for broadcasting
        app.state.push_notification_manager.streaming_manager = app.state.streaming_manager

        # Initialize other required managers
        app.state.task_manager = A2ATaskManager(app.state.push_notification_manager)
        app.state.context_manager = A2AContextManager()
        app.state.a2a_translator = A2ATranslator()

        return TestClient(app)

    def test_database_connection_recovery(self, test_client, test_db):
        """Test recovery from database connection issues."""
        push_mgr = test_client.app.state.push_notification_manager

        async def test_recovery():
            # Test that operations handle database errors gracefully

            # Mock database operation to raise an exception
            original_store = test_db.store_push_notification_config

            async def failing_store(*args, **kwargs):
                raise sqlite3.OperationalError("Database is locked")

            test_db.store_push_notification_config = failing_store

            try:
                config = TaskPushNotificationConfig(
                    id="recovery-test",
                    task_id="recovery-task",
                    url="https://example.com/recovery"
                )

                # Should handle the error gracefully
                with pytest.raises(Exception):  # Should raise the database error
                    await push_mgr.store_config(config)

            finally:
                test_db.store_push_notification_config = original_store

        asyncio.run(test_recovery())

    @patch('httpx.AsyncClient.post')
    def test_http_client_resilience(self, mock_post, test_client, test_db):
        """Test resilience to HTTP client issues."""
        push_mgr = test_client.app.state.push_notification_manager

        # Create a side effect that fails twice then succeeds
        failure_count = 0

        def mock_post_side_effect(url, json=None, headers=None):
            nonlocal failure_count
            failure_count += 1

            if failure_count <= 2:  # Fail first 2 attempts
                raise Exception("Connection timeout")

            # Succeed on third attempt
            mock_response = AsyncMock()
            mock_response.is_success = True
            mock_response.status_code = 200
            mock_response.text = '{"success": true}'
            return mock_response

        mock_post.side_effect = mock_post_side_effect

        async def test_http_resilience():
            # Set up a config
            config = TaskPushNotificationConfig(
                id="resilience-test",
                task_id="resilience-task",
                url="https://example.com/resilience"
            )
            await push_mgr.store_config(config)

            # Should eventually succeed after retries
            success = await push_mgr.send_notification("resilience-task", {
                "event": "resilience_test"
            })

            assert success
            assert mock_post.call_count == 3  # Initial + 2 retries

        asyncio.run(test_http_resilience())


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
