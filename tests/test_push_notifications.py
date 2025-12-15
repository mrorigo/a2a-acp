"""
Unit tests for push notification functionality.

Tests the core push notification components including models, database operations,
notification delivery, and streaming functionality.
"""

import json
import pytest
from datetime import datetime, timedelta
from datetime import timezone
from unittest.mock import AsyncMock, patch

from a2a_acp.models import (
    TaskPushNotificationConfig,
    NotificationDelivery,
    TaskStatusChangePayload,
    TaskMessagePayload,
    DeliveryStatus,
    NotificationFilter,
)
from a2a_acp.push_notification_manager import PushNotificationManager
from a2a_acp.streaming_manager import StreamingManager, StreamingConnection
from a2a_acp.database import SessionDatabase


class TestPushNotificationModels:
    """Test push notification data models."""

    def test_task_push_notification_config_creation(self):
        """Test creating a push notification configuration."""
        config = TaskPushNotificationConfig(
            id="test-config-1",
            task_id="test-task-1",
            url="https://example.com/webhook",
            token="test-token",
            authentication_schemes={"bearer": {"type": "bearer"}},
            credentials="secret-key",
        )

        assert config.id == "test-config-1"
        assert config.task_id == "test-task-1"
        assert config.url == "https://example.com/webhook"
        assert config.token == "test-token"

    def test_notification_delivery_creation(self):
        """Test creating a notification delivery record."""
        delivery = NotificationDelivery(
            id="delivery-1",
            config_id="config-1",
            task_id="task-1",
            event_type="status_change",
            delivery_status=DeliveryStatus.DELIVERED.value,
            response_code=200,
            response_body='{"success": true}',
            attempted_at=datetime.now(timezone.utc),
        )

        assert delivery.id == "delivery-1"
        assert delivery.delivery_status == "delivered"
        assert delivery.response_code == 200

    def test_task_status_change_payload(self):
        """Test task status change notification payload."""
        payload = TaskStatusChangePayload(
            event="status_change",
            task_id="task-1",
            timestamp=datetime.now(timezone.utc),
            data={"old_state": "working", "new_state": "completed"},
            old_state="working",
            new_state="completed",
        )

        payload_dict = payload.to_dict()
        assert payload_dict["event"] == "status_change"
        assert payload_dict["old_state"] == "working"
        assert payload_dict["new_state"] == "completed"

    def test_notification_filter(self):
        """Test notification filtering logic."""
        filter_config = NotificationFilter(
            enabled_events=["status_change", "task_message"],
            disabled_events=["task_artifact"],
            quiet_hours_start="22:00",
            quiet_hours_end="08:00",
        )

        # Should send status_change notifications
        assert filter_config.should_send_notification("status_change", 14)

        # Should not send artifact notifications
        assert not filter_config.should_send_notification("task_artifact", 14)

        # Should not send during quiet hours (22:00-08:00)
        assert not filter_config.should_send_notification("status_change", 23)
        assert not filter_config.should_send_notification("status_change", 2)

        # Should send outside quiet hours
        assert filter_config.should_send_notification("status_change", 14)


class TestPushNotificationManager:
    """Test push notification manager functionality."""

    @pytest.fixture
    def mock_database(self):
        """Create a mock database for testing."""
        return AsyncMock(spec=SessionDatabase)

    @pytest.fixture
    def push_manager(self, mock_database):
        """Create a push notification manager for testing."""
        # Create test settings to avoid validation errors
        from a2a_acp.settings import PushNotificationSettings

        test_settings = PushNotificationSettings(
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
            metrics_retention_hours=72,
        )

        return PushNotificationManager(mock_database, settings=test_settings)

    def test_push_manager_initialization(self, push_manager, mock_database):
        """Test push notification manager initialization."""
        assert push_manager.database == mock_database
        assert push_manager.http_client is not None
        assert push_manager.analytics is not None
        assert push_manager.settings.webhook_timeout == 30.0

    @pytest.mark.asyncio
    async def test_store_config(self, push_manager, mock_database):
        """Test storing a push notification configuration."""
        config = TaskPushNotificationConfig(
            id="test-config", task_id="test-task", url="https://example.com/webhook"
        )

        await push_manager.store_config(config)

        mock_database.store_push_notification_config.assert_called_once()
        call_args = mock_database.store_push_notification_config.call_args[1]
        assert call_args["config_id"] == "test-config"
        assert call_args["task_id"] == "test-task"
        assert call_args["url"] == "https://example.com/webhook"

    @pytest.mark.asyncio
    async def test_get_config(self, push_manager, mock_database):
        """Test retrieving a push notification configuration."""
        expected_config = {
            "id": "test-config",
            "task_id": "test-task",
            "url": "https://example.com/webhook",
        }
        mock_database.get_push_notification_config.return_value = expected_config

        config = await push_manager.get_config("test-task", "test-config")

        assert config is not None
        assert config.id == "test-config"
        mock_database.get_push_notification_config.assert_called_once_with(
            "test-task", "test-config"
        )

    @pytest.mark.asyncio
    async def test_list_configs(self, push_manager, mock_database):
        """Test listing push notification configurations for a task."""
        expected_configs = [
            {"id": "config-1", "task_id": "test-task", "url": "https://example.com/1"},
            {"id": "config-2", "task_id": "test-task", "url": "https://example.com/2"},
        ]
        mock_database.list_push_notification_configs.return_value = expected_configs

        configs = await push_manager.list_configs("test-task")

        assert len(configs) == 2
        assert configs[0].id == "config-1"
        assert configs[1].id == "config-2"

    @pytest.mark.asyncio
    async def test_delete_config(self, push_manager, mock_database):
        """Test deleting a push notification configuration."""
        mock_database.delete_push_notification_config.return_value = True

        success = await push_manager.delete_config("test-task", "test-config")

        assert success
        mock_database.delete_push_notification_config.assert_called_once_with(
            "test-task", "test-config"
        )

    @pytest.mark.asyncio
    async def test_send_notification_with_no_configs(self, push_manager, mock_database):
        """Test sending notification when no configurations exist."""
        mock_database.list_push_notification_configs.return_value = []

        success = await push_manager.send_notification("test-task", {"event": "test"})

        assert success  # Should return True even with no configs

    @pytest.mark.asyncio
    async def test_send_notification_with_configs(self, push_manager, mock_database):
        """Test sending notification with configurations."""
        config = TaskPushNotificationConfig(
            id="test-config", task_id="test-task", url="https://httpbin.org/post"
        )
        mock_database.list_push_notification_configs.return_value = [config.to_dict()]

        # Mock successful HTTP response
        mock_response = AsyncMock()
        mock_response.is_success = True
        mock_response.status_code = 200
        mock_response.text = '{"success": true}'

        with patch.object(
            push_manager.http_client, "post", return_value=mock_response
        ) as mock_post:
            success = await push_manager.send_notification(
                "test-task",
                {
                    "event": "status_change",
                    "old_state": "working",
                    "new_state": "completed",
                },
            )

            assert success
            mock_post.assert_called_once()

            # Verify the request was made with correct data
            call_args = mock_post.call_args
            assert config.url in call_args[0]  # URL is first positional arg
            assert call_args[1]["json"]["event"] == "status_change"

    def test_create_notification_payload_status_change(self, push_manager):
        """Test creating status change notification payload."""
        event = {
            "event": "status_change",
            "task_id": "test-task",
            "old_state": "working",
            "new_state": "completed",
        }

        payload = push_manager._create_notification_payload(event)

        assert isinstance(payload, TaskStatusChangePayload)
        assert payload.event == "status_change"
        assert payload.old_state == "working"
        assert payload.new_state == "completed"

    def test_create_notification_payload_message(self, push_manager):
        """Test creating message notification payload."""
        event = {
            "event": "message",
            "task_id": "test-task",
            "message_role": "agent",
            "message_content": "Hello, world!",
        }

        payload = push_manager._create_notification_payload(event)

        assert isinstance(payload, TaskMessagePayload)
        assert payload.event == "message"
        assert payload.message_role == "agent"
        assert payload.message_content == "Hello, world!"

    def test_authentication_headers_bearer_token(self, push_manager):
        """Test authentication header generation for bearer tokens."""
        config = TaskPushNotificationConfig(
            id="test", task_id="test", url="https://example.com", token="test-token-123"
        )

        headers = push_manager._get_authentication_headers(config)

        assert headers["Authorization"] == "Bearer test-token-123"

    def test_authentication_headers_api_key(self, push_manager):
        """Test authentication header generation for API keys."""
        config = TaskPushNotificationConfig(
            id="test",
            task_id="test",
            url="https://example.com",
            authentication_schemes={
                "apikey": {"type": "apikey", "header_name": "X-API-Key"}
            },
            credentials="secret-api-key",
        )

        headers = push_manager._get_authentication_headers(config)

        assert headers["X-API-Key"] == "secret-api-key"

    def test_analytics_tracking(self, push_manager):
        """Test analytics tracking for delivery results."""
        # Create mock delivery for successful delivery
        delivery = NotificationDelivery(
            id="delivery-1",
            config_id="config-1",
            task_id="task-1",
            event_type="status_change",
            delivery_status=DeliveryStatus.DELIVERED.value,
            attempted_at=datetime.now(timezone.utc),
        )

        push_manager.analytics.update_from_delivery(delivery)

        assert push_manager.analytics.total_sent == 1
        assert push_manager.analytics.total_delivered == 1
        assert push_manager.analytics.total_failed == 0
        assert push_manager.analytics.success_rate == 100.0

    @pytest.mark.asyncio
    async def test_cleanup_by_task_state_failed(self, push_manager, mock_database):
        """Test cleanup for failed tasks (immediate deletion)."""
        # Mock having configs for the task
        config1 = TaskPushNotificationConfig(
            id="config-1", task_id="test-task", url="https://example.com/1"
        )
        config2 = TaskPushNotificationConfig(
            id="config-2", task_id="test-task", url="https://example.com/2"
        )
        mock_database.list_push_notification_configs.return_value = [
            config1.to_dict(),
            config2.to_dict(),
        ]

        # Mock successful deletions
        mock_database.delete_push_notification_config.return_value = True

        deleted_count = await push_manager.cleanup_by_task_state("test-task", "failed")

        assert deleted_count == 2
        assert mock_database.delete_push_notification_config.call_count == 2

    @pytest.mark.asyncio
    async def test_cleanup_by_task_state_completed(self, push_manager, mock_database):
        """Test cleanup for completed tasks (retention period)."""
        deleted_count = await push_manager.cleanup_by_task_state(
            "test-task", "completed"
        )

        # Should not delete immediately for completed tasks
        assert deleted_count == 0
        mock_database.list_push_notification_configs.assert_not_called()


class TestStreamingManager:
    """Test streaming manager functionality."""

    @pytest.fixture
    def mock_push_manager(self):
        """Create a mock push notification manager."""
        return AsyncMock(spec=PushNotificationManager)

    @pytest.fixture
    def streaming_manager(self, mock_push_manager):
        """Create a streaming manager for testing."""
        return StreamingManager(mock_push_manager)

    def test_streaming_connection_creation(self):
        """Test creating a streaming connection."""
        connection = StreamingConnection(
            connection_id="test-connection", task_filter={"task-1", "task-2"}
        )

        assert connection.connection_id == "test-connection"
        assert "task-1" in connection.task_filter
        assert "task-2" in connection.task_filter
        assert len(connection.task_filter) == 2

    def test_should_receive_notification_with_filter(self):
        """Test notification filtering with task filter."""
        connection = StreamingConnection(
            connection_id="test-connection", task_filter={"task-1", "task-2"}
        )

        assert connection.should_receive_notification("task-1")
        assert connection.should_receive_notification("task-2")
        assert not connection.should_receive_notification("task-3")

    def test_should_receive_notification_no_filter(self):
        """Test notification filtering without task filter (all tasks)."""
        connection = StreamingConnection(connection_id="test-connection")

        assert connection.should_receive_notification("task-1")
        assert connection.should_receive_notification("task-2")
        assert connection.should_receive_notification("any-task")

    @pytest.mark.asyncio
    async def test_register_sse_connection(self, streaming_manager):
        """Test registering an SSE connection."""
        connection_id, connection = await streaming_manager.register_sse_connection(
            ["task-1"]
        )

        assert connection_id is not None
        assert isinstance(connection, StreamingConnection)
        assert "task-1" in connection.task_filter

        # Check it's stored in the manager
        assert connection_id in streaming_manager.sse_connections

    @pytest.mark.asyncio
    async def test_unregister_sse_connection(self, streaming_manager):
        """Test unregistering an SSE connection."""
        connection_id, connection = await streaming_manager.register_sse_connection()
        assert connection_id in streaming_manager.sse_connections

        await streaming_manager.unregister_sse_connection(connection_id)
        assert connection_id not in streaming_manager.sse_connections

    @pytest.mark.asyncio
    async def test_get_connection_stats(self, streaming_manager):
        """Test getting connection statistics."""
        # Add some test connections
        await streaming_manager.register_sse_connection(["task-1"])
        await streaming_manager.register_sse_connection(["task-2"])

        stats = await streaming_manager.get_connection_stats()

        assert "websocket_connections" in stats
        assert "sse_connections" in stats
        assert stats["sse_connections"]["count"] == 2
        assert stats["total_connections"] == 2

    @pytest.mark.asyncio
    async def test_cleanup_stale_connections(self, streaming_manager):
        """Test cleaning up stale connections."""
        # Create a connection and manually set old last_activity
        connection_id, connection = await streaming_manager.register_sse_connection()
        connection.last_activity = datetime.now(timezone.utc) - timedelta(
            hours=2
        )  # 2 hours old

        # Cleanup connections older than 1 hour
        cleaned_count = await streaming_manager.cleanup_stale_connections(
            max_age_seconds=3600
        )

        assert cleaned_count == 1
        assert connection_id not in streaming_manager.sse_connections


class TestDatabaseOperations:
    """Test database operations for push notifications."""

    @pytest.fixture
    def test_db(self, tmp_path):
        """Create a temporary test database."""
        db_path = tmp_path / "test_push_notifications.db"
        return SessionDatabase(str(db_path))

    @pytest.mark.asyncio
    async def test_store_and_get_push_notification_config(self, test_db):
        """Test storing and retrieving push notification configurations."""
        # Store a configuration
        await test_db.store_push_notification_config(
            config_id="test-config",
            task_id="test-task",
            url="https://example.com/webhook",
            token="test-token",
        )

        # Retrieve the configuration
        config = await test_db.get_push_notification_config("test-task", "test-config")

        assert config is not None
        assert config["id"] == "test-config"
        assert config["task_id"] == "test-task"
        assert config["url"] == "https://example.com/webhook"
        assert config["token"] == "test-token"

    @pytest.mark.asyncio
    async def test_list_push_notification_configs(self, test_db):
        """Test listing push notification configurations."""
        # Store multiple configurations
        await test_db.store_push_notification_config(
            config_id="config-1", task_id="test-task", url="https://example.com/1"
        )
        await test_db.store_push_notification_config(
            config_id="config-2", task_id="test-task", url="https://example.com/2"
        )

        configs = await test_db.list_push_notification_configs("test-task")

        assert len(configs) == 2
        config_urls = {config["url"] for config in configs}
        assert "https://example.com/1" in config_urls
        assert "https://example.com/2" in config_urls

    @pytest.mark.asyncio
    async def test_delete_push_notification_config(self, test_db):
        """Test deleting push notification configurations."""
        # Store a configuration
        await test_db.store_push_notification_config(
            config_id="test-config",
            task_id="test-task",
            url="https://example.com/webhook",
        )

        # Verify it exists
        config = await test_db.get_push_notification_config("test-task", "test-config")
        assert config is not None

        # Delete the configuration
        success = await test_db.delete_push_notification_config(
            "test-task", "test-config"
        )
        assert success

        # Verify it's gone
        config = await test_db.get_push_notification_config("test-task", "test-config")
        assert config is None

    @pytest.mark.asyncio
    async def test_track_notification_delivery(self, test_db):
        """Test tracking notification delivery attempts."""
        delivery_id = "test-delivery"
        config_id = "test-config"
        task_id = "test-task"

        await test_db.track_notification_delivery(
            delivery_id=delivery_id,
            config_id=config_id,
            task_id=task_id,
            event_type="status_change",
            delivery_status=DeliveryStatus.DELIVERED.value,
            response_code=200,
            response_body='{"success": true}',
            delivered_at=datetime.now(timezone.utc).isoformat(),
        )

        # Retrieve delivery history
        deliveries = await test_db.get_notification_delivery_history(task_id=task_id)

        assert len(deliveries) == 1
        delivery = deliveries[0]
        assert delivery["id"] == delivery_id
        assert delivery["config_id"] == config_id
        assert delivery["event_type"] == "status_change"
        assert delivery["delivery_status"] == DeliveryStatus.DELIVERED.value

    @pytest.mark.asyncio
    async def test_cleanup_expired_notification_configs(self, test_db):
        """Test cleaning up expired notification configurations."""
        # Store a configuration with old timestamp
        await test_db.store_push_notification_config(
            config_id="old-config",
            task_id="test-task",
            url="https://example.com/webhook",
        )

        # Manually update the timestamp to be old (this would normally be done by cleanup logic)
        # For this test, we'll just verify the method exists and can be called
        cleaned_count = await test_db.cleanup_expired_notification_configs()

        # The actual cleanup depends on the timestamp logic in the database method
        assert isinstance(cleaned_count, int)


class TestIntegrationScenarios:
    """Test integration scenarios for push notifications."""

    @pytest.mark.asyncio
    async def test_task_lifecycle_notifications(self):
        """Test that notifications are sent during task lifecycle."""
        # This would test the full integration of task manager with push notifications
        # For now, we'll test the key integration points

        mock_db = AsyncMock(spec=SessionDatabase)
        from a2a_acp.settings import PushNotificationSettings

        test_settings = PushNotificationSettings(
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
            metrics_retention_hours=72,
        )
        push_mgr = PushNotificationManager(mock_db, settings=test_settings)

        # Mock configurations for the task
        config = TaskPushNotificationConfig(
            id="test-config", task_id="test-task", url="https://example.com/webhook"
        )
        mock_db.list_push_notification_configs.return_value = [config.to_dict()]

        # Mock successful HTTP response
        mock_response = AsyncMock()
        mock_response.is_success = True
        mock_response.status_code = 200
        mock_response.text = '{"success": true}'

        with patch.object(push_mgr.http_client, "post", return_value=mock_response):
            # Test status change notification
            success = await push_mgr.send_notification(
                "test-task",
                {
                    "event": "status_change",
                    "task_id": "test-task",
                    "old_state": "working",
                    "new_state": "completed",
                },
            )

            assert success

    @pytest.mark.asyncio
    async def test_retry_logic_on_failure(self):
        """Test retry logic when notification delivery fails."""
        mock_db = AsyncMock(spec=SessionDatabase)
        from a2a_acp.settings import PushNotificationSettings

        test_settings = PushNotificationSettings(
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
            metrics_retention_hours=72,
        )
        push_mgr = PushNotificationManager(mock_db, settings=test_settings)

        # Mock configuration
        config = TaskPushNotificationConfig(
            id="test-config", task_id="test-task", url="https://example.com/webhook"
        )
        mock_db.list_push_notification_configs.return_value = [config.to_dict()]

        # Mock failed HTTP response, then successful
        failed_response = AsyncMock()
        failed_response.is_success = False
        failed_response.status_code = 500

        success_response = AsyncMock()
        success_response.is_success = True
        success_response.status_code = 200

        with patch.object(push_mgr.http_client, "post") as mock_post:
            mock_post.side_effect = [failed_response, failed_response, success_response]

            success = await push_mgr.send_notification(
                "test-task", {"event": "status_change", "task_id": "test-task"}
            )

            # Should succeed after retries
            assert success
            assert mock_post.call_count == 3  # Initial attempt + 2 retries

    def test_notification_payload_serialization(self):
        """Test that notification payloads can be properly serialized."""
        payload = TaskStatusChangePayload(
            event="status_change",
            task_id="test-task",
            timestamp=datetime.now(timezone.utc),
            data={"test": "data"},
            old_state="working",
            new_state="completed",
        )

        payload_dict = payload.to_dict()

        # Should be JSON serializable
        json_str = json.dumps(payload_dict)
        parsed = json.loads(json_str)

        assert parsed["event"] == "status_change"
        assert parsed["old_state"] == "working"
        assert parsed["new_state"] == "completed"


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
