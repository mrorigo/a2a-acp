"""
Push Notification Manager for A2A-ACP.

This module handles the configuration, delivery, and management of push notifications
for A2A-ACP tasks. It provides HTTP webhook notifications with authentication,
retry logic, and delivery tracking.
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import httpx
from httpx import AsyncClient, Response, Timeout

from .models import (
    TaskPushNotificationConfig,
    NotificationDelivery,
    NotificationPayload,
    TaskStatusChangePayload,
    TaskMessagePayload,
    TaskArtifactPayload,
    DeliveryStatus,
    EventType,
    NotificationFilter,
    NotificationAnalytics,
    DevelopmentToolEvent,
    DevelopmentToolEventKind
)
from .database import SessionDatabase
# from .streaming_manager import StreamingManager  # Avoid circular import

logger = logging.getLogger(__name__)


class PushNotificationManager:
    """
    Manages push notifications for tasks.

    Handles configuration storage, HTTP delivery, authentication,
    retry logic, and delivery tracking.
    """

    def __init__(
        self,
        database: SessionDatabase,
        base_url: Optional[str] = None,
        streaming_manager: Optional[Any] = None,
        settings: Optional[Any] = None
    ):
        """Initialize the push notification manager."""
        from .settings import get_settings

        self.database = database
        self.base_url = base_url or "http://localhost:8001"
        self.streaming_manager = streaming_manager
        self.http_client: Optional[AsyncClient] = None
        self.analytics = NotificationAnalytics()

        # Get settings (either passed in or from environment)
        self.settings = settings or get_settings().push_notifications

        # Initialize HTTP client
        self._init_http_client()

    def _init_http_client(self) -> None:
        """Initialize the HTTP client with proper configuration."""
        timeout = Timeout(self.settings.webhook_timeout, connect=10.0)
        self.http_client = AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            max_redirects=3
        )

    async def store_config(
        self,
        config: TaskPushNotificationConfig
    ) -> None:
        """Store a push notification configuration."""
        try:
            await self.database.store_push_notification_config(
                config_id=config.id,
                task_id=config.task_id,
                url=config.url,
                token=config.token,
                authentication_schemes=config.authentication_schemes,
                credentials=config.credentials,
                enabled_events=config.enabled_events,
                disabled_events=config.disabled_events,
                quiet_hours_start=config.quiet_hours_start,
                quiet_hours_end=config.quiet_hours_end
            )
            logger.info(
                "Stored push notification config",
                extra={
                    "config_id": config.id,
                    "task_id": config.task_id,
                    "url": config.url
                }
            )
        except Exception as e:
            logger.error(
                "Failed to store push notification config",
                extra={
                    "config_id": config.id,
                    "task_id": config.task_id,
                    "error": str(e)
                }
            )
            raise

    async def get_config(
        self,
        task_id: str,
        config_id: str
    ) -> Optional[TaskPushNotificationConfig]:
        """Retrieve a push notification configuration."""
        try:
            config_data = await self.database.get_push_notification_config(task_id, config_id)
            if config_data:
                return TaskPushNotificationConfig.from_dict(config_data)
            return None
        except Exception as e:
            logger.error(
                "Failed to get push notification config",
                extra={
                    "config_id": config_id,
                    "task_id": task_id,
                    "error": str(e)
                }
            )
            return None

    async def list_configs(
        self,
        task_id: str
    ) -> List[TaskPushNotificationConfig]:
        """List all push notification configurations for a task."""
        try:
            config_data_list = await self.database.list_push_notification_configs(task_id)
            configs = []
            for config_data in config_data_list:
                config = TaskPushNotificationConfig.from_dict(config_data)
                configs.append(config)
            return configs
        except Exception as e:
            logger.error(
                "Failed to list push notification configs",
                extra={"task_id": task_id, "error": str(e)}
            )
            return []

    async def delete_config(
        self,
        task_id: str,
        config_id: str
    ) -> bool:
        """Delete a push notification configuration."""
        try:
            success = await self.database.delete_push_notification_config(task_id, config_id)
            if success:
                logger.info(
                    "Deleted push notification config",
                    extra={"config_id": config_id, "task_id": task_id}
                )
            return success
        except Exception as e:
            logger.error(
                "Failed to delete push notification config",
                extra={
                    "config_id": config_id,
                    "task_id": task_id,
                    "error": str(e)
                }
            )
            return False

    async def send_notification(
        self,
        task_id: str,
        event: Dict[str, Any]
    ) -> bool:
        """Send notifications for a task event to all configured endpoints."""
        try:
            # Get all notification configurations for this task
            configs = await self.list_configs(task_id)

            if not configs:
                logger.debug(
                    "No notification configs found for task",
                    extra={"task_id": task_id}
                )
                return True

            # Send to all configured endpoints
            success_count = 0
            total_count = len(configs)

            for config in configs:
                try:
                    # Check if we should send this notification (filtering logic)
                    if not self._should_send_notification(config, event):
                        logger.debug(
                            "Skipping notification due to filter rules",
                            extra={
                                "config_id": config.id,
                                "task_id": task_id,
                                "event_type": event.get("event")
                            }
                        )
                        continue

                    # Send the notification
                    delivery_success = await self._send_single_notification(config, event)

                    if delivery_success:
                        success_count += 1

                except Exception as e:
                    logger.error(
                        "Error sending notification to config",
                        extra={
                            "config_id": config.id,
                            "task_id": task_id,
                            "error": str(e)
                        }
                    )

            # Broadcast to streaming connections for real-time updates
            if self.streaming_manager:
                try:
                    await self.streaming_manager.broadcast_notification(task_id, event)
                except Exception as e:
                    logger.error(
                        "Failed to broadcast to streaming connections",
                        extra={"task_id": task_id, "error": str(e)}
                    )

            # Track delivery analytics
            self._track_deliveries(configs, event, success_count, total_count)

            # Return True if at least one notification succeeded
            return success_count > 0

        except Exception as e:
            logger.error(
                "Failed to send notifications",
                extra={"task_id": task_id, "error": str(e)}
            )
            return False

    def _should_send_notification(
        self,
        config: TaskPushNotificationConfig,
        event: Dict[str, Any]
    ) -> bool:
        """Check if notification should be sent based on configuration."""
        event_type_str = event.get("event", "")

        # Skip certain internal events unless explicitly requested
        internal_events = {"notification_sent", "notification_failed"}
        if event_type_str in internal_events:
            return False

        # Try to map the event type to enum for consistent filtering
        event_type_mapping = {
            "status_change": EventType.TASK_STATUS_CHANGE.value,
            "message": EventType.TASK_MESSAGE.value,
            "artifact": EventType.TASK_ARTIFACT.value,
            "input_required": EventType.TASK_INPUT_REQUIRED.value,
            "task_created": EventType.TASK_STATUS_CHANGE.value,
            "completed": EventType.TASK_COMPLETED.value,
            "failed": EventType.TASK_FAILED.value,
            "concurrent_test": EventType.TASK_STATUS_CHANGE.value,
            "test_event": EventType.TASK_STATUS_CHANGE.value,
            "resilience_test": EventType.TASK_STATUS_CHANGE.value,
            "large_payload_test": EventType.TASK_STATUS_CHANGE.value,
            "performance_test": EventType.TASK_STATUS_CHANGE.value
        }
        normalized_event_type = event_type_mapping.get(event_type_str, EventType.TASK_STATUS_CHANGE.value)

        # Ensure we always have a valid string for filtering
        if not normalized_event_type:
            normalized_event_type = event_type_str or "unknown"

        # Use NotificationFilter for sophisticated filtering
        # Create filter from config (extend TaskPushNotificationConfig to include filter settings)
        filter_config = NotificationFilter(
            enabled_events=getattr(config, 'enabled_events', []),
            disabled_events=getattr(config, 'disabled_events', []),
            quiet_hours_start=getattr(config, 'quiet_hours_start', None),
            quiet_hours_end=getattr(config, 'quiet_hours_end', None)
        )

        # Get current hour for quiet hours filtering
        current_hour = datetime.now().hour

        # Apply filtering rules
        return filter_config.should_send_notification(normalized_event_type, current_hour)

    async def _handle_notification_error(
        self,
        config: TaskPushNotificationConfig,
        event: Dict[str, Any],
        error: Exception,
        attempt: int,
        start_time: float,
        response: Optional[Response] = None
    ) -> bool:
        """Handle notification delivery errors with consistent logging and tracking."""
        should_retry = attempt < self.settings.retry_attempts

        if isinstance(error, httpx.TimeoutException):
            logger.warning(
                "Notification timeout",
                extra={
                    "config_id": config.id,
                    "task_id": config.task_id,
                    "url": config.url,
                    "attempt": attempt + 1,
                    "error": str(error)
                }
            )
        else:
            logger.error(
                "Notification delivery error",
                extra={
                    "config_id": config.id,
                    "task_id": config.task_id,
                    "url": config.url,
                    "attempt": attempt + 1,
                    "error": str(error)
                }
            )

        # If this is the last attempt, track the failure
        if not should_retry:
            # Track failure synchronously instead of async task
            try:
                await self._track_delivery_attempt(
                    config,
                    event,
                    response=response,
                    response_time=time.time() - start_time,
                    error=str(error)
                )
            except Exception as track_error:
                logger.error(
                    "Failed to track delivery attempt",
                    extra={
                        "config_id": config.id,
                        "task_id": config.task_id,
                        "track_error": str(track_error)
                    }
                )

        return should_retry

    async def _send_single_notification(
        self,
        config: TaskPushNotificationConfig,
        event: Dict[str, Any]
    ) -> bool:
        """Send a notification to a single endpoint with retry logic."""
        if not self.http_client:
            logger.error("HTTP client not initialized")
            return False

        event_type_str = event.get("event", "")
        # Create notification payload
        payload = self._create_notification_payload(event)
        payload_dict = payload.to_dict()
        dev_tool_meta = event.get("development_tool_metadata")
        if dev_tool_meta:
            kind_map = {
                EventType.TASK_STATUS_CHANGE.value: DevelopmentToolEventKind.TOOL_CALL_UPDATE,
                EventType.TASK_MESSAGE.value: DevelopmentToolEventKind.THOUGHT,
                EventType.TASK_ARTIFACT.value: DevelopmentToolEventKind.TOOL_CALL_UPDATE,
                EventType.TASK_INPUT_REQUIRED.value: DevelopmentToolEventKind.TOOL_CALL_CONFIRMATION,
            }
            kind = kind_map.get(event_type_str, DevelopmentToolEventKind.UNSPECIFIED)
            dev_event = DevelopmentToolEvent(
                kind=kind,
                data=dev_tool_meta
            )
            payload_dict["development_tool_event"] = dev_event.to_dict()

        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "A2A-ACP-PushNotificationManager/1.0"
        }

        # Add authentication headers
        auth_headers = self._get_authentication_headers(config)
        headers.update(auth_headers)

        # Attempt delivery with retries
        for attempt in range(self.settings.retry_attempts + 1):
            start_time = time.time()
            try:
                response: Response = await self.http_client.post(
                    config.url,
                    json=payload_dict,
                    headers=headers
                )

                response_time = time.time() - start_time

                # Track delivery attempt
                await self._track_delivery_attempt(
                    config,
                    event,
                    response,
                    response_time
                )

                if response.is_success:
                    logger.info(
                        "Notification delivered successfully",
                        extra={
                            "config_id": config.id,
                            "task_id": config.task_id,
                            "url": config.url,
                            "response_time": response_time,
                            "status_code": response.status_code
                        }
                    )
                    return True
                else:
                    logger.warning(
                        "Notification delivery failed",
                        extra={
                            "config_id": config.id,
                            "task_id": config.task_id,
                            "url": config.url,
                            "status_code": response.status_code,
                            "response_body": response.text[:500],  # Limit log size
                            "attempt": attempt + 1
                        }
                    )

                    # If this is the last attempt, track the failure
                    if attempt == self.settings.retry_attempts:
                        asyncio.create_task(self._track_delivery_attempt(
                            config,
                            event,
                            response=response,
                            response_time=response_time,
                            error=f"HTTP {response.status_code}"
                        ))
                        return False

            except Exception as e:
                # Use helper method to handle errors consistently
                should_retry = await self._handle_notification_error(config, event, e, attempt, start_time)
                if not should_retry:
                    return False

            # Wait before retry (exponential backoff)
            if attempt < self.settings.retry_attempts:
                delay = min(self.settings.retry_delay * (2 ** attempt), self.settings.max_retry_delay)
                await asyncio.sleep(delay)

        return False

    def _create_notification_payload(self, event: Dict[str, Any]) -> NotificationPayload:
        """Create a standardized notification payload from event data."""
        event_type_str = event.get("event", "unknown")
        task_id = event.get("task_id", "")
        timestamp = datetime.now(timezone.utc)
    
        # Handle legacy event type strings by mapping them to enum values
        event_type_mapping = {
            "status_change": EventType.TASK_STATUS_CHANGE,
            "message": EventType.TASK_MESSAGE,
            "artifact": EventType.TASK_ARTIFACT,
            "input_required": EventType.TASK_INPUT_REQUIRED,
            "task_created": EventType.TASK_STATUS_CHANGE,  # Map task creation to status change
            "completed": EventType.TASK_COMPLETED,
            "failed": EventType.TASK_FAILED,
            "concurrent_test": EventType.TASK_STATUS_CHANGE,
            "test_event": EventType.TASK_STATUS_CHANGE,
            "resilience_test": EventType.TASK_STATUS_CHANGE,
            "large_payload_test": EventType.TASK_STATUS_CHANGE,
            "performance_test": EventType.TASK_STATUS_CHANGE
        }
        event_type = event_type_mapping.get(event_type_str, EventType.TASK_STATUS_CHANGE)
    
        # Preserve original event string for payload (for backward compatibility)
        # but use enum for internal type determination
        payload_event = event_type_str  # Use original string from event
    
        # Create base payload
        if event_type in [EventType.TASK_STATUS_CHANGE, EventType.TASK_COMPLETED, EventType.TASK_FAILED]:
            base_payload = TaskStatusChangePayload(
                event=payload_event,
                task_id=task_id,
                timestamp=timestamp,
                data=event,
                old_state=event.get("old_state", ""),
                new_state=event.get("new_state", "")
            )
        elif event_type == EventType.TASK_MESSAGE:
            base_payload = TaskMessagePayload(
                event=payload_event,
                task_id=task_id,
                timestamp=timestamp,
                data=event,
                message_role=event.get("message_role", ""),
                message_content=event.get("message_content", "")
            )
        elif event_type == EventType.TASK_ARTIFACT:
            base_payload = TaskArtifactPayload(
                event=payload_event,
                task_id=task_id,
                timestamp=timestamp,
                data=event,
                artifact_type=event.get("artifact_type", ""),
                artifact_name=event.get("artifact_name", ""),
                artifact_size=event.get("artifact_size")
            )
        else:
            # Generic payload for unknown event types
            base_payload = NotificationPayload(
                event=payload_event,
                task_id=task_id,
                timestamp=timestamp,
                data=event
            )

        return base_payload

    def _get_authentication_headers(
        self,
        config: TaskPushNotificationConfig
    ) -> Dict[str, str]:
        """Get authentication headers for the notification request."""
        headers = {}

        if config.token:
            headers["Authorization"] = f"Bearer {config.token}"

        # Handle custom authentication schemes
        if config.authentication_schemes:
            for scheme_name, scheme_config in config.authentication_schemes.items():
                scheme_type = scheme_config.get("type", "").lower()

                if scheme_type == "apikey":
                    header_name = scheme_config.get("header_name", "X-API-Key")
                    headers[header_name] = config.credentials or ""
                elif scheme_type == "basic":
                    # Basic auth would need username:password in credentials
                    if config.credentials:
                        import base64
                        headers["Authorization"] = f"Basic {base64.b64encode(config.credentials.encode()).decode()}"
                elif scheme_type == "custom":
                    header_name = scheme_config.get("header_name", "")
                    header_value = scheme_config.get("header_value", "")
                    if header_name and header_value:
                        headers[header_name] = header_value

        return headers

    async def _track_delivery_attempt(
        self,
        config: TaskPushNotificationConfig,
        event: Dict[str, Any],
        response: Optional[Response],
        response_time: float,
        error: Optional[str] = None
    ) -> None:
        """Track a delivery attempt in the database."""
        delivery_id = str(uuid.uuid4())
        event_type = event.get("event", "unknown")

        # Determine delivery status
        if error:
            delivery_status = DeliveryStatus.FAILED.value
            response_code = None
            response_body = error
        elif response and response.is_success:
            delivery_status = DeliveryStatus.DELIVERED.value
            response_code = response.status_code
            response_body = response.text[:1000] if response.text else None
        else:
            delivery_status = DeliveryStatus.FAILED.value
            response_code = response.status_code if response else None
            response_body = response.text[:1000] if response and response.text else None

        # Track in database
        try:
            await self.database.track_notification_delivery(
                delivery_id=delivery_id,
                config_id=config.id,
                task_id=config.task_id,
                event_type=event_type,
                delivery_status=delivery_status,
                response_code=response_code,
                response_body=response_body,
                delivered_at=datetime.now(timezone.utc).isoformat() if delivery_status == DeliveryStatus.DELIVERED.value else None
            )
        except Exception as e:
            logger.error(
                "Failed to track delivery attempt",
                extra={
                    "delivery_id": delivery_id,
                    "config_id": config.id,
                    "error": str(e)
                }
            )

    def _track_deliveries(
        self,
        configs: List[TaskPushNotificationConfig],
        event: Dict[str, Any],
        success_count: int,
        total_count: int
    ) -> None:
        """Update analytics with delivery results."""
        # Create mock delivery records for analytics
        event_type = event.get("event", "unknown")

        for i, config in enumerate(configs):
            delivery = NotificationDelivery(
                id=str(uuid.uuid4()),
                config_id=config.id,
                task_id=config.task_id,
                event_type=event_type,
                delivery_status=DeliveryStatus.DELIVERED.value if i < success_count else DeliveryStatus.FAILED.value,
                attempted_at=datetime.now(timezone.utc)
            )

            self.analytics.update_from_delivery(delivery)

    async def cleanup_expired_configs(self) -> int:
        """Clean up expired notification configurations."""
        try:
            cleaned_count = await self.database.cleanup_expired_notification_configs()
            if cleaned_count > 0:
                logger.info(
                    "Cleaned up expired notification configurations",
                    extra={"count": cleaned_count}
                )
            return cleaned_count
        except Exception as e:
            logger.error(
                "Failed to cleanup expired notification configs",
                extra={"error": str(e)}
            )
            return 0

    async def cleanup_by_task_state(self, task_id: str, task_state: str) -> int:
        """Clean up notification configurations based on task state."""
        cleanup_strategy = {
            "failed": 0,      # Delete immediately for failed tasks
            "cancelled": 0,   # Delete immediately for cancelled tasks
            "completed": 24,  # Keep for 24 hours for completed tasks
            "input-required": 168,  # Keep for 7 days for auth-required tasks (may be resumed)
        }

        retention_hours = cleanup_strategy.get(task_state.lower(), 24)

        try:
            # If retention is 0, delete immediately
            if retention_hours == 0:
                # Delete all configs for this task
                configs = await self.list_configs(task_id)
                deleted_count = 0

                for config in configs:
                    success = await self.delete_config(task_id, config.id)
                    if success:
                        deleted_count += 1

                logger.info(
                    "Cleaned up notification configs for task",
                    extra={
                        "task_id": task_id,
                        "task_state": task_state,
                        "deleted_count": deleted_count
                    }
                )
                return deleted_count

            # For tasks with retention period, mark them for later cleanup
            # This would typically update a cleanup timestamp in the database
            logger.debug(
                "Marked notification configs for cleanup",
                extra={
                    "task_id": task_id,
                    "task_state": task_state,
                    "retention_hours": retention_hours
                }
            )

            return 0  # No immediate cleanup for tasks with retention

        except Exception as e:
            logger.error(
                "Failed to cleanup configs by task state",
                extra={"task_id": task_id, "task_state": task_state, "error": str(e)}
            )
            return 0

    async def get_delivery_history(
        self,
        task_id: Optional[str] = None,
        config_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[NotificationDelivery]:
        """Get notification delivery history."""
        try:
            delivery_data = await self.database.get_notification_delivery_history(
                task_id=task_id,
                config_id=config_id,
                limit=limit
            )

            deliveries = []
            for data in delivery_data:
                delivery = NotificationDelivery.from_dict(data)
                deliveries.append(delivery)

            return deliveries
        except Exception as e:
            logger.error(
                "Failed to get delivery history",
                extra={"error": str(e)}
            )
            return []

    def get_analytics(self) -> NotificationAnalytics:
        """Get current notification analytics."""
        return self.analytics

    async def close(self) -> None:
        """Close the HTTP client and cleanup resources."""
        if self.http_client:
            await self.http_client.aclose()
            self.http_client = None
