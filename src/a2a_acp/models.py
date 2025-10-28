"""
ZedACP-specific models for push notifications.

This module defines the data models used by ZedACP for handling push notification
configurations, delivery tracking, and related functionality.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import field
from enum import Enum


class DeliveryStatus(Enum):
    """Enumeration of possible notification delivery statuses."""
    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"


class EventType(Enum):
    """Enumeration of possible notification event types."""
    TASK_STATUS_CHANGE = "task_status_change"
    TASK_MESSAGE = "task_message"
    TASK_ARTIFACT = "task_artifact"
    TASK_INPUT_REQUIRED = "task_input_required"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_GOVERNOR_FOLLOWUP = "task_governor_followup"
    TASK_FEEDBACK_REQUIRED = "task_feedback_required"


@dataclass
class TaskPushNotificationConfig:
    """Configuration for push notifications for a specific task."""

    id: str
    task_id: str
    url: str  # Webhook URL or endpoint
    token: Optional[str] = None  # Bearer token for authentication
    authentication_schemes: Optional[Dict[str, Any]] = None  # Custom auth schemes
    credentials: Optional[str] = None  # API key or other credentials
    # Notification filtering settings
    enabled_events: Optional[List[str]] = None
    disabled_events: Optional[List[str]] = None
    quiet_hours_start: Optional[str] = None  # HH:MM format
    quiet_hours_end: Optional[str] = None    # HH:MM format
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        if self.created_at:
            data['created_at'] = self.created_at.isoformat()
        if self.updated_at:
            data['updated_at'] = self.updated_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskPushNotificationConfig':
        """Create from dictionary."""
        if 'created_at' in data and data['created_at']:
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data and data['updated_at']:
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)


@dataclass
class NotificationDelivery:
    """Represents a single notification delivery attempt."""

    id: str
    config_id: str
    task_id: str
    event_type: str
    delivery_status: str
    response_code: Optional[int] = None
    response_body: Optional[str] = None
    attempted_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        if self.attempted_at:
            data['attempted_at'] = self.attempted_at.isoformat()
        if self.delivered_at:
            data['delivered_at'] = self.delivered_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NotificationDelivery':
        """Create from dictionary."""
        if 'attempted_at' in data and data['attempted_at']:
            data['attempted_at'] = datetime.fromisoformat(data['attempted_at'])
        if 'delivered_at' in data and data['delivered_at']:
            data['delivered_at'] = datetime.fromisoformat(data['delivered_at'])
        return cls(**data)


@dataclass
class NotificationPayload:
    """Standardized payload structure for push notifications."""

    event: str
    task_id: str
    timestamp: datetime
    data: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class TaskStatusChangePayload(NotificationPayload):
    """Payload for task status change notifications."""

    old_state: str
    new_state: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = super().to_dict()
        data.update({
            'event_type': 'task_status_change',
            'old_state': self.old_state,
            'new_state': self.new_state
        })
        return data


@dataclass
class TaskMessagePayload(NotificationPayload):
    """Payload for task message notifications."""

    message_role: str
    message_content: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = super().to_dict()
        data.update({
            'event_type': 'task_message',
            'message_role': self.message_role,
            'message_content': self.message_content
        })
        return data


@dataclass
class TaskArtifactPayload(NotificationPayload):
    """Payload for task artifact notifications."""

    artifact_type: str
    artifact_name: str
    artifact_size: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = super().to_dict()
        data.update({
            'event_type': 'task_artifact',
            'artifact_type': self.artifact_type,
            'artifact_name': self.artifact_name,
            'artifact_size': self.artifact_size
        })
        return data


@dataclass
class NotificationFilter:
    """Configuration for filtering which notifications to send."""

    enabled_events: List[str]
    disabled_events: Optional[List[str]] = field(default_factory=list)
    minimum_priority: Optional[str] = None
    quiet_hours_start: Optional[str] = None  # HH:MM format
    quiet_hours_end: Optional[str] = None    # HH:MM format

    def __post_init__(self):
        if self.disabled_events is None:
            self.disabled_events = []

    def should_send_notification(self, event_type: str, current_hour: Optional[int] = None) -> bool:
        """Check if notification should be sent based on filter rules."""
        # Check if event is explicitly disabled
        if self.disabled_events and event_type in self.disabled_events:
            return False

        # Check if event is in enabled list (if specified)
        if self.enabled_events and event_type not in self.enabled_events:
            return False

        # Check quiet hours
        if current_hour is not None and self.quiet_hours_start and self.quiet_hours_end:
            quiet_start = int(self.quiet_hours_start.split(':')[0])
            quiet_end = int(self.quiet_hours_end.split(':')[0])

            if quiet_start <= quiet_end:
                # Same day quiet hours (e.g., 22:00 to 08:00)
                if quiet_start <= current_hour < quiet_end:
                    return False
            else:
                # Overnight quiet hours (e.g., 22:00 to 08:00 next day)
                if current_hour >= quiet_start or current_hour < quiet_end:
                    return False

        return True


@dataclass
class NotificationAnalytics:
    """Analytics data for notification delivery performance."""

    total_sent: int = 0
    total_delivered: int = 0
    total_failed: int = 0
    average_response_time_ms: float = 0.0
    success_rate: float = 0.0
    events_by_type: Optional[Dict[str, int]] = field(default_factory=dict)

    def __post_init__(self):
        if self.events_by_type is None:
            self.events_by_type = {}

    def update_from_delivery(self, delivery: NotificationDelivery) -> None:
        """Update analytics from a delivery record."""
        self.total_sent += 1

        if delivery.delivery_status == DeliveryStatus.DELIVERED.value:
            self.total_delivered += 1
        elif delivery.delivery_status == DeliveryStatus.FAILED.value:
            self.total_failed += 1

        # Track events by type
        if self.events_by_type is None:
            self.events_by_type = {}
        self.events_by_type[delivery.event_type] = self.events_by_type.get(delivery.event_type, 0) + 1

        # Update success rate
        if self.total_sent > 0:
            self.success_rate = (self.total_delivered / self.total_sent) * 100.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
