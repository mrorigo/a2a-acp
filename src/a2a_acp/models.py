"""
ZedACP-specific models for push notifications.

This module defines the data models used by ZedACP for handling push notification
configurations, delivery tracking, and related functionality.
"""

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union


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

class ToolCallStatus(Enum):
    """Enumeration of tool call statuses."""
    PENDING = "pending"
    EXECUTING = "executing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

class DevelopmentToolEventKind(Enum):
    """Enumeration of development tool event kinds."""
    UNSPECIFIED = "unspecified"
    TOOL_CALL_CONFIRMATION = "tool_call_confirmation"
    TOOL_CALL_UPDATE = "tool_call_update"
    TEXT_CONTENT = "text_content"
    STATE_CHANGE = "state_change"
    THOUGHT = "thought"
    GOVERNANCE_EVENT = "governance_event"

class CommandExecutionStatus(Enum):
    """Enumeration of command execution statuses."""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class ConfirmationOption:
    """An option for user confirmation in a confirmation request."""
    id: str
    name: str
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConfirmationOption':
        """Create from dictionary."""
        return cls(**data)

@dataclass
class GenericDetails:
    """Generic details for a confirmation request or tool output."""
    description: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GenericDetails':
        """Create from dictionary."""
        return cls(**data)

@dataclass
class FileDiff:
    """Details for a file diff in confirmation or output."""
    path: str
    old_content: Optional[str] = None
    new_content: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FileDiff':
        """Create from dictionary."""
        return cls(**data)

@dataclass
class McpDetails:
    """Details for an MCP tool in confirmation or output."""
    tool_name: str
    server_id: Optional[str] = None
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'McpDetails':
        """Create from dictionary."""
        return cls(**data)

@dataclass
class ExecuteDetails:
    """Details for an execute operation in confirmation or output."""
    stdout: str = ""
    stderr: Optional[str] = None
    exit_code: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExecuteDetails':
        """Create from dictionary."""
        return cls(**data)

@dataclass
class ToolOutput:
    """Output from a successful tool call."""
    content: str
    details: Optional[Union[ExecuteDetails, FileDiff, McpDetails]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        if self.details:
            data['details'] = self.details.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolOutput':
        """Create from dictionary, handling union for details."""
        details = None
        details_data = data.get('details')
        if details_data:
            if isinstance(details_data, dict):
                if 'path' in details_data:
                    details = FileDiff.from_dict(details_data)
                elif 'stdout' in details_data or 'exit_code' in details_data:
                    details = ExecuteDetails.from_dict(details_data)
                elif 'tool_name' in details_data:
                    details = McpDetails.from_dict(details_data)
                else:
                    details = None
            else:
                details = details_data
        return cls(content=data['content'], details=details)

@dataclass
class ErrorDetails:
    """Error details from a failed tool call."""
    message: str
    code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        if self.details:
            data['details'] = json.dumps(self.details) if isinstance(self.details, dict) else self.details
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ErrorDetails':
        """Create from dictionary."""
        details = data.get('details')
        if isinstance(details, str):
            try:
                details = json.loads(details)
            except json.JSONDecodeError:
                details = {}
        return cls(message=data['message'], code=data.get('code'), details=details)

@dataclass
class ConfirmationRequest:
    """Request for user confirmation on a tool call."""
    options: List[ConfirmationOption]
    details: Union[GenericDetails, ExecuteDetails, FileDiff, McpDetails]
    title: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['options'] = [option.to_dict() for option in self.options]
        data['details'] = self.details.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConfirmationRequest':
        """Create from dictionary, handling union for details."""
        options = [ConfirmationOption.from_dict(opt) for opt in data['options']]
        details_data = data.get('details', {})
        details = None
        if isinstance(details_data, dict):
            if 'description' in details_data:
                details = GenericDetails.from_dict(details_data)
            elif 'path' in details_data:
                details = FileDiff.from_dict(details_data)
            elif 'stdout' in details_data or 'exit_code' in details_data:
                details = ExecuteDetails.from_dict(details_data)
            elif 'tool_name' in details_data:
                details = McpDetails.from_dict(details_data)
            else:
                details = GenericDetails.from_dict({'description': str(details_data)})
        else:
            details = details_data
        other = {k: v for k, v in data.items() if k not in ['options', 'details']}
        return cls(options=options, details=details, **other)

@dataclass
class ToolCall:
    """A tool call in the development tool extension."""
    tool_call_id: str
    status: ToolCallStatus
    tool_name: str
    input_parameters: Dict[str, Any]
    confirmation_request: Optional[ConfirmationRequest] = None
    result: Optional[Union[ToolOutput, ErrorDetails]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        if isinstance(self.status, ToolCallStatus):
            data['status'] = self.status.value
        if self.confirmation_request:
            data['confirmation_request'] = self.confirmation_request.to_dict()
        if self.result:
            data['result'] = self.result.to_dict()
        # Handle datetimes if added later
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolCall':
        """Create from dictionary, handling nested objects and unions."""
        params = data.copy()
        if 'status' in params and isinstance(params['status'], str):
            params['status'] = ToolCallStatus(params['status'])
        confirmation_request = None
        if 'confirmation_request' in data and data['confirmation_request'] is not None:
            confirmation_request = ConfirmationRequest.from_dict(data['confirmation_request'])
        result = None
        if 'result' in data and data['result'] is not None:
            res_data = data['result']
            if isinstance(res_data, dict):
                if 'content' in res_data:
                    result = ToolOutput.from_dict(res_data)
                elif 'message' in res_data:
                    result = ErrorDetails.from_dict(res_data)
                else:
                    result = None
            else:
                result = res_data
        # input_parameters remains dict
        return cls(
            tool_call_id=data.get('tool_call_id'),
            status=params['status'],
            tool_name=data.get('tool_name'),
            input_parameters=data.get('input_parameters', {}),
            confirmation_request=confirmation_request,
            result=result
        )

@dataclass
class AgentThought:
    """An agent's thought for transparency."""
    content: str
    timestamp: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        if self.timestamp:
            data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentThought':
        """Create from dictionary."""
        if 'timestamp' in data and data['timestamp']:
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

@dataclass
class DevelopmentToolEvent:
    """An event in the development tool extension."""
    kind: DevelopmentToolEventKind
    model: Optional[str] = None
    user_tier: Optional[str] = None
    error: Optional[str] = None
    data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data_dict = asdict(self)
        if isinstance(self.kind, DevelopmentToolEventKind):
            data_dict["kind"] = self.kind.value
        if self.data:
            data_dict['data'] = self.data  # Assume already serializable
        return data_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DevelopmentToolEvent':
        """Create from dictionary."""
        if 'kind' in data and isinstance(data['kind'], str):
            data['kind'] = DevelopmentToolEventKind(data['kind'])
        return cls(**data)

@dataclass
class ToolCallConfirmation:
    """Confirmation response for a tool call."""
    tool_call_id: str
    selected_option_id: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolCallConfirmation':
        """Create from dictionary."""
        return cls(**data)

@dataclass
class AgentSettings:
    """Settings for the agent in the development tool extension."""
    workspace_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentSettings':
        """Create from dictionary."""
        return cls(**data)

@dataclass
class SlashCommandArgument:
    """An argument for a slash command."""
    name: str
    type: str  # e.g., "string", "number"
    description: str
    required: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SlashCommandArgument':
        """Create from dictionary."""
        return cls(**data)

@dataclass
class SlashCommand:
    """A slash command definition."""
    name: str
    description: str
    arguments: List[SlashCommandArgument] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['arguments'] = [arg.to_dict() for arg in self.arguments]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SlashCommand':
        """Create from dictionary."""
        arguments = [SlashCommandArgument.from_dict(arg) for arg in data.get('arguments', [])]
        other = {k: v for k, v in data.items() if k != 'arguments'}
        return cls(arguments=arguments, **other)

@dataclass
class ExecuteSlashCommandRequest:
    """Request to execute a slash command."""
    command: str
    arguments: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExecuteSlashCommandRequest':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ExecuteSlashCommandResponse:
    """Response from executing a slash command."""
    execution_id: str
    status: CommandExecutionStatus

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        if isinstance(self.status, str):
            data['status'] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExecuteSlashCommandResponse':
        """Create from dictionary."""
        if 'status' in data and isinstance(data['status'], str):
            data['status'] = CommandExecutionStatus(data['status'])
        return cls(**data)


@dataclass
class GetAllSlashCommandsResponse:
    """Response containing all available slash commands."""
    commands: List[SlashCommand]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['commands'] = [cmd.to_dict() for cmd in self.commands]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GetAllSlashCommandsResponse':
        """Create from dictionary."""
        commands = [SlashCommand.from_dict(cmd) for cmd in data.get('commands', [])]
        return cls(commands=commands)
