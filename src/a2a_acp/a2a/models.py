"""
A2A (Agent-to-Agent) Protocol Data Models

Complete Python implementation of A2A types based on the TypeScript definitions
in docs/a2a/types.ts. These models provide type safety and validation for the
A2A protocol implementation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


# ===== FOUNDATIONAL TYPES =====

class TransportProtocol(str, Enum):
    """Supported A2A transport protocols."""
    JSONRPC = "JSONRPC"
    GRPC = "GRPC"
    HTTP_JSON = "HTTP+JSON"


class TaskState(str, Enum):
    """Defines the lifecycle states of a Task."""
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    REJECTED = "rejected"
    AUTH_REQUIRED = "auth-required"
    UNKNOWN = "unknown"


# ===== SECURITY SCHEMES =====

class SecuritySchemeBase(BaseModel):
    """Base properties shared by all security scheme objects."""
    description: Optional[str] = None


class APIKeySecurityScheme(SecuritySchemeBase):
    """Defines a security scheme using an API key."""
    type: Literal["apiKey"] = "apiKey"
    in_: str = Field(alias="in", description="The location of the API key")
    name: str = Field(description="The name of the header, query, or cookie parameter")

    @field_validator("in_")
    def validate_in(cls, v):
        if v not in ["query", "header", "cookie"]:
            raise ValueError("API key location must be 'query', 'header', or 'cookie'")
        return v


class HTTPAuthSecurityScheme(SecuritySchemeBase):
    """Defines a security scheme using HTTP authentication."""
    type: Literal["http"] = "http"
    scheme: str = Field(description="The HTTP authentication scheme")
    bearerFormat: Optional[str] = Field(None, description="Bearer token format hint")


class MutualTLSSecurityScheme(SecuritySchemeBase):
    """Defines a security scheme using mTLS authentication."""
    type: Literal["mutualTLS"] = "mutualTLS"


class OAuthFlows(BaseModel):
    """Defines the configuration for the supported OAuth 2.0 flows."""
    authorizationCode: Optional[AuthorizationCodeOAuthFlow] = None
    clientCredentials: Optional[ClientCredentialsOAuthFlow] = None
    implicit: Optional[ImplicitOAuthFlow] = None
    password: Optional[PasswordOAuthFlow] = None


class AuthorizationCodeOAuthFlow(BaseModel):
    """Defines configuration details for the OAuth 2.0 Authorization Code flow."""
    authorizationUrl: str
    tokenUrl: str
    refreshUrl: Optional[str] = None
    scopes: Dict[str, str]


class ClientCredentialsOAuthFlow(BaseModel):
    """Defines configuration details for the OAuth 2.0 Client Credentials flow."""
    tokenUrl: str
    refreshUrl: Optional[str] = None
    scopes: Dict[str, str]


class ImplicitOAuthFlow(BaseModel):
    """Defines configuration details for the OAuth 2.0 Implicit flow."""
    authorizationUrl: str
    refreshUrl: Optional[str] = None
    scopes: Dict[str, str]


class PasswordOAuthFlow(BaseModel):
    """Defines configuration details for the OAuth 2.0 Resource Owner Password flow."""
    tokenUrl: str
    refreshUrl: Optional[str] = None
    scopes: Dict[str, str]


class OAuth2SecurityScheme(SecuritySchemeBase):
    """Defines a security scheme using OAuth 2.0."""
    type: Literal["oauth2"] = "oauth2"
    flows: OAuthFlows
    oauth2MetadataUrl: Optional[str] = None


class OpenIdConnectSecurityScheme(SecuritySchemeBase):
    """Defines a security scheme using OpenID Connect."""
    type: Literal["openIdConnect"] = "openIdConnect"
    openIdConnectUrl: str


# Union type for all security schemes
SecurityScheme = Union[
    APIKeySecurityScheme,
    HTTPAuthSecurityScheme,
    OAuth2SecurityScheme,
    OpenIdConnectSecurityScheme,
    MutualTLSSecurityScheme
]


# ===== AGENT CARD COMPONENTS =====

class AgentProvider(BaseModel):
    """Represents the service provider of an agent."""
    organization: str
    url: str


class AgentCapabilities(BaseModel):
    """Defines optional capabilities supported by an agent."""
    streaming: Optional[bool] = None
    pushNotifications: Optional[bool] = None
    stateTransitionHistory: Optional[bool] = None
    extensions: Optional[List[AgentExtension]] = None


class AgentExtension(BaseModel):
    """A declaration of a protocol extension supported by an Agent."""
    uri: str
    description: Optional[str] = None
    required: Optional[bool] = None
    params: Optional[Dict[str, Any]] = None


class AgentSkill(BaseModel):
    """Represents a distinct capability or function that an agent can perform."""
    id: str
    name: str
    description: str
    tags: List[str]
    examples: Optional[List[str]] = None
    inputModes: Optional[List[str]] = None
    outputModes: Optional[List[str]] = None
    security: Optional[List[Dict[str, List[str]]]] = None


class AgentInterface(BaseModel):
    """Declares a combination of a target URL and a transport protocol."""
    url: str
    transport: Union[TransportProtocol, str]


class AgentCardSignature(BaseModel):
    """AgentCardSignature represents a JWS signature of an AgentCard."""
    protected: str
    signature: str
    header: Optional[Dict[str, Any]] = None


class AgentCard(BaseModel):
    """The AgentCard is a self-describing manifest for an agent."""
    protocolVersion: str = "0.3.0"
    name: str
    description: str
    url: str
    preferredTransport: Optional[Union[TransportProtocol, str]] = None
    additionalInterfaces: Optional[List[AgentInterface]] = None
    iconUrl: Optional[str] = None
    provider: Optional[AgentProvider] = None
    version: str
    documentationUrl: Optional[str] = None
    capabilities: AgentCapabilities
    securitySchemes: Optional[Dict[str, SecurityScheme]] = None
    security: Optional[List[Dict[str, List[str]]]] = None
    defaultInputModes: List[str] = ["text/plain"]
    defaultOutputModes: List[str] = ["text/plain"]
    skills: List[AgentSkill]
    supportsAuthenticatedExtendedCard: Optional[bool] = None
    signatures: Optional[List[AgentCardSignature]] = None


# ===== TASK AND MESSAGE TYPES =====

class TaskStatus(BaseModel):
    """Represents the status of a task at a specific point in time."""
    state: TaskState
    message: Optional[Message] = None
    timestamp: Optional[str] = None


class Task(BaseModel):
    """Represents a single, stateful operation or conversation between a client and an agent."""
    id: str
    contextId: str
    status: TaskStatus
    history: Optional[List[Message]] = None
    artifacts: Optional[List[Artifact]] = None
    metadata: Optional[Dict[str, Any]] = None
    kind: Literal["task"] = "task"


class Message(BaseModel):
    """Represents a single message in the conversation between a user and an agent."""
    role: Literal["user", "agent"]
    parts: List[Part]
    messageId: str
    taskId: Optional[str] = None
    contextId: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    extensions: Optional[List[str]] = None
    referenceTaskIds: Optional[List[str]] = None
    kind: Literal["message"] = "message"


# ===== CONTENT PARTS =====

class PartBase(BaseModel):
    """Defines base properties common to all message or artifact parts."""
    metadata: Optional[Dict[str, Any]] = None


class TextPart(PartBase):
    """Represents a text segment within a message or artifact."""
    kind: Literal["text"] = "text"
    text: str


class FileBase(BaseModel):
    """Defines base properties for a file."""
    name: Optional[str] = None
    mimeType: Optional[str] = None


class FileWithBytes(FileBase):
    """Represents a file with its content provided directly as a base64-encoded string."""
    bytes: str
    uri: Optional[str] = None  # Must be absent when bytes is present


class FileWithUri(FileBase):
    """Represents a file with its content located at a specific URI."""
    uri: str
    bytes: Optional[str] = None  # Must be absent when uri is present


class FilePart(PartBase):
    """Represents a file segment within a message or artifact."""
    kind: Literal["file"] = "file"
    file: Union[FileWithBytes, FileWithUri]


class DataPart(PartBase):
    """Represents a structured data segment within a message or artifact."""
    kind: Literal["data"] = "data"
    data: Dict[str, Any]


# Union type for all parts (simple union without discriminator)
Part = Union[TextPart, FilePart, DataPart]


# ===== ARTIFACTS =====

class Artifact(BaseModel):
    """Represents a file, data structure, or other resource generated by an agent."""
    artifactId: str
    name: Optional[str] = None
    description: Optional[str] = None
    parts: List[Part]
    metadata: Optional[Dict[str, Any]] = None
    extensions: Optional[List[str]] = None


# ===== PUSH NOTIFICATIONS =====

class PushNotificationAuthenticationInfo(BaseModel):
    """Defines authentication details for a push notification endpoint."""
    schemes: List[str]
    credentials: Optional[str] = None


class PushNotificationConfig(BaseModel):
    """Defines the configuration for setting up push notifications."""
    id: Optional[str] = None
    url: str
    token: Optional[str] = None
    authentication: Optional[PushNotificationAuthenticationInfo] = None


class TaskPushNotificationConfig(BaseModel):
    """A container associating a push notification configuration with a specific task."""
    taskId: str
    pushNotificationConfig: PushNotificationConfig


# ===== JSON-RPC 2.0 TYPES =====

class JSONRPCMessage(BaseModel):
    """Base structure for any JSON-RPC 2.0 request, response, or notification."""
    jsonrpc: Literal["2.0"] = "2.0"
    id: Optional[Union[int, str, None]] = None


class JSONRPCRequest(JSONRPCMessage):
    """Represents a JSON-RPC 2.0 Request object."""
    method: str
    params: Optional[Dict[str, Any]] = None


class JSONRPCError(BaseModel):
    """Represents a JSON-RPC 2.0 Error object."""
    code: int
    message: str
    data: Optional[Any] = None


class JSONRPCSuccessResponse(BaseModel):
    """Represents a successful JSON-RPC 2.0 Response object."""
    jsonrpc: Literal["2.0"] = "2.0"
    id: Union[int, str]  # Required in responses
    result: Any
    error: Optional[None] = None  # Must not exist in success response


class JSONRPCErrorResponse(BaseModel):
    """Represents a JSON-RPC 2.0 Error Response object."""
    jsonrpc: Literal["2.0"] = "2.0"
    id: Union[int, str, None] = None  # Can be null for error responses
    result: Optional[None] = None  # Must not exist in error response
    error: JSONRPCError


# ===== A2A REQUEST/RESPONSE TYPES =====

class MessageSendConfiguration(BaseModel):
    """Configuration options for message/send or message/stream requests."""
    acceptedOutputModes: Optional[List[str]] = None
    historyLength: Optional[int] = None
    pushNotificationConfig: Optional[PushNotificationConfig] = None
    blocking: Optional[bool] = None


class MessageSendParams(BaseModel):
    """Parameters for a request to send a message to an agent."""
    message: Message
    configuration: Optional[MessageSendConfiguration] = None
    metadata: Optional[Dict[str, Any]] = None


class TaskIdParams(BaseModel):
    """Parameters containing a task ID for simple task operations."""
    id: str
    metadata: Optional[Dict[str, Any]] = None


class TaskQueryParams(TaskIdParams):
    """Parameters for querying a task with optional history length."""
    historyLength: Optional[int] = None


class ListTasksParams(BaseModel):
    """Parameters for listing tasks with optional filtering criteria."""
    contextId: Optional[str] = None
    status: Optional[TaskState] = None
    pageSize: Optional[int] = None
    pageToken: Optional[str] = None
    historyLength: Optional[int] = None
    lastUpdatedAfter: Optional[int] = None
    includeArtifacts: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None


class ListTasksResult(BaseModel):
    """Result object for tasks/list method containing tasks and pagination."""
    tasks: List[Task]
    totalSize: int
    pageSize: int
    nextPageToken: str


# ===== INPUT-REQUIRED TYPES =====

class InputRequiredNotification(BaseModel):
    """Represents a notification that a task requires user input."""
    taskId: str
    contextId: str
    kind: Literal["input-required"] = "input-required"
    message: str
    inputTypes: List[str] = ["text/plain"]
    timeout: Optional[int] = None  # Timeout in seconds
    metadata: Optional[Dict[str, Any]] = None


# ===== EVENT TYPES =====

class TaskStatusUpdateEvent(BaseModel):
    """An event sent by the agent to notify the client of a change in a task's status."""
    taskId: str
    contextId: str
    kind: Literal["status-update"] = "status-update"
    status: TaskStatus
    final: bool
    metadata: Optional[Dict[str, Any]] = None


class TaskArtifactUpdateEvent(BaseModel):
    """An event sent by the agent to notify the client that an artifact has been generated."""
    taskId: str
    contextId: str
    kind: Literal["artifact-update"] = "artifact-update"
    artifact: Artifact
    append: Optional[bool] = None
    lastChunk: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None


# ===== A2A ERROR TYPES =====

class TaskNotFoundError(BaseModel):
    """An A2A-specific error indicating that the requested task ID was not found."""
    code: int = -32001
    message: str = "Task not found"
    data: Optional[Any] = None


class TaskNotCancelableError(BaseModel):
    """An A2A-specific error indicating that the task is in a state where it cannot be canceled."""
    code: int = -32002
    message: str = "Task cannot be canceled"
    data: Optional[Any] = None


class PushNotificationNotSupportedError(BaseModel):
    """An A2A-specific error indicating that the agent does not support push notifications."""
    code: int = -32003
    message: str = "Push Notification is not supported"
    data: Optional[Any] = None


class UnsupportedOperationError(BaseModel):
    """An A2A-specific error indicating that the requested operation is not supported."""
    code: int = -32004
    message: str = "This operation is not supported"
    data: Optional[Any] = None


class ContentTypeNotSupportedError(BaseModel):
    """An A2A-specific error indicating an incompatibility between content types."""
    code: int = -32005
    message: str = "Incompatible content types"
    data: Optional[Any] = None


class InvalidAgentResponseError(BaseModel):
    """An A2A-specific error indicating that the agent returned an invalid response."""
    code: int = -32006
    message: str = "Invalid agent response"
    data: Optional[Any] = None


class AuthenticatedExtendedCardNotConfiguredError(BaseModel):
    """An A2A-specific error indicating that the agent does not have an extended card configured."""
    code: int = -32007
    message: str = "Authenticated Extended Card is not configured"
    data: Optional[Any] = None


# Union type for all A2A errors (for use in JSON-RPC error responses)
A2AError = Union[
    TaskNotFoundError,
    TaskNotCancelableError,
    PushNotificationNotSupportedError,
    UnsupportedOperationError,
    ContentTypeNotSupportedError,
    InvalidAgentResponseError,
    AuthenticatedExtendedCardNotConfiguredError
]


# ===== UTILITY FUNCTIONS =====

def generate_id(prefix: str = "") -> str:
    """Generate a unique ID for A2A entities."""
    return f"{prefix}{uuid4()}" if prefix else str(uuid4())


def create_task_id() -> str:
    """Generate a unique task ID."""
    return generate_id("task_")


def create_context_id() -> str:
    """Generate a unique context ID."""
    return generate_id("ctx_")


def create_message_id() -> str:
    """Generate a unique message ID."""
    return generate_id("msg_")


def current_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp, as required by the A2A spec."""
    return datetime.now(timezone.utc).isoformat()


def create_artifact_id() -> str:
    """Generate a unique artifact ID."""
    return generate_id("art_")


def get_current_timestamp() -> str:
    """Get current timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat() + "Z"
