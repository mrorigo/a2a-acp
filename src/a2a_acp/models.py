from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Literal, Optional, Sequence

from pydantic import BaseModel, Field, field_validator


class RunMode(str, Enum):
    """Execution modes supported by the proxy."""

    sync = "sync"
    stream = "stream"


class RunStatus(str, Enum):
    """Lifecycle states for IBMACP runs."""

    queued = "queued"
    in_progress = "in_progress"
    completed = "completed"
    failed = "failed"
    cancelling = "cancelling"
    cancelled = "cancelled"


class MessagePart(BaseModel):
    """Minimal IBMACP message part representation."""

    type: Literal["text"] = "text"
    text: str


class Message(BaseModel):
    """Minimal IBMACP message format."""

    role: Literal["user", "assistant", "system"]
    content: List[MessagePart]

    @field_validator("content")
    def validate_content(cls, value: Sequence[MessagePart]) -> Sequence[MessagePart]:
        if not value:
            raise ValueError("Message content may not be empty")
        return value


class RunCreateRequest(BaseModel):
    """Request payload for POST /runs."""

    agent: str = Field(..., description="Agent name from configuration.")
    input: Message = Field(..., description="The initial user message to send to the agent.")
    mode: RunMode = Field(default=RunMode.sync, description="Execution mode.")
    session_id: Optional[str] = Field(None, description="Optional session ID for stateful runs.")


class ErrorDetail(BaseModel):
    """Structured error detail aligned with IBMACP schema."""

    code: str
    message: str
    data: Optional[dict] = None


class Run(BaseModel):
    """Representation of an IBMACP run."""

    id: str
    agent: str
    status: RunStatus
    mode: RunMode
    created_at: datetime
    updated_at: datetime
    output: Optional[Message] = None
    stop_reason: Optional[str] = None
    error: Optional[ErrorDetail] = None


class AgentConfig(BaseModel):
    """Configuration entry loaded from agents.json."""

    name: str
    command: List[str]
    description: Optional[str] = None
    version: Optional[str] = None
    api_key: Optional[str] = None


class AgentSummary(BaseModel):
    """Public agent listing entry."""

    name: str
    description: Optional[str] = None


class AgentManifestCapabilities(BaseModel):
    """Capabilities for an agent manifest."""

    modes: List[RunMode]
    supports_streaming: bool = True
    supports_cancellation: bool = True


class AgentManifest(BaseModel):
    """Public agent manifest returned by the API."""

    name: str
    description: str
    version: str
    capabilities: AgentManifestCapabilities
