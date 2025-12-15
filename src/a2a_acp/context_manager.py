"""
A2A Context Manager

A2A-native context management that bridges A2A contexts to ACP sessions.
Replaces SessionManager with proper A2A terminology.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timezone

from a2a_acp.a2a.models import Task, Message, generate_id
from .database import SessionDatabase

# Global database instance
_db = SessionDatabase()

logger = logging.getLogger(__name__)


@dataclass
class ContextState:
    """A2A context state with ACP session mapping."""

    context_id: str
    agent_name: str
    acp_session_id: Optional[str] = None
    tasks: Optional[List[Task]] = None
    messages: Optional[List[Message]] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self):
        if self.tasks is None:
            self.tasks = []
        if self.messages is None:
            self.messages = []
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.updated_at is None:
            self.updated_at = datetime.now(timezone.utc)


class A2AContextManager:
    """
    A2A-native context manager that bridges A2A contexts to ACP sessions.

    Provides A2A-compliant context lifecycle while maintaining ACP compatibility.
    """

    def __init__(self) -> None:
        self._active_contexts: Dict[str, ContextState] = {}
        self._lock: Optional[asyncio.Lock] = None

    @property
    def lock(self) -> asyncio.Lock:
        """Get or create the asyncio lock for thread-safe operations."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def create_context(
        self, agent_name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new A2A context."""
        async with self.lock:
            context_id = generate_id("ctx_")

            # Create ACP session for this context
            acp_session_id = generate_id("sess_")
            await _db.create_a2a_context(
                context_id=acp_session_id,
                agent=agent_name,
                cwd=".",
                zed_session_id="",  # Will be set later when ZedACP session is created
                metadata=metadata or {},
            )

            # Create context state
            context_state = ContextState(
                context_id=context_id,
                agent_name=agent_name,
                acp_session_id=acp_session_id,
                metadata=metadata or {},
            )

            self._active_contexts[context_id] = context_state

            logger.info(
                "Created A2A context",
                extra={
                    "context_id": context_id,
                    "agent": agent_name,
                    "acp_session": acp_session_id,
                },
            )

            return context_id

    async def get_context(self, context_id: str) -> Optional[ContextState]:
        """Retrieve a context by ID."""
        if context_id in self._active_contexts:
            return self._active_contexts[context_id]

        # Try to load from database
        acp_session = await _db.get_a2a_context(context_id)
        if acp_session:
            # Convert ACP session to A2A context
            context_state = ContextState(
                context_id=context_id,
                agent_name=acp_session.agent_name,
                acp_session_id=acp_session.context_id,
                created_at=acp_session.created_at,
                updated_at=acp_session.updated_at,
            )
            self._active_contexts[context_id] = context_state
            return context_state

        return None

    async def add_task_to_context(self, context_id: str, task: Task) -> None:
        """Add a task to an A2A context."""
        async with self.lock:
            if context_id not in self._active_contexts:
                await self.get_context(context_id)  # Load from DB if needed

            if context_id in self._active_contexts:
                context_state = self._active_contexts[context_id]
                if context_state.tasks is not None:
                    context_state.tasks.append(task)
                context_state.updated_at = datetime.now(timezone.utc)

                logger.debug(
                    "Added task to context",
                    extra={"context_id": context_id, "task_id": task.id},
                )

    async def add_message_to_context(self, context_id: str, message: Message) -> None:
        """Add a message to an A2A context."""
        async with self.lock:
            if context_id in self._active_contexts:
                context_state = self._active_contexts[context_id]
                if context_state.messages is not None:
                    context_state.messages.append(message)
                context_state.updated_at = datetime.now(timezone.utc)

                logger.debug(
                    "Added message to context",
                    extra={"context_id": context_id, "message_id": message.messageId},
                )

    async def list_contexts(self) -> List[ContextState]:
        """List all active contexts."""
        # For now, just return active contexts in memory
        # TODO: Implement proper database loading when needed
        contexts = []
        for context_state in self._active_contexts.values():
            contexts.append(context_state)

        return contexts

    async def get_context_tasks(self, context_id: str) -> List[Task]:
        """Get all tasks for a context."""
        if context_id in self._active_contexts:
            context_state = self._active_contexts[context_id]
            return context_state.tasks or []

        # Load from database if available
        retrieved_context = await self.get_context(context_id)
        return (
            retrieved_context.tasks
            if retrieved_context and retrieved_context.tasks
            else []
        )

    async def get_context_messages(self, context_id: str) -> List[Message]:
        """Get all messages for a context."""
        if context_id in self._active_contexts:
            context_state = self._active_contexts[context_id]
            return context_state.messages or []

        # Load from database if available
        retrieved_context = await self.get_context(context_id)
        return (
            retrieved_context.messages
            if retrieved_context and retrieved_context.messages
            else []
        )

    async def close_context(self, context_id: str) -> bool:
        """Close an A2A context."""
        async with self.lock:
            if context_id in self._active_contexts:
                context_state = self._active_contexts[context_id]

                # Mark as inactive in database
                # Note: This would need to be implemented in the database layer
                pass  # Placeholder for now

                # Remove from memory
                del self._active_contexts[context_id]

                logger.info(
                    "Closed A2A context",
                    extra={
                        "context_id": context_id,
                        "acp_session": context_state.acp_session_id,
                    },
                )
                return True

            return False

    async def cleanup_old_contexts(self) -> int:
        """Clean up old inactive contexts."""
        # Mark old contexts as inactive (older than 24 hours)
        cutoff_time = datetime.now(timezone.utc).timestamp() - (24 * 3600)

        # Use the database cleanup method
        cleaned_count = await _db.cleanup_inactive_sessions(
            days_old=1
        )  # 24 hours = 1 day

        # Remove from memory
        to_remove = []
        for context_id, context_state in self._active_contexts.items():
            if (
                not context_state.updated_at
                or context_state.updated_at.timestamp() < cutoff_time
            ):
                to_remove.append(context_id)

        for context_id in to_remove:
            del self._active_contexts[context_id]

        if cleaned_count > 0:
            logger.info("Cleaned up old contexts", extra={"count": cleaned_count})

        return cleaned_count


# Global A2A-native context manager
a2a_context_manager = A2AContextManager()
