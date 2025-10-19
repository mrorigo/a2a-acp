"""
A2A ↔ ZedACP Protocol Translation Layer

This module handles the translation between A2A protocol messages and ZedACP protocol messages.
It enables seamless communication between A2A clients and ZedACP agents.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

from .models import (
    Message, MessageSendParams, Task, TextPart, FilePart, DataPart,
    TaskStatus, TaskState, generate_id, create_task_id, create_context_id,
    create_message_id, MessageSendConfiguration
)

logger = logging.getLogger(__name__)


class A2ATranslator:
    """
    Translates between A2A and ZedACP protocol formats.

    Handles the conversion of messages, tasks, contexts, and other
    protocol-specific data structures between the two protocols.
    """

    def __init__(self):
        self._active_sessions: Dict[str, Dict[str, Any]] = {}
        self._context_sessions: Dict[str, str] = {}  # context_id -> session_id

    def a2a_to_zedacp_message(self, a2a_message: Message) -> List[Dict[str, Any]]:
        """
        Convert A2A Message to ZedACP prompt format.

        Args:
            a2a_message: The A2A message to convert

        Returns:
            List of ZedACP prompt parts
        """
        zedacp_parts = []

        for part in a2a_message.parts:
            if isinstance(part, TextPart):
                zedacp_parts.append({
                    "type": "text",
                    "text": part.text
                })
            elif isinstance(part, FilePart):
                # Handle file content translation
                if hasattr(part.file, 'bytes') and part.file.bytes:
                    # File with bytes content
                    zedacp_parts.append({
                        "type": "file",
                        "name": part.file.name or "unnamed_file",
                        "content": part.file.bytes
                    })
                elif hasattr(part.file, 'uri') and part.file.uri:
                    # File with URI reference
                    zedacp_parts.append({
                        "type": "file",
                        "name": part.file.name or "unnamed_file",
                        "uri": part.file.uri
                    })
            elif isinstance(part, DataPart):
                # Handle structured data
                zedacp_parts.append({
                    "type": "data",
                    "data": part.data
                })

        logger.debug("Converted A2A message to ZedACP format",
                    extra={"parts_count": len(zedacp_parts)})
        return zedacp_parts

    def zedacp_to_a2a_message(self, zedacp_response: Dict[str, Any],
                             context_id: str, task_id: str) -> Message:
        """
        Convert ZedACP response to A2A Message format.

        Args:
            zedacp_response: The ZedACP response to convert
            context_id: The A2A context ID
            task_id: The A2A task ID

        Returns:
            A2A Message object
        """
        # Extract text content from ZedACP response
        text_content = ""

        if isinstance(zedacp_response, dict):
            # Try different possible response formats
            if "result" in zedacp_response:
                result = zedacp_response["result"]
                if isinstance(result, str):
                    text_content = result
                elif isinstance(result, dict) and "text" in result:
                    text_content = result["text"]
                elif isinstance(result, dict) and "content" in result:
                    content = result["content"]
                    if isinstance(content, str):
                        text_content = content
                    elif isinstance(content, list):
                        # Concatenate multiple content parts
                        for item in content:
                            if isinstance(item, dict) and "text" in item:
                                text_content += item["text"]
            elif "text" in zedacp_response:
                text_content = zedacp_response["text"]
            elif isinstance(zedacp_response, str):
                text_content = zedacp_response

        # Create A2A message parts
        parts: List[Any] = [TextPart(kind="text", text=text_content)] if text_content else []

        message = Message(
            role="agent",
            parts=parts,
            messageId=create_message_id(),
            taskId=task_id,
            contextId=context_id,
            metadata={"source": "zedacp"}
        )

        logger.debug("Converted ZedACP response to A2A message",
                    extra={"text_length": len(text_content)})
        return message

    def create_a2a_task_from_zedacp_session(self, session_id: str,
                                          context_id: Optional[str] = None) -> Task:
        """
        Create A2A Task from ZedACP session context.

        Args:
            session_id: The ZedACP session ID
            context_id: Optional existing context ID

        Returns:
            A2A Task object
        """
        if context_id is None:
            context_id = create_context_id()

        task = Task(
            id=create_task_id(),
            contextId=context_id,
            status=TaskStatus(
                state=TaskState.SUBMITTED,
                timestamp=generate_id("ts_")
            ),
            metadata={
                "zedacp_session_id": session_id,
                "source": "zedacp"
            }
        )

        # Track the mapping
        self._context_sessions[context_id] = session_id

        logger.debug("Created A2A task from ZedACP session",
                    extra={"task_id": task.id, "context_id": context_id})
        return task

    def get_zedacp_session_for_context(self, context_id: str) -> Optional[str]:
        """Get ZedACP session ID for an A2A context."""
        return self._context_sessions.get(context_id)

    def register_session_context(self, context_id: str, session_id: str):
        """Register the mapping between A2A context and ZedACP session."""
        self._context_sessions[context_id] = session_id
        logger.debug("Registered session context mapping",
                    extra={"context_id": context_id, "session_id": session_id})

    def unregister_session_context(self, context_id: str):
        """Remove the mapping between A2A context and ZedACP session."""
        if context_id in self._context_sessions:
            del self._context_sessions[context_id]
            logger.debug("Unregistered session context mapping",
                        extra={"context_id": context_id})

    def create_message_send_params(self, message: Message,
                                 configuration: Optional[Dict[str, Any]] = None) -> MessageSendParams:
        """Create MessageSendParams from a Message and optional configuration."""
        # Convert dict configuration to MessageSendConfiguration if needed
        config_obj = None
        if configuration:
            config_obj = MessageSendConfiguration(**configuration)

        return MessageSendParams(
            message=message,
            configuration=config_obj,
            metadata={"translated": True}
        )

    def extract_task_from_zedacp_prompt_result(self, zedacp_result: Dict[str, Any],
                                             task_id: str, context_id: str) -> Task:
        """Extract and update Task information from ZedACP prompt result."""
        # Update task status based on ZedACP result
        status_state = TaskState.COMPLETED
        if "error" in zedacp_result:
            status_state = TaskState.FAILED

        return Task(
            id=task_id,
            contextId=context_id,
            status=TaskStatus(
                state=status_state,
                timestamp=generate_id("ts_")
            ),
            metadata={"zedacp_result": zedacp_result}
        )


# Global translator instance
translator = A2ATranslator()


# Convenience functions
def a2a_to_zedacp_message(a2a_message: Message) -> List[Dict[str, Any]]:
    """Convert A2A Message to ZedACP prompt format."""
    return translator.a2a_to_zedacp_message(a2a_message)


def zedacp_to_a2a_message(zedacp_response: Dict[str, Any],
                         context_id: str, task_id: str) -> Message:
    """Convert ZedACP response to A2A Message format."""
    return translator.zedacp_to_a2a_message(zedacp_response, context_id, task_id)


def create_a2a_task_from_zedacp_session(session_id: str,
                                      context_id: Optional[str] = None) -> Task:
    """Create A2A Task from ZedACP session context."""
    return translator.create_a2a_task_from_zedacp_session(session_id, context_id)