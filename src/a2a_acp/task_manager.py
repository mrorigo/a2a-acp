"""
A2A Task Manager

A2A-native task lifecycle management that bridges A2A tasks to ZedACP runs.
Replaces RunManager with proper A2A terminology.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass
from datetime import datetime

from a2a.models import (
    Task,
    TaskStatus,
    TaskState,
    Message,
    generate_id,
    create_task_id,
    InputRequiredNotification,
    current_timestamp,
)

from .zed_agent import ZedAgentConnection, AgentProcessError, PromptCancelled
from .push_notification_manager import PushNotificationManager
from .models import EventType

logger = logging.getLogger(__name__)


@dataclass
class TaskExecutionContext:
    """A2A task execution context with ZedACP mapping."""
    task: Task
    agent_name: str
    zedacp_session_id: Optional[str] = None
    working_directory: Optional[str] = None
    cancel_event: Optional[asyncio.Event] = None
    created_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.cancel_event is None:
            self.cancel_event = asyncio.Event()


class A2ATaskManager:
    """
    A2A-native task manager that bridges A2A tasks to ZedACP runs.

    Provides A2A-compliant task lifecycle while maintaining ZedACP compatibility.
    """

    def __init__(self, push_notification_manager: Optional[PushNotificationManager] = None):
        self._active_tasks: Dict[str, TaskExecutionContext] = {}
        self._lock: Optional[asyncio.Lock] = None
        self.push_notification_manager = push_notification_manager

    @property
    def lock(self) -> asyncio.Lock:
        """Get or create the asyncio lock for thread-safe operations."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def _send_task_notification(self, task_id: str, event: str, event_data: Dict[str, Any]) -> None:
        """Send a push notification for a task event."""
        if self.push_notification_manager:
            try:
                await self.push_notification_manager.send_notification(task_id, {
                    "event": event,
                    "task_id": task_id,
                    **event_data
                })
            except Exception as e:
                logger.error(
                    "Failed to send task notification",
                    extra={"task_id": task_id, "event": event, "error": str(e)}
                )

    async def _handle_task_error(
        self,
        task_id: str,
        old_state: str,
        error: Exception,
        context: str = ""
    ) -> None:
        """Handle task errors with consistent status update, notification, and cleanup."""
        try:
            # Update task status to failed
            if task_id in self._active_tasks:
                task = self._active_tasks[task_id].task
                task.status.state = TaskState.FAILED
                task.status.timestamp = current_timestamp()

            # Send notification for failure
            error_message = f"{context}: {str(error)}" if context else str(error)
            asyncio.create_task(self._send_task_notification(task_id, EventType.TASK_STATUS_CHANGE.value, {
                "old_state": old_state,
                "new_state": TaskState.FAILED.value,
                "message": error_message,
                "error": str(error)
            }))

            # Clean up notification configs for failed tasks (immediate deletion)
            if self.push_notification_manager:
                asyncio.create_task(self.push_notification_manager.cleanup_by_task_state(
                    task_id, TaskState.FAILED.value
                ))

            logger.error(f"Task failed: {error_message}", extra={"task_id": task_id, "error": str(error)})
        except Exception as e:
            logger.error(
                "Error in task error handler",
                extra={"task_id": task_id, "original_error": str(error), "handler_error": str(e)}
            )

    async def _handle_task_cancellation(self, task_id: str, old_state: str, context: str = "") -> None:
        """Handle task cancellation with consistent status update and notification."""
        try:
            # Update task status to cancelled
            if task_id in self._active_tasks:
                task = self._active_tasks[task_id].task
                task.status.state = TaskState.CANCELLED
                task.status.timestamp = current_timestamp()

            # Send notification for cancellation
            cancel_message = f"Task was cancelled{': ' + context if context else ''}"
            asyncio.create_task(self._send_task_notification(task_id, EventType.TASK_STATUS_CHANGE.value, {
                "old_state": old_state,
                "new_state": TaskState.CANCELLED.value,
                "message": cancel_message
            }))

            logger.info(f"Task cancelled: {cancel_message}", extra={"task_id": task_id})
        except Exception as e:
            logger.error(
                "Error in task cancellation handler",
                extra={"task_id": task_id, "context": context, "error": str(e)}
            )

    def _extract_message_content(self, message) -> str:
        """Extract content from a message for notification purposes."""
        if message.parts:
            first_part = message.parts[0]
            if first_part.kind == "text":
                return first_part.text
            elif first_part.kind == "data":
                return str(first_part.data)
            elif first_part.kind == "file":
                return f"File: {first_part.file.name or 'unnamed'}"
            else:
                return str(first_part)
        return ""

    def _response_has_agent_content(self, response: dict) -> bool:
        """Check whether the agent response includes any textual content."""

        def _has_text(value: Any) -> bool:
            return isinstance(value, str) and value.strip() != ""

        if not isinstance(response, dict):
            return False

        # Direct text field (some agents return text at top level)
        if _has_text(response.get("text")):
            return True

        result = response.get("result")
        if isinstance(result, str):
            return _has_text(result)
        if isinstance(result, dict):
            if _has_text(result.get("text")):
                return True

            content = result.get("content")
            if isinstance(content, str):
                return _has_text(content)
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and _has_text(item.get("text")):
                        return True

        return False

    def _is_input_required_from_response(self, response: dict) -> tuple[bool, str]:
        """Protocol-compliant detection of input-required state using Zed ACP response."""
        stop_reason = response.get("stopReason")
        tool_calls = response.get("toolCalls", [])
        has_agent_content = self._response_has_agent_content(response)

        if stop_reason == "input_required":
            return True, "Agent explicitly requested additional input"

        if stop_reason == "end_turn" and not tool_calls and not has_agent_content:
            return True, "Agent completed turn without actions"

        if has_agent_content:
            return False, "Agent provided response content"

        return False, f"Turn ended with reason: {stop_reason}"

    def _extract_input_types_from_response(self, response: dict) -> list[str]:
        """Extract input types from response metadata if available."""
        # Check for _meta hints in final response for enhanced detection
        meta = response.get("_meta", {})
        if isinstance(meta, dict) and "input_types" in meta:
            return meta["input_types"]

        # Fallback to default types
        return ["text/plain"]

    async def _send_message_notification(self, task_id: str, message, message_type: str) -> None:
        """Send a notification for a task message with extracted content."""
        content = self._extract_message_content(message)

        asyncio.create_task(self._send_task_notification(task_id, EventType.TASK_MESSAGE.value, {
            "message_role": message.role,
            "message_content": content,
            "message_type": message_type
        }))

    async def create_task(
        self,
        context_id: str,
        agent_name: str,
        initial_message: Optional[Message] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Task:
        """Create a new A2A task."""
        async with self.lock:
            task_id = create_task_id()

            # Create initial task status
            status = TaskStatus(
                state=TaskState.SUBMITTED,
                timestamp=current_timestamp()
            )

            # Create A2A task
            history = None
            if initial_message:
                history = [Message(**initial_message.model_dump(exclude_none=True))]

            task = Task(
                id=task_id,
                contextId=context_id,
                status=status,
                history=history,
                artifacts=None,
                metadata=metadata or {},
                kind="task"
            )

            # Create execution context
            context = TaskExecutionContext(
                task=task,
                agent_name=agent_name
            )

            self._active_tasks[task_id] = context
    
            logger.info("Created A2A task",
                       extra={"task_id": task_id, "context_id": context_id, "agent": agent_name})
    
            # Send notification for task creation
            asyncio.create_task(self._send_task_notification(task_id, "task_created", {
                "old_state": None,
                "new_state": TaskState.SUBMITTED.value,
                "context_id": context_id,
                "agent_name": agent_name,
                "creation_event": True
            }))
    
            return task

    async def execute_task(
        self,
        task_id: str,
        agent_command: List[str],
        api_key: Optional[str] = None,
        working_directory: str = ".",
        mcp_servers: Optional[List[Dict]] = None,
        stream_handler: Optional[Callable[[str], Awaitable[None]]] = None
    ) -> Task:
        """Execute an A2A task using ZedACP agent with input-required support."""
        async with self.lock:
            if task_id not in self._active_tasks:
                raise ValueError(f"Unknown task: {task_id}")

            context = self._active_tasks[task_id]
            task = context.task

            try:
                # Update status to working
                old_state = task.status.state.value
                task.status.state = TaskState.WORKING
                task.status.timestamp = current_timestamp()

                # Send notification for status change
                asyncio.create_task(self._send_task_notification(task_id, EventType.TASK_STATUS_CHANGE.value, {
                    "old_state": old_state,
                    "new_state": TaskState.WORKING.value,
                    "message": "Task execution started"
                }))

                # Execute via ZedACP agent
                async with ZedAgentConnection(agent_command, api_key=api_key) as connection:
                    await connection.initialize()

                    # Create ZedACP session for this task
                    zed_session_id = await connection.start_session(
                        cwd=working_directory,
                        mcp_servers=mcp_servers or []
                    )

                    # Update context with ZedACP mapping
                    context.zedacp_session_id = zed_session_id
                    context.working_directory = working_directory

                    # Convert A2A message to ZedACP format if we have history
                    if task.history and len(task.history) > 0:
                        user_message = task.history[0]  # First message should be user input
                        from ..a2a.translator import A2ATranslator
                        translator = A2ATranslator()
                        zedacp_parts = translator.a2a_to_zedacp_message(user_message)

                        # Execute the task with protocol-compliant input-required detection
                        cancel_event = context.cancel_event

                        # Chunk handler that accumulates content and handles artifacts
                        accumulated_content = []

                        async def on_chunk(text: str) -> None:
                            # Accumulate all chunks for the final response
                            accumulated_content.append(text)

                            # Forward to streaming handler if provided
                            if stream_handler:
                                await stream_handler(text)

                            # Also check for artifact creation notifications
                            upper_text = text.upper()
                            if ("ARTIFACT:" in upper_text or
                                "FILE:" in upper_text or
                                "CREATED:" in upper_text):

                                # Send notification for potential artifact creation
                                asyncio.create_task(self._send_task_notification(task_id, EventType.TASK_ARTIFACT.value, {
                                    "artifact_type": "detected",
                                    "artifact_name": "processing",
                                    "detection_text": text[:100]
                                }))

                        result = await connection.prompt(
                            zed_session_id,
                            zedacp_parts,
                            on_chunk=on_chunk,
                            cancel_event=cancel_event
                        )

                        # Add accumulated content to the result
                        if accumulated_content:
                            full_text = "".join(accumulated_content)
                            if "result" not in result:
                                result["result"] = {}
                            result["result"]["text"] = full_text
                            logger.debug("Added accumulated content to result",
                                       extra={"content_length": len(full_text), "chunks": len(accumulated_content)})

                        # Protocol-compliant input-required detection using stopReason and toolCalls
                        is_input_required, reason = self._is_input_required_from_response(result)
                        logger.info("Protocol analysis of ZedACP response",
                                   extra={
                                       "task_id": task_id,
                                       "stop_reason": result.get("stopReason"),
                                       "tool_calls_count": len(result.get("toolCalls", [])),
                                       "is_input_required": is_input_required,
                                       "reason": reason
                                   })

                        if is_input_required:
                            # Update task to input-required state using protocol semantics
                            old_state = task.status.state.value
                            task.status.state = TaskState.INPUT_REQUIRED
                            task.status.timestamp = current_timestamp()

                            # Extract input types from response metadata if available
                            input_types = self._extract_input_types_from_response(result)

                            # Send notification for input required
                            asyncio.create_task(self._send_task_notification(task_id, EventType.TASK_INPUT_REQUIRED.value, {
                                "old_state": old_state,
                                "new_state": TaskState.INPUT_REQUIRED.value,
                                "message": "Additional input required",
                                "input_types": input_types,
                                "detection_method": "protocol_compliant"
                            }))

                            # Create input-required notification
                            input_notification = InputRequiredNotification(
                                taskId=task_id,
                                contextId=task.contextId,
                                message="Additional input required",
                                inputTypes=input_types,
                                timeout=300  # 5 minutes default timeout
                            )

                            logger.info("Task requires input (protocol-compliant detection)",
                                        extra={
                                            "task_id": task_id,
                                            "notification": input_notification.model_dump(),
                                            "input_types": input_types,
                                            "detection_reason": reason
                                        })
                            return task

                        # Convert ZedACP response back to A2A message
                        response_message = translator.zedacp_to_a2a_message(result, task.contextId, task_id)

                        # Add response to task history
                        task.history.append(response_message)

                        # Send notification for new message
                        asyncio.create_task(self._send_message_notification(task_id, response_message, "agent_response"))

                        # Mark as completed
                        old_state = task.status.state.value
                        task.status.state = TaskState.COMPLETED
                        task.status.timestamp = current_timestamp()

                        # Send notification for completion
                        asyncio.create_task(self._send_task_notification(task_id, EventType.TASK_STATUS_CHANGE.value, {
                            "old_state": old_state,
                            "new_state": TaskState.COMPLETED.value,
                            "message": "Task completed successfully"
                        }))

                        # Clean up notification configs based on task state
                        if self.push_notification_manager:
                            asyncio.create_task(self.push_notification_manager.cleanup_by_task_state(
                                task_id, TaskState.COMPLETED.value
                            ))

                        logger.info("Task completed successfully",
                                   extra={"task_id": task_id, "agent": context.agent_name})

                        return task
                    else:
                        # No message to execute
                        task.status.state = TaskState.COMPLETED
                        return task

            except PromptCancelled:
                await self._handle_task_cancellation(task_id, task.status.state.value)
                return task

            except AgentProcessError as e:
                await self._handle_task_error(task_id, task.status.state.value, e)
                raise

            except Exception as e:
                await self._handle_task_error(task_id, task.status.state.value, e, "Unexpected error")
                raise

    async def get_task(self, task_id: str) -> Optional[Task]:
        """Retrieve a task by ID."""
        if task_id in self._active_tasks:
            return self._active_tasks[task_id].task
        return None

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""
        if task_id not in self._active_tasks:
            return False

        context = self._active_tasks[task_id]
        if context.cancel_event:
            context.cancel_event.set()

        # Update task status
        task = context.task
        old_state = task.status.state.value
        task.status.state = TaskState.CANCELLED
        task.status.timestamp = current_timestamp()

        # Send notification for cancellation
        asyncio.create_task(self._send_task_notification(task_id, EventType.TASK_STATUS_CHANGE.value, {
            "old_state": old_state,
            "new_state": TaskState.CANCELLED.value,
            "message": "Task was cancelled by user"
        }))

        logger.info("Cancelled A2A task", extra={"task_id": task_id})
        return True

    async def list_tasks(self, context_id: Optional[str] = None) -> List[Task]:
        """List tasks with optional context filtering."""
        tasks = [ctx.task for ctx in self._active_tasks.values()]

        if context_id:
            tasks = [t for t in tasks if t.contextId == context_id]

        return tasks

    async def provide_input_and_continue(
        self,
        task_id: str,
        user_input: Message,
        agent_command: List[str],
        api_key: Optional[str] = None,
        working_directory: Optional[str] = None,
        mcp_servers: Optional[List[Dict]] = None
    ) -> Task:
        """Provide input for an input-required task and continue execution."""
        async with self.lock:
            if task_id not in self._active_tasks:
                raise ValueError(f"Unknown task: {task_id}")

            context = self._active_tasks[task_id]
            task = context.task

            # Verify task is in input-required state
            if task.status.state != TaskState.INPUT_REQUIRED:
                raise ValueError(f"Task {task_id} is not in input-required state")

            try:
                # Update status back to working
                old_state = task.status.state.value
                task.status.state = TaskState.WORKING
                task.status.timestamp = current_timestamp()

                # Send notification for status change back to working
                asyncio.create_task(self._send_task_notification(task_id, EventType.TASK_STATUS_CHANGE.value, {
                    "old_state": old_state,
                    "new_state": TaskState.WORKING.value,
                    "message": "Task resumed after input provided"
                }))

                # Ensure task history exists
                if task.history is None:
                    task.history = []

                # Add user input to task history
                task.history.append(user_input)

                # Send notification for user message
                asyncio.create_task(self._send_message_notification(task_id, user_input, "user_input"))

                # Use existing ZedACP session if available, otherwise create new connection
                if context.zedacp_session_id and working_directory == context.working_directory:
                    # Continue with existing session
                    async with ZedAgentConnection(agent_command, api_key=api_key) as connection:
                        await connection.initialize()

                        # Load existing session
                        await connection.load_session(context.zedacp_session_id, context.working_directory or ".")

                        from ..a2a.translator import A2ATranslator
                        translator = A2ATranslator()
                        zedacp_parts = translator.a2a_to_zedacp_message(user_input)

                        cancel_event = context.cancel_event

                        # Chunk handler for continuation that accumulates content
                        continuation_content = []

                        async def on_chunk(text: str) -> None:
                            continuation_content.append(text)
                            logger.debug("Continue task output chunk",
                                       extra={"task_id": task_id, "chunk": text[:100]})

                        result = await connection.prompt(
                            context.zedacp_session_id,
                            zedacp_parts,
                            on_chunk=on_chunk,
                            cancel_event=cancel_event
                        )

                        # Add accumulated content to the result
                        if continuation_content:
                            full_text = "".join(continuation_content)
                            if "result" not in result:
                                result["result"] = {}
                            result["result"]["text"] = full_text
                            logger.debug("Added continuation content to result",
                                       extra={"content_length": len(full_text), "chunks": len(continuation_content)})

                        # Convert response back to A2A message
                        response_message = translator.zedacp_to_a2a_message(result, task.contextId, task_id)

                        # Ensure history still exists before appending
                        if task.history is None:
                            task.history = []
                        task.history.append(response_message)

                        # Check for artifacts in the response and send notifications
                        if response_message.parts:
                            for part in response_message.parts:
                                if part.kind == "file":
                                    asyncio.create_task(self._send_task_notification(task_id, EventType.TASK_ARTIFACT.value, {
                                        "artifact_type": "file",
                                        "artifact_name": part.file.name or "unnamed",
                                        "artifact_size": len(part.file.bytes) if hasattr(part.file, 'bytes') and part.file.bytes else 0
                                    }))

                        # Protocol-compliant check if more input is required
                        is_input_required, reason = self._is_input_required_from_response(result)
                        if is_input_required:
                            old_state = task.status.state.value
                            task.status.state = TaskState.INPUT_REQUIRED
                            task.status.timestamp = current_timestamp()

                            # Extract input types from response metadata if available
                            input_types = self._extract_input_types_from_response(result)

                            # Send notification for additional input required
                            asyncio.create_task(self._send_task_notification(task_id, EventType.TASK_INPUT_REQUIRED.value, {
                                "old_state": old_state,
                                "new_state": TaskState.INPUT_REQUIRED.value,
                                "message": "Additional input required after continuation",
                                "input_types": input_types,
                                "detection_method": "protocol_compliant"
                            }))

                            logger.info("Additional input required after continuation",
                                       extra={
                                           "task_id": task_id,
                                           "detection_reason": reason,
                                           "input_types": input_types
                                       })
                            return task

                        # Mark as completed
                        old_state = task.status.state.value
                        task.status.state = TaskState.COMPLETED
                        task.status.timestamp = current_timestamp()

                        # Send notification for completion
                        asyncio.create_task(self._send_task_notification(task_id, EventType.TASK_STATUS_CHANGE.value, {
                            "old_state": old_state,
                            "new_state": TaskState.COMPLETED.value,
                            "message": "Task completed after input continuation"
                        }))

                        logger.info("Task completed after input",
                                   extra={"task_id": task_id, "agent": context.agent_name})
                        return task
                else:
                    # Need to restart with new connection
                    return await self.execute_task(
                        task_id, agent_command, api_key, working_directory or ".",
                        mcp_servers
                    )

            except PromptCancelled:
                await self._handle_task_cancellation(task_id, task.status.state.value, "during input continuation")
                return task

            except AgentProcessError as e:
                await self._handle_task_error(task_id, task.status.state.value, e, "during continuation")
                raise

            except Exception as e:
                await self._handle_task_error(task_id, task.status.state.value, e, "during continuation")
                raise

    async def cleanup_completed_tasks(self) -> int:
        """Clean up old completed tasks."""
        to_remove = []
        for task_id, context in self._active_tasks.items():
            if context.task.status.state in [TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED]:
                # Keep for a while for history, then remove
                if context.created_at:
                    age = (datetime.utcnow() - context.created_at).total_seconds()
                    if age > 3600:  # Remove after 1 hour
                        to_remove.append(task_id)

        for task_id in to_remove:
            del self._active_tasks[task_id]

        return len(to_remove)

    async def get_input_required_tasks(self) -> List[Task]:
        """Get all tasks currently in input-required state."""
        input_required_tasks = []
        for context in self._active_tasks.values():
            if context.task.status.state == TaskState.INPUT_REQUIRED:
                input_required_tasks.append(context.task)
        return input_required_tasks


# Global A2A-native task manager
a2a_task_manager = A2ATaskManager()
