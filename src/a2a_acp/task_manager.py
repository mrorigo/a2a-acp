"""
A2A Task Manager

A2A-native task lifecycle management that bridges A2A tasks to ZedACP runs.
Replaces RunManager with proper A2A terminology.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from ..a2a.models import Task, TaskStatus, TaskState, Message, generate_id, create_task_id
from .models import RunMode
from .zed_agent import ZedAgentConnection, AgentProcessError, PromptCancelled

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

    def __init__(self):
        self._active_tasks: Dict[str, TaskExecutionContext] = {}
        self._lock = asyncio.Lock()

    async def create_task(
        self,
        context_id: str,
        agent_name: str,
        initial_message: Optional[Message] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Task:
        """Create a new A2A task."""
        async with self._lock:
            task_id = create_task_id()

            # Create initial task status
            status = TaskStatus(
                state=TaskState.SUBMITTED,
                timestamp=generate_id("ts_")
            )

            # Create A2A task
            task = Task(
                id=task_id,
                contextId=context_id,
                status=status,
                history=[initial_message] if initial_message else [],
                artifacts=[],
                metadata=metadata or {}
            )

            # Create execution context
            context = TaskExecutionContext(
                task=task,
                agent_name=agent_name
            )

            self._active_tasks[task_id] = context

            logger.info("Created A2A task",
                       extra={"task_id": task_id, "context_id": context_id, "agent": agent_name})

            return task

    async def execute_task(
        self,
        task_id: str,
        agent_command: List[str],
        api_key: Optional[str] = None,
        working_directory: str = ".",
        mcp_servers: Optional[List[Dict]] = None
    ) -> Task:
        """Execute an A2A task using ZedACP agent."""
        async with self._lock:
            if task_id not in self._active_tasks:
                raise ValueError(f"Unknown task: {task_id}")

            context = self._active_tasks[task_id]
            task = context.task

            try:
                # Update status to working
                task.status.state = TaskState.WORKING
                task.status.timestamp = generate_id("ts_")

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

                        # Execute the task
                        cancel_event = context.cancel_event

                        async def on_chunk(text: str) -> None:
                            # Could emit A2A message updates here
                            logger.debug("Task output chunk", extra={"task_id": task_id, "chunk": text[:100]})

                        result = await connection.prompt(
                            zed_session_id,
                            zedacp_parts,
                            on_chunk=on_chunk,
                            cancel_event=cancel_event
                        )

                        # Convert ZedACP response back to A2A message
                        response_message = translator.zedacp_to_a2a_message(result, task.contextId, task_id)

                        # Add response to task history
                        task.history.append(response_message)

                        # Mark as completed
                        task.status.state = TaskState.COMPLETED
                        task.status.timestamp = generate_id("ts_")

                        logger.info("Task completed successfully",
                                   extra={"task_id": task_id, "agent": context.agent_name})

                        return task
                    else:
                        # No message to execute
                        task.status.state = TaskState.COMPLETED
                        return task

            except PromptCancelled:
                task.status.state = TaskState.CANCELLED
                logger.info("Task cancelled", extra={"task_id": task_id})
                return task

            except AgentProcessError as e:
                task.status.state = TaskState.FAILED
                logger.error("Task failed", extra={"task_id": task_id, "error": str(e)})
                raise

            except Exception as e:
                task.status.state = TaskState.FAILED
                logger.exception("Unexpected error executing task", extra={"task_id": task_id})
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
        task.status.state = TaskState.CANCELLED
        task.status.timestamp = generate_id("ts_")

        logger.info("Cancelled A2A task", extra={"task_id": task_id})
        return True

    async def list_tasks(self, context_id: Optional[str] = None) -> List[Task]:
        """List tasks with optional context filtering."""
        tasks = [ctx.task for ctx in self._active_tasks.values()]

        if context_id:
            tasks = [t for t in tasks if t.contextId == context_id]

        return tasks

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


# Global A2A-native task manager
a2a_task_manager = A2ATaskManager()