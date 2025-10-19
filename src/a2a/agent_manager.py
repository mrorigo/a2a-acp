"""
A2A Agent Manager

Manages ZedACP agent connections and integrates them with the A2A protocol.
Handles agent lifecycle, session management, and message routing.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, Optional, Any, List
from pathlib import Path

from .models import (
    Message, Task, TaskStatus, TaskState, generate_id, create_task_id,
    create_context_id, create_message_id, MessageSendParams
)
from .translator import translator, a2a_to_zedacp_message, zedacp_to_a2a_message
from ..a2a_acp.agent_registry import AgentRegistry
from ..a2a_acp.zed_agent import ZedAgentConnection, PromptCancelled

logger = logging.getLogger(__name__)


class A2AAgentManager:
    """
    Manages ZedACP agent connections for A2A protocol.

    Handles the lifecycle of ZedACP agent connections, session management,
    and coordinates message passing between A2A clients and ZedACP agents.
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.agent_registry = AgentRegistry(config_path)
        self._connections: Dict[str, ZedAgentConnection] = {}
        self._connection_locks: Dict[str, asyncio.Lock] = {}
        self._active_tasks: Dict[str, Dict[str, Any]] = {}

    async def initialize_agent_connection(self, agent_name: str) -> ZedAgentConnection:
        """
        Initialize a ZedACP agent connection.

        Args:
            agent_name: Name of the agent to connect to

        Returns:
            Active ZedAgentConnection instance
        """
        if agent_name in self._connections:
            return self._connections[agent_name]

        # Get agent configuration
        try:
            agent_config = self.agent_registry.get(agent_name)
        except KeyError as e:
            raise ValueError(f"Unknown agent: {agent_name}") from e

        # Create connection lock for this agent
        if agent_name not in self._connection_locks:
            self._connection_locks[agent_name] = asyncio.Lock()

        async with self._connection_locks[agent_name]:
            # Double-check in case another coroutine created it
            if agent_name in self._connections:
                return self._connections[agent_name]

            # Create new ZedACP connection
            logger.info("Creating new ZedACP connection",
                       extra={"agent_name": agent_name, "command": agent_config.command})

            connection = ZedAgentConnection(
                command=agent_config.command,
                api_key=agent_config.api_key,
                log=logger.getChild(f"ZedAgent-{agent_name}")
            )

            try:
                # Initialize the connection
                await connection.initialize()
                logger.info("ZedACP connection initialized", extra={"agent_name": agent_name})

                # Store the connection
                self._connections[agent_name] = connection
                return connection

            except Exception as e:
                logger.error("Failed to initialize ZedACP connection",
                           extra={"agent_name": agent_name, "error": str(e)})
                raise

    async def send_message(self, message_params: MessageSendParams,
                          agent_name: str) -> Task:
        """
        Send a message to a ZedACP agent via A2A protocol.

        Args:
            message_params: A2A message parameters
            agent_name: Name of the target agent

        Returns:
            A2A Task representing the operation
        """
        # Create task and context
        task = Task(
            id=create_task_id(),
            contextId=create_context_id(),
            status=TaskStatus(state=TaskState.SUBMITTED),
            metadata={"agent_name": agent_name, "source": "a2a"}
        )

        # Track active task
        self._active_tasks[task.id] = {
            "task": task,
            "agent_name": agent_name,
            "message_params": message_params
        }

        # Update task status to working
        task.status.state = TaskState.WORKING

        try:
            # Get agent connection
            connection = await self.initialize_agent_connection(agent_name)

            # Convert A2A message to ZedACP format
            zedacp_prompt = a2a_to_zedacp_message(message_params.message)

            # Create ZedACP session
            session_id = await connection.start_session(cwd=".")

            # Register the session-context mapping
            translator.register_session_context(task.contextId, session_id)

            # Send prompt to ZedACP agent
            logger.info("Sending prompt to ZedACP agent",
                       extra={"task_id": task.id, "agent_name": agent_name,
                             "session_id": session_id, "prompt_length": len(zedacp_prompt)})

            # Execute the prompt (non-streaming for now)
            zedacp_result = await connection.prompt(
                session_id=session_id,
                prompt=zedacp_prompt
            )

            # Convert ZedACP response back to A2A format
            response_message = zedacp_to_a2a_message(
                zedacp_result,
                task.contextId,
                task.id
            )

            # Update task with completion
            task.status.state = TaskState.COMPLETED
            task.history = [message_params.message, response_message]

            logger.info("Task completed successfully",
                       extra={"task_id": task.id, "agent_name": agent_name})

        except PromptCancelled:
            logger.warning("Task was cancelled", extra={"task_id": task.id})
            task.status.state = TaskState.CANCELLED
        except Exception as e:
            logger.error("Task failed", extra={"task_id": task.id, "error": str(e)})
            task.status.state = TaskState.FAILED
            task.status.message = Message(
                role="agent",
                parts=[],
                messageId=create_message_id(),
                metadata={"error": str(e)}
            )
        finally:
            # Clean up active task tracking
            if task.id in self._active_tasks:
                del self._active_tasks[task.id]

        return task

    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel an active task.

        Args:
            task_id: ID of the task to cancel

        Returns:
            True if cancellation was successful
        """
        if task_id not in self._active_tasks:
            logger.warning("Task not found for cancellation", extra={"task_id": task_id})
            return False

        task_info = self._active_tasks[task_id]
        agent_name = task_info["agent_name"]

        try:
            connection = await self.initialize_agent_connection(agent_name)

            # Get the ZedACP session ID for this context
            context_id = task_info["task"].contextId
            session_id = translator.get_zedacp_session_for_context(context_id)

            if session_id:
                await connection.cancel(session_id)
                logger.info("Task cancelled successfully",
                           extra={"task_id": task_id, "session_id": session_id})
                return True
            else:
                logger.warning("No ZedACP session found for context",
                             extra={"task_id": task_id, "context_id": context_id})
                return False

        except Exception as e:
            logger.error("Failed to cancel task",
                        extra={"task_id": task_id, "error": str(e)})
            return False

    async def get_task_status(self, task_id: str) -> Optional[Task]:
        """Get the current status of a task."""
        if task_id in self._active_tasks:
            return self._active_tasks[task_id]["task"]
        return None

    async def list_active_tasks(self) -> List[Task]:
        """List all currently active tasks."""
        return [info["task"] for info in self._active_tasks.values()]

    async def close_all_connections(self):
        """Close all ZedACP agent connections."""
        for agent_name, connection in self._connections.items():
            try:
                await connection.close()
                logger.info("Closed ZedACP connection", extra={"agent_name": agent_name})
            except Exception as e:
                logger.error("Error closing ZedACP connection",
                           extra={"agent_name": agent_name, "error": str(e)})

        self._connections.clear()
        self._active_tasks.clear()

    def get_available_agents(self) -> List[str]:
        """Get list of available agent names."""
        return [agent.name for agent in self.agent_registry.list()]


# Global agent manager instance
agent_manager = A2AAgentManager()


# Convenience functions
async def send_message_to_agent(message_params: MessageSendParams,
                               agent_name: str) -> Task:
    """Send a message to an agent using the global agent manager."""
    return await agent_manager.send_message(message_params, agent_name)


async def cancel_task(task_id: str) -> bool:
    """Cancel a task using the global agent manager."""
    return await agent_manager.cancel_task(task_id)


def get_available_agents() -> List[str]:
    """Get available agent names using the global agent manager."""
    return agent_manager.get_available_agents()