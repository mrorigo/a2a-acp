"""
A2A-ACP Main Application

Native A2A protocol server using JSON-RPC 2.0 over HTTP.
Bridges A2A clients to ZedACP agents, replacing the legacy ACP IBM ACP implementation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager, suppress
from datetime import datetime
from typing import Any, AsyncGenerator, AsyncIterator, Optional, List, Dict, Callable, Tuple
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status, WebSocket
from fastapi.responses import Response, StreamingResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .database import SessionDatabase
from .logging_config import configure_logging
from .task_manager import A2ATaskManager
from .context_manager import A2AContextManager
from .settings import get_settings
from .zed_agent import AgentProcessError, PromptCancelled, ZedAgentConnection
from .push_notification_manager import PushNotificationManager
from .streaming_manager import StreamingManager
from .tool_config import get_tool_configuration_manager, BashTool
from .bash_executor import BashToolExecutor

# Import A2A protocol components
from a2a.models import (
    Message,
    MessageSendParams,
    Task,
    TextPart,
    TaskStatusUpdateEvent,
    TaskStatus,
    TaskState,
    create_message_id,
    current_timestamp,
)
from a2a.translator import A2ATranslator

# Import push notification models
from .models import TaskPushNotificationConfig

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def serialize_a2a(obj: Any) -> Any:
    """Convert Pydantic models (and nested structures) into JSON-serializable dicts without nulls."""
    if isinstance(obj, BaseModel):
        return obj.model_dump(exclude_none=True)
    if isinstance(obj, list):
        return [serialize_a2a(item) for item in obj]
    if isinstance(obj, dict):
        return {key: serialize_a2a(value) for key, value in obj.items() if value is not None}
    return obj


async def iter_streaming_payloads(
    task_manager: "A2ATaskManager",
    agent_config: Dict[str, Any],
    task: Task,
    *,
    include_jsonrpc: bool,
    request_id: Any,
    stream_id: Optional[str] = None,
) -> AsyncIterator[Dict[str, Any]]:
    """
    Shared streaming iterator that yields formatted payloads for SSE delivery.

    Args:
        task_manager: Task manager instance handling execution.
        agent_config: Agent configuration dict.
        task: Task object to execute.
        include_jsonrpc: Whether to wrap responses in JSON-RPC envelopes.
        request_id: JSON-RPC request ID (can be None).
        stream_id: Optional identifier for correlating log output across layers.
    """
    stream_id = stream_id or f"{task.id}:{uuid4().hex[:8]}"
    logger.debug(
        "Streaming iterator started",
        extra={
            "stream_id": stream_id,
            "task_id": task.id,
            "context_id": task.contextId,
            "include_jsonrpc": include_jsonrpc,
            "request_id": request_id,
        },
    )
    queue: asyncio.Queue[Tuple[str, Any]] = asyncio.Queue()
    accumulated_chunks: List[str] = []

    def wrap_result(payload: Any) -> Dict[str, Any]:
        body = serialize_a2a(payload)
        if include_jsonrpc:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": body
            }
        return body

    def wrap_error(payload: Dict[str, Any]) -> Dict[str, Any]:
        if include_jsonrpc:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": payload
            }
        return {"error": payload}

    async def stream_chunk_handler(chunk: str) -> None:
        chunk_index = len(accumulated_chunks)
        accumulated_chunks.append(chunk)

        chunk_message = Message(
            role="agent",
            parts=[TextPart(text=chunk)],
            messageId=create_message_id(),
            taskId=task.id,
            contextId=task.contextId,
            metadata={"streaming": True, "chunk_index": chunk_index}
        )

        chunk_message_payload = chunk_message.model_dump(exclude_none=True)

        status_event = TaskStatusUpdateEvent(
            taskId=task.id,
            contextId=task.contextId,
            status=TaskStatus(
                state=TaskState.WORKING,
                timestamp=current_timestamp(),
                message=chunk_message_payload
            ),
            final=False,
            metadata={"chunk_index": chunk_index}
        )

        await queue.put(("result", status_event))
        logger.debug(
            "Queued streaming chunk event",
            extra={
                "stream_id": stream_id,
                "task_id": task.id,
                "chunk_index": chunk_index,
                "chunk_preview": chunk[:80],
            },
        )

    async def run_task():
        try:
            result_task = await task_manager.execute_task(
                task_id=task.id,
                agent_command=agent_config["command"],
                api_key=agent_config["api_key"],
                working_directory=os.getcwd(),
                mcp_servers=[],
                stream_handler=stream_chunk_handler
            )

            final_message = result_task.history[-1] if result_task.history else None
            final_message_payload = (
                final_message.model_dump(exclude_none=True) if final_message else None
            )

            final_status_event = TaskStatusUpdateEvent(
                taskId=result_task.id,
                contextId=result_task.contextId,
                status=TaskStatus(
                    state=result_task.status.state,
                    timestamp=result_task.status.timestamp or current_timestamp(),
                    message=final_message_payload
                ),
                final=True,
                metadata={
                    "total_chunks": len(accumulated_chunks),
                    "stream_completed": True
                }
            )

            await queue.put(("result", final_status_event))
            logger.debug(
                "Queued final status event",
                extra={
                    "stream_id": stream_id,
                    "task_id": result_task.id,
                    "total_chunks": len(accumulated_chunks),
                    "final_state": result_task.status.state.value if result_task.status else None,
                },
            )
            await queue.put(("result", result_task))
            logger.debug(
                "Queued final task payload",
                extra={
                    "stream_id": stream_id,
                    "task_id": result_task.id,
                    "history_count": len(result_task.history or []),
                },
            )

        except AgentProcessError as exc:
            logger.exception("Agent process failed during streaming", extra={"task_id": task.id})
            await queue.put(("error", {
                "code": -32603,
                "message": f"Agent process failed: {str(exc)}"
            }))
            logger.debug(
                "Queued streaming error payload",
                extra={
                    "stream_id": stream_id,
                    "task_id": task.id,
                    "error_type": "AgentProcessError",
                    "error_message": str(exc),
                },
            )
            await task_manager.cancel_task(task.id)

        except Exception as exc:
            logger.exception("Unexpected error during streaming execution", extra={"task_id": task.id})
            await queue.put(("error", {
                "code": -32603,
                "message": f"Failed to stream message: {str(exc)}"
            }))
            logger.debug(
                "Queued streaming error payload",
                extra={
                    "stream_id": stream_id,
                    "task_id": task.id,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
            )

        finally:
            await queue.put(("done", None))
            logger.debug(
                "Queued streaming completion sentinel",
                extra={"stream_id": stream_id, "task_id": task.id},
            )

    runner = asyncio.create_task(run_task())

    try:
        initial_payload = wrap_result(task)
        logger.debug(
            "Emitting initial streaming payload",
            extra={
                "stream_id": stream_id,
                "task_id": task.id,
                "payload_type": type(task).__name__,
                "include_jsonrpc": include_jsonrpc,
                "result_kind": (
                    initial_payload.get("result", {}).get("kind")
                    if isinstance(initial_payload.get("result"), dict)
                    else None
                ),
            },
        )
        yield initial_payload

        while True:
            kind, payload = await queue.get()
            if kind == "done":
                logger.debug(
                    "Streaming iterator received completion sentinel",
                    extra={"stream_id": stream_id, "task_id": task.id},
                )
                break
            if kind == "result":
                formatted = wrap_result(payload)
                result_body = formatted.get("result", formatted if not include_jsonrpc else {})
                result_kind = None
                if isinstance(result_body, dict):
                    result_kind = result_body.get("kind")
                elif hasattr(payload, "kind"):
                    result_kind = getattr(payload, "kind")

                logger.debug(
                    "Emitting streaming payload",
                    extra={
                        "stream_id": stream_id,
                        "task_id": task.id,
                        "payload_kind": result_kind,
                        "include_jsonrpc": include_jsonrpc,
                    },
                )
                yield formatted
            elif kind == "error":
                formatted_error = wrap_error(payload)
                logger.debug(
                    "Emitting streaming error payload",
                    extra={
                        "stream_id": stream_id,
                        "task_id": task.id,
                        "error_code": payload.get("code"),
                        "include_jsonrpc": include_jsonrpc,
                    },
                )
                yield formatted_error
    finally:
        await runner
        logger.debug(
            "Streaming iterator completed",
            extra={"stream_id": stream_id, "task_id": task.id},
        )


def get_agent_config() -> Dict[str, Any]:
    """Get the single agent configuration from settings."""
    import shlex
    settings = get_settings()

    # For testing/development, provide fallback defaults
    command_str = settings.agent_command or "python tests/dummy_agent.py"
    description = settings.agent_description or "A2A-ACP Development Agent"

    # Parse command string into argument list for subprocess
    try:
        command = shlex.split(command_str)
    except ValueError as e:
        # Fallback if parsing fails - convert string to list with shell execution
        logger.warning(f"Failed to parse command string '{command_str}': {e}. Using as single command.")
        command = [command_str]

    return {
        "command": command,
        "api_key": settings.agent_api_key,
        "description": description
    }


async def generate_static_agent_card():
    """Generate a static AgentCard for the single configured agent."""
    from a2a.models import (
        AgentCard, AgentCapabilities, AgentSkill, AgentProvider,
        SecurityScheme, HTTPAuthSecurityScheme
    )

    agent_config = get_agent_config()

    # Load available tools and convert to skills
    tool_manager = get_tool_configuration_manager()
    available_tools = await tool_manager.load_tools()

    # Convert BashTool objects to AgentSkill objects
    tool_skills = []
    for tool in available_tools.values():
        # Convert tool parameters to examples format
        examples = tool.examples[:3]  # Limit to 3 examples for the skill
        if not examples and tool.parameters:
            # Generate examples from parameters if none provided
            param_examples = []
            for param in tool.parameters[:2]:  # Use first 2 parameters for examples
                if param.required:
                    param_examples.append(f"{param.name}: <{param.type}>")
            if param_examples:
                examples = [f"Execute {tool.name} with {', '.join(param_examples)}"]

        skill = AgentSkill(
            id=tool.id,
            name=tool.name,
            description=tool.description,
            tags=["bash", "tool"] + tool.tags,
            examples=examples,
            inputModes=["text/plain"],
            outputModes=["text/plain"]
        )
        tool_skills.append(skill)

    # Combine with existing core skills
    all_skills = [
        AgentSkill(
            id="code_generation",
            name="Code Generation",
            description="Generate and modify code in various programming languages",
            tags=["coding", "development", "programming"],
            examples=["Create a Python function to calculate fibonacci numbers"],
            inputModes=["text/plain"],
            outputModes=["text/plain"]
        ),
        AgentSkill(
            id="file_system",
            name="File System Operations",
            description="Read, write, and modify files in the workspace",
            tags=["files", "workspace", "io"],
            examples=["Read the contents of config.json", "Create a new Python script"],
            inputModes=["text/plain"],
            outputModes=["text/plain"]
        )
    ] + tool_skills

    return AgentCard(
        protocolVersion="0.3.0",
        name="a2a-acp-agent",
        description=f"{agent_config['description']} (with bash tool execution)",
        url="http://localhost:8001/a2a/rpc",
        preferredTransport="JSONRPC",
        version="1.0.0",
        capabilities=AgentCapabilities(
            streaming=True,
            pushNotifications=True,
            stateTransitionHistory=True
        ),
        securitySchemes={
            "bearer": HTTPAuthSecurityScheme(
                type="http",
                scheme="bearer",
                description="JWT bearer token authentication",
                bearerFormat="JWT"
            )
        } if get_settings().auth_token else None,
        defaultInputModes=["text/plain"],
        defaultOutputModes=["text/plain"],
        skills=all_skills
    )


def format_sse(event: str, data: Any) -> bytes:
    """Serialize data as a server-sent event."""
    encoded = jsonable_encoder(data)
    return f"event: {event}\ndata: {json.dumps(encoded)}\n\n".encode("utf-8")


def require_authorization(authorization: Optional[str] = Header(default=None)) -> None:
    """FastAPI dependency enforcing bearer token authentication."""
    settings = get_settings()
    token = settings.auth_token

    # If no token is configured, allow access (development mode)
    if not token:
        return

    # If token is configured, enforce authentication
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
    provided = authorization.split(" ", 1)[1]
    if provided != token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid bearer token")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    configure_logging()

    # Initialize components
    app.state.database = SessionDatabase()

    # Initialize push notification manager first with settings
    settings = get_settings()
    app.state.push_notification_manager = PushNotificationManager(
        app.state.database,
        settings=settings.push_notifications
    )

    # Initialize streaming manager with push notification manager and settings
    app.state.streaming_manager = StreamingManager(
        app.state.push_notification_manager,
        max_websocket_connections=settings.push_notifications.max_websocket_connections,
        max_sse_connections=settings.push_notifications.max_sse_connections,
        cleanup_interval=settings.push_notifications.connection_cleanup_interval
    )

    # Connect streaming manager back to push notification manager for broadcasting
    app.state.push_notification_manager.streaming_manager = app.state.streaming_manager

    # Initialize task manager with push notification manager
    app.state.task_manager = A2ATaskManager(app.state.push_notification_manager)

    app.state.context_manager = A2AContextManager()

    # Initialize A2A translator
    app.state.a2a_translator = A2ATranslator()

    # Initialize bash executor with push notification manager and task manager for event emission
    from .bash_executor import BashToolExecutor
    app.state.bash_executor = BashToolExecutor(
        push_notification_manager=app.state.push_notification_manager,
        task_manager=app.state.task_manager
    )

    logger.info("A2A-ACP proxy initialized with A2A ↔ ZedACP translation")

    # Start background cleanup task for push notifications
    async def periodic_cleanup():
        """Periodically clean up expired push notification configurations."""
        cleanup_interval = settings.push_notifications.cleanup_interval
        while True:
            try:
                await asyncio.sleep(cleanup_interval)
                cleanup_count = await app.state.push_notification_manager.cleanup_expired_configs()
                if cleanup_count > 0:
                    logger.info(f"Periodic cleanup removed {cleanup_count} expired notification configs")

                # Also cleanup stale streaming connections
                stale_count = await app.state.streaming_manager.cleanup_stale_connections()
                if stale_count > 0:
                    logger.info(f"Periodic cleanup removed {stale_count} stale streaming connections")

            except Exception as e:
                logger.error("Error in periodic cleanup", extra={"error": str(e)})

    # Start the cleanup task
    cleanup_task = asyncio.create_task(periodic_cleanup())
    app.state.cleanup_task = cleanup_task

    yield

    # Cleanup
    # Cancel the background cleanup task
    if hasattr(app.state, 'cleanup_task'):
        app.state.cleanup_task.cancel()
        try:
            await app.state.cleanup_task
        except asyncio.CancelledError:
            pass

    await app.state.push_notification_manager.close()
    await app.state.streaming_manager.close()
    app.state.database.close()
    logger.info("A2A-ACP proxy shutdown")


def get_task_manager(request: Request) -> A2ATaskManager:
    return request.app.state.task_manager


def get_context_manager(request: Request) -> A2AContextManager:
    return request.app.state.context_manager


def get_database(request: Request) -> SessionDatabase:
    return request.app.state.database


def get_a2a_translator(request: Request) -> A2ATranslator:
    return request.app.state.a2a_translator


def get_push_notification_manager(request: Request) -> PushNotificationManager:
    return request.app.state.push_notification_manager


def get_streaming_manager(request: Request) -> StreamingManager:
    return request.app.state.streaming_manager


def get_bash_executor(request: Request):
    return request.app.state.bash_executor


async def get_agent_card(request: Request):
    """Get the static AgentCard for this server."""
    if request.app.state.agent_card is None:
        request.app.state.agent_card = await generate_static_agent_card()
    return request.app.state.agent_card


def handle_push_notification_config_method(
    method_name: str,
    params: Dict[str, Any],
    request_id: Any,
    handler_func: Callable,
    param_requirements: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Handle push notification configuration methods with consistent error handling.

    Args:
        method_name: Name of the method for error logging
        params: Request parameters
        request_id: JSON-RPC request ID
        handler_func: Async function to handle the method logic
        param_requirements: List of required parameter names

    Returns:
        JSON-RPC response dictionary
    """
    try:
        # Validate required parameters
        if param_requirements:
            missing_params = [param for param in param_requirements if param not in params]
            if missing_params:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32602,
                        "message": f"Missing required parameters: {', '.join(missing_params)}"
                    }
                }

        # Call the handler function
        result = asyncio.run(handler_func(params))

        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result
        }

    except Exception as e:
        logger.exception(f"Error in {method_name}")
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32603,
                "message": f"Failed to {method_name}: {str(e)}"
            }
        }


async def handle_message_send_jsonrpc(params: Dict[str, Any], request: Request, request_id: Any) -> Dict[str, Any]:
    """Handle message/send via JSON-RPC."""
    try:
        from a2a.models import MessageSendParams

        # Parse MessageSendParams
        message_params = MessageSendParams(**params)

        # Get dependencies
        task_manager = get_task_manager(request)
        context_manager = get_context_manager(request)

        # Get single agent configuration
        agent_config = get_agent_config()

        # Create A2A context for this task
        context_id = message_params.message.contextId or await context_manager.create_context("default-agent")

        # Create A2A task
        task = await task_manager.create_task(
            context_id=context_id,
            agent_name="default-agent",
            initial_message=message_params.message,
            metadata={"mode": "sync"}
        )

        try:
            async with ZedAgentConnection(agent_config["command"], api_key=agent_config["api_key"]) as connection:
                await connection.initialize()

                # Create or load ZedACP session if context is provided
                session_id = await connection.start_session(cwd=os.getcwd(), mcp_servers=[])

                # Execute the task (task manager handles translation and history updates)
                result = await task_manager.execute_task(
                    task_id=task.id,
                    agent_command=agent_config["command"],
                    api_key=agent_config["api_key"],
                    working_directory=os.getcwd(),
                    mcp_servers=[]
                )

                # Add completed task to context
                await context_manager.add_task_to_context(context_id, result)

                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": serialize_a2a(result)
                }

        except AgentProcessError as exc:
            logger.exception("Agent process failed during A2A message", extra={"task_id": task.id})
            # Mark task as failed
            await task_manager.cancel_task(task.id)

            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": f"Agent process failed: {str(exc)}"
                }
            }

    except Exception as e:
        logger.exception("Error in message/send JSON-RPC handler")
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32603,
                "message": f"Failed to send message: {str(e)}"
            }
        }


async def handle_tasks_get_jsonrpc(params: Dict[str, Any], request: Request, request_id: Any) -> Dict[str, Any]:
    """Handle tasks/get via JSON-RPC."""
    try:
        from a2a.models import TaskQueryParams

        # Parse TaskQueryParams
        task_query = TaskQueryParams(**params)

        # Get task from task manager
        task_manager = get_task_manager(request)
        task = await task_manager.get_task(task_query.id)

        if task:
            # Include message history if requested
            if task_query.historyLength and task_query.historyLength > 0:
                # Limit history to requested length
                if task.history and len(task.history) > task_query.historyLength:
                    task.history = task.history[-task_query.historyLength:]

            logger.info("Task retrieved successfully",
                       extra={"task_id": task.id, "status": task.status.state.value})
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": serialize_a2a(task)
            }
        else:
            # Task not found
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32001,
                    "message": "Task not found"
                }
            }

    except Exception as e:
        logger.exception("Error in tasks/get JSON-RPC handler")
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32603,
                "message": f"Failed to get task: {str(e)}"
            }
        }


async def handle_tasks_list_jsonrpc(params: Dict[str, Any], request: Request, request_id: Any) -> Dict[str, Any]:
    """Handle tasks/list via JSON-RPC."""
    try:
        from a2a.models import ListTasksParams, ListTasksResult

        # Parse ListTasksParams
        list_params = ListTasksParams(**params)

        # Get active tasks from task manager
        task_manager = get_task_manager(request)
        active_tasks = await task_manager.list_tasks()

        # Apply filtering
        filtered_tasks = active_tasks

        # Filter by context ID if specified
        if list_params.contextId:
            filtered_tasks = [t for t in filtered_tasks if t.contextId == list_params.contextId]

        # Filter by status if specified
        if list_params.status:
            filtered_tasks = [t for t in filtered_tasks if t.status.state == list_params.status]

        # Apply pagination
        start_index = 0
        if list_params.pageToken:
            try:
                start_index = int(list_params.pageToken)
            except (ValueError, TypeError):
                start_index = 0

        page_size = list_params.pageSize or 50
        end_index = start_index + page_size
        paginated_tasks = filtered_tasks[start_index:end_index]

        # Generate next page token if there are more results
        next_page_token = ""
        if len(filtered_tasks) > end_index:
            next_page_token = str(end_index)

        # Create result with history length consideration
        result_tasks = []
        for task in paginated_tasks:
            # Include history if requested
            if list_params.historyLength and list_params.historyLength > 0:
                if task.history and len(task.history) > list_params.historyLength:
                    task.history = task.history[-list_params.historyLength:]

            # Include artifacts if requested
            if not list_params.includeArtifacts:
                task.artifacts = None

            result_tasks.append(task)

        result = ListTasksResult(
            tasks=result_tasks,
            totalSize=len(filtered_tasks),
            pageSize=page_size,
            nextPageToken=next_page_token
        )

        logger.info("Tasks listed successfully",
                   extra={"total_tasks": len(filtered_tasks),
                         "returned_tasks": len(result_tasks),
                         "next_page_token": next_page_token})

        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": serialize_a2a(result)
        }

    except Exception as e:
        logger.exception("Error in tasks/list JSON-RPC handler")
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32603,
                "message": f"Failed to list tasks: {str(e)}"
            }
        }


async def handle_tasks_cancel_jsonrpc(params: Dict[str, Any], request: Request, request_id: Any) -> Dict[str, Any]:
    """Handle tasks/cancel via JSON-RPC."""
    try:
        from a2a.models import TaskIdParams

        # Parse TaskIdParams
        task_id_params = TaskIdParams(**params)

        # Cancel task via task manager
        task_manager = get_task_manager(request)
        success = await task_manager.cancel_task(task_id_params.id)

        if success:
            # Get updated task status
            task = await task_manager.get_task(task_id_params.id)
            if task:
                logger.info("Task cancelled successfully",
                           extra={"task_id": task_id_params.id})
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": task
                }
            else:
                # Task was cancelled and removed from active tasks
                from a2a.models import Task, TaskStatus, TaskState
                cancelled_task = Task(
                    id=task_id_params.id,
                    contextId="unknown",
                    status=TaskStatus(state=TaskState.CANCELLED),
                    metadata={"cancelled": True}
                )
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": cancelled_task
                }
        else:
            # Task not found or cancellation failed
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32001,
                    "message": "Task not found or could not be cancelled"
                }
            }

    except Exception as e:
        logger.exception("Error in tasks/cancel JSON-RPC handler")
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32603,
                "message": f"Failed to cancel task: {str(e)}"
            }
        }


async def handle_message_stream_jsonrpc(params: Dict[str, Any], request: Request, request_id: Any) -> StreamingResponse:
    """Handle message/stream via JSON-RPC with SSE streaming support."""
    try:
        # Parse MessageSendParams
        message_params = MessageSendParams(**params)

        # Get dependencies
        task_manager = get_task_manager(request)
        context_manager = get_context_manager(request)

        # Get single agent configuration
        agent_config = get_agent_config()

        # Create A2A context for this task
        context_id = message_params.message.contextId or await context_manager.create_context("default-agent")

        # Create A2A task
        task = await task_manager.create_task(
            context_id=context_id,
            agent_name="default-agent",
            initial_message=message_params.message,
            metadata={"mode": "streaming"}
        )

        stream_trace_id = f"{task.id}:{uuid4().hex[:8]}"
        logger.info(
            "Starting JSON-RPC streaming response",
            extra={
                "stream_id": stream_trace_id,
                "task_id": task.id,
                "context_id": task.contextId,
                "request_id": request_id,
            },
        )

        async def sse_event_generator():
            """Generate SSE events for JSON-RPC streaming response."""
            logger.debug(
                "JSON-RPC streaming generator started",
                extra={"stream_id": stream_trace_id, "task_id": task.id},
            )
            try:
                async for payload in iter_streaming_payloads(
                    task_manager,
                    agent_config,
                    task,
                    include_jsonrpc=True,
                    request_id=request_id,
                    stream_id=stream_trace_id,
                ):
                    logger.debug(
                        "Forwarding JSON-RPC streaming payload",
                        extra={"stream_id": stream_trace_id, "task_id": task.id},
                    )
                    yield f"data: {json.dumps(payload)}\n\n"
            except Exception as e:
                logger.exception("Error in JSON-RPC streaming event generator")
                error_response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32603,
                        "message": f"Failed to stream message: {str(e)}"
                    }
                }
                yield f"data: {json.dumps(error_response)}\n\n"
            finally:
                logger.debug(
                    "JSON-RPC streaming generator completed",
                    extra={"stream_id": stream_trace_id, "task_id": task.id},
                )

        return StreamingResponse(
            sse_event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control",
            }
        )

    except Exception as e:
        logger.exception("Error in message/stream JSON-RPC handler")
        error_message = f"Failed to stream message: {str(e)}"

        async def error_sse_generator(message: str):
            error_response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": message
                }
            }
            yield f"data: {json.dumps(error_response)}\n\n"

        return StreamingResponse(
            error_sse_generator(error_message),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control",
            }
        )


def create_app() -> FastAPI:
    """
    Create the A2A-ACP FastAPI application.

    Returns:
        Configured FastAPI application serving both ACP and A2A protocols
    """
    app = FastAPI(
        title="A2A-ACP Server",
        description="Native A2A protocol server bridging ZedACP agents to A2A clients",
        version="0.1.0",
        lifespan=lifespan
    )

    # Add CORS middleware for web clients
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Simple ping endpoint for health checks
    @app.get("/ping")
    async def ping() -> dict[str, str]:
        return {"status": "ok"}

    # Comprehensive health check endpoint
    @app.get("/health")
    async def health_check(
        request: Request,
        authorization: Optional[str] = Header(default=None)
    ) -> dict[str, Any]:
        """Comprehensive health check including push notification system status."""
        require_authorization(authorization)

        health_info = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "database": "unknown",
                "push_notifications": "unknown",
                "streaming": "unknown"
            },
            "version": "1.0.0"
        }

        try:
            # Check database connection - simple check if database exists and is accessible
            database = get_database(request)
            # Try to execute a simple query to verify database is working
            health_info["services"]["database"] = "healthy"
        except Exception as e:
            health_info["services"]["database"] = "unhealthy"
            health_info["status"] = "degraded"
            logger.error("Database health check failed", extra={"error": str(e)})

        try:
            # Check push notification manager - basic availability check
            push_mgr = get_push_notification_manager(request)
            health_info["services"]["push_notifications"] = "healthy"
        except Exception as e:
            health_info["services"]["push_notifications"] = "unhealthy"
            health_info["status"] = "unhealthy"
            logger.error("Push notification health check failed", extra={"error": str(e)})

        try:
            # Check streaming manager - basic availability check
            streaming_mgr = get_streaming_manager(request)
            health_info["services"]["streaming"] = "healthy"
        except Exception as e:
            health_info["services"]["streaming"] = "unhealthy"
            health_info["status"] = "degraded"
            logger.error("Streaming health check failed", extra={"error": str(e)})

        return health_info

    # Push notification metrics endpoint
    @app.get("/metrics/push-notifications")
    async def push_notification_metrics(
        request: Request,
        authorization: Optional[str] = Header(default=None)
    ) -> dict[str, Any]:
        """Get push notification system metrics."""
        require_authorization(authorization)

        push_mgr = get_push_notification_manager(request)

        metrics = {
            "timestamp": datetime.now().isoformat(),
            "delivery_stats": {
                "total_deliveries": 0,
                "successful_deliveries": 0,
                "failed_deliveries": 0,
                "success_rate": 0.0
            },
            "configuration_stats": {
                "total_configs": 0,
                "active_configs": 0,
                "expired_configs": 0
            },
            "performance_stats": {
                "average_response_time": 0.0,
                "requests_per_minute": 0.0,
                "error_rate": 0.0
            }
        }

        try:
            # Get basic delivery statistics from delivery history
            deliveries = await push_mgr.get_delivery_history()
            total_deliveries = len(deliveries)
            successful_deliveries = len([d for d in deliveries if d.delivery_status == "delivered"])
            failed_deliveries = len([d for d in deliveries if d.delivery_status == "failed"])

            metrics["delivery_stats"].update({
                "total_deliveries": total_deliveries,
                "successful_deliveries": successful_deliveries,
                "failed_deliveries": failed_deliveries,
                "success_rate": (successful_deliveries / total_deliveries * 100) if total_deliveries > 0 else 0.0
            })

            # Get configuration statistics by counting configs for all tasks
            # Current approach loads all deliveries and queries per task - acceptable for moderate usage
            # where each task has limited notification volume
            all_task_ids = set()
            for delivery in deliveries:
                all_task_ids.add(delivery.task_id)

            total_configs = 0
            active_configs = 0
            for task_id in all_task_ids:
                configs = await push_mgr.list_configs(task_id)
                total_configs += len(configs)
                # Consider configs active if they have recent deliveries
                recent_deliveries = [d for d in deliveries if d.task_id == task_id]
                if recent_deliveries:
                    active_configs += len(configs)

            metrics["configuration_stats"].update({
                "total_configs": total_configs,
                "active_configs": active_configs,
                "expired_configs": total_configs - active_configs
            })

            # Basic performance metrics
            metrics["performance_stats"].update({
                "average_response_time": 0.0,  # Would need timing data to calculate
                "requests_per_minute": total_deliveries / 60.0,  # Rough estimate
                "error_rate": (failed_deliveries / total_deliveries * 100) if total_deliveries > 0 else 0.0
            })

        except Exception as e:
            logger.error("Error retrieving push notification metrics", extra={"error": str(e)})
            metrics["error"] = str(e)

        return metrics

    # System metrics endpoint
    @app.get("/metrics/system")
    async def system_metrics(
        request: Request,
        authorization: Optional[str] = Header(default=None)
    ) -> dict[str, Any]:
        """Get system-wide metrics."""
        require_authorization(authorization)

        streaming_mgr = get_streaming_manager(request)

        metrics = {
            "timestamp": datetime.now().isoformat(),
            "streaming_connections": {
                "websocket": {
                    "active": 0,
                    "total": 0,
                    "limit": 1000
                },
                "sse": {
                    "active": 0,
                    "total": 0,
                    "limit": 500
                }
            },
            "background_tasks": {
                "cleanup_running": False,
                "last_cleanup": None
            }
        }

        try:
            # Get streaming connection stats
            connection_stats = await streaming_mgr.get_connection_stats()
            metrics["streaming_connections"]["websocket"]["active"] = connection_stats.get("websocket_active", 0)
            metrics["streaming_connections"]["websocket"]["total"] = connection_stats.get("websocket_total", 0)
            metrics["streaming_connections"]["sse"]["active"] = connection_stats.get("sse_active", 0)
            metrics["streaming_connections"]["sse"]["total"] = connection_stats.get("sse_total", 0)

            # Get connection limits from settings
            settings = get_settings()
            metrics["streaming_connections"]["websocket"]["limit"] = settings.push_notifications.max_websocket_connections
            metrics["streaming_connections"]["sse"]["limit"] = settings.push_notifications.max_sse_connections

            # Check if cleanup task is running
            cleanup_task = request.app.state.get("cleanup_task")
            metrics["background_tasks"]["cleanup_running"] = cleanup_task is not None and not cleanup_task.done()

        except Exception as e:
            logger.error("Error retrieving system metrics", extra={"error": str(e)})
            metrics["error"] = str(e)

        return metrics

    # Store static AgentCard for A2A responses
    # Note: We'll initialize this when first accessed since we can't await here
    app.state.agent_card = None

    # Well-known Agent Card discovery endpoint (publicly accessible)
    @app.get("/.well-known/agent-card.json")
    async def get_agent_card_well_known(request: Request) -> Any:
        """Agent Card discovery endpoint for A2A protocol compatibility.

        This endpoint allows A2A clients to discover agent capabilities
        without authentication, following well-known URL conventions.
        """
        try:
            agent_card = await get_agent_card(request)
            return serialize_a2a(agent_card)
        except Exception as e:
            logger.error("Error serving well-known agent card", extra={"error": str(e)})
            raise HTTPException(status_code=500, detail="Failed to generate agent card")

    # A2A JSON-RPC endpoint - delegate to A2A server
    @app.post("/a2a/rpc")
    async def a2a_jsonrpc(
        request: Request,
        authorization: Optional[str] = Header(default=None)
    ):
        """Handle A2A JSON-RPC 2.0 requests."""
        require_authorization(authorization)

        try:
            # Get the JSON-RPC request body
            body_bytes = await request.body()
            if not body_bytes:
                # Return method listing for empty requests
                return {
                    "jsonrpc": "2.0",
                    "id": None,
                    "result": {
                        "message": "A2A-ACP JSON-RPC endpoint ready",
                        "supported_methods": [
                            "message/send", "message/stream", "tasks/get",
                            "tasks/list", "tasks/cancel", "agent/getAuthenticatedExtendedCard",
                            "tasks/pushNotificationConfig/set", "tasks/pushNotificationConfig/get",
                            "tasks/pushNotificationConfig/list", "tasks/pushNotificationConfig/delete"
                        ],
                        "http_endpoints": {
                            "streaming": "POST /a2a/message/stream (SSE format)",
                            "non_streaming": "POST /a2a/message/send (JSON response)"
                        },
                        "notes": "JSON-RPC message/stream supports streaming via SSE format with proper A2A response objects."
                    }
                }

            body = json.loads(body_bytes)
            logger.info("Received A2A JSON-RPC request", extra={"method": body.get("method")})

            # Handle A2A methods directly
            method = body.get("method")
            params = body.get("params", {})
            request_id = body.get("id")

            if method == "message/send":
                # Handle message/send via JSON-RPC
                return await handle_message_send_jsonrpc(params, request, request_id)
            elif method == "message/stream":
                # Handle message/stream via JSON-RPC
                return await handle_message_stream_jsonrpc(params, request, request_id)
            elif method == "tasks/get":
                # Handle tasks/get via JSON-RPC
                return await handle_tasks_get_jsonrpc(params, request, request_id)
            elif method == "tasks/list":
                # Handle tasks/list via JSON-RPC
                return await handle_tasks_list_jsonrpc(params, request, request_id)
            elif method == "tasks/cancel":
                # Handle tasks/cancel via JSON-RPC
                return await handle_tasks_cancel_jsonrpc(params, request, request_id)
            elif method == "agent/getAuthenticatedExtendedCard":
                # Return the static AgentCard
                agent_card = await get_agent_card(request)
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": serialize_a2a(agent_card)
                }
            elif method == "tasks/pushNotificationConfig/set":
                # Handle push notification config set
                push_mgr = get_push_notification_manager(request)
                config_params = params

                try:
                    config = TaskPushNotificationConfig(
                        id=config_params["id"],
                        task_id=config_params["taskId"],
                        url=config_params["url"],
                        token=config_params.get("token"),
                        authentication_schemes=config_params.get("authenticationSchemes"),
                        credentials=config_params.get("credentials")
                    )

                    await push_mgr.store_config(config)

                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {"success": True}
                    }

                except Exception as e:
                    logger.exception("Error setting push notification config")
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {
                            "code": -32603,
                            "message": f"Failed to set push notification config: {str(e)}"
                        }
                    }

            elif method == "tasks/pushNotificationConfig/get":
                # Handle push notification config get
                async def handle_get(params):
                    push_mgr = get_push_notification_manager(request)
                    task_id = params.get("taskId")
                    config_id = params.get("id")

                    config = await push_mgr.get_config(task_id, config_id)

                    if config:
                        return config.to_dict()
                    else:
                        return {
                            "jsonrpc": "2.0",
                            "id": request_id,
                            "error": {
                                "code": -32602,
                                "message": "Push notification config not found"
                            }
                        }

                return handle_push_notification_config_method(
                    "tasks/pushNotificationConfig/get",
                    params,
                    request_id,
                    handle_get,
                    ["taskId", "id"]
                )

            elif method == "tasks/pushNotificationConfig/list":
                # Handle push notification config list
                push_mgr = get_push_notification_manager(request)
                task_id = params.get("taskId")

                if not task_id:
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {
                            "code": -32602,
                            "message": "Missing required parameter: taskId"
                        }
                    }

                try:
                    configs = await push_mgr.list_configs(task_id)

                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "configs": [config.to_dict() for config in configs]
                        }
                    }

                except Exception as e:
                    logger.exception("Error listing push notification configs")
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {
                            "code": -32603,
                            "message": f"Failed to list push notification configs: {str(e)}"
                        }
                    }

            elif method == "tasks/pushNotificationConfig/delete":
                # Handle push notification config delete
                async def handle_delete(params):
                    push_mgr = get_push_notification_manager(request)
                    task_id = params.get("taskId")
                    config_id = params.get("id")

                    success = await push_mgr.delete_config(task_id, config_id)
                    return {"success": success}

                return handle_push_notification_config_method(
                    "tasks/pushNotificationConfig/delete",
                    params,
                    request_id,
                    handle_delete,
                    ["taskId", "id"]
                )

            else:
                # For now, return method not found for other methods
                # TODO: Implement other A2A methods as needed
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method '{method}' not implemented in streamlined version"
                    }
                }

        except json.JSONDecodeError as e:
            logger.warning("Invalid JSON in A2A JSON-RPC request", extra={"error": str(e)})
            raise HTTPException(status_code=400, detail="Invalid JSON payload")
        except Exception as e:
            logger.exception("Error handling A2A JSON-RPC request")
            raise HTTPException(status_code=500, detail=str(e))

    # A2A HTTP API endpoints (simplified single-agent)
    @app.post("/a2a/message/send")
    async def a2a_message_send(
        params: MessageSendParams,
        task_manager: A2ATaskManager = Depends(get_task_manager),
        context_manager: A2AContextManager = Depends(get_context_manager),
        translator: A2ATranslator = Depends(get_a2a_translator),
        authorization: Optional[str] = Header(default=None)
    ):
        """Send an A2A message and return response."""
        require_authorization(authorization)

        # Get single agent configuration
        agent_config = get_agent_config()

        # Create A2A context for this task
        context_id = params.message.contextId or await context_manager.create_context("default-agent")

        # Create A2A task
        task = await task_manager.create_task(
            context_id=context_id,
            agent_name="default-agent",
            initial_message=params.message,
            metadata={"mode": "sync"}
        )

        try:
            async with ZedAgentConnection(agent_config["command"], api_key=agent_config["api_key"]) as connection:
                await connection.initialize()

                # Create or load ZedACP session if context is provided
                session_id = await connection.start_session(cwd=os.getcwd(), mcp_servers=[])

                # Execute the task
                result = await task_manager.execute_task(
                    task_id=task.id,
                    agent_command=agent_config["command"],
                    api_key=agent_config["api_key"],
                    working_directory=os.getcwd(),
                    mcp_servers=[]
                )

                # Extract the ZedACP result from the task for translation
                zedacp_result = {}  # The task execution result would need to be extracted from the Task object

                # Convert ZedACP response back to A2A format
                from a2a.models import TaskStatus, TaskState
                a2a_response = translator.zedacp_to_a2a_message(zedacp_result, task.id, task.id)

                # Update task with response
                if task.history is not None:
                    task.history.append(a2a_response)

                # Add task to context
                await context_manager.add_task_to_context(context_id, task)
                await context_manager.add_message_to_context(context_id, a2a_response)

                return {"task": task}

        except AgentProcessError as exc:
            logger.exception("Agent process failed during A2A message", extra={"task_id": task.id})
            # Mark task as failed
            await task_manager.cancel_task(task.id)
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))

    @app.post("/a2a/message/stream")
    async def a2a_message_stream(
        params: MessageSendParams,
        task_manager: A2ATaskManager = Depends(get_task_manager),
        context_manager: A2AContextManager = Depends(get_context_manager),
        authorization: Optional[str] = Header(default=None)
    ):
        """Stream an A2A message conversation using Server-Sent Events."""
        require_authorization(authorization)

        # Get single agent configuration
        agent_config = get_agent_config()

        # Create A2A context for this task
        context_id = params.message.contextId or await context_manager.create_context("default-agent")

        # Create A2A task
        task = await task_manager.create_task(
            context_id=context_id,
            agent_name="default-agent",
            initial_message=params.message,
            metadata={"mode": "streaming"}
        )

        stream_trace_id = f"{task.id}:{uuid4().hex[:8]}"
        logger.info(
            "Starting HTTP SSE streaming response",
            extra={
                "stream_id": stream_trace_id,
                "task_id": task.id,
                "context_id": task.contextId,
            },
        )

        async def event_generator():
            """Generate SSE events for the streaming conversation."""
            logger.debug(
                "HTTP SSE streaming generator started",
                extra={"stream_id": stream_trace_id, "task_id": task.id},
            )
            try:
                async for payload in iter_streaming_payloads(
                    task_manager,
                    agent_config,
                    task,
                    include_jsonrpc=True,
                    request_id=None,
                    stream_id=stream_trace_id,
                ):
                    logger.debug(
                        "Forwarding HTTP SSE streaming payload",
                        extra={"stream_id": stream_trace_id, "task_id": task.id},
                    )
                    yield f"data: {json.dumps(payload)}\n\n"
            except Exception as e:
                logger.exception("Error in streaming event generator")
                error_event = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32603,
                        "message": f"Failed to stream message: {str(e)}"
                    }
                }
                yield f"data: {json.dumps(error_event)}\n\n"
            finally:
                logger.debug(
                    "HTTP SSE streaming generator completed",
                    extra={"stream_id": stream_trace_id, "task_id": task.id},
                )

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control",
            }
        )

    # Real-time streaming endpoints
    @app.websocket("/streaming/websocket")
    async def websocket_notifications(
        websocket: WebSocket,
        task_filter: Optional[str] = None,  # Comma-separated task IDs
        authorization: Optional[str] = Header(default=None)
    ):
        """WebSocket endpoint for real-time task notifications."""
        require_authorization(authorization)

        # For WebSocket, we need to access app state directly
        streaming_mgr = websocket.app.state.streaming_manager

        # Parse task filter
        task_ids = None
        if task_filter:
            task_ids = [tid.strip() for tid in task_filter.split(",")]

        # Register connection
        connection_id = await streaming_mgr.register_websocket_connection(websocket, task_ids)

        try:
            # Handle incoming messages
            await streaming_mgr.handle_websocket_messages(websocket, connection_id)
        except Exception as e:
            logger.error(
                "WebSocket error",
                extra={"connection_id": connection_id, "error": str(e)}
            )
        finally:
            await streaming_mgr.unregister_websocket_connection(connection_id)

    @app.get("/streaming/sse")
    async def sse_notifications(
        request: Request,
        task_filter: Optional[str] = None,  # Comma-separated task IDs
        authorization: Optional[str] = Header(default=None)
    ):
        """Server-Sent Events endpoint for real-time task notifications."""
        require_authorization(authorization)

        streaming_mgr = get_streaming_manager(request)

        # Parse task filter
        task_ids = None
        if task_filter:
            task_ids = [tid.strip() for tid in task_filter.split(",")]

        # Register SSE connection
        connection_id, connection = await streaming_mgr.register_sse_connection(task_ids)

        # Return SSE streaming response
        return await streaming_mgr.create_sse_response(connection_id, connection)

    @app.get("/streaming/stats")
    async def streaming_stats(
        request: Request,
        authorization: Optional[str] = Header(default=None)
    ):
        """Get statistics about current streaming connections."""
        require_authorization(authorization)

        streaming_mgr = get_streaming_manager(request)
        return await streaming_mgr.get_connection_stats()

    @app.post("/streaming/cleanup")
    async def cleanup_streaming_connections(
        request: Request,
        authorization: Optional[str] = Header(default=None)
    ):
        """Clean up stale streaming connections."""
        require_authorization(authorization)

        streaming_mgr = get_streaming_manager(request)
        cleaned_count = await streaming_mgr.cleanup_stale_connections()

        return {
            "success": True,
            "cleaned_connections": cleaned_count,
            "message": f"Cleaned up {cleaned_count} stale connections"
        }

    logger.info("A2A-ACP application created successfully")
    return app


# Convenience function for running the server
def run_server(host: str = "localhost", port: int = 8000) -> None:
    """Run the A2A-ACP server directly."""
    import uvicorn

    app = create_app()
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )
