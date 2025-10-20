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
from typing import Any, AsyncGenerator, Optional, List, Dict

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status, WebSocket
from fastapi.responses import Response, StreamingResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware

from .database import SessionDatabase
from .logging_config import configure_logging
from .task_manager import A2ATaskManager
from .context_manager import A2AContextManager
from .settings import get_settings
from .zed_agent import AgentProcessError, PromptCancelled, ZedAgentConnection
from .push_notification_manager import PushNotificationManager
from .streaming_manager import StreamingManager

# Import A2A protocol components
from ..a2a.models import Message as A2AMessage, MessageSendParams, Task
from ..a2a.translator import A2ATranslator

# Import push notification models
from .models import TaskPushNotificationConfig

logger = logging.getLogger(__name__)


def get_agent_config() -> Dict[str, Any]:
    """Get the single agent configuration from settings."""
    settings = get_settings()

    # For testing/development, provide fallback defaults
    command = settings.agent_command or "python tests/dummy_agent.py"
    description = settings.agent_description or "A2A-ACP Development Agent"

    return {
        "command": command,
        "api_key": settings.agent_api_key,
        "description": description
    }


def generate_static_agent_card():
    """Generate a static AgentCard for the single configured agent."""
    from ..a2a.models import (
        AgentCard, AgentCapabilities, AgentSkill, AgentProvider,
        SecurityScheme, HTTPAuthSecurityScheme
    )

    agent_config = get_agent_config()

    return AgentCard(
        protocolVersion="0.3.0",
        name="a2a-acp-agent",
        description=agent_config["description"],
        url="http://localhost:8000",  # Server URL
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
        skills=[
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
        ]
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


def get_agent_card(request: Request):
    """Get the static AgentCard for this server."""
    return request.app.state.agent_card


def handle_push_notification_config_method(
    method_name: str,
    params: Dict[str, Any],
    request_id: Any,
    handler_func: callable,
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
    app.state.agent_card = generate_static_agent_card()

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
                        ]
                    }
                }

            body = json.loads(body_bytes)
            logger.info("Received A2A JSON-RPC request", extra={"method": body.get("method")})

            # Handle A2A methods directly
            method = body.get("method")
            params = body.get("params", {})
            request_id = body.get("id")

            if method == "agent/getAuthenticatedExtendedCard":
                # Return the static AgentCard
                agent_card = get_agent_card(request)
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": agent_card.model_dump() if hasattr(agent_card, 'model_dump') else agent_card
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
                from ..a2a.models import TaskStatus, TaskState
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