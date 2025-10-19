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
from typing import Any, AsyncGenerator, Optional, List, Dict

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import Response, StreamingResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware

from .agent_registry import AgentRegistry
from .database import SessionDatabase
from .logging_config import configure_logging
from .models import AgentManifest, AgentSummary, Run, RunCreateRequest, RunMode, RunStatus, Message, MessagePart
from .task_manager import A2ATaskManager
from .context_manager import A2AContextManager
from .settings import get_settings
from .zed_agent import AgentProcessError, PromptCancelled, ZedAgentConnection

# Import A2A protocol components
from ..a2a.models import Message as A2AMessage, MessageSendParams, Task
from ..a2a.translator import A2ATranslator

logger = logging.getLogger(__name__)


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
    app.state.registry = AgentRegistry()
    app.state.task_manager = A2ATaskManager()
    app.state.context_manager = A2AContextManager()

    # Load agent configuration
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "agents.json")
    with open(config_path, 'r') as f:
        agent_config = json.load(f)

    # Initialize database
    app.state.database = SessionDatabase()

    # Initialize A2A translator
    app.state.a2a_translator = A2ATranslator()

    logger.info("A2A-ACP proxy initialized with A2A ↔ ZedACP translation")
    yield

    # Cleanup
    app.state.database.close()
    logger.info("A2A-ACP proxy shutdown")


def get_registry(request: Request) -> AgentRegistry:
    return request.app.state.registry


def get_task_manager(request: Request) -> A2ATaskManager:
    return request.app.state.task_manager


def get_context_manager(request: Request) -> A2AContextManager:
    return request.app.state.context_manager


def get_database(request: Request) -> SessionDatabase:
    return request.app.state.database


def get_a2a_translator(request: Request) -> A2ATranslator:
    return request.app.state.a2a_translator


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

    # Legacy ACP endpoints (for backward compatibility during transition)
    @app.get("/ping", dependencies=[Depends(require_authorization)])
    async def ping() -> dict[str, str]:
        return {"status": "ok"}

    @app.get(
        "/agents",
        response_model=list[AgentSummary],
        dependencies=[Depends(require_authorization)],
    )
    async def list_agents(registry: AgentRegistry = Depends(get_registry)) -> list[AgentSummary]:
        agents = [
            AgentSummary(name=agent.name, description=agent.description)
            for agent in registry.list()
        ]
        return agents

    @app.get(
        "/agents/{name}",
        response_model=AgentManifest,
        dependencies=[Depends(require_authorization)],
    )
    async def agent_manifest(name: str, registry: AgentRegistry = Depends(get_registry)) -> AgentManifest:
        try:
            return registry.manifest_for(name)
        except KeyError:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found") from None

    # Initialize A2A server for JSON-RPC handling
    from ..a2a.server import create_a2a_server
    a2a_server = create_a2a_server()

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
                            "tasks/list", "tasks/cancel", "agent/getAuthenticatedExtendedCard"
                        ]
                    }
                }

            body = json.loads(body_bytes)
            logger.info("Received A2A JSON-RPC request", extra={"method": body.get("method")})

            # Use the A2A server's request handling
            response = await a2a_server._handle_single_request(body)

            if response is None:
                # This was a notification with no response expected
                return Response(content="", status_code=204)

            return response

        except json.JSONDecodeError as e:
            logger.warning("Invalid JSON in A2A JSON-RPC request", extra={"error": str(e)})
            raise HTTPException(status_code=400, detail="Invalid JSON payload")
        except Exception as e:
            logger.exception("Error handling A2A JSON-RPC request")
            raise HTTPException(status_code=500, detail=str(e))

    # A2A HTTP API endpoints
    @app.post("/a2a/message/send")
    async def a2a_message_send(
        params: MessageSendParams,
        registry: AgentRegistry = Depends(get_registry),
        task_manager: A2ATaskManager = Depends(get_task_manager),
        context_manager: A2AContextManager = Depends(get_context_manager),
        translator: A2ATranslator = Depends(get_a2a_translator),
        authorization: Optional[str] = Header(default=None)
    ):
        """Send an A2A message and return response."""
        require_authorization(authorization)

        # Extract agent name from message metadata
        agent_name = params.metadata.get("agent_name") if params.metadata else None
        if not agent_name:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Agent name must be specified in message metadata")

        try:
            agent = registry.get(agent_name)
        except KeyError:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found") from None

        # Create A2A context for this task
        context_id = params.message.contextId or await context_manager.create_context(agent.name)

        # Create A2A task
        task = await task_manager.create_task(
            context_id=context_id,
            agent_name=agent.name,
            initial_message=params.message,
            metadata={"mode": "sync"}
        )

        try:
            async with ZedAgentConnection(agent.command, api_key=agent.api_key) as connection:
                await connection.initialize()

                # Create or load ZedACP session if context is provided
                session_id = await connection.start_session(cwd=os.getcwd(), mcp_servers=[])

                # Execute the task
                result = await task_manager.execute_task(
                    task_id=task.id,
                    agent_command=agent.command,
                    api_key=agent.api_key,
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

    logger.info("A2A-ACP application created successfully")
    return app


# Convenience function for running the server
def run_server(host: str = "localhost", port: int = 8001) -> None:
    """Run the A2A-ACP server directly."""
    import uvicorn

    app = create_app()
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )