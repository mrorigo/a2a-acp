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

# Removed: from .agent_registry import AgentRegistry
from .database import SessionDatabase
from .logging_config import configure_logging
from .task_manager import A2ATaskManager
from .context_manager import A2AContextManager
from .settings import get_settings
from .zed_agent import AgentProcessError, PromptCancelled, ZedAgentConnection

# Import A2A protocol components
from ..a2a.models import Message as A2AMessage, MessageSendParams, Task
from ..a2a.translator import A2ATranslator

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
        url="http://localhost:8001",  # Server URL
        preferredTransport="JSONRPC",
        version="1.0.0",
        capabilities=AgentCapabilities(
            streaming=True,
            pushNotifications=False,
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
    app.state.task_manager = A2ATaskManager()
    app.state.context_manager = A2AContextManager()

    # Initialize database
    app.state.database = SessionDatabase()

    # Initialize A2A translator
    app.state.a2a_translator = A2ATranslator()

    logger.info("A2A-ACP proxy initialized with A2A ↔ ZedACP translation")
    yield

    # Cleanup
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


def get_agent_card(request: Request):
    """Get the static AgentCard for this server."""
    return request.app.state.agent_card


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
                            "tasks/list", "tasks/cancel", "agent/getAuthenticatedExtendedCard"
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