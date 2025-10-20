"""
A2A JSON-RPC 2.0 HTTP Server Implementation

Pure Python implementation of a JSON-RPC 2.0 server for the A2A protocol.
No external dependencies - handles HTTP transport and JSON-RPC protocol directly.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Optional, Union, Callable, Awaitable

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import StreamingResponse

from typing import List
from .models import (
    JSONRPCRequest, JSONRPCSuccessResponse, JSONRPCErrorResponse,
    JSONRPCError, AgentCard, Task, Message, TaskStatus, TaskState,
    MessageSendParams, TaskQueryParams, ListTasksParams, TaskIdParams,
    ListTasksResult, TaskPushNotificationConfig, TaskStatusUpdateEvent,
    TextPart, TaskNotFoundError,
    UnsupportedOperationError, AuthenticatedExtendedCardNotConfiguredError,
    generate_id
)
from .agent_manager import agent_manager
from .agent_card import agent_card_manager

logger = logging.getLogger(__name__)


class A2AServer:
    """
    A2A JSON-RPC 2.0 server implementation.

    Handles HTTP requests, parses JSON-RPC messages, dispatches to method
    handlers, and returns appropriate responses.
    """

    def __init__(self):
        self.methods: Dict[str, Callable] = {}
        self.id_counter = 1
        self.app = FastAPI(title="A2A-ACP Server", version="0.1.0")

        # Register HTTP endpoints
        self._setup_routes()

    def _setup_routes(self):
        """Set up FastAPI routes for A2A endpoints."""

        @self.app.post("/")
        async def handle_jsonrpc(request: Request) -> Response:
            """Main JSON-RPC 2.0 endpoint."""
            request_data = None
            try:
                body = await request.body()
                if not body:
                    raise HTTPException(status_code=400, detail="Empty request body")

                # Parse JSON-RPC request(s)
                try:
                    request_data = json.loads(body)
                except json.JSONDecodeError as e:
                    logger.error("Invalid JSON in request body", extra={"error": str(e)})
                    return Response(
                        content=json.dumps(self._create_error_response(
                            id=None,
                            code=-32700,
                            message="Invalid JSON payload"
                        )),
                        media_type="application/json"
                    )

                # Handle both single requests and batch requests
                if isinstance(request_data, list):
                    responses = []
                    for req in request_data:
                        response = await self._handle_single_request(req)
                        if response:
                            responses.append(response)
                    return Response(
                        content=json.dumps(responses),
                        media_type="application/json"
                    )
                else:
                    response = await self._handle_single_request(request_data)
                    if response is None:
                        # This was a notification with no response expected
                        return Response(content="", status_code=204)
                    return Response(
                        content=json.dumps(response),
                        media_type="application/json"
                    )

            except Exception as e:
                logger.exception("Unexpected error handling JSON-RPC request")
                request_id = None
                if 'request_data' in locals():
                    request_id = getattr(request_data, 'id', None)
                return Response(
                    content=json.dumps(self._create_error_response(
                        id=request_id,
                        code=-32603,
                        message="Internal server error"
                    )),
                    media_type="application/json"
                )

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "ok", "protocol": "A2A", "version": "0.1.0"}

    async def _handle_single_request(self, request_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle a single JSON-RPC request."""
        try:
            # Parse and validate request
            request = JSONRPCRequest(**request_data)
        except Exception as e:
            logger.error("Invalid JSON-RPC request format", extra={"error": str(e)})
            return self._create_error_response(
                id=request_data.get("id"),
                code=-32600,
                message="Invalid request format"
            )

        request_id = request.id
        method_name = request.method
        params = request.params or {}

        logger.info("Processing A2A request", extra={
            "method": method_name,
            "id": request_id,
            "has_params": params is not None
        })

        # Dispatch to method handler
        if method_name not in self.methods:
            logger.warning("Method not found", extra={"method": method_name})
            return self._create_error_response(
                id=request_id,
                code=-32601,
                message="Method not found"
            )

        try:
            handler = self.methods[method_name]
            result = await handler(params)

            # Return success response
            # Handle None request_id for notifications
            if request_id is None:
                # For notifications, we don't return a response
                return None

            return JSONRPCSuccessResponse(
                id=request_id,
                result=result
            ).model_dump(mode="json")

        except Exception as e:
            # Handle unexpected errors in method handlers
            logger.exception("Unexpected error in method handler", extra={
                "method": method_name,
                "error": str(e)
            })
            return self._create_error_response(
                id=request_id,
                code=-32603,
                message="Internal server error"
            )

    def _create_error_response(self, id: Any, code: int, message: str, data: Any = None) -> Dict[str, Any]:
        """Create a JSON-RPC error response."""
        try:
            return JSONRPCErrorResponse(
                id=id,
                error=JSONRPCError(code=code, message=message, data=data)
            ).model_dump(mode="json")
        except Exception as e:
            # Fallback error response if validation fails
            logger.warning("Error response validation failed, using fallback", extra={"error": str(e)})
            return {
                "jsonrpc": "2.0",
                "id": id,
                "error": {
                    "code": -32603,
                    "message": "Internal server error",
                    "data": {"original_error": message}
                }
            }

    def register_method(self, method_name: str, handler: Callable[[Dict[str, Any]], Awaitable[Union[Dict[str, Any], Any]]]):
        """Register an A2A method handler."""
        self.methods[method_name] = handler
        logger.debug("Registered A2A method handler", extra={"method": method_name})

    def get_fastapi_app(self) -> FastAPI:
        """Get the underlying FastAPI application."""
        return self.app


class A2AMethodHandlers:
    """
    Default A2A method handlers.

    Provides stub implementations for all core A2A methods.
    Should be extended or replaced with actual business logic.
    """

    def __init__(self, server: A2AServer):
        self.server = server
        self._register_default_handlers()

    def _register_default_handlers(self):
        """Register default method handlers."""

        # Message operations
        self.server.register_method("message/send", self.handle_message_send)
        self.server.register_method("message/stream", self.handle_message_stream)

        # Task management
        self.server.register_method("tasks/get", self.handle_tasks_get)
        self.server.register_method("tasks/list", self.handle_tasks_list)
        self.server.register_method("tasks/cancel", self.handle_tasks_cancel)

        # Push notification configuration
        self.server.register_method("tasks/pushNotificationConfig/set", self.handle_push_config_set)
        self.server.register_method("tasks/pushNotificationConfig/get", self.handle_push_config_get)
        self.server.register_method("tasks/pushNotificationConfig/list", self.handle_push_config_list)
        self.server.register_method("tasks/pushNotificationConfig/delete", self.handle_push_config_delete)

        # Agent information
        self._register_agent_handlers()

    def _create_push_notification_error_response(self, id: Any, message: str, error_code: int = -32603) -> Dict[str, Any]:
        """Create a standardized error response for push notification operations."""
        error = UnsupportedOperationError(
            code=error_code,
            message=message
        )
        return self.server._create_error_response(
            id=id,
            code=error.code,
            message=error.message,
            data=error.data
        )

    def _validate_push_notification_params(self, params: Dict[str, Any], required_params: List[str]) -> Optional[Dict[str, Any]]:
        """Validate push notification parameters and return error response if invalid."""
        for param in required_params:
            if param not in params:
                return self._create_push_notification_error_response(
                    id=list(params.keys())[0] if params else None,
                    message=f"Missing required parameter: {param}",
                    error_code=-32602
                )
        return None

    # Agent information
    def _register_agent_handlers(self):
        """Register agent-related method handlers."""
        self.server.register_method("agent/getAuthenticatedExtendedCard", self.handle_get_agent_card)

    async def handle_message_send(self, params: Dict[str, Any]) -> Union[Message, Task]:
        """Handle message/send requests."""
        logger.info("message/send called", extra={"params_keys": list(params.keys())})

        try:
            # Parse MessageSendParams
            message_params = MessageSendParams(**params)

            # TODO: Extract agent name from message metadata or configuration
            # For now, use a default agent name - in the full implementation,
            # this would come from the message metadata or routing configuration
            agent_name = message_params.metadata.get("agent_name") if message_params.metadata else None

            if not agent_name:
                # If no agent specified, return an error indicating this is required
                from .models import UnsupportedOperationError
                error = UnsupportedOperationError(
                    code=-32004,
                    message="Agent name must be specified in message metadata"
                )
                # This will be handled by the JSON-RPC layer as an error response
                raise ValueError("Agent name required")

            # Send message via agent manager
            task = await agent_manager.send_message(message_params, agent_name)

            logger.info("Message processed successfully",
                       extra={"task_id": task.id, "agent_name": agent_name})

            return task

        except Exception as e:
            logger.exception("Error in message/send handler", extra={"error": str(e)})
            # Re-raise the exception to be handled by the JSON-RPC layer
            raise

    async def handle_message_stream(self, params: Dict[str, Any]):
        """Handle message/stream requests with Server-Sent Events."""
        logger.info("message/stream called", extra={"params_keys": list(params.keys())})

        try:
            # Parse MessageSendParams
            message_params = MessageSendParams(**params)

            # TODO: Extract agent name from message metadata
            agent_name = message_params.metadata.get("agent_name") if message_params.metadata else None

            if not agent_name:
                error = UnsupportedOperationError(
                    code=-32004,
                    message="Agent name must be specified in message metadata for streaming"
                )
                # Return error as JSON-RPC response
                error_response_data = self.server._create_error_response(
                    id=list(params.keys())[0] if params else "streaming-setup",
                    code=error.code,
                    message=error.message
                )
                return JSONRPCErrorResponse(
                    id=list(params.keys())[0] if params else "streaming-setup",
                    error=error_response_data["error"]
                )

            async def event_stream():
                try:
                    # Create initial task
                    task = Task(
                        id=generate_id("task_"),
                        contextId=generate_id("ctx_"),
                        status=TaskStatus(state=TaskState.SUBMITTED),
                        metadata={"agent_name": agent_name, "streaming": True}
                    )

                    # Send initial task response
                    yield JSONRPCSuccessResponse(
                        id=self.server.id_counter,
                        result=task
                    ).model_dump(mode="json")

                    # Update task status to working
                    task.status.state = TaskState.WORKING
                    status_update = TaskStatusUpdateEvent(
                        taskId=task.id,
                        contextId=task.contextId,
                        status=task.status,
                        final=False
                    )

                    yield JSONRPCSuccessResponse(
                        id=self.server.id_counter + 1,
                        result=status_update
                    ).model_dump(mode="json")

                    # TODO: Implement actual streaming with ZedACP agent
                    # For now, simulate streaming responses

                    # Simulate agent thinking/planning
                    await asyncio.sleep(0.2)
                    thinking_message = Message(
                        role="agent",
                        parts=[TextPart(kind="text", text="Thinking about your request...")],
                        messageId=generate_id("msg_"),
                        taskId=task.id,
                        contextId=task.contextId
                    )

                    yield JSONRPCSuccessResponse(
                        id=self.server.id_counter + 2,
                        result=thinking_message
                    ).model_dump(mode="json")

                    # Simulate streaming response chunks
                    response_chunks = [
                        "Here's my analysis of your request.\n\n",
                        "Based on the information provided, I can help you with several aspects:\n\n",
                        "1. Code generation and implementation\n",
                        "2. File system operations\n",
                        "3. Text processing and analysis\n\n",
                        "The A2A protocol integration is working well with the ZedACP agent bridge."
                    ]

                    for i, chunk in enumerate(response_chunks):
                        await asyncio.sleep(0.3)  # Simulate processing time

                        chunk_message = Message(
                            role="agent",
                            parts=[TextPart(kind="text", text=chunk)],
                            messageId=generate_id("msg_"),
                            taskId=task.id,
                            contextId=task.contextId
                        )

                        yield JSONRPCSuccessResponse(
                            id=self.server.id_counter + 3 + i,
                            result=chunk_message
                        ).model_dump(mode="json")

                    # Send completion status
                    task.status.state = TaskState.COMPLETED
                    final_update = TaskStatusUpdateEvent(
                        taskId=task.id,
                        contextId=task.contextId,
                        status=task.status,
                        final=True
                    )

                    yield JSONRPCSuccessResponse(
                        id=self.server.id_counter + len(response_chunks) + 3,
                        result=final_update
                    ).model_dump(mode="json")

                    logger.info("Streaming completed successfully",
                               extra={"task_id": task.id, "chunks_sent": len(response_chunks) + 2})

                except Exception as e:
                    logger.exception("Error in streaming response", extra={"error": str(e)})

                    # Send error status - use fallback IDs if task variables are not available
                    fallback_task_id = generate_id("task_")
                    fallback_context_id = generate_id("ctx_")

                    error_update = TaskStatusUpdateEvent(
                        taskId=fallback_task_id,
                        contextId=fallback_context_id,
                        status=TaskStatus(state=TaskState.FAILED),
                        final=True
                    )

                    yield JSONRPCSuccessResponse(
                        id=self.server.id_counter + 100,  # High ID for error responses
                        result=error_update
                    ).model_dump(mode="json")

            return StreamingResponse(
                self._jsonrpc_stream_generator(event_stream()),
                media_type="application/json"
            )

        except Exception as e:
            logger.exception("Error setting up message/stream", extra={"error": str(e)})
            # Return error response for setup errors
            error_response_data = self.server._create_error_response(
                id=list(params.keys())[0] if params else "stream-setup",
                code=-32004,
                message=f"Streaming setup failed: {str(e)}"
            )
            return JSONRPCErrorResponse(
                id=list(params.keys())[0] if params else "stream-setup",
                error=error_response_data["error"]
            )

    async def _jsonrpc_stream_generator(self, async_gen):
        """Convert async generator to newline-delimited JSON-RPC responses."""
        try:
            async for item in async_gen:
                yield json.dumps(item) + "\n"
        except Exception as e:
            logger.exception("Error in streaming response")
            error_response = self.server._create_error_response(
                id=None, code=-32603, message="Streaming error"
            )
            yield json.dumps(error_response) + "\n"

    async def handle_tasks_get(self, params: Dict[str, Any]) -> Union[Task, Dict[str, Any]]:
        """Handle tasks/get requests."""
        logger.info("tasks/get called", extra={"params": params})

        try:
            # Parse TaskQueryParams
            task_query = TaskQueryParams(**params)

            # Get task from agent manager
            task = await agent_manager.get_task_status(task_query.id)

            if task:
                # Include message history if requested
                if task_query.historyLength and task_query.historyLength > 0:
                    # Limit history to requested length
                    if task.history and len(task.history) > task_query.historyLength:
                        task.history = task.history[-task_query.historyLength:]

                logger.info("Task retrieved successfully",
                           extra={"task_id": task.id, "status": task.status.state.value})
                return task
            else:
                # Task not found
                error = TaskNotFoundError(code=-32001, message="Task not found")
                return self.server._create_error_response(
                    id=list(params.keys())[0] if params else None,
                    code=error.code,
                    message=error.message,
                    data=error.data
                )

        except Exception as e:
            logger.exception("Error in tasks/get handler", extra={"error": str(e)})
            error = TaskNotFoundError(code=-32001, message=f"Error retrieving task: {str(e)}")
            return self.server._create_error_response(
                id=list(params.keys())[0] if params else None,
                code=error.code,
                message=error.message,
                data=error.data
            )

    async def handle_tasks_list(self, params: Dict[str, Any]) -> ListTasksResult:
        """Handle tasks/list requests."""
        logger.info("tasks/list called", extra={"params": params})

        list_params = None
        try:
            # Parse ListTasksParams
            list_params = ListTasksParams(**params)

            # Get active tasks from agent manager
            active_tasks = await agent_manager.list_active_tasks()

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

            return result

        except Exception as e:
            logger.exception("Error in tasks/list handler", extra={"error": str(e)})
            # Return empty result on error
            default_page_size = 50
            if list_params:
                default_page_size = list_params.pageSize or 50

            return ListTasksResult(
                tasks=[],
                totalSize=0,
                pageSize=default_page_size,
                nextPageToken=""
            )

    async def handle_tasks_cancel(self, params: Dict[str, Any]) -> Union[Task, Dict[str, Any]]:
        """Handle tasks/cancel requests."""
        logger.info("tasks/cancel called", extra={"params": params})

        try:
            # Parse TaskIdParams
            task_id_params = TaskIdParams(**params)

            # Cancel task via agent manager
            success = await agent_manager.cancel_task(task_id_params.id)

            if success:
                # Get updated task status
                task = await agent_manager.get_task_status(task_id_params.id)
                if task:
                    logger.info("Task cancelled successfully",
                               extra={"task_id": task_id_params.id})
                    return task
                else:
                    # Task was cancelled and removed from active tasks
                    cancelled_task = Task(
                        id=task_id_params.id,
                        contextId="unknown",
                        status=TaskStatus(state=TaskState.CANCELLED),
                        metadata={"cancelled": True}
                    )
                    return cancelled_task
            else:
                # Task not found or cancellation failed
                error = TaskNotFoundError(code=-32001, message="Task not found or could not be cancelled")
                return self.server._create_error_response(
                    id=list(params.keys())[0] if params else None,
                    code=error.code,
                    message=error.message,
                    data=error.data
                )

        except Exception as e:
            logger.exception("Error in tasks/cancel handler", extra={"error": str(e)})
            # Return error as per A2A spec
            error = TaskNotFoundError(code=-32001, message=f"Cancellation failed: {str(e)}")
            return self.server._create_error_response(
                id=list(params.keys())[0] if params else None,
                code=error.code,
                message=error.message,
                data=error.data
            )

    async def handle_push_config_set(self, params: Dict[str, Any]) -> Union[TaskPushNotificationConfig, Dict[str, Any]]:
        """Handle push notification config set requests."""
        logger.info("tasks/pushNotificationConfig/set called", extra={"params_keys": list(params.keys())})

        try:
            # Parse the A2A TaskPushNotificationConfig structure
            task_push_config = TaskPushNotificationConfig(**params)

            # TODO: Integrate with PushNotificationManager when available in A2A server context
            # For now, return success response
            logger.info("Push notification config set (stub)",
                       extra={"task_id": task_push_config.taskId,
                              "config_url": task_push_config.pushNotificationConfig.url})

            return {"success": True}

        except Exception as e:
            logger.exception("Error in push notification config set", extra={"error": str(e)})
            error = UnsupportedOperationError(
                code=-32603,
                message=f"Failed to set push notification config: {str(e)}"
            )
            return self.server._create_error_response(
                id=list(params.keys())[0] if params else None,
                code=error.code,
                message=error.message,
                data=error.data
            )

    async def handle_push_config_get(self, params: Dict[str, Any]) -> Union[TaskPushNotificationConfig, Dict[str, Any]]:
        """Handle push notification config get requests."""
        logger.info("tasks/pushNotificationConfig/get called", extra={"params_keys": list(params.keys())})

        # Validate parameters
        validation_error = self._validate_push_notification_params(params, ["taskId", "id"])
        if validation_error:
            return validation_error

        try:
            # Extract parameters for getting config
            task_id = params.get("taskId")
            config_id = params.get("id")

            # TODO: Integrate with PushNotificationManager when available
            # For now, return not found error
            logger.info("Push notification config get (stub)",
                        extra={"task_id": task_id, "config_id": config_id})

            return self._create_push_notification_error_response(
                id=list(params.keys())[0] if params else None,
                message="Push notification config not found",
                error_code=-32001
            )

        except Exception as e:
            logger.exception("Error in push notification config get", extra={"error": str(e)})
            return self._create_push_notification_error_response(
                id=list(params.keys())[0] if params else None,
                message=f"Failed to get push notification config: {str(e)}"
            )

    async def handle_push_config_list(self, params: Dict[str, Any]) -> Union[List[TaskPushNotificationConfig], Dict[str, Any]]:
        """Handle push notification config list requests."""
        logger.info("tasks/pushNotificationConfig/list called", extra={"params_keys": list(params.keys())})

        try:
            # Extract task ID for listing configs
            task_id = params.get("taskId")

            if not task_id:
                error = UnsupportedOperationError(
                    code=-32602,
                    message="Missing required parameter: taskId"
                )
                return self.server._create_error_response(
                    id=list(params.keys())[0] if params else None,
                    code=error.code,
                    message=error.message,
                    data=error.data
                )

            # TODO: Integrate with PushNotificationManager when available
            # For now, return empty list
            logger.info("Push notification config list (stub)",
                       extra={"task_id": task_id})

            return {"configs": []}

        except Exception as e:
            logger.exception("Error in push notification config list", extra={"error": str(e)})
            error = UnsupportedOperationError(
                code=-32603,
                message=f"Failed to list push notification configs: {str(e)}"
            )
            return self.server._create_error_response(
                id=list(params.keys())[0] if params else None,
                code=error.code,
                message=error.message,
                data=error.data
            )

    async def handle_push_config_delete(self, params: Dict[str, Any]) -> Union[None, Dict[str, Any]]:
        """Handle push notification config delete requests."""
        logger.info("tasks/pushNotificationConfig/delete called", extra={"params_keys": list(params.keys())})

        # Validate parameters
        validation_error = self._validate_push_notification_params(params, ["taskId", "id"])
        if validation_error:
            return validation_error

        try:
            # Extract parameters for deleting config
            task_id = params.get("taskId")
            config_id = params.get("id")

            # TODO: Integrate with PushNotificationManager when available
            # For now, return success
            logger.info("Push notification config delete (stub)",
                        extra={"task_id": task_id, "config_id": config_id})

            return {"success": True}

        except Exception as e:
            logger.exception("Error in push notification config delete", extra={"error": str(e)})
            return self._create_push_notification_error_response(
                id=list(params.keys())[0] if params else None,
                message=f"Failed to delete push notification config: {str(e)}"
            )

    async def handle_get_agent_card(self, params: Dict[str, Any]) -> Union[AgentCard, Dict[str, Any]]:
        """Handle agent/getAuthenticatedExtendedCard requests."""
        logger.info("agent/getAuthenticatedExtendedCard called")

        try:
            # Debug: Check what agents are available
            available_agents = agent_card_manager.list_available_agents()
            logger.info("Available agents for card generation",
                       extra={"agents": available_agents, "count": len(available_agents)})

            # For now, return a default agent card
            # In the full implementation, this might be based on the requesting client
            # or specific agent selection from params

            # Get the first available agent as default
            if not available_agents:
                logger.error("No agents available for card generation")
                error = AuthenticatedExtendedCardNotConfiguredError(
                    code=-32007,
                    message="No agents configured"
                )
                return self.server._create_error_response(
                    id=list(params.keys())[0] if params else None,
                    code=error.code,
                    message=error.message,
                    data=error.data
                )

            # Generate card for the first available agent
            agent_name = available_agents[0]
            logger.info("Generating card for agent", extra={"agent_name": agent_name})

            agent_card = agent_card_manager.get_agent_card(agent_name)

            if agent_card is None:
                logger.error("Agent card generation returned None",
                           extra={"agent_name": agent_name})
                error = AuthenticatedExtendedCardNotConfiguredError(
                    code=-32007,
                    message=f"Failed to generate agent card for {agent_name}: returned None"
                )
                return self.server._create_error_response(
                    id=list(params.keys())[0] if params else None,
                    code=error.code,
                    message=error.message,
                    data=error.data
                )

            logger.info("Agent card generated successfully",
                        extra={"agent_name": agent_card.name, "skills_count": len(agent_card.skills)})

            return agent_card

        except Exception as e:
            logger.exception("Error generating agent card", extra={"error": str(e)})
            error = AuthenticatedExtendedCardNotConfiguredError(
                code=-32007,
                message=f"Failed to generate agent card: {str(e)}"
            )
            return self.server._create_error_response(
                id=list(params.keys())[0] if params else None,
                code=error.code,
                message=error.message,
                data=error.data
            )


# Convenience function to create A2A server with default handlers
def create_a2a_server() -> A2AServer:
    """Create a new A2A server with default method handlers."""
    server = A2AServer()
    A2AMethodHandlers(server)
    return server


# Export commonly used symbols
__all__ = [
    "A2AServer",
    "A2AMethodHandlers",
    "create_a2a_server"
]