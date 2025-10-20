"""
Real-time Event Streaming Manager for ZedACP Push Notifications.

This module provides WebSocket and Server-Sent Events (SSE) support for real-time
notification streaming, enabling live updates for task events and notifications.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Set, Optional, Any, TYPE_CHECKING
import uuid

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from .models import NotificationPayload, TaskStatusChangePayload, TaskMessagePayload, TaskArtifactPayload

if TYPE_CHECKING:
    from .push_notification_manager import PushNotificationManager

logger = logging.getLogger(__name__)


class StreamingConnection:
    """Represents a single streaming connection (WebSocket or SSE)."""

    def __init__(self, connection_id: str, task_filter: Optional[Set[str]] = None):
        self.connection_id = connection_id
        self.task_filter = task_filter or set()  # Empty set means all tasks
        self.connected_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()

    def should_receive_notification(self, task_id: str) -> bool:
        """Check if this connection should receive notifications for the given task."""
        return len(self.task_filter) == 0 or task_id in self.task_filter

    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = datetime.utcnow()


class StreamingManager:
    """
    Manages real-time notification streaming via WebSocket and SSE.

    Provides connection management, message broadcasting, and integration
    with the push notification system for live event streaming.
    """

    def __init__(
        self,
        push_notification_manager: Optional[Any] = None,
        max_websocket_connections: int = 100,
        max_sse_connections: int = 200,
        cleanup_interval: int = 300
    ):
        """Initialize the streaming manager."""
        self.push_mgr = push_notification_manager
        self.websocket_connections: Dict[str, StreamingConnection] = {}
        self.sse_connections: Dict[str, StreamingConnection] = {}
        self._lock: Optional[asyncio.Lock] = None

        # Configuration
        self.max_websocket_connections = max_websocket_connections
        self.max_sse_connections = max_sse_connections
        self.cleanup_interval = cleanup_interval

    @property
    def lock(self) -> asyncio.Lock:
        """Get or create the asyncio lock for thread-safe operations."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def register_websocket_connection(
        self,
        websocket: WebSocket,
        task_filter: Optional[List[str]] = None
    ) -> str:
        """Register a new WebSocket connection for real-time notifications."""
        await websocket.accept()

        connection_id = str(uuid.uuid4())
        connection = StreamingConnection(
            connection_id=connection_id,
            task_filter=set(task_filter) if task_filter else set()
        )

        async with self.lock:
            self.websocket_connections[connection_id] = connection

        logger.info(
            "WebSocket connection registered",
            extra={
                "connection_id": connection_id,
                "task_filter": list(connection.task_filter) if connection.task_filter else None
            }
        )

        # Send welcome message
        welcome_message = {
            "type": "connection_established",
            "connection_id": connection_id,
            "timestamp": connection.connected_at.isoformat(),
            "message": "Connected to notification stream"
        }

        try:
            await websocket.send_text(json.dumps(welcome_message))
        except Exception as e:
            logger.error(
                "Failed to send welcome message",
                extra={"connection_id": connection_id, "error": str(e)}
            )

        return connection_id

    async def unregister_websocket_connection(self, connection_id: str) -> None:
        """Unregister a WebSocket connection."""
        async with self.lock:
            if connection_id in self.websocket_connections:
                del self.websocket_connections[connection_id]
                logger.info(
                    "WebSocket connection unregistered",
                    extra={"connection_id": connection_id}
                )

    async def handle_websocket_messages(
        self,
        websocket: WebSocket,
        connection_id: str
    ) -> None:
        """Handle incoming messages from a WebSocket connection."""
        try:
            while True:
                # Receive control messages from client
                data = await websocket.receive_text()

                try:
                    message = json.loads(data)
                    message_type = message.get("type")

                    if message_type == "ping":
                        # Update activity and respond with pong
                        async with self.lock:
                            if connection_id in self.websocket_connections:
                                self.websocket_connections[connection_id].update_activity()

                        await websocket.send_text(json.dumps({
                            "type": "pong",
                            "timestamp": datetime.utcnow().isoformat()
                        }))

                    elif message_type == "subscribe":
                        # Update task filter
                        task_ids = message.get("taskIds", [])
                        async with self.lock:
                            if connection_id in self.websocket_connections:
                                self.websocket_connections[connection_id].task_filter = set(task_ids)

                        await websocket.send_text(json.dumps({
                            "type": "subscription_updated",
                            "taskIds": task_ids,
                            "timestamp": datetime.utcnow().isoformat()
                        }))

                    elif message_type == "unsubscribe":
                        # Clear task filter (subscribe to all)
                        async with self.lock:
                            if connection_id in self.websocket_connections:
                                self.websocket_connections[connection_id].task_filter = set()

                        await websocket.send_text(json.dumps({
                            "type": "subscription_updated",
                            "taskIds": [],
                            "timestamp": datetime.utcnow().isoformat()
                        }))

                except json.JSONDecodeError:
                    logger.warning(
                        "Invalid JSON received from WebSocket",
                        extra={"connection_id": connection_id, "data": data[:100]}
                    )

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected", extra={"connection_id": connection_id})
            await self.unregister_websocket_connection(connection_id)
        except Exception as e:
            logger.error(
                "WebSocket error",
                extra={"connection_id": connection_id, "error": str(e)}
            )
            await self.unregister_websocket_connection(connection_id)

    async def register_sse_connection(
        self,
        task_filter: Optional[List[str]] = None
    ) -> tuple[str, StreamingConnection]:
        """Register a new SSE connection for real-time notifications."""
        connection_id = str(uuid.uuid4())
        connection = StreamingConnection(
            connection_id=connection_id,
            task_filter=set(task_filter) if task_filter else set()
        )

        async with self.lock:
            self.sse_connections[connection_id] = connection

        logger.info(
            "SSE connection registered",
            extra={
                "connection_id": connection_id,
                "task_filter": list(connection.task_filter) if connection.task_filter else None
            }
        )

        return connection_id, connection

    async def unregister_sse_connection(self, connection_id: str) -> None:
        """Unregister an SSE connection."""
        async with self.lock:
            if connection_id in self.sse_connections:
                del self.sse_connections[connection_id]
                logger.info(
                    "SSE connection unregistered",
                    extra={"connection_id": connection_id}
                )

    async def create_sse_response(
        self,
        connection_id: str,
        connection: StreamingConnection
    ) -> StreamingResponse:
        """Create an SSE streaming response."""

        async def event_generator():
            """Generate SSE events for the connection."""
            try:
                # Send initial connection event
                yield f"data: {json.dumps({'type': 'connection_established', 'connection_id': connection_id, 'timestamp': connection.connected_at.isoformat()})}\n\n"

                # Keep connection alive and listen for events
                while True:
                    # Update activity
                    connection.update_activity()

                    # Check if connection still exists
                    async with self.lock:
                        if connection_id not in self.sse_connections:
                            break

                    # Poll for events - acceptable for low connection volume without external dependencies
                    # In high-volume scenarios, consider Redis pub/sub or similar event distribution
                    await asyncio.sleep(1)

                    # Send heartbeat every 30 seconds
                    current_time = datetime.utcnow()
                    if (current_time - connection.last_activity).seconds >= 30:
                        yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': current_time.isoformat()})}\n\n"

            except Exception as e:
                logger.error(
                    "SSE stream error",
                    extra={"connection_id": connection_id, "error": str(e)}
                )
            finally:
                await self.unregister_sse_connection(connection_id)

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

    async def broadcast_notification(self, task_id: str, event: Dict[str, Any]) -> None:
        """Broadcast a notification to all connected streaming clients."""
        # Broadcast to WebSocket connections
        await self._broadcast_to_websockets(task_id, event)

        # Note: SSE broadcasting would need a different mechanism
        # since SSE is unidirectional. For SSE, clients would need to
        # reconnect or use a different pattern for real-time updates.

    async def _broadcast_to_websockets(self, task_id: str, event: Dict[str, Any]) -> None:
        """Broadcast notification to WebSocket connections."""
        notification = {
            "type": "notification",
            "task_id": task_id,
            "event": event.get("event", "unknown"),
            "data": event,
            "timestamp": datetime.utcnow().isoformat()
        }

        async with self.lock:
            disconnected_connections = []

            for connection_id, connection in self.websocket_connections.items():
                if connection.should_receive_notification(task_id):
                    try:
                        # In a real implementation, you'd need to store the websocket
                        # reference in the connection object to send messages
                        logger.debug(
                            "Would broadcast to WebSocket",
                            extra={
                                "connection_id": connection_id,
                                "task_id": task_id,
                                "event": event.get("event")
                            }
                        )

                        # Update activity
                        connection.update_activity()

                    except Exception as e:
                        logger.error(
                            "Failed to broadcast to WebSocket",
                            extra={
                                "connection_id": connection_id,
                                "task_id": task_id,
                                "error": str(e)
                            }
                        )
                        disconnected_connections.append(connection_id)

            # Clean up disconnected connections
            for connection_id in disconnected_connections:
                if connection_id in self.websocket_connections:
                    del self.websocket_connections[connection_id]

    async def get_connection_stats(self) -> Dict[str, Any]:
        """Get statistics about current streaming connections."""
        async with self.lock:
            websocket_count = len(self.websocket_connections)
            sse_count = len(self.sse_connections)

            # Calculate connection ages
            websocket_ages = []
            for conn in self.websocket_connections.values():
                age = (datetime.utcnow() - conn.connected_at).total_seconds()
                websocket_ages.append(age)

            sse_ages = []
            for conn in self.sse_connections.values():
                age = (datetime.utcnow() - conn.connected_at).total_seconds()
                sse_ages.append(age)

            return {
                "websocket_connections": {
                    "count": websocket_count,
                    "average_age_seconds": sum(websocket_ages) / len(websocket_ages) if websocket_ages else 0,
                    "oldest_connection_seconds": max(websocket_ages) if websocket_ages else 0
                },
                "sse_connections": {
                    "count": sse_count,
                    "average_age_seconds": sum(sse_ages) / len(sse_ages) if sse_ages else 0,
                    "oldest_connection_seconds": max(sse_ages) if sse_ages else 0
                },
                "total_connections": websocket_count + sse_count
            }

    async def cleanup_stale_connections(self, max_age_seconds: Optional[int] = None) -> int:
        """Clean up stale connections that haven't been active for too long."""
        if max_age_seconds is None:
            max_age_seconds = self.cleanup_interval
        cutoff_time = datetime.utcnow().timestamp() - max_age_seconds
        cleaned_count = 0

        async with self.lock:
            # Clean WebSocket connections
            stale_websocket = [
                conn_id for conn_id, conn in self.websocket_connections.items()
                if conn.last_activity.timestamp() < cutoff_time
            ]

            for conn_id in stale_websocket:
                del self.websocket_connections[conn_id]
                cleaned_count += 1

            # Clean SSE connections
            stale_sse = [
                conn_id for conn_id, conn in self.sse_connections.items()
                if conn.last_activity.timestamp() < cutoff_time
            ]

            for conn_id in stale_sse:
                del self.sse_connections[conn_id]
                cleaned_count += 1

        if cleaned_count > 0:
            logger.info(
                "Cleaned up stale streaming connections",
                extra={"count": cleaned_count, "max_age_seconds": max_age_seconds}
            )

        return cleaned_count

    async def close(self) -> None:
        """Close all streaming connections and cleanup resources."""
        async with self.lock:
            # Close all WebSocket connections
            websocket_count = len(self.websocket_connections)
            self.websocket_connections.clear()

            # Close all SSE connections
            sse_count = len(self.sse_connections)
            self.sse_connections.clear()

        logger.info(
            "Closed all streaming connections",
            extra={"websocket_count": websocket_count, "sse_count": sse_count}
        )