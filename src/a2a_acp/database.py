"""
Database layer for A2A-ACP context and task persistence.

This module provides SQLite-based storage for A2A contexts, tasks, and message history,
enabling stateful agent conversations across multiple tasks.
"""

import json
import logging
import sqlite3
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from a2a_acp.a2a.models import Message, Task

logger = logging.getLogger(__name__)


def serialize_dataclass_with_dates(obj: Any) -> Dict[str, Any]:
    """Convert dataclass to dictionary with proper datetime and metadata handling."""
    data = asdict(obj)
    # Handle datetime fields
    for field_name in ["created_at", "updated_at"]:
        if field_name in data and isinstance(data[field_name], datetime):
            data[field_name] = data[field_name].isoformat()
    # Handle metadata field
    if "metadata" in data and data["metadata"] is not None:
        data["metadata"] = json.dumps(data["metadata"])
    return data


def deserialize_dataclass_with_dates(cls: Any, data: Dict[str, Any]) -> Any:
    """Create dataclass from dictionary with proper datetime and metadata handling."""
    # Handle datetime fields
    for field_name in ["created_at", "updated_at"]:
        if field_name in data and data[field_name]:
            data[field_name] = datetime.fromisoformat(data[field_name])
    # Handle metadata field
    if "metadata" in data and data["metadata"]:
        data["metadata"] = json.loads(data["metadata"])
    return cls(**data)


@dataclass
class A2AContext:
    """Represents an A2A context with ZedACP mapping."""

    context_id: str
    agent_name: str
    zed_session_id: str
    working_directory: str
    created_at: datetime
    updated_at: datetime
    is_active: bool = True
    last_task_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return serialize_dataclass_with_dates(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "A2AContext":
        """Create from dictionary."""
        return deserialize_dataclass_with_dates(cls, data)


@dataclass
class ACPSession:
    """Represents an ACP session with ZedACP mapping."""

    acp_session_id: str
    agent_name: str
    zed_session_id: str
    working_directory: str
    created_at: datetime
    updated_at: datetime
    is_active: bool = True
    last_run_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return serialize_dataclass_with_dates(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ACPSession":
        """Create from dictionary."""
        return deserialize_dataclass_with_dates(cls, data)


@dataclass
class SessionHistory:
    """Represents a message in session history."""

    id: Optional[int]
    acp_session_id: str
    run_id: str
    message_role: str  # 'user' | 'assistant'
    message_data: Dict[str, Any]  # Full IBM ACP Message
    created_at: datetime
    sequence_number: int
    zed_message_data: Optional[Dict[str, Any]] = None  # ZedACP format

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        data["message_data"] = json.dumps(self.message_data)
        data["zed_message_data"] = (
            json.dumps(self.zed_message_data) if self.zed_message_data else None
        )
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionHistory":
        """Create from dictionary."""
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["message_data"] = json.loads(data["message_data"])
        data["zed_message_data"] = (
            json.loads(data["zed_message_data"]) if data["zed_message_data"] else None
        )
        return cls(**data)


class SessionDatabase:
    """
    SQLite database for ACP session persistence.

    Uses WAL mode for better concurrency and creates tables as needed.
    Thread-safe through connection pooling.
    """

    def __init__(self, db_path: str = "a2a_acp.db"):
        """Initialize database connection."""
        self.db_path = db_path
        self._local = threading.local()
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "connection"):
            self._local.connection = sqlite3.connect(
                self.db_path, check_same_thread=False
            )
            # Enable WAL mode for better concurrency
            self._local.connection.execute("PRAGMA journal_mode=WAL")
            self._local.connection.execute("PRAGMA synchronous=NORMAL")
            self._local.connection.execute("PRAGMA cache_size=10000")
            self._local.connection.execute("PRAGMA temp_store=memory")

        return self._local.connection

    def _init_database(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS a2a_contexts (
                    context_id TEXT PRIMARY KEY,
                    agent_name TEXT NOT NULL,
                    zed_session_id TEXT NOT NULL,
                    working_directory TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    is_active BOOLEAN DEFAULT 1,
                    last_task_id TEXT,
                    metadata TEXT
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS a2a_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    context_id TEXT NOT NULL REFERENCES a2a_contexts(context_id),
                    task_id TEXT NOT NULL,
                    message_role TEXT NOT NULL,
                    message_data TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    sequence_number INTEGER,
                    zed_message_data TEXT
                )
            """
            )

            # Push notification configuration table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS push_notification_configs (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    url TEXT NOT NULL,
                    token TEXT,
                    authentication_schemes TEXT,
                    credentials TEXT,
                    enabled_events TEXT,
                    disabled_events TEXT,
                    quiet_hours_start TEXT,
                    quiet_hours_end TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Push notification delivery tracking table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS notification_deliveries (
                    id TEXT PRIMARY KEY,
                    config_id TEXT NOT NULL REFERENCES push_notification_configs(id),
                    task_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    delivery_status TEXT NOT NULL,
                    response_code INTEGER,
                    response_body TEXT,
                    attempted_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    delivered_at TEXT
                )
            """
            )

            # Create indexes for performance
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_a2a_contexts_agent_active
                ON a2a_contexts(agent_name, is_active)
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_a2a_messages_context
                ON a2a_messages(context_id, created_at)
            """
            )

            # Push notification indexes
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_push_configs_task
                ON push_notification_configs(task_id)
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_notification_deliveries_config
                ON notification_deliveries(config_id, attempted_at)
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_notification_deliveries_task_event
                ON notification_deliveries(task_id, event_type, delivery_status)
            """
            )

            conn.commit()

    async def create_a2a_context(
        self,
        context_id: str,
        agent: str,
        cwd: str,
        zed_session_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> A2AContext:
        """Create a new A2A context record."""
        now = datetime.now(timezone.utc)
        context = A2AContext(
            context_id=context_id,
            agent_name=agent,
            zed_session_id=zed_session_id,
            working_directory=cwd,
            created_at=now,
            updated_at=now,
            metadata=metadata,
        )

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO a2a_contexts
                (context_id, agent_name, zed_session_id, working_directory,
                 created_at, updated_at, is_active, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    context.context_id,
                    context.agent_name,
                    context.zed_session_id,
                    context.working_directory,
                    context.created_at.isoformat(),
                    context.updated_at.isoformat(),
                    context.is_active,
                    json.dumps(context.metadata) if context.metadata else None,
                ),
            )
            conn.commit()

        return context

    async def get_a2a_context(self, context_id: str) -> Optional[A2AContext]:
        """Retrieve an A2A context by ID."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM a2a_contexts WHERE context_id = ?
            """,
                (context_id,),
            )

            row = cursor.fetchone()
            if row:
                return A2AContext.from_dict(dict(row))
            return None

    async def update_zed_session_id(
        self, acp_session_id: str, zed_session_id: str
    ) -> None:
        """Update the ZedACP session ID for an ACP session."""
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE acp_sessions
                SET zed_session_id = ?, updated_at = ?
                WHERE acp_session_id = ?
            """,
                (
                    zed_session_id,
                    datetime.now(timezone.utc).isoformat(),
                    acp_session_id,
                ),
            )
            conn.commit()

    async def append_message_history(
        self,
        acp_session_id: str,
        run_id: str,
        message: Message,
        sequence_number: int,
        zed_message: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append a message to session history."""
        # Determine role from message (this is a simplified approach)
        # In practice, the role should be determined by the context of the run
        # For now, we'll use a simple heuristic based on sequence number
        message_role = "user" if sequence_number == 0 else "assistant"

        history_entry = SessionHistory(
            id=None,
            acp_session_id=acp_session_id,
            run_id=run_id,
            message_role=message_role,
            message_data={
                "role": message.role,
                "parts": [part.model_dump() for part in message.parts],
            },
            created_at=datetime.now(timezone.utc),
            sequence_number=sequence_number,
            zed_message_data=zed_message,
        )

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO session_history
                (acp_session_id, run_id, message_role, message_data, created_at,
                 sequence_number, zed_message_data)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    history_entry.acp_session_id,
                    history_entry.run_id,
                    history_entry.message_role,
                    json.dumps(history_entry.message_data),
                    history_entry.created_at.isoformat(),
                    history_entry.sequence_number,
                    (
                        json.dumps(history_entry.zed_message_data)
                        if history_entry.zed_message_data
                        else None
                    ),
                ),
            )
            conn.commit()

    async def get_session_history(
        self, acp_session_id: str, limit: Optional[int] = None
    ) -> List[SessionHistory]:
        """Retrieve message history for a session."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            query = """
                SELECT * FROM session_history
                WHERE acp_session_id = ?
                ORDER BY sequence_number ASC
            """
            params = [acp_session_id]

            if limit:
                query += " LIMIT ?"
                params.append(str(limit))

            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

            return [SessionHistory.from_dict(dict(row)) for row in rows]

    async def list_a2a_contexts(
        self, agent_name: Optional[str] = None, active_only: bool = True
    ) -> List[A2AContext]:
        """List ACP sessions with optional filtering."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            query = "SELECT * FROM acp_sessions WHERE 1=1"
            params = []

            if agent_name:
                query += " AND agent_name = ?"
                params.append(agent_name)

            if active_only:
                query += " AND is_active = 1"

            query += " ORDER BY updated_at DESC"

            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

            return [A2AContext.from_dict(dict(row)) for row in rows]

    async def delete_a2a_context(self, context_id: str) -> bool:
        """Delete an A2A context and all its messages."""
        with self._get_connection() as conn:
            # Delete messages first (foreign key constraint)
            conn.execute("DELETE FROM a2a_messages WHERE context_id = ?", (context_id,))

            # Delete the context
            cursor = conn.execute(
                "DELETE FROM a2a_contexts WHERE context_id = ?", (context_id,)
            )
            conn.commit()

            return cursor.rowcount > 0

    async def cleanup_inactive_sessions(self, days_old: int = 30) -> int:
        """Clean up old inactive sessions. Returns number of sessions deleted."""
        cutoff_date = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                DELETE FROM acp_sessions
                WHERE is_active = 0 AND updated_at < datetime(?, '-' || ? || ' days')
            """,
                (cutoff_date, days_old),
            )
            deleted_count = cursor.rowcount
            conn.commit()
            return deleted_count

    async def store_push_notification_config(
        self,
        config_id: str,
        task_id: str,
        url: str,
        token: Optional[str] = None,
        authentication_schemes: Optional[Dict[str, Any]] = None,
        credentials: Optional[str] = None,
        enabled_events: Optional[List[str]] = None,
        disabled_events: Optional[List[str]] = None,
        quiet_hours_start: Optional[str] = None,
        quiet_hours_end: Optional[str] = None,
    ) -> None:
        """Store a push notification configuration."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO push_notification_configs
                (id, task_id, url, token, authentication_schemes, credentials,
                 enabled_events, disabled_events, quiet_hours_start, quiet_hours_end, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
                (
                    config_id,
                    task_id,
                    url,
                    token,
                    (
                        json.dumps(authentication_schemes)
                        if authentication_schemes
                        else None
                    ),
                    credentials,
                    json.dumps(enabled_events) if enabled_events else None,
                    json.dumps(disabled_events) if disabled_events else None,
                    quiet_hours_start,
                    quiet_hours_end,
                ),
            )
            conn.commit()

    async def get_push_notification_config(
        self, task_id: str, config_id: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a push notification configuration."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM push_notification_configs
                WHERE task_id = ? AND id = ?
            """,
                (task_id, config_id),
            )

            row = cursor.fetchone()
            if row:
                config = dict(row)
                config["authentication_schemes"] = (
                    json.loads(config["authentication_schemes"])
                    if config["authentication_schemes"]
                    else None
                )
                config["enabled_events"] = (
                    json.loads(config["enabled_events"])
                    if config["enabled_events"]
                    else None
                )
                config["disabled_events"] = (
                    json.loads(config["disabled_events"])
                    if config["disabled_events"]
                    else None
                )
                return config
            return None

    async def list_push_notification_configs(
        self, task_id: str
    ) -> List[Dict[str, Any]]:
        """List all push notification configurations for a task."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM push_notification_configs
                WHERE task_id = ?
                ORDER BY created_at DESC
            """,
                (task_id,),
            )

            rows = cursor.fetchall()
            configs = []
            for row in rows:
                config = dict(row)
                config["authentication_schemes"] = (
                    json.loads(config["authentication_schemes"])
                    if config["authentication_schemes"]
                    else None
                )
                config["enabled_events"] = (
                    json.loads(config["enabled_events"])
                    if config["enabled_events"]
                    else None
                )
                config["disabled_events"] = (
                    json.loads(config["disabled_events"])
                    if config["disabled_events"]
                    else None
                )
                configs.append(config)

            return configs

    async def delete_push_notification_config(
        self, task_id: str, config_id: str
    ) -> bool:
        """Delete a push notification configuration."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                DELETE FROM push_notification_configs
                WHERE task_id = ? AND id = ?
            """,
                (task_id, config_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    async def track_notification_delivery(
        self,
        delivery_id: str,
        config_id: str,
        task_id: str,
        event_type: str,
        delivery_status: str,
        response_code: Optional[int] = None,
        response_body: Optional[str] = None,
        delivered_at: Optional[str] = None,
    ) -> None:
        """Track a notification delivery attempt."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO notification_deliveries
                (id, config_id, task_id, event_type, delivery_status, response_code, response_body, delivered_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    delivery_id,
                    config_id,
                    task_id,
                    event_type,
                    delivery_status,
                    response_code,
                    response_body,
                    delivered_at,
                ),
            )
            conn.commit()

    async def get_notification_delivery_history(
        self,
        task_id: Optional[str] = None,
        config_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get notification delivery history with optional filtering."""
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            query = "SELECT * FROM notification_deliveries WHERE 1=1"
            params = []

            if task_id:
                query += " AND task_id = ?"
                params.append(task_id)

            if config_id:
                query += " AND config_id = ?"
                params.append(config_id)

            query += " ORDER BY attempted_at DESC"

            if limit:
                query += " LIMIT ?"
                params.append(str(limit))

            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

            deliveries = []
            for row in rows:
                delivery = dict(row)
                deliveries.append(delivery)

            return deliveries

    async def cleanup_expired_notification_configs(self) -> int:
        """Clean up old notification configurations. Returns number of configs deleted."""
        # For now, implement a simple cleanup strategy
        # This could be extended with more sophisticated lifecycle management
        cutoff_date = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                DELETE FROM push_notification_configs
                WHERE updated_at < datetime(?, '-24 hours')
            """,
                (cutoff_date,),
            )
            deleted_count = cursor.rowcount
            conn.commit()
            return deleted_count

    async def store_task(self, task: Task) -> None:
        """Persist task metadata; currently a placeholder for future persistence."""
        try:
            logger.debug("Storing task metadata", extra={"task_id": task.id})
        except Exception as exc:
            logger.warning("Failed to log task persistence", extra={"error": str(exc)})

    def close(self) -> None:
        """Close all database connections."""
        if hasattr(self._local, "connection"):
            self._local.connection.close()
