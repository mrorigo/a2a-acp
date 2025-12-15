"""
Audit and Compliance System for Tool Execution

Comprehensive audit logging for all tool executions, security events, and access attempts.
Provides detailed logging for compliance, monitoring, and forensic analysis.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
import uuid
from dataclasses import dataclass, asdict, replace
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, cast

from .tool_config import BashTool
from .sandbox import ExecutionContext

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of auditable events."""

    TOOL_EXECUTION_STARTED = "tool_execution_started"
    TOOL_EXECUTION_COMPLETED = "tool_execution_completed"
    TOOL_EXECUTION_FAILED = "tool_execution_failed"
    TOOL_CONFIRMATION_REQUESTED = "tool_confirmation_requested"
    TOOL_CONFIRMATION_APPROVED = "tool_confirmation_approved"
    TOOL_CONFIRMATION_DENIED = "tool_confirmation_denied"
    SECURITY_VIOLATION = "security_violation"
    PARAMETER_VALIDATION_FAILED = "parameter_validation_failed"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    CIRCUIT_BREAKER_OPENED = "circuit_breaker_opened"
    CIRCUIT_BREAKER_CLOSED = "circuit_breaker_closed"
    OAUTH_LOGIN_STARTED = "oauth_login_started"
    OAUTH_LOGIN_SUCCESS = "oauth_login_success"
    OAUTH_LOGIN_FAILED = "oauth_login_failed"
    OAUTH_REFRESH_SUCCESS = "oauth_refresh_success"
    OAUTH_TOKEN_REMOVED = "oauth_token_removed"


@dataclass
class AuditEvent:
    """Represents an auditable event."""

    id: str
    timestamp: datetime
    event_type: AuditEventType
    user_id: str
    session_id: str
    task_id: str
    tool_id: Optional[str] = None
    severity: str = "info"  # info, warning, error, critical
    details: Optional[Dict[str, Any]] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["event_type"] = self.event_type.value
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AuditEvent:
        """Create from dictionary."""
        data = dict(data)
        data["event_type"] = AuditEventType(data["event_type"])
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        data.pop("created_at", None)
        return cls(**data)


class AuditDatabase:
    """SQLite database for audit events."""

    def __init__(self, db_path: str = "audit.db"):
        """Initialize audit database."""
        self.db_path = db_path
        self._local = threading.local()
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "connection"):
            connection = sqlite3.connect(self.db_path, check_same_thread=False)
            connection.execute("PRAGMA journal_mode=WAL")
            self._local.connection = connection
        return cast(sqlite3.Connection, self._local.connection)

    def _init_database(self) -> None:
        """Initialize audit database schema."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_events (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    tool_id TEXT,
                    severity TEXT NOT NULL,
                    details TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Create indexes for efficient querying
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_audit_events_timestamp
                ON audit_events(timestamp)
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_audit_events_user
                ON audit_events(user_id)
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_audit_events_tool
                ON audit_events(tool_id)
            """
            )

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_audit_events_type
                ON audit_events(event_type)
            """
            )

            conn.commit()

    def log_event(self, event: AuditEvent) -> None:
        """Log an audit event to the database."""
        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO audit_events
                    (id, timestamp, event_type, user_id, session_id, task_id, tool_id,
                     severity, details, ip_address, user_agent)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        event.id,
                        event.timestamp.isoformat(),
                        event.event_type.value,
                        event.user_id,
                        event.session_id,
                        event.task_id,
                        event.tool_id,
                        event.severity,
                        json.dumps(event.details) if event.details else None,
                        event.ip_address,
                        event.user_agent,
                    ),
                )
                conn.commit()
        except Exception as e:
            logger.error(
                f"Failed to log audit event: {e}", extra={"event_id": event.id}
            )

    def get_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_type: Optional[AuditEventType] = None,
        user_id: Optional[str] = None,
        tool_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[AuditEvent]:
        """Retrieve audit events with optional filtering."""
        try:
            with self._get_connection() as conn:
                conn.row_factory = sqlite3.Row
                query = "SELECT * FROM audit_events WHERE 1=1"
                params = []

                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time.isoformat())

                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time.isoformat())

                if event_type:
                    query += " AND event_type = ?"
                    params.append(event_type.value)

                if user_id:
                    query += " AND user_id = ?"
                    params.append(user_id)

                if tool_id:
                    query += " AND tool_id = ?"
                    params.append(tool_id)

                query += " ORDER BY timestamp DESC"

                if limit:
                    query += " LIMIT ?"
                    params.append(str(limit))

                cursor = conn.execute(query, params)
                rows = cursor.fetchall()

                events = []
                for row in rows:
                    event_dict = dict(row)
                    event_dict["details"] = (
                        json.loads(event_dict["details"])
                        if event_dict["details"]
                        else None
                    )
                    events.append(AuditEvent.from_dict(event_dict))

                return events

        except Exception as e:
            logger.error(f"Failed to retrieve audit events: {e}")
            return []

    def get_event_stats(self) -> Dict[str, Any]:
        """Get audit event statistics."""
        try:
            with self._get_connection() as conn:
                # Get event counts by type
                cursor = conn.execute(
                    """
                    SELECT event_type, severity, COUNT(*) as count
                    FROM audit_events
                    WHERE timestamp >= datetime('now', '-24 hours')
                    GROUP BY event_type, severity
                    ORDER BY event_type, severity
                """
                )

                event_stats: dict[str, dict[str, int]] = {}
                for row in cursor.fetchall():
                    event_type, severity, count = row
                    if event_type not in event_stats:
                        event_stats[event_type] = {}
                    event_stats[event_type][severity] = count

                # Get tool usage stats
                cursor = conn.execute(
                    """
                    SELECT tool_id, COUNT(*) as executions,
                           COUNT(CASE WHEN event_type = 'tool_execution_failed' THEN 1 END) as failures
                    FROM audit_events
                    WHERE event_type IN ('tool_execution_started', 'tool_execution_completed', 'tool_execution_failed')
                    AND timestamp >= datetime('now', '-24 hours')
                    AND tool_id IS NOT NULL
                    GROUP BY tool_id
                    ORDER BY executions DESC
                """
                )

                tool_stats: list[Dict[str, Any]] = []
                for row in cursor.fetchall():
                    tool_stats.append(
                        {
                            "tool_id": row[0],
                            "executions": row[1],
                            "failures": row[2],
                            "success_rate": (
                                (row[1] - row[2]) / row[1] * 100 if row[1] > 0 else 0
                            ),
                        }
                    )

                return {
                    "event_stats": event_stats,
                    "tool_stats": tool_stats,
                    "total_events_24h": sum(
                        count
                        for stats in event_stats.values()
                        for count in stats.values()
                    ),
                    "most_used_tools": tool_stats[:10],  # Top 10
                }

        except Exception as e:
            logger.error(f"Failed to get audit statistics: {e}")
            return {}


class AuditLogger:
    """Main audit logging interface."""

    def __init__(
        self, database: Optional[AuditDatabase] = None, db_path: str = "audit.db"
    ):
        """Initialize audit logger.

        Args:
            database: Optional database instance. If None, creates new one.
            db_path: Database path if creating new database.
        """
        self.database = database or AuditDatabase(db_path)
        self._lock: Optional[asyncio.Lock] = None

    async def log_tool_execution(
        self,
        event_type: AuditEventType,
        context: ExecutionContext,
        tool: Optional[BashTool] = None,
        **additional_details,
    ) -> None:
        """Log a tool execution event.

        Args:
            event_type: Type of audit event
            context: Execution context
            tool: Tool being executed (if applicable)
            **additional_details: Additional event details
        """
        event_timestamp = datetime.now()
        event = AuditEvent(
            id=self._generate_event_id(context.task_id, event_timestamp),
            timestamp=event_timestamp,
            event_type=event_type,
            user_id=context.user_id,
            session_id=context.session_id,
            task_id=context.task_id,
            tool_id=tool.id if tool else None,
            severity=self._determine_severity(event_type),
            details=additional_details,
        )

        log_extra = {
            "user_id": context.user_id,
            "session_id": context.session_id,
            "task_id": context.task_id,
            "tool_id": tool.id if tool else None,
            **additional_details,
        }
        await self._persist_and_log(event, log_extra)

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def _persist_and_log(
        self, event: AuditEvent, log_extra: Dict[str, Any]
    ) -> AuditEvent:
        """Persist the event and emit a structured log record."""
        lock = self._get_lock()
        async with lock:
            loop = asyncio.get_event_loop()
            persisted_event = await loop.run_in_executor(
                None, self._persist_event_with_retry, event
            )

        payload = {"audit_event_type": event.event_type.value, **log_extra}
        payload["severity"] = persisted_event.severity
        logger.info(f"Audit Event: {event.event_type.value}", extra=payload)
        return persisted_event

    def _generate_event_id(
        self, task_id: Optional[str], timestamp: Optional[datetime] = None
    ) -> str:
        """Generate a globally unique audit event identifier."""
        reference_time = timestamp or datetime.now()
        timestamp_part = reference_time.strftime("%Y%m%d_%H%M%S_%f")
        task_fragment = (task_id or "task")[:8]
        safe_task_fragment = "".join(
            ch if ch.isalnum() else "_" for ch in task_fragment
        )
        random_fragment = uuid.uuid4().hex[:8]
        return f"audit_{timestamp_part}_{safe_task_fragment}_{random_fragment}"

    def _persist_event_with_retry(
        self, event: AuditEvent, attempts: int = 3
    ) -> AuditEvent:
        """Persist an audit event, regenerating IDs if collisions are detected."""
        current_event = event
        for attempt in range(attempts):
            try:
                self.database.log_event(current_event)
                return current_event
            except sqlite3.IntegrityError:
                new_id = self._generate_event_id(current_event.task_id)
                current_event = replace(current_event, id=new_id)
        # Final attempt without swallowing the exception
        self.database.log_event(current_event)
        return current_event

    def _determine_severity(self, event_type: AuditEventType) -> str:
        """Determine severity level for an event type."""
        severity_mapping = {
            AuditEventType.TOOL_EXECUTION_STARTED: "info",
            AuditEventType.TOOL_EXECUTION_COMPLETED: "info",
            AuditEventType.TOOL_EXECUTION_FAILED: "warning",
            AuditEventType.TOOL_CONFIRMATION_REQUESTED: "info",
            AuditEventType.TOOL_CONFIRMATION_APPROVED: "info",
            AuditEventType.TOOL_CONFIRMATION_DENIED: "warning",
            AuditEventType.SECURITY_VIOLATION: "error",
            AuditEventType.PARAMETER_VALIDATION_FAILED: "warning",
            AuditEventType.CACHE_HIT: "info",
            AuditEventType.CACHE_MISS: "info",
            AuditEventType.RATE_LIMIT_EXCEEDED: "warning",
            AuditEventType.CIRCUIT_BREAKER_OPENED: "warning",
            AuditEventType.CIRCUIT_BREAKER_CLOSED: "info",
            AuditEventType.OAUTH_LOGIN_STARTED: "info",
            AuditEventType.OAUTH_LOGIN_SUCCESS: "info",
            AuditEventType.OAUTH_LOGIN_FAILED: "warning",
            AuditEventType.OAUTH_REFRESH_SUCCESS: "info",
            AuditEventType.OAUTH_TOKEN_REMOVED: "info",
        }

        return severity_mapping.get(event_type, "info")

    async def log_event(
        self,
        event_type: AuditEventType,
        user_id: str,
        session_id: str,
        task_id: str,
        tool_id: Optional[str] = None,
        severity: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        """Log a general-purpose audit event (not tied to a tool execution)."""
        event_timestamp = datetime.now()
        event_details = dict(details or {})
        event = AuditEvent(
            id=self._generate_event_id(task_id, event_timestamp),
            timestamp=event_timestamp,
            event_type=event_type,
            user_id=user_id,
            session_id=session_id,
            task_id=task_id,
            tool_id=tool_id,
            severity=severity or self._determine_severity(event_type),
            details=event_details,
        )

        log_extra = {
            "user_id": user_id,
            "session_id": session_id,
            "task_id": task_id,
            "tool_id": tool_id,
            **event_details,
        }
        return await self._persist_and_log(event, log_extra)

    async def log_security_event(
        self, context: ExecutionContext, violation_type: str, details: Dict[str, Any]
    ) -> None:
        """Log a security-related event.

        Args:
            context: Execution context
            violation_type: Type of security violation
            details: Violation details
        """
        await self.log_tool_execution(
            AuditEventType.SECURITY_VIOLATION,
            context,
            severity="error",
            violation_type=violation_type,
            **details,
        )

    async def get_audit_report(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        user_id: Optional[str] = None,
        tool_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate an audit report for the specified time period.

        Args:
            start_time: Start of reporting period
            end_time: End of reporting period
            user_id: Filter by user ID
            tool_id: Filter by tool ID

        Returns:
            Audit report with statistics and events
        """
        # Get events for the period
        events = self.database.get_events(
            start_time=start_time,
            end_time=end_time,
            user_id=user_id,
            tool_id=tool_id,
            limit=1000,  # Reasonable limit for reports
        )

        # Get statistics
        stats = self.database.get_event_stats()

        # Generate summary
        events_by_type: dict[str, int] = {}
        events_by_severity: dict[str, int] = {}
        security_incidents = 0

        # Count by type and severity
        for event in events:
            event_type = event.event_type.value
            events_by_type[event_type] = events_by_type.get(event_type, 0) + 1
            events_by_severity[event.severity] = (
                events_by_severity.get(event.severity, 0) + 1
            )

            if event.severity in ["error", "critical"]:
                security_incidents += 1

        # Get top tools and users
        tool_usage: dict[str, int] = {}
        user_activity: dict[str, int] = {}

        for event in events:
            if event.tool_id:
                tool_usage[event.tool_id] = tool_usage.get(event.tool_id, 0) + 1
            if event.user_id:
                user_activity[event.user_id] = user_activity.get(event.user_id, 0) + 1

        top_tools = sorted(tool_usage.items(), key=lambda x: x[1], reverse=True)[:10]
        top_users = sorted(user_activity.items(), key=lambda x: x[1], reverse=True)[:10]

        summary: Dict[str, Any] = {
            "report_period": {
                "start": start_time.isoformat() if start_time else None,
                "end": end_time.isoformat() if end_time else None,
            },
            "total_events": len(events),
            "events_by_type": events_by_type,
            "events_by_severity": events_by_severity,
            "top_tools": top_tools,
            "top_users": top_users,
            "security_incidents": security_incidents,
        }

        return {
            "summary": summary,
            "statistics": stats,
            "recent_events": [
                event.to_dict() for event in events[:100]
            ],  # Last 100 events
        }


# Global audit logger instance
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Get the global audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


async def log_tool_execution(
    event_type: AuditEventType,
    context: ExecutionContext,
    tool: Optional[BashTool] = None,
    **details,
) -> None:
    """Convenience function to log tool execution events."""
    logger = get_audit_logger()
    await logger.log_tool_execution(event_type, context, tool, **details)


async def log_security_event(
    context: ExecutionContext, violation_type: str, **details
) -> None:
    """Convenience function to log security events."""
    logger = get_audit_logger()
    await logger.log_security_event(context, violation_type, details)


async def log_oauth_event(
    event_type: AuditEventType,
    agent_id: str,
    details: Optional[Dict[str, Any]] = None,
) -> AuditEvent:
    """Convenience function to log OAuth-specific events."""
    logger = get_audit_logger()
    event_details = dict(details or {})
    event_details["agent_id"] = agent_id
    return await logger.log_event(
        event_type=event_type,
        user_id="oauth",
        session_id=agent_id,
        task_id="oauth_flow",
        details=event_details,
    )


def generate_audit_report(
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    user_id: Optional[str] = None,
    tool_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate an audit report synchronously."""
    logger = get_audit_logger()

    # Run async function in new event loop if needed
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Create new event loop for sync context
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(
                    logger.get_audit_report(start_time, end_time, user_id, tool_id)
                )
            finally:
                new_loop.close()
        else:
            return loop.run_until_complete(
                logger.get_audit_report(start_time, end_time, user_id, tool_id)
            )
    except RuntimeError:
        # No event loop, create one
        return asyncio.run(
            logger.get_audit_report(start_time, end_time, user_id, tool_id)
        )
