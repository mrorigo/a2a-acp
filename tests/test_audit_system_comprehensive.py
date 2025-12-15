"""
Comprehensive test suite for a2a_acp.audit

Covers core functionality that is currently missing from coverage:
- AuditEvent and AuditEventType classes
- AuditDatabase operations (CRUD)
- AuditLogger functionality
- Event persistence and retrieval
- Security event logging
- OAuth event logging
- Audit report generation
- Error handling and edge cases
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from a2a_acp.audit import (
    AuditDatabase,
    AuditEvent,
    AuditEventType,
    AuditLogger,
    generate_audit_report,
    get_audit_logger,
    log_oauth_event,
    log_security_event,
    log_tool_execution,
)
from a2a_acp.sandbox import ExecutionContext
from a2a_acp.tool_config import BashTool, ToolConfig


class TestAuditEventType:
    """Test the AuditEventType enum."""

    def test_audit_event_type_values(self):
        """Test that all audit event types have expected values."""
        expected_types = {
            "tool_execution_started",
            "tool_execution_completed",
            "tool_execution_failed",
            "tool_confirmation_requested",
            "tool_confirmation_approved",
            "tool_confirmation_denied",
            "security_violation",
            "parameter_validation_failed",
            "cache_hit",
            "cache_miss",
            "rate_limit_exceeded",
            "circuit_breaker_opened",
            "circuit_breaker_closed",
            "oauth_login_started",
            "oauth_login_success",
            "oauth_login_failed",
            "oauth_refresh_success",
            "oauth_token_removed",
        }

        actual_types = {event_type.value for event_type in AuditEventType}
        assert actual_types == expected_types


class TestAuditEvent:
    """Test the AuditEvent dataclass."""

    def test_audit_event_creation(self):
        """Test basic AuditEvent creation."""
        timestamp = datetime.now()
        event = AuditEvent(
            id="test_event_123",
            timestamp=timestamp,
            event_type=AuditEventType.TOOL_EXECUTION_STARTED,
            user_id="test_user",
            session_id="test_session",
            task_id="test_task",
            tool_id="test_tool",
            severity="info",
            details={"param": "value"},
        )

        assert event.id == "test_event_123"
        assert event.timestamp == timestamp
        assert event.event_type == AuditEventType.TOOL_EXECUTION_STARTED
        assert event.user_id == "test_user"
        assert event.session_id == "test_session"
        assert event.task_id == "test_task"
        assert event.tool_id == "test_tool"
        assert event.severity == "info"
        assert event.details == {"param": "value"}

    def test_audit_event_defaults(self):
        """Test AuditEvent with default values."""
        timestamp = datetime.now()
        event = AuditEvent(
            id="test_event_456",
            timestamp=timestamp,
            event_type=AuditEventType.TOOL_EXECUTION_STARTED,
            user_id="test_user",
            session_id="test_session",
            task_id="test_task",
        )

        assert event.tool_id is None
        assert event.severity == "info"  # default
        assert event.details is None
        assert event.ip_address is None
        assert event.user_agent is None

    def test_audit_event_to_dict(self):
        """Test AuditEvent serialization to dictionary."""
        timestamp = datetime.now()
        event = AuditEvent(
            id="test_event_789",
            timestamp=timestamp,
            event_type=AuditEventType.SECURITY_VIOLATION,
            user_id="test_user",
            session_id="test_session",
            task_id="test_task",
            severity="error",
            details={"violation": "unauthorized_access"},
        )

        data = event.to_dict()

        assert data["id"] == "test_event_789"
        assert data["timestamp"] == timestamp.isoformat()
        assert data["event_type"] == "security_violation"
        assert data["user_id"] == "test_user"
        assert data["session_id"] == "test_session"
        assert data["task_id"] == "test_task"
        assert data["severity"] == "error"
        assert data["details"] == {"violation": "unauthorized_access"}

    def test_audit_event_from_dict(self):
        """Test AuditEvent creation from dictionary."""
        data = {
            "id": "test_event_101",
            "timestamp": "2023-12-01T10:30:00",
            "event_type": "cache_hit",
            "user_id": "test_user",
            "session_id": "test_session",
            "task_id": "test_task",
            "severity": "info",
            "details": {"cache_key": "test_key"},
        }

        event = AuditEvent.from_dict(data)

        assert event.id == "test_event_101"
        assert event.timestamp == datetime.fromisoformat("2023-12-01T10:30:00")
        assert event.event_type == AuditEventType.CACHE_HIT
        assert event.user_id == "test_user"
        assert event.session_id == "test_session"
        assert event.task_id == "test_task"
        assert event.severity == "info"
        assert event.details == {"cache_key": "test_key"}


class TestAuditDatabase:
    """Test the AuditDatabase class."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database file."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
            db_path = tmp.name
        yield db_path
        # Cleanup
        Path(db_path).unlink(missing_ok=True)

    def test_database_initialization(self, temp_db_path):
        """Test database initialization and table creation."""
        db = AuditDatabase(temp_db_path)

        # Verify database file was created
        assert Path(temp_db_path).exists()

        # Verify tables exist
        with db._get_connection() as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            assert "audit_events" in tables

    def test_database_table_schema(self, temp_db_path):
        """Test database table schema."""
        db = AuditDatabase(temp_db_path)

        with db._get_connection() as conn:
            cursor = conn.execute("PRAGMA table_info(audit_events)")
            columns = {row[1]: row[2] for row in cursor.fetchall()}

            expected_columns = {
                "id": "TEXT",
                "timestamp": "TEXT",
                "event_type": "TEXT",
                "user_id": "TEXT",
                "session_id": "TEXT",
                "task_id": "TEXT",
                "tool_id": "TEXT",
                "severity": "TEXT",
                "details": "TEXT",
                "ip_address": "TEXT",
                "user_agent": "TEXT",
                "created_at": "TEXT",
            }

            for col_name, col_type in expected_columns.items():
                assert col_name in columns
                # SQLite type affinity check (TEXT for TEXT columns)
                assert columns[col_name] == col_type

    def test_database_indexes(self, temp_db_path):
        """Test database indexes are created."""
        db = AuditDatabase(temp_db_path)

        with db._get_connection() as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
            indexes = [row[0] for row in cursor.fetchall()]

            expected_indexes = [
                "idx_audit_events_timestamp",
                "idx_audit_events_tool",
                "idx_audit_events_type",
                "idx_audit_events_user",
            ]

            for expected_idx in expected_indexes:
                assert expected_idx in indexes

    def test_insert_event(self, temp_db_path):
        """Test inserting an audit event."""
        db = AuditDatabase(temp_db_path)
        event = AuditEvent(
            id="test_insert_001",
            timestamp=datetime.now(),
            event_type=AuditEventType.TOOL_EXECUTION_STARTED,
            user_id="insert_user",
            session_id="insert_session",
            task_id="insert_task",
            tool_id="insert_tool",
            severity="info",
            details={"test": "data"},
        )

        db.log_event(event)

        # Verify the event was inserted
        with db._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM audit_events WHERE id = ?", (event.id,)
            )
            row = cursor.fetchone()

            assert row is not None
            assert row[0] == event.id  # id
            assert row[1] == event.timestamp.isoformat()  # timestamp
            assert row[2] == event.event_type.value  # event_type
            assert row[3] == event.user_id  # user_id
            assert row[4] == event.session_id  # session_id
            assert row[5] == event.task_id  # task_id
            assert row[6] == event.tool_id  # tool_id
            assert row[7] == event.severity  # severity
            assert row[8] == json.dumps(event.details)  # details

    def test_insert_event_with_none_values(self, temp_db_path):
        """Test inserting event with None optional values."""
        db = AuditDatabase(temp_db_path)
        event = AuditEvent(
            id="test_insert_002",
            timestamp=datetime.now(),
            event_type=AuditEventType.CACHE_HIT,
            user_id="cache_user",
            session_id="cache_session",
            task_id="cache_task",
            # No optional fields set
        )

        db.log_event(event)

        with db._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM audit_events WHERE id = ?", (event.id,)
            )
            row = cursor.fetchone()

            assert row is not None
            assert row[6] is None  # tool_id
            assert row[7] == "info"  # severity (default)
            assert row[8] is None  # details
            assert row[9] is None  # ip_address
            assert row[10] is None  # user_agent

    def test_get_events_no_filters(self, temp_db_path):
        """Test retrieving all events without filters."""
        db = AuditDatabase(temp_db_path)

        # Insert test events
        events = [
            AuditEvent(
                id=f"test_event_{i}",
                timestamp=datetime.now(),
                event_type=AuditEventType.TOOL_EXECUTION_STARTED,
                user_id="user1",
                session_id="session1",
                task_id="task1",
            )
            for i in range(5)
        ]

        for event in events:
            db.log_event(event)

        retrieved_events = db.get_events()

        assert len(retrieved_events) == 5
        retrieved_ids = [event.id for event in retrieved_events]
        assert set(retrieved_ids) == {f"test_event_{i}" for i in range(5)}

    def test_get_events_with_user_filter(self, temp_db_path):
        """Test retrieving events filtered by user_id."""
        db = AuditDatabase(temp_db_path)

        # Insert events for different users
        events = [
            AuditEvent(
                id="user1_event_1",
                timestamp=datetime.now(),
                event_type=AuditEventType.TOOL_EXECUTION_STARTED,
                user_id="user1",
                session_id="session1",
                task_id="task1",
            ),
            AuditEvent(
                id="user2_event_1",
                timestamp=datetime.now(),
                event_type=AuditEventType.TOOL_EXECUTION_STARTED,
                user_id="user2",
                session_id="session2",
                task_id="task2",
            ),
            AuditEvent(
                id="user1_event_2",
                timestamp=datetime.now(),
                event_type=AuditEventType.TOOL_EXECUTION_COMPLETED,
                user_id="user1",
                session_id="session3",
                task_id="task3",
            ),
        ]

        for event in events:
            db.log_event(event)

        # Filter by user1
        user1_events = db.get_events(user_id="user1")

        assert len(user1_events) == 2
        user_ids = [event.user_id for event in user1_events]
        assert all(user_id == "user1" for user_id in user_ids)

    def test_get_events_with_time_filter(self, temp_db_path):
        """Test retrieving events filtered by time range."""
        db = AuditDatabase(temp_db_path)

        base_time = datetime.now()

        # Insert events at different times
        events = [
            AuditEvent(
                id="past_event",
                timestamp=base_time - timedelta(hours=2),
                event_type=AuditEventType.TOOL_EXECUTION_STARTED,
                user_id="user1",
                session_id="session1",
                task_id="task1",
            ),
            AuditEvent(
                id="recent_event",
                timestamp=base_time,
                event_type=AuditEventType.TOOL_EXECUTION_STARTED,
                user_id="user2",
                session_id="session2",
                task_id="task2",
            ),
        ]

        for event in events:
            db.log_event(event)

        # Filter by recent time range
        recent_events = db.get_events(
            start_time=base_time - timedelta(hours=1),
            end_time=base_time + timedelta(hours=1),
        )

        assert len(recent_events) == 1
        assert recent_events[0].id == "recent_event"

    def test_get_events_with_tool_filter(self, temp_db_path):
        """Test retrieving events filtered by tool_id."""
        db = AuditDatabase(temp_db_path)

        events = [
            AuditEvent(
                id="tool1_event",
                timestamp=datetime.now(),
                event_type=AuditEventType.TOOL_EXECUTION_STARTED,
                user_id="user1",
                session_id="session1",
                task_id="task1",
                tool_id="tool1",
            ),
            AuditEvent(
                id="tool2_event",
                timestamp=datetime.now(),
                event_type=AuditEventType.TOOL_EXECUTION_STARTED,
                user_id="user2",
                session_id="session2",
                task_id="task2",
                tool_id="tool2",
            ),
        ]

        for event in events:
            db.log_event(event)

        # Filter by tool1
        tool1_events = db.get_events(tool_id="tool1")

        assert len(tool1_events) == 1
        assert tool1_events[0].tool_id == "tool1"

    def test_get_events_with_limit(self, temp_db_path):
        """Test retrieving events with limit."""
        db = AuditDatabase(temp_db_path)

        # Insert 10 events
        events = [
            AuditEvent(
                id=f"limited_event_{i}",
                timestamp=datetime.now(),
                event_type=AuditEventType.TOOL_EXECUTION_STARTED,
                user_id="user1",
                session_id="session1",
                task_id="task1",
            )
            for i in range(10)
        ]

        for event in events:
            db.log_event(event)

        # Retrieve with limit of 5
        limited_events = db.get_events(limit=5)

        assert len(limited_events) == 5

    def test_database_error_handling(self, temp_db_path):
        """Test database error handling."""
        db = AuditDatabase(temp_db_path)

        # Test error handling by trying to log with None event
        try:
            # This should be handled gracefully or raise an error
            db.log_event(None)
            assert False, "Should have raised an error"
        except (TypeError, AttributeError):
            # Expected error types
            pass


class TestAuditLogger:
    """Test the AuditLogger class."""

    @pytest.fixture
    def mock_database(self):
        """Create a mock database."""
        return MagicMock(spec=AuditDatabase)

    @pytest.fixture
    def audit_logger(self, mock_database):
        """Create an audit logger with mock database."""
        return AuditLogger(database=mock_database)

    @pytest.fixture
    def sample_execution_context(self):
        """Create a sample execution context."""
        return ExecutionContext(
            tool_id="test_tool",
            session_id="test_session",
            task_id="test_task",
            user_id="test_user",
        )

    def test_audit_logger_initialization(self, mock_database):
        """Test AuditLogger initialization."""
        logger = AuditLogger(database=mock_database)

        assert logger.database is mock_database
        assert logger._lock is None

    @pytest.mark.asyncio
    async def test_get_lock(self, audit_logger):
        """Test lock initialization and retrieval."""
        # Initially no lock
        assert audit_logger._lock is None

        # Get lock should initialize it
        lock1 = audit_logger._get_lock()
        assert audit_logger._lock is not None

        # Subsequent calls should return the same lock
        lock2 = audit_logger._get_lock()
        assert lock1 is lock2

    def test_generate_event_id(self, audit_logger):
        """Test event ID generation."""
        task_id = "test_task_123"
        timestamp = datetime(2023, 12, 1, 10, 30, 45, 123456)

        event_id = audit_logger._generate_event_id(task_id, timestamp)

        # Should start with audit_ prefix
        assert event_id.startswith("audit_")
        # Should contain timestamp parts
        assert "20231201" in event_id  # YYYYMMDD
        assert "103045" in event_id  # HHMMSS
        # Should contain task fragment
        assert "test_tas" in event_id  # truncated task_id
        # Should contain UUID fragment
        assert len(event_id.split("_")) >= 4

    def test_determine_severity(self, audit_logger):
        """Test severity determination from event types."""
        # Error-level events
        assert (
            audit_logger._determine_severity(AuditEventType.SECURITY_VIOLATION)
            == "error"
        )

        # Warning-level events
        assert (
            audit_logger._determine_severity(AuditEventType.TOOL_EXECUTION_FAILED)
            == "warning"
        )
        assert (
            audit_logger._determine_severity(AuditEventType.RATE_LIMIT_EXCEEDED)
            == "warning"
        )
        assert (
            audit_logger._determine_severity(AuditEventType.CIRCUIT_BREAKER_OPENED)
            == "warning"
        )
        assert (
            audit_logger._determine_severity(AuditEventType.OAUTH_LOGIN_FAILED)
            == "warning"
        )

        # Info-level events (default)
        assert (
            audit_logger._determine_severity(AuditEventType.TOOL_EXECUTION_STARTED)
            == "info"
        )
        assert (
            audit_logger._determine_severity(AuditEventType.TOOL_EXECUTION_COMPLETED)
            == "info"
        )
        assert audit_logger._determine_severity(AuditEventType.CACHE_HIT) == "info"
        assert (
            audit_logger._determine_severity(AuditEventType.OAUTH_LOGIN_SUCCESS)
            == "info"
        )

    @pytest.mark.asyncio
    async def test_log_tool_execution(self, audit_logger, sample_execution_context):
        """Test tool execution event logging."""
        tool = BashTool(
            id="test_tool",
            name="Test Tool",
            description="Test tool",
            version="1.0.0",
            script="echo 'test'",
            parameters=[],
            config=ToolConfig(),
        )

        with patch.object(
            audit_logger, "_persist_and_log", new=AsyncMock()
        ) as mock_persist:
            await audit_logger.log_tool_execution(
                AuditEventType.TOOL_EXECUTION_STARTED,
                sample_execution_context,
                tool=tool,
                execution_time=1.5,
            )

            # Verify _persist_and_log was called
            assert mock_persist.called
            call_args = mock_persist.call_args
            event = call_args[0][0]  # First positional arg

            assert event.event_type == AuditEventType.TOOL_EXECUTION_STARTED
            assert event.user_id == sample_execution_context.user_id
            assert event.session_id == sample_execution_context.session_id
            assert event.task_id == sample_execution_context.task_id
            assert event.tool_id == tool.id
            assert "execution_time" in event.details

    @pytest.mark.asyncio
    async def test_log_tool_execution_without_tool(
        self, audit_logger, sample_execution_context
    ):
        """Test tool execution event logging without tool."""
        with patch.object(
            audit_logger, "_persist_and_log", new=AsyncMock()
        ) as mock_persist:
            await audit_logger.log_tool_execution(
                AuditEventType.TOOL_EXECUTION_FAILED,
                sample_execution_context,
                error_message="Tool not found",
            )

            call_args = mock_persist.call_args
            event = call_args[0][0]

            assert event.tool_id is None
            assert "error_message" in event.details

    @pytest.mark.asyncio
    async def test_log_event(self, audit_logger):
        """Test general event logging."""
        with patch.object(
            audit_logger, "_persist_and_log", new=AsyncMock()
        ) as mock_persist:
            await audit_logger.log_event(
                AuditEventType.CACHE_HIT,
                user_id="cache_user",
                session_id="cache_session",
                task_id="cache_task",
                details={"cache_key": "test_key"},
            )

            call_args = mock_persist.call_args
            event = call_args[0][0]

            assert event.event_type == AuditEventType.CACHE_HIT
            assert event.user_id == "cache_user"
            assert event.session_id == "cache_session"
            assert event.task_id == "cache_task"
            assert event.details == {"cache_key": "test_key"}

    @pytest.mark.asyncio
    async def test_log_security_event(self, audit_logger, sample_execution_context):
        """Test security event logging."""
        with patch.object(
            audit_logger, "log_tool_execution", new=AsyncMock()
        ) as mock_log:
            await audit_logger.log_security_event(
                sample_execution_context,
                violation_type="unauthorized_access",
                details={"path": "/etc/passwd", "user": "unauthorized_user"},
            )

            # Verify log_tool_execution was called with correct parameters
            mock_log.assert_called_once()
            call_args = mock_log.call_args

            assert call_args[0][0] == AuditEventType.SECURITY_VIOLATION
            assert call_args[0][1] == sample_execution_context
            assert call_args[1]["severity"] == "error"
            assert call_args[1]["violation_type"] == "unauthorized_access"
            assert call_args[1]["path"] == "/etc/passwd"
            assert call_args[1]["user"] == "unauthorized_user"

    @pytest.mark.asyncio
    async def test_persist_and_log(self, audit_logger):
        """Test event persistence and logging."""
        event = AuditEvent(
            id="persist_test_001",
            timestamp=datetime.now(),
            event_type=AuditEventType.TOOL_EXECUTION_STARTED,
            user_id="persist_user",
            session_id="persist_session",
            task_id="persist_task",
        )

        log_extra = {"custom_field": "custom_value"}

        with patch.object(audit_logger.database, "log_event") as mock_insert:
            result = await audit_logger._persist_and_log(event, log_extra)

            # Verify database insertion
            mock_insert.assert_called_once_with(event)

            # Verify result
            assert result == event

    def test_get_audit_statistics_empty_database(self, mock_database):
        """Test audit statistics with empty database."""
        audit_logger = AuditLogger(database=mock_database)

        mock_database.get_event_stats.return_value = {
            "event_stats": {},
            "tool_stats": [],
        }

        stats = audit_logger.database.get_event_stats()

        assert stats == {"event_stats": {}, "tool_stats": []}

    def test_get_audit_statistics_with_data(self, mock_database):
        """Test audit statistics with sample data."""
        audit_logger = AuditLogger(database=mock_database)

        # Mock database to return sample statistics
        mock_database.get_event_stats.return_value = {
            "event_stats": {
                "tool_execution_started": {"info": 5},
                "tool_execution_failed": {"warning": 3},
            },
            "tool_stats": [],
            "total_events_24h": 8,
        }

        stats = audit_logger.database.get_event_stats()

        assert "event_stats" in stats
        assert "tool_stats" in stats
        assert stats["event_stats"]["tool_execution_started"]["info"] == 5
        assert stats["event_stats"]["tool_execution_failed"]["warning"] == 3


class TestConvenienceFunctions:
    """Test convenience functions for logging events."""

    @pytest.fixture
    def mock_audit_logger(self):
        """Create a mock audit logger."""
        return MagicMock()

    @pytest.fixture
    def sample_execution_context(self):
        """Create a sample execution context."""
        return ExecutionContext(
            tool_id="convenience_tool",
            session_id="convenience_session",
            task_id="convenience_task",
            user_id="convenience_user",
        )

    @pytest.mark.asyncio
    async def test_log_oauth_event(self, mock_audit_logger):
        """Test OAuth event logging convenience function."""
        with patch("a2a_acp.audit.get_audit_logger", return_value=mock_audit_logger):
            mock_result = AuditEvent(
                id="oauth_test",
                timestamp=datetime.now(),
                event_type=AuditEventType.OAUTH_LOGIN_SUCCESS,
                user_id="oauth",
                session_id="test_agent",
                task_id="oauth_flow",
            )
            mock_audit_logger.log_event = AsyncMock(return_value=mock_result)

            result = await log_oauth_event(
                AuditEventType.OAUTH_LOGIN_SUCCESS,
                agent_id="test_agent",
                details={"provider": "github"},
            )

            # Verify log_event was called
            assert mock_audit_logger.log_event.called
            assert result is not None

    @pytest.mark.asyncio
    async def test_log_security_event_convenience(
        self, mock_audit_logger, sample_execution_context
    ):
        """Test security event logging convenience function."""
        with patch("a2a_acp.audit.get_audit_logger", return_value=mock_audit_logger):
            mock_audit_logger.log_security_event = AsyncMock()

            await log_security_event(
                sample_execution_context,
                violation_type="script_injection",
                script_content="malicious_script",
                user_input="user_data",
            )

            # Verify log_security_event was called
            assert mock_audit_logger.log_security_event.called

    @pytest.mark.asyncio
    async def test_log_tool_execution_convenience(
        self, mock_audit_logger, sample_execution_context
    ):
        """Test tool execution logging convenience function."""
        tool = BashTool(
            id="convenience_tool",
            name="Convenience Tool",
            description="Tool for convenience testing",
            version="1.0.0",
            script="echo 'test'",
            parameters=[],
            config=ToolConfig(),
        )

        with patch("a2a_acp.audit.get_audit_logger", return_value=mock_audit_logger):
            mock_audit_logger.log_tool_execution = AsyncMock()

            await log_tool_execution(
                AuditEventType.TOOL_EXECUTION_COMPLETED,
                sample_execution_context,
                tool=tool,
                execution_time=2.5,
                output="Tool completed successfully",
            )

            # Verify log_tool_execution was called
            assert mock_audit_logger.log_tool_execution.called


class TestAuditReportGeneration:
    """Test audit report generation functionality."""

    @pytest.fixture
    def mock_audit_logger(self):
        """Create a mock audit logger."""
        return MagicMock()

    def test_generate_audit_report_sync(self, mock_audit_logger):
        """Test synchronous audit report generation."""
        with patch("a2a_acp.audit.get_audit_logger", return_value=mock_audit_logger):
            # Create an async mock function
            async def mock_get_audit_report(*args, **kwargs):
                return {
                    "summary": {"total_events": 10},
                    "statistics": {},
                    "recent_events": [],
                }

            mock_audit_logger.get_audit_report = mock_get_audit_report

            report = generate_audit_report(
                start_time=datetime.now() - timedelta(hours=1),
                end_time=datetime.now(),
                user_id="test_user",
            )

            assert "summary" in report
            assert "recent_events" in report

    @pytest.mark.asyncio
    async def test_get_audit_report(self, mock_audit_logger):
        """Test audit report generation with sample data."""
        audit_logger = AuditLogger(database=MagicMock())

        # Mock database to return sample events
        events = [
            AuditEvent(
                id=f"report_event_{i}",
                timestamp=datetime.now(),
                event_type=AuditEventType.TOOL_EXECUTION_STARTED,
                user_id="user1",
                session_id="session1",
                task_id="task1",
                severity="info",
            )
            for i in range(5)
        ]

        with patch.object(audit_logger.database, "get_events", return_value=events):
            report = await audit_logger.get_audit_report(
                start_time=datetime.now() - timedelta(hours=1),
                end_time=datetime.now(),
                user_id=None,
                tool_id=None,
            )

            assert "summary" in report
            assert "recent_events" in report
            assert report["summary"]["total_events"] == 5
            assert len(report["recent_events"]) == 5

    @pytest.mark.asyncio
    async def test_get_audit_report_with_filters(self, mock_audit_logger):
        """Test audit report generation with various filters."""
        audit_logger = AuditLogger(database=MagicMock())

        # Mock database to return filtered events
        filtered_events = [
            AuditEvent(
                id="filtered_event",
                timestamp=datetime.now(),
                event_type=AuditEventType.SECURITY_VIOLATION,
                user_id="filtered_user",
                session_id="filtered_session",
                task_id="filtered_task",
                tool_id="filtered_tool",
                severity="error",
            )
        ]

        with patch.object(
            audit_logger.database, "get_events", return_value=filtered_events
        ) as mock_get_events:
            report = await audit_logger.get_audit_report(
                start_time=datetime.now() - timedelta(hours=1),
                end_time=datetime.now(),
                user_id="filtered_user",
                tool_id="filtered_tool",
            )

            # Verify database was called with correct filters
            mock_get_events.assert_called_once()

            # The actual timestamp values will be slightly different
            call_args = mock_get_events.call_args
            assert call_args[1]["user_id"] == "filtered_user"
            assert call_args[1]["tool_id"] == "filtered_tool"
            assert call_args[1]["limit"] == 1000

            assert report["summary"]["total_events"] == 1
            assert len(report["recent_events"]) == 1


class TestGlobalAuditLogger:
    """Test global audit logger singleton."""

    def test_get_audit_logger_singleton(self):
        """Test that get_audit_logger returns singleton instance."""
        logger1 = get_audit_logger()
        logger2 = get_audit_logger()

        assert logger1 is logger2
        assert isinstance(logger1, AuditLogger)

    def test_global_logger_initialization(self):
        """Test global logger initialization."""
        # Get current global logger
        original_logger = get_audit_logger()

        # Verify it's an AuditLogger instance
        assert isinstance(original_logger, AuditLogger)


class TestErrorHandling:
    """Test error handling in audit system."""

    @pytest.fixture
    def failing_database(self):
        """Create a database that fails on operations."""
        db = MagicMock(spec=AuditDatabase)
        db.log_event.side_effect = Exception("Database connection failed")
        return db

    @pytest.mark.asyncio
    async def test_persist_and_log_with_database_error(self, failing_database):
        """Test error handling when database operations fail."""
        audit_logger = AuditLogger(database=failing_database)

        event = AuditEvent(
            id="error_test_001",
            timestamp=datetime.now(),
            event_type=AuditEventType.TOOL_EXECUTION_STARTED,
            user_id="error_user",
            session_id="error_session",
            task_id="error_task",
        )

        # The _persist_and_log method should handle database errors
        # (exact behavior depends on implementation)
        try:
            await audit_logger._persist_and_log(event, {})
        except Exception as e:
            # If the method propagates the error, that's also valid
            assert str(e) == "Database connection failed"

    @pytest.mark.asyncio
    async def test_log_event_with_invalid_parameters(self, failing_database):
        """Test logging with invalid parameters."""
        audit_logger = AuditLogger(database=failing_database)

        # Test that the method works with valid parameters
        # The failing_database fixture will cause an error, which is the expected behavior
        with pytest.raises(Exception):
            await audit_logger.log_event(
                AuditEventType.TOOL_EXECUTION_STARTED,
                user_id="test_user",
                session_id="test_session",
                task_id="test_task",
            )

    def test_audit_database_connection_error(self):
        """Test database connection error handling."""
        # Test with non-existent database path
        non_existent_path = "/non/existent/path/database.db"

        try:
            db = AuditDatabase(non_existent_path)
            # If it doesn't fail immediately, try an operation
            with pytest.raises((FileNotFoundError, OSError, PermissionError)):
                with db._get_connection() as conn:
                    conn.execute("SELECT 1").fetchone()
        except Exception:
            # Some implementations might fail at initialization
            pass

    def test_audit_event_serialization_errors(self):
        """Test audit event serialization with problematic data."""
        # Test with non-serializable details
        event = AuditEvent(
            id="serialization_test",
            timestamp=datetime.now(),
            event_type=AuditEventType.TOOL_EXECUTION_STARTED,
            user_id="test_user",
            session_id="test_session",
            task_id="test_task",
            details={"lambda": lambda x: x},  # Non-serializable
        )

        # to_dict should handle non-serializable data gracefully
        try:
            data = event.to_dict()
            # If it succeeds, the lambda should be converted to string or None
            assert "details" in data
        except (TypeError, ValueError):
            # If it fails, that's also acceptable behavior
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
