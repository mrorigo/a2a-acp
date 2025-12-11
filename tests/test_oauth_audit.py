from __future__ import annotations

import pytest

from a2a_acp.audit import AuditDatabase, AuditEventType, AuditLogger, log_oauth_event


@pytest.mark.asyncio
async def test_log_oauth_event_records_event(monkeypatch, tmp_path) -> None:
    """Ensure OAuth audit helper persists events with agent metadata."""
    import a2a_acp.audit as audit_module

    db_path = tmp_path / "oauth_events.db"
    db = AuditDatabase(str(db_path))
    audit_logger = AuditLogger(database=db)
    monkeypatch.setattr(audit_module, "_audit_logger", audit_logger)

    await log_oauth_event(
        AuditEventType.OAUTH_LOGIN_SUCCESS,
        "audit-agent",
        {"expires": 120_000},
    )

    events = audit_logger.database.get_events(
        event_type=AuditEventType.OAUTH_LOGIN_SUCCESS
    )
    assert len(events) == 1
    event = events[0]
    assert event.event_type == AuditEventType.OAUTH_LOGIN_SUCCESS
    assert event.details["agent_id"] == "audit-agent"
    assert event.details["expires"] == 120_000
