# a2a-acp/tests/test_auth_endpoints.py
"""
Integration tests for OAuth-related REST endpoints:
- POST /a2a/agents/{agent_id}/auth/start
- GET  /a2a/agents/{agent_id}/auth/callback
- POST /a2a/agents/{agent_id}/auth/manual
- GET  /a2a/agents/{agent_id}/auth/status
- POST /a2a/agents/{agent_id}/auth/refresh
- DELETE /a2a/agents/{agent_id}/auth
- POST /a2a/agents/{agent_id}/auth/check-auth

Notes:
- The token endpoint used by oauth_manager is patched to a dummy async httpx client.
- Agent interactions (initialize/authenticate) are patched by replacing `ZedAgentConnection`
  in the `a2a_acp.api.auth_endpoints` module with a dummy context manager class that
  simulates behavior for accepting or rejecting tokens.
- These tests focus on end-to-end calling of REST endpoints via TestClient and assert:
  - proper success/failure responses,
  - token file written/read/removed in the codex_home,
  - refresh replacement behavior,
  - manual-flow parsing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from a2a_acp import oauth_manager
from a2a_acp.audit import AuditDatabase, AuditEventType, AuditLogger
from a2a_acp.main import create_app
from a2a_acp.settings import get_settings
from a2a_acp.utils.file_utils import atomic_write_json


def _clear_settings_cache() -> None:
    """Clear lru_cache on get_settings() between tests to ensure env changes get applied."""
    try:
        get_settings.cache_clear()
    except Exception:
        # Not fatal in case get_settings hasn't been cached in this environment
        pass


def _auth_file_path(tmp_path: Path) -> Path:
    """Return the path to auth file used by the settings pointing at `tmp_path`."""
    _clear_settings_cache()
    auth_name = get_settings().agent_auth_file_name
    return Path(tmp_path) / auth_name


class DummyResponse:
    """Minimal emulation of httpx.Response used by oauth_manager."""

    def __init__(self, payload: Dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        # Do nothing - simulates success
        return None

    def json(self) -> Dict[str, Any]:
        return self._payload


class DummyAsyncClient:
    """AsyncClient stub to patch out httpx.AsyncClient in oauth_manager."""

    def __init__(self, *args, **kwargs) -> None:
        self._last_post_args = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, exc_tb) -> None:
        return False

    async def post(self, url: str, data=None, headers=None):
        # Return token if the flow requests it; callers will set the expected response
        # via `DummyAsyncClient._configured_response` attribute.
        resp = getattr(self, "_configured_response", None)
        if resp is None:
            # Default: return a no-op token just to satisfy the code paths.
            default = {
                "access_token": "default-access",
                "refresh_token": "default-refresh",
                "expires_in": 3600,
            }
            return DummyResponse(default)
        return DummyResponse(resp)


class DummyZedAgentConnection:
    """Simple context manager to simulate agent handshake for auth endpoints."""

    def __init__(self, *args, **kwargs):
        # Optionally allow configured behavior
        self.raise_on_auth = kwargs.pop("raise_on_auth", False)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def initialize(self):
        # Simulate returning a structure similar to initialize RPC result
        return {"capabilities": {}}

    async def authenticate(self, method_id: str):
        if self.raise_on_auth:
            raise RuntimeError("Agent rejected authentication")
        return {"authenticated": True}


@pytest.fixture(autouse=True)
def reset_state_and_settings(monkeypatch, tmp_path):
    """
    Test-wide fixture to ensure settings are fresh for each test and the
    PKCE state store is cleared.
    """
    # Clear settings cache and state store between tests
    _clear_settings_cache()
    # default minimal environment to pass settings validation
    monkeypatch.setenv("A2A_AGENT_COMMAND", "true")
    # ensure OAuth gating is true for tests that need it; individual tests can overwrite
    # We'll leave it off by default to avoid accidentally gating tests that don't need it.
    try:
        # Clear the in-memory state store synchronously to avoid creating a coroutine
        oauth_manager._state_store._store.clear()
    except Exception:
        # Not required; it's safe to ignore
        pass


def make_test_client_with_oauth(monkeypatch, tmp_path: Path) -> TestClient:
    """Apply environment variables for OAuth-enabled tests and return a TestClient."""
    monkeypatch.setenv("A2A_AGENT_OAUTH_ENABLED", "true")
    monkeypatch.setenv("A2A_AGENT_CODEX_HOME", str(tmp_path))
    monkeypatch.setenv("A2A_AGENT_AUTH_FILE_NAME", "auth.json")
    _clear_settings_cache()
    app = create_app()
    return TestClient(app)


def _configure_dummy_httpx(
    client: DummyAsyncClient, response_payload: Dict[str, Any]
) -> None:
    """Configure the DummyAsyncClient to return the given payload from post()."""
    setattr(client, "_configured_response", response_payload)


def test_auth_start_returns_authorize_url_and_state(tmp_path, monkeypatch):
    """POST /auth/start should generate an authorize_url, state and expiry."""
    client = make_test_client_with_oauth(monkeypatch, tmp_path)

    # POST start with no overrides in body
    response = client.post("/a2a/agents/test-agent/auth/start", json={})
    assert response.status_code == 200, response.text
    data = response.json()

    assert "authorize_url" in data and "state" in data and "expires_at" in data
    assert "code_challenge" in data["authorize_url"]
    assert "code_challenge_method=S256" in data["authorize_url"]


def test_auth_start_emits_audit_event(monkeypatch, tmp_path):
    """Audit log records an OAuth login start event."""
    import a2a_acp.audit as audit_module

    db_path = tmp_path / "audit_events.db"
    db = AuditDatabase(str(db_path))
    audit_logger = AuditLogger(database=db)
    monkeypatch.setattr(audit_module, "_audit_logger", audit_logger)

    client = make_test_client_with_oauth(monkeypatch, tmp_path)
    response = client.post("/a2a/agents/test-agent/auth/start", json={})
    assert response.status_code == 200, response.text

    events = audit_logger.database.get_events(
        event_type=AuditEventType.OAUTH_LOGIN_STARTED
    )
    assert len(events) == 1
    event = events[0]
    assert event.details.get("state") == response.json()["state"]
    assert event.details.get("agent_id") == "test-agent"


def test_auth_callback_completes_and_writes_token(tmp_path, monkeypatch):
    """
    Full auth flow:
    - POST /auth/start to create PKCE state
    - GET /auth/callback?code=<code>&state=<state> to finalize and write tokens.
    """
    # Prepare app with OAuth enabled and codex_home
    client = make_test_client_with_oauth(monkeypatch, tmp_path)

    # Start session to obtain state
    start_resp = client.post("/a2a/agents/test-agent/auth/start", json={})
    assert start_resp.status_code == 200
    state = start_resp.json()["state"]

    # Setup httpx Dummy client to return an access/refresh token
    dummy_http = DummyAsyncClient()
    token_payload = {
        "access_token": "endpoint-access-token",
        "refresh_token": "endpoint-refresh-token",
        "expires_in": 3600,
        "id_token": "endpoint-id-token",
    }
    _configure_dummy_httpx(dummy_http, token_payload)

    # Patch the AsyncClient in oauth_manager module and ZedAgentConnection in endpoints
    with patch(
        "a2a_acp.oauth_manager.httpx.AsyncClient", lambda *args, **kwargs: dummy_http
    ):
        with patch(
            "a2a_acp.api.auth_endpoints.ZedAgentConnection", DummyZedAgentConnection
        ):
            # Call callback: code is arbitrary since our dummy token returns regardless
            cb_resp = client.get(
                f"/a2a/agents/test-agent/auth/callback?code=code-1&state={state}"
            )
            assert cb_resp.status_code == 200, cb_resp.text
            cb_data = cb_resp.json()
            assert cb_data.get("success") is True

    # Verify auth.json exists and contains the expected tokens
    auth_path = _auth_file_path(tmp_path)
    assert auth_path.exists()
    with auth_path.open("r", encoding="utf-8") as f:
        token_json = json.load(f)
    assert token_json["access"] == token_payload["access_token"]
    assert token_json["refresh"] == token_payload["refresh_token"]


def test_auth_manual_paste_flow(tmp_path, monkeypatch):
    """
    Manual paste flow should parse the callback URL and complete the same flow.
    """
    client = make_test_client_with_oauth(monkeypatch, tmp_path)

    # Start session to get state
    start_resp = client.post("/a2a/agents/test-agent/auth/start", json={})
    assert start_resp.status_code == 200
    state = start_resp.json()["state"]

    dummy_http = DummyAsyncClient()
    token_payload = {
        "access_token": "manual-access-token",
        "refresh_token": "manual-refresh-token",
        "expires_in": 3600,
    }
    _configure_dummy_httpx(dummy_http, token_payload)

    with patch(
        "a2a_acp.oauth_manager.httpx.AsyncClient", lambda *args, **kwargs: dummy_http
    ):
        with patch(
            "a2a_acp.api.auth_endpoints.ZedAgentConnection", DummyZedAgentConnection
        ):
            callback_url = f"http://localhost:8001/a2a/agents/test-agent/auth/callback?code=manual-code&state={state}"
            resp = client.post(
                "/a2a/agents/test-agent/auth/manual",
                json={"callback_url": callback_url},
            )
            assert resp.status_code == 200, resp.text
            assert resp.json().get("success") is True

    # Confirm the token file exists and matches
    auth_path = _auth_file_path(tmp_path)
    assert auth_path.exists()


def test_auth_status_reflects_token_presence(tmp_path, monkeypatch):
    """GET /auth/status should return signed_in true when token is present and false otherwise."""
    client = make_test_client_with_oauth(monkeypatch, tmp_path)

    # Initially, no token file exists
    resp_no = client.get("/a2a/agents/test-agent/auth/status")
    assert resp_no.status_code == 200
    assert resp_no.json()["signed_in"] is False

    # Write a valid token using atomic_write_json helper
    token_payload = {
        "type": "oauth",
        "access": "status-access",
        "refresh": "status-refresh",
        "expires": oauth_manager.now_ms() + 3600 * 1000,
        "created_at": oauth_manager.now_ms(),
    }
    atomic_write_json(_auth_file_path(tmp_path), token_payload)

    resp_yes = client.get("/a2a/agents/test-agent/auth/status")
    assert resp_yes.status_code == 200
    data = resp_yes.json()
    assert data["signed_in"] is True
    assert data["authentication_method"] == "chatgpt"


def test_auth_refresh_replaces_token_and_updates_file(tmp_path, monkeypatch):
    """POST /auth/refresh should call OAuth refresh and update the persisted auth.json file."""
    client = make_test_client_with_oauth(monkeypatch, tmp_path)

    # Write expired token to be refreshed
    initial = {
        "type": "oauth",
        "access": "initial-access",
        "refresh": "initial-refresh",
        "expires": oauth_manager.now_ms() - 1000,
        "created_at": oauth_manager.now_ms() - 3600000,
    }
    atomic_write_json(_auth_file_path(tmp_path), initial)

    # Configure a dummy refresh response
    refreshed = {
        "access_token": "refreshed-access",
        "refresh_token": "refreshed-refresh",
        "expires_in": 7200,
    }
    dummy_http = DummyAsyncClient()
    _configure_dummy_httpx(dummy_http, refreshed)

    with patch(
        "a2a_acp.oauth_manager.httpx.AsyncClient", lambda *args, **kwargs: dummy_http
    ):
        response = client.post("/a2a/agents/test-agent/auth/refresh", json={})
        assert response.status_code == 200, response.text
        data = response.json()
        assert "expires" in data

    # Validate the persisted file has been updated
    with _auth_file_path(tmp_path).open("r", encoding="utf-8") as f:
        persisted = json.load(f)
    assert persisted["access"] == refreshed["access_token"]
    assert persisted["refresh"] == refreshed["refresh_token"]


def test_auth_delete_removes_token_and_returns_204(tmp_path, monkeypatch):
    """DELETE /auth should remove auth.json and return 204 if token existed."""
    client = make_test_client_with_oauth(monkeypatch, tmp_path)

    # Ensure a token exists
    initial = {
        "type": "oauth",
        "access": "to-delete",
        "refresh": "r-to-delete",
        "expires": oauth_manager.now_ms() + 1000,
        "created_at": oauth_manager.now_ms(),
    }
    atomic_write_json(_auth_file_path(tmp_path), initial)

    # Patch ZedAgentConnection to a dummy that won't fail on authenticate
    with patch(
        "a2a_acp.api.auth_endpoints.ZedAgentConnection", DummyZedAgentConnection
    ):
        resp = client.delete("/a2a/agents/test-agent/auth")
        assert resp.status_code == 204

    # File should be removed
    assert not _auth_file_path(tmp_path).exists()


def test_check_auth_returns_200_when_agent_accepts_and_401_when_rejected(
    tmp_path, monkeypatch
):
    """POST /auth/check-auth uses agent initialize + authenticate to confirm token validity."""
    client = make_test_client_with_oauth(monkeypatch, tmp_path)

    # Write a valid token
    token = {
        "type": "oauth",
        "access": "check-access",
        "refresh": "check-refresh",
        "expires": oauth_manager.now_ms() + 3600_000,
        "created_at": oauth_manager.now_ms(),
    }
    atomic_write_json(_auth_file_path(tmp_path), token)

    # 1) Agent accepts auth
    with patch(
        "a2a_acp.api.auth_endpoints.ZedAgentConnection", DummyZedAgentConnection
    ):
        resp_ok = client.post("/a2a/agents/test-agent/auth/check-auth")
        assert resp_ok.status_code == 200
        assert resp_ok.json().get("success") is True

    # 2) Agent rejects auth
    class RejectingZedAgentConnection(DummyZedAgentConnection):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.raise_on_auth = True

    with patch(
        "a2a_acp.api.auth_endpoints.ZedAgentConnection", RejectingZedAgentConnection
    ):
        resp_reject = client.post("/a2a/agents/test-agent/auth/check-auth")
        assert resp_reject.status_code == 401


def test_auth_delete_removes_fallback_auth_file(tmp_path, monkeypatch):
    """
    If token is stored in a fallback auth file (e.g. openai-auth.json),
    DELETE /auth should remove that file as well.
    """
    client = make_test_client_with_oauth(monkeypatch, tmp_path)

    # Create fallback file (openai-auth.json) and write token
    token = {
        "type": "oauth",
        "access": "fallback-delete-access",
        "refresh": "fallback-delete-refresh",
        "expires": oauth_manager.now_ms() + 3600 * 1000,
        "created_at": oauth_manager.now_ms(),
    }
    fallback_path = Path(tmp_path) / "openai-auth.json"
    atomic_write_json(fallback_path, token)

    # Confirm status shows signed_in true (read fallback name)
    resp_status = client.get("/a2a/agents/test-agent/auth/status")
    assert resp_status.status_code == 200
    assert resp_status.json()["signed_in"] is True

    # Delete should succeed (204) and remove the fallback file
    with patch(
        "a2a_acp.api.auth_endpoints.ZedAgentConnection", DummyZedAgentConnection
    ):
        resp = client.delete("/a2a/agents/test-agent/auth")
        assert resp.status_code == 204, resp.text

    assert not fallback_path.exists()


def test_auto_refresh_on_status_triggers_refresh(tmp_path, monkeypatch):
    """
    If the token is within the auto-refresh window (<= 5 minutes from expiry),
    `GET /auth/status` should trigger an automatic refresh and the persisted token
    should be updated accordingly.
    """
    client = make_test_client_with_oauth(monkeypatch, tmp_path)

    # Prepare an initial token that is near expiry (4 minutes ahead) so auto-refresh triggers.
    initial_token = {
        "type": "oauth",
        "access": "near-expiry-access",
        "refresh": "initial-refresh",
        "expires": oauth_manager.now_ms() + 4 * 60 * 1000,  # 4 minutes
        "created_at": oauth_manager.now_ms(),
    }
    atomic_write_json(_auth_file_path(tmp_path), initial_token)

    # Configure a dummy AsyncClient to return a refreshed token payload on refresh call.
    refreshed_payload = {
        "access_token": "refreshed-access",
        "refresh_token": "refreshed-refresh",
        "expires_in": 7200,
    }
    dummy_http = DummyAsyncClient()
    _configure_dummy_httpx(dummy_http, refreshed_payload)

    # Patch the AsyncClient used by oauth_manager so refresh uses our dummy response.
    with patch(
        "a2a_acp.oauth_manager.httpx.AsyncClient", lambda *args, **kwargs: dummy_http
    ):
        # When status is requested, get_token_info_for_agent (used by status) will attempt auto-refresh.
        resp = client.get("/a2a/agents/test-agent/auth/status")
        assert resp.status_code == 200, resp.text
        # Confirm that refresh happened by checking the persisted auth file
        with _auth_file_path(tmp_path).open("r", encoding="utf-8") as fh:
            persisted = json.load(fh)

    assert persisted["access"] == refreshed_payload["access_token"]
    assert persisted["refresh"] == refreshed_payload["refresh_token"]
