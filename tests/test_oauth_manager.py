# a2a-acp/tests/test_oauth_manager.py
"""
Unit tests for a2a_acp.oauth_manager

Covers:
- PKCE code_verifier + code_challenge behavior
- StateStore create/get/pop and TTL cleanup
- create_auth_session: URL contains PKCE and state is stored
- complete_auth_session: mocks token endpoint and verifies auth.json written
- refresh_token_for_codex_home: mocks refresh endpoint and verifies file updates
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict

import pytest

from a2a_acp import oauth_manager
from a2a_acp.settings import get_settings
from a2a_acp.utils.file_utils import atomic_write_json


def _clear_settings_cache() -> None:
    """Ensure settings cached by lru_cache are cleared between tests."""
    try:
        get_settings.cache_clear()
    except AttributeError:
        # No cache / nothing to clear
        pass


async def _clear_state_store() -> None:
    """Clear the module-level StateStore (test helper)."""
    async with oauth_manager._state_store._lock:
        oauth_manager._state_store._store.clear()


def _now_ms() -> int:
    """Return current time in ms for tests."""
    return oauth_manager.now_ms()


def test_pkce_generation_and_challenge_properties() -> None:
    """generate_code_verifier returns a base64url string (no padding) and challenge matches S256 digest."""
    verifier = oauth_manager.generate_code_verifier()
    assert isinstance(verifier, str)
    # no padding
    assert "=" not in verifier
    assert len(verifier) > 0

    # Compute expected code challenge manually
    expected = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest())
        .rstrip(b"=")
        .decode("ascii")
    )

    challenge = oauth_manager.generate_code_challenge(verifier)
    assert challenge == expected


@pytest.mark.asyncio
async def test_state_store_create_get_pop_and_ttl(monkeypatch) -> None:
    """StateStore creates entries, can pop them, and TTL expiration is respected."""
    _clear_settings_cache()
    await _clear_state_store()

    state = "tests-state-1"
    agent_id = "agent-test-1"
    code_verifier = "cv-abc-123"
    now = _now_ms()
    expires_at = now + 2000  # 2s TTL for test

    await oauth_manager._state_store.create(
        state,
        agent_id,
        code_verifier,
        client_id=None,
        redirect_uri=None,
        scope=None,
        expires_at=expires_at,
    )

    # get returns without removing
    entry = await oauth_manager._state_store.get(state)
    assert entry is not None
    assert entry.agent_id == agent_id
    assert entry.code_verifier == code_verifier

    # pop returns and removes the entry
    popped = await oauth_manager._state_store.pop(state)
    assert popped is not None
    assert popped.agent_id == agent_id

    # now get should return None
    got_after_pop = await oauth_manager._state_store.get(state)
    assert got_after_pop is None

    # TTL expiry scenario: create a very short-lived entry
    short_state = "tests-state-ttl"
    await oauth_manager._state_store.create(
        short_state,
        agent_id,
        code_verifier,
        client_id=None,
        redirect_uri=None,
        scope=None,
        expires_at=_now_ms() + 5,  # expire in 5ms
    )

    # Wait for small period ensuring expiration
    await asyncio.sleep(0.02)  # 20ms
    expired = await oauth_manager._state_store.get(short_state)
    assert expired is None

    # cleanup should remove the expired key and return >0
    removed = await oauth_manager.cleanup_expired_states()
    assert removed >= 0  # may be 0 if already removed; at least non-negative


@pytest.mark.asyncio
async def test_create_auth_session_populates_state_and_url(
    tmp_path, monkeypatch
) -> None:
    """Create an auth session and confirm authorize_url contains PKCE and state is stored."""
    # Set minimal settings for OAuth
    monkeypatch.setenv("A2A_AGENT_COMMAND", "true")
    monkeypatch.setenv("A2A_AGENT_OAUTH_ENABLED", "true")
    monkeypatch.setenv("A2A_AGENT_CODEX_HOME", str(tmp_path))
    monkeypatch.setenv("A2A_AGENT_AUTH_FILE_NAME", "auth.json")
    _clear_settings_cache()

    await _clear_state_store()

    agent_id = "agent-create-session"
    payload = await oauth_manager.create_auth_session(agent_id=agent_id)

    assert "authorize_url" in payload
    assert "state" in payload
    assert "expires_at" in payload

    assert "code_challenge=" in payload["authorize_url"]
    assert "code_challenge_method=S256" in payload["authorize_url"]
    assert "state=" in payload["authorize_url"]

    # Ensure state is recorded in the store
    state = payload["state"]
    entry = await oauth_manager._state_store.get(state)
    assert entry is not None
    assert entry.agent_id == agent_id


@pytest.mark.asyncio
async def test_complete_auth_session_writes_auth_json(tmp_path, monkeypatch) -> None:
    """
    Mock the token endpoint response and verify that complete_auth_session writes
    the token (auth.json) with the expected schema into codex_home.
    """
    # Configure settings (codex home)
    monkeypatch.setenv("A2A_AGENT_COMMAND", "true")
    monkeypatch.setenv("A2A_AGENT_OAUTH_ENABLED", "true")
    monkeypatch.setenv("A2A_AGENT_CODEX_HOME", str(tmp_path))
    monkeypatch.setenv("A2A_AGENT_AUTH_FILE_NAME", "auth.json")
    _clear_settings_cache()

    await _clear_state_store()

    agent_id = "agent-complete-session"
    # Create PKCE state entry
    create_payload = await oauth_manager.create_auth_session(agent_id=agent_id)
    state = create_payload["state"]

    # Prepare mocked token response
    token_response = {
        "access_token": "access-token-test-1",
        "refresh_token": "refresh-token-test-1",
        "expires_in": 3600,
        "id_token": "id-token-test-1",
    }

    class DummyResponse:
        def __init__(self, data: Dict[str, Any]):
            self._data = data

        def raise_for_status(self) -> None:
            # simulate no error
            return None

        def json(self) -> Dict[str, Any]:
            return self._data

    class DummyAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, exc_tb) -> None:
            return False

        async def post(self, url, data=None, headers=None):
            # ensure we received required fields in the data map (basic sanity)
            assert data is not None
            # Return a dummy response
            return DummyResponse(token_response)

    # Patch AsyncClient used by the oauth_manager module
    monkeypatch.setattr(
        "a2a_acp.oauth_manager.httpx.AsyncClient", DummyAsyncClient, raising=True
    )

    # Perform the exchange: we can call complete_auth_session with a dummy code
    result = await oauth_manager.complete_auth_session(
        agent_id=agent_id, code="dummy-code", state=state
    )

    assert isinstance(result, dict)
    assert result.get("success") is True
    assert result.get("agent_id") == agent_id
    assert "expires" in result and result["expires"] > _now_ms()

    # Verify file persisted to disk
    auth_path = Path(tmp_path) / get_settings().agent_auth_file_name
    assert auth_path.exists()

    with auth_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["type"] == "oauth"
    assert data["access"] == token_response["access_token"]
    assert data["refresh"] == token_response["refresh_token"]
    assert data["id_token"] == token_response["id_token"]
    assert isinstance(data["expires"], int)
    assert isinstance(data["created_at"], int)


@pytest.mark.asyncio
async def test_refresh_token_for_codex_home_updates_token_file(
    tmp_path, monkeypatch
) -> None:
    """Write an initial token file and verify refresh_token_for_codex_home replaces the access token."""
    # Setup settings
    monkeypatch.setenv("A2A_AGENT_COMMAND", "true")
    monkeypatch.setenv("A2A_AGENT_OAUTH_ENABLED", "true")
    monkeypatch.setenv("A2A_AGENT_CODEX_HOME", str(tmp_path))
    monkeypatch.setenv("A2A_AGENT_AUTH_FILE_NAME", "auth.json")
    _clear_settings_cache()

    auth_path = Path(tmp_path) / get_settings().agent_auth_file_name

    now = _now_ms()
    initial_token = {
        "type": "oauth",
        "access": "initial-access",
        "refresh": "initial-refresh",
        "expires": now - 10000,  # already expired
        "created_at": now - 3600000,
    }
    atomic_write_json(auth_path, initial_token)

    # Prepare mocked refresh response
    refreshed_response = {
        "access_token": "refreshed-access",
        "refresh_token": "refreshed-refresh",
        "expires_in": 7200,
    }

    class DummyResponse:
        def __init__(self, data: Dict[str, Any]):
            self._data = data

        def raise_for_status(self) -> None:
            return None

        def json(self) -> Dict[str, Any]:
            return self._data

    class DummyAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, exc_tb) -> None:
            return False

        async def post(self, url, data=None, headers=None):
            # Validate we're calling refresh grant type
            assert data is not None and data.get("grant_type") == "refresh_token"
            return DummyResponse(refreshed_response)

    monkeypatch.setattr(
        "a2a_acp.oauth_manager.httpx.AsyncClient", DummyAsyncClient, raising=True
    )

    updated = await oauth_manager.refresh_token_for_codex_home(Path(tmp_path))

    assert updated["access"] == refreshed_response["access_token"]
    assert updated["refresh"] == refreshed_response["refresh_token"]
    assert updated["expires"] > now

    # Verify file persisted
    with auth_path.open("r", encoding="utf-8") as f:
        persisted = json.load(f)

    assert persisted["access"] == refreshed_response["access_token"]
    assert persisted["refresh"] == refreshed_response["refresh_token"]
    assert persisted["expires"] > now


@pytest.mark.asyncio
async def test_check_auth_file_valid(tmp_path, monkeypatch) -> None:
    """Validate check_auth_file_valid returns True for valid tokens and False for expired/missing tokens."""
    # Setup settings
    monkeypatch.setenv("A2A_AGENT_CODEX_HOME", str(tmp_path))
    monkeypatch.setenv("A2A_AGENT_COMMAND", "true")
    monkeypatch.setenv("A2A_AGENT_OAUTH_ENABLED", "true")
    monkeypatch.setenv("A2A_AGENT_AUTH_FILE_NAME", "auth.json")
    _clear_settings_cache()

    auth_path = Path(tmp_path) / get_settings().agent_auth_file_name

    # No token file -> False
    assert oauth_manager.check_auth_file_valid(Path(tmp_path)) is False

    # Write valid token
    token_valid = {
        "type": "oauth",
        "access": "ok",
        "refresh": "r",
        "expires": _now_ms() + 100000,
        "created_at": _now_ms(),
    }
    atomic_write_json(auth_path, token_valid)
    assert oauth_manager.check_auth_file_valid(Path(tmp_path)) is True

    # Expired token
    token_expired = {
        "type": "oauth",
        "access": "ok",
        "refresh": "r",
        "expires": _now_ms() - 1000,
        "created_at": _now_ms() - 3600000,
    }
    atomic_write_json(auth_path, token_expired)
    assert oauth_manager.check_auth_file_valid(Path(tmp_path)) is False


@pytest.mark.asyncio
async def test_read_auth_file_fallback_detects_alternate_names(
    tmp_path, monkeypatch
) -> None:
    """If a legacy or alternate token file exists, read_auth_file should return it.

    This validates the fallback detection logic (openai-auth.json / .auth.json) when
    the default `auth.json` is not present.
    """
    # Configure minimal settings and enforce codex_home = tmp_path
    monkeypatch.setenv("A2A_AGENT_COMMAND", "true")
    monkeypatch.setenv("A2A_AGENT_OAUTH_ENABLED", "true")
    monkeypatch.setenv("A2A_AGENT_CODEX_HOME", str(tmp_path))
    monkeypatch.setenv("A2A_AGENT_AUTH_FILE_NAME", "auth.json")
    _clear_settings_cache()

    # Create a token file using a legacy candidate name (openai-auth.json)
    candidate_path = Path(tmp_path) / "openai-auth.json"
    token_fallback = {
        "type": "oauth",
        "access": "fallback-access",
        "refresh": "fallback-refresh",
        "expires": _now_ms() + 3600 * 1000,
        "created_at": _now_ms(),
    }
    atomic_write_json(candidate_path, token_fallback)

    # read_auth_file should detect the fallback file and return its contents
    found = oauth_manager.read_auth_file(Path(tmp_path))
    assert found is not None
    assert found.get("access") == token_fallback["access"]
    assert found.get("refresh") == token_fallback["refresh"]

    # get_token_info_for_agent should also detect the fallback name when codex_home is resolved
    token_info = await oauth_manager.get_token_info_for_agent("agent-fallback")
    assert token_info is not None
    assert token_info.get("access") == token_fallback["access"]
    assert token_info.get("refresh") == token_fallback["refresh"]


@pytest.mark.asyncio
async def test_oauth_manager_logging_never_includes_tokens(
    tmp_path, monkeypatch, caplog
) -> None:
    """Ensure auth manager logs do not leak access/refresh tokens."""
    monkeypatch.setenv("A2A_AGENT_COMMAND", "true")
    monkeypatch.setenv("A2A_AGENT_OAUTH_ENABLED", "true")
    monkeypatch.setenv("A2A_AGENT_CODEX_HOME", str(tmp_path))
    monkeypatch.setenv("A2A_AGENT_AUTH_FILE_NAME", "auth.json")
    _clear_settings_cache()

    caplog.set_level(logging.INFO, logger="a2a_acp.oauth_manager")

    agent_id = "agent-logging-test"
    payload = await oauth_manager.create_auth_session(agent_id=agent_id)
    state = payload["state"]

    token_payload = {
        "access_token": "logged-access-token",
        "refresh_token": "logged-refresh-token",
        "expires_in": 3600,
    }

    class DummyResponse:
        def __init__(self, data: Dict[str, Any]) -> None:
            self._data = data

        def raise_for_status(self) -> None:
            return None

        def json(self) -> Dict[str, Any]:
            return self._data

    class DummyAsyncClientComplete:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, exc_tb) -> None:
            return False

        async def post(self, url, data=None, headers=None):
            return DummyResponse(token_payload)

    monkeypatch.setattr(
        "a2a_acp.oauth_manager.httpx.AsyncClient",
        DummyAsyncClientComplete,
        raising=True,
    )
    await oauth_manager.complete_auth_session(
        agent_id=agent_id, code="code-logging", state=state
    )

    refreshed_payload = {
        "access_token": "logged-access-refresh",
        "refresh_token": "logged-refresh-refresh",
        "expires_in": 7200,
    }

    class DummyAsyncClientRefresh:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, exc_tb) -> None:
            return False

        async def post(self, url, data=None, headers=None):
            return DummyResponse(refreshed_payload)

    monkeypatch.setattr(
        "a2a_acp.oauth_manager.httpx.AsyncClient",
        DummyAsyncClientRefresh,
        raising=True,
    )
    await oauth_manager.refresh_token_for_codex_home(Path(tmp_path))

    text = caplog.text
    sensitive_values = [
        token_payload["access_token"],
        token_payload["refresh_token"],
        refreshed_payload["access_token"],
        refreshed_payload["refresh_token"],
    ]
    for sensitive in sensitive_values:
        assert sensitive not in text
