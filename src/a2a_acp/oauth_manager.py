"""
OAuth manager for ChatGPT (OpenAI) authentication flows.

Implements:
- PKCE generation (code verifier + code challenge)
- State store for OAuth flows (state -> code_verifier mapping)
- Authorization URL generation (PKCE, state)
- Code exchange: authorization_code -> access/refresh tokens
- Token file write and read (atomic, permission controlled)
- Token refresh flow
- Token removal and validation helpers

This module aims to be agnostic to the HTTP framework (FastAPI) while
providing async helpers that REST endpoints in `api/auth_endpoints.py`
will call for the actual flows.

See docs/CHATGPT_AUTH_PLAN.md for the API contract and exact behaviors.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import secrets
import shlex
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, cast

import httpx

from .settings import get_settings
from .utils.file_utils import atomic_remove, atomic_write_json

logger = logging.getLogger(__name__)

# Default state store TTL (ms) - 10 minutes
STATE_TTL_MS = 10 * 60 * 1000


def now_ms() -> int:
    """Return current time in milliseconds since epoch."""
    return int(time.time() * 1000)


def generate_state() -> str:
    """Generate a random state string (hex 32 bytes)."""
    return secrets.token_hex(32)


def generate_code_verifier(length: int = 64) -> str:
    """
    Generate a high-entropy code verifier for PKCE flows.

    The verifier is a base64url string without padding derived from `length` random bytes.
    """
    raw = secrets.token_bytes(length)
    b64 = base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")
    return b64


def generate_code_challenge(verifier: str) -> str:
    """
    Generate the S256 code challenge for a given verifier.

    Code challenge algorithm: base64url_encode(sha256(verifier)) without padding.
    """
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return challenge


@dataclass
class StateEntry:
    """Representation of a stored authorization session state."""

    agent_id: str
    code_verifier: str
    client_id: Optional[str]
    redirect_uri: Optional[str]
    scope: Optional[str]
    created_at: int
    expires_at: int


class StateStore:
    """
    An in-memory TTL-backed store for OAuth state -> entry mappings.

    Usage pattern:
      - create: store.create(...)
      - consume: store.pop(state)  # returns entry or None
      - background cleanup can call store.cleanup()
    """

    def __init__(self) -> None:
        self._store: Dict[str, StateEntry] = {}
        self._lock = asyncio.Lock()

    async def create(
        self,
        state: str,
        agent_id: str,
        code_verifier: str,
        client_id: Optional[str],
        redirect_uri: Optional[str],
        scope: Optional[str],
        expires_at: int,
    ) -> None:
        entry = StateEntry(
            agent_id=agent_id,
            code_verifier=code_verifier,
            client_id=client_id,
            redirect_uri=redirect_uri,
            scope=scope,
            created_at=now_ms(),
            expires_at=expires_at,
        )
        async with self._lock:
            self._store[state] = entry
            logger.debug(
                "Stored PKCE state", extra={"state": state, "agent_id": agent_id}
            )

    async def pop(self, state: str) -> Optional[StateEntry]:
        """Pop and return the entry if present and not expired (returns None otherwise)."""
        async with self._lock:
            entry = self._store.pop(state, None)
        if not entry:
            return None
        if entry.expires_at <= now_ms():
            logger.debug(
                "PKCE state expired", extra={"state": state, "agent_id": entry.agent_id}
            )
            return None
        return entry

    async def get(self, state: str) -> Optional[StateEntry]:
        """Return entry if present and not expired (does not remove it)."""
        async with self._lock:
            entry = self._store.get(state)
        if not entry:
            return None
        if entry.expires_at <= now_ms():
            return None
        return entry

    async def cleanup(self) -> int:
        """Remove expired states and return the number of removed items."""
        removed = 0
        now = now_ms()
        async with self._lock:
            keys = list(self._store.keys())
            for k in keys:
                if self._store[k].expires_at <= now:
                    del self._store[k]
                    removed += 1
        if removed > 0:
            logger.debug("Cleaned up expired PKCE states", extra={"removed": removed})
        return removed


# Module-level state store
_state_store = StateStore()


def _get_settings() -> Any:
    return get_settings()


def _get_agent_codex_home(agent_id: str) -> Optional[Path]:
    """
    Resolve the codex_home path for the given agent.

    Current logic (single-agent focused):
      1. If `A2A_AGENT_CODEX_HOME` is set return it
      2. Inspect the `A2A_AGENT_COMMAND` for `--codex-home` argument
      3. Fall back to `A2A_DEFAULT_CODEX_HOME`
      4. If none found return None
    """
    settings = _get_settings()

    # 1) per-agent codex home from environment
    if getattr(settings, "agent_codex_home", None):
        return Path(settings.agent_codex_home)

    # 2) inspect agent_command for `--codex-home` tokens
    command_str = settings.agent_command or ""
    if command_str:
        try:
            parts = shlex.split(command_str)
            for idx, token in enumerate(parts):
                if token == "--codex-home" and idx + 1 < len(parts):
                    return Path(parts[idx + 1])
                if token.startswith("--codex-home="):
                    return Path(token.split("=", 1)[1])
        except Exception:
            # fallback to substring check if shlex parsing fails
            if "--codex-home=" in command_str:
                try:
                    # best effort parse
                    for token in command_str.split():
                        if token.startswith("--codex-home="):
                            return Path(token.split("=", 1)[1])
                except Exception:
                    pass

    # 3) fallback to default codex home from settings
    if getattr(settings, "default_codex_home", None):
        return Path(settings.default_codex_home)

    return None


def _auth_file_candidates(codex_home: Path) -> list[Path]:
    """
    Return ordered list of candidate auth file paths for the given codex_home.
    Order:
      1. <codex_home>/<agent_auth_file_name> (default)
      2. <codex_home>/.<agent_auth_file_name> (dot-prefixed)
      3. <codex_home>/openai-auth.json
    """
    settings = _get_settings()
    file_name = getattr(settings, "agent_auth_file_name", "auth.json")

    candidates: list[Path] = []
    if not codex_home:
        return candidates

    # Primary
    candidates.append(Path(codex_home) / file_name)

    # Dot-prefixed fallback, only if different from primary
    dot_name = f".{file_name}" if not file_name.startswith(".") else file_name
    if dot_name and dot_name != file_name:
        candidates.append(Path(codex_home) / dot_name)

    # Legacy OpenAI file name fallback
    if file_name != "openai-auth.json":
        candidates.append(Path(codex_home) / "openai-auth.json")

    return candidates


def _find_existing_auth_file(codex_home: Path) -> Optional[Path]:
    """Return the first existing auth file from candidate list or None."""
    for candidate in _auth_file_candidates(codex_home):
        try:
            if candidate.exists():
                return candidate
        except Exception:
            # If file system operations fail (e.g., permission), continue to next candidate.
            continue
    return None


def _auth_file_path(codex_home: Path) -> Path:
    """Return the absolute path to the auth file for a given codex_home (default path)."""
    settings = _get_settings()
    file_name = getattr(settings, "agent_auth_file_name", "auth.json")
    return Path(codex_home) / file_name


async def create_auth_session(
    agent_id: str,
    client_id: Optional[str] = None,
    redirect_uri: Optional[str] = None,
    scope: Optional[str] = None,
    ttl_ms: int = STATE_TTL_MS,
) -> Dict[str, Any]:
    """
    Create PKCE and state for OAuth authorization redirect.

    Returns:
        {
            "authorize_url": "<string>",
            "state": "<string>",
            "expires_at": <ms since epoch>
        }
    """
    settings = _get_settings()

    client_id = client_id or settings.oauth_client_id
    redirect_uri = redirect_uri or settings.oauth_redirect
    scope = scope or settings.oauth_scope

    # Generate pkce & state
    code_verifier = generate_code_verifier()
    code_challenge = generate_code_challenge(code_verifier)
    state = generate_state()
    expires_at = now_ms() + ttl_ms

    # Store state mapping
    await _state_store.create(
        state, agent_id, code_verifier, client_id, redirect_uri, scope, expires_at
    )

    # Build authorize URL
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": scope,
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }

    # Build query string safely
    from urllib.parse import urlencode

    authorize_url = f"{settings.oauth_authorize_url}?{urlencode(params)}"
    return {"authorize_url": authorize_url, "state": state, "expires_at": expires_at}


class OAuthError(Exception):
    """Generic OAuth-related error for oauth_manager operations."""


async def complete_auth_session(
    agent_id: str, code: str, state: str, client_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Complete an OAuth session by exchanging the authorization code for access/refresh tokens,
    writing `auth.json` into the agent `codex_home`, and returning a summary of the result.

    Returns:
        {
            "success": True,
            "agent_id": "<agent_id>",
            "account_id": "<account_id or None>",
            "expires": <ms since epoch>
        }

    Raises:
        OAuthError on invalid state / token exchange / write errors
    """
    settings = _get_settings()

    # Retrieve state entry (pop)
    entry = await _state_store.pop(state)
    if not entry:
        raise OAuthError("Invalid or expired state")

    if entry.agent_id != agent_id:
        raise OAuthError("State does not match agent")

    client_id = client_id or entry.client_id or settings.oauth_client_id
    redirect_uri = entry.redirect_uri or settings.oauth_redirect
    code_verifier = entry.code_verifier

    # Exchange authorization code for token
    token_url = settings.oauth_token_url

    data = {
        "grant_type": "authorization_code",
        "client_id": client_id,
        "code": code,
        "code_verifier": code_verifier,
        "redirect_uri": redirect_uri,
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            resp = await client.post(token_url, data=data, headers=headers)
            resp.raise_for_status()
            token_resp = resp.json()
    except httpx.HTTPError as exc:
        logger.debug("Token exchange HTTP error", exc_info=True)
        raise OAuthError(f"Token exchange failed: {str(exc)}")

    # Map token response into our auth.json schema
    access_token = token_resp.get("access_token")
    refresh_token = token_resp.get("refresh_token")
    expires_in = token_resp.get("expires_in")
    id_token = token_resp.get("id_token")
    # Optional: derive account_id if present (best-effort) - token provider specific
    account_id = None

    if access_token is None:
        raise OAuthError("Token response missing access_token")

    expires_ts = now_ms() + (int(expires_in) * 1000 if expires_in else 0)
    token_json = {
        "type": "oauth",
        "access": access_token,
        "refresh": refresh_token,
        "expires": expires_ts,
        "id_token": id_token,
        "account_id": account_id,
        "created_at": now_ms(),
    }

    # Determine codex_home for the agent and write file atomically (with fallbacks)
    codex_home = _get_agent_codex_home(agent_id)
    if not codex_home:
        raise OAuthError("Agent `codex_home` not configured for writing auth token")

    try:
        written_path = write_auth_file(codex_home, token_json)
        logger.info(
            "Wrote auth token file",
            extra={"agent_id": agent_id, "path": str(written_path)},
        )
    except Exception as exc:
        logger.exception(
            "Failed to write auth file to any candidate path",
            extra={"agent_id": agent_id, "codex_home": str(codex_home)},
        )
        raise OAuthError(f"Failed to persist token for agent: {str(exc)}")

    return {
        "success": True,
        "agent_id": agent_id,
        "account_id": account_id,
        "expires": expires_ts,
    }


def write_auth_file(codex_home: Path, token_json: Dict[str, Any]) -> Path:
    """Synchronous helper to write an auth token JSON with file locking and enforced permissions.

    Attempts writing to the default path first, then the dot-prefixed and legacy
    candidate names. Returns the Path that was successfully written to.
    Raises the last encountered exception if no candidate succeeded.
    """
    candidates = _auth_file_candidates(codex_home)
    last_exc: Optional[Exception] = None

    for cand in candidates:
        try:
            atomic_write_json(cand, token_json)
            return cand
        except Exception as exc:
            # Collect and continue on errors, but keep previous behavior by trying next candidate.
            last_exc = exc
            logger.debug(
                "Failed to write auth file to candidate",
                extra={"candidate": str(cand), "error": str(exc)},
            )
            continue

    # If none of the candidates worked, raise the last error to fail the operation.
    if last_exc:
        raise last_exc
    raise Exception("No auth file candidates available to write to")


def read_auth_file(codex_home: Path) -> Optional[Dict[str, Any]]:
    """Read the token JSON from `codex_home` if present, else return None.

    This function searches for authentication files using the candidate paths and returns
    JSON content from the first path that is present. This enables discovery of legacy
    or alternative token file names.
    """
    path = _find_existing_auth_file(codex_home)
    if not path or not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = cast(Dict[str, Any], json.load(f))
        return data
    except Exception:
        logger.exception("Failed to read auth file", extra={"path": str(path)})
        return None


def remove_auth_file(codex_home: Path) -> None:
    """Remove the auth file for the given codex_home.

    This removes any existing auth file found among the candidate names and logs the paths removed.
    The function will not raise if no candidate was found or if removal failed with missing_ok=True.
    """
    candidates = _auth_file_candidates(codex_home)
    removed_any = False
    for cand in candidates:
        if cand.exists():
            try:
                atomic_remove(cand, missing_ok=True)
                logger.info("Removed auth file", extra={"path": str(cand)})
                removed_any = True
            except Exception:
                logger.exception(
                    "Failed to remove auth file", extra={"path": str(cand)}
                )
                raise
    if not removed_any:
        # Nothing to remove; keep behavior idempotent.
        logger.debug(
            "No auth file found to remove", extra={"codex_home": str(codex_home)}
        )


def check_auth_file_valid(codex_home: Path) -> bool:
    """Return True if token exists and `expires` is in future."""
    token = read_auth_file(codex_home)
    if not token:
        return False
    expires = token.get("expires")
    if not isinstance(expires, (int, float)):
        return False
    return int(expires) > now_ms()


async def refresh_token_for_codex_home(
    codex_home: Path, client_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Refresh tokens using the stored `refresh` token in `<codex_home>/auth.json`.

    Returns the updated token JSON dict on success.

    Raises OAuthError on failures.
    """
    settings = _get_settings()
    token_info = read_auth_file(codex_home)
    if not token_info:
        raise OAuthError("No token file found to refresh")

    refresh_token = token_info.get("refresh")
    if not refresh_token:
        raise OAuthError("No refresh token available")

    cid = client_id or settings.oauth_client_id
    token_url = settings.oauth_token_url

    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": cid,
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            resp = await client.post(token_url, data=data, headers=headers)
            resp.raise_for_status()
            token_resp = resp.json()
    except httpx.HTTPError as exc:
        logger.debug("Token refresh HTTP error", exc_info=True)
        raise OAuthError(f"Token refresh failed: {str(exc)}")

    new_access = token_resp.get("access_token")
    new_refresh = token_resp.get(
        "refresh_token", refresh_token
    )  # update if provider returned new refresh
    expires_in = token_resp.get("expires_in")
    id_token = token_resp.get("id_token", token_info.get("id_token"))

    if not new_access:
        raise OAuthError("Token refresh response missing access_token")

    updated_token_json = {
        "type": "oauth",
        "access": new_access,
        "refresh": new_refresh,
        "expires": now_ms() + (int(expires_in) * 1000 if expires_in else 0),
        "id_token": id_token,
        "account_id": token_info.get("account_id"),
        "created_at": now_ms(),
    }

    # Persist updated token: prefer writing back to the same existing auth file if present,
    # otherwise use the default/candidate write path.
    try:
        existing = _find_existing_auth_file(codex_home)
        if existing is not None:
            atomic_write_json(existing, updated_token_json)
            written_path = existing
        else:
            written_path = write_auth_file(codex_home, updated_token_json)
        logger.info(
            "Refreshed auth token and persisted new token",
            extra={"path": str(written_path)},
        )
    except Exception as exc:
        logger.exception(
            "Failed to write refreshed auth file",
            extra={
                "path": str(
                    _find_existing_auth_file(codex_home) or _auth_file_path(codex_home)
                )
            },
        )
        raise OAuthError(f"Failed to persist refreshed token: {str(exc)}")

    return updated_token_json


async def get_token_info_for_agent(agent_id: str) -> Optional[Dict[str, Any]]:
    """
    Convenience wrapper to return token JSON for an agent based on codex_home resolution.

    Additionally, this function performs an automatic refresh of tokens when the token's
    'expires' time is within the auto-refresh margin (default 5 minutes).
    If a refresh is successful, the updated token JSON (persisted on disk by the refresh)
    is returned; otherwise, the previously persisted token info is returned (or None).
    """
    codex_home = _get_agent_codex_home(agent_id)
    if not codex_home:
        return None

    # Read the current token (detecting fallback auth file names).
    token_info = read_auth_file(codex_home)
    if not token_info:
        return None

    # Auto-refresh window: if the token expires within this margin (in ms), attempt refresh.
    settings = _get_settings()
    AUTO_REFRESH_MARGIN_MS = (
        int(getattr(settings, "oauth_auto_refresh_margin", 300)) * 1000
    )

    try:
        expires = token_info.get("expires")
        if isinstance(expires, (int, float)):
            # If the token is about to expire (<= 5 minutes), attempt refresh
            if int(expires) - now_ms() <= AUTO_REFRESH_MARGIN_MS:
                logger.debug(
                    "Token near expiry; attempting auto-refresh",
                    extra={"codex_home": str(codex_home)},
                )
                try:
                    refreshed = await refresh_token_for_codex_home(codex_home)
                    # Return the refreshed token dict
                    return refreshed
                except OAuthError:
                    # Refresh failed — log the failure but return the previously persisted token
                    logger.debug(
                        "Auto-refresh failed; returning existing token",
                        extra={"codex_home": str(codex_home)},
                    )
                except Exception:
                    # For any other unexpected failure, log it and continue returning existing token
                    logger.exception(
                        "Unexpected error during auto-refresh",
                        extra={"codex_home": str(codex_home)},
                    )
    except Exception:
        # If anything goes wrong computing expiry or other checks, fall back to returning the token_info
        logger.exception(
            "Failed to determine token expiry for auto-refresh check",
            extra={"codex_home": str(codex_home)},
        )

    return token_info


# State cleanup: optionally the server can call this periodically to remove expired states
async def cleanup_expired_states() -> int:
    """Cleanup expired PKCE states and return the number removed."""
    return await _state_store.cleanup()


async def housekeeping_pass(agent_id: str = "default") -> Dict[str, int]:
    """
    Perform a single housekeeping pass:
      - Cleanup expired PKCE states.
      - If OAuth is enabled, check tokens and refresh those which are within the auto-refresh margin.

    Returns:
        {
            "states_removed": <int>,
            "tokens_refreshed": <int>
        }
    """
    settings = _get_settings()
    states_removed = await _state_store.cleanup()
    tokens_refreshed = 0

    # If OAuth is disabled or no codex_home is present for the single agent, nothing to do
    if not getattr(settings, "agent_oauth_enabled", False):
        return {"states_removed": states_removed, "tokens_refreshed": tokens_refreshed}

    codex_home = _get_agent_codex_home(agent_id)
    if not codex_home:
        return {"states_removed": states_removed, "tokens_refreshed": tokens_refreshed}

    # Read token using the fallback detector
    token_info = read_auth_file(codex_home)
    if not token_info:
        return {"states_removed": states_removed, "tokens_refreshed": tokens_refreshed}

    try:
        expires = token_info.get("expires")
        margin_ms = int(getattr(settings, "oauth_auto_refresh_margin", 300)) * 1000
        if isinstance(expires, (int, float)) and expires <= now_ms() + margin_ms:
            # Attempt a refresh; refresh_token_for_codex_home will raise OAuthError if this isn't
            # possible (e.g. refresh token missing) or other HTTP errors occur.
            try:
                await refresh_token_for_codex_home(codex_home)
                tokens_refreshed += 1
                logger.info(
                    "OAuth housekeeping refreshed token",
                    extra={"codex_home": str(codex_home)},
                )
            except Exception as exc:
                # Non-fatal: log details and continue with the housekeeping pass. Avoid logging tokens.
                logger.debug(
                    "OAuth housekeeping token refresh failed",
                    extra={"codex_home": str(codex_home), "error": str(exc)},
                )
    except Exception:
        logger.exception(
            "OAuth housekeeping failed to determine token expiry",
            extra={"codex_home": str(codex_home)},
        )

    return {"states_removed": states_removed, "tokens_refreshed": tokens_refreshed}
