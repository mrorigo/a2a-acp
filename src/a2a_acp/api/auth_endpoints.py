"""
OAuth-related endpoints for ChatGPT/OpenAI integration with codex-acp.

Endpoints (under prefix /a2a/agents/{agent_id}/auth):
- POST /start -> Generate PKCE state and authorize URL
- GET /callback -> OAuth callback (code & state)
- POST /manual -> Manual paste fallback for headless servers
- GET /status -> Inspect token presence and expiry
- DELETE / -> Remove token file
- POST /refresh -> Force token refresh
- POST /check-auth -> Ask agent to initialize & authenticate to verify token

Notes:
- This module expects oauth_manager functions to be available (async).
- `ZedAgentConnection` is reused to drive `initialize` & `authenticate`.
- Authorization is checked using the same approach as in main; requires `A2A_AUTH_TOKEN`
  unless one is not configured (i.e., in dev mode).
"""

from __future__ import annotations

import logging
import shlex
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from a2a_acp.audit import AuditEventType, log_oauth_event
from a2a_acp.oauth_manager import (
    _get_agent_codex_home,  # internal helper, used to detect codex_home
    complete_auth_session,
    create_auth_session,
    get_token_info_for_agent,
    refresh_token_for_codex_home,
    remove_auth_file,
)
from a2a_acp import settings as settings_module
from a2a_acp.zed_agent import ZedAgentConnection

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/a2a/agents", tags=["agent-auth"])


#
# Helper / Models
#
class StartRequest(BaseModel):
    client_id: Optional[str] = None
    redirect_uri: Optional[str] = None
    scope: Optional[str] = None


class ManualAuthRequest(BaseModel):
    callback_url: str
    state: Optional[str] = None


def _require_authorization(authorization: Optional[str] = Header(default=None)) -> None:
    """
    Dependency to enforce authorization header based on A2A_AUTH_TOKEN in settings.

    If no `A2A_AUTH_TOKEN` is configured, the request is allowed (development mode).
    """
    settings = settings_module.get_settings()
    token = settings.auth_token
    if not isinstance(token, str) or not token:
        # dev mode; don't enforce auth
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token"
        )
    provided = authorization.split(" ", 1)[1]
    if provided != token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid bearer token"
        )


def _resolve_agent_command_and_api_key() -> tuple[list[str], Optional[str]]:
    """
    Re-implementation of the minimal `get_agent_config` parsing to avoid circular imports.
    Returns (command_list, api_key)
    """
    settings = settings_module.get_settings()
    command_str = settings.agent_command or "python tests/dummy_agent.py"
    api_key = settings.agent_api_key
    try:
        command = shlex.split(command_str)
    except ValueError:
        # Fallback: treat entire string as a shell command
        command = [command_str]
    return command, api_key


def _get_codex_home_for_agent(agent_id: str):
    """Helper that wraps oauth_manager _get_agent_codex_home with fallback semantics."""
    # Delegated to oauth_manager helper to ensure single source of truth
    codex_home = _get_agent_codex_home(agent_id)
    return codex_home


#
# Endpoints
#


@router.post("/{agent_id}/auth/start")
async def auth_start(
    agent_id: str,
    body: StartRequest,
    authorization: Optional[str] = Depends(_require_authorization),
) -> Dict[str, Any]:
    """
    Generate authorization URL & start session using PKCE.

    Request JSON:
      { "client_id": "<optional>", "redirect_uri": "<optional>", "scope": "<optional>" }

    Response:
      200:
        {
          "authorize_url": "<string>",
          "state": "<string>",
          "expires_at": <ms since epoch>
        }
    """
    settings = settings_module.get_settings()
    if not settings.agent_oauth_enabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="OAuth flows are disabled on this server",
        )

    # Ensure that codex_home is configured for this agent before starting the flow.
    codex_home = _get_codex_home_for_agent(agent_id)
    if not codex_home:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="agent_missing_codex_home",
        )

    try:
        payload = await create_auth_session(
            agent_id=agent_id,
            client_id=body.client_id,
            redirect_uri=body.redirect_uri,
            scope=body.scope,
        )
        await log_oauth_event(
            AuditEventType.OAUTH_LOGIN_STARTED,
            agent_id,
            {"state": payload.get("state")},
        )
    except Exception as exc:
        logger.exception(
            "Failed to create authorization session", extra={"agent_id": agent_id}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )

    return payload


@router.get("/{agent_id}/auth/callback", response_model=None)
async def auth_callback(
    request: Request,
    agent_id: str,
    code: Optional[str] = None,
    state: Optional[str] = None,
    authorization: Optional[str] = Depends(_require_authorization),
) -> Dict[str, Any]:
    """
    OAuth callback endpoint. Query parameters: `code` and `state`.

    On success writes the `auth.json` token into codex_home and tries to ask the agent to authenticate
    with `authenticate("chatgpt")`. If the agent rejects the token, the file is removed and an internal
    error is returned.

    Responses:
      200 (success): { "success": true, "agent_id": "<>", "account_id": "<>|null", "expires": <ms> }
      400: invalid request
      401: unauthorized (invalid token exchange)
      500: internal error writing token or agent rejects
    """
    if not code or not state:
        await log_oauth_event(
            AuditEventType.OAUTH_LOGIN_FAILED,
            agent_id,
            {"phase": "missing_code_or_state"},
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Missing code or state"
        )

    settings = settings_module.get_settings()
    if not settings.agent_oauth_enabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="OAuth flows are disabled"
        )

    # Ensure codex_home configured
    codex_home = _get_codex_home_for_agent(agent_id)
    if not codex_home:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="agent_missing_codex_home"
        )

    try:
        result = await complete_auth_session(agent_id=agent_id, code=code, state=state)
    except Exception as exc:
        logger.exception(
            "OAuth complete flow failed", extra={"agent_id": agent_id, "state": state}
        )
        await log_oauth_event(
            AuditEventType.OAUTH_LOGIN_FAILED,
            agent_id,
            {"state": state, "error": str(exc)},
        )
        # Consider returning 401 if code/state invalid or 500 otherwise
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc))

    # Now validate with the agent by calling initialize & authenticate("chatgpt")
    try:
        command, api_key = _resolve_agent_command_and_api_key()
        async with ZedAgentConnection(command, api_key=api_key) as conn:
            await conn.initialize()
            # Call authenticate("chatgpt") - the agent should read token file on its own
            await conn.authenticate("chatgpt")
    except Exception as exc:
        # Agent rejected the token; remove the token file and return error per plan
        try:
            remove_auth_file(codex_home)
        except Exception:
            logger.exception(
                "Failed to remove token after agent rejected it",
                extra={"codex_home": str(codex_home)},
            )
        logger.exception("Agent did not accept token", extra={"agent_id": agent_id})
        await log_oauth_event(
            AuditEventType.OAUTH_LOGIN_FAILED,
            agent_id,
            {"phase": "agent_rejected", "state": state, "error": str(exc)},
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"agent_not_accepting_auth: {str(exc)}",
        )

    await log_oauth_event(
        AuditEventType.OAUTH_LOGIN_SUCCESS,
        agent_id,
        {"state": state, "expires": result.get("expires")},
    )

    # Success: return details
    return {
        "success": True,
        "agent_id": agent_id,
        "account_id": result.get("account_id") if isinstance(result, dict) else None,
        "expires": result.get("expires") if isinstance(result, dict) else None,
    }


@router.post("/{agent_id}/auth/manual", response_model=None)
async def auth_manual(
    request: Request,
    agent_id: str,
    body: ManualAuthRequest,
    authorization: Optional[str] = Depends(_require_authorization),
) -> Dict[str, Any]:
    """
    Manual callback paste support for headless servers.
    Accepts:
      { "callback_url": "http://host:port/auth/callback?code=...&state=...", "state": "<optional>" }
    """
    settings = settings_module.get_settings()
    if not settings.agent_oauth_enabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="OAuth flows disabled"
        )

    # Extract code/state from callback_url
    try:
        parsed = urlparse(body.callback_url)
        params = parse_qs(parsed.query)
        code = params.get("code", [None])[0]
        state = params.get("state", [body.state])[0]
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse callback_url: {exc}",
        )

    if not code or not state:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing code or state in callback_url",
        )

    # Reuse callback handling logic
    return await auth_callback(
        request=request,
        agent_id=agent_id,
        code=code,
        state=state,
        authorization=authorization,
    )


@router.get("/{agent_id}/auth/status")
async def auth_status(
    agent_id: str,
    authorization: Optional[str] = Depends(_require_authorization),
) -> Dict[str, Any]:
    """
    Return whether the agent has a valid token stored and the token's expiry info.

    Response:
      {
        "signed_in": true|false,
        "expires": <msSinceEpoch|null>,
        "account_id": "<string|null>",
        "authentication_method": "chatgpt"
      }
    """
    settings_module.get_settings()
    codex_home = _get_codex_home_for_agent(agent_id)
    if not codex_home:
        return {
            "signed_in": False,
            "expires": None,
            "account_id": None,
            "authentication_method": "chatgpt",
        }

    # Use get_token_info_for_agent (async) - this will automatically refresh tokens if they are
    # nearing expiry and return the up-to-date token JSON if successful.
    token = await get_token_info_for_agent(agent_id)

    signed_in = bool(token and token.get("access"))
    expires = token.get("expires") if token else None
    account_id = token.get("account_id") if token else None

    return {
        "signed_in": signed_in,
        "expires": expires,
        "account_id": account_id,
        "authentication_method": "chatgpt",
    }


@router.delete("/{agent_id}/auth")
async def auth_delete(
    agent_id: str,
    authorization: Optional[str] = Depends(_require_authorization),
):
    """
    Remove stored token file for the agent.
    Responses:
      204: success
      404: token not found
    """
    settings_module.get_settings()
    codex_home = _get_codex_home_for_agent(agent_id)
    if not codex_home:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="agent_missing_codex_home"
        )

    # Check existence first using `get_token_info_for_agent` to allow fallback file names
    token = await get_token_info_for_agent(agent_id)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="No token to delete"
        )

    try:
        # Remove any existing auth file found among candidate names (default, dot-prefixed, legacy)
        remove_auth_file(Path(codex_home))
        await log_oauth_event(AuditEventType.OAUTH_TOKEN_REMOVED, agent_id)
    except Exception as exc:
        logger.exception("Failed to delete auth file", extra={"agent_id": agent_id})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        )

    # Confirm agent's new auth status: call initialize/authenticate to confirm the removal is recognized
    try:
        command, api_key = _resolve_agent_command_and_api_key()
        async with ZedAgentConnection(command, api_key=api_key) as conn:
            await conn.initialize()
            await conn.authenticate("chatgpt")
    except Exception:
        # If authenticate fails, it's acceptable since we've already removed the token
        logger.debug(
            "Agent authenticate failed after removal (expected); continuing",
            extra={"agent_id": agent_id},
        )

    return JSONResponse(status_code=status.HTTP_204_NO_CONTENT, content=None)


@router.post("/{agent_id}/auth/refresh")
async def auth_refresh(
    agent_id: str,
    authorization: Optional[str] = Depends(_require_authorization),
) -> Dict[str, Any]:
    """
    Force token refresh for the agent using the stored `refresh` token.
    """
    settings = settings_module.get_settings()
    if not settings.agent_oauth_enabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="OAuth flows disabled"
        )

    codex_home = _get_codex_home_for_agent(agent_id)
    if not codex_home:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="agent_missing_codex_home"
        )

    try:
        new_token = await refresh_token_for_codex_home(Path(codex_home))
        await log_oauth_event(
            AuditEventType.OAUTH_REFRESH_SUCCESS,
            agent_id,
            {"expires": new_token.get("expires")},
        )
    except Exception as exc:
        logger.exception("Token refresh failed", extra={"agent_id": agent_id})
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc))

    return {
        "agent_id": agent_id,
        "expires": new_token.get("expires"),
        "created_at": new_token.get("created_at"),
    }


@router.post("/{agent_id}/auth/check-auth")
async def auth_check_auth(
    agent_id: str,
    authorization: Optional[str] = Depends(_require_authorization),
):
    """
    Ask the agent to initialize and authenticate("chatgpt") to confirm the agent accepts tokens.
    - Returns 200 if agent accepts
    - 401 if agent rejects (token missing/invalid)
    - 500 for internal errors
    """
    settings_module.get_settings()
    codex_home = _get_codex_home_for_agent(agent_id)
    if not codex_home:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="agent_missing_codex_home"
        )

    # If no token, return 401. Use the async helper which performs auto-refresh for nearing-expiry tokens.
    token = await get_token_info_for_agent(agent_id)
    if not token or not token.get("access"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Token missing"
        )

    command, api_key = _resolve_agent_command_and_api_key()
    try:
        async with ZedAgentConnection(command, api_key=api_key) as conn:
            await conn.initialize()
            await conn.authenticate("chatgpt")
    except Exception as exc:
        logger.exception(
            "Agent rejected token during check-auth", extra={"agent_id": agent_id}
        )
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc))

    # Agent accepted authentication
    return {"success": True, "agent_id": agent_id}


# Export router for consumption in main.create_app()
__all__ = ["router"]
