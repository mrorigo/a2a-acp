"""
Unit tests for a2a_acp.settings, focusing on defaults and environment overrides.

Tests included:
- Default values of ChatGPT/OAuth related settings
- Environment variable overrides
- Validation behavior when OAuth is enabled but codex_home is missing
- Parsing codex_home from the A2A_AGENT_COMMAND via get_agent_config()
"""

from __future__ import annotations

from pathlib import Path

import pytest

from a2a_acp import main as main_mod
from a2a_acp.settings import get_settings


def _clear_settings_cache():
    """Helper to clear lru_cache on get_settings() between tests."""
    try:
        get_settings.cache_clear()
    except AttributeError:
        # If the function is not cached (already cleared), it's fine
        pass


def test_settings_defaults(monkeypatch):
    """When no override env vars are set expect the default values to apply."""
    # Make sure the agent command is present to avoid validation errors
    monkeypatch.setenv("A2A_AGENT_COMMAND", "true")

    # Ensure all relevant OAuth env vars are unset to test defaults
    for k in [
        "A2A_OAUTH_CLIENT_ID",
        "A2A_OAUTH_AUTHORIZE_URL",
        "A2A_OAUTH_TOKEN_URL",
        "A2A_OAUTH_SCOPE",
        "A2A_OAUTH_REDIRECT",
        "A2A_AGENT_OAUTH_ENABLED",
        "A2A_AGENT_AUTH_FILE_NAME",
        "A2A_DEFAULT_CODEX_HOME",
        "A2A_AGENT_CODEX_HOME",
    ]:
        monkeypatch.delenv(k, raising=False)

    _clear_settings_cache()
    settings = get_settings()

    # Defaults as defined in the CHATGPT_AUTH_PLAN
    assert settings.oauth_client_id == "app_EMoamEEZ73f0CkXaXp7hrann"
    assert settings.oauth_authorize_url == "https://auth.openai.com/oauth/authorize"
    assert settings.oauth_token_url == "https://auth.openai.com/oauth/token"
    assert settings.oauth_scope == "openid profile email offline_access"
    assert (
        settings.oauth_redirect
        == "http://localhost:8001/a2a/agents/{agent_id}/auth/callback"
    )
    assert settings.agent_oauth_enabled is False
    assert settings.agent_auth_file_name == "auth.json"
    assert settings.default_codex_home is None
    assert settings.agent_codex_home is None


def test_settings_env_overrides(monkeypatch):
    """Check that environment variables override default settings values."""
    monkeypatch.setenv("A2A_AGENT_COMMAND", "true")
    monkeypatch.setenv("A2A_OAUTH_CLIENT_ID", "override_clientid")
    monkeypatch.setenv("A2A_OAUTH_AUTHORIZE_URL", "https://example.com/oauth/authorize")
    monkeypatch.setenv("A2A_OAUTH_TOKEN_URL", "https://example.com/oauth/token")
    monkeypatch.setenv("A2A_OAUTH_SCOPE", "openid email")
    monkeypatch.setenv("A2A_OAUTH_REDIRECT", "http://localhost/callback/some-agent")
    monkeypatch.setenv("A2A_AGENT_OAUTH_ENABLED", "true")
    monkeypatch.setenv("A2A_AGENT_AUTH_FILE_NAME", "override_auth.json")
    monkeypatch.setenv("A2A_AGENT_CODEX_HOME", "/tmp/codex-instance")
    monkeypatch.setenv("A2A_DEFAULT_CODEX_HOME", "/opt/codex-default")

    _clear_settings_cache()
    settings = get_settings()

    assert settings.oauth_client_id == "override_clientid"
    assert settings.oauth_authorize_url == "https://example.com/oauth/authorize"
    assert settings.oauth_token_url == "https://example.com/oauth/token"
    assert settings.oauth_scope == "openid email"
    assert settings.oauth_redirect == "http://localhost/callback/some-agent"
    assert settings.agent_oauth_enabled is True
    assert settings.agent_auth_file_name == "override_auth.json"
    assert isinstance(settings.agent_codex_home, Path)
    assert str(settings.agent_codex_home) == "/tmp/codex-instance"
    assert isinstance(settings.default_codex_home, Path)
    assert str(settings.default_codex_home) == "/opt/codex-default"


def test_oauth_requires_codex_home_raises(monkeypatch):
    """
    get_settings() should raise Configuration validation ValueError when
    A2A_AGENT_OAUTH_ENABLED=true and no codex_home is configured via:
      - A2A_AGENT_CODEX_HOME
      - `--codex-home` in A2A_AGENT_COMMAND
      - or A2A_DEFAULT_CODEX_HOME
    """
    # Set agent command to something benign (no codex-home flag)
    monkeypatch.setenv("A2A_AGENT_COMMAND", "my-agent --version")

    # Ensure default & agent codex home values are *not* present
    monkeypatch.delenv("A2A_AGENT_CODEX_HOME", raising=False)
    monkeypatch.delenv("A2A_DEFAULT_CODEX_HOME", raising=False)

    # Enable OAuth gating
    monkeypatch.setenv("A2A_AGENT_OAUTH_ENABLED", "true")

    # The oauth client ID default is present so we don't need to set it;
    # however the lack of a codex_home should cause a validation error.
    _clear_settings_cache()
    with pytest.raises(ValueError):
        # get_settings() validates and should error since no codex_home source
        # exists.
        get_settings()


def test_oauth_allowed_with_default_codex_home(monkeypatch):
    """If A2A_DEFAULT_CODEX_HOME is present and OAuth gating is enabled, settings should be valid."""
    monkeypatch.setenv("A2A_AGENT_COMMAND", "my-agent --version")
    monkeypatch.setenv("A2A_AGENT_OAUTH_ENABLED", "true")

    # Provide default codex home (global fallback)
    monkeypatch.setenv("A2A_DEFAULT_CODEX_HOME", "/opt/codex-fallback")

    _clear_settings_cache()
    settings = get_settings()
    assert settings.agent_oauth_enabled is True
    assert isinstance(settings.default_codex_home, Path)
    assert str(settings.default_codex_home) == "/opt/codex-fallback"


def test_agent_command_codex_home_parsing(monkeypatch):
    """If the A2A_AGENT_COMMAND contains `--codex-home` it should be parsed into get_agent_config() result."""
    # A2A_AGENT_COMMAND contains flag format `--codex-home /path` (space separated)
    monkeypatch.setenv(
        "A2A_AGENT_COMMAND",
        "/usr/local/bin/codex-acp --codex-home /var/lib/codex-instance --debug",
    )
    monkeypatch.setenv("A2A_AGENT_OAUTH_ENABLED", "true")
    monkeypatch.setenv("A2A_AGENT_CODEX_HOME", "")  # ensure env override not present
    monkeypatch.delenv("A2A_AGENT_CODEX_HOME", raising=False)

    # Clear any cached settings
    _clear_settings_cache()
    # get_agent_config parses A2A_AGENT_COMMAND; it will use get_settings internally.
    agent_cfg = main_mod.get_agent_config()
    assert "codex_home" in agent_cfg
    assert agent_cfg["codex_home"] == "/var/lib/codex-instance"

    # Now check support for flag-style `--codex-home=/path` (equals syntax)
    monkeypatch.setenv(
        "A2A_AGENT_COMMAND",
        "/usr/local/bin/codex-acp --codex-home=/var/lib/codex-equals",
    )
    _clear_settings_cache()
    agent_cfg2 = main_mod.get_agent_config()
    assert "codex_home" in agent_cfg2
    assert agent_cfg2["codex_home"] == "/var/lib/codex-equals"
