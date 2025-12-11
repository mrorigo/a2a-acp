from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from .error_profiles import ErrorProfile, parse_error_profile


@dataclass(frozen=True)
class PushNotificationSettings:
    """Push notification specific settings."""

    # Core settings
    enabled: bool
    webhook_timeout: float
    retry_attempts: int
    retry_delay: float
    max_retry_delay: float

    # Batch processing
    batch_size: int
    max_concurrent_notifications: int

    # Cleanup configuration
    cleanup_enabled: bool
    cleanup_interval: int  # seconds
    retention_completed_hours: int
    retention_failed_hours: int
    retention_auth_required_hours: int

    # Security
    hmac_secret: Optional[str]
    rate_limit_per_minute: int

    # Real-time streaming
    streaming_enabled: bool
    max_websocket_connections: int
    max_sse_connections: int
    connection_cleanup_interval: int

    # Performance monitoring
    enable_metrics: bool
    metrics_retention_hours: int


@dataclass(frozen=True)
class Settings:
    """Simple settings container sourced from environment variables."""

    auth_token: Optional[str]
    agents_config_path: Path

    # Single agent configuration (streamlined approach)
    agent_command: Optional[str]
    agent_api_key: Optional[str]
    agent_description: str

    # Per-agent and default codex home fallback
    default_codex_home: Optional[Path]
    agent_codex_home: Optional[Path]

    # ChatGPT/OpenAI OAuth Settings (A2A_OAUTH_*)
    oauth_client_id: Optional[str]
    oauth_authorize_url: str
    oauth_token_url: str
    oauth_scope: str
    oauth_redirect: str
    agent_oauth_enabled: bool
    agent_auth_file_name: str
    # Housekeeping configuration for OAuth flows:
    # - `oauth_housekeeping_interval` defines how often (in seconds) the background
    #   housekeeping job should run (cleanup states, attempt proactive refreshes).
    # - `oauth_auto_refresh_margin` defines the threshold (in seconds) before expiry
    #   where the system should attempt a proactive refresh.
    oauth_housekeeping_interval: int
    oauth_auto_refresh_margin: int

    # Push notification settings
    push_notifications: PushNotificationSettings

    # Error handling profile
    error_profile: ErrorProfile

    # Development tool extension
    development_tool_extension_enabled: bool = True


def _get_push_notification_settings() -> PushNotificationSettings:
    """Get push notification specific settings from environment variables."""
    return PushNotificationSettings(
        # Core settings
        enabled=os.getenv("PUSH_NOTIFICATIONS_ENABLED", "true").lower() == "true",
        webhook_timeout=float(os.getenv("PUSH_NOTIFICATION_WEBHOOK_TIMEOUT", "30")),
        retry_attempts=int(os.getenv("PUSH_NOTIFICATION_RETRY_ATTEMPTS", "3")),
        retry_delay=float(os.getenv("PUSH_NOTIFICATION_RETRY_DELAY", "1.0")),
        max_retry_delay=float(os.getenv("PUSH_NOTIFICATION_MAX_RETRY_DELAY", "60.0")),
        # Batch processing
        batch_size=int(os.getenv("PUSH_NOTIFICATION_BATCH_SIZE", "10")),
        max_concurrent_notifications=int(
            os.getenv("PUSH_NOTIFICATION_MAX_CONCURRENT", "50")
        ),
        # Cleanup configuration
        cleanup_enabled=os.getenv("PUSH_NOTIFICATION_CLEANUP_ENABLED", "true").lower()
        == "true",
        cleanup_interval=int(os.getenv("PUSH_NOTIFICATION_CLEANUP_INTERVAL", "3600")),
        retention_completed_hours=int(
            os.getenv("PUSH_NOTIFICATION_RETENTION_COMPLETED_HOURS", "24")
        ),
        retention_failed_hours=int(
            os.getenv("PUSH_NOTIFICATION_RETENTION_FAILED_HOURS", "0")
        ),
        retention_auth_required_hours=int(
            os.getenv("PUSH_NOTIFICATION_RETENTION_AUTH_REQUIRED_HOURS", "168")
        ),
        # Security
        hmac_secret=os.getenv("PUSH_NOTIFICATION_HMAC_SECRET"),
        rate_limit_per_minute=int(
            os.getenv("PUSH_NOTIFICATION_RATE_LIMIT_PER_MINUTE", "60")
        ),
        # Real-time streaming
        streaming_enabled=os.getenv(
            "PUSH_NOTIFICATION_STREAMING_ENABLED", "true"
        ).lower()
        == "true",
        max_websocket_connections=int(
            os.getenv("PUSH_NOTIFICATION_MAX_WS_CONNECTIONS", "100")
        ),
        max_sse_connections=int(
            os.getenv("PUSH_NOTIFICATION_MAX_SSE_CONNECTIONS", "200")
        ),
        connection_cleanup_interval=int(
            os.getenv("PUSH_NOTIFICATION_CONNECTION_CLEANUP_INTERVAL", "300")
        ),
        # Performance monitoring
        enable_metrics=os.getenv("PUSH_NOTIFICATION_ENABLE_METRICS", "true").lower()
        == "true",
        metrics_retention_hours=int(
            os.getenv("PUSH_NOTIFICATION_METRICS_RETENTION_HOURS", "72")
        ),
    )


def validate_settings(settings: Settings) -> None:
    """Validate settings and raise ValueError for invalid configurations."""
    errors = []

    # Validate agent configuration
    if not settings.agent_command:
        errors.append("A2A_AGENT_COMMAND environment variable is required")

    # Validate push notification settings
    push_settings = settings.push_notifications

    if push_settings.enabled:
        if push_settings.webhook_timeout <= 0:
            errors.append("PUSH_NOTIFICATION_WEBHOOK_TIMEOUT must be positive")

        if push_settings.retry_attempts < 0:
            errors.append("PUSH_NOTIFICATION_RETRY_ATTEMPTS must be non-negative")

        if push_settings.retry_delay <= 0:
            errors.append("PUSH_NOTIFICATION_RETRY_DELAY must be positive")

        if push_settings.batch_size <= 0:
            errors.append("PUSH_NOTIFICATION_BATCH_SIZE must be positive")

        if push_settings.max_concurrent_notifications <= 0:
            errors.append("PUSH_NOTIFICATION_MAX_CONCURRENT must be positive")

        if push_settings.cleanup_interval <= 0:
            errors.append("PUSH_NOTIFICATION_CLEANUP_INTERVAL must be positive")

        if push_settings.rate_limit_per_minute <= 0:
            errors.append("PUSH_NOTIFICATION_RATE_LIMIT_PER_MINUTE must be positive")

        if push_settings.max_websocket_connections <= 0:
            errors.append("PUSH_NOTIFICATION_MAX_WS_CONNECTIONS must be positive")

        if push_settings.max_sse_connections <= 0:
            errors.append("PUSH_NOTIFICATION_MAX_SSE_CONNECTIONS must be positive")

        # Validate retention policies make sense
        if push_settings.retention_failed_hours < 0:
            errors.append(
                "PUSH_NOTIFICATION_RETENTION_FAILED_HOURS must be non-negative"
            )

        if push_settings.retention_completed_hours < 0:
            errors.append(
                "PUSH_NOTIFICATION_RETENTION_COMPLETED_HOURS must be non-negative"
            )

        if push_settings.retention_auth_required_hours < 0:
            errors.append(
                "PUSH_NOTIFICATION_RETENTION_AUTH_REQUIRED_HOURS must be non-negative"
            )

        # Warning: if HMAC secret is not set, webhook signatures cannot be validated
        if not push_settings.hmac_secret:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                "PUSH_NOTIFICATION_HMAC_SECRET not set - webhook signature validation disabled"
            )

    # Validate OAuth gating and minimal OAuth settings
    if getattr(settings, "agent_oauth_enabled", False):
        if not getattr(settings, "oauth_client_id", None):
            errors.append(
                "A2A_OAUTH_CLIENT_ID must be set when A2A_AGENT_OAUTH_ENABLED is true"
            )
        if not getattr(settings, "oauth_authorize_url", None) or not getattr(
            settings, "oauth_token_url", None
        ):
            errors.append(
                "A2A_OAUTH_AUTHORIZE_URL and A2A_OAUTH_TOKEN_URL must be set when A2A_AGENT_OAUTH_ENABLED is true"
            )

        # Validate housekeeping interval and auto-refresh margin values
        if getattr(settings, "oauth_housekeeping_interval", 0) <= 0:
            errors.append("A2A_OAUTH_HOUSEKEEPING_INTERVAL must be positive")
        if getattr(settings, "oauth_auto_refresh_margin", 0) < 0:
            errors.append("A2A_OAUTH_AUTO_REFRESH_MARGIN must be non-negative")

        # Ensure codex_home is present when OAuth is enabled:
        # - per-agent codex home (A2A_AGENT_CODEX_HOME) OR
        # - repo-wide default (A2A_DEFAULT_CODEX_HOME) OR
        # - `--codex-home <path>` provided on `A2A_AGENT_COMMAND`
        def _command_has_codex_home(command: Optional[str]) -> bool:
            """True if command contains a `--codex-home` flag (either `--codex-home <dir>` or `--codex-home=<dir>`)."""
            if not command:
                return False
            try:
                import shlex

                parts = shlex.split(command)
            except Exception:
                # Fallback to a substring check if parsing fails for any reason
                return "--codex-home" in command

            for i, p in enumerate(parts):
                if p == "--codex-home":
                    # presence of the flag with a following argument is valid
                    return i + 1 < len(parts)
                if p.startswith("--codex-home="):
                    return True
            return False

        if not (
            getattr(settings, "agent_codex_home", None)
            or getattr(settings, "default_codex_home", None)
            or _command_has_codex_home(getattr(settings, "agent_command", None))
        ):
            errors.append(
                "codex_home must be configured when A2A_AGENT_OAUTH_ENABLED is true: set A2A_AGENT_CODEX_HOME, A2A_DEFAULT_CODEX_HOME, or include `--codex-home <path>` in A2A_AGENT_COMMAND"
            )

    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(
            f"  - {error}" for error in errors
        )
        raise ValueError(error_msg)


@lru_cache()
def get_settings() -> Settings:
    """Return cached and validated settings."""
    auth_token = os.getenv("A2A_AUTH_TOKEN")
    config_path_raw = os.getenv("A2A_AGENTS_CONFIG", "config/agents.json")

    # Single agent configuration (streamlined approach)
    agent_command = os.getenv("A2A_AGENT_COMMAND")
    agent_api_key = os.getenv("A2A_AGENT_API_KEY")
    agent_description = os.getenv("A2A_AGENT_DESCRIPTION", "A2A-ACP Agent")

    # ChatGPT / OpenAI OAuth settings
    oauth_client_id = os.getenv("A2A_OAUTH_CLIENT_ID", "app_EMoamEEZ73f0CkXaXp7hrann")
    oauth_authorize_url = os.getenv(
        "A2A_OAUTH_AUTHORIZE_URL", "https://auth.openai.com/oauth/authorize"
    )
    oauth_token_url = os.getenv(
        "A2A_OAUTH_TOKEN_URL", "https://auth.openai.com/oauth/token"
    )
    oauth_scope = os.getenv("A2A_OAUTH_SCOPE", "openid profile email offline_access")
    oauth_redirect = os.getenv(
        "A2A_OAUTH_REDIRECT",
        "http://localhost:8001/a2a/agents/{agent_id}/auth/callback",
    )
    agent_oauth_enabled = (
        os.getenv("A2A_AGENT_OAUTH_ENABLED", "false").lower() == "true"
    )
    agent_auth_file_name = os.getenv("A2A_AGENT_AUTH_FILE_NAME", "auth.json")
    oauth_housekeeping_interval = int(
        os.getenv("A2A_OAUTH_HOUSEKEEPING_INTERVAL", "300")
    )
    oauth_auto_refresh_margin = int(os.getenv("A2A_OAUTH_AUTO_REFRESH_MARGIN", "300"))
    default_codex_home_raw = os.getenv("A2A_DEFAULT_CODEX_HOME")
    default_codex_home = (
        Path(default_codex_home_raw) if default_codex_home_raw else None
    )
    agent_codex_home_raw = os.getenv("A2A_AGENT_CODEX_HOME")
    agent_codex_home = Path(agent_codex_home_raw) if agent_codex_home_raw else None

    development_tool_extension_enabled = (
        os.getenv("DEVELOPMENT_TOOL_EXTENSION_ENABLED", "true").lower() == "true"
    )

    error_profile = parse_error_profile(os.getenv("A2A_ERROR_PROFILE"))

    # Push notification settings
    push_notification_settings = _get_push_notification_settings()

    settings = Settings(
        auth_token=auth_token,
        agents_config_path=Path(config_path_raw),
        agent_command=agent_command,
        agent_api_key=agent_api_key,
        agent_description=agent_description,
        default_codex_home=default_codex_home,
        agent_codex_home=agent_codex_home,
        oauth_client_id=oauth_client_id,
        oauth_authorize_url=oauth_authorize_url,
        oauth_token_url=oauth_token_url,
        oauth_scope=oauth_scope,
        oauth_redirect=oauth_redirect,
        agent_oauth_enabled=agent_oauth_enabled,
        agent_auth_file_name=agent_auth_file_name,
        oauth_housekeeping_interval=oauth_housekeeping_interval,
        oauth_auto_refresh_margin=oauth_auto_refresh_margin,
        development_tool_extension_enabled=development_tool_extension_enabled,
        push_notifications=push_notification_settings,
        error_profile=error_profile,
    )

    # Validate settings
    validate_settings(settings)

    return settings


def get_push_notification_settings() -> PushNotificationSettings:
    """Get push notification settings directly (convenience function)."""
    return get_settings().push_notifications


def is_push_notifications_enabled() -> bool:
    """Check if push notifications are enabled."""
    return get_settings().push_notifications.enabled


def create_example_env_file() -> str:
    """Generate an example .env file with push notification settings."""
    return """# A2A-ACP Configuration
A2A_AUTH_TOKEN=your-auth-token-here
A2A_AGENT_COMMAND=python your_agent.py
A2A_AGENT_API_KEY=your-agent-api-key
A2A_AGENT_DESCRIPTION="Your A2A-ACP Agent"

# Optional: ChatGPT / OpenAI OAuth configuration (for codex-acp)
# ChatGPT / OpenAI OAuth settings
A2A_OAUTH_CLIENT_ID=app_EMoamEEZ73f0CkXaXp7hrann
A2A_OAUTH_AUTHORIZE_URL=https://auth.openai.com/oauth/authorize
A2A_OAUTH_TOKEN_URL=https://auth.openai.com/oauth/token
A2A_OAUTH_SCOPE=openid profile email offline_access
# Must match the redirect registered with the OAuth client.
A2A_OAUTH_REDIRECT=http://localhost:8001/a2a/agents/{agent_id}/auth/callback
# Gate: default false; enable per-agent OAuth flows via agent config AND this setting
A2A_AGENT_OAUTH_ENABLED=false
# Token file written to agent codex_home: <codex_home>/$A2A_AGENT_AUTH_FILE_NAME
A2A_AGENT_AUTH_FILE_NAME=auth.json
# Housekeeping: interval for background OAuth housekeeping job (seconds)
A2A_OAUTH_HOUSEKEEPING_INTERVAL=300
# Auto-refresh margin: seconds before expiry to attempt a proactive refresh
A2A_OAUTH_AUTO_REFRESH_MARGIN=300
# Optional fallback (discouraged in production): per-agent `codex_home` is preferred
+A2A_DEFAULT_CODEX_HOME=/var/lib/codex-data
+A2A_AGENT_CODEX_HOME=/var/lib/codex-data/agent-1

# Development Tool Extension
DEVELOPMENT_TOOL_EXTENSION_ENABLED=true

# Push Notification Configuration
PUSH_NOTIFICATIONS_ENABLED=true
PUSH_NOTIFICATION_WEBHOOK_TIMEOUT=30
PUSH_NOTIFICATION_RETRY_ATTEMPTS=3
PUSH_NOTIFICATION_RETRY_DELAY=1.0
PUSH_NOTIFICATION_MAX_RETRY_DELAY=60.0

# Batch processing
PUSH_NOTIFICATION_BATCH_SIZE=10
PUSH_NOTIFICATION_MAX_CONCURRENT=50

# Cleanup configuration
PUSH_NOTIFICATION_CLEANUP_ENABLED=true
PUSH_NOTIFICATION_CLEANUP_INTERVAL=3600
PUSH_NOTIFICATION_RETENTION_COMPLETED_HOURS=24
PUSH_NOTIFICATION_RETENTION_FAILED_HOURS=0
PUSH_NOTIFICATION_RETENTION_AUTH_REQUIRED_HOURS=168

# Security
PUSH_NOTIFICATION_HMAC_SECRET=your-hmac-secret-for-webhook-validation
PUSH_NOTIFICATION_RATE_LIMIT_PER_MINUTE=60

# Real-time streaming
PUSH_NOTIFICATION_STREAMING_ENABLED=true
PUSH_NOTIFICATION_MAX_WS_CONNECTIONS=100
PUSH_NOTIFICATION_MAX_SSE_CONNECTIONS=200
PUSH_NOTIFICATION_CONNECTION_CLEANUP_INTERVAL=300

# Performance monitoring
PUSH_NOTIFICATION_ENABLE_METRICS=true
PUSH_NOTIFICATION_METRICS_RETENTION_HOURS=72
"""
