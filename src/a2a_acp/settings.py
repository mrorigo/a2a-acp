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
        max_concurrent_notifications=int(os.getenv("PUSH_NOTIFICATION_MAX_CONCURRENT", "50")),

        # Cleanup configuration
        cleanup_enabled=os.getenv("PUSH_NOTIFICATION_CLEANUP_ENABLED", "true").lower() == "true",
        cleanup_interval=int(os.getenv("PUSH_NOTIFICATION_CLEANUP_INTERVAL", "3600")),
        retention_completed_hours=int(os.getenv("PUSH_NOTIFICATION_RETENTION_COMPLETED_HOURS", "24")),
        retention_failed_hours=int(os.getenv("PUSH_NOTIFICATION_RETENTION_FAILED_HOURS", "0")),
        retention_auth_required_hours=int(os.getenv("PUSH_NOTIFICATION_RETENTION_AUTH_REQUIRED_HOURS", "168")),

        # Security
        hmac_secret=os.getenv("PUSH_NOTIFICATION_HMAC_SECRET"),
        rate_limit_per_minute=int(os.getenv("PUSH_NOTIFICATION_RATE_LIMIT_PER_MINUTE", "60")),

        # Real-time streaming
        streaming_enabled=os.getenv("PUSH_NOTIFICATION_STREAMING_ENABLED", "true").lower() == "true",
        max_websocket_connections=int(os.getenv("PUSH_NOTIFICATION_MAX_WS_CONNECTIONS", "100")),
        max_sse_connections=int(os.getenv("PUSH_NOTIFICATION_MAX_SSE_CONNECTIONS", "200")),
        connection_cleanup_interval=int(os.getenv("PUSH_NOTIFICATION_CONNECTION_CLEANUP_INTERVAL", "300")),

        # Performance monitoring
        enable_metrics=os.getenv("PUSH_NOTIFICATION_ENABLE_METRICS", "true").lower() == "true",
        metrics_retention_hours=int(os.getenv("PUSH_NOTIFICATION_METRICS_RETENTION_HOURS", "72"))
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
            errors.append("PUSH_NOTIFICATION_RETENTION_FAILED_HOURS must be non-negative")

        if push_settings.retention_completed_hours < 0:
            errors.append("PUSH_NOTIFICATION_RETENTION_COMPLETED_HOURS must be non-negative")

        if push_settings.retention_auth_required_hours < 0:
            errors.append("PUSH_NOTIFICATION_RETENTION_AUTH_REQUIRED_HOURS must be non-negative")

        # Warning: if HMAC secret is not set, webhook signatures cannot be validated
        if not push_settings.hmac_secret:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                "PUSH_NOTIFICATION_HMAC_SECRET not set - webhook signature validation disabled"
            )

    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
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

    development_tool_extension_enabled = os.getenv("DEVELOPMENT_TOOL_EXTENSION_ENABLED", "true").lower() == "true"

    error_profile = parse_error_profile(os.getenv("A2A_ERROR_PROFILE"))

    # Push notification settings
    push_notification_settings = _get_push_notification_settings()

    settings = Settings(
        auth_token=auth_token,
        agents_config_path=Path(config_path_raw),
        agent_command=agent_command,
        agent_api_key=agent_api_key,
        agent_description=agent_description,
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
    return '''# A2A-ACP Configuration
A2A_AUTH_TOKEN=your-auth-token-here
A2A_AGENT_COMMAND=python your_agent.py
A2A_AGENT_API_KEY=your-agent-api-key
A2A_AGENT_DESCRIPTION="Your A2A-ACP Agent"

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
'''
