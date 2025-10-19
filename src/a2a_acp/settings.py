from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Settings:
    """Simple settings container sourced from environment variables."""

    auth_token: Optional[str]
    agents_config_path: Path

    # Single agent configuration (streamlined approach)
    agent_command: Optional[str]
    agent_api_key: Optional[str]
    agent_description: str


@lru_cache()
def get_settings() -> Settings:
    """Return cached settings."""
    auth_token = os.getenv("A2A_AUTH_TOKEN")
    config_path_raw = os.getenv("A2A_AGENTS_CONFIG", "config/agents.json")

    # Single agent configuration (streamlined approach)
    agent_command = os.getenv("A2A_AGENT_COMMAND")
    agent_api_key = os.getenv("A2A_AGENT_API_KEY")
    agent_description = os.getenv("A2A_AGENT_DESCRIPTION", "A2A-ACP Agent")

    return Settings(
        auth_token=auth_token,
        agents_config_path=Path(config_path_raw),
        agent_command=agent_command,
        agent_api_key=agent_api_key,
        agent_description=agent_description
    )
