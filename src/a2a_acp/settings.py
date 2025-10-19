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


@lru_cache()
def get_settings() -> Settings:
    """Return cached settings."""
    auth_token = os.getenv("A2A_AUTH_TOKEN")
    config_path_raw = os.getenv("A2A_AGENTS_CONFIG", "config/agents.json")
    return Settings(auth_token=auth_token, agents_config_path=Path(config_path_raw))
