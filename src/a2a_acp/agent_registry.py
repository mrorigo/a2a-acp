from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable

from .models import AgentConfig, AgentManifest, AgentManifestCapabilities, RunMode
from .settings import get_settings

logger = logging.getLogger(__name__)


class AgentRegistry:
    """In-memory registry for configured agents."""

    def __init__(self, config_path: Path | None = None) -> None:
        settings = get_settings()
        self._config_path = Path(config_path or settings.agents_config_path)
        self._agents: Dict[str, AgentConfig] = {}
        self.reload()

    def reload(self) -> None:
        """Reload configuration from disk."""
        logger.debug("Loading agents configuration", extra={"path": str(self._config_path)})
        if not self._config_path.exists():
            raise FileNotFoundError(f"Agents configuration not found: {self._config_path}")
        data = json.loads(self._config_path.read_text())
        self._agents = {name: AgentConfig(**payload) for name, payload in data.items()}

    def list(self) -> Iterable[AgentConfig]:
        """Iterate over configured agents."""
        return self._agents.values()

    def get(self, name: str) -> AgentConfig:
        """Retrieve a single agent."""
        try:
            return self._agents[name]
        except KeyError as exc:
            raise KeyError(f"Unknown agent: {name}") from exc

    def manifest_for(self, name: str) -> AgentManifest:
        """Return a static manifest for a given agent."""
        agent = self.get(name)
        description = agent.description or f"ZedACP agent '{agent.name}' exposed over IBMACP."
        version = agent.version or "0.1.0"
        capabilities = AgentManifestCapabilities(
            modes=[RunMode.sync, RunMode.stream],
            supports_streaming=True,
            supports_cancellation=True,
        )
        return AgentManifest(
            name=agent.name,
            description=description,
            version=version,
            capabilities=capabilities,
        )
