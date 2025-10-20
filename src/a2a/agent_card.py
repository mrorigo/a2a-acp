"""
A2A Agent Card Generation System

Generates A2A AgentCard objects from ZedACP agent configurations.
Provides dynamic agent discovery and capability advertisement.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional
from pathlib import Path

from .models import (
    AgentCard, AgentCapabilities, AgentSkill, SecurityScheme,
    APIKeySecurityScheme, HTTPAuthSecurityScheme, TransportProtocol
)
from a2a_acp.settings import get_settings

logger = logging.getLogger(__name__)


class AgentCardGenerator:
    """
    Generates A2A AgentCard objects from ZedACP agent configurations.

    Creates comprehensive agent cards that advertise the capabilities,
    security schemes, and skills of ZedACP agents in A2A format.
    """

    def __init__(self, config_path: Optional[Path] = None):
        # Single agent configuration - use settings instead of registry
        settings = get_settings()

        self.agent_config = {
            "name": "default-agent",
            "command": settings.agent_command or "python tests/dummy_agent.py",
            "api_key": settings.agent_api_key,
            "description": settings.agent_description or "A2A-ACP Development Agent"
        }
        self._base_url = "http://localhost:8001"  # TODO: Make configurable

    def generate_agent_card(self, agent_name: str) -> AgentCard:
        """
        Generate an A2A AgentCard for a specific agent.

        Args:
            agent_name: Name of the agent to generate card for

        Returns:
            Complete AgentCard with all capabilities and metadata
        """
        # For single-agent architecture, only "default-agent" is supported
        if agent_name != "default-agent":
            raise ValueError(f"Unknown agent: {agent_name}. Only 'default-agent' is supported in single-agent mode.")

        agent_config = self.agent_config

        # Generate base agent information
        card = AgentCard(
            protocolVersion="0.3.0",
            name=agent_config["name"],
            description=agent_config["description"] or f"A2A-compatible ZedACP agent '{agent_config['name']}'",
            url=f"{self._base_url}/a2a/{agent_config['name']}",
            preferredTransport=TransportProtocol.JSONRPC,
            version="1.0.0",  # Single agent doesn't have version in config
            capabilities=self._generate_capabilities(),
            securitySchemes=self._generate_security_schemes(),
            defaultInputModes=["text/plain"],
            defaultOutputModes=["text/plain"],
            skills=self._generate_agent_skills(agent_config)
        )

        logger.info("Generated AgentCard",
                   extra={"agent_name": agent_name, "skills_count": len(card.skills)})
        return card

    def generate_all_agent_cards(self) -> Dict[str, AgentCard]:
        """
        Generate AgentCards for all configured agents.

        Returns:
            Dictionary mapping agent names to their AgentCards
        """
        cards = {}
        # Single agent only
        try:
            card = self.generate_agent_card("default-agent")
            cards["default-agent"] = card
        except Exception as e:
            logger.error("Failed to generate AgentCard",
                        extra={"agent_name": "default-agent", "error": str(e)})

        logger.info("Generated AgentCards for all agents", extra={"count": len(cards)})
        return cards

    def _generate_capabilities(self) -> AgentCapabilities:
        """Generate standard capabilities for ZedACP agents."""
        return AgentCapabilities(
            streaming=True,
            pushNotifications=False,  # Not yet supported
            stateTransitionHistory=True
        )

    def _generate_security_schemes(self) -> Dict[str, SecurityScheme]:
        """Generate available security schemes."""
        return {
            "bearer": HTTPAuthSecurityScheme(
                type="http",
                scheme="bearer",
                bearerFormat="JWT",
                description="JWT bearer token authentication"
            ),
            "apikey": APIKeySecurityScheme(
                type="apiKey",
                **{"in": "header"},
                name="X-API-Key",
                description="API key authentication via header"
            )
        }

    def _generate_agent_skills(self, agent_config) -> List[AgentSkill]:
        """Generate A2A skills based on ZedACP agent capabilities."""
        # Default skills that most ZedACP agents support
        skills = [
            AgentSkill(
                id="code_generation",
                name="Code Generation",
                description="Generate and modify code in various programming languages",
                tags=["coding", "development", "programming"],
                examples=[
                    "Create a Python function to calculate fibonacci numbers",
                    "Write a JavaScript function to handle HTTP requests",
                    "Generate SQL queries for data analysis"
                ],
                inputModes=["text/plain"],
                outputModes=["text/plain"]
            ),
            AgentSkill(
                id="file_system",
                name="File System Operations",
                description="Read, write, and modify files in the workspace",
                tags=["files", "workspace", "io"],
                examples=[
                    "Read the contents of config.json",
                    "Create a new Python script",
                    "List all files in the current directory"
                ],
                inputModes=["text/plain"],
                outputModes=["text/plain"]
            ),
            AgentSkill(
                id="text_processing",
                name="Text Processing and Analysis",
                description="Analyze, summarize, and process text content",
                tags=["text", "nlp", "analysis"],
                examples=[
                    "Summarize this article",
                    "Extract key points from the text",
                    "Translate this content to Spanish"
                ],
                inputModes=["text/plain"],
                outputModes=["text/plain"]
            ),
            AgentSkill(
                id="data_analysis",
                name="Data Analysis and Visualization",
                description="Analyze data and generate insights",
                tags=["data", "analysis", "visualization"],
                examples=[
                    "Analyze this dataset and create a summary",
                    "Generate a chart from the CSV data",
                    "Find patterns in this log file"
                ],
                inputModes=["text/plain"],
                outputModes=["text/plain"]
            )
        ]

        # Add agent-specific skills based on description
        if agent_config["description"]:
            description_lower = agent_config["description"].lower()

            if any(term in description_lower for term in ["code", "programming", "development"]):
                skills.append(AgentSkill(
                    id="debugging",
                    name="Debugging and Troubleshooting",
                    description="Debug code and resolve technical issues",
                    tags=["debugging", "troubleshooting", "errors"],
                    examples=[
                        "Debug this Python script that's throwing errors",
                        "Find the issue in this JavaScript code",
                        "Help me understand this error message"
                    ],
                    inputModes=["text/plain"],
                    outputModes=["text/plain"]
                ))

            if any(term in description_lower for term in ["data", "analysis", "ml", "machine learning"]):
                skills.append(AgentSkill(
                    id="machine_learning",
                    name="Machine Learning",
                    description="Build and train machine learning models",
                    tags=["ml", "machine-learning", "ai"],
                    examples=[
                        "Create a simple linear regression model",
                        "Train a classifier on this dataset",
                        "Explain how this ML algorithm works"
                    ],
                    inputModes=["text/plain"],
                    outputModes=["text/plain"]
                ))

        logger.debug("Generated agent skills",
                    extra={"agent_name": agent_config["name"], "skills_count": len(skills)})
        return skills


class A2AAgentCardManager:
    """
    Manages A2A Agent Cards for the server.

    Provides caching and dynamic generation of Agent Cards
    for agent discovery and capability advertisement.
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.generator = AgentCardGenerator(config_path)
        self._card_cache: Dict[str, AgentCard] = {}
        self._cache_timestamp: Optional[float] = None

    def get_agent_card(self, agent_name: str, use_cache: bool = True) -> AgentCard:
        """
        Get AgentCard for a specific agent.

        Args:
            agent_name: Name of the agent
            use_cache: Whether to use cached card if available

        Returns:
            AgentCard for the requested agent
        """
        if use_cache and agent_name in self._card_cache:
            return self._card_cache[agent_name]

        card = self.generator.generate_agent_card(agent_name)
        self._card_cache[agent_name] = card
        return card

    def get_all_agent_cards(self, use_cache: bool = True) -> Dict[str, AgentCard]:
        """
        Get AgentCards for all configured agents.

        Args:
            use_cache: Whether to use cached cards if available

        Returns:
            Dictionary of all AgentCards
        """
        if use_cache and self._card_cache:
            return self._card_cache.copy()

        cards = self.generator.generate_all_agent_cards()
        self._card_cache = cards
        return cards

    def refresh_cache(self) -> Dict[str, AgentCard]:
        """Refresh the AgentCard cache."""
        self._card_cache.clear()
        return self.get_all_agent_cards(use_cache=False)

    def list_available_agents(self) -> List[str]:
        """List all available agent names."""
        return [self.generator.agent_config["name"]]


# Global Agent Card manager instance
agent_card_manager = A2AAgentCardManager()


# Convenience functions
def generate_agent_card(agent_name: str) -> AgentCard:
    """Generate an AgentCard for a specific agent."""
    return agent_card_manager.get_agent_card(agent_name)


def get_all_agent_cards() -> Dict[str, AgentCard]:
    """Get AgentCards for all configured agents."""
    return agent_card_manager.get_all_agent_cards()


def list_available_agents() -> List[str]:
    """List all available agent names."""
    return agent_card_manager.list_available_agents()