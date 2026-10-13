"""Exception hierarchy for the agentic analyst.

Every error raised by this package descends from :class:`AnalystError`, so a
caller can catch the whole domain with one clause or narrow to a single layer.
"""

from __future__ import annotations


class AnalystError(Exception):
    """Base class for every error raised by this package."""


class ConfigError(AnalystError):
    """Raised when configuration is missing or invalid."""


class ValidationError(AnalystError):
    """Raised when input fails validation at a public API boundary."""


class LLMError(AnalystError):
    """Raised when a language-model provider call fails or returns nothing usable."""


class ToolError(AnalystError):
    """Raised when a tool cannot execute the request it was given."""


class AgentError(AnalystError):
    """Raised when an agent cannot complete its task."""


class LoopLimitExceeded(AgentError):
    """Raised when an agent's tool-use loop hits its iteration ceiling."""

    def __init__(self, agent_name: str, limit: int) -> None:
        super().__init__(
            f"agent {agent_name!r} exceeded its tool-use loop limit of {limit} iterations"
        )
        self.agent_name = agent_name
        self.limit = limit
