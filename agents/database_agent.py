"""Agent that answers questions from the analytics database."""

from __future__ import annotations

from typing import Optional

from agents.base import Agent
from config.settings import AgentConfig, DatabaseConfig
from llm.base import LLMClient
from prompts.manager import load_prompt
from tools.sql_tools import SQLToolset


class DatabaseAgent(Agent):
    """Owns every tool that reads the analytics database."""

    def __init__(
        self,
        client: LLMClient,
        db_config: DatabaseConfig,
        agent_config: Optional[AgentConfig] = None,
        prompt_version: Optional[int] = None,
    ) -> None:
        self._prompt_version = prompt_version
        super().__init__(client, [SQLToolset(db_config)], agent_config)

    @property
    def name(self) -> str:
        return "database_agent"

    @property
    def system_prompt(self) -> str:
        return load_prompt("database_agent", self._prompt_version)
