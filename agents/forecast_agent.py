"""Agent that projects future demand with a fitted regression model."""

from __future__ import annotations

from typing import Optional

from agents.base import Agent
from config.settings import AgentConfig, DatabaseConfig, ForecastConfig
from llm.base import LLMClient
from prompts.manager import load_prompt
from tools.forecast_tools import ForecastToolset


class ForecastAgent(Agent):
    """Owns the forecasting tools and the error reporting that goes with them."""

    def __init__(
        self,
        client: LLMClient,
        db_config: DatabaseConfig,
        forecast_config: Optional[ForecastConfig] = None,
        agent_config: Optional[AgentConfig] = None,
        prompt_version: Optional[int] = None,
    ) -> None:
        self._prompt_version = prompt_version
        toolset = ForecastToolset(db_config, forecast_config or ForecastConfig())
        super().__init__(client, [toolset], agent_config)

    @property
    def name(self) -> str:
        return "forecast_agent"

    @property
    def system_prompt(self) -> str:
        return load_prompt("forecast_agent", self._prompt_version)
