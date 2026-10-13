"""Agent that renders a chart from data another agent retrieved."""

from __future__ import annotations

from typing import Optional

from agents.base import Agent
from config.settings import AgentConfig
from llm.base import LLMClient
from prompts.manager import load_prompt
from tools.chart_tools import ChartToolset


class VisualizationAgent(Agent):
    """Owns chart rendering and holds the most recent chart produced."""

    def __init__(
        self,
        client: LLMClient,
        agent_config: Optional[AgentConfig] = None,
        prompt_version: Optional[int] = None,
    ) -> None:
        self._prompt_version = prompt_version
        self._charts = ChartToolset()
        super().__init__(client, [self._charts], agent_config)

    @property
    def name(self) -> str:
        return "visualization_agent"

    @property
    def system_prompt(self) -> str:
        return load_prompt("visualization_agent", self._prompt_version)

    @property
    def last_chart(self) -> Optional[str]:
        """Base64 PNG of the most recent chart, or None if none rendered yet."""
        return self._charts.last_chart
