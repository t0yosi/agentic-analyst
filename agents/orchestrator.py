"""The coordinating agent.

The Orchestrator's tools *are* the other agents. Delegation therefore happens
through the same tool-calling mechanism as everything else, rather than through
a hardcoded pipeline, which means the model decides which specialists a given
question actually needs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agents.base import Agent, AgentResult
from agents.database_agent import DatabaseAgent
from agents.forecast_agent import ForecastAgent
from agents.visualization_agent import VisualizationAgent
from config.settings import AppConfig
from llm.base import LLMClient
from prompts.manager import load_prompt
from utils.errors import ToolError

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorResult:
    """The final answer plus everything gathered on the way to it."""

    answer: str
    chart_base64: Optional[str] = None
    agents_called: List[str] = field(default_factory=list)
    iterations: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class _AgentDelegationToolset:
    """Presents the specialist agents to the orchestrator as callable tools."""

    def __init__(
        self,
        database_agent: DatabaseAgent,
        forecast_agent: ForecastAgent,
        visualization_agent: VisualizationAgent,
    ) -> None:
        self._agents: Dict[str, Agent] = {
            "call_database_agent": database_agent,
            "call_forecast_agent": forecast_agent,
            "call_visualization_agent": visualization_agent,
        }
        self.calls: List[str] = []
        self.sub_results: List[AgentResult] = []

    @property
    def definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "call_database_agent",
                "description": (
                    "Ask the database specialist a question about stations, riders, "
                    "trips, or weather. It has read-only SQL access and returns the "
                    "figures it retrieved. It cannot see the user's question, so state "
                    "the full request in the instruction."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "instruction": {
                            "type": "string",
                            "description": "A complete, self-contained data request.",
                        }
                    },
                    "required": ["instruction"],
                },
            },
            {
                "name": "call_forecast_agent",
                "description": (
                    "Ask the forecasting specialist to project future daily demand. It "
                    "fits a regression model and returns the projection with holdout "
                    "error metrics. State the horizon and any borough filter."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "instruction": {
                            "type": "string",
                            "description": "What to forecast, over what horizon.",
                        }
                    },
                    "required": ["instruction"],
                },
            },
            {
                "name": "call_visualization_agent",
                "description": (
                    "Ask the charting specialist to render one chart. It has no data "
                    "access of its own, so the instruction must contain the actual "
                    "labels and values to plot."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "instruction": {
                            "type": "string",
                            "description": "Chart request including the data to plot.",
                        }
                    },
                    "required": ["instruction"],
                },
            },
        ]

    def dispatch(self, name: str, arguments: Dict[str, Any]) -> str:
        agent = self._agents.get(name)
        if agent is None:
            raise ToolError(f"unknown delegation tool: {name!r}")

        instruction = str(arguments.get("instruction", "")).strip()
        if not instruction:
            raise ToolError(f"{name} requires a non-empty 'instruction' argument")

        self.calls.append(name)
        logger.info("[ROUTE] orchestrator -> %s query=%.80s", agent.name, instruction)
        result = agent.run(instruction)
        self.sub_results.append(result)
        return result.text

    @property
    def visualization_agent(self) -> VisualizationAgent:
        return self._agents["call_visualization_agent"]  # type: ignore[return-value]


class Orchestrator(Agent):
    """Top-level agent that plans, delegates, correlates, and synthesises."""

    def __init__(
        self,
        client: LLMClient,
        config: Optional[AppConfig] = None,
        prompt_version: Optional[int] = None,
    ) -> None:
        self._app_config = config or AppConfig()
        self._prompt_version = prompt_version
        self._delegation = _AgentDelegationToolset(
            DatabaseAgent(client, self._app_config.database, self._app_config.agent),
            ForecastAgent(
                client,
                self._app_config.database,
                self._app_config.forecast,
                self._app_config.agent,
            ),
            VisualizationAgent(client, self._app_config.agent),
        )
        self._history: List[Dict[str, Any]] = []
        super().__init__(client, [self._delegation], self._app_config.agent, max_tokens=4096)

    @property
    def name(self) -> str:
        return "orchestrator"

    @property
    def system_prompt(self) -> str:
        return load_prompt("orchestrator", self._prompt_version)

    def ask(self, question: str, remember: bool = True) -> OrchestratorResult:
        """Answer a business question end to end.

        Args:
            question: The user's natural-language question.
            remember: When True, append this exchange to conversation memory.

        Returns:
            The synthesised answer, any chart produced, and run statistics.

        Raises:
            ValueError: If ``question`` is empty.
        """
        if not question or not question.strip():
            raise ValueError("question must be a non-empty string, got empty")

        self._delegation.calls.clear()
        self._delegation.sub_results.clear()

        result = self.run(question, history=list(self._history))

        sub_in = sum(r.input_tokens for r in self._delegation.sub_results)
        sub_out = sum(r.output_tokens for r in self._delegation.sub_results)

        if remember:
            self._history.append({"role": "user", "content": question})
            self._history.append({"role": "assistant", "content": result.text})
            self._trim_history()

        logger.info(
            "[TOKENS] run total in=%d out=%d agents=%s",
            result.input_tokens + sub_in,
            result.output_tokens + sub_out,
            self._delegation.calls,
        )
        return OrchestratorResult(
            answer=result.text,
            chart_base64=self._delegation.visualization_agent.last_chart,
            agents_called=list(self._delegation.calls),
            iterations=result.iterations,
            input_tokens=result.input_tokens + sub_in,
            output_tokens=result.output_tokens + sub_out,
        )

    def reset_memory(self) -> None:
        """Clear conversation history."""
        self._history.clear()

    def _trim_history(self) -> None:
        limit = self._app_config.agent.memory_turns * 2
        if limit and len(self._history) > limit:
            self._history = self._history[-limit:]
