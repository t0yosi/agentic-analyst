"""The tool-use loop every agent runs.

The loop is written directly against the neutral :class:`LLMClient` interface
rather than a third-party agent framework. For a system this size the framework
costs more in opacity than it saves in code, and a hand-written loop makes the
exact sequence of tool calls inspectable in tests and in logs.

Subclasses supply three things: a name, a system prompt, and a toolset that can
both describe its tools and dispatch them.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from config.settings import AgentConfig
from llm.base import LLMClient, LLMResponse
from utils.errors import AgentError, LoopLimitExceeded, ToolError

logger = logging.getLogger(__name__)


class Toolset(Protocol):
    """Anything that can describe its tools and execute them by name."""

    @property
    def definitions(self) -> List[Dict[str, Any]]: ...

    def dispatch(self, name: str, arguments: Dict[str, Any]) -> str: ...


@dataclass
class AgentResult:
    """What one agent run produced.

    Attributes:
        text: The agent's final prose answer.
        tool_calls: Names of tools invoked, in call order.
        iterations: How many model turns the loop consumed.
        input_tokens: Prompt tokens billed across the run.
        output_tokens: Completion tokens billed across the run.
    """

    text: str
    tool_calls: List[str] = field(default_factory=list)
    iterations: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class Agent(ABC):
    """Base class implementing the shared tool-use loop."""

    def __init__(
        self,
        client: LLMClient,
        toolsets: List[Toolset],
        config: Optional[AgentConfig] = None,
        max_tokens: int = 2048,
    ) -> None:
        if not toolsets:
            raise ValueError(f"agent {self.name!r} requires at least one toolset, got none")
        self._client = client
        self._toolsets = toolsets
        self._config = config or AgentConfig()
        self._max_tokens = max_tokens

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in logs and delegation."""

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """System instruction sent on every call."""

    @property
    def tool_definitions(self) -> List[Dict[str, Any]]:
        """Every tool this agent may call, flattened across its toolsets."""
        definitions: List[Dict[str, Any]] = []
        for toolset in self._toolsets:
            definitions.extend(toolset.definitions)
        return definitions

    def run(self, query: str, history: Optional[List[Dict[str, Any]]] = None) -> AgentResult:
        """Answer ``query``, calling tools until the model stops asking.

        Args:
            query: The natural-language task for this agent.
            history: Optional prior turns to prepend for continuity.

        Returns:
            The agent's result.

        Raises:
            ValueError: If ``query`` is empty.
            LoopLimitExceeded: If the loop hits ``max_iterations``.
            AgentError: If the model returns neither text nor a tool call.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string, got empty")

        messages: List[Dict[str, Any]] = list(history or [])
        messages.append({"role": "user", "content": query})

        result = AgentResult(text="")
        tools = self.tool_definitions
        logger.info("[STAGE] %s start tools=%d", self.name, len(tools))

        for iteration in range(1, self._config.max_iterations + 1):
            response = self._client.complete(
                messages=messages,
                system_prompt=self.system_prompt,
                tools=tools,
                max_tokens=self._max_tokens,
            )
            result.iterations = iteration
            result.input_tokens += response.input_tokens
            result.output_tokens += response.output_tokens

            if not response.wants_tools:
                if not response.text:
                    raise AgentError(
                        f"agent {self.name!r} returned neither text nor a tool call "
                        f"on iteration {iteration}"
                    )
                result.text = response.text
                logger.info(
                    "[STAGE] %s done iterations=%d tools_used=%d tokens=%d",
                    self.name, iteration, len(result.tool_calls), result.total_tokens,
                )
                return result

            messages.append(self._assistant_turn(response))
            messages.append(self._tool_results_turn(response, result))

        raise LoopLimitExceeded(self.name, self._config.max_iterations)

    def _assistant_turn(self, response: LLMResponse) -> Dict[str, Any]:
        content: List[Dict[str, Any]] = []
        if response.text:
            content.append({"type": "text", "text": response.text})
        for call in response.tool_calls:
            content.append(
                {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
            )
        return {"role": "assistant", "content": content}

    def _tool_results_turn(self, response: LLMResponse, result: AgentResult) -> Dict[str, Any]:
        blocks: List[Dict[str, Any]] = []
        for call in response.tool_calls:
            result.tool_calls.append(call.name)
            logger.info("[ROUTE] %s -> %s args=%.100s", self.name, call.name, call.arguments)
            try:
                output = self._dispatch(call.name, call.arguments)
                is_error = False
            except Exception as exc:
                # Tool failures are reported back to the model rather than raised,
                # so it can correct a bad query instead of aborting the run.
                output = f"Tool error: {exc}"
                is_error = True
                logger.warning("[ROUTE] %s tool %s failed: %s", self.name, call.name, exc)
            blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": output,
                    "is_error": is_error,
                }
            )
        return {"role": "user", "content": blocks}

    def _dispatch(self, name: str, arguments: Dict[str, Any]) -> str:
        for toolset in self._toolsets:
            if any(d["name"] == name for d in toolset.definitions):
                return toolset.dispatch(name, arguments)
        raise ToolError(f"no toolset registered for tool {name!r} on agent {self.name!r}")
