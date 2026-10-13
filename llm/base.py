"""Provider-agnostic language-model interface.

The runner that drives a tool-use loop must not know which provider it is
talking to, nor which tools exist. Both are parameters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ToolCall:
    """A single tool invocation requested by the model."""

    id: str
    name: str
    arguments: Dict[str, Any]

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("ToolCall.id must be a non-empty string, got empty")
        if not self.name:
            raise ValueError("ToolCall.name must be a non-empty string, got empty")


@dataclass
class LLMResponse:
    """One turn of model output.

    Attributes:
        text: Any prose the model produced this turn.
        tool_calls: Tools the model asked to run. Empty means the turn is final.
        input_tokens: Prompt tokens billed for this call.
        output_tokens: Completion tokens billed for this call.
        stop_reason: Provider-reported reason the turn ended.
    """

    text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: Optional[str] = None

    @property
    def wants_tools(self) -> bool:
        """True when the model requested at least one tool call."""
        return bool(self.tool_calls)


class LLMClient(ABC):
    """Abstract language-model client.

    Implementations must be safe to call repeatedly and must never mutate the
    ``messages`` list they are given.
    """

    @abstractmethod
    def complete(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Produce one turn of model output.

        Args:
            messages: Conversation so far, in provider-neutral form.
            system_prompt: System instruction for this call.
            tools: JSON-schema tool definitions the model may call.
            max_tokens: Ceiling on generated tokens.

        Returns:
            The model's response for this turn.

        Raises:
            LLMError: If the provider call fails.
        """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier of the underlying model, for logging."""
