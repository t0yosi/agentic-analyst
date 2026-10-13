"""Deterministic mock model client for tests.

Responses are queued explicitly, so a test states exactly what the model will
do. There is no magic-mock behaviour and no network access.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Deque, Dict, List, Optional

from llm.base import LLMClient, LLMResponse
from utils.errors import LLMError

logger = logging.getLogger(__name__)


class MockLLMClient(LLMClient):
    """An :class:`LLMClient` that replays a queue of prepared responses."""

    def __init__(self, model_name: str = "mock-model") -> None:
        if not model_name:
            raise ValueError("model_name must be a non-empty string, got empty")
        self._queue: Deque[LLMResponse] = deque()
        self._model_name = model_name
        self.calls: List[Dict[str, Any]] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    def queue(self, response: LLMResponse) -> "MockLLMClient":
        """Append one response to the replay queue. Returns self for chaining."""
        if not isinstance(response, LLMResponse):
            raise TypeError(f"response must be an LLMResponse, got {type(response).__name__}")
        self._queue.append(response)
        return self

    def queue_text(self, text: str) -> "MockLLMClient":
        """Queue a plain final answer with no tool calls."""
        return self.queue(LLMResponse(text=text, stop_reason="end_turn"))

    @property
    def pending(self) -> int:
        """How many queued responses remain unconsumed."""
        return len(self._queue)

    def complete(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        self.calls.append(
            {
                "messages": list(messages),
                "system_prompt": system_prompt,
                "tools": [t["name"] for t in (tools or [])],
                "max_tokens": max_tokens,
            }
        )
        if not self._queue:
            raise LLMError(
                f"MockLLMClient queue is empty but complete() was called "
                f"(call #{len(self.calls)}); queue another response in the test"
            )
        return self._queue.popleft()
