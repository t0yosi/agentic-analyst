"""Anthropic implementation of :class:`LLMClient`."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from llm.base import LLMClient, LLMResponse, ToolCall
from utils.errors import LLMError

logger = logging.getLogger(__name__)


class AnthropicClient(LLMClient):
    """Talks to the Anthropic Messages API.

    The SDK import is deferred to ``__init__`` so that importing this module
    does not require the package to be installed, which keeps the test suite
    free of provider dependencies.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        timeout_seconds: float = 60.0,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must be a non-empty string, got empty")
        if not model:
            raise ValueError("model must be a non-empty string, got empty")

        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise LLMError(
                "the 'anthropic' package is required for AnthropicClient; "
                "install it with 'pip install anthropic'"
            ) from exc

        self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout_seconds)
        self._model = model
        self._temperature = temperature

    @property
    def model_name(self) -> str:
        return self._model

    def complete(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        kwargs: Dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": self._temperature,
            "system": system_prompt,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools

        try:
            raw = self._client.messages.create(**kwargs)
        except Exception as exc:
            raise LLMError(f"Anthropic request failed for model {self._model!r}: {exc}") from exc

        return self._parse(raw)

    def _parse(self, raw: Any) -> LLMResponse:
        """Convert a provider response into the neutral :class:`LLMResponse`."""
        text_parts: List[str] = []
        tool_calls: List[ToolCall] = []

        for block in getattr(raw, "content", []) or []:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_parts.append(getattr(block, "text", ""))
            elif block_type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=getattr(block, "id", ""),
                        name=getattr(block, "name", ""),
                        arguments=dict(getattr(block, "input", {}) or {}),
                    )
                )

        usage = getattr(raw, "usage", None)
        response = LLMResponse(
            text="\n".join(p for p in text_parts if p),
            tool_calls=tool_calls,
            input_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
            stop_reason=getattr(raw, "stop_reason", None),
        )
        logger.info(
            "[TOKENS] model=%s in=%d out=%d tools=%d",
            self._model,
            response.input_tokens,
            response.output_tokens,
            len(response.tool_calls),
        )
        return response
