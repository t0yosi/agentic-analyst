"""Factory that builds a concrete :class:`LLMClient` from configuration."""

from __future__ import annotations

import logging

from config.settings import LLMConfig
from llm.base import LLMClient
from llm.mock_client import MockLLMClient
from utils.errors import ConfigError

logger = logging.getLogger(__name__)


def build_llm_client(config: LLMConfig, use_mock: bool = False) -> LLMClient:
    """Return the client implied by ``config``.

    Args:
        config: Validated provider settings.
        use_mock: When True, return a :class:`MockLLMClient` and never touch
            the network. Tests and offline demos use this.

    Returns:
        A ready-to-use client.

    Raises:
        ConfigError: If a live client is requested without an API key.
    """
    if use_mock:
        logger.info("[CONFIG] llm_client=mock")
        return MockLLMClient()

    if not config.api_key:
        raise ConfigError(
            "no API key configured; set ANTHROPIC_API_KEY or call with use_mock=True"
        )

    from llm.anthropic_client import AnthropicClient

    logger.info("[CONFIG] llm_client=anthropic model=%s", config.model)
    return AnthropicClient(
        api_key=config.api_key,
        model=config.model,
        temperature=config.temperature,
        timeout_seconds=config.timeout_seconds,
    )
