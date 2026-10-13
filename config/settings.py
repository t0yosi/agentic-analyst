"""Typed configuration for every layer of the analyst.

Each layer owns a dataclass that validates its own values in ``__post_init__``.
Modules receive a config object and read attributes; nothing in this package
ever indexes a raw dict for configuration.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from utils.errors import ConfigError

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5"
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


@dataclass
class LLMConfig:
    """Settings for the language-model provider."""

    model: str = DEFAULT_MODEL
    api_key: Optional[str] = None
    max_tokens: int = 2048
    temperature: float = 0.0
    timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model must be a non-empty string, got empty")
        if self.max_tokens <= 0:
            raise ValueError(f"max_tokens must be > 0, got {self.max_tokens}")
        if not 0.0 <= self.temperature <= 1.0:
            raise ValueError(f"temperature must be in [0.0, 1.0], got {self.temperature}")
        if self.timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be > 0, got {self.timeout_seconds}")


@dataclass
class DatabaseConfig:
    """Settings for the SQLite analytics database."""

    path: Path = field(default_factory=lambda: Path("bikeshare.db"))
    max_rows: int = 500
    query_timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        if isinstance(self.path, str):
            self.path = Path(self.path)
        if self.max_rows <= 0:
            raise ValueError(f"max_rows must be > 0, got {self.max_rows}")
        if self.query_timeout_seconds <= 0:
            raise ValueError(
                f"query_timeout_seconds must be > 0, got {self.query_timeout_seconds}"
            )


@dataclass
class ForecastConfig:
    """Settings for the forecasting model."""

    min_training_points: int = 14
    max_horizon_days: int = 90
    test_fraction: float = 0.2

    def __post_init__(self) -> None:
        if self.min_training_points < 3:
            raise ValueError(
                f"min_training_points must be >= 3, got {self.min_training_points}"
            )
        if self.max_horizon_days <= 0:
            raise ValueError(f"max_horizon_days must be > 0, got {self.max_horizon_days}")
        if not 0.0 < self.test_fraction < 1.0:
            raise ValueError(
                f"test_fraction must be in (0.0, 1.0), got {self.test_fraction}"
            )


@dataclass
class AgentConfig:
    """Settings shared by every agent's tool-use loop."""

    max_iterations: int = 8
    memory_turns: int = 6

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            raise ValueError(f"max_iterations must be > 0, got {self.max_iterations}")
        if self.memory_turns < 0:
            raise ValueError(f"memory_turns must be >= 0, got {self.memory_turns}")


@dataclass
class AppConfig:
    """The single composed configuration object for the application."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    forecast: ForecastConfig = field(default_factory=ForecastConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    log_level: LogLevel = "INFO"

    def log_active_config(self) -> None:
        """Emit every active hyperparameter so a run's logs are self-contained."""
        logger.info(
            "[CONFIG] model=%s max_tokens=%d temperature=%.2f",
            self.llm.model,
            self.llm.max_tokens,
            self.llm.temperature,
        )
        logger.info(
            "[CONFIG] db=%s max_rows=%d loop_max_iter=%d memory_turns=%d",
            self.database.path,
            self.database.max_rows,
            self.agent.max_iterations,
            self.agent.memory_turns,
        )
        logger.info(
            "[CONFIG] forecast_min_points=%d max_horizon=%d test_fraction=%.2f",
            self.forecast.min_training_points,
            self.forecast.max_horizon_days,
            self.forecast.test_fraction,
        )


def load_app_config(require_api_key: bool = False) -> AppConfig:
    """Build an :class:`AppConfig` from environment variables.

    Args:
        require_api_key: When True, a missing ``ANTHROPIC_API_KEY`` is fatal.
            Leave False for offline work and tests, which use the mock client.

    Returns:
        A fully validated configuration object.

    Raises:
        ConfigError: If ``require_api_key`` is True and no key is present.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY") or None
    if require_api_key and not api_key:
        raise ConfigError(
            "ANTHROPIC_API_KEY is not set; export it or copy .env.example to .env"
        )

    level = os.environ.get("ANALYST_LOG_LEVEL", "INFO").upper()
    if level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        raise ConfigError(
            f"ANALYST_LOG_LEVEL must be a valid level name, got {level!r}"
        )

    return AppConfig(
        llm=LLMConfig(
            model=os.environ.get("ANALYST_MODEL", DEFAULT_MODEL),
            api_key=api_key,
        ),
        database=DatabaseConfig(
            path=Path(os.environ.get("ANALYST_DB_PATH", "bikeshare.db"))
        ),
        log_level=level,  # type: ignore[arg-type]
    )
