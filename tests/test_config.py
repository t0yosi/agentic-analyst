from __future__ import annotations

import pytest

from config.settings import (
    AgentConfig,
    AppConfig,
    DatabaseConfig,
    ForecastConfig,
    LLMConfig,
    load_app_config,
)
from utils.errors import ConfigError


class TestLLMConfig:
    def test_defaults_are_valid(self) -> None:
        assert LLMConfig().max_tokens == 2048

    @pytest.mark.parametrize("value", [0, -1, -2048])
    def test_rejects_non_positive_max_tokens(self, value: int) -> None:
        with pytest.raises(ValueError, match="max_tokens must be > 0"):
            LLMConfig(max_tokens=value)

    @pytest.mark.parametrize("value", [-0.1, 1.1, 5.0])
    def test_rejects_out_of_range_temperature(self, value: float) -> None:
        with pytest.raises(ValueError, match=r"temperature must be in \[0.0, 1.0\]"):
            LLMConfig(temperature=value)

    def test_rejects_empty_model(self) -> None:
        with pytest.raises(ValueError, match="model must be a non-empty string"):
            LLMConfig(model="")

    def test_error_message_names_the_bad_value(self) -> None:
        with pytest.raises(ValueError, match="got -5"):
            LLMConfig(max_tokens=-5)


class TestDatabaseConfig:
    def test_coerces_string_path_to_path(self) -> None:
        from pathlib import Path

        assert isinstance(DatabaseConfig(path="x.db").path, Path)

    @pytest.mark.parametrize("value", [0, -10])
    def test_rejects_non_positive_max_rows(self, value: int) -> None:
        with pytest.raises(ValueError, match="max_rows must be > 0"):
            DatabaseConfig(max_rows=value)


class TestForecastConfig:
    def test_rejects_too_few_training_points(self) -> None:
        with pytest.raises(ValueError, match="min_training_points must be >= 3"):
            ForecastConfig(min_training_points=2)

    @pytest.mark.parametrize("value", [0.0, 1.0, -0.5, 2.0])
    def test_rejects_out_of_range_test_fraction(self, value: float) -> None:
        with pytest.raises(ValueError, match=r"test_fraction must be in \(0.0, 1.0\)"):
            ForecastConfig(test_fraction=value)

    def test_rejects_non_positive_horizon(self) -> None:
        with pytest.raises(ValueError, match="max_horizon_days must be > 0"):
            ForecastConfig(max_horizon_days=0)


class TestAgentConfig:
    def test_rejects_non_positive_iterations(self) -> None:
        with pytest.raises(ValueError, match="max_iterations must be > 0"):
            AgentConfig(max_iterations=0)

    def test_rejects_negative_memory(self) -> None:
        with pytest.raises(ValueError, match="memory_turns must be >= 0"):
            AgentConfig(memory_turns=-1)

    def test_zero_memory_is_allowed(self) -> None:
        assert AgentConfig(memory_turns=0).memory_turns == 0


class TestLoadAppConfig:
    def test_reads_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        monkeypatch.setenv("ANALYST_MODEL", "custom-model")
        assert load_app_config().llm.model == "custom-model"

    def test_missing_key_is_fatal_when_required(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY is not set"):
            load_app_config(require_api_key=True)

    def test_missing_key_is_fine_when_not_required(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert load_app_config().llm.api_key is None

    def test_rejects_bad_log_level(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANALYST_LOG_LEVEL", "LOUD")
        with pytest.raises(ConfigError, match="ANALYST_LOG_LEVEL"):
            load_app_config()

    def test_log_active_config_emits_every_layer(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("INFO"):
            AppConfig().log_active_config()
        assert sum("[CONFIG]" in r.message for r in caplog.records) == 3
