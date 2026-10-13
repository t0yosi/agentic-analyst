from __future__ import annotations

import os
from pathlib import Path

import pytest

from config.settings import (
    DEFAULT_ENV_FILE,
    PROJECT_ROOT,
    AgentConfig,
    AppConfig,
    DatabaseConfig,
    ForecastConfig,
    LLMConfig,
    load_app_config,
    load_env_file,
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


class TestLoadEnvFile:
    def test_missing_file_is_not_an_error(self, tmp_path: Path) -> None:
        assert load_env_file(tmp_path / "absent.env") is False

    def test_reads_values_into_the_environment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANALYST_MODEL", raising=False)
        env = tmp_path / ".env"
        env.write_text("ANALYST_MODEL=from-the-file\n")
        assert load_env_file(env) is True
        assert os.environ["ANALYST_MODEL"] == "from-the-file"

    def test_does_not_overwrite_an_exported_value(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANALYST_MODEL", "from-the-export")
        env = tmp_path / ".env"
        env.write_text("ANALYST_MODEL=from-the-file\n")
        load_env_file(env)
        assert os.environ["ANALYST_MODEL"] == "from-the-export"

    def test_ignores_comments_and_blank_lines(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANALYST_DB_PATH", raising=False)
        env = tmp_path / ".env"
        env.write_text("# a comment\n\nANALYST_DB_PATH=from-file.db\n")
        load_env_file(env)
        assert os.environ["ANALYST_DB_PATH"] == "from-file.db"

    def test_default_path_is_resolved_from_the_package_not_the_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        assert DEFAULT_ENV_FILE == PROJECT_ROOT / ".env"
        assert DEFAULT_ENV_FILE.is_absolute()


class TestLoadAppConfigReadsEnvFile:
    def test_api_key_comes_from_the_env_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The shipped bug: SETUP.md told users to write .env, but nothing read
        # it, so this raised ConfigError with a key sitting right there.
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        env = tmp_path / ".env"
        env.write_text("ANTHROPIC_API_KEY=key-from-file\n")
        config = load_app_config(require_api_key=True, env_file=env)
        assert config.llm.api_key == "key-from-file"

    def test_model_comes_from_the_env_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANALYST_MODEL", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        env = tmp_path / ".env"
        env.write_text("ANALYST_MODEL=file-model\n")
        assert load_app_config(env_file=env).llm.model == "file-model"

    def test_still_raises_when_no_key_anywhere(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY is not set"):
            load_app_config(require_api_key=True, env_file=tmp_path / "absent.env")
