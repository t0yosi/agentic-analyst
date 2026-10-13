from __future__ import annotations

from pathlib import Path

import pytest

from prompts.manager import PromptManager, load_prompt
from utils.errors import ConfigError

COMPONENTS = ["orchestrator", "database_agent", "forecast_agent", "visualization_agent"]


class TestPromptManager:
    @pytest.mark.parametrize("component", COMPONENTS)
    def test_every_component_has_a_prompt(self, component: str) -> None:
        assert load_prompt(component)

    @pytest.mark.parametrize("component", COMPONENTS)
    def test_versions_are_discovered(self, component: str) -> None:
        assert PromptManager().available_versions(component) == [1]

    def test_unknown_component_raises(self) -> None:
        with pytest.raises(ConfigError, match="no prompt directory"):
            PromptManager().load("nonexistent_agent")

    def test_unknown_version_reports_what_exists(self) -> None:
        with pytest.raises(ConfigError, match="available versions are"):
            PromptManager().load("orchestrator", version=99)

    def test_caches_repeated_loads(self) -> None:
        manager = PromptManager()
        assert manager.load("orchestrator") is manager.load("orchestrator")

    def test_empty_prompt_file_is_rejected(self, tmp_path: Path) -> None:
        (tmp_path / "blank").mkdir()
        (tmp_path / "blank" / "v1.txt").write_text("   ")
        with pytest.raises(ConfigError, match="is empty"):
            PromptManager(tmp_path).load("blank")

    def test_directory_without_versions_is_rejected(self, tmp_path: Path) -> None:
        (tmp_path / "empty").mkdir()
        with pytest.raises(ConfigError, match="no vN.txt files"):
            PromptManager(tmp_path).load("empty")

    def test_resolves_the_highest_version(self, tmp_path: Path) -> None:
        (tmp_path / "multi").mkdir()
        (tmp_path / "multi" / "v1.txt").write_text("old")
        (tmp_path / "multi" / "v2.txt").write_text("new")
        assert PromptManager(tmp_path).load("multi") == "new"

    def test_explicit_version_wins(self, tmp_path: Path) -> None:
        (tmp_path / "multi").mkdir()
        (tmp_path / "multi" / "v1.txt").write_text("old")
        (tmp_path / "multi" / "v2.txt").write_text("new")
        assert PromptManager(tmp_path).load("multi", version=1) == "old"


class TestPromptContent:
    def test_orchestrator_states_the_protocol(self) -> None:
        prompt = load_prompt("orchestrator")
        for step in ("PLAN", "GATHER", "CORRELATE", "SYNTHESISE"):
            assert step in prompt

    def test_orchestrator_forbids_inventing_numbers(self) -> None:
        assert "Never invent a number" in load_prompt("orchestrator")

    def test_visualization_prompt_forbids_inventing_data(self) -> None:
        assert "Never invent data points" in load_prompt("visualization_agent")
