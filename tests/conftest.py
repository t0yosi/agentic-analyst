from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import AgentConfig, AppConfig, DatabaseConfig, ForecastConfig, LLMConfig
from database.seed import seed_database
from llm.base import LLMResponse, ToolCall
from llm.mock_client import MockLLMClient


@pytest.fixture(scope="session")
def seeded_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small seeded database shared across the session."""
    path = tmp_path_factory.mktemp("db") / "test.db"
    seed_database(path, n_stations=6, n_riders=40)
    return path


@pytest.fixture
def db_config(seeded_db: Path) -> DatabaseConfig:
    return DatabaseConfig(path=seeded_db, max_rows=200)


@pytest.fixture
def app_config(seeded_db: Path) -> AppConfig:
    return AppConfig(
        llm=LLMConfig(model="mock-model", api_key="test-key"),
        database=DatabaseConfig(path=seeded_db, max_rows=200),
        forecast=ForecastConfig(min_training_points=5),
        agent=AgentConfig(max_iterations=5, memory_turns=2),
    )


@pytest.fixture
def mock_client() -> MockLLMClient:
    return MockLLMClient()


def tool_response(name: str, arguments: dict, call_id: str = "call_1") -> LLMResponse:
    """Build a response in which the model requests one tool call."""
    return LLMResponse(
        tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
        stop_reason="tool_use",
        input_tokens=10,
        output_tokens=5,
    )


@pytest.fixture
def make_tool_response():
    return tool_response
