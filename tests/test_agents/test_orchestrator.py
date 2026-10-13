from __future__ import annotations

import pytest

from agents.orchestrator import Orchestrator, OrchestratorResult
from config.settings import AgentConfig, AppConfig
from llm.base import LLMResponse, ToolCall
from llm.mock_client import MockLLMClient


def delegate(agent_tool: str, instruction: str, call_id: str = "d1") -> LLMResponse:
    return LLMResponse(
        tool_calls=[ToolCall(id=call_id, name=agent_tool, arguments={"instruction": instruction})],
        stop_reason="tool_use",
        input_tokens=12,
        output_tokens=6,
    )


class TestDelegation:
    def test_exposes_exactly_three_agent_tools(
        self, mock_client: MockLLMClient, app_config: AppConfig
    ) -> None:
        names = {d["name"] for d in Orchestrator(mock_client, app_config).tool_definitions}
        assert names == {
            "call_database_agent", "call_forecast_agent", "call_visualization_agent"
        }

    def test_answers_without_delegating_when_unnecessary(
        self, mock_client: MockLLMClient, app_config: AppConfig
    ) -> None:
        mock_client.queue_text("No data needed.")
        result = Orchestrator(mock_client, app_config).ask("hello")
        assert result.agents_called == []
        assert result.answer == "No data needed."

    def test_delegates_to_the_database_agent(
        self, mock_client: MockLLMClient, app_config: AppConfig
    ) -> None:
        mock_client.queue(delegate("call_database_agent", "count the trips"))
        mock_client.queue(delegate("execute_sql", ""))  # consumed by the sub-agent
        mock_client.queue_text("There are many trips.")
        mock_client.queue_text("Final answer.")
        # sub-agent turn order: orchestrator -> db agent (2 turns) -> orchestrator
        result = Orchestrator(mock_client, app_config).ask("how many trips?")
        assert "call_database_agent" in result.agents_called

    def test_rejects_empty_question(
        self, mock_client: MockLLMClient, app_config: AppConfig
    ) -> None:
        with pytest.raises(ValueError, match="question must be a non-empty string"):
            Orchestrator(mock_client, app_config).ask("  ")

    def test_empty_instruction_is_reported_as_a_tool_error(
        self, mock_client: MockLLMClient, app_config: AppConfig
    ) -> None:
        mock_client.queue(delegate("call_database_agent", "")).queue_text("recovered")
        result = Orchestrator(mock_client, app_config).ask("q")
        assert result.answer == "recovered"
        blocks = mock_client.calls[1]["messages"][-1]["content"]
        assert "non-empty 'instruction'" in blocks[0]["content"]


class TestMemory:
    def test_remembers_across_turns(
        self, mock_client: MockLLMClient, app_config: AppConfig
    ) -> None:
        orchestrator = Orchestrator(mock_client, app_config)
        mock_client.queue_text("first").queue_text("second")
        orchestrator.ask("one")
        orchestrator.ask("two")
        assert mock_client.calls[1]["messages"][0]["content"] == "one"

    def test_remember_false_skips_history(
        self, mock_client: MockLLMClient, app_config: AppConfig
    ) -> None:
        orchestrator = Orchestrator(mock_client, app_config)
        mock_client.queue_text("a").queue_text("b")
        orchestrator.ask("one", remember=False)
        orchestrator.ask("two")
        assert mock_client.calls[1]["messages"][0]["content"] == "two"

    def test_history_is_trimmed_to_the_configured_window(
        self, mock_client: MockLLMClient, app_config: AppConfig
    ) -> None:
        orchestrator = Orchestrator(mock_client, app_config)  # memory_turns=2
        for i in range(5):
            mock_client.queue_text(f"answer {i}")
        for i in range(5):
            orchestrator.ask(f"question {i}")
        # window of 2 turns == 4 messages, plus the current question
        assert len(mock_client.calls[-1]["messages"]) == 5

    def test_reset_clears_history(
        self, mock_client: MockLLMClient, app_config: AppConfig
    ) -> None:
        orchestrator = Orchestrator(mock_client, app_config)
        mock_client.queue_text("a").queue_text("b")
        orchestrator.ask("one")
        orchestrator.reset_memory()
        orchestrator.ask("two")
        assert len(mock_client.calls[1]["messages"]) == 1


class TestResult:
    def test_total_tokens_sums_both_directions(self) -> None:
        result = OrchestratorResult(answer="a", input_tokens=5, output_tokens=6)
        assert result.total_tokens == 11

    def test_chart_is_none_when_nothing_was_plotted(
        self, mock_client: MockLLMClient, app_config: AppConfig
    ) -> None:
        mock_client.queue_text("no chart here")
        assert Orchestrator(mock_client, app_config).ask("q").chart_base64 is None

    def test_reports_iteration_count(
        self, mock_client: MockLLMClient, app_config: AppConfig
    ) -> None:
        mock_client.queue_text("done")
        assert Orchestrator(mock_client, app_config).ask("q").iterations == 1
