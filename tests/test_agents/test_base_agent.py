from __future__ import annotations

from typing import Any, Dict, List

import pytest

from agents.base import Agent, AgentResult
from config.settings import AgentConfig
from llm.base import LLMResponse, ToolCall
from llm.mock_client import MockLLMClient
from utils.errors import AgentError, LoopLimitExceeded, ToolError


class RecordingToolset:
    """A toolset that records calls and returns canned output."""

    def __init__(self, output: str = "tool output", fail: bool = False) -> None:
        self.output = output
        self.fail = fail
        self.calls: List[Dict[str, Any]] = []

    @property
    def definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "echo",
                "description": "Echo the argument back.",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            }
        ]

    def dispatch(self, name: str, arguments: Dict[str, Any]) -> str:
        self.calls.append({"name": name, "arguments": arguments})
        if self.fail:
            raise ToolError("deliberate failure")
        return self.output


class EchoAgent(Agent):
    @property
    def name(self) -> str:
        return "echo_agent"

    @property
    def system_prompt(self) -> str:
        return "You are a test agent."


def _tool_turn(call_id: str = "c1") -> LLMResponse:
    return LLMResponse(
        tool_calls=[ToolCall(id=call_id, name="echo", arguments={"x": 1})],
        stop_reason="tool_use",
        input_tokens=10,
        output_tokens=4,
    )


class TestConstruction:
    def test_requires_at_least_one_toolset(self, mock_client: MockLLMClient) -> None:
        with pytest.raises(ValueError, match="requires at least one toolset"):
            EchoAgent(mock_client, [])

    def test_flattens_tool_definitions(self, mock_client: MockLLMClient) -> None:
        agent = EchoAgent(mock_client, [RecordingToolset(), RecordingToolset()])
        assert len(agent.tool_definitions) == 2


class TestRun:
    def test_rejects_empty_query(self, mock_client: MockLLMClient) -> None:
        agent = EchoAgent(mock_client, [RecordingToolset()])
        with pytest.raises(ValueError, match="query must be a non-empty string"):
            agent.run("   ")

    def test_returns_immediately_when_no_tools_requested(self, mock_client: MockLLMClient) -> None:
        mock_client.queue_text("the answer")
        result = agent_run(mock_client, "question")
        assert result.text == "the answer"
        assert result.iterations == 1
        assert result.tool_calls == []

    def test_executes_a_requested_tool_then_answers(self, mock_client: MockLLMClient) -> None:
        toolset = RecordingToolset()
        mock_client.queue(_tool_turn()).queue_text("final")
        agent = EchoAgent(mock_client, [toolset])
        result = agent.run("q")
        assert result.text == "final"
        assert result.tool_calls == ["echo"]
        assert result.iterations == 2
        assert toolset.calls[0]["arguments"] == {"x": 1}

    def test_accumulates_tokens_across_turns(self, mock_client: MockLLMClient) -> None:
        mock_client.queue(_tool_turn()).queue(
            LLMResponse(text="done", input_tokens=7, output_tokens=3)
        )
        result = EchoAgent(mock_client, [RecordingToolset()]).run("q")
        assert result.input_tokens == 17
        assert result.output_tokens == 7
        assert result.total_tokens == 24

    def test_tool_failure_is_reported_to_the_model_not_raised(
        self, mock_client: MockLLMClient
    ) -> None:
        mock_client.queue(_tool_turn()).queue_text("recovered")
        agent = EchoAgent(mock_client, [RecordingToolset(fail=True)])
        result = agent.run("q")
        assert result.text == "recovered"
        blocks = mock_client.calls[1]["messages"][-1]["content"]
        assert blocks[0]["is_error"] is True
        assert "deliberate failure" in blocks[0]["content"]

    def test_raises_when_the_loop_never_terminates(self, mock_client: MockLLMClient) -> None:
        for i in range(4):
            mock_client.queue(_tool_turn(f"c{i}"))
        agent = EchoAgent(mock_client, [RecordingToolset()], AgentConfig(max_iterations=3))
        with pytest.raises(LoopLimitExceeded, match="limit of 3 iterations"):
            agent.run("q")

    def test_loop_limit_names_the_agent(self, mock_client: MockLLMClient) -> None:
        mock_client.queue(_tool_turn())
        agent = EchoAgent(mock_client, [RecordingToolset()], AgentConfig(max_iterations=1))
        with pytest.raises(LoopLimitExceeded, match="echo_agent"):
            agent.run("q")

    def test_empty_final_turn_is_an_error(self, mock_client: MockLLMClient) -> None:
        mock_client.queue(LLMResponse(text="", stop_reason="end_turn"))
        with pytest.raises(AgentError, match="neither text nor a tool call"):
            EchoAgent(mock_client, [RecordingToolset()]).run("q")

    def test_unknown_tool_is_surfaced_as_an_error_result(
        self, mock_client: MockLLMClient
    ) -> None:
        mock_client.queue(
            LLMResponse(tool_calls=[ToolCall(id="c", name="ghost", arguments={})])
        ).queue_text("ok")
        EchoAgent(mock_client, [RecordingToolset()]).run("q")
        blocks = mock_client.calls[1]["messages"][-1]["content"]
        assert "no toolset registered" in blocks[0]["content"]

    def test_history_is_prepended(self, mock_client: MockLLMClient) -> None:
        mock_client.queue_text("ok")
        history = [{"role": "user", "content": "earlier"}]
        EchoAgent(mock_client, [RecordingToolset()]).run("now", history=history)
        assert mock_client.calls[0]["messages"][0]["content"] == "earlier"
        assert mock_client.calls[0]["messages"][-1]["content"] == "now"

    def test_system_prompt_is_sent_every_turn(self, mock_client: MockLLMClient) -> None:
        mock_client.queue(_tool_turn()).queue_text("done")
        EchoAgent(mock_client, [RecordingToolset()]).run("q")
        assert all(c["system_prompt"] == "You are a test agent." for c in mock_client.calls)

    def test_handles_two_tool_calls_in_one_turn(self, mock_client: MockLLMClient) -> None:
        toolset = RecordingToolset()
        mock_client.queue(
            LLMResponse(
                tool_calls=[
                    ToolCall(id="a", name="echo", arguments={"n": 1}),
                    ToolCall(id="b", name="echo", arguments={"n": 2}),
                ]
            )
        ).queue_text("both done")
        result = EchoAgent(mock_client, [toolset]).run("q")
        assert result.tool_calls == ["echo", "echo"]
        assert len(toolset.calls) == 2


class TestAgentResult:
    def test_total_tokens_sums_both_directions(self) -> None:
        assert AgentResult(text="x", input_tokens=3, output_tokens=4).total_tokens == 7


def agent_run(client: MockLLMClient, query: str) -> AgentResult:
    return EchoAgent(client, [RecordingToolset()]).run(query)
