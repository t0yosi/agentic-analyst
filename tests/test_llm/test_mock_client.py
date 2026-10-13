from __future__ import annotations

import pytest

from llm.base import LLMResponse, ToolCall
from llm.mock_client import MockLLMClient
from utils.errors import LLMError


class TestToolCall:
    def test_rejects_empty_id(self) -> None:
        with pytest.raises(ValueError, match="ToolCall.id must be a non-empty string"):
            ToolCall(id="", name="t", arguments={})

    def test_rejects_empty_name(self) -> None:
        with pytest.raises(ValueError, match="ToolCall.name must be a non-empty string"):
            ToolCall(id="1", name="", arguments={})


class TestLLMResponse:
    def test_wants_tools_is_false_without_calls(self) -> None:
        assert LLMResponse(text="done").wants_tools is False

    def test_wants_tools_is_true_with_calls(self) -> None:
        response = LLMResponse(tool_calls=[ToolCall(id="1", name="t", arguments={})])
        assert response.wants_tools is True


class TestMockLLMClient:
    def test_replays_in_order(self) -> None:
        client = MockLLMClient()
        client.queue_text("first").queue_text("second")
        assert client.complete([], "sys").text == "first"
        assert client.complete([], "sys").text == "second"

    def test_empty_queue_raises_a_useful_error(self) -> None:
        with pytest.raises(LLMError, match="queue another response in the test"):
            MockLLMClient().complete([], "sys")

    def test_error_reports_the_call_number(self) -> None:
        client = MockLLMClient()
        client.queue_text("only")
        client.complete([], "sys")
        with pytest.raises(LLMError, match="call #2"):
            client.complete([], "sys")

    def test_records_every_call(self) -> None:
        client = MockLLMClient()
        client.queue_text("x")
        client.complete([{"role": "user", "content": "hi"}], "sys-prompt", max_tokens=99)
        assert client.calls[0]["system_prompt"] == "sys-prompt"
        assert client.calls[0]["max_tokens"] == 99

    def test_records_tool_names_offered(self) -> None:
        client = MockLLMClient()
        client.queue_text("x")
        client.complete([], "s", tools=[{"name": "a"}, {"name": "b"}])
        assert client.calls[0]["tools"] == ["a", "b"]

    def test_pending_tracks_the_queue(self) -> None:
        client = MockLLMClient()
        assert client.pending == 0
        client.queue_text("a").queue_text("b")
        assert client.pending == 2
        client.complete([], "s")
        assert client.pending == 1

    def test_queue_returns_self_for_chaining(self) -> None:
        client = MockLLMClient()
        assert client.queue_text("a") is client

    def test_rejects_non_response_objects(self) -> None:
        with pytest.raises(TypeError, match="must be an LLMResponse"):
            MockLLMClient().queue("just a string")  # type: ignore[arg-type]

    def test_rejects_empty_model_name(self) -> None:
        with pytest.raises(ValueError, match="model_name must be a non-empty string"):
            MockLLMClient(model_name="")

    def test_exposes_its_model_name(self) -> None:
        assert MockLLMClient("m1").model_name == "m1"

    def test_does_not_mutate_the_caller_messages(self) -> None:
        client = MockLLMClient()
        client.queue_text("x")
        messages = [{"role": "user", "content": "hi"}]
        client.complete(messages, "s")
        assert messages == [{"role": "user", "content": "hi"}]
