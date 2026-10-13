from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from config.settings import AppConfig
from llm.base import LLMResponse, ToolCall
from llm.mock_client import MockLLMClient


@pytest.fixture
def client(app_config: AppConfig, mock_client: MockLLMClient) -> TestClient:
    return TestClient(create_app(app_config, mock_client))


class TestHealth:
    def test_reports_ok(self, client: TestClient) -> None:
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["database_ready"] is True

    def test_reports_the_model_name(self, client: TestClient) -> None:
        assert client.get("/health").json()["model"] == "mock-model"

    def test_reports_table_counts(self, client: TestClient) -> None:
        assert client.get("/health").json()["tables"]["stations"] == 6


class TestIndex:
    def test_serves_the_chat_page(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert "Agentic Analyst" in response.text


class TestAsk:
    def test_returns_an_answer(self, client: TestClient, mock_client: MockLLMClient) -> None:
        mock_client.queue_text("42 trips.")
        body = client.post("/ask", json={"question": "how many?"}).json()
        assert body["answer"] == "42 trips."
        assert body["agents_called"] == []

    def test_rejects_an_empty_question(self, client: TestClient) -> None:
        assert client.post("/ask", json={"question": ""}).status_code == 422

    def test_rejects_a_missing_question(self, client: TestClient) -> None:
        assert client.post("/ask", json={}).status_code == 422

    def test_rejects_an_oversized_question(self, client: TestClient) -> None:
        assert client.post("/ask", json={"question": "x" * 2001}).status_code == 422

    def test_returns_a_chart_when_one_was_rendered(
        self, client: TestClient, mock_client: MockLLMClient
    ) -> None:
        mock_client.queue(
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="v1",
                        name="call_visualization_agent",
                        arguments={"instruction": "plot 1 and 2"},
                    )
                ]
            )
        )
        mock_client.queue(
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="create_chart",
                        arguments={
                            "kind": "bar",
                            "labels": ["a", "b"],
                            "values": [1, 2],
                            "title": "T",
                        },
                    )
                ]
            )
        )
        mock_client.queue_text("Chart done.")
        mock_client.queue_text("Here is your chart.")
        body = client.post("/ask", json={"question": "chart it"}).json()
        assert body["chart_base64"]
        assert body["agents_called"] == ["call_visualization_agent"]

    def test_surfaces_an_analyst_error_as_422(
        self, client: TestClient, mock_client: MockLLMClient
    ) -> None:
        mock_client.queue(LLMResponse(text="", stop_reason="end_turn"))
        assert client.post("/ask", json={"question": "q"}).status_code == 422


class TestReset:
    def test_clears_memory(self, client: TestClient, mock_client: MockLLMClient) -> None:
        mock_client.queue_text("a").queue_text("b")
        client.post("/ask", json={"question": "one"})
        assert client.post("/reset").json()["status"] == "memory cleared"
        client.post("/ask", json={"question": "two"})
        assert len(mock_client.calls[1]["messages"]) == 1
