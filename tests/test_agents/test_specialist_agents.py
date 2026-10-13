from __future__ import annotations

import pytest

from agents.database_agent import DatabaseAgent
from agents.forecast_agent import ForecastAgent
from agents.visualization_agent import VisualizationAgent
from config.settings import AppConfig
from llm.base import LLMResponse, ToolCall
from llm.mock_client import MockLLMClient


class TestDatabaseAgent:
    def test_name_and_prompt(self, mock_client: MockLLMClient, app_config: AppConfig) -> None:
        agent = DatabaseAgent(mock_client, app_config.database)
        assert agent.name == "database_agent"
        assert "Database Agent" in agent.system_prompt

    def test_prompt_forbids_inventing_figures(
        self, mock_client: MockLLMClient, app_config: AppConfig
    ) -> None:
        prompt = DatabaseAgent(mock_client, app_config.database).system_prompt
        assert "never estimate" in prompt.lower()

    def test_offers_the_sql_tools(
        self, mock_client: MockLLMClient, app_config: AppConfig
    ) -> None:
        names = {d["name"] for d in DatabaseAgent(mock_client, app_config.database).tool_definitions}
        assert "execute_sql" in names

    def test_runs_a_real_query_end_to_end(
        self, mock_client: MockLLMClient, app_config: AppConfig
    ) -> None:
        mock_client.queue(
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="q1",
                        name="execute_sql",
                        arguments={"sql": "SELECT COUNT(*) AS n FROM trips"},
                    )
                ]
            )
        ).queue_text("Counted them.")
        result = DatabaseAgent(mock_client, app_config.database).run("count trips")
        assert result.tool_calls == ["execute_sql"]
        tool_output = mock_client.calls[1]["messages"][-1]["content"][0]["content"]
        assert '"n"' in tool_output


class TestForecastAgent:
    def test_name_and_prompt(self, mock_client: MockLLMClient, app_config: AppConfig) -> None:
        agent = ForecastAgent(mock_client, app_config.database, app_config.forecast)
        assert agent.name == "forecast_agent"
        assert "Forecast Agent" in agent.system_prompt

    def test_prompt_requires_reporting_error(
        self, mock_client: MockLLMClient, app_config: AppConfig
    ) -> None:
        prompt = ForecastAgent(mock_client, app_config.database).system_prompt
        assert "holdout_mae" in prompt

    def test_offers_the_forecast_tools(
        self, mock_client: MockLLMClient, app_config: AppConfig
    ) -> None:
        names = {
            d["name"]
            for d in ForecastAgent(mock_client, app_config.database).tool_definitions
        }
        assert names == {"forecast_daily_trips", "summarise_series"}


class TestVisualizationAgent:
    def test_name_and_prompt(self, mock_client: MockLLMClient) -> None:
        agent = VisualizationAgent(mock_client)
        assert agent.name == "visualization_agent"
        assert "Visualization Agent" in agent.system_prompt

    def test_starts_with_no_chart(self, mock_client: MockLLMClient) -> None:
        assert VisualizationAgent(mock_client).last_chart is None

    def test_renders_and_retains_a_chart(self, mock_client: MockLLMClient) -> None:
        mock_client.queue(
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="create_chart",
                        arguments={
                            "kind": "bar",
                            "labels": ["Mon", "Tue"],
                            "values": [10, 20],
                            "title": "Trips",
                        },
                    )
                ]
            )
        ).queue_text("Here is the chart.")
        agent = VisualizationAgent(mock_client)
        agent.run("plot it")
        assert agent.last_chart is not None
