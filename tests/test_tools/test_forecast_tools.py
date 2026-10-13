from __future__ import annotations

import json

import pytest

from config.settings import DatabaseConfig, ForecastConfig
from tools.forecast_tools import ForecastToolset
from utils.errors import ToolError, ValidationError


@pytest.fixture
def toolset(db_config: DatabaseConfig) -> ForecastToolset:
    return ForecastToolset(db_config, ForecastConfig(min_training_points=5))


class TestSummariseSeries:
    def test_returns_expected_keys(self, toolset: ForecastToolset) -> None:
        payload = json.loads(toolset.summarise_series())
        for key in (
            "days_observed", "first_date", "last_date", "mean_trips_per_day",
            "std_trips_per_day", "min_trips", "max_trips", "busiest_date", "quietest_date",
        ):
            assert key in payload

    def test_bounds_are_ordered(self, toolset: ForecastToolset) -> None:
        payload = json.loads(toolset.summarise_series())
        assert payload["min_trips"] <= payload["mean_trips_per_day"] <= payload["max_trips"]
        assert payload["first_date"] <= payload["last_date"]


class TestForecastDailyTrips:
    def test_returns_the_requested_horizon(self, toolset: ForecastToolset) -> None:
        payload = json.loads(toolset.forecast_daily_trips(7))
        assert len(payload["forecast"]) == 7

    def test_always_reports_error_metrics(self, toolset: ForecastToolset) -> None:
        payload = json.loads(toolset.forecast_daily_trips(3))
        assert "holdout_mae" in payload and "holdout_r2" in payload

    def test_predictions_are_never_negative(self, toolset: ForecastToolset) -> None:
        payload = json.loads(toolset.forecast_daily_trips(30))
        assert all(p["predicted_trips"] >= 0 for p in payload["forecast"])

    def test_dates_continue_after_the_last_observation(self, toolset: ForecastToolset) -> None:
        summary = json.loads(toolset.summarise_series())
        payload = json.loads(toolset.forecast_daily_trips(5))
        assert payload["forecast"][0]["date"] > summary["last_date"]

    def test_forecast_dates_are_strictly_increasing(self, toolset: ForecastToolset) -> None:
        dates = [p["date"] for p in json.loads(toolset.forecast_daily_trips(10))["forecast"]]
        assert dates == sorted(dates) and len(set(dates)) == len(dates)

    @pytest.mark.parametrize("horizon", [0, -1, 91, 1000])
    def test_rejects_out_of_range_horizon(self, toolset: ForecastToolset, horizon: int) -> None:
        with pytest.raises(ValidationError, match="horizon_days must be between"):
            toolset.forecast_daily_trips(horizon)

    def test_accepts_the_boundary_horizons(self, toolset: ForecastToolset) -> None:
        assert json.loads(toolset.forecast_daily_trips(1))["forecast"]
        assert len(json.loads(toolset.forecast_daily_trips(90))["forecast"]) == 90

    def test_unknown_borough_is_a_tool_error(self, toolset: ForecastToolset) -> None:
        with pytest.raises(ToolError, match="no trips found"):
            toolset.forecast_daily_trips(5, "Atlantis")

    def test_rejects_injection_in_borough(self, toolset: ForecastToolset) -> None:
        with pytest.raises(ValidationError, match="borough must be a simple name"):
            toolset.forecast_daily_trips(5, "x' OR '1'='1")

    def test_defaults_to_all_boroughs(self, toolset: ForecastToolset) -> None:
        assert json.loads(toolset.forecast_daily_trips(3))["borough"] == "all"


class TestDispatch:
    def test_exposes_two_tools(self, toolset: ForecastToolset) -> None:
        assert {d["name"] for d in toolset.definitions} == {
            "forecast_daily_trips", "summarise_series"
        }

    def test_routes_both_tools(self, toolset: ForecastToolset) -> None:
        assert toolset.dispatch("summarise_series", {})
        assert toolset.dispatch("forecast_daily_trips", {"horizon_days": 2})

    def test_rejects_unknown_tool(self, toolset: ForecastToolset) -> None:
        with pytest.raises(ToolError, match="unknown tool"):
            toolset.dispatch("predict_lottery", {})
