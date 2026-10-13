"""Forecasting tools backed by scikit-learn.

The forecast agent calls these to fit a model on a daily series pulled from the
database and project it forward. Every fit reports held-out error, so the
agent can tell the user how much to trust the projection.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from config.settings import DatabaseConfig, ForecastConfig
from tools.sql_tools import SQLToolset
from utils.errors import ToolError, ValidationError

logger = logging.getLogger(__name__)

_DAY = "%Y-%m-%d"


def _design_matrix(days: Sequence[int], weekdays: Sequence[int]) -> np.ndarray:
    """Build features: trend index plus a one-hot weekday encoding."""
    trend = np.asarray(days, dtype=float).reshape(-1, 1)
    onehot = np.zeros((len(days), 7), dtype=float)
    onehot[np.arange(len(days)), np.asarray(weekdays, dtype=int)] = 1.0
    return np.hstack([trend, onehot])


class ForecastToolset:
    """Fits and projects daily series drawn from the analytics database."""

    def __init__(self, db_config: DatabaseConfig, forecast_config: ForecastConfig) -> None:
        self._sql = SQLToolset(db_config)
        self._config = forecast_config

    @property
    def definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "forecast_daily_trips",
                "description": (
                    "Fit a ridge regression on daily trip counts, with a linear trend "
                    "and day-of-week effects, then project forward. Returns the "
                    "projection alongside held-out mean absolute error and R squared, "
                    "so report the error whenever you quote a forecast. Optionally "
                    "restrict to one borough."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "horizon_days": {
                            "type": "integer",
                            "description": (
                                "Days to project beyond the last observed date, "
                                f"1 to {self._config.max_horizon_days}."
                            ),
                        },
                        "borough": {
                            "type": "string",
                            "description": "Optional borough name to filter start stations by.",
                        },
                    },
                    "required": ["horizon_days"],
                },
            },
            {
                "name": "summarise_series",
                "description": (
                    "Return descriptive statistics for daily trip counts: count, mean, "
                    "standard deviation, min, max, and the busiest and quietest dates. "
                    "Use this before forecasting to understand the shape of the data."
                ),
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
        ]

    def dispatch(self, name: str, arguments: Dict[str, Any]) -> str:
        if name == "forecast_daily_trips":
            return self.forecast_daily_trips(
                int(arguments.get("horizon_days", 14)),
                arguments.get("borough"),
            )
        if name == "summarise_series":
            return self.summarise_series()
        raise ToolError(f"unknown tool for ForecastToolset: {name!r}")

    def _daily_series(self, borough: Any = None) -> Tuple[List[str], List[int]]:
        where = ""
        if borough:
            if not isinstance(borough, str) or not borough.replace(" ", "").isalnum():
                raise ValidationError(f"borough must be a simple name, got {borough!r}")
            where = (
                " JOIN stations s ON s.station_id = t.start_station_id "
                f" WHERE s.borough = '{borough}'"
            )
        sql = (
            "SELECT substr(t.started_at, 1, 10) AS day, COUNT(*) AS trips "
            f"FROM trips t{where} GROUP BY day ORDER BY day"
        )
        raw = self._sql.execute_sql(sql)
        if raw.startswith("The query returned no rows"):
            raise ToolError(
                f"no trips found for borough={borough!r}; check the borough name against the stations table"
            )
        rows = json.loads(raw)
        return [r["day"] for r in rows], [int(r["trips"]) for r in rows]

    def summarise_series(self) -> str:
        days, counts = self._daily_series()
        arr = np.asarray(counts, dtype=float)
        busiest = days[int(arr.argmax())]
        quietest = days[int(arr.argmin())]
        payload = {
            "days_observed": len(days),
            "first_date": days[0],
            "last_date": days[-1],
            "mean_trips_per_day": round(float(arr.mean()), 1),
            "std_trips_per_day": round(float(arr.std(ddof=1)), 1),
            "min_trips": int(arr.min()),
            "max_trips": int(arr.max()),
            "busiest_date": busiest,
            "quietest_date": quietest,
        }
        logger.info("[DATA] summarise_series days=%d mean=%.1f", len(days), arr.mean())
        return json.dumps(payload, indent=2)

    def forecast_daily_trips(self, horizon_days: int, borough: Any = None) -> str:
        if not 1 <= horizon_days <= self._config.max_horizon_days:
            raise ValidationError(
                f"horizon_days must be between 1 and {self._config.max_horizon_days}, "
                f"got {horizon_days}"
            )

        days, counts = self._daily_series(borough)
        if len(days) < self._config.min_training_points:
            raise ToolError(
                f"need at least {self._config.min_training_points} days of history to "
                f"forecast, found {len(days)}"
            )

        dates = [datetime.strptime(d, _DAY) for d in days]
        index = list(range(len(dates)))
        weekdays = [d.weekday() for d in dates]
        features = _design_matrix(index, weekdays)
        target = np.asarray(counts, dtype=float)

        split = int(len(target) * (1.0 - self._config.test_fraction))
        model = make_pipeline(PolynomialFeatures(degree=1, include_bias=False), Ridge(alpha=1.0))
        model.fit(features[:split], target[:split])
        held_out = model.predict(features[split:])
        mae = float(mean_absolute_error(target[split:], held_out))
        r2 = float(r2_score(target[split:], held_out))

        model.fit(features, target)
        future_dates = [dates[-1] + timedelta(days=i + 1) for i in range(horizon_days)]
        future = _design_matrix(
            [len(dates) + i for i in range(horizon_days)],
            [d.weekday() for d in future_dates],
        )
        projection = np.maximum(0.0, model.predict(future))

        logger.info(
            "[LATENCY] forecast fitted n=%d horizon=%d mae=%.1f r2=%.3f",
            len(target), horizon_days, mae, r2,
        )
        return json.dumps(
            {
                "borough": borough or "all",
                "training_days": len(target),
                "holdout_mae": round(mae, 1),
                "holdout_r2": round(r2, 3),
                "mean_observed_trips_per_day": round(float(target.mean()), 1),
                "forecast": [
                    {"date": d.strftime(_DAY), "predicted_trips": int(round(v))}
                    for d, v in zip(future_dates, projection)
                ],
            },
            indent=2,
        )
