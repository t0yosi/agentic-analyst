from __future__ import annotations

import base64
import json

import pytest

from tools.chart_tools import MAX_POINTS, ChartToolset, render_chart
from utils.errors import ToolError, ValidationError


class TestRenderChart:
    @pytest.mark.parametrize("kind", ["line", "bar", "barh"])
    def test_renders_every_supported_kind(self, kind: str) -> None:
        encoded = render_chart(kind, ["a", "b"], [1.0, 2.0], "Title")
        assert base64.b64decode(encoded)[:4] == b"\x89PNG"

    def test_rejects_unknown_kind(self) -> None:
        with pytest.raises(ValidationError, match="kind must be one of"):
            render_chart("pie", ["a"], [1.0], "T")  # type: ignore[arg-type]

    def test_rejects_length_mismatch(self) -> None:
        with pytest.raises(ValidationError, match="same length, got 2 and 3"):
            render_chart("bar", ["a", "b"], [1.0, 2.0, 3.0], "T")

    @pytest.mark.parametrize(
        "labels,values", [([], [1.0]), (["a"], []), ([], [])]
    )
    def test_rejects_empty_series(self, labels, values) -> None:
        with pytest.raises(ValidationError, match="non-empty"):
            render_chart("bar", labels, values, "T")

    def test_rejects_oversized_series(self) -> None:
        n = MAX_POINTS + 1
        with pytest.raises(ValidationError, match="aggregate first"):
            render_chart("line", [str(i) for i in range(n)], [1.0] * n, "T")

    def test_accepts_exactly_the_maximum(self) -> None:
        n = MAX_POINTS
        assert render_chart("line", [str(i) for i in range(n)], [1.0] * n, "T")

    def test_single_point_is_allowed(self) -> None:
        assert render_chart("bar", ["only"], [42.0], "T")

    def test_accepts_integer_values(self) -> None:
        assert render_chart("bar", ["a", "b"], [1, 2], "T")

    def test_handles_negative_and_zero(self) -> None:
        assert render_chart("bar", ["a", "b", "c"], [-5.0, 0.0, 5.0], "T")


class TestChartToolset:
    def test_exposes_one_tool(self) -> None:
        assert [d["name"] for d in ChartToolset().definitions] == ["create_chart"]

    def test_dispatch_stores_the_chart(self) -> None:
        toolset = ChartToolset()
        assert toolset.last_chart is None
        payload = json.loads(
            toolset.dispatch(
                "create_chart",
                {"kind": "bar", "labels": ["a"], "values": [1.0], "title": "T"},
            )
        )
        assert payload["status"] == "rendered"
        assert toolset.last_chart is not None

    def test_dispatch_rejects_unknown_tool(self) -> None:
        with pytest.raises(ToolError, match="unknown tool"):
            ChartToolset().dispatch("draw_it", {})

    def test_dispatch_propagates_validation_errors(self) -> None:
        with pytest.raises(ValidationError):
            ChartToolset().dispatch(
                "create_chart", {"kind": "bar", "labels": ["a"], "values": [], "title": "T"}
            )

    def test_second_chart_replaces_the_first(self) -> None:
        toolset = ChartToolset()
        args = {"kind": "bar", "labels": ["a"], "values": [1.0], "title": "One"}
        toolset.dispatch("create_chart", args)
        first = toolset.last_chart
        toolset.dispatch("create_chart", {**args, "title": "Two", "values": [9.0]})
        assert toolset.last_chart != first
