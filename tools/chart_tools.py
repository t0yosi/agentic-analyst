"""Chart rendering tools.

Charts are returned as base64 PNG so the API can embed them directly in a JSON
response without a static file server in the loop.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from typing import Any, Dict, List, Literal, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

from utils.errors import ToolError, ValidationError  # noqa: E402

logger = logging.getLogger(__name__)

ChartKind = Literal["line", "bar", "barh"]
_KINDS = ("line", "bar", "barh")
MAX_POINTS = 400


def render_chart(
    kind: ChartKind,
    labels: Sequence[str],
    values: Sequence[float],
    title: str,
    y_label: str = "",
) -> str:
    """Render a chart and return it as a base64-encoded PNG.

    Args:
        kind: One of ``line``, ``bar``, ``barh``.
        labels: X-axis categories, one per value.
        values: Numeric series to plot.
        title: Chart title.
        y_label: Optional axis label.

    Returns:
        Base64-encoded PNG bytes.

    Raises:
        ValidationError: If the series is empty, mismatched, oversized, or the
            chart kind is unknown.
    """
    if kind not in _KINDS:
        raise ValidationError(f"kind must be one of {_KINDS}, got {kind!r}")
    if not labels or not values:
        raise ValidationError("labels and values must both be non-empty")
    if len(labels) != len(values):
        raise ValidationError(
            f"labels and values must be the same length, got {len(labels)} and {len(values)}"
        )
    if len(values) > MAX_POINTS:
        raise ValidationError(
            f"series must be at most {MAX_POINTS} points, got {len(values)}; aggregate first"
        )

    numeric = [float(v) for v in values]
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=110)
    try:
        if kind == "line":
            ax.plot(range(len(numeric)), numeric, linewidth=1.8, color="#2b6cb0")
            ax.set_xticks(_tick_positions(len(labels)))
            ax.set_xticklabels(
                [labels[i] for i in _tick_positions(len(labels))], rotation=45, ha="right"
            )
        elif kind == "bar":
            ax.bar(range(len(numeric)), numeric, color="#2b6cb0")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
        else:
            ax.barh(range(len(numeric)), numeric, color="#2b6cb0")
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels)

        ax.set_title(title, fontsize=12, pad=12)
        if y_label:
            ax.set_ylabel(y_label)
        ax.grid(axis="y" if kind != "barh" else "x", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png")
        buffer.seek(0)
        encoded = base64.b64encode(buffer.read()).decode("ascii")
    finally:
        plt.close(fig)

    logger.info("[STAGE] chart rendered kind=%s points=%d bytes=%d", kind, len(numeric), len(encoded))
    return encoded


def _tick_positions(n: int, max_ticks: int = 12) -> List[int]:
    """Choose evenly spaced tick indices so dense axes stay readable."""
    if n <= max_ticks:
        return list(range(n))
    step = max(1, n // max_ticks)
    return list(range(0, n, step))


class ChartToolset:
    """Exposes chart rendering as a model-callable tool."""

    def __init__(self) -> None:
        self.last_chart: Optional[str] = None

    @property
    def definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "create_chart",
                "description": (
                    "Render a chart from data you already have and return it to the "
                    "user. Use 'line' for anything over time, 'bar' for comparisons "
                    "across a handful of categories, and 'barh' when category names "
                    f"are long. At most {MAX_POINTS} points, so aggregate before plotting."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "enum": list(_KINDS)},
                        "labels": {"type": "array", "items": {"type": "string"}},
                        "values": {"type": "array", "items": {"type": "number"}},
                        "title": {"type": "string"},
                        "y_label": {"type": "string"},
                    },
                    "required": ["kind", "labels", "values", "title"],
                },
            }
        ]

    def dispatch(self, name: str, arguments: Dict[str, Any]) -> str:
        if name != "create_chart":
            raise ToolError(f"unknown tool for ChartToolset: {name!r}")
        encoded = render_chart(
            kind=arguments.get("kind", "line"),
            labels=list(arguments.get("labels", [])),
            values=list(arguments.get("values", [])),
            title=str(arguments.get("title", "Chart")),
            y_label=str(arguments.get("y_label", "")),
        )
        self.last_chart = encoded
        return json.dumps(
            {"status": "rendered", "points": len(arguments.get("values", []))}
        )
