"""Centralised logging configuration.

Call :func:`setup_logging` once at an entry point. Library modules must never
configure logging themselves; they only ever create a module-level logger.
"""

from __future__ import annotations

import logging
import sys
from typing import Literal, Optional

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

_FORMAT = "%(asctime)s %(levelname)-7s %(name)-28s %(message)s"
_DATEFMT = "%H:%M:%S"


def setup_logging(level: LogLevel = "INFO", stream: Optional[object] = None) -> None:
    """Configure root logging once, idempotently.

    Args:
        level: Minimum level to emit.
        stream: Optional stream to write to. Defaults to stderr.

    Raises:
        ValueError: If ``level`` is not a recognised logging level name.
    """
    numeric = getattr(logging, level, None)
    if not isinstance(numeric, int):
        raise ValueError(
            f"level must be one of DEBUG, INFO, WARNING, ERROR, CRITICAL, got {level!r}"
        )

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    root.addHandler(handler)
    root.setLevel(numeric)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)
