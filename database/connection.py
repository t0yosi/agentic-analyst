"""SQLite connection handling and schema introspection."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Dict, List

from utils.errors import ValidationError

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def connect(db_path: Path, read_only: bool = False) -> sqlite3.Connection:
    """Open a connection with row access by column name.

    Args:
        db_path: Path to the SQLite file.
        read_only: When True, open via a URI in read-only mode. The database
            file must already exist.

    Returns:
        An open connection.

    Raises:
        ValidationError: If ``read_only`` is requested for a missing file.
    """
    if read_only:
        if not db_path.exists():
            raise ValidationError(
                f"cannot open a missing database read-only: {db_path}; run the seed step first"
            )
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    else:
        conn = sqlite3.connect(db_path)

    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def apply_schema(conn: sqlite3.Connection) -> None:
    """Create every table and index if it does not already exist."""
    conn.executescript(SCHEMA_PATH.read_text())
    conn.commit()
    logger.info("[STAGE] schema applied")


def describe_schema(conn: sqlite3.Connection) -> str:
    """Render the live schema as text an agent can read.

    Returns:
        One block per table listing columns, types, and row counts.
    """
    tables = [
        row["name"]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]

    blocks: List[str] = []
    for table in tables:
        columns: List[str] = []
        for col in conn.execute(f"PRAGMA table_info({table})"):
            flag = " PRIMARY KEY" if col["pk"] else ""
            columns.append(f"  {col['name']} {col['type']}{flag}")
        count = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
        blocks.append(f"TABLE {table} ({count} rows)\n" + "\n".join(columns))

    logger.info("[DATA] schema described tables=%d", len(tables))
    return "\n\n".join(blocks)


def table_counts(conn: sqlite3.Connection) -> Dict[str, int]:
    """Return a mapping of table name to row count."""
    counts: Dict[str, int] = {}
    for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ):
        name = row["name"]
        counts[name] = conn.execute(f"SELECT COUNT(*) AS n FROM {name}").fetchone()["n"]
    return counts
