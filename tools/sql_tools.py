"""Read-only SQL tools exposed to the database agent.

Safety note: the model can propose any string, so the guard here is a
whitelist, not a blacklist. Only a single ``SELECT`` (or ``WITH ... SELECT``)
statement is accepted, and a row ceiling is always applied.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

from config.settings import DatabaseConfig
from database.connection import connect, describe_schema
from utils.errors import ToolError, ValidationError

logger = logging.getLogger(__name__)

_ALLOWED_START = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)
_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|ATTACH|DETACH|PRAGMA|VACUUM)\b",
    re.IGNORECASE,
)
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_select(sql: str, max_statements: int = 1) -> str:
    """Return ``sql`` if it is a single read-only SELECT, else raise.

    Args:
        sql: Candidate statement.
        max_statements: Ceiling on statements in the string.

    Returns:
        The stripped, validated statement.

    Raises:
        ValidationError: If the statement is empty, is not a SELECT, contains a
            mutating keyword, or bundles multiple statements.
    """
    if not sql or not sql.strip():
        raise ValidationError("sql must be a non-empty statement, got empty")

    stripped = sql.strip().rstrip(";").strip()

    if not _ALLOWED_START.match(stripped):
        raise ValidationError(
            f"only SELECT or WITH queries are permitted, got: {stripped[:80]!r}"
        )
    if _FORBIDDEN.search(stripped):
        raise ValidationError(
            f"query contains a forbidden mutating keyword: {stripped[:80]!r}"
        )
    if stripped.count(";") >= max_statements:
        raise ValidationError(
            f"only {max_statements} statement is permitted per call, got {stripped.count(';') + 1}"
        )
    return stripped


class SQLToolset:
    """Bundles the database tools with the connection settings they need."""

    def __init__(self, config: DatabaseConfig) -> None:
        if not isinstance(config, DatabaseConfig):
            raise TypeError(f"config must be a DatabaseConfig, got {type(config).__name__}")
        self._config = config

    @property
    def definitions(self) -> List[Dict[str, Any]]:
        """JSON-schema definitions for every tool in this set."""
        return [
            {
                "name": "get_database_schema",
                "description": (
                    "Return every table, its columns and types, and its current row "
                    "count. Call this first when you do not already know the schema."
                ),
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "execute_sql",
                "description": (
                    "Run one read-only SQL SELECT against the analytics database and "
                    f"return rows as JSON. At most {self._config.max_rows} rows come "
                    "back, so aggregate in SQL rather than selecting raw rows, and add "
                    "an explicit LIMIT when you only need a sample. Only SELECT and "
                    "WITH are accepted; anything that writes is rejected."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "sql": {
                            "type": "string",
                            "description": "A single valid SQLite SELECT statement.",
                        }
                    },
                    "required": ["sql"],
                },
            },
            {
                "name": "get_table_sample",
                "description": (
                    "Return a handful of example rows from one table so you can see "
                    "the shape and typical values of the data before querying it."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "table": {"type": "string", "description": "Table name."},
                        "limit": {
                            "type": "integer",
                            "description": "Rows to return, 1 to 20. Defaults to 5.",
                        },
                    },
                    "required": ["table"],
                },
            },
        ]

    def dispatch(self, name: str, arguments: Dict[str, Any]) -> str:
        """Route a tool call to its implementation and return a string result."""
        if name == "get_database_schema":
            return self.get_database_schema()
        if name == "execute_sql":
            return self.execute_sql(str(arguments.get("sql", "")))
        if name == "get_table_sample":
            return self.get_table_sample(
                str(arguments.get("table", "")), int(arguments.get("limit", 5))
            )
        raise ToolError(f"unknown tool for SQLToolset: {name!r}")

    def get_database_schema(self) -> str:
        conn = self._open()
        try:
            return describe_schema(conn)
        finally:
            conn.close()

    def execute_sql(self, sql: str) -> str:
        statement = validate_select(sql)
        conn = self._open()
        try:
            rows = conn.execute(statement).fetchmany(self._config.max_rows)
        except sqlite3.Error as exc:
            raise ToolError(f"SQL execution failed for {statement[:120]!r}: {exc}") from exc
        finally:
            conn.close()

        payload = [dict(row) for row in rows]
        logger.info("[DATA] execute_sql rows=%d", len(payload))
        if not payload:
            return "The query returned no rows."
        return json.dumps(payload, default=str, indent=2)

    def get_table_sample(self, table: str, limit: int = 5) -> str:
        if not _IDENTIFIER.match(table or ""):
            raise ValidationError(
                f"table must be a plain identifier, got {table!r}"
            )
        if not 1 <= limit <= 20:
            raise ValidationError(f"limit must be between 1 and 20, got {limit}")

        conn = self._open()
        try:
            known = {
                row["name"]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            if table not in known:
                raise ValidationError(
                    f"table {table!r} does not exist; known tables are {sorted(known)}"
                )
            rows = conn.execute(f"SELECT * FROM {table} LIMIT ?", (limit,)).fetchall()
        finally:
            conn.close()
        return json.dumps([dict(r) for r in rows], default=str, indent=2)

    def _open(self) -> sqlite3.Connection:
        path: Path = self._config.path
        if not path.exists():
            raise ToolError(
                f"analytics database not found at {path}; run 'python main.py seed' first"
            )
        return connect(path, read_only=True)
