from __future__ import annotations

import json

import pytest

from config.settings import DatabaseConfig
from tools.sql_tools import SQLToolset, validate_select
from utils.errors import ToolError, ValidationError


class TestValidateSelect:
    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT 1",
            "select * from trips",
            "  SELECT COUNT(*) FROM trips  ",
            "WITH t AS (SELECT 1 AS x) SELECT x FROM t",
            "SELECT 1;",
        ],
    )
    def test_accepts_read_only_queries(self, sql: str) -> None:
        assert validate_select(sql)

    @pytest.mark.parametrize(
        "sql",
        [
            "DROP TABLE trips",
            "DELETE FROM trips",
            "UPDATE trips SET duration_minutes = 0",
            "INSERT INTO trips VALUES (1)",
            "ALTER TABLE trips ADD COLUMN x INT",
            "CREATE TABLE evil (id INT)",
            "PRAGMA table_info(trips)",
            "VACUUM",
            "ATTACH DATABASE 'other.db' AS o",
        ],
    )
    def test_rejects_every_mutating_statement(self, sql: str) -> None:
        with pytest.raises(ValidationError):
            validate_select(sql)

    def test_rejects_mutation_smuggled_after_a_select(self) -> None:
        with pytest.raises(ValidationError, match="forbidden mutating keyword"):
            validate_select("SELECT 1; DROP TABLE trips")

    def test_rejects_stacked_selects(self) -> None:
        with pytest.raises(ValidationError, match="only 1 statement is permitted"):
            validate_select("SELECT 1; SELECT 2")

    @pytest.mark.parametrize("sql", ["", "   ", "\n\t"])
    def test_rejects_empty(self, sql: str) -> None:
        with pytest.raises(ValidationError, match="non-empty"):
            validate_select(sql)

    def test_error_shows_the_offending_query(self) -> None:
        with pytest.raises(ValidationError, match="EXPLAIN"):
            validate_select("EXPLAIN SELECT 1")


class TestSQLToolset:
    def test_rejects_wrong_config_type(self) -> None:
        with pytest.raises(TypeError, match="config must be a DatabaseConfig"):
            SQLToolset({"path": "x"})  # type: ignore[arg-type]

    def test_exposes_three_tools(self, db_config: DatabaseConfig) -> None:
        names = {d["name"] for d in SQLToolset(db_config).definitions}
        assert names == {"get_database_schema", "execute_sql", "get_table_sample"}

    def test_every_definition_has_a_schema(self, db_config: DatabaseConfig) -> None:
        for definition in SQLToolset(db_config).definitions:
            assert definition["input_schema"]["type"] == "object"
            assert definition["description"]

    def test_row_ceiling_is_documented_in_the_description(self, db_config: DatabaseConfig) -> None:
        execute = next(
            d for d in SQLToolset(db_config).definitions if d["name"] == "execute_sql"
        )
        assert str(db_config.max_rows) in execute["description"]

    def test_schema_lists_all_tables(self, db_config: DatabaseConfig) -> None:
        schema = SQLToolset(db_config).get_database_schema()
        for table in ("stations", "riders", "trips", "weather"):
            assert f"TABLE {table}" in schema

    def test_executes_an_aggregate(self, db_config: DatabaseConfig) -> None:
        rows = json.loads(SQLToolset(db_config).execute_sql("SELECT COUNT(*) AS n FROM trips"))
        assert rows[0]["n"] > 0

    def test_reports_empty_result_in_words(self, db_config: DatabaseConfig) -> None:
        out = SQLToolset(db_config).execute_sql("SELECT * FROM trips WHERE trip_id < 0")
        assert "no rows" in out

    def test_caps_returned_rows(self, seeded_db) -> None:
        toolset = SQLToolset(DatabaseConfig(path=seeded_db, max_rows=7))
        assert len(json.loads(toolset.execute_sql("SELECT * FROM trips"))) == 7

    def test_bad_sql_raises_tool_error(self, db_config: DatabaseConfig) -> None:
        with pytest.raises(ToolError, match="SQL execution failed"):
            SQLToolset(db_config).execute_sql("SELECT nope FROM trips")

    def test_missing_database_names_the_fix(self, tmp_path) -> None:
        toolset = SQLToolset(DatabaseConfig(path=tmp_path / "absent.db"))
        with pytest.raises(ToolError, match="run 'python main.py seed' first"):
            toolset.execute_sql("SELECT 1")

    @pytest.mark.parametrize("table", ["trips; DROP TABLE trips", "trips--", "", "1trips"])
    def test_sample_rejects_non_identifiers(self, db_config: DatabaseConfig, table: str) -> None:
        with pytest.raises(ValidationError):
            SQLToolset(db_config).get_table_sample(table)

    def test_sample_rejects_unknown_table(self, db_config: DatabaseConfig) -> None:
        with pytest.raises(ValidationError, match="does not exist"):
            SQLToolset(db_config).get_table_sample("ghosts")

    @pytest.mark.parametrize("limit", [0, -1, 21, 100])
    def test_sample_rejects_bad_limit(self, db_config: DatabaseConfig, limit: int) -> None:
        with pytest.raises(ValidationError, match="limit must be between 1 and 20"):
            SQLToolset(db_config).get_table_sample("trips", limit)

    def test_sample_honours_limit(self, db_config: DatabaseConfig) -> None:
        assert len(json.loads(SQLToolset(db_config).get_table_sample("stations", 3))) == 3

    def test_dispatch_routes_each_tool(self, db_config: DatabaseConfig) -> None:
        toolset = SQLToolset(db_config)
        assert "TABLE trips" in toolset.dispatch("get_database_schema", {})
        assert toolset.dispatch("execute_sql", {"sql": "SELECT 1 AS x"})
        assert toolset.dispatch("get_table_sample", {"table": "stations", "limit": 2})

    def test_dispatch_rejects_unknown_tool(self, db_config: DatabaseConfig) -> None:
        with pytest.raises(ToolError, match="unknown tool"):
            SQLToolset(db_config).dispatch("rm_rf", {})
