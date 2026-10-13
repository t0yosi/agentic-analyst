from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from database.connection import connect, describe_schema, table_counts
from database.seed import seed_database
from utils.errors import ValidationError


class TestSeedValidation:
    @pytest.mark.parametrize("n", [0, -1])
    def test_rejects_non_positive_stations(self, tmp_path: Path, n: int) -> None:
        with pytest.raises(ValueError, match="n_stations must be > 0"):
            seed_database(tmp_path / "a.db", n_stations=n)

    @pytest.mark.parametrize("n", [0, -5])
    def test_rejects_non_positive_riders(self, tmp_path: Path, n: int) -> None:
        with pytest.raises(ValueError, match="n_riders must be > 0"):
            seed_database(tmp_path / "a.db", n_riders=n)

    def test_rejects_more_stations_than_names(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="available names"):
            seed_database(tmp_path / "a.db", n_stations=999)


class TestSeededData:
    def test_populates_every_table(self, seeded_db: Path) -> None:
        conn = connect(seeded_db, read_only=True)
        try:
            counts = table_counts(conn)
        finally:
            conn.close()
        assert counts["stations"] == 6
        assert counts["riders"] == 40
        assert counts["weather"] == 365
        assert counts["trips"] > 1000

    def test_is_deterministic(self, tmp_path: Path) -> None:
        a = seed_database(tmp_path / "a.db", n_stations=5, n_riders=20)
        b = seed_database(tmp_path / "b.db", n_stations=5, n_riders=20)
        assert a == b

    def test_reseeding_replaces_rather_than_appends(self, tmp_path: Path) -> None:
        path = tmp_path / "r.db"
        first = seed_database(path, n_stations=5, n_riders=20)
        second = seed_database(path, n_stations=5, n_riders=20)
        assert first["trips"] == second["trips"]

    def test_trips_reference_real_stations(self, seeded_db: Path) -> None:
        conn = connect(seeded_db, read_only=True)
        try:
            orphans = conn.execute(
                "SELECT COUNT(*) AS n FROM trips t "
                "LEFT JOIN stations s ON s.station_id = t.start_station_id "
                "WHERE s.station_id IS NULL"
            ).fetchone()["n"]
        finally:
            conn.close()
        assert orphans == 0

    def test_durations_are_positive(self, seeded_db: Path) -> None:
        conn = connect(seeded_db, read_only=True)
        try:
            bad = conn.execute(
                "SELECT COUNT(*) AS n FROM trips WHERE duration_minutes <= 0"
            ).fetchone()["n"]
        finally:
            conn.close()
        assert bad == 0

    def test_memberships_are_from_the_allowed_set(self, seeded_db: Path) -> None:
        conn = connect(seeded_db, read_only=True)
        try:
            values = {r["membership"] for r in conn.execute("SELECT DISTINCT membership FROM riders")}
        finally:
            conn.close()
        assert values <= {"casual", "monthly", "annual"}

    def test_weather_covers_a_full_year(self, seeded_db: Path) -> None:
        conn = connect(seeded_db, read_only=True)
        try:
            row = conn.execute(
                "SELECT MIN(observed_date) AS lo, MAX(observed_date) AS hi FROM weather"
            ).fetchone()
        finally:
            conn.close()
        assert row["lo"] == "2025-01-01" and row["hi"] == "2025-12-31"


class TestConnection:
    def test_read_only_blocks_writes(self, seeded_db: Path) -> None:
        conn = connect(seeded_db, read_only=True)
        try:
            with pytest.raises(sqlite3.OperationalError):
                conn.execute("DELETE FROM trips")
        finally:
            conn.close()

    def test_read_only_requires_an_existing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError, match="run the seed step first"):
            connect(tmp_path / "nope.db", read_only=True)

    def test_describe_schema_reports_row_counts(self, seeded_db: Path) -> None:
        conn = connect(seeded_db, read_only=True)
        try:
            text = describe_schema(conn)
        finally:
            conn.close()
        assert "TABLE stations (6 rows)" in text
