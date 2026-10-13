"""Deterministic synthetic data generator.

Every value here is fabricated. The generator is seeded, so two runs on the
same version produce byte-identical databases, which is what lets the test
suite assert on concrete aggregate values.
"""

from __future__ import annotations

import logging
import math
import random
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

from database.connection import apply_schema, connect, table_counts

logger = logging.getLogger(__name__)

RANDOM_SEED = 20260828
START_DATE = date(2025, 1, 1)
END_DATE = date(2025, 12, 31)

BOROUGHS = ("Riverside", "Old Town", "Harbour", "Northgate", "University")
STATION_NAMES = (
    "Mill Bridge", "Canal Head", "Foundry Lane", "Market Cross", "Beacon Hill",
    "Quay Street", "Elm Terrace", "Station Approach", "Abbey Gate", "Willow Park",
    "Copper Row", "Lantern Yard", "Kiln Road", "Weaver Walk", "Cobble Square",
    "Granary Wharf", "Priory Fields", "Signal Box", "Tanner Street", "Vaulted Arch",
)
MEMBERSHIPS = ("casual", "monthly", "annual")
MEMBERSHIP_WEIGHTS = (0.45, 0.25, 0.30)


def _daily_demand(day: date, temp: float, precip: float) -> int:
    """Model expected trips for a day from season, weekday, and weather."""
    seasonal = 1.0 + 0.45 * math.sin((day.timetuple().tm_yday - 100) / 365.0 * 2 * math.pi)
    weekday = 1.0 if day.weekday() < 5 else 0.72
    warmth = max(0.35, min(1.35, (temp + 4.0) / 22.0))
    rain = max(0.30, 1.0 - precip * 0.06)
    return max(6, int(120 * seasonal * weekday * warmth * rain))


def _generate_weather(rng: random.Random) -> List[Tuple[str, float, float, float]]:
    rows: List[Tuple[str, float, float, float]] = []
    day = START_DATE
    while day <= END_DATE:
        seasonal = 11.0 + 9.5 * math.sin((day.timetuple().tm_yday - 100) / 365.0 * 2 * math.pi)
        temp = round(seasonal + rng.gauss(0, 2.6), 1)
        precip = round(max(0.0, rng.gauss(1.1, 2.8)), 1)
        wind = round(max(2.0, rng.gauss(14.0, 5.0)), 1)
        rows.append((day.isoformat(), temp, precip, wind))
        day += timedelta(days=1)
    return rows


def seed_database(db_path: Path, n_stations: int = 20, n_riders: int = 600) -> Dict[str, int]:
    """Create and populate the analytics database.

    Args:
        db_path: Destination SQLite file. An existing file is replaced.
        n_stations: Number of stations to generate. Must be positive.
        n_riders: Number of riders to generate. Must be positive.

    Returns:
        Row counts per table.

    Raises:
        ValueError: If a count argument is not positive, or ``n_stations``
            exceeds the pool of available station names.
    """
    if n_stations <= 0:
        raise ValueError(f"n_stations must be > 0, got {n_stations}")
    if n_riders <= 0:
        raise ValueError(f"n_riders must be > 0, got {n_riders}")
    if n_stations > len(STATION_NAMES):
        raise ValueError(
            f"n_stations must be <= {len(STATION_NAMES)} (available names), got {n_stations}"
        )

    rng = random.Random(RANDOM_SEED)
    if db_path.exists():
        db_path.unlink()

    conn = connect(db_path)
    try:
        apply_schema(conn)
        _insert_stations(conn, rng, n_stations)
        _insert_riders(conn, rng, n_riders)
        weather = _generate_weather(rng)
        conn.executemany("INSERT INTO weather VALUES (?, ?, ?, ?)", weather)
        _insert_trips(conn, rng, n_stations, n_riders, weather)
        conn.commit()
        counts = table_counts(conn)
    finally:
        conn.close()

    logger.info(
        "[DATA] seeded stations=%d riders=%d trips=%d weather=%d",
        counts.get("stations", 0),
        counts.get("riders", 0),
        counts.get("trips", 0),
        counts.get("weather", 0),
    )
    return counts


def _insert_stations(conn: sqlite3.Connection, rng: random.Random, n: int) -> None:
    rows = []
    for i in range(n):
        rows.append(
            (
                i + 1,
                STATION_NAMES[i],
                BOROUGHS[i % len(BOROUGHS)],
                rng.choice((12, 16, 20, 24, 32)),
                round(53.79 + rng.uniform(-0.05, 0.05), 5),
                round(-1.55 + rng.uniform(-0.05, 0.05), 5),
            )
        )
    conn.executemany("INSERT INTO stations VALUES (?, ?, ?, ?, ?, ?)", rows)


def _insert_riders(conn: sqlite3.Connection, rng: random.Random, n: int) -> None:
    rows = []
    for i in range(n):
        joined = START_DATE - timedelta(days=rng.randint(0, 900))
        rows.append(
            (
                i + 1,
                rng.choices(MEMBERSHIPS, weights=MEMBERSHIP_WEIGHTS, k=1)[0],
                joined.isoformat(),
                rng.randint(1955, 2007) if rng.random() > 0.08 else None,
            )
        )
    conn.executemany("INSERT INTO riders VALUES (?, ?, ?, ?)", rows)


def _insert_trips(
    conn: sqlite3.Connection,
    rng: random.Random,
    n_stations: int,
    n_riders: int,
    weather: List[Tuple[str, float, float, float]],
) -> None:
    trip_id = 1
    batch: List[Tuple[int, int, int, int, str, float, float]] = []

    for observed_date, temp, precip, _wind in weather:
        day = datetime.strptime(observed_date, "%Y-%m-%d").date()
        for _ in range(_daily_demand(day, temp, precip)):
            hour = rng.choices(
                range(24),
                weights=[1, 1, 1, 1, 2, 5, 12, 26, 34, 18, 12, 13,
                         16, 15, 13, 16, 27, 35, 26, 16, 10, 6, 3, 2],
                k=1,
            )[0]
            start = rng.randint(1, n_stations)
            end = rng.randint(1, n_stations)
            duration = round(max(2.0, rng.lognormvariate(2.5, 0.55)), 1)
            batch.append(
                (
                    trip_id,
                    rng.randint(1, n_riders),
                    start,
                    end,
                    f"{observed_date} {hour:02d}:{rng.randint(0, 59):02d}:00",
                    duration,
                    round(max(0.2, duration * rng.uniform(0.16, 0.24)), 2),
                )
            )
            trip_id += 1

    conn.executemany("INSERT INTO trips VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
