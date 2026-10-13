-- Synthetic city bikeshare analytics schema.
-- All data is generated; no real riders, stations, or trips are represented.

CREATE TABLE IF NOT EXISTS stations (
    station_id   INTEGER PRIMARY KEY,
    name         TEXT    NOT NULL,
    borough      TEXT    NOT NULL,
    capacity     INTEGER NOT NULL,
    latitude     REAL    NOT NULL,
    longitude    REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS riders (
    rider_id       INTEGER PRIMARY KEY,
    membership     TEXT    NOT NULL CHECK (membership IN ('casual', 'monthly', 'annual')),
    joined_date    TEXT    NOT NULL,
    birth_year     INTEGER
);

CREATE TABLE IF NOT EXISTS trips (
    trip_id            INTEGER PRIMARY KEY,
    rider_id           INTEGER NOT NULL REFERENCES riders(rider_id),
    start_station_id   INTEGER NOT NULL REFERENCES stations(station_id),
    end_station_id     INTEGER NOT NULL REFERENCES stations(station_id),
    started_at         TEXT    NOT NULL,
    duration_minutes   REAL    NOT NULL,
    distance_km        REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS weather (
    observed_date   TEXT PRIMARY KEY,
    temp_celsius    REAL NOT NULL,
    precip_mm       REAL NOT NULL,
    wind_kph        REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_trips_started_at ON trips(started_at);
CREATE INDEX IF NOT EXISTS idx_trips_start_station ON trips(start_station_id);
CREATE INDEX IF NOT EXISTS idx_trips_rider ON trips(rider_id);
