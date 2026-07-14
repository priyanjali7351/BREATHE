"""
api/database.py — SQLite persistence layer for the BREATHE FastAPI backend.

Tables
------
sensors         : one row per physical/simulated node (device_id, lat, lon)
readings        : time-series of PM2.5/PM10/temp/humidity per device
vitals          : time-series of HR/SpO2 per user (from Health Connect/HealthKit later)
user_baselines  : rolling 7-day median HR/SpO2 per user (used by the biometric modifier — step 5)
"""

import sqlite3
from pathlib import Path
from contextlib import contextmanager

DB_PATH = Path(__file__).parent / "breathe.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS sensors (
    device_id   TEXT PRIMARY KEY,
    name        TEXT,
    lat         REAL NOT NULL,
    lon         REAL NOT NULL,
    is_real     INTEGER NOT NULL DEFAULT 1,   -- 0 = simulated node (for IDW demo)
    created_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS readings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id   TEXT NOT NULL,
    pm25        REAL,
    pm10        REAL,
    temp_c      REAL,
    humidity    REAL,
    timestamp   TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (device_id) REFERENCES sensors(device_id)
);

CREATE TABLE IF NOT EXISTS vitals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT NOT NULL,
    hr          REAL,
    spo2        REAL,
    timestamp   TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS user_baselines (
    user_id         TEXT PRIMARY KEY,
    baseline_hr     REAL,
    baseline_spo2   REAL,
    updated_at      TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_readings_device_ts ON readings(device_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_vitals_user_ts ON vitals(user_id, timestamp);
"""


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        conn.commit()


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()