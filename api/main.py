"""
api/main.py — BREATHE FastAPI backend (Step 1 of the prototype build).

Endpoints: /health, /ingest, /predict, /vitals

Known placeholders in THIS step (to be replaced in later steps, not hidden):
  - /predict uses IDW (p=2, default 500m trust radius) per the doc's Claim 4.
    If no sensor is within radius, falls back to nearest-sensor-regardless-
    of-distance (flagged in the response `source` field) rather than the
    doc's city-snapshot fallback, which isn't built yet.
  - pollutants passed to compute_phrs only include PM2.5/PM10 (what the
    node reports) -> other pollutants default to 0 until a fuller sensor
    or city-snapshot fallback is added
  - No trained AQI Forecaster model is loaded (models/*.joblib are not in
    the repo — data/INDIA_AQI_COMPLETE_20251126.csv referenced by train.py
    is also missing). /predict returns the CPCB-converted current AQI only;
    trend/forecast field is omitted rather than faked.
  - No auth — fine for a local demo, not for anything public-facing
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # import repo-root modules

from fastapi import FastAPI, HTTPException
from generate_profiles import compute_phrs, HealthProfile, phrs_category

from database import init_db, get_conn
from schemas import SensorReadingIn, VitalsIn, PredictIn, PredictOut
from aqi_convert import pm25_to_aqi
from idw import idw_estimate, SensorPoint, _haversine_m

app = FastAPI(title="BREATHE API", version="0.1.0-step1")

@app.get("/")
def root():
    return {"status": "ok"}


@app.on_event("startup")
def _startup():
    init_db()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
def ingest(reading: SensorReadingIn):
    with get_conn() as conn:
        existing = conn.execute(
            "SELECT device_id FROM sensors WHERE device_id = ?", (reading.device_id,)
        ).fetchone()
        if existing is None:
            if reading.lat is None or reading.lon is None:
                raise HTTPException(
                    400, "First reading from a new device_id must include lat/lon."
                )
            conn.execute(
                "INSERT INTO sensors (device_id, lat, lon, is_real) VALUES (?, ?, ?, ?)",
                (reading.device_id, reading.lat, reading.lon, int(reading.is_real)),
            )
        conn.execute(
            """INSERT INTO readings (device_id, pm25, pm10, temp_c, humidity)
               VALUES (?, ?, ?, ?, ?)""",
            (reading.device_id, reading.pm25, reading.pm10, reading.temp_c, reading.humidity),
        )
        conn.commit()
    return {"status": "stored", "device_id": reading.device_id}


@app.post("/vitals")
def vitals(v: VitalsIn):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO vitals (user_id, hr, spo2) VALUES (?, ?, ?)",
            (v.user_id, v.hr, v.spo2),
        )
        conn.commit()
    return {"status": "stored", "user_id": v.user_id}


def _latest_reading_per_sensor(conn) -> list[SensorPoint]:
    """One SensorPoint per device, using its most recent reading."""
    rows = conn.execute(
        """
        SELECT s.device_id, s.lat, s.lon, r.pm25, r.pm10
        FROM sensors s
        JOIN readings r ON r.id = (
            SELECT id FROM readings WHERE device_id = s.device_id
            ORDER BY timestamp DESC LIMIT 1
        )
        """
    ).fetchall()
    return [
        SensorPoint(device_id=r["device_id"], lat=r["lat"], lon=r["lon"],
                    pm25=r["pm25"], pm10=r["pm10"])
        for r in rows
    ]


@app.post("/predict", response_model=PredictOut)
def predict(req: PredictIn):
    with get_conn() as conn:
        points = _latest_reading_per_sensor(conn)

    if not points:
        return PredictOut(
            phrs=0, category="Unknown", color="#888888",
            aqi_used=0, source="no_data", device_id=None,
        )

    result = idw_estimate(req.lat, req.lon, points,
                           power=2.0, trust_radius_m=req.trust_radius_m)
    source = "idw_mesh"

    if result is None:
        # No sensor within trust radius — fall back to nearest regardless of
        # distance. NOT the doc's city-snapshot fallback (not built yet).
        nearest = min(points, key=lambda s: _haversine_m(req.lat, req.lon, s.lat, s.lon))
        result = {"pm25": nearest.pm25, "pm10": nearest.pm10,
                   "n_sensors_used": 1, "distances_m": None}
        source = "nearest_fallback_out_of_radius"

    aqi = pm25_to_aqi(result["pm25"])
    pollutants = {"PM2.5": result["pm25"], "PM10": result["pm10"]}
    profile = HealthProfile(
        age=req.age,
        condition=req.conditions[0] if req.conditions else "Healthy",
        activity_level=req.activity_level,
        hours_outdoors=req.hours_outdoors,
    )

    phrs = compute_phrs(
        aqi=aqi,
        pollutants=pollutants,
        profile=profile,
        conditions=req.conditions,
    )
    category, color = phrs_category(phrs)

    return PredictOut(
        phrs=phrs, category=category, color=color,
        aqi_used=aqi, source=source,
        device_id=None, n_sensors_used=result["n_sensors_used"],
    )