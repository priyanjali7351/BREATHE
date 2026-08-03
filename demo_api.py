"""
demo_api.py — BREATHE / BioAQI live demo backend.

Fuses the three real data sources into one live PHRS for the dashboard:
  1. City air quality      — realtime.py (Open-Meteo, cached)      → "Auto mode"
  2. Local particulate air — gp2y10.py (GP2Y10 dust sensor)       → "Sensor mode"
  3. Body / exertion       — watch_hr.py (BLE smartwatch HR)      → activity level

Serves the static dashboard in frontend/ and a single JSON endpoint the page
polls once a second. Hardware (BLE watch, serial dust sensor) is started
ON DEMAND via /api/hardware/connect so the server runs fine with nothing
plugged in — and a `sim` mode lets you drive HR/PM by hand for filming or as a
live fallback.

Run:  uvicorn demo_api:app --reload
Open: http://127.0.0.1:8000
"""

import time
from pathlib import Path

import sys

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# After the project reorg the ML pipeline modules (generate_profiles, preprocess)
# live in ml/pipelines/ and use bare sibling imports, so put that dir on sys.path
# and import them as bare modules — matching how they're designed to run. This
# also lets realtime.py's `from preprocess import ...` resolve.
_ML_PIPELINES = Path(__file__).resolve().parent / "ml" / "pipelines"
if _ML_PIPELINES.exists():
    sys.path.insert(0, str(_ML_PIPELINES))

import realtime
import gp2y10
import watch_hr
from generate_profiles import (
    compute_phrs, compute_phrs_horizons, condition_weight_for,
    HealthProfile, phrs_category,
)
from models import predict_aqi_smooth

app = FastAPI(title="BREATHE Live Demo", version="1.0")

FRONTEND_DIR = Path(__file__).parent / "frontend"

# ── City AQI cache ────────────────────────────────────────────────────────────
# City air changes slowly; cache each city's fetch so polling /api/live every
# second never hammers Open-Meteo (and stays fast — the network call is ~2s).
_AQI_TTL = 300  # seconds
_aqi_cache: dict[str, tuple[float, dict]] = {}
_last_hr_log = None  # throttle the per-request HR debug log
_last_sensor_log = None  # throttle the per-request sensor debug log


def _city_aqi(city: str) -> dict:
    now = time.time()
    hit = _aqi_cache.get(city)
    if hit and (now - hit[0]) < _AQI_TTL:
        return hit[1]
    data = realtime.fetch_realtime_data(city)
    if not data.get("error"):
        _aqi_cache[city] = (now, data)
    return data


# ── HR → activity zone (demo-side mapping; the in-formula version is task #3) ──
# Maps a live heart rate to the activity level compute_phrs already understands.
# Framing: HR proxies ventilation/exertion, which scales inhaled dose. This does
# NOT edit the teammate-owned formula — it just picks the existing activity input.
def hr_to_activity(hr: float | None) -> str:
    if hr is None:
        return "Sedentary"
    if hr < 90:
        return "Sedentary"
    if hr < 110:
        return "Moderate"
    if hr < 140:
        return "Active"
    return "Athlete"


def _trend_forecast(row, current_aqi):
    """Fallback near-term AQI estimate — damped-momentum extrapolation of the
    live AQI trend the row carries (AQI_lag6h). Only used if the trained
    forecaster models can't be loaded/run for some reason."""
    def g(k, d):
        try:
            return float(row.get(k, d))
        except Exception:
            return d

    cur = float(current_aqi)
    lag6 = g("AQI_lag6h", cur)
    slope6 = (cur - lag6) / 6.0          # AQI/hour over the last 6h
    fc6 = cur + slope6 * 6.0             # continue the 6h trend
    fc24 = cur + slope6 * 24.0 * 0.4     # damped extrapolation (trends decay)
    clip = lambda v: int(round(max(0.0, min(500.0, v))))
    return {"h6": clip(fc6), "h24": clip(fc24), "source": "trend"}


def _forecast_aqi(row, current_aqi):
    """Near-term AQI estimate for +6h/+24h, from the trained XGBoost AQI
    forecaster models (models/aqi_forecaster_h6.joblib, _h24.joblib) fed with
    the live feature row from realtime.py. Falls back to a trend
    extrapolation only if the models/row aren't usable."""
    if row is None:
        return None

    row_dict = row.to_dict() if hasattr(row, "to_dict") else row
    cur = float(current_aqi)
    try:
        fc6 = predict_aqi_smooth(row_dict, cur, 6)
        fc24 = predict_aqi_smooth(row_dict, cur, 24)
        clip = lambda v: int(round(max(0.0, min(500.0, v))))
        return {"h6": clip(fc6), "h24": clip(fc24), "source": "ml"}
    except Exception as exc:
        print(f"[api] AQI forecaster unavailable ({type(exc).__name__}: {exc}), falling back to trend", flush=True)
        return _trend_forecast(row, current_aqi)


# ── Hardware control ──────────────────────────────────────────────────────────

@app.post("/api/hardware/connect")
def connect_hardware(watch: bool = False, sensor: bool = False,
                     sensor_port: str | None = None, watch_addr: str | None = None):
    """Start the BLE watch and/or serial dust-sensor background readers."""
    started = {}
    if watch and not watch_hr.is_running():
        watch_hr.start(address=watch_addr)
        started["watch"] = True
    if sensor and sensor_port and not gp2y10.is_running():
        print(f"[api] starting GP2Y10 reader on {sensor_port}", flush=True)
        gp2y10.start(sensor_port)
        started["sensor"] = sensor_port
    return {"started": started,
            "watch_running": watch_hr.is_running(),
            "sensor_running": gp2y10.is_running()}


@app.get("/api/ports")
def serial_ports():
    """Available serial ports (to pick the dust sensor's COM port)."""
    return {"ports": gp2y10.list_ports()}


@app.get("/api/watch/scan")
async def watch_scan():
    """Scan nearby BLE devices so the user can pick the watch (its address
    rotates, so auto-connect is unreliable — an explicit pick is robust)."""
    from bleak import BleakScanner
    devs = await BleakScanner.discover(timeout=6.0, return_adv=True)
    out = []
    for d, adv in devs.values():
        name = d.name or adv.local_name or ""
        out.append({"name": name or "(unknown)", "address": d.address, "rssi": adv.rssi})
    out.sort(key=lambda x: (x["name"] == "(unknown)", -x["rssi"]))  # named + strongest first
    return {"devices": out}


@app.get("/api/cities")
def cities():
    return {"cities": sorted(realtime.CITY_COORDS.keys())}


@app.get("/api/metrics")
def metrics():
    """Serve the trained-model metrics JSON files for the metrics page."""
    import json as _json
    models_dir = Path(__file__).resolve().parent / "models"
    out = {}
    for name in ["metrics", "nhanes_metrics", "phrs_calibration", "sanity_caps"]:
        p = models_dir / f"{name}.json"
        if p.exists():
            try:
                with open(p) as f:
                    out[name] = _json.load(f)
            except Exception as exc:
                out[name] = {"error": str(exc)}
    return out


# ── The one endpoint the dashboard polls ──────────────────────────────────────

@app.get("/api/live")
def live(
    city: str = "Delhi",
    mode: str = "auto",              # "auto" = city air | "sensor" = local dust sensor
    age: int = 30,
    condition: str = "Healthy",
    hours_outdoors: float = 2.0,
    sim: bool = False,               # simulate mode (drive HR/PM by hand)
    sim_hr: float | None = None,
    sim_pm: float | None = None,
):
    # ── Heart rate: hardware if live, else simulated ─────────────────────────
    hr_snap = watch_hr.get_hr()
    if sim and sim_hr is not None:
        hr, hr_source, hr_stale = float(sim_hr), "sim", False
    elif hr_snap.get("bpm") is not None and not hr_snap.get("stale"):
        hr, hr_source, hr_stale = float(hr_snap["bpm"]), "watch", False
    else:
        hr, hr_source, hr_stale = None, "none", True

    activity = hr_to_activity(hr)

    # Debug log — prints only when the HR decision changes (not every second).
    global _last_hr_log
    _key = (hr, hr_source)
    if _key != _last_hr_log:
        print(f"[api] HR used={hr} source={hr_source} stale={hr_stale} activity={activity} "
              f"| watch snapshot: bpm={hr_snap.get('bpm')} stale={hr_snap.get('stale')} "
              f"err={hr_snap.get('error')} | sim={sim}", flush=True)
        _last_hr_log = _key

    # ── Air: city API (auto) or local dust sensor (sensor) ───────────────────
    pollutants: dict[str, float] = {}
    temp_c = humidity = None
    row_for_fc = None
    if mode == "sensor":
        sensor_snap = gp2y10.get_reading()
        global _last_sensor_log
        if not sim:
            _sk = (sensor_snap.get("pm25"), sensor_snap.get("error"))
            if _sk != _last_sensor_log:
                print(f"[api] sensor snapshot: {sensor_snap} running={gp2y10.is_running()}", flush=True)
                _last_sensor_log = _sk
        if sim and sim_pm is not None:
            pm25, air_source = float(sim_pm), "sim"
        elif sensor_snap.get("pm25") is not None:
            pm25, air_source = float(sensor_snap["pm25"]), "sensor"
        else:
            pm25, air_source = 0.0, "sensor_no_data"
        aqi = gp2y10.pm25_to_aqi(pm25)
        # The GP2Y10 measures particulates only — it cannot isolate PM10 or any
        # gas. Report just the measured PM2.5 (approximate); no fabricated values.
        # (Auto mode below carries the full real pollutant set from the API.)
        pollutants = {"PM2.5": pm25}
    else:  # auto
        data = _city_aqi(city)
        aqi = float(data.get("current_aqi", 0.0))
        pollutants = data.get("pollutants", {})
        pm25 = float(pollutants.get("PM2.5", 0.0))
        temp_c = data.get("temp_c")
        humidity = data.get("humidity")
        air_source = "api_error" if data.get("error") else "city_api"
        row_for_fc = data.get("row")

    # ── Near-term AQI forecast (+6h / +24h) — real trained forecaster ────────
    forecast = _forecast_aqi(row_for_fc, aqi) if mode != "sensor" else None

    # ── Fuse into PHRS: NHANES-derived condition weight (condition_weight_for,
    # used inside compute_phrs_horizons) applied to now AND to the forecasted
    # +6h/+24h AQI, so "future PHRS" reflects the ML forecaster's predicted
    # AQI, not just the current reading ──────────────────────────────────────
    profile = HealthProfile(age=age, condition=condition,
                            activity_level=activity, hours_outdoors=hours_outdoors)
    phrs_error = None
    phrs_forecast = None
    try:
        if forecast is not None:
            aqi_by_h = {"now": aqi, "+6h": float(forecast["h6"]), "+24h": float(forecast["h24"])}
            # Future pollutant composition isn't forecast — reuse the current
            # mix (only the AQI level itself is projected forward).
            poll_by_h = {"now": pollutants, "+6h": pollutants, "+24h": pollutants}
            result = compute_phrs_horizons(aqi_by_h, poll_by_h, profile,
                                           conditions=[condition],
                                           temp_c=temp_c, humidity=humidity)
            now_h = result["horizons"]["now"]
            phrs, category, color = now_h["phrs"], now_h["band"], now_h["color"]
            phrs_forecast = {"h6": result["horizons"]["+6h"], "h24": result["horizons"]["+24h"]}
        else:
            phrs = compute_phrs(aqi=aqi, pollutants=pollutants, profile=profile,
                                temp_c=temp_c, humidity=humidity, conditions=[condition])
            category, color = phrs_category(phrs)
    except Exception as exc:
        phrs, category, color = None, "Model offline", "#7f918a"
        phrs_error = f"{type(exc).__name__}"

    return JSONResponse({
        "forecast": forecast,
        "phrs_error": phrs_error,
        "ts": time.time(),
        "mode": mode,
        "city": city,
        "aqi": round(aqi, 1),
        "pm25": round(pm25, 1),
        "pollutants": {k: round(float(v), 1) for k, v in pollutants.items()},
        "air_source": air_source,
        "temp_c": temp_c,
        "humidity": humidity,
        "hr": hr,
        "hr_source": hr_source,
        "hr_stale": hr_stale,
        "activity": activity,
        "phrs": phrs,
        "category": category,
        "color": color,
        "phrs_forecast": phrs_forecast,
        "profile": {"age": age, "condition": condition},
        "hardware": {"watch": watch_hr.is_running(), "sensor": gp2y10.is_running()},
    }, headers={"Cache-Control": "no-store, max-age=0"})


# ── Serve the dashboard (mounted last so /api/* wins) ─────────────────────────
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
