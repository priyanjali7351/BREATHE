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
from generate_profiles import compute_phrs, HealthProfile, phrs_category

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

    # ── Fuse into PHRS ───────────────────────────────────────────────────────
    profile = HealthProfile(age=age, condition=condition,
                            activity_level=activity, hours_outdoors=hours_outdoors)
    phrs = compute_phrs(aqi=aqi, pollutants=pollutants, profile=profile,
                        temp_c=temp_c, humidity=humidity, conditions=[condition])
    category, color = phrs_category(phrs)

    return JSONResponse({
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
        "profile": {"age": age, "condition": condition},
        "hardware": {"watch": watch_hr.is_running(), "sensor": gp2y10.is_running()},
    }, headers={"Cache-Control": "no-store, max-age=0"})


# ── Serve the dashboard (mounted last so /api/* wins) ─────────────────────────
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
