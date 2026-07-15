"""
realtime.py — Real-time AQI & weather data via Open-Meteo APIs (no API key required)

NOT currently wired into app.py or api/main.py (both serve from the trained
INDIA_AQI_COMPLETE dataset instead) — this module exists for a future live-fetch
feature. Updated here to match the India-only, hourly AQI Forecaster feature
schema (see preprocess.AQI_FEATURE_COLS) rather than the old daily/3-source one.

Open-Meteo cannot supply every feature the hourly model was trained on (Dust_ugm3,
PM_Ratio, AOD, Dew_Point_C, Wind_Gusts_kmh, Pressure_MSL_hPa, Surface_Pressure_hPa,
Solar_Radiation_Wm2, Cloud_Cover_Percent, Sunshine_Seconds, Rain_mm, Festival_Period,
Crop_Burning_Season are not in its free hourly air-quality/weather responses) — those
default to 0 / False and are flagged in the returned row via "_missing_features".

Entry point: fetch_realtime_data(city) -> dict
  {
      "row":         pd.Series with model features (or None on error),
      "current_aqi": float,
      "pollutants":  dict[str, float],  # PM2.5/PM10/NO2/SO2/CO/O3 keys
      "temp_c":      float,
      "humidity":    float,
      "error":       str | None,
  }
"""

import datetime
import math

import numpy as np
import pandas as pd
import requests

from preprocess import CITY_ENC_MAP as _CITY_ENC

# ─── City coordinates (lat, lon) ──────────────────────────────────────────────

CITY_COORDS: dict[str, tuple[float, float]] = {
    "Ahmedabad":     (23.0225,  72.5714),
    "Bengaluru":     (12.9716,  77.5946),
    "Bhopal":        (23.2599,  77.4126),
    "Chandigarh":    (30.7333,  76.7794),
    "Chennai":       (13.0827,  80.2707),
    "Delhi":         (28.6139,  77.2090),
    "Gurugram":      (28.4595,  77.0266),
    "Guwahati":      (26.1445,  91.7362),
    "Hyderabad":     (17.3850,  78.4867),
    "Jaipur":        (26.9124,  75.7873),
    "Kolkata":       (22.5726,  88.3639),
    "Lucknow":       (26.8467,  80.9462),
    "Mumbai":        (19.0760,  72.8777),
    "Patna":         (25.5941,  85.1376),
    "Visakhapatnam": (17.6868,  83.2185),
}

# _CITY_ENC is imported from preprocess.CITY_ENC_MAP — single source of truth.

_NORTH_INDIA: set[str] = {
    "Delhi", "Lucknow", "Patna", "Chandigarh",
    "Gurugram", "Jaipur", "Ahmedabad", "Bhopal",
}

_AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
_WX_URL = "https://api.open-meteo.com/v1/forecast"
_TIMEOUT = 12  # seconds

# Features the hourly model expects that Open-Meteo's free endpoints don't
# provide — defaulted to 0 and reported so callers know accuracy is reduced.
_UNAVAILABLE_FEATURES = [
    "Dust_ugm3", "PM_Ratio", "AOD",
    "Dew_Point_C", "Wind_Gusts_kmh", "Pressure_MSL_hPa", "Surface_Pressure_hPa",
    "Solar_Radiation_Wm2", "Cloud_Cover_Percent", "Sunshine_Seconds", "Rain_mm",
    "Festival_Period", "Crop_Burning_Season",
]


# ─── US AQI from PM2.5 (piecewise linear, EPA breakpoints) ───────────────────

def _pm25_to_us_aqi(pm25: float) -> float:
    breakpoints = [
        (0.0,   12.0,    0,  50),
        (12.1,  35.4,   51, 100),
        (35.5,  55.4,  101, 150),
        (55.5, 150.4,  151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 500.4, 301, 500),
    ]
    pm25 = max(0.0, min(float(pm25 or 0.0), 500.4))
    for c_lo, c_hi, aqi_lo, aqi_hi in breakpoints:
        if pm25 <= c_hi:
            return aqi_lo + (pm25 - c_lo) / (c_hi - c_lo) * (aqi_hi - aqi_lo)
    return 500.0


# ─── Open-Meteo API calls (hourly, not daily-aggregated) ─────────────────────

def _fetch_air_quality(lat: float, lon: float) -> dict:
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": "pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide,ozone",
        "past_days": 3, "forecast_days": 0, "timezone": "Asia/Kolkata",
    }
    r = requests.get(_AQ_URL, params=params, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def _fetch_weather(lat: float, lon: float) -> dict:
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,precipitation,is_day",
        "past_days": 3, "forecast_days": 0, "timezone": "Asia/Kolkata",
        "wind_speed_unit": "kmh",
    }
    r = requests.get(_WX_URL, params=params, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def _hourly_series(json_data: dict, var: str) -> pd.Series:
    times = pd.to_datetime(json_data["hourly"]["time"])
    vals = json_data["hourly"].get(var, [None] * len(times))
    return pd.Series(vals, index=times)


def _at_hour(series: pd.Series, target: pd.Timestamp, default: float) -> float:
    if target not in series.index:
        return default
    val = series.loc[target]
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return default
    return float(val)


# ─── Feature row builder ──────────────────────────────────────────────────────

def _build_row(city: str, aq_json: dict, wx_json: dict) -> pd.Series:
    """Construct a pd.Series with the model's hourly feature set for the
    latest available hour. Fields Open-Meteo can't supply default to 0/False
    (see _UNAVAILABLE_FEATURES)."""
    pm25_s = _hourly_series(aq_json, "pm2_5")
    now = pm25_s.dropna().index.max()
    if now is None or pd.isna(now):
        now = pd.Timestamp(datetime.datetime.now()).floor("h")

    def aq(var, default=0.0):
        return _at_hour(_hourly_series(aq_json, var), now, default)

    def wx(var, default=0.0):
        return _at_hour(_hourly_series(wx_json, var), now, default)

    pm25 = aq("pm2_5")
    pm10 = aq("pm10")
    no2 = aq("nitrogen_dioxide")
    so2 = aq("sulphur_dioxide")
    co = aq("carbon_monoxide")
    o3 = aq("ozone")

    temp = wx("temperature_2m", 25.0)
    humidity = wx("relative_humidity_2m", 50.0)
    wind_speed = wx("wind_speed_10m", 10.0)
    wind_dir = wx("wind_direction_10m", 0.0)
    precip = wx("precipitation", 0.0)
    is_day = wx("is_day", 1.0)

    aqi = _pm25_to_us_aqi(pm25)

    month = now.month
    hour = now.hour
    wind_stagnation = float(wind_speed < 5.0)
    temp_inversion = int(month in (11, 12, 1, 2) and city in _NORTH_INDIA and wind_speed < 5.0)

    def _aqi_h_ago(hours_ago: int) -> float:
        target = now - pd.Timedelta(hours=hours_ago)
        pm25_lag = _at_hour(pm25_s, target, pm25)
        return _pm25_to_us_aqi(pm25_lag)

    lag_hours = [1, 3, 6, 24, 48]
    lags = {h: _aqi_h_ago(h) for h in lag_hours}

    roll24 = [_aqi_h_ago(h) for h in range(24)]
    roll168 = [_aqi_h_ago(h) for h in range(0, 168, 6)]  # sparse sample — 3 past days of hourly data only

    season = (
        "Monsoon" if month in (6, 7, 8, 9) else
        "Post_Monsoon" if month in (10, 11) else
        "Winter" if month in (12, 1, 2) else "Summer"
    )

    row = {
        "PM2_5_ugm3": pm25, "PM10_ugm3": pm10, "NO2_ugm3": no2,
        "SO2_ugm3": so2, "CO_ugm3": co, "O3_ugm3": o3,
        "Dust_ugm3": 0.0, "PM_Ratio": (pm25 / pm10) if pm10 else 0.0, "AOD": 0.0,
        "Temp_2m_C": temp, "Humidity_Percent": humidity, "Dew_Point_C": 0.0,
        "Wind_Speed_10m_kmh": wind_speed, "Wind_Dir_10m": wind_dir,
        "Wind_Gusts_kmh": 0.0, "Wind_Stagnation": wind_stagnation,
        "Precipitation_mm": precip, "Rain_mm": 0.0,
        "Pressure_MSL_hPa": 0.0, "Surface_Pressure_hPa": 0.0,
        "Solar_Radiation_Wm2": 0.0, "Cloud_Cover_Percent": 0.0, "Sunshine_Seconds": 0.0,
        "Is_Daytime": is_day, "Is_Raining": float(precip > 0), "Heavy_Rain": float(precip > 7.6),
        "Temp_Inversion": temp_inversion, "Festival_Period": 0, "Crop_Burning_Season": 0,
        "Month": float(month), "Day_of_Week": float(now.dayofweek),
        "Is_Weekend": float(now.dayofweek >= 5), "Quarter": float((month - 1) // 3 + 1),
        "city_enc": float(_CITY_ENC.get(city, 0)),
        "season_Winter": float(season == "Winter"),
        "season_Monsoon": float(season == "Monsoon"),
        "season_Post_Monsoon": float(season == "Post_Monsoon"),
        "season_Summer": float(season == "Summer"),
        "Hour_sin": float(np.sin(2 * np.pi * hour / 24)),
        "Hour_cos": float(np.cos(2 * np.pi * hour / 24)),
        "AQI_lag1h": lags[1], "AQI_lag3h": lags[3], "AQI_lag6h": lags[6],
        "AQI_lag24h": lags[24], "AQI_lag48h": lags[48],
        "AQI_roll24h_mean": float(np.mean(roll24)), "AQI_roll24h_std": float(np.std(roll24)),
        "AQI_roll168h_mean": float(np.mean(roll168)), "AQI_roll168h_std": float(np.std(roll168)),
        # ── Display-only (not passed to models) ───────────────────────────────
        "US_AQI": aqi,
        "City": city,
        "_missing_features": _UNAVAILABLE_FEATURES,
    }
    return pd.Series(row)


# ─── Main entry point ─────────────────────────────────────────────────────────

def fetch_realtime_data(city: str) -> dict:
    """Fetch real-time AQI + weather for *city* via Open-Meteo APIs (hourly).
    Returns a dict — always check the "error" key first (None = success)."""
    _empty = {
        "row": None, "current_aqi": 0.0,
        "pollutants": {}, "temp_c": 25.0, "humidity": 50.0,
    }

    coords = CITY_COORDS.get(city)
    if coords is None:
        return {**_empty, "error": f"No coordinates defined for '{city}'."}

    lat, lon = coords

    try:
        aq_json = _fetch_air_quality(lat, lon)
        wx_json = _fetch_weather(lat, lon)
    except requests.exceptions.RequestException as exc:
        return {**_empty, "error": f"API request failed: {exc}"}
    except Exception as exc:
        return {**_empty, "error": f"Unexpected fetch error: {exc}"}

    try:
        row = _build_row(city, aq_json, wx_json)
        return {
            "row": row,
            "current_aqi": float(row["US_AQI"]),
            "pollutants": {
                "PM2.5": float(row.get("PM2_5_ugm3", 0.0)),
                "PM10": float(row.get("PM10_ugm3", 0.0)),
                "NO2": float(row.get("NO2_ugm3", 0.0)),
                "SO2": float(row.get("SO2_ugm3", 0.0)),
                "CO": float(row.get("CO_ugm3", 0.0)),
                "O3": float(row.get("O3_ugm3", 0.0)),
            },
            "temp_c": float(row.get("Temp_2m_C", 25.0)),
            "humidity": float(row.get("Humidity_Percent", 50.0)),
            "error": None,
        }
    except Exception as exc:
        return {**_empty, "error": f"Data processing error: {exc}"}
