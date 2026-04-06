"""
realtime.py — Real-time AQI & weather data via Open-Meteo APIs (no API key required)

Entry point: fetch_realtime_data(city) → dict
  {
      "row":         pd.Series with all 34 model features (or None on error),
      "current_aqi": float,
      "pollutants":  dict[str, float],
      "temp_c":      float,
      "humidity":    float,
      "error":       str | None,
  }
"""

import datetime
import math

import pandas as pd
import requests

from preprocess import normalize_aqi_india, CITY_ENC_MAP as _CITY_ENC

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

# North India cities where crop burning occurs in Oct–Nov
_NORTH_INDIA: set[str] = {
    "Delhi", "Lucknow", "Patna", "Chandigarh",
    "Gurugram", "Jaipur", "Ahmedabad", "Bhopal",
}

# Health reference values per AQI ceiling (bin-averaged from air_quality_health_impact_data.csv)
_HEALTH_REF: dict[int, tuple[float, float]] = {
    50:  (0.15,  12.0),
    100: (0.28,  35.0),
    150: (0.42,  68.0),
    200: (0.56, 112.0),
    300: (0.72, 178.0),
    500: (0.90, 280.0),
}

_AQ_URL  = "https://air-quality-api.open-meteo.com/v1/air-quality"
_WX_URL  = "https://api.open-meteo.com/v1/forecast"
_TIMEOUT = 12  # seconds


# ─── India CPCB AQI from PM2.5 (piecewise linear) ────────────────────────────

def _pm25_to_aqi(pm25: float) -> float:
    """Convert PM2.5 (µg/m³) to India CPCB AQI sub-index via piecewise interpolation."""
    breakpoints = [
        (0.0,   30.0,   0,   50),
        (30.0,  60.0,  51,  100),
        (60.0,  90.0, 101,  200),
        (90.0, 120.0, 201,  300),
        (120.0,250.0, 301,  400),
        (250.0,500.0, 401,  500),
    ]
    pm25 = max(0.0, min(float(pm25 or 0.0), 500.0))
    for c_lo, c_hi, aqi_lo, aqi_hi in breakpoints:
        if pm25 <= c_hi:
            return aqi_lo + (pm25 - c_lo) / (c_hi - c_lo) * (aqi_hi - aqi_lo)
    return 500.0


def _lookup_health_ref(aqi: float) -> tuple[float, float]:
    for ceiling, vals in sorted(_HEALTH_REF.items()):
        if aqi <= ceiling:
            return vals
    return (0.90, 280.0)


# ─── Open-Meteo API calls ─────────────────────────────────────────────────────

def _fetch_air_quality(lat: float, lon: float) -> dict:
    """Fetch hourly AQ data from Open-Meteo (past 7 days + today)."""
    params = {
        "latitude":      lat,
        "longitude":     lon,
        "hourly":        "pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide,ozone",
        "past_days":     7,
        "forecast_days": 0,
        "timezone":      "Asia/Kolkata",
    }
    r = requests.get(_AQ_URL, params=params, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


def _fetch_weather(lat: float, lon: float) -> dict:
    """Fetch hourly weather data from Open-Meteo (past 7 days + today)."""
    params = {
        "latitude":        lat,
        "longitude":       lon,
        "hourly":          "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
        "past_days":       7,
        "forecast_days":   0,
        "timezone":        "Asia/Kolkata",
        "wind_speed_unit": "kmh",
    }
    r = requests.get(_WX_URL, params=params, timeout=_TIMEOUT)
    r.raise_for_status()
    return r.json()


# ─── Hourly JSON → daily DataFrame ───────────────────────────────────────────

def _hourly_to_daily(
    json_data: dict,
    variables: list[str],
    agg_rules: dict[str, str],
) -> pd.DataFrame:
    """Convert Open-Meteo hourly JSON to a daily-aggregated DataFrame."""
    times = pd.to_datetime(json_data["hourly"]["time"])
    df = pd.DataFrame({"datetime": times})
    df["Date"] = df["datetime"].dt.normalize()

    for var in variables:
        vals = json_data["hourly"].get(var)
        if vals is not None:
            df[var] = vals

    agg = {var: agg_rules.get(var, "mean") for var in variables if var in df.columns}
    daily = df.groupby("Date").agg(agg).reset_index()
    return daily


# ─── Wind stagnation ──────────────────────────────────────────────────────────

def _compute_wind_stagnation(wx_json: dict, today_str: str) -> float:
    """Fraction of today's hourly readings where wind_speed_10m < 5 km/h."""
    times  = wx_json["hourly"]["time"]
    speeds = wx_json["hourly"].get("wind_speed_10m", [])
    if not speeds:
        return 0.0
    today_speeds = [
        s for t, s in zip(times, speeds)
        if t.startswith(today_str) and s is not None
    ]
    if not today_speeds:
        return 0.0
    return sum(1 for s in today_speeds if s < 5.0) / len(today_speeds)


# ─── Season helper ────────────────────────────────────────────────────────────

def _month_to_season(month: int) -> str:
    if month in (6, 7, 8, 9):   return "Monsoon"
    if month in (10, 11):        return "Post_Monsoon"
    if month in (12, 1, 2):      return "Winter"
    return "Summer"


# ─── Feature row builder ──────────────────────────────────────────────────────

def _get_val(df: pd.DataFrame, col: str, days_back: int,
             today: pd.Timestamp, default: float) -> float:
    """Read a value from `days_back` days before today in df."""
    target = today - pd.Timedelta(days=days_back)
    sub = df[df["Date"].dt.normalize() == target]
    if sub.empty or col not in sub.columns:
        return default
    val = sub.iloc[0][col]
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return default
    return float(val)


def _build_row(
    city: str,
    aq_df: pd.DataFrame,
    wx_df: pd.DataFrame,
    wind_stagnation: float,
) -> pd.Series:
    """Construct a pd.Series with all 34 model features for today."""
    today = pd.Timestamp(datetime.date.today())

    # ── Pollutants (today's daily mean) ───────────────────────────────────────
    pm25 = _get_val(aq_df, "pm2_5",            0, today, 0.0)
    pm10 = _get_val(aq_df, "pm10",             0, today, 0.0)
    no2  = _get_val(aq_df, "nitrogen_dioxide", 0, today, 0.0)
    so2  = _get_val(aq_df, "sulphur_dioxide",  0, today, 0.0)
    co   = _get_val(aq_df, "carbon_monoxide",  0, today, 0.0)
    o3   = _get_val(aq_df, "ozone",            0, today, 0.0)

    # ── Weather (today) ───────────────────────────────────────────────────────
    temp     = _get_val(wx_df, "temperature_2m",       0, today, 25.0)
    humidity = _get_val(wx_df, "relative_humidity_2m", 0, today, 50.0)
    wind     = _get_val(wx_df, "wind_speed_10m",       0, today, 10.0)
    precip   = _get_val(wx_df, "precipitation",        0, today,  0.0)

    # ── AQI from PM2.5 sub-index ──────────────────────────────────────────────
    aqi = _pm25_to_aqi(pm25)

    # ── Date/season features ──────────────────────────────────────────────────
    now    = datetime.datetime.now()
    month  = now.month
    dow    = now.weekday()
    season = _month_to_season(month)

    # ── Binary event flags ────────────────────────────────────────────────────
    crop_burning   = int(month in (10, 11) and city in _NORTH_INDIA)
    temp_inversion = int(season == "Winter" and wind < 5.0)

    # ── AQI lag features (derived from past PM2.5 values) ─────────────────────
    def _aqi_d(d: int) -> float:
        return _pm25_to_aqi(_get_val(aq_df, "pm2_5", d, today, pm25))

    aqi_lag1 = _aqi_d(1)
    aqi_lag3 = _aqi_d(3)
    aqi_lag7 = _aqi_d(7)

    aqi_series     = [_aqi_d(d) for d in range(7)]
    s              = pd.Series(aqi_series)
    aqi_roll7_mean = float(s.mean())
    aqi_roll7_std  = float(s.std(ddof=0)) if len(s) > 1 else 0.0

    aqi_delta1 = aqi - aqi_lag1
    aqi_delta3 = aqi - aqi_lag3

    # ── Pollutant lag features (1 day) ────────────────────────────────────────
    pm25_lag1 = _get_val(aq_df, "pm2_5",            1, today, pm25)
    pm10_lag1 = _get_val(aq_df, "pm10",             1, today, pm10)
    no2_lag1  = _get_val(aq_df, "nitrogen_dioxide", 1, today, no2)

    # ── Health reference ──────────────────────────────────────────────────────
    ref_health_score, ref_resp_cases = _lookup_health_ref(aqi)

    return pd.Series({
        # ── 34 model features ─────────────────────────────────────────────────
        "PM2.5":               pm25,
        "PM10":                pm10,
        "NO2":                 no2,
        "SO2":                 so2,
        "CO":                  co,
        "O3":                  o3,
        "Temp_2m_C":           temp,
        "Humidity_Percent":    humidity,
        "Wind_Speed_kmh":      wind,
        "Precipitation_mm":    precip,
        "Wind_Stagnation":     wind_stagnation,
        "Temp_Inversion":      float(temp_inversion),
        "Festival_Period":     0.0,
        "Crop_Burning_Season": float(crop_burning),
        "month":               float(month),
        "dayofweek":           float(dow),
        "city_enc":            float(_CITY_ENC.get(city, 0)),
        "season_Winter":       float(season == "Winter"),
        "season_Monsoon":      float(season == "Monsoon"),
        "season_Post_Monsoon": float(season == "Post_Monsoon"),
        "season_Summer":       float(season == "Summer"),
        "AQI_lag1":            aqi_lag1,
        "AQI_lag3":            aqi_lag3,
        "AQI_lag7":            aqi_lag7,
        "AQI_roll7_mean":      aqi_roll7_mean,
        "AQI_roll7_std":       aqi_roll7_std,
        "AQI_delta1":          aqi_delta1,
        "AQI_delta3":          aqi_delta3,
        "AQI_norm_india":      normalize_aqi_india(aqi),
        "PM2.5_lag1":          pm25_lag1,
        "PM10_lag1":           pm10_lag1,
        "NO2_lag1":            no2_lag1,
        "ref_health_score":    ref_health_score,
        "ref_resp_cases":      ref_resp_cases,
        # ── Display-only (not passed to models) ───────────────────────────────
        "AQI":                 aqi,
        "City":                city,
    })


# ─── Main entry point ─────────────────────────────────────────────────────────

def fetch_realtime_data(city: str) -> dict:
    """
    Fetch real-time AQI + weather for *city* via Open-Meteo APIs.

    Returns a dict — always check the "error" key first (None = success).
    """
    _empty = {
        "row": None, "current_aqi": 0.0,
        "pollutants": {}, "temp_c": 25.0, "humidity": 50.0,
    }

    coords = CITY_COORDS.get(city)
    if coords is None:
        return {**_empty, "error": f"No coordinates defined for '{city}'."}

    lat, lon  = coords
    today_str = datetime.date.today().strftime("%Y-%m-%d")

    try:
        aq_json = _fetch_air_quality(lat, lon)
        wx_json = _fetch_weather(lat, lon)
    except requests.exceptions.RequestException as exc:
        return {**_empty, "error": f"API request failed: {exc}"}
    except Exception as exc:
        return {**_empty, "error": f"Unexpected fetch error: {exc}"}

    try:
        aq_vars = ["pm2_5", "pm10", "nitrogen_dioxide",
                   "sulphur_dioxide", "carbon_monoxide", "ozone"]
        aq_df = _hourly_to_daily(
            aq_json, aq_vars,
            agg_rules={v: "mean" for v in aq_vars},
        )

        wx_vars = ["temperature_2m", "relative_humidity_2m",
                   "wind_speed_10m", "precipitation"]
        wx_df = _hourly_to_daily(
            wx_json, wx_vars,
            agg_rules={
                "temperature_2m":       "mean",
                "relative_humidity_2m": "mean",
                "wind_speed_10m":       "mean",
                "precipitation":        "sum",
            },
        )

        wind_stagnation = _compute_wind_stagnation(wx_json, today_str)
        row = _build_row(city, aq_df, wx_df, wind_stagnation)

        return {
            "row":         row,
            "current_aqi": float(row["AQI"]),
            "pollutants": {
                "PM2.5": float(row.get("PM2.5", 0.0)),
                "PM10":  float(row.get("PM10",  0.0)),
                "NO2":   float(row.get("NO2",   0.0)),
                "SO2":   float(row.get("SO2",   0.0)),
                "CO":    float(row.get("CO",    0.0)),
                "O3":    float(row.get("O3",    0.0)),
            },
            "temp_c":   float(row.get("Temp_2m_C", 25.0)),
            "humidity": float(row.get("Humidity_Percent", 50.0)),
            "error":    None,
        }

    except Exception as exc:
        return {**_empty, "error": f"Data processing error: {exc}"}
