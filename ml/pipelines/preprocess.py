"""
BioAQI — Data Preprocessing Module (v3, India-only, hourly)

Single source of truth: data/INDIA_AQI_COMPLETE_20251126.csv
  - Hourly, 29 cities, 2022-08-05 -> 2025-11-26, pollutants + weather + AQI sub-indices.

city_day.csv and air_quality_health_impact_data.csv (and the ref_health_score /
ref_resp_cases features derived from the latter) have been removed from this
pipeline — both were confirmed dead weight that caused a daily/hourly and
CPCB/US-AQI scale mismatch.

Target: US_AQI (the complete AQI signal in this file — there is no CPCB column).
US_AQI is clipped to [0, 500] before training; values above that are known
computation artefacts in the source data (max observed: 2742), not real events.

Leakage: US_AQI = max(US_AQI_PM25, US_AQI_PM10, US_AQI_NO2, US_AQI_O3, US_AQI_CO).
Those sub-indices (and the EU_AQI family, and the two category columns) are the
answer sheet and are dropped before feature engineering. Past values of the
target itself (US_AQI lags) are allowed and are the strongest predictive signal.
"""

import numpy as np
import pandas as pd


# ─── Constants ─────────────────────────────────────────────────────────────────

MAJOR_CITIES = [
    "Delhi", "Mumbai", "Bengaluru", "Hyderabad", "Chennai",
    "Kolkata", "Ahmedabad", "Jaipur", "Lucknow", "Patna",
    "Chandigarh", "Gurugram", "Guwahati", "Visakhapatnam", "Bhopal",
]

# Canonical city encoding — alphabetical, fixed regardless of which cities
# survive a given filter. Import this in realtime.py / inference code to
# guarantee consistent city_enc values at train and predict time.
CITY_ENC_MAP: dict[str, int] = {
    city: i for i, city in enumerate(sorted(MAJOR_CITIES))
}

# Columns that are 100% null in INDIA_AQI_COMPLETE — dropped entirely.
ALL_NULL_COLS = [
    "Temp_80m_C", "Temp_120m_C", "Temp_180m_C",
    "Wind_Speed_80m_kmh", "Wind_Speed_120m_kmh",
    "UV_Index", "NH3_ugm3", "Inversion_Strength_C",
]

# US_AQI = max() of these sub-indices -> answer sheet, never used as features.
# EU_AQI (and its sub-indices) are a different scale built from the same raw
# pollutants -> also leakage. Category columns are AQI restated as text.
LEAKAGE_COLS = [
    "US_AQI_PM25", "US_AQI_PM10", "US_AQI_NO2", "US_AQI_O3", "US_AQI_CO",
    "EU_AQI", "EU_AQI_PM25", "EU_AQI_PM10",
    "AQI_Category", "PM25_Category_India",
]

TARGET_COL = "US_AQI"

POLLUTANT_COLS = [
    "PM2_5_ugm3", "PM10_ugm3", "NO2_ugm3", "SO2_ugm3", "CO_ugm3", "O3_ugm3",
    "Dust_ugm3", "PM_Ratio", "AOD",
]

WEATHER_COLS = [
    "Temp_2m_C", "Humidity_Percent", "Dew_Point_C", "Wind_Speed_10m_kmh",
    "Wind_Dir_10m", "Wind_Gusts_kmh", "Wind_Stagnation", "Precipitation_mm",
    "Rain_mm", "Pressure_MSL_hPa", "Surface_Pressure_hPa",
    "Solar_Radiation_Wm2", "Cloud_Cover_Percent", "Sunshine_Seconds",
    "Is_Daytime", "Is_Raining", "Heavy_Rain",
]

EVENT_COLS = ["Temp_Inversion", "Festival_Period", "Crop_Burning_Season"]

CALENDAR_BASE_COLS = ["Month", "Day_of_Week", "Is_Weekend", "Quarter", "city_enc"]
SEASON_COLS = ["season_Winter", "season_Monsoon", "season_Post_Monsoon", "season_Summer"]
HOUR_CYCLICAL_COLS = ["Hour_sin", "Hour_cos"]

# Target-history features (built from past US_AQI only — never leaks the future)
LAG_HOURS = [1, 3, 6, 24, 48]
ROLL_WINDOWS_H = [24, 168]
AQI_LAG_COLS = [f"AQI_lag{h}h" for h in LAG_HOURS]
AQI_ROLL_COLS = [f"AQI_roll{w}h_{stat}" for w in ROLL_WINDOWS_H for stat in ("mean", "std")]

# Final ordered feature set for the AQI forecasters
AQI_FEATURE_COLS = (
    POLLUTANT_COLS
    + WEATHER_COLS
    + EVENT_COLS
    + CALENDAR_BASE_COLS
    + SEASON_COLS
    + HOUR_CYCLICAL_COLS
    + AQI_LAG_COLS
    + AQI_ROLL_COLS
)

# Feature-group membership, used for grouped feature-importance reporting
FEATURE_GROUPS = {
    "pollutants": POLLUTANT_COLS,
    "weather": WEATHER_COLS,
    "events": EVENT_COLS,
    "calendar": CALENDAR_BASE_COLS + SEASON_COLS + HOUR_CYCLICAL_COLS,
    "aqi_history": AQI_LAG_COLS + AQI_ROLL_COLS,
}


# ─── AQI normalization (kept for the PHRS pipeline — generic CPCB utility,
#     not tied to any particular dataset) ────────────────────────────────────

_AQI_BREAKPOINTS = [
    (0,   0.00),
    (50,  0.15),
    (100, 0.30),
    (200, 0.55),
    (300, 0.75),
    (400, 0.90),
    (500, 1.00),
]
_BP_RAW = [b[0] for b in _AQI_BREAKPOINTS]
_BP_NORM = [b[1] for b in _AQI_BREAKPOINTS]


def normalize_aqi_india(aqi: float) -> float:
    """Piecewise-linear normalization using the India CPCB AQI scale (0-500 -> 0-1)."""
    return float(np.interp(np.clip(aqi, 0, 500), _BP_RAW, _BP_NORM))


def _month_to_season(m: int) -> str:
    if m in (6, 7, 8, 9):
        return "Monsoon"
    if m in (10, 11):
        return "Post_Monsoon"
    if m in (12, 1, 2):
        return "Winter"
    return "Summer"


# ─── Loading ───────────────────────────────────────────────────────────────────

def load_india_hourly(path: str) -> pd.DataFrame:
    """
    Load INDIA_AQI_COMPLETE_20251126.csv, filter to MAJOR_CITIES, drop the
    100%-null columns and every leakage column, and clip the target to [0, 500].
    """
    df = pd.read_csv(path, parse_dates=["Datetime"], low_memory=False)
    df = df[df["City"].isin(MAJOR_CITIES)].copy()

    drop_cols = [c for c in ALL_NULL_COLS + LEAKAGE_COLS if c in df.columns]
    df = df.drop(columns=drop_cols)

    df[TARGET_COL] = df[TARGET_COL].clip(0, 500)

    return df.sort_values(["City", "Datetime"]).reset_index(drop=True)


# ─── Hourly grid alignment (fixes shift()-counts-rows bug) ───────────────────

def reindex_hourly_grid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reindex every city onto a complete hourly Datetime grid so that shift(-h)
    always means exactly h hours, never h rows. Any hour missing from the
    source becomes a NaN row (dropped later once lag/target features are built
    and rows lacking a valid target are removed).
    """
    pieces = []
    for city, g in df.groupby("City", sort=False):
        g = g.set_index("Datetime").sort_index()
        full_idx = pd.date_range(g.index.min(), g.index.max(), freq="h")
        g = g.reindex(full_idx)
        g["City"] = city
        g.index.name = "Datetime"
        pieces.append(g)

    out = pd.concat(pieces).reset_index().rename(columns={"index": "Datetime"})
    return out.sort_values(["City", "Datetime"]).reset_index(drop=True)


# ─── Imputation ────────────────────────────────────────────────────────────────

def _impute_per_city(df: pd.DataFrame) -> pd.DataFrame:
    """Fill gap rows created by reindex_hourly_grid: per-city ffill/bfill, then
    global median fallback. Does not touch the target column (US_AQI) — rows
    with a missing target are dropped downstream instead of imputed."""
    df = df.copy()
    impute_cols = POLLUTANT_COLS + WEATHER_COLS + EVENT_COLS

    for col in impute_cols:
        if col not in df.columns:
            continue
        df[col] = df.groupby("City")[col].transform(lambda s: s.ffill().bfill())
        df[col] = df[col].fillna(df[col].median())

    return df


# ─── Feature engineering ──────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build calendar/cyclical/season/city features and the AQI-history
    lag + rolling features. Assumes df has already been through
    reindex_hourly_grid (so per-city rows are exactly 1 hour apart).
    """
    df = df.copy()

    # ── Calendar (recomputed from Datetime — gap rows had NaN source values) ─
    dt = df["Datetime"]
    df["Month"] = dt.dt.month
    df["Day_of_Week"] = dt.dt.dayofweek
    df["Is_Weekend"] = (df["Day_of_Week"] >= 5).astype(int)
    df["Quarter"] = dt.dt.quarter
    hour = dt.dt.hour
    df["Hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["Hour_cos"] = np.cos(2 * np.pi * hour / 24)

    df["Season"] = df["Month"].map(_month_to_season)
    s_dummies = pd.get_dummies(df["Season"], prefix="season")
    for col in SEASON_COLS:
        if col not in s_dummies.columns:
            s_dummies[col] = 0
    df = pd.concat([df, s_dummies[SEASON_COLS]], axis=1)

    df["city_enc"] = df["City"].map(CITY_ENC_MAP)

    df["Wind_Stagnation"] = df["Wind_Stagnation"].clip(0.0, 1.0).fillna(0.0)

    # ── Event flags: gap rows get 0 (no known event) ─────────────────────────
    for col in EVENT_COLS:
        df[col] = df[col].fillna(0).astype(int)

    # ── Target-history lags (past US_AQI only) ───────────────────────────────
    grp = df.groupby("City")[TARGET_COL]
    for h in LAG_HOURS:
        df[f"AQI_lag{h}h"] = grp.shift(h)

    for w in ROLL_WINDOWS_H:
        df[f"AQI_roll{w}h_mean"] = (
            df.groupby("City")[TARGET_COL]
            .transform(lambda s, w=w: s.shift(1).rolling(w, min_periods=max(3, w // 8)).mean())
        )
        df[f"AQI_roll{w}h_std"] = (
            df.groupby("City")[TARGET_COL]
            .transform(lambda s, w=w: s.shift(1).rolling(w, min_periods=max(3, w // 8)).std().fillna(0))
        )

    return df


# ─── Main pipeline ─────────────────────────────────────────────────────────────

def build_training_df(india_path: str = "data/INDIA_AQI_COMPLETE_20251126.csv") -> dict:
    """
    Full India-only hourly preprocessing pipeline.

    Returns
    -------
    dict with keys:
      "full_df"      -> engineered DataFrame (unscaled), sorted by Datetime.
                        Includes Datetime, City, US_AQI, and every feature in
                        AQI_FEATURE_COLS. Per-horizon targets are NOT built
                        here — see models.build_aqi_targets().
      "feature_cols" -> ordered list of feature column names
    """
    print("  Loading INDIA_AQI_COMPLETE (hourly, India-only) …")
    df = load_india_hourly(india_path)
    n_raw = len(df)

    print("  Reindexing each city onto a complete hourly grid …")
    df = reindex_hourly_grid(df)
    n_gridded = len(df)
    n_gap_rows = n_gridded - n_raw
    print(f"      {n_raw:,} source rows -> {n_gridded:,} grid rows ({n_gap_rows:,} gap rows filled as NaN)")

    print("  Imputing gap rows (per-city ffill/bfill + median fallback) …")
    df = _impute_per_city(df)

    print("  Engineering features (calendar, cyclical hour, season, AQI history) …")
    df = engineer_features(df)

    feat_cols = [c for c in AQI_FEATURE_COLS if c in df.columns]
    meta = ["City", "Datetime", TARGET_COL]
    df = df[[c for c in meta + feat_cols if c in df.columns]].copy()
    df = df.sort_values("Datetime").reset_index(drop=True)

    assert not any(c.startswith(("US_AQI_", "EU_AQI")) or c in ("AQI_Category", "PM25_Category_India")
                   for c in feat_cols), "Leakage column leaked into feature_cols!"

    print(
        f"  Done — {len(df):,} total rows | Features: {len(feat_cols)} | "
        f"Cities: {df['City'].nunique()}"
    )

    return {
        "full_df": df,
        "feature_cols": feat_cols,
    }


# ─── Standalone smoke-test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    pipeline = build_training_df()
    full_df = pipeline["full_df"]
    feat_cols = pipeline["feature_cols"]

    print(f"\nFeatures ({len(feat_cols)}):")
    for f in feat_cols:
        print(f"  {f}")

    print(f"\nUS_AQI stats:\n{full_df[TARGET_COL].describe()}")
    print(f"\nDatetime range: {full_df['Datetime'].min()} -> {full_df['Datetime'].max()}")

    print("\nNull counts in feature cols:")
    nulls = full_df[feat_cols].isnull().sum()
    print(nulls[nulls > 0] if nulls.any() else "  None")
