"""
BioAQI — Data Preprocessing Module (v2)
Combines three datasets:
  1. city_day.csv                      (daily AQI + pollutants, 2015-2020, 26 cities)
  2. INDIA_AQI_COMPLETE_20251126.csv   (hourly AQI + weather, 2022-2025, 29 cities)
  3. air_quality_health_impact_data.csv (health impact reference, no city/date)

Key improvements:
  - Per-city seasonal outlier treatment (festival/inversion rows get relaxed fences)
  - Weather features (temp, humidity, wind, precipitation) integrated with AQI
  - Season, festival, crop-burning, and inversion flags as features
  - Health impact reference features derived from the third dataset
  - Useless/redundant features removed to reduce overfitting
  - 80/20 temporal train/test split; scalers fit on train only (no data leakage)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# ─── Constants ─────────────────────────────────────────────────────────────────

MAJOR_CITIES = [
    "Delhi", "Mumbai", "Bengaluru", "Hyderabad", "Chennai",
    "Kolkata", "Ahmedabad", "Jaipur", "Lucknow", "Patna",
    "Chandigarh", "Gurugram", "Guwahati", "Visakhapatnam", "Bhopal",
]

# Core pollutants: NH3 dropped — 100% null in INDIA_AQI_COMPLETE, 35% missing in city_day
POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]

# North Indian cities prone to winter temperature inversions and crop burning
NORTH_INDIA_CITIES = {
    "Delhi", "Lucknow", "Patna", "Chandigarh", "Gurugram",
    "Jaipur", "Ahmedabad", "Bhopal",
}

# India CPCB AQI piecewise breakpoints for normalization
# Stretches the 100-400 range where Indian cities spend most of their time
_AQI_BREAKPOINTS = [
    (0,   0.00),
    (50,  0.15),   # Good ceiling
    (100, 0.30),   # Satisfactory ceiling
    (200, 0.55),   # Moderate ceiling  ← stretched vs naive 0.40
    (300, 0.75),   # Poor ceiling
    (400, 0.90),   # Very Poor ceiling
    (500, 1.00),   # Severe ceiling
]
_BP_RAW  = [b[0] for b in _AQI_BREAKPOINTS]
_BP_NORM = [b[1] for b in _AQI_BREAKPOINTS]

# AQI bins for health reference join (India CPCB categories)
_AQI_BINS       = [0, 50, 100, 200, 300, 400, 501]
_AQI_BIN_LABELS = ["Good", "Satisfactory", "Moderate", "Poor", "VeryPoor", "Severe"]

# Columns scaled with StandardScaler (fit on train only)
_SCALE_STANDARD = [
    "PM2.5", "PM10", "NO2", "SO2", "CO", "O3",
    "Temp_2m_C", "Humidity_Percent", "Wind_Speed_kmh", "Precipitation_mm",
    "AQI_lag1", "AQI_lag3", "AQI_lag7",
    "AQI_roll7_mean", "AQI_roll7_std",
    "AQI_delta1", "AQI_delta3",
    "PM2.5_lag1", "PM10_lag1", "NO2_lag1",
    "ref_health_score", "ref_resp_cases",
]
# Columns scaled with MinMaxScaler (already bounded)
_SCALE_MINMAX = ["AQI_norm_india", "Wind_Stagnation"]

# Final ordered feature set for AQI forecasting models
AQI_FEATURE_COLS = [
    # Raw pollutant concentrations (6)
    "PM2.5", "PM10", "NO2", "SO2", "CO", "O3",
    # Weather conditions (5) — affect pollutant formation and dispersion
    "Temp_2m_C", "Humidity_Percent", "Wind_Speed_kmh", "Precipitation_mm", "Wind_Stagnation",
    # Event flags (3) — known pollution spike causes
    "Temp_Inversion", "Festival_Period", "Crop_Burning_Season",
    # Calendar + city identity (3)
    "month", "dayofweek", "city_enc",
    # Season one-hot (4) — captures seasonal pollution patterns
    "season_Winter", "season_Monsoon", "season_Post_Monsoon", "season_Summer",
    # AQI temporal memory (8) — autocorrelation and trend signals
    "AQI_lag1", "AQI_lag3", "AQI_lag7",
    "AQI_roll7_mean", "AQI_roll7_std",
    "AQI_delta1", "AQI_delta3",
    "AQI_norm_india",
    # Key pollutant lags (3) — PM2.5/PM10/NO2 drive most AQI variance
    "PM2.5_lag1", "PM10_lag1", "NO2_lag1",
    # Health impact reference (2) — AQI-bin-level health burden signal
    "ref_health_score", "ref_resp_cases",
]


# ─── AQI normalization ─────────────────────────────────────────────────────────

def normalize_aqi_india(aqi: float) -> float:
    """
    Piecewise-linear normalization using India CPCB AQI scale.
    Unlike a simple /500 divide, this correctly weights the bands where
    Indian cities spend most of their time (100-400 range).
    """
    return float(np.interp(np.clip(aqi, 0, 500), _BP_RAW, _BP_NORM))


# Backwards-compatible alias
def normalize_aqi(aqi: float) -> float:
    return normalize_aqi_india(aqi)


# ─── Dataset loaders ───────────────────────────────────────────────────────────

def _load_city_day(path: str) -> pd.DataFrame:
    """
    Load city_day.csv, filter to MAJOR_CITIES, keep core pollutant + AQI columns.
    Drops: NO, NOx, Benzene, Toluene, Xylene (missing-heavy or redundant),
           NH3 (35% missing, no match in INDIA_AQI_COMPLETE),
           AQI_Bucket (categorical label, not a feature).
    """
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df[df["City"].isin(MAJOR_CITIES)].copy()

    keep = ["City", "Date"] + POLLUTANTS + ["AQI"]
    df = df[[c for c in keep if c in df.columns]]

    df["_source"] = "city_day"
    return df.sort_values(["City", "Date"]).reset_index(drop=True)


def _load_india_complete(path: str) -> pd.DataFrame:
    """
    Load INDIA_AQI_COMPLETE, read only needed columns, aggregate hourly → daily.
    Uses US_AQI as the AQI column (most complete signal; both US and India scales
    run 0-500, so the India CPCB normalization applied later is still valid).
    Drops 100%-null columns and all redundant AQI sub-indices.
    """
    needed_cols = [
        "City", "Datetime",
        "PM2_5_ugm3", "PM10_ugm3", "CO_ugm3", "NO2_ugm3", "SO2_ugm3", "O3_ugm3",
        "US_AQI",
        "Temp_2m_C", "Humidity_Percent", "Wind_Speed_10m_kmh",
        "Precipitation_mm", "Temp_Inversion", "Festival_Period",
        "Crop_Burning_Season", "Wind_Stagnation", "Season",
    ]

    df = pd.read_csv(
        path,
        usecols=needed_cols,
        parse_dates=["Datetime"],
        low_memory=False,
    )
    df = df[df["City"].isin(MAJOR_CITIES)].copy()

    # Rename pollutant columns to match city_day convention
    df = df.rename(columns={
        "PM2_5_ugm3":       "PM2.5",
        "PM10_ugm3":        "PM10",
        "CO_ugm3":          "CO",
        "NO2_ugm3":         "NO2",
        "SO2_ugm3":         "SO2",
        "O3_ugm3":          "O3",
        "US_AQI":           "AQI",
        "Wind_Speed_10m_kmh": "Wind_Speed_kmh",
    })

    df["Date"] = df["Datetime"].dt.normalize()

    # Aggregate hourly → daily per city
    agg = {p: "mean" for p in POLLUTANTS}
    agg.update({
        "AQI":               "mean",
        "Temp_2m_C":         "mean",
        "Humidity_Percent":  "mean",
        "Wind_Speed_kmh":    "mean",
        "Precipitation_mm":  "sum",   # total daily rainfall
        "Temp_Inversion":    "max",   # any hour with inversion → flag the day
        "Festival_Period":   "max",
        "Crop_Burning_Season": "max",
        "Wind_Stagnation":   "mean",  # fraction of stagnant hours (0–1)
        "Season":            lambda x: x.mode().iloc[0] if len(x) > 0 else np.nan,
    })
    agg = {k: v for k, v in agg.items() if k in df.columns}

    daily = df.groupby(["City", "Date"]).agg(agg).reset_index()
    daily["_source"] = "india_complete"
    return daily.sort_values(["City", "Date"]).reset_index(drop=True)


def _load_health_impact(path: str) -> pd.DataFrame:
    """
    Load air_quality_health_impact_data.csv (no city/date — global synthetic reference).
    Computes median HealthImpactScore and RespiratoryCases per India CPCB AQI bin.
    Returns a small lookup: AQI_bin → (ref_health_score, ref_resp_cases), both 0–1.
    These features add a learned health burden baseline per pollution level.
    """
    df = pd.read_csv(path)

    df["AQI_bin"] = pd.cut(
        df["AQI"],
        bins=_AQI_BINS,
        labels=_AQI_BIN_LABELS,
        include_lowest=True,
    )

    ref = (
        df.groupby("AQI_bin", observed=True)
        .agg(
            ref_health_score=("HealthImpactScore", "median"),
            ref_resp_cases=("RespiratoryCases", "median"),
        )
        .reset_index()
    )

    # Normalize to 0–1
    ref["ref_health_score"] = ref["ref_health_score"] / ref["ref_health_score"].max()
    ref["ref_resp_cases"]   = ref["ref_resp_cases"]   / ref["ref_resp_cases"].max()

    return ref


# ─── Dataset merge ─────────────────────────────────────────────────────────────

def _merge_datasets(
    city_day: pd.DataFrame,
    india: pd.DataFrame,
    health_ref: pd.DataFrame,
) -> pd.DataFrame:
    """
    Concatenate city_day (2015-2020) and india_complete (2022-2025) — no date overlap.
    Derive Season / Festival / CropBurning / Inversion for city_day rows.
    Impute weather columns for city_day rows using city×month medians from INDIA data.
    Add health reference features by AQI bin join.
    """
    # Align columns: add missing weather cols to city_day as NaN
    weather_cols = [
        "Temp_2m_C", "Humidity_Percent", "Wind_Speed_kmh", "Precipitation_mm",
        "Temp_Inversion", "Festival_Period", "Crop_Burning_Season",
        "Wind_Stagnation", "Season",
    ]
    for col in weather_cols:
        if col not in city_day.columns:
            city_day = city_day.copy()
            city_day[col] = np.nan

    combined = pd.concat([city_day, india], ignore_index=True, sort=False)
    combined = combined.sort_values(["City", "Date"]).reset_index(drop=True)

    month = combined["Date"].dt.month

    # ── Derive Season for city_day rows ──────────────────────────────────────
    def _month_to_season(m: int) -> str:
        if m in (6, 7, 8, 9):
            return "Monsoon"
        elif m in (10, 11):
            return "Post_Monsoon"
        elif m in (12, 1, 2):
            return "Winter"
        return "Summer"   # 3, 4, 5

    missing_season = combined["Season"].isna()
    combined.loc[missing_season, "Season"] = month[missing_season].map(_month_to_season)

    # ── Derive Festival_Period (Diwali: Oct 20 – Nov 10 each year) ───────────
    missing_fest = combined["Festival_Period"].isna()
    diwali_mask  = (
        ((month == 10) & (combined["Date"].dt.day >= 20)) |
        ((month == 11) & (combined["Date"].dt.day <= 10))
    )
    combined.loc[missing_fest, "Festival_Period"] = diwali_mask[missing_fest].astype(int)
    combined["Festival_Period"] = combined["Festival_Period"].fillna(0).astype(int)

    # ── Derive Crop_Burning_Season (Oct–Nov, north India only) ───────────────
    missing_crop = combined["Crop_Burning_Season"].isna()
    crop_mask    = month.isin([10, 11]) & combined["City"].isin(NORTH_INDIA_CITIES)
    combined.loc[missing_crop, "Crop_Burning_Season"] = crop_mask[missing_crop].astype(int)
    combined["Crop_Burning_Season"] = combined["Crop_Burning_Season"].fillna(0).astype(int)

    # ── Derive Temp_Inversion (Nov–Feb, north India — where inversions form) ─
    missing_inv   = combined["Temp_Inversion"].isna()
    inversion_mask = month.isin([11, 12, 1, 2]) & combined["City"].isin(NORTH_INDIA_CITIES)
    combined.loc[missing_inv, "Temp_Inversion"] = inversion_mask[missing_inv].astype(int)
    combined["Temp_Inversion"] = combined["Temp_Inversion"].fillna(0).astype(int)

    # ── Impute weather for city_day rows using INDIA city×month medians ───────
    # Compute medians from INDIA rows (actual measurements)
    india_rows = combined[combined["_source"] == "india_complete"].copy()
    india_rows["_month"] = india_rows["Date"].dt.month

    continuous_weather = [
        "Temp_2m_C", "Humidity_Percent", "Wind_Speed_kmh",
        "Precipitation_mm", "Wind_Stagnation",
    ]
    combined["_month"] = combined["Date"].dt.month

    for wcol in continuous_weather:
        if wcol not in india_rows.columns:
            continue

        med_lookup = (
            india_rows.groupby(["City", "_month"])[wcol]
            .median()
            .rename(f"_med_{wcol}")
            .reset_index()
        )
        combined = combined.merge(med_lookup, on=["City", "_month"], how="left")

        # Fill NaN weather with city×month median from INDIA data
        mask = combined[wcol].isna()
        combined.loc[mask, wcol] = combined.loc[mask, f"_med_{wcol}"]

        # Final fallback: global median
        combined[wcol] = combined[wcol].fillna(combined[wcol].median())
        combined = combined.drop(columns=[f"_med_{wcol}"])

    combined = combined.drop(columns=["_month"], errors="ignore")

    # ── Add health reference features by AQI bin ──────────────────────────────
    combined["AQI_bin"] = pd.cut(
        combined["AQI"],
        bins=_AQI_BINS,
        labels=_AQI_BIN_LABELS,
        include_lowest=True,
    )
    combined = combined.merge(health_ref, on="AQI_bin", how="left")
    combined["ref_health_score"] = combined["ref_health_score"].fillna(0.5)
    combined["ref_resp_cases"]   = combined["ref_resp_cases"].fillna(0.3)
    combined = combined.drop(columns=["AQI_bin"])

    return combined


# ─── Per-city seasonal imputation ─────────────────────────────────────────────

def _impute_per_city(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing pollutant and weather values per city.
    Strategy: forward-fill + backward-fill per city (preserves temporal continuity),
    then city×season median, then city median, then global median.
    """
    df = df.copy()
    impute_cols = POLLUTANTS + [
        "Temp_2m_C", "Humidity_Percent", "Wind_Speed_kmh", "Precipitation_mm",
    ]

    for col in impute_cols:
        if col not in df.columns:
            continue

        # Time-aware fill within each city
        df[col] = df.groupby("City")[col].transform(lambda s: s.ffill().bfill())

        # City × season median for remaining gaps
        cs_median = df.groupby(["City", "Season"])[col].transform("median")
        df[col] = df[col].fillna(cs_median)

        # City median → global median as final fallbacks
        c_median = df.groupby("City")[col].transform("median")
        df[col] = df[col].fillna(c_median).fillna(df[col].median())

    return df


# ─── Per-city seasonal outlier treatment ──────────────────────────────────────

def _treat_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clip outliers per city × season using IQR fences.

    Normal rows:      upper = Q3 + 3.0 × IQR  (generous — avoids clipping real spikes)
    Event rows        upper = Q3 + 5.0 × IQR  (Festival_Period=1, Crop_Burning=1, or
    (known spikes):                             Temp_Inversion=1 → preserve actual spikes)
    Lower fence:      max(0, Q1 - 1.5 × IQR)  (pollutants can't be negative)

    Using 3× IQR (not the standard 1.5×) because Indian air quality has genuine
    heavy-pollution events — tightening clips legitimate readings.
    """
    df = df.copy()
    treat_cols = POLLUTANTS + ["AQI"]

    event_col_mask = (
        (df["Festival_Period"] == 1) |
        (df["Crop_Burning_Season"] == 1) |
        (df["Temp_Inversion"] == 1)
    )

    for city in df["City"].unique():
        city_mask = df["City"] == city

        for season in df["Season"].unique():
            group_mask = city_mask & (df["Season"] == season)
            if group_mask.sum() < 10:
                continue

            is_event  = group_mask & event_col_mask
            is_normal = group_mask & ~event_col_mask

            for col in treat_cols:
                if col not in df.columns:
                    continue

                vals = df.loc[group_mask, col]
                q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
                iqr = q3 - q1
                if iqr == 0:
                    continue

                lower         = max(0.0, q1 - 1.5 * iqr)
                upper_normal  = q3 + 3.0 * iqr
                upper_event   = q3 + 5.0 * iqr

                if is_normal.any():
                    df.loc[is_normal, col] = df.loc[is_normal, col].clip(lower, upper_normal)
                if is_event.any():
                    df.loc[is_event,  col] = df.loc[is_event,  col].clip(lower, upper_event)

    return df


# ─── Feature engineering ──────────────────────────────────────────────────────

def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the final feature set. Drops redundant or overfitting-prone features:
      - quarter (redundant with month)
      - dayofyear (high cardinality, captured by month + season)
      - AQI_roll3 (redundant with AQI_lag1)
      - AQI_delta_trend / acceleration (noisy)
      - per-pollutant roll/lag3/lag7 (collinear with AQI lags)
    """
    df = df.copy()

    # ── Calendar ──────────────────────────────────────────────────────────────
    df["month"]     = df["Date"].dt.month
    df["dayofweek"] = df["Date"].dt.dayofweek

    # ── City label encoding (ordinal, 0-indexed alphabetically) ──────────────
    city_order = sorted(df["City"].unique())
    df["city_enc"] = df["City"].map({c: i for i, c in enumerate(city_order)})

    # ── Season one-hot (4 dummies) ────────────────────────────────────────────
    df["Season"] = df["Season"].str.strip()
    s_dummies = pd.get_dummies(df["Season"], prefix="season")
    for col in ["season_Winter", "season_Monsoon", "season_Post_Monsoon", "season_Summer"]:
        if col not in s_dummies.columns:
            s_dummies[col] = 0
    s_dummies = s_dummies[
        ["season_Winter", "season_Monsoon", "season_Post_Monsoon", "season_Summer"]
    ]
    df = pd.concat([df, s_dummies], axis=1)

    # ── AQI lag features (1, 3, 7 days) ──────────────────────────────────────
    for lag in [1, 3, 7]:
        df[f"AQI_lag{lag}"] = df.groupby("City")["AQI"].shift(lag)

    # ── AQI rolling stats (7-day; 3-day dropped — redundant with lag1) ────────
    df["AQI_roll7_mean"] = (
        df.groupby("City")["AQI"]
        .transform(lambda s: s.shift(1).rolling(7, min_periods=3).mean())
    )
    df["AQI_roll7_std"] = (
        df.groupby("City")["AQI"]
        .transform(lambda s: s.shift(1).rolling(7, min_periods=3).std().fillna(0))
    )

    # ── AQI velocity (1-day and 3-day; acceleration dropped — noisy) ─────────
    df["AQI_delta1"] = df.groupby("City")["AQI"].diff(1)
    df["AQI_delta3"] = df.groupby("City")["AQI"].diff(3)

    # ── India CPCB normalized AQI ─────────────────────────────────────────────
    df["AQI_norm_india"] = df["AQI"].apply(normalize_aqi_india)

    # ── Pollutant lag1 (PM2.5, PM10, NO2 drive most AQI variance) ────────────
    df["PM2.5_lag1"] = df.groupby("City")["PM2.5"].shift(1)
    df["PM10_lag1"]  = df.groupby("City")["PM10"].shift(1)
    df["NO2_lag1"]   = df.groupby("City")["NO2"].shift(1)

    # ── Wind stagnation: ensure 0–1 range ────────────────────────────────────
    df["Wind_Stagnation"] = df["Wind_Stagnation"].clip(0.0, 1.0).fillna(0.0)

    # ── Drop rows where lag features are NaN (first few rows per city) ────────
    lag_check = ["AQI_lag1", "AQI_lag3", "AQI_lag7",
                 "PM2.5_lag1", "PM10_lag1", "NO2_lag1"]
    df = df.dropna(subset=lag_check).reset_index(drop=True)

    return df


# ─── Scaling ──────────────────────────────────────────────────────────────────

def _fit_scalers(X_train: pd.DataFrame):
    """Fit StandardScaler and MinMaxScaler on training data only."""
    std_cols = [c for c in _SCALE_STANDARD if c in X_train.columns]
    mm_cols  = [c for c in _SCALE_MINMAX   if c in X_train.columns]

    std_scaler = StandardScaler()
    mm_scaler  = MinMaxScaler()

    if std_cols:
        std_scaler.fit(X_train[std_cols])
    if mm_cols:
        mm_scaler.fit(X_train[mm_cols])

    return std_scaler, mm_scaler, std_cols, mm_cols


def _apply_scalers(
    X: pd.DataFrame, std_scaler, mm_scaler, std_cols: list, mm_cols: list
) -> pd.DataFrame:
    """Apply pre-fitted scalers (no re-fitting) to prevent data leakage."""
    X = X.copy()
    if std_cols:
        X[std_cols] = std_scaler.transform(X[std_cols])
    if mm_cols:
        X[mm_cols]  = mm_scaler.transform(X[mm_cols])
    return X


# ─── Main pipeline ─────────────────────────────────────────────────────────────

def build_training_df(
    city_day_path: str = "data/city_day.csv",
    india_path:    str = "data/INDIA_AQI_COMPLETE_20251126.csv",
    health_path:   str = "data/air_quality_health_impact_data.csv",
    test_size:     float = 0.20,
) -> dict:
    """
    Full preprocessing pipeline combining all three datasets.

    Returns
    -------
    dict with keys:
      "train"       → {"X": scaled_features, "y": AQI, "raw": unscaled_df,
                        "dates": DateSeries, "cities": CitySeries}
      "test"        → same structure as "train"
      "full_df"     → complete processed DataFrame (unscaled, all cols) for AQI forecaster
      "scalers"     → {"standard": std_scaler, "minmax": mm_scaler,
                        "std_cols": list, "mm_cols": list}
      "feature_cols"→ ordered list of feature column names
    """
    # Step 1 — Load
    print("  Loading city_day …")
    df_city = _load_city_day(city_day_path)

    print("  Loading INDIA_AQI_COMPLETE (large file, may take ~30s) …")
    df_india = _load_india_complete(india_path)

    print("  Loading health impact reference …")
    health_ref = _load_health_impact(health_path)

    # Step 2 — Merge
    print("  Merging datasets …")
    df = _merge_datasets(df_city, df_india, health_ref)

    # Drop rows with missing AQI target
    df = df.dropna(subset=["AQI"]).copy()

    # Step 3 — Impute missing values per city
    print("  Imputing missing values per city …")
    df = _impute_per_city(df)

    # Step 4 — Per-city seasonal outlier treatment
    print("  Treating outliers per city × season …")
    df = _treat_outliers(df)

    # Step 5 — Feature engineering
    print("  Engineering features …")
    df = _engineer_features(df)

    # Keep only the feature cols + metadata, drop internal helper columns
    feat_cols = [c for c in AQI_FEATURE_COLS if c in df.columns]
    meta      = ["City", "Date", "AQI", "Season", "_source"]
    df = df[[c for c in meta + feat_cols if c in df.columns]].copy()

    # Sort by date for temporal split
    df = df.sort_values("Date").reset_index(drop=True)

    # Step 6 — 80/20 temporal train/test split
    split_idx = int(len(df) * (1.0 - test_size))
    train_df  = df.iloc[:split_idx].copy()
    test_df   = df.iloc[split_idx:].copy()

    X_train = train_df[feat_cols]
    y_train = train_df["AQI"]
    X_test  = test_df[feat_cols]
    y_test  = test_df["AQI"]

    # Step 7 — Fit scalers on training data only (no leakage)
    std_scaler, mm_scaler, std_cols, mm_cols = _fit_scalers(X_train)
    X_train_sc = _apply_scalers(X_train, std_scaler, mm_scaler, std_cols, mm_cols)
    X_test_sc  = _apply_scalers(X_test,  std_scaler, mm_scaler, std_cols, mm_cols)

    print(
        f"  Done — {len(df):,} total rows | "
        f"Train: {len(train_df):,} | Test: {len(test_df):,} | "
        f"Features: {len(feat_cols)} | Cities: {df['City'].nunique()}"
    )

    return {
        "train": {
            "X":       X_train_sc,
            "y":       y_train,
            "raw":     train_df,
            "dates":   train_df["Date"],
            "cities":  train_df["City"],
        },
        "test": {
            "X":       X_test_sc,
            "y":       y_test,
            "raw":     test_df,
            "dates":   test_df["Date"],
            "cities":  test_df["City"],
        },
        "full_df":     df,            # unscaled, used by AQI forecaster (trees don't need scaling)
        "scalers": {
            "standard": std_scaler,
            "minmax":   mm_scaler,
            "std_cols": std_cols,
            "mm_cols":  mm_cols,
        },
        "feature_cols": feat_cols,
    }


# ─── Standalone smoke-test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    pipeline = build_training_df()
    train, test = pipeline["train"], pipeline["test"]
    feat_cols   = pipeline["feature_cols"]

    print(f"\nFeatures ({len(feat_cols)}):")
    for f in feat_cols:
        print(f"  {f}")

    print(f"\nTrain AQI stats:\n{train['y'].describe()}")
    print(f"\nTest  AQI stats:\n{test['y'].describe()}")
    print(f"\nTrain date range: {train['dates'].min().date()} → {train['dates'].max().date()}")
    print(f"Test  date range: {test['dates'].min().date()}  → {test['dates'].max().date()}")

    print("\nNull counts in train X:")
    nulls = train["X"].isnull().sum()
    print(nulls[nulls > 0] if nulls.any() else "  None")

    print("\nNormalization sanity check (CPCB piecewise vs /500):")
    for v in [0, 50, 100, 150, 200, 250, 300, 400, 500]:
        cpcb  = normalize_aqi_india(v)
        naive = v / 500
        print(f"  AQI {v:3d} → CPCB {cpcb:.3f}  naive {naive:.3f}  diff {cpcb-naive:+.3f}")
