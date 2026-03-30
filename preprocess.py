"""
BioAQI — Data Preprocessing Module
Loads city_day.csv, cleans it, engineers features,
and prepares it for model training.
"""

import pandas as pd
import numpy as np
from pathlib import Path


# Pollutants used in PHRS and AQI forecasting
POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "NH3"]

# India CPCB AQI piecewise breakpoints
# Each tuple: (raw_aqi, normalized_0_to_1)
# Reflects that most Indian cities sit in the 100-400 range, so the middle
# bands are stretched — a "Moderate" day (AQI 200) maps to 0.55, not 0.40.
_AQI_BREAKPOINTS = [
    (0,   0.00),
    (50,  0.15),   # Good ceiling
    (100, 0.30),   # Satisfactory ceiling
    (200, 0.55),   # Moderate ceiling
    (300, 0.75),   # Poor ceiling
    (400, 0.90),   # Very Poor ceiling
    (500, 1.00),   # Severe ceiling
]
_BP_RAW  = [b[0] for b in _AQI_BREAKPOINTS]
_BP_NORM = [b[1] for b in _AQI_BREAKPOINTS]


def normalize_aqi_india(aqi: float) -> float:
    """
    Piecewise-linear normalization using India CPCB AQI scale.
    Unlike a simple /500 divide, this correctly weights the bands where
    Indian cities actually spend most of their time (100-400 range).

    AQI   0 → 0.00
    AQI  50 → 0.15  (Good/Satisfactory boundary)
    AQI 100 → 0.30  (Satisfactory/Moderate boundary)
    AQI 200 → 0.55  (Moderate/Poor boundary)  ← was 0.40 with /500
    AQI 300 → 0.75  (Poor/Very Poor boundary)
    AQI 400 → 0.90  (Very Poor/Severe boundary)
    AQI 500 → 1.00  (cap)
    """
    return float(np.interp(np.clip(aqi, 0, 500), _BP_RAW, _BP_NORM))


def load_data(path: str = "data/city_day.csv") -> pd.DataFrame:
    """Load raw Kaggle AQI dataset."""
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values(["City", "Date"]).reset_index(drop=True)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Drop rows with no AQI target
    - Forward-fill then median-fill pollutant columns per city
    - Cap extreme outliers at 99.5th percentile
    """
    df = df.dropna(subset=["AQI"]).copy()

    for col in POLLUTANTS:
        if col in df.columns:
            # Forward-fill then back-fill within each city
            # Using .ffill()/.bfill() — fillna(method=...) is deprecated in pandas 2+
            df[col] = df.groupby("City")[col].transform(
                lambda s: s.ffill().bfill()
            )
            # Remaining NaNs → city median, then global median
            city_med = df.groupby("City")[col].transform("median")
            df[col] = df[col].fillna(city_med).fillna(df[col].median())

            # Cap outliers at 99.5th percentile
            p995 = df[col].quantile(0.995)
            df[col] = df[col].clip(upper=p995)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based and lag features for AQI forecasting.
    Operates per-city so lags don't bleed across cities.
    """
    df = df.copy()

    # Calendar features
    df["month"]      = df["Date"].dt.month
    df["dayofweek"]  = df["Date"].dt.dayofweek
    df["dayofyear"]  = df["Date"].dt.dayofyear
    df["quarter"]    = df["Date"].dt.quarter

    # Lag features (1, 3, 7 days) and rolling stats
    lag_cols = ["AQI"] + [p for p in POLLUTANTS if p in df.columns]
    for lag in [1, 3, 7]:
        for col in lag_cols:
            df[f"{col}_lag{lag}"] = df.groupby("City")[col].shift(lag)

    for window in [3, 7]:
        for col in lag_cols:
            df[f"{col}_roll{window}_mean"] = (
                df.groupby("City")[col]
                .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
            )
            df[f"{col}_roll{window}_std"] = (
                df.groupby("City")[col]
                .transform(lambda s: s.shift(1).rolling(window, min_periods=1).std().fillna(0))
            )

    # First-order delta (velocity) and second-order delta (acceleration).
    # The acceleration term tells the model whether a trend is speeding up or
    # slowing down — a strong signal for multi-day forecasting.
    df["AQI_delta1"] = df.groupby("City")["AQI"].diff(1)
    df["AQI_delta3"] = df.groupby("City")["AQI"].diff(3)
    df["AQI_delta_trend"] = df.groupby("City")["AQI_delta1"].diff(1)  # acceleration

    # India-normalised AQI as an explicit feature (captures non-linear scale)
    df["AQI_norm_india"] = df["AQI"].apply(normalize_aqi_india)

    # Drop rows with NaN lags (first few rows per city)
    df = df.dropna(subset=[c for c in df.columns if "lag" in c]).reset_index(drop=True)

    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return feature column names (excludes metadata and target)."""
    exclude = {"City", "Date", "AQI", "AQI_Bucket"}
    return [c for c in df.columns if c not in exclude]


def normalize_aqi(aqi: float) -> float:
    """Backwards-compatible wrapper — now uses India CPCB piecewise scale."""
    return normalize_aqi_india(aqi)


def build_training_df(path: str = "data/city_day.csv") -> tuple[pd.DataFrame, list[str]]:
    """Full preprocessing pipeline. Returns (df, feature_cols)."""
    raw       = load_data(path)
    cleaned   = clean_data(raw)
    featured  = engineer_features(cleaned)
    feat_cols = get_feature_cols(featured)
    return featured, feat_cols


if __name__ == "__main__":
    df, feat_cols = build_training_df()
    print(f"Shape: {df.shape}")
    print(f"Features ({len(feat_cols)}): {feat_cols[:10]} ...")
    print(df[["City", "Date", "AQI"]].head(10))
    print("\nNormalization sanity check (CPCB piecewise vs /500):")
    for v in [0, 50, 100, 150, 200, 250, 300, 400, 500]:
        cpcb  = normalize_aqi_india(v)
        naive = v / 500
        print(f"  AQI {v:3d} → CPCB {cpcb:.3f}  naive {naive:.3f}  diff {cpcb-naive:+.3f}")
