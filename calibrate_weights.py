"""
calibrate_weights.py — Data-driven PHRS weight calibration

Regresses HealthImpactScore (from air_quality_health_impact_data.csv) against
the existing PHRS formula sub-components to derive evidence-based weights for
W_AQI, W_POLLUTANT, and W_WEATHER.

W_PROFILE and W_TREND are NOT calibrated here (no age/activity/temporal data).

Usage:
    python calibrate_weights.py [--health PATH]

Output: prints recommended weights to terminal. To apply, manually update
W_AQI / W_POLLUTANT / W_PROFILE / W_TREND in generate_profiles.py, then
run: python train.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score

from generate_profiles import _aqi_component, _pollutant_component


def main():
    parser = argparse.ArgumentParser(description="Calibrate PHRS weights from health data")
    parser.add_argument("--health", default="data/air_quality_health_impact_data.csv",
                        help="Path to air_quality_health_impact_data.csv")
    args = parser.parse_args()

    health_path = Path(args.health)
    if not health_path.exists():
        print(f"ERROR: {health_path} not found. Download the dataset first.")
        return

    print(f"Loading {health_path} ...")
    df = pd.read_csv(health_path)

    required = {"AQI", "PM2_5", "PM10", "NO2", "SO2", "O3",
                "Temperature", "Humidity", "HealthImpactScore"}
    missing = required - set(df.columns)
    if missing:
        print(f"ERROR: Missing columns in dataset: {missing}")
        return

    df = df.dropna(subset=list(required))
    print(f"  {len(df):,} rows after dropping NA")

    # ── Compute PHRS sub-components ───────────────────────────────────────────
    print("Computing formula sub-components...")

    df["aqi_comp"] = df["AQI"].apply(_aqi_component)

    def _poll(row):
        return _pollutant_component(
            {
                "PM2.5": row["PM2_5"],
                "PM10":  row["PM10"],
                "NO2":   row["NO2"],
                "SO2":   row["SO2"],
                "O3":    row["O3"],
            },
            conditions=["Healthy"],
        )

    df["poll_comp"] = df.apply(_poll, axis=1)

    # Normalize temperature [10–50°C] → [0–1] and humidity [0–100%] → [0–1]
    df["temp_norm"]  = np.clip((df["Temperature"] - 10.0) / 40.0, 0.0, 1.0)
    df["humid_norm"] = np.clip(df["Humidity"] / 100.0, 0.0, 1.0)

    # ── Ridge regression ──────────────────────────────────────────────────────
    feature_names = ["aqi_comp", "poll_comp", "temp_norm", "humid_norm"]
    X = df[feature_names].values
    y = df["HealthImpactScore"].values

    # Scale features to [0, 1] for comparable coefficients
    scaler = MinMaxScaler()
    X_sc = scaler.fit_transform(X)

    model = Ridge(alpha=1.0, positive=True, fit_intercept=True)
    model.fit(X_sc, y)

    # 5-fold CV R²
    cv_scores = cross_val_score(model, X_sc, y, cv=5, scoring="r2")
    r2_cv    = cv_scores.mean()
    r2_std   = cv_scores.std()

    coefs = np.array(model.coef_)
    coefs_clipped = np.clip(coefs, 0.0, None)

    # ── Renormalize to sum = 1 across calibratable weights ───────────────────
    total = coefs_clipped.sum()
    if total == 0:
        print("WARNING: All coefficients are zero. Check that dataset has variance.")
        return

    raw_shares = coefs_clipped / total

    # These three correspond to W_AQI, W_POLLUTANT, and W_WEATHER
    w_aqi_raw     = raw_shares[0]
    w_poll_raw    = raw_shares[1]
    w_weather_raw = raw_shares[2] + raw_shares[3]  # temp + humidity combined

    # Fixed weights (paper-based, not calibratable from this dataset)
    W_PROFILE_FIXED = 0.20
    W_TREND_FIXED   = 0.05
    remaining       = 1.0 - W_PROFILE_FIXED - W_TREND_FIXED  # = 0.75 to distribute

    w_aqi_cal     = round(w_aqi_raw     * remaining, 3)
    w_poll_cal    = round(w_poll_raw    * remaining, 3)
    w_weather_cal = round(w_weather_raw * remaining, 3)

    # Adjust for rounding so all 5 sum to exactly 1.0
    total_so_far  = w_aqi_cal + w_poll_cal + w_weather_cal + W_PROFILE_FIXED + W_TREND_FIXED
    w_aqi_cal    += round(1.0 - total_so_far, 3)

    # ── Print results ─────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("PHRS WEIGHT CALIBRATION RESULTS")
    print("=" * 60)
    print(f"Dataset rows used: {len(df):,}")
    print(f"Ridge regression — 5-fold CV R²: {r2_cv:.4f} ± {r2_std:.4f}")
    print()
    print("Raw coefficient shares (calibratable features only):")
    for name, share in zip(["aqi_comp", "poll_comp", "temp_norm", "humid_norm"], raw_shares):
        print(f"  {name:<15}: {share:.3f}")
    print()
    print("Recommended weights (calibrated, summing to 1.0):")
    print(f"  W_AQI       = {w_aqi_cal:.3f}   (current: 0.500)")
    print(f"  W_POLLUTANT = {w_poll_cal:.3f}   (current: 0.250)")
    print(f"  W_WEATHER   = {w_weather_cal:.3f}   (not in current formula — new)")
    print(f"  W_PROFILE   = {W_PROFILE_FIXED:.3f}   (fixed — paper-based, not regression)")
    print(f"  W_TREND     = {W_TREND_FIXED:.3f}   (fixed — paper-based, not regression)")
    total_check = w_aqi_cal + w_poll_cal + w_weather_cal + W_PROFILE_FIXED + W_TREND_FIXED
    print(f"  Sum         = {total_check:.3f}")
    print()
    print("To apply: update generate_profiles.py lines 61–64, then run: python train.py")
    print()

    # ── Interpretation ────────────────────────────────────────────────────────
    if r2_cv < 0.30:
        print("NOTE: Low R² suggests HealthImpactScore in the dataset is only weakly")
        print("      explained by these sub-components alone. Current hand-tuned weights")
        print("      may be reasonable. Consider this output as a directional guide only.")
    elif r2_cv > 0.70:
        print("NOTE: Good R² — calibrated weights are strongly supported by the data.")

    print("=" * 60)


if __name__ == "__main__":
    main()
