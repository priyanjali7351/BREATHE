"""
BioAQI — Training Script
Run: python train.py
Trains all models and saves them + metrics to models/

AQI Forecaster: India-only, hourly (INDIA_AQI_COMPLETE_20251126.csv), horizons
+6h/+12h/+24h/+48h. PHRS: unchanged legacy 3-source daily pipeline (out of
scope for the hourly rewrite — see legacy_phrs_pipeline.py).
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from preprocess import build_training_df
from legacy_phrs_pipeline import build_phrs_training_df
from generate_profiles import generate_phrs_dataset
from models import (
    HORIZONS_HOURS,
    MAX_HOURLY_CHANGE,
    compute_sanity_caps,
    train_aqi_forecaster,
    train_phrs_model,
    feature_importance,
    feature_importance_by_group,
)


def main(
    india_path: str = "data/INDIA_AQI_COMPLETE_20251126.csv",
    city_day_path: str = "data/city_day.csv",
    health_path: str = "data/air_quality_health_impact_data.csv",
    profiles_per_row: int = 5,
):
    print("=" * 60)
    print("  BioAQI — Training Pipeline (India-only, hourly)")
    print("=" * 60)

    all_metrics: dict = {}

    # ── Step 1: Preprocess — India-only hourly pipeline ──────────────────────
    print("\n[1/5] Loading & preprocessing India AQI data (hourly) …")
    t0 = time.time()
    pipeline = build_training_df(india_path=india_path)
    full_df = pipeline["full_df"]
    feat_cols = pipeline["feature_cols"]
    print(f"      {len(full_df):,} total records · {len(feat_cols)} features  ({time.time()-t0:.1f}s)")

    print("\n      Acceptance check — no *_AQI_*/*_Category leakage column in feature_cols:")
    leak_hits = [c for c in feat_cols if "AQI_" in c and not c.startswith("AQI_lag") and not c.startswith("AQI_roll")]
    leak_hits = [c for c in leak_hits if c not in ("Crop_Burning_Season",)]
    print(f"      {'PASS — none found' if not leak_hits else 'FAIL — ' + str(leak_hits)}")

    # ── Step 2: Sanity-clip caps — 95th pct |US_AQI(t+h) - US_AQI(t)| ────────
    print("\n[2/5] Computing sanity-clip caps (95th pct hourly AQI deltas, train split only) …")
    train_cutoff = int(len(full_df) * 0.80)
    train_only_df = full_df.sort_values("Datetime").iloc[:train_cutoff]
    caps = compute_sanity_caps(train_only_df, HORIZONS_HOURS)
    MAX_HOURLY_CHANGE.update(caps)
    with open("models/sanity_caps.json", "w") as f:
        json.dump(caps, f, indent=2)
    print(f"      Saved -> models/sanity_caps.json")

    # ── Step 3: Train AQI Forecasters (hourly horizons) ──────────────────────
    print("\n[3/5] Training AQI Forecasters …")
    importance_24h = None
    for h in HORIZONS_HOURS:
        print(f"\n  Horizon +{h}h:")
        result = train_aqi_forecaster(full_df, horizon=h)
        all_metrics[f"aqi_h{h}"] = {
            "label": f"AQI Forecaster +{h}h",
            "horizon_hours": h,
            **result["metrics"],
            "train_r2": result["train_metrics"]["r2"],
            "train_mae": result["train_metrics"]["mae"],
            "train_rmse": result["train_metrics"]["rmse"],
        }
        if h == 24:
            importance_24h = result

    # ── Step 4: Generate PHRS dataset & train PHRS model (legacy, unchanged) ─
    print(f"\n[4/5] Generating synthetic PHRS dataset ({profiles_per_row} profiles/row) — legacy pipeline …")
    t0 = time.time()
    phrs_pipeline = build_phrs_training_df(
        city_day_path=city_day_path, india_path=india_path, health_path=health_path,
    )
    train_raw = phrs_pipeline["train"]["raw"]
    phrs_df = generate_phrs_dataset(train_raw, n_profiles_per_row=profiles_per_row)
    print(f"      {len(phrs_df):,} PHRS samples  ({time.time()-t0:.1f}s)")
    phrs_df.to_csv("data/phrs_dataset.csv", index=False)
    print("      Saved -> data/phrs_dataset.csv")

    print("\n  Training PHRS Predictor …")
    result_phrs = train_phrs_model(phrs_df)
    all_metrics["phrs"] = {
        "label": "PHRS Predictor",
        **result_phrs["metrics"],
        "train_r2": result_phrs["train_metrics"]["r2"],
        "train_mae": result_phrs["train_metrics"]["mae"],
        "train_rmse": result_phrs["train_metrics"]["rmse"],
        "cv_r2": result_phrs["cv_r2"],
        "cv_r2_std": result_phrs["cv_r2_std"],
    }

    # ── Step 5: Save metrics & feature importance ─────────────────────────────
    print("\n[5/5] Saving metrics and feature importance …")

    metrics_path = Path("models/metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"      Saved -> {metrics_path}")

    print("\n  Feature importance (PHRS model, top 10):")
    imp = feature_importance(result_phrs, top_n=10)
    print(imp.to_string(index=False))

    if importance_24h is not None:
        print("\n  Per-group feature importance — AQI Forecaster +24h:")
        grp_imp = feature_importance_by_group(importance_24h)
        print(grp_imp.to_string(index=False))

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Metrics Summary")
    print("=" * 60)
    print(f"  {'Model':<24} {'Test R²':>8} {'Test MAE':>10} {'Test RMSE':>11} {'Train R²':>9}")
    print("  " + "-" * 65)
    for key, m in all_metrics.items():
        print(
            f"  {m['label']:<24} {m['r2']:>8.4f} {m['mae']:>10.2f} "
            f"{m['rmse']:>11.2f} {m['train_r2']:>9.4f}"
        )
    print("\n  All models trained and saved to models/")
    print("  Run: streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BioAQI Training Script")
    parser.add_argument("--india", default="data/INDIA_AQI_COMPLETE_20251126.csv",
                        help="Path to INDIA_AQI_COMPLETE CSV")
    parser.add_argument("--city_day", default="data/city_day.csv",
                        help="Path to city_day.csv (PHRS pipeline only)")
    parser.add_argument("--health", default="data/air_quality_health_impact_data.csv",
                        help="Path to air_quality_health_impact_data.csv (PHRS pipeline only)")
    parser.add_argument("--profiles", default=5, type=int,
                        help="Synthetic profiles per AQI row (PHRS)")
    args = parser.parse_args()

    Path("data").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    main(args.india, args.city_day, args.health, args.profiles)
