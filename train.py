"""
BioAQI — Training Script
Run: python train.py
Trains all models and saves them + metrics to models/
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from preprocess import build_training_df
from generate_profiles import generate_phrs_dataset
from models import (
    train_aqi_forecaster,
    train_phrs_model,
    feature_importance,
)


def main(data_path: str = "data/city_day.csv", profiles_per_row: int = 5):
    print("=" * 60)
    print("  BioAQI — Training Pipeline")
    print("=" * 60)

    all_metrics: dict = {}

    # ── Step 1: Preprocess AQI data ──────────────────────────────
    print("\n[1/4] Loading & preprocessing AQI data …")
    t0 = time.time()
    df, feat_cols = build_training_df(data_path)
    print(f"      {len(df):,} records · {len(feat_cols)} features  ({time.time()-t0:.1f}s)")

    # ── Step 2: Train AQI Forecasters ────────────────────────────
    print("\n[2/4] Training AQI Forecasters …")
    for h in [1, 3, 7]:
        print(f"\n  Horizon +{h} day(s):")
        result = train_aqi_forecaster(df, horizon=h)
        all_metrics[f"aqi_h{h}"] = {
            "label":         f"AQI Forecaster +{h}d",
            "horizon_days":  h,
            **result["metrics"],
            "train_r2":      result["train_metrics"]["r2"],
            "train_mae":     result["train_metrics"]["mae"],
            "train_rmse":    result["train_metrics"]["rmse"],
        }

    # ── Step 3: Generate PHRS dataset & train PHRS model ─────────
    print(f"\n[3/4] Generating synthetic PHRS dataset ({profiles_per_row} profiles/row) …")
    t0 = time.time()
    phrs_df = generate_phrs_dataset(df, n_profiles_per_row=profiles_per_row)
    print(f"      {len(phrs_df):,} PHRS samples  ({time.time()-t0:.1f}s)")
    phrs_df.to_csv("data/phrs_dataset.csv", index=False)
    print("      Saved → data/phrs_dataset.csv")

    print("\n  Training PHRS Predictor …")
    result_phrs = train_phrs_model(phrs_df)
    all_metrics["phrs"] = {
        "label":     "PHRS Predictor",
        **result_phrs["metrics"],
        "train_r2":  result_phrs["train_metrics"]["r2"],
        "train_mae": result_phrs["train_metrics"]["mae"],
        "train_rmse":result_phrs["train_metrics"]["rmse"],
        "cv_r2":     result_phrs["cv_r2"],
        "cv_r2_std": result_phrs["cv_r2_std"],
    }

    # ── Step 4: Save metrics & feature importance ─────────────────
    print("\n[4/4] Saving metrics and feature importance …")

    metrics_path = Path("models/metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"      Saved → {metrics_path}")

    print("\n  Feature importance (PHRS model, top 10):")
    imp = feature_importance(result_phrs, top_n=10)
    print(imp.to_string(index=False))

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Metrics Summary")
    print("=" * 60)
    print(f"  {'Model':<24} {'Test R²':>8} {'Test MAE':>10} {'Test RMSE':>11} {'Train R²':>9}")
    print("  " + "-" * 65)
    for key, m in all_metrics.items():
        print(f"  {m['label']:<24} {m['r2']:>8.4f} {m['mae']:>10.2f} "
              f"{m['rmse']:>11.2f} {m['train_r2']:>9.4f}")
    print("\n  All models trained and saved to models/")
    print("  Run: streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BioAQI Training Script")
    parser.add_argument("--data",     default="data/city_day.csv",
                        help="Path to city_day.csv")
    parser.add_argument("--profiles", default=5, type=int,
                        help="Synthetic profiles per AQI row")
    args = parser.parse_args()

    Path("data").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    main(args.data, args.profiles)
