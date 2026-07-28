"""
BioAQI — ML Models
  Model 1: AQI Forecaster  — predicts US_AQI +6h / +12h / +24h / +48h ahead
  Model 2: PHRS Predictor  — predicts personalized health risk score
Both use gradient-boosted trees (XGBoost).
"""

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from preprocess import AQI_FEATURE_COLS, FEATURE_GROUPS, TARGET_COL


MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

HORIZONS_HOURS = [6, 12, 24, 48]

# Hard caps on how much US_AQI can physically change per hourly horizon.
# Populated by compute_sanity_caps() at train time (95th percentile of
# |US_AQI(t+h) - US_AQI(t)| on the training split) and persisted to
# models/sanity_caps.json so the serving path can load it without retraining.
MAX_HOURLY_CHANGE: dict[int, float] = {}
# Best-effort load at import time so serving code (app.py, api/) gets the
# trained caps without needing an explicit call; train.py overwrites this
# dict again right after computing fresh caps.


def compute_sanity_caps(df: pd.DataFrame, horizons: list[int] = HORIZONS_HOURS) -> dict[int, float]:
    """
    95th percentile of |US_AQI(t+h) - US_AQI(t)|, measured per city on the
    hourly-gridded data (so t+h means exactly h hours, not h rows).
    """
    caps = {}
    for h in horizons:
        deltas = []
        for _, g in df.groupby("City")[TARGET_COL]:
            g = g.reset_index(drop=True)
            deltas.append((g.shift(-h) - g).abs())
        all_deltas = pd.concat(deltas).dropna()
        cap = float(np.percentile(all_deltas, 95))
        caps[h] = round(cap, 1)
        print(f"  95th-pct |US_AQI(t+{h}h) - US_AQI(t)| = {cap:.1f}")
    return caps


def load_sanity_caps() -> dict[int, float]:
    """Load the caps written by train.py (models/sanity_caps.json) into
    MAX_HOURLY_CHANGE for the serving path. Falls back to a linear 3 AQI/h
    estimate for any horizon missing from the file."""
    path = MODELS_DIR / "sanity_caps.json"
    if path.exists():
        with open(path) as f:
            caps = {int(k): float(v) for k, v in json.load(f).items()}
        MAX_HOURLY_CHANGE.update(caps)
    return MAX_HOURLY_CHANGE


load_sanity_caps()   # populate MAX_HOURLY_CHANGE at import time, if already trained


# ─── Shared helpers ────────────────────────────────────────────────────────────

def regression_report(y_true, y_pred, label: str = "") -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    print(f"  [{label}] MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")
    return {"mae": round(mae, 3), "rmse": round(rmse, 3), "r2": round(r2, 4)}


# ─── Model 1: AQI Forecaster ──────────────────────────────────────────────────

AQI_FORECAST_FEATURES = AQI_FEATURE_COLS


def build_aqi_targets(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Add the +horizon-hour US_AQI target per city, using the hourly-gridded
    (gap-free) DataFrame so shift(-horizon) means exactly `horizon` hours.
    Verifies the alignment on a sample of rows before returning.
    """
    df = df.copy()
    target_col = f"AQI_future{horizon}h"

    df[target_col] = df.groupby("City")[TARGET_COL].shift(-horizon)
    df["_target_dt"] = df.groupby("City")["Datetime"].shift(-horizon)

    valid = df[target_col].notna()
    if valid.any():
        sample = df.loc[valid].sample(min(1000, valid.sum()), random_state=42)
        actual_delta = sample["_target_dt"] - sample["Datetime"]
        expected = pd.Timedelta(hours=horizon)
        assert (actual_delta == expected).all(), (
            f"Shift alignment broken for horizon={horizon}h — "
            f"found deltas other than {expected}"
        )

    df = df.drop(columns=["_target_dt"])
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    return df


def train_aqi_forecaster(df: pd.DataFrame, horizon: int) -> dict:
    """
    Train XGBoost model to predict US_AQI `horizon` hours ahead.
    Returns trained model + metrics dict (model vs persistence vs rolling-24h baselines).
    """
    target_col = f"AQI_future{horizon}h"
    df_t = build_aqi_targets(df, horizon)
    df_t = df_t.dropna(subset=AQI_FORECAST_FEATURES + [target_col]).reset_index(drop=True)
    df_t = df_t.sort_values("Datetime").reset_index(drop=True)

    feat_cols = [f for f in AQI_FORECAST_FEATURES if f in df_t.columns]
    X = df_t[feat_cols].values
    y = df_t[target_col].values

    # Chronological split: train (fit) / val (early stopping) / test (report only)
    n = len(df_t)
    i_test = int(n * 0.80)
    i_val = int(i_test * 0.85)

    X_train, y_train = X[:i_val], y[:i_val]
    X_val, y_val = X[i_val:i_test], y[i_val:i_test]
    X_test, y_test = X[i_test:], y[i_test:]

    model = XGBRegressor(
        n_estimators=700,
        max_depth=6,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        early_stopping_rounds=40,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],      # ← val, NOT test
        verbose=False,
    )

    y_pred = np.clip(model.predict(X_test), 0, 500)
    metrics = regression_report(y_test, y_pred, label=f"AQI Forecaster +{horizon}h")
    metrics["best_iteration"] = int(model.best_iteration) if model.best_iteration else 700

    # ── Baselines on the same test slice ────────────────────────────────────
    aqi_now = df_t[TARGET_COL].values[i_test:]
    roll24 = df_t["AQI_roll24h_mean"].values[i_test:]
    metrics["baseline_persistence"] = regression_report(
        y_test, aqi_now, label=f"BASELINE persistence +{horizon}h")
    metrics["baseline_roll24h"] = regression_report(
        y_test, roll24, label=f"BASELINE roll24h +{horizon}h")

    # Training-set metrics to expose bias/variance in the dashboard
    y_train_pred = np.clip(model.predict(X_train), 0, 500)
    train_metrics = regression_report(y_train, y_train_pred,
                                       label=f"AQI Forecaster +{horizon}h [train]")

    path = MODELS_DIR / f"aqi_forecaster_h{horizon}.joblib"
    joblib.dump({"model": model, "feat_cols": feat_cols}, path)
    print(f"  Saved -> {path}  (stopped at iteration {metrics['best_iteration']} / 700)")

    return {
        "model": model,
        "feat_cols": feat_cols,
        "metrics": metrics,
        "train_metrics": train_metrics,
    }


# ─── Model 2: PHRS Predictor ──────────────────────────────────────────────────
# NOTE: out of scope for this task — left untouched.

PHRS_FEATURES = [
    "AQI", "AQI_norm_india",
    "PM2.5", "PM10", "NO2", "SO2", "CO", "O3",
    "age", "condition_enc", "activity_enc", "hours_outdoors",
    "month", "dayofweek", "city_enc",
    "season_Winter", "season_Monsoon", "season_Post_Monsoon", "season_Summer",
    "Temp_2m_C", "Humidity_Percent", "Wind_Speed_kmh", "Wind_Stagnation",
    "Temp_Inversion", "Festival_Period",
    "AQI_lag1", "AQI_lag3", "AQI_roll7_mean",
    "AQI_delta1", "AQI_delta3",
    "ref_health_score", "ref_resp_cases",
]


def train_phrs_model(phrs_df: pd.DataFrame) -> dict:
    """
    Train XGBoost model on the synthetic PHRS dataset.
    Returns trained model + metrics dict.
    """
    from sklearn.model_selection import train_test_split

    feat_cols = [f for f in PHRS_FEATURES if f in phrs_df.columns]
    X = phrs_df[feat_cols].values
    y = phrs_df["PHRS"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, shuffle=True
    )

    model = XGBRegressor(
        n_estimators=800,
        max_depth=7,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        reg_alpha=0.05,
        reg_lambda=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train, verbose=False)

    y_pred = np.clip(model.predict(X_test), 0, 100)
    metrics = regression_report(y_test, y_pred, label="PHRS Predictor")

    y_train_pred = np.clip(model.predict(X_train), 0, 100)
    train_metrics = regression_report(y_train, y_train_pred,
                                       label="PHRS Predictor [train]")

    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2", n_jobs=-1)
    print(f"  CV R² = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    path = MODELS_DIR / "phrs_model.joblib"
    joblib.dump({"model": model, "feat_cols": feat_cols}, path)
    print(f"  Saved -> {path}")

    return {
        "model": model,
        "feat_cols": feat_cols,
        "metrics": metrics,
        "train_metrics": train_metrics,
        "cv_r2": round(float(cv_scores.mean()), 4),
        "cv_r2_std": round(float(cv_scores.std()), 4),
    }


# ─── Inference helpers ─────────────────────────────────────────────────────────

def load_aqi_forecaster(horizon: int) -> dict:
    path = MODELS_DIR / f"aqi_forecaster_h{horizon}.joblib"
    return joblib.load(path)


def load_phrs_model() -> dict:
    return joblib.load(MODELS_DIR / "phrs_model.joblib")


def predict_aqi(row: dict, horizon: int) -> float:
    """Raw model prediction — no temporal smoothing applied."""
    bundle = load_aqi_forecaster(horizon)
    model = bundle["model"]
    feat_cols = bundle["feat_cols"]
    x = np.array([[row.get(f, 0) for f in feat_cols]])
    pred = float(np.clip(model.predict(x)[0], 0, 500))
    return round(pred, 1)


def predict_aqi_smooth(row: dict, current_aqi: float, horizon: int) -> float:
    """
    Raw model prediction, hard-clipped to a physically plausible range around
    current_aqi using the 95th-percentile |US_AQI(t+h) - US_AQI(t)| cap for
    this horizon (MAX_HOURLY_CHANGE). No momentum blending — a prior version
    blended 30% momentum extrapolation in, which amplified single-hour noise
    and hurt longer-horizon accuracy; the raw model + sanity clip performs
    better and is easier to reason about.

    Parameters
    ----------
    row         : feature dict for the current hour
    current_aqi : this hour's raw US_AQI (continuity anchor)
    horizon     : forecast horizon in hours (6, 12, 24, or 48)
    """
    raw_pred = predict_aqi(row, horizon)
    max_delta = MAX_HOURLY_CHANGE.get(horizon, 3.0 * horizon)

    bounded = np.clip(raw_pred,
                       current_aqi - max_delta,
                       current_aqi + max_delta)

    return round(float(np.clip(bounded, 0, 500)), 1)


def predict_phrs(row: dict) -> float:
    """Predict PHRS for a given feature row dict."""
    bundle = load_phrs_model()
    model = bundle["model"]
    feat_cols = bundle["feat_cols"]
    x = np.array([[row.get(f, 0) for f in feat_cols]])
    pred = float(np.clip(model.predict(x)[0], 0, 100))
    return round(pred, 2)


# ─── Feature importance ────────────────────────────────────────────────────────

def feature_importance(bundle: dict, top_n: int = 15) -> pd.DataFrame:
    model = bundle["model"]
    feat_cols = bundle["feat_cols"]
    return (
        pd.DataFrame({"feature": feat_cols, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )


def feature_importance_by_group(bundle: dict, groups: dict = FEATURE_GROUPS) -> pd.DataFrame:
    """Sum feature_importances_ within each named feature group (see
    preprocess.FEATURE_GROUPS) so relative contribution of e.g. weather vs
    pollutants vs AQI-history can be compared directly."""
    model = bundle["model"]
    feat_cols = bundle["feat_cols"]
    imp = dict(zip(feat_cols, model.feature_importances_))

    rows = []
    for group_name, cols in groups.items():
        total = sum(imp.get(c, 0.0) for c in cols if c in imp)
        rows.append({"group": group_name, "importance": total})

    out = pd.DataFrame(rows).sort_values("importance", ascending=False)
    out["importance_pct"] = 100 * out["importance"] / out["importance"].sum()
    return out
