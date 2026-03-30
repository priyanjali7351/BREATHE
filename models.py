"""
BioAQI — ML Models
  Model 1: AQI Forecaster  — predicts AQI 1 / 3 / 7 days ahead
  Model 2: PHRS Predictor  — predicts personalized health risk score
Both use gradient-boosted trees (XGBoost).
"""

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# Hard caps on how much AQI can physically change per horizon.
# Derived from 95th-percentile of |AQI_delta| in the India city_day dataset.
# These prevent the model from predicting impossibly fast AQI swings.
MAX_DAILY_CHANGE = {1: 60, 3: 120, 7: 200}


# ─── Shared helpers ────────────────────────────────────────────────────────────

def regression_report(y_true, y_pred, label: str = "") -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = r2_score(y_true, y_pred)
    print(f"  [{label}] MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")
    return {"mae": round(mae, 3), "rmse": round(rmse, 3), "r2": round(r2, 4)}


# ─── Model 1: AQI Forecaster ──────────────────────────────────────────────────

AQI_FORECAST_FEATURES = [
    # Current pollutant readings
    "PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "NH3",
    # Calendar / seasonality
    "month", "dayofweek", "dayofyear", "quarter",
    # AQI lag values (memory of recent air quality)
    "AQI_lag1", "AQI_lag3", "AQI_lag7",
    # AQI rolling statistics (smoothed baseline + volatility)
    "AQI_roll3_mean", "AQI_roll7_mean",
    "AQI_roll3_std",  "AQI_roll7_std",
    # AQI velocity (first-order) and acceleration (second-order)
    "AQI_delta1", "AQI_delta3", "AQI_delta_trend",
    # India-normalised AQI (non-linear scale encodes CPCB bands)
    "AQI_norm_india",
    # Primary pollutant lags (PM2.5 and O3 drive most AQI variability)
    "PM2.5_lag1", "PM2.5_lag3", "PM2.5_roll7_mean",
    "NO2_lag1",   "NO2_lag3",
    "O3_lag1",    "O3_lag3",
]


def build_aqi_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add future AQI targets (+1, +3, +7 days) per city."""
    df = df.copy()
    for h in [1, 3, 7]:
        df[f"AQI_future{h}"] = df.groupby("City")["AQI"].shift(-h)
    df = df.dropna(subset=["AQI_future1", "AQI_future3", "AQI_future7"])
    return df


def train_aqi_forecaster(df: pd.DataFrame, horizon: int = 1) -> dict:
    """
    Train XGBoost model to predict AQI `horizon` days ahead.
    Returns trained model + metrics dict.
    """
    target_col = f"AQI_future{horizon}"
    df_t = build_aqi_targets(df)

    feat_cols = [f for f in AQI_FORECAST_FEATURES if f in df_t.columns]
    X = df_t[feat_cols].values
    y = df_t[target_col].values

    # Temporal split: last 15% is the test period (no shuffle preserves time order)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, shuffle=False
    )

    model = XGBRegressor(
        n_estimators=700,
        max_depth=6,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        early_stopping_rounds=40,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred = np.clip(model.predict(X_test), 0, 500)
    metrics = regression_report(y_test, y_pred, label=f"AQI Forecaster +{horizon}d")

    # Also compute training-set metrics to expose bias/variance in the dashboard
    y_train_pred = np.clip(model.predict(X_train), 0, 500)
    train_metrics = regression_report(y_train, y_train_pred,
                                      label=f"AQI Forecaster +{horizon}d [train]")

    path = MODELS_DIR / f"aqi_forecaster_h{horizon}.joblib"
    joblib.dump({"model": model, "feat_cols": feat_cols}, path)
    print(f"  Saved → {path}")

    return {
        "model": model,
        "feat_cols": feat_cols,
        "metrics": metrics,
        "train_metrics": train_metrics,
    }


# ─── Model 2: PHRS Predictor ──────────────────────────────────────────────────

PHRS_FEATURES = [
    "AQI",
    "PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "NH3",
    "age", "condition_enc", "activity_enc", "hours_outdoors",
    "month", "dayofweek",
    "AQI_lag1", "AQI_lag3", "AQI_roll7_mean",
    "AQI_delta1", "AQI_delta3", "AQI_delta_trend",
    "AQI_norm_india",
]


def train_phrs_model(phrs_df: pd.DataFrame) -> dict:
    """
    Train XGBoost model on the synthetic PHRS dataset.
    Returns trained model + metrics dict.
    """
    feat_cols = [f for f in PHRS_FEATURES if f in phrs_df.columns]
    X = phrs_df[feat_cols].values
    y = phrs_df["PHRS"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, shuffle=True
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

    # 5-fold cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2", n_jobs=-1)
    print(f"  CV R² = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    path = MODELS_DIR / "phrs_model.joblib"
    joblib.dump({"model": model, "feat_cols": feat_cols}, path)
    print(f"  Saved → {path}")

    return {
        "model": model,
        "feat_cols": feat_cols,
        "metrics": metrics,
        "train_metrics": train_metrics,
        "cv_r2": round(float(cv_scores.mean()), 4),
        "cv_r2_std": round(float(cv_scores.std()), 4),
    }


# ─── Inference helpers ─────────────────────────────────────────────────────────

def load_aqi_forecaster(horizon: int = 1) -> dict:
    path = MODELS_DIR / f"aqi_forecaster_h{horizon}.joblib"
    return joblib.load(path)


def load_phrs_model() -> dict:
    return joblib.load(MODELS_DIR / "phrs_model.joblib")


def predict_aqi(row: dict, horizon: int = 1) -> float:
    """Raw model prediction — no temporal smoothing applied."""
    bundle    = load_aqi_forecaster(horizon)
    model     = bundle["model"]
    feat_cols = bundle["feat_cols"]
    x    = np.array([[row.get(f, 0) for f in feat_cols]])
    pred = float(np.clip(model.predict(x)[0], 0, 500))
    return round(pred, 1)


def predict_aqi_smooth(row: dict, current_aqi: float, horizon: int = 1) -> float:
    """
    Temporally-constrained AQI prediction.

    AQI is a physical quantity governed by atmospheric dispersal — it cannot
    jump or drop by hundreds of units overnight. This function:

      1. Gets the raw model prediction.
      2. Blends it 70/30 with a simple momentum extrapolation
         (current + velocity * horizon), reducing outlier predictions.
      3. Hard-clips the result to a physically plausible range around
         current_aqi based on observed 95th-percentile daily AQI deltas
         in the India dataset:
           +1 day: ±60,  +3 days: ±120,  +7 days: ±200

    Parameters
    ----------
    row         : feature dict for the current day
    current_aqi : today's raw AQI (used as the continuity anchor)
    horizon     : forecast horizon in days (1, 3, or 7)
    """
    raw_pred  = predict_aqi(row, horizon)
    max_delta = MAX_DAILY_CHANGE.get(horizon, 60 * horizon)

    # Momentum extrapolation: current trend projected forward
    velocity  = float(row.get("AQI_delta1", 0))
    momentum  = current_aqi + velocity * horizon

    # 70% model confidence, 30% momentum — reduces cold-start overshoot
    blended   = 0.70 * raw_pred + 0.30 * momentum

    # Hard physical cap: AQI cannot change faster than observed extremes
    bounded   = np.clip(blended,
                        current_aqi - max_delta,
                        current_aqi + max_delta)

    return round(float(np.clip(bounded, 0, 500)), 1)


def predict_phrs(row: dict) -> float:
    """Predict PHRS for a given feature row dict."""
    bundle    = load_phrs_model()
    model     = bundle["model"]
    feat_cols = bundle["feat_cols"]
    x    = np.array([[row.get(f, 0) for f in feat_cols]])
    pred = float(np.clip(model.predict(x)[0], 0, 100))
    return round(pred, 2)


# ─── Feature importance ────────────────────────────────────────────────────────

def feature_importance(bundle: dict, top_n: int = 15) -> pd.DataFrame:
    model     = bundle["model"]
    feat_cols = bundle["feat_cols"]
    return (
        pd.DataFrame({"feature": feat_cols, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )
