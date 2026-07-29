"""
nhanes_scorer.py — Serving-time wrapper around the NHANES susceptibility model.

Loads the calibrated classifier (models/nhanes_susceptibility.joblib) and the
lo/hi/K stretch parameters (models/nhanes_scaler.json) produced by
train_nhanes_susceptibility.py, and exposes a single scoring function that
generate_profiles.aggregate_condition_weight() (or its replacement) can call
in place of the hardcoded CONDITION_WEIGHTS lookup.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

FEATURE_COLS = ["age", "female", "bmi", "ever_smoker", "current_smoker"]

_model = None
_scaler = None


def _load():
    global _model, _scaler
    if _model is None:
        _model = joblib.load(MODELS_DIR / "nhanes_susceptibility.joblib")
        with open(MODELS_DIR / "nhanes_scaler.json") as f:
            _scaler = json.load(f)
    return _model, _scaler


def _normalize_gender(gender) -> int:
    """1 -> female, 0 -> male. Accepts RIAGENDR code (1=male,2=female),
    a bool ("is_female"), or a string ("male"/"female"/"m"/"f")."""
    if isinstance(gender, str):
        return 1 if gender.strip().lower().startswith("f") else 0
    if isinstance(gender, bool):
        return int(gender)
    return 1 if int(gender) == 2 else 0


def _normalize_tri(value) -> int:
    """Maps to the training encoding: 1=yes, 0=no, -1=unknown."""
    if value is None:
        return -1
    if isinstance(value, bool):
        return int(value)
    v = int(value)
    return v if v in (1, 0) else -1


def susceptibility_and_weight(age, gender, bmi, ever_smoker, current_smoker) -> dict:
    """
    Score a single person and return their data-derived susceptibility and
    the PHRS condition weight it maps to.

    age:             years
    gender:          RIAGENDR code (1=male,2=female), bool is_female, or str
    bmi:             kg/m^2, or None/NaN if unknown (model handles missing)
    ever_smoker:     1/True=yes, 0/False=no, None=unknown
    current_smoker:  1/True=yes, 0/False=no, None=unknown

    Returns {"susceptibility": float in [0,1], "condition_weight": float in [1.0, 1.0+K]}
    """
    model, scaler = _load()

    row = pd.DataFrame([{
        "age": age,
        "female": _normalize_gender(gender),
        "bmi": np.nan if bmi is None else float(bmi),
        "ever_smoker": _normalize_tri(ever_smoker),
        "current_smoker": _normalize_tri(current_smoker),
    }])[FEATURE_COLS]

    raw_score = float(model.predict_proba(row)[0, 1])

    lo, hi, K = scaler["lo"], scaler["hi"], scaler["K"]
    stretched = float(np.clip((raw_score - lo) / (hi - lo), 0.0, 1.0))
    weight = 1.0 + stretched * K

    return {"susceptibility": stretched, "condition_weight": weight}
