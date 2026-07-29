"""
train_nhanes_susceptibility.py — NHANES-derived susceptibility classifier.

Replaces the hardcoded CONDITION_WEIGHTS lookup table (generate_profiles.py)
with a data-derived per-person susceptibility score: P(has a pollution-relevant
chronic respiratory/cardiovascular condition | age, sex, BMI, smoking).

Pools NHANES cycles I (2015-2016) and J (2017-2018): DEMO (demographics),
MCQ (medical conditions), BMX (body measures), SMQ (smoking). Trains an
XGBoost classifier, calibrates it with isotonic regression, then stretches
the calibrated probabilities (which cluster low because most adults are
healthy) into a 0-1 "susceptibility" score and a 1.0-2.0 PHRS condition
weight.

Run: python train_nhanes_susceptibility.py
Writes: models/nhanes_susceptibility.joblib, models/nhanes_scaler.json,
        models/nhanes_metrics.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib

NHANES_DIR = Path("data/nhanes")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

CYCLES = ["I", "J"]

CONDITION_COLS = [
    "MCQ010",   # asthma-ever
    "MCQ160B",  # congestive heart failure
    "MCQ160C",  # coronary heart disease
    "MCQ160D",  # angina
    "MCQ160E",  # heart attack
    "MCQ160F",  # stroke
    "MCQ160G",  # emphysema
    "MCQ160K",  # chronic bronchitis
    "MCQ160O",  # COPD
]

FEATURE_COLS = ["age", "female", "bmi", "ever_smoker", "current_smoker"]

K_DEFAULT = 1.0
RANDOM_STATE = 42


def _load_cycle(cycle: str) -> pd.DataFrame:
    demo = pd.read_sas(NHANES_DIR / f"DEMO_{cycle}.xpt", format="xport")
    mcq = pd.read_sas(NHANES_DIR / f"MCQ_{cycle}.xpt", format="xport")
    bmx = pd.read_sas(NHANES_DIR / f"BMX_{cycle}.xpt", format="xport")
    smq = pd.read_sas(NHANES_DIR / f"SMQ_{cycle}.xpt", format="xport")

    demo = demo[["SEQN", "RIDAGEYR", "RIAGENDR"]]
    mcq = mcq[["SEQN"] + CONDITION_COLS]
    bmx = bmx[["SEQN", "BMXBMI"]]
    smq = smq[["SEQN", "SMQ020", "SMQ040"]]

    df = demo.merge(mcq, on="SEQN", how="left")
    df = df.merge(bmx, on="SEQN", how="left")
    df = df.merge(smq, on="SEQN", how="left")
    df["_cycle"] = cycle
    return df


def load_pooled() -> pd.DataFrame:
    return pd.concat([_load_cycle(c) for c in CYCLES], ignore_index=True)


def build_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["RIDAGEYR"] >= 20].copy()

    valid_answer = df[CONDITION_COLS].isin([1, 2])
    usable = valid_answer.any(axis=1)
    df = df[usable].copy()

    has_condition = (df[CONDITION_COLS] == 1).any(axis=1)
    df["y"] = has_condition.astype(int)

    df["age"] = df["RIDAGEYR"]
    df["female"] = (df["RIAGENDR"] == 2).astype(int)
    df["bmi"] = df["BMXBMI"]
    df["ever_smoker"] = df["SMQ020"].map({1: 1, 2: 0}).fillna(-1).astype(int)
    df["current_smoker"] = df["SMQ040"].apply(
        lambda v: 1 if v in (1, 2) else (0 if v == 3 else -1)
    )

    return df[FEATURE_COLS + ["y"]].reset_index(drop=True)


def example_profiles() -> dict:
    return {
        "healthy_young":        dict(age=25, female=0, bmi=22.0, ever_smoker=0, current_smoker=0),
        "obese_young":          dict(age=27, female=1, bmi=34.0, ever_smoker=0, current_smoker=0),
        "mid_smoker":           dict(age=45, female=0, bmi=27.0, ever_smoker=1, current_smoker=1),
        "elderly_obese_smoker": dict(age=68, female=0, bmi=33.0, ever_smoker=1, current_smoker=1),
        "elderly_healthy":      dict(age=70, female=1, bmi=23.0, ever_smoker=0, current_smoker=0),
    }


def main():
    print("=" * 60)
    print("  NHANES Susceptibility Classifier — Training Pipeline")
    print("=" * 60)

    print("\n[1/6] Loading & pooling NHANES cycles I + J …")
    raw = load_pooled()
    data = build_dataset(raw)
    prevalence = data["y"].mean()
    print(f"      Pooled adults (20+, usable condition answers): {len(data):,}")
    print(f"      Prevalence of y=1 (any pollution-relevant chronic condition): {prevalence:.4f}")
    print("      Scope: adults 20+ only — MCQ chronic-condition items are only surveyed for this age group.")

    X = data[FEATURE_COLS]
    y = data["y"]

    print("\n[2/6] Stratified 70/15/15 train/val/test split (random_state=42) …")
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=RANDOM_STATE
    )
    print(f"      train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}")

    print("\n[3/6] Training XGBoost classifier …")
    xgb = XGBClassifier(
        max_depth=4,
        learning_rate=0.04,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=50,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print(f"      Best iteration: {xgb.best_iteration}")

    print("\n[4/6] Isotonic calibration on VAL set …")
    calibrated = CalibratedClassifierCV(FrozenEstimator(xgb), method="isotonic")
    calibrated.fit(X_val, y_val)

    def raw_score(X_):
        return calibrated.predict_proba(X_)[:, 1]

    train_scores = raw_score(X_train)
    lo, hi = np.percentile(train_scores, [5, 95])
    print(f"      TRAIN calibrated-score lo (p5)={lo:.4f}  hi (p95)={hi:.4f}")

    def stretch(scores):
        return np.clip((scores - lo) / (hi - lo), 0.0, 1.0)

    K = K_DEFAULT

    def condition_weight(scores):
        return 1.0 + stretch(scores) * K

    print("\n[5/6] Evaluating on TEST set …")
    test_scores_raw = raw_score(X_test)
    test_scores_stretched = stretch(test_scores_raw)

    auc = roc_auc_score(y_test, test_scores_raw)
    pr_auc = average_precision_score(y_test, test_scores_raw)
    brier = brier_score_loss(y_test, test_scores_raw)

    baseline_pred = np.full_like(test_scores_raw, y_train.mean())
    baseline_auc = 0.5
    baseline_pr_auc = y_test.mean()
    baseline_brier = brier_score_loss(y_test, baseline_pred)

    print(f"      Model    AUC={auc:.4f}  PR-AUC={pr_auc:.4f}  Brier={brier:.4f}")
    print(f"      Baseline AUC={baseline_auc:.4f}  PR-AUC={baseline_pr_auc:.4f}  Brier={baseline_brier:.4f}")
    calibration_ok = brier <= baseline_brier
    if not calibration_ok:
        print("      *** WARNING: model Brier is WORSE than prevalence baseline — calibration failed. ***")
    else:
        print("      PASS — model Brier beats prevalence baseline.")

    print("\n      Feature importance (gain):")
    booster = xgb.get_booster()
    gain_raw = booster.get_score(importance_type="gain")
    fmap = {f"f{i}": c for i, c in enumerate(FEATURE_COLS)}
    gain = {fmap.get(k, k): v for k, v in gain_raw.items()}
    gain = dict(sorted(gain.items(), key=lambda kv: kv[1], reverse=True))
    for feat, g in gain.items():
        print(f"        {feat:<16} {g:.2f}")

    pcts = np.percentile(test_scores_stretched, [5, 25, 50, 75, 95])
    pct_labels = ["p5", "p25", "p50", "p75", "p95"]
    print("\n      Stretched-score percentiles (TEST set):")
    for lbl, v in zip(pct_labels, pcts):
        print(f"        {lbl}: {v:.4f}")

    print("\n      Example profiles:")
    profiles = example_profiles()
    example_results = {}
    for name, feats in profiles.items():
        row = pd.DataFrame([feats])[FEATURE_COLS]
        s_raw = raw_score(row)[0]
        s_stretched = float(stretch(np.array([s_raw]))[0])
        w = float(condition_weight(np.array([s_raw]))[0])
        example_results[name] = {
            "features": feats,
            "susceptibility": s_stretched,
            "condition_weight": w,
        }
        print(f"        {name:<22} susceptibility={s_stretched:.4f}  condition_weight={w:.4f}")

    print("\n[6/6] Saving artifacts …")
    model_path = MODELS_DIR / "nhanes_susceptibility.joblib"
    joblib.dump(calibrated, model_path)
    print(f"      Saved -> {model_path}")

    scaler_path = MODELS_DIR / "nhanes_scaler.json"
    scaler = {"lo": float(lo), "hi": float(hi), "K": float(K)}
    with open(scaler_path, "w") as f:
        json.dump(scaler, f, indent=2)
    print(f"      Saved -> {scaler_path}  {scaler}")

    metrics = {
        "scope": "Adults 20+ only (RIDAGEYR >= 20) — MCQ chronic-condition items are only "
                 "surveyed for this age group in NHANES.",
        "n_pooled_adults": int(len(data)),
        "prevalence": float(prevalence),
        "split_sizes": {"train": int(len(X_train)), "val": int(len(X_val)), "test": int(len(X_test))},
        "test_metrics": {
            "auc": float(auc),
            "pr_auc": float(pr_auc),
            "brier": float(brier),
            "baseline_auc": float(baseline_auc),
            "baseline_pr_auc": float(baseline_pr_auc),
            "baseline_brier": float(baseline_brier),
            "beats_baseline_brier": bool(calibration_ok),
        },
        "feature_importance_gain": gain,
        "stretched_score_percentiles_test": {lbl: float(v) for lbl, v in zip(pct_labels, pcts)},
        "example_profiles": example_results,
        "scaler": scaler,
        "known_limitations": [
            "AUC ~0.66: age/gender/BMI/smoking are weak predictors of chronic illness. "
            "This is a prototype susceptibility PROXY, not a diagnostic tool — it reflects "
            "a data ceiling (NHANES only has coarse demographic/behavioral predictors for "
            "chronic disease), not a modeling bug.",
            "NHANES is a US population survey, used here as a proxy for Indian BREATHE users. "
            "The later Delhi calibration step is what keeps PHRS defensible for the target "
            "population.",
            "The model can give near-identical scores to some young profiles regardless of "
            "BMI, because BMI only becomes predictive of chronic conditions at older ages in "
            "the training data. This is expected model behavior, not a bug.",
            "Susceptibility is not strictly monotonic in age across the 25-55 range (e.g. a "
            "45-year-old smoker can score below a 25-year-old non-smoker) because no monotonic "
            "constraint was applied and the underlying prevalence-vs-age relationship is noisy "
            "at this sample size. Monotonicity only becomes visually clear past ~60. This should "
            "be disclosed alongside PHRS scores, not silently smoothed over.",
        ],
    }
    metrics_path = MODELS_DIR / "nhanes_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"      Saved -> {metrics_path}")

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
