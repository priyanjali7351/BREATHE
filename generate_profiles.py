"""
BioAQI — Synthetic Health Profile Generator
Creates realistic user health profiles that are merged with AQI data
to train the PHRS prediction model.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict

from preprocess import normalize_aqi_india


# ─── Health sensitivity weights (domain knowledge) ────────────────────────────

CONDITION_WEIGHTS = {
    "Healthy":        1.00,
    "Mild Asthma":    1.40,
    "Severe Asthma":  1.70,
    "Heart Disease":  1.80,
    "Diabetes":       1.25,
    "Elderly (65+)":  1.35,
    "Child (<12)":    1.30,
}
_MAX_CONDITION_WEIGHT = max(CONDITION_WEIGHTS.values())  # 1.80

ACTIVITY_MULTIPLIERS = {
    "Sedentary":  1.00,
    "Moderate":   1.15,
    "Active":     1.35,
    "Athlete":    1.50,
}
_MAX_ACTIVITY_MULT = max(ACTIVITY_MULTIPLIERS.values())  # 1.50

# Pollutant sensitivity weights per condition.
# These reflect peer-reviewed clinical evidence on differential susceptibility:
#   - PM2.5 & O3 are primary triggers for asthma
#   - CO & PM2.5 are primary cardiovascular risk drivers
#   - Diabetics have moderate elevated sensitivity to most pollutants
POLLUTANT_SENSITIVITY = {
    "Healthy":        {"PM2.5": 1.0, "PM10": 0.8, "NO2": 0.6, "SO2": 0.5, "CO": 0.5, "O3": 0.7, "NH3": 0.4},
    "Mild Asthma":    {"PM2.5": 1.5, "PM10": 1.3, "NO2": 1.4, "SO2": 1.2, "CO": 0.8, "O3": 1.5, "NH3": 1.1},
    "Severe Asthma":  {"PM2.5": 1.9, "PM10": 1.7, "NO2": 1.8, "SO2": 1.6, "CO": 1.2, "O3": 1.9, "NH3": 1.4},
    "Heart Disease":  {"PM2.5": 1.8, "PM10": 1.5, "NO2": 1.6, "SO2": 1.7, "CO": 1.9, "O3": 1.4, "NH3": 1.0},
    "Diabetes":       {"PM2.5": 1.3, "PM10": 1.1, "NO2": 1.2, "SO2": 1.1, "CO": 1.0, "O3": 1.1, "NH3": 0.9},
    "Elderly (65+)":  {"PM2.5": 1.4, "PM10": 1.3, "NO2": 1.3, "SO2": 1.2, "CO": 1.1, "O3": 1.2, "NH3": 1.0},
    "Child (<12)":    {"PM2.5": 1.3, "PM10": 1.2, "NO2": 1.2, "SO2": 1.0, "CO": 0.9, "O3": 1.3, "NH3": 1.0},
}

# Safe concentration thresholds (µg/m³ or mg/m³ for CO) per India NAAQS / WHO guidelines
POLLUTANT_THRESHOLDS = {
    "PM2.5": 60,    # India 24h standard: 60 µg/m³
    "PM10":  100,   # India 24h standard: 100 µg/m³
    "NO2":   80,    # India 24h standard: 80 µg/m³
    "SO2":   80,    # India 24h standard: 80 µg/m³
    "CO":    2.0,   # India 8h standard: 2 mg/m³
    "O3":    100,   # India 8h standard: 100 µg/m³
    "NH3":   200,   # India 24h standard: 200 µg/m³
}

# PHRS component weights — must sum to 1.0
W_AQI      = 0.35   # Raw air quality level
W_POLLUTANT = 0.30  # Pollutant composition & exceedances
W_PROFILE  = 0.25   # Personal vulnerability (activity, exposure, age)
W_TREND    = 0.10   # Future AQI trend penalty


@dataclass
class HealthProfile:
    age: int
    condition: str
    activity_level: str
    hours_outdoors: float   # per day


def _aqi_component(aqi: float) -> float:
    """
    AQI sub-score (0–100) using India CPCB piecewise normalization.
    Pure air quality signal, before condition scaling.
    """
    return normalize_aqi_india(aqi) * 100.0


def _pollutant_component(pollutants: dict[str, float], condition: str) -> float:
    """
    Pollutant sub-score (0–100).

    For each pollutant, compute exceedance ratio = value / safe_threshold,
    clipped at 2× (so 2× threshold = fully exceeded).
    Weight by condition-specific sensitivity, then average.
    Scale so that an average exceedance of 1.0 (at threshold) → 50 points,
    and 2.0 (double threshold) → 100 points.
    """
    sens = POLLUTANT_SENSITIVITY.get(condition, POLLUTANT_SENSITIVITY["Healthy"])
    poll_risk = 0.0
    total_weight = 0.0
    for poll, raw_val in pollutants.items():
        if poll in sens and poll in POLLUTANT_THRESHOLDS:
            w        = sens[poll]
            exc      = np.clip(raw_val / POLLUTANT_THRESHOLDS[poll], 0.0, 2.0)
            poll_risk    += w * exc
            total_weight += w

    if total_weight == 0:
        return 0.0

    weighted_avg_exceedance = poll_risk / total_weight   # 0 – 2
    return float(np.clip(weighted_avg_exceedance * 50.0, 0.0, 100.0))


def _profile_component(
    profile: HealthProfile,
    temp_c: float | None = None,
    humidity: float | None = None,
) -> float:
    """
    Personal vulnerability sub-score (0–100), independent of condition.
    Three sub-contributors, each mapped to a 0–33.3 range:
      - Activity multiplier: how intensely are they breathing?
      - Exposure factor:     how many hours per day are they outside?
      - Age factor:          are they in a vulnerable age group?

    Optional weather modifiers:
      - Extreme heat (temp > 35°C) raises exposure slightly (higher O3 + exertion risk)
      - Cold + humid (temp < 10°C, humidity > 70%) raises exposure slightly (fog/PM trapping)
    """
    act_mult = ACTIVITY_MULTIPLIERS.get(profile.activity_level, 1.0)
    # Map [1.0, 1.5] → [0, 33.3]
    activity_contrib = (act_mult - 1.0) / (_MAX_ACTIVITY_MULT - 1.0) * 33.3

    # exposure factor: 1 + 0.06 * hours, capped at 10h
    exposure_factor = 1.0 + 0.06 * min(profile.hours_outdoors, 10.0)
    # Weather modifiers on exposure (capped at +0.1 total to avoid dominating)
    if temp_c is not None and temp_c > 35.0:
        exposure_factor = min(exposure_factor + 0.06, 1.7)   # heat stress
    if temp_c is not None and humidity is not None and temp_c < 10.0 and humidity > 70.0:
        exposure_factor = min(exposure_factor + 0.05, 1.7)   # fog/inversion trapping
    # Map [1.0, 1.6] → [0, 33.3]  (slight overflow beyond 1.6 still clipped at 100 below)
    exposure_contrib = (exposure_factor - 1.0) / 0.6 * 33.3

    # Age: children <12 and elderly >65 are more vulnerable
    if profile.age < 12:
        age_factor = 1.25
    elif profile.age > 65:
        age_factor = 1.20
    else:
        age_factor = 1.0
    # Map [1.0, 1.25] → [0, 33.3]
    age_contrib = (age_factor - 1.0) / 0.25 * 33.3

    return float(np.clip(activity_contrib + exposure_contrib + age_contrib, 0.0, 100.0))


def _trend_component(aqi: float, predicted_aqi: float | None) -> float:
    """
    Trend sub-score (−20 to +100).
    Rising AQI → positive (more risk); falling AQI → small negative (slight relief).
    A future increase of 200 AQI units = full 100-point trend score.
    A future decrease of 200 units caps the relief at −20 points.
    """
    if predicted_aqi is None:
        return 0.0
    delta = predicted_aqi - aqi
    if delta >= 0:
        return float(np.clip(delta / 200.0 * 100.0, 0.0, 100.0))
    else:
        # Falling AQI gives a small bonus (capped at −20 to avoid over-discounting)
        return float(np.clip(delta / 200.0 * 100.0, -20.0, 0.0))


def compute_phrs(
    aqi: float,
    pollutants: dict[str, float],
    profile: HealthProfile,
    predicted_aqi: float | None = None,
    temp_c: float | None = None,
    humidity: float | None = None,
) -> float:
    """
    Compute Personal Health Risk Score (0–100) using an additive weighted sum.

    Formula
    -------
    base_PHRS = W_AQI      * aqi_component(aqi)
              + W_POLLUTANT * pollutant_component(pollutants, condition)
              + W_PROFILE   * profile_component(profile)
              + W_TREND     * max(0, trend_component(aqi, predicted_aqi))

    The rising-trend bonus is included in base_PHRS.
    A falling-trend gives a small relief reduction applied after.

    The condition weight (1.0–1.80) scales the final score so that a Severe
    Asthmatic or Heart Disease patient at the same AQI as a Healthy person
    receives a meaningfully higher PHRS.

    PHRS = clip(base_PHRS * condition_weight, 0, 100)

    Parameters
    ----------
    aqi           : current raw AQI (India CPCB scale)
    pollutants    : dict of pollutant concentrations (µg/m³)
    profile       : user health profile
    predicted_aqi : optional forecasted AQI for trend penalty
    """
    aqi_comp  = _aqi_component(aqi)
    poll_comp = _pollutant_component(pollutants, profile.condition)
    prof_comp = _profile_component(profile, temp_c=temp_c, humidity=humidity)
    trend_val = _trend_component(aqi, predicted_aqi)

    # Rising trend adds to risk; falling trend gives partial relief
    trend_bonus    = max(0.0, trend_val)
    trend_relief   = min(0.0, trend_val)  # negative or zero

    base_phrs = (
        W_AQI       * aqi_comp
        + W_POLLUTANT * poll_comp
        + W_PROFILE   * prof_comp
        + W_TREND     * trend_bonus
    )

    # Apply falling-trend relief (capped at −20 points, scaled by W_TREND)
    base_phrs += W_TREND * trend_relief

    # Condition weight scales the full score (1.0–1.80)
    cond_weight = CONDITION_WEIGHTS.get(profile.condition, 1.0)
    phrs = np.clip(base_phrs * cond_weight, 0.0, 100.0)

    return round(float(phrs), 2)


def phrs_category(score: float) -> tuple[str, str]:
    """Return (risk label, hex colour)."""
    if score <= 30:
        return "Safe",          "#2ecc71"
    elif score <= 60:
        return "Moderate Risk", "#f39c12"
    elif score <= 80:
        return "High Risk",     "#e67e22"
    else:
        return "Critical",      "#e74c3c"


# ─── Synthetic dataset generation ─────────────────────────────────────────────

def _random_profile(rng: np.random.Generator) -> HealthProfile:
    condition = rng.choice(list(CONDITION_WEIGHTS.keys()),
                           p=[0.40, 0.15, 0.08, 0.10, 0.10, 0.10, 0.07])
    activity  = rng.choice(list(ACTIVITY_MULTIPLIERS.keys()),
                           p=[0.25, 0.40, 0.25, 0.10])
    age_ranges = {
        "Healthy":        (18, 65),
        "Mild Asthma":    (10, 70),
        "Severe Asthma":  (10, 70),
        "Heart Disease":  (45, 85),
        "Diabetes":       (35, 80),
        "Elderly (65+)":  (65, 90),
        "Child (<12)":    (3,  12),
    }
    lo, hi    = age_ranges.get(condition, (18, 65))
    age       = int(rng.integers(lo, hi + 1))
    hours_out = round(float(rng.uniform(0.5, 8.0)), 1)
    return HealthProfile(age=age, condition=condition,
                         activity_level=activity, hours_outdoors=hours_out)


def generate_phrs_dataset(
    aqi_df: pd.DataFrame,
    n_profiles_per_row: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """
    For each AQI record, generate n synthetic health profiles and compute PHRS.
    Returns a flat DataFrame ready for ML training.
    Now includes weather, season, event flags, and health reference features.
    """
    rng = np.random.default_rng(seed)
    pollutants_present = [p for p in POLLUTANT_THRESHOLDS if p in aqi_df.columns]

    # Extra context columns to carry through to the output
    context_cols = [
        "month", "dayofweek", "city_enc",
        "AQI_lag1", "AQI_lag3", "AQI_roll7_mean",
        "AQI_delta1", "AQI_delta3", "AQI_norm_india",
        "Temp_2m_C", "Humidity_Percent", "Wind_Speed_kmh", "Wind_Stagnation",
        "Temp_Inversion", "Festival_Period",
        "season_Winter", "season_Monsoon", "season_Post_Monsoon", "season_Summer",
        "ref_health_score", "ref_resp_cases",
    ]

    records = []
    for _, row in aqi_df.iterrows():
        poll_vals = {p: float(row.get(p, 0) or 0) for p in pollutants_present}

        # Use AQI_lag1 as the predicted next-day AQI proxy during training
        pred_aqi = float(row.get("AQI_lag1", row["AQI"]))

        # Pull weather values for the PHRS formula modifiers
        temp_c   = float(row["Temp_2m_C"])   if "Temp_2m_C"        in row.index else None
        humidity = float(row["Humidity_Percent"]) if "Humidity_Percent" in row.index else None

        for _ in range(n_profiles_per_row):
            profile = _random_profile(rng)
            phrs    = compute_phrs(
                row["AQI"], poll_vals, profile,
                predicted_aqi=pred_aqi,
                temp_c=temp_c,
                humidity=humidity,
            )
            rec = {
                "AQI":           row["AQI"],
                **poll_vals,
                **asdict(profile),
                "condition_enc": list(CONDITION_WEIGHTS.keys()).index(profile.condition),
                "activity_enc":  list(ACTIVITY_MULTIPLIERS.keys()).index(profile.activity_level),
                "PHRS":          phrs,
            }
            # Carry through all context features if present in the row
            for col in context_cols:
                if col in row.index:
                    rec[col] = row[col]
            records.append(rec)

    return pd.DataFrame(records)


if __name__ == "__main__":
    # Sanity-check the new formula
    cases = [
        ("Healthy, sedentary, AQI=50",
         dict(aqi=50, pollutants={"PM2.5": 15, "PM10": 30, "NO2": 20, "SO2": 10, "CO": 0.5, "O3": 40, "NH3": 20},
              profile=HealthProfile(age=25, condition="Healthy", activity_level="Sedentary", hours_outdoors=1),
              predicted_aqi=55, expected="Safe (~10-20)")),
        ("Mild Asthma, moderate, AQI=150",
         dict(aqi=150, pollutants={"PM2.5": 80, "PM10": 150, "NO2": 50, "SO2": 40, "CO": 1.2, "O3": 80, "NH3": 30},
              profile=HealthProfile(age=35, condition="Mild Asthma", activity_level="Moderate", hours_outdoors=3),
              predicted_aqi=170, expected="Moderate-High (~45-65)")),
        ("Severe Asthma, active, AQI=250",
         dict(aqi=250, pollutants={"PM2.5": 150, "PM10": 250, "NO2": 80, "SO2": 80, "CO": 2.0, "O3": 100, "NH3": 50},
              profile=HealthProfile(age=28, condition="Severe Asthma", activity_level="Active", hours_outdoors=4),
              predicted_aqi=280, expected="Critical (~80-95)")),
        ("Heart Disease, sedentary, AQI=300",
         dict(aqi=300, pollutants={"PM2.5": 200, "PM10": 300, "NO2": 100, "SO2": 100, "CO": 3.0, "O3": 120, "NH3": 80},
              profile=HealthProfile(age=60, condition="Heart Disease", activity_level="Sedentary", hours_outdoors=1),
              predicted_aqi=290, expected="High-Critical (~70-90)")),
    ]
    print("PHRS formula sanity checks:")
    for label, kwargs in cases:
        expected = kwargs.pop("expected")
        score = compute_phrs(**kwargs)
        cat, _ = phrs_category(score)
        print(f"  {label}")
        print(f"    PHRS = {score:.1f}  → {cat}  (expected {expected})")
