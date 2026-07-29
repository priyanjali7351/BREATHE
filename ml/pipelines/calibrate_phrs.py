"""
calibrate_phrs.py — PHRS prototype / internal-consistency calibration.

This is NOT calibration against real health outcomes in Delhi — we don't have
that data yet. It's a sanity harness: run a representative pollution scenario
through compute_phrs_horizons() for a handful of hand-picked profiles and
check that the RELATIVE ordering makes clinical sense (asthma raises risk,
elderly+smoker+obese is worse than young+clean, etc). Confirms the wiring
between the NHANES classifier and the PHRS calculator behaves sensibly before
any real outcome data is available to calibrate against.

Usage (from repo root):
    python ml/pipelines/calibrate_phrs.py
"""

import json
from pathlib import Path

from generate_profiles import (
    HealthProfile,
    compute_phrs_horizons,
    ASTHMA_BUMP,
    PHRS_HORIZONS,
)

MODELS_DIR = Path("models")

# Representative worsening-then-improving pollution scenario (India CPCB AQI).
SCENARIO_AQI = {"now": 180, "+6h": 220, "+24h": 150}

SCENARIO_POLLUTANTS = {
    "now":  {"PM2.5": 120, "PM10": 180, "NO2": 60, "SO2": 40, "CO": 1.5, "O3": 70, "NH3": 30},
    "+6h":  {"PM2.5": 160, "PM10": 230, "NO2": 75, "SO2": 55, "CO": 1.9, "O3": 90, "NH3": 40},
    "+24h": {"PM2.5": 100, "PM10": 150, "NO2": 50, "SO2": 35, "CO": 1.2, "O3": 60, "NH3": 25},
}

# (label, age, gender, bmi, ever_smoker, current_smoker, conditions)
PROFILES = [
    ("young_clean",                  28, "M", 22, 0, 0, ["Healthy"]),
    ("young_asthmatic",              28, "M", 22, 0, 0, ["Mild Asthma"]),
    ("mid_smoker",                   45, "M", 27, 1, 1, ["Healthy"]),
    ("elderly_obese_smoker",         68, "M", 33, 1, 1, ["Healthy"]),
    ("elderly_obese_smoker_asthma",  68, "M", 33, 1, 1, ["Mild Asthma"]),
    ("elderly_clean",                70, "F", 24, 0, 0, ["Healthy"]),
]


def _profile_result(age, gender, bmi, ever_smoker, current_smoker, conditions):
    profile = HealthProfile(age=age, condition=conditions[0], activity_level="Moderate",
                            hours_outdoors=2.0)
    return compute_phrs_horizons(
        SCENARIO_AQI, SCENARIO_POLLUTANTS, profile,
        gender=gender, bmi=bmi, ever_smoker=ever_smoker, current_smoker=current_smoker,
        conditions=conditions,
    )


def main():
    results = {}
    for label, age, gender, bmi, ever_smoker, current_smoker, conditions in PROFILES:
        results[label] = _profile_result(age, gender, bmi, ever_smoker, current_smoker, conditions)

    # ── Table: profile x horizon ────────────────────────────────────────────
    col_w = 16
    header = f"{'profile':<28}" + "".join(f"{h:>{col_w}}" for h in PHRS_HORIZONS) + f"{'cond_weight':>14}"
    print(header)
    print("-" * len(header))
    for label, res in results.items():
        row = f"{label:<28}"
        for h in PHRS_HORIZONS:
            hz = res["horizons"][h]
            row += f"{hz['phrs']:>7.2f} {hz['band'][:7]:<8}"
        row += f"{res['condition']['condition_weight']:>14.3f}"
        print(row)

    # ── Consistency checks ──────────────────────────────────────────────────
    print("\nConsistency checks:")

    cw_clean = results["young_clean"]["condition"]["condition_weight"]
    cw_asthma = results["young_asthmatic"]["condition"]["condition_weight"]
    asthma_ratio = cw_asthma / cw_clean
    print(f"  asthma effect (young): {cw_clean:.3f} -> {cw_asthma:.3f}  "
          f"(ratio {asthma_ratio:.3f}, expected ~{ASTHMA_BUMP})")
    assert results["young_asthmatic"]["condition"]["asthma_applied"] is True
    assert results["young_clean"]["condition"]["asthma_applied"] is False

    cw_elderly = results["elderly_obese_smoker"]["condition"]["condition_weight"]
    elderly_vs_young_ratio = cw_elderly / cw_clean
    print(f"  elderly_obese_smoker vs young_clean weight ratio: {elderly_vs_young_ratio:.3f} "
          f"(expected > 1.0)")
    assert cw_elderly > cw_clean, "elderly smoker should have higher condition weight than young clean"

    for h in PHRS_HORIZONS:
        assert (results["young_asthmatic"]["horizons"][h]["phrs"]
                >= results["young_clean"]["horizons"][h]["phrs"]), \
            f"asthma should not lower PHRS at horizon {h}"

    print("  all checks passed.")

    # ── Save calibration record ─────────────────────────────────────────────
    with open(MODELS_DIR / "nhanes_scaler.json") as f:
        K = json.load(f)["K"]

    record = {
        "note": (
            "PROTOTYPE internal-consistency calibration only — NOT calibrated "
            "against real (Delhi) health outcomes, which we don't have yet. "
            "This checks that the NHANES-classifier-derived condition weight "
            "and the declared-asthma bump produce clinically sensible relative "
            "ordering across profiles, not absolute accuracy."
        ),
        "scenario_aqi": SCENARIO_AQI,
        "asthma_bump": ASTHMA_BUMP,
        "K": K,
        "profiles": {
            label: {
                "condition": res["condition"],
                "horizons": res["horizons"],
            }
            for label, res in results.items()
        },
    }

    out_path = MODELS_DIR / "phrs_calibration.json"
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\nSaved calibration record to {out_path}")


if __name__ == "__main__":
    main()
