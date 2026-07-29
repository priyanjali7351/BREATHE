# BREATHE — Personalized Air Quality Risk Platform

BREATHE does not report a generic AQI number. It reports a **Personal Health
Risk Score (PHRS)**: how risky the current/near-future air actually is *for
you*, given your age, activity level, exposure, and declared health
conditions — computed separately for **now**, **+6h**, and **+24h**.

## Two pipelines, combined at request time

1. **Hourly AQI Forecaster** (`ml/pipelines/train.py`, `preprocess.py`,
   `models.py`) — XGBoost models trained on
   `data/INDIA_AQI_COMPLETE_20251126.csv` (hourly, 29 Indian cities,
   2022–2025) that forecast AQI at multiple horizons
   (`models/aqi_forecaster_h*.joblib`).
2. **NHANES Susceptibility Classifier** (`ml/pipelines/train_nhanes_susceptibility.py`,
   `nhanes_scorer.py`) — an XGBoost classifier trained on pooled NHANES
   cycles I/J (demographics, BMI, smoking, chronic-condition labels) that
   predicts P(pollution-relevant chronic condition | age, sex, BMI, smoking),
   calibrated and stretched into a data-derived **condition weight**
   (`models/nhanes_susceptibility.joblib`, `models/nhanes_scaler.json`).

The **PHRS calculator** (`ml/pipelines/generate_profiles.py`) combines both:
for each horizon it takes the forecaster's AQI and scores it against the
classifier's condition weight (computed once per request, reused across
horizons — it doesn't depend on AQI).

## PHRS formula

```
base_PHRS = W_AQI(0.52) × aqi_component(aqi)
          + W_POLLUTANT(0.26) × pollutant_component(pollutants, conditions)
          + W_PROFILE(0.22) × profile_component(age, activity, exposure, weather)

condition_weight = nhanes_scorer.susceptibility_and_weight(age, gender, bmi,
                        ever_smoker, current_smoker)["condition_weight"]
if any declared condition contains "asthma":
    condition_weight *= ASTHMA_BUMP (1.15)
condition_weight = clip(condition_weight, 1.0, 2.0)

PHRS = clip(base_PHRS × condition_weight, 0, 100)
```

This runs **once per horizon** (now / +6h / +24h), each against its own
forecasted AQI — there is no single blended score with a "trend" nudge
anymore (the old `W_TREND` term is gone; `compute_phrs_horizons()` in
`generate_profiles.py` returns all three scores directly).

Bands: **Safe** 0–30 · **Moderate** 31–60 · **High** 61–80 · **Critical** 81–100.

### Why asthma is a declared-fact rule, not a learned weight

The NHANES classifier predicts susceptibility from body metrics (age, sex,
BMI, smoking) — it has no signal that would let it infer asthma specifically
from those inputs. Rather than pretend the model knows something it doesn't,
a user-declared "Asthma" condition applies a fixed 1.15× multiplicative bump
directly in the calculator (`condition_weight_for()` in
`generate_profiles.py`), on top of whatever the classifier already computed
from demographics.

## Repo layout

```
demo_api.py             FastAPI live demo backend — the UI (see below)
frontend/                static dashboard demo_api.py serves (index.html, app.js, style.css)
realtime.py              Open-Meteo city AQI/weather fetch, used by demo_api.py "auto" mode
gp2y10.py                GP2Y10 dust sensor reader, used by demo_api.py "sensor" mode
watch_hr.py              BLE smartwatch HR reader, used by demo_api.py for activity level
backend/api/             separate FastAPI service for the IoT sensor-mesh /predict endpoint
                         (main.py, database.py, idw.py, schemas.py) — no UI of its own
ml/pipelines/            preprocess.py, models.py, train.py       — AQI forecaster
                         train_nhanes_susceptibility.py,
                         nhanes_scorer.py                         — susceptibility classifier
                         generate_profiles.py                     — PHRS calculator
                         calibrate_phrs.py                        — PHRS prototype calibration
                         calibrate_weights.py                     — W_AQI/W_POLLUTANT calibration
models/                  trained artifacts (*.joblib, *.json) — gitignored
data/                    input datasets — gitignored
```

## Setup

```bash
pip install -r requirements.txt
```

Datasets to add (gitignored, not in the repo) — only needed to retrain the
two ML pipelines, not to run the live demo itself:
- `data/INDIA_AQI_COMPLETE_20251126.csv` — hourly AQI + weather, 2022–2025
- `data/nhanes/` — 8 NHANES cycle I/J .xpt files: `DEMO_I.xpt`, `DEMO_J.xpt`,
  `MCQ_I.xpt`, `MCQ_J.xpt`, `BMX_I.xpt`, `BMX_J.xpt`, `SMQ_I.xpt`, `SMQ_J.xpt`

Train:

```bash
python ml/pipelines/train.py                          # AQI forecaster (+ legacy PHRS model)
python ml/pipelines/train_nhanes_susceptibility.py     # susceptibility classifier
python ml/pipelines/calibrate_phrs.py                  # prototype PHRS consistency check
```

Run the live demo (UI):

```bash
uvicorn demo_api:app --reload
# open http://127.0.0.1:8000
```

`demo_api.py` fuses live city AQI (`realtime.py`, Open-Meteo) or a local
GP2Y10 dust sensor, a BLE smartwatch heart rate (mapped to activity level),
and `generate_profiles.compute_phrs()` into one JSON endpoint (`/api/live`)
that `frontend/app.js` polls once a second. Hardware is optional — a `sim`
mode lets you drive HR/PM by hand, and city "auto" mode works with nothing
plugged in. Note: this live demo is single-condition/single-horizon (age +
one condition, "now" only) — it does not yet expose the full
multi-condition, multi-horizon (`compute_phrs_horizons`) calculator or the
demographic (gender/BMI/smoking) inputs the NHANES classifier can use; those
default to the same conservative fallback `condition_weight_for()` uses
when demographics are unavailable.

Run the sensor-mesh predict API (separate service, no UI):

```bash
uvicorn backend.api.main:app --reload
```

## Honest limitations

- **Classifier ceiling, not a bug**: the NHANES susceptibility classifier
  tops out around **AUC ≈ 0.67** on held-out data (see
  `models/nhanes_metrics.json`). Chronic-condition risk from age/sex/BMI/
  smoking alone has a real information ceiling — this isn't a tuning
  problem, it's what five demographic features can predict.
- **Asthma blind spot**: the classifier cannot detect asthma from body
  metrics at all, hence the declared-fact bump described above rather than
  a learned weight.
- **US-population proxy**: the susceptibility model is trained on NHANES
  (a US population survey), not Indian health data, and is used as a proxy
  for relative individual susceptibility, not an absolute risk estimate for
  the Indian population.
- **Prototype calibration only**: `calibrate_phrs.py` checks internal
  consistency (asthma raises risk, elderly/smoker/obese ranks above
  young/healthy, etc.) across a synthetic scenario — it is **not** calibrated
  against real Delhi health outcomes, which we don't yet have.
