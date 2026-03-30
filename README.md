# BioAQI — Personalized Air Quality Risk Platform

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Get the dataset
#    Download city_day.csv from Kaggle:
#    https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india
#    Place it in:  data/city_day.csv

# 3. Train models (takes ~5-10 min)
python train.py

# 4. Launch dashboard
streamlit run app.py
```

## File Overview

| File | Purpose |
|------|---------|
| `preprocess.py` | Load, clean, and feature-engineer city_day.csv |
| `generate_profiles.py` | Synthetic health profiles + PHRS formula |
| `models.py` | XGBoost AQI Forecaster + PHRS Predictor |
| `train.py` | End-to-end training script |
| `app.py` | Streamlit dashboard |

## PHRS Formula

```
PHRS = AQI_norm × condition_weight × activity_multiplier
       × exposure_factor × age_factor × pollutant_risk × trend_penalty
```

Mapped to: **Safe** (0–30) · **Moderate** (31–60) · **High** (61–80) · **Critical** (81–100)
