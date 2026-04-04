# BioAQI — Personalized Air Quality Risk Platform

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add datasets to data/
#    Required:
#      data/city_day.csv                        (Kaggle: rohanrao/air-quality-data-in-india)
#      data/INDIA_AQI_COMPLETE_20251126.csv      (hourly AQI + weather, 2022-2025)
#      data/air_quality_health_impact_data.csv   (health impact reference)

# 3. Train models (~10-20 min on first run — processes ~870K rows)
python train.py

# 4. Launch dashboard
streamlit run app.py
```

On first launch you will land on the **Login / Sign Up** page. Create an account, fill in your health profile, and you will be routed to your personal dashboard.

## File Overview

| File | Purpose |
|------|---------|
| `preprocess.py` | Load & merge 3 datasets, per-city seasonal cleaning, feature engineering, 80/20 train/test split with separate scaling pipelines |
| `generate_profiles.py` | Synthetic health profiles + PHRS formula (weather-aware) |
| `models.py` | XGBoost AQI Forecaster (×3 horizons) + PHRS Predictor |
| `train.py` | End-to-end training script |
| `app.py` | Streamlit dashboard — login/signup landing page + main dashboard |

## Datasets

| File | Rows | Coverage |
|------|------|----------|
| `city_day.csv` | 29 K | Daily AQI + pollutants, 26 cities, 2015–2020 |
| `INDIA_AQI_COMPLETE_20251126.csv` | 842 K | Hourly AQI + weather, 29 cities, 2022–2025 |
| `air_quality_health_impact_data.csv` | 5.8 K | Health impact reference (respiratory/CV cases per AQI bin) |

Only the 15 major cities present in both AQI datasets are used: Delhi, Mumbai, Bengaluru, Hyderabad, Chennai, Kolkata, Ahmedabad, Jaipur, Lucknow, Patna, Chandigarh, Gurugram, Guwahati, Visakhapatnam, Bhopal.

## Training CLI Options

```bash
python train.py \
  --city_day  data/city_day.csv \
  --india     data/INDIA_AQI_COMPLETE_20251126.csv \
  --health    data/air_quality_health_impact_data.csv \
  --profiles  5          # synthetic health profiles per AQI row
```

## PHRS Formula

```
PHRS = W_AQI(0.35) × aqi_component
     + W_POLLUTANT(0.30) × pollutant_exceedance × condition_sensitivity
     + W_PROFILE(0.25) × (activity + exposure + age + weather_modifiers)
     + W_TREND(0.10) × future_aqi_delta_penalty
     × condition_weight (1.0–1.80)
```

Mapped to: **Safe** (0–30) · **Moderate Risk** (31–60) · **High Risk** (61–80) · **Critical** (81–100)

Weather modifiers (new): extreme heat >35 °C raises exposure score; cold (<10 °C) + high humidity (>70%) raises it further to capture fog/inversion trapping.

## User Accounts

User profiles are stored locally in `data/users.json` (passwords SHA-256 hashed). Each profile records:
- Age, activity level, hours outdoors, city
- **Multiple health conditions** (checkboxes — all that apply)

PHRS is computed using the highest-risk selected condition. All fields are editable from the sidebar after login.
