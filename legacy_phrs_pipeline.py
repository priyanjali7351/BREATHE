"""
legacy_phrs_pipeline.py — PHRS-only data pipeline (city_day + INDIA_AQI + health impact).

The AQI Forecaster pipeline (preprocess.py) was rebuilt to be India-only and
hourly. The PHRS model is explicitly out of scope for that rewrite, and its
synthetic training data (generate_profiles.generate_phrs_dataset) depends on
a daily, 3-source feature set (AQI_lag1/lag3/roll7, ref_health_score,
ref_resp_cases, etc.) that the new hourly pipeline no longer produces.

This module is the old preprocess.py's dataset-merge logic, kept verbatim and
isolated here so PHRS training in train.py is completely unaffected by the
AQI Forecaster rewrite. Do not extend this module for forecaster use — it is
PHRS-only and duplicates data sources (city_day.csv, air_quality_health_impact_data.csv)
that the forecaster pipeline has intentionally dropped.
"""

import pandas as pd
import numpy as np

from preprocess import MAJOR_CITIES, CITY_ENC_MAP, normalize_aqi_india

POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]

NORTH_INDIA_CITIES = {
    "Delhi", "Lucknow", "Patna", "Chandigarh", "Gurugram",
    "Jaipur", "Ahmedabad", "Bhopal",
}

_AQI_BINS = [0, 50, 100, 200, 300, 400, 501]
_AQI_BIN_LABELS = ["Good", "Satisfactory", "Moderate", "Poor", "VeryPoor", "Severe"]

PHRS_CONTEXT_FEATURE_COLS = [
    "PM2.5", "PM10", "NO2", "SO2", "CO", "O3",
    "Temp_2m_C", "Humidity_Percent", "Wind_Speed_kmh", "Precipitation_mm", "Wind_Stagnation",
    "Temp_Inversion", "Festival_Period", "Crop_Burning_Season",
    "month", "dayofweek", "city_enc",
    "season_Winter", "season_Monsoon", "season_Post_Monsoon", "season_Summer",
    "AQI_lag1", "AQI_lag3", "AQI_lag7",
    "AQI_roll7_mean", "AQI_roll7_std",
    "AQI_delta1", "AQI_delta3",
    "AQI_norm_india",
    "PM2.5_lag1", "PM10_lag1", "NO2_lag1",
    "ref_health_score", "ref_resp_cases",
]


def _load_city_day(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df[df["City"].isin(MAJOR_CITIES)].copy()
    keep = ["City", "Date"] + POLLUTANTS + ["AQI"]
    df = df[[c for c in keep if c in df.columns]]
    df["_source"] = "city_day"
    return df.sort_values(["City", "Date"]).reset_index(drop=True)


def _load_india_complete(path: str) -> pd.DataFrame:
    needed_cols = [
        "City", "Datetime",
        "PM2_5_ugm3", "PM10_ugm3", "CO_ugm3", "NO2_ugm3", "SO2_ugm3", "O3_ugm3",
        "US_AQI",
        "Temp_2m_C", "Humidity_Percent", "Wind_Speed_10m_kmh",
        "Precipitation_mm", "Temp_Inversion", "Festival_Period",
        "Crop_Burning_Season", "Wind_Stagnation", "Season",
    ]
    df = pd.read_csv(path, usecols=needed_cols, parse_dates=["Datetime"], low_memory=False)
    df = df[df["City"].isin(MAJOR_CITIES)].copy()
    df = df.rename(columns={
        "PM2_5_ugm3": "PM2.5", "PM10_ugm3": "PM10", "CO_ugm3": "CO",
        "NO2_ugm3": "NO2", "SO2_ugm3": "SO2", "O3_ugm3": "O3",
        "US_AQI": "AQI", "Wind_Speed_10m_kmh": "Wind_Speed_kmh",
    })
    df["Date"] = df["Datetime"].dt.normalize()

    agg = {p: "mean" for p in POLLUTANTS}
    agg.update({
        "AQI": "mean", "Temp_2m_C": "mean", "Humidity_Percent": "mean",
        "Wind_Speed_kmh": "mean", "Precipitation_mm": "sum",
        "Temp_Inversion": "max", "Festival_Period": "max", "Crop_Burning_Season": "max",
        "Wind_Stagnation": "mean",
        "Season": lambda x: x.mode().iloc[0] if len(x) > 0 else np.nan,
    })
    agg = {k: v for k, v in agg.items() if k in df.columns}
    daily = df.groupby(["City", "Date"]).agg(agg).reset_index()
    daily["_source"] = "india_complete"
    return daily.sort_values(["City", "Date"]).reset_index(drop=True)


def _load_health_impact(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["AQI_bin"] = pd.cut(df["AQI"], bins=_AQI_BINS, labels=_AQI_BIN_LABELS, include_lowest=True)
    ref = (
        df.groupby("AQI_bin", observed=True)
        .agg(ref_health_score=("HealthImpactScore", "median"),
             ref_resp_cases=("RespiratoryCases", "median"))
        .reset_index()
    )
    ref["ref_health_score"] = ref["ref_health_score"] / ref["ref_health_score"].max()
    ref["ref_resp_cases"] = ref["ref_resp_cases"] / ref["ref_resp_cases"].max()
    return ref


def _merge_datasets(city_day, india, health_ref):
    weather_cols = [
        "Temp_2m_C", "Humidity_Percent", "Wind_Speed_kmh", "Precipitation_mm",
        "Temp_Inversion", "Festival_Period", "Crop_Burning_Season",
        "Wind_Stagnation", "Season",
    ]
    for col in weather_cols:
        if col not in city_day.columns:
            city_day = city_day.copy()
            city_day[col] = np.nan

    combined = pd.concat([city_day, india], ignore_index=True, sort=False)
    combined = combined.sort_values(["City", "Date"]).reset_index(drop=True)
    month = combined["Date"].dt.month

    def _month_to_season(m):
        if m in (6, 7, 8, 9):
            return "Monsoon"
        elif m in (10, 11):
            return "Post_Monsoon"
        elif m in (12, 1, 2):
            return "Winter"
        return "Summer"

    missing_season = combined["Season"].isna()
    combined.loc[missing_season, "Season"] = month[missing_season].map(_month_to_season)

    missing_fest = combined["Festival_Period"].isna()
    diwali_mask = (
        ((month == 10) & (combined["Date"].dt.day >= 20)) |
        ((month == 11) & (combined["Date"].dt.day <= 10))
    )
    combined.loc[missing_fest, "Festival_Period"] = diwali_mask[missing_fest].astype(int)
    combined["Festival_Period"] = combined["Festival_Period"].fillna(0).astype(int)

    missing_crop = combined["Crop_Burning_Season"].isna()
    crop_mask = month.isin([10, 11]) & combined["City"].isin(NORTH_INDIA_CITIES)
    combined.loc[missing_crop, "Crop_Burning_Season"] = crop_mask[missing_crop].astype(int)
    combined["Crop_Burning_Season"] = combined["Crop_Burning_Season"].fillna(0).astype(int)

    missing_inv = combined["Temp_Inversion"].isna()
    inversion_mask = month.isin([11, 12, 1, 2]) & combined["City"].isin(NORTH_INDIA_CITIES)
    combined.loc[missing_inv, "Temp_Inversion"] = inversion_mask[missing_inv].astype(int)
    combined["Temp_Inversion"] = combined["Temp_Inversion"].fillna(0).astype(int)

    india_rows = combined[combined["_source"] == "india_complete"].copy()
    india_rows["_month"] = india_rows["Date"].dt.month
    continuous_weather = ["Temp_2m_C", "Humidity_Percent", "Wind_Speed_kmh", "Precipitation_mm", "Wind_Stagnation"]
    combined["_month"] = combined["Date"].dt.month

    for wcol in continuous_weather:
        if wcol not in india_rows.columns:
            continue
        med_lookup = (
            india_rows.groupby(["City", "_month"])[wcol]
            .median().rename(f"_med_{wcol}").reset_index()
        )
        combined = combined.merge(med_lookup, on=["City", "_month"], how="left")
        mask = combined[wcol].isna()
        combined.loc[mask, wcol] = combined.loc[mask, f"_med_{wcol}"]
        combined[wcol] = combined[wcol].fillna(combined[wcol].median())
        combined = combined.drop(columns=[f"_med_{wcol}"])

    combined = combined.drop(columns=["_month"], errors="ignore")

    combined["AQI_bin"] = pd.cut(combined["AQI"], bins=_AQI_BINS, labels=_AQI_BIN_LABELS, include_lowest=True)
    combined = combined.merge(health_ref, on="AQI_bin", how="left")
    combined["ref_health_score"] = combined["ref_health_score"].fillna(0.5)
    combined["ref_resp_cases"] = combined["ref_resp_cases"].fillna(0.3)
    combined = combined.drop(columns=["AQI_bin"])
    return combined


def _impute_per_city(df):
    df = df.copy()
    impute_cols = POLLUTANTS + ["Temp_2m_C", "Humidity_Percent", "Wind_Speed_kmh", "Precipitation_mm"]
    for col in impute_cols:
        if col not in df.columns:
            continue
        df[col] = df.groupby("City")[col].transform(lambda s: s.ffill().bfill())
        cs_median = df.groupby(["City", "Season"])[col].transform("median")
        df[col] = df[col].fillna(cs_median)
        c_median = df.groupby("City")[col].transform("median")
        df[col] = df[col].fillna(c_median).fillna(df[col].median())
    return df


def _treat_outliers(df):
    df = df.copy()
    treat_cols = POLLUTANTS + ["AQI"]
    event_col_mask = (
        (df["Festival_Period"] == 1) | (df["Crop_Burning_Season"] == 1) | (df["Temp_Inversion"] == 1)
    )
    for city in df["City"].unique():
        city_mask = df["City"] == city
        for season in df["Season"].unique():
            group_mask = city_mask & (df["Season"] == season)
            if group_mask.sum() < 10:
                continue
            is_event = group_mask & event_col_mask
            is_normal = group_mask & ~event_col_mask
            for col in treat_cols:
                if col not in df.columns:
                    continue
                vals = df.loc[group_mask, col]
                q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
                iqr = q3 - q1
                if iqr == 0:
                    continue
                lower = max(0.0, q1 - 1.5 * iqr)
                upper_normal = q3 + 3.0 * iqr
                upper_event = q3 + 5.0 * iqr
                if is_normal.any():
                    df.loc[is_normal, col] = df.loc[is_normal, col].clip(lower, upper_normal)
                if is_event.any():
                    df.loc[is_event, col] = df.loc[is_event, col].clip(lower, upper_event)
    return df


def _engineer_features(df):
    df = df.copy()
    df["month"] = df["Date"].dt.month
    df["dayofweek"] = df["Date"].dt.dayofweek
    df["city_enc"] = df["City"].map(CITY_ENC_MAP)

    df["Season"] = df["Season"].str.strip()
    s_dummies = pd.get_dummies(df["Season"], prefix="season")
    for col in ["season_Winter", "season_Monsoon", "season_Post_Monsoon", "season_Summer"]:
        if col not in s_dummies.columns:
            s_dummies[col] = 0
    s_dummies = s_dummies[["season_Winter", "season_Monsoon", "season_Post_Monsoon", "season_Summer"]]
    df = pd.concat([df, s_dummies], axis=1)

    for lag in [1, 3, 7]:
        df[f"AQI_lag{lag}"] = df.groupby("City")["AQI"].shift(lag)

    df["AQI_roll7_mean"] = df.groupby("City")["AQI"].transform(lambda s: s.shift(1).rolling(7, min_periods=3).mean())
    df["AQI_roll7_std"] = df.groupby("City")["AQI"].transform(lambda s: s.shift(1).rolling(7, min_periods=3).std().fillna(0))
    df["AQI_delta1"] = df.groupby("City")["AQI"].diff(1)
    df["AQI_delta3"] = df.groupby("City")["AQI"].diff(3)
    df["AQI_norm_india"] = df["AQI"].apply(normalize_aqi_india)

    df["PM2.5_lag1"] = df.groupby("City")["PM2.5"].shift(1)
    df["PM10_lag1"] = df.groupby("City")["PM10"].shift(1)
    df["NO2_lag1"] = df.groupby("City")["NO2"].shift(1)

    df["Wind_Stagnation"] = df["Wind_Stagnation"].clip(0.0, 1.0).fillna(0.0)

    lag_check = ["AQI_lag1", "AQI_lag3", "AQI_lag7", "PM2.5_lag1", "PM10_lag1", "NO2_lag1"]
    df = df.dropna(subset=lag_check).reset_index(drop=True)
    return df


def build_phrs_training_df(
    city_day_path: str = "data/city_day.csv",
    india_path: str = "data/INDIA_AQI_COMPLETE_20251126.csv",
    health_path: str = "data/air_quality_health_impact_data.csv",
    test_size: float = 0.20,
) -> dict:
    """Old 3-source daily merge, used only to build PHRS synthetic training rows."""
    df_city = _load_city_day(city_day_path)
    df_india = _load_india_complete(india_path)
    health_ref = _load_health_impact(health_path)

    df = _merge_datasets(df_city, df_india, health_ref)
    df = df.dropna(subset=["AQI"]).copy()
    df = _impute_per_city(df)
    df = _treat_outliers(df)
    df = _engineer_features(df)

    feat_cols = [c for c in PHRS_CONTEXT_FEATURE_COLS if c in df.columns]
    meta = ["City", "Date", "AQI", "Season", "_source"]
    df = df[[c for c in meta + feat_cols if c in df.columns]].copy()
    df = df.sort_values("Date").reset_index(drop=True)

    split_idx = int(len(df) * (1.0 - test_size))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    return {
        "train": {"raw": train_df},
        "test": {"raw": test_df},
        "full_df": df,
        "feature_cols": feat_cols,
    }
