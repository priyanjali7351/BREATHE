"""
BioAQI — Streamlit Dashboard
Run: streamlit run app.py
"""

import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import joblib

from generate_profiles import (
    HealthProfile, compute_phrs, phrs_category,
    CONDITION_WEIGHTS, ACTIVITY_MULTIPLIERS,
)
from models import MAX_DAILY_CHANGE

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BioAQI — Personal Air Quality Risk",
    page_icon="🌬️",
    layout="wide",
)

# ─── Helper functions ─────────────────────────────────────────────────────────

def _aqi_bucket(aqi: float) -> str:
    if aqi <= 50:   return "Good"
    if aqi <= 100:  return "Satisfactory"
    if aqi <= 200:  return "Moderate"
    if aqi <= 300:  return "Poor"
    if aqi <= 400:  return "Very Poor"
    return "Severe"


def _risk_color_for_aqi(aqi: float) -> str:
    if aqi <= 50:   return "#2ecc71"
    if aqi <= 100:  return "#a8d86e"
    if aqi <= 200:  return "#f39c12"
    if aqi <= 300:  return "#e67e22"
    return "#e74c3c"


def get_recommendations(phrs: float, profile: HealthProfile,
                         current_aqi: float, pred_aqi: float | None) -> list[str]:
    recs = []
    if phrs <= 30:
        recs.append("✅ Air quality is safe for you today. Normal activities are fine.")
    elif phrs <= 60:
        recs.append("⚠️ Moderate risk. Limit prolonged strenuous outdoor activity.")
        if profile.activity_level in ("Active", "Athlete"):
            recs.append("⚠️ Consider moving workouts indoors or to early morning when AQI is lower.")
    elif phrs <= 80:
        recs.append("🔴 High risk. Avoid outdoor activity — especially exercise.")
        recs.append("🏠 Stay indoors with windows closed. Use an air purifier if available.")
        if profile.condition in ("Mild Asthma", "Severe Asthma"):
            recs.append("💊 Keep your rescue inhaler accessible at all times.")
        if profile.condition == "Heart Disease":
            recs.append("❤️ Monitor for chest tightness or shortness of breath.")
    else:
        recs.append("🚨 Critical risk. Stay indoors. Avoid all outdoor exposure.")
        recs.append("😷 If going outside is unavoidable, wear an N95 mask.")
        recs.append("💊 Follow your doctor's emergency action plan.")

    if pred_aqi is not None and pred_aqi > current_aqi + 50:
        recs.append(f"📈 AQI is forecast to rise to ~{pred_aqi:.0f} tomorrow. Plan accordingly.")

    if profile.age < 12 or profile.age > 65:
        recs.append("👶/👴 Extra caution is recommended for your age group.")

    if profile.hours_outdoors >= 6 and phrs > 40:
        recs.append(
            "⏱️ Reduce time outdoors — your high daily exposure significantly increases your risk."
        )
    return recs


# ─── Load models & data ────────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    bundle = {}
    for h in [1, 3, 7]:
        p = Path(f"models/aqi_forecaster_h{h}.joblib")
        if p.exists():
            bundle[f"aqi_h{h}"] = joblib.load(p)
    p = Path("models/phrs_model.joblib")
    if p.exists():
        bundle["phrs"] = joblib.load(p)
    return bundle


@st.cache_data
def load_metrics() -> dict | None:
    p = Path("models/metrics.json")
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


@st.cache_data
def load_aqi_data():
    p = Path("data/city_day.csv")
    if not p.exists():
        return None
    from preprocess import build_training_df
    df, _ = build_training_df(str(p))
    return df


MODELS       = load_models()
METRICS      = load_metrics()
AQI_DF       = load_aqi_data()
MODELS_READY = bool(MODELS)

# ─── Sidebar — Health Profile ──────────────────────────────────────────────────

with st.sidebar:
    st.title("🧬 Your Health Profile")
    st.markdown("---")

    age       = st.slider("Age", 5, 90, 25)
    condition = st.selectbox("Medical Condition", list(CONDITION_WEIGHTS.keys()))
    activity  = st.selectbox("Activity Level", list(ACTIVITY_MULTIPLIERS.keys()))
    hours_out = st.slider("Hours Outdoors / Day", 0.0, 12.0, 2.0, 0.5)

    st.markdown("---")
    st.markdown("**Manual AQI Override**")
    manual_aqi = st.number_input("Current AQI (leave 0 to use city data)",
                                  min_value=0, max_value=500, value=0)

    if AQI_DF is not None:
        cities        = sorted(AQI_DF["City"].unique())
        selected_city = st.selectbox("City", cities)
    else:
        selected_city = None

    st.markdown("---")
    st.caption("BioAQI v1.1 · Personalized Air Quality Risk")


profile = HealthProfile(
    age=age,
    condition=condition,
    activity_level=activity,
    hours_outdoors=hours_out,
)

# ─── Main layout ──────────────────────────────────────────────────────────────

st.title("🌬️ BioAQI — Personal Health Risk Score")
st.markdown(
    "Standard AQI treats everyone equally. **BioAQI** computes *your* personal risk "
    "based on your health profile, activity, and predicted air quality trends."
)

if not MODELS_READY:
    st.warning("⚠️ Models not found. Run `python train.py` first to train all models.")

# ── Get latest AQI row for selected city ──────────────────────────────────────

def get_latest_row(city: str) -> pd.Series | None:
    if AQI_DF is None or city not in AQI_DF["City"].values:
        return None
    city_df = AQI_DF[AQI_DF["City"] == city].sort_values("Date")
    return city_df.iloc[-1]


def predict_future_aqi_smooth(row: pd.Series, current_aqi: float,
                               horizon: int) -> float | None:
    """
    Temporally-constrained AQI forecast.
    Blends the XGBoost prediction (70%) with a momentum extrapolation (30%),
    then hard-clips to a physically plausible range around current_aqi.
    """
    key = f"aqi_h{horizon}"
    if key not in MODELS:
        return None
    bundle    = MODELS[key]
    feat_cols = bundle["feat_cols"]
    x         = np.array([[row.get(f, 0) for f in feat_cols]])
    raw_pred  = float(np.clip(bundle["model"].predict(x)[0], 0, 500))

    max_delta = MAX_DAILY_CHANGE.get(horizon, 60 * horizon)
    velocity  = float(row.get("AQI_delta1", 0))
    momentum  = current_aqi + velocity * horizon
    blended   = 0.70 * raw_pred + 0.30 * momentum
    bounded   = np.clip(blended, current_aqi - max_delta, current_aqi + max_delta)
    return round(float(np.clip(bounded, 0, 500)), 1)


latest_row = get_latest_row(selected_city) if selected_city else None

if manual_aqi > 0:
    current_aqi = float(manual_aqi)
    pollutants  = {}
    pred_aqi    = None
elif latest_row is not None:
    current_aqi = float(latest_row["AQI"])
    pollutants  = {
        p: float(latest_row.get(p, 0) or 0)
        for p in ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "NH3"]
        if p in latest_row.index
    }
    pred_aqi = (predict_future_aqi_smooth(latest_row, current_aqi, 1)
                if MODELS_READY else None)
else:
    current_aqi = 0.0
    pollutants  = {}
    pred_aqi    = None

# ── PHRS Computation ──────────────────────────────────────────────────────────

phrs_score            = compute_phrs(current_aqi, pollutants, profile,
                                     predicted_aqi=pred_aqi)
risk_label, risk_color = phrs_category(phrs_score)

# ─── Row 1: KPI cards ─────────────────────────────────────────────────────────

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Current AQI", f"{current_aqi:.0f}")
    st.caption(_aqi_bucket(current_aqi))

with c2:
    st.metric("Your PHRS", f"{phrs_score:.1f} / 100")
    st.markdown(
        f"<span style='color:{risk_color}; font-weight:bold; font-size:1.1em'>"
        f"{risk_label}</span>",
        unsafe_allow_html=True,
    )

with c3:
    if pred_aqi is not None:
        delta_val = round(pred_aqi - current_aqi, 1)
        delta_str = f"+{delta_val}" if delta_val > 0 else str(delta_val)
        st.metric("Predicted AQI (+1 day)", f"{pred_aqi:.0f}",
                  delta=delta_str, delta_color="inverse")
    else:
        st.metric("Predicted AQI (+1 day)", "—")
        st.caption("Train models to see forecast")

with c4:
    cw = CONDITION_WEIGHTS.get(condition, 1.0)
    am = ACTIVITY_MULTIPLIERS.get(activity, 1.0)
    st.metric("Sensitivity Multiplier", f"{cw * am:.2f}×")
    st.caption(f"Condition {cw}× · Activity {am}×")

st.markdown("---")

# ─── Row 2: PHRS gauge + AQI forecast chart ──────────────────────────────────

col_gauge, col_forecast = st.columns([1, 2])

with col_gauge:
    st.subheader("Personal Risk Gauge")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=phrs_score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": risk_label, "font": {"color": risk_color, "size": 18}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar":  {"color": risk_color},
            "steps": [
                {"range": [0,  30], "color": "#2ecc71"},
                {"range": [30, 60], "color": "#f39c12"},
                {"range": [60, 80], "color": "#e67e22"},
                {"range": [80,100], "color": "#e74c3c"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.75,
                "value": phrs_score,
            },
        },
    ))
    fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_forecast:
    st.subheader("AQI Forecast (Temporally Constrained)")
    if latest_row is not None and MODELS_READY:
        aqi_vals = [current_aqi]
        for h in [1, 3, 7]:
            v = predict_future_aqi_smooth(latest_row, current_aqi, h)
            aqi_vals.append(v if v is not None else current_aqi)
        labels = ["Now", "+1 day", "+3 days", "+7 days"]

        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(
            x=labels, y=aqi_vals,
            mode="lines+markers",
            marker=dict(size=10, color=[_risk_color_for_aqi(v) for v in aqi_vals]),
            line=dict(color="#3498db", width=2),
            name="AQI",
        ))
        for threshold, label, color in [
            (50,  "Good",     "#2ecc71"),
            (100, "Moderate", "#f39c12"),
            (200, "Poor",     "#e67e22"),
        ]:
            fig_fc.add_hline(y=threshold, line_dash="dot", line_color=color,
                             annotation_text=label, annotation_position="right")
        fig_fc.update_layout(
            height=280, xaxis_title="Horizon", yaxis_title="AQI",
            margin=dict(l=20, r=70, t=30, b=20),
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font=dict(color="white"),
        )
        st.plotly_chart(fig_fc, use_container_width=True)
    else:
        st.info("Select a city and train models (`python train.py`) to see the AQI forecast.")

st.markdown("---")

# ─── Row 3: Pollutant breakdown ───────────────────────────────────────────────

st.subheader("Pollutant Breakdown")

if pollutants:
    thresholds = {"PM2.5": 60, "PM10": 100, "NO2": 80,
                  "SO2": 80, "CO": 2, "O3": 100, "NH3": 200}
    poll_df = pd.DataFrame([
        {
            "Pollutant":       p,
            "Value":           round(v, 2),
            "Safe Threshold":  thresholds.get(p, 100),
            "% of Threshold":  round(v / thresholds.get(p, 100) * 100, 1),
        }
        for p, v in pollutants.items() if v > 0
    ])
    poll_df["Status"] = poll_df["% of Threshold"].apply(
        lambda x: "Safe" if x <= 60 else ("Moderate" if x <= 100 else "Exceeded")
    )
    fig_poll = px.bar(
        poll_df, x="Pollutant", y="Value",
        color="Status",
        color_discrete_map={"Safe": "#2ecc71", "Moderate": "#f39c12", "Exceeded": "#e74c3c"},
        text="Value", height=300,
    )
    fig_poll.update_layout(
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="white"), margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig_poll, use_container_width=True)
    with st.expander("Pollutant Data Table"):
        st.dataframe(poll_df, use_container_width=True)
else:
    st.info("No pollutant data available. Select a city from the sidebar.")

st.markdown("---")

# ─── Row 4: Sensitivity comparison ───────────────────────────────────────────

st.subheader("How Different Profiles Compare (Same AQI)")

if current_aqi > 0:
    comparison_rows = []
    for cond in CONDITION_WEIGHTS:
        for act in ["Sedentary", "Active"]:
            p = HealthProfile(age=30, condition=cond, activity_level=act, hours_outdoors=3)
            s = compute_phrs(current_aqi, pollutants, p)
            comparison_rows.append({"Condition": cond, "Activity": act, "PHRS": s})
    comp_df = pd.DataFrame(comparison_rows)
    fig_comp = px.bar(
        comp_df, x="Condition", y="PHRS", color="Activity", barmode="group",
        color_discrete_map={"Sedentary": "#3498db", "Active": "#e74c3c"},
        height=320,
    )
    fig_comp.add_hline(y=phrs_score, line_dash="dash", line_color="yellow",
                       annotation_text=f"Your Score: {phrs_score:.1f}",
                       annotation_position="top right")
    fig_comp.update_layout(
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="white"), margin=dict(t=20),
        xaxis_tickangle=-30,
    )
    st.plotly_chart(fig_comp, use_container_width=True)

st.markdown("---")

# ─── Row 5: Historical AQI trend ─────────────────────────────────────────────

if AQI_DF is not None and selected_city:
    st.subheader(f"Historical AQI — {selected_city} (last 90 days)")
    city_hist = (
        AQI_DF[AQI_DF["City"] == selected_city]
        .sort_values("Date")
        .tail(90)
    )
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=city_hist["Date"], y=city_hist["AQI"],
        fill="tozeroy", mode="lines",
        line=dict(color="#3498db"),
        name="AQI",
    ))
    for level, color, label in [
        (50,  "#2ecc71", "Good"),
        (100, "#f39c12", "Moderate"),
        (200, "#e67e22", "Poor"),
        (300, "#e74c3c", "V.Poor"),
    ]:
        fig_hist.add_hline(y=level, line_dash="dot", line_color=color,
                           annotation_text=label, annotation_position="right")
    fig_hist.update_layout(
        height=300, xaxis_title="Date", yaxis_title="AQI",
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="white"), margin=dict(t=10, r=70),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("---")

# ─── Row 6: Recommendations ───────────────────────────────────────────────────

st.subheader("Personalized Recommendations")
recs = get_recommendations(phrs_score, profile, current_aqi, pred_aqi)
for rec in recs:
    st.markdown(f"- {rec}")

st.markdown("---")

# ─── Row 7: Model Performance ─────────────────────────────────────────────────

st.subheader("📊 Model Performance")
st.caption("Metrics computed on the held-out test set (last 15% of temporal data for AQI; random 15% for PHRS).")

if METRICS is None:
    st.info("No metrics found. Run `python train.py` to train the models and generate metrics.")
else:
    # ── 7a: Metrics table ─────────────────────────────────────────────────────
    rows = []
    for key, m in METRICS.items():
        row = {
            "Model":      m.get("label", key),
            "Test R²":    m.get("r2", "—"),
            "Test MAE":   m.get("mae", "—"),
            "Test RMSE":  m.get("rmse", "—"),
            "Train R²":   m.get("train_r2", "—"),
        }
        if key == "phrs":
            row["CV R² (5-fold)"] = f"{m.get('cv_r2', '—')} ± {m.get('cv_r2_std', '—')}"
        else:
            row["CV R² (5-fold)"] = "—"
        rows.append(row)

    metrics_df = pd.DataFrame(rows)

    # Style R² columns: green if good, yellow if ok, red if poor
    def _color_r2(val):
        try:
            v = float(val)
        except (TypeError, ValueError):
            return ""
        if v >= 0.85:
            return "color: #2ecc71; font-weight: bold"
        if v >= 0.70:
            return "color: #f39c12"
        return "color: #e74c3c"

    styled = (
        metrics_df.style
        .applymap(_color_r2, subset=["Test R²", "Train R²"])
        .format({
            "Test R²":  "{:.4f}",
            "Test MAE": "{:.2f}",
            "Test RMSE":"{:.2f}",
            "Train R²": "{:.4f}",
        }, na_rep="—")
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── 7b: R² comparison bar chart ───────────────────────────────────────────
    col_r2, col_err = st.columns(2)

    with col_r2:
        st.markdown("**R² by Model (Test vs Train)**")
        r2_rows = []
        for key, m in METRICS.items():
            lbl = m.get("label", key)
            r2_rows.append({"Model": lbl, "R²": m.get("r2", 0), "Set": "Test"})
            r2_rows.append({"Model": lbl, "R²": m.get("train_r2", 0), "Set": "Train"})
        r2_df = pd.DataFrame(r2_rows)

        fig_r2 = px.bar(
            r2_df, x="Model", y="R²", color="Set", barmode="group",
            color_discrete_map={"Test": "#3498db", "Train": "#95a5a6"},
            range_y=[0, 1.05], height=320,
        )
        fig_r2.add_hline(y=0.85, line_dash="dot", line_color="#2ecc71",
                         annotation_text="Good (0.85)", annotation_position="right")
        fig_r2.add_hline(y=0.70, line_dash="dot", line_color="#f39c12",
                         annotation_text="Acceptable (0.70)", annotation_position="right")
        fig_r2.update_layout(
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font=dict(color="white"), margin=dict(t=10, r=100),
            xaxis_tickangle=-20,
        )
        st.plotly_chart(fig_r2, use_container_width=True)

    with col_err:
        st.markdown("**MAE & RMSE by Model**")
        err_rows = []
        for key, m in METRICS.items():
            lbl = m.get("label", key)
            err_rows.append({"Model": lbl, "Error": m.get("mae", 0),  "Metric": "MAE"})
            err_rows.append({"Model": lbl, "Error": m.get("rmse", 0), "Metric": "RMSE"})
        err_df = pd.DataFrame(err_rows)

        fig_err = px.bar(
            err_df, x="Model", y="Error", color="Metric", barmode="group",
            color_discrete_map={"MAE": "#e74c3c", "RMSE": "#e67e22"},
            height=320,
        )
        fig_err.update_layout(
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font=dict(color="white"), margin=dict(t=10),
            xaxis_tickangle=-20,
        )
        st.plotly_chart(fig_err, use_container_width=True)

    # ── 7c: AQI forecast accuracy by horizon ─────────────────────────────────
    st.markdown("**AQI Forecast Accuracy vs Horizon**")
    st.caption(
        "R² degrades with longer horizons — this is expected (uncertainty grows over time). "
        "MAE and RMSE are in AQI units."
    )
    horizon_rows = []
    for key in ["aqi_h1", "aqi_h3", "aqi_h7"]:
        if key not in METRICS:
            continue
        m = METRICS[key]
        horizon_rows.append({
            "Horizon": f"+{m.get('horizon_days', '?')} day(s)",
            "Test R²": m.get("r2", 0),
            "MAE (AQI units)":  m.get("mae", 0),
            "RMSE (AQI units)": m.get("rmse", 0),
        })
    if horizon_rows:
        h_df = pd.DataFrame(horizon_rows)
        fig_h = go.Figure()
        fig_h.add_trace(go.Scatter(
            x=h_df["Horizon"], y=h_df["Test R²"],
            mode="lines+markers+text",
            text=[f"{v:.3f}" for v in h_df["Test R²"]],
            textposition="top center",
            marker=dict(size=12, color="#3498db"),
            line=dict(width=2, color="#3498db"),
            name="Test R²", yaxis="y1",
        ))
        fig_h.add_trace(go.Bar(
            x=h_df["Horizon"], y=h_df["MAE (AQI units)"],
            name="MAE", marker_color="#e74c3c",
            opacity=0.7, yaxis="y2",
        ))
        fig_h.update_layout(
            height=320,
            yaxis=dict(title="R²", range=[0, 1.1], side="left",
                       gridcolor="#333", color="white"),
            yaxis2=dict(title="MAE (AQI units)", overlaying="y", side="right",
                        color="#e74c3c"),
            legend=dict(orientation="h", y=1.1),
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font=dict(color="white"), margin=dict(t=30, r=80),
        )
        st.plotly_chart(fig_h, use_container_width=True)

    # ── 7d: PHRS CV score badge ───────────────────────────────────────────────
    if "phrs" in METRICS:
        pm = METRICS["phrs"]
        cv  = pm.get("cv_r2", None)
        std = pm.get("cv_r2_std", None)
        if cv is not None:
            color = "#2ecc71" if cv >= 0.85 else ("#f39c12" if cv >= 0.70 else "#e74c3c")
            st.markdown(
                f"<div style='background:#1a1a2e;border-left:4px solid {color};"
                f"padding:12px;border-radius:4px;margin-top:8px'>"
                f"<b style='color:{color}'>PHRS Model — 5-Fold CV R²: {cv:.4f} ± {std:.4f}</b>"
                f"<br><span style='color:#aaa;font-size:0.9em'>"
                f"Cross-validation R² measures how well the model generalises across different "
                f"subsets of the PHRS training data. Values ≥ 0.85 indicate strong fit.</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

# ─── Footer ───────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    "BioAQI v1.1 · Personal Health Risk Score (PHRS) · "
    "Data: India AQI Dataset (Kaggle — city_day.csv) · For educational/research use only."
)
