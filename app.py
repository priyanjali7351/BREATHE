"""
BioAQI — Streamlit Dashboard
Run: streamlit run app.py

Flow:
  Landing page (Login / Sign Up with health profile)
    └─► Main Dashboard (sidebar profile editor + multi-condition checkboxes)
"""

import hashlib
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

# ─── User store helpers (flat JSON file) ──────────────────────────────────────

_USERS_FILE = Path("data/users.json")


def _load_users() -> dict:
    if _USERS_FILE.exists():
        with open(_USERS_FILE) as f:
            return json.load(f)
    return {}


def _save_users(users: dict):
    _USERS_FILE.parent.mkdir(exist_ok=True)
    with open(_USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def _hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


def _authenticate(username: str, password: str) -> dict | None:
    """Return user record if credentials match, else None."""
    users = _load_users()
    user  = users.get(username.strip().lower())
    if user and user["password_hash"] == _hash_pw(password):
        return user
    return None


def _register(username: str, password: str, profile: dict) -> bool:
    """Create new user. Returns False if username already taken."""
    users = _load_users()
    key   = username.strip().lower()
    if key in users:
        return False
    users[key] = {"password_hash": _hash_pw(password), "profile": profile}
    _save_users(users)
    return True


def _update_profile(username: str, profile: dict):
    users = _load_users()
    key   = username.strip().lower()
    if key in users:
        users[key]["profile"] = profile
        _save_users(users)


# ─── Session state bootstrap ──────────────────────────────────────────────────

def _init_session():
    defaults = {
        "logged_in":  False,
        "username":   "",
        "profile":    {},
        "auth_tab":   "Login",   # "Login" | "Sign Up"
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_session()

# ─── Helper: derive primary condition from list ────────────────────────────────

def _primary_condition(conditions: list[str]) -> str:
    """Return the highest-risk condition from a list (for PHRS computation)."""
    if not conditions:
        return "Healthy"
    return max(conditions, key=lambda c: CONDITION_WEIGHTS.get(c, 1.0))


# ─── App helpers ──────────────────────────────────────────────────────────────

def _aqi_bucket(aqi: float) -> str:
    if aqi <= 50:   return "Good"
    if aqi <= 100:  return "Satisfactory — minor breathing discomfort for sensitive groups"
    if aqi <= 150:  return "Moderate — breathing discomfort for people with lung/heart disease"
    if aqi <= 200:  return "Unhealthy — breathing discomfort for most people"
    if aqi <= 300:  return "Hazardous — serious respiratory risk for everyone"
    return "Severe — emergency conditions, entire population affected"


def _risk_color_for_aqi(aqi: float) -> str:
    if aqi <= 50:   return "#2ecc71"
    if aqi <= 100:  return "#a8d86e"
    if aqi <= 150:  return "#edf10f"
    if aqi <= 200:  return "#f39c12"
    if aqi <= 300:  return "#e62c22"
    return "#741663"


def get_recommendations(phrs: float, profile: HealthProfile,
                        conditions: list[str],
                        current_aqi: float, pred_aqi: float | None) -> list[str]:
    recs = []
    if phrs <= 30:
        recs.append("Air quality is safe for you today. Normal activities are fine.")
    elif phrs <= 60:
        recs.append("Moderate risk. Limit prolonged strenuous outdoor activity.")
        if profile.activity_level in ("Active", "Athlete"):
            recs.append("Consider moving workouts indoors or to early morning when AQI is lower.")
    elif phrs <= 80:
        recs.append("High risk. Avoid outdoor activity — especially exercise.")
        recs.append("Stay indoors with windows closed. Use an air purifier if available.")
        if any(c in conditions for c in ("Mild Asthma", "Severe Asthma")):
            recs.append("Keep your rescue inhaler accessible at all times.")
        if "Heart Disease" in conditions:
            recs.append("Monitor for chest tightness or shortness of breath.")
    else:
        recs.append("Critical risk. Stay indoors. Avoid all outdoor exposure.")
        recs.append("If going outside is unavoidable, wear an N95 mask.")
        recs.append("Follow your doctor's emergency action plan.")

    if pred_aqi is not None and pred_aqi > current_aqi + 50:
        recs.append(f"AQI is forecast to rise to ~{pred_aqi:.0f} tomorrow. Plan accordingly.")

    if profile.age < 12 or profile.age > 65:
        recs.append("Extra caution is recommended for your age group.")

    if profile.hours_outdoors >= 6 and phrs > 40:
        recs.append(
            "Reduce time outdoors — your high daily exposure significantly increases your risk."
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
    """
    Load and preprocess AQI data using the full 3-dataset pipeline.
    Falls back to city_day only if the large INDIA file is absent.
    """
    from preprocess import build_training_df

    city_day = Path("data/city_day.csv")
    india    = Path("data/INDIA_AQI_COMPLETE_20251126.csv")
    health   = Path("data/air_quality_health_impact_data.csv")

    if not city_day.exists():
        return None

    if india.exists() and health.exists():
        try:
            pipeline = build_training_df(
                city_day_path=str(city_day),
                india_path=str(india),
                health_path=str(health),
            )
            return pipeline["full_df"]
        except Exception:
            pass

    # Minimal fallback: just city_day with old-style preprocessing
    try:
        from preprocess import _load_city_day, _impute_per_city, _treat_outliers, _engineer_features
        df = _load_city_day(str(city_day))
        df["Season"] = df["Date"].dt.month.map(
            lambda m: "Monsoon" if m in (6,7,8,9) else
                      "Post_Monsoon" if m in (10,11) else
                      "Winter" if m in (12,1,2) else "Summer"
        )
        for col in ["Festival_Period", "Crop_Burning_Season", "Temp_Inversion",
                    "Temp_2m_C", "Humidity_Percent", "Wind_Speed_kmh",
                    "Precipitation_mm", "Wind_Stagnation",
                    "ref_health_score", "ref_resp_cases"]:
            df[col] = 0 if col in ("Festival_Period", "Crop_Burning_Season",
                                   "Temp_Inversion") else 0.0
        df = df.dropna(subset=["AQI"])
        df = _impute_per_city(df)
        df = _treat_outliers(df)
        df = _engineer_features(df)
        return df
    except Exception:
        return None


MODELS       = load_models()
METRICS      = load_metrics()
AQI_DF       = load_aqi_data()
MODELS_READY = bool(MODELS)

ALL_CONDITIONS = list(CONDITION_WEIGHTS.keys())
ALL_ACTIVITIES = list(ACTIVITY_MULTIPLIERS.keys())
ALL_CITIES     = sorted(AQI_DF["City"].unique().tolist()) if AQI_DF is not None else []


# ─── Condition checkboxes widget ──────────────────────────────────────────────

def _condition_checkboxes(
    label: str,
    defaults: list[str],
    key_prefix: str,
) -> list[str]:
    """
    Render a set of checkboxes for all health conditions.
    Returns the list of selected condition strings.
    At least one must remain selected (enforced below).
    """
    st.markdown(f"**{label}**")
    selected = []
    cols = st.columns(2)
    for i, cond in enumerate(ALL_CONDITIONS):
        col = cols[i % 2]
        checked = col.checkbox(
            cond,
            value=(cond in defaults),
            key=f"{key_prefix}_cond_{cond}",
        )
        if checked:
            selected.append(cond)
    if not selected:
        selected = ["Healthy"]   # always keep at least one
    return selected


# ─── Prediction helpers ────────────────────────────────────────────────────────

def get_latest_row(city: str) -> pd.Series | None:
    if AQI_DF is None or city not in AQI_DF["City"].values:
        return None
    city_df = AQI_DF[AQI_DF["City"] == city].sort_values("Date")
    return city_df.iloc[-1]


def predict_future_aqi_smooth(row: pd.Series, current_aqi: float,
                               horizon: int) -> float | None:
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


# ══════════════════════════════════════════════════════════════════════════════
# LANDING PAGE — Login / Sign Up
# ══════════════════════════════════════════════════════════════════════════════

def _show_auth_page():
    # Centre card layout
    _, centre, _ = st.columns([1, 2, 1])
    with centre:
        st.markdown(
            "<h1 style='text-align:center; margin-bottom:0'>🌬️ BioAQI</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align:center; color:#aaa; margin-top:4px'>"
            "Personalized Air Quality Risk Platform</p>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        tab_login, tab_signup = st.tabs(["Login", "Sign Up"])

        # ── Login tab ──────────────────────────────────────────────────────
        with tab_login:
            st.markdown("#### Welcome back")
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pw")

            if st.button("Login", use_container_width=True, type="primary"):
                if not username or not password:
                    st.error("Please enter your username and password.")
                else:
                    user = _authenticate(username, password)
                    if user is None:
                        st.error("Invalid username or password.")
                    else:
                        st.session_state.logged_in = True
                        st.session_state.username  = username.strip().lower()
                        st.session_state.profile   = user["profile"]
                        st.rerun()

        # ── Sign Up tab ────────────────────────────────────────────────────
        with tab_signup:
            st.markdown("#### Create your account")
            new_user = st.text_input("Choose a username", key="su_user")
            new_pw   = st.text_input("Choose a password", type="password", key="su_pw")
            new_pw2  = st.text_input("Confirm password",  type="password", key="su_pw2")

            st.markdown("---")
            st.markdown("### Your Health Profile")
            st.caption(
                "This helps BioAQI personalise your Air Quality Risk Score. "
                "You can update it anytime from the dashboard."
            )

            su_age = st.slider("Age", 5, 90, 25, key="su_age")
            su_activity = st.selectbox(
                "Activity Level", ALL_ACTIVITIES, key="su_activity"
            )
            su_hours = st.slider(
                "Hours Outdoors / Day", 0.0, 12.0, 2.0, 0.5, key="su_hours"
            )

            su_city = st.selectbox(
                "Your City",
                ALL_CITIES if ALL_CITIES else ["Delhi"],
                key="su_city",
            )

            st.markdown("<br>", unsafe_allow_html=True)
            su_conditions = _condition_checkboxes(
                "Health Conditions (select all that apply)",
                defaults=["Healthy"],
                key_prefix="su",
            )

            if st.button("Create Account", use_container_width=True, type="primary"):
                if not new_user or not new_pw:
                    st.error("Username and password are required.")
                elif new_pw != new_pw2:
                    st.error("Passwords do not match.")
                elif len(new_pw) < 4:
                    st.error("Password must be at least 4 characters.")
                else:
                    profile = {
                        "age":           su_age,
                        "conditions":    su_conditions,
                        "activity_level": su_activity,
                        "hours_outdoors": su_hours,
                        "city":           su_city,
                    }
                    ok = _register(new_user, new_pw, profile)
                    if not ok:
                        st.error("Username already taken. Please choose another.")
                    else:
                        st.success(
                            "Account created! Switch to the Login tab to sign in."
                        )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def _show_dashboard():
    profile_data = st.session_state.profile
    username     = st.session_state.username

    # ── Sidebar — Health Profile Editor ───────────────────────────────────────
    with st.sidebar:
        st.markdown(
            f"<p style='color:#aaa; font-size:0.85em'>Signed in as "
            f"<b style='color:white'>{username}</b></p>",
            unsafe_allow_html=True,
        )
        if st.button("Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username  = ""
            st.session_state.profile   = {}
            st.rerun()

        st.markdown("---")
        st.markdown("### Your Health Profile")

        sb_age = st.slider(
            "Age", 5, 90,
            int(profile_data.get("age", 25)),
            key="sb_age",
        )
        sb_activity = st.selectbox(
            "Activity Level",
            ALL_ACTIVITIES,
            index=ALL_ACTIVITIES.index(
                profile_data.get("activity_level", "Moderate")
            ),
            key="sb_activity",
        )
        sb_hours = st.slider(
            "Hours Outdoors / Day",
            0.0, 12.0,
            float(profile_data.get("hours_outdoors", 2.0)),
            0.5,
            key="sb_hours",
        )

        city_options = ALL_CITIES if ALL_CITIES else ["Delhi"]
        saved_city   = profile_data.get("city", city_options[0])
        city_idx     = city_options.index(saved_city) if saved_city in city_options else 0
        sb_city = st.selectbox("City", city_options, index=city_idx, key="sb_city")

        st.markdown("---")
        st.markdown("**Health Conditions**")
        st.caption("Select all that apply.")
        saved_conditions = profile_data.get("conditions", ["Healthy"])

        sb_conditions = []
        for cond in ALL_CONDITIONS:
            checked = st.checkbox(
                cond,
                value=(cond in saved_conditions),
                key=f"sb_cond_{cond}",
            )
            if checked:
                sb_conditions.append(cond)
        if not sb_conditions:
            sb_conditions = ["Healthy"]

        st.markdown("---")
        st.markdown("**Manual AQI Override**")
        manual_aqi = st.number_input(
            "Current AQI (0 = use city data)",
            min_value=0, max_value=1000, value=0,
            key="sb_manual_aqi",
        )

        if st.button("Save Profile", use_container_width=True, type="primary"):
            updated = {
                "age":            sb_age,
                "conditions":     sb_conditions,
                "activity_level": sb_activity,
                "hours_outdoors": sb_hours,
                "city":           sb_city,
            }
            st.session_state.profile = updated
            _update_profile(username, updated)
            st.success("Profile saved!")

        st.markdown("---")
        st.caption("BioAQI v2.0 · Personalized Air Quality Risk")

    # ── Derive primary condition (highest risk weight) ─────────────────────────
    primary_condition = _primary_condition(sb_conditions)

    profile = HealthProfile(
        age=sb_age,
        condition=primary_condition,
        activity_level=sb_activity,
        hours_outdoors=sb_hours,
    )

    # ── Title ──────────────────────────────────────────────────────────────────
    st.title("🌬️ BioAQI — Personal Health Risk Score")
    st.markdown(
        "Standard AQI treats everyone equally. **BioAQI** computes *your* personal risk "
        "based on your health profile, activity, and predicted air quality trends."
    )

    if sb_conditions and len(sb_conditions) > 1:
        labels = " · ".join(sb_conditions)
        cw_max = CONDITION_WEIGHTS.get(primary_condition, 1.0)
        st.info(
            f"**Active conditions:** {labels}  \n"
            f"Risk is computed using your highest-risk condition: "
            f"**{primary_condition}** (weight {cw_max}×)"
        )

    if not MODELS_READY:
        st.warning("⚠️ Models not found. Run `python train.py` first to train all models.")

    # ── Fetch city data ────────────────────────────────────────────────────────
    latest_row = get_latest_row(sb_city) if sb_city else None

    if manual_aqi > 0:
        current_aqi = float(manual_aqi)
        pollutants  = {}
        pred_aqi    = None
    elif latest_row is not None:
        current_aqi = float(latest_row["AQI"])
        pollutants  = {
            p: float(latest_row.get(p, 0) or 0)
            for p in ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
            if p in latest_row.index
        }
        pred_aqi = (
            predict_future_aqi_smooth(latest_row, current_aqi, 1)
            if MODELS_READY else None
        )
    else:
        current_aqi = 0.0
        pollutants  = {}
        pred_aqi    = None

    # ── PHRS computation ───────────────────────────────────────────────────────
    phrs_score             = compute_phrs(current_aqi, pollutants, profile,
                                          predicted_aqi=pred_aqi)
    risk_label, risk_color = phrs_category(phrs_score)

    # ── KPI cards ──────────────────────────────────────────────────────────────
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
        cw = CONDITION_WEIGHTS.get(primary_condition, 1.0)
        am = ACTIVITY_MULTIPLIERS.get(sb_activity, 1.0)
        st.metric("Sensitivity Multiplier", f"{cw * am:.2f}×")
        st.caption(f"Condition {cw}× · Activity {am}×")

    st.markdown("---")

    # ── Row 2: PHRS gauge + AQI forecast chart ─────────────────────────────────
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
            st.info("Select a city and train models to see the AQI forecast.")

    st.markdown("---")

    # ── Row 3: Pollutant breakdown ─────────────────────────────────────────────
    st.subheader("Pollutant Breakdown")

    if pollutants:
        thresholds = {"PM2.5": 60, "PM10": 100, "NO2": 80, "SO2": 80, "CO": 2, "O3": 100}
        poll_df = pd.DataFrame([
            {
                "Pollutant":      p,
                "Value":          round(v, 2),
                "Safe Threshold": thresholds.get(p, 100),
                "% of Threshold": round(v / thresholds.get(p, 100) * 100, 1),
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
        st.info("No pollutant data available. Select a city or enter a manual AQI.")

    st.markdown("---")

    # ── Row 4: Sensitivity comparison ─────────────────────────────────────────
    st.subheader("How Different Profiles Compare (Same AQI)")

    if current_aqi > 0:
        comparison_rows = []
        for cond in CONDITION_WEIGHTS:
            for act in ["Sedentary", "Active"]:
                p = HealthProfile(age=30, condition=cond, activity_level=act, hours_outdoors=3)
                s = compute_phrs(current_aqi, pollutants, p)
                comparison_rows.append({"Condition": cond, "Activity": act, "PHRS": s})
        comp_df = pd.DataFrame(comparison_rows)
        # Highlight user's conditions
        comp_df["IsUser"] = comp_df["Condition"].isin(sb_conditions)

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

    # ── Row 5: Historical AQI trend ───────────────────────────────────────────
    if AQI_DF is not None and sb_city:
        st.subheader(f"Historical AQI — {sb_city} (last 90 days)")
        city_hist = (
            AQI_DF[AQI_DF["City"] == sb_city]
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

    # ── Row 6: Recommendations ────────────────────────────────────────────────
    st.subheader("Personalized Recommendations")
    recs = get_recommendations(phrs_score, profile, sb_conditions, current_aqi, pred_aqi)
    for rec in recs:
        st.markdown(f"- {rec}")

    st.markdown("---")

    # ── Row 7: Model Performance ──────────────────────────────────────────────
    st.subheader("Model Performance")
    st.caption("Metrics computed on the held-out test set (last 20% of temporal data for AQI; random 20% for PHRS).")

    if METRICS is None:
        st.info("No metrics found. Run `python train.py` to train the models.")
    else:
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
                "Test R²":   "{:.4f}",
                "Test MAE":  "{:.2f}",
                "Test RMSE": "{:.2f}",
                "Train R²":  "{:.4f}",
            }, na_rep="—")
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

        col_r2, col_err = st.columns(2)

        with col_r2:
            st.markdown("**R² by Model (Test vs Train)**")
            r2_rows = []
            for key, m in METRICS.items():
                lbl = m.get("label", key)
                r2_rows.append({"Model": lbl, "R²": m.get("r2", 0),       "Set": "Test"})
                r2_rows.append({"Model": lbl, "R²": m.get("train_r2", 0), "Set": "Train"})
            r2_df = pd.DataFrame(r2_rows)
            fig_r2 = px.bar(
                r2_df, x="Model", y="R²", color="Set", barmode="group",
                color_discrete_map={"Test": "#3498db", "Train": "#95a5a6"},
                range_y=[0, 1.05], height=320,
            )
            fig_r2.add_hline(y=0.85, line_dash="dot", line_color="#2ecc71",
                             annotation_text="Good (0.85)", annotation_position="right")
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

        # AQI forecast accuracy by horizon
        st.markdown("**AQI Forecast Accuracy vs Horizon**")
        horizon_rows = []
        for key in ["aqi_h1", "aqi_h3", "aqi_h7"]:
            if key not in METRICS:
                continue
            m = METRICS[key]
            horizon_rows.append({
                "Horizon": f"+{m.get('horizon_days', '?')} day(s)",
                "Test R²": m.get("r2", 0),
                "MAE":     m.get("mae", 0),
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
                x=h_df["Horizon"], y=h_df["MAE"],
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

        if "phrs" in METRICS:
            pm  = METRICS["phrs"]
            cv  = pm.get("cv_r2")
            std = pm.get("cv_r2_std")
            if cv is not None:
                color = "#2ecc71" if cv >= 0.85 else ("#f39c12" if cv >= 0.70 else "#e74c3c")
                st.markdown(
                    f"<div style='background:#1a1a2e;border-left:4px solid {color};"
                    f"padding:12px;border-radius:4px;margin-top:8px'>"
                    f"<b style='color:{color}'>PHRS Model — 5-Fold CV R²: {cv:.4f} ± {std:.4f}</b>"
                    f"<br><span style='color:#aaa;font-size:0.9em'>"
                    f"Cross-validation R² measures generalisation across PHRS training subsets."
                    f"</span></div>",
                    unsafe_allow_html=True,
                )

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "BioAQI v2.0 · Personal Health Risk Score (PHRS) · "
        "Data: India AQI Dataset + INDIA_AQI_COMPLETE · For educational/research use only."
    )


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.logged_in:
    _show_dashboard()
else:
    _show_auth_page()
