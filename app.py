"""
BREATHE — Streamlit Dashboard
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
    aggregate_condition_weight,
    CONDITION_WEIGHTS, ACTIVITY_MULTIPLIERS,
)
from models import MAX_DAILY_CHANGE
from preprocess import normalize_aqi_india
import sensor as _sensor

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BREATHE — Personal Air Quality Risk",
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
        "logged_in":    False,
        "username":     "",
        "profile":      {},
        "auth_tab":     "Login",   # "Login" | "Sign Up"
        "manual_mode":  False,     # Auto=False | Manual=True
        "horizon_view": 0,         # 0=Today, 1=+1d, 3=+3d, 7=+7d
        "sensor_aqi":   150.0,
        "manual_live_aqi": 150.0,
        "manual_aqi_follow_sensor": False,
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


# ─── Season helper ────────────────────────────────────────────────────────────

_MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

def _month_to_season(month: int) -> str:
    if month in (6, 7, 8, 9):   return "Monsoon"
    if month in (10, 11):        return "Post_Monsoon"
    if month in (12, 1, 2):      return "Winter"
    return "Summer"


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
            "<h1 style='text-align:center; margin-bottom:0'>BREATHE</h1>",
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
                "This helps BREATHE personalise your Air Quality Risk Score. "
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


# ── MQ135 live panel — auto-refreshes every 2 s when sensor is active ─────────

@st.fragment(run_every=2)
def _sensor_live_panel():
    """Fragment that polls the background sensor thread every 2 seconds."""
    if not st.session_state.get("sensor_active"):
        return
    r = _sensor.get_reading()
    if r["error"]:
        st.warning(f"Sensor error: {r['error']}")
        return
    if r["ppm"] is None:
        st.info("Waiting for sensor data... (allow 2–5 min warm-up)")
        return

    aqi   = r["aqi"]
    ppm   = r["ppm"]
    raw   = r["raw"]

    # Map AQI to a label for context
    if aqi <= 50:
        cat = "Good"
    elif aqi <= 100:
        cat = "Satisfactory"
    elif aqi <= 200:
        cat = "Moderate"
    elif aqi <= 300:
        cat = "Poor"
    elif aqi <= 400:
        cat = "Very Poor"
    else:
        cat = "Severe"

    raw_info = f" · RAW ADC: {raw}" if raw is not None else ""
    st.info(
        f"**MQ135 Live →**  {ppm} PPM  |  AQI ~**{aqi}** ({cat}){raw_info}  "
        f"*(slider pre-filled; drag to override)*"
    )
    # Keep raw sensor telemetry separate from the manual AQI input state.
    prev_sensor_aqi = st.session_state.get("sensor_aqi")
    st.session_state["sensor_aqi"] = aqi
    if (
        st.session_state.get("manual_mode")
        and st.session_state.get("manual_aqi_follow_sensor")
        and st.session_state.get("manual_live_aqi") != aqi
        and prev_sensor_aqi != aqi
    ):
        st.session_state["manual_live_aqi"] = float(aqi)
        st.rerun()


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
        st.markdown("### MQ135 Sensor")
        _available_ports = _sensor.list_ports()
        if _available_ports:
            _sel_port = st.selectbox("COM Port", _available_ports, key="sensor_port")
            _already_active = st.session_state.get("sensor_active")
            if not _already_active:
                if st.button("Connect Sensor", use_container_width=True):
                    _sensor.start(_sel_port)
                    st.session_state["sensor_active"] = True
                    st.session_state["manual_aqi_follow_sensor"] = True
                    st.rerun()
            else:
                if st.button("Disconnect", use_container_width=True):
                    st.session_state["sensor_active"] = False
                    st.session_state["manual_aqi_follow_sensor"] = False
                    st.rerun()
                _r = _sensor.get_reading()
                if _r["error"]:
                    st.error(f"Error: {_r['error']}")
                elif _r["ppm"] is not None:
                    st.success(f"Connected · {_r['ppm']} PPM")
                else:
                    st.info("Connecting... (waiting for data)")
        else:
            st.caption("No serial ports detected. Plug in your Arduino.")

        st.markdown("---")
        st.caption("BREATHE v2.0 · Personalized Air Quality Risk")

    # ── Derive primary condition (highest risk weight) ─────────────────────────
    primary_condition = _primary_condition(sb_conditions)

    profile = HealthProfile(
        age=sb_age,
        condition=primary_condition,
        activity_level=sb_activity,
        hours_outdoors=sb_hours,
    )

    # ── Title ──────────────────────────────────────────────────────────────────
    st.title("BREATHE — Personal Health Risk Score")
    st.markdown(
        "Standard AQI treats everyone equally. **BREATHE** computes *your* personal risk "
        "based on your health profile, activity, weather conditions, and predicted air quality trends."
    )

    if sb_conditions and len(sb_conditions) > 1:
        labels     = " · ".join(sb_conditions)
        eff_weight = aggregate_condition_weight(sb_conditions)
        st.info(
            f"**Active conditions:** {labels}  \n"
            f"Effective condition weight: **{eff_weight:.2f}×** "
            f"(all co-morbidities included via additive penalty)"
        )

    if not MODELS_READY:
        st.warning("Models not found. Run `python train.py` first to train all models.")

    # ══════════════════════════════════════════════════════════════════════════
    # MODE TOGGLE — Auto vs Manual
    # ══════════════════════════════════════════════════════════════════════════

    mode_col, _ = st.columns([1, 3])
    with mode_col:
        manual_mode = st.toggle(
            "Manual Input Mode",
            value=st.session_state.manual_mode,
            key="mode_toggle",
            help="Auto: loads latest recorded city data.  Manual: enter your own scenario values.",
        )
        if manual_mode != st.session_state.manual_mode:
            st.session_state.manual_mode  = manual_mode
            st.session_state.horizon_view = 0   # reset horizon on mode switch
            st.rerun()

    if manual_mode:
        st.caption("Manual mode — enter your own AQI and weather to explore any scenario.")
    else:
        st.caption("Auto mode — using the latest recorded data for your selected city.")

    # ── Manual mode input panel ────────────────────────────────────────────────
    if manual_mode:
        with st.expander("Manual Input Values", expanded=True):
            # ── MQ135 live reading — auto-refreshes every 2 s ─────────────
            _sensor_live_panel()
            sensor_is_active = st.session_state.get("sensor_active", False)
            follow_sensor = st.checkbox(
                "Use live sensor for Current AQI",
                key="manual_aqi_follow_sensor",
                disabled=not sensor_is_active,
                help="When enabled, the Current AQI slider follows the MQ135 live reading.",
            )
            if sensor_is_active and follow_sensor:
                sensor_aqi = float(min(st.session_state.get("sensor_aqi", 150), 500))
                if st.session_state.get("manual_live_aqi") != sensor_aqi:
                    st.session_state["manual_live_aqi"] = sensor_aqi
            elif not sensor_is_active:
                st.caption("Connect the MQ135 sensor to enable live AQI sync.")

            mc1, mc2, mc3, mc4 = st.columns(4)
            with mc1:
                manual_aqi = st.slider(
                    "Current AQI",
                    0,
                    500,
                    key="manual_live_aqi",
                    disabled=follow_sensor and sensor_is_active,
                )
            with mc2:
                manual_temp  = st.slider("Temperature (°C)", -5, 50, 25, key="m_temp")
            with mc3:
                manual_humid = st.slider("Humidity (%)", 0, 100, 50, key="m_humid")
            with mc4:
                manual_month_name = st.selectbox(
                    "Month", _MONTH_NAMES, index=0, key="m_month"
                )
                manual_month = _MONTH_NAMES.index(manual_month_name) + 1

    # ══════════════════════════════════════════════════════════════════════════
    # HORIZON SELECTOR — Today / +1 Day / +3 Days / +7 Days
    # ══════════════════════════════════════════════════════════════════════════

    st.markdown("**View Forecast Horizon:**")
    h_cols = st.columns(4)
    _HORIZONS = [(0, "Today"), (1, "+1 Day"), (3, "+3 Days"), (7, "+7 Days")]
    for col, (h, label) in zip(h_cols, _HORIZONS):
        is_active = st.session_state.horizon_view == h
        if col.button(
            label,
            use_container_width=True,
            type="primary" if is_active else "secondary",
            key=f"hbtn_{h}",
        ):
            st.session_state.horizon_view = h
            st.rerun()

    horizon = st.session_state.horizon_view
    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # DATA LOADING — Auto or Manual
    # ══════════════════════════════════════════════════════════════════════════

    latest_row = get_latest_row(sb_city) if sb_city else None
    _POLL_COLS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]

    if manual_mode:
        # Build a synthetic row from the last known city row (for lag context)
        # then override with user-entered values
        if latest_row is not None:
            row = latest_row.copy()
        else:
            row = pd.Series(dtype=float)

        current_aqi = float(manual_aqi)
        temp_c      = float(manual_temp)
        humidity    = float(manual_humid)
        season      = _month_to_season(manual_month)

        # Override the AQI-related fields with the manual values
        row["AQI"]              = current_aqi
        row["Temp_2m_C"]        = temp_c
        row["Humidity_Percent"] = humidity
        row["month"]            = manual_month
        row["AQI_lag1"]         = current_aqi   # assume stable history
        row["AQI_lag3"]         = current_aqi
        row["AQI_lag7"]         = current_aqi
        row["AQI_roll7_mean"]   = current_aqi
        row["AQI_roll7_std"]    = 0.0
        row["AQI_delta1"]       = 0.0
        row["AQI_delta3"]       = 0.0
        row["AQI_norm_india"]   = normalize_aqi_india(current_aqi)
        for s in ["Winter", "Monsoon", "Post_Monsoon", "Summer"]:
            row[f"season_{s}"] = int(season == s)

        pollutants = {}   # no per-pollutant data in manual mode

    else:
        if latest_row is not None:
            row         = latest_row
            current_aqi = float(latest_row["AQI"])
            pollutants  = {
                p: float(latest_row.get(p, 0) or 0)
                for p in _POLL_COLS if p in latest_row.index
            }
            temp_c   = float(latest_row.get("Temp_2m_C", 25.0) or 25.0)
            humidity = float(latest_row.get("Humidity_Percent", 50.0) or 50.0)
        else:
            row         = pd.Series(dtype=float)
            current_aqi = 0.0
            pollutants  = {}
            temp_c      = 25.0
            humidity    = 50.0

    # Pre-compute forecasts for all horizons (used by both chart and KPI cards)
    _forecasts = {}   # horizon → predicted AQI (or None)
    if MODELS_READY and not row.empty:
        for h in [1, 3, 7]:
            _forecasts[h] = predict_future_aqi_smooth(row, current_aqi, h)

    # ── Select display_aqi based on chosen horizon ─────────────────────────────
    if horizon == 0:
        display_aqi  = current_aqi
        horizon_label = "Current AQI"
        is_forecast   = False
    else:
        display_aqi   = _forecasts.get(horizon, current_aqi) or current_aqi
        horizon_label = f"Predicted AQI (+{horizon} day{'s' if horizon > 1 else ''})"
        is_forecast   = True

    # For the PHRS trend component: use the next-horizon prediction as "future"
    _next_h = {1: 3, 3: 7, 7: 7}
    trend_aqi = (
        _forecasts.get(_next_h.get(horizon, 1))
        if horizon > 0
        else (_forecasts.get(1))
    )

    # ── PHRS computation (weather-aware, multi-condition) ─────────────────────
    phrs_score             = compute_phrs(
        display_aqi, pollutants, profile,
        predicted_aqi=trend_aqi,
        temp_c=temp_c,
        humidity=humidity,
        conditions=sb_conditions,
    )
    risk_label, risk_color = phrs_category(phrs_score)

    # ══════════════════════════════════════════════════════════════════════════
    # KPI CARDS
    # ══════════════════════════════════════════════════════════════════════════

    if is_forecast:
        st.markdown(
            f"<div style='background:#1a1a2e;border-left:4px solid #f39c12;"
            f"padding:8px 14px;border-radius:4px;margin-bottom:12px'>"
            f"<b style='color:#f39c12'>Viewing forecast: +{horizon} day{'s' if horizon > 1 else ''}</b>"
            f" — AQI and PHRS below reflect predicted conditions, not current readings."
            f"</div>",
            unsafe_allow_html=True,
        )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(horizon_label, f"{display_aqi:.0f}")
        st.caption(_aqi_bucket(display_aqi))

    with c2:
        phrs_card_label = f"PHRS (+{horizon}d)" if is_forecast else "Your PHRS"
        st.metric(phrs_card_label, f"{phrs_score:.1f} / 100")
        st.markdown(
            f"<span style='color:{risk_color}; font-weight:bold; font-size:1.1em'>"
            f"{risk_label}</span>",
            unsafe_allow_html=True,
        )

    with c3:
        next_h_for_card = _next_h.get(horizon, 1) if horizon > 0 else 1
        next_aqi = _forecasts.get(next_h_for_card)
        ref_aqi  = display_aqi if not is_forecast else current_aqi
        if next_aqi is not None:
            delta_val = round(next_aqi - ref_aqi, 1)
            delta_str = f"+{delta_val}" if delta_val > 0 else str(delta_val)
            label_c3  = (
                f"Predicted AQI (+{next_h_for_card}d)"
                if not is_forecast
                else f"Next Forecast (+{next_h_for_card}d)"
            )
            st.metric(label_c3, f"{next_aqi:.0f}", delta=delta_str, delta_color="inverse")
        else:
            st.metric("Predicted AQI (+1 day)", "—")
            st.caption("Train models to see forecast")

    with c4:
        eff_cw = aggregate_condition_weight(sb_conditions)
        am     = ACTIVITY_MULTIPLIERS.get(sb_activity, 1.0)
        st.metric("Sensitivity Multiplier", f"{eff_cw * am:.2f}×")
        st.caption(f"Condition {eff_cw:.2f}× · Activity {am}×  \nTemp {temp_c:.0f}°C · Humidity {humidity:.0f}%")

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
        if not row.empty and MODELS_READY and _forecasts:
            fc_labels  = ["Now", "+1 day", "+3 days", "+7 days"]
            fc_aqi     = [
                current_aqi,
                _forecasts.get(1, current_aqi),
                _forecasts.get(3, current_aqi),
                _forecasts.get(7, current_aqi),
            ]
            # Map selected horizon to its label for highlight
            _h_label_map = {0: "Now", 1: "+1 day", 3: "+3 days", 7: "+7 days"}
            selected_label = _h_label_map[horizon]

            fig_fc = go.Figure()
            # Main forecast line
            fig_fc.add_trace(go.Scatter(
                x=fc_labels, y=fc_aqi,
                mode="lines+markers",
                marker=dict(size=10, color=[_risk_color_for_aqi(v) for v in fc_aqi]),
                line=dict(color="#3498db", width=2),
                name="AQI",
            ))
            # Highlight the currently-viewed horizon with a large yellow ring
            fig_fc.add_trace(go.Scatter(
                x=[selected_label],
                y=[fc_aqi[fc_labels.index(selected_label)]],
                mode="markers",
                marker=dict(size=22, color="rgba(0,0,0,0)",
                            line=dict(color="#f1c40f", width=3)),
                name="Viewing",
                showlegend=True,
            ))
            for threshold, lbl, color in [
                (50,  "Good",     "#2ecc71"),
                (100, "Moderate", "#f39c12"),
                (200, "Poor",     "#e67e22"),
            ]:
                fig_fc.add_hline(y=threshold, line_dash="dot", line_color=color,
                                 annotation_text=lbl, annotation_position="right")
            fig_fc.update_layout(
                height=280, xaxis_title="Horizon", yaxis_title="AQI",
                margin=dict(l=20, r=70, t=30, b=20),
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                font=dict(color="white"),
                legend=dict(orientation="h", y=1.12),
            )
            st.plotly_chart(fig_fc, use_container_width=True)
        else:
            st.info("Select a city with available data and trained models to see the AQI forecast.")

    st.markdown("---")

    # ── Row 3: Personalized Recommendations (moved here — just below gauge) ────
    st.subheader("Personalized Recommendations")
    recs = get_recommendations(phrs_score, profile, sb_conditions, display_aqi, trend_aqi)
    if is_forecast:
        st.caption(f"Recommendations based on predicted conditions at +{horizon} day{'s' if horizon > 1 else ''}.")
    for rec in recs:
        st.markdown(f"- {rec}")

    st.markdown("---")

    # ── Row 4: Pollutant breakdown ─────────────────────────────────────────────
    st.subheader("Pollutant Breakdown")

    if manual_mode:
        st.info(
            "Pollutant breakdown is not available in Manual mode — "
            "only overall AQI was entered. Switch to Auto mode to see per-pollutant data."
        )
    elif pollutants:
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
    elif not manual_mode:
        st.info("No pollutant data available. Select a city from the sidebar.")

    st.markdown("---")

    # ── Row 4: Sensitivity comparison ─────────────────────────────────────────
    st.subheader("How Different Profiles Compare (Same AQI)")

    if display_aqi > 0:
        comparison_rows = []
        for cond in CONDITION_WEIGHTS:
            for act in ["Sedentary", "Active"]:
                p = HealthProfile(age=30, condition=cond, activity_level=act, hours_outdoors=3)
                s = compute_phrs(display_aqi, pollutants, p,
                                 temp_c=temp_c, humidity=humidity)
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
            .map(_color_r2, subset=["Test R²", "Train R²"])
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
        "BREATHE v2.0 · Personal Health Risk Score (PHRS) · "
        "Data: India AQI Dataset + INDIA_AQI_COMPLETE · For educational/research use only."
    )


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.logged_in:
    _show_dashboard()
else:
    _show_auth_page()
