# Outbreak Detective — Prediction & Response (North Campus Wastewater)
# Streamlit app with biologically framed interventions, Days Delayed KPI,
# and a Viral Load Reduction metric.

import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# SciPy is optional: smooth fallback if it's missing
try:
    from scipy.optimize import curve_fit
    SCIPY_OK = True
except Exception:
    curve_fit = None
    SCIPY_OK = False

# -------------------- THEME / STYLES --------------------
HAS_RED   = "#E84428"  # North Campus / key curve
HAS_NAVY  = "#0A1C33"
HAS_AMBER = "#F5A623"
HAS_TEAL  = "#00E0B8"
HAS_GRAY  = "#8896A5"

st.set_page_config(
    page_title="Outbreak Detective — Forecast & Response",
    page_icon="🧠",
    layout="wide",
)

st.markdown(
    """
    <style>
      .stApp { background-color: #0b1220; }
      .metric-box {
        padding: 14px 16px;
        border-radius: 14px;
        background: #0f1a30;
        border: 1px solid #182544;
        color: #DDE6F1;
      }
      .subtle { color: #A7B4C3; font-size: 0.9rem; }
      .kpi-title { color: #DDE6F1; }
      .card {
        background:#0f1a30;border:1px solid #182544;border-radius:12px;
        padding:12px;margin-bottom:10px;
      }
      .card h4 { margin:0 0 6px 0;color:#DDE6F1;font-size:1rem; }
      .tag { color:#A7B4C3;font-size:0.8rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- DEMO TUNING --------------------
# Makes plan effects visually obvious (affects r, <1 = stronger reduction)
DEMO_EXAGGERATION = 0.60
LINE_SHAPE = "spline"
Y_CAP_MULT = 1.6
Y_MIN_FRACTION = 0.5

# -------------------- HELPERS --------------------
def logistic(t, K, r, t0):
    return K / (1.0 + np.exp(-r * (t - t0)))

def exp_curve(t, A, r):
    return A * np.exp(r * t)

def fit_growth_from_series(dates, values):
    """Fit exponential to last ~5 points; return (A, r). Safe fallback if SciPy absent."""
    y = np.array(values, dtype=float)
    t = (pd.to_datetime(dates) - pd.to_datetime(dates).min()).days.values
    idx = max(0, len(t) - 5)
    t_fit = t[idx:]
    y_fit = np.clip(y[idx:], 1e-6, None)

    if SCIPY_OK and len(y_fit) >= 2:
        try:
            popt, _ = curve_fit(
                lambda tt, A, r: exp_curve(tt, A, r),
                t_fit - t_fit[0], y_fit,
                p0=[y_fit[0], 0.25], maxfev=2000
            )
            A_est, r_est = popt
            r_est = float(np.clip(r_est, 0.05, 0.6))
            return max(float(A_est), 1.0), r_est
        except Exception:
            pass  # fall through to log-linear fallback

    # Fallback: log-linear on last points
    if len(y_fit) >= 2 and y_fit[-2] > 0:
        r_guess = float(np.clip(np.log(y_fit[-1] / y_fit[-2] + 1e-9), 0.08, 0.35))
    else:
        r_guess = 0.25
    return float(max(y_fit[0], 8.0)), r_guess

def detect_threshold_cross(y, threshold):
    idx = np.where(y >= threshold)[0]
    return int(idx[0]) if len(idx) else None

def pct_reduction(a, b):
    if a <= 0:
        return 0.0
    return 100.0 * (a - b) / a

# --- Smooth transition simulator (continuous-time with smooth r(t)) ---
def simulate_forecast_smooth(
    days,
    y0,
    r0,
    r_mult=1.0,
    K=None,
    t_int=0,
    trans_days=2.0,
    substeps=12,
):
    """
    r(t) ramps from r0 to r0*r_mult over trans_days using a cubic smoothstep.
    Logistic if K is set, else exponential. Euler integration with substeps/day.
    Returns daily samples of length `days`.
    """
    import numpy as _np

    days = int(days)
    if days <= 0:
        return _np.array([], dtype=float)

    n = (days - 1) * substeps + 1
    t = _np.linspace(0.0, float(days - 1), n)
    y = _np.zeros(n, dtype=float)
    y[0] = max(float(y0), 1e-6)

    def _smoothstep(u):
        u = _np.clip(u, 0.0, 1.0)
        return u * u * (3 - 2 * u)

    for i in range(1, n):
        dt = t[i] - t[i - 1]
        u = (t[i - 1] - float(t_int)) / max(float(trans_days), 1e-6)
        s = _smoothstep(u)  # 0→1 across the window
        r_t = (1.0 - s) * r0 + s * (r0 * r_mult)

        if K is not None and K > 0:
            dy = r_t * y[i - 1] * (1.0 - y[i - 1] / float(K))
        else:
            dy = r_t * y[i - 1]

        y[i] = max(y[i - 1] + dy * dt, 0.0)

    y_daily = y[::substeps]
    return y_daily

# -------------------- SIDEBAR CONTROLS --------------------
st.sidebar.markdown("### 🧭 Controls")

danger = st.sidebar.number_input(
    "Danger threshold (copies/mL, normalized)",
    min_value=50, max_value=20000, value=1000, step=50
)
days_ahead = st.sidebar.slider("Forecast horizon (days)", 7, 28, 14)
use_logistic = st.sidebar.checkbox("Use logistic growth (carrying capacity)", value=True)
K = st.sidebar.number_input(
    "If logistic: carrying capacity (K)",
    min_value=500.0, max_value=50000.0, value=5000.0, step=100.0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Budget Game (10 points)")
budget = 10

# --- FINAL INTERVENTIONS: TR vs SR ---
# TR = Transmission Reducers (change r / slope)
# SR = Shedding Reducers (change wastewater height via shed_mult)
INTERVENTIONS = {
    # Transmission Reducers (TR)
    "Masking Policy": {
        "cost": 4,
        "desc": "Fewer infected droplets in the air → fewer new infections (strong TR).",
        "r_mult": 0.70,
        "shed_mult": 1.00,
    },
    "Improved Dorm Ventilation": {
        "cost": 3,
        "desc": "Cleaner indoor air → fewer exposures become infections (medium TR).",
        "r_mult": 0.80,
        "shed_mult": 1.00,
    },
    "Stay-Home-When-Sick Campaign": {
        "cost": 1,
        "desc": "Keeps mildly sick students out of shared spaces (light TR).",
        "r_mult": 0.90,
        "shed_mult": 1.00,
    },

    # Shedding Reducers (SR)
    "Rapid Testing Blitz": {
        "cost": 6,
        "desc": "Finds hidden positives early → removes silent shedders (strong SR).",
        "r_mult": 1.00,
        "shed_mult": 0.60,
    },
    "Dedicated Isolation Dorms": {
        "cost": 3,
        "desc": "Moves positives to separate plumbing → lowers RNA in wastewater (medium SR).",
        "r_mult": 1.00,
        "shed_mult": 0.75,
    },
    "Same-Day Test Results": {
        "cost": 2,
        "desc": "Faster test → isolation → fewer days shedding (light SR).",
        "r_mult": 1.00,
        "shed_mult": 0.85,
    },
}

chosen = st.sidebar.multiselect(
    f"Choose interventions (Budget {budget} pts):",
    list(INTERVENTIONS.keys()),
    default=[]
)

total_cost = sum(INTERVENTIONS[k]["cost"] for k in chosen)
over_budget = total_cost > budget
applied_interventions = chosen if not over_budget and total_cost > 0 else []

if applied_interventions:
    combined_r_mult = float(np.prod([INTERVENTIONS[k]["r_mult"] for k in applied_interventions]))
    combined_shed_mult = float(np.prod([INTERVENTIONS[k]["shed_mult"] for k in applied_interventions]))
else:
    combined_r_mult = 1.0
    combined_shed_mult = 1.0

if total_cost > budget:
    st.sidebar.error(f"Over budget by {total_cost - budget} point(s). Remove something.")

st.sidebar.markdown("**Selected:**")
for k in chosen:
    st.sidebar.markdown(f"- {k} ({INTERVENTIONS[k]['cost']} pts)")
st.sidebar.markdown(f"**Total cost:** {total_cost} / {budget}")

# -------------------- DATA INGEST / FIT (NO UPLOADS) --------------------
default_dates = pd.date_range("2025-11-05", periods=7, freq="D")
default_values = np.array([120, 140, 180, 260, 420, 520, 560])  # North Campus-ish

series_dates = default_dates
series_vals  = default_values
A0, r0 = fit_growth_from_series(series_dates, series_vals)
base_dates = pd.date_range(series_dates.min(), periods=len(series_vals) + days_ahead, freq="D")
horizon = len(base_dates)

# Optional: align with Detection Phase "today"
DAYS_SHIFT = 6
series_dates = series_dates + pd.Timedelta(days=DAYS_SHIFT)
base_dates   = base_dates + pd.Timedelta(days=DAYS_SHIFT)

# -------------------- MODEL / SCENARIOS --------------------
def run_scenario(r_mult=1.0, shed_mult=1.0, trans_days=2.0):
    """
    r_mult: affects growth rate (transmission).
    shed_mult: scales the forecast part of the curve (shedding).
    """
    rr0 = r0
    K_use = K if use_logistic else None

    y = np.zeros(horizon, dtype=float)

    # 1) Copy ALL observed points directly (shared history up to "Today")
    y[: len(series_vals)] = series_vals[:]

    # 2) Forecast starts the day AFTER the last observed point ("Today")
    start_idx = len(series_vals)          # index of the first future day
    future_days = horizon - start_idx     # how many future days we need

    if future_days > 0:
        # Simulate future trajectory starting from the last observed value
        y_forward = simulate_forecast_smooth(
            days=future_days + 1,         # +1 so we can discard the t=0 point
            y0=series_vals[-1],
            r0=rr0,
            r_mult=r_mult,
            K=K_use,
            t_int=0.0,
            trans_days=trans_days,
            substeps=12,
        )

        # First element of y_forward is the starting value at "Today"
        # We drop it and only use the future points.
        y_future = y_forward[1:] * shed_mult
        y[start_idx:] = y_future

    return y

# Baseline and plan curves
y_base = run_scenario(r_mult=1.0, shed_mult=1.0, trans_days=2.0)

if applied_interventions and not over_budget:
    # Make plan effects slightly stronger visually
    r_mult_effective = combined_r_mult * DEMO_EXAGGERATION
    y_combo = run_scenario(r_mult=r_mult_effective, shed_mult=combined_shed_mult, trans_days=2.0)
else:
    y_combo = y_base.copy()

# -------------------- METRICS --------------------
cross_base = detect_threshold_cross(y_base, danger)
cross_combo = detect_threshold_cross(y_combo, danger)

# Days delayed (numeric + display)
days_delay_num = 0
if cross_base is not None and cross_combo is not None:
    days_delay_num = int(cross_combo - cross_base)
    delayed_text = f"{days_delay_num} day(s)"
elif cross_base is not None and cross_combo is None:
    # Plan avoids threshold in window; treat as at least days_ahead delay
    days_delay_num = int(days_ahead)
    delayed_text = f">{days_ahead} day(s)"
elif cross_base is None and cross_combo is not None:
    # Baseline never crosses but plan does; negative delay (worse)
    days_delay_num = -int(cross_combo)
    delayed_text = f"{days_delay_num} day(s)"
else:
    delayed_text = "N/A"
    days_delay_num = 0

# Cumulative load reduction
cum_base  = float(np.sum(y_base))
cum_combo = float(np.sum(y_combo))
reduction_cum = pct_reduction(cum_base, cum_combo)

# -------------------- EFFECTIVENESS SCORE --------------------

def has_TR(applied):
    """True if the plan includes at least one transmission reducer."""
    return any(INTERVENTIONS[name]["r_mult"] < 1.0 for name in applied)

def has_SR(applied):
    """True if the plan includes at least one shedding reducer."""
    return any(INTERVENTIONS[name]["shed_mult"] < 1.0 for name in applied)

if applied_interventions and not over_budget and total_cost > 0:
    # Core biological impacts (never reward negative changes)
    delay_score = max(days_delay_num, 0)
    rna_score   = max(reduction_cum, 0.0)

    # Normalize so score is mostly between 0 and 100
    # Treat "delay by forecast window" and "80 percent reduction" as excellent
    delay_norm = min(delay_score, days_ahead) / float(days_ahead)
    rna_norm   = min(rna_score, 80.0) / 80.0

    # Delay is slightly more important than total load
    base = 60.0 * delay_norm + 40.0 * rna_norm

    # Synergy bonus if plan mixes TR and SR
    if has_TR(applied_interventions) and has_SR(applied_interventions):
        base *= 1.20
    elif has_TR(applied_interventions) or has_SR(applied_interventions):
        base *= 1.05

    # Small bonus for using more than one tool
    # This keeps a strong combo from looking worse than a strong single pick
    complexity_bonus = max(0, len(applied_interventions) - 1) * 4.0

    effectiveness_score = min(base + complexity_bonus, 100.0)
    eff_text = f"{effectiveness_score:.1f}"
else:
    effectiveness_score = None
    eff_text = "N/A"


# -------------------- LAYOUT --------------------
st.markdown(
    "<h2 style='color:#DDE6F1;'>Forecasting North Campus Wastewater</h2>",
    unsafe_allow_html=True
)
st.markdown(
    "<div class='subtle'>Forecast model: We fit a short-term exponential growth rate from the latest data, then simulate forward with a logistic-like curve. Transmission-focused interventions bend the slope; shedding-focused interventions lower the wastewater curve.</div>",
    unsafe_allow_html=True
)
st.write("")

st.markdown(
    """
**Your mission:**  
With a **budget of 10 points**, pick a combination of interventions that

1. **Delays** the curve from crossing the danger threshold (more **Days Until Danger Zone**), and  
2. **Decreases the total Viral Load in Sewage** (higher **Wastewater Viral Load Reduction**).
"""
)

col_plot, col_kpi = st.columns([7, 5])

# ---------- Plot ----------
with col_plot:
    fig = go.Figure()

    # Define “Today” (last observed date) and forecast end
    obs_end = pd.to_datetime(series_dates.max())
    base_max = pd.to_datetime(base_dates.max())

    # Shade forecast window (after Today)
    fig.add_shape(
        type="rect",
        x0=obs_end, x1=base_max, y0=0, y1=1,
        xref="x", yref="paper",
        fillcolor="rgba(255,255,255,0.05)",
        line=dict(width=0)
    )

    # Vertical “Today” line
    fig.add_shape(
        type="line",
        x0=obs_end, x1=obs_end, y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color="#AAB2BF", dash="dash", width=2)
    )
    fig.add_annotation(
        x=obs_end, y=1.02,
        xref="x", yref="paper",
        text="Today",
        showarrow=False,
        font=dict(color="#DDE6F1")
    )

    # Baseline future (no intervention)
    fig.add_trace(go.Scatter(
        x=base_dates, y=y_base, mode="lines",
        name="No Intervention",
        line=dict(color=HAS_RED, width=3, dash="dash"),
        line_shape=LINE_SHAPE
    ))

    # Plan future
    plan_active = bool(applied_interventions) and not over_budget
    color_combo = HAS_TEAL if plan_active else HAS_GRAY
    plan_opacity = 0.95 if plan_active else (0.35 if applied_interventions else 0.6)
    plan_dash = None if plan_active else "dot"
    plan_name = "Your Plan" if plan_active else ("Your Plan (over budget)" if over_budget else "No interventions selected")

    fig.add_trace(go.Scatter(
        x=base_dates, y=y_combo, mode="lines",
        name=plan_name,
        line=dict(color=color_combo, width=4, dash=plan_dash),
        line_shape=LINE_SHAPE,
        opacity=plan_opacity
    ))

    # Danger threshold line
    fig.add_shape(
        type="line",
        x0=base_dates.min(), x1=base_dates.max(),
        y0=danger, y1=danger,
        xref="x", yref="y",
        line=dict(color="#DDE6F1", dash="dot", width=2)
    )
    fig.add_annotation(
        x=base_dates.min(), y=danger,
        xref="x", yref="y",
        text="DANGER ZONE",
        showarrow=False, xanchor="left", yanchor="bottom",
        font=dict(color="#DDE6F1")
    )

    # Annotate crossing days
    if cross_base is not None:
        x_cb = pd.to_datetime(base_dates[cross_base])
        fig.add_annotation(
            x=x_cb, y=danger, xref="x", yref="y",
            text="Baseline crosses",
            showarrow=False, yshift=-12, font=dict(color="#DDE6F1")
        )
    if cross_combo is not None:
        x_cc = pd.to_datetime(base_dates[cross_combo])
        fig.add_annotation(
            x=x_cc, y=danger, xref="x", yref="y",
            text="Your plan crosses",
            showarrow=False, yshift=10, font=dict(color=HAS_TEAL)
        )
    if cross_base is not None and cross_combo is None and plan_active:
        fig.add_annotation(
            x=obs_end, y=danger * 1.02, xref="x", yref="y",
            text="Your plan avoids threshold in window",
            showarrow=False, font=dict(color=HAS_TEAL)
        )

    # Tight y-axis zoom around threshold (but include start and a bit above)
    all_vals = np.concatenate([series_vals, y_base, y_combo])
    y_min_candidate = min(all_vals.min(), danger * Y_MIN_FRACTION)
    y_min = max(0.0, y_min_candidate * 0.95)
    y_max_candidate = max(danger * 1.25, np.percentile(all_vals, 90))
    y_max_cap = max(danger * 1.3, danger * Y_CAP_MULT)
    y_max = min(y_max_candidate * 1.05, y_max_cap)
    if y_max <= danger * 1.1:
        y_max = danger * 1.3

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0b1220",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Date",
        yaxis_title="Viral Load Concentration (copies/mL)",
    )
    fig.update_yaxes(range=[y_min, y_max])

    st.plotly_chart(fig, use_container_width=True)

# ---------- KPI + BUTTON + STORY ----------
with col_kpi:
    st.markdown("<div class='metric-box'>", unsafe_allow_html=True)
    st.markdown("<h4 class='kpi-title'>Scenario Metrics</h4>", unsafe_allow_html=True)

    st.metric(
        label="Days Delayed",
        value=delayed_text,
        help="Higher = outbreak stays out of the danger zone for longer."
    )

    st.metric(
        label="Wastewater Viral Load Reduction (%)",
        value=f"{reduction_cum:.1f}%",
        help="Higher = less total viral RNA shed into sewage over the window."
    )

    if over_budget:
        st.error("Over budget. Interventions ignored until ≤ 10 points.")

    # Button to reveal effectiveness and the biological story
    if applied_interventions and not over_budget and effectiveness_score is not None:
        if st.button("Compute Effectiveness Score"):
            st.markdown(f"**Effectiveness Score:** {effectiveness_score:.1f}")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Intervention Menu (visible to students) ----------
st.markdown("## 🧰 Intervention Menu")
colA, colB = st.columns(2)
items = list(INTERVENTIONS.items())
half = (len(items) + 1) // 2

for col, chunk in zip((colA, colB), (items[:half], items[half:])):
    with col:
        for name, info in chunk:
            st.markdown(
                f"""
                <div class="card">
                  <h4>{name}</h4>
                  <div class="tag">Cost: {info['cost']} pts &nbsp;&nbsp;</div>
                  <div style="height:6px;"></div>
                  <div style="color:#DDE6F1;font-size:0.92rem;">{info['desc']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

# ---------- Presenter notes (kept minimal) ----------
with st.expander("ℹ️ Presenter Notes"):
    st.markdown("""
- Red dashed is **no intervention**. Teal is the **team's plan**.
- **Transmission reducers (TR)** mainly bend the slope → more **Days Until Danger Zone**.
- **Shedding reducers (SR)** mainly lower the curve → more **Wastewater Viral Load Reduction**.
- Effectiveness score rewards combos that both delay the danger zone and shrink total viral load,
  with a soft penalty for cost (so smart combos can beat a single cheap pick).
    """)