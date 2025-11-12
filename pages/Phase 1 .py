# pages/1_Detection_Phase.py
import os
from pathlib import Path
import numpy as np
import streamlit as st

# --------------------------------
# PAGE CONFIG
# --------------------------------
st.set_page_config(page_title="Detection Phase", page_icon="🧠", layout="wide")

# --------------------------------
# PATHS: HARD-CODED ASSETS
# --------------------------------
def _assets_dir() -> Path:
    try:
        here = Path(__file__).parent
    except NameError:
        here = Path.cwd()
    return (here / "assets").resolve()

ASSETS = _assets_dir()
MAP_PATH = ASSETS / "map.png"
RNA_PATH = ASSETS / "Viral RNA in Sewage.png"
POS_PATH = ASSETS / "Rapid Test Positivity by Campus Zone.png"
ER_PATH  = ASSETS / "ER Visits by Campus Zone.png"

def _exists(p: Path) -> bool:
    return p is not None and p.exists() and p.is_file()

# --------------------------------
# SIDEBAR CONTROLS
# --------------------------------
st.sidebar.markdown("### Presenter Controls")
presenter_mode = st.sidebar.toggle("Presenter mode (step through exhibits)", value=True)
show_missing_note = st.sidebar.checkbox("Show 'missing sampling dates' note", value=True)
show_host_notes = st.sidebar.toggle("Show host notes on screen", value=False)

# --------------------------------
# SESSION: EXHIBIT STEPPER
# --------------------------------
if "step" not in st.session_state:
    st.session_state.step = 1  # 1=A(RNA), 2=B(Pos), 3=C(ER), 4=Final

def goto(delta):
    st.session_state.step = int(np.clip(st.session_state.step + delta, 1, 4))

# --------------------------------
# HEADER
# --------------------------------
st.title("🧠 Detection Phase")
st.markdown("""
**Mission:** You will see three exhibits. Use clues from each to decide where the outbreak actually started.  
Note: **Think about which dataset most reliably tells us this**.
""")

# Orientation map (if present)
if _exists(MAP_PATH):
    st.image(str(MAP_PATH), caption="Campus zones map (orientation)", use_container_width=True)
st.divider()

with st.expander("Legend (quick mental model)", expanded=False):
    st.markdown(
        "🧬 **Viral RNA in Sewage** = early, population-level signal\n"
        "🧪 **% Positive Tests** = behavior & access matter; small sample sizes can mislead\n"
        "🏥 **ER Visits** = late clinical signal (often lags infections)"
    )

# --------------------------------
# NAV
# --------------------------------
if presenter_mode:
    col_nav = st.columns([1, 6, 1])
    with col_nav[0]:
        st.button("⬅️ Prev", on_click=lambda: goto(-1), use_container_width=True)
    with col_nav[2]:
        st.button("Next ➡️", on_click=lambda: goto(1), use_container_width=True)

def exhibit_title(step):
    return {
        1: "Exhibit A — Viral RNA in Sewage by Campus Zone",
        2: "Exhibit B — % of People Testing Positive by Zone",
        3: "Exhibit C — ER Visits by Campus Zone",
        4: "Final — Put it Together"
    }[step]

st.subheader(exhibit_title(st.session_state.step))

# --------------------------------
# EXHIBITS
# --------------------------------
if st.session_state.step == 1:
    st.caption("Daily viral RNA measured from sewage samples by campus zone.")
    if _exists(RNA_PATH):
        st.image(str(RNA_PATH), use_container_width=True)
    else:
        st.error("❌ Viral RNA image not found in /pages/assets/")
    st.markdown("**What to look for:** Earlier **sustained** rise by zone. One-day spikes may be noise.")
    if show_missing_note:
        st.markdown("- Note: Some sampling dates are missing (gaps are expected in real monitoring).")
    if show_host_notes:
        st.info("Host notes: RNA can lead clinical data by days; watch timing more than peak height.")

elif st.session_state.step == 2:
    st.caption("Percent of daily rapid antigen tests returning positive per zone (sample size matters).")
    if _exists(POS_PATH):
        st.image(str(POS_PATH), use_container_width=True)
    else:
        st.error("❌ Test positivity image not found in /pages/assets/")
    st.markdown("**What to look for:** High positivity **with enough tests** is meaningful. Small-n spikes can mislead.")
    if show_host_notes:
        st.info("Host notes: Surge testing can lower positivity while cases rise; focus on trend + sample size.")

elif st.session_state.step == 3:
    st.caption("Daily ER visits for respiratory-like symptoms by zone (or overall time series).")
    if _exists(ER_PATH):
        st.image(str(ER_PATH), use_container_width=True)
    else:
        st.error("❌ ER Visits image not found in /pages/assets/")
    st.markdown("**What to look for:** ER tends to **lag** infections. Late heat doesn’t always show the **origin**.")
    if show_host_notes:
        st.info("Host notes: Use ER as confirmatory evidence, not early detection.")

else:
    st.caption("Use all exhibits together. Think timing (biology), reliability (stats), and localization (zones).")
    c1, c2 = st.columns(2)
    with c1:
        if _exists(RNA_PATH):
            st.image(str(RNA_PATH), caption="Exhibit A — Viral RNA (by zone)", use_container_width=True)
        if _exists(POS_PATH):
            st.image(str(POS_PATH), caption="Exhibit B — % Positive (by zone)", use_container_width=True)
    with c2:
        if _exists(ER_PATH):
            st.image(str(ER_PATH), caption="Exhibit C — ER Visits (by zone)", use_container_width=True)
        if _exists(MAP_PATH):
            st.image(str(MAP_PATH), caption="Campus Map (reference)", use_container_width=True)

    st.divider()
    st.markdown("### 📝 Final Questions")
    st.markdown("1) **Where did the outbreak likely start?**  \n2) **(Tie-breaker)** Which exhibit gave the **first real clue**, and why?")

# --------------------------------
# OPTIONAL HINTS & NOTES
# --------------------------------
st.divider()
with st.expander("💡 Nudge (optional)"):
    st.markdown(
        "- Visual height isn’t always the earliest clue.\n"
        "- Small sample sizes can inflate percentages.\n"
        "- Clinical data confirms; wastewater often leads."
    )

st.text_area("Team notes (not submitted):", height=100, placeholder="Jot quick evidence or timestamps here…")