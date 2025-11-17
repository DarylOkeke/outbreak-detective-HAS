import streamlit as st

# ---------------------------------------------------
# PAGE CONFIG + THEME
# ---------------------------------------------------
st.set_page_config(
    page_title="Outbreak Detective",
    page_icon="🧠",
    layout="wide",
)

st.markdown(
    """
    <style>
      .stApp {
        background-color: #0b1220;
        color: #DDE6F1;
      }
      .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #FF4B4B;
        text-align: center;
      }
      .hero-subtitle {
        font-size: 1.15rem;
        color: #A7B4C3;
        text-align: center;
        max-width: 780px;
        margin: 0 auto 10px auto;
      }
      .pill {
        display:inline-block;
        padding:4px 10px;
        border-radius:999px;
        background:#182544;
        color:#A7B4C3;
        font-size:0.8rem;
        margin-bottom:6px;
      }
      .phase-card {
        background:#0f1a30;
        border-radius:14px;
        border:1px solid #182544;
        padding:14px 14px 12px 14px;
        height:100%;
      }
      .phase-card h3 {
        font-size:1.05rem;
        margin-bottom:4px;
        color:#DDE6F1;
      }
      .phase-tag {
        font-size:0.8rem;
        color:#A7B4C3;
        margin-bottom:6px;
      }
      .phase-body {
        font-size:0.9rem;
        color:#A7B4C3;
      }
      .game-box {
        background:#0f1a30;
        border-radius:14px;
        border:1px solid #182544;
        padding:16px;
        margin-top:12px;
      }
      .game-box h3 {
        font-size:1.05rem;
        margin-bottom:8px;
        color:#DDE6F1;
      }
      .game-box ul {
        padding-left:1.1rem;
        margin-bottom:0;
      }
      .game-box li {
        font-size:0.9rem;
        color:#A7B4C3;
        margin-bottom:4px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------
# HERO / TOP SECTION
# ---------------------------------------------------
st.markdown("<div class='hero-title'>OUTBREAK ALERT!!</div>", unsafe_allow_html=True)
st.markdown(
    """
    <p class='hero-subtitle'>
    Campus surveillance has detected early signs of an outbreak. 
    As the public health analyst team, your mission is to identify where it started and choose the most effective interventions to slow the spread and limit the impact.
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown("")
st.markdown("---")

# ---------------------------------------------------------------------
# HOW THE GAME WORKS
# ---------------------------------------------------------------------
st.markdown("""
## How the Game Works

### **Phase 1 — Detection**
🧬 Three independent datasets  
You’ll examine three different exhibits, each showing a different signal from around campus.  
Your team’s goal is simple: **figure out which zone the outbreak most likely began in** based on the evidence.

---

### **Phase 2 — Forecast**
📈 What happens if we do nothing?  
Once your team has chosen a starting zone, you’ll zoom in on that area and look at its **trendline over time**.  
This shows what the outbreak is projected to do **if no one intervenes**.

---

### **Phase 3 — Response**
🛠️ Choose your interventions  
You’ll get a list of possible interventions 
Your mission is to build a plan that **slows the outbreak and limits its impact**, all within a fixed budget.

---

### **What Happens After Your Plan**
🏆 Wrap-up discussion  
We reveal what the true starting zone was and walk through how different intervention choices changed the trajectory.

---

### **🎯 How to Play**
1. Go to the **Detection Phase** page and study all three exhibits.  
2. As a team, decide which zone is the most likely origin.  
3. Move to the **Forecast** tab to see what happens if no action is taken.  
4. Go to the **Response** tab to build your intervention plan.  
5. In the wrap-up, we’ll break down which clues were strongest and why certain interventions worked better.)
""")