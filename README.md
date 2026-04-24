# Outbreak Detective (HAS Event Game)

## Preface
This game was built for a Health Analytics Society event under tight time constraints.
I used Claude as a development assistant to help me move faster while designing and implementing the experience.

## Overview
Outbreak Detective is an interactive Streamlit game designed to help students think through real public health surveillance tradeoffs.
Players investigate multi-source outbreak signals, decide where an outbreak likely started, and then choose interventions under a fixed budget to compare outcomes.

## Learning Goals
- Practice interpreting surveillance data with different strengths and weaknesses.
- Distinguish early signals (wastewater/RNA) from lagging clinical indicators.
- Make intervention decisions with tradeoffs in cost, timing, and expected impact.

## Game Flow
1. Phase 1 (Detection): Teams review three exhibits and identify the likely origin zone.
2. Phase 2 (Forecast): Teams view projected trend behavior if no intervention is applied.
3. Phase 3 (Response): Teams build a response plan with a 10-point budget and evaluate impact.

## Tech Stack
- Python
- Streamlit
- NumPy
- Pandas
- Plotly
- SciPy (optional fallback logic exists if unavailable)
- Altair

## Project Structure
- `app.py`: Landing page and game instructions.
- `pages/Phase 1 .py`: Detection phase with exhibit stepping and facilitator controls.
- `pages/Phase 2 & 3.py`: Forecast + intervention simulator with score and KPI outputs.
- `pages/assets/`: Static visual assets used by the game.

## Quick Start
1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

4. Open the local URL Streamlit prints (typically `http://localhost:8501`).

## Notes
- The app is designed for facilitated group play during events/workshops.
- Phase files currently include spaces/symbols in filenames to match your existing setup.