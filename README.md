WakeWatch - Satellite-to-suspect oil spill detection and vessel attribution.

Overview

WakeWatch is an automated pipeline that takes a satellite radar (SAR) image of a suspected marine oil spill and produces a ranked list of the vessels most likely responsible for it.

What it does:
- Detects and characterises oil slicks in Sentinel-1 SAR imagery (area, fragment count, age estimate)
- Hindcasts the slick backward - using real wind and ocean-current data - to estimate where and when it was released, and forecasts where it will spread next
- Reconstructs AIS vessel traffic around that estimated origin and scores every candidate vessel on proximity, behavioural anomaly, dwell time, and route deviation
- Presents the full chain of evidence on an interactive map-based console


## How to Run

Requires **Python 3.11**.

# 1. Create and activate a virtual environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies (CPU build by default, see requirements.txt)
pip install -r requirements.txt

# 3. Verify model weights load correctly(optional)
python run.py --check

# 4. Run the console (Streamlit UI)
python run.py
# -> http://localhost:8501

# Other run modes:
python run.py --api          # inference API only  -> http://localhost:8000/docs
python run.py --both         # console + API together


Project Structure:

WakeWatch/
|-- run.py
|-- requirements.txt
|-- .streamlit/
|   `-- config.toml
|-- multimodal_oil_spill_detection.ipynb
|-- models/
|   |-- best_sar_model.pth
|   |-- trajectory_lstm_baseline.pth
|   |-- ais_phase1_autoencoder.pth
|   `-- ais_phase1_scaler.joblib
|-- assets/
|   |-- ais/
|   `-- sar_samples/
|-- wakewatch/
|   |-- config.py
|   |-- registry.py
|   |-- weights.py
|   |-- inference/
|   |   |-- sar.py
|   |   |-- sar_vessel.py
|   |   |-- trajectory.py
|   |   |-- trajectory_domain_check.py
|   |   `-- ais.py
|   |-- drift.py
|   |-- drift_validation.py
|   |-- fusion.py
|   |-- attribution_ablation.py
|   |-- dark_vessel.py
|   |-- pre_spill_dark.py
|   |-- investigation.py
|   |-- synthetic_ais.py
|   |-- report_export.py
|   `-- scenario/
|       |-- wakashio.py
|       |-- real_ais.py
|       |-- sar_scenes.py
|       `-- indian_ocean_demo.py
|-- ui/
|   |-- app.py
|   |-- engine.py
|   |-- maps.py
|   |-- charts.py
|   |-- theme.py
|   `-- views/
|       |-- overview.py
|       |-- sar_view.py
|       |-- drift_view.py
|       |-- ais_view.py
|       |-- trajectory_view.py
|       |-- attribution_view.py
|       `-- new_investigation_view.py
|-- api/
|   `-- main.py
|-- docs/
|   |-- MODEL_NOTES.md
|   |-- model_interface_spec.md
|   |-- drift_validation.md
|   |-- attribution_weight_ablation.md
|   `-- VERSIONS.txt
`-- tests/