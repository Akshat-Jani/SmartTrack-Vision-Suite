# SmartTrack Vision Suite

**SmartTrack Vision Suite** — real-time object tracking + analytics + Streamlit dashboard.  
Includes: detector → tracker → analytics pipeline and a Streamlit dashboard for visual exploration.

## Quick highlights
- YOLOv8-based detection (detector in separate repo/package)
- ByteTrack-style tracking pipeline
- Analytics engine: counts, trajectories, speeds, heatmaps, CSV/JSON outputs
- Streamlit dashboard for interactive inspection

## Quick start (developer)
```bash
# create venv (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -e .
# if model not present:
scripts/download_model.sh

# run CLI demo
smarttrack --source demos/demo.mp4
# or run the Streamlit dashboard
streamlit run streamlit_app/streamlit_dashboard.py

Model weights
Large model weights are available in Releases (v1.0.0). Use scripts/download_model.sh to download into models/.
Notes about assets
Screenshots and high-res images are hosted externally (or in the Releases) to keep the repo lightweight. If you want to host images externally, replace demos/demo.gif in this README with the absolute URL.

Project structure
.
├─ src/
│  ├─ smarttrack/                # CLI + small wrappers
│  └─ smarttrack_tracker/        # tracker module
├─ streamlit_app/
├─ demos/
├─ docs/
├─ scripts/
├─ README.md
└─ pyproject.toml

License
MIT

