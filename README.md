# SmartTrack Vision Suite

**SmartTrack Vision Suite** — real-time object tracking + analytics + Streamlit dashboard.  
Includes: detector → tracker → analytics pipeline and a Streamlit dashboard for visual exploration.

## Quick highlights
- YOLOv8-based detection (detector in separate repo/package)
- ByteTrack-style tracking pipeline
- Analytics engine: counts, trajectories, speeds, heatmaps, CSV/JSON outputs
- Streamlit dashboard for interactive inspection

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


