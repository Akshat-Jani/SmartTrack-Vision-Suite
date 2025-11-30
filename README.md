# SmartTrack Vision Suite

**SmartTrack Vision Suite** — Real-time object tracking + Advanced analytics Dashboard (YOLOv8 + ByteTrack + Streamlit)  
Includes: detector → tracker → analytics pipeline, and a Streamlit dashboard for visual exploration.

[Demo]  
(assets/test.gif)

## Quick highlights
- YOLOv8-based detection (detector in separate repo/package)
- ByteTrack-style tracking pipeline
- Analytics engine: counts, trajectories, fps, heatmaps, CSV/JSON outputs
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


