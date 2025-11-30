# SmartTrack Vision Suite

**SmartTrack Vision Suite** — Real-time object tracking + Advanced analytics Dashboard (YOLOv8 + ByteTrack + Streamlit)  
Includes: detector → tracker → analytics pipeline, and a Streamlit dashboard for visual exploration.

[Demo]  (assets/test.gif)

## Quick highlights
- YOLOv8-based detection (detector in separate repo/package)
- ByteTrack-style tracking pipeline
- Analytics engine: counts, trajectories, fps, heatmaps, CSV/JSON outputs
- Streamlit dashboard for interactive inspection

## 🔥 Features

- 🚗 YOLOv8-based object detection  
- 🧭 ByteTrack-style multi-object tracking  
- 📈 Detailed analytics:  
  - object counts  
  - trajectories  
  - speeds  
  - heatmaps  
  - time-series graphs  
- 📊 Complete Streamlit dashboard  
- 🎥 Support for any MP4 input video  
- 📁 Auto-generated CSV/JSON stats  
- 🖼 Modular codebase for detectors, trackers, and analytics  

## 📦 Installation

```bash
git clone https://github.com/Akshat-Jani/SmartTrack-Vision-Suite.git
cd SmartTrack-Vision-Suite

# Create environment (recommended)
python -m venv .venv
.\.venv\Scripts\activate       # Windows
```

## Install dependencies
```bash
pip install -r requirements.txt
```

## 🚀 Run the Streamlit Dashboard

```bash
streamlit run streamlit_app/streamlit_dashboard.py
```

## Project structure
├── src/
│ └── smarttrack/ # tracking + analytics Python modules
├── streamlit_app/ # Streamlit dashboard
├── scripts/ # helper scripts (optional)
├── configs/ # YOLO/ByteTrack configs
├── demos/ # demo videos/gifs
├── assets/ # images used in README
├── docs/ # project docs, explanation PDFs
├── requirements.txt
├── README.md
└── pyproject.toml


