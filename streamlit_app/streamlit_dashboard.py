# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 16:57:39 2025

@author: acer
"""

# streamlit_app/streamlit_dashboard.py

import streamlit as st
from pathlib import Path

from upload import create_run_dir, save_video, run_tracker_on_video
from analyzer_wrapper import process_run
import viz as viz

# -------------------------- Config --------------------------
st.set_page_config(
    page_title="SmartTrack Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

RUNS_BASE = Path("runs")
DEMO_VIDEO = Path("data/demo.mp4")

if "last_run_dir" not in st.session_state:
    st.session_state["last_run_dir"] = None

if "last_analysis" not in st.session_state:
    st.session_state["last_analysis"] = None

st.title("SmartTrack — Interactive Dashboard")
st.write("Upload video → Run Tracker → Analyze → Explore.")


# -------------------------- Sidebar --------------------------
with st.sidebar:
    st.header("Options")

    save_plots = st.checkbox("Save analyzer PNGs", value=True)

    show_files = st.checkbox("Show run folder file list", value=True)

    st.markdown("---")
    st.caption("Runs are stored in:")
    st.code(str(RUNS_BASE.resolve()))


# -------------------------- Tabs --------------------------
tab_upload, tab_runs, tab_about = st.tabs(["Upload & Run", "Explore Runs", "About"])


# ==========================================================
#                     TAB 1: Upload & Run
# ==========================================================
with tab_upload:
    st.header("Upload Video")

    uploaded_file = None

    # ----- USER UPLOAD -----
    uploaded_file = st.file_uploader("Upload video", 
                                     type=["mp4", "mov", "avi"],)

    if uploaded_file and st.button("Run tracker"):
        run_dir = create_run_dir(RUNS_BASE)
        saved_path = save_video(uploaded_file, run_dir)
    
        # DEBUG: show paths before running tracker
        st.write("Saved uploaded file to:", str(saved_path))
        st.write("Run dir (before tracker):", str(run_dir))
        try:
            before = sorted([p.name for p in run_dir.iterdir()])
        except Exception:
            before = []
        st.write("Files already in run_dir (before):", before)
    
        # Run tracker and capture logs
        with st.spinner("Running tracker..."):
            logs = run_tracker_on_video(saved_path, run_dir)
    
        # show subprocess stdout/stderr and return value
        st.subheader("Tracker subprocess output")
        st.text_area("stdout", logs.get("stdout", ""), height=160)
        st.text_area("stderr", logs.get("stderr", ""), height=160)
        st.write("Tracker error field:", logs.get("error", None))
        st.write("Tracker rc:", logs.get("rc", None))
        st.write("Run dir (after tracker):", str(run_dir))
    
        # DEBUG: list files in run_dir and the 'track' folder (if exists)
        try:
            after = sorted([p.name for p in run_dir.iterdir()])
        except Exception:
            after = []
        st.write("Files in run_dir (after):", after)
    
        track_folder = Path("track")  # common ultralytics default
        if track_folder.exists():
            try:
                track_list = [(p.name, sorted([q.name for q in p.iterdir()]) if p.is_dir() else []) for p in track_folder.iterdir()]
            except Exception:
                track_list = []
            st.write("Contents of track/ (top-level):", track_list)
        else:
            st.write("No 'track/' folder present at repo root.")
    
        if logs.get("error"):
            st.error("Tracker error: " + str(logs["error"]))
        else:
            st.success("Tracker finished.")
            st.session_state["last_run_dir"] = str(run_dir)


    # ----- ANALYZE LAST RUN -----

    # ensure flag exists to avoid KeyError
    if "analyzing" not in st.session_state:
        st.session_state["analyzing"] = False
    
    if st.session_state.get("last_run_dir"):
        st.markdown("---")
        st.write("Last run:", st.session_state["last_run_dir"])
    
        run_dir = Path(st.session_state["last_run_dir"])
        metrics = run_dir / "metrics.csv"      # your filenames (you said you'll fix names later)
        dets = run_dir / "detections.csv"
    
        if st.button("Analyze last run") and not st.session_state.get("analyzing", False):
            st.session_state["analyzing"] = True
            # clear any previous short log
            st.session_state["analysis_logs"] = ""
            try:
                with st.spinner("Analyzing results..."):
                    res = process_run(
                        metrics_path=str(metrics) if metrics.exists() else None,
                        detections_path=str(dets) if dets.exists() else None,
                        out_dir=str(run_dir),
                        save_plots=True,
                        plot_n=20,
                    )
    
                st.session_state["last_analysis"] = res
                st.success("Analysis completed. Go to **Explore Runs** tab.")
                # capture any logs returned by the wrapper (optional)
                if isinstance(res, dict) and res.get("logs"):
                    st.session_state["analysis_logs"] = res.get("logs")
            except Exception as e:
                # minimal error reporting + save to session_state for inspection
                err_txt = f"Analysis failed: {e}"
                st.session_state["analysis_logs"] = err_txt
                st.error(err_txt)
            finally:
                st.session_state["analyzing"] = False
    
        # show logs if present
        if st.session_state.get("analysis_logs"):
            st.subheader("Analyzer logs")
            st.text_area("logs", st.session_state["analysis_logs"], height=180)


# ==========================================================
#                      TAB 2: Explore Runs
# ==========================================================
with tab_runs:
    st.header("Explore Previous Runs")

    # List runs
    if RUNS_BASE.exists():
        run_dirs = sorted(RUNS_BASE.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
        run_choices = [str(p) for p in run_dirs]
    else:
        run_choices = []

    if not run_choices:
        st.info("No runs found yet. Upload a video first.")
    else:
        chosen = st.selectbox("Choose run:", run_choices, index=0)
        chosen_dir = Path(chosen)

        # Load analyzer results
        if st.button("Analyze selected run"):
            metrics = chosen_dir / "metrics.csv"
            dets = chosen_dir / "detections.csv"

            with st.spinner("Analyzing..."):
                res = process_run(
                    metrics_path=str(metrics) if metrics.exists() else None,
                    detections_path=str(dets) if dets.exists() else None,
                    out_dir=str(chosen_dir),
                    save_plots=True,
                    plot_n=20,
                )

            st.session_state["last_analysis"] = res
            st.success("Finished.")

    res = st.session_state.get("last_analysis")

    if res:
        st.subheader("Summary")

        if res.get("combined_summary"):
            cs = res["combined_summary"]
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Avg FPS", cs.get("avg_fps", "N/A"))
                st.metric("FPS Std", cs.get("fps_std", "N/A"))
                st.metric("FPS Stability", cs.get("fps_stability", "N/A"))
            with col2:
                st.metric("Avg Objects/Frame", cs.get("avg_objects_per_frame", "N/A"))
                st.metric("Total Tracks", cs.get("total_tracks", "N/A"))
                st.metric("Scene Density", cs.get("scene_density", "N/A"))

        # ---------------- PNG ARTIFACTS ----------------
        st.markdown("### Static Artifacts (from Analyzer)")
        for name, p in res.get("artifact_paths", {}).items():
            if p and p.endswith(".png") and Path(p).exists():
                st.image(p, caption=name, width="content")

        # ---------------- INTERACTIVE CHARTS ----------------
        st.markdown("### Interactive Charts")

        # 1) FPS time-series
        if res.get("metrics_df") is not None:
            st.plotly_chart(viz.plot_fps_time(res["metrics_df"]), width="content")

        # 2) Per-class stacked — prefer metrics_df
        if res.get("metrics_df") is not None:
            st.plotly_chart(viz.plot_per_class_stack_from_metrics(res["metrics_df"]), 
                            width="content")
        elif res.get("detections_df") is not None:
            st.plotly_chart(viz.plot_per_class_stack_from_detections(res["detections_df"]),width="content")

        # 3) Centers scatter (all frames)
        if res.get("detections_df") is not None:
            st.markdown("#### Centroids (all frames)")
            st.plotly_chart(viz.plot_centers_scatter(res["detections_df"]), 
                            width="content")
            
            # Frame selector
            max_frame = int(res["detections_df"]["frame"].max())
            frame_sel = st.slider("Frame", 0, max_frame, 0)
            st.plotly_chart(viz.plot_centers_scatter(res["detections_df"], frame=frame_sel), 
                            width="content")

        # ---------------- FILE VIEW ----------------
        if show_files:
            st.markdown("### Run Folder Contents")
            try:
                files = sorted([f.name for f in Path(res["artifact_paths"]["out_dir"]).iterdir()])
                st.code("\n".join(files))
            except:
                st.write("Could not list files.")

# ==========================================================
#                          TAB 3: About
# ==========================================================
with tab_about:
    st.header("About SmartTrack Dashboard")
    st.write("""
    - Upload → Track → Analyze → Visualize.
    - Analyzer produces static artifacts (.png, .csv).
    - Dashboard provides interactive insights (Plotly).
    - PNGs are kept for GitHub, offline reports, screenshots.
    """)
