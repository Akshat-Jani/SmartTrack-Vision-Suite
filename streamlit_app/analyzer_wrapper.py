# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 11:25:15 2025

@author: acer
"""

# streamlit_app/analyzer_wrapper.py
"""
Wrapper to call src.smarttrack.analyze_all (analyze_all.py) functions from Streamlit.
Provides a convenience function `process_run(...)` that:
 - accepts paths to metrics/detections and out_dir
 - calls analyzer functions in the right order
 - returns a dict with DataFrames/summary dicts and artifact paths

Adjust DEFAULT_ANALYZER_MODULE if your analyzer import path differs.
"""

# streamlit_app/analyzer_wrapper.py

from pathlib import Path
from typing import Optional, Dict, Any
import importlib
import traceback
import pandas as pd

DEFAULT_ANALYZER_MODULE = "smarttrack.analyze_all"


def _import_analyzer(module_path: str = DEFAULT_ANALYZER_MODULE):
    return importlib.import_module(module_path)


def _safe_exists(path):
    return path and Path(path).exists()


def _maybe_add(artifacts, key, path):
    p = Path(path)
    if p.exists():
        artifacts[key] = str(p)


def process_run(
    metrics_path: Optional[str] = None,
    detections_path: Optional[str] = None,
    out_dir: Optional[str] = None,
    save_plots: bool = True,
    plot_n: int = 20,
    analyzer_module_path: str = DEFAULT_ANALYZER_MODULE
) -> Dict[str, Any]:

    logs = []
    artifacts = {}
    results = {}

    # Import analyzer
    try:
        mod = _import_analyzer(analyzer_module_path)
    except Exception as e:
        return {
            "metrics_df": pd.DataFrame(),
            "detections_df": pd.DataFrame(),
            "tracks_df": pd.DataFrame(),
            "metrics_summary": {},
            "combined_summary": {},
            "artifact_paths": {},
            "median_fps": None,
            "logs": f"[wrapper] Failed to import analyzer:\n{traceback.format_exc()}"
        }

    # Ensure out_dir
    if out_dir:
        out_dir = Path(out_dir)
    else:
        out_dir = Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts["out_dir"] = str(out_dir)

    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------
    metrics_df = pd.DataFrame()
    metrics_summary = {}

    try:
        if _safe_exists(metrics_path):
            metrics_df = mod.load_metrics(Path(metrics_path))
            try:
                metrics_df, classes = mod.expand_counts_json(metrics_df)
            except Exception:
                logs.append("[wrapper] expand_counts_json failed:\n" + traceback.format_exc())
                classes = []

            try:
                metrics_summary = mod.compute_metrics_summary(metrics_df, classes)
            except Exception:
                logs.append("[wrapper] compute_metrics_summary failed:\n" + traceback.format_exc())
                metrics_summary = {}

            results["metrics_df"] = metrics_df
            results["metrics_summary"] = metrics_summary
            logs.append("[wrapper] Metrics processed.")

            if save_plots:
                try:
                    mod.plot_time_series(metrics_df, out_dir)
                    _maybe_add(artifacts, "objects_time_series", out_dir / "objects_time_series.png")
                except Exception:
                    logs.append("[wrapper] plot_time_series failed:\n" + traceback.format_exc())



        else:
            logs.append("[wrapper] Metrics file missing — skipping metrics.")
            results["metrics_df"] = pd.DataFrame()
            results["metrics_summary"] = {}

    except Exception:
        logs.append("[wrapper] Metrics block failed:\n" + traceback.format_exc())
        results["metrics_df"] = pd.DataFrame()
        results["metrics_summary"] = {}

    # -----------------------------------------------------
    # DETECTIONS (This MUST be outside the metrics except block)
    # -----------------------------------------------------
    detections_df = pd.DataFrame()
    tracks_df = pd.DataFrame()
    median_fps = None

    try:
        if _safe_exists(detections_path):

            try:
                detections_df = mod.load_detections(Path(detections_path))
            except Exception:
                logs.append("[wrapper] load_detections failed:\n" + traceback.format_exc())
                detections_df = pd.DataFrame()

            if detections_df.empty:
                logs.append("[wrapper] detections_df empty — skipping tracking.")
            else:
                try:
                    detections_df = mod.compute_centers(detections_df)
                except Exception:
                    logs.append("[wrapper] compute_centers failed:\n" + traceback.format_exc())

                try:
                    tracks_df, median_fps = mod.per_track_metrics(detections_df)
                except Exception:
                    logs.append("[wrapper] per_track_metrics failed:\n" + traceback.format_exc())
                    tracks_df = pd.DataFrame()
                    median_fps = None

                if tracks_df is None:
                    tracks_df = pd.DataFrame()

                results["detections_df"] = detections_df
                results["tracks_df"] = tracks_df
                results["median_fps"] = median_fps

                logs.append("[wrapper] Detections processed.")

                if save_plots:
                    try:
                        mod.plot_trajectories(detections_df, out_dir, plot_n)
                        _maybe_add(artifacts, "trajectories", out_dir / "trajectories.png")
                    except Exception:
                        logs.append("[wrapper] plot_trajectories failed:\n" + traceback.format_exc())

                    try:
                        mod.plot_heatmap(detections_df, out_dir)
                        _maybe_add(artifacts, "heatmap", out_dir / "heatmap.png")
                    except Exception:
                        logs.append("[wrapper] plot_heatmap failed:\n" + traceback.format_exc())

        else:
            logs.append("[wrapper] Detections file missing — skipping detections.")
            results["detections_df"] = pd.DataFrame()
            results["tracks_df"] = pd.DataFrame()
            results["median_fps"] = None

    except Exception:
        logs.append("[wrapper] Detections block crashed:\n" + traceback.format_exc())
        results["detections_df"] = pd.DataFrame()
        results["tracks_df"] = pd.DataFrame()
        results["median_fps"] = None

    # -----------------------------------------------------
    # COMBINED SUMMARY
    # -----------------------------------------------------
    combined_summary = {}

    try:
        safe_metrics = results.get("metrics_summary", {}) or {}
        safe_tracks = results.get("tracks_df", pd.DataFrame())

        combined_summary = mod.overall_summary(
            safe_metrics,
            safe_tracks,
            results.get("median_fps"),
            out_dir
        )

        results["combined_summary"] = combined_summary
        _maybe_add(artifacts, "summary_run_csv", out_dir / "summary_run.csv")
        logs.append("[wrapper] Combined summary computed.")

    except Exception:
        logs.append("[wrapper] overall_summary failed:\n" + traceback.format_exc())
        results["combined_summary"] = {}

    # -----------------------------------------------------
    # Finalize results
    # -----------------------------------------------------
    results["artifact_paths"] = artifacts
    results["logs"] = "\n".join(logs)

    return results
