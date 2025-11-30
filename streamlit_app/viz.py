# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 08:10:00 2025

@author: acer
"""

from typing import Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import math

# ---------------------------------------------------
# 1) FPS time series
# ---------------------------------------------------
def plot_fps_time(metrics_df: pd.DataFrame) -> go.Figure:
    """Interactive FPS vs frame chart."""
    if metrics_df is None or "fps" not in metrics_df.columns:
        fig = go.Figure()
        fig.update_layout(title="No FPS data available")
        return fig

    df = metrics_df.dropna(subset=["frame", "fps"]).sort_values("frame")
    fig = px.line(df, x="frame", y="fps", title="FPS over Frames")
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    return fig


# ---------------------------------------------------
# 2) Per-class stacked (from METRICS df)
# ---------------------------------------------------
def plot_per_class_stack_from_metrics(metrics_df: pd.DataFrame) -> go.Figure:
    """Stacked area chart of per-class counts over frames (from metrics_df)."""
    if metrics_df is None:
        fig = go.Figure()
        fig.update_layout(title="No per-class metrics available")
        return fig

    # All class columns = anything added by expand_counts_json
    ignore = {"frame", "fps", "num_tracks", "counts_json"}
    class_cols = [c for c in metrics_df.columns if c not in ignore]

    if not class_cols:
        fig = go.Figure()
        fig.update_layout(title="No per-class metrics available")
        return fig

    df = metrics_df[["frame"] + class_cols].copy().sort_values("frame")
    long_df = df.melt(id_vars="frame", value_vars=class_cols,
                      var_name="class", value_name="count")

    fig = px.area(long_df, x="frame", y="count", color="class",
                  title="Per-Class Counts Over Time")
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    return fig


# ---------------------------------------------------
# 3) Per-class stacked (from DETECTIONS df)
# ---------------------------------------------------
def plot_per_class_stack_from_detections(detections_df: pd.DataFrame) -> go.Figure:
    """Stacked area chart of per-class detection counts over frames."""
    if detections_df is None or "cls_name" not in detections_df.columns:
        fig = go.Figure()
        fig.update_layout(title="No per-class detections available")
        return fig

    df = detections_df.copy()
    
    # determine number of unique frames and choose bin size automatically
    n_frames = int(df["frame"].max() - df["frame"].min() + 1) if not df["frame"].isnull().all() else int(df["frame"].nunique())
    TARGET_POINTS = 200  # target number of x points for the chart (reasonable default)
    bin_size = max(1, math.ceil(n_frames / TARGET_POINTS))
   
    if bin_size > 1:
        df["bin"] = (df["frame"] // bin_size) * bin_size
        group = df.groupby(["bin", "cls_name"]).size().reset_index(name="count")
        x_col = "bin"
    else:
        group = df.groupby(["frame", "cls_name"]).size().reset_index(name="count")
        x_col = "frame"

    fig = px.area(group, x=x_col, y="count", color="cls_name",
                  title=f"Per-Class Counts (bin={bin_size})")
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    return fig


# ---------------------------------------------------
# 4) Center scatter (single frame or entire video)
# ---------------------------------------------------
def plot_centers_scatter(detections_df: pd.DataFrame, frame: Optional[int] = None) -> go.Figure:
    """Scatter of detection centers. If frame provided, filters to that frame."""
    if detections_df is None or "cx" not in detections_df.columns or "cy" not in detections_df.columns:
        fig = go.Figure()
        fig.update_layout(title="No center data available")
        return fig

    df = detections_df.copy()

    if frame is not None:
        df = df[df["frame"] == frame]

    df = df.dropna(subset=["cx", "cy"])
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No centers for the selected frame")
        return fig

    # Use color by ID if available
    color_col = "id" if "id" in df.columns else None
    hover_cols = ["frame"] + ([ "id" ] if "id" in df.columns else [])

    fig = px.scatter(df, x="cx", y="cy", color=color_col, hover_data=hover_cols)
    fig.update_yaxes(autorange="reversed")  # image coordinate system
    fig.update_layout(
        title=f"Centers (frame={frame})" if frame is not None else "Centers (all frames)",
        margin=dict(l=10, r=10, t=30, b=10)
    )
    return fig




