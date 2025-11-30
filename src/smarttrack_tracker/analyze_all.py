#!/usr/bin/env python3
#!/usr/bin/env python3
"""
analyze_all.py - Combined analyzer for SmartTrack outputs

Processes:
 - metrics.csv (frame-summary): frame,fps,num_tracks,counts_json
 - detections.csv (detailed): frame,id,cls_id,cls_name,conf,x1,y1,x2,y2,fps,video_width,video_height

Outputs (into --out folder):
 - summary_run.csv          (combined run-level summary)
 - objects_time_series.png  (if --save-plots and metrics provided)
 - class_totals.png         (if --save-plots and metrics provided)
 - per_track_summary.csv    (if --save-plots or detections provided)
 - trajectories.png         (top N tracks - if --save-plots and detections provided)
 - heatmap.png              (if --save-plots and detections provided)

Usage examples:
  # metrics only, print summary
  python analyze_all.py --metrics runs/pipeline_video/metrics.csv

  # metrics + detections, save plots & outputs
  python analyze_all.py --metrics runs/pipeline_video/metrics.csv \
      --detections runs/pipeline_video/detections.csv \
      --out runs/pipeline_video/analysis --save-plots --plot-n 30 --assume-fps 30
"""

from pathlib import Path
import argparse
import sys
import json


import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt

# -------------------------
# ---- Metrics functions
# -------------------------
def load_metrics(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded metrics: {len(df)} frames from {csv_path}")
    return df

def expand_counts_json(df: pd.DataFrame):
    if "counts_json" not in df.columns:
        print("[WARN] counts_json not found in metrics; skipping expansion.")
        return df, []
    parsed = []
    classes = set()
    for cell in df["counts_json"].fillna(""):
        try:
            d = json.loads(cell) if isinstance(cell, str) and cell.strip() else {}
        except Exception:
            d = {}
        parsed.append(d)
        classes.update(d.keys())
    classes = sorted(classes)
    for cls in classes:
        df[cls] = [int(d.get(cls, 0)) for d in parsed]
    return df, classes

def compute_metrics_summary(df: pd.DataFrame, classes: list):
    s = {}
    s["total_frames"] = int(len(df))
    s["avg_fps"] = float(df["fps"].mean()) if "fps" in df.columns else None
    s["fps_std"] = float(df["fps"].std()) if "fps" in df.columns else None
    s["avg_objects_per_frame"] = float(df["num_tracks"].mean()) if "num_tracks" in df.columns else None
    s["max_objects_in_frame"] = int(df["num_tracks"].max()) if "num_tracks" in df.columns else None
    s["class_totals"] = {cls: int(df[cls].sum()) for cls in classes} if classes else {}
    if s["fps_std"] is None:
        s["fps_stability"] = "Unknown"
    else:
        s["fps_stability"] = "Stable" if s["fps_std"] < 1.0 else "Unstable"
    avg = s["avg_objects_per_frame"] or 0.0
    s["scene_density"] = "Low" if avg < 2 else "Medium" if avg < 5 else "High"
    return s

def plot_time_series(df: pd.DataFrame, out_dir: Path):
    if "frame" not in df.columns or "num_tracks" not in df.columns:
        print("[WARN] Missing frame/num_tracks; skipping time series plot.")
        return
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df["frame"], df["num_tracks"], linewidth=1)
    ax.set_xlabel("Frame"); ax.set_ylabel("Objects"); ax.set_title("Objects per Frame")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    out_path = out_dir / "objects_time_series.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved time-series plot to: {out_path}")


# -------------------------
# ---- Detections functions
# -------------------------
def load_detections(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded detections: {len(df)} rows from {csv_path}")
    required = ["frame","id","x1","y1","x2","y2"]
    for c in required:
        if c not in df.columns:
            raise SystemExit(f"[ERROR] Required column missing in detections CSV: {c}")
    return df

def compute_centers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cx"] = (df["x1"] + df["x2"]) / 2.0
    df["cy"] = (df["y1"] + df["y2"]) / 2.0
    df["id"] = df["id"].fillna(-1).astype(int)
    return df

def per_track_metrics(df: pd.DataFrame):
    tracks = []
    fps_series = df["fps"] if "fps" in df.columns else None
    median_fps = float(fps_series.median())
    grouped = df.groupby("id")
    for tid, g in grouped:
        if tid < 0:
            continue
        g = g.sort_values("frame")
        frames = g["frame"].values.astype(int) # array
        cx = g["cx"].values # array
        cy = g["cy"].values
        confs = g["conf"].values if "conf" in g.columns else np.full(len(g), np.nan)
        if len(cx) > 1:
            dx = np.diff(cx); dy = np.diff(cy) #computes consecutive differences:
            dists = np.sqrt(dx*dx + dy*dy)
            total_dist = float(np.nansum(dists))
            if median_fps and median_fps > 0:
                duration_s = (frames[-1] - frames[0]) / median_fps if frames[-1] != frames[0] else (len(frames)-1)/median_fps
                avg_speed = total_dist / max(1e-9, duration_s)
            else:
                duration_s = None; avg_speed = None
        else:
            total_dist = 0.0; duration_s = 0.0; avg_speed = 0.0
        mean_conf = float(np.nanmean(confs)) if len(confs)>0 else None
        track_info = {
            "id": int(tid),
            "start_frame": int(frames[0]),
            "end_frame": int(frames[-1]),
            "num_frames": int(len(frames)),
            "total_distance (in pixels)": round(total_dist,3),
            "duration_s": round(duration_s,3) if duration_s is not None else "",
            "avg_speed (pixel/sec.)": round(avg_speed,3) if avg_speed is not None else "",
            "mean_conf": round(mean_conf,3) if mean_conf is not None and not np.isnan(mean_conf) else "",
           
        }
        tracks.append(track_info)
    tracks_df = pd.DataFrame(tracks).sort_values("num_frames", ascending=False)
    return tracks_df, median_fps

def save_tracks_csv(tracks_df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "per_track_summary.csv"
    tracks_df.to_csv(out_file, index=False)
    print(f"[INFO] Saved per-track summary to: {out_file}")

def plot_trajectories(df: pd.DataFrame, out_dir: Path, plot_n):
    # choose top tracks by unique frames
    lengths = df.groupby("id")["frame"].count().sort_values(ascending=False)
    top_ids = lengths.index.tolist()[:plot_n]
    plt.figure(figsize=(6,6))
    ax = plt.gca()
    for tid in top_ids:
        g = df[df["id"]==tid].sort_values("frame")
        if len(g)==0: continue
        ax.plot(g["cx"], g["cy"], marker="o", markersize=2, linewidth=1, label=f"id {int(tid)}")
    ax.set_xlabel("cx (pixels)"); ax.set_ylabel("cy (pixels)")
    ax.set_title(f"Top {len(top_ids)} Trajectories (pixels)")
    ax.invert_yaxis()
    ax.legend(fontsize="small", ncol=2)
    plt.tight_layout()
    out_path = out_dir / "trajectories.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved trajectories plot to: {out_path}")

def plot_heatmap(df: pd.DataFrame, out_dir: Path, bins=(200,120)):
    xs = df["cx"].values; ys = df["cy"].values
    if len(xs)==0:
        print("[WARN] No points for heatmap; skipping.")
        return
    h, xedges, yedges = np.histogram2d(xs, ys, bins=bins)
    plt.figure(figsize=(6,5))
    plt.gca().invert_yaxis()
    plt.imshow(h.T, origin="upper", cmap="hot", interpolation="nearest", aspect="auto")
    plt.title("Detection Center Heatmap (pixels)")
    plt.xlabel("X bin"); plt.ylabel("Y bin")
    plt.colorbar(label="counts")
    plt.tight_layout()
    out_path = out_dir / "heatmap.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved heatmap to: {out_path}")

def overall_summary(metrics_summary, tracks_df, median_fps, out_dir: Path):
    # combine metrics & detection insights into one run-level summary
    s = {}
    # include metrics-level fields if available
    if metrics_summary:
        s.update({
            "total_frames": metrics_summary.get("total_frames",""),
            "avg_fps": metrics_summary.get("avg_fps",""),
            "fps_std": metrics_summary.get("fps_std",""),
            "fps_stability": metrics_summary.get("fps_stability",""),
            "avg_objects_per_frame": metrics_summary.get("avg_objects_per_frame",""),
            "max_objects_in_frame": metrics_summary.get("max_objects_in_frame",""),
            "scene_density": metrics_summary.get("scene_density","")
        })
    # detection-level fields
    s["total_tracks"] = int(len(tracks_df)) 
    s["median_track_length_frames"] = float(tracks_df["num_frames"].median()) 
    s["median_fps_used"] = float(median_fps) 
    avg_speed = tracks_df["avg_speed (pixel/sec.)"].replace("", np.nan).dropna().astype(float).mean() 
    s["mean_track_speed_(pixel/sec.)"] = round(float(avg_speed),3) 
    # save final combined run summary
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "summary_run.csv"
    pd.DataFrame([s]).to_csv(out_file, index=False)
    print(f"[INFO] Saved combined summary to: {out_file}")
    return s

# -------------------------
# ---- CLI / main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="SmartTrack combined analyzer (metrics + detections)")
    parser.add_argument("--metrics", default=None, help="path to metrics.csv (frame-summary)")
    parser.add_argument("--detections", default=None, help="path to detections.csv (per-detection rows)")
    parser.add_argument("--out", default=None, help="output folder to save results (required for saving)")
    parser.add_argument("--save-plots", action="store_true", help="save plots to disk (requires --out)")
    parser.add_argument("--plot-n", type=int, default=20, help="top-N tracks to plot")
    parser.add_argument("--assume-fps", type=float, default=None, help="fallback FPS if fps missing in detections")
    parser.add_argument("--skip-detections", action="store_true", help="skip processing detections even if provided")
    args = parser.parse_args()

    metrics_df = None; metrics_summary = None
    detections_df = None; tracks_df = None; median_fps = None

    # If no outputs requested and no metrics/detections provided -> show help
    if not args.metrics and not args.detections:    
        print("[ERROR] Provide at least --metrics or --detections. Use --help for usage.")
        sys.exit(1)

    # out_dir validation (required when --save-plots or when we must write outputs)
    out_dir = Path(args.out) if args.out else None
    if args.save_plots and out_dir is None:
        print("[ERROR] --save-plots requires --out to be specified.")
        sys.exit(1)
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    # --- Metrics processing (if provided)
    if args.metrics:
        metrics_path = Path(args.metrics)
        if not metrics_path.exists():
            print(f"[ERROR] metrics file not found: {metrics_path}"); sys.exit(1)
        metrics_df = load_metrics(metrics_path)
        metrics_df, classes = expand_counts_json(metrics_df)
        metrics_summary = compute_metrics_summary(metrics_df, classes)
        print("\n=== METRICS SUMMARY ===")
        for k,v in metrics_summary.items(): # prints all the details/key value pairs except class totals
            if k != "class_totals":
                print(f"{k:30s}: {v}")
        if metrics_summary.get("class_totals"): # prints the class total dictionary
            print("Per-class totals:")
            for cls,cnt in metrics_summary["class_totals"].items():
                print(f"  {cls:>12s} : {cnt}")
        # save metrics plots if requested
        if args.save_plots and out_dir:
            plot_time_series(metrics_df, out_dir)

    # --- Detections processing (if provided and not skipped)
    if args.detections:
        det_path = Path(args.detections)
        if not det_path.exists():
            print(f"[ERROR] detections file not found: {det_path}"); sys.exit(1)
        detections_df = load_detections(det_path)
        detections_df = compute_centers(detections_df)
        tracks_df, median_fps = per_track_metrics(detections_df)
        if out_dir:
            save_tracks_csv(tracks_df, out_dir)
        if args.save_plots and out_dir:
            # trajectory & heatmap
            plot_trajectories(detections_df, out_dir, plot_n=args.plot_n)
            plot_heatmap(detections_df, out_dir)

    # --- Combined summary
    combined_summary = overall_summary(metrics_summary, tracks_df, median_fps, out_dir if out_dir else Path("."))
    print("\n=== COMBINED RUN SUMMARY ===")
    for k,v in combined_summary.items():
        print(f"{k:30s}: {v}")

if __name__ == "__main__":
    main()
