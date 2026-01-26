# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 08:44:34 2025

@author: acer
"""

# streamlit_app/upload.py
"""
Helpers to save uploaded videos and run the SmartTrack pipeline (best-effort).
Functions:
 - save_video(uploaded_file, runs_base) -> Path (saved video path)
 - create_run_dir(runs_base, prefix="run") -> Path (new run directory)
 - run_tracker_on_video(video_path, run_dir, module_path="src.smarttrack.pipeline_tracker") -> dict
"""

from pathlib import Path
import subprocess
import traceback
import time
from typing import Dict

TMP_RUNS_BASE = Path("runs")  # default base for run outputs; adjust if needed

def create_run_dir(runs_base: Path = TMP_RUNS_BASE, prefix: str = "run") -> Path:
    """
    Create a new run directory under runs_base with a timestamp.-
    Returns the Path object for the new run.
    """
    runs_base = Path(runs_base)
    runs_base.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = runs_base / f"{prefix}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def save_video(uploaded_file, run_dir: Path) -> Path:
    """
    Save a Streamlit UploadedFile to run_dir and return the saved path.
    uploaded_file: file-like object from st.file_uploader
    run_dir: Path where the file will be written
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / uploaded_file.name
    with open(out_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return out_path

def run_tracker_on_video(video_path: Path, run_dir: Path, module_path: str = "src.smarttrack.pipeline_tracker", timeout=3600) -> Dict:
    """
   Minimal runner: always execute the tracker via subprocess (python -m ...).
   Returns diagnostic dict: {rc, stdout, stderr, error, run_dir}
   """
    res = {"rc": None, "stdout": "", "stderr": "", "error": None, "run_dir": str(run_dir), "--project": str(Path(run_dir).parent),
    "--name": str(Path(run_dir).name)}
    try:
        cmd = ["python", "-m", module_path, "--source", str(video_path), "--source", str(video_path),
    "--project", str(run_dir.parent), "--name", str(run_dir.name) ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False, timeout= timeout)
        res["rc"] = proc.returncode
        res["stdout"] = proc.stdout
        res["stderr"] = proc.stderr
        if proc.returncode != 0:
            res["error"] = f"Subprocess returned non-zero exit code {proc.returncode}"
    except subprocess.TimeoutExpired as te:
        res["error"] = f"Subprocess timed out after {timeout}s: {te}"
        res["stderr"] = getattr(te, "stderr", "") or res["stderr"]
        res["stdout"] = getattr(te, "stdout", "") or res["stdout"]
    except Exception as e:
        res["error"] = str(e) + "\n" + traceback.format_exc()
    return res
