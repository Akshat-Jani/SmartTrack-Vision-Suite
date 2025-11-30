# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 13:25:56 2025

@author: acer
"""

#!/usr/bin/env python3
"""
measure_stats_v2.py
Reliable measurement script for ultralytics YOLO .pt files and model-names.
Produces resume_stats.json with:
 - model params
 - model file size & sha256
 - inference timings (mean/median/p95)
 - sampled peak RSS
Usage:
  python measure_stats_v2.py --model-path ./yolov8n.pt --input-size 640 640 --device cuda --iters 200
  python measure_stats_v2.py --model-name yolov8n --input-size 640 640 --device cpu --iters 100
"""
import argparse, json, os, time, statistics, hashlib, sys
import torch

try:
    import psutil
except Exception:
    psutil = None

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def load_ultralytics_model(path_or_name, device):
    """Always try ultralytics.YOLO(path_or_name) — this worked in your env."""
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError(f"ultralytics not installed or import failed: {e}")
    try:
        y = YOLO(path_or_name)
        # underlying torch Module is y.model
        y.model.to(device)
        return y.model, y  # return raw module and wrapper
    except Exception as e:
        raise RuntimeError(f"YOLO load failed for {path_or_name}: {e}")

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def bench_inference(model, device, input_size, warmup=20, iters=200, sample_rss_every=10):
    model.eval()
    bs = 1
    H, W = input_size
    dummy = torch.randn(bs, 3, H, W, device=device)
    times = []
    rss_samples = []
    with torch.no_grad():
        for _ in range(warmup):
            out = model(dummy) if callable(model) else model(dummy)
            if device.type.startswith("cuda"):
                torch.cuda.synchronize()
        for i in range(iters):
            t0 = time.perf_counter()
            out = model(dummy) if callable(model) else model(dummy)
            if device.type.startswith("cuda"):
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
            if psutil and (i % sample_rss_every == 0):
                try:
                    rss_samples.append(psutil.Process().memory_info().rss)
                except Exception:
                    pass
    times_sorted = sorted(times)
    idx95 = max(0, int(0.95 * len(times_sorted)) - 1)
    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "p95_ms": times_sorted[idx95],
        "iters": iters,
        "rss_peak_bytes_sampled": max(rss_samples) if rss_samples else None,
        "times_sample_ms": times[:10]
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=str, default=None)
    ap.add_argument("--model-name", type=str, default=None, help="yolov8n / yolov8l etc. (downloads if not present)")
    ap.add_argument("--input-size", type=int, nargs=2, default=[640,640])
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--out", type=str, default="stats.json")
    ap.add_argument("--skip-bench", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device)
    model = None
    model_file_size = None
    model_sha256 = None
    model_name_hint = args.model_name or args.model_path

    if args.model_path and os.path.exists(args.model_path):
        try:
            model_file_size = os.path.getsize(args.model_path)
        except Exception:
            model_file_size = None
        try:
            model_sha256 = sha256(args.model_path)
        except Exception:
            model_sha256 = None

    if not args.skip_bench:
        try:
            if args.model_path:
                model, wrapper = load_ultralytics_model(args.model_path, device)
            elif args.model_name:
                model, wrapper = load_ultralytics_model(args.model_name, device)
        except Exception as e:
            print("Model load warning:", e, file=sys.stderr)
            model = None

    result = {
        "run_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": str(device),
        "input_size": args.input_size,
        "model_file_size_bytes": model_file_size,
        "model_sha256": model_sha256,
        "model_name_hint": model_name_hint
    }

    if model is not None:
        try:
            result["num_params"] = count_params(model)
        except Exception as e:
            print("Param counting warning:", e, file=sys.stderr)
        try:
            bench = bench_inference(model, device, tuple(args.input_size), warmup=args.warmup, iters=args.iters)
            result["inference"] = bench
            result["inference"]["inference_only_fps"] = (1000.0 / bench["mean_ms"]) if bench.get("mean_ms") else None
        except Exception as e:
            print("Benchmark error:", e, file=sys.stderr)

    if psutil:
        try:
            result["rss_bytes_after_start"] = psutil.Process().memory_info().rss
        except Exception:
            pass

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print("Wrote", args.out)
    summary_keys = ("device","model_name_hint","model_file_size_bytes","num_params")
    print(json.dumps({k: result.get(k) for k in summary_keys}, indent=2))

if __name__ == "__main__":
    main()

