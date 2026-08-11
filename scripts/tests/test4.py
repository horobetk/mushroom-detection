#!/usr/bin/env python3
# Test 4 – Per-frame latency breakdown (preprocess / inference / postprocess).
#
# Measures millisecond-level timings on GPU (default) using Ultralytics speed
# statistics, after a short warm-up. Results support the performance tables in
# Chapter 5 of the thesis.
#
# Author: Kiril Horobets
# Warsaw University of Technology, 2026

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import torch
from ultralytics import YOLO

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from _paths import DEFAULT_MODEL, DEFAULT_OUTPUT  # noqa: E402


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and torch.cuda.is_available():
        return "0"
    return requested


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test 4: latency breakdown (preprocess / inference / NMS)."
    )
    parser.add_argument("--model", "-m", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    print("[INFO] TEST 4: Latency Breakdown")
    print(f"[INFO] Model:   {args.model}")
    print(f"[INFO] Device:  {device}")
    print(f"[INFO] Warmup:  {args.warmup}")
    print(f"[INFO] Runs:    {args.runs}")

    model = YOLO(str(args.model))

    # Synthetic RGB frame (H, W, 3) – matches Ultralytics numpy input path.
    dummy = np.random.randint(
        0, 255, (args.imgsz, args.imgsz, 3), dtype=np.uint8
    )

    print("Warming up...")
    for _ in range(args.warmup):
        model.predict(dummy, imgsz=args.imgsz, device=device, verbose=False)
    if torch.cuda.is_available() and device != "cpu":
        torch.cuda.synchronize()

    preprocess_ms: list[float] = []
    inference_ms: list[float] = []
    postprocess_ms: list[float] = []
    wall_ms: list[float] = []

    print("Measuring...")
    for _ in range(args.runs):
        t0 = time.perf_counter()
        results = model.predict(
            dummy, imgsz=args.imgsz, device=device, verbose=False
        )
        if torch.cuda.is_available() and device != "cpu":
            torch.cuda.synchronize()
        wall_ms.append((time.perf_counter() - t0) * 1000.0)

        speed = results[0].speed  # keys: preprocess, inference, postprocess
        preprocess_ms.append(float(speed.get("preprocess", 0.0)))
        inference_ms.append(float(speed.get("inference", 0.0)))
        postprocess_ms.append(float(speed.get("postprocess", 0.0)))

    def stats(values: list[float]) -> dict[str, float]:
        arr = np.asarray(values, dtype=np.float64)
        return {
            "mean_ms": float(arr.mean()),
            "std_ms": float(arr.std()),
            "min_ms": float(arr.min()),
            "max_ms": float(arr.max()),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
        }

    summary = {
        "device": device,
        "imgsz": args.imgsz,
        "runs": args.runs,
        "preprocess": stats(preprocess_ms),
        "inference": stats(inference_ms),
        "postprocess_nms": stats(postprocess_ms),
        "end_to_end_wall": stats(wall_ms),
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available() and device != "cpu"
            else "cpu"
        ),
    }

    out_json = args.output / "latency_breakdown.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nAverage latency per frame:")
    print(f"  Preprocess : {summary['preprocess']['mean_ms']:.2f} ms")
    print(f"  Inference  : {summary['inference']['mean_ms']:.2f} ms")
    print(f"  Postprocess: {summary['postprocess_nms']['mean_ms']:.2f} ms")
    print(f"  Wall clock : {summary['end_to_end_wall']['mean_ms']:.2f} ms")
    print(f"  GPU        : {summary['gpu_name']}")
    print(f"Saved: {out_json}")
    print("[OK] Test 4 finished")


if __name__ == "__main__":
    main()
