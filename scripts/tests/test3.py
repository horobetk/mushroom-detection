#!/usr/bin/env python3
# Test 3 – Confidence threshold sweep (F1 justification)
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from _paths import (  # noqa: E402
    ANDROID_CONF_THRESHOLD,
    DEFAULT_DATA,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT,
)


def f1_score(precision: float, recall: float) -> float:
    # Harmonic mean of precision and recall.
    denom = precision + recall
    if denom <= 0:
        return 0.0
    return 2.0 * precision * recall / denom


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test 3: confidence threshold sweep with F1-score curve."
    )
    parser.add_argument("--model", "-m", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", "-d", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf-min", type=float, default=0.05)
    parser.add_argument("--conf-max", type=float, default=0.95)
    parser.add_argument("--conf-step", type=float, default=0.05)
    parser.add_argument(
        "--highlight",
        type=float,
        default=ANDROID_CONF_THRESHOLD,
        help="Confidence value to mark on the plot (Android default).",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    thresholds = np.round(
        np.arange(args.conf_min, args.conf_max + 1e-9, args.conf_step), 4
    )

    print("[INFO] TEST 3: Confidence Threshold Sweep")
    print(f"[INFO] Model:      {args.model}")
    print(f"[INFO] Range:      {args.conf_min:.2f} .. {args.conf_max:.2f} "
          f"(step {args.conf_step})")
    print(f"[INFO] Highlight:  {args.highlight:.3f} (Android)")

    model = YOLO(str(args.model))
    rows: list[dict[str, float]] = []

    for conf in thresholds:
        metrics = model.val(
            data=str(args.data),
            split="test",
            conf=float(conf),
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )
        p = float(metrics.box.mp)
        r = float(metrics.box.mr)
        map50 = float(metrics.box.map50)
        f1 = f1_score(p, r)
        rows.append(
            {
                "conf": float(conf),
                "precision": p,
                "recall": r,
                "f1": f1,
                "map50": map50,
            }
        )
        print(
            f"Conf: {conf:.2f} | P: {p:.3f} | R: {r:.3f} | "
            f"F1: {f1:.3f} | mAP50: {map50:.3f}"
        )

    csv_path = args.output / "threshold_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["conf", "precision", "recall", "f1", "map50"]
        )
        writer.writeheader()
        writer.writerows(rows)

    confs = [r["conf"] for r in rows]
    f1s = [r["f1"] for r in rows]
    ps = [r["precision"] for r in rows]
    rs = [r["recall"] for r in rows]

    best = max(rows, key=lambda r: r["f1"])

    plt.figure(figsize=(8, 5))
    plt.plot(confs, ps, label="Precision", marker="o", markersize=3)
    plt.plot(confs, rs, label="Recall", marker="o", markersize=3)
    plt.plot(confs, f1s, label="F1-Score", marker="o", markersize=3, linewidth=2)
    plt.axvline(
        args.highlight,
        color="crimson",
        linestyle="--",
        label=f"Android conf={args.highlight:.3f}",
    )
    plt.xlabel("Confidence threshold")
    plt.ylabel("Score")
    plt.title("Confidence Threshold Sweep on Test Split")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig_path = args.output / "threshold_sweep_f1.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()

    summary = {
        "best_f1_conf": best["conf"],
        "best_f1": best["f1"],
        "android_conf": args.highlight,
        "android_nearest_row": min(rows, key=lambda r: abs(r["conf"] - args.highlight)),
        "csv": str(csv_path),
        "figure": str(fig_path),
    }
    summary_path = args.output / "threshold_sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nBest F1 at conf={best['conf']:.2f} (F1={best['f1']:.3f})")
    print(f"Saved: {csv_path}")
    print(f"Saved: {fig_path}")
    print(f"Saved: {summary_path}")
    print("[OK] Test 3 finished")


if __name__ == "__main__":
    main()
