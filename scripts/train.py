#!/usr/bin/env python3
# Train YOLO11 mushroom detector (transfer learning).
# Defaults: epochs=200, imgsz=640, batch=32.
# Headless-safe for remote clusters (matplotlib Agg, Ultralytics show=False).
# Author: Kiril Horobets, WUT 2026

import argparse
import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

# Must be set before importing matplotlib
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("YOLO_VERBOSE", "True")

import matplotlib  # noqa: E402

matplotlib.use("Agg", force=True)

import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

DEFAULT_MODEL = "weights/yolo11m.pt"
DEFAULT_DATA = "mushrooms.yaml"


def resolve_device(requested: str) -> str:
    cuda_available = torch.cuda.is_available()

    if requested == "auto":
        if cuda_available:
            print(f"[device] GPU: {torch.cuda.get_device_name(0)} -> '0'")
            return "0"
        print("[device] No CUDA, using CPU")
        return "cpu"

    if requested == "cpu":
        print("[device] Forced CPU")
        return "cpu"

    if not cuda_available:
        print(f"[device] Requested '{requested}' but CUDA unavailable, using CPU")
        return "cpu"

    device = "0" if requested == "cuda" else requested
    print(f"[device] Using '{device}'")
    return device


def train(args) -> str:
    device = resolve_device(args.device)

    # Absolute project path avoids Ultralytics global runs_dir overrides
    project = str(Path(args.project).resolve())

    print("[INFO] Training YOLO11m")
    print(f"  model={args.model} data={args.data}")
    print(
        f"  epochs={args.epochs} imgsz={args.imgsz} batch={args.batch} "
        f"device={device} workers={args.workers}"
    )
    print(f"  project={project}")

    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        patience=args.patience,
        project=project,
        name=args.name,
        exist_ok=args.exist_ok,
        resume=args.resume,
        plots=True,
        show=False,
        verbose=True,
    )

    save_dir = getattr(results, "save_dir", None)
    if save_dir is None:
        save_dir = os.path.join(project, args.name)
    best_path = os.path.join(str(save_dir), "weights", "best.pt")

    print("[OK] Training finished")
    print(f"  results: {save_dir}")
    print(f"  best:    {best_path}")
    return best_path


def main():
    parser = argparse.ArgumentParser(
        description="Train mushroom detector (YOLO11m)."
    )
    parser.add_argument(
        "--model", "-m", default=DEFAULT_MODEL,
        help=f"Base weights (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--data", "-d",
        default="E:/Kiril_Horobets/mushroom_data/mushroom_final/data.yaml",
        help="Path to data.yaml",
    )
    parser.add_argument("--epochs", "-e", type=int, default=200)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", "-b", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument(
        "--project", default="E:/Kiril_Horobets/mushroom_data/runs",
        help="Output directory for runs",
    )
    parser.add_argument("--name", default="train")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
