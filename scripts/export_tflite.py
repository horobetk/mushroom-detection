#!/usr/bin/env python3
# Export YOLO11 .pt weights to TensorFlow Lite via SavedModel intermediate.
# Direct TFLite/LiteRT export is blocked on Windows by Ultralytics, so:
#   1) model.export(format='saved_model')
#   2) tf.lite.TFLiteConverter.from_saved_model(...)
# Modes: int8 (needs --data), fp16, fp32. INT8 failure falls back to fp32
# unless --no-fallback is set.
# Author: Kiril Horobets, WUT 2026

import argparse
import os
import shutil
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

os.environ.setdefault("MPLBACKEND", "Agg")

from ultralytics import YOLO  # noqa: E402

DEFAULT_WEIGHTS = "C:/mushroom_data/runs/train-4/weights/best.pt"
DEFAULT_DATA = "mushrooms.yaml"


def _run_export(weights: Path, quant: str, imgsz: int, data: str) -> Path:
    import tensorflow as tf

    model = YOLO(str(weights))

    export_kwargs = {
        "format": "saved_model",
        "imgsz": imgsz,
    }
    if quant == "int8":
        export_kwargs["int8"] = True
        export_kwargs["data"] = data

    saved_model_path = Path(str(model.export(**export_kwargs)))

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_path))

    if quant == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif quant == "fp16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    # fp32: converter defaults

    tflite_model = converter.convert()

    tflite_path = saved_model_path.parent / f"{saved_model_path.name}_{quant}.tflite"
    tflite_path.write_bytes(tflite_model)

    return tflite_path


def export_model(args) -> Path:
    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(
            f"Weights not found: {weights}\n"
            "Run training first (scripts/train.py)."
        )

    print("[INFO] Export to TFLite (SavedModel pipeline)")
    print(f"  weights={weights} quant={args.quant} imgsz={args.imgsz}")
    if args.quant == "int8":
        print(f"  calibration={args.data}")

    try:
        exported_path = _run_export(weights, args.quant, args.imgsz, args.data)
    except Exception as exc:  # noqa: BLE001
        if args.quant == "int8" and not args.no_fallback:
            print(f"[WARN] INT8 export failed: {exc}")
            print("[WARN] Falling back to fp32")
            exported_path = _run_export(weights, "fp32", args.imgsz, args.data)
        else:
            raise

    print(f"[OK] Exported: {exported_path}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(exported_path, out_path)
        print(f"[OK] Copied to {out_path.resolve()}")
        exported_path = out_path

    size_mb = exported_path.stat().st_size / (1024 * 1024)
    print(f"[INFO] Model size: {size_mb:.2f} MB")
    return exported_path


def main():
    parser = argparse.ArgumentParser(
        description="Export YOLO11 to TFLite via SavedModel (Windows-compatible)."
    )
    parser.add_argument(
        "--weights", "-w", default=DEFAULT_WEIGHTS,
        help=f".pt weights (default: {DEFAULT_WEIGHTS})",
    )
    parser.add_argument(
        "--quant", "-q", default="int8", choices=["int8", "fp16", "fp32"],
    )
    parser.add_argument("--data", "-d", default=DEFAULT_DATA)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument(
        "--no-fallback", action="store_true",
        help="Disable int8 -> fp32 fallback",
    )

    args = parser.parse_args()
    export_model(args)


if __name__ == "__main__":
    main()
