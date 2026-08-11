#!/usr/bin/env python3
# Test 2 – Robustness to forest imaging noise.
#
# Simulates field conditions typical for mushroom foraging:
#   - dark dense-forest lighting (brightness reduction),
#   - hand tremor / motion blur (Gaussian blur),
#   - digital camera noise (additive Gaussian noise).
#
# Compares mAP50 on the clean test split versus a corrupted copy of the same
# images and reports relative degradation.
#
# Author: Kiril Horobets
# Warsaw University of Technology, 2026

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

os.environ.setdefault("MPLBACKEND", "Agg")

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from _paths import DEFAULT_DATA, DEFAULT_MODEL, DEFAULT_OUTPUT  # noqa: E402


def apply_forest_noise(
    image_bgr: np.ndarray,
    brightness_factor: float = 0.6,
    blur_kernel: int = 5,
    noise_sigma: float = 8.0,
) -> np.ndarray:
    # Darken + blur + Gaussian noise (forest-like degradation).
    img = cv2.convertScaleAbs(image_bgr, alpha=brightness_factor, beta=0)
    k = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
    img = cv2.GaussianBlur(img, (k, k), 0)
    noise = np.random.normal(0.0, noise_sigma, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def resolve_split_images(data_yaml: Path, split: str = "test") -> Path:
    # Resolve absolute path to images/<split> from data.yaml.
    with data_yaml.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    split_key = cfg.get(split, f"images/{split}")
    images_dir = Path(split_key)
    if not images_dir.is_absolute():
        images_dir = (root / images_dir).resolve()
    return images_dir


def build_noisy_dataset(
    data_yaml: Path,
    work_dir: Path,
    brightness: float,
    blur_kernel: int,
    noise_sigma: float,
) -> Path:
    # Copy the YOLO dataset layout into work_dir and replace test images with
    # noise-corrupted versions. Labels are copied unchanged.
    with data_yaml.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()

    dst_root = work_dir / "noisy_dataset"
    for split in ("train", "val", "test"):
        for kind in ("images", "labels"):
            src = root / kind / split
            dst = dst_root / kind / split
            if not src.exists():
                continue
            dst.mkdir(parents=True, exist_ok=True)
            if kind == "labels" or split != "test":
                for item in src.iterdir():
                    if item.is_file():
                        shutil.copy2(item, dst / item.name)
            else:
                for item in src.iterdir():
                    if item.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                        continue
                    img = cv2.imread(str(item))
                    if img is None:
                        continue
                    noisy = apply_forest_noise(
                        img,
                        brightness_factor=brightness,
                        blur_kernel=blur_kernel,
                        noise_sigma=noise_sigma,
                    )
                    cv2.imwrite(str(dst / item.name), noisy)

    noisy_yaml = {
        "path": str(dst_root.resolve()),
        "train": cfg.get("train", "images/train"),
        "val": cfg.get("val", "images/val"),
        "test": cfg.get("test", "images/test"),
        "nc": cfg.get("nc"),
        "names": cfg.get("names"),
    }
    yaml_path = dst_root / "data_noisy.yaml"
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(noisy_yaml, f, sort_keys=False)
    return yaml_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test 2: robustness to forest noise (blur / darkness / sensor noise)."
    )
    parser.add_argument("--model", "-m", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", "-d", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--brightness", type=float, default=0.6)
    parser.add_argument("--blur-kernel", type=int, default=5)
    parser.add_argument("--noise-sigma", type=float, default=8.0)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    print("[INFO] TEST 2: Forest Noise Robustness")
    print(f"[INFO] Model:       {args.model}")
    print(f"[INFO] Brightness:  {args.brightness}")
    print(f"[INFO] Blur kernel: {args.blur_kernel}")
    print(f"[INFO] Noise sigma: {args.noise_sigma}")

    model = YOLO(str(args.model))

    print("\n[1/2] Evaluating clean test split...")
    clean = model.val(
        data=str(args.data),
        split="test",
        imgsz=args.imgsz,
        device=args.device,
        verbose=False,
    )
    clean_map50 = float(clean.box.map50)

    print("[2/2] Building noisy test images and re-evaluating...")
    with tempfile.TemporaryDirectory(prefix="forest_noise_") as tmp:
        noisy_yaml = build_noisy_dataset(
            args.data,
            Path(tmp),
            brightness=args.brightness,
            blur_kernel=args.blur_kernel,
            noise_sigma=args.noise_sigma,
        )
        noisy = model.val(
            data=str(noisy_yaml),
            split="test",
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )
        noisy_map50 = float(noisy.box.map50)

    degradation = clean_map50 - noisy_map50
    relative = (degradation / clean_map50 * 100.0) if clean_map50 > 0 else float("nan")

    summary = {
        "clean_map50": clean_map50,
        "noisy_map50": noisy_map50,
        "absolute_degradation": degradation,
        "relative_degradation_percent": relative,
        "brightness_factor": args.brightness,
        "blur_kernel": args.blur_kernel,
        "noise_sigma": args.noise_sigma,
    }
    out_json = args.output / "robustness_noise_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nClean  mAP50: {clean_map50:.4f}")
    print(f"Noisy  mAP50: {noisy_map50:.4f}")
    print(f"Delta         : {degradation:.4f} ({relative:.2f}%)")
    print(f"Saved: {out_json}")
    print("[OK] Test 2 finished")


if __name__ == "__main__":
    main()
