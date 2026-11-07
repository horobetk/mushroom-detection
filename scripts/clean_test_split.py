#!/usr/bin/env python3
"""Quarantine test images whose obs_id also appears in train/val."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections import defaultdict
from typing import Dict, Optional, Set

IMG_EXTS = (".jpg", ".jpeg", ".png")


def obs_id_of(filename: str) -> str:
    """Extract the observation id from ``{obs_id}_{photo_id}.jpg``."""
    return os.path.basename(filename).split("_")[0]


def list_images(folder: str) -> list[str]:
    if not os.path.isdir(folder):
        return []
    return [f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS)]


def collect_obs_ids(*folders: str) -> Set[str]:
    ids: Set[str] = set()
    for folder in folders:
        for name in list_images(folder):
            ids.add(obs_id_of(name))
    return ids


def class_id_of_label(label_path: str) -> Optional[int]:
    """Return the class id from the first row of a YOLO label file, or None."""
    try:
        with open(label_path, "r", encoding="utf-8") as fh:
            first = fh.readline().strip()
        if not first:
            return None
        return int(first.split()[0])
    except (OSError, ValueError):
        return None


def per_class_counts(img_dir: str, lbl_dir: str) -> Dict[int, int]:
    counts: Dict[int, int] = defaultdict(int)
    for img_name in list_images(img_dir):
        base = os.path.splitext(img_name)[0]
        cid = class_id_of_label(os.path.join(lbl_dir, base + ".txt"))
        if cid is not None:
            counts[cid] += 1
    return dict(counts)


def move_pair(img_name: str, img_dir: str, lbl_dir: str, dst_img: str, dst_lbl: str) -> None:
    base = os.path.splitext(img_name)[0]
    os.makedirs(dst_img, exist_ok=True)
    os.makedirs(dst_lbl, exist_ok=True)
    shutil.move(os.path.join(img_dir, img_name), os.path.join(dst_img, img_name))
    src_lbl = os.path.join(lbl_dir, base + ".txt")
    if os.path.exists(src_lbl):
        shutil.move(src_lbl, os.path.join(dst_lbl, base + ".txt"))


def purge_leaked_test(base_dir: str, apply_changes: bool) -> dict:
    """Move leaked test samples to a quarantine folder. Returns a report dict."""
    img_train = os.path.join(base_dir, "images", "train")
    img_val = os.path.join(base_dir, "images", "val")
    img_test = os.path.join(base_dir, "images", "test")
    lbl_test = os.path.join(base_dir, "labels", "test")

    q_img = os.path.join(base_dir, "images", "test_leaked_quarantine")
    q_lbl = os.path.join(base_dir, "labels", "test_leaked_quarantine")

    train_val_ids = collect_obs_ids(img_train, img_val)
    test_images = list_images(img_test)

    before = per_class_counts(img_test, lbl_test)

    leaked = [f for f in test_images if obs_id_of(f) in train_val_ids]
    leaked_obs = {obs_id_of(f) for f in leaked}

    print(f"[INFO] train/val distinct obs_id : {len(train_val_ids)}")
    print(f"[INFO] test images total         : {len(test_images)}")
    print(f"[INFO] leaked test images        : {len(leaked)} "
          f"({len(leaked_obs)} distinct obs_id)")

    if apply_changes:
        for img_name in leaked:
            move_pair(img_name, img_test, lbl_test, q_img, q_lbl)
        print(f"[OK] Moved {len(leaked)} leaked pairs -> {q_img}")
    else:
        print("[DRY-RUN] No files moved. Re-run with --apply to quarantine them.")

    after = per_class_counts(img_test, lbl_test) if apply_changes else {
        cid: before.get(cid, 0)
        - sum(1 for f in leaked if class_id_of_label(
            os.path.join(lbl_test, os.path.splitext(f)[0] + ".txt")) == cid)
        for cid in before
    }

    emptied = sorted(cid for cid in before if after.get(cid, 0) == 0)
    if emptied:
        print(f"[WARN] Classes with 0 test images after purge: {emptied}")

    return {
        "train_val_obs_ids": len(train_val_ids),
        "test_images_before": len(test_images),
        "leaked_images": len(leaked),
        "leaked_obs_ids": len(leaked_obs),
        "test_images_after": len(test_images) - (len(leaked) if apply_changes else 0),
        "classes_emptied": emptied,
        "per_class_before": before,
        "per_class_after": after,
        "applied": apply_changes,
    }


def run_validation(base_dir: str, weights: str, data_yaml: str,
                   imgsz: int, device: str) -> Optional[dict]:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[SKIP] ultralytics not installed - validation skipped.")
        return None
    if not os.path.exists(weights):
        print(f"[SKIP] weights not found: {weights}")
        return None
    if not os.path.exists(data_yaml):
        print(f"[SKIP] data.yaml not found: {data_yaml}")
        return None

    print(f"[INFO] Validating {weights} on split=test ...")
    model = YOLO(weights)
    results = model.val(data=data_yaml, split="test", imgsz=imgsz, device=device)
    box = results.box
    metrics = {
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "num_classes": int(len(box.maps)) if box.maps is not None else None,
    }
    print("[OK] Metrics:", json.dumps(metrics, indent=2))
    return metrics


def save_report(out_dir: str, report: dict, metrics: Optional[dict]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    payload = {"purge": report, "metrics": metrics}
    json_path = os.path.join(out_dir, "clean_test_metrics.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"[OK] Wrote {json_path}")

    csv_path = os.path.join(out_dir, "clean_test_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        writer.writerow(["test_images_before", report["test_images_before"]])
        writer.writerow(["leaked_images", report["leaked_images"]])
        writer.writerow(["test_images_after", report["test_images_after"]])
        if metrics:
            for key in ("mAP50", "mAP50_95", "precision", "recall"):
                writer.writerow([key, metrics[key]])
    print(f"[OK] Wrote {csv_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-dir", default="E:/Kiril_Horobets/mushroom_data/mushroom_final",
                   help="Dataset root containing images/ and labels/.")
    p.add_argument("--weights", default="E:/Kiril_Horobets/mushroom_data/runs/"
                   "fine_147classes_v2/weights/best.pt",
                   help="Path to the YOLO best.pt checkpoint for validation.")
    p.add_argument("--data-yaml", default="E:/Kiril_Horobets/mushroom_data/"
                   "mushroom_final/data.yaml",
                   help="Ultralytics data.yaml describing train/val/test.")
    p.add_argument("--out-dir", default="docs/thesis_results",
                   help="Where to write the JSON/CSV metric report.")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--apply", action="store_true",
                   help="Actually move leaked files. Without it the run is a dry run.")
    p.add_argument("--no-val", action="store_true",
                   help="Only purge, do not run model.val().")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    report = purge_leaked_test(args.base_dir, apply_changes=args.apply)

    metrics = None
    if not args.no_val and args.apply:
        metrics = run_validation(args.base_dir, args.weights, args.data_yaml,
                                 args.imgsz, args.device)
    elif not args.apply:
        print("[INFO] Dry run: skipping validation (run with --apply to evaluate).")

    save_report(args.out_dir, report, metrics)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
