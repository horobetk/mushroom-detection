#!/usr/bin/env python3
# Test 1 - Edibility safety matrix with asymmetric Android thresholds
from __future__ import annotations

import argparse
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
import pandas as pd
import seaborn as sns
import yaml
from ultralytics import YOLO

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from _paths import (  # noqa: E402
    CONF_EDIBLE,
    CONF_NEUTRAL_MAX,
    CONF_NEUTRAL_MIN,
    CONF_TOXIC,
    DEFAULT_DATA,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT,
)

SAFE = "SAFE"
UNSAFE = "UNSAFE"
NEUTRAL = "NEUTRAL"

_DEADLY_KEYWORDS = (
    "phalloides", "virosa", "galerina", "rubellus", "orellanus", "satanas",
)
_POISONOUS_KEYWORDS = (
    "muscaria", "pantherina", "emetica", "gyromitra", "paxillus", "inocybe",
    "hypholoma", "entoloma", "xanthodermus", "omphalotus", "scleroderma",
    "psilocybe", "lepiota", "cristata",
)
_CAUTION_KEYWORDS = (
    "equestre", "helvella", "torminosus", "piperatus", "rufus", "helvus",
)

IOU_MATCH = 0.50


def load_class_names(data_yaml: Path) -> list[str]:
    with data_yaml.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("names", [])
    if isinstance(names, dict):
        return [names[i] for i in sorted(names.keys(), key=int)]
    return list(names)


def infer_toxicity(name: str) -> str:
    key = name.lower().replace("_", " ")
    if any(k in key for k in _DEADLY_KEYWORDS) or any(
        k in key for k in _POISONOUS_KEYWORDS
    ):
        return UNSAFE
    if any(k in key for k in _CAUTION_KEYWORDS):
        return SAFE
    return SAFE


def load_toxicity_map(names: list[str], toxicity_csv: Path | None) -> dict[int, str]:
    mapping: dict[int, str] = {}
    if toxicity_csv is not None and toxicity_csv.exists():
        df = pd.read_csv(toxicity_csv)
        for _, row in df.iterrows():
            cid = int(row["class_id"])
            raw = str(row["toxicity"]).strip().upper()
            if raw in ("POISONOUS", "DEADLY", "UNSAFE"):
                mapping[cid] = UNSAFE
            elif raw in ("INEDIBLE", "NEUTRAL"):
                mapping[cid] = NEUTRAL
            else:
                mapping[cid] = SAFE
        return mapping

    for i, name in enumerate(names):
        mapping[i] = infer_toxicity(name)
    return mapping


def predicted_bucket(tox: str, conf: float) -> str | None:
    # Mirror Android displayStatus without the live temporal vote.
    # UNSAFE >= conf_toxic, SAFE >= conf_edible, mid-gap -> NEUTRAL.
    if tox == UNSAFE:
        if conf >= CONF_TOXIC:
            return UNSAFE
        return None

    if tox == SAFE:
        if conf >= CONF_EDIBLE:
            return SAFE
        if conf >= CONF_NEUTRAL_MIN:
            return NEUTRAL
        return None

    # Predicted class is biologically NEUTRAL (inedible / unknown).
    if conf >= CONF_NEUTRAL_MIN:
        return NEUTRAL
    return None


def resolve_dataset_root(data_yaml: Path) -> Path:
    with data_yaml.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    return root, cfg


def yolo_to_xyxy(cx: float, cy: float, w: float, h: float, img_w: int, img_h: int):
    x1 = (cx - w / 2.0) * img_w
    y1 = (cy - h / 2.0) * img_h
    x2 = (cx + w / 2.0) * img_w
    y2 = (cy + h / 2.0) * img_h
    return x1, y1, x2, y2


def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_gt_boxes(label_path: Path, img_w: int, img_h: int) -> list[tuple[int, tuple]]:
    boxes = []
    if not label_path.exists():
        return boxes
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:5])
            boxes.append((cls_id, yolo_to_xyxy(cx, cy, w, h, img_w, img_h)))
    return boxes


def build_asymmetric_matrix(
    model: YOLO,
    data_yaml: Path,
    tox: dict[int, str],
    imgsz: int,
    device: str,
) -> tuple[np.ndarray, list[str], dict[str, int]]:
    # Per-GT matching on the test split with asymmetric conf mapping.
    root, cfg = resolve_dataset_root(data_yaml)
    test_key = cfg.get("test", "images/test")
    images_dir = Path(test_key)
    if not images_dir.is_absolute():
        images_dir = (root / images_dir).resolve()
    labels_dir = root / "labels" / "test"

    labels = [SAFE, NEUTRAL, UNSAFE]
    idx = {lab: i for i, lab in enumerate(labels)}
    coarse = np.zeros((3, 3), dtype=np.float64)
    counts = {"matched": 0, "unmatched_gt": 0, "preds_used": 0}

    image_files = sorted(
        [
            p
            for p in images_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ]
    )
    print(f"[INFO] Evaluating {len(image_files)} test images")

    for img_path in image_files:
        results = model.predict(
            source=str(img_path),
            imgsz=imgsz,
            conf=CONF_TOXIC,
            device=device,
            verbose=False,
        )
        r0 = results[0]
        img_h, img_w = r0.orig_shape
        gt_boxes = load_gt_boxes(labels_dir / f"{img_path.stem}.txt", img_w, img_h)

        preds = []
        if r0.boxes is not None and len(r0.boxes) > 0:
            xyxy = r0.boxes.xyxy.cpu().numpy()
            confs = r0.boxes.conf.cpu().numpy()
            clss = r0.boxes.cls.cpu().numpy().astype(int)
            for i in range(len(clss)):
                bucket = predicted_bucket(tox.get(int(clss[i]), NEUTRAL), float(confs[i]))
                if bucket is None:
                    continue
                preds.append(
                    {
                        "box": tuple(map(float, xyxy[i])),
                        "bucket": bucket,
                        "used": False,
                    }
                )

        for gt_cls, gt_box in gt_boxes:
            gt_bucket = tox.get(gt_cls, NEUTRAL)
            best_i = -1
            best_iou = IOU_MATCH
            for i, pred in enumerate(preds):
                if pred["used"]:
                    continue
                score = iou_xyxy(gt_box, pred["box"])
                if score > best_iou:
                    best_iou = score
                    best_i = i

            if best_i < 0:
                # No qualifying prediction: count as NEUTRAL (no SAFE/UNSAFE assert).
                coarse[idx[gt_bucket], idx[NEUTRAL]] += 1.0
                counts["unmatched_gt"] += 1
                continue

            preds[best_i]["used"] = True
            pred_bucket = preds[best_i]["bucket"]
            coarse[idx[gt_bucket], idx[pred_bucket]] += 1.0
            counts["matched"] += 1
            counts["preds_used"] += 1

    return coarse, labels, counts


def critical_false_positive_rate(coarse: np.ndarray, labels: list[str]) -> float:
    # CFPR = P(pred=SAFE | gt=UNSAFE) under asymmetric thresholds.
    i_unsafe = labels.index(UNSAFE)
    i_safe = labels.index(SAFE)
    unsafe_total = coarse[i_unsafe].sum()
    if unsafe_total <= 0:
        return 0.0
    return float(coarse[i_unsafe, i_safe] / unsafe_total)


def false_alarm_rate(coarse: np.ndarray, labels: list[str]) -> float:
    # FAR = P(pred=UNSAFE | gt=SAFE); accepted safety tradeoff.
    i_safe = labels.index(SAFE)
    i_unsafe = labels.index(UNSAFE)
    safe_total = coarse[i_safe].sum()
    if safe_total <= 0:
        return 0.0
    return float(coarse[i_safe, i_unsafe] / safe_total)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test 1: asymmetric edibility safety matrix "
        f"(conf_edible={CONF_EDIBLE}, conf_toxic={CONF_TOXIC})."
    )
    parser.add_argument("--model", "-m", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", "-d", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--toxicity-csv", type=Path, default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    names = load_class_names(args.data)
    tox = load_toxicity_map(names, args.toxicity_csv)

    print("[INFO] TEST 1: Asymmetric edibility safety matrix")
    print(f"[INFO] Model:        {args.model}")
    print(f"[INFO] Data:         {args.data}")
    print(f"[INFO] Classes:      {len(names)}")
    print(f"[INFO] Unsafe:       {sum(1 for v in tox.values() if v == UNSAFE)}")
    print(f"[INFO] conf_edible:  {CONF_EDIBLE}")
    print(f"[INFO] conf_toxic:   {CONF_TOXIC}")
    print(f"[INFO] neutral band: [{CONF_NEUTRAL_MIN}, {CONF_NEUTRAL_MAX}]")

    model = YOLO(str(args.model))

    print("[INFO] Running Ultralytics val() for mAP reference")
    results = model.val(
        data=str(args.data),
        split="test",
        conf=CONF_TOXIC,
        imgsz=args.imgsz,
        device=args.device,
        plots=False,
        verbose=False,
    )

    print("[INFO] Building asymmetric confusion matrix")
    coarse, labels, counts = build_asymmetric_matrix(
        model, args.data, tox, args.imgsz, args.device
    )
    cfpr = critical_false_positive_rate(coarse, labels)
    far = false_alarm_rate(coarse, labels)

    df = pd.DataFrame(coarse, index=labels, columns=labels)
    csv_path = args.output / "safety_matrix.csv"
    df.to_csv(csv_path)

    plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(df, annot=True, fmt=".0f", cmap="Reds", cbar=True)
    plt.title(
        f"Asymmetric Safety Matrix "
        f"(edible>={CONF_EDIBLE}, toxic>={CONF_TOXIC})"
    )
    plt.ylabel("Ground truth")
    plt.xlabel("Prediction (after thresholds)")
    plt.tight_layout()
    fig_path = args.output / "safety_matrix_heatmap.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()

    summary = {
        "num_classes": len(names),
        "conf_edible": CONF_EDIBLE,
        "conf_toxic": CONF_TOXIC,
        "conf_neutral_min": CONF_NEUTRAL_MIN,
        "conf_neutral_max": CONF_NEUTRAL_MAX,
        "critical_false_positive_rate": cfpr,
        "false_alarm_rate_safe_to_unsafe": far,
        "match_counts": counts,
        "map50": float(getattr(results.box, "map50", float("nan"))),
        "map50_95": float(getattr(results.box, "map", float("nan"))),
        "precision": float(getattr(results.box, "mp", float("nan"))),
        "recall": float(getattr(results.box, "mr", float("nan"))),
        "matrix_csv": str(csv_path),
        "heatmap": str(fig_path),
    }
    summary_path = args.output / "safety_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[INFO] CFPR (pred=SAFE | gt=UNSAFE): {cfpr:.4f}")
    print(f"[INFO] FAR  (pred=UNSAFE | gt=SAFE): {far:.4f}")
    print(f"[OK] Saved {csv_path}")
    print(f"[OK] Saved {fig_path}")
    print(f"[OK] Saved {summary_path}")
    print("[OK] Test 1 finished")


if __name__ == "__main__":
    main()
