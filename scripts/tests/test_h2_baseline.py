#!/usr/bin/env python3


from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
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
    CONF_TOXIC,
    DEFAULT_DATA,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT,
)

IOU_MATCH = 0.50
GLOBAL_TAU = 0.30
PRED_CONF_FLOOR = 0.05

GT_LABELS = ("EDIBLE", "INEDIBLE", "TOXIC")
PRED_LABELS = ("JADALNY", "NEUTRAL", "OSTRZEZENIE")

LOOKALIKE_PAIRS = [
    ("Amanita phalloides", "Agaricus campestris"),
    ("Amanita phalloides", "Amanita citrina"),
    ("Amanita virosa", "Agaricus campestris"),
    ("Omphalotus olearius", "Cantharellus cibarius"),
    ("Hygrophoropsis aurantiaca", "Cantharellus cibarius"),
    ("Galerina marginata", "Kuehneromyces mutabilis"),
]


def load_class_names(data_yaml: Path) -> list[str]:
    with data_yaml.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("names", [])
    if isinstance(names, dict):
        return [names[i] for i in sorted(names.keys(), key=int)]
    return list(names)


def load_registry(csv_path: Path) -> dict[int, str]:
    mapping: dict[int, str] = {}
    with csv_path.open("r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            raw = row["toxicity"].strip().upper()
            cid = int(row["class_id"])
            if raw in ("POISONOUS", "DEADLY"):
                mapping[cid] = "TOXIC"
            elif raw == "INEDIBLE":
                mapping[cid] = "INEDIBLE"
            else:
                mapping[cid] = "EDIBLE"
    return mapping


def name_to_id(names: list[str]) -> dict[str, int]:
    out = {}
    for i, n in enumerate(names):
        out[n.replace("_", " ")] = i
        out[n] = i
    return out


def resolve_dataset_root(data_yaml: Path):
    with data_yaml.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    return root, cfg


def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    return (
        (cx - w / 2.0) * img_w,
        (cy - h / 2.0) * img_h,
        (cx + w / 2.0) * img_w,
        (cy + h / 2.0) * img_h,
    )


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


def load_gt_boxes(label_path: Path, img_w: int, img_h: int):
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


def pred_asymmetric(gt_tox_of_pred: str, conf: float) -> str:
    if gt_tox_of_pred == "TOXIC" and conf >= CONF_TOXIC:
        return "OSTRZEZENIE"
    if gt_tox_of_pred == "EDIBLE" and conf >= CONF_EDIBLE:
        return "JADALNY"
    return "NEUTRAL"


def pred_global(gt_tox_of_pred: str, conf: float, tau: float) -> str:
    if conf < tau:
        return "NEUTRAL"
    if gt_tox_of_pred == "TOXIC":
        return "OSTRZEZENIE"
    if gt_tox_of_pred == "EDIBLE":
        return "JADALNY"
    return "NEUTRAL"


def empty_matrix() -> np.ndarray:
    return np.zeros((3, 3), dtype=np.float64)


def add_cell(mat: np.ndarray, gt: str, pred: str) -> None:
    mat[GT_LABELS.index(gt), PRED_LABELS.index(pred)] += 1.0


def cfpr(mat: np.ndarray) -> float:
    row = mat[GT_LABELS.index("TOXIC")]
    total = row.sum()
    return float(row[PRED_LABELS.index("JADALNY")] / total) if total else 0.0


def far_edible(mat: np.ndarray) -> float:
    row = mat[GT_LABELS.index("EDIBLE")]
    total = row.sum()
    return float(row[PRED_LABELS.index("OSTRZEZENIE")] / total) if total else 0.0


def save_heatmap(mat: np.ndarray, title: str, path: Path) -> None:
    df = pd.DataFrame(mat, index=GT_LABELS, columns=PRED_LABELS)
    plt.figure(figsize=(7.2, 5.6))
    sns.heatmap(df, annot=True, fmt=".0f", cmap="Reds", cbar=True)
    plt.title(title)
    plt.ylabel("Ground truth (rejestr toksykologiczny)")
    plt.xlabel("Predykcja po progach (displayStatus)")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    df.to_csv(path.with_suffix(".csv"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", "-m", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", "-d", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--toxicity-csv",
        type=Path,
        default=_TESTS_DIR / "toxicity_registry.csv",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--global-tau", type=float, default=GLOBAL_TAU)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    names = load_class_names(args.data)
    tox = load_registry(args.toxicity_csv)
    n2i = name_to_id(names)

    print("[INFO] H2 dual-policy safety + look-alikes")
    print(f"[INFO] model={args.model}")
    print(f"[INFO] classes={len(names)} registry={len(tox)}")
    print(
        f"[INFO] EDIBLE={sum(v=='EDIBLE' for v in tox.values())} "
        f"INEDIBLE={sum(v=='INEDIBLE' for v in tox.values())} "
        f"TOXIC={sum(v=='TOXIC' for v in tox.values())}"
    )

    missing_pairs = [
        (a, b) for a, b in LOOKALIKE_PAIRS if a not in n2i or b not in n2i
    ]
    if missing_pairs:
        print("[WARN] look-alike names not in data.yaml:", missing_pairs)

    root, cfg = resolve_dataset_root(args.data)
    test_key = cfg.get("test", "images/test")
    images_dir = Path(test_key)
    if not images_dir.is_absolute():
        images_dir = (root / images_dir).resolve()
    labels_dir = root / "labels" / "test"

    image_files = sorted(
        p
        for p in images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    print(f"[INFO] test images: {len(image_files)}")

    model = YOLO(str(args.model))
    mat_asymm = empty_matrix()
    mat_glob = empty_matrix()
    pair_counts = {
        f"{a} -> {b}": 0 for a, b in LOOKALIKE_PAIRS
    }
    pair_counts.update({f"{b} -> {a}": 0 for a, b in LOOKALIKE_PAIRS})
    pair_gt = defaultdict(int)
    n_gt = 0

    pair_ids = []
    for a, b in LOOKALIKE_PAIRS:
        if a in n2i and b in n2i:
            pair_ids.append((n2i[a], n2i[b], a, b))

    for k, img_path in enumerate(image_files, 1):
        if k % 500 == 0:
            print(f"[INFO] {k}/{len(image_files)}")
        results = model.predict(
            source=str(img_path),
            imgsz=args.imgsz,
            conf=PRED_CONF_FLOOR,
            device=args.device,
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
                preds.append(
                    {
                        "box": tuple(map(float, xyxy[i])),
                        "cls": int(clss[i]),
                        "conf": float(confs[i]),
                        "used": False,
                    }
                )

        for gt_cls, gt_box in gt_boxes:
            n_gt += 1
            gt_row = tox.get(gt_cls, "INEDIBLE")
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
                add_cell(mat_asymm, gt_row, "NEUTRAL")
                add_cell(mat_glob, gt_row, "NEUTRAL")
                continue

            preds[best_i]["used"] = True
            pcls = preds[best_i]["cls"]
            pconf = preds[best_i]["conf"]
            ptox = tox.get(pcls, "INEDIBLE")
            add_cell(mat_asymm, gt_row, pred_asymmetric(ptox, pconf))
            add_cell(mat_glob, gt_row, pred_global(ptox, pconf, args.global_tau))

            for ia, ib, na, nb in pair_ids:
                if gt_cls == ia:
                    pair_gt[na] += 1
                    if pcls == ib:
                        pair_counts[f"{na} -> {nb}"] += 1
                elif gt_cls == ib:
                    pair_gt[nb] += 1
                    if pcls == ia:
                        pair_counts[f"{nb} -> {na}"] += 1

    cfpr_a = cfpr(mat_asymm)
    cfpr_g = cfpr(mat_glob)
    far_a = far_edible(mat_asymm)
    far_g = far_edible(mat_glob)

    heat_a = args.output / "safety_matrix_asymmetric.png"
    heat_g = args.output / "safety_matrix_global.png"
    save_heatmap(
        mat_asymm,
        f"Asymmetric 0.18/0.60 (CFPR={cfpr_a:.4f})",
        heat_a,
    )
    save_heatmap(
        mat_glob,
        f"Global tau={args.global_tau:.2f} (CFPR={cfpr_g:.4f})",
        heat_g,
    )

    cmp_path = args.output / "safety_cfpr_comparison.csv"
    with cmp_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "policy",
                "tau_toxic_or_global",
                "tau_edible",
                "n_gt_toxic",
                "cfp_jadaly_given_toxic",
                "cfpr",
                "n_gt_edible",
                "warnings_given_edible",
                "far_edible",
            ]
        )
        n_tox = float(mat_asymm[GT_LABELS.index("TOXIC")].sum())
        n_ed = float(mat_asymm[GT_LABELS.index("EDIBLE")].sum())
        cfp_a = float(mat_asymm[GT_LABELS.index("TOXIC"), PRED_LABELS.index("JADALNY")])
        cfp_g = float(mat_glob[GT_LABELS.index("TOXIC"), PRED_LABELS.index("JADALNY")])
        warn_a = float(mat_asymm[GT_LABELS.index("EDIBLE"), PRED_LABELS.index("OSTRZEZENIE")])
        warn_g = float(mat_glob[GT_LABELS.index("EDIBLE"), PRED_LABELS.index("OSTRZEZENIE")])
        w.writerow(
            [
                "asymmetric",
                CONF_TOXIC,
                CONF_EDIBLE,
                int(n_tox),
                int(cfp_a),
                cfpr_a,
                int(n_ed),
                int(warn_a),
                far_a,
            ]
        )
        w.writerow(
            [
                "global",
                args.global_tau,
                args.global_tau,
                int(n_tox),
                int(cfp_g),
                cfpr_g,
                int(n_ed),
                int(warn_g),
                far_g,
            ]
        )

    look_path = args.output / "lookalike_confusions.csv"
    with look_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "gt_species",
                "pred_species",
                "gt_instances_matched_or_unmatched",
                "confusions",
                "rate",
            ]
        )
        rows_out = []
        for a, b in LOOKALIKE_PAIRS:
            for src, dst in ((a, b), (b, a)):
                key = f"{src} -> {dst}"
                n = pair_gt.get(src, 0)
                c = pair_counts.get(key, 0)
                rate = (c / n) if n else 0.0
                w.writerow([src, dst, n, c, rate])
                rows_out.append(
                    {
                        "gt": src,
                        "pred": dst,
                        "n_gt": n,
                        "n_confuse": c,
                        "rate": rate,
                    }
                )

    summary = {
        "n_test_images": len(image_files),
        "n_gt_instances": n_gt,
        "global_tau": args.global_tau,
        "conf_toxic": CONF_TOXIC,
        "conf_edible": CONF_EDIBLE,
        "cfpr_asymmetric": cfpr_a,
        "cfpr_global": cfpr_g,
        "far_edible_asymmetric": far_a,
        "far_edible_global": far_g,
        "matrix_asymmetric": mat_asymm.tolist(),
        "matrix_global": mat_glob.tolist(),
        "gt_labels": list(GT_LABELS),
        "pred_labels": list(PRED_LABELS),
        "lookalikes": rows_out,
        "h2_holds": bool(cfpr_a < cfpr_g),
    }
    summary_path = args.output / "h2_cfpr_lookalikes.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[OK] CFPR asymmetric={cfpr_a:.6f}  global={cfpr_g:.6f}")
    print(f"[OK] H2 (asymm < global): {summary['h2_holds']}")
    print(f"[OK] {cmp_path}")
    print(f"[OK] {look_path}")
    print(f"[OK] {summary_path}")


if __name__ == "__main__":
    main()
