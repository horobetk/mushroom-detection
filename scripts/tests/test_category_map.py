#!/usr/bin/env python3
"""Per-class and per-toxicity-category mAP, P and R on the test split."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
from ultralytics import YOLO

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from _paths import DEFAULT_DATA, DEFAULT_MODEL, DEFAULT_OUTPUT  # noqa: E402

GROUPS = {
    "EDIBLE": ("EDIBLE",),
    "INEDIBLE": ("INEDIBLE",),
    "TOXIC": ("POISONOUS", "DEADLY"),
    "DEADLY": ("DEADLY",),
}


def load_toxicity(csv_path: Path) -> dict[int, tuple[str, str]]:
    out: dict[int, tuple[str, str]] = {}
    with csv_path.open("r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[int(row["class_id"])] = (row["name"], row["toxicity"].strip().upper())
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", "-m", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--data", "-d", type=Path, default=DEFAULT_DATA)
    p.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--toxicity-csv", type=Path, default=_TESTS_DIR / "toxicity_registry.csv")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    args = p.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    tox = load_toxicity(args.toxicity_csv)

    model = YOLO(str(args.model))
    res = model.val(
        data=str(args.data),
        split="test",
        imgsz=args.imgsz,
        device=args.device,
        plots=False,
        verbose=False,
    )
    box = res.box
    idx = [int(c) for c in box.ap_class_index]
    ap50 = np.asarray(box.ap50, dtype=float)
    ap = np.asarray(box.ap, dtype=float)
    prec = np.asarray(box.p, dtype=float)
    rec = np.asarray(box.r, dtype=float)

    per_class_path = args.output / "per_class_test_metrics.csv"
    with per_class_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["class_id", "species", "toxicity", "ap50", "ap50_95", "precision", "recall"])
        for k, cid in enumerate(idx):
            name, t = tox.get(cid, ("?", "UNKNOWN"))
            w.writerow([cid, name, t, ap50[k], ap[k], prec[k], rec[k]])

    groups = {}
    for gname, members in GROUPS.items():
        sel = [k for k, cid in enumerate(idx) if tox.get(cid, ("", ""))[1] in members]
        n_reg = sum(1 for _, t in tox.values() if t in members)
        groups[gname] = {
            "n_classes_registry": n_reg,
            "n_classes_with_gt": len(sel),
            "mAP50": float(ap50[sel].mean()) if sel else None,
            "mAP50_95": float(ap[sel].mean()) if sel else None,
            "precision": float(prec[sel].mean()) if sel else None,
            "recall": float(rec[sel].mean()) if sel else None,
        }

    summary = {
        "model": str(args.model),
        "data": str(args.data),
        "split": "test",
        "toxicity_csv": str(args.toxicity_csv),
        "all": {
            "n_classes_with_gt": len(idx),
            "mAP50": float(box.map50),
            "mAP50_95": float(box.map),
            "precision": float(box.mp),
            "recall": float(box.mr),
        },
        "groups": groups,
    }
    out = args.output / "category_test_metrics.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[OK] all: mAP50={box.map50:.4f} mAP50-95={box.map:.4f} P={box.mp:.4f} R={box.mr:.4f}")
    for gname, g in groups.items():
        if g["mAP50"] is None:
            continue
        print(
            f"[OK] {gname:9s} ({g['n_classes_with_gt']} cls): mAP50={g['mAP50']:.4f} "
            f"mAP50-95={g['mAP50_95']:.4f} P={g['precision']:.4f} R={g['recall']:.4f}"
        )
    print(f"[OK] {per_class_path}")
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()
