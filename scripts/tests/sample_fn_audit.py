#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from _paths import DEFAULT_DATA, DEFAULT_OUTPUT  # noqa: E402
import yaml


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


def resolve_dataset_root(data_yaml: Path):
    with data_yaml.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    return root, cfg


def first_class(label_path: Path) -> int | None:
    if not label_path.exists():
        return None
    with label_path.open("r", encoding="utf-8") as fh:
        line = fh.readline().strip()
    if not line:
        return None
    return int(float(line.split()[0]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--toxicity-csv", type=Path, default=_TESTS_DIR / "toxicity_registry.csv")
    p.add_argument("--n-toxic", type=int, default=100)
    p.add_argument("--n-edible", type=int, default=80)
    p.add_argument("--n-inedible", type=int, default=70)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    tox = load_registry(args.toxicity_csv)
    root, cfg = resolve_dataset_root(args.data)
    test_key = cfg.get("test", "images/test")
    images_dir = Path(test_key)
    if not images_dir.is_absolute():
        images_dir = (root / images_dir).resolve()
    labels_dir = root / "labels" / "test"

    buckets: dict[str, list[Path]] = defaultdict(list)
    for img in images_dir.iterdir():
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        cid = first_class(labels_dir / f"{img.stem}.txt")
        if cid is None:
            continue
        buckets[tox.get(cid, "INEDIBLE")].append(img)

    rng = random.Random(args.seed)
    want = {"TOXIC": args.n_toxic, "EDIBLE": args.n_edible, "INEDIBLE": args.n_inedible}
    chosen: list[tuple[str, Path]] = []
    for key, n in want.items():
        pool = buckets[key]
        rng.shuffle(pool)
        if len(pool) < n:
            print(f"[WARN] {key}: only {len(pool)} images, taking all")
            take = pool
        else:
            take = pool[:n]
        chosen.extend((key, pth) for pth in take)

    rng.shuffle(chosen)
    args.output.mkdir(parents=True, exist_ok=True)
    out = args.output / "fn_audit_sheet.csv"
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "image_path",
                "gt_bucket",
                "n_visible_fruitbodies",
                "n_boxes_ok",
                "n_fn_missed",
                "n_fp_extra",
                "notes",
            ]
        )
        for bucket, path in chosen:
            w.writerow([str(path), bucket, "", "", "", "", ""])

    howto = args.output / "fn_audit_howto.txt"
    howto.write_text(
        "\n".join(
            [
                "FN audit of GroundingDINO boxes on the isolated TEST split.",
                f"Sample size: {len(chosen)} (seed={args.seed}).",
                "",
                "Open each image_path. Count visible fruitbodies yourself.",
                "Fill four integers:",
                "  n_visible_fruitbodies = how many mushrooms you see",
                "  n_boxes_ok            = YOLO txt boxes that sit on a mushroom",
                "  n_fn_missed           = mushrooms with no box",
                "  n_fp_extra            = boxes with no mushroom",
                "Check: n_fn_missed + n_boxes_ok should be close to n_visible.",
                "",
                "When finished, send fn_audit_sheet.csv back.",
            ]
        ),
        encoding="utf-8",
    )
    print(f"[OK] {out} n={len(chosen)}")
    print(f"[OK] {howto}")


if __name__ == "__main__":
    main()
