#!/usr/bin/env python3


from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

CSV = Path(__file__).resolve().parents[2] / "docs" / "thesis_results" / "fn_audit_sheet.csv"
OUT = CSV.parent / "fn_audit_summary.json"


def main() -> None:
    rows = list(csv.DictReader(CSV.open(encoding="utf-8-sig")))
    empty = 0
    notes_n = 0
    inconsist = 0
    buckets: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "n_images": 0,
            "n_visible": 0,
            "n_boxes_ok": 0,
            "n_fn": 0,
            "n_fp": 0,
            "images_with_fn": 0,
            "images_with_fp": 0,
        }
    )
    for r in rows:
        vis_s = (r.get("n_visible_fruitbodies") or "").strip()
        if not vis_s:
            empty += 1
            continue
        vis = int(vis_s)
        ok = int(r["n_boxes_ok"])
        fn = int(r["n_fn_missed"])
        fp = int(r["n_fp_extra"])
        b = (r.get("gt_bucket") or "UNK").strip()
        s = buckets[b]
        s["n_images"] += 1
        s["n_visible"] += vis
        s["n_boxes_ok"] += ok
        s["n_fn"] += fn
        s["n_fp"] += fp
        if fn > 0:
            s["images_with_fn"] += 1
        if fp > 0:
            s["images_with_fp"] += 1
        if abs((ok + fn) - vis) > 2:
            inconsist += 1
        if (r.get("notes") or "").strip():
            notes_n += 1

    tot = {
        "n_images": 0,
        "n_visible": 0,
        "n_boxes_ok": 0,
        "n_fn": 0,
        "n_fp": 0,
        "images_with_fn": 0,
        "images_with_fp": 0,
    }
    for s in buckets.values():
        for k in tot:
            tot[k] += s[k]

    def rates(s: dict[str, int]) -> dict[str, float]:
        vis = s["n_visible"]
        imgs = s["n_images"]
        return {
            "fn_rate_fruitbodies": (s["n_fn"] / vis) if vis else 0.0,
            "coverage_boxes_ok_over_visible": (s["n_boxes_ok"] / vis) if vis else 0.0,
            "images_with_fn_frac": (s["images_with_fn"] / imgs) if imgs else 0.0,
            "images_with_fp_frac": (s["images_with_fp"] / imgs) if imgs else 0.0,
        }

    payload = {
        "source_csv": "docs/thesis_results/fn_audit_sheet.csv",
        "n_csv_rows": len(rows),
        "n_empty": empty,
        "n_notes": notes_n,
        "n_inconsistent_ok_fn_vs_visible_gt2": inconsist,
        "total": {**tot, **rates(tot)},
        "by_bucket": {b: {**s, **rates(s)} for b, s in sorted(buckets.items())},
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
