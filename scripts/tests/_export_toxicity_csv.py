#!/usr/bin/env python3

from __future__ import annotations

import csv
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KT = (
    REPO
    / "android"
    / "app"
    / "src"
    / "main"
    / "java"
    / "com"
    / "pw"
    / "mushroom"
    / "model"
    / "MushroomRegistry.kt"
)
OUT = Path(__file__).resolve().parent / "toxicity_registry.csv"
PAT = re.compile(
    r'(\d+)\s+to\s+MushroomSpecies\(\d+,\s+"([^"]+)",\s+Toxicity\.(\w+)\)'
)


def csv_n(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as fh:
        return sum(1 for _ in csv.DictReader(fh))


def main() -> None:
    n_csv = csv_n(OUT)
    if n_csv == 147:
        print(f"[OK] reuse {OUT} n=147 (skip Kotlin)")
        return

    if not KT.is_file():
        raise SystemExit(
            f"no {KT} and CSV has {n_csv} rows (need 147). "
            "Copy scripts/tests/toxicity_registry.csv from the thesis laptop."
        )

    text = KT.read_text(encoding="utf-8")
    rows = PAT.findall(text)
    if len(rows) == 147:
        with OUT.open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["class_id", "name", "toxicity"])
            w.writerows(rows)
        print(f"[OK] wrote {OUT.name} n={len(rows)}")
        return

    raise SystemExit(
        f"expected 147 species, Kotlin gave {len(rows)}, CSV has {n_csv}. "
        "Copy scripts/tests/toxicity_registry.csv onto this PC and re-run."
    )


if __name__ == "__main__":
    main()
