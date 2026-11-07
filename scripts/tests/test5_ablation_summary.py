#!/usr/bin/env python3
# Test 5 – Ablation / architecture comparison summary table

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from _paths import DEFAULT_OUTPUT  # noqa: E402

# Static ablation figures measured during thesis experiments.
# Update these values if a new training / export run is performed.
ABLATION_ROWS = [
    {
        "Model Architecture": "YOLO11m (PyTorch)",
        "Params (M)": "20.1",
        "File Size (MB)": "38.8",
        "mAP50 (Test)": "82.8%",
        "mAP50-95 (Test)": "78.2%",
        "Inference Speed (GPU)": "5.1 ms",
    },
    {
        "Model Architecture": "YOLO11m (TFLite FP16)",
        "Params (M)": "20.1",
        "File Size (MB)": "38.58",
        "mAP50 (Test)": "82.8%",
        "mAP50-95 (Test)": "78.2%",
        "Inference Speed (GPU)": "~15-25 ms (NPU/mobile)",
    },
    {
        "Model Architecture": "YOLO11n (Baseline)",
        "Params (M)": "2.66",
        "File Size (MB)": "5.6",
        "mAP50 (Test)": "48.0%",
        "mAP50-95 (Test)": "42.1%",
        "Inference Speed (GPU)": "1.1 ms",
    },
]

COLUMNS = list(ABLATION_ROWS[0].keys())


def to_markdown(rows: list[dict[str, str]]) -> str:
    header = "| " + " | ".join(COLUMNS) + " |"
    sep = "| " + " | ".join("---" for _ in COLUMNS) + " |"
    body = [
        "| " + " | ".join(str(row[c]) for c in COLUMNS) + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body])


def to_latex(rows: list[dict[str, str]]) -> str:
    # Export a booktabs-style tabular for the thesis.
    lines = [
        r"\begin{tabular}{|l|c|c|c|c|c|}",
        r"\hline",
        r"\textbf{Model} & \textbf{Params (M)} & \textbf{Size (MB)} & "
        r"$\boldsymbol{mAP_{50}}$ & $\boldsymbol{mAP_{50\text{--}95}}$ & "
        r"\textbf{GPU latency} \\",
        r"\hline \hline",
    ]
    for row in rows:
        lines.append(
            f"{row['Model Architecture']} & "
            f"{row['Params (M)']} & "
            f"{row['File Size (MB)']} & "
            f"{row['mAP50 (Test)']} & "
            f"{row['mAP50-95 (Test)']} & "
            f"{row['Inference Speed (GPU)']} \\\\"
        )
        lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test 5: write ablation study summary table for the thesis."
    )
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    print("[INFO] Ablation study summary table")
    print()
    print(to_markdown(ABLATION_ROWS))

    csv_path = args.output / "ablation_study.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(ABLATION_ROWS)

    tex_path = args.output / "ablation_study.tex"
    tex_path.write_text(to_latex(ABLATION_ROWS), encoding="utf-8")

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {tex_path}")
    print("[OK] Test 5 finished")


if __name__ == "__main__":
    main()
