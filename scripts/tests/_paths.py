#!/usr/bin/env python3


from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL = Path(
    "E:/Kiril_Horobets/mushroom_data/runs/fine_147classes_v2/weights/best.pt"
)
DEFAULT_DATA = Path("E:/Kiril_Horobets/mushroom_data/mushroom_final/data.yaml")
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "thesis_results"

# Must match the asymmetric thresholds in the Android app.
CONF_TOXIC = 0.18
CONF_EDIBLE = 0.60
CONF_NEUTRAL_MIN = 0.20

# Single value highlighted by test3_threshold_sweep.
ANDROID_CONF_THRESHOLD = CONF_EDIBLE
