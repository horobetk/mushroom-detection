#!/usr/bin/env python3


from pathlib import Path

# Repository root: scripts/tests/ -> ../..
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL = Path(
    "E:/Kiril_Horobets/mushroom_data/runs/fine_147classes_v2/weights/best.pt"
)
DEFAULT_DATA = Path("E:/Kiril_Horobets/mushroom_data/mushroom_final/data.yaml")
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "thesis_results"

# Asymmetric Android display thresholds (see MushroomRegistry).
CONF_TOXIC = 0.18
CONF_EDIBLE = 0.60
CONF_NEUTRAL_MIN = 0.20
CONF_NEUTRAL_MAX = 0.55

# Legacy single-value alias used by threshold_sweep highlight.
ANDROID_CONF_THRESHOLD = CONF_EDIBLE
