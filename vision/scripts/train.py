#!/usr/bin/env python3
"""Thin CLI over training.detection.yolo_trainer (2026-07-10 platform
refactor) -- see src/training/detection/yolo_trainer.py for the
implementation. CLI args and behavior are unchanged from the pre-refactor
scripts/train.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from training.detection.yolo_trainer import (  # noqa: E402,F401
    CLASS_NAMES,
    build_s_plus_r_yaml,
    build_s_v2_yaml,
    main,
)

if __name__ == "__main__":
    main()
