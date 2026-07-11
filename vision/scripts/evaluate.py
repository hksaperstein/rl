#!/usr/bin/env python3
"""Thin CLI over evaluation.detection_eval (2026-07-10 platform refactor) --
see src/evaluation/detection_eval.py for the implementation. CLI args and
behavior are unchanged from the pre-refactor scripts/evaluate.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.detection_eval import (  # noqa: E402,F401
    CLASS_NAMES,
    MERGED_CLASS_NAMES,
    eval_real,
    main,
    merge_d10,
    yolo_label_to_gt,
)

if __name__ == "__main__":
    main()
