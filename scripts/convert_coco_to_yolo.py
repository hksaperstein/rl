#!/usr/bin/env python3
"""Thin CLI over datasets.coco_yolo (2026-07-10 platform refactor) -- see
src/datasets/coco_yolo.py for the implementation. CLI args and behavior are
unchanged from the pre-refactor scripts/convert_coco_to_yolo.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets.coco_yolo import CLASS_NAMES, build_split, coco_bbox_to_yolo, main  # noqa: E402,F401

if __name__ == "__main__":
    main()
