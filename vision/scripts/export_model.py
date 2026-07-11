#!/usr/bin/env python3
"""Thin CLI over export.onnx_export (2026-07-10 platform refactor) -- see
src/export/onnx_export.py for the implementation.

Usage:
  python scripts/export_model.py --weights models/runs/s_plus_r/weights/best.pt --name s_plus_r

Produces models/export/<name>/{model.onnx,manifest.json}.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from export.onnx_export import export, main  # noqa: E402,F401

if __name__ == "__main__":
    main()
