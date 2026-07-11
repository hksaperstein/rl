#!/usr/bin/env python3
"""Thin CLI over datasets.real_ingest (2026-07-10 platform refactor) -- see
src/datasets/real_ingest.py for the implementation. CLI args and behavior
are unchanged from the pre-refactor scripts/prepare_real_data.py. Requires
ROBOFLOW_API_KEY in the environment. Never commit the key.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets.real_ingest import (  # noqa: E402,F401
    CLASS_NAMES,
    SOURCE_NAME_MAP,
    SOURCES,
    EXCLUDE_STEM_PREFIXES,
    group_pairs_by_augmentation,
    main,
    remap_label_line,
    should_exclude_stem,
    split_groups,
)

if __name__ == "__main__":
    main()
