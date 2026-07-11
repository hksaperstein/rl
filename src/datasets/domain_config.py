"""Loads a domain's config/plugin metadata from configs/domains/<name>.yaml.

Domain = config + datagen plugin (see
docs/superpowers/specs/2026-07-10-training-platform-architecture-design.md):
adding a new detection domain means a new configs/domains/*.yaml file, not
edits to datasets/training/evaluation/export library code. This loader is
the one place every one of those packages goes to read a domain's class
list/order, real-data sources, remap tables, and exclusions.
"""
import functools
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DOMAINS_DIR = REPO_ROOT / "configs" / "domains"


@functools.lru_cache(maxsize=None)
def load_domain_config(name: str = "dice") -> dict:
    """Load and parse configs/domains/<name>.yaml. Cached per name."""
    path = DOMAINS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no domain config at {path}")
    return yaml.safe_load(path.read_text())
