#!/usr/bin/env python3
"""Export a trained detector to ONNX + a versioned model manifest.

This is the platform's deployment-contract implementation (see
docs/superpowers/specs/2026-07-10-training-platform-architecture-design.md,
"The three contracts binding the two repos" #1): the `rl`/robot side is
meant to consume only these artifacts (models/export/<variant>/model.onnx +
manifest.json), never this repo's internals. The manifest records what a
downstream consumer needs to use the model correctly and to know how
trustworthy it is: class list/order, input size, checkpoint provenance
(git SHA, dataset version), and the eval mAP50 table already written to
docs/results/2026-07-dice-detector-v1/eval_<variant>.md by
evaluation/detection_eval.py -- this module reads that table rather than
re-running evaluation, since the manifest is a checkpoint-provenance
artifact, not a place to recompute metrics.
"""
import argparse
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from datasets.domain_config import load_domain_config

REPO = Path(__file__).resolve().parents[2]

# Bumped when the underlying data (detection_v1 synthetic render + the real
# finetune/test ingest) changes; see dataset-manifest discipline in the
# platform spec ("Datasets are versioned artifacts with manifests"). No
# actual data-versioning registry exists yet, so this is a hand-maintained
# string, not a hash -- good enough for provenance until one exists.
DATASET_VERSION = "detection_v1+real_v1"

_TABLE_ROW_RE = re.compile(r"^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*$")


def git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO
    ).decode().strip()


def parse_eval_real_table(eval_md_path: Path) -> dict | None:
    """Parse the '## Real test' markdown table in an eval_<variant>.md file.

    Returns {class_name: {"mAP50": float, "mAP50_95": float}}, or None if
    the file doesn't exist (export can still proceed without it -- the
    manifest just records eval_real_test: null rather than failing export
    over a missing results doc).
    """
    if not eval_md_path.exists():
        return None
    lines = eval_md_path.read_text().splitlines()
    in_section = False
    results = {}
    for line in lines:
        if line.startswith("## Real test"):
            in_section = True
            continue
        if in_section and line.startswith("##"):
            break
        if not in_section:
            continue
        m = _TABLE_ROW_RE.match(line)
        if not m:
            continue
        name, map50, map5095 = (g.strip() for g in m.groups())
        if name in ("class", "---"):
            continue
        try:
            results[name] = {"mAP50": float(map50), "mAP50_95": float(map5095)}
        except ValueError:
            continue
    return results or None


def export(weights: Path, name: str, imgsz: int = 640,
           dataset_version: str = DATASET_VERSION) -> Path:
    from ultralytics import YOLO

    cfg = load_domain_config("dice")
    classes = list(cfg["classes"])

    out_dir = REPO / "models" / "export" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    exported_path = Path(model.export(format="onnx", imgsz=imgsz))
    final_onnx = out_dir / "model.onnx"
    shutil.copy(exported_path, final_onnx)

    eval_md = REPO / f"docs/results/2026-07-dice-detector-v1/eval_{name}.md"
    eval_real_test = parse_eval_real_table(eval_md)

    manifest = {
        "variant": name,
        "classes": classes,
        "imgsz": imgsz,
        "git_sha": git_sha(),
        "dataset_version": dataset_version,
        "weights_source": str(weights),
        "eval_source_doc": str(eval_md.relative_to(REPO)) if eval_md.exists() else None,
        "eval_real_test_map50": eval_real_test,
        "export_date_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--name", required=True, help="export variant label, e.g. s_plus_r")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--dataset-version", default=DATASET_VERSION)
    args = ap.parse_args()

    out_dir = export(args.weights, args.name, imgsz=args.imgsz,
                      dataset_version=args.dataset_version)
    print(f"[DONE] exported to {out_dir}")


if __name__ == "__main__":
    main()
