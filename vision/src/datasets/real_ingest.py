#!/usr/bin/env python3
"""Download public real-photo dice datasets from Roboflow, remap labels to
this repo's 7-class taxonomy, split into finetune + frozen test, and render
spot-check overlays for human label-quality review.

Requires ROBOFLOW_API_KEY in the environment. Never commit the key.

Moved from scripts/prepare_real_data.py (2026-07-10 platform refactor, see
docs/superpowers/specs/2026-07-10-training-platform-architecture-design.md)
-- scripts/prepare_real_data.py is now a thin CLI over this module.
Behavior is unchanged; CLASS_NAMES/SOURCE_NAME_MAP/SOURCES/
EXCLUDE_STEM_PREFIXES now come from configs/domains/dice.yaml instead of
being hardcoded here.
"""
import argparse
import os
import random
import shutil
from pathlib import Path

import cv2
import yaml

from datasets.domain_config import load_domain_config

_CFG = load_domain_config("dice")
CLASS_NAMES = list(_CFG["classes"])

# source class name (lowercased) -> our class name; None = drop the box.
# Real sets can't distinguish percentile d10s reliably -> merge into d10.
SOURCE_NAME_MAP = dict(_CFG["real_data"]["source_name_map"])

SOURCES = [dict(s) for s in _CFG["real_data"]["sources"]]

# Source-specific exclusions: stem prefixes to skip (before .rf. suffix).
# dd_dice's "all_dice_*" images are multi-dice scenes with only 1-2 boxes labeled;
# under-labeled images poison detection training (unlabeled dice become hard negatives).
# Visually verified 2026-07-10: 14 images (2 originals, 7 augmented copies each) excluded.
# All other dd_dice families single-die, correctly labeled; dnd_dices has no exclusions.
EXCLUDE_STEM_PREFIXES = {
    k: tuple(v) for k, v in _CFG["real_data"]["exclude_stem_prefixes"].items()
}


def should_exclude_stem(source_slug, stem_before_rf):
    """Check if a stem should be excluded for a given source."""
    if source_slug not in EXCLUDE_STEM_PREFIXES:
        return False
    for prefix in EXCLUDE_STEM_PREFIXES[source_slug]:
        if stem_before_rf.startswith(prefix):
            return True
    return False


def remap_label_line(line, src_names):
    """Remap one YOLO label line from a source dataset's class order to ours.

    Returns the remapped line, None to drop the box, or raises KeyError on a
    source class name we have no explicit decision for (forces a human call).

    Handles two source line shapes: a plain YOLO bbox (class + 4 values) is
    passed through unchanged apart from the class index; a YOLO polygon/
    segmentation annotation (class + an even count > 4 of x,y point pairs --
    one real source mixes both per image) is converted to its axis-aligned
    bounding box.
    """
    parts = line.split()
    src_name = src_names[int(parts[0])].lower()
    our_name = SOURCE_NAME_MAP[src_name]  # KeyError on purpose if unknown
    if our_name is None:
        return None
    out_idx = str(CLASS_NAMES.index(our_name))
    coord_parts = parts[1:]
    if len(coord_parts) == 4:
        return " ".join([out_idx] + coord_parts)
    # polygon annotation: convert to an axis-aligned bounding box
    coords = [float(v) for v in coord_parts]
    xs, ys = coords[0::2], coords[1::2]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    bw, bh = maxx - minx, maxy - miny
    return " ".join([out_idx] + [f"{v:.6f}" for v in (cx, cy, bw, bh)])


def group_pairs_by_augmentation(pairs):
    """Group (img, lbl) pairs by their augmentation stem (before .rf.).

    Roboflow creates augmented variants of the same original image by varying
    the .rf.<hash> suffix; we group all variants by their common stem to ensure
    augmented copies don't leak across train/test splits.

    Returns dict where key is stem_before_rf and value is list of (img, lbl) pairs.
    """
    groups = {}
    for img, lbl in pairs:
        # img.stem is like "d10_top002_jpg.rf.66e9e33a888ec47e61bdb36870c1c74e"
        # Extract the part before .rf.
        stem_before_rf = img.stem.split(".rf.")[0]
        if stem_before_rf not in groups:
            groups[stem_before_rf] = []
        groups[stem_before_rf].append((img, lbl))
    return groups


def split_groups(groups, test_frac, seed):
    """Split image groups into test and finetune.

    Groups are shuffled as atomic units (all augmented variants of the same
    original stay together), then split by group count to avoid test-set leakage.

    Args:
        groups: dict of stem_key -> list of (img, lbl) pairs
        test_frac: fraction of groups to assign to test
        seed: random seed for reproducibility

    Returns:
        (test_groups, finetune_groups) tuple of dicts with same structure
    """
    rng = random.Random(seed)
    group_keys = sorted(groups.keys())
    rng.shuffle(group_keys)
    n_test = round(len(group_keys) * test_frac)
    test_keys = set(group_keys[:n_test])

    test_groups = {k: groups[k] for k in test_keys}
    finetune_groups = {k: groups[k] for k in group_keys if k not in test_keys}

    return test_groups, finetune_groups


def download_source(src, dest_root):
    from roboflow import Roboflow
    rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
    project = rf.workspace(src["workspace"]).project(src["project"])
    versions = project.versions()
    version = max(versions, key=lambda v: int(str(v.version).split("/")[-1]))
    loc = str(dest_root / "raw" / src["slug"])
    version.download("yolov8", location=loc, overwrite=False)
    return Path(loc)


def collect_pairs(raw_dir):
    """Yield (img_path, label_path) across the download's train/valid/test dirs."""
    for split_dir in raw_dir.iterdir():
        img_dir = split_dir / "images"
        if not img_dir.is_dir():
            continue
        for img in sorted(img_dir.iterdir()):
            lbl = split_dir / "labels" / (img.stem + ".txt")
            if lbl.exists():
                yield img, lbl


def spotcheck(img_path, label_path, out_path):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    for line in label_path.read_text().splitlines():
        c, cx, cy, bw, bh = line.split()
        cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
        p1 = (int(cx - bw / 2), int(cy - bh / 2))
        p2 = (int(cx + bw / 2), int(cy + bh / 2))
        cv2.rectangle(img, p1, p2, (0, 255, 0), 2)
        cv2.putText(img, CLASS_NAMES[int(c)], (p1[0], max(p1[1] - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite(str(out_path), img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/real"))
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--spotcheck-n", type=int, default=20)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    for sub in ("finetune/images", "finetune/labels", "test/images",
                "test/labels", "spotcheck"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)

    total = {"finetune": 0, "test": 0}
    for src in SOURCES:
        raw = download_source(src, args.out)
        data_yaml = yaml.safe_load((raw / "data.yaml").read_text())
        src_names = data_yaml["names"]
        print(f"[{src['slug']}] source classes: {src_names}")

        pairs = list(collect_pairs(raw))
        # Group by augmentation stem (before .rf.) to avoid test-set leakage
        groups = group_pairs_by_augmentation(pairs)
        test_groups, finetune_groups = split_groups(groups, args.test_frac, args.seed)

        # Process test groups: keep only one image per group (lex-first)
        for stem_key, group_pairs in test_groups.items():
            if should_exclude_stem(src["slug"], stem_key):
                continue  # skip excluded stems (e.g., under-labeled multi-dice scenes)
            sorted_pairs = sorted(group_pairs, key=lambda p: str(p[0]))
            for idx, (img, lbl) in enumerate(sorted_pairs):
                if idx > 0:
                    continue  # keep only the first (lex-smallest) filename
                remapped = [
                    r for line in lbl.read_text().splitlines() if line.strip()
                    if (r := remap_label_line(line, src_names)) is not None
                ]
                if not remapped:
                    continue  # image with no mappable boxes: skip entirely
                stem = f"{src['slug']}_{img.stem}"
                shutil.copy(img, args.out / "test" / "images" / f"{stem}{img.suffix}")
                (args.out / "test" / "labels" / f"{stem}.txt").write_text(
                    "\n".join(remapped) + "\n")
                total["test"] += 1

        # Process finetune groups: keep all images (augmentation for training)
        for stem_key, group_pairs in finetune_groups.items():
            if should_exclude_stem(src["slug"], stem_key):
                continue  # skip excluded stems (e.g., under-labeled multi-dice scenes)
            for img, lbl in group_pairs:
                remapped = [
                    r for line in lbl.read_text().splitlines() if line.strip()
                    if (r := remap_label_line(line, src_names)) is not None
                ]
                if not remapped:
                    continue  # image with no mappable boxes: skip entirely
                stem = f"{src['slug']}_{img.stem}"
                shutil.copy(img, args.out / "finetune" / "images" / f"{stem}{img.suffix}")
                (args.out / "finetune" / "labels" / f"{stem}.txt").write_text(
                    "\n".join(remapped) + "\n")
                total["finetune"] += 1

        # spot-check overlays from this source's finetune slice
        done = 0
        for img in sorted((args.out / "finetune" / "images").glob(f"{src['slug']}_*")):
            if done >= args.spotcheck_n:
                break
            lbl = args.out / "finetune" / "labels" / (img.stem + ".txt")
            spotcheck(img, lbl, args.out / "spotcheck" / img.name)
            done += 1

    yaml_text = (
        f"path: {args.out.resolve()}\n"
        "train: finetune/images\n"
        "val: test/images\n"
        "names:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES))
    )
    (args.out / "real.yaml").write_text(yaml_text)
    print(f"[DONE] finetune={total['finetune']} test={total['test']}")


if __name__ == "__main__":
    main()
