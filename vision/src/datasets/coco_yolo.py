#!/usr/bin/env python3
"""Convert a COCO detection json + images dir to ultralytics YOLO layout.

Images are symlinked (not copied); labels are written as YOLO txt.
Split is image-level random (see spec amendment 2026-07-10: an
asset-set-level split is infeasible because 99.5% of images mix sets).

Moved from scripts/convert_coco_to_yolo.py (2026-07-10 platform refactor,
see docs/superpowers/specs/2026-07-10-training-platform-architecture-design.md)
-- scripts/convert_coco_to_yolo.py is now a thin CLI over this module.
Behavior is unchanged; CLASS_NAMES now comes from configs/domains/dice.yaml
instead of being hardcoded (this repo previously had this same list
duplicated 4x across scripts/).
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from datasets.domain_config import load_domain_config

CLASS_NAMES = list(load_domain_config("dice")["classes"])


def coco_bbox_to_yolo(bbox, img_w, img_h):
    """COCO [x, y, w, h] (top-left, pixels) -> YOLO (cx, cy, w, h) normalized."""
    x, y, w, h = bbox
    return ((x + w / 2) / img_w, (y + h / 2) / img_h, w / img_w, h / img_h)


def build_split(image_ids, val_frac=0.1, seed=42):
    """Deterministic random split. Returns (train_ids, val_ids) as sets."""
    ids = sorted(image_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = round(len(ids) * val_frac)
    return set(ids[n_val:]), set(ids[:n_val])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data/detection_v1"))
    ap.add_argument("--out", type=Path, default=Path("data/yolo"))
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    coco = json.loads((args.src / "coco.json").read_text())
    cat_to_idx = {}
    for cat in coco["categories"]:
        cat_to_idx[cat["id"]] = CLASS_NAMES.index(cat["name"])

    anns_by_image = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_image[a["image_id"]].append(a)

    train_ids, val_ids = build_split(
        [im["id"] for im in coco["images"]], args.val_frac, args.seed
    )

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0}
    for im in coco["images"]:
        split = "train" if im["id"] in train_ids else "val"
        stem = Path(im["file_name"]).stem
        lines = []
        for a in anns_by_image[im["id"]]:
            cx, cy, w, h = coco_bbox_to_yolo(a["bbox"], im["width"], im["height"])
            cls = cat_to_idx[a["category_id"]]
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        (args.out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        link = args.out / "images" / split / im["file_name"]
        if not link.exists():
            link.symlink_to((args.src / im["file_name"]).resolve())
        counts[split] += 1

    yaml_text = (
        f"path: {args.out.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES))
    )
    (args.out / "dice.yaml").write_text(yaml_text)
    print(f"[DONE] train={counts['train']} val={counts['val']} -> {args.out}")


if __name__ == "__main__":
    main()
