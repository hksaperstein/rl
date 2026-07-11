#!/usr/bin/env python3
"""Evaluate a trained dice detector on (a) the synthetic val split and
(b) the frozen real test set, with d10_pct merged into d10 for (b).

Real-set metrics are computed via pycocotools in the merged 6-class space;
synthetic metrics come from ultralytics' own val() in the full 7-class space.

Moved from scripts/evaluate.py (2026-07-10 platform refactor, see
docs/superpowers/specs/2026-07-10-training-platform-architecture-design.md)
-- scripts/evaluate.py is now a thin CLI over this module. Behavior is
unchanged; CLASS_NAMES/MERGED_CLASS_NAMES now come from
configs/domains/dice.yaml. This is the first (and, for now, only)
Evaluator-protocol implementation the platform spec describes.
"""
import argparse
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from datasets.domain_config import load_domain_config

_CFG = load_domain_config("dice")
CLASS_NAMES = list(_CFG["classes"])
MERGED_CLASS_NAMES = list(_CFG["merged_classes"])
REPO = Path(__file__).resolve().parents[2]


def merge_d10(class_idx):
    """Map 7-class index to merged 6-class index (d10_pct -> d10)."""
    if class_idx == 4:
        return 3
    return class_idx if class_idx < 4 else class_idx - 1


def eval_real(pred_records, gt_records, class_names):
    """Per-class (mAP50, mAP50-95) via COCOeval.

    gt_records: [{image_id, class_idx (merged), bbox [x,y,w,h] px, width, height}]
    pred_records: [{image_id, class_idx (merged), bbox [x,y,w,h] px, score}]
    """
    images = {}
    for r in gt_records:
        images[r["image_id"]] = {"id": r["image_id"],
                                 "width": r["width"], "height": r["height"]}
    gt = {
        "images": list(images.values()),
        "annotations": [
            {"id": i + 1, "image_id": r["image_id"],
             "category_id": r["class_idx"] + 1, "bbox": r["bbox"],
             "area": r["bbox"][2] * r["bbox"][3], "iscrowd": 0}
            for i, r in enumerate(gt_records)
        ],
        "categories": [{"id": i + 1, "name": n} for i, n in enumerate(class_names)],
    }
    coco_gt = COCO()
    coco_gt.dataset = gt
    coco_gt.createIndex()
    dets = [
        {"image_id": r["image_id"], "category_id": r["class_idx"] + 1,
         "bbox": r["bbox"], "score": r["score"]}
        for r in pred_records
    ]
    if not dets:
        # COCOeval cannot be constructed with a None detections object;
        # with no predictions at all, every class trivially scores 0.
        return {name: (0.0, 0.0) for name in class_names}
    coco_dt = coco_gt.loadRes(dets)
    results = {}
    for i, name in enumerate(class_names):
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.params.catIds = [i + 1]
        ev.evaluate(); ev.accumulate(); ev.summarize()
        # stats[1] = mAP50, stats[0] = mAP50-95 (COCO summarize convention)
        results[name] = (max(ev.stats[1], 0.0), max(ev.stats[0], 0.0))
    return results


def yolo_label_to_gt(label_path, image_id, w, h):
    recs = []
    for line in label_path.read_text().splitlines():
        c, cx, cy, bw, bh = (float(v) for v in line.split())
        recs.append({
            "image_id": image_id,
            "class_idx": merge_d10(int(c)),
            "bbox": [(cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h],
            "width": w, "height": h,
        })
    return recs


def main():
    from ultralytics import YOLO

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--name", required=True, help="variant label, e.g. s")
    ap.add_argument("--overlay-n", type=int, default=30)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    model = YOLO(str(args.weights))
    out_md = REPO / f"docs/results/2026-07-dice-detector-v1/eval_{args.name}.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    overlay_dir = REPO / f"models/eval/{args.name}/overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    # (a) synthetic val, full 7-class space, ultralytics metrics
    synth = model.val(data=str(REPO / "data/yolo/dice.yaml"), split="val")
    synth_rows = [
        f"| {CLASS_NAMES[c]} | {synth.box.ap50[i]:.3f} | {synth.box.ap[i]:.3f} |"
        for i, c in enumerate(synth.box.ap_class_index)
    ]

    # (b) frozen real test, merged 6-class space
    test_images = sorted((REPO / "data/real/test/images").iterdir())
    gt_records, pred_records = [], []
    for image_id, img_path in enumerate(test_images, start=1):
        res = model.predict(str(img_path), conf=args.conf, verbose=False)[0]
        h, w = res.orig_shape
        lbl = REPO / "data/real/test/labels" / (img_path.stem + ".txt")
        gt_records += yolo_label_to_gt(lbl, image_id, w, h)
        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            pred_records.append({
                "image_id": image_id,
                "class_idx": merge_d10(int(box.cls)),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(box.conf),
            })
        if image_id <= args.overlay_n:
            res.save(filename=str(overlay_dir / img_path.name))

    real = eval_real(pred_records, gt_records, MERGED_CLASS_NAMES)
    real_rows = [f"| {n} | {m50:.3f} | {m5095:.3f} |"
                 for n, (m50, m5095) in real.items()]

    out_md.write_text(
        f"# Eval: {args.name}\n\nWeights: `{args.weights}`\n\n"
        "## Synthetic val (7 classes — optimistic, see spec amendment)\n\n"
        "| class | mAP50 | mAP50-95 |\n|---|---|---|\n"
        + "\n".join(synth_rows) +
        "\n\n## Real test (frozen, d10_pct merged into d10)\n\n"
        "| class | mAP50 | mAP50-95 |\n|---|---|---|\n"
        + "\n".join(real_rows) +
        f"\n\nOverlays: `models/eval/{args.name}/overlays/` "
        f"({min(len(test_images), args.overlay_n)} images)\n"
    )
    print(f"[DONE] wrote {out_md}")


if __name__ == "__main__":
    main()
