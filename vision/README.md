# Dice Detection

Purpose: train models to recognize and classify dice by shape/sides (d4, d6, d8, d10, d12, d20).

Contents:
- data/ : dataset (raw and processed). Keep large files out of VCS.
- notebooks/ : EDA and training notebooks
- src/ : data loaders, augmentation, model code
- models/ : saved checkpoints (gitignored)
- scripts/ : training and evaluation scripts

Suggested approaches: object detection (YOLO/Detectron) or segmentation + classification. Label format: COCO or YOLO.

Evaluation: per-class accuracy, mAP, confusion matrix.

Workflow notes: collect images for each dice type in varied lighting and orientations, include synthetic renders, annotate with bounding boxes or masks.
