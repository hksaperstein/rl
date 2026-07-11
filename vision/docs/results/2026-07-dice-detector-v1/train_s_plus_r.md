# Variant S+R: Synthetic + Real Finetune Training

## Configuration

- **Variant**: S+R (synthetic-only train + real finetune slice)
- **Dataset**: `data/yolo/dice_s_plus_r.yaml` — 9000 synthetic train images + 13,255 real finetune images (from `data/real/finetune`) mixed as training data, 1000 synthetic val images
- **Model**: YOLO11s pretrained
- **Hyperparameters**:
  - Seed: 42
  - Image size (imgsz): 640
  - Epochs: 60
  - Batch size: 32

## Final Validation Metrics (Epoch 60)

| Class | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| all | 0.963 | 0.943 | 0.984 | 0.963 |
| d4 | 0.989 | 0.974 | 0.992 | 0.97 |
| d6 | 0.98 | 0.961 | 0.99 | 0.97 |
| d8 | 0.954 | 0.932 | 0.98 | 0.945 |
| d10 | 0.92 | 0.864 | 0.966 | 0.936 |
| d10_pct | 0.925 | 0.9 | 0.974 | 0.941 |
| d12 | 0.984 | 0.978 | 0.994 | 0.985 |
| d20 | 0.99 | 0.988 | 0.995 | 0.991 |

**Note**: Synthetic validation set shares die instances with training set (see spec amendment for rationale) — these metrics are optimistic. Real-world generalization metrics from held-out real test set are reported separately in `eval_s_plus_r.md`.

## Timing

- **Wall time**: 6757.8 seconds (~1.88 hours)

## Environment

- **Ultralytics**: 8.4.92
- **PyTorch**: 2.11.0+cu128
