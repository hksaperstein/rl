# Variant S: Synthetic-Only Training

## Configuration

- **Variant**: S (synthetic-only baseline)
- **Dataset**: `data/yolo/dice.yaml` — 9000 synthetic train images, 1000 synthetic val images
- **Model**: YOLO11s pretrained
- **Hyperparameters**:
  - Seed: 42
  - Image size (imgsz): 640
  - Epochs: 60
  - Batch size: 32

## Final Validation Metrics (Epoch 60)

| Class | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| all | 0.978 | 0.934 | 0.984 | 0.963 |
| d4 | 0.987 | 0.962 | 0.991 | 0.972 |
| d6 | 0.989 | 0.957 | 0.989 | 0.972 |
| d8 | 0.98 | 0.918 | 0.978 | 0.942 |
| d10 | 0.954 | 0.859 | 0.966 | 0.935 |
| d10_pct | 0.958 | 0.884 | 0.977 | 0.944 |
| d12 | 0.99 | 0.976 | 0.994 | 0.983 |
| d20 | 0.99 | 0.986 | 0.995 | 0.991 |

**Note**: Synthetic validation set shares die instances with training set (see spec amendment for rationale) — these metrics are optimistic. Real-world generalization metrics from held-out real test set are reported separately in `eval_s.md`.

## Timing

- **Wall time**: 2489.3 seconds (~41.5 minutes)

## Environment

- **Ultralytics**: 8.4.92
- **PyTorch**: 2.11.0+cu128
