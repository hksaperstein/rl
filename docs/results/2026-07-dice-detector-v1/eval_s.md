# Eval: s

Weights: `models/runs/s/weights/best.pt`

## Synthetic val (7 classes — optimistic, see spec amendment)

| class | mAP50 | mAP50-95 |
|---|---|---|
| d4 | 0.991 | 0.971 |
| d6 | 0.989 | 0.971 |
| d8 | 0.978 | 0.943 |
| d10 | 0.966 | 0.936 |
| d10_pct | 0.977 | 0.947 |
| d12 | 0.994 | 0.983 |
| d20 | 0.995 | 0.991 |

## Real test (frozen, d10_pct merged into d10)

| class | mAP50 | mAP50-95 |
|---|---|---|
| d4 | 0.695 | 0.162 |
| d6 | 0.519 | 0.132 |
| d8 | 0.090 | 0.018 |
| d10 | 0.097 | 0.034 |
| d12 | 0.936 | 0.403 |
| d20 | 0.855 | 0.264 |

Overlays: `models/eval/s/overlays/` (30 images)
