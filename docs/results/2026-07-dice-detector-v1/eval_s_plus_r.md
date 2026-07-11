# Eval: s_plus_r

Weights: `models/runs/s_plus_r/weights/best.pt`

## Synthetic val (7 classes — optimistic, see spec amendment)

| class | mAP50 | mAP50-95 |
|---|---|---|
| d4 | 0.992 | 0.970 |
| d6 | 0.990 | 0.969 |
| d8 | 0.979 | 0.946 |
| d10 | 0.966 | 0.936 |
| d10_pct | 0.974 | 0.943 |
| d12 | 0.994 | 0.984 |
| d20 | 0.995 | 0.991 |

## Real test (frozen, d10_pct merged into d10)

| class | mAP50 | mAP50-95 |
|---|---|---|
| d4 | 1.000 | 0.768 |
| d6 | 1.000 | 0.756 |
| d8 | 1.000 | 0.754 |
| d10 | 0.989 | 0.714 |
| d12 | 1.000 | 0.767 |
| d20 | 1.000 | 0.757 |

Overlays: `models/eval/s_plus_r/overlays/` (30 images)
