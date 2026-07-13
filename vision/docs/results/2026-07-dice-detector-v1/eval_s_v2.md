# Eval: s_v2

Weights: `models/runs/s_v2/weights/best.pt`

## Synthetic val (7 classes — optimistic, see spec amendment)

| class | mAP50 | mAP50-95 |
|---|---|---|
| d4 | 0.992 | 0.972 |
| d6 | 0.989 | 0.970 |
| d8 | 0.974 | 0.947 |
| d10 | 0.967 | 0.940 |
| d10_pct | 0.975 | 0.945 |
| d12 | 0.994 | 0.985 |
| d20 | 0.995 | 0.990 |

## Real test (frozen, d10_pct merged into d10)

| class | mAP50 | mAP50-95 |
|---|---|---|
| d4 | 0.784 | 0.174 |
| d6 | 0.275 | 0.077 |
| d8 | 0.442 | 0.105 |
| d10 | 0.534 | 0.233 |
| d12 | 0.946 | 0.410 |
| d20 | 0.907 | 0.284 |

Overlays: `models/eval/s_v2/overlays/` (30 images)
