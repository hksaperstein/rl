# Dice Detector v1 — Results Summary & Sim-to-Real Findings

**Date:** 2026-07-10
**Spec:** `docs/superpowers/specs/2026-07-10-dice-detector-v1-design.md`
**Weights:** `models/runs/{s,s_plus_r}/weights/best.pt` (gitignored)

## Headline result

Training on synthetic data alone (variant S) produces a detector that is
near-perfect on synthetic validation (mAP50 0.984) but **fails severely and
class-specifically on real photos** (per-class mean mAP50 0.532, with d8 at
0.090 and d10 at 0.097). Adding the real fine-tune slice (variant S+R)
closes the gap entirely *within the test set's own photo collections* (real
mAP50 ≥0.989 every class; see within-collection caveat below) — the model
and task are not the limiter; **the synthetic data's coverage is.**

All real-set numbers in this document use the committed pycocotools
protocol (`eval_s.md` / `eval_s_plus_r.md`: d10_pct merged into d10,
conf 0.25). The un-merged 7-class `yolo val` runs used for the confusion
matrices report slightly different absolute values (e.g. d10 0.074);
rankings and conclusions are identical.

## S vs S+R on the frozen real test set (1,376 images, d10_pct merged into d10)

| class | S real mAP50 | S+R real mAP50 | transfer gap |
|---|---|---|---|
| d4  | 0.695 | 1.000 | 0.305 |
| d6  | 0.519 | 1.000 | 0.481 |
| d8  | **0.090** | 1.000 | **0.910** |
| d10 | **0.097** | 0.989 | **0.892** |
| d12 | 0.936 | 1.000 | 0.064 |
| d20 | 0.855 | 1.000 | 0.145 |

(Variant S synthetic-val mAP50 was ≥0.966 on every class — the synthetic
val split predicts nothing about this per-class transfer ranking.)

## Failure mechanism (confusion matrix + overlay review, variant S)

Not a detection failure — a **systematic classification shift up the
shape-complexity ladder** on real close-ups:

- True d10 → predicted d12 (37%) or d20 (41%); only 4% correct.
- True d8 → predicted d20 (52%); only 15% correct.
- True d6 → 26% predicted d20.
- d12 (100%) and d20 (96%) are correct — the top of the ladder has nowhere
  to shift to.
- 38% of background false positives are labeled d20 (over-prediction of the
  largest class).
- Overlays confirm: real d10s are boxed confidently (conf ≈0.94) and labeled
  d12. Localization works; the class head is wrong.

**Working hypothesis (for datagen v2):** in the synthetic scenes, dice are
rendered at realistic *relative* physical sizes in multi-die tabletop
scenes, so apparent in-frame size correlates strongly with class (a d20 is
physically larger than a d8). The real test photos are single-die extreme
close-ups — every die is huge in frame, which the model reads as "large die
⇒ d12/d20". Secondary suspects: synthetic glyph-style variety (e.g.
`cjk_numerals`, present in the synthetic annotations) vs. plain arabic
numerals on real dice; real-photo blur/noise absent from clean renders.

## Recommendations back to the data-generation pipeline

1. **Decouple apparent size from class** (highest priority, directly tests
   the hypothesis): render single-die and few-die close-ups across the full
   focal/framing range so every class appears at every apparent scale.
2. **Split-designated asset pools** so a leak-free synthetic val exists
   (carried over from the spec amendment; today's synthetic val is
   saturated at 0.98+ and predicts nothing about transfer).
3. **Weight glyph-style distribution toward real-world conventions**
   (arabic numerals dominant); keep exotic styles as a minority slice.
4. **Add photographic degradation** (defocus blur, sensor noise, lower-key
   lighting) — real mAP50-95 is deflated even where mAP50 is perfect,
   though loose community GT boxes confound that column (see caveats).
5. **d8 and d10 geometry/material review**: the two worst classes; verify
   the synthetic octahedron/trapezohedron proportions and engraving
   legibility against real dice photos.

## Caveats on the S+R result

- **Within-collection evaluation.** The fine-tune slice and the frozen test
  set come from the same two photo collections (same physical dice, same
  cameras/settings). Group-aware splitting removed image-level duplicate
  leakage (verified: zero group overlap), but the same *physical die*
  appears in both splits in different photos. S+R's 0.995 is therefore
  within-collection performance, not proof of open-world generalization.
- **Single-die bias.** Both real sources are ~entirely single-die close-ups;
  multi-die clutter scenes (which the synthetic data covers heavily) are
  essentially untested on real photos.
- **Loose GT boxes** in the community datasets deflate real mAP50-95 for
  both variants; mAP50 is the trustworthy real-set column.
- A stronger iteration-2 test set: a third, disjoint source — ideally the
  user's own dice photographed with the robot's actual camera (deferred in
  the spec).

## Run/environment facts

- YOLO11s pretrained, imgsz 640, 60 epochs, batch 32, seed 42,
  deterministic; ultralytics 8.4.92, torch 2.11.0+cu128, RTX 5070 Ti.
- Wall time: S 41.5 min (9,000 train images), S+R 1.88 h (22,255).
- Full tables: `train_s.md`, `train_s_plus_r.md`, `eval_s.md`,
  `eval_s_plus_r.md`. Confusion matrices:
  `models/eval/{s,s_plus_r}/real_cm/` (gitignored artifacts).
- Data-quality defects caught during ingest (all fixed pre-training):
  Roboflow augmented-duplicate split leakage (group-aware split),
  under-labeled multi-die scenes (excluded), polygon/bbox label format mix
  (converted).
