# Vision platform (`vision/` subtree)

Monorepo-merged 2026-07-10 (former standalone Dice-Detection repo, full
git history preserved via subtree; spec:
`docs/superpowers/specs/2026-07-10-monorepo-merge-design.md`). Blender
synthetic-data generation → dataset plumbing → YOLO detection training/
eval → ONNX+manifest export. Runtime: `vision/.venv` (cu128 torch), never
Isaac's python — path decides the interpreter.

## First study: dice-detector-v1 (2026-07-10)

Full write-up: `vision/docs/results/2026-07-dice-detector-v1/summary.md`.
Synthetic-only YOLO11s: synthetic val mAP50 0.984 (leaky/optimistic) but
real-photo per-class mean 0.532 — d12 0.936/d20 0.855 transfer, d8 0.090/
d10 0.097 collapse via systematic *upward* misclassification (d8→d20,
d10→d12/d20). Detection/localization fine; classifier head wrong.
Hypothesis: apparent-size-as-class-cue confound (synthetic multi-die
scenes make in-frame size predict class; real test photos are single-die
close-ups). +13k real fine-tune images closes everything (≥0.989,
within-collection caveat). Tested by datagen-v2 close-up slice
(see [[sim-to-real-transfer]] once written).

## Second study: datagen-v2 close-up slice (verdict 2026-07-13)

Spec + verdict:
`vision/docs/superpowers/specs/2026-07-11-datagen-v2-closeup-design.md`.
**Hypothesis SUPPORTED** — adding a 3,000-image close-up slice whose
camera distance is sampled independently of die class (target
frame-height fraction per scene) raised real-test mAP50 d8 0.090→0.442,
d10 0.097→0.534 (pre-registered ≥0.40 threshold: both pass) with the
d12/d20 guard intact (0.946/0.907, slightly up). Confirms the
apparent-size confound as a major contributor; the model can read
geometry/numeral cues when the data stops letting it cheat on size.

Two open items for iteration 3: (1) **d6 regressed 0.519→0.275**
(outside the pre-registered guard, mechanism unknown — possibly d6's
cue set is more scale-sensitive); (2) absolute mAP50-95 still far below
`s_plus_r`'s real-fine-tuned 0.71–0.77 — close-up slice fixes the class
confound, not the full synthetic-to-real gap (glyph styles, photographic
degradation remain the deferred suspects).

Methodology note: synthetic-val was ~identical for `s` and `s_v2`
(≥0.966 everywhere) while real-test moved by ±0.4 — synthetic val is
structurally blind to this failure mode; only the frozen real test set
can issue verdicts (see [[one-authoritative-eval-protocol]] practice).

Methodology lessons that generalize: frozen real test set as the
cross-iteration benchmark; group-aware splits for augmented-duplicate
community data; never tune hyperparameters against the frozen test set.
