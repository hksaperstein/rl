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
within-collection caveat). Being tested by datagen-v2 close-up slice
(see [[sim-to-real-transfer]] once written).

Methodology lessons that generalize: frozen real test set as the
cross-iteration benchmark; group-aware splits for augmented-duplicate
community data; never tune hyperparameters against the frozen test set.
