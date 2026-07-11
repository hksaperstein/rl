# Dice Detector v1 — Training & Sim-to-Real Evaluation

**Date:** 2026-07-10
**Status:** Approved (first iteration; findings feed back into data generation)

## Goal

Train a real-time-capable object detector for the 7 dice classes on the
synthetic `data/detection_v1` dataset, and **measure** (not assume) how well
the synthetic data transfers to real photographs. The detector's eventual
destination is a robot camera in the manipulation platform, which biases
choices toward single-stage real-time models and easy export (ONNX/TensorRT).

This is explicitly iteration 1 of a loop: the real-image evaluation exists to
produce actionable feedback for the Blender data-generation pipeline
(`src/dice_gen/`), per this repo's mission of proving data trustworthiness
rather than asserting it.

## Success criteria

1. Training completes on the local RTX 5070 Ti and produces per-class
   mAP50 / mAP50-95 on both a synthetic val split and a **real-photo test
   set** the model never trained on.
2. The synthetic-only vs synthetic+real comparison (below) yields a concrete
   transfer-gap number per class.
3. Visual verification: prediction-overlay images on real photos are
   produced and reviewed, not just scalar metrics.

No specific mAP threshold is a pass/fail gate for iteration 1 — the
deliverable is the measurement and the trained baseline.

## Approach (decided)

**Ultralytics YOLO11s, fine-tuned from the COCO-pretrained checkpoint.**
Rationale: pretrained-on-real-photos backbone is the single largest
sim-to-real mitigation available for free; built-in training loop,
augmentation, COCO-style metrics, confusion matrix; one-line ONNX/TensorRT
export for robot deployment; fast iteration (~tens of minutes per run on a
5070 Ti). AGPL license acceptable for this private/personal project.

Rejected alternatives: hand-rolled torchvision RetinaNet/Faster R-CNN (more
code, slower iteration, Faster R-CNN not real-time), RT-DETR (heavier,
overkill for 7 rigid classes).

## Data

### Synthetic (training + synthetic val)

- Source: `data/detection_v1/` — 10,000 renders (1024×768), 51,505 COCO
  boxes, 7 classes (`d4, d6, d8, d10, d10_pct, d12, d20`), ~7k boxes/class.
- Converted COCO → YOLO txt format into `data/yolo/` (gitignored, derived
  artifact).
- **Split: image-level random 90/10 (seed 42).** The originally-specified
  asset-set-level split is infeasible: 9,948 of 10,000 images contain dice
  from more than one of the 600 asset sets, so no image partition keeps
  sets disjoint. Consequence: synthetic val mAP is optimistic (same die
  instances appear in train and val) and is used only for training
  monitoring — the frozen real test set is the sole generalization
  measure. Feedback to the generator for the next dataset iteration:
  sample each image's dice from a split-designated pool of asset sets so a
  leak-free synthetic val exists.

### Real (fine-tune slice + frozen test set)

- Sources (Roboflow Universe, downloaded via free API key):
  - "DND Dices" by Sunib (~5,900 images, d4/d6/d8/d10/d12/d20 boxes)
  - "D&D Dice Detection" by Thomas Phillips (~1,000 images)
  - Exact class lists verified at download time; any additional public
    d4–d20 detection set may be substituted/added if one of these proves
    unusable (broken labels, license, download failure).
- Label remapping to our taxonomy at ingest. Real sets do not distinguish
  `d10` vs `d10_pct`; therefore **real-image evaluation merges `d10_pct`
  into `d10`** (model still has 7 classes; merge applied at eval time).
- Split: a fine-tune slice (for variant S+R) and a **frozen real test set**
  (target ≥300 images, stratified across both sources) that no variant ever
  trains or validates on.
- Stored under `data/real/` (gitignored); the ingest script and remap tables
  are committed.

## Experiments

Two training runs, identical eval protocol:

| Variant | Train data | Purpose |
|---------|-----------|---------|
| **S**   | synthetic only | Measures raw sim-to-real transfer of `detection_v1` |
| **S+R** | synthetic + real fine-tune slice | Upper reference: what a bit of real data buys |

Both evaluated on (a) synthetic val split, (b) frozen real test set.
The per-class gap between S and S+R on (b) is the primary output — it
localizes which classes/appearances the synthetic pipeline fails to cover.

## Training configuration

- Model: `yolo11s.pt` (pretrained), imgsz 640, ~60 epochs, early-patience
  default; batch sized to 16 GB VRAM (start 32, back off on OOM).
- Default Ultralytics augmentation (mosaic, HSV, flips). No custom
  augmentation in iteration 1.
- Determinism: fixed seed recorded in the run config; runs logged under
  `models/runs/` (gitignored) with a committed results summary.
- **Environment gate (step zero):** RTX 5070 Ti is Blackwell (sm_120) and
  requires PyTorch ≥ 2.7 built for CUDA 12.8. Verify
  `torch.cuda.is_available()` and a smoke-train before real runs. Dedicated
  venv for this repo (Isaac Lab's python is not used here — this project is
  Isaac-free).

## Evaluation & deliverables

- Per-class mAP50 and mAP50-95 on synthetic val and real test, for S and
  S+R (4 result tables); confusion matrices.
- Prediction-overlay grids on ~30 real test images per variant, reviewed
  visually.
- Filled-in `scripts/train.py` and `scripts/evaluate.py` (currently empty
  stubs), plus `scripts/convert_coco_to_yolo.py` and
  `scripts/prepare_real_data.py`.
- A short results write-up in `docs/` including the transfer-gap table and
  concrete recommendations back to the data-generation pipeline (e.g.
  under-covered materials/lighting/scale ranges).

## Out of scope (iteration 1)

- Face-value reading (pip/number recognition) — classes are die *types* only.
- Custom augmentation tuning, model-size sweeps, TensorRT deployment.
- Photographing the user's own dice (deferred; the robot-camera test set is
  a later iteration once the camera exists).
- Changes to the Blender generator itself — this iteration only produces
  the evidence for those changes.

## Risks

- **Roboflow download requires a free API key** — user-provided at execution
  time; fallback is another public mirror or a different public dataset.
- Real-set label quality is unverified community data; ingest includes a
  spot-check (visualize ~20 random labeled images per source before trust).
- `d10` vs `d10_pct` ambiguity in real labels is handled by eval-time merge,
  but if a real source *does* label percentile dice as d10, the S+R
  fine-tune could mislabel-train `d10_pct`; ingest spot-check covers this.
