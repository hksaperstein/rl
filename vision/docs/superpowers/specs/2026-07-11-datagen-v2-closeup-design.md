# Datagen v2 — Close-Up Slice (Tier 1)

**Date:** 2026-07-11
**Status:** Proposed
**Prior art:** `docs/results/2026-07-dice-detector-v1/summary.md` (failure
analysis), `docs/superpowers/specs/2026-07-10-dice-detector-v1-design.md`
(eval protocol reused unchanged here)

## Hypothesis (falsifiable)

The synthetic-only YOLO11s detector's severe, class-specific real-photo
failure on d8 (mAP50 0.090) and d10 (0.097) — with systematic
upward misclassification (d8→d20 52%, d10→d12 37%/d20 41%) while d12/d20
transfer well (0.86–0.94) — is substantially caused by an
**apparent-size-as-class-cue confound** in `data/detection_v1`: training
scenes place 3–8 dice per image under one shared camera-distance/lens
sample per scene, so in-frame apparent size correlates with the die's
physical size (hence class), and the model has learned to read "large in
frame" as "d12/d20" rather than reading engraved-numeral/geometry cues.
Real test photos are near-uniformly single-die extreme close-ups (see
"Real framing measurement" below), which the model reads via the same
learned size cue as belonging to the large classes regardless of true
class.

**Falsifiable prediction:** adding a close-up synthetic slice, in which
camera distance is decoupled from die class (sampled per-scene from a
target frame-height fraction independent of which class is being
rendered), and mixing it into training, will raise real-test d8/d10
mAP50 substantially above baseline (0.090/0.097) without materially
damaging d12/d20's existing good transfer.

**Falsification condition:** if d8/d10 real mAP50 stays roughly at
baseline (say, doesn't clear ~0.40) despite the close-up slice
existing and being included in training, the apparent-size confound is
not the dominant mechanism (or the fix didn't actually decouple size
from class as intended) — the secondary suspects named in the v1 summary
(glyph-style distribution, real-photo blur/noise, d8/d10 geometry itself)
would move up in priority for iteration 3, not another scale-focused
attempt.

## Literature grounding

One research pass (2026-07-11), three real, arXiv-verified citations on
camera-distance/scale domain randomization for sim-to-real detection
transfer:

1. **Tobin, Fong, Ray, Schneider, Zaremba, Abbeel, "Domain Randomization
   for Transferring Deep Neural Networks from Simulation to the Real
   World"** (IROS 2017, arXiv:1703.06907). Trains object localization
   purely on simulated images with randomized textures, lighting, and
   **camera position/distance**; shows sufficient randomization of these
   scene/camera parameters (not photorealism) drives sim-to-real
   transfer. Supports camera-distance randomization as a validated,
   load-bearing domain-randomization axis, not a cosmetic addition.
2. **Tremblay, Prakash, Acuna, Brophy, Jampani, Anil, To, Cameracci,
   Boochoon, Birchfield, "Training Deep Networks with Synthetic Data:
   Bridging the Reality Gap by Domain Randomization"** (CVPR Workshops
   2018, arXiv:1804.06516). Randomizes camera viewpoint/distance,
   lighting, pose, and distractors for synthetic *detection* training
   data (cars); non-realistic randomization transfers to real photos with
   zero real training data. Directly analogous prior art for a
   detection (not just classification) task.
3. **Dwibedi, Misra, Hebert, "Cut, Paste and Learn: Surprisingly Easy
   Synthesis for Instance Detection"** (ICCV 2017, arXiv:1708.01642).
   Composites instances onto backgrounds "with no regard to preserving
   scale," arguing explicitly that letting object scale vary independent
   of natural scene statistics prevents a detector from exploiting
   spurious scale/context correlations in training data. This is the
   closest match to our specific failure mode: it is direct prior
   evidence that (a) detectors do learn incidental scale correlations
   from synthetically-composed training scenes when scale isn't
   deliberately randomized, and (b) breaking that correlation by
   randomizing scale independent of content is an effective, previously
   validated countermeasure — the same move this spec proposes for
   breaking the scale↔class correlation specifically.

All three verified as real via direct arXiv lookup (title/authors/ID
match), not from memory alone, per this repo's citation-verification
practice.

## Real framing measurement (grounds the target range empirically, not by guess)

Before picking a target apparent-size range, the frozen real test set's
own YOLO labels were measured directly (`data/real/test/labels/*.txt`,
1,376 boxes): bounding-box height as a fraction of image height (the
single-die-dominant real photos are the transfer target, so this is the
actual distribution to match, not an assumption):

| stat | value |
|---|---|
| min | 0.308 |
| 5th pct | 0.394 |
| 25th pct | 0.475 |
| median | 0.530 |
| 75th pct | 0.583 |
| 95th pct | 0.661 |
| max | 0.788 |

This is materially higher than the dispatch brief's suggested 15–60%
range — real dice consistently occupy roughly a third to four-fifths of
frame height, not as low as 15%. **Decision: sample the close-up slice's
target frame-height fraction from Uniform(0.28, 0.82)** (covers the full
observed real range with a small margin on each side) rather than the
originally-suggested 15–60%, since the goal is to match real framing, and
the measured real distribution is the ground truth for what "close-up"
means here.

## Method

**Additive slice, not a regeneration.** `data/detection_v1` (10,000
images) is untouched. A new slice is rendered to
`data/detection_v2_closeup/` (gitignored, same as `detection_v1`) using
the *same* renderer (`scripts/render_detection_dataset.py`), extended
with a `--closeup` mode rather than a new script — the existing
COCO/ID-pass/physics-settle/material pipeline is reused unchanged; only
scene composition (dice count, camera placement) differs under the flag.

### `--closeup` mode changes to `compose()`

- `n_dice = rng.randint(1, 2)` instead of `rng.randint(3, 8)` — matches
  real photos being single-die (occasionally two dice touching/adjacent
  in a few real images).
- Dice placement radius tightened (`rng.uniform(0.0, 0.02)` instead of
  `0.075`) so a 2-die scene doesn't spread a second die out of a tight
  close-up frame.
- **Camera distance is computed, not sampled directly**, to decouple
  apparent size from class:
  1. Sample `target_frac ~ Uniform(0.28, 0.82)` (see measurement above).
  2. Sample lens as before (`Uniform(35, 80)`); read Blender's own
     `cam_data.angle_y` (vertical FOV, already correctly accounting for
     sensor_fit/aspect — avoids hand-deriving and risking a mismatched
     FOV formula) after setting the lens and the scene's resolution.
  3. Measure `size = max(die.dimensions for die in dice)` — world-space
     AABB extent post-scale, read after physics settle. `dimensions` is
     rotation-invariant (local-axis bounding box × scale), which is a
     deliberate, acknowledged approximation: real per-frame apparent
     height still varies with the random settled orientation, but dice
     are roughly isotropic solids (manifest `size_mm` spans only
     14–24mm across all 7 classes, no long/thin outlier shape), so this
     approximation's error is small relative to the confound being
     removed, and does not depend on class in a way that could
     reintroduce a size↔class correlation.
  4. `dist = size / (2 * target_frac * tan(angle_y / 2))`, clamped to a
     minimum of 0.05m (clip-plane safety margin).
  5. Camera azimuth/elevation sampled from the existing ranges
     (`Uniform(0, 2π)` / `Uniform(15°, 65°)`), look-at target is the
     centroid of the placed dice (not the fixed world origin, so 2-die
     scenes stay centered).
- Distractor count and HDRI/lighting/ground-material sampling unchanged
  from `detection_v1` — only dice count and camera placement change.
- Everything else (physics settle, ID-pass occlusion masking, JPEG
  noise/exposure/DOF realism, COCO export, checkpointed sharding/resume)
  is reused byte-identical from the existing pipeline.

### Volume and shape

Target ~3,000 images. `detection_v1`'s 10,000-image, 6-shard run
measured 4,819s wall time end-to-end (≈2.08 img/s across 6 parallel
Blender workers) — at that rate 3,000 images is ≈24 minutes, far under
the 6-hour budget ceiling; no need to cut to 1,500. Same 6-way sharding
convention (`--shard k --shards 6`), seed distinct from `detection_v1`'s
(`--seed 5150` vs. `424242`) so scene sampling doesn't collide.

### Training: variant `s_v2`

Extend `src/training/detection/yolo_trainer.py` minimally, mirroring the
existing `build_s_plus_r_yaml()` multi-train-dir pattern: a
`build_s_v2_yaml()` that lists both `data/yolo/images/train` (from
`detection_v1`) and a converted-to-YOLO `data/yolo_closeup/images/all`
(from `detection_v2_closeup`, converted via the existing
`scripts/convert_coco_to_yolo.py`) as `train:` entries, keeping the same
`val:` split as variant `s` (synthetic val stays `detection_v1`-only —
this experiment's target metric is the frozen real test set, not
synthetic val). Add `"s_v2"` to `--variant` choices. Same hyperparameters
as variant `s`: `yolo11s.pt` pretrained, imgsz 640, 60 epochs, batch 32,
seed 42, deterministic.

### Evaluation

`scripts/evaluate.py --weights models/runs/s_v2/weights/best.pt --name
s_v2` — identical protocol to variants `s`/`s_plus_r` (frozen real test
set, d10_pct merged into d10, pycocotools mAP50/mAP50-95, conf 0.25,
overlays). **Never trains or validates on `data/real/test`.**

## Success criteria

- **Primary (hypothesis-supported threshold, pre-registered):** real-test
  mAP50 for **both** d8 and d10 ≥ **0.40** (up from 0.090/0.097). This is
  well below d12/d20's current 0.86–0.94, deliberately conservative — the
  claim being tested is "the confound is a major contributor," not "this
  one slice fully closes the gap to S+R's real-fine-tuned 0.99+."
- **Guard (regression check):** d12 and d20 real-test mAP50 must not drop
  by more than 0.05 from variant `s`'s baseline (0.936 → ≥0.886; 0.855 →
  ≥0.805). A guard failure means the close-up slice's added scene
  diversity cost existing good transfer and the net tradeoff needs
  re-examination even if d8/d10 improve.
- Full per-class table (`eval_s_v2.md`) reported side-by-side with
  `eval_s.md` in the verdict note appended to this spec after the run.

## Out of scope (this iteration)

- Regenerating or modifying `detection_v1` itself.
- Glyph-style rebalancing, photographic degradation, d8/d10 geometry
  review (summary's other recommendations — deferred to a later
  iteration depending on this one's verdict).
- A leak-free synthetic val split (unrelated to this hypothesis).

## Verdict (2026-07-13)

**Hypothesis SUPPORTED — both pre-registered criteria met.**

Training completed 60/60 epochs on 2026-07-12 (run `s_v2`); authoritative
eval (`scripts/evaluate.py`, frozen real test set, conf 0.25) run
2026-07-13 morning. Side-by-side, real-test mAP50 (mAP50-95 in parens),
variant `s` (detection_v1 only) vs `s_v2` (v1 + close-up slice):

| class | s (baseline) | s_v2 | Δ mAP50 |
|---|---|---|---|
| d4 | 0.695 (0.162) | 0.784 (0.174) | +0.089 |
| d6 | 0.519 (0.132) | 0.275 (0.077) | **−0.244** |
| d8 | 0.090 (0.018) | **0.442** (0.105) | +0.352 |
| d10 | 0.097 (0.034) | **0.534** (0.233) | +0.437 |
| d12 | 0.936 (0.403) | 0.946 (0.410) | +0.010 |
| d20 | 0.855 (0.264) | 0.907 (0.284) | +0.052 |

- **Primary criterion (pre-registered): PASS.** d8 0.442 ≥ 0.40 and
  d10 0.534 ≥ 0.40 (up from 0.090/0.097). The apparent-size-as-class-cue
  confound is a major contributor to the v1 d8/d10 transfer failure, as
  hypothesized.
- **Guard (d12/d20 regression): PASS.** d12 0.946 ≥ 0.886 and d20 0.907
  ≥ 0.805 — both actually improved slightly.
- Controller spot-check (seeded random, 3 overlays): single-die extreme
  close-up d10s now correctly classified d10 at 0.95/0.96/0.61 conf —
  exactly the framing v1 systematically misread as d12/d20.

**Open item the guard did not cover: d6 regressed 0.519 → 0.275.** The
guard was defined only over d12/d20, so this does not falsify anything
pre-registered, but it roughly halves the one cubic die's transfer and
is the top candidate question for iteration 3 (alongside the summary's
deferred suspects: glyph-style distribution, photographic degradation,
absolute mAP50-95 still low across the board vs `s_plus_r`'s
real-fine-tuned 0.71–0.77). ~~Hypothesis-shaped lead: scale
sensitivity of d6's cue set~~ — **analyzed 2026-07-13, lead killed**
(`vision/docs/results/2026-07-dice-detector-v1/d6_regression_analysis.md`):
d6's close-up training size distribution matches real d6 well, and the
regression is broad across framing subgroups. The actual mechanism: the
regression is almost entirely d6→d10 reassignment (6.2%→49.7%), and
d10 became a network-wide attractor class after the close-up slice made
its heavily-arabic-numeral glyph convention (~78% by domain-config
design vs ~28% for d6) legible at scale — real d6 photos are ~100%
arabic numerals, matching d10's training convention better than d6's
own. Two falsifiable iteration-3 hypotheses recorded in the analysis
doc (H1 glyph-mix rebalance; H2 d10 decision-boundary overshoot /
hard-negative pass).

Synthetic-val numbers stayed ~identical between `s` and `s_v2` (both
≥0.966 mAP50 everywhere) — reconfirming the spec's premise that
synthetic val is blind to this real-transfer failure mode and cannot be
used as the verdict metric.
