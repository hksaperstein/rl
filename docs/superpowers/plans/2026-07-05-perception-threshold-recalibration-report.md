# Perception planarity-threshold recalibration: measurement, recalibration, and verification report

Date: 2026-07-05 (session run 2026-07-06 sim-time). Branch:
`worktree-sphere-grasp-bonus`. Uncommitted — left for Principal review.

## TL;DR

`PLANARITY_RESIDUAL_THRESHOLD` was recalibrated from `0.0008` to `0.0045` (mean +
3*std of an empirical measurement), exactly as instructed. **This does not fix
ROADMAP item 2 and introduces a new regression.** The root cause is not camera/
render noise (measured negligible, ~30nm) but a genuine geometric artifact: any
object not directly under the top-down camera shows a real, non-noisy sliver of
its own side wall in the segmented cluster, which biases the SVD plane fit. This
artifact is *larger*, at the objects' real fixed scene positions, for the flat
cube (2.93mm) than the actual sphere's own curvature-driven residual (2.10mm) —
an ordering inversion that makes it **mathematically impossible** for any single
`PLANARITY_RESIDUAL_THRESHOLD` value to correctly classify both. Recommend the
Principal decide between reverting this value, accepting a still-broken
classifier for now, or scoping a structural point-cloud-level fix (see
Recommendation section). I did not make that call myself — it's a judgment call
above what "recalibrate a constant" scoped me to decide.

## Task 1: measurement

### Dead-center, motionless (as literally specified)

Per the original instructions, first measured with the cube held fixed at
`CUBE_X=0.0, CUBE_Y=0.31` (dead center under `perception_camera`, reusing
`perception_calibration.py`'s `CUBE_Y`/`CUBE_Z`), across 20 captured frames (5
warmup steps). Result: **perfectly deterministic across all 20 frames**
(std across frames = 0.0 — Isaac Sim's depth render has no frame-to-frame
noise for this static scene) with residual ≈ **2.9e-8 m** (~30 nanometers,
floating-point-precision level). This confirms the camera's depth buffer is
exactly what the background doc says: a plain, noise-free ray-traced/
rasterized render, not a physically-noisy RGB-D sensor model.

This measurement is real, but **not representative of the actual bug** for two
reasons discovered during investigation:
1. World `(0, 0.31)` happens to coincide with where the AR4 arm's resting
   end-effector/gripper sits after reset, which at some frames fully occludes
   the cube from the top-down camera (confirmed via a `height≈0.35m`,
   `pixels≈62800` cluster nearest to that point in a follow-up sweep — clearly
   the robot, not the 18mm cube).
2. A perfectly axis-aligned, dead-center view is exactly the one case where the
   real geometric effect below (oblique side-wall visibility) *cannot* occur.
   Measuring only there hides the actual failure mode.

### Realistic (production-position) measurement — used for calibration

`objects_cfg.py`'s objects are **not randomized** — they sit at fixed
world positions (`CUBE_CFG` at `(0.20, 0.28, 0.009)`, `RECT_PRISM_CFG` at
`(0.20, 0.34, 0.015)`, `SPHERE_CFG` at `(-0.20, 0.28, 0.009)`, `WEDGE_CFG` at
`(-0.20, 0.34, 0.009)`), all ~20cm off to the side of the camera (mounted at
world `(0, 0.31, 0.55)`, looking straight down). That offset is where the real
pipeline actually operates.

Re-measured the cube held motionless at its real resting pose `(0.20, 0.28,
0.009)`: again perfectly deterministic across 20 frames, residual ≈
**0.0029336 m** constant (std=0 — since a fixed pose has no source of frame-
to-frame variance in this noise-free renderer). To get a meaningful *spread*
(the "several frames" var­iance the task anticipated turned out to live in
*position*, not *time*, given the renderer is deterministic), sampled the same
cube across the `perception_calibration.py` slide sweep (`x` in
`[-0.35, 0.35]`, step ≈ every simulation step, `y=0.31`), filtering to frames
where the detected cluster's height exactly matched the cube's true 18mm (to
exclude frames where the sweep passed near the robot's resting pose and merged
with/was occluded by it):

- **n = 220 samples**, x spanning `[-0.338, 0.341]`
- mean = **0.002922 m**, std = **0.000530 m**
- min = 0.0000486 m (near FOV edge, low pixel count), max = 0.0038511 m
- **mean + 3·std = 0.004512 m**

For comparison, at their own fixed production positions (single motionless
measurement each, same methodology):

| object | pos (x,y) | residual | height | tilt | circularity |
|---|---|---|---|---|---|
| cube | (0.20, 0.28) | 0.002934 m | 0.01800 m | 23.5° | 0.786 |
| rect_prism | (0.20, 0.34) | 0.004348 m | 0.03000 m | 71.5° | 0.790 |
| sphere | (-0.20, 0.28) | 0.002097 m | 0.01792 m | 22.1° | 0.956 |
| wedge | (-0.20, 0.34) | 0.003851 m | 0.01800 m | 53.5° | 0.605 |

And sampling the sphere across the same slide sweep as a control (n=233):
mean=0.002360 m, std=0.000558 m, range [0.000675, 0.003851] m — **this range
overlaps almost entirely with the cube's distribution above.**

## Task 2: recalibration

Set `PLANARITY_RESIDUAL_THRESHOLD = 0.0045` (rounded `mean+3*std` from the
220-sample realistic measurement above) in `perception/shape_classifier.py`,
with a comment documenting the measurement, its provenance
(`scripts/measure_planarity_residual.py`), and the caveat below.

## Task 3: verification

### pytest

`python3 -m pytest perception/tests/ -v`: **24 passed, 1 failed**
(`test_classifies_curved_round_cap_as_sphere`).

Diagnosed why (per the "understand why, don't just weaken the test" mandate):
the test's synthetic sphere cap (radius 9mm, points sampled at `r <= 0.9*radius`
from an idealized noiseless hemisphere) has its own analytic SVD-fit residual
of only **0.001337 m** — computed directly via
`perception.shape_classifier._fit_plane` — which is *below* the new 0.0045
threshold, so `classify_shape` now calls it `cube` instead of `sphere`. This
is not the "noise too tight" problem the other tests had (0.2mm Gaussian jitter
vs. real render noise) — it's that a genuinely perfect small-radius sphere
cap's *intrinsic* curvature, sampled over only its near-top portion (mirroring
how the real camera's `ground_margin` crop only exposes the upper cap), is
smaller than the newly-calibrated threshold. This is the exact same structural
finding as the real-camera measurement, appearing independently in idealized
synthetic data. **Did not weaken this test** — it is correctly failing, and
its failure is diagnostic signal, not test rot.

### Real end-to-end check (`scripts/perception_classification_check.py`)

Wrote a small permanent script (kept at
`scripts/perception_classification_check.py` — useful going forward as a
one-shot regression check for any `shape_classifier.py` change) that classifies
all four objects at their real, fixed scene positions via the real
`perception_camera` + full `segment_objects`/`build_world_point_grid`/
`classify_shape` pipeline (no synthetic data).

**Before** (threshold 0.0008): `cube → sphere` (wrong), `rect_prism → sphere`
(wrong — this matches ROADMAP item 2 exactly), `sphere → sphere` (correct),
`wedge → wedge` (correct). **2/4 correct.**

**After** (threshold 0.0045): `cube → wedge` (still wrong), `rect_prism →
wedge` (still wrong), `sphere → wedge` (**newly wrong — regression**), `wedge
→ wedge` (correct). **1/4 correct.**

Raising the residual threshold doesn't fix cube/rect_prism — it just changes
their wrong label from `sphere` to `wedge`, because once an object's residual
no longer exceeds the (now higher) threshold, `classify_shape` falls through
to the tilt check — and the *same* oblique-viewing artifact that inflates
residual also inflates the fitted plane's tilt (23.5°–71.5° measured above,
all well past `TILT_THRESHOLD_RAD`'s 15°). And the real sphere, whose own
residual (0.0021 m) is now below the new threshold, falls through to that same
tilt check and gets mislabeled `wedge` too — a genuine new regression on an
object that was previously correctly classified.

## Why no threshold value can fix this (worth stating precisely)

For cube to stop reading as `sphere`, `PLANARITY_RESIDUAL_THRESHOLD` must be
`>= 0.00293` (cube's own residual at its fixed position). For the real sphere
to keep reading as `sphere`, the threshold must be `< 0.00210` (the sphere's
own residual at its fixed position). `0.00293 > 0.00210`, so these two
requirements cannot both hold — there is no single threshold value that gets
both right. Circularity doesn't rescue this either: cube/rect_prism's
footprint circularity (0.786/0.790) exceeds `CIRCULARITY_THRESHOLD` (0.7) at
~89% of sampled off-center positions (checked across the same 220-sample
sweep), so the circularity gate doesn't reliably exclude them from the sphere
branch.

## Root cause (for the record, beyond this task's scope to fix)

The real depth camera has essentially zero sensor/render noise (measured
~30nm). The actual source of "residual" is that `segment_objects` +
`build_world_point_grid` hand `classify_shape` the *entire* segmented cluster,
including pixels along the object's silhouette that — under oblique
(non-nadir) viewing from a camera not directly overhead — genuinely see a
sliver of the object's vertical side wall, not just its top face. That's real
3D geometry, not noise, and an SVD plane fit over a "flat top + partial side
wall" point set produces a non-trivial residual and tilt that happens to scale
with how far the object sits from directly under the camera — and, for these
object sizes (9-30mm) at this camera geometry (0.55m altitude, ~20cm lateral
offset), that artifact is comparable to or larger than an actual 9mm sphere's
intrinsic curvature signature.

## Recommendation

I do not recommend shipping `PLANARITY_RESIDUAL_THRESHOLD = 0.0045` as a real
fix — it satisfies the literal instruction (empirically-measured mean+3*std,
not a fabricated literature number) but measurably makes end-to-end
classification worse (1/4 vs. 2/4 objects correct) and breaks the previously-
working sphere case. Three options, in my judgment, roughly in order of
attractiveness, but this is a call for the Principal:

1. **Revert to `0.0008`** for now (keeps the status quo — cube/rect_prism
   still misclassify, but sphere/wedge still work) and treat ROADMAP item 2 as
   requiring a structural fix, not a threshold tweak. Least churn.
2. **Fix the actual geometry**: before fitting the plane in `classify_shape`
   (or upstream in `pipeline.py`/`segmentation.py`), restrict the point cloud
   to a top height-band (e.g., points within some mm of the cluster's own max
   Z) so oblique side-wall points are excluded before the SVD fit — this
   should let a genuinely flat top read near-zero residual regardless of
   camera offset, while a genuinely curved cap still shows real curvature
   within just its own top band. Untested by me — flagging as the likely real
   fix, not implementing it (out of this task's scope).
3. Keep `0.0045` if the Principal judges the cube/rect_prism-as-sphere bug
   worse than a wedge-vs-sphere regression for the current priorities — but
   I'd flag this as likely net-negative given the 2/4→1/4 real measurement.

## Script disposition

- `scripts/measure_planarity_residual.py` — kept permanently (Task 1's
  measurement tool; mirrors `perception_calibration.py`'s pattern; referenced
  from the `shape_classifier.py` comment for future recalibration).
- `scripts/perception_classification_check.py` — kept permanently (Task 3's
  real end-to-end regression check across all four objects; useful for any
  future `shape_classifier.py` change).
- Two purely exploratory scripts used to build the case above (a slide-sweep
  variant classifying the cube across the FOV, and the same for the sphere)
  were deleted after their findings were captured in this report — they added
  nothing over `perception_classification_check.py` + this document once the
  investigation was done.

## Note: unrelated pre-existing worktree state

`git status` in this worktree also shows uncommitted modifications to
`tasks/ar4/pickplace_env_cfg.py` (an experimental base-mounted LiDAR sensor
addition) and `tasks/ar4/robot_cfg.py` (a gripper PD-gain rescale experiment),
plus new files `scripts/lidar_calibration.py` and two
`docs/superpowers/specs/research/2026-07-05-grasp-scale-literature-*.md`
files. **None of these are from this task** — I never opened or edited those
files; they appear to be uncommitted work from a different,
concurrent/previous task in this same worktree. Flagging so they aren't
mistaken for part of this deliverable.
