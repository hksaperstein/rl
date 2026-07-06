# Perception side-wall-contamination fix: height-band plane-fit filter, report

Date: 2026-07-05 (session run 2026-07-06 sim-time). Branch:
`worktree-sphere-grasp-bonus`. Uncommitted — left for Principal review.

## TL;DR

Implemented the height-band point-cloud filter recommended (but not
implemented) by the prior threshold-recalibration investigation: `classify_shape`
now fits the plane (residual + tilt) only to the points within `TOP_BAND_MARGIN`
(4mm, empirically tuned) of the cluster's own max Z, instead of the full
segmented cluster, to exclude the oblique-viewing side-wall sliver before the
SVD fit. **This fixes 3 of 4 objects, robustly** (cube, rect_prism, sphere all
correctly classified across repeated runs) **but introduces a new regression on
the 4th** (wedge, previously correct, now reads as `cube`) for a real,
characterized reason: the wedge's own top face spans nearly its whole height
range, so cropping to a thin top band destroys the very tilt signal that
identifies it as a wedge. This is a genuine, understood limitation of the
single-global-margin approach, not a tuning miss — no margin value fixes all
four simultaneously (see "Why wedge can't be fixed this way" below). Net: **3/4
correct** (up from 2/4 before this change), a real improvement but not full
resolution of ROADMAP item 2.

## Task 1: implementation

### Code change

`perception/shape_classifier.py`:

1. Added `TOP_BAND_MARGIN = 0.004` (m) with a derivation comment.
2. Added `_restrict_to_top_band(points, margin)`: keeps only points with
   `z >= max_z - margin`; falls back to the full cluster if the band would
   leave fewer than 3 points (avoids an unstable/degenerate SVD).
3. `classify_shape` now computes `_fit_plane` (and therefore both `residual`
   and `tilt_rad`) on `_restrict_to_top_band(points, TOP_BAND_MARGIN)` instead
   of the raw `points`. `height` (uses `points[:, 2].max()`) and `circularity`
   (uses the full XY footprint) are **unchanged** — deliberately still computed
   on the full cluster, since the object's overall top-down silhouette (not
   just its topmost slice) is what should read as round vs. square/rectangular,
   and height just needs the cluster's true apex.

Net diff shape:

```python
 PLANARITY_RESIDUAL_THRESHOLD = 0.0008  # unchanged
 TILT_THRESHOLD_RAD = np.radians(15.0)  # unchanged
 CIRCULARITY_THRESHOLD = 0.7  # unchanged

+TOP_BAND_MARGIN = 0.004  # m, plane-fit input is restricted to this band below the cluster's own max Z

+def _restrict_to_top_band(points, margin):
+    max_z = points[:, 2].max()
+    band = points[points[:, 2] >= max_z - margin]
+    if band.shape[0] < 3:
+        return points
+    return band

 def classify_shape(points, ground_z=0.0):
     ...
     height = float(points[:, 2].max() - ground_z)
-    normal, residual = _fit_plane(points)
+    top_band_points = _restrict_to_top_band(points, TOP_BAND_MARGIN)
+    normal, residual = _fit_plane(top_band_points)
     tilt_rad = float(np.arccos(np.clip(abs(normal[2]), -1.0, 1.0)))
     circularity = _footprint_circularity(points[:, :2])  # still full cluster
```

`HEIGHT_THRESHOLD`, `PLANARITY_RESIDUAL_THRESHOLD`, `TILT_THRESHOLD_RAD`, and
`CIRCULARITY_THRESHOLD` are all unchanged, per the task's scoping instruction —
no justified reason emerged to touch them; the fix is entirely in what point
set feeds the plane fit.

### Margin value reasoning (4mm, not the initially-estimated 3mm)

Started from a geometric estimate: at the objects' real fixed scene positions
(`tasks/ar4/objects_cfg.py`), the camera `(0, 0.31, 0.55)` views them from
roughly `atan(0.20 / 0.52) ≈ 21°` off nadir, so the visible side-wall sliver's
vertical depth below the top edge is approximately the object's own top-face
half-width times `tan(21°)` — for these 16-18mm-wide objects (half-width
8-9mm), that's `≈ 3-3.5mm`.

Rather than ship that estimate directly, empirically swept every real object's
top-band residual/tilt (via a one-off script reusing
`perception/pipeline.py`/`perception/segmentation.py`/`perception_camera`, same
pattern as `scripts/measure_planarity_residual.py`; not kept — its findings are
captured here) across margins from 0.5mm to 10mm, and to full-cluster (no
restriction). Key numbers (residual in meters, tilt in degrees):

| margin | cube resid/tilt | rect_prism resid/tilt | sphere resid/tilt | wedge resid/tilt |
|---|---|---|---|---|
| 1.5mm | 0.000328 / 1.62° | 0.000250 / 1.43° | 0.000408 / 1.50° | 0.000159 / 0.60° |
| 2.5mm | 0.000328 / 1.62° | 0.000250 / 1.43° | 0.000727 / 0.66° | 0.000413 / 1.34° |
| **3mm** | 0.000328 / 1.62° | 0.000250 / 1.43° | **0.000816** / 2.84° | 0.000548 / 2.29° |
| **4mm** | 0.000328 / 1.62° | 0.000250 / 1.43° | **0.001077** / 6.05° | 0.000767 / 3.54° |
| 5mm | 0.000328 / 1.62° | 0.001087 / 6.51° | 0.001385 / 5.86° | 0.001054 / 4.43° |
| 6mm | 0.001189 / 6.44° | 0.001087 / 6.51° | 0.001471 / 9.64° | 0.001439 / 5.47° |
| 10mm | 0.002105 / 13.87° | 0.001963 / 14.44° | 0.001979 / 22.39° | 0.002523 / 16.76° |
| full cluster | 0.002934 / 23.54° | 0.004348 / 71.51° | 0.002003 / 23.51° | 0.003851 / 53.49° |

(`PLANARITY_RESIDUAL_THRESHOLD = 0.0008`, `TILT_THRESHOLD_RAD = 15°`.)

Observations that drove the final value:
- Cube's and rect_prism's top-band residual **plateaus** at their low, flat
  values (0.000328m / 0.000250m) across margins 1.5mm-4mm — there's a
  discrete gap in their point clouds' Z distribution in that range (a pixel/
  ray-density quantization artifact), so no new side-wall points enter the
  band until ~5-6mm. This gives real headroom, not a knife-edge.
- The real sphere's top-band residual rises **smoothly** with margin. At 3mm
  it clears the 0.0008 threshold by only ~2% (0.000816 vs 0.0008) — too close
  to trust; **confirmed non-robust empirically** (see below). At 4mm it clears
  by ~35% (0.001077 vs 0.0008), comfortably outside the noise floor of
  reset-to-reset variation.
- 4mm is the **largest** margin still fully inside the cube/rect_prism safe
  zone (rect_prism's residual jumps above threshold at 5mm), so it's the
  value that maximizes the sphere's separation margin without reintroducing
  the cube/rect_prism regression.

Confirmed 3mm's fragility directly: three separate `perception_classification_check.py`
runs at `TOP_BAND_MARGIN = 0.003` gave sphere residuals of 0.0007877, 0.0008162,
and 0.0008591 — straddling the 0.0008 threshold and flipping the sphere's label
between `cube` and `sphere` run to run (Isaac Sim logs "Seed not set for the
environment... may not be deterministic", consistent with tiny frame-to-frame
settling variation shifting a few boundary pixels in/out of the cluster). At
4mm, three repeated runs all gave sphere residuals in [0.0010858, 0.0012128] —
consistently well clear of the threshold, all classified `sphere`.

## Task 2: verification

### pytest

`python3 -m pytest perception/tests/ -v`: **25 passed, 0 failed.** All
pre-existing tests (idealized synthetic point clouds, not off-center) pass
unaffected, including `test_classifies_curved_round_cap_as_sphere` — its exact,
noiseless synthetic sphere cap retains enough of its own curvature within a 4mm
top band that its residual still clears the threshold.

### Real end-to-end check (`scripts/perception_classification_check.py`)

**Before this task** (per the prior report, threshold reverted to 0.0008, no
top-band filter): `cube → sphere` (wrong), `rect_prism → sphere` (wrong,
ROADMAP item 2), `sphere → sphere` (correct), `wedge → wedge` (correct).
**2/4 correct.**

**After this task** (`TOP_BAND_MARGIN = 0.004`), run 3 times to check
robustness:

| object | run 1 | run 2 | run 3 |
|---|---|---|---|
| cube | `cube` (correct) | `cube` | `cube` |
| rect_prism | `rectangular_prism` (correct) | `rectangular_prism` | `rectangular_prism` |
| sphere | `sphere` (correct) | `sphere` | `sphere` |
| wedge | `cube` (**wrong** — regression) | `cube` | `cube` |

**3/4 correct, robustly (all 3 runs identical).** Cube and rect_prism are
fixed (the two objects ROADMAP item 2 names explicitly). Sphere stays correct.
Wedge regresses from correct to `cube`.

Full per-object numbers (one representative run):

```
cube         height=0.01800 residual=0.0003278 tilt_deg=1.62  circ=0.786  -> cube
rect_prism   height=0.03000 residual=0.0002502 tilt_deg=1.43  circ=0.790  -> rectangular_prism
sphere       height=0.01787 residual=0.0012128 tilt_deg=3.47  circ=0.966  -> sphere
wedge        height=0.01800 residual=0.0007670 tilt_deg=3.54  circ=0.605  -> cube (was wedge)
```

## Task 3: why wedge isn't fixed, and why no margin fixes all four

This is a real, characterized limitation of the single-global-margin
height-band approach — not a case of picking the wrong number.

The wedge is misclassified because its **tilt signal collapses** once cropped
to a thin top band: full-cluster tilt is ~53° (correctly `> TILT_THRESHOLD_RAD`,
i.e. correctly reads as wedge), but within a 4mm top band it drops to ~3.5°
(nowhere near the 15° threshold), so `classify_shape` falls through to the
height check and labels it `cube`. The reason is structural, not a tuning
artifact: cube/rect_prism's contamination is a *thin sliver* of side wall
around the edge of an otherwise genuinely flat top, so cropping a thin band
removes almost only that sliver. The wedge's situation is different — its real
top surface is a **single tilted plane spanning nearly its whole height range**
(the whole point of a wedge shape), so a thin top-of-cluster band captures only
a small, spatially narrow patch near the tallest ridge, which is *not*
representative of the full face's orientation. The height-band crop, applied
uniformly, can't tell "thin contamination sliver on top of a flat face" apart
from "genuinely large tilted face" — it just always keeps the top slice, which
helps the former case and actively destroys the signal in the latter.

Checked whether any margin avoids this trade-off: from the sweep table above,
wedge's tilt only climbs past `TILT_THRESHOLD_RAD` (15°) at margin ≈10mm
(measured 16.76°) — but at that same margin, cube's residual is already
0.0021 and rect_prism's is 0.00196, both roughly 2.5x over
`PLANARITY_RESIDUAL_THRESHOLD`, reintroducing the original cube/rect_prism-as-
sphere bug. The "cube/rect_prism stay correct" window (margin ≲4mm) and the
"wedge tilt recovers" window (margin ≳10mm) do not overlap at all — this is
the same kind of structural impossibility the prior threshold-only
investigation found, just relocated from `PLANARITY_RESIDUAL_THRESHOLD` to
`TOP_BAND_MARGIN`.

Also checked whether computing tilt from the *full* cluster (while keeping
residual on the top band) could rescue wedge without breaking cube/rect_prism:
it can't — the same oblique-viewing artifact that inflates residual also
inflates tilt for *every* off-center object, correctly-shaped or not. Full-
cluster tilt is ~23.5° for cube, ~71.5° for rect_prism, and ~23.5° for the
*correctly round* sphere — all already past `TILT_THRESHOLD_RAD` regardless of
shape. Using full-cluster tilt would misclassify cube, rect_prism, *and*
sphere as wedge; it isn't a viable partial fix, it's strictly worse.

**Real fix, out of this task's scope**: the wedge likely needs a genuinely
different check — e.g., a robust/RANSAC-style dominant-plane fit that
identifies and discounts a minority of outlier points (the side-wall sliver)
regardless of their Z-height, rather than a height-based crop, since that
would treat "flat top + thin outlier sliver" (cube/rect_prism) and "one large
genuinely tilted face" (wedge) differently based on how well each fits a
single dominant plane, not on Z-position alone. Flagging for the Principal to
decide whether this is worth pursuing versus accepting a wedge/cube confusion
(is the wedge's own use in the pick-and-place task even shape-sensitive downstream, or is
this misclassification tolerable?) as a known limitation.

## Script disposition

- `scripts/perception_classification_check.py`,
  `scripts/measure_planarity_residual.py` — untouched, reused as-is (already
  existed from the prior task).
- A one-off margin-sweep script (recomputed `_fit_plane` at 12 candidate
  margins against a single real-camera capture of all four objects) was
  written to build the case above and then **deleted** after its findings were
  captured in this report and in `shape_classifier.py`'s derivation comment —
  same disposition convention as the prior report's exploratory scripts.

## Note: unrelated pre-existing worktree state

`git status` also shows uncommitted changes to `tasks/ar4/pickplace_env_cfg.py`
and `tasks/ar4/robot_cfg.py`, plus `scripts/lidar_calibration.py` and two
`docs/superpowers/specs/research/2026-07-05-grasp-scale-literature-*.md`
files — from other concurrent tasks in this same worktree. None of these were
touched by this task.
