# Shape classifier perception debugging

## Overview

A production shape classifier misclassified cube/rectangular-prism as "sphere" against real depth-camera data. The bug report triggered a multi-phase investigation: literature pass on sensing modality and feature-robustness (catching fabricated citations in the process), empirical root-cause diagnosis, a structural fix that resolved the originally-reported bug but exposed a new wedge-shape regression, and a LiDAR side-investigation that confirmed RGB-D is the only viable perception modality in this stack.

**Current status:** 3/4 shapes correct (cube, rect_prism, sphere); wedge regression identified and documented; RGB-D modality confirmed structurally sound; LiDAR ruled out.

---

## Root cause: not sensor noise, real geometry

Initial hypothesis: `PLANARITY_RESIDUAL_THRESHOLD` (tuned on near-noiseless synthetic data at 0.0008m) doesn't generalize to real sensor noise.

**Empirical diagnosis** (`scripts/measure_planarity_residual.py`): measured the real camera's planarity-residual distribution at the objects' actual off-center positions (~20cm to the side of the top-down camera). Finding: the render is noise-free (~30nm), but at off-center scene positions, the segmented cluster for cube and rectangular-prism genuinely includes a sliver of the object's oblique-visible side wall — real 3D geometry, not sensor noise. This inverts the residual ordering at working positions: cube (0.00293m) > sphere (0.00210m), meaning no single threshold can classify both correctly.

Recalibration attempt (to empirically-measured mean + 3σ = 0.0045m) was tried and reverted: it measurably worsened end-to-end classification (2/4 → 1/4 correct; cube and rect_prism stayed wrong, sphere broke). Root cause: the oblique side-wall sliver is a structural property of the camera's viewing angle, not a parameter-tuning miss.

---

## Literature research and verification

Delegated to dedicated research specs with post-publication senior citation review (catching a fabricated verbatim quote and a fabricated "3-6mm RMS depth noise at 0.5m" figure — see [[citation-verification-practice]]). **What survived verification:**

### Modality: RGB-D remains correct choice
- LiDAR's angular resolution creates mm-scale uncertainty on 9-30mm objects at 0.5m (correctly-labeled trig: ~8.7mm per 1° beam divergence, not dressed as a citation). No modality change needed.
- Existing top-down RGB-D camera is the right sensor.

### Empirical threshold recalibration (Phase 1, falsified)
Standard next step: empirically measure the real planarity-residual distribution on a flat reference surface at the actual 0.55m working distance via `scripts/perception_calibration.py` (can extend this tool; perception camera already has a realistic noise model in sim), and set threshold to mean + 3σ from that measurement. **This did not work** (see Root cause section above) — the residual ordering inversion is structural, not noise-driven.

### Phase 2 fallback: spin-image local descriptor
Spin image + lightweight classifier — verbatim-verified 85–91% accuracy on ModelNet10 (DOI:10.3390/s24237749/Sensors 2024) — outperforms FPFH in that paper (junior initially misapplied the paper's result to FPFH; the paper explicitly rejected FPFH in favor of spin image). Not yet implemented; available if top-band structural fix insufficient.

### Phase 3 fallback: learned classifier
PointNet-style end-to-end learned classifier — verbatim-verified noise/corruption robust (3.8% accuracy drop at 50% missing points, >80% accuracy at 20% outlier points; arXiv:1612.00593). Needs training data but robust to noise by construction rather than via hand-tuned thresholds. Backlog if Phase 2 insufficient.

### Multi-view fusion (separate enhancement, not part of this fix)
MVTN: 13% accuracy gain at 50% occlusion (arXiv:2011.13244). Requires new camera angle — bigger architecture change than this bug fix scope. Noted for future; not pursued here.

---

## Structural fix: top-band restriction (3/4 correct, wedge regression)

**Implemented:** `_restrict_to_top_band()` / `TOP_BAND_MARGIN` (4mm, geometrically derived then empirically swept 0.5–10mm against real camera) in `perception/shape_classifier.py`. Plane-fit residual and tilt now use only points within the margin of the cluster's own top, excluding the oblique side-wall sliver at the source rather than threshold-filtering its effect.

**Verification** (3 repeated real end-to-end runs via `scripts/perception_classification_check.py`): **cube → cube, rect_prism → rectangular_prism, sphere → sphere — all correct.** The originally-reported bug is fixed. All 25 `perception/tests/` unit tests still pass.

**New regression:** wedge → cube. Root cause is structural, not tuning: the wedge's tilted face spans nearly its full height, so a 4mm top-band crop thin enough to exclude cube/rect_prism side-wall sliver destroys the wedge's own tilt signal (measured tilt: 53°→3.5° within the band). Tilt only recovers past ~10mm margin, which reintroduces the cube/rect_prism regression — no single margin fixes all four shapes simultaneously.

**Recommended follow-up** (not yet done): give the wedge's tilt check a RANSAC-style robust plane fit over the full cluster (robust to the side-wall sliver as an outlier population) instead of relying on the same top-band-restricted fit for residual/circularity. Decided to ship this net improvement (3/4 > 2/4, fixes the specific originally-reported bug) rather than block on a fully general fix.

Full margin-sweep data in `docs/superpowers/plans/2026-07-05-perception-sidewall-fix-report.md`.

---

## LiDAR side-investigation: tried, reverted, architecturally blind

Per direct user request ("create a lidar with higher resolution"), investigated whether higher resolution could compensate for perceived sensor limitations. Two independent, code-verified structural blockers:

1. **Static-mesh-only limitation:** `RayCaster._initialize_warp_meshes` bakes the target mesh's world-space vertices into an immutable `wp.Mesh` once at sim initialization (gated by `_is_initialized` flag only cleared on full timeline stop, never per-env reset). Empirically confirmed: teleporting a mesh-typed object left RayCaster's `ray_hits_w` centroid completely unchanged. (Note: cube/rect_prism/sphere are procedural USD primitives, not meshes, see point 2.)

2. **Incompatible object representation:** This repo's graspable objects (cube/rect_prism/sphere in `objects_cfg.py`) are analytic USD primitives, not `UsdGeom.Mesh` prims. Pointing `mesh_prim_paths` at any of them raises `RuntimeError: Invalid mesh prim path` immediately, reproducible across runs. Independent of blocker 1.

**Conclusion:** No resolution setting or target-selection change can make this Isaac Lab version's `RayCaster` see a dynamic graspable object. RGB-D remains the only viable perception modality in this stack.

Negative result documented in `docs/superpowers/plans/2026-07-05-ar4-base-lidar-report.md` for future reference (if this project ever returns to LiDAR investigation or needs to prove the architectural limitation isn't Isaac-Lab-specific).

---

## Related concepts

[[citation-verification-practice]] — this saga's literature pass caught a fabricated verbatim quote and a fabricated noise figure, directly illustrating the citation-hygiene challenge the practice article addresses.

---

## Sources

- `docs/superpowers/specs/research/2026-07-05-perception-sensing-literature-junior.md` — initial literature pass (raw research)
- `docs/superpowers/specs/research/2026-07-05-perception-sensing-literature-senior-review.md` — citation verification and errors caught
- `docs/superpowers/plans/2026-07-05-perception-threshold-recalibration-report.md` — empirical root-cause diagnosis
- `docs/superpowers/plans/2026-07-05-perception-sidewall-fix-report.md` — structural fix details and margin-sweep data
- `docs/superpowers/plans/2026-07-05-ar4-base-lidar-report.md` — LiDAR investigation and architectural blockers
