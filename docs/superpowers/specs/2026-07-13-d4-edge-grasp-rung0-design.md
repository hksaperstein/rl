# d4 edge-grasp, rung 0: opposite-edge antipodal axis for the scripted pick

**Date:** 2026-07-13. **Branch:** `franka-panda-pivot`.
**Prior result:** dice-pick demo (`kb/wiki/experiments/dice-pick-demo.md`) —
4/5 die types pick successfully with the scripted DiffIK straight-down
pinch; the d4 is the sole failure: sub-mm centroid convergence, then the
flat parallel jaws close on the triangular cross-section and eject the die
16.3mm sideways, reproduced across white and colored runs.
**Research grounding:** `.superpowers/sdd/research-d4-grasp.md` (six-axis
survey + four-rung ladder + independent citation review, 2026-07-12; no
fabricated citations; two should-address findings from that review are
incorporated here — see "Extrapolation ownership" below).

## Hypothesis (falsifiable)

> Aligning the scripted pick controller's gripper-closing axis with one of
> the d4's three opposite-edge pairs (computed from the die's resting
> orientation), instead of the current straight-down pinch on the
> triangular cross-section, will yield pick success (same z-gain
> verification as the other four dice) in ≥4/5 seeded trials with lateral
> ejection ≤5mm at closure — because clamping along the common
> perpendicular of two opposite edges places each jaw plane in full
> line contact with an edge (the two edges of a pair each lie in planes
> perpendicular to their common perpendicular), eliminating the
> angled-surface twist mechanism that Wang et al. (CASE 2019) identify
> for pinches on sloped faces and that Montana's convex-vertex
> non-robustness result (cited in Smith et al., ICRA 1999) predicts for
> the current near-vertex contact.

**Falsification condition:** if ≥2/5 seeded trials still eject the die
>5mm at closure despite a converged, correctly-tilted edge-pair approach
(convergence and axis alignment verified from logged EE pose, not
assumed), the edge-grasp geometry is not sufficient with these rigid flat
pads, and the ladder climbs to rung 1 (fingertip pad modification, Guo et
al. ICRA 2017: 93.7% vs 28.7% under comparable position error) per the
pre-registered climb rule in the research doc.

## Extrapolation ownership (review findings 1 & 2, addressed)

Smith et al. (ICRA 1999) is a strictly 2D polygonal-cross-section theory;
its own Discussion names 3D extension as unsolved future work. **The
line-contact construction in this spec's hypothesis is this project's own
3D extrapolation, not a result of that paper.** What the paper does
supply, and what we use it for: the friction-cone condition
(|φ| ≤ arctan μ) and ε-margin vocabulary for the desk check below, and
(via its citation of Montana) the explanation for why the current
near-vertex pinch is fragile. Likewise, Wang et al.'s headline
100%-predicted/13%-actual number belongs to their **15°-slope**
adversarial object (their 10° object scored 80%) — the number is used
here only as evidence that angled-face pinches twist objects out of
grippers, which holds at either slope. The empirical seeded-trial gate,
not the borrowed theory, carries the verdict.

## Desk check (plan Task 0 — must pass before any sim run)

Geometry of a regular tetrahedron, edge a = 30.3mm, resting on a face:

- Each of the 3 bottom edges pairs with the opposite lateral edge; the
  closing axis is their common perpendicular, running midpoint to
  midpoint. Span between midpoints: a/√2 ≈ **21.4mm** — comfortably
  inside the Franka gripper's 80mm max opening.
- The axis is tilted **arctan(1/√2) ≈ 35.26° from horizontal**, so the
  wrist must tilt 35.26° from the demo's canonical straight-down
  orientation, rotating about the direction of the bottom edge of the
  chosen pair. Three pair choices per resting face; pick the one whose
  required wrist yaw is closest to the current joint configuration.
- Nominal contact is line contact on both edges (φ ≈ 0), so the friction
  condition is satisfied with margin; with the demo's ~3.4° orientation
  tolerance plus die-pose error, φ stays far inside arctan(μ) for any
  plausible pad μ ≥ 0.2. **Task 0 must read the actual contact-material μ
  from the asset/physics material, not assume it.**
- Positional window along the edge direction: contact must stay on the
  edge, i.e. |offset| ≤ a/2 − w/2 where w = jaw-pad width (measure from
  the Franka hand asset at plan time; ~±6mm for an ~18mm pad). Rung 0
  runs on ground-truth pose (below), so the error budget is IK-only
  (measured 1–5mm) — inside the window, with the detector's 2–8mm error
  deliberately excluded from this rung.
- **Table-clearance constraint (new, identified here):** the bottom edge
  of every pair lies ON the table, so the lower jaw cannot contact it
  exactly — the real contact lands slightly above, on the adjacent faces.
  Task 0 must compute the resulting φ at a chosen contact-height offset
  (a few mm along the closing axis) and confirm it stays inside the
  friction cone; if it cannot, that is a desk-stage falsification and the
  ladder climbs to rung 1 without burning sim runs.

### Desk-check corrections (2026-07-13, post-Task-0 measurement)

Task 0 measured the real assets (full arithmetic:
`.superpowers/sdd/task-d4-rung0-tasks01-report.md`); three numbers above
change, none alter the hypothesis or criteria:

- **Edge length a = 23.591mm measured** from the actual d4 mesh (the
  30.3mm assumed above was the d20's size, carried over in error). Span
  between edge midpoints: 16.68mm. Tilt 35.26° is edge-length-independent.
- **Positional window is ±3.05mm** ((a − w)/2 with measured pad width
  w = 17.5mm), not ±6mm — the upper end of the 1–5mm IK error budget
  exceeds it. This narrows the expected pass margin; the pre-registered
  ≥4/5 gate and climb rule stand unchanged, and per-trial logging must
  record the closure-time positional residual against ±3.05mm explicitly.
- **μ = 0.5 verified** (neither the Franka hand nor the die USD authors a
  physics material; Isaac Lab's `RigidBodyMaterialCfg()` default 0.5/0.5
  applies), so arctan μ = 26.6°. The table-clearance analysis passes for
  small contact-height offsets (φ ≈ 0 while the pad still spans the
  bottom edge) but φ jumps discontinuously to 54.7° (half the dihedral
  angle's complement geometry) if contact falls entirely onto the wedge
  faces — Task 2 must verify actual contact via instrumentation, not
  assume the small-δ regime held.

## Design

- **Scope:** `scripts/dice_pick_demo.py` gate G only, d4-only branch. The
  code path for d8/d10/d12/d20 must remain byte-identical (regression
  guard below).
- **Pose source — pre-registered scope decision:** rung 0 reads the d4's
  resting orientation from **sim ground truth** (the demo already reads GT
  for verification). This isolates the grasp-mechanism variable; the
  detector currently provides centroid only, and perception-side yaw/pose
  estimation is a separate, explicitly-deferred axis (research doc
  confound 1). A rung-0 pass therefore claims "the grasp mechanism works,"
  not "the vision-driven demo picks the d4" — the follow-up to close that
  gap is recorded in ROADMAP, not smuggled into this rung.
- **Axis computation:** from the GT quaternion, classify which face is
  down (handle all 4 resting faces — research doc confound 3; reject and
  log if the die is not settled flat), enumerate the 3 opposite-edge
  pairs, compute each pair's common-perpendicular axis in world frame,
  pick the reachability-best pair, derive the tilted grasp orientation
  quat and the two staged waypoints (approach along the axis normal,
  then close). Write the geometry as a reusable
  "antipodal edge-pair axis from mesh + pose" helper, not d4-hardcoded
  constants — the computation itself is shape-general machinery
  (North Star), even though only the d4 exercises it in this rung.
- **Verification instrumentation (standing practice):** log EE pose at
  closure, die displacement during the closure window (the lateral-
  ejection metric), and z-gain over the lift, exactly as the demo's GT
  check does today; video per the full-arm framing rule.

## Success criteria (pre-registered)

- **Primary:** ≥4/5 seeded trials (seeds 42, 123, 7, 1000, 2026; same
  layout-sampling machinery as the demo) end in pick success — die in
  gripper with z-gain ≥ 200mm sustained (the other dice measured
  237–241mm) — with lateral ejection at closure ≤5mm.
- **Regression guard:** non-d4 code path byte-identical by diff; one d20
  smoke pick re-run to confirm no behavioral drift.
- **Climb rule (from the research doc, unchanged):** on failure of the
  primary criterion, or if the per-resting-pose axis computation proves
  brittle, climb to rung 1 (pad modification) — do not iterate rung 0
  parameters past one bounded debugging pass.

## North Star call (explicit, not defaulted)

The research doc flagged the tension: rungs 0–1 are cheap but per-shape;
rung 2 (suction) is shape-general but CPU-only in current Isaac Lab.
Decision: **run rung 0 first as the cheapest empirical probe**, with two
deliberate generality hedges — (1) the axis computation is written as
general antipodal-pair machinery, reusable for any convex polyhedron;
(2) the rung-2 suction option is kept alive as the recorded next
structural alternative if rungs 0–1 both fail, with its CPU-only cost to
be scoped only if we actually reach it. Choosing cheapest-first here is
an evidence-ordering decision, not a commitment to per-shape fixes as
the platform's grasp strategy.

## Out of scope (this rung)

- Perception-side pose/yaw estimation (rung 0 uses GT orientation).
- Fingertip/pad geometry changes (rung 1), suction (rung 2), pre-grasp
  toppling (rung 3).
- Any RL training; this is scripted-controller work only.
- The other four dice beyond the regression smoke.

## Verdict

**FALSIFIED at the implementation layer - 0/5 seeded trials, but the
grasp-mechanism hypothesis itself was never actually exercised.**
Full per-trial data: `.superpowers/sdd/task-d4-rung0-trials-report.md`.

All 5 seeded trials (42, 123, 7, 1000, 2026) failed identically: the d4
branch's `stage2_descend_d4` waypoint (descent to the tilted grasp height
along the computed edge-pair axis) never converged within its 400-step
budget (final position residual 26.6-40.1mm against a 5mm tolerance -
and against the desk check's corrected +-3.05mm positional window, worse
still). The gripper never closed on any trial. Consequently:

- **Zero grasp attempts across all 5 trials** - `waypoint_status` in
  every trial's verdict JSON is `{"error": "...stage2_descend_d4 did NOT
  converge..."}`, populated before the code path that would ever measure
  closure-window lateral ejection, read the new contact-force sensors
  (`tasks/franka/dice_scene_cfg.py`'s `d4_leftfinger_contact`/
  `d4_rightfinger_contact`, added this task specifically to verify the
  desk check's flagged phi-regime risk), or attempt a lift.
- **The die was never touched or perturbed** in any trial (verified
  directly, not assumed): `z_now == z_before` and `xy_drift ~= 0` (all
  sub-micron, i.e. simulation noise floor) for the d4 in all 5 verdict
  tables.
- **This does not test the hypothesis.** The spec's own falsification
  condition requires "a converged, correctly-tilted edge-pair approach
  (convergence and axis alignment verified from logged EE pose, not
  assumed)" before an ejection outcome counts as evidence against the
  line-contact mechanism. No trial reached that state, so rung 0's
  actual grasp-mechanism question (does opposite-edge-pair contact
  survive closure without ejecting the die) remains **untested**, not
  falsified.
- **Diagnostic pattern is consistent with an implementation/reachability
  gap, not per-trial grasp instability**: all 5 trials fail via the
  identical mechanism (stage2 non-convergence) regardless of which of
  the 3 edge-pairs was selected (pair_id 0 picked 3x, pair_id 2 picked
  2x, spanning wrist_yaw -20.9deg to +21.2deg) or seed. Stage 1
  (approach, looser 15mm tolerance) "converges" in every trial but with
  a suspiciously consistent large residual concentrated in one axis
  (dz = -12.6mm to -13.4mm in all 5 trials, same sign and magnitude
  regardless of pair/yaw) - a systematic bias, not seed noise. Stage 2
  then fails outright, and in 3/5 trials orientation error also grows
  substantially during the failed stage-2 attempt (rot_err reaching
  0.33-0.64 rad, i.e. 19-36deg, despite converging to 0.0007rad after
  stage 1) - the bounded relative-step IK controller (`_step_toward`)
  appears to fight itself on the combined tilted-orientation +
  off-axis-position target rather than smoothly converge. The Task 1
  report's own item #6 flagged this exact gap in advance: the d4 path
  has no XY/Z refine fallback for a stalled stage2, unlike the non-d4
  path's proven fallback for its own (much smaller, ~14mm) oscillation
  floor.
- **Per the climb rule's own carve-out** ("do NOT iterate rung-0
  parameters beyond one bounded debugging pass, and only if a trial
  failure is clearly implementation... rather than mechanism"), this
  failure signature qualifies as clearly implementation, not mechanism -
  100% reproducible non-convergence with a consistent directional bias,
  not stochastic per-seed ejection variance. Task 2/3's implementer did
  NOT spend the one authorized bounded debugging pass under this
  dispatch (out of scope for a trials+verdict task, and root-causing the
  IK oscillation properly needs dedicated investigation rather than a
  guess burned against the one-pass budget) - flagged back to the
  controller as the recommended next step (a scoped Task 4) rather than
  climbing straight to rung 1: the pad-geometry mechanism change rung 1
  represents would not even address this failure mode, since it never
  reaches the point where pad geometry matters.
- **Regression guard**: non-d4 code path confirmed byte-identical
  (`git diff -w` shows zero changed lines inside the pre-existing
  `else:` branch). The d20 smoke (seed 42, run twice) both came back
  FAIL (0.0mm z-gain, 38.5mm lateral drift, byte-identical between the
  two reruns) - traced directly to the perception subprocess reporting
  an 8.4mm detector-vs-GT xy error for this exact seed's rendered scene,
  which exceeds the ~8mm lateral squeeze-out margin already documented
  as a pre-existing, seed-dependent d20 fragility in
  `kb/wiki/experiments/dice-pick-demo.md` ("occasionally exceeds the
  ~8mm lateral squeeze-out margin even for the d20... 1 FAIL in 3 d20
  runs") predating this task. The detector subprocess is causally
  isolated from every change in this task's diff (independent process,
  reads only the saved camera frame; dice settle positions and the
  non-d4 control path are both confirmed unchanged) - this is not
  attributed to the d4 rung-0 work.

**Disposition**: rung 0's edge-grasp geometry/orientation math is
implemented and unit-tested correct (Task 1), but the scripted pick
controller cannot currently reach the tilted grasp waypoint reliably
enough to test it in sim. Recommend a scoped, bounded follow-up (root-
cause the stage2 IK non-convergence - likely candidates: the standoff/
waypoint computation's consistent dz bias, or an XY+Z refine fallback
analogous to the non-d4 path's) before treating rung 0 as falsified and
climbing to rung 1 per the research doc's ladder. This is a controller/
Principal-level scoping decision, not unilaterally executed here.
