# Dice-pick demo (2026-07-11) — first perception-in-the-loop pick

**Result: the convergence milestone (dice + Franka + detection) is met.**
Commanded die type ∈ {d4, d8, d10, d12, d20} → the trained [[vision-platform]]
detector identifies and 3D-localizes the correct die on a five-die table
(depth deprojection, sim ground truth verification-only) → a scripted
staged DiffIK sequence picks it up. 4/5 die types pass on seed 42
(z-gains 237-241mm, video + GT verified); d4 is the sole, pre-declared
permitted failure (flat-pad closure squeezes the tetrahedron out despite
sub-mm convergence — open follow-up). Videos:
`outputs/dice_demo/gate_v/dice_pick_<die>.mp4`. Branch
`franka-panda-pivot`; gates A/P/G/V history in
`.superpowers/sdd/dice-demo-report.md`.

This is a **scripted controller**, not RL — it proves the perception
bridge and the platform (assets, camera, IK, gripper), not learned
grasping. Phase I (detector-derived state inside a trained policy)
remains open.

## Transferable findings

- **mm-as-m assets:** `vision/`'s dice USDs are authored in
  millimeters-as-units; the detector's own Blender renders apply
  `MM_TO_M` at import. A uniform `scale=(0.001,)*3` on `UsdFileCfg`
  reproduces exactly what the detector was trained on (per-die factors
  would distort the size distribution — a class cue per the
  dice-detector-v1 confound).
- **Schema-less USDs get NO physics from `RigidObjectCfg`:**
  `rigid_props`/`collision_props`/`mass_props` only *modify* existing
  schemas — on a visual-only USD they silently no-op. Runtime fix:
  `.Apply()` RigidBodyAPI/CollisionAPI/MeshCollisionAPI(convexHull) then
  call `modify_*_properties` to apply the tuned values (see
  `scripts/dice_pick_demo.py::apply_convex_hull_collision`).
- **Camera sensors need explicit lighting + convergence:** a
  DomeLight-only scene renders near-black to a `CameraCfg` sensor;
  add a DistantLight and render extra RTX frames before reading
  `camera.data.output`. Also: `scene.reset()` after `sim.reset()` or
  the camera's `pos_w`/`quat_w_ros` stay zero/NaN. Related:
  [[sim-physics-fidelity]] (the repo's camera-convention bug history).
- **Scripted DiffIK descent from the Franka ready pose:** holding the
  *default ready-pose orientation* rigidly funnels the arm into
  joint-limit branches (target-independent attractors); use the
  canonical straight-down quat (0,1,0,0 wxyz) + a joint-space
  "ready-to-descend" prep stage + bounded per-step relative commands.
  DLS damping tuned for stability throttles convergence speed —
  bounded relative stepping makes low damping safe again.
- **Tolerance must scale with object size:** a 15mm grasp-position
  tolerance passes 30mm dice and loses 15-18mm dice (residual >
  object radius → fingers close beside it). Tightened to ~5mm at
  grasp height.
- **Detector on a known scene deserves a contract:** a geometric
  plausibility filter (deprojected position must lie in the physical
  band above the table) rejected a table-hole false positive that
  deprojected *below* the table; a one-per-class recovery sweep with
  tight per-candidate recrops recovered a washed-out die. Region-crop
  upscaling alone does nothing — ultralytics letterboxes every crop to
  the same network input size; only the object's *fraction of the
  crop* matters.
- **Ops:** Isaac Kit reliably hangs in teardown after the script's
  `[DONE]` (kill the Kit PID, then the orphaned `Omniverse Hub` holding
  the flock fd via `fuser`); startup can hang silently with zero log
  output (kill + relaunch once if the log hasn't grown in 10 min);
  never let a watcher's `pgrep -f` pattern appear verbatim in its own
  command line (split the string) — two separate self-match incidents
  this session, one of which killed healthy runs.

## Colored-dice repeat (2026-07-12)

Question: do the dice render in their authored colors, and does the
detector/pick pipeline survive them? (`--colored-dice` +
`--light-scale` flags, commit fdc7164; full report
`.superpowers/sdd/dice-demo-colored-report.md`.)

- Material authorship was never the problem: every set_00000 die USD
  already carries a correctly authored + bound UsdPreviewSurface whose
  diffuseColor exactly matches its manifest HSV (all five dice share one
  blue-violet, HSV (0.677, 0.406, 0.416)).
- Only `--colored-dice` runs blow out the render — the white baseline
  never did, and the blowout covers the whole frame including the table
  the flag never touches. Working theory: RTX auto-exposure reacting to
  the runtime material rebind, NOT the linear doubled-light-energy story
  originally written into the WIP docstring (open mechanism question).
- `--light-scale 0.3` empirically fixes exposure. Detection at ls=0.3 is
  the best of all tested conditions on every die; d10 was found on the
  primary pass for the first time (no missing-class recovery ladder).
- Pixel-verified nuance (controller): at ls=0.3 the dice show the
  authored blue-violet hue direction (+5..+7 blue-excess) but still
  render far lighter than the authored albedo — "correct hue tint,
  still strongly lightened," not "colors restored."
- Franka arm's washed-out look: same scene-wide exposure artifact,
  confirmed via the dedicated whole-arm diagnostic camera across light
  scales.
- Phase C picks (seed 42, `--colored-dice --light-scale 0.3`): **4/5
  PASS** (d20 237.1mm / d8 240.9mm / d12 238.6mm / d10 239.3mm z-gain,
  zero non-target drift in every run), d4 the documented permitted-fail
  (0.0mm gain, 16.3mm sideways squeeze-out, IK converged fine) — the
  colored+fixed-lighting pipeline reproduces the white baseline's
  results exactly, including d10 passing without perception assistance.
  Videos: `outputs/dice_demo/colored/gate_v/dice_pick_<die>.mp4`.
- Ops note from Phase C: every gate run hit the Kit teardown hang after
  its DONE line (killed per procedure each time); one near-miss from a
  fuzzy `pgrep|tail` PID match on a stale process — match the full
  command line with `ps aux` instead.

## Open follow-ups

- **Pick fragility (quantified 2026-07-11 post-review):** detector xy
  error varies 2-8.4mm run-to-run with render lighting; stacked on the
  ~1-5mm IK residual it occasionally exceeds the ~8mm lateral squeeze-out
  margin even for the d20 (observed: die pushed 32mm sideways, 1 FAIL in
  3 d20 runs). Hardening options: multi-frame detection averaging at
  capture; detect-again retry on zero post-lift z-gain.
  **Reproduced again 2026-07-13** (d4 rung-0 regression smoke, seed 42,
  run twice, byte-identical both times): this exact seed's rendered
  scene deterministically yields an 8.4mm detector-vs-GT xy error,
  enough to push the d20 38.5mm sideways with 0.0mm z-gain. Confirms
  this fragility is seed/scene-deterministic, not run-to-run stochastic
  noise as the "run-to-run" framing above might suggest — see
  `.superpowers/sdd/task-d4-rung0-trials-report.md`.

- **d4 grasp strategy — rung 0 (opposite-edge antipodal axis) attempted
  2026-07-13, FALSIFIED at the implementation layer, mechanism itself
  untested.** Spec
  `docs/superpowers/specs/2026-07-13-d4-edge-grasp-rung0-design.md`;
  geometry/orientation math implemented and unit-tested correct
  (`tasks/franka/antipodal_edge_grasp.py`, 17 tests) but all 5 seeded
  trials (42/123/7/1000/2026) failed identically at the tilted-axis
  descent waypoint (`stage2_descend_d4` never converged, 26.6-40.1mm
  residual vs a 5mm tolerance) — the gripper never closed in any trial,
  so the actual grasp-mechanism question (does opposite-edge-pair line
  contact survive closure without ejecting the die) remains untested,
  not falsified. Diagnostic pattern (100% reproducible non-convergence,
  a consistent ~13mm z-bias already present at the end of stage 1
  across all 5 trials regardless of which edge-pair was picked) points
  at an IK-reachability/waypoint-math gap in the scripted controller,
  not per-trial grasp instability. Full data:
  `.superpowers/sdd/task-d4-rung0-trials-report.md`. Next step
  (flagged, not yet scoped): root-cause the stage2 non-convergence
  before deciding whether to retry rung 0 or climb to rung 1 (pad
  geometry, research doc's ladder) — rung 1 wouldn't help with this
  particular failure mode either, since it's upstream of pad contact
  ever occurring.
- Phase I: detection-derived observations inside a trained policy
  (the RL lift line), then learned shape-general grasping.
- Camera angle: single fixed view occludes the die once gripped;
  Gate V verification used convergent evidence — a second angle would
  make future video evidence unambiguous.
- **d4 may be a systematically weak detection class, not just
  occasional noise (2026-07-15, d4 rung-1 cloud trials).** Distinct
  from the "Fragility attribution closed" finding above (which is about
  one seed's occasional few-mm offset): all 5 seeded trials
  (42/123/7/1000/2026) run for the rung-1 V-notch fixture work returned
  **zero** `d4`-class detections — not an offset, a total miss, in
  every trial, across 5 different scene layouts. The one trial where a
  `d4` candidate appeared at all (seed 123) was low-confidence
  (0.27-0.36) and got displaced by a same-location, higher-confidence
  `d10` candidate. 5/5 identical-shaped failures is a stronger,
  more-systematic-looking pattern than a single seed's noise — reads as
  "d4 is a weak/marginal class for this detector on this scene region,"
  not investigated further as of this pass (out of scope for the task
  that found it). See ROADMAP.md's 2026-07-15 entry and
  `.superpowers/sdd/task-2-report.md` for the full per-trial data. This
  blocks any future grasp-mechanism test that depends on this demo's
  perception step to find the d4 at all — not just rung 1's own V-notch
  hypothesis.
  **Diagnosed 2026-07-15** (`.superpowers/sdd/research-d4-detector-weakness.md`):
  d4 is NOT weak in training/eval (0.992-1.000 mAP50, among the best
  classes), not meaningfully underrepresented, not the smallest die in
  this exact scene (larger than d8/d10, which detect fine in the same
  frames), and confirmed fully visible/unoccluded in all 5 failing
  renders (direct 3D-to-pixel reprojection). Leading hypothesis, not yet
  tested: under this scene's degraded, near-textureless rendering, the
  detector may fall back on residual 3D shape cues (facet edges, apex
  highlights, shading gradients) to classify the other 4 dice, and d4's
  flat-face rest pose is the one silhouette with none of those cues — a
  shape/silhouette-flatness confound, sharper than but related to this
  page's own apparent-size-as-class-cue precedent
  ([[vision-platform]]). Needs a controlled ablation to confirm.

- **d4 rung 1 (V-notch fixture): FALSIFIED via ground-truth bypass
  (2026-07-15).** Once the perception blocker above was routed around
  (`--gt-xy-bypass`, `docs/superpowers/specs/2026-07-15-d4-rung1-pad-geometry-design.md`'s
  addendum — first build had a real bug, only protected against an
  inaccurate detection not a total miss, fixed and re-verified before
  the retry), 3/5 seeds genuinely reached the grasp mechanism and 0/3
  met the primary criterion: closure-window lateral ejection
  172.0mm/18.8mm/57.7mm (threshold ≤5mm), z-gain ~zero, zero contact
  force in all 3. Confirmed visually (frame extraction, not metrics
  alone): at closure the gripper is fully closed at the die's original
  position while the d4 sits undisturbed several cm away — the notch
  swept the die aside without ever engaging it, not a subtle
  grasp-then-eject. 2/5 seeds hit an unrelated, reproducible CUDA crash
  (hardcoded contact-sensor buffer `maxContactDataCount=4` overflowing
  under denser contact — flagged, not fixed). Open questions: whether
  the notch fixture's collision geometry is actually correctly placed
  under real dynamic closure load (only its static position was ever
  verified); whether 110°/~10mm/~4mm is a tuning problem or the
  symmetric-notch-on-flat-jaws strategy itself is wrong. Full data:
  `.superpowers/sdd/task-2-report.md`.

## Fragility attribution closed (2026-07-13 archaeology)

The d20-seed42 pick, PASS on 2026-07-11, now FAILS with an 8.4mm
detector-vs-GT error — on clean HEAD, on HEAD+instrumentation, AND on
the pre-d4 commit (f5b77c7, isolated worktree): identical 8.4mm error
everywhere, byte-identical within a session. Conclusion: code-
independent, cross-session RTX render nondeterminism moves detection a
few mm against the ~8mm squeeze-out margin. The demo's success is
knife-edge by design margin, not by any regression. Hardening options
already listed in this page (multi-frame detection averaging, tighter
recrop) are the fix path if/when the demo needs to be robust across
sessions — the razor margin is also exactly what the
noise-robustness principle (N≥20 varied-seed evals) is meant to
surface.

## Colored-dice / exposure mystery RESOLVED (2026-07-13)

Root cause of the "white dice" and every blowout observation since
2026-07-11: **doubled light energy** — the scene stacked a
DistantLight(3000) "stage light" on the default DomeLight(3000). Not
auto-exposure: `/rtx/post/histogram/enabled` was measured False all
along (manual exposure: f/5.0, ISO 100, 1/50s), killing the earlier
"RTX auto-exposure reacting to material rebind" theory. The dice USDs
always carried correct authored UsdPreviewSurface materials (verified
by pxr inspection + from-scratch regeneration reproducing the shipped
set). With the DistantLight removed (user directive, commit 1f860a9),
pixel samples match the healthy documented baseline within 1 RGB unit
and the dice render their authored pale blue-violet with legible
numerals — frames: outputs/dice_demo/exposure_check/gate_a/.
`--colored-dice` runtime rebinding and `--light-scale` workarounds are
now obsolete for the standard scene. Caveats recorded in the task
report: n=1 per lighting condition; residual run-to-run RTX render
variance exists independently (see the d20-seed42 fragility section
above). If more saturated dice are wanted, that's a datagen manifest
parameter (HSV saturation range), not a rendering fix.
