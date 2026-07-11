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

## Open follow-ups

- d4 grasp strategy (reorient/edge-grasp/push-assist — flat-pad
  mid-height squeeze fails on a tetrahedron).
- Phase I: detection-derived observations inside a trained policy
  (the RL lift line), then learned shape-general grasping.
- Camera angle: single fixed view occludes the die once gripped;
  Gate V verification used convergent evidence — a second angle would
  make future video evidence unambiguous.
