# AR4 Perception Integration — Camera, Training Adapter, Interactive Demo — Design

Date: 2026-07-04

## Context

Follow-on to `docs/superpowers/specs/2026-07-04-ar4-pickplace-rl-design.md`, which
trained a PPO policy on *privileged* simulation state (ground-truth cube pose read
directly from physics — "no camera, no perception pipeline"). That spec explicitly
deferred any vision-based observation path to "area 4" of the original roadmap
(`2026-07-01-ar4-sim-foundation-design.md`).

This spec is that area-4 work, scoped to stay inside simulation (no ROS2/real
hardware yet): add a real RGB-D camera and a perception pipeline modeled on what
would actually run on the real robot, use it to identify objects by shape (not by
a privileged scene-entity name), wire it into inference/demo time without
disturbing the already-working training loop, and add an interactive demo plus a
README so this is runnable end-to-end by a human, not just via scripts.

## Scope

- **Camera**: a single static RGB-D sensor, mounted directly above the workspace
  looking straight down (not robot-mounted — no wrist camera available). Modeled
  on a real depth camera (e.g. Intel RealSense D435-class specs: 640x480 RGB-D,
  ~87°x58° FOV), mounted close (~0.5m above the ground plane) since the objects
  here are small (9-18mm) and depth precision degrades with range.
- **Perception pipeline** (classical, not a learned model — see "Approaches
  considered" below): RANSAC ground-plane removal on the depth point cloud,
  Euclidean clustering of the remainder into per-object point clusters, then a
  geometric shape classifier per cluster (bounding-box extent ratios for
  cube-vs-rectangular-prism, sphere-fit residual, planar-normal detection for the
  wedge's slanted face). A lightweight per-object tracker (nearest-centroid
  matching frame-to-frame) reports each object's position and shape class, and
  keeps reporting the last-known position with a staleness counter when an
  object is temporarily occluded or otherwise not visible.
- **Perception calibration clip**: a standalone script/scene, run once as a
  sanity check (not during training), where a cube slides across the camera's
  field of view for 5-10s. Output is an mp4 with the detected mask/bounding
  box/shape label burned into each frame.
- **Training**: unchanged task logic (`Ar4PickPlaceEnvCfg` observations,
  rewards, terminations, action space — all reused as-is from the prior spec),
  still trained on privileged simulation state for speed. Only the scale
  changes: `num_envs` 512 -> 4096 (verify VRAM headroom with a short smoke test
  first, since training doesn't render/run perception per-env, physics-only
  scale-up). Algorithm stays PPO via `rsl_rl` (see "Approaches considered").
- **Inference-time perception adapter**: `eval_loop.py` and the new interactive
  demo build the same observation tensor shape the policy was trained on, but
  source "cube position" from the perception pipeline's tracked detection
  classified as *cube* (falling back to its last-known/stale estimate) instead
  of a privileged sim-state read. Training code path is untouched.
- **Interactive demo** (`rl/scripts/interactive_demo.py`): GUI, `num_envs=1`,
  trained policy loaded. Continuously runs perception; once the tracked cube's
  position has been stable for ~1s *and* the detection is both fresh
  (non-stale) and within a defined reachable workspace region, triggers the
  policy to pick it up and place it at the fixed "other side" target region
  (same range used in training). Out-of-view or out-of-bounds cube positions
  are ignored (no trigger, timer reset) rather than acted on.
- **`rl/README.md`** (new): end-to-end instructions — build assets, run the
  perception calibration clip, train (with the num_envs=4096 VRAM-smoke-test
  caveat), evaluate a checkpoint, run the interactive demo — each with an
  exact, explicit launch command and what a healthy result looks like.
- **Training diagnostics**: document the per-reward-term TensorBoard logging
  Isaac Lab already does automatically; add one new custom scalar (success
  rate: fraction of episodes ending via `cube_reached_goal` vs. timeout) so
  training health and failure mode are both visible at a glance. Both the
  calibration clip and every eval/demo video get the perception overlay
  burned in, so perception correctness is visible during actual policy runs,
  not just in isolation.
- **Script-launch convention fix**: `train.py` and `eval_loop.py`'s docstrings
  currently show `./isaaclab.sh -p rl/scripts/...`, which silently assumes
  `isaaclab.sh` is aliased onto the PATH from this repo's root — it isn't
  (`isaaclab.sh` lives in the separate IsaacLab install at
  `/home/saps/IsaacLab/`, `rl/` lives in this repo). Fix both docstrings, and
  write the new demo script's docstring and the README, using the explicit
  form: `cd` to this repo's root, then invoke IsaacLab's launcher by its real
  absolute path, e.g.:
  ```bash
  cd /home/saps/projects/6DoF
  /home/saps/IsaacLab/isaaclab.sh -p rl/scripts/interactive_demo.py --checkpoint rl/logs/train/<run>/model_1500.pt
  ```

## Approaches considered

**Perception ground truth vs. real sensing**: Isaac Sim can produce free,
perfect ground-truth segmentation/bounding boxes from scene metadata. Rejected
in favor of an actual RGB-D-camera-driven pipeline, since all perception work
here is meant to reflect what would run on the real robot (this is area-4,
sim-to-real-facing work) — ground-truth annotations don't transfer to real
hardware at all.

**Localization+classification model choice**: considered fine-tuning an
off-the-shelf detector (e.g. YOLO) directly on cube/sphere/prism/wedge classes
using Isaac-Sim-rendered synthetic training data. Rejected for this first pass
in favor of classical depth segmentation (plane removal + Euclidean clustering)
plus geometric shape classification: no model training/data-generation step
needed before perception can be evaluated at all, directly exercises depth data
to answer "can depth distinguish a cube from a rectangular prism" (yes — via
3D extent fitting), and degrades gracefully under partial occlusion since
geometric fitting works from partial point clouds. Revisit with a learned
model only if this proves insufficient in practice.

**Perception-in-the-loop training vs. privileged-state training +
inference-time adapter**: running the full perception pipeline for all 4096
parallel training environments at every simulation step was considered and
rejected — it would mean rendering + CV inference 4096x per step instead of a
free physics-state read, likely making training impractically slow. Training
keeps reading privileged state (already fast, already working), shaped
identically to what perception provides, so the same trained policy consumes
either source unchanged.

**RL algorithm**: PPO via `rsl_rl` (already implemented in the prior spec) was
re-confirmed rather than revisited. Alternatives — SAC/TD3 (off-policy,
more sample-efficient per env-step, but don't benefit as much from thousands
of parallel GPU envs and need more delicate tuning) and DQN/DDQN-style
value-based methods (don't natively handle continuous joint-target actions
without lossy discretization) — don't fit this massively-parallel,
continuous-action setting as well. PPO is also Isaac Lab's own first-class,
best-supported integration, with proven hyperparameters to start from (already
adapted from Isaac Lab's Franka lift-task example).

**Demo trigger**: continuous automatic stabilization-detection was chosen over
a manual "press Enter when ready" trigger, despite the latter being simpler
and having no heuristic to tune — automatic detection is the more hands-off,
"interactive" experience the demo is meant to showcase.

## Components

### Camera + scene: `rl/tasks/ar4/pickplace_env_cfg.py` (extended)

Add a new `Ar4PickPlacePerceptionSceneCfg(Ar4PickPlaceSceneCfg)` subclass
carrying a `Camera` sensor (RGB + `distance_to_image_plane` depth), mounted on
a fixed world prim above the workspace center, looking straight down, not
attached to the robot. `train.py` keeps using the base
`Ar4PickPlaceSceneCfg`/`Ar4PickPlaceEnvCfg` unchanged — camera rendering isn't
free even if its output goes unused, so it must not run during the 4096-env
training loop. `eval_loop.py` and the interactive demo use a corresponding
`Ar4PickPlacePerceptionEnvCfg(Ar4PickPlaceEnvCfg)` that swaps in the
perception scene, so only those two entry points pay for the camera.

### Perception module: `rl/perception/`

New package, independent of `rl/tasks/`:
- `segmentation.py`: RANSAC ground-plane removal + Euclidean clustering on a
  depth-derived point cloud.
- `shape_classifier.py`: per-cluster geometric classification (bounding-box
  extents, sphere-fit residual, planar-normal detection).
- `tracker.py`: frame-to-frame nearest-centroid matching, last-known-pose +
  staleness-counter persistence per tracked object.
- `overlay.py`: draws mask/bbox/shape-label annotations onto rendered RGB
  frames, shared by the calibration clip, `eval_loop.py`, and the interactive
  demo.

### Calibration script: `rl/scripts/perception_calibration.py`

Spawns just the camera and a cube scripted to slide across the frame over
5-10s (no arm needed). Runs the perception pipeline every frame, overlays
detections, writes `rl/logs/videos/perception_calibration.mp4`. Run once,
manually, before trusting perception in the eval/demo scripts — not part of
any training or automated flow.

### Training: `rl/tasks/ar4/pickplace_env_cfg.py`, `rl/scripts/train.py`

`num_envs` default raised 512 -> 4096 in both the env cfg and `train.py`'s
`--num_envs` default. No other change — observations, rewards, terminations,
actions, and the PPO agent config (`Ar4PickPlacePPORunnerCfg`) are reused
as-is from the prior spec.

### Eval script: `rl/scripts/eval_loop.py` (extended)

Adds the camera-enabled scene variant, runs the perception pipeline each step,
sources the "cube position" observation from the tracked cube detection
(falling back to last-known/stale pose) instead of the privileged
`object_position_in_robot_root_frame` read, and burns the perception overlay
into the recorded mp4. Docstring launch command fixed per the convention
above.

### Interactive demo: `rl/scripts/interactive_demo.py`

GUI, `num_envs=1`, trained policy + camera-enabled scene. Per frame: run
perception, update the tracker, check whether the cube detection is
(a) fresh (non-stale) and (b) within a defined reachable workspace bounding
box; if both hold and its position has stayed within a small tolerance for
~1s, trigger a pick-and-place rollout to the fixed target region, using the
same manual observation-construction path as `eval_loop.py`. Otherwise stay
idle (log a status line) and keep watching — an out-of-view or out-of-bounds
cube never triggers an attempt, and any such state simply resets the
stabilization timer. Recorded to mp4 with the perception overlay, same as
eval.

### `rl/README.md`

New file. Ordered walkthrough: build assets -> perception calibration clip ->
train (smoke-test the 4096-env VRAM footprint first, then full run,
referencing TensorBoard reward-curve and success-rate scalars to judge when
training has converged, rather than a fixed iteration count) -> eval -> 
interactive demo. Every step includes its exact launch command (explicit `cd`
+ absolute launcher path, not the ambiguous `./isaaclab.sh` shorthand) and a
description of what a healthy result looks like.

### Docstring fixes: `rl/scripts/train.py`, `rl/scripts/eval_loop.py`

Update the existing `.. code-block::` launch examples to the explicit
`cd <repo-root>` + `/home/saps/IsaacLab/isaaclab.sh -p ...` form, matching the
new scripts, instead of the ambiguous `./isaaclab.sh -p ...` shorthand.

### Training diagnostics

No new logging infrastructure needed for per-term reward curves (already
automatic via Isaac Lab's `extras` -> `rsl_rl`'s TensorBoard writer) — document
which scalars to watch in the README. Add one new custom scalar, success rate
(fraction of episodes ending via the `cube_reached_goal` termination vs.
timeout), logged once per training iteration alongside the existing reward
curves.

## Data flow

Camera sensor (rendered RGB-D, eval/demo only) -> perception module
(segmentation -> shape classification -> tracking) -> tracked per-object
{position, shape class, staleness} list -> inference-time adapter selects the
*cube*-classified detection -> observation tensor (same shape as training) ->
loaded PPO policy -> joint position + gripper actions -> env step -> recorded
mp4 with perception overlay.

Training itself has a separate, simpler data flow, unchanged from the prior
spec: `pickplace_env_cfg.py` (privileged state) -> `train.py` (PPO, 4096
envs) -> checkpoints/TensorBoard logs in `rl/logs/train/<run-name>/`.

## Error handling

Same fail-loud approach as prior specs: missing checkpoint paths error
clearly (already the case in `eval_loop.py`; same in the new demo script).
Out-of-view/out-of-bounds cube detections are not errors — they're an
expected, handled state (idle/wait), described above under "Interactive
demo." No retry/fallback logic beyond that.

## Testing / verification

- Perception: watch `perception_calibration.mp4` — the cube should be
  correctly masked, boxed, and labeled "cube" (not confused with another
  shape) throughout its slide across the frame.
- Training: watch TensorBoard — per-term reward curves climbing, and the new
  success-rate scalar rising off zero, rather than judging convergence from
  a fixed iteration count.
- Eval/demo: watch the recorded mp4s — perception overlay correctly labels
  all four objects (not just the cube) throughout, and the arm actually
  picks up and places the cube. Manually test the demo's validity gating by
  dragging the cube out of the camera's view and out of the workspace bounds,
  confirming the arm stays idle in both cases and only acts once the cube is
  back in a valid, stable position.
- No automated test suite, consistent with the rest of `rl/`.

## Out of scope (deferred)

- Wrist-mounted / multi-camera perception (would resolve more occlusion
  cases, not available/needed for this pass).
- A learned CV detector (fine-tuned or otherwise) — revisit only if classical
  depth segmentation + geometric shape classification proves insufficient.
- Perception-in-the-loop training.
- User-adjustable demo target location — the "other side" target stays the
  fixed range used in training; only the cube's start position is
  interactive.
- ROS2 integration, real-hardware deployment (area 4's remaining scope
  beyond this spec).
- Any object other than the cube being interactively moved/tracked as a task
  target (sphere/prism/wedge remain fixed scene context, same as the prior
  spec).
