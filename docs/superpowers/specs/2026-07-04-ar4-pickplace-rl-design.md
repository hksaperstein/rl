# AR4 Pick-and-Place RL Task — Design

Date: 2026-07-04

## Context

Follow-on to `docs/superpowers/specs/2026-07-04-ar4-gripper-objects-design.md`
(gripper + objects scene, verified working via a scripted IK reach/grasp
demo). This spec covers "area 2" from the original roadmap
(`2026-07-01-ar4-sim-foundation-design.md`): task/reward logic and training,
now made concrete as a single pick-and-place task.

The scripted `grasp_demo.py` reach was hand-tuned and imprecise (no
guarantee it actually contacts the cube). Training a policy replaces that
guesswork: the policy observes the cube's ground-truth pose (read directly
from Isaac Sim's physics state — no camera, no perception pipeline; this is
"privileged" simulation state, not available on the real robot) and learns
to reach, grasp, lift, and place it.

## Scope

- **Task**: pick up the cube from a randomized start position (near its
  current row) and place it at a randomized target position near the
  sphere/wedge on the other side. Only the cube is dynamic per-episode; the
  other three objects stay fixed as visual/spatial context, not part of the
  task yet.
- **Algorithm**: PPO via `rsl_rl`, Isaac Lab's default RL integration
  (already present as a dependency).
- **Training scale**: `num_envs=512` for training, tunable if VRAM headroom
  allows (RTX 5070 Ti, 16GB).
- **Local-only training infra**: TensorBoard logs + checkpoints written
  locally. No cloud compute or experiment tracking — that remains scoped to
  "area 3" of the original roadmap, not this spec.
- **Eval deliverable**: a fixed number of episodes (default configurable,
  e.g. 10) run with a trained checkpoint, `num_envs=1`, in the Isaac Sim
  GUI, each recorded to an `.mp4`.

## Components

### Task env: `rl/tasks/ar4/pickplace_env_cfg.py`

A new env config building on the existing scene/robot/objects configs from
the gripper+objects scene (`rl/tasks/ar4/env_cfg.py`, `robot_cfg.py`,
`objects_cfg.py`) — not a modification of `Ar4EnvCfg` in place, since this
adds genuinely new task machinery (reward, randomization, termination) that
the sim-foundation env deliberately excludes.

- **Randomization (event term, on reset)**: reposition the cube to a random
  point within ~±2cm of its current start-row position, and generate a
  random target position within ~±2cm of a point near the sphere/wedge row.
  The target is stored as part of the env's command/state (not a physical
  object) — nothing is rendered at the target location for this pass.
- **Observations**: arm+gripper joint pos/vel (16 dims); cube position
  relative to the end-effector; cube position relative to its target (goal
  error); target position relative to the robot base. Relative positions
  throughout, following Isaac Lab's own Franka lift-task observation
  pattern, since relative quantities generalize much better than raw world
  coordinates.
- **Actions**: same as the sim-foundation env — 6 arm joint position
  targets + 1 binary gripper open/close command.
- **Rewards** (dense, staged — same shape as Isaac Lab's built-in lift
  task):
  1. Reaching: negative gripper-to-cube distance.
  2. Lifting: bonus once the cube crosses a height threshold.
  3. Goal tracking: negative cube-to-target distance, weighted higher once
     lifted, so the policy doesn't try to "place" before actually holding
     the cube.
  4. Action penalty: small regularization on action rate / joint velocity.
  5. Success bonus: sparse bonus when the cube is within tolerance of the
     target.
- **Termination**: success (cube at target, held briefly) ends the episode
  early. Otherwise, a fixed max-steps timeout (truncation, not failure). No
  "dropped off the table" failure state for this first pass — kept simple
  deliberately; revisit only if training behaves badly without it.

### Training script: `rl/scripts/train.py`

Launches Isaac Lab's standard `rsl_rl` PPO training loop against the
pick-and-place env, `num_envs=512`, headless. Writes TensorBoard logs and
periodic checkpoints to `rl/logs/train/<run-name>/` (gitignored, same
pattern as existing `rl/logs/`).

### Eval/loop script: `rl/scripts/eval_loop.py`

Loads a trained checkpoint and runs a fixed number of episodes (default
configurable) with `num_envs=1` in the Isaac Sim GUI, so the run can be
watched live. Each episode is recorded to `rl/logs/videos/*.mp4` using
Isaac Lab's existing `gymnasium` `RecordVideo` integration.

## Data flow

`pickplace_env_cfg.py` (task definition) → `train.py` runs PPO for many
parallel envs, writing checkpoints/TensorBoard logs to
`rl/logs/train/<run-name>/` → `eval_loop.py` loads a checkpoint, runs N
episodes with `num_envs=1` in the GUI, writes `rl/logs/videos/*.mp4`.

## Error handling

Same minimal, fail-loud approach as prior specs in this area: `eval_loop.py`
errors clearly if the requested checkpoint path doesn't exist. No
retry/fallback logic. Training itself is expected to require iteration
(reward tuning, hyperparameter adjustment) — that is normal RL development,
not an error condition to design around.

## Testing / verification

No automated test suite, consistent with the rest of `rl/`. Verified by:
watching TensorBoard reward curves climb during training, and watching the
eval `.mp4`s show the cube actually being picked up and placed near the
other objects. A headless smoke test (few steps, assert no crash, assert
expected tensor shapes) remains a reasonable future addition once this task
is stable, but is not required for this pass.

## Out of scope (deferred)

- The other three objects (sphere, rectangular prism, wedge) as part of the
  task — cube only for this pass.
- Cloud compute, experiment tracking, shipping checkpoints across machines
  (area 3 of the original roadmap).
- Any vision-based/camera-sensor observation path — this task uses
  privileged simulation state only. Swapping in real perception is a
  sim-to-real concern (area 4).
- ROS2 integration, real-hardware deployment (area 4).
- ANY foundation-model / pretrained-policy approach (GR00T, Octo, etc.) —
  considered and rejected earlier in this project (see
  `2026-07-04-ar4-gripper-objects-design.md`).
