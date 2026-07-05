# AR4 sphere pick-and-place: retargeting the RL task from Cube to Sphere

## Goal

Extend the existing AR4 pick-and-place RL task to manipulate the **Sphere**
instead of the Cube, as the first step in this repo's stated direction of
growing beyond a single fixed manipulation task. This is a genuinely
different grasp geometry (round, no flat face to align a gripper against)
rather than a cosmetic rename, so it exercises real reward-function research
and iteration — not a find-and-replace exercise.

## Why the Sphere, and why not a new robot/scene

`tasks/ar4/objects_cfg.py` already defines four fully-simulated rigid
objects (Cube, RectPrism, Sphere, Wedge — real collision/mass/physics,
sized for the gripper), and `env_cfg.py` already registers all four as scene
attributes (`cube`, `sphere`, etc. — see `Ar4SceneCfg`). Only the Cube is
currently wired into the RL task itself
(`tasks/ar4/pickplace_env_cfg.py`); the other three exist as static
perception-calibration props. This makes the Sphere the lowest-risk path to
a genuinely new manipulation task: no new asset authoring, no new robot, no
new sensor work — the research energy goes entirely into making the task
actually succeed with a different object geometry, per the brainstorming
decision to build on validated infrastructure rather than start from zero.

## Scope

**In scope:**
- Retarget `Ar4PickPlaceEnvCfg` (and its scene/observations/events/rewards/
  terminations) from the Cube to the Sphere as the manipulated object.
- Redesign the reward function for the Sphere's geometry — audit each
  existing reward term (`reaching_cube`, `lifting_cube`,
  `cube_goal_tracking`, `cube_goal_tracking_fine_grained`) for whether it
  transfers as-is or needs rework for a round object (e.g., grasp-approach
  shaping, height/orientation-sensitivity of "lifted").
- Move the target placement region so it lands in genuinely empty space —
  reusing the Cube's old target region as-is is wrong once the Cube is no
  longer the manipulated object, since that region overlaps the
  RectPrism/Wedge's static positions (`objects_cfg.py`'s existing layout).
  Exact new coordinates are worked out and verified empirically during
  implementation (checked against real object positions, not just derived
  on paper), not pinned down in this spec.
- Train, evaluate, and iterate (reward shaping / hyperparameters) using the
  same verification standard as the existing Cube task: TensorBoard metrics
  (particularly the goal-reached termination rate), real eval runs with
  video, per `rl-for-manipulators`.
- Decide whether the Cube remains in the scene as a static prop (like
  RectPrism/Wedge already are) or is removed — driven by whether keeping it
  creates any collision/placement conflict with the new target region.

**Out of scope:**
- New robot, new sensors, or camera changes — the existing top-down RGB-D
  setup is reused unchanged.
- Real-camera perception accuracy for the Sphere — perception is only used
  by eval/demo entry points, not training (training uses privileged sim
  state), and the existing shape-classifier accuracy issue (tracked in
  `ROADMAP.md`) is explicitly out of scope for this task.
- RectPrism and Wedge as manipulated objects — future iterations, not this
  one.
- Any change to the AR4 robot model, gripper, or scene assets themselves.

## Success criteria

A trained policy reliably picks up the Sphere and places it at the new
target region, evaluated the same way the Cube task was: `Episode_Termination
/cube_reached_goal`-equivalent success rate climbs and plateaus in
TensorBoard, and real `eval_loop.py` runs show the sphere consistently
reaching the target region across episodes (video-verified, not just exit
code).

## Verification

- Smoke test (`--num_envs 16 --max_iterations 2 --headless`) confirms the
  retargeted env config runs and writes a checkpoint before committing to a
  full training run.
- Full training run's TensorBoard curves reviewed per the metrics already
  documented in `README.md` (reward trend, lifting signal, goal-tracking
  precision, success-termination rate).
- `eval_loop.py` run against a trained checkpoint, video inspected directly
  (per this repo's verification standard) to confirm the sphere is actually
  reaching the target, not just that the process exited cleanly.
- `perception/tests/` suite still passes unmodified (this work doesn't
  touch the perception package).
