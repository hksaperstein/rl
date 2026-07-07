# Classical perception-driven pick-and-place demo (no RL)

## Purpose

A fully classical (zero learned components) demo: the camera detects the
cube live, its position is transformed from camera/world frame into the
robot's root frame, a 5-waypoint Cartesian path is planned live from that
detection, and a live differential-IK controller executes it — pregrasp,
grasp, lift, transit, place, release, home. This complements, but is
distinct from, the two existing demo scripts:

- `grasp_demo.py`: classical, but joints are hardcoded/precomputed offline
  against a known fixed cube position — no camera, no live planning.
- `interactive_demo.py`: uses live camera perception, but drives a
  *trained RL policy*, not a classical planner.

Neither exercises "detect live → transform camera-to-EE → plan a path
live → execute with classical IK" end-to-end. This demo is also a direct,
useful diagnostic for the Experiment 11-13 line of work: if a pure
classical controller reliably completes the full pick-and-place sequence
on the same cube/scene, that's evidence the RL policies' difficulty is a
genuine exploration/learning problem, not a sign the task is physically
infeasible for this arm/gripper/object; if the classical controller also
struggles (e.g., can't maintain a stable grasp through the lift), that
points the diagnosis somewhere else entirely (contact/grasp mechanics)
rather than at RL specifically.

## Design

New script `scripts/classical_pickplace_demo.py` and a new minimal scene
cfg `Ar4ClassicalDemoSceneCfg` in a new file
`tasks/ar4/classical_demo_env_cfg.py` — additive, does not modify
`env_cfg.py`, `objects_cfg.py`, or any `pickplace_*.py` file.

**Scene**: robot + single cube (`CUBE_CFG`, from `objects_cfg.py`,
unmodified) + `ee_frame` (`FrameTransformerCfg`, same `_EE_OFFSET`
convention as every other task) + `perception_camera` (`CameraCfg`, same
`_PERCEPTION_CAMERA_POS`/`_PERCEPTION_CAMERA_QUAT_WORLD` constants as
`pickplace_env_cfg.py`/`pickplace_single_object_env_cfg.py`). Closely
modeled on `Ar4PickPlaceSingleObjectSceneCfg`
(`tasks/ar4/pickplace_single_object_env_cfg.py:34-73`), but with `CUBE_CFG`
instead of `SPHERE_CFG`, matching this repo's current single-object
scope (CLAUDE.md: "one AR4 arm, one cube"). `num_envs=1` (single demo
instance, matching `interactive_demo.py`'s convention). Actions: plain,
unmodified `isaaclab_mdp.DifferentialInverseKinematicsActionCfg` (same
`scale=0.05`, `body_name="link_6"`, `body_offset=_EE_OFFSET`,
`command_type="position"`, `use_relative_mode=True`, `ik_method="dls"` as
every other task-space env cfg) + `BinaryJointPositionActionCfg` for the
gripper — **not** `ResidualDifferentialIKActionCfg`: this demo has no
policy, so there's no residual to add; the script commands the base
pursuit step directly as the raw action each step. A bare
`ManagerBasedEnv` (not the RL variant), matching `grasp_demo.py`'s
precedent — no reward/termination managers needed for a scripted
demonstration.

**Perception → EE-frame transform**: reuse `_perception_adapter.py`'s
existing pattern exactly (`perceive_sphere`'s internals,
`scripts/_perception_adapter.py:41-67`), generalized to a shape-label
parameter instead of being hardcoded to `"sphere"` — add a new
`perceive_object(env, camera, tracker, ground_z, shape_label, env_index=0)`
function there (small, targeted addition; `perceive_sphere` itself stays
as a thin wrapper calling it with `shape_label="sphere"`, so
`interactive_demo.py`/`eval_loop.py --perception` need no changes). Reuses
`perception.pipeline.run_perception` and `perception.tracker.find_by_shape`
unmodified — `find_by_shape` is already shape-label-generic, only the
demo-facing wrapper was sphere-specific.

**Path planning**: a new pure function `plan_waypoints(cube_pos_b,
goal_pos_b, lift_minimal_height, pregrasp_hover, lift_margin,
carry_height) -> torch.Tensor` (shape `(5, 3)`) in the new
`scripts/classical_pickplace_demo.py` (not `tasks/ar4/mdp.py` — this is
demo-script logic operating on plain tensors from live perception, not an
Isaac Lab event term operating on privileged env state; keeping it
separate avoids conflating the two). Same geometry as
`compute_path_waypoints` (`tasks/ar4/mdp.py:372-422`): pregrasp (cube
position + hover height), grasp (cube position), lift (cube xy, fixed
lift height), transit (midpoint xy, carry height), place (goal position).
Deliberately duplicated rather than refactored out of `mdp.py`, per this
repo's YAGNI/don't-touch-working-code discipline — the duplication is ~10
lines of vector arithmetic, not worth coupling a live perception-driven
script to an Isaac Lab `EventTerm`'s signature (`env`, `env_ids`) it
doesn't need.

**Execution loop**: each control step:
1. Run perception on the current camera frame (`perceive_object`,
   `shape_label="cube"`) → cube position in robot-root frame, or `None`
   if not detected this frame (in which case: hold position, don't
   advance the plan — same "no override this frame" philosophy
   `_perception_adapter.py` already uses elsewhere).
2. On the first successful detection only, call `plan_waypoints(...)`
   once with a fixed goal position (not randomized — a single
   demonstration run showing the pipeline works, not a generalization
   test) and initialize `waypoint_idx = 0`.
3. Compute the current EE position (`ee_frame` `FrameTransformer`, robot
   frame) and a bounded pursuit step toward
   `waypoints[waypoint_idx]` — identical math to
   `ResidualDifferentialIKAction._compute_base_delta()`
   (`tasks/ar4/residual_ik_action.py`, Experiment 13's Task 1): direction
   normalized, magnitude capped at the same `_BASE_MAX_STEP = 0.05`.
4. Gripper command: open for `waypoint_idx < 2`, closed from
   `waypoint_idx >= 2` onward — same schedule as `gripper_schedule_bonus`
   (`tasks/ar4/mdp.py:525-554`), reimplemented inline here since this
   script has no `ManagerBasedRLEnv`/`RewardManager` to call it through.
5. Step the env with `[pursuit_step (3,), gripper_command (1,)]`.
6. Advance `waypoint_idx` (capped at 4) when within `advance_tolerance`
   of the current waypoint — same `advance_tolerance=0.03` used
   throughout this repo's path-tracking reward terms.
7. After waypoint 4 (place) is reached: open the gripper (release), then
   command a return-to-home pose, then stop.
8. Timeout: a generous fixed step budget (matching `episode_length_s=5.0`
   x some multiple, e.g. 3x = 15s, since there's no RL-episode-length
   constraint pushing for a tight budget here — this is a demo, not a
   training signal) — if exceeded without reaching waypoint 4, stop and
   report incomplete rather than looping forever.

**Video/visualization**: record an mp4 via `imageio` (matching every
other demo's convention), overlaying detections via
`perception.overlay.draw_detections` (already used by
`interactive_demo.py`) so the recorded video shows exactly what the
camera pipeline detected each frame, not just the raw scene — this is
the whole point of a "camera-driven" demo being visibly verifiable.
Non-headless by default (GUI visible), matching `grasp_demo.py`/
`interactive_demo.py`'s convention for demo scripts (as opposed to
`train.py`'s `--headless` default) — reuse `_raise_window_in_background`
(`scripts/grasp_demo.py`, or extract to a shared tiny helper if this
starts feeling duplicated across three call sites; for now, duplicate,
matching the existing precedent of `grasp_demo.py` and `interactive_demo.py`
not sharing this helper via a common module either).

## What this does NOT do

- No RL, no policy, no training, no reward function.
- No modification to any existing task/scene/reward file.
- No goal randomization (single fixed goal per run — this demonstrates
  the pipeline working, it isn't a generalization benchmark).
- No interactive drag-and-trigger loop like `interactive_demo.py` — the
  cube is placed once (either via a scripted random reset event, matching
  the RL task's spawn distribution, or left at its scene-default position
  for the first run) and the demo executes a single pass.

## Verification plan

1. Syntax-check both new files.
2. Headless smoke test (a few seconds of stepping) confirming the env
   constructs, perception runs without exception, and the action/gripper
   tensors have the right shape.
3. A real, non-headless run, with the resulting video personally
   inspected (frame-by-frame, same standard as every experiment's eval
   video in this session) for: does perception correctly detect the cube
   each frame (visible in the overlay), does the arm reach and grasp it,
   does it lift, carry, and place it in the fixed goal region, does it
   release and return home. Record the outcome as a factual account (not
   spun toward a predetermined conclusion) — this is a diagnostic, and a
   failure here is exactly as informative as a success, per this
   project's standing practice of recording negative results.
