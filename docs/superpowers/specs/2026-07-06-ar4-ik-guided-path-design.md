# AR4 sphere pick-place — classical IK-guided path reward — Design

## Goal

An eighth attempt on the "grasp/lift never emerges" sub-problem (seven
prior reward/scene-only experiments falsified — see `ROADMAP.md`), this
time user-directed: set up a classical inverse-kinematics path-planning
mechanism and reward the RL policy for tracking it, rather than relying
purely on reward shaping of end-state achievement (reach/grasp/lift/goal
distances) to make the policy discover reasonable intermediate motion on
its own.

## Why this might succeed where reward-shaping alone didn't

Every prior experiment rewarded *outcomes* (is the object gripped? is it
above a height threshold? is it near the goal?) but never gave the policy
any information about *how to get there* — a reasonable arm trajectory,
or when to close the gripper relative to arm position. A classical IK
solver can supply exactly that: a geometrically sensible sequence of
poses (hover above the object, descend, grasp, lift, carry, place) and
concrete joint targets toward each. Tracking that path/target directly is
a much denser, more specific signal than end-state proximity alone.

## IK tool

Isaac Lab's built-in `isaaclab.controllers.DifferentialIKController`
(decided over vendoring the sibling `6DoF` project's analytical
`ar4_mk5_kinematics.py`): already GPU-batched across all environments,
zero new dependency, matches this repo's existing frame conventions with
no offset-porting risk. Trade-off accepted: iterative/numerical rather
than closed-form, but this is used here as a *reward-shaping guide*, not
a precision controller commanding the arm directly — approximate
suggestions are enough.

## Architecture

New parallel task file `tasks/ar4/pickplace_ik_guided_env_cfg.py`,
following this session's established convention: does not modify
`env_cfg.py`, `objects_cfg.py`, `pickplace_env_cfg.py`, or
`pickplace_mirror_env_cfg.py`. Reuses the mirror-scene's proven scene
(`Ar4PickPlaceMirrorSceneCfg`, including the shrunk 12mm sphere — the
sphere-shrink experiment showed size isn't the bottleneck, so keep it
fixed rather than reintroducing a second changed variable), its
spawn-randomization and mirrored-goal mechanism (`set_mirrored_goal`,
`env._target_pos_w`), and its `contact_grasp_bonus`/`stillness_penalty`
reward terms unchanged. Only the staged-progress reward term
(`staged_milestone_bonus`) is replaced.

### Important implementation refinement found during design

`DifferentialIKController` is a **closed-loop** controller: `compute()`
needs a live, physics-derived Jacobian and current end-effector
pose/joint state for every call — there is no offline "solve this
Cartesian waypoint into a joint-space snapshot" API without actually
stepping physics. Precomputing a full joint-space path once at reset
(the originally-discussed design) would require either silently
stepping a shadow simulation before the episode "starts" (complex,
touches `ManagerBasedRLEnv`'s reset internals) or a second phantom
robot instance (doubles simulation cost). Neither is worth the
complexity for what is fundamentally a reward-shaping signal.

**Resolution:** compute the IK guidance **live, every step**, using the
actual current physics state (exactly how `DifferentialIKController` is
used in Isaac Lab's own tutorial,
`scripts/tutorials/05_controllers/run_diff_ik.py`) — ask "given where the
arm actually is right now, what would classical IK suggest as the next
joint target toward the current active waypoint?" and reward the
policy's actual behavior for tracking that suggestion. This is simpler,
requires no shadow simulation, and is the controller's intended usage
pattern.

### The Cartesian path (5 waypoints, purely geometric, no IK needed to define them)

Computed once per episode at reset (cheap, vectorized torch ops, no
iteration), stored in a new per-env buffer `env._path_waypoints_w` (shape
`(num_envs, 5, 3)`, world frame):

1. **Pre-grasp**: sphere spawn position + `(0, 0, 0.05)` (5cm hover).
2. **Grasp**: sphere spawn position (at sphere height, matches
   `contact_grasp_bonus`'s target implicitly).
3. **Lift**: sphere spawn position (x, y) + z = `lift_minimal_height +
   0.02` (2cm past the existing lift threshold).
4. **Transit**: midpoint between spawn and `env._target_pos_w` (x, y),
   z = a fixed safe carry height (`0.10`).
5. **Place**: `env._target_pos_w` (the existing mirrored-goal buffer).

### Jacobian indexing (fixed- vs. floating-base)

Follow `run_diff_ik.py`'s own pattern exactly rather than assuming:
`ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1 if robot.is_fixed_base
else robot_entity_cfg.body_ids[0]` — AR4 is bolted to a stationary base
and behaves as fixed-base in every observation this session, but the
robot's own `is_fixed_base` property (not a hardcoded assumption) is what
determines the correct offset, since a wrong index silently produces a
garbage Jacobian rather than an error.

### Per-env waypoint progress tracking

A new per-env buffer `env._path_waypoint_idx` (shape `(num_envs,)`, int,
0-4), advanced when the end-effector (`ee_frame.data.target_pos_w`) comes
within `0.03m` of the current waypoint. Monotonic (never decreases within
an episode) — reset to 0 alongside the other stateful buffers.

### Reward: replaces `staged_milestone_bonus`

New function `ik_guided_path_bonus`, structured as the same *undiscounted
running-max* pattern proven correct in the mirror-scene/sphere-shrink
experiments (never the buggy discounted `staged_potential_progress`
formula) — two sub-signals combined:

1. **Waypoint proximity** (`1 - tanh(dist_to_current_waypoint / 0.1)`,
   weighted by `(waypoint_idx + 1) / 5` so later waypoints dominate,
   mirroring the existing staged weighting philosophy) — a generalization
   of the old `reach_term`/`lift_term`/`goal_term` into one continuous
   5-stage signal.
2. **IK-action-matching bonus**: each step, call
   `DifferentialIKController.compute()` (live Jacobian/ee_pose/joint_pos
   from `robot.root_physx_view.get_jacobians()`, `robot.data.body_pose_w`,
   `robot.data.joint_pos`, following the exact pattern in
   `run_diff_ik.py`) targeting the current waypoint, producing
   `joint_pos_des`. Reward
   `1 - tanh(||current_joint_pos - joint_pos_des|| / 0.5)` — how close the
   arm's *actual* joint configuration is to what classical IK suggests
   right now. This is the concrete "IK path guidance" signal, separate
   from raw Cartesian proximity.

Raw combined signal (per step) is the sum of both sub-signals, run through
the same running-max/undiscounted-delta wrapper as
`staged_milestone_bonus`, so the final reward is 0 at a plateau and
positive only on genuine new progress.

`contact_grasp_bonus` and `stillness_penalty` remain unchanged and
additive alongside this term — grasp truth still comes from real contact
sensors, not from IK guidance.

### Approximation accepted

The IK guidance targets the raw `link_6` body pose (Isaac Lab's Jacobian
is computed per rigid body), not the `_EE_OFFSET`-adjusted gripper pinch
point the rest of this task's reward uses. This 3.6cm discrepancy is
acceptable for a shaping *guide* (not a precision controller) — noted
explicitly so it isn't mistaken for an oversight later.

## Gripper-state-matching bonus (new)

Small additional reward term `gripper_schedule_bonus`: the classical plan
knows the gripper should be open through waypoints 0-1 (pre-grasp,
grasp-approach) and closed from waypoint 2 onward (lift, transit, place).
Reward `+0.1` when the commanded gripper action matches the expected
open/closed state for the current waypoint index, `0` otherwise. Cheap,
direct signal for grasp *timing* — nothing in any of the seven prior
experiments explicitly rewarded when to close the gripper relative to arm
position.

## `RewardsCfg` for the new task

```python
ik_guided_path_bonus = RewTerm(func=ar4_mdp.ik_guided_path_bonus, weight=25.0, params={...})
gripper_schedule_bonus = RewTerm(func=ar4_mdp.gripper_schedule_bonus, weight=1.0, params={...})
contact_grasp_bonus = RewTerm(func=ar4_mdp.contact_grasp_bonus, weight=20.0, params={...})  # reused unchanged, now a direct additive term since it's no longer folded into the staged signal
stillness_penalty = RewTerm(func=ar4_mdp.stillness_penalty, weight=2.0, params={...})  # unchanged from the sphere-shrink experiment
action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4, params={...})
```

Note `contact_grasp_bonus` moves from being a sub-term *inside* the staged
signal (as it was in `_raw_lift_progress`/`_raw_lift_progress_mirrored`)
to a standalone top-level `RewardsCfg` term here, since the new staged
signal (`ik_guided_path_bonus`) is built from waypoint/IK proximity, not
from `_raw_lift_progress`'s reach/grasp/lift/goal sub-terms.

## Verification plan

1. Smoke test (`--num_envs 16 --max_iterations 2`): exits 0, confirms 6
   reward terms active, no exceptions on any new stateful buffer
   (`_path_waypoints_w`, `_path_waypoint_idx`) regardless of
   reward-fn/reset-event ordering.
2. Verify the IK guidance itself in isolation before the full run: a
   throwaway script that resets a small env, steps a few times, and
   confirms `joint_pos_des` from `DifferentialIKController.compute()` is
   finite (no NaN from a singular Jacobian) and changes sensibly as the
   waypoint index advances.
3. Full 1500-iteration run at `num_envs=4096`, then the same real-eval-
   video decision gate as every prior experiment (10 episodes, extract
   frames, visually inspect **frame-by-frame, not just coarse samples** —
   two prior experiments' coarse-sample checks misread an accidental
   collision-launch as a genuine lift).
4. Record outcome in `ROADMAP.md` regardless of result.

## Global constraints for the plan

- Do not modify `env_cfg.py`, `objects_cfg.py`, `pickplace_env_cfg.py`,
  `pickplace_mirror_env_cfg.py`, or any existing function in `mdp.py`
  (including the mirror-scene/sphere-shrink task's own functions:
  `contact_grasp_bonus`, `stillness_penalty`, `reset_stillness_buffers`,
  `set_mirrored_goal`, `mirrored_target_position_in_robot_root_frame`,
  `object_reached_mirrored_goal`) — this is a new, parallel task.
- `DifferentialIKController` config: `command_type="position"`,
  `ik_method="dls"` (damped least-squares — most robust near
  singularities, matching the tutorial's default), default `ik_params`.
- Waypoint advance tolerance `0.03m`; proximity std `0.1`; IK-joint-match
  std `0.5` (radians); carry height `0.10m`; pre-grasp hover `0.05m`; lift
  margin `0.02m` above `lift_minimal_height=0.03` — decided values, not
  placeholders.
- `num_envs=4096` for the real training run, matching every prior
  experiment this session.
- Reuses `Ar4PickPlacePPORunnerCfg` unchanged (no new hyperparameters).
