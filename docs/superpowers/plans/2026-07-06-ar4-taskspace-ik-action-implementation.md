# AR4 Task-Space IK-Driven Action Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Experiment 11 — replace the AR4 cube task's joint-space action (`JointPositionActionCfg`) with a task-space action (`DifferentialInverseKinematicsActionCfg`), so the policy outputs Cartesian end-effector deltas and a live differential-IK controller (in the control loop, not just the reward) converts them to joint targets — testing whether offloading low-level joint coordination to a classical solver fixes the precision bottleneck Experiment 10 exposed (antipodal grasp condition regressed to exactly 0).

**Architecture:** New parallel task file `tasks/ar4/pickplace_taskspace_env_cfg.py` reusing `Ar4PickPlaceMirrorSceneCfg` (scene/cube/contact-sensors, unchanged) and Experiment 10's reward fixes (`antipodal_grasp_bonus` at `-0.7071`, `gripper_schedule_bonus`, `stillness_penalty`, `action_rate`, `joint_vel`, all unchanged), with a new `ActionsCfg` (task-space IK action) and a new, simpler `path_proximity_bonus` reward function (in `tasks/ar4/mdp.py`) that drops the IK-controller/Jacobian computation `ik_guided_path_bonus` needed when IK was reward-only.

**Tech Stack:** Isaac Lab `ManagerBasedRLEnvCfg`, `isaaclab.envs.mdp.DifferentialInverseKinematicsActionCfg` + `isaaclab.controllers.DifferentialIKControllerCfg`, PyTorch, rsl_rl PPO (`Ar4PickPlacePPORunnerCfg`, unchanged).

## Global Constraints

- Do not modify `env_cfg.py`, `objects_cfg.py`, `pickplace_env_cfg.py`, `pickplace_mirror_env_cfg.py`, or `pickplace_ik_guided_env_cfg.py`.
- Do not modify any existing function in `tasks/ar4/mdp.py` (`ik_guided_path_bonus`, `compute_path_waypoints`, `antipodal_grasp_bonus`, `contact_grasp_bonus`, `gripper_schedule_bonus`, `stillness_penalty`, `set_mirrored_goal`, etc.) — only add the new `path_proximity_bonus` function.
- Decided values (verbatim, not placeholders): action `scale=0.05`, `body_name="link_6"`, `body_offset.pos=_EE_OFFSET=(0.0, 0.0, 0.036)`, `command_type="position"`, controller `use_relative_mode=True`, `ik_method="dls"`.
- `num_envs=4096` for the real training run.
- Reuse `Ar4PickPlacePPORunnerCfg` unchanged — no new PPO hyperparameters.
- Always invoke scripts via `/home/saps/IsaacLab/isaaclab.sh -p scripts/<script>.py` from the repo root, never plain `python`.
- Verify completion via files (checkpoint counts, `model_<N>.pt` existence, event-file mtimes), not console text or exit codes alone — this session's established practice against false positives from Isaac Sim's unreliable clean-shutdown/stdout behavior.

---

### Task 1: `path_proximity_bonus` reward function

**Files:**
- Modify: `tasks/ar4/mdp.py` (append new function at end of file, after `antipodal_grasp_bonus` which ends at line 591)
- Test: none (this repo has no unit tests for reward functions that depend on live Isaac Sim state; verification happens via the smoke test in Task 3)

**Interfaces:**
- Consumes: `env._path_waypoints_w`, `env._path_waypoint_idx`, `env._ik_milestone_max` — all three are lazily created AND per-episode reset by the existing, unmodified `compute_path_waypoints` event function (`tasks/ar4/mdp.py:372-422`). This new function's own lazy-init block (below) only matters if `path_proximity_bonus` is ever evaluated before the first reset event fires, matching the identical defensive pattern `ik_guided_path_bonus` already uses.
- Produces: `path_proximity_bonus(env, ee_frame_cfg, proximity_std, advance_tolerance) -> torch.Tensor` of shape `(num_envs,)` — a `RewTerm` function, used by Task 2's `RewardsCfg`.

- [ ] **Step 1: Append the new function to `tasks/ar4/mdp.py`**

Add this function at the end of the file (after `antipodal_grasp_bonus`):

```python
def path_proximity_bonus(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg,
    proximity_std: float,
    advance_tolerance: float,
) -> torch.Tensor:
    """Undiscounted running-max bonus (same corrected pattern as
    staged_milestone_bonus/ik_guided_path_bonus - see those functions'
    docstrings for the decay bug this avoids) for Cartesian proximity to
    the current path waypoint (env._path_waypoints_w[:, env._path_waypoint_idx]),
    weighted so later waypoints dominate.

    Unlike ik_guided_path_bonus, this drops the IK-action-matching
    sub-signal entirely: this task's arm action is driven by a live
    DifferentialInverseKinematicsAction (see pickplace_taskspace_env_cfg.py's
    ActionsCfg), so the arm tracks IK's suggestion by construction -
    scoring "does the joint configuration match what IK suggests" would
    be close to tautological here. See
    docs/superpowers/specs/2026-07-06-ar4-taskspace-ik-action-design.md.

    The waypoint index itself advances (monotonically, capped at 4)
    whenever the end-effector comes within advance_tolerance of the
    current waypoint - identical mechanism to ik_guided_path_bonus.
    Reuses env._ik_milestone_max (initialized/reset by
    compute_path_waypoints, unchanged) as its running-max buffer rather
    than introducing a new one, since compute_path_waypoints already
    owns that buffer's lazy-init and per-episode reset.
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    if not hasattr(env, "_path_waypoints_w"):
        env._path_waypoints_w = torch.zeros(env.num_envs, 5, 3, device=env.device)
        env._path_waypoint_idx = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._ik_milestone_max = torch.zeros(env.num_envs, device=env.device)

    current_waypoint = torch.gather(
        env._path_waypoints_w, 1, env._path_waypoint_idx.view(-1, 1, 1).expand(-1, 1, 3)
    ).squeeze(1)

    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    dist_to_waypoint = torch.norm(ee_pos_w - current_waypoint, dim=-1)

    reached = dist_to_waypoint < advance_tolerance
    env._path_waypoint_idx = torch.where(
        reached & (env._path_waypoint_idx < 4), env._path_waypoint_idx + 1, env._path_waypoint_idx
    )

    proximity_term = (1.0 - torch.tanh(dist_to_waypoint / proximity_std)) * (
        env._path_waypoint_idx.float() + 1.0
    ) / 5.0

    prev = env._ik_milestone_max.clone()
    env._ik_milestone_max = torch.maximum(env._ik_milestone_max, proximity_term)
    return env._ik_milestone_max - prev
```

No new imports needed: `torch`, `SceneEntityCfg`, and `FrameTransformer` (under `TYPE_CHECKING`) are already imported at the top of `tasks/ar4/mdp.py`.

- [ ] **Step 2: Syntax-check the file**

Run: `python3 -c "import ast; ast.parse(open('tasks/ar4/mdp.py').read())"`
Expected: no output (parses cleanly). This is a pure syntax check — it does not require Isaac Sim, unlike actually importing the module.

- [ ] **Step 3: Commit**

```bash
git add tasks/ar4/mdp.py
git commit -m "Add path_proximity_bonus reward function for Experiment 11 (task-space IK action)"
```

---

### Task 2: New task-space env cfg file

**Files:**
- Create: `tasks/ar4/pickplace_taskspace_env_cfg.py`

**Interfaces:**
- Consumes: `Ar4PickPlaceMirrorSceneCfg` (from `tasks/ar4/pickplace_mirror_env_cfg.py`, unmodified), `ar4_mdp.path_proximity_bonus` (Task 1), `ar4_mdp.antipodal_grasp_bonus` / `ar4_mdp.gripper_schedule_bonus` / `ar4_mdp.stillness_penalty` / `ar4_mdp.set_mirrored_goal` / `ar4_mdp.compute_path_waypoints` / `ar4_mdp.mirrored_target_position_in_robot_root_frame` / `ar4_mdp.object_reached_mirrored_goal` (all pre-existing, unmodified, from `tasks/ar4/mdp.py`), `_EE_OFFSET` (from `tasks/ar4/pickplace_env_cfg.py`), `ARM_JOINT_NAMES` / `GRIPPER_JOINT_NAMES` / `GRIPPER_OPEN_POS` / `GRIPPER_CLOSED_POS` (from `tasks/ar4/robot_cfg.py`).
- Produces: `Ar4PickPlaceTaskspaceEnvCfg` class — consumed by Task 3 (script wiring).

- [ ] **Step 1: Write the new file**

```python
# tasks/ar4/pickplace_taskspace_env_cfg.py
"""Task-space IK-driven action variant of the AR4 mirror-goal cube task
(Experiment 11): identical scene/spawn-randomization/mirrored-goal/reward
fixes as pickplace_ik_guided_env_cfg.py, but the arm's action space is
replaced with Isaac Lab's built-in DifferentialInverseKinematicsActionCfg
(a Cartesian end-effector delta each step, converted to joint targets by
a live differential-IK controller inside the control loop) instead of
JointPositionActionCfg (direct joint-angle deltas). See
docs/superpowers/specs/2026-07-06-ar4-taskspace-ik-action-design.md.

Additive/parallel to every other pickplace_*.py file: does NOT modify
env_cfg.py, objects_cfg.py, pickplace_env_cfg.py, pickplace_mirror_env_cfg.py,
or pickplace_ik_guided_env_cfg.py. Reuses Ar4PickPlaceMirrorSceneCfg
directly (same cube scene, same contact sensors, same ee_frame) - only the
action space and the path-tracking reward's internals differ from
pickplace_ik_guided_env_cfg.py.

Import this module only after an Isaac Sim/Isaac Lab AppLauncher has been
created.
"""

import isaaclab.envs.mdp as isaaclab_mdp
import isaaclab.sim as sim_utils
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.manipulation.lift import mdp

from . import mdp as ar4_mdp
from .pickplace_env_cfg import _EE_OFFSET
from .pickplace_mirror_env_cfg import Ar4PickPlaceMirrorSceneCfg
from .robot_cfg import ARM_JOINT_NAMES, GRIPPER_CLOSED_POS, GRIPPER_JOINT_NAMES, GRIPPER_OPEN_POS

# Same values as pickplace_ik_guided_env_cfg.py's EventCfg reuse.
_LIFT_MINIMAL_HEIGHT = 0.03
_PREGRASP_HOVER = 0.05
_LIFT_MARGIN = 0.02
_CARRY_HEIGHT = 0.10


@configclass
class ActionsCfg:
    """Task-space action specifications: the arm is controlled via
    incremental Cartesian position commands converted to joint targets by
    a live differential-IK solver (Isaac Lab's built-in
    DifferentialInverseKinematicsActionCfg), rather than direct
    joint-position targets (contrast pickplace_mirror_env_cfg.py's
    ActionsCfg, used by every prior experiment this session).

    - command_type="position": 3D Cartesian only, not full 6-DOF pose -
      orientation isn't critical for this task's fixed top-down approach
      geometry, keeping the action space small like every prior experiment.
    - controller.use_relative_mode=True: DifferentialInverseKinematicsAction
      recomputes the current end-effector pose every step
      (process_actions -> self._compute_frame_pose()) and passes it to
      the controller's set_command() alongside the scaled raw action; with
      use_relative_mode=True the controller adds the scaled action to that
      CURRENT pose each step, i.e. the policy outputs incremental Cartesian
      deltas, not absolute positions - confirmed via
      isaaclab/envs/mdp/actions/task_space_actions.py's process_actions
      and isaaclab/controllers/differential_ik.py's set_command. There is
      only one use_relative_mode flag in this action term and it lives on
      the controller cfg (DifferentialInverseKinematicsActionCfg itself
      has no separate relative-mode field).
    - body_name="link_6", body_offset=OffsetCfg(pos=_EE_OFFSET): reuses the
      already-measured-and-verified 0.036m gripper-pinch-point offset (the
      same constant used throughout this session's ee_frame
      FrameTransformerCfg), so the controlled point is the actual gripper
      tip, not link_6 itself.
    - scale=0.05: 5cm maximum Cartesian step per unit of policy output,
      reasoned from the workspace's ~0.3-0.5m scale.
    - ik_method="dls": matches the damped-least-squares choice already
      validated in Experiments 8-10's reward-side IK usage
      (ik_guided_path_bonus).

    DifferentialInverseKinematicsAction (isaaclab/envs/mdp/actions/
    task_space_actions.py) already handles the fixed-base Jacobian
    indexing internally (self._asset.is_fixed_base branch) - the exact
    same logic ik_guided_path_bonus had to hand-implement for its
    reward-only IK usage. No new indexing code needed here.
    """

    arm_action = isaaclab_mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINT_NAMES,
        body_name="link_6",
        body_offset=isaaclab_mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=_EE_OFFSET),
        scale=0.05,
        controller=DifferentialIKControllerCfg(command_type="position", use_relative_mode=True, ik_method="dls"),
    )
    gripper_position = isaaclab_mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=GRIPPER_JOINT_NAMES,
        open_command_expr={name: GRIPPER_OPEN_POS for name in GRIPPER_JOINT_NAMES},
        close_command_expr={name: GRIPPER_CLOSED_POS for name in GRIPPER_JOINT_NAMES},
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP - identical to
    pickplace_ik_guided_env_cfg.py's ObservationsCfg (same scene, same
    goal mechanism); the action term's internals changed but these
    observation functions read joint/object state directly, not the
    action term, so they remain valid unmodified. last_action will now
    report the 3D Cartesian delta + 2D binary gripper command instead of
    the previous 6D joint-delta + 2D gripper command - a smaller
    observation vector, no code change needed since it's already generic."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        cube_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame, params={"object_cfg": SceneEntityCfg("cube")}
        )
        target_object_position = ObsTerm(
            func=ar4_mdp.mirrored_target_position_in_robot_root_frame,
            params={"robot_cfg": SceneEntityCfg("robot")},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset events, in registration order - identical to
    pickplace_ik_guided_env_cfg.py's EventCfg:
    1. reset_all - whole scene back to default.
    2. reset_cube_position - randomize the cube across the full workspace.
    3. randomize_goal - reads the cube's now-updated position, sets the mirrored goal.
    4. compute_path_waypoints - reads both the cube's position and the
       goal, computes the 5-waypoint path, and resets path-progress state
       (including env._ik_milestone_max, reused by path_proximity_bonus)."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_cube_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.30, 0.30), "y": (-0.175, 0.175), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube"),
        },
    )

    randomize_goal = EventTerm(
        func=ar4_mdp.set_mirrored_goal,
        mode="reset",
        params={
            "object_cfg": SceneEntityCfg("cube"),
            "goal_y_range": (0.10, 0.45),
            "goal_z_range": (0.0, 0.02),
        },
    )

    compute_path_waypoints = EventTerm(
        func=ar4_mdp.compute_path_waypoints,
        mode="reset",
        params={
            "object_cfg": SceneEntityCfg("cube"),
            "lift_minimal_height": _LIFT_MINIMAL_HEIGHT,
            "pregrasp_hover": _PREGRASP_HOVER,
            "lift_margin": _LIFT_MARGIN,
            "carry_height": _CARRY_HEIGHT,
        },
    )


@configclass
class TerminationsCfg:
    """Success (cube at the mirrored goal) ends the episode early;
    otherwise a fixed timeout."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    cube_reached_goal = DoneTerm(
        func=ar4_mdp.object_reached_mirrored_goal,
        params={"threshold": 0.02, "object_cfg": SceneEntityCfg("cube")},
    )


@configclass
class RewardsCfg:
    """path_proximity_bonus replaces ik_guided_path_bonus (drops the
    now-redundant IK-action-matching sub-signal, since IK is now part of
    the control loop itself - see path_proximity_bonus's docstring).
    antipodal_grasp_bonus (Experiment 10's physics-corrected -0.7071
    threshold), gripper_schedule_bonus, stillness_penalty, action_rate,
    and joint_vel all carry over unchanged from
    pickplace_ik_guided_env_cfg.py - this experiment isolates the
    action-space variable specifically."""

    path_proximity_bonus = RewTerm(
        func=ar4_mdp.path_proximity_bonus,
        weight=25.0,
        params={
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "proximity_std": 0.1,
            "advance_tolerance": 0.03,
        },
    )

    gripper_schedule_bonus = RewTerm(
        func=ar4_mdp.gripper_schedule_bonus,
        weight=0.1,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "gripper_joint_names": GRIPPER_JOINT_NAMES,
            "open_pos": GRIPPER_OPEN_POS,
            "closed_pos": GRIPPER_CLOSED_POS,
        },
    )

    antipodal_grasp_bonus = RewTerm(
        func=ar4_mdp.antipodal_grasp_bonus,
        weight=3.0,
        params={
            "force_threshold": 0.05,
            "antipodal_cos_threshold": -0.7071,
            "jaw1_contact_cfg": SceneEntityCfg("gripper_jaw1_contact"),
            "jaw2_contact_cfg": SceneEntityCfg("gripper_jaw2_contact"),
        },
    )

    stillness_penalty = RewTerm(
        func=ar4_mdp.stillness_penalty,
        weight=2.0,
        params={
            "object_cfg": SceneEntityCfg("cube"),
            "jaw1_contact_cfg": SceneEntityCfg("gripper_jaw1_contact"),
            "jaw2_contact_cfg": SceneEntityCfg("gripper_jaw2_contact"),
            "force_threshold": 0.05,
            "still_bound": 0.005,
            "patience_steps": 25,
        },
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class Ar4PickPlaceTaskspaceEnvCfg(ManagerBasedRLEnvCfg):
    """AR4 task-space IK-driven-action task: same scene/spawn/goal as the
    mirror and ik_guided tasks, but the arm's action space is Cartesian
    deltas converted to joint targets by a live differential-IK
    controller in the control loop, instead of direct joint-position
    deltas. num_envs=4096 default (a real training-scale run) -
    scripts/train.py's --num_envs flag overrides this per-run same as
    every other env cfg in this repo."""

    scene: Ar4PickPlaceMirrorSceneCfg = Ar4PickPlaceMirrorSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 5.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0)
        self.viewer.eye = (1.5, 1.5, 1.2)
        self.viewer.lookat = (0.0, 0.0, 0.4)
```

- [ ] **Step 2: Syntax-check the file**

Run: `python3 -c "import ast; ast.parse(open('tasks/ar4/pickplace_taskspace_env_cfg.py').read())"`
Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add tasks/ar4/pickplace_taskspace_env_cfg.py
git commit -m "Add Ar4PickPlaceTaskspaceEnvCfg: task-space IK-driven action (Experiment 11)"
```

---

### Task 3: Wire `--taskspace` into `scripts/train.py` and `scripts/eval_loop.py`, then smoke-test

**Files:**
- Modify: `scripts/train.py` (add `--taskspace` flag, import, and `env_cfg_cls` branch)
- Modify: `scripts/eval_loop.py` (same)

**Interfaces:**
- Consumes: `Ar4PickPlaceTaskspaceEnvCfg` (Task 2).
- Produces: `--taskspace` CLI flag on both scripts, verified via a headless 2-iteration smoke test writing a result file.

- [ ] **Step 1: Add the flag and import to `scripts/train.py`**

In `scripts/train.py`, after the existing `--ik_guided` `parser.add_argument(...)` block (ends at line 59, right before `AppLauncher.add_app_launcher_args(parser)` at line 60), insert:

```python
parser.add_argument(
    "--taskspace",
    action="store_true",
    default=False,
    help=(
        "Train on the task-space IK-driven-action variant of the mirror-goal scene: the arm's action "
        "is a Cartesian end-effector delta converted to joint targets by a live differential-IK "
        "controller in the control loop, instead of direct joint-position deltas. See "
        "docs/superpowers/specs/2026-07-06-ar4-taskspace-ik-action-design.md."
    ),
)
```

Then add the import next to the other `pickplace_*_env_cfg` imports (after line 91's `Ar4PickPlaceMirrorEnvCfg` import):

```python
from tasks.ar4.pickplace_taskspace_env_cfg import Ar4PickPlaceTaskspaceEnvCfg  # noqa: E402
```

Then change the `env_cfg_cls` selection (currently lines 98-105):

```python
    if args_cli.taskspace:
        env_cfg_cls = Ar4PickPlaceTaskspaceEnvCfg
    elif args_cli.ik_guided:
        env_cfg_cls = Ar4PickPlaceIkGuidedEnvCfg
    elif args_cli.mirror:
        env_cfg_cls = Ar4PickPlaceMirrorEnvCfg
    elif args_cli.perception:
        env_cfg_cls = Ar4PickPlaceSingleObjectEnvCfg
    else:
        env_cfg_cls = Ar4PickPlaceEnvCfg
```

- [ ] **Step 2: Add the flag and import to `scripts/eval_loop.py`**

After the existing `--ik_guided` block (ends at line 37, right before `AppLauncher.add_app_launcher_args(parser)` at line 38), insert:

```python
parser.add_argument(
    "--taskspace",
    action="store_true",
    default=False,
    help="Evaluate the task-space IK-driven-action scene (see scripts/train.py --taskspace) instead of the four-object scene.",
)
```

Add the import next to the other `pickplace_*_env_cfg` imports (after line 69's `Ar4PickPlaceMirrorEnvCfg` import):

```python
from tasks.ar4.pickplace_taskspace_env_cfg import Ar4PickPlaceTaskspaceEnvCfg  # noqa: E402
```

Change the `env_cfg_cls` selection (currently lines 75-82):

```python
    if args_cli.taskspace:
        env_cfg_cls = Ar4PickPlaceTaskspaceEnvCfg
    elif args_cli.ik_guided:
        env_cfg_cls = Ar4PickPlaceIkGuidedEnvCfg
    elif args_cli.mirror:
        env_cfg_cls = Ar4PickPlaceMirrorEnvCfg
    elif args_cli.perception:
        env_cfg_cls = Ar4PickPlacePerceptionEnvCfg
    else:
        env_cfg_cls = Ar4PickPlaceEnvCfg
```

And change the `name_prefix` selection (currently lines 105-110):

```python
        if args_cli.taskspace:
            name_prefix = "ar4_pickplace_taskspace"
        elif args_cli.ik_guided:
            name_prefix = "ar4_pickplace_ik_guided"
        elif args_cli.mirror:
            name_prefix = "ar4_pickplace_mirror"
        else:
            name_prefix = "ar4_pickplace"
```

- [ ] **Step 3: Syntax-check both files**

Run: `python3 -c "import ast; ast.parse(open('scripts/train.py').read()); ast.parse(open('scripts/eval_loop.py').read())"`
Expected: no output.

- [ ] **Step 4: Smoke test — 2-iteration headless training run**

Run (from repo root, foreground, allow up to 5 minutes):
```bash
rm -f /tmp/taskspace_smoke_stdout.log
timeout 280 /home/saps/IsaacLab/isaaclab.sh -p scripts/train.py --taskspace --num_envs 16 --max_iterations 2 --headless > /tmp/taskspace_smoke_stdout.log 2>&1
echo "exit_code=$?"
```

Expected: the command may hit the `timeout` (Isaac Sim's clean-shutdown hang is a known false-negative pattern per this session's established practice) — do NOT treat a nonzero/timeout exit code alone as failure. Instead verify via files:

```bash
ls -la logs/train/ | tail -5
```

Find the newest `logs/train/<timestamp>/` directory (created after this command started) and check:
```bash
find logs/train/<newest_timestamp_dir> -name "model_*.pt"
grep -o "path_proximity_bonus\|gripper_schedule_bonus\|antipodal_grasp_bonus\|stillness_penalty\|action_rate\|joint_vel" /tmp/taskspace_smoke_stdout.log | sort -u
```

Expected: `model_0.pt` and `model_1.pt` both exist (2-iteration smoke pattern, matching every prior experiment's smoke-test convention in this repo), and all 6 reward term names appear (confirming the `RewardsCfg` wired correctly with no exceptions during construction or the reward computation step). If any exception/traceback appears in `/tmp/taskspace_smoke_stdout.log`, that IS a real failure — read the traceback and fix the underlying config error before proceeding (most likely culprits: a typo in `body_name`, a mismatched `OffsetCfg` import, or `ARM_JOINT_NAMES` not matching the robot's actual joint names — cross-check against `tasks/ar4/robot_cfg.py` if this happens).

- [ ] **Step 5: Commit**

```bash
git add scripts/train.py scripts/eval_loop.py
git commit -m "Wire --taskspace flag into train.py and eval_loop.py for Experiment 11"
```

---

### Task 4: Full 1500-iteration training run + TensorBoard verification report

**Files:**
- Create: `docs/superpowers/plans/2026-07-06-ar4-experiment11-report.md` (training verification report, same format as `2026-07-06-ar4-experiment10-report.md`)

**Interfaces:**
- Consumes: the smoke-tested `--taskspace` flag (Task 3).
- Produces: a verified, complete 1500-iteration training run under `logs/train/`, and a written report with full scalar trajectories for every reward term plus `Episode_Termination/cube_reached_goal`.

- [ ] **Step 1: Launch the full training run**

Run (from repo root, background — this takes significantly longer than the smoke test):
```bash
nohup /home/saps/IsaacLab/isaaclab.sh -p scripts/train.py --taskspace --num_envs 4096 --headless > /tmp/exp11_train_stdout.log 2>&1 &
echo "PID=$!"
```

Do not trust stdout alone for completion — poll for the result files instead (see Step 2).

- [ ] **Step 2: Verify completion via files**

Poll until the run's log directory (`logs/train/<timestamp>/`, created at launch time) contains `model_1499.pt` (or the run's `save_interval`-aligned final checkpoint number — check `tasks/ar4/agents/rsl_rl_ppo_cfg.py`'s `Ar4PickPlacePPORunnerCfg.max_iterations` and `save_interval` if unsure, matching Experiment 10's `save_interval=50` convention):

```bash
find logs/train/<newest_timestamp_dir> -name "model_*.pt" | wc -l
find logs/train/<newest_timestamp_dir> -name "model_1499.pt"
ls -la logs/train/<newest_timestamp_dir>/events.out.tfevents.*
```

Expected: 31 checkpoints (0, 50, 100, ..., 1450, 1499 — same pattern as every prior experiment this session), `model_1499.pt` exists, and the event file's mtime matches the run's actual completion time (not stale from an earlier failed attempt).

- [ ] **Step 3: Extract full scalar trajectories**

For each of `Episode_Reward/path_proximity_bonus`, `Episode_Reward/gripper_schedule_bonus`, `Episode_Reward/antipodal_grasp_bonus`, `Episode_Reward/stillness_penalty`, `Episode_Termination/cube_reached_goal`, extract the full 1500-point trajectory from the TensorBoard event file (use the same extraction method as Experiment 10's report — e.g. `tensorboard.backend.event_processing.event_accumulator` or an existing extraction script in this repo if one exists; check `docs/superpowers/plans/2026-07-06-ar4-experiment10-report.md` for the exact method used there).

- [ ] **Step 4: Write the report**

Write `docs/superpowers/plans/2026-07-06-ar4-experiment11-report.md` following the same structure as `2026-07-06-ar4-experiment10-report.md` (checkpoint integrity checks, per-term summary + sampled trajectory, and — critically, learning from Experiment 10's caught error — compare **final-value-to-final-value only** against Experiment 10's own final values, never a cumulative-sum-across-iterations against a single-episode value):

- Experiment 10 final `antipodal_grasp_bonus`: 0.000000 (the baseline this experiment must beat)
- Experiment 10 final `Episode_Termination/cube_reached_goal`: 0.002848

State plainly whether `antipodal_grasp_bonus`'s final value is now nonzero (the specific regression Experiment 11 targets), and whether `cube_reached_goal` improved, without yet drawing a final success/failure conclusion — that requires Task 5's video inspection.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/plans/2026-07-06-ar4-experiment11-report.md
git commit -m "Record Experiment 11 training run: task-space IK-driven action scalar trajectories"
```

---

### Task 5: Real eval + video inspection, ROADMAP record

**Files:**
- Modify: `ROADMAP.md` (append Experiment 11's outcome, following the existing format for Experiments 8-10)

**Interfaces:**
- Consumes: `model_1499.pt` from Task 4's training run.

- [ ] **Step 1: Run eval with video recording**

```bash
/home/saps/IsaacLab/isaaclab.sh -p scripts/eval_loop.py --taskspace --checkpoint logs/train/<experiment11_run_dir>/model_1499.pt --episodes 10
```

Verify the output video file exists under `logs/videos/` (named `ar4_pickplace_taskspace...mp4` per Task 3's `name_prefix` wiring) before treating this as complete.

- [ ] **Step 2: Watch the video frame-by-frame**

Use the Read tool (or equivalent frame-extraction) to inspect the recorded episodes, not just file existence — per this session's established standard (two prior false positives came from trusting TensorBoard scalars or coarse sampling alone). Specifically look for: does the gripper close around the cube, does the cube leave the ground, is any lift held for more than an instantaneous contact, and does the arm's motion look qualitatively different (smoother/more direct Cartesian paths) compared to Experiment 8-10's joint-space-action videos.

- [ ] **Step 3: Record the outcome in `ROADMAP.md`**

Append a new entry after the Experiment 10 entry, in the same format (date, hypothesis, what changed, quantitative result from Task 4's report, qualitative video finding from Step 2, and — regardless of outcome — an explicit statement of whether the "grasp/lift never emerges" problem is now resolved, improved-but-unresolved, or unchanged/regressed).

- [ ] **Step 4: Commit and push**

```bash
git add ROADMAP.md
git commit -m "Record Experiment 11 outcome (task-space IK-driven action) in ROADMAP"
git push origin main
```
