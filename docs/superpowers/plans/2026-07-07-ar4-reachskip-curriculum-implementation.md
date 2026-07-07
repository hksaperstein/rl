# Experiment 14 Implementation Plan: Reach-Skip Curriculum

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Experiment 14 — add a one-shot IK reset event that teleports the arm to the pregrasp pose for each episode's randomized cube position, removing the reach sub-problem (already reliably solved across Experiments 11-13) so the full step budget goes toward the actually-unsolved grasp→lift→carry→place sequence.

**Architecture:** New `reset_arm_to_pregrasp_pose` `EventTerm` in `tasks/ar4/mdp.py` (one-shot `DifferentialIKController` solve, reusing `ik_guided_path_bonus`'s controller-construction/gripper-offset pattern), wired into a new `Ar4PickPlaceReachskipEnvCfg` (`tasks/ar4/pickplace_reachskip_env_cfg.py`) built on Experiment 12's clean, non-regressed baseline (plain task-space action, Experiment 12's exact reward weights) — isolates the one new variable (starting state) against the last known-good state, not Experiment 13's unresolved regression.

**Tech Stack:** Isaac Lab `ManagerBasedRLEnvCfg`, `isaaclab.controllers.DifferentialIKController` (one-shot, not per-step), `Articulation.write_joint_position_to_sim`/`set_joint_position_target`, rsl_rl PPO (`Ar4PickPlaceTaskspacePPORunnerCfg`, unchanged).

## Global Constraints

- Do not modify `tasks/ar4/pickplace_taskspace_env_cfg.py`, `tasks/ar4/pickplace_residual_env_cfg.py`, `tasks/ar4/residual_ik_action.py`, or any existing function in `tasks/ar4/mdp.py` — purely additive (one new function appended, one new env cfg file).
- Reward weights must be copied verbatim from `pickplace_taskspace_env_cfg.py`'s current `RewardsCfg`: `path_proximity_bonus` weight 25.0, `gripper_schedule_bonus` weight 0.1, `antipodal_grasp_bonus` weight 3.0 with `antipodal_cos_threshold=-0.7071`, `stillness_penalty` weight **5.0** (post-Experiment-12 fix, not the stale 2.0) with `patience_steps=25`/`still_bound=0.005`, `action_rate` weight -1e-4, `joint_vel` weight -1e-4.
- Action term must be the plain `isaaclab_mdp.DifferentialInverseKinematicsActionCfg` (Experiment 12's mechanism) — **not** `ResidualDifferentialIKActionCfg` (Experiment 13's unresolved regression).
- Must reuse `Ar4PickPlaceTaskspacePPORunnerCfg` (`clip_actions=5.0`) unchanged — no new PPORunnerCfg subclass.
- `_LIFT_MINIMAL_HEIGHT=0.03`, `_PREGRASP_HOVER=0.05`, `_LIFT_MARGIN=0.02`, `_CARRY_HEIGHT=0.10`, `advance_tolerance=0.03` — same values used throughout this repo's waypoint-planning code.
- Always invoke scripts via `/home/saps/IsaacLab/isaaclab.sh -p scripts/<script>.py` (or `/home/saps/IsaacLab/isaaclab.sh -p -c "<snippet>"` for one-off Python) from the repo root, never plain `python`.
- Verify via file evidence (checkpoints, TensorBoard event files, `params/env.yaml`) — this repo has no pytest-based unit tests for Isaac-Sim-dependent code.
- **Any subagent dispatched to launch or wait on a training run must be given the literal blocking poll command in its dispatch prompt** (not just told to "poll" in prose) — this exact mistake ("wait for a background notification" that never comes) has recurred multiple times this session even when explicitly warned in prose only.

---

### Task 1: `reset_arm_to_pregrasp_pose` event function

**Files:**
- Modify: `tasks/ar4/mdp.py` (append new function at end of file)

**Interfaces:**
- Consumes: `DifferentialIKController`/`DifferentialIKControllerCfg` (already imported at the top of `mdp.py`), `SceneEntityCfg`, `subtract_frame_transforms`/`quat_apply` (already imported), `Articulation.write_joint_position_to_sim`/`set_joint_position_target` (Isaac Lab, verified signatures: `write_joint_position_to_sim(position, joint_ids=..., env_ids=...)` and `set_joint_position_target(target, joint_ids=..., env_ids=...)`, both accept `env_ids` as an index sequence — a `torch.Tensor` works, matching this file's own existing `env_ids` usage pattern, e.g. `reset_stillness_buffers`).
- Produces: `reset_arm_to_pregrasp_pose(env, env_ids, object_cfg, robot_cfg, pregrasp_hover, gripper_tool_offset) -> None` — an `EventTerm` function, consumed by Task 2's `EventCfg`.

- [ ] **Step 1: Append the new function to `tasks/ar4/mdp.py`**

Add this function at the end of the file:

```python
def reset_arm_to_pregrasp_pose(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    object_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    pregrasp_hover: float,
    gripper_tool_offset: tuple[float, float, float],
) -> None:
    """Event term (mode="reset"): one-shot IK solve that teleports the
    arm's joints so the gripper's pinch point starts AT the pregrasp
    waypoint (this episode's randomized cube position + pregrasp_hover in
    z), instead of starting from a fixed home pose every episode. Must be
    registered AFTER the cube's position has been randomized (reads
    object.data.root_pos_w) and BEFORE compute_path_waypoints - the full
    5-waypoint path is still computed unchanged; since the arm now starts
    already at/near waypoint 0, path_proximity_bonus's own
    advance-tolerance check naturally credits it almost immediately, with
    no reward-function change needed.

    Reuses the same live-DifferentialIKController construction and
    gripper-tool-offset correction ik_guided_path_bonus already uses
    (same file, above), but as a single one-shot solve for only env_ids
    at reset time - NOT cached across calls, since env_ids' length varies
    between calls (all envs on the very first reset, a smaller subset on
    later per-env resets during training), and DifferentialIKController
    allocates internal buffers sized to whatever num_envs it's
    constructed with. Only the env-agnostic SceneEntityCfg/jacobian-index
    lookups are cached; the controller itself is constructed fresh each
    call, sized to len(env_ids). See
    docs/superpowers/specs/2026-07-07-ar4-reachskip-curriculum-design.md.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    if not hasattr(env, "_reachskip_robot_entity_cfg"):
        env._reachskip_robot_entity_cfg = SceneEntityCfg(
            robot_cfg.name, joint_names=robot_cfg.joint_names, body_names=["link_6"]
        )
        env._reachskip_robot_entity_cfg.resolve(env.scene)
        env._reachskip_jacobi_idx = (
            env._reachskip_robot_entity_cfg.body_ids[0] - 1
            if robot.is_fixed_base
            else env._reachskip_robot_entity_cfg.body_ids[0]
        )

    ik_cfg = DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls")
    ik_controller = DifferentialIKController(ik_cfg, num_envs=len(env_ids), device=env.device)

    object_pos_w = object.data.root_pos_w[env_ids]
    pregrasp_w = object_pos_w.clone()
    pregrasp_w[:, 2] += pregrasp_hover

    root_pose_w = robot.data.root_pose_w[env_ids]
    ee_pose_w = robot.data.body_pose_w[env_ids, env._reachskip_robot_entity_cfg.body_ids[0]]

    # Same gripper-tool-offset correction as ik_guided_path_bonus: the
    # waypoint targets the pinch point, but the IK controller/Jacobian
    # operate on the raw link_6 body - subtract the offset (rotated into
    # world frame by link_6's current orientation) before commanding IK.
    offset_vec = torch.tensor(gripper_tool_offset, device=env.device).expand(len(env_ids), 3)
    offset_w = quat_apply(ee_pose_w[:, 3:7], offset_vec)
    ik_target_w = pregrasp_w - offset_w
    waypoint_command_b, _ = subtract_frame_transforms(root_pose_w[:, 0:3], root_pose_w[:, 3:7], ik_target_w)

    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )

    jacobian_all = robot.root_physx_view.get_jacobians()[env_ids]
    jacobian = jacobian_all[:, env._reachskip_jacobi_idx, :, env._reachskip_robot_entity_cfg.joint_ids]
    joint_pos = robot.data.joint_pos[env_ids][:, env._reachskip_robot_entity_cfg.joint_ids]

    ik_controller.set_command(waypoint_command_b, ee_pos=ee_pos_b, ee_quat=ee_quat_b)
    joint_pos_des = ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

    # Teleport the actual physics state AND set the PD drive's target to
    # the same value, so the drive doesn't immediately fight to move away
    # from the teleported pose on the very first control step.
    robot.write_joint_position_to_sim(
        joint_pos_des, joint_ids=env._reachskip_robot_entity_cfg.joint_ids, env_ids=env_ids
    )
    robot.set_joint_position_target(
        joint_pos_des, joint_ids=env._reachskip_robot_entity_cfg.joint_ids, env_ids=env_ids
    )
```

No new imports needed: `DifferentialIKController`, `DifferentialIKControllerCfg`, `SceneEntityCfg`, `subtract_frame_transforms`, `quat_apply`, `torch`, `Articulation`, `RigidObject` are all already imported at the top of `tasks/ar4/mdp.py` (confirmed — the same imports `ik_guided_path_bonus` already uses).

- [ ] **Step 2: Syntax-check the file**

Run: `python3 -c "import ast; ast.parse(open('tasks/ar4/mdp.py').read())"`
Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add tasks/ar4/mdp.py
git commit -m "Add reset_arm_to_pregrasp_pose event function for Experiment 14 (reach-skip curriculum)"
```

---

### Task 2: New reach-skip env cfg file

**Files:**
- Create: `tasks/ar4/pickplace_reachskip_env_cfg.py`

**Interfaces:**
- Consumes: `Ar4PickPlaceMirrorSceneCfg` (from `tasks/ar4/pickplace_mirror_env_cfg.py`, unmodified), `ar4_mdp.reset_arm_to_pregrasp_pose` (Task 1), `ar4_mdp.path_proximity_bonus`/`gripper_schedule_bonus`/`antipodal_grasp_bonus`/`stillness_penalty`/`set_mirrored_goal`/`compute_path_waypoints`/`mirrored_target_position_in_robot_root_frame`/`object_reached_mirrored_goal` (all pre-existing, unmodified), `_EE_OFFSET` (from `pickplace_env_cfg.py`), `ARM_JOINT_NAMES`/`GRIPPER_JOINT_NAMES`/`GRIPPER_OPEN_POS`/`GRIPPER_CLOSED_POS` (from `robot_cfg.py`), `Ar4PickPlaceTaskspacePPORunnerCfg` (from `pickplace_taskspace_env_cfg.py`, unmodified, reused directly — not redefined).
- Produces: `Ar4PickPlaceReachskipEnvCfg` class — consumed by Task 3 (script wiring).

- [ ] **Step 1: Write the new file**

```python
# tasks/ar4/pickplace_reachskip_env_cfg.py
"""Reach-skip curriculum variant of the AR4 mirror-goal cube task
(Experiment 14): identical scene/action/reward as
pickplace_taskspace_env_cfg.py (Experiment 12's clean, non-regressed
baseline - NOT Experiment 13's unresolved residual-action regression),
but the arm starts each episode already at the pregrasp pose for that
episode's randomized cube position (via a one-shot IK reset event)
instead of a fixed home pose. Removes the reach sub-problem - already
reliably solved across Experiments 11-13 - so the full step budget goes
toward the sub-problem those experiments never solved:
grasp->lift->carry->place. See
docs/superpowers/specs/2026-07-07-ar4-reachskip-curriculum-design.md.

Additive/parallel to every other pickplace_*.py file: does NOT modify
env_cfg.py, objects_cfg.py, pickplace_env_cfg.py, pickplace_mirror_env_cfg.py,
pickplace_ik_guided_env_cfg.py, pickplace_taskspace_env_cfg.py, or
pickplace_residual_env_cfg.py. Reuses Ar4PickPlaceMirrorSceneCfg and
Ar4PickPlaceTaskspacePPORunnerCfg directly - only the EventCfg (new
reset_arm_to_pregrasp_pose step) differs from pickplace_taskspace_env_cfg.py.

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
from .pickplace_taskspace_env_cfg import Ar4PickPlaceTaskspacePPORunnerCfg
from .robot_cfg import ARM_JOINT_NAMES, GRIPPER_CLOSED_POS, GRIPPER_JOINT_NAMES, GRIPPER_OPEN_POS

# Same values as pickplace_taskspace_env_cfg.py's EventCfg reuse.
_LIFT_MINIMAL_HEIGHT = 0.03
_PREGRASP_HOVER = 0.05
_LIFT_MARGIN = 0.02
_CARRY_HEIGHT = 0.10


@configclass
class ActionsCfg:
    """Identical to pickplace_taskspace_env_cfg.py's ActionsCfg - the
    plain (non-residual) DifferentialInverseKinematicsActionCfg."""

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
    """Identical to pickplace_taskspace_env_cfg.py's ObservationsCfg."""

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
    """Reset events, in registration order - ONE new step relative to
    pickplace_taskspace_env_cfg.py's EventCfg:
    1. reset_all - whole scene back to default.
    2. reset_cube_position - randomize the cube across the full workspace.
    3. reset_arm_to_pregrasp_pose (NEW) - one-shot IK teleport of the arm
       to the pregrasp pose for THIS episode's now-randomized cube
       position. Must run after reset_cube_position (needs the cube's
       real position) and before compute_path_waypoints.
    4. randomize_goal - reads the cube's position, sets the mirrored goal.
    5. compute_path_waypoints - unchanged, still computes the full
       5-waypoint path; the arm now starts at/near waypoint 0."""

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

    reset_arm_to_pregrasp_pose = EventTerm(
        func=ar4_mdp.reset_arm_to_pregrasp_pose,
        mode="reset",
        params={
            "object_cfg": SceneEntityCfg("cube"),
            "robot_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES),
            "pregrasp_hover": _PREGRASP_HOVER,
            "gripper_tool_offset": _EE_OFFSET,
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
    """Identical weights to pickplace_taskspace_env_cfg.py's RewardsCfg
    (post-Experiment-12: stillness_penalty weight 5.0) - this experiment
    isolates the starting-state variable alone, reward function unchanged."""

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
        weight=5.0,
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
class Ar4PickPlaceReachskipEnvCfg(ManagerBasedRLEnvCfg):
    """AR4 reach-skip curriculum task: same scene/action/reward as
    Experiment 12's clean baseline, but the arm starts each episode
    already at the pregrasp pose for that episode's cube position,
    instead of a fixed home pose. num_envs=4096 default (a real
    training-scale run) - scripts/train.py's --num_envs flag overrides
    this per-run same as every other env cfg in this repo."""

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

Run: `python3 -c "import ast; ast.parse(open('tasks/ar4/pickplace_reachskip_env_cfg.py').read())"`
Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add tasks/ar4/pickplace_reachskip_env_cfg.py
git commit -m "Add Ar4PickPlaceReachskipEnvCfg: arm starts at pregrasp pose (Experiment 14)"
```

---

### Task 3: Wire `--reachskip` into `scripts/train.py` and `scripts/eval_loop.py`, then smoke-test

**Files:**
- Modify: `scripts/train.py`
- Modify: `scripts/eval_loop.py`

**Interfaces:**
- Consumes: `Ar4PickPlaceReachskipEnvCfg` (Task 2), `Ar4PickPlaceTaskspacePPORunnerCfg` (pre-existing, already imported by both scripts for `--taskspace`/`--residual`).
- Produces: `--reachskip` CLI flag on both scripts, verified via a headless 2-iteration smoke test.

- [ ] **Step 1: Add the flag and import to `scripts/train.py`**

Immediately after the existing `--residual` `parser.add_argument(...)` block (currently lines 71-81, ending right before `AppLauncher.add_app_launcher_args(parser)` at line 82), insert:

```python
parser.add_argument(
    "--reachskip",
    action="store_true",
    default=False,
    help=(
        "Train on the reach-skip curriculum variant of the task-space scene: the arm starts each "
        "episode already at the pregrasp pose for that episode's randomized cube position (via a "
        "one-shot IK reset), instead of a fixed home pose - removing the reach sub-problem so the "
        "full step budget goes toward grasp/lift/carry/place. See "
        "docs/superpowers/specs/2026-07-07-ar4-reachskip-curriculum-design.md."
    ),
)
```

Add the import next to the existing `pickplace_residual_env_cfg` import (currently line 115):

```python
from tasks.ar4.pickplace_reachskip_env_cfg import Ar4PickPlaceReachskipEnvCfg  # noqa: E402
```

Change the `env_cfg_cls` selection (currently lines 125-136):

```python
    if args_cli.reachskip:
        env_cfg_cls = Ar4PickPlaceReachskipEnvCfg
    elif args_cli.residual:
        env_cfg_cls = Ar4PickPlaceResidualEnvCfg
    elif args_cli.taskspace:
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

Change the `agent_cfg` selection (currently lines 141-144) so `--reachskip` reuses the same already-verified PPO runner cfg:

```python
    if args_cli.taskspace or args_cli.residual or args_cli.reachskip:
        agent_cfg = Ar4PickPlaceTaskspacePPORunnerCfg()
    else:
        agent_cfg = Ar4PickPlacePPORunnerCfg()
```

- [ ] **Step 2: Add the flag and import to `scripts/eval_loop.py`**

Immediately after the existing `--residual` `parser.add_argument(...)` block (currently lines 44-49, ending right before `AppLauncher.add_app_launcher_args(parser)` at line 50), insert:

```python
parser.add_argument(
    "--reachskip",
    action="store_true",
    default=False,
    help="Evaluate the reach-skip curriculum scene (see scripts/train.py --reachskip) instead of the four-object scene.",
)
```

Add the import next to the existing `pickplace_residual_env_cfg` import (currently line 82):

```python
from tasks.ar4.pickplace_reachskip_env_cfg import Ar4PickPlaceReachskipEnvCfg  # noqa: E402
```

Change the `env_cfg_cls` selection (currently lines 92-103):

```python
    if args_cli.reachskip:
        env_cfg_cls = Ar4PickPlaceReachskipEnvCfg
    elif args_cli.residual:
        env_cfg_cls = Ar4PickPlaceResidualEnvCfg
    elif args_cli.taskspace:
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

Change the `agent_cfg` selection (currently lines 108-111):

```python
    if args_cli.taskspace or args_cli.residual or args_cli.reachskip:
        agent_cfg = Ar4PickPlaceTaskspacePPORunnerCfg()
    else:
        agent_cfg = Ar4PickPlacePPORunnerCfg()
```

And change the `name_prefix` selection (currently lines 129-138):

```python
        if args_cli.reachskip:
            name_prefix = "ar4_pickplace_reachskip"
        elif args_cli.residual:
            name_prefix = "ar4_pickplace_residual"
        elif args_cli.taskspace:
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

This is the FIRST time `reset_arm_to_pregrasp_pose`/`Ar4PickPlaceReachskipEnvCfg` will actually run inside Isaac Sim — Tasks 1-2 only had syntax checks. If it throws an exception, that's real information (a bug pure syntax-checking couldn't catch, e.g. a Jacobian/tensor-shape mismatch in the one-shot IK solve), not something to route around — read the traceback and fix it.

Run (from repo root, foreground, allow up to 5 minutes):
```bash
rm -f /tmp/reachskip_smoke_stdout.log
timeout 280 /home/saps/IsaacLab/isaaclab.sh -p scripts/train.py --reachskip --num_envs 16 --max_iterations 2 --headless > /tmp/reachskip_smoke_stdout.log 2>&1
echo "exit_code=$?"
```

A `timeout`/nonzero exit code alone is NOT proof of failure (Isaac Sim's shutdown sequence sometimes hangs after genuinely finishing) — verify via files:

```bash
grep -i "error\|exception\|traceback" /tmp/reachskip_smoke_stdout.log
ls -la logs/train/ | tail -5
```

Find the newest `logs/train/<timestamp>/` directory and check:
```bash
find logs/train/<newest_timestamp_dir> -name "model_*.pt"
cat logs/train/<newest_timestamp_dir>/params/env.yaml | grep -A3 "arm_action\|reset_arm_to_pregrasp"
```

Expected: `model_0.pt` and `model_1.pt` both exist, `env.yaml` confirms `Ar4PickPlaceReachskipEnvCfg`'s events (including `reset_arm_to_pregrasp_pose`) are present, and no traceback in the stdout log. If an exception appears, the most likely culprits given this task's new code are: (a) a tensor shape mismatch in the Jacobian indexing (`jacobian_all[:, env._reachskip_jacobi_idx, :, ...]` — verify `env._reachskip_jacobi_idx` is a plain int, not a tensor, matching `ik_guided_path_bonus`'s identical pattern); (b) `DifferentialIKController.compute()` rejecting a batch size that doesn't match how it was constructed (this plan constructs it fresh each call with `num_envs=len(env_ids)` specifically to avoid this — if it still errors, the constructor or `compute()` may expect something else, worth reading `isaaclab/controllers/differential_ik.py` directly to confirm); (c) `write_joint_position_to_sim`/`set_joint_position_target` rejecting `env_ids` as a `torch.Tensor` (unlikely, given `reset_stillness_buffers` and other existing event functions already index tensors with `env_ids` directly, but confirm if this specific write-to-sim call behaves differently).

- [ ] **Step 5: Commit**

```bash
git add scripts/train.py scripts/eval_loop.py
git commit -m "Wire --reachskip flag into train.py and eval_loop.py for Experiment 14"
```

---

### Task 4: Diagnostic run (300 iterations) — verify the new reset event is stable before committing to a full run

**Files:**
- None created — this is a verification-only task. If the diagnostic looks wrong, stop and report; do not proceed to Task 5.

**Interfaces:**
- Consumes: `Ar4PickPlaceReachskipEnvCfg` (Task 2) via the `--reachskip` flag (Task 3).
- Produces: a confirmed-healthy diagnostic run directory under `logs/train/`, gating Task 5.

- [ ] **Step 1: Launch the diagnostic run**

```bash
nohup /home/saps/IsaacLab/isaaclab.sh -p scripts/train.py --reachskip --num_envs 4096 --max_iterations 300 --headless > /tmp/exp14_diagnostic_stdout.log 2>&1 &
echo "PID=$!"
```

- [ ] **Step 2: Poll for completion — BLOCK on this yourself, do not end your turn early**

Run this exact command (a real blocking loop — if one call's timeout is hit before the run finishes, re-issue the same command again):
```bash
until find logs/train/ -name "model_299.pt" -newer tasks/ar4/pickplace_reachskip_env_cfg.py 2>/dev/null | grep -q .; do sleep 30; done
echo "diagnostic complete"
```
Based on this repo's prior 300-iteration diagnostic runs, expect roughly 3-5 minutes of real wall-clock time — do not assume failure before at least 10-15 minutes have elapsed.

- [ ] **Step 3: Extract and check the diagnostic scalars**

```bash
LATEST=$(ls -dt logs/train/*/ | head -1)
/home/saps/IsaacLab/isaaclab.sh -p -c "
from tensorboard.backend.event_processing import event_accumulator
import glob
path = sorted(glob.glob('${LATEST}events.out.tfevents.*'))[-1]
ea = event_accumulator.EventAccumulator(path)
ea.Reload()
for tag in ['Loss/value_function', 'Episode_Reward/antipodal_grasp_bonus', 'Episode_Reward/stillness_penalty',
            'Episode_Reward/path_proximity_bonus', 'Episode_Termination/cube_reached_goal']:
    if tag in ea.Tags()['scalars']:
        vals = ea.Scalars(tag)
        nonzero = sum(1 for v in vals if v.value != 0.0)
        print(tag, '-> first:', vals[0].value, 'last:', vals[-1].value, 'max:', max(v.value for v in vals),
              'min:', min(v.value for v in vals), 'nonzero:', nonzero, '/', len(vals))
    else:
        print(tag, '-> NOT FOUND')
"
```

- [ ] **Step 4: Evaluate the diagnostic against these two gate checks**

1. **`Loss/value_function` stays bounded (no sustained exponential growth).** This new reset event touches joint state directly via `write_joint_position_to_sim` before the first control step — new surface area, worth the same scrutiny prior new mechanisms got in Experiments 11 and 13 (both of which hit real critic-divergence bugs the first time a new state-manipulating mechanism was introduced). A small transient spike with immediate recovery is fine (matching Experiment 11/12/13's own precedent); a sustained climb is not.
2. **No exceptions/tracebacks in `/tmp/exp14_diagnostic_stdout.log`.**

If both checks pass, proceed to Task 5. If either fails, stop, do not proceed, and report the finding instead — this would itself be a notable result (the one-shot IK reset producing a bad initial state that destabilizes training), not just a blocker to silently patch around.

---

### Task 5: Full 1500-iteration training run + TensorBoard verification report

**Files:**
- Create: `docs/superpowers/plans/2026-07-07-ar4-experiment14-report.md`

**Interfaces:**
- Consumes: the Task 4-verified reset event.
- Produces: a complete 1500-iteration training run under `logs/train/`, and a written report with full scalar trajectories.

- [ ] **Step 1: Launch the full training run**

```bash
nohup /home/saps/IsaacLab/isaaclab.sh -p scripts/train.py --reachskip --num_envs 4096 --headless > /tmp/exp14_train_stdout.log 2>&1 &
echo "PID=$!"
```

- [ ] **Step 2: Poll for completion — BLOCK on this yourself, do not end your turn early**

```bash
until find logs/train/ -name "model_1499.pt" -newer tasks/ar4/pickplace_reachskip_env_cfg.py 2>/dev/null | grep -q .; do sleep 60; done
echo "full run complete"
```
Expect roughly 15-20 minutes of real wall-clock time based on this repo's prior 1500-iteration runs at `num_envs=4096` — treat that as a rough guide, not a hard cutoff; keep re-issuing the blocking command if a single tool call's own timeout is hit before the checkpoint appears.

Once found, confirm checkpoint integrity:
```bash
LATEST=$(ls -dt logs/train/*/ | head -1)
find "$LATEST" -name "model_*.pt" | wc -l
ls -la "${LATEST}"events.out.tfevents.*
```
Expected: 31 checkpoints (0, 50, 100, ..., 1450, 1499 — `save_interval=50`), `model_1499.pt` exists, event file mtime matches the run's actual completion time.

- [ ] **Step 3: Extract full scalar trajectories**

```bash
LATEST=$(ls -dt logs/train/*/ | head -1)
/home/saps/IsaacLab/isaaclab.sh -p -c "
from tensorboard.backend.event_processing import event_accumulator
import glob
path = sorted(glob.glob('${LATEST}events.out.tfevents.*'))[-1]
ea = event_accumulator.EventAccumulator(path)
ea.Reload()
tags = ['Episode_Reward/path_proximity_bonus', 'Episode_Reward/gripper_schedule_bonus',
        'Episode_Reward/antipodal_grasp_bonus', 'Episode_Reward/stillness_penalty',
        'Episode_Termination/cube_reached_goal', 'Loss/value_function']
for tag in tags:
    if tag in ea.Tags()['scalars']:
        vals = ea.Scalars(tag)
        nonzero = sum(1 for v in vals if v.value != 0.0)
        print(f'=== {tag} ===')
        print('  points:', len(vals), 'first:', vals[0].value, 'last:', vals[-1].value,
              'max:', max(v.value for v in vals), 'min:', min(v.value for v in vals),
              'nonzero:', nonzero, '/', len(vals))
        for i in range(0, len(vals), 150):
            print(f'  iteration={vals[i].step:4d}, value={vals[i].value:.6f}')
        print(f'  iteration={vals[-1].step:4d}, value={vals[-1].value:.6f}')
    else:
        print(tag, '-> NOT FOUND')
"
```

- [ ] **Step 4: Write the report**

Write `docs/superpowers/plans/2026-07-07-ar4-experiment14-report.md` following the structure of `docs/superpowers/plans/2026-07-07-ar4-experiment13-report.md` (checkpoint integrity, `Loss/value_function` sanity check, per-term summary + sampled trajectory for all 5 tags above). Include a "Key Comparison" section against **Experiment 12's exact final values** (final-iteration snapshot vs. final-iteration snapshot only, per this project's established correction protocol):

- Experiment 12 final `Episode_Reward/antipodal_grasp_bonus`: 0.012777
- Experiment 12 final `Episode_Reward/stillness_penalty`: -0.001857
- Experiment 12 final `Episode_Reward/path_proximity_bonus`: 0.064421
- Experiment 12 final `Episode_Termination/cube_reached_goal`: 0.010773

State the scalar comparison factually. **Do not draw a final success/failure conclusion from scalars alone** — per this project's own established lesson (Experiment 12's original report misread a scalar drop as failure and had to be corrected by the controller). That conclusion requires Task 6's video inspection.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/plans/2026-07-07-ar4-experiment14-report.md
git commit -m "Record Experiment 14 training run: reach-skip curriculum scalar trajectories"
```

---

### Task 6: Real eval + multi-episode video frame extraction (mechanical only), ROADMAP record

**Files:**
- Modify: `ROADMAP.md` (append Experiment 14's outcome — this step is done by the controller personally, not this task's dispatched subagent, see below)

**Interfaces:**
- Consumes: `model_1499.pt` from Task 5's training run.

This task is split: the dispatched subagent does ONLY the mechanical eval + frame extraction. The actual visual judgment and the ROADMAP entry are written by the controller (Principal) personally afterward, matching this session's established pattern for decisive evidence (Experiments 11-13 were all personally video-reviewed by the controller, not delegated to a subagent's description of what it saw).

- [ ] **Step 1: Run eval with video recording**

```bash
LATEST=$(ls -dt logs/train/*/ | head -1)
/home/saps/IsaacLab/isaaclab.sh -p scripts/eval_loop.py --reachskip --checkpoint "${LATEST}model_1499.pt" --episodes 10
```

Verify 10 output video files exist under `logs/videos/` (named `ar4_pickplace_reachskip-step-*.mp4`, one per episode).

- [ ] **Step 2: Extract frames from at least 3 of the 10 episodes**

Per the established lesson (a single episode is not a representative sample given this task's low success rate):

```bash
mkdir -p /tmp/exp14_eval_frames_ep1 /tmp/exp14_eval_frames_ep2 /tmp/exp14_eval_frames_ep3
ffmpeg -y -i logs/videos/ar4_pickplace_reachskip-step-0.mp4 -vf fps=5 /tmp/exp14_eval_frames_ep1/frame_%03d.png
ffmpeg -y -i logs/videos/ar4_pickplace_reachskip-step-250.mp4 -vf fps=5 /tmp/exp14_eval_frames_ep2/frame_%03d.png
ffmpeg -y -i logs/videos/ar4_pickplace_reachskip-step-500.mp4 -vf fps=5 /tmp/exp14_eval_frames_ep3/frame_%03d.png
```

List the extracted frame files (`ls -la /tmp/exp14_eval_frames_ep1/ /tmp/exp14_eval_frames_ep2/ /tmp/exp14_eval_frames_ep3/`) and confirm each has ~25 frames covering the full ~5s episode. Do NOT describe what the frames show or draw any conclusion — that's the controller's job in Step 3, done outside this dispatched task.

- [ ] **Step 3 (controller, not the dispatched subagent): personally inspect the frames, write the ROADMAP entry**

Following the format used for Experiments 9-13 (hypothesis, what changed, quantitative result from Task 5's report, qualitative multi-episode video finding, and an explicit statement of whether "grasp/lift never emerges" / "pick up and move" is resolved, improved-but-unresolved, unchanged, or regressed — plus, per the success criteria in the design spec, an explicit read on whether the policy reaches waypoint index ≥2 (lift) more than prior experiments, since that's the specific bar this experiment targets).

- [ ] **Step 4: Commit and push**

```bash
git add ROADMAP.md
git commit -m "Record Experiment 14 outcome (reach-skip curriculum) in ROADMAP"
git push origin main
```
