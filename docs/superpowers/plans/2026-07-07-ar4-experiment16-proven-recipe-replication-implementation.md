# Experiment 16 Implementation Plan: Proven Recipe Replication

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Experiment 16 — replace this repo's entire reward/action design with a from-scratch replication of two independently-proven Isaac-ecosystem manipulation recipes (Isaac Lab's own Franka Cube Lift task, IsaacGymEnvs' FrankaCubeStack), testing whether removing the standalone grasp reward and gating goal-tracking reward on lift produces genuine lift+carry where 15 prior ad hoc iterations did not.

**Architecture:** New `mirrored_goal_distance_gated` function in `tasks/ar4/mdp.py` (a direct adaptation of the reference's `object_goal_distance` formula to this repo's mirrored-goal buffer), new `Ar4PickPlaceProvenRecipeEnvCfg` (`tasks/ar4/pickplace_provenrecipe_env_cfg.py`) reusing the reference's own `object_ee_distance`/`object_is_lifted` functions directly off the shelf, plain joint-space action (matching both references, not this repo's task-space/IK lineage), and Isaac Lab's curriculum manager (new to this repo) replicating the reference's regularization-weight curriculum.

**Tech Stack:** Isaac Lab `ManagerBasedRLEnvCfg`, `CurriculumTermCfg`/`isaaclab.envs.mdp.modify_reward_weight` (new to this repo), `isaaclab_tasks.manager_based.manipulation.lift.mdp.object_ee_distance`/`object_is_lifted` (reused directly, not reimplemented), rsl_rl PPO (`Ar4PickPlacePPORunnerCfg`, unchanged, no task-space `clip_actions` override).

## Global Constraints

- Do not modify `tasks/ar4/pickplace_taskspace_env_cfg.py`, `pickplace_residual_env_cfg.py`, `pickplace_reachskip_env_cfg.py`, `pickplace_baseproximity_env_cfg.py`, `pickplace_mirror_env_cfg.py`, or any existing function in `tasks/ar4/mdp.py` — purely additive (one new function, one new env cfg file).
- Reward terms, exact values (from the design spec): `reaching_object` (reused `mdp.object_ee_distance`, weight 1.0, `std=0.1`), `lifting_object` (reused `mdp.object_is_lifted`, weight 15.0, `minimal_height=0.03`), `object_goal_tracking` (new `ar4_mdp.mirrored_goal_distance_gated`, weight 16.0, `std=0.3`, `minimal_height=0.03`), `object_goal_tracking_fine_grained` (same function, weight 5.0, `std=0.05`, `minimal_height=0.03`), `action_rate` (weight -1e-4), `joint_vel` (weight -1e-4). **No** `antipodal_grasp_bonus`/`stillness_penalty`/`ground_penalty`/`base_proximity_penalty`/`gripper_schedule_bonus`/`path_proximity_bonus` — none of these exist in this design.
- Action space: plain `isaaclab_mdp.JointPositionActionCfg(asset_name="robot", joint_names=ARM_JOINT_NAMES, scale=0.5)` — **not** task-space/IK. PPO runner cfg: plain `Ar4PickPlacePPORunnerCfg` — **not** `Ar4PickPlaceTaskspacePPORunnerCfg` (no `clip_actions=5.0` override; that override exists specifically for the task-space/IK action term this experiment doesn't use).
- Curriculum (new mechanism): `CurriculumTermCfg` via `mdp.modify_reward_weight` (already available through this repo's existing `from isaaclab_tasks.manager_based.manipulation.lift import mdp` import — that module re-exports `isaaclab.envs.mdp`'s contents via `from isaaclab.envs.mdp import *`, confirmed by reading `isaaclab_tasks/manager_based/manipulation/lift/mdp/__init__.py` directly), bumping `action_rate`/`joint_vel` from -1e-4 to -1e-1 after `num_steps=10000`. `ManagerBasedRLEnvCfg`'s base class field is `curriculum: object | None = None` (confirmed by reading `isaaclab/envs/manager_based_rl_env_cfg.py` directly) — any `@configclass`-decorated instance works as an override, same pattern as `rewards`/`observations`/etc.
- Events: `reset_all`, `reset_cube_position` (same `pose_range` as every prior experiment), `set_mirrored_goal` (reused). **No** `compute_path_waypoints` — there is no waypoint system in this design.
- Episode length/sim settings: `episode_length_s=5.0`, `decimation=2`, `sim.dt=0.01` — unchanged from every prior experiment.
- Always invoke scripts via `/home/saps/IsaacLab/isaaclab.sh -p scripts/<script>.py` (or `/home/saps/IsaacLab/isaaclab.sh -p -c "<snippet>"` for one-off Python) from the repo root, never plain `python`.
- Verify via file evidence (checkpoints, TensorBoard event files, `params/env.yaml`) — this repo has no pytest-based unit tests for Isaac-Sim-dependent code.
- **Any subagent dispatched to launch or wait on a training run must be given the literal blocking poll command in its dispatch prompt** (not just told to "poll" in prose) — this exact mistake has recurred multiple times this session even when warned only in prose.

---

### Task 1: `mirrored_goal_distance_gated` reward function

**Files:**
- Modify: `tasks/ar4/mdp.py` (append new function at end of file — current file ends at line 797, after `reset_arm_to_pregrasp_pose`)

**Interfaces:**
- Consumes: `SceneEntityCfg`, `torch` (already imported at the top of `mdp.py`), `RigidObject` (already imported under `TYPE_CHECKING`), `env._target_pos_w` (existing stateful buffer, lazily initialized by `set_mirrored_goal`, already read by `mirrored_target_position_in_robot_root_frame`/`object_reached_mirrored_goal` in this same file — reuse the identical lazy-init guard pattern).
- Produces: `mirrored_goal_distance_gated(env, std, minimal_height, object_cfg) -> torch.Tensor` — a `RewardTermCfg` function, consumed by Task 2's `RewardsCfg` (registered twice, at two different `std`/weight values).

- [ ] **Step 1: Append the new function to `tasks/ar4/mdp.py`**

Add this function at the end of the file:

```python
def mirrored_goal_distance_gated(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Direct adaptation of isaaclab_tasks.manager_based.manipulation.lift.mdp.object_goal_distance's
    exact tanh-kernel-distance-gated-on-lift formula to this repo's
    mirrored-goal buffer (env._target_pos_w, already world-frame, set by
    set_mirrored_goal) instead of the command manager - see
    docs/superpowers/specs/2026-07-07-ar4-experiment16-proven-recipe-replication-design.md
    for why the command manager can't be used here (this repo's goal is a
    function of the object's own random spawn) and why this is otherwise
    an unmodified replication of the reference formula, not a new design.
    """
    object: RigidObject = env.scene[object_cfg.name]
    if not hasattr(env, "_target_pos_w"):
        env._target_pos_w = torch.zeros(env.num_envs, 3, device=env.device)
    distance = torch.norm(env._target_pos_w - object.data.root_pos_w, dim=-1)
    lifted = (object.data.root_pos_w[:, 2] > minimal_height).float()
    return lifted * (1.0 - torch.tanh(distance / std))
```

No new imports needed: `SceneEntityCfg`, `torch`, `RigidObject`, `ManagerBasedRLEnv` are all already imported/type-checked at the top of `tasks/ar4/mdp.py`.

- [ ] **Step 2: Syntax-check the file**

Run: `python3 -c "import ast; ast.parse(open('tasks/ar4/mdp.py').read())"`
Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add tasks/ar4/mdp.py
git commit -m "Add mirrored_goal_distance_gated reward function for Experiment 16"
```

---

### Task 2: New proven-recipe env cfg file

**Files:**
- Create: `tasks/ar4/pickplace_provenrecipe_env_cfg.py`

**Interfaces:**
- Consumes: `Ar4PickPlaceMirrorSceneCfg` (from `tasks/ar4/pickplace_mirror_env_cfg.py`, unmodified), `ar4_mdp.mirrored_goal_distance_gated` (Task 1), `ar4_mdp.set_mirrored_goal`/`mirrored_target_position_in_robot_root_frame`/`object_reached_mirrored_goal` (all pre-existing, unmodified), `mdp.object_ee_distance`/`object_is_lifted`/`modify_reward_weight` (from `isaaclab_tasks.manager_based.manipulation.lift.mdp`, reused directly — not reimplemented), `ARM_JOINT_NAMES`/`GRIPPER_JOINT_NAMES`/`GRIPPER_OPEN_POS`/`GRIPPER_CLOSED_POS` (from `robot_cfg.py`), `Ar4PickPlacePPORunnerCfg` (from `tasks/ar4/agents/rsl_rl_ppo_cfg.py`, unmodified, reused directly — not redefined).
- Produces: `Ar4PickPlaceProvenRecipeEnvCfg` class — consumed by Task 3 (script wiring).

- [ ] **Step 1: Write the new file**

```python
# tasks/ar4/pickplace_provenrecipe_env_cfg.py
"""From-scratch replication of two proven, independently-published
Isaac-ecosystem manipulation reward recipes (Experiment 16): Isaac Lab's
own Franka Cube Lift task
(isaaclab_tasks/manager_based/manipulation/lift/lift_env_cfg.py,
mdp/rewards.py, config/franka/joint_pos_env_cfg.py) and IsaacGymEnvs'
FrankaCubeStack task (isaacgymenvs/tasks/franka_cube_stack.py) - both read
directly from source. Unlike every prior experiment this session, this
env cfg has NO standalone grasp-quality reward (no antipodal/contact
sensing), a PLAIN BINARY per-step lift reward (not a milestone/running-max
bonus), goal-tracking reward MULTIPLICATIVELY GATED on the lift condition,
and plain joint-space action (not task-space/IK) - see
docs/superpowers/specs/2026-07-07-ar4-experiment16-proven-recipe-replication-design.md
for the full hypothesis and cited research.

Additive/parallel to every other pickplace_*.py file: does NOT modify
env_cfg.py, objects_cfg.py, pickplace_mirror_env_cfg.py,
pickplace_taskspace_env_cfg.py, pickplace_residual_env_cfg.py,
pickplace_reachskip_env_cfg.py, or pickplace_baseproximity_env_cfg.py.
Reuses Ar4PickPlaceMirrorSceneCfg and Ar4PickPlacePPORunnerCfg directly.

Import this module only after an Isaac Sim/Isaac Lab AppLauncher has been
created.
"""

import isaaclab.envs.mdp as isaaclab_mdp
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.manipulation.lift import mdp

from . import mdp as ar4_mdp
from .pickplace_mirror_env_cfg import Ar4PickPlaceMirrorSceneCfg
from .agents.rsl_rl_ppo_cfg import Ar4PickPlacePPORunnerCfg
from .robot_cfg import ARM_JOINT_NAMES, GRIPPER_CLOSED_POS, GRIPPER_JOINT_NAMES, GRIPPER_OPEN_POS


@configclass
class ActionsCfg:
    """Plain joint-space action, matching both proven references exactly
    (neither uses task-space/IK-driven control). scale=0.5 matches
    pickplace_mirror_env_cfg.py's own ActionsCfg, which already cites the
    same Franka lift-task precedent for this value."""

    joint_positions = isaaclab_mdp.JointPositionActionCfg(asset_name="robot", joint_names=ARM_JOINT_NAMES, scale=0.5)
    gripper_position = isaaclab_mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=GRIPPER_JOINT_NAMES,
        open_command_expr={name: GRIPPER_OPEN_POS for name in GRIPPER_JOINT_NAMES},
        close_command_expr={name: GRIPPER_CLOSED_POS for name in GRIPPER_JOINT_NAMES},
    )


@configclass
class ObservationsCfg:
    """Identical structure to every prior experiment's ObservationsCfg -
    this axis was already well-aligned with the proven references, no
    change needed."""

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
    """Reset events, in registration order - deliberately simpler than
    every prior experiment's EventCfg: NO compute_path_waypoints, since
    this design has no waypoint/milestone system.
    1. reset_all - whole scene back to default.
    2. reset_cube_position - randomize the cube across the full workspace.
    3. randomize_goal - reads the cube's now-updated position, sets the
       mirrored goal into env._target_pos_w."""

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


@configclass
class TerminationsCfg:
    """Success (cube at the mirrored goal) ends the episode early;
    otherwise a fixed timeout - this repo's own established success
    definition, kept since the proven references' tabletop-drop
    termination doesn't apply to this repo's ground-level scene."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    cube_reached_goal = DoneTerm(
        func=ar4_mdp.object_reached_mirrored_goal,
        params={"threshold": 0.02, "object_cfg": SceneEntityCfg("cube")},
    )


@configclass
class RewardsCfg:
    """Six terms, matching the proven references' count and structure
    exactly. reaching_object/lifting_object are REUSED DIRECTLY from
    isaaclab_tasks.manager_based.manipulation.lift.mdp - not
    reimplemented, since both are already fully generic (parameterized by
    object_cfg/ee_frame_cfg, no Franka-specific assumptions).
    object_goal_tracking/object_goal_tracking_fine_grained use the new
    mirrored_goal_distance_gated (Task 1) - the SAME gating formula as
    the reference's object_goal_distance, adapted only to this repo's
    goal-storage mechanism. NO standalone grasp-quality reward (no
    antipodal/contact-force term) - grasp is purely instrumental, matching
    both references. See
    docs/superpowers/specs/2026-07-07-ar4-experiment16-proven-recipe-replication-design.md."""

    reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        weight=1.0,
        params={"std": 0.1, "object_cfg": SceneEntityCfg("cube"), "ee_frame_cfg": SceneEntityCfg("ee_frame")},
    )

    lifting_object = RewTerm(
        func=mdp.object_is_lifted,
        weight=15.0,
        params={"minimal_height": 0.03, "object_cfg": SceneEntityCfg("cube")},
    )

    object_goal_tracking = RewTerm(
        func=ar4_mdp.mirrored_goal_distance_gated,
        weight=16.0,
        params={"std": 0.3, "minimal_height": 0.03, "object_cfg": SceneEntityCfg("cube")},
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=ar4_mdp.mirrored_goal_distance_gated,
        weight=5.0,
        params={"std": 0.05, "minimal_height": 0.03, "object_cfg": SceneEntityCfg("cube")},
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class CurriculumCfg:
    """New to this repo: Isaac Lab's curriculum manager, replicating the
    proven reference's regularization-weight curriculum exactly
    (mdp.modify_reward_weight is a framework-provided function, already
    available via this file's `from isaaclab_tasks.manager_based.manipulation.lift
    import mdp` - that module re-exports isaaclab.envs.mdp's contents)."""

    action_rate_curr = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    )

    joint_vel_curr = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    )


@configclass
class Ar4PickPlaceProvenRecipeEnvCfg(ManagerBasedRLEnvCfg):
    """AR4 proven-recipe task (Experiment 16): from-scratch replication of
    Isaac Lab's Franka Cube Lift and IsaacGymEnvs' FrankaCubeStack reward
    structure and action space on the AR4+cube scene. num_envs=4096
    default - scripts/train.py's --num_envs flag overrides this per-run
    same as every other env cfg in this repo."""

    scene: Ar4PickPlaceMirrorSceneCfg = Ar4PickPlaceMirrorSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

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

Run: `python3 -c "import ast; ast.parse(open('tasks/ar4/pickplace_provenrecipe_env_cfg.py').read())"`
Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add tasks/ar4/pickplace_provenrecipe_env_cfg.py
git commit -m "Add Ar4PickPlaceProvenRecipeEnvCfg: from-scratch proven-recipe replication (Experiment 16)"
```

---

### Task 3: Wire `--provenrecipe` into `scripts/train.py` and `scripts/eval_loop.py`, then smoke-test

**Files:**
- Modify: `scripts/train.py`
- Modify: `scripts/eval_loop.py`

**Interfaces:**
- Consumes: `Ar4PickPlaceProvenRecipeEnvCfg` (Task 2), `Ar4PickPlacePPORunnerCfg` (pre-existing, already imported by both scripts).
- Produces: `--provenrecipe` CLI flag on both scripts, verified via a headless 2-iteration smoke test.

- [ ] **Step 1: Add the flag and import to `scripts/train.py`**

Immediately after the existing `--baseproximity` `parser.add_argument(...)` block (currently lines 94-105, ending right before `AppLauncher.add_app_launcher_args(parser)` at line 106), insert:

```python
parser.add_argument(
    "--provenrecipe",
    action="store_true",
    default=False,
    help=(
        "Train on the from-scratch proven-recipe replication: no standalone grasp reward, a plain "
        "binary lift reward, goal-tracking reward gated on lift, and plain joint-space action - "
        "replicating Isaac Lab's own Franka Cube Lift task and IsaacGymEnvs' FrankaCubeStack task, "
        "both read directly from source. See "
        "docs/superpowers/specs/2026-07-07-ar4-experiment16-proven-recipe-replication-design.md."
    ),
)
```

Add the import next to the existing `pickplace_baseproximity_env_cfg` import (currently line 139):

```python
from tasks.ar4.pickplace_provenrecipe_env_cfg import Ar4PickPlaceProvenRecipeEnvCfg  # noqa: E402
```

Change the `env_cfg_cls` selection (currently lines 151-166) to add `--provenrecipe` as the first branch:

```python
    if args_cli.provenrecipe:
        env_cfg_cls = Ar4PickPlaceProvenRecipeEnvCfg
    elif args_cli.baseproximity:
        env_cfg_cls = Ar4PickPlaceBaseProximityEnvCfg
    elif args_cli.reachskip:
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

**Do NOT change the `agent_cfg` selection** (currently lines 171-174) — `--provenrecipe` must fall through to the `else` branch (plain `Ar4PickPlacePPORunnerCfg`, no `clip_actions=5.0` override), exactly like `--mirror` already does, since this experiment uses plain joint-space action, not task-space/IK. Leave this code exactly as it is:

```python
    if args_cli.taskspace or args_cli.residual or args_cli.reachskip or args_cli.baseproximity:
        agent_cfg = Ar4PickPlaceTaskspacePPORunnerCfg()
    else:
        agent_cfg = Ar4PickPlacePPORunnerCfg()
```

- [ ] **Step 2: Add the flag and import to `scripts/eval_loop.py`**

Immediately after the existing `--baseproximity` `parser.add_argument(...)` block (currently lines 56-61, ending right before `AppLauncher.add_app_launcher_args(parser)` at line 62), insert:

```python
parser.add_argument(
    "--provenrecipe",
    action="store_true",
    default=False,
    help="Evaluate the from-scratch proven-recipe replication (see scripts/train.py --provenrecipe) instead of the four-object scene.",
)
```

Add the import next to the existing `pickplace_baseproximity_env_cfg` import (currently line 94):

```python
from tasks.ar4.pickplace_provenrecipe_env_cfg import Ar4PickPlaceProvenRecipeEnvCfg  # noqa: E402
```

Change the `env_cfg_cls` selection (currently lines 106-121) to add `--provenrecipe` as the first branch:

```python
    if args_cli.provenrecipe:
        env_cfg_cls = Ar4PickPlaceProvenRecipeEnvCfg
    elif args_cli.baseproximity:
        env_cfg_cls = Ar4PickPlaceBaseProximityEnvCfg
    elif args_cli.reachskip:
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

**Do NOT change the `agent_cfg` selection** (currently lines 126-129) — same reasoning as `train.py`, leave exactly as-is:

```python
    if args_cli.taskspace or args_cli.residual or args_cli.reachskip or args_cli.baseproximity:
        agent_cfg = Ar4PickPlaceTaskspacePPORunnerCfg()
    else:
        agent_cfg = Ar4PickPlacePPORunnerCfg()
```

And change the `name_prefix` selection (currently lines 147-160) to add `--provenrecipe` as the first branch:

```python
        if args_cli.provenrecipe:
            name_prefix = "ar4_pickplace_provenrecipe"
        elif args_cli.baseproximity:
            name_prefix = "ar4_pickplace_baseproximity"
        elif args_cli.reachskip:
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

This is the FIRST time `Ar4PickPlaceProvenRecipeEnvCfg` will actually run inside Isaac Sim — Tasks 1-2 only had syntax checks. It's also the first time this repo's `CurriculumCfg` mechanism runs at all (no prior experiment has used it). If it throws an exception, that's real information — read the traceback and fix it, don't route around it.

Run (from repo root, foreground, allow up to 5 minutes):
```bash
rm -f /tmp/provenrecipe_smoke_stdout.log
timeout 280 /home/saps/IsaacLab/isaaclab.sh -p scripts/train.py --provenrecipe --num_envs 16 --max_iterations 2 --headless > /tmp/provenrecipe_smoke_stdout.log 2>&1
echo "exit_code=$?"
```

A `timeout`/nonzero exit code alone is NOT proof of failure (Isaac Sim's shutdown sequence sometimes hangs after genuinely finishing) — verify via files:

```bash
grep -i "error\|exception\|traceback" /tmp/provenrecipe_smoke_stdout.log
ls -la logs/train/ | tail -5
```

Find the newest `logs/train/<timestamp>/` directory and check:
```bash
find logs/train/<newest_timestamp_dir> -name "model_*.pt"
cat logs/train/<newest_timestamp_dir>/params/env.yaml | grep -A5 "curriculum\|lifting_object\|object_goal_tracking\|joint_positions"
```

Expected: `model_0.pt` and `model_1.pt` both exist, `env.yaml` confirms `Ar4PickPlaceProvenRecipeEnvCfg`'s `RewardsCfg` (no antipodal/stillness/ground/base-proximity terms present) and `CurriculumCfg` (both `action_rate_curr`/`joint_vel_curr` terms present) are wired in, plain `JointPositionActionCfg` for `joint_positions` (not `DifferentialInverseKinematicsActionCfg`), and no traceback in the stdout log. If an exception appears, the most likely culprits given this task's new mechanisms are: (a) the `curriculum` field name/type mismatch on `ManagerBasedRLEnvCfg` (re-confirm against `isaaclab/envs/manager_based_rl_env_cfg.py`'s actual field if this errors); (b) `modify_reward_weight` not resolving via the `mdp` import (re-confirm `isaaclab_tasks/manager_based/manipulation/lift/mdp/__init__.py`'s re-export); (c) a params mismatch on the reused `object_ee_distance`/`object_is_lifted` functions (re-check their exact signatures in `isaaclab_tasks/manager_based/manipulation/lift/mdp/rewards.py`).

- [ ] **Step 5: Commit**

```bash
git add scripts/train.py scripts/eval_loop.py
git commit -m "Wire --provenrecipe flag into train.py and eval_loop.py for Experiment 16"
```

---

### Task 4: Diagnostic run (300 iterations) — verify the new action space + curriculum mechanism is stable before committing to a full run

**Files:**
- None created — this is a verification-only task. If the diagnostic looks wrong, stop and report; do not proceed to Task 5.

**Interfaces:**
- Consumes: `Ar4PickPlaceProvenRecipeEnvCfg` (Task 2) via the `--provenrecipe` flag (Task 3).
- Produces: a confirmed-healthy diagnostic run directory under `logs/train/`, gating Task 5.

- [ ] **Step 1: Launch the diagnostic run**

```bash
nohup /home/saps/IsaacLab/isaaclab.sh -p scripts/train.py --provenrecipe --num_envs 4096 --max_iterations 300 --headless > /tmp/exp16_diagnostic_stdout.log 2>&1 &
echo "PID=$!"
```

- [ ] **Step 2: Poll for completion — BLOCK on this yourself, do not end your turn early**

Run this exact command (a real blocking loop — if one call's timeout is hit before the run finishes, re-issue the same command again):
```bash
until find logs/train/ -name "model_299.pt" -newer tasks/ar4/pickplace_provenrecipe_env_cfg.py 2>/dev/null | grep -q .; do sleep 30; done
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
for tag in ['Loss/value_function', 'Episode_Reward/lifting_object', 'Episode_Reward/object_goal_tracking',
            'Episode_Reward/reaching_object', 'Episode_Termination/cube_reached_goal']:
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

1. **`Loss/value_function` stays bounded (no sustained exponential growth).** This experiment combines a new action space (reverting to joint-space after 5 experiments on task-space/IK) with a new curriculum-manager mechanism this repo has never used — real new surface area, worth the same scrutiny prior new mechanisms got in Experiments 11, 13, and 14. A small transient spike with immediate recovery is fine (matching this project's own established precedent, including Experiment 15's own unusually-large-but-benign 17.66 spike); a sustained climb is not.
2. **No exceptions/tracebacks in `/tmp/exp16_diagnostic_stdout.log`.**

If both checks pass, proceed to Task 5. If either fails, stop, do not proceed, and report the finding instead — this would itself be a notable result.

---

### Task 5: Full 1500-iteration training run + TensorBoard verification report

**Files:**
- Create: `docs/superpowers/plans/2026-07-07-ar4-experiment16-report.md`

**Interfaces:**
- Consumes: the Task 4-verified reward/action/curriculum configuration.
- Produces: a complete 1500-iteration training run under `logs/train/`, and a written report with full scalar trajectories.

- [ ] **Step 1: Launch the full training run**

```bash
nohup /home/saps/IsaacLab/isaaclab.sh -p scripts/train.py --provenrecipe --num_envs 4096 --headless > /tmp/exp16_train_stdout.log 2>&1 &
echo "PID=$!"
```

- [ ] **Step 2: Poll for completion — BLOCK on this yourself, do not end your turn early**

```bash
until find logs/train/ -name "model_1499.pt" -newer tasks/ar4/pickplace_provenrecipe_env_cfg.py 2>/dev/null | grep -q .; do sleep 60; done
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
tags = ['Episode_Reward/reaching_object', 'Episode_Reward/lifting_object',
        'Episode_Reward/object_goal_tracking', 'Episode_Reward/object_goal_tracking_fine_grained',
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

Write `docs/superpowers/plans/2026-07-07-ar4-experiment16-report.md` following the structure of `docs/superpowers/plans/2026-07-07-ar4-experiment15-report.md` (checkpoint integrity, `Loss/value_function` sanity check, per-term summary + sampled trajectory for all 6 tags above). Include a "Key Comparison" section against **Experiment 15's exact final value** (the best of the session so far) and **Experiment 12's exact final value** (the original task-space baseline), final-iteration snapshot vs. final-iteration snapshot only, per this project's established correction protocol:

- Experiment 12 final `Episode_Termination/cube_reached_goal`: 0.010773
- Experiment 15 final `Episode_Termination/cube_reached_goal`: 0.017202

Additionally, per the design spec's success criteria, report explicitly: (a) `Episode_Reward/lifting_object`'s nonzero rate and trend across the run (this experiment's direct per-step lift indicator — compare the first 150 iterations' nonzero rate against the last 150 iterations'), and (b) `Episode_Reward/object_goal_tracking`'s nonzero rate and trend (confirming whether the lift-gate is actually unlocking over training — if `lifting_object` grows but `object_goal_tracking` stays at zero, that would indicate the gate itself has a bug worth flagging, not just a training outcome).

State the scalar comparison factually. **Do not draw a final success/failure conclusion from scalars alone** — per this project's own established lesson (Experiment 12's original report misread a scalar drop as failure and had to be corrected by the controller). Final judgment on lift/no-lift requires video inspection, done separately by the controller outside this plan (no Task 6 in this plan — matches this session's established pattern).

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/plans/2026-07-07-ar4-experiment16-report.md
git commit -m "Record Experiment 16 training run: proven-recipe replication scalar trajectories"
```
