# Experiment 17 Implementation Plan: Grasp-Verification-Gated Lift/Goal-Tracking

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Experiment 17 — fix Experiment 16's confirmed "stage leakage" exploit (the policy wedges the cube against its own wrist/gripper-housing geometry to satisfy a height-only lift/goal-tracking reward, with zero jaw contact force at any point in the episode) by gating both reward terms on genuine bilateral antipodal contact, not just height.

**Architecture:** Three new functions in `tasks/ar4/mdp.py` (`genuine_grasp_and_lift`, `lifting_object_grasp_gated`, `mirrored_goal_distance_grasp_gated`) reusing the already-proven `antipodal_grasp_bonus` force-closure check, wired into a new `Ar4PickPlaceGraspGatedEnvCfg` (`tasks/ar4/pickplace_graspgated_env_cfg.py`) that is otherwise identical to Experiment 16's `Ar4PickPlaceProvenRecipeEnvCfg` (same scene, action, observations, events, terminations, curriculum, PPO runner cfg) — isolating the grasp-gate as the only new variable.

**Tech Stack:** Isaac Lab `ManagerBasedRLEnvCfg`, `ContactSensor.data.force_matrix_w` (via the already-proven `antipodal_grasp_bonus`), rsl_rl PPO (`Ar4PickPlacePPORunnerCfg`, unchanged, no task-space `clip_actions` override).

## Global Constraints

- Do not modify `tasks/ar4/pickplace_provenrecipe_env_cfg.py` or any existing function in `tasks/ar4/mdp.py` (including `antipodal_grasp_bonus` and `mirrored_goal_distance_gated`, both reused/called, not edited) — purely additive (three new functions, one new env cfg file).
- Reward weights, exact values (identical to Experiment 16 — isolating the grasp-gate as the only new variable): `reaching_object` (reused `mdp.object_ee_distance`, weight 1.0, `std=0.1`), `lifting_object` (new `ar4_mdp.lifting_object_grasp_gated`, weight 15.0, `minimal_height=0.03`, `force_threshold=0.05`, `antipodal_cos_threshold=-0.7071`), `object_goal_tracking` (new `ar4_mdp.mirrored_goal_distance_grasp_gated`, weight 16.0, `std=0.3`, same grasp params), `object_goal_tracking_fine_grained` (same function, weight 5.0, `std=0.05`, same grasp params), `action_rate` (weight -1e-4), `joint_vel` (weight -1e-4).
- Action space: plain `isaaclab_mdp.JointPositionActionCfg(asset_name="robot", joint_names=ARM_JOINT_NAMES, scale=0.5)` — **not** task-space/IK, identical to Experiment 16. PPO runner cfg: plain `Ar4PickPlacePPORunnerCfg` — **not** `Ar4PickPlaceTaskspacePPORunnerCfg`.
- `ActionsCfg`, `ObservationsCfg`, `EventCfg`, `TerminationsCfg`, `CurriculumCfg` are all identical to Experiment 16's — copy verbatim, no changes to their content.
- Always invoke scripts via `/home/saps/IsaacLab/isaaclab.sh -p scripts/<script>.py` (or `/home/saps/IsaacLab/isaaclab.sh -p -c "<snippet>"` for one-off Python) from the repo root, never plain `python`.
- Verify via file evidence (checkpoints, TensorBoard event files, `params/env.yaml`) — this repo has no pytest-based unit tests for Isaac-Sim-dependent code.
- **Any subagent dispatched to launch or wait on a real Isaac Sim training/rollout job must be given the literal blocking poll command in its dispatch prompt** (not just told to "poll" in prose) — this exact mistake has recurred multiple times this session even when warned only in prose.
- **This experiment's final verdict may NOT rest on video alone.** Experiment 16's own mistake was exactly this — trusting video for a mechanism claim ("genuine grasp") that turned out to be wrong when checked with real contact-sensor instrumentation. Task 6 (instrumented contact-force verification) is mandatory before any "genuine grasp" claim is written to ROADMAP.md.

---

### Task 1: Grasp-gating reward functions

**Files:**
- Modify: `tasks/ar4/mdp.py` (append three new functions at end of file — current file ends at line 820, after `mirrored_goal_distance_gated`)

**Interfaces:**
- Consumes: `antipodal_grasp_bonus(env, force_threshold, antipodal_cos_threshold, jaw1_contact_cfg, jaw2_contact_cfg) -> torch.Tensor` (already in `mdp.py`, confirmed signature — do not modify), `SceneEntityCfg`, `torch`, `RigidObject` (all already imported/type-checked at the top of `mdp.py`).
- Produces: `genuine_grasp_and_lift(env, object_cfg, jaw1_contact_cfg, jaw2_contact_cfg, force_threshold, antipodal_cos_threshold, minimal_height) -> torch.Tensor`, `lifting_object_grasp_gated(env, object_cfg, jaw1_contact_cfg, jaw2_contact_cfg, force_threshold, antipodal_cos_threshold, minimal_height) -> torch.Tensor`, `mirrored_goal_distance_grasp_gated(env, std, minimal_height, object_cfg, jaw1_contact_cfg, jaw2_contact_cfg, force_threshold, antipodal_cos_threshold) -> torch.Tensor` — all three consumed by Task 2's `RewardsCfg`.

- [ ] **Step 1: Append the three new functions to `tasks/ar4/mdp.py`**

Add these functions at the end of the file:

```python
def genuine_grasp_and_lift(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    jaw1_contact_cfg: SceneEntityCfg,
    jaw2_contact_cfg: SceneEntityCfg,
    force_threshold: float,
    antipodal_cos_threshold: float,
    minimal_height: float,
) -> torch.Tensor:
    """Shared gating condition for Experiment 17: the object is lifted
    ONLY if both the height condition AND a genuine bilateral antipodal
    grasp (reusing antipodal_grasp_bonus's own force-closure check, not
    reimplementing it) hold simultaneously - fixes Experiment 16's
    "stage leakage" exploit (Xu et al. 2026, arXiv:2606.31377), confirmed
    via direct contact-sensor instrumentation to have let the policy wedge
    the cube against its own wrist/gripper-housing geometry with zero jaw
    contact force. See
    docs/superpowers/specs/2026-07-07-ar4-experiment17-grasp-gated-lift-design.md.
    """
    object: RigidObject = env.scene[object_cfg.name]
    height_ok = object.data.root_pos_w[:, 2] > minimal_height
    grasp_ok = antipodal_grasp_bonus(
        env, force_threshold, antipodal_cos_threshold, jaw1_contact_cfg, jaw2_contact_cfg
    ) > 0.5
    return (height_ok & grasp_ok).float()


def lifting_object_grasp_gated(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    jaw1_contact_cfg: SceneEntityCfg,
    jaw2_contact_cfg: SceneEntityCfg,
    force_threshold: float,
    antipodal_cos_threshold: float,
    minimal_height: float,
) -> torch.Tensor:
    """Same binary reward shape as isaaclab_tasks' object_is_lifted
    (1.0/0.0), but ONLY pays out when genuine_grasp_and_lift's stricter
    condition holds - see that function's docstring. See
    docs/superpowers/specs/2026-07-07-ar4-experiment17-grasp-gated-lift-design.md.
    """
    return genuine_grasp_and_lift(
        env, object_cfg, jaw1_contact_cfg, jaw2_contact_cfg, force_threshold, antipodal_cos_threshold, minimal_height
    )


def mirrored_goal_distance_grasp_gated(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    object_cfg: SceneEntityCfg,
    jaw1_contact_cfg: SceneEntityCfg,
    jaw2_contact_cfg: SceneEntityCfg,
    force_threshold: float,
    antipodal_cos_threshold: float,
) -> torch.Tensor:
    """Same tanh-kernel goal-distance formula as
    mirrored_goal_distance_gated (Experiment 16), but gated on
    genuine_grasp_and_lift's height-AND-grasp condition instead of height
    alone. See
    docs/superpowers/specs/2026-07-07-ar4-experiment17-grasp-gated-lift-design.md.
    """
    object: RigidObject = env.scene[object_cfg.name]
    if not hasattr(env, "_target_pos_w"):
        env._target_pos_w = torch.zeros(env.num_envs, 3, device=env.device)
    distance = torch.norm(env._target_pos_w - object.data.root_pos_w, dim=-1)
    gate = genuine_grasp_and_lift(
        env, object_cfg, jaw1_contact_cfg, jaw2_contact_cfg, force_threshold, antipodal_cos_threshold, minimal_height
    )
    return gate * (1.0 - torch.tanh(distance / std))
```

No new imports needed: `SceneEntityCfg`, `torch`, `RigidObject`, `ManagerBasedRLEnv` are all already imported/type-checked at the top of `tasks/ar4/mdp.py`, and `antipodal_grasp_bonus` is already defined earlier in the same file (do not re-import or redefine it).

- [ ] **Step 2: Syntax-check the file**

Run: `python3 -c "import ast; ast.parse(open('tasks/ar4/mdp.py').read())"`
Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add tasks/ar4/mdp.py
git commit -m "Add grasp-gated lift/goal-tracking reward functions for Experiment 17"
```

---

### Task 2: New grasp-gated env cfg file

**Files:**
- Create: `tasks/ar4/pickplace_graspgated_env_cfg.py`

**Interfaces:**
- Consumes: `Ar4PickPlaceMirrorSceneCfg` (from `tasks/ar4/pickplace_mirror_env_cfg.py`, unmodified), `ar4_mdp.lifting_object_grasp_gated`/`mirrored_goal_distance_grasp_gated` (Task 1), `ar4_mdp.set_mirrored_goal`/`mirrored_target_position_in_robot_root_frame`/`object_reached_mirrored_goal` (all pre-existing, unmodified), `mdp.object_ee_distance`/`modify_reward_weight` (from `isaaclab_tasks.manager_based.manipulation.lift.mdp`, reused directly), `ARM_JOINT_NAMES`/`GRIPPER_JOINT_NAMES`/`GRIPPER_OPEN_POS`/`GRIPPER_CLOSED_POS` (from `robot_cfg.py`), `Ar4PickPlacePPORunnerCfg` (from `tasks/ar4/agents/rsl_rl_ppo_cfg.py`, unmodified, reused directly).
- Produces: `Ar4PickPlaceGraspGatedEnvCfg` class — consumed by Task 3 (script wiring).

- [ ] **Step 1: Write the new file**

```python
# tasks/ar4/pickplace_graspgated_env_cfg.py
"""Grasp-verification-gated variant of the proven-recipe replication
(Experiment 17): identical to Experiment 16's Ar4PickPlaceProvenRecipeEnvCfg
(tasks/ar4/pickplace_provenrecipe_env_cfg.py) in every respect except the
lift and goal-tracking reward terms, which now require genuine bilateral
antipodal jaw contact (not just object height) before they pay out - fixes
a confirmed "stage leakage" exploit (Xu et al. 2026, arXiv:2606.31377)
found in Experiment 16 via direct contact-sensor instrumentation: the
policy was wedging the cube against its own wrist/gripper-housing geometry
to satisfy the height-only reward, registering ZERO gripper-jaw contact
force at every step of the "held" period. See
docs/superpowers/specs/2026-07-07-ar4-experiment17-grasp-gated-lift-design.md.

Additive/parallel to every other pickplace_*.py file: does NOT modify
env_cfg.py, objects_cfg.py, pickplace_mirror_env_cfg.py,
pickplace_taskspace_env_cfg.py, pickplace_residual_env_cfg.py,
pickplace_reachskip_env_cfg.py, pickplace_baseproximity_env_cfg.py, or
pickplace_provenrecipe_env_cfg.py. Reuses Ar4PickPlaceMirrorSceneCfg and
Ar4PickPlacePPORunnerCfg directly.

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
    """Plain joint-space action, identical to Experiment 16's
    ActionsCfg."""

    joint_positions = isaaclab_mdp.JointPositionActionCfg(asset_name="robot", joint_names=ARM_JOINT_NAMES, scale=0.5)
    gripper_position = isaaclab_mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=GRIPPER_JOINT_NAMES,
        open_command_expr={name: GRIPPER_OPEN_POS for name in GRIPPER_JOINT_NAMES},
        close_command_expr={name: GRIPPER_CLOSED_POS for name in GRIPPER_JOINT_NAMES},
    )


@configclass
class ObservationsCfg:
    """Identical to Experiment 16's ObservationsCfg."""

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
    """Identical to Experiment 16's EventCfg - no waypoint system, no
    grasp-related reset state (the grasp gate is computed live from
    contact sensors each step, no stateful buffer needed)."""

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
    """Identical to Experiment 16's TerminationsCfg."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    cube_reached_goal = DoneTerm(
        func=ar4_mdp.object_reached_mirrored_goal,
        params={"threshold": 0.02, "object_cfg": SceneEntityCfg("cube")},
    )


@configclass
class RewardsCfg:
    """Six terms, weights identical to Experiment 16 - the grasp-gate is
    the only new variable. lifting_object and object_goal_tracking/
    object_goal_tracking_fine_grained now use the new grasp-gated
    functions (Task 1) instead of Experiment 16's height-only versions.
    reaching_object/action_rate/joint_vel are unchanged. See
    docs/superpowers/specs/2026-07-07-ar4-experiment17-grasp-gated-lift-design.md."""

    reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        weight=1.0,
        params={"std": 0.1, "object_cfg": SceneEntityCfg("cube"), "ee_frame_cfg": SceneEntityCfg("ee_frame")},
    )

    lifting_object = RewTerm(
        func=ar4_mdp.lifting_object_grasp_gated,
        weight=15.0,
        params={
            "object_cfg": SceneEntityCfg("cube"),
            "jaw1_contact_cfg": SceneEntityCfg("gripper_jaw1_contact"),
            "jaw2_contact_cfg": SceneEntityCfg("gripper_jaw2_contact"),
            "force_threshold": 0.05,
            "antipodal_cos_threshold": -0.7071,
            "minimal_height": 0.03,
        },
    )

    object_goal_tracking = RewTerm(
        func=ar4_mdp.mirrored_goal_distance_grasp_gated,
        weight=16.0,
        params={
            "std": 0.3,
            "minimal_height": 0.03,
            "object_cfg": SceneEntityCfg("cube"),
            "jaw1_contact_cfg": SceneEntityCfg("gripper_jaw1_contact"),
            "jaw2_contact_cfg": SceneEntityCfg("gripper_jaw2_contact"),
            "force_threshold": 0.05,
            "antipodal_cos_threshold": -0.7071,
        },
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=ar4_mdp.mirrored_goal_distance_grasp_gated,
        weight=5.0,
        params={
            "std": 0.05,
            "minimal_height": 0.03,
            "object_cfg": SceneEntityCfg("cube"),
            "jaw1_contact_cfg": SceneEntityCfg("gripper_jaw1_contact"),
            "jaw2_contact_cfg": SceneEntityCfg("gripper_jaw2_contact"),
            "force_threshold": 0.05,
            "antipodal_cos_threshold": -0.7071,
        },
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class CurriculumCfg:
    """Identical to Experiment 16's CurriculumCfg."""

    action_rate_curr = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    )

    joint_vel_curr = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    )


@configclass
class Ar4PickPlaceGraspGatedEnvCfg(ManagerBasedRLEnvCfg):
    """AR4 grasp-gated task (Experiment 17): identical to Experiment 16
    except lifting_object/object_goal_tracking now require genuine
    bilateral antipodal jaw contact, not just object height. num_envs=4096
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

Run: `python3 -c "import ast; ast.parse(open('tasks/ar4/pickplace_graspgated_env_cfg.py').read())"`
Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add tasks/ar4/pickplace_graspgated_env_cfg.py
git commit -m "Add Ar4PickPlaceGraspGatedEnvCfg: grasp-verification-gated lift/goal-tracking (Experiment 17)"
```

---

### Task 3: Wire `--graspgated` into `scripts/train.py` and `scripts/eval_loop.py`, then smoke-test

**Files:**
- Modify: `scripts/train.py`
- Modify: `scripts/eval_loop.py`

**Interfaces:**
- Consumes: `Ar4PickPlaceGraspGatedEnvCfg` (Task 2), `Ar4PickPlacePPORunnerCfg` (pre-existing, already imported by both scripts).
- Produces: `--graspgated` CLI flag on both scripts, verified via a headless 2-iteration smoke test.

- [ ] **Step 1: Add the flag and import to `scripts/train.py`**

Immediately after the existing `--provenrecipe` `parser.add_argument(...)` block (currently lines 106-117, ending right before `AppLauncher.add_app_launcher_args(parser)` at line 118), insert:

```python
parser.add_argument(
    "--graspgated",
    action="store_true",
    default=False,
    help=(
        "Train on the grasp-verification-gated variant of the proven-recipe scene: identical to "
        "--provenrecipe except the lift and goal-tracking reward terms now require genuine bilateral "
        "antipodal jaw contact, not just object height, fixing a confirmed 'stage leakage' exploit "
        "where the policy wedged the cube against its own wrist geometry instead of grasping it. See "
        "docs/superpowers/specs/2026-07-07-ar4-experiment17-grasp-gated-lift-design.md."
    ),
)
```

Add the import next to the existing `pickplace_provenrecipe_env_cfg` import (currently line 152):

```python
from tasks.ar4.pickplace_graspgated_env_cfg import Ar4PickPlaceGraspGatedEnvCfg  # noqa: E402
```

Change the `env_cfg_cls` selection (currently lines 164-181) to add `--graspgated` as the first branch:

```python
    if args_cli.graspgated:
        env_cfg_cls = Ar4PickPlaceGraspGatedEnvCfg
    elif args_cli.provenrecipe:
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

**Do NOT add `--graspgated` to the `agent_cfg` selection** (currently lines 186-189) — the current condition is exactly `if args_cli.taskspace or args_cli.residual or args_cli.reachskip or args_cli.baseproximity:` (4 flags — `--provenrecipe` was correctly excluded in Experiment 16, and `--graspgated` must be excluded the same way, since this experiment uses plain joint-space action). Leave this code exactly as it is:

```python
    if args_cli.taskspace or args_cli.residual or args_cli.reachskip or args_cli.baseproximity:
        agent_cfg = Ar4PickPlaceTaskspacePPORunnerCfg()
    else:
        agent_cfg = Ar4PickPlacePPORunnerCfg()
```

- [ ] **Step 2: Add the flag and import to `scripts/eval_loop.py`**

Immediately after the existing `--provenrecipe` `parser.add_argument(...)` block (currently lines 62-67, ending right before `AppLauncher.add_app_launcher_args(parser)` at line 68), insert:

```python
parser.add_argument(
    "--graspgated",
    action="store_true",
    default=False,
    help="Evaluate the grasp-verification-gated scene (see scripts/train.py --graspgated) instead of the four-object scene.",
)
```

Add the import next to the existing `pickplace_provenrecipe_env_cfg` import (currently line 101):

```python
from tasks.ar4.pickplace_graspgated_env_cfg import Ar4PickPlaceGraspGatedEnvCfg  # noqa: E402
```

Change the `env_cfg_cls` selection (currently lines 113-130) to add `--graspgated` as the first branch:

```python
    if args_cli.graspgated:
        env_cfg_cls = Ar4PickPlaceGraspGatedEnvCfg
    elif args_cli.provenrecipe:
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

**Do NOT add `--graspgated` to the `agent_cfg` selection** (currently lines 135-138) — same reasoning as `train.py`, leave exactly as-is:

```python
    if args_cli.taskspace or args_cli.residual or args_cli.reachskip or args_cli.baseproximity:
        agent_cfg = Ar4PickPlaceTaskspacePPORunnerCfg()
    else:
        agent_cfg = Ar4PickPlacePPORunnerCfg()
```

And change the `name_prefix` selection (currently lines 156-171) to add `--graspgated` as the first branch:

```python
        if args_cli.graspgated:
            name_prefix = "ar4_pickplace_graspgated"
        elif args_cli.provenrecipe:
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

This is the FIRST time `Ar4PickPlaceGraspGatedEnvCfg` will actually run inside Isaac Sim — Tasks 1-2 only had syntax checks. The new reward functions call `antipodal_grasp_bonus` (already proven across many prior experiments) inside a new gating wrapper — the main new risk is a params/signature mismatch in how the gate is composed, not the underlying contact-sensor mechanics themselves.

Run (from repo root, foreground, allow up to 5 minutes):
```bash
rm -f /tmp/graspgated_smoke_stdout.log
timeout 280 /home/saps/IsaacLab/isaaclab.sh -p scripts/train.py --graspgated --num_envs 16 --max_iterations 2 --headless > /tmp/graspgated_smoke_stdout.log 2>&1
echo "exit_code=$?"
```

A `timeout`/nonzero exit code alone is NOT proof of failure (Isaac Sim's shutdown sequence sometimes hangs after genuinely finishing) — verify via files:

```bash
grep -i "error\|exception\|traceback" /tmp/graspgated_smoke_stdout.log
ls -la logs/train/ | tail -5
```

Find the newest `logs/train/<timestamp>/` directory and check:
```bash
find logs/train/<newest_timestamp_dir> -name "model_*.pt"
cat logs/train/<newest_timestamp_dir>/params/env.yaml | grep -A6 "lifting_object\|object_goal_tracking"
```

Expected: `model_0.pt` and `model_1.pt` both exist, `env.yaml` confirms `lifting_object`'s function is `lifting_object_grasp_gated` (not `object_is_lifted`) and `object_goal_tracking`'s function is `mirrored_goal_distance_grasp_gated` (not `mirrored_goal_distance_gated`), both with `force_threshold`/`antipodal_cos_threshold` params present, and no traceback in the stdout log. If an exception appears, the most likely culprit is a params mismatch between `RewardsCfg`'s `params={...}` dicts and the actual function signatures written in Task 1 — re-check both against each other directly.

- [ ] **Step 5: Commit**

```bash
git add scripts/train.py scripts/eval_loop.py
git commit -m "Wire --graspgated flag into train.py and eval_loop.py for Experiment 17"
```

---

### Task 4: Diagnostic run (300 iterations) — verify the grasp-gated reward is stable before committing to a full run

**Files:**
- None created — this is a verification-only task. If the diagnostic looks wrong, stop and report; do not proceed to Task 5.

**Interfaces:**
- Consumes: `Ar4PickPlaceGraspGatedEnvCfg` (Task 2) via the `--graspgated` flag (Task 3).
- Produces: a confirmed-healthy diagnostic run directory under `logs/train/`, gating Task 5.

- [ ] **Step 1: Launch the diagnostic run**

```bash
nohup /home/saps/IsaacLab/isaaclab.sh -p scripts/train.py --graspgated --num_envs 4096 --max_iterations 300 --headless > /tmp/exp17_diagnostic_stdout.log 2>&1 &
echo "PID=$!"
```

- [ ] **Step 2: Poll for completion — BLOCK on this yourself, do not end your turn early**

Run this exact command (a real blocking loop — if one call's timeout is hit before the run finishes, re-issue the same command again):
```bash
until find logs/train/ -name "model_299.pt" -newer tasks/ar4/pickplace_graspgated_env_cfg.py 2>/dev/null | grep -q .; do sleep 30; done
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

1. **`Loss/value_function` stays bounded (no sustained exponential growth).** This experiment reuses an already-proven action space (joint-space, validated fully in Experiment 16) and already-proven contact-sensor code (`antipodal_grasp_bonus`, validated across Experiments 9-15) — lower novel-mechanism risk than Experiment 16's own diagnostic, but the diagnostic gate is this project's uniform standing practice regardless of perceived risk. A small transient spike with immediate recovery, or even a larger spike that clearly declines by the run's end (matching Experiment 16's own precedent of a curriculum-driven spike at iteration ~417 that fully resolved), is acceptable; a sustained, non-recovering climb is not.
2. **No exceptions/tracebacks in `/tmp/exp17_diagnostic_stdout.log`.**

If both checks pass, proceed to Task 5. If either fails, stop, do not proceed, and report the finding instead.

---

### Task 5: Full 1500-iteration training run + TensorBoard verification report

**Files:**
- Create: `docs/superpowers/plans/2026-07-07-ar4-experiment17-report.md`

**Interfaces:**
- Consumes: the Task 4-verified grasp-gated reward configuration.
- Produces: a complete 1500-iteration training run under `logs/train/`, and a written report with full scalar trajectories.

- [ ] **Step 1: Launch the full training run**

```bash
nohup /home/saps/IsaacLab/isaaclab.sh -p scripts/train.py --graspgated --num_envs 4096 --headless > /tmp/exp17_train_stdout.log 2>&1 &
echo "PID=$!"
```

- [ ] **Step 2: Poll for completion — BLOCK on this yourself, do not end your turn early**

```bash
until find logs/train/ -name "model_1499.pt" -newer tasks/ar4/pickplace_graspgated_env_cfg.py 2>/dev/null | grep -q .; do sleep 60; done
echo "full run complete"
```
Expect roughly 15-25 minutes of real wall-clock time based on this repo's prior 1500-iteration runs at `num_envs=4096` — treat that as a rough guide, not a hard cutoff; keep re-issuing the blocking command if a single tool call's own timeout is hit before the checkpoint appears.

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

Write `docs/superpowers/plans/2026-07-07-ar4-experiment17-report.md` following the structure of `docs/superpowers/plans/2026-07-07-ar4-experiment16-report.md` (checkpoint integrity, `Loss/value_function` sanity check, per-term summary + sampled trajectory for all 6 tags above). Include a "Key Comparison" section against **Experiment 16's exact final value** and **Experiment 12's exact final value**, final-iteration snapshot vs. final-iteration snapshot only, per this project's established correction protocol:

- Experiment 12 final `Episode_Termination/cube_reached_goal`: 0.010773
- Experiment 16 final `Episode_Termination/cube_reached_goal`: 0.008962

Additionally, report explicitly whether `Episode_Reward/lifting_object`'s nonzero rate is much lower/sparser than Experiment 16's own (Experiment 16's final values: `lifting_object` first-150/last-150 nonzero rates were 81.3%/100.0%, per that experiment's report) — this experiment's grasp-gate makes the condition strictly harder to satisfy than Experiment 16's height-only check, so a real drop in nonzero rate is an expected, informative possibility, not necessarily a bug. Report the actual numbers, don't assume either direction.

State the scalar comparison factually. **Do not draw a final success/failure conclusion from scalars OR video alone** — per this project's own established lesson (Experiment 16's own report/ROADMAP entry initially misread video evidence as confirming genuine grasp, and had to be corrected after direct contact-sensor instrumentation showed otherwise). Final judgment requires Task 6's instrumented contact-force verification, done next in this same plan.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/plans/2026-07-07-ar4-experiment17-report.md
git commit -m "Record Experiment 17 training run: grasp-gated lift/goal-tracking scalar trajectories"
```

---

### Task 6: Instrumented contact-force verification (mandatory — this experiment's verdict cannot rest on video alone)

**Files:**
- Create: a standalone diagnostic script in a scratch location (not part of `scripts/`, this is a one-off verification tool, not permanent repo infrastructure) — e.g. under this session's scratchpad directory. Exact path is the implementer's choice; report it clearly.

**Interfaces:**
- Consumes: `logs/train/<Task 5's run>/model_1499.pt` (Task 5's final checkpoint), `Ar4PickPlaceGraspGatedEnvCfg` (Task 2).
- Produces: a written report stating, with real numeric evidence, whether genuine bilateral jaw contact force is present during any elevated-hold behavior this checkpoint exhibits.

**Context on why this task exists and how it must work:** Experiment 16's ROADMAP entry originally claimed "genuine, sustained lift" based on video alone, and had to be corrected the same day after a controller-run instrumented rollout showed **zero gripper-jaw contact force at every one of 250 logged steps** of the episode, including the initial approach — the cube was being wedged against the wrist/gripper-housing body (`link_6`/`gripper_base_link`), not gripped. This task is that same check, run against Experiment 17's own trained checkpoint, to determine whether the grasp-gate (Task 1) actually fixed the exploit or not.

- [ ] **Step 1: Load Articulation and ContactSensor APIs**

Read `Articulation.find_bodies()` and the `body_pos_w` data field directly from the installed Isaac Lab source at `/home/saps/IsaacLab/source/isaaclab/isaaclab/assets/articulation/articulation.py` (used to resolve `gripper_jaw1_link`/`gripper_jaw2_link`/`link_6`/`gripper_base_link` body indices by name, then index `robot.data.body_pos_w[:, body_id]` for world positions), and the `ContactSensor` class's `data.force_matrix_w` field under `/home/saps/IsaacLab/source/isaaclab/isaaclab/sensors/contact_sensor/` (same field `antipodal_grasp_bonus` in `tasks/ar4/mdp.py` already reads — use that function's exact indexing pattern, `sensor.data.force_matrix_w.view(env.num_envs, 3)` then `torch.linalg.vector_norm(..., dim=-1)`, as your reference, don't reinvent the indexing).

- [ ] **Step 2: Write a standalone instrumented rollout script**

Follow `scripts/eval_loop.py`'s exact checkpoint-loading pattern (env cfg construction with `num_envs=1`, `Ar4PickPlacePPORunnerCfg` — not the task-space runner cfg, `RslRlVecEnvWrapper`, `OnPolicyRunner.load(checkpoint)`, `runner.get_inference_policy(...)`) but skip the video-recording wrapper entirely — this script's only job is running one full episode (250 steps) with the policy and printing per-step diagnostics, not producing a video. At minimum every ~10 steps (finer resolution — every step — near where the cube's height first crosses `minimal_height=0.03`, since that transition is where the mechanism matters most), print: the cube's world z-position, both jaw contact sensors' force magnitudes (via the pattern from Step 1), both gripper joint positions, and the cube's distance to `gripper_jaw1_link`/`gripper_jaw2_link` vs. `link_6`/`gripper_base_link` (via `find_bodies()` + `body_pos_w` indexing). Invoke it the same way `scripts/eval_loop.py` is invoked — via `/home/saps/IsaacLab/isaaclab.sh -p <your_script_path>.py`, never plain `python`, since it needs Isaac Sim/`AppLauncher` to load the checkpoint and step the environment.

**CRITICAL — this is a real Isaac Sim rollout and will take real wall-clock time (likely a few minutes for a 250-step episode plus Isaac Sim startup overhead). Block on it yourself; do not end your turn while it's running.** Launch it with `nohup ... > /tmp/exp17_grasp_diagnostic_stdout.log 2>&1 &` and poll with a real blocking loop (e.g. `until grep -q "<some marker your script prints on completion>" /tmp/exp17_grasp_diagnostic_stdout.log 2>/dev/null; do sleep 15; done`, re-issuing if one tool call's timeout is hit) rather than assuming it's done after a fixed wait.

- [ ] **Step 3: Interpret the results and write the report**

Look specifically for: (a) does the cube's z-position ever sustain above `0.03` (the `minimal_height` threshold) for more than a handful of consecutive steps (i.e. does `lifting_object` actually fire at all under the new, stricter gate — a real possible outcome is that it doesn't, meaning the gate successfully closed the exploit but the policy hasn't yet learned genuine grasping within 1500 iterations, which is itself a valid, reportable, non-failure result the spec's own success criteria anticipated); (b) if the cube IS elevated for a sustained period, are BOTH jaw contact sensors simultaneously showing force above `0.05` (the `force_threshold` used in the reward) during that period, correlated step-by-step, not just at isolated moments; (c) is the cube closer to the jaws than to `link_6`/`gripper_base_link` during any such elevated period (the opposite of Experiment 16's finding). Write a plain, direct finding: either "genuine bilateral jaw contact confirmed during elevated-hold, at approximately these force magnitudes, and the exploit pattern from Experiment 16 (zero force, cube closer to wrist) does not reproduce" or "the exploit still reproduces / a new one appears" or "lift essentially does not happen under the stricter gate within this run." Do not soften or hedge past what the actual logged numbers show.

- [ ] **Step 4: Report status**

This is the final task in this plan — no Task 7. The controller (Principal) reads this task's report and personally writes the ROADMAP.md entry for Experiment 17 afterward (not delegated to this task's implementer), matching this session's established pattern for decisive evidence. This task's own deliverable is the instrumented script + its report with real numbers, nothing more.
