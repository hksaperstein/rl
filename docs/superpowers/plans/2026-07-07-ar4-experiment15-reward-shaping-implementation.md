# Experiment 15 Implementation Plan: Ground/Base-Proximity Penalties + Higher Grasp Weight

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Experiment 15 — wire the existing unused `ground_penalty` function into the reward, add a new `base_proximity_penalty` function (cube x/y distance to the robot's own base), and raise `antipodal_grasp_bonus`'s weight (with a matched `stillness_penalty` raise to preserve the already-verified anti-freeze reward-rate margin), built on Experiment 12's clean baseline.

**Architecture:** Two new/reused reward functions in `tasks/ar4/mdp.py` (`base_proximity_penalty` new, `ground_penalty` pre-existing but never wired in), a new `Ar4PickPlaceBaseProximityEnvCfg` (`tasks/ar4/pickplace_baseproximity_env_cfg.py`) that reuses Experiment 12's scene/action/observations/events/terminations/PPO-runner-cfg unchanged and only changes `RewardsCfg`, wired into `scripts/train.py`/`scripts/eval_loop.py` via a new `--baseproximity` flag.

**Tech Stack:** Isaac Lab `ManagerBasedRLEnvCfg`, `RewardTermCfg`, rsl_rl PPO (`Ar4PickPlaceTaskspacePPORunnerCfg`, unchanged).

## Global Constraints

- Do not modify `tasks/ar4/pickplace_taskspace_env_cfg.py`, `tasks/ar4/pickplace_residual_env_cfg.py`, `tasks/ar4/pickplace_reachskip_env_cfg.py`, `tasks/ar4/residual_ik_action.py`, or any existing function in `tasks/ar4/mdp.py` (including `ground_penalty` itself, reused as-is) — purely additive (one new function appended, one new env cfg file).
- Action term, observations, events, and terminations must be identical to `pickplace_taskspace_env_cfg.py`'s (Experiment 12's clean, non-regressed baseline) — plain `isaaclab_mdp.DifferentialInverseKinematicsActionCfg`, **not** `ResidualDifferentialIKActionCfg` (Experiment 13) and **not** the reach-skip reset event (Experiment 14). This experiment isolates the reward function as the only new variable.
- Reward weights, exact values (from the design spec):
  - `ground_penalty`: weight `0.1`, `ground_height_threshold=0.015`, `object_cfg=SceneEntityCfg("cube")`.
  - `base_proximity_penalty`: weight `0.1`, `base_xy_threshold=0.08`, `object_cfg=SceneEntityCfg("cube")`, `robot_cfg=SceneEntityCfg("robot")`.
  - `antipodal_grasp_bonus`: weight `4.0` (raised from 3.0), `antipodal_cos_threshold=-0.7071` (unchanged), `force_threshold=0.05` (unchanged).
  - `stillness_penalty`: weight `6.0` (raised from 5.0, matched raise preserving the -2.0/step net margin), `still_bound=0.005`/`patience_steps=25` (unchanged).
  - `path_proximity_bonus` (weight 25.0), `gripper_schedule_bonus` (weight 0.1), `action_rate` (weight -1e-4), `joint_vel` (weight -1e-4): copied verbatim, unchanged from `pickplace_taskspace_env_cfg.py`'s current `RewardsCfg`.
- Must reuse `Ar4PickPlaceTaskspacePPORunnerCfg` (`clip_actions=5.0`) unchanged — no new PPORunnerCfg subclass.
- Always invoke scripts via `/home/saps/IsaacLab/isaaclab.sh -p scripts/<script>.py` (or `/home/saps/IsaacLab/isaaclab.sh -p -c "<snippet>"` for one-off Python) from the repo root, never plain `python`.
- Verify via file evidence (checkpoints, TensorBoard event files, `params/env.yaml`) — this repo has no pytest-based unit tests for Isaac-Sim-dependent code.
- **Any subagent dispatched to launch or wait on a training run must be given the literal blocking poll command in its dispatch prompt** (not just told to "poll" in prose) — this exact mistake ("wait for a background notification" that never comes) has recurred multiple times this session even when explicitly warned in prose only.

---

### Task 1: `base_proximity_penalty` reward function

**Files:**
- Modify: `tasks/ar4/mdp.py` (append new function at end of file, after `reset_arm_to_pregrasp_pose`)

**Interfaces:**
- Consumes: `SceneEntityCfg`, `torch` (already imported at the top of `mdp.py`), `RigidObject` (already imported under `TYPE_CHECKING`).
- Produces: `base_proximity_penalty(env, object_cfg, robot_cfg, base_xy_threshold) -> torch.Tensor` — a `RewardTermCfg` function, consumed by Task 2's `RewardsCfg`.

- [ ] **Step 1: Append the new function to `tasks/ar4/mdp.py`**

Add this function at the end of the file:

```python
def base_proximity_penalty(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    base_xy_threshold: float,
) -> torch.Tensor:
    """Penalty for the cube being horizontally close to the robot's own
    base, independent of height - unlike ground_penalty (z-height only,
    fires for any low cube position anywhere in the workspace), this
    specifically targets the cube sitting at or sliding into the base
    column, a distinct failure mode from "not yet lifted." Direct user
    request (2026-07-07): "negative reward for the cube contacting the
    base of the robot" - explicitly requested as a new function, separate
    from ground_penalty. x/y distance only (not z): a cube directly above
    the base at carry height should not be penalized by this term, only
    one sitting/sliding into the base footprint itself. See
    docs/superpowers/specs/2026-07-07-ar4-experiment15-reward-shaping-design.md.
    """
    object: RigidObject = env.scene[object_cfg.name]
    robot: RigidObject = env.scene[robot_cfg.name]
    object_xy = object.data.root_pos_w[:, :2]
    robot_xy = robot.data.root_pos_w[:, :2]
    xy_dist = torch.norm(object_xy - robot_xy, dim=-1)
    too_close = xy_dist < base_xy_threshold
    return -too_close.float()
```

No new imports needed: `SceneEntityCfg`, `torch`, `RigidObject`, `ManagerBasedRLEnv` are all already imported/type-checked at the top of `tasks/ar4/mdp.py` (confirmed — the same imports `ground_penalty` and `stillness_penalty` already use for an identical `RigidObject`/`SceneEntityCfg` pattern).

- [ ] **Step 2: Best-effort check of the base footprint against the 0.08m threshold (informational only, not blocking)**

The `0.08` constant in the design spec is reasoned from workspace-area proportion, not measured robot geometry. Attempt to confirm it's not obviously wrong by checking the robot's base-link collision bounds:

```bash
/home/saps/IsaacLab/isaaclab.sh -p -c "
import sys
sys.path.insert(0, '.')
from tasks.ar4.robot_cfg import _resolve_usd_path
print('USD path:', _resolve_usd_path())
"
```

If the USD path resolves, optionally inspect it with `usdview` or a text search for the `base_link` prim's extent/bbox attributes (e.g. `grep -A5 'base_link' <path> | grep -i extent`). This is genuinely best-effort — if extracting an exact bounding box from the USD turns out to be nontrivial (no straightforward extent attribute, needs a full USD-stage load), do not block on it. Note whatever is found (or that it wasn't easily extractable) in the task report. **Do not change the `0.08` constant based on this check** unless it reveals the base footprint is clearly larger than 0.08m in some horizontal dimension (in which case, flag this as a concern in the report rather than silently changing the spec's chosen value — the controller will decide whether to adjust).

- [ ] **Step 3: Syntax-check the file**

Run: `python3 -c "import ast; ast.parse(open('tasks/ar4/mdp.py').read())"`
Expected: no output.

- [ ] **Step 4: Commit**

```bash
git add tasks/ar4/mdp.py
git commit -m "Add base_proximity_penalty reward function for Experiment 15"
```

---

### Task 2: New base-proximity env cfg file

**Files:**
- Create: `tasks/ar4/pickplace_baseproximity_env_cfg.py`

**Interfaces:**
- Consumes: `Ar4PickPlaceMirrorSceneCfg` (from `tasks/ar4/pickplace_mirror_env_cfg.py`, unmodified), `ar4_mdp.ground_penalty` (pre-existing, unmodified), `ar4_mdp.base_proximity_penalty` (Task 1), `ar4_mdp.path_proximity_bonus`/`gripper_schedule_bonus`/`antipodal_grasp_bonus`/`stillness_penalty`/`set_mirrored_goal`/`compute_path_waypoints`/`mirrored_target_position_in_robot_root_frame`/`object_reached_mirrored_goal` (all pre-existing, unmodified), `_EE_OFFSET` (from `pickplace_env_cfg.py`), `ARM_JOINT_NAMES`/`GRIPPER_JOINT_NAMES`/`GRIPPER_OPEN_POS`/`GRIPPER_CLOSED_POS` (from `robot_cfg.py`), `Ar4PickPlaceTaskspacePPORunnerCfg` (from `pickplace_taskspace_env_cfg.py`, unmodified, reused directly — not redefined).
- Produces: `Ar4PickPlaceBaseProximityEnvCfg` class — consumed by Task 3 (script wiring).

- [ ] **Step 1: Write the new file**

```python
# tasks/ar4/pickplace_baseproximity_env_cfg.py
"""Reward-shaping variant of the AR4 mirror-goal cube task (Experiment 15):
identical scene/action/observations/events/terminations as
pickplace_taskspace_env_cfg.py (Experiment 12's clean, non-regressed
baseline - NOT Experiment 13's residual action or Experiment 14's
reach-skip reset), with only RewardsCfg changed: the existing but
never-activated ground_penalty function wired in, a new
base_proximity_penalty function (cube x/y distance to the robot's own
base), and antipodal_grasp_bonus's weight raised (with a matched
stillness_penalty raise preserving the already-verified anti-freeze
reward-rate margin). See
docs/superpowers/specs/2026-07-07-ar4-experiment15-reward-shaping-design.md.

Additive/parallel to every other pickplace_*.py file: does NOT modify
env_cfg.py, objects_cfg.py, pickplace_env_cfg.py, pickplace_mirror_env_cfg.py,
pickplace_ik_guided_env_cfg.py, pickplace_taskspace_env_cfg.py,
pickplace_residual_env_cfg.py, or pickplace_reachskip_env_cfg.py. Reuses
Ar4PickPlaceMirrorSceneCfg and Ar4PickPlaceTaskspacePPORunnerCfg directly -
only RewardsCfg differs from pickplace_taskspace_env_cfg.py.

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
    """Identical to pickplace_taskspace_env_cfg.py's EventCfg (no reset
    event changes in this experiment - reward function only):
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
    """Experiment 15: adds ground_penalty (existing function, never
    previously wired into any RewardsCfg) and base_proximity_penalty (new
    function, Task 1) on top of Experiment 12's exact RewardsCfg, and
    raises antipodal_grasp_bonus's weight 3.0 -> 4.0 with a matched
    stillness_penalty raise 5.0 -> 6.0 (preserves the exact -2.0/step net
    margin Experiment 12 verified for the grasped-and-stagnant state - see
    the design spec's section 3 for the full arithmetic).
    path_proximity_bonus/gripper_schedule_bonus/action_rate/joint_vel are
    unchanged from pickplace_taskspace_env_cfg.py. See
    docs/superpowers/specs/2026-07-07-ar4-experiment15-reward-shaping-design.md."""

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

    # weight raised 3.0 -> 4.0 (Experiment 15, direct user request: "higher
    # reward for the cube being in the grasp position"). stillness_penalty
    # below is raised in matched proportion to preserve the exact -2.0/step
    # net margin Experiment 12 verified for the grasped-and-stagnant state.
    antipodal_grasp_bonus = RewTerm(
        func=ar4_mdp.antipodal_grasp_bonus,
        weight=4.0,
        params={
            "force_threshold": 0.05,
            "antipodal_cos_threshold": -0.7071,
            "jaw1_contact_cfg": SceneEntityCfg("gripper_jaw1_contact"),
            "jaw2_contact_cfg": SceneEntityCfg("gripper_jaw2_contact"),
        },
    )

    # weight raised 5.0 -> 6.0 (Experiment 15): 4.0 - 6.0 = -2.0/step net,
    # identical margin to Experiment 12's 3.0 - 5.0 = -2.0/step. See
    # docs/superpowers/specs/2026-07-07-ar4-experiment15-reward-shaping-design.md.
    stillness_penalty = RewTerm(
        func=ar4_mdp.stillness_penalty,
        weight=6.0,
        params={
            "object_cfg": SceneEntityCfg("cube"),
            "jaw1_contact_cfg": SceneEntityCfg("gripper_jaw1_contact"),
            "jaw2_contact_cfg": SceneEntityCfg("gripper_jaw2_contact"),
            "force_threshold": 0.05,
            "still_bound": 0.005,
            "patience_steps": 25,
        },
    )

    # New (Experiment 15): existing function, never previously wired into
    # any RewardsCfg. Direct user request: "negative reward for contacting
    # the ground." weight=0.1 kept small since this fires almost every
    # step until any lift happens (unlike the running-max milestone
    # bonuses) - see the design spec's section 1 for the per-episode
    # magnitude reasoning.
    ground_penalty = RewTerm(
        func=ar4_mdp.ground_penalty,
        weight=0.1,
        params={
            "object_cfg": SceneEntityCfg("cube"),
            "ground_height_threshold": 0.015,
        },
    )

    # New (Experiment 15): new function (Task 1). Direct user request:
    # "negative reward for the cube contacting the base of the robot" -
    # explicitly a new function, distinct from ground_penalty (x/y
    # proximity to the robot's own base, not z-height). weight=0.1, same
    # magnitude reasoning as ground_penalty.
    base_proximity_penalty = RewTerm(
        func=ar4_mdp.base_proximity_penalty,
        weight=0.1,
        params={
            "object_cfg": SceneEntityCfg("cube"),
            "robot_cfg": SceneEntityCfg("robot"),
            "base_xy_threshold": 0.08,
        },
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class Ar4PickPlaceBaseProximityEnvCfg(ManagerBasedRLEnvCfg):
    """AR4 reward-shaping task (Experiment 15): same scene/action/events as
    Experiment 12's clean baseline, with ground_penalty and
    base_proximity_penalty newly wired into the reward, and
    antipodal_grasp_bonus/stillness_penalty weights raised in matched
    proportion. num_envs=4096 default (a real training-scale run) -
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

Run: `python3 -c "import ast; ast.parse(open('tasks/ar4/pickplace_baseproximity_env_cfg.py').read())"`
Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add tasks/ar4/pickplace_baseproximity_env_cfg.py
git commit -m "Add Ar4PickPlaceBaseProximityEnvCfg: ground/base-proximity penalties + higher grasp weight (Experiment 15)"
```

---

### Task 3: Wire `--baseproximity` into `scripts/train.py` and `scripts/eval_loop.py`, then smoke-test

**Files:**
- Modify: `scripts/train.py`
- Modify: `scripts/eval_loop.py`

**Interfaces:**
- Consumes: `Ar4PickPlaceBaseProximityEnvCfg` (Task 2), `Ar4PickPlaceTaskspacePPORunnerCfg` (pre-existing, already imported by both scripts for `--taskspace`/`--residual`/`--reachskip`).
- Produces: `--baseproximity` CLI flag on both scripts, verified via a headless 2-iteration smoke test.

- [ ] **Step 1: Add the flag and import to `scripts/train.py`**

Immediately after the existing `--reachskip` `parser.add_argument(...)` block (currently lines 82-93, ending right before `AppLauncher.add_app_launcher_args(parser)` at line 94), insert:

```python
parser.add_argument(
    "--baseproximity",
    action="store_true",
    default=False,
    help=(
        "Train on the reward-shaping variant of the task-space scene: adds a ground-contact penalty "
        "and a new cube-to-robot-base proximity penalty, and raises the antipodal grasp bonus's weight "
        "(with a matched stillness-penalty raise preserving the anti-freeze reward-rate margin), on top "
        "of Experiment 12's clean baseline reward. See "
        "docs/superpowers/specs/2026-07-07-ar4-experiment15-reward-shaping-design.md."
    ),
)
```

Add the import next to the existing `pickplace_reachskip_env_cfg` import (currently line 127):

```python
from tasks.ar4.pickplace_baseproximity_env_cfg import Ar4PickPlaceBaseProximityEnvCfg  # noqa: E402
```

Change the `env_cfg_cls` selection (currently lines 138-151) to add `--baseproximity` as the first branch:

```python
    if args_cli.baseproximity:
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

Change the `agent_cfg` selection (currently lines 156-159) so `--baseproximity` reuses the same already-verified PPO runner cfg:

```python
    if args_cli.taskspace or args_cli.residual or args_cli.reachskip or args_cli.baseproximity:
        agent_cfg = Ar4PickPlaceTaskspacePPORunnerCfg()
    else:
        agent_cfg = Ar4PickPlacePPORunnerCfg()
```

- [ ] **Step 2: Add the flag and import to `scripts/eval_loop.py`**

Immediately after the existing `--reachskip` `parser.add_argument(...)` block (currently lines 50-55, ending right before `AppLauncher.add_app_launcher_args(parser)` at line 56), insert:

```python
parser.add_argument(
    "--baseproximity",
    action="store_true",
    default=False,
    help="Evaluate the reward-shaping scene (see scripts/train.py --baseproximity) instead of the four-object scene.",
)
```

Add the import next to the existing `pickplace_reachskip_env_cfg` import (currently line 88):

```python
from tasks.ar4.pickplace_baseproximity_env_cfg import Ar4PickPlaceBaseProximityEnvCfg  # noqa: E402
```

Change the `env_cfg_cls` selection (currently lines 99-112) to add `--baseproximity` as the first branch:

```python
    if args_cli.baseproximity:
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

Change the `agent_cfg` selection (currently lines 117-120):

```python
    if args_cli.taskspace or args_cli.residual or args_cli.reachskip or args_cli.baseproximity:
        agent_cfg = Ar4PickPlaceTaskspacePPORunnerCfg()
    else:
        agent_cfg = Ar4PickPlacePPORunnerCfg()
```

And change the `name_prefix` selection (currently lines 138-149) to add `--baseproximity` as the first branch:

```python
        if args_cli.baseproximity:
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

This is the FIRST time `base_proximity_penalty`/`Ar4PickPlaceBaseProximityEnvCfg` will actually run inside Isaac Sim — Tasks 1-2 only had syntax checks. If it throws an exception, that's real information (a bug pure syntax-checking couldn't catch, e.g. a tensor-shape mismatch between `object.data.root_pos_w[:, :2]` and `robot.data.root_pos_w[:, :2]`), not something to route around — read the traceback and fix it.

Run (from repo root, foreground, allow up to 5 minutes):
```bash
rm -f /tmp/baseproximity_smoke_stdout.log
timeout 280 /home/saps/IsaacLab/isaaclab.sh -p scripts/train.py --baseproximity --num_envs 16 --max_iterations 2 --headless > /tmp/baseproximity_smoke_stdout.log 2>&1
echo "exit_code=$?"
```

A `timeout`/nonzero exit code alone is NOT proof of failure (Isaac Sim's shutdown sequence sometimes hangs after genuinely finishing) — verify via files:

```bash
grep -i "error\|exception\|traceback" /tmp/baseproximity_smoke_stdout.log
ls -la logs/train/ | tail -5
```

Find the newest `logs/train/<timestamp>/` directory and check:
```bash
find logs/train/<newest_timestamp_dir> -name "model_*.pt"
cat logs/train/<newest_timestamp_dir>/params/env.yaml | grep -A3 "ground_penalty\|base_proximity_penalty\|antipodal_grasp_bonus\|stillness_penalty"
```

Expected: `model_0.pt` and `model_1.pt` both exist, `env.yaml` confirms `Ar4PickPlaceBaseProximityEnvCfg`'s `RewardsCfg` is present with `ground_penalty` weight 0.1, `base_proximity_penalty` weight 0.1, `antipodal_grasp_bonus` weight 4.0, `stillness_penalty` weight 6.0, and no traceback in the stdout log. If an exception appears, the most likely culprit given this task's new code is a shape mismatch in `base_proximity_penalty` (`object.data.root_pos_w[:, :2]` and `robot.data.root_pos_w[:, :2]` should both be `(num_envs, 2)` — confirm via the traceback which tensor's shape is unexpected).

- [ ] **Step 5: Commit**

```bash
git add scripts/train.py scripts/eval_loop.py
git commit -m "Wire --baseproximity flag into train.py and eval_loop.py for Experiment 15"
```

---

### Task 4: Diagnostic run (300 iterations) — verify the new reward terms are stable before committing to a full run

**Files:**
- None created — this is a verification-only task. If the diagnostic looks wrong, stop and report; do not proceed to Task 5.

**Interfaces:**
- Consumes: `Ar4PickPlaceBaseProximityEnvCfg` (Task 2) via the `--baseproximity` flag (Task 3).
- Produces: a confirmed-healthy diagnostic run directory under `logs/train/`, gating Task 5.

- [ ] **Step 1: Launch the diagnostic run**

```bash
nohup /home/saps/IsaacLab/isaaclab.sh -p scripts/train.py --baseproximity --num_envs 4096 --max_iterations 300 --headless > /tmp/exp15_diagnostic_stdout.log 2>&1 &
echo "PID=$!"
```

- [ ] **Step 2: Poll for completion — BLOCK on this yourself, do not end your turn early**

Run this exact command (a real blocking loop — if one call's timeout is hit before the run finishes, re-issue the same command again):
```bash
until find logs/train/ -name "model_299.pt" -newer tasks/ar4/pickplace_baseproximity_env_cfg.py 2>/dev/null | grep -q .; do sleep 30; done
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
            'Episode_Reward/path_proximity_bonus', 'Episode_Reward/ground_penalty',
            'Episode_Reward/base_proximity_penalty', 'Episode_Termination/cube_reached_goal']:
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

1. **`Loss/value_function` stays bounded (no sustained exponential growth).** This experiment only changes reward weights/adds reward terms (no new action/reset mechanism touching physics state directly), a lower-risk change than Experiments 11/13/14's new mechanisms — but the diagnostic gate is this project's uniform standing practice for every experiment regardless of perceived risk. A small transient spike with immediate recovery is fine (matching prior experiments' own precedent); a sustained climb is not.
2. **No exceptions/tracebacks in `/tmp/exp15_diagnostic_stdout.log`.**
3. **Sanity check the two new terms fired at all**: `Episode_Reward/ground_penalty` and `Episode_Reward/base_proximity_penalty` should both show `nonzero` counts > 0 (confirms the terms are actually wired into the reward computation, not silently no-op'ing due to a params mismatch) — `ground_penalty` in particular should be nonzero on the large majority of the 300 iterations, since the cube starts every episode on the ground.

If all checks pass, proceed to Task 5. If any fails, stop, do not proceed, and report the finding instead.

---

### Task 5: Full 1500-iteration training run + TensorBoard verification report

**Files:**
- Create: `docs/superpowers/plans/2026-07-07-ar4-experiment15-report.md`

**Interfaces:**
- Consumes: the Task 4-verified reward configuration.
- Produces: a complete 1500-iteration training run under `logs/train/`, and a written report with full scalar trajectories.

- [ ] **Step 1: Launch the full training run**

```bash
nohup /home/saps/IsaacLab/isaaclab.sh -p scripts/train.py --baseproximity --num_envs 4096 --headless > /tmp/exp15_train_stdout.log 2>&1 &
echo "PID=$!"
```

- [ ] **Step 2: Poll for completion — BLOCK on this yourself, do not end your turn early**

```bash
until find logs/train/ -name "model_1499.pt" -newer tasks/ar4/pickplace_baseproximity_env_cfg.py 2>/dev/null | grep -q .; do sleep 60; done
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
        'Episode_Reward/ground_penalty', 'Episode_Reward/base_proximity_penalty',
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

Write `docs/superpowers/plans/2026-07-07-ar4-experiment15-report.md` following the structure of `docs/superpowers/plans/2026-07-07-ar4-experiment14-report.md` (checkpoint integrity, `Loss/value_function` sanity check, per-term summary + sampled trajectory for all 8 tags above). Include a "Key Comparison" section against **both Experiment 12's and Experiment 14's exact final values** (final-iteration snapshot vs. final-iteration snapshot only, per this project's established correction protocol):

- Experiment 12 final `Episode_Reward/antipodal_grasp_bonus`: 0.012777
- Experiment 12 final `Episode_Reward/stillness_penalty`: -0.001857
- Experiment 12 final `Episode_Reward/path_proximity_bonus`: 0.064421
- Experiment 12 final `Episode_Termination/cube_reached_goal`: 0.010773
- Experiment 14 final `Episode_Termination/cube_reached_goal`: 0.011393

Additionally, for the two new terms, report explicitly (per the design spec's success criteria) whether `Episode_Reward/ground_penalty`'s nonzero rate trends down across the run (comparing the first 150 iterations' nonzero rate against the last 150 iterations' nonzero rate — evidence the cube is spending less time on the ground as training progresses, not just accumulating a flat penalty the policy never learns to reduce) and whether `Episode_Reward/base_proximity_penalty`'s nonzero rate stays low across the whole run (report the overall nonzero rate; a low, stable rate is consistent with the term firing only for the specific base-collapse-adjacent cases it targets rather than fighting the cube's legitimate spawn distribution).

State the scalar comparison factually. **Do not draw a final success/failure conclusion from scalars alone** — per this project's own established lesson (Experiment 12's original report misread a scalar drop as failure and had to be corrected by the controller). Final judgment requires video inspection, done separately by the controller outside this plan (no Task 6 in this plan — matches this session's established pattern of the controller personally reviewing eval video and writing the ROADMAP entry after the plan's tasks complete).

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/plans/2026-07-07-ar4-experiment15-report.md
git commit -m "Record Experiment 15 training run: ground/base-proximity + higher grasp weight scalar trajectories"
```
