# Experiment 13 Implementation Plan: Residual RL over a Classical Waypoint-Seeking Base Controller

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Experiment 13 — replace the AR4 cube task-space action's raw-policy-only Cartesian delta with a residual: a bounded proportional ("seek") step toward the already-computed active waypoint (the classical base controller) plus the policy's own scaled output on top (the RL residual), per Silver et al. 2018 (arXiv:1812.06298) and Johannink et al. 2019 (ICRA)'s additive-superposition pattern — testing whether giving the policy a classical prior to correct, rather than asking it to discover the entire pick-and-place motion via reward shaping alone, unlocks the lift/carry/place behavior Experiments 11-12 didn't reach.

**Architecture:** New parallel task file `tasks/ar4/pickplace_residual_env_cfg.py` reusing `Ar4PickPlaceMirrorSceneCfg` (scene/cube/contact-sensors, unchanged) and Experiment 12's exact reward weights (unchanged), with a new `ResidualDifferentialIKActionCfg`/`ResidualDifferentialIKAction` pair (`tasks/ar4/residual_ik_action.py`) subclassing Isaac Lab's `DifferentialInverseKinematicsAction` and overriding only `process_actions()`.

**Tech Stack:** Isaac Lab `ManagerBasedRLEnvCfg`, `isaaclab.envs.mdp.actions.task_space_actions.DifferentialInverseKinematicsAction` (subclassed), PyTorch, rsl_rl PPO (`Ar4PickPlaceTaskspacePPORunnerCfg`, unchanged, reused from Experiment 11/12), TensorBoard event files for scalar verification.

## Global Constraints

- Do not modify `tasks/ar4/mdp.py`, `env_cfg.py`, `objects_cfg.py`, `pickplace_env_cfg.py`, `pickplace_mirror_env_cfg.py`, `pickplace_ik_guided_env_cfg.py`, or `pickplace_taskspace_env_cfg.py` — purely additive, per this repo's established per-experiment-file convention.
- Do not modify Isaac Lab's own source (`/home/saps/IsaacLab/...`) — subclass, don't patch.
- Reward weights must be copied verbatim from `pickplace_taskspace_env_cfg.py`'s current `RewardsCfg` (post-Experiment-12: `path_proximity_bonus` weight 25.0, `gripper_schedule_bonus` weight 0.1, `antipodal_grasp_bonus` weight 3.0 with `antipodal_cos_threshold=-0.7071`, `stillness_penalty` weight **5.0** with `patience_steps=25`/`still_bound=0.005`, `action_rate` weight -1e-4, `joint_vel` weight -1e-4) — this experiment isolates the action-term variable alone.
- Decided values (verbatim, not placeholders): `_BASE_MAX_STEP = 0.05` (base controller's max per-step pursuit distance in meters), action `scale=0.05` (residual's own scale, unchanged from Experiment 11/12), `body_name="link_6"`, `body_offset.pos=_EE_OFFSET=(0.0, 0.0, 0.036)`, `command_type="position"`, `use_relative_mode=True`, `ik_method="dls"`.
- Reuse `Ar4PickPlaceTaskspacePPORunnerCfg` unchanged (`clip_actions=5.0`) — no new PPO hyperparameters, no new PPORunnerCfg subclass in this experiment.
- `num_envs=4096` for the real training run.
- Always invoke scripts via `/home/saps/IsaacLab/isaaclab.sh -p scripts/<script>.py` (or `/home/saps/IsaacLab/isaaclab.sh -p -c "<snippet>"` for one-off Python) from the repo root, never plain `python`.
- Verify completion via files (checkpoint counts, `model_<N>.pt` existence, event-file mtimes) and actual TensorBoard scalar data, not console text or exit codes alone.
- **Any subagent dispatched to launch a training run must be given the literal blocking poll command up front** (e.g. `until find <path> -name <checkpoint> 2>/dev/null | grep -q .; do sleep 30; done`), not just told to "poll" — prior subagents in this session have repeatedly defaulted to "wait for a background notification" that never comes, even when warned in prose only.

---

### Task 1: `ResidualDifferentialIKAction` + `ResidualDifferentialIKActionCfg`

**Files:**
- Create: `tasks/ar4/residual_ik_action.py`

**Interfaces:**
- Consumes: `isaaclab.envs.mdp.actions.task_space_actions.DifferentialInverseKinematicsAction` (Isaac Lab base class — verified present at that exact import path; NOT re-exported via `isaaclab.envs.mdp`'s wildcard import, must be imported from the submodule directly), `isaaclab.managers.ActionTerm` (verified exported at `isaaclab.managers` top level), `isaaclab.envs.mdp.DifferentialInverseKinematicsActionCfg` (verified exported via `isaaclab.envs.mdp`, already used the same way in `pickplace_taskspace_env_cfg.py`), `isaaclab.utils.math.subtract_frame_transforms` (already used identically in `tasks/ar4/mdp.py`'s `ik_guided_path_bonus`), `env._path_waypoints_w` / `env._path_waypoint_idx` (lazily created and per-episode reset by the existing, unmodified `compute_path_waypoints` event, `tasks/ar4/mdp.py:372-422`).
- Produces: `ResidualDifferentialIKAction` (ActionTerm subclass) and `ResidualDifferentialIKActionCfg` (its paired Cfg, `class_type=ResidualDifferentialIKAction`) — consumed by Task 2's `ActionsCfg`.

- [ ] **Step 1: Write the new file**

```python
# tasks/ar4/residual_ik_action.py
"""Residual-over-classical-controller action term (Experiment 13): adds a
bounded proportional ("seek") step toward the currently-active waypoint
(env._path_waypoints_w[env._path_waypoint_idx], the same 5-waypoint path
compute_path_waypoints already computes and path_proximity_bonus/
ik_guided_path_bonus already track) to the policy's own scaled raw action,
before handing the combined Cartesian delta to the same live differential-
IK controller every other task-space action term already uses. Additive
superposition (base + residual), per Silver et al. 2018 "Residual Policy
Learning" (arXiv:1812.06298) and Johannink et al. 2019 "Residual
Reinforcement Learning for Robot Control" (ICRA). See
docs/superpowers/specs/2026-07-07-ar4-residual-ik-action-design.md.

Import this module only after an Isaac Sim/Isaac Lab AppLauncher has been
created (matches every other tasks/ar4/*.py module in this repo).
"""

import torch

import isaaclab.envs.mdp as isaaclab_mdp
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.managers import ActionTerm
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import subtract_frame_transforms

_BASE_MAX_STEP = 0.05
"""Max per-step Cartesian pursuit distance (meters) the base controller
contributes toward the active waypoint - deliberately identical to
ActionsCfg's own scale=0.05 for the policy's raw-action contribution, so
base and residual are comparably-sized (neither dominates by
construction). See design spec's "Design" section."""


class ResidualDifferentialIKAction(DifferentialInverseKinematicsAction):
    """Same as DifferentialInverseKinematicsAction, except process_actions()
    adds a bounded pursuit step toward the active waypoint to the policy's
    own scaled action, instead of using the policy's action alone as the
    full commanded delta. apply_actions() (the actual IK solve + joint
    command) is inherited unchanged - only the Cartesian delta fed into it
    changes."""

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        base_delta = self._compute_base_delta()
        self._processed_actions[:] = base_delta + self.raw_actions * self._scale
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        self._ik_controller.set_command(self._processed_actions, ee_pos_curr, ee_quat_curr)

    def _compute_base_delta(self) -> torch.Tensor:
        """Bounded proportional ("seek") step toward the currently-active
        waypoint, in the same body-frame convention process_actions expects
        (command_type="position", use_relative_mode=True - the controller
        adds this delta to the CURRENT ee pose each step). Returns zeros
        before the first reset event (compute_path_waypoints) has run, since
        env._path_waypoints_w doesn't exist yet at that point - matches the
        identical defensive pattern path_proximity_bonus/ik_guided_path_bonus
        already use in tasks/ar4/mdp.py."""
        env = self._env
        if not hasattr(env, "_path_waypoints_w"):
            return torch.zeros(self.num_envs, 3, device=self.device)
        current_waypoint_w = torch.gather(
            env._path_waypoints_w, 1, env._path_waypoint_idx.view(-1, 1, 1).expand(-1, 1, 3)
        ).squeeze(1)
        root_pose_w = self._asset.data.root_pose_w
        target_b, _ = subtract_frame_transforms(root_pose_w[:, 0:3], root_pose_w[:, 3:7], current_waypoint_w)
        ee_pos_curr, _ = self._compute_frame_pose()
        direction = target_b - ee_pos_curr
        dist = torch.norm(direction, dim=-1, keepdim=True)
        step = torch.clamp(dist, max=_BASE_MAX_STEP)
        return direction / (dist + 1e-8) * step


@configclass
class ResidualDifferentialIKActionCfg(isaaclab_mdp.DifferentialInverseKinematicsActionCfg):
    """Same fields as DifferentialInverseKinematicsActionCfg - only
    class_type differs, pointing to ResidualDifferentialIKAction instead of
    the base DifferentialInverseKinematicsAction. @configclass is REQUIRED
    here (not optional/cosmetic): configclass wraps dataclasses.dataclass,
    which regenerates __init__ with fresh field defaults at decoration
    time - without re-decorating this subclass, the inherited __init__
    would silently keep the PARENT's original class_type default
    (DifferentialInverseKinematicsAction), not this override, with no
    exception. Caught by Task 1's review (Experiment 13); every other
    class_type-overriding *ActionCfg in Isaac Lab's own actions_cfg.py is
    @configclass-decorated for exactly this reason."""

    class_type: type[ActionTerm] = ResidualDifferentialIKAction
```

- [ ] **Step 2: Syntax-check the file**

Run: `python3 -c "import ast; ast.parse(open('tasks/ar4/residual_ik_action.py').read())"`
Expected: no output (parses cleanly). This is a pure syntax check — it does not require Isaac Sim, unlike actually importing the module (which needs an AppLauncher first, per this file's own docstring).

- [ ] **Step 3: Commit**

```bash
git add tasks/ar4/residual_ik_action.py
git commit -m "Add ResidualDifferentialIKAction: base+residual action term (Experiment 13)"
```

---

### Task 2: New residual task-space env cfg file

**Files:**
- Create: `tasks/ar4/pickplace_residual_env_cfg.py`

**Interfaces:**
- Consumes: `Ar4PickPlaceMirrorSceneCfg` (from `tasks/ar4/pickplace_mirror_env_cfg.py`, unmodified), `ResidualDifferentialIKActionCfg` (Task 1), `ar4_mdp.path_proximity_bonus` / `ar4_mdp.gripper_schedule_bonus` / `ar4_mdp.antipodal_grasp_bonus` / `ar4_mdp.stillness_penalty` / `ar4_mdp.set_mirrored_goal` / `ar4_mdp.compute_path_waypoints` / `ar4_mdp.mirrored_target_position_in_robot_root_frame` / `ar4_mdp.object_reached_mirrored_goal` (all pre-existing, unmodified, from `tasks/ar4/mdp.py`), `_EE_OFFSET` (from `tasks/ar4/pickplace_env_cfg.py`), `ARM_JOINT_NAMES` / `GRIPPER_JOINT_NAMES` / `GRIPPER_OPEN_POS` / `GRIPPER_CLOSED_POS` (from `tasks/ar4/robot_cfg.py`).
- Produces: `Ar4PickPlaceResidualEnvCfg` class — consumed by Task 3 (script wiring). Note: this experiment does NOT define its own PPORunnerCfg subclass — Task 3 imports `Ar4PickPlaceTaskspacePPORunnerCfg` from `tasks/ar4/pickplace_taskspace_env_cfg.py` directly (already-verified `clip_actions=5.0` fix, reused unchanged).

- [ ] **Step 1: Write the new file**

```python
# tasks/ar4/pickplace_residual_env_cfg.py
"""Residual-action variant of the AR4 mirror-goal cube task (Experiment
13): identical scene/spawn-randomization/mirrored-goal/reward as
pickplace_taskspace_env_cfg.py, but the arm's action space uses
ResidualDifferentialIKActionCfg (a bounded pursuit step toward the active
waypoint, i.e. a classical base controller, plus the policy's own scaled
action as an RL residual on top) instead of plain
DifferentialInverseKinematicsActionCfg (policy action only). See
docs/superpowers/specs/2026-07-07-ar4-residual-ik-action-design.md.

Additive/parallel to every other pickplace_*.py file: does NOT modify
env_cfg.py, objects_cfg.py, pickplace_env_cfg.py, pickplace_mirror_env_cfg.py,
pickplace_ik_guided_env_cfg.py, or pickplace_taskspace_env_cfg.py. Reuses
Ar4PickPlaceMirrorSceneCfg directly (same cube scene, same contact
sensors, same ee_frame) and Ar4PickPlaceTaskspacePPORunnerCfg directly
(same clip_actions=5.0 fix, no new PPO hyperparameters) - only the arm
action term and this file's own name/docstrings differ from
pickplace_taskspace_env_cfg.py.

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
from .residual_ik_action import ResidualDifferentialIKActionCfg
from .robot_cfg import ARM_JOINT_NAMES, GRIPPER_CLOSED_POS, GRIPPER_JOINT_NAMES, GRIPPER_OPEN_POS

# Same values as pickplace_taskspace_env_cfg.py's EventCfg reuse.
_LIFT_MINIMAL_HEIGHT = 0.03
_PREGRASP_HOVER = 0.05
_LIFT_MARGIN = 0.02
_CARRY_HEIGHT = 0.10


@configclass
class ActionsCfg:
    """Task-space action specifications: the arm is controlled via a
    RESIDUAL Cartesian action - a bounded pursuit step toward the active
    waypoint (classical base controller, ResidualDifferentialIKAction's own
    _compute_base_delta) plus the policy's own scaled raw action (RL
    residual), summed before being converted to joint targets by the same
    live differential-IK solver every other task-space action term uses.
    Contrast pickplace_taskspace_env_cfg.py's ActionsCfg, where the policy's
    raw action is the ENTIRE commanded delta with no base controller.

    All fields identical to pickplace_taskspace_env_cfg.py's arm_action
    (body_name="link_6", body_offset=_EE_OFFSET, scale=0.05,
    command_type="position", use_relative_mode=True, ik_method="dls") -
    only the action term CLASS differs (ResidualDifferentialIKActionCfg
    instead of isaaclab_mdp.DifferentialInverseKinematicsActionCfg)."""

    arm_action = ResidualDifferentialIKActionCfg(
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
    """Identical to pickplace_taskspace_env_cfg.py's ObservationsCfg (same
    scene, same goal mechanism); the action term's internals changed but
    these observation functions read joint/object state directly, not the
    action term, so they remain valid unmodified."""

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
    """Identical to pickplace_taskspace_env_cfg.py's EventCfg, in
    registration order:
    1. reset_all - whole scene back to default.
    2. reset_cube_position - randomize the cube across the full workspace.
    3. randomize_goal - reads the cube's now-updated position, sets the mirrored goal.
    4. compute_path_waypoints - reads both the cube's position and the
       goal, computes the 5-waypoint path, and resets path-progress state
       (including env._path_waypoints_w/env._path_waypoint_idx, now also
       read every step by ResidualDifferentialIKAction's _compute_base_delta,
       not just by path_proximity_bonus)."""

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
    """Identical weights to pickplace_taskspace_env_cfg.py's RewardsCfg
    (post-Experiment-12: stillness_penalty weight 5.0) - this experiment
    isolates the action-term variable alone, reward function unchanged."""

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
class Ar4PickPlaceResidualEnvCfg(ManagerBasedRLEnvCfg):
    """AR4 residual-action task: same scene/spawn/goal/rewards as the
    taskspace task, but the arm's action is a classical waypoint-seeking
    base controller plus the policy's own action as a residual correction,
    instead of the policy's action alone. num_envs=4096 default (a real
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

Run: `python3 -c "import ast; ast.parse(open('tasks/ar4/pickplace_residual_env_cfg.py').read())"`
Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add tasks/ar4/pickplace_residual_env_cfg.py
git commit -m "Add Ar4PickPlaceResidualEnvCfg: residual action over classical waypoint controller (Experiment 13)"
```

---

### Task 3: Wire `--residual` into `scripts/train.py` and `scripts/eval_loop.py`, then smoke-test

**Files:**
- Modify: `scripts/train.py`
- Modify: `scripts/eval_loop.py`

**Interfaces:**
- Consumes: `Ar4PickPlaceResidualEnvCfg` (Task 2), `Ar4PickPlaceTaskspacePPORunnerCfg` (pre-existing, from `tasks/ar4/pickplace_taskspace_env_cfg.py`, already imported by both scripts for `--taskspace`).
- Produces: `--residual` CLI flag on both scripts, verified via a headless 2-iteration smoke test writing result files.

- [ ] **Step 1: Add the flag and import to `scripts/train.py`**

In `scripts/train.py`, immediately after the existing `--taskspace` `parser.add_argument(...)` block (currently lines 60-70, ending right before `AppLauncher.add_app_launcher_args(parser)` at line 71), insert:

```python
parser.add_argument(
    "--residual",
    action="store_true",
    default=False,
    help=(
        "Train on the residual-action variant of the task-space scene: the arm's action is a bounded "
        "pursuit step toward the active waypoint (classical base controller) plus the policy's own "
        "scaled action (RL residual) on top, instead of the policy's action alone. See "
        "docs/superpowers/specs/2026-07-07-ar4-residual-ik-action-design.md."
    ),
)
```

Then add the import next to the existing `pickplace_taskspace_env_cfg` import (currently lines 104-107):

```python
from tasks.ar4.pickplace_residual_env_cfg import Ar4PickPlaceResidualEnvCfg  # noqa: E402
```

Then change the `env_cfg_cls` selection (currently lines 113-122):

```python
    if args_cli.residual:
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

Then change the `agent_cfg` selection (currently lines 127-130) so `--residual` reuses the same already-verified PPO runner cfg as `--taskspace`:

```python
    if args_cli.taskspace or args_cli.residual:
        agent_cfg = Ar4PickPlaceTaskspacePPORunnerCfg()
    else:
        agent_cfg = Ar4PickPlacePPORunnerCfg()
```

- [ ] **Step 2: Add the flag and import to `scripts/eval_loop.py`**

In `scripts/eval_loop.py`, immediately after the existing `--taskspace` `parser.add_argument(...)` block (currently lines 38-43, ending right before `AppLauncher.add_app_launcher_args(parser)` at line 44), insert:

```python
parser.add_argument(
    "--residual",
    action="store_true",
    default=False,
    help="Evaluate the residual-action scene (see scripts/train.py --residual) instead of the four-object scene.",
)
```

Add the import next to the existing `pickplace_taskspace_env_cfg` import (currently lines 76-79):

```python
from tasks.ar4.pickplace_residual_env_cfg import Ar4PickPlaceResidualEnvCfg  # noqa: E402
```

Change the `env_cfg_cls` selection (currently lines 85-94):

```python
    if args_cli.residual:
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

Change the `agent_cfg` selection (currently lines 99-102):

```python
    if args_cli.taskspace or args_cli.residual:
        agent_cfg = Ar4PickPlaceTaskspacePPORunnerCfg()
    else:
        agent_cfg = Ar4PickPlacePPORunnerCfg()
```

And change the `name_prefix` selection (currently lines 120-127):

```python
        if args_cli.residual:
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

Run (from repo root, foreground, allow up to 5 minutes):
```bash
rm -f /tmp/residual_smoke_stdout.log
timeout 280 /home/saps/IsaacLab/isaaclab.sh -p scripts/train.py --residual --num_envs 16 --max_iterations 2 --headless > /tmp/residual_smoke_stdout.log 2>&1
echo "exit_code=$?"
```

Expected: the command may hit the `timeout` (Isaac Sim's clean-shutdown hang is a known false-negative pattern in this repo — do NOT treat a nonzero/timeout exit code alone as failure). Instead verify via files:

```bash
ls -la logs/train/ | tail -5
```

Find the newest `logs/train/<timestamp>/` directory (created after this command started) and check:
```bash
find logs/train/<newest_timestamp_dir> -name "model_*.pt"
grep -o "path_proximity_bonus\|gripper_schedule_bonus\|antipodal_grasp_bonus\|stillness_penalty\|action_rate\|joint_vel" /tmp/residual_smoke_stdout.log | sort -u
```

Expected: `model_0.pt` and `model_1.pt` both exist, and all 6 reward term names appear (confirming `RewardsCfg` wired correctly with no exceptions during construction or the reward computation step). If any exception/traceback appears in `/tmp/residual_smoke_stdout.log`, that IS a real failure — read the traceback and fix before proceeding. Likely culprits specific to this task's new code: (a) `DifferentialInverseKinematicsAction` not importable from `isaaclab.envs.mdp.actions.task_space_actions` (re-verify the exact module path against `/home/saps/IsaacLab/source/isaaclab/isaaclab/envs/mdp/actions/task_space_actions.py` if this happens — it was confirmed present there during design, but re-check if the import fails); (b) `self._env` not available inside `_compute_base_delta()` (verify `ActionTerm.__init__` sets `self._env`, confirmed at `isaaclab/managers/action_manager.py:54` during design); (c) a shape mismatch in `_compute_base_delta()`'s tensor ops (add `print(direction.shape, dist.shape, step.shape)` temporarily inside the method, re-run the smoke test, and remove the print once the shapes are confirmed `(num_envs, 3)`/`(num_envs, 1)`/`(num_envs, 1)` respectively).

- [ ] **Step 5: Commit**

```bash
git add scripts/train.py scripts/eval_loop.py
git commit -m "Wire --residual flag into train.py and eval_loop.py for Experiment 13"
```

---

### Task 4: Diagnostic run (300 iterations) — verify the new action term is stable before committing to a full run

**Files:**
- None created — this is a verification-only task. If the diagnostic looks wrong, stop and report; do not proceed to Task 5.

**Interfaces:**
- Consumes: `Ar4PickPlaceResidualEnvCfg` (Task 2) via the `--residual` flag (Task 3).
- Produces: a confirmed-healthy diagnostic run directory under `logs/train/`, gating Task 5.

- [ ] **Step 1: Launch the diagnostic run**

Run (from repo root, background):
```bash
nohup /home/saps/IsaacLab/isaaclab.sh -p scripts/train.py --residual --num_envs 4096 --max_iterations 300 --headless > /tmp/exp13_diagnostic_stdout.log 2>&1 &
echo "PID=$!"
```

- [ ] **Step 2: Poll for completion — BLOCK on this yourself, do not end your turn early**

Run this exact command (a real blocking loop, not a background job you then wait on separately — if one call's timeout isn't enough, re-issue the same command again):
```bash
until find logs/train/ -name "model_299.pt" -newer tasks/ar4/pickplace_residual_env_cfg.py 2>/dev/null | grep -q .; do sleep 30; done
echo "diagnostic complete"
```
This repo's diagnostic runs for a comparable env/iteration count have taken roughly 3-5 minutes real wall-clock time in prior experiments — do not assume failure before at least 10 minutes have elapsed, and re-issue the blocking command again if your tool call's own timeout is hit before then.

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

1. **`Loss/value_function` stays bounded (no sustained exponential growth).** This is the critical check for this task specifically: Experiment 11 hit a real critic-divergence bug the first time a *new* differential-IK-based action term was introduced (`Mean value_function loss` exploding from ~0 to ~5.2e23 over the run), traced to an outlier raw policy action driving the IK solve into a discontinuous jump. `ResidualDifferentialIKAction` is a new action term of the same general family (differential-IK-driven), so this exact risk applies again — check the full 300-point trajectory for any point exceeding roughly 100x its neighbors that doesn't immediately recover within 1-2 iterations (a small transient spike immediately recovering, as seen in Experiment 11's *fixed* run, max 7.88, is fine; a sustained climb is not). If this fails, the likely fix (do not implement yet, just report it) is the same `clip_actions` mechanism already proven in Experiment 11/12 — but note `Ar4PickPlaceTaskspacePPORunnerCfg` is already reused here with `clip_actions=5.0`, so if divergence still occurs despite that, this is a **new** finding (the residual's base-controller contribution isn't itself bounded by `clip_actions`, which only clips the raw policy output before scaling — the base delta is added on top and is bounded to `_BASE_MAX_STEP=0.05` by construction, so this would suggest the interaction between the two is the problem, not either alone).
2. **No exceptions/tracebacks in `/tmp/exp13_diagnostic_stdout.log`.** Run `grep -i "error\|exception\|traceback" /tmp/exp13_diagnostic_stdout.log` — if this finds anything beyond expected Isaac Sim startup/shutdown noise, read the context and fix before proceeding.

If both checks pass, proceed to Task 5. If either fails, stop, do not proceed to Task 5, and report the finding instead (do not attempt a fix within this task — report back to the controller, matching Experiment 12's Task 2 gate precedent of treating a genuine failure as a result to report, not a blocker to silently patch around).

---

### Task 5: Full 1500-iteration training run + TensorBoard verification report

**Files:**
- Create: `docs/superpowers/plans/2026-07-07-ar4-experiment13-report.md`

**Interfaces:**
- Consumes: the Task 4-verified action term.
- Produces: a complete 1500-iteration training run under `logs/train/`, and a written report with full scalar trajectories.

- [ ] **Step 1: Launch the full training run**

```bash
nohup /home/saps/IsaacLab/isaaclab.sh -p scripts/train.py --residual --num_envs 4096 --headless > /tmp/exp13_train_stdout.log 2>&1 &
echo "PID=$!"
```

- [ ] **Step 2: Poll for completion — BLOCK on this yourself, do not end your turn early**

```bash
until find logs/train/ -name "model_1499.pt" -newer tasks/ar4/pickplace_residual_env_cfg.py 2>/dev/null | grep -q .; do sleep 60; done
echo "full run complete"
```
Based on this repo's prior 1500-iteration runs at `num_envs=4096`, expect roughly 15-20 minutes of real wall-clock time (this machine's GPU has run comparable 1500-iteration jobs in that window before) — but treat that as a rough guide, not a hard cutoff; keep re-issuing the blocking command if a single tool call's own timeout is hit before the checkpoint appears, rather than assuming failure.

Once found, confirm checkpoint integrity:
```bash
LATEST=$(ls -dt logs/train/*/ | head -1)
find "$LATEST" -name "model_*.pt" | wc -l
ls -la "${LATEST}"events.out.tfevents.*
```

Expected: 31 checkpoints (0, 50, 100, ..., 1450, 1499 — `save_interval=50`, matching every prior experiment this session), `model_1499.pt` exists, event file mtime matches the run's actual completion time.

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

Write `docs/superpowers/plans/2026-07-07-ar4-experiment13-report.md` following the structure of `docs/superpowers/plans/2026-07-07-ar4-experiment12-report.md` (checkpoint integrity, `Loss/value_function` sanity check, per-term summary + sampled trajectory for all 5 reward/termination tags above). Include a "Key Comparison" section against **Experiment 12's exact final values** (final-iteration snapshot vs. final-iteration snapshot only, per this project's established correction protocol against cumulative-vs-single-episode comparison errors):

- Experiment 12 final `Episode_Reward/antipodal_grasp_bonus`: 0.012777
- Experiment 12 final `Episode_Reward/stillness_penalty`: -0.001857
- Experiment 12 final `Episode_Reward/path_proximity_bonus`: 0.064421
- Experiment 12 final `Episode_Termination/cube_reached_goal`: 0.010773

State the scalar comparison plainly but **do not draw a final success/failure conclusion from scalars alone** — per Experiment 12's own corrected report, scalar comparisons in this task have already proven ambiguous/misleading in isolation (a proxy-term drop was previously misread as failure when the underlying behavior may have genuinely improved). That conclusion requires Task 6's video inspection.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/plans/2026-07-07-ar4-experiment13-report.md
git commit -m "Record Experiment 13 training run: residual action scalar trajectories"
```

---

### Task 6: Real eval + multi-episode video inspection, ROADMAP record

**Files:**
- Modify: `ROADMAP.md` (append Experiment 13's outcome, following the existing format for Experiments 9-12)

**Interfaces:**
- Consumes: `model_1499.pt` from Task 5's training run.

- [ ] **Step 1: Run eval with video recording**

```bash
LATEST=$(ls -dt logs/train/*/ | head -1)
/home/saps/IsaacLab/isaaclab.sh -p scripts/eval_loop.py --residual --checkpoint "${LATEST}model_1499.pt" --episodes 10
```

Verify 10 output video files exist under `logs/videos/` (named `ar4_pickplace_residual-step-*.mp4`, one per episode per this repo's `RecordVideo` `step_trigger`/`video_length=250` convention — confirmed in Experiment 12's Task 4 that this produces one file per episode, not a merged multi-episode file) before treating this as complete.

- [ ] **Step 2: Extract and inspect frames from AT LEAST 3 of the 10 episodes**

Per Experiment 12's own finding: with a ~1% `cube_reached_goal` training-time success rate, a single inspected episode is not a representative sample. Extract frames from at least 3 episodes (e.g. `ar4_pickplace_residual-step-0.mp4`, `-step-250.mp4`, `-step-500.mp4`):

```bash
mkdir -p /tmp/exp13_eval_frames_ep1 /tmp/exp13_eval_frames_ep2 /tmp/exp13_eval_frames_ep3
ffmpeg -y -i logs/videos/ar4_pickplace_residual-step-0.mp4 -vf fps=5 /tmp/exp13_eval_frames_ep1/frame_%03d.png
ffmpeg -y -i logs/videos/ar4_pickplace_residual-step-250.mp4 -vf fps=5 /tmp/exp13_eval_frames_ep2/frame_%03d.png
ffmpeg -y -i logs/videos/ar4_pickplace_residual-step-500.mp4 -vf fps=5 /tmp/exp13_eval_frames_ep3/frame_%03d.png
```

Inspect the extracted frames directly (via the Read tool or equivalent — not a text description of what a video player shows), checking specifically for: (1) does the cube visibly leave the ground and stay elevated for more than an instantaneous contact; (2) does the arm move the cube toward the goal after any lift; (3) how does the arm's motion compare qualitatively to Experiment 12's video (more direct/assertive toward each waypoint, given the base controller's pursuit-step contribution, or similar). If frames are ambiguous about a cube's position relative to the gripper at the render resolution, crop and upscale the gripper region (e.g. via `PIL.Image.crop()` + `.resize()` with `LANCZOS` resampling) before concluding — this was necessary in Experiment 12's own inspection to distinguish a held grasp from an empty gripper at default zoom.

- [ ] **Step 3: Record the outcome in `ROADMAP.md`**

Append a new entry after the Experiment 12 entry, in the same format used for Experiments 9-12 (hypothesis, what changed, literature citations, quantitative result from Task 5's report, qualitative multi-episode video finding from Step 2, and an explicit statement of whether "grasp/lift never emerges" / "pick up and move" is now resolved, improved-but-unresolved, unchanged, or regressed — plus an explicit note on sample size limitations if the video evidence remains ambiguous, matching Experiment 12's own honesty standard on this point).

- [ ] **Step 4: Commit and push**

```bash
git add ROADMAP.md
git commit -m "Record Experiment 13 outcome (residual action over classical waypoint controller) in ROADMAP"
git push origin main
```
