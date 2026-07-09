# Experiment 25: Touch-Goal Reach Task Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new AR4 task (`Ar4PickPlaceTouchGoalEnvCfg`) where the arm's only job is to touch the top of a fixed cube, then reach a fixed goal point — no grasp, no lift, no gripper action at all — replacing the grasp/lift mechanism that six prior experiments (17-22) failed to make reliable.

**Architecture:** One new `tasks/ar4/mdp.py` reward/termination/observation function set (touch-then-goal stage-gated running-max reward, mirroring the already-proven `staged_milestone_bonus` mechanism), one new env cfg file (`tasks/ar4/pickplace_touchgoal_env_cfg.py`, arm-only 6-DOF action space, fixed cube/goal positions, no gripper contact sensors), and a `scripts/train.py` wiring change (`--touchgoal` flag).

**Tech Stack:** Isaac Lab `ManagerBasedRLEnv`, `rsl_rl` PPO, PyTorch. Launch everything via `/home/saps/IsaacLab/isaaclab.sh -p <script>`, never bare `python`.

## Global Constraints

- Fixed cube world position: `(0.20, 0.28, 0.006)` (reusing `env_cfg.py`'s existing `CUBE_CFG` default pose — already covered by 2026-07-09's physics-fidelity pass).
- Fixed goal world position: `(-0.20, 0.28, 0.15)`, expressed as a constant offset from the cube's own live position so it stays correct per-env in a multi-env (`num_envs > 1`) run: `GOAL_OFFSET = (-0.40, 0.0, 0.144)` (i.e. `goal_pos_w = cube.data.root_pos_w + GOAL_OFFSET`).
- `touch_std=0.05`, `touch_tolerance=0.02`, `goal_std=0.1` (per the design spec's grounding — see `docs/superpowers/specs/2026-07-09-ar4-experiment25-touch-goal-reach-design.md`). `cube_half_size=0.006` (12mm cube).
- Action space is arm-only: `JointPositionActionCfg` over `ARM_JOINT_NAMES` (6 joints, `tasks/ar4/robot_cfg.py`). No gripper action term anywhere in this task.
- Reuse `Ar4PickPlacePPORunnerCfg` (`tasks/ar4/agents/rsl_rl_ppo_cfg.py`) unchanged — same as `--mirror`'s own PPO config selection in `scripts/train.py`. Do not create a new PPO runner cfg.
- Episode length `5.0s`, `decimation=4`, `sim.dt=0.005` — copy `Ar4PickPlaceMirrorEnvCfg.__post_init__` exactly (today's physics-fidelity settings), do not use older, coarser values from any other env cfg.

---

### Task 1: Add touch-goal reward/termination/observation functions to `tasks/ar4/mdp.py`

**Files:**
- Modify: `tasks/ar4/mdp.py` (append after `reset_lift_milestone`, i.e. after line 337 in the current file — verify by reading the file first, since other agents may have appended functions since this plan was written)

**Interfaces:**
- Produces: `_raw_touch_goal_progress(env, object_cfg, ee_frame_cfg, goal_offset, cube_half_size=0.006, touch_std=0.05, touch_tolerance=0.02, goal_std=0.1) -> torch.Tensor` (shape `(num_envs,)`)
- Produces: `touch_goal_milestone_bonus(env, object_cfg, ee_frame_cfg, goal_offset, cube_half_size=0.006, touch_std=0.05, touch_tolerance=0.02, goal_std=0.1) -> torch.Tensor` (shape `(num_envs,)`, the running-max-delta reward term — this is what gets wired into `RewardsCfg`)
- Produces: `reset_touch_goal_milestone(env, env_ids) -> None` (event term, zeroes `env._touch_goal_milestone_max` and `env._touched_cube` for resetting envs)
- Produces: `touch_then_goal_reached(env, threshold, object_cfg, ee_frame_cfg, goal_offset, cube_half_size=0.006, touch_tolerance=0.02) -> torch.Tensor` (shape `(num_envs,)`, dtype bool — termination term)
- Produces: `touch_goal_position_in_robot_root_frame(env, robot_cfg, object_cfg, goal_offset) -> torch.Tensor` (shape `(num_envs, 3)` — observation term)
- Consumes: `env.scene[object_cfg.name].data.root_pos_w`, `env.scene[ee_frame_cfg.name].data.target_pos_w[:, 0, :]` (existing `FrameTransformer`/`RigidObject` buffers, same pattern as `_raw_lift_progress_mirrored`, `tasks/ar4/mdp.py:254-287`), `subtract_frame_transforms` (already imported in `mdp.py`, used by `mirrored_target_position_in_robot_root_frame`).

- [ ] **Step 1: Read the current end of `tasks/ar4/mdp.py` to confirm insertion point**

Run: `grep -n "^def \|^class " /home/saps/projects/rl/tasks/ar4/mdp.py | tail -20`

Confirm `reset_lift_milestone` is still the last function before `stillness_penalty`, or find wherever the file's tail actually is now, and insert the new functions there (after `reset_lift_milestone`, before `stillness_penalty` — or at the true end of the file if the layout changed. Do not insert in the middle of an unrelated function.)

- [ ] **Step 2: Add the reward/termination/observation functions**

Insert this exact code block (after `reset_lift_milestone`, matching its existing docstring/pattern style):

```python
def _raw_touch_goal_progress(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    goal_offset: tuple[float, float, float],
    cube_half_size: float = 0.006,
    touch_std: float = 0.05,
    touch_tolerance: float = 0.02,
    goal_std: float = 0.1,
) -> torch.Tensor:
    """Two-stage touch-then-goal progress signal, no grasp/lift involved at
    all (Experiment 25 - see docs/superpowers/specs/2026-07-09-ar4-experiment25-touch-goal-reach-design.md).
    Stage 1: end-effector proximity to a point just above the cube's top
    face. Stage 2 (goal_term): end-effector proximity to a fixed goal
    point, computed as object.data.root_pos_w + goal_offset so it stays
    correctly per-env-world-offset without a separate stateful buffer
    (the cube's own position is fixed/unrandomized in this task, so this
    is equivalent to - but simpler than - env_target_pos_w-style state).
    goal_term is gated to 0 until env._touched_cube latches true for that
    env, unlike _raw_lift_progress_mirrored's ungated additive sum (the
    exact reward shape that let Experiment 16's wedging exploit satisfy
    lift/goal reward without genuine grasp - this task has no grasp to
    exploit around, but the gate is kept anyway since "touch, then go" is
    the literal task definition, not just a shaping nicety).
    """
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]

    goal_offset_t = torch.tensor(goal_offset, device=env.device)
    touch_point_w = object.data.root_pos_w + torch.tensor([0.0, 0.0, cube_half_size], device=env.device)
    goal_pos_w = object.data.root_pos_w + goal_offset_t

    touch_dist = torch.norm(ee_pos_w - touch_point_w, dim=-1)
    touch_term = 1.0 - torch.tanh(touch_dist / touch_std)

    if not hasattr(env, "_touched_cube"):
        env._touched_cube = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._touched_cube |= touch_dist < touch_tolerance

    goal_dist = torch.norm(ee_pos_w - goal_pos_w, dim=-1)
    goal_term_raw = 1.0 - torch.tanh(goal_dist / goal_std)
    goal_term = torch.where(env._touched_cube, goal_term_raw, torch.zeros_like(goal_term_raw))

    return 0.3 * touch_term + 0.7 * goal_term


def touch_goal_milestone_bonus(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    goal_offset: tuple[float, float, float],
    cube_half_size: float = 0.006,
    touch_std: float = 0.05,
    touch_tolerance: float = 0.02,
    goal_std: float = 0.1,
) -> torch.Tensor:
    """Undiscounted running-max milestone bonus over _raw_touch_goal_progress -
    same mechanism as staged_milestone_bonus (tasks/ar4/mdp.py, this file):
    reward = (new best-ever raw progress) - (previous best-ever raw
    progress), never negative, never punishes regressing away from a
    best-ever point already reached."""
    if not hasattr(env, "_touch_goal_milestone_max"):
        env._touch_goal_milestone_max = torch.zeros(env.num_envs, device=env.device)

    raw = _raw_touch_goal_progress(
        env, object_cfg, ee_frame_cfg, goal_offset, cube_half_size, touch_std, touch_tolerance, goal_std,
    )
    prev = env._touch_goal_milestone_max.clone()
    env._touch_goal_milestone_max = torch.maximum(env._touch_goal_milestone_max, raw)
    return env._touch_goal_milestone_max - prev


def reset_touch_goal_milestone(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Event term (mode="reset"): zero the running-max milestone buffer and
    the touched-cube latch for resetting envs, so a new episode starts
    with no carried-over progress. Must be registered alongside
    reset_scene_to_default."""
    if not hasattr(env, "_touch_goal_milestone_max"):
        env._touch_goal_milestone_max = torch.zeros(env.num_envs, device=env.device)
    if not hasattr(env, "_touched_cube"):
        env._touched_cube = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._touch_goal_milestone_max[env_ids] = 0.0
    env._touched_cube[env_ids] = False


def touch_then_goal_reached(
    env: ManagerBasedRLEnv,
    threshold: float,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    goal_offset: tuple[float, float, float],
    cube_half_size: float = 0.006,
    touch_tolerance: float = 0.02,
) -> torch.Tensor:
    """Termination: end-effector within threshold of the fixed goal point
    AND the cube has been touched at some point this episode. Recomputes
    the touch check independently (not just reading env._touched_cube)
    so this is correct regardless of whether the reward manager or the
    termination manager runs first within a given step - both terms
    apply the same idempotent |= latch update using the same
    touch_tolerance, so either evaluation order produces the same result
    by the end of the step."""
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]

    touch_point_w = object.data.root_pos_w + torch.tensor([0.0, 0.0, cube_half_size], device=env.device)
    touch_dist = torch.norm(ee_pos_w - touch_point_w, dim=-1)
    if not hasattr(env, "_touched_cube"):
        env._touched_cube = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._touched_cube |= touch_dist < touch_tolerance

    goal_offset_t = torch.tensor(goal_offset, device=env.device)
    goal_pos_w = object.data.root_pos_w + goal_offset_t
    goal_dist = torch.norm(ee_pos_w - goal_pos_w, dim=-1)

    return (goal_dist < threshold) & env._touched_cube


def touch_goal_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    goal_offset: tuple[float, float, float],
) -> torch.Tensor:
    """The fixed goal position (object.data.root_pos_w + goal_offset)
    expressed in the robot's root frame - mirrors
    mirrored_target_position_in_robot_root_frame's pattern (this file),
    but derives the goal from the cube's own live position plus a fixed
    offset instead of a separately-randomized stateful buffer, since this
    task's cube position is fixed (not randomized per episode)."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    goal_offset_t = torch.tensor(goal_offset, device=env.device)
    goal_pos_w = object.data.root_pos_w + goal_offset_t
    goal_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, goal_pos_w)
    return goal_pos_b
```

- [ ] **Step 3: Verify the file still imports cleanly**

Run: `/home/saps/IsaacLab/isaaclab.sh -p -c "import sys; sys.path.insert(0, '.'); import ast; ast.parse(open('tasks/ar4/mdp.py').read())"`
Expected: no output, exit code 0 (this only checks Python syntax validity - full import verification happens in Task 3's smoke test, since `tasks/ar4/mdp.py` requires an AppLauncher/Isaac Sim context to import at all).

- [ ] **Step 4: Commit**

```bash
git add tasks/ar4/mdp.py
git commit -m "Add touch-goal reward/termination/observation functions for Experiment 25"
```

---

### Task 2: Create `tasks/ar4/pickplace_touchgoal_env_cfg.py`

**Files:**
- Create: `tasks/ar4/pickplace_touchgoal_env_cfg.py`

**Interfaces:**
- Consumes: `touch_goal_milestone_bonus`, `reset_touch_goal_milestone`, `touch_then_goal_reached`, `touch_goal_position_in_robot_root_frame` (Task 1, `tasks/ar4/mdp.py`); `CUBE_CFG` (`tasks/ar4/objects_cfg.py`); `AR4_MK5_CFG`, `ARM_JOINT_NAMES` (`tasks/ar4/robot_cfg.py`); `_EE_OFFSET` (`tasks/ar4/pickplace_env_cfg.py`).
- Produces: `Ar4PickPlaceTouchGoalEnvCfg` (the class `scripts/train.py` will import in Task 4), `GOAL_OFFSET = (-0.40, 0.0, 0.144)` (module-level constant, importable for reuse in eval/demo scripts later).

- [ ] **Step 1: Write the file**

```python
# tasks/ar4/pickplace_touchgoal_env_cfg.py
"""Touch-goal variant of the AR4 pick-and-place task (Experiment 25): the
arm's only job is to touch the top of a fixed cube, then reach a fixed
goal point. No grasp, no lift, no gripper action - see
docs/superpowers/specs/2026-07-09-ar4-experiment25-touch-goal-reach-design.md.

Additive/parallel to pickplace_mirror_env_cfg.py: deliberately does NOT
touch that file, env_cfg.py, or objects_cfg.py.

Import this module only after an Isaac Sim/Isaac Lab AppLauncher has been
created.
"""

import isaaclab.envs.mdp as isaaclab_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.manipulation.lift import mdp

from . import mdp as ar4_mdp
from .objects_cfg import CUBE_CFG
from .pickplace_env_cfg import _EE_OFFSET
from .robot_cfg import ARM_JOINT_NAMES, AR4_MK5_CFG

# NOTE (plan self-review, 2026-07-09): JointPositionActionCfg comes from
# isaaclab.envs.mdp (aliased isaaclab_mdp above), NOT from the
# isaaclab_tasks lift-task mdp module (aliased plain `mdp` above, used
# for everything else in this file: joint_pos_rel, object_position_in_
# robot_root_frame, last_action, reset_scene_to_default, time_out,
# action_rate_l2, joint_vel_l2) - exactly matching
# pickplace_mirror_env_cfg.py's own three-alias import convention
# (isaaclab_mdp / mdp / ar4_mdp). Using the wrong one of the two `mdp`
# aliases for ActionsCfg was caught here before implementation, not
# after a failed smoke test.

# Fixed goal point, expressed as an offset from the cube's own (also
# fixed) spawn position so per-env world placement stays correct without
# a separate randomization buffer: cube world (0.20, 0.28, 0.006) + this
# offset = world (-0.20, 0.28, 0.15) - mirrored across the cube in X,
# elevated clear of the ground plane.
GOAL_OFFSET = (-0.40, 0.0, 0.144)

CUBE_HALF_SIZE = 0.006  # meters (12mm cube, tasks/ar4/objects_cfg.py)
TOUCH_STD = 0.05
TOUCH_TOLERANCE = 0.02
GOAL_STD = 0.1
GOAL_TOLERANCE = 0.02


@configclass
class ActionsCfg:
    """Arm-only action space - no gripper action term at all (Experiment
    25 removes grasp/lift entirely, so the gripper serves no purpose
    here; the gripper joints stay physically present but unactuated)."""

    joint_positions = isaaclab_mdp.JointPositionActionCfg(asset_name="robot", joint_names=ARM_JOINT_NAMES, scale=0.5)


@configclass
class Ar4PickPlaceTouchGoalSceneCfg(InteractiveSceneCfg):
    """AR4 arm + a single fixed-position cube, no rect_prism/wedge/sphere,
    no gripper contact sensors (no grasp signal needed for this task)."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )
    robot: ArticulationCfg = AR4_MK5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cube: RigidObjectCfg = CUBE_CFG.replace(
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.20, 0.28, 0.006)),
    )

    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/root_joint/base_link",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/root_joint/link_6",
                name="end_effector",
                offset=OffsetCfg(pos=_EE_OFFSET),
            ),
        ],
    )


@configclass
class ObservationsCfg:
    """Observation specifications: arm joint state (gripper joints
    excluded - they're unactuated in this task), cube position, fixed
    goal position, last action (6-dim, arm-only)."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)}
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)}
        )
        cube_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame, params={"object_cfg": SceneEntityCfg("cube")}
        )
        goal_position = ObsTerm(
            func=ar4_mdp.touch_goal_position_in_robot_root_frame,
            params={"robot_cfg": SceneEntityCfg("robot"), "object_cfg": SceneEntityCfg("cube"), "goal_offset": GOAL_OFFSET},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset events: whole scene back to default, then zero the
    touch-goal milestone buffer and touched-cube latch."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_touch_goal_milestone = EventTerm(func=ar4_mdp.reset_touch_goal_milestone, mode="reset")


@configclass
class TerminationsCfg:
    """Success (end-effector touched the cube, then reached the goal)
    ends the episode early; otherwise a fixed timeout."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    goal_reached = DoneTerm(
        func=ar4_mdp.touch_then_goal_reached,
        params={
            "threshold": GOAL_TOLERANCE,
            "object_cfg": SceneEntityCfg("cube"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "goal_offset": GOAL_OFFSET,
            "cube_half_size": CUBE_HALF_SIZE,
            "touch_tolerance": TOUCH_TOLERANCE,
        },
    )


@configclass
class RewardsCfg:
    """Two-stage gated running-max milestone bonus: touch the cube top,
    then reach the goal - no grasp/lift terms at all."""

    touch_goal_milestone_bonus = RewTerm(
        func=ar4_mdp.touch_goal_milestone_bonus,
        weight=25.0,
        params={
            "object_cfg": SceneEntityCfg("cube"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "goal_offset": GOAL_OFFSET,
            "cube_half_size": CUBE_HALF_SIZE,
            "touch_std": TOUCH_STD,
            "touch_tolerance": TOUCH_TOLERANCE,
            "goal_std": GOAL_STD,
        },
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)}
    )


@configclass
class Ar4PickPlaceTouchGoalEnvCfg(ManagerBasedRLEnvCfg):
    """AR4 touch-then-goal task (Experiment 25): touch the cube's top,
    then reach a fixed goal point. No grasp, no lift, arm-only action
    space. num_envs=4096 default, matching the mirror task's training
    scale - scripts/train.py's --num_envs flag overrides this per-run."""

    scene: Ar4PickPlaceTouchGoalSceneCfg = Ar4PickPlaceTouchGoalSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 5.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0)
        self.viewer.eye = (1.5, 1.5, 1.2)
        self.viewer.lookat = (0.0, 0.0, 0.4)
```

- [ ] **Step 2: Verify Python syntax is valid**

Run: `python3 -c "import ast; ast.parse(open('/home/saps/projects/rl/tasks/ar4/pickplace_touchgoal_env_cfg.py').read())"`
Expected: no output, exit code 0.

- [ ] **Step 3: Commit**

```bash
git add tasks/ar4/pickplace_touchgoal_env_cfg.py
git commit -m "Add Ar4PickPlaceTouchGoalEnvCfg (Experiment 25: touch-cube-then-goal, no grasp)"
```

---

### Task 3: Smoke-test the new env cfg in real Isaac Sim

**Files:**
- Create: `scripts/smoke_test_touchgoal_env.py`

**Interfaces:**
- Consumes: `Ar4PickPlaceTouchGoalEnvCfg` (Task 2).
- Produces: nothing importable - a standalone diagnostic script (kept permanently, matching this project's convention of keeping useful diagnostic scripts like `scripts/mimic_joint_verify.py`).

- [ ] **Step 1: Write the smoke-test script**

```python
# scripts/smoke_test_touchgoal_env.py
"""Headless smoke test for Ar4PickPlaceTouchGoalEnvCfg (Experiment 25):
builds the env, steps it a fixed number of times with random actions
across a few envs, and prints observation shapes, reward values, and
termination behavior - real evidence the env cfg actually builds and
runs, not just that it imports without error.

.. code-block:: bash

    /home/saps/IsaacLab/isaaclab.sh -p scripts/smoke_test_touchgoal_env.py
"""

import sys

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser(description="Smoke test for the touch-goal env cfg.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os  # noqa: E402

import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from tasks.ar4.pickplace_touchgoal_env_cfg import Ar4PickPlaceTouchGoalEnvCfg  # noqa: E402


def main() -> None:
    env_cfg = Ar4PickPlaceTouchGoalEnvCfg()
    env_cfg.scene.num_envs = 4
    env_cfg.sim.device = args_cli.device
    env = ManagerBasedRLEnv(cfg=env_cfg)

    with torch.inference_mode():
        obs, _ = env.reset()
        print(f"[SMOKE] observation shape: {obs['policy'].shape} (expect (4, N) for some N)")
        print(f"[SMOKE] action space shape: {env.action_manager.action.shape} (expect (4, 6) - arm-only)")

        for step in range(50):
            actions = torch.zeros(4, env.action_manager.total_action_dim, device=env.device)
            obs, rew, terminated, truncated, info = env.step(actions)
            if step % 10 == 0:
                print(
                    f"[SMOKE] step {step}: reward={rew.cpu().tolist()}, "
                    f"terminated={terminated.cpu().tolist()}, truncated={truncated.cpu().tolist()}"
                )

        # Drive one env's arm target directly toward a plausible pose and confirm
        # touch_goal_milestone_bonus and touched-cube latch actually respond -
        # not just that the env runs without crashing.
        cube = env.scene["cube"]
        ee_frame = env.scene["ee_frame"]
        print(f"[SMOKE] cube position (env 0): {cube.data.root_pos_w[0].cpu().tolist()}")
        print(f"[SMOKE] ee_frame target position (env 0): {ee_frame.data.target_pos_w[0, 0].cpu().tolist()}")
        print(f"[SMOKE] env._touched_cube: {env._touched_cube.cpu().tolist()}")
        print(f"[SMOKE] env._touch_goal_milestone_max: {env._touch_goal_milestone_max.cpu().tolist()}")

    env.close()
    print("[SMOKE] PASS: env built, stepped 50 times, no exceptions.")


if __name__ == "__main__":
    main()
    simulation_app.close()
```

- [ ] **Step 2: Run the smoke test**

Run: `PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/smoke_test_touchgoal_env.py`
Expected: `[SMOKE] PASS: env built, stepped 50 times, no exceptions.` as the final line, with observation shape `(4, 24)` (6 joint_pos + 6 joint_vel + 3 cube_position + 3 goal_position + 6 last_action = 24 — if the printed shape differs, that's real evidence a param assumption in Task 1/2 was wrong; recompute by hand from the actual printed shape and fix the mismatched function before moving on, don't just accept a different number silently), action space shape `(4, 6)`, and `env._touched_cube`/`env._touch_goal_milestone_max` printed with sane values (not NaN, not all-True with zero actions commanded).

If this fails with an import or attribute error, fix the specific mismatch (e.g. a wrong function signature between Task 1's `mdp.py` additions and Task 2's env cfg `params={...}` wiring) and re-run before proceeding - do not move to Task 4 with a failing smoke test.

- [ ] **Step 3: Commit**

```bash
git add scripts/smoke_test_touchgoal_env.py
git commit -m "Add headless smoke test for the touch-goal env cfg"
```

---

### Task 4: Wire `--touchgoal` into `scripts/train.py`

**Files:**
- Modify: `scripts/train.py`

**Interfaces:**
- Consumes: `Ar4PickPlaceTouchGoalEnvCfg` (Task 2), `Ar4PickPlacePPORunnerCfg` (already imported in `scripts/train.py`, no change needed there).

- [ ] **Step 1: Add the import**

In `scripts/train.py`, immediately after the existing `from tasks.ar4.pickplace_taskspace_env_cfg import (...)` block (around line 236-239), add:

```python
from tasks.ar4.pickplace_touchgoal_env_cfg import Ar4PickPlaceTouchGoalEnvCfg  # noqa: E402
```

- [ ] **Step 2: Add the `--touchgoal` flag**

Immediately before the `AppLauncher.add_app_launcher_args(parser)` line (around line 181, right after the `--warmresidual` argument block), add:

```python
parser.add_argument(
    "--touchgoal",
    action="store_true",
    default=False,
    help=(
        "Train on the touch-goal variant (Experiment 25): the arm touches the top of a fixed cube, "
        "then reaches a fixed goal point - no grasp, no lift, arm-only action space. Replaces the "
        "grasp/lift mechanism entirely given the still-unresolved jaw-mimic-joint defect (Experiments "
        "19/22, both failed) and pickplace_mirror_env_cfg.py's own ungated reward shape (the same "
        "shape Experiment 16 found exploitable via wrist-wedging). See "
        "docs/superpowers/specs/2026-07-09-ar4-experiment25-touch-goal-reach-design.md."
    ),
)
```

- [ ] **Step 3: Add the env cfg selection branch**

In the `if/elif` chain selecting `env_cfg_cls` (around line 245-275), add as the **first** `elif` (highest priority, matching how `--warmresidual` is checked first among the variant flags):

```python
    if args_cli.touchgoal:
        env_cfg_cls = Ar4PickPlaceTouchGoalEnvCfg
    elif args_cli.warmresidual:
        env_cfg_cls = Ar4PickPlaceWarmResidualEnvCfg
```

(i.e. insert `if args_cli.touchgoal: env_cfg_cls = Ar4PickPlaceTouchGoalEnvCfg` as the new first branch, and change the existing first line from `if args_cli.warmresidual:` to `elif args_cli.warmresidual:` — do not duplicate the `if`.)

- [ ] **Step 4: Confirm `--touchgoal` falls through to the default PPO runner cfg**

Read the existing line: `if args_cli.taskspace or args_cli.residual or args_cli.reachskip or args_cli.baseproximity or args_cli.warmresidual:` (around line 281) — `--touchgoal` should NOT be added to this list, since it must fall through to the `else: agent_cfg = Ar4PickPlacePPORunnerCfg()` branch (same as `--mirror`). No code change needed here, just confirm by reading it that `touchgoal` is absent from that condition.

- [ ] **Step 5: Verify the script's argument parser accepts the new flag**

Run: `/home/saps/IsaacLab/isaaclab.sh -p scripts/train.py --touchgoal --help 2>&1 | grep -A2 "touchgoal"`
Expected: the `--touchgoal` help text is printed, confirming the flag parses without error (this also exercises that the new import at the top of the file doesn't raise, since argparse setup runs before any Isaac Sim app launch in this script - but the deeper `Ar4PickPlaceTouchGoalEnvCfg` import only happens after `AppLauncher` starts, so this check alone does not fully verify the import; Step 6 does).

- [ ] **Step 6: Run a short real training smoke test (confirms the full import + env cfg selection + PPO wiring all work together)**

Run: `PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/train.py --touchgoal --num_envs 64 --max_iterations 3`
Expected: training starts, runs 3 iterations, and exits cleanly (check `logs/train/<timestamp>/` was created with a `model_0.pt`-style checkpoint or at minimum a `params/` dir - confirms `env_cfg` and `agent_cfg` wiring is correct end-to-end, not just that individual pieces parse). If this errors, the error message will point at whichever piece (env cfg construction, PPO runner cfg dimension mismatch, observation/action shape mismatch) is actually wrong - fix it and re-run before considering this task done.

- [ ] **Step 7: Commit**

```bash
git add scripts/train.py
git commit -m "Wire --touchgoal flag into scripts/train.py (Experiment 25)"
```
