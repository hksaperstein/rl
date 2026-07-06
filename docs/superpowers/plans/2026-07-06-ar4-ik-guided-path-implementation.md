# AR4 classical-IK-guided path reward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new, parallel AR4 pick-and-place task that replaces
end-state-only reward shaping with a classical IK-guided path: geometric
Cartesian waypoints (pre-grasp, grasp, lift, transit, place) plus a live,
per-step reward comparing the policy's actual joint configuration against
what Isaac Lab's `DifferentialIKController` suggests as the next step
toward the current waypoint.

**Architecture:** New file `tasks/ar4/pickplace_ik_guided_env_cfg.py`
reusing the mirror-scene's scene/spawn/goal/shrunk-sphere infrastructure
unchanged, with a new `RewardsCfg` built from new `tasks/ar4/mdp.py`
functions (`compute_path_waypoints`, `ik_guided_path_bonus`,
`gripper_schedule_bonus`). `contact_grasp_bonus` and `stillness_penalty`
are reused unchanged as standalone top-level reward terms (no longer
folded inside a staged reach/grasp/lift/goal signal).

**Tech Stack:** Isaac Lab / Isaac Sim (`ManagerBasedRLEnv`),
`isaaclab.controllers.DifferentialIKController`, PyTorch, `rsl_rl` PPO.

## Global Constraints

- Design spec: `docs/superpowers/specs/2026-07-06-ar4-ik-guided-path-design.md`
  — read it, especially the "Important implementation refinement" section
  explaining why IK guidance is computed live (per step) rather than as
  an offline precomputed joint-space path.
- Always invoke via `/home/saps/IsaacLab/isaaclab.sh -p scripts/<script>.py`
  from the repo root — never plain `python` for anything importing
  `isaaclab`.
- Do **not** modify `tasks/ar4/env_cfg.py`, `tasks/ar4/objects_cfg.py`,
  `tasks/ar4/pickplace_env_cfg.py`, `tasks/ar4/pickplace_mirror_env_cfg.py`,
  or any existing function in `tasks/ar4/mdp.py` (including the
  mirror-scene/sphere-shrink task's own functions: `contact_grasp_bonus`,
  `stillness_penalty`, `reset_stillness_buffers`, `set_mirrored_goal`,
  `mirrored_target_position_in_robot_root_frame`,
  `object_reached_mirrored_goal`, `_raw_lift_progress_mirrored`,
  `staged_milestone_bonus`, `reset_lift_milestone`) — this is a new,
  parallel task, not a change to the existing one.
- `DifferentialIKController` config: `command_type="position"`,
  `use_relative_mode=False`, `ik_method="dls"`.
- Decided values, not placeholders: pre-grasp hover `0.05m`, lift margin
  `0.02m` above `lift_minimal_height=0.03`, carry height `0.10m`,
  waypoint-advance tolerance `0.03m`, Cartesian proximity `std=0.1`,
  IK-joint-match `std=0.5` (radians), `gripper_schedule_bonus` weight
  `0.1` (the function itself returns `1.0`/`0.0`, matching-vs-not — the
  `+0.1` magnitude the spec describes comes from this weight, not from
  the function's own return value).
- `num_envs=4096` for the real training run, matching every prior
  experiment this session.
- Reuses `Ar4PickPlacePPORunnerCfg` unchanged (no new PPO hyperparameters).
- Jacobian indexing must use `robot.is_fixed_base` (a live property
  check), never a hardcoded assumption — see Task 2.

---

### Task 1: Path waypoint computation + new scene/observations/events/terminations file

**Files:**
- Modify: `tasks/ar4/mdp.py` (add `compute_path_waypoints`)
- Create: `tasks/ar4/pickplace_ik_guided_env_cfg.py` (scene,
  `ObservationsCfg`, `EventCfg`, `TerminationsCfg` — `RewardsCfg` and the
  assembled env cfg come in Task 3)

**Interfaces:**
- Produces: `compute_path_waypoints(env, env_ids, sphere_cfg,
  lift_minimal_height, pregrasp_hover, lift_margin, carry_height) ->
  None` (EventTerm function, `mode="reset"`), writes
  `env._path_waypoints_w` (shape `(num_envs, 5, 3)`, world frame) and
  `env._path_waypoint_idx` (shape `(num_envs,)`, `torch.long`) and
  `env._ik_milestone_max` (shape `(num_envs,)`) — all three reset
  together since they represent one coupled piece of per-episode state
  (the path and where progress stands on it).
- Reuses (unchanged, imported): `Ar4PickPlaceMirrorSceneCfg`,
  `mirrored_target_position_in_robot_root_frame`,
  `object_reached_mirrored_goal`, `set_mirrored_goal`, from
  `tasks/ar4/pickplace_mirror_env_cfg.py`.
- Consumes: nothing from other tasks (this is the first task).

- [ ] **Step 1: Append `compute_path_waypoints` to `tasks/ar4/mdp.py`**

Add at the end of the file:

```python
def compute_path_waypoints(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    sphere_cfg: SceneEntityCfg,
    lift_minimal_height: float,
    pregrasp_hover: float,
    lift_margin: float,
    carry_height: float,
) -> None:
    """Event term (mode="reset"): must be registered AFTER
    reset_sphere_position (sphere's spawn) and AFTER randomize_goal
    (env._target_pos_w) in the same EventCfg, since it reads both.
    Computes 5 Cartesian waypoints (pre-grasp, grasp, lift, transit,
    place) purely geometrically - no IK is used to define them. IK
    guidance happens later, live, per step (ik_guided_path_bonus), by
    asking what classical IK would suggest toward whichever waypoint is
    currently active - see
    docs/superpowers/specs/2026-07-06-ar4-ik-guided-path-design.md for
    why an offline joint-space path isn't computed here.
    """
    sphere: RigidObject = env.scene[sphere_cfg.name]
    if not hasattr(env, "_path_waypoints_w"):
        env._path_waypoints_w = torch.zeros(env.num_envs, 5, 3, device=env.device)
        env._path_waypoint_idx = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._ik_milestone_max = torch.zeros(env.num_envs, device=env.device)

    sphere_pos = sphere.data.root_pos_w[env_ids]
    goal_pos = env._target_pos_w[env_ids]

    pregrasp = sphere_pos.clone()
    pregrasp[:, 2] += pregrasp_hover

    grasp = sphere_pos.clone()

    lift = sphere_pos.clone()
    lift[:, 2] = lift_minimal_height + lift_margin

    transit = torch.zeros_like(sphere_pos)
    transit[:, 0] = (sphere_pos[:, 0] + goal_pos[:, 0]) / 2.0
    transit[:, 1] = (sphere_pos[:, 1] + goal_pos[:, 1]) / 2.0
    transit[:, 2] = carry_height

    place = goal_pos.clone()

    env._path_waypoints_w[env_ids, 0] = pregrasp
    env._path_waypoints_w[env_ids, 1] = grasp
    env._path_waypoints_w[env_ids, 2] = lift
    env._path_waypoints_w[env_ids, 3] = transit
    env._path_waypoints_w[env_ids, 4] = place
    env._path_waypoint_idx[env_ids] = 0
    env._ik_milestone_max[env_ids] = 0.0
```

- [ ] **Step 2: Create `tasks/ar4/pickplace_ik_guided_env_cfg.py`**

```python
# tasks/ar4/pickplace_ik_guided_env_cfg.py
"""Classical-IK-guided variant of the AR4 mirror-goal pick-and-place task:
same scene/spawn-randomization/mirrored-goal as pickplace_mirror_env_cfg.py
(sphere-only, shrunk to 12mm diameter, spawn randomized across the full
workspace, goal on the opposite side of the robot), but the staged reward
is replaced by a classical-IK-guided path: 5 geometric Cartesian waypoints
(pre-grasp, grasp, lift, transit, place) plus a live, per-step comparison
between the policy's actual joint configuration and what
isaaclab.controllers.DifferentialIKController suggests toward the current
waypoint. See
docs/superpowers/specs/2026-07-06-ar4-ik-guided-path-design.md.

Additive/parallel to pickplace_mirror_env_cfg.py: does NOT modify that
file, env_cfg.py, objects_cfg.py, or pickplace_env_cfg.py. Reuses
Ar4PickPlaceMirrorSceneCfg and the mirrored-goal observation/termination
functions directly (unchanged) - only the scene's identity and the goal
mechanism are shared, not the reward.

Import this module only after an Isaac Sim/Isaac Lab AppLauncher has been
created.
"""

import isaaclab.sim as sim_utils
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
from .env_cfg import ActionsCfg
from .pickplace_mirror_env_cfg import Ar4PickPlaceMirrorSceneCfg

# Same values as pickplace_mirror_env_cfg.py's EventCfg/RewardsCfg reuse.
_LIFT_MINIMAL_HEIGHT = 0.03
_PREGRASP_HOVER = 0.05
_LIFT_MARGIN = 0.02
_CARRY_HEIGHT = 0.10


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP - identical to
    pickplace_mirror_env_cfg.py's ObservationsCfg (same scene, same goal
    mechanism), duplicated here rather than imported since this file's
    RewardsCfg/EventCfg differ enough that keeping all manager configs
    together in one file is clearer for this new task."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        sphere_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame, params={"object_cfg": SceneEntityCfg("sphere")}
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
    """Reset events, in registration order:
    1. reset_all - whole scene back to default.
    2. reset_sphere_position - randomize the sphere across the full workspace.
    3. randomize_goal - reads the sphere's now-updated position, sets the mirrored goal.
    4. compute_path_waypoints - reads both the sphere's position and the
       goal, computes the 5-waypoint path, and resets path-progress state."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_sphere_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.30, 0.30), "y": (-0.175, 0.175), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("sphere"),
        },
    )

    randomize_goal = EventTerm(
        func=ar4_mdp.set_mirrored_goal,
        mode="reset",
        params={
            "sphere_cfg": SceneEntityCfg("sphere"),
            "goal_y_range": (0.10, 0.45),
            "goal_z_range": (0.0, 0.02),
        },
    )

    compute_path_waypoints = EventTerm(
        func=ar4_mdp.compute_path_waypoints,
        mode="reset",
        params={
            "sphere_cfg": SceneEntityCfg("sphere"),
            "lift_minimal_height": _LIFT_MINIMAL_HEIGHT,
            "pregrasp_hover": _PREGRASP_HOVER,
            "lift_margin": _LIFT_MARGIN,
            "carry_height": _CARRY_HEIGHT,
        },
    )


@configclass
class TerminationsCfg:
    """Success (sphere at the mirrored goal) ends the episode early;
    otherwise a fixed timeout."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    sphere_reached_goal = DoneTerm(
        func=ar4_mdp.object_reached_mirrored_goal,
        params={"threshold": 0.02, "object_cfg": SceneEntityCfg("sphere")},
    )
```

Note: `RewardsCfg` and `Ar4PickPlaceIkGuidedEnvCfg` are added in Task 3 —
this file has no assembled env cfg yet until then.

- [ ] **Step 3: Verify the waypoint computation directly**

Create a throwaway verification script (delete after use, do not commit
it) at `/tmp/verify_ik_path.py`:

```python
"""Throwaway check: confirm compute_path_waypoints produces 5 finite,
geometrically sensible waypoints for every env, and that
_path_waypoint_idx starts at 0. Not part of the test suite - delete
after running. Route the result to a file, not just stdout - this
environment's Isaac Sim clean-shutdown path has repeatedly dropped
buffered stdout even on full success."""
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args(["--headless"])
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

sys.path.insert(0, "/home/saps/projects/rl")

import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.managers import RewardTermCfg as RewTerm  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.utils.configclass import configclass  # noqa: E402
from isaaclab_tasks.manager_based.manipulation.lift import mdp  # noqa: E402

from tasks.ar4.env_cfg import ActionsCfg  # noqa: E402
from tasks.ar4.pickplace_ik_guided_env_cfg import EventCfg, ObservationsCfg, TerminationsCfg  # noqa: E402
from tasks.ar4.pickplace_mirror_env_cfg import Ar4PickPlaceMirrorSceneCfg  # noqa: E402


@configclass
class _MinimalRewardsCfg:
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)


@configclass
class _CheckCfg(ManagerBasedRLEnvCfg):
    scene: Ar4PickPlaceMirrorSceneCfg = Ar4PickPlaceMirrorSceneCfg(num_envs=8, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: _MinimalRewardsCfg = _MinimalRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 5.0
        self.sim.dt = 0.01


env = ManagerBasedRLEnv(cfg=_CheckCfg())
env.reset()

sphere = env.scene["sphere"]
waypoints = env._path_waypoints_w.clone()
idx = env._path_waypoint_idx.clone()
sphere_pos = sphere.data.root_pos_w.clone()
goal_pos = env._target_pos_w.clone()

with open("/tmp/verify_ik_path_result.txt", "w") as f:
    f.write(f"waypoint_idx (should be all 0): {idx.tolist()}\n")
    f.write(f"all_finite: {torch.isfinite(waypoints).all().item()}\n")
    f.write(f"waypoint 0 (pregrasp) matches sphere_pos + hover:\n")
    f.write(f"  waypoints[:,0,:]: {waypoints[:, 0, :].tolist()}\n")
    f.write(f"  sphere_pos: {sphere_pos.tolist()}\n")
    f.write(f"waypoint 4 (place) matches goal_pos:\n")
    f.write(f"  waypoints[:,4,:]: {waypoints[:, 4, :].tolist()}\n")
    f.write(f"  goal_pos: {goal_pos.tolist()}\n")
    pregrasp_matches = torch.allclose(waypoints[:, 0, :2], sphere_pos[:, :2], atol=1e-4) and torch.allclose(
        waypoints[:, 0, 2], sphere_pos[:, 2] + 0.05, atol=1e-4
    )
    place_matches = torch.allclose(waypoints[:, 4, :], goal_pos, atol=1e-4)
    f.write(f"PREGRASP_MATCHES: {pregrasp_matches}\n")
    f.write(f"PLACE_MATCHES: {place_matches}\n")
    f.write("CHECK PASSED\n" if (pregrasp_matches and place_matches and torch.isfinite(waypoints).all()) else "CHECK FAILED\n")

simulation_app.close()
```

Run: `/home/saps/IsaacLab/isaaclab.sh -p /tmp/verify_ik_path.py`, then
read `/tmp/verify_ik_path_result.txt` (not stdout — this repo's Isaac Sim
setup has repeatedly dropped buffered console output on clean shutdown
even on full success, so file-based results are the reliable signal).

Expected: `CHECK PASSED`, `waypoint_idx` all zeros, `PREGRASP_MATCHES:
True`, `PLACE_MATCHES: True`.

Delete `/tmp/verify_ik_path.py` and `/tmp/verify_ik_path_result.txt`
after this passes (throwaway, not part of the repo).

- [ ] **Step 4: Commit**

```bash
cd /home/saps/projects/rl
git add tasks/ar4/mdp.py tasks/ar4/pickplace_ik_guided_env_cfg.py
git commit -m "Add classical-IK-guided path waypoint computation and scene/observations/events for AR4 sphere task"
```

---

### Task 2: Live IK-guided reward

**Files:**
- Modify: `tasks/ar4/mdp.py` (add `ik_guided_path_bonus`, plus new
  imports)

**Interfaces:**
- Consumes: `env._path_waypoints_w`, `env._path_waypoint_idx`,
  `env._ik_milestone_max` (Task 1, via lazy `hasattr` fallback since
  Task 1's event always runs first in practice but the guard matches
  this repo's established convention).
- Produces: `ik_guided_path_bonus(env, robot_cfg, ee_frame_cfg,
  proximity_std, advance_tolerance, ik_joint_std,
  gripper_tool_offset) -> torch.Tensor`, shape `(num_envs,)`.
  `gripper_tool_offset` corrects a tool-offset mismatch found during
  this task's review: the waypoints are defined for the gripper's pinch
  point (`_EE_OFFSET` in `pickplace_env_cfg.py`), not the raw `link_6`
  body the IK controller/Jacobian operate on — without correcting for
  it, IK's suggested target is systematically off by the offset's
  magnitude (3.6cm, larger than `advance_tolerance`), so the IK-match
  sub-signal could never reach its maximum even at the objectively
  correct grasp pose.

- [ ] **Step 1: Add imports to `tasks/ar4/mdp.py`**

Change:

```python
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, sample_uniform, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor, FrameTransformer
```

to:

```python
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_apply, sample_uniform, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor, FrameTransformer
```

(`DifferentialIKController`/`DifferentialIKControllerCfg` are
instantiated at runtime inside the reward function, not just used for
type hints, so they're real imports, not `TYPE_CHECKING`-only.)

- [ ] **Step 2: Append `ik_guided_path_bonus` to `tasks/ar4/mdp.py`**

```python
def ik_guided_path_bonus(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    proximity_std: float,
    advance_tolerance: float,
    ik_joint_std: float,
    gripper_tool_offset: tuple[float, float, float],
) -> torch.Tensor:
    """Undiscounted running-max bonus (same corrected pattern as
    staged_milestone_bonus - see that function's docstring for the decay
    bug this avoids) combining two sub-signals:

    1. Cartesian proximity to the current path waypoint
       (env._path_waypoints_w[:, env._path_waypoint_idx]), weighted so
       later waypoints dominate - a direct generalization of the old
       reach/lift/goal staged terms into one continuous 5-stage signal.
    2. How closely the arm's actual joint configuration matches what a
       LIVE classical IK controller (DifferentialIKController) suggests
       as the next joint target toward that same waypoint, computed
       fresh every step from the real physics state (jacobian, joint
       pos, ee pose) - see
       docs/superpowers/specs/2026-07-06-ar4-ik-guided-path-design.md's
       "Important implementation refinement" section for why this is
       live rather than a precomputed offline path.

    The waypoint index itself advances (monotonically, capped at 4)
    whenever the end-effector comes within advance_tolerance of the
    current waypoint.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    if not hasattr(env, "_path_waypoints_w"):
        env._path_waypoints_w = torch.zeros(env.num_envs, 5, 3, device=env.device)
        env._path_waypoint_idx = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._ik_milestone_max = torch.zeros(env.num_envs, device=env.device)

    if not hasattr(env, "_ik_controller"):
        ik_cfg = DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls")
        env._ik_controller = DifferentialIKController(ik_cfg, num_envs=env.num_envs, device=env.device)
        env._ik_robot_entity_cfg = SceneEntityCfg("robot", joint_names=robot_cfg.joint_names, body_names=["link_6"])
        env._ik_robot_entity_cfg.resolve(env.scene)
        env._ik_jacobi_idx = (
            env._ik_robot_entity_cfg.body_ids[0] - 1
            if robot.is_fixed_base
            else env._ik_robot_entity_cfg.body_ids[0]
        )

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

    jacobian = robot.root_physx_view.get_jacobians()[:, env._ik_jacobi_idx, :, env._ik_robot_entity_cfg.joint_ids]
    root_pose_w = robot.data.root_pose_w
    ee_pose_w = robot.data.body_pose_w[:, env._ik_robot_entity_cfg.body_ids[0]]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )
    joint_pos = robot.data.joint_pos[:, env._ik_robot_entity_cfg.joint_ids]

    # The waypoint is defined for the gripper's pinch point (ee_frame's
    # target_pos_w - link_6 offset by gripper_tool_offset along its own
    # local +Z, see _EE_OFFSET in pickplace_env_cfg.py), but the IK
    # controller/Jacobian operate on the raw link_6 body. Subtract the
    # offset (rotated into world frame by link_6's current world
    # orientation) from the waypoint before commanding IK, so the
    # suggested joint target places the PINCH POINT - not link_6 itself -
    # at the waypoint. Without this, IK's suggested target is
    # systematically off by the offset's magnitude (3.6cm, larger than
    # advance_tolerance), so the IK-match sub-signal could never reach
    # its maximum even at the objectively correct grasp pose.
    offset_vec = torch.tensor(gripper_tool_offset, device=env.device).expand(env.num_envs, 3)
    offset_w = quat_apply(ee_pose_w[:, 3:7], offset_vec)
    ik_target_w = current_waypoint - offset_w
    waypoint_command_b, _ = subtract_frame_transforms(root_pose_w[:, 0:3], root_pose_w[:, 3:7], ik_target_w)
    env._ik_controller.set_command(waypoint_command_b, ee_pos=ee_pos_b, ee_quat=ee_quat_b)
    joint_pos_des = env._ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

    joint_dist = torch.norm(joint_pos - joint_pos_des, dim=-1)
    ik_match_term = 1.0 - torch.tanh(joint_dist / ik_joint_std)

    raw = proximity_term + ik_match_term
    prev = env._ik_milestone_max.clone()
    env._ik_milestone_max = torch.maximum(env._ik_milestone_max, raw)
    return env._ik_milestone_max - prev
```

- [ ] **Step 3: Verify the IK guidance produces finite, sensible output**

Reuse the same throwaway-script pattern as Task 1's Step 3 (a fresh
`/tmp/verify_ik_guidance.py`, delete after use, route results to a file
not stdout). Build a `_CheckCfg` identical to Task 1's verification
script but with a full `RewardsCfg` containing exactly one term:

```python
@configclass
class _CheckRewardsCfg:
    ik_guided_path_bonus = RewTerm(
        func=ar4_mdp.ik_guided_path_bonus,
        weight=1.0,
        params={
            "robot_cfg": SceneEntityCfg("robot", joint_names=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "proximity_std": 0.1,
            "advance_tolerance": 0.03,
            "ik_joint_std": 0.5,
            "gripper_tool_offset": (0.0, 0.0, 0.036),
        },
    )
```

After `env.reset()`, step the environment 10 times with zero actions
(`torch.zeros(env.num_envs, env.action_manager.total_action_dim,
device=env.device)`) and write to a result file: whether every reward
value across all 10 steps is finite (`torch.isfinite(...).all()` on the
collected rewards) and non-negative (the undiscounted running-max
formula guarantees this — a negative value here means a real bug, not
just an uninteresting result), plus the final `env._path_waypoint_idx`
values. Read the result file, not stdout, per this environment's known
buffering quirk.

Expected: `ALL_FINITE: True`, `ALL_NON_NEGATIVE: True`. A `NaN` here
would most likely mean a singular Jacobian (e.g., the arm started at a
degenerate pose) — if this happens, check that `ik_method="dls"`
(damped least-squares, robust near singularities) is actually what got
used, not an accidentally-different method.

Delete both throwaway files after this passes.

- [ ] **Step 4: Commit**

```bash
cd /home/saps/projects/rl
git add tasks/ar4/mdp.py
git commit -m "Add live classical-IK-guided path reward for AR4 sphere task"
```

---

### Task 3: Gripper-schedule bonus + full `RewardsCfg` + assembled env cfg

**Files:**
- Modify: `tasks/ar4/mdp.py` (add `gripper_schedule_bonus`)
- Modify: `tasks/ar4/pickplace_ik_guided_env_cfg.py` (add `RewardsCfg`,
  add `Ar4PickPlaceIkGuidedEnvCfg`)

**Interfaces:**
- Consumes: `env._path_waypoint_idx` (Task 1), `ik_guided_path_bonus`
  (Task 2), `contact_grasp_bonus`/`stillness_penalty`/
  `reset_stillness_buffers` (existing, from the mirror-scene task,
  imported unchanged).
- Produces: `gripper_schedule_bonus(env, robot_cfg, gripper_joint_names,
  open_pos, closed_pos) -> torch.Tensor`, shape `(num_envs,)`.
  Produces: `Ar4PickPlaceIkGuidedEnvCfg(ManagerBasedRLEnvCfg)`, the
  complete assembled env config, `num_envs=4096` default.

- [ ] **Step 1: Append `gripper_schedule_bonus` to `tasks/ar4/mdp.py`**

```python
def gripper_schedule_bonus(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    gripper_joint_names: list[str],
    open_pos: float,
    closed_pos: float,
) -> torch.Tensor:
    """Reward matching the classical plan's expected gripper state for
    the current path waypoint: open through waypoints 0-1 (pre-grasp,
    grasp-approach), closed from waypoint 2 onward (lift, transit,
    place). Uses the actual gripper joint position (not the commanded
    action) as ground truth, consistent with contact_grasp_bonus reading
    real physical state rather than commands. Returns 1.0/0.0
    (matches/doesn't) - the "+0.1" magnitude described in the design
    spec comes from this term's RewardsCfg weight (0.1), not from this
    function's own return value. See
    docs/superpowers/specs/2026-07-06-ar4-ik-guided-path-design.md.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    if not hasattr(env, "_path_waypoint_idx"):
        env._path_waypoint_idx = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    gripper_joint_ids, _ = robot.find_joints(gripper_joint_names)
    gripper_pos = robot.data.joint_pos[:, gripper_joint_ids].mean(dim=-1)
    midpoint = (open_pos + closed_pos) / 2.0
    is_open = gripper_pos > midpoint

    expected_open = env._path_waypoint_idx < 2
    matches = is_open == expected_open
    return matches.float()
```

- [ ] **Step 2: Add `RewardsCfg` and `Ar4PickPlaceIkGuidedEnvCfg` to `tasks/ar4/pickplace_ik_guided_env_cfg.py`**

Add these imports to the top of the file (alongside the existing ones):

```python
from .robot_cfg import ARM_JOINT_NAMES, GRIPPER_CLOSED_POS, GRIPPER_JOINT_NAMES, GRIPPER_OPEN_POS
```

Append at the end of the file:

```python
@configclass
class RewardsCfg:
    """Classical-IK-guided path reward: replaces the old staged
    reach/grasp/lift/goal signal with waypoint-sequenced Cartesian
    proximity + live IK-action-matching (ik_guided_path_bonus), plus a
    gripper-open/closed timing bonus. contact_grasp_bonus and
    stillness_penalty are reused unchanged from the mirror-scene task as
    standalone additive terms (no longer folded inside a staged signal)."""

    ik_guided_path_bonus = RewTerm(
        func=ar4_mdp.ik_guided_path_bonus,
        weight=25.0,
        params={
            "robot_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "proximity_std": 0.1,
            "advance_tolerance": 0.03,
            "ik_joint_std": 0.5,
            "gripper_tool_offset": (0.0, 0.0, 0.036),
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

    contact_grasp_bonus = RewTerm(
        func=ar4_mdp.contact_grasp_bonus,
        weight=20.0,
        params={
            "force_threshold": 0.05,
            "jaw1_contact_cfg": SceneEntityCfg("gripper_jaw1_contact"),
            "jaw2_contact_cfg": SceneEntityCfg("gripper_jaw2_contact"),
        },
    )

    stillness_penalty = RewTerm(
        func=ar4_mdp.stillness_penalty,
        weight=2.0,
        params={
            "object_cfg": SceneEntityCfg("sphere"),
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
class Ar4PickPlaceIkGuidedEnvCfg(ManagerBasedRLEnvCfg):
    """AR4 classical-IK-guided task: same scene/spawn/goal as the mirror
    task, but reach/grasp/lift/carry is shaped by a live classical-IK
    path-tracking reward instead of ad hoc end-state distances.
    num_envs=4096 default (a real training-scale run) -
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

- [ ] **Step 3: Smoke test**

```bash
cd /home/saps/projects/rl
/home/saps/IsaacLab/isaaclab.sh -p -c "
from isaaclab.app import AppLauncher
import argparse
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args(['--headless'])
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv
from tasks.ar4.pickplace_ik_guided_env_cfg import Ar4PickPlaceIkGuidedEnvCfg

cfg = Ar4PickPlaceIkGuidedEnvCfg()
cfg.scene.num_envs = 16
env = ManagerBasedRLEnv(cfg=cfg)
env.reset()
zeros = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
for _ in range(5):
    env.step(zeros)
with open('/tmp/task3_smoke_result.txt', 'w') as f:
    f.write('reward terms: ' + str(list(env.reward_manager.active_terms)) + '\n')
    f.write('SMOKE TEST PASSED\n')
simulation_app.close()
"
cat /tmp/task3_smoke_result.txt
rm -f /tmp/task3_smoke_result.txt
```

Expected: `/tmp/task3_smoke_result.txt` contains `reward terms:
['ik_guided_path_bonus', 'gripper_schedule_bonus', 'contact_grasp_bonus',
'stillness_penalty', 'action_rate', 'joint_vel']` and `SMOKE TEST
PASSED`. No exceptions on any stateful buffer regardless of
reward-fn/reset-event ordering.

- [ ] **Step 4: Commit**

```bash
cd /home/saps/projects/rl
git add tasks/ar4/mdp.py tasks/ar4/pickplace_ik_guided_env_cfg.py
git commit -m "Add gripper-schedule bonus and assemble full AR4 classical-IK-guided env cfg"
```

---

### Task 4: Wire `--ik-guided` flag into `scripts/train.py` and `scripts/eval_loop.py`

**Files:**
- Modify: `scripts/train.py`
- Modify: `scripts/eval_loop.py`

**Interfaces:**
- Consumes: `Ar4PickPlaceIkGuidedEnvCfg` (Task 3).
- Produces: `--ik_guided` CLI flag on both scripts.

- [ ] **Step 1: Add the `--ik_guided` flag and env-selection branch to `scripts/train.py`**

Change:

```python
parser.add_argument(
    "--mirror",
    action="store_true",
    default=False,
    help=(
        "Train on the mirror-goal scene (sphere only, spawn randomized across the full workspace, goal "
        "always on the opposite side of the robot from the spawn), with the corrected undiscounted "
        "milestone-bonus reward and a grasp-gated stillness penalty. See "
        "docs/superpowers/specs/2026-07-06-ar4-sphere-mirror-scene-design.md."
    ),
)
AppLauncher.add_app_launcher_args(parser)
```

to:

```python
parser.add_argument(
    "--mirror",
    action="store_true",
    default=False,
    help=(
        "Train on the mirror-goal scene (sphere only, spawn randomized across the full workspace, goal "
        "always on the opposite side of the robot from the spawn), with the corrected undiscounted "
        "milestone-bonus reward and a grasp-gated stillness penalty. See "
        "docs/superpowers/specs/2026-07-06-ar4-sphere-mirror-scene-design.md."
    ),
)
parser.add_argument(
    "--ik_guided",
    action="store_true",
    default=False,
    help=(
        "Train on the classical-IK-guided variant of the mirror-goal scene: reach/grasp/lift/carry is "
        "shaped by a live classical-IK path-tracking reward instead of ad hoc end-state distances. See "
        "docs/superpowers/specs/2026-07-06-ar4-ik-guided-path-design.md."
    ),
)
AppLauncher.add_app_launcher_args(parser)
```

Change:

```python
from tasks.ar4.pickplace_env_cfg import GROUND_Z, Ar4PickPlaceEnvCfg  # noqa: E402
from tasks.ar4.pickplace_mirror_env_cfg import Ar4PickPlaceMirrorEnvCfg  # noqa: E402
from tasks.ar4.pickplace_single_object_env_cfg import Ar4PickPlaceSingleObjectEnvCfg  # noqa: E402
```

to:

```python
from tasks.ar4.pickplace_env_cfg import GROUND_Z, Ar4PickPlaceEnvCfg  # noqa: E402
from tasks.ar4.pickplace_ik_guided_env_cfg import Ar4PickPlaceIkGuidedEnvCfg  # noqa: E402
from tasks.ar4.pickplace_mirror_env_cfg import Ar4PickPlaceMirrorEnvCfg  # noqa: E402
from tasks.ar4.pickplace_single_object_env_cfg import Ar4PickPlaceSingleObjectEnvCfg  # noqa: E402
```

Change:

```python
def main() -> None:
    if args_cli.mirror:
        env_cfg_cls = Ar4PickPlaceMirrorEnvCfg
    elif args_cli.perception:
        env_cfg_cls = Ar4PickPlaceSingleObjectEnvCfg
    else:
        env_cfg_cls = Ar4PickPlaceEnvCfg
    env_cfg = env_cfg_cls()
```

to:

```python
def main() -> None:
    if args_cli.ik_guided:
        env_cfg_cls = Ar4PickPlaceIkGuidedEnvCfg
    elif args_cli.mirror:
        env_cfg_cls = Ar4PickPlaceMirrorEnvCfg
    elif args_cli.perception:
        env_cfg_cls = Ar4PickPlaceSingleObjectEnvCfg
    else:
        env_cfg_cls = Ar4PickPlaceEnvCfg
    env_cfg = env_cfg_cls()
```

- [ ] **Step 2: Smoke-test `scripts/train.py --ik_guided`**

```bash
cd /home/saps/projects/rl
/home/saps/IsaacLab/isaaclab.sh -p scripts/train.py --ik_guided --num_envs 16 --max_iterations 2 --headless
```

Expected: exits 0. Verify via file evidence (this environment's Isaac
Sim setup has repeatedly dropped final console text even on success):
the newest `logs/train/<timestamp>/` directory has `model_0.pt` and
`model_1.pt`, and its `params/env.yaml` shows the 6 reward terms from
Task 3.

- [ ] **Step 3: Add the `--ik_guided` flag and env-selection branch to `scripts/eval_loop.py`**

Change:

```python
parser.add_argument(
    "--mirror",
    action="store_true",
    default=False,
    help="Evaluate the mirror-goal scene (see scripts/train.py --mirror) instead of the four-object scene.",
)
AppLauncher.add_app_launcher_args(parser)
```

to:

```python
parser.add_argument(
    "--mirror",
    action="store_true",
    default=False,
    help="Evaluate the mirror-goal scene (see scripts/train.py --mirror) instead of the four-object scene.",
)
parser.add_argument(
    "--ik_guided",
    action="store_true",
    default=False,
    help="Evaluate the classical-IK-guided scene (see scripts/train.py --ik_guided) instead of the four-object scene.",
)
AppLauncher.add_app_launcher_args(parser)
```

Change:

```python
from tasks.ar4.pickplace_env_cfg import GROUND_Z, Ar4PickPlaceEnvCfg, Ar4PickPlacePerceptionEnvCfg  # noqa: E402
from tasks.ar4.pickplace_mirror_env_cfg import Ar4PickPlaceMirrorEnvCfg  # noqa: E402
```

to:

```python
from tasks.ar4.pickplace_env_cfg import GROUND_Z, Ar4PickPlaceEnvCfg, Ar4PickPlacePerceptionEnvCfg  # noqa: E402
from tasks.ar4.pickplace_ik_guided_env_cfg import Ar4PickPlaceIkGuidedEnvCfg  # noqa: E402
from tasks.ar4.pickplace_mirror_env_cfg import Ar4PickPlaceMirrorEnvCfg  # noqa: E402
```

Change:

```python
def main() -> None:
    if args_cli.mirror:
        env_cfg_cls = Ar4PickPlaceMirrorEnvCfg
    elif args_cli.perception:
        env_cfg_cls = Ar4PickPlacePerceptionEnvCfg
    else:
        env_cfg_cls = Ar4PickPlaceEnvCfg
    env_cfg = env_cfg_cls()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device
```

to:

```python
def main() -> None:
    if args_cli.ik_guided:
        env_cfg_cls = Ar4PickPlaceIkGuidedEnvCfg
    elif args_cli.mirror:
        env_cfg_cls = Ar4PickPlaceMirrorEnvCfg
    elif args_cli.perception:
        env_cfg_cls = Ar4PickPlacePerceptionEnvCfg
    else:
        env_cfg_cls = Ar4PickPlaceEnvCfg
    env_cfg = env_cfg_cls()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device
```

Change (the `else` branch's `RecordVideo` call — gives the IK-guided
scene's eval videos a distinct filename prefix):

```python
        name_prefix = "ar4_pickplace_mirror" if args_cli.mirror else "ar4_pickplace"
```

to:

```python
        if args_cli.ik_guided:
            name_prefix = "ar4_pickplace_ik_guided"
        elif args_cli.mirror:
            name_prefix = "ar4_pickplace_mirror"
        else:
            name_prefix = "ar4_pickplace"
```

- [ ] **Step 4: Smoke-test `scripts/eval_loop.py --ik_guided`**

```bash
cd /home/saps/projects/rl
LATEST=$(ls -dt logs/train/*/ | head -1)
/home/saps/IsaacLab/isaaclab.sh -p scripts/eval_loop.py --ik_guided --checkpoint "${LATEST}model_1.pt" --episodes 1
```

Expected: exits 0, produces `logs/videos/ar4_pickplace_ik_guided-step-0.mp4`.

- [ ] **Step 5: Commit**

```bash
cd /home/saps/projects/rl
git add scripts/train.py scripts/eval_loop.py
git commit -m "Wire --ik_guided flag into train.py and eval_loop.py for the AR4 classical-IK-guided task"
```

---

### Task 5: Full 1500-iteration training run (4096 envs)

**Files:** none (no code changes — this task runs the training loop and
inspects its output).

**Interfaces:**
- Consumes: `Ar4PickPlaceIkGuidedEnvCfg` (Task 3), `--ik_guided` flag (Task 4).
- Produces: `logs/train/<timestamp>/model_1499.pt` and TensorBoard event
  logs, consumed by Task 6.

- [ ] **Step 1: Run the full 1500-iteration training run**

```bash
cd /home/saps/projects/rl
/home/saps/IsaacLab/isaaclab.sh -p scripts/train.py --ik_guided --num_envs 4096 --headless
```

Expected wall-clock time: roughly 15-30 minutes (may run somewhat slower
than prior experiments this session due to the per-step
`DifferentialIKController`/Jacobian computation inside the reward
function — this is expected, not a bug, if it happens).

Verify completion via file evidence — `logs/train/<timestamp>/model_1499.pt`
existing — not console text.

- [ ] **Step 2: Pull the key TensorBoard scalars**

```bash
cd /home/saps/projects/rl
LATEST=$(ls -dt logs/train/*/ | head -1)
/home/saps/IsaacLab/isaaclab.sh -p -c "
from tensorboard.backend.event_processing import event_accumulator
import glob
path = sorted(glob.glob('${LATEST}events.out.tfevents.*'))[-1]
ea = event_accumulator.EventAccumulator(path)
ea.Reload()
for tag in ['Episode_Reward/ik_guided_path_bonus', 'Episode_Reward/gripper_schedule_bonus',
            'Episode_Reward/contact_grasp_bonus', 'Episode_Reward/stillness_penalty',
            'Episode_Termination/sphere_reached_goal']:
    if tag in ea.Tags()['scalars']:
        vals = ea.Scalars(tag)
        n = len(vals)
        samples = [vals[i].value for i in range(0, n, max(1, n // 10))]
        print(tag, '-> first:', vals[0].value, 'last:', vals[-1].value, 'max:', max(v.value for v in vals),
              'min:', min(v.value for v in vals),
              'trajectory (10 samples):', [round(s, 6) for s in samples])
    else:
        print(tag, '-> NOT FOUND')
"
```

Record all five lines verbatim in the Task 5 report. Check specifically
whether `ik_guided_path_bonus` stays non-negative throughout (the
undiscounted running-max formula guarantees this — a negative value here
would mean a real bug, same class of issue found in the mirror-scene
experiment's first attempt).

- [ ] **Step 3: Write the report (create the file)**

Create `docs/superpowers/plans/2026-07-06-ar4-ik-guided-path-report.md`
with a "Task 5" section containing: the log directory path, the five
scalar lines from Step 2, and one factual sentence on the trajectories —
no success/failure judgment yet, that's Task 6's job after the eval
video.

```bash
git add docs/superpowers/plans/2026-07-06-ar4-ik-guided-path-report.md
git commit -m "Record AR4 classical-IK-guided path experiment training run results"
```

---

### Task 6: Real eval + video inspection (decision gate)

**Files:** none (no code changes — this task runs eval and visually
inspects output).

**Interfaces:**
- Consumes: `logs/train/<timestamp>/model_1499.pt` from Task 5.
- Produces: eval videos in `logs/videos/`, a final pass/fail verdict
  consumed by Task 7.

- [ ] **Step 1: Run eval for 10 episodes**

```bash
cd /home/saps/projects/rl
/home/saps/IsaacLab/isaaclab.sh -p scripts/eval_loop.py --ik_guided --checkpoint logs/train/<RUN_DIR>/model_1499.pt --episodes 10
```

(substitute the actual `<RUN_DIR>` from Task 5). Expected: 10 files
`logs/videos/ar4_pickplace_ik_guided-step-0.mp4` through `-step-2250.mp4`.
If fewer than 10 fresh files appear (checked via file modification time,
not just existence — a prior experiment this session found a stale file
left over from an earlier eval run), delete all
`ar4_pickplace_ik_guided-step-*.mp4` files and re-run eval once to get a
clean, complete set before proceeding.

- [ ] **Step 2: Extract frames from every episode video**

```bash
cd /home/saps/projects/rl
rm -rf logs/videos/frames_ik_guided
mkdir -p logs/videos/frames_ik_guided
for f in logs/videos/ar4_pickplace_ik_guided-step-*.mp4; do
  name=$(basename "$f" .mp4)
  mkdir -p "logs/videos/frames_ik_guided/$name"
  ffmpeg -y -i "$f" -vf fps=10 "logs/videos/frames_ik_guided/$name/frame_%03d.png" -loglevel error
done
```

- [ ] **Step 3: Visually inspect all 10 episodes, frame-by-frame**

Use the Read tool to view frames from each of the 10 episode
directories. **Check every ~5th frame across the whole episode (not just
start/25%/50%/75%/end)** — two prior experiments this session found that
a coarse 5-point sample can misread an accidental gripper-body collision
launching the sphere into the air as a genuine lift. The real signature
of a genuine grasp-and-lift is the sphere's position **tracking the
gripper's position across consecutive frames** (moving together); a
knock/launch shows the sphere separating from a gripper that stays
static, or a motion-blur streak at the moment of the collision. For each
episode, determine: does the sphere spawn at a different position than
other episodes (confirms randomization still works), does the sphere
ever move together with the gripper while elevated, and does it end up
carried toward the opposite side of the robot (the mirror-goal
objective)?

- [ ] **Step 4: Apply the decision gate**

- **If 8/10 or more episodes show the sphere genuinely lifted (tracking
  the gripper, not just briefly airborne) and carried toward the
  target:** success.
- **If fewer than 8/10 do, but some episodes show real (even brief)
  tracked lifting that never happened in prior experiments:** partial
  progress — describe precisely what's different, and double-check it
  isn't another accidental-launch false positive before reporting it as
  progress.
- **If 0/10 show any genuine tracked lift:** this experiment is
  falsified. This is the eighth real attempt on this sub-problem. Flag
  back to the Principal/user rather than attempting a ninth tweak
  unilaterally.

- [ ] **Step 5: Commit the report update**

```bash
git add docs/superpowers/plans/2026-07-06-ar4-ik-guided-path-report.md
git commit -m "Record AR4 classical-IK-guided path experiment eval video inspection results"
```

---

### Task 7: Record the outcome in `ROADMAP.md`

**Files:**
- Modify: `ROADMAP.md`

**Interfaces:** none (documentation-only task, terminal in this plan).

- [ ] **Step 1: Write the outcome into `ROADMAP.md`'s "grasp/lift never
  emerges" follow-up**

Add a new bullet after the sphere-shrink sub-bullet already recorded
there, following the same evidentiary detail as the existing
sub-bullets. Use whichever template applies, filling in real numbers
from Tasks 5-6:

**If Task 6's decision gate passed (8/10+):**

```markdown
   - **Follow-up experiment: classical-IK-guided path reward
     (SUCCESS).** Replaced end-state-only reward shaping with a
     classical IK path (5 Cartesian waypoints: pre-grasp, grasp, lift,
     transit, place) and a live per-step reward comparing the policy's
     actual joint configuration against what
     isaaclab.controllers.DifferentialIKController suggests toward the
     current waypoint, plus a gripper-open/closed timing bonus, per
     `docs/superpowers/specs/2026-07-06-ar4-ik-guided-path-design.md`.
     Full run data:
     `docs/superpowers/plans/2026-07-06-ar4-ik-guided-path-report.md`.
     **Result: [X]/10 real eval episodes show the sphere genuinely
     lifted (tracking the gripper) and carried to the opposite side.**
     This resolves the "grasp/lift never emerges" follow-up.
```

**If Task 6's decision gate did not pass (0/10 or partial):**

```markdown
   - **Follow-up experiment: classical-IK-guided path reward
     ([falsified | partial progress]).** Replaced end-state-only reward
     shaping with a classical IK path (5 Cartesian waypoints: pre-grasp,
     grasp, lift, transit, place) and a live per-step reward comparing
     the policy's actual joint configuration against what
     isaaclab.controllers.DifferentialIKController suggests toward the
     current waypoint, plus a gripper-open/closed timing bonus, per
     `docs/superpowers/specs/2026-07-06-ar4-ik-guided-path-design.md` -
     user-directed, the eighth real attempt on this sub-problem. Full
     run data:
     `docs/superpowers/plans/2026-07-06-ar4-ik-guided-path-report.md`.
     **Result: [X]/10 real eval episodes show any genuine (gripper-
     tracking, not accidental-launch) lift** — [one to two sentences on
     what the video actually showed]. Flagged back to the
     Principal/user; the reward/scene axis has now been tuned eight
     different ways without a single confirmed controlled grasp.
```

- [ ] **Step 2: Commit**

```bash
git add ROADMAP.md
git commit -m "Record AR4 classical-IK-guided path experiment outcome in ROADMAP"
```

---

## Plan self-review notes

- **Spec coverage:** Task 1 covers the design's Cartesian waypoint
  computation and new scene/observations/events/terminations. Task 2
  covers the live IK-guidance reward (the design's core novel piece,
  including its own dedicated verification step given the risk of a
  singular-Jacobian NaN). Task 3 covers the gripper-schedule bonus and
  full reward/env assembly. Task 4 covers script wiring (needed for
  Tasks 5-6 to be runnable). Task 5 covers the full-run half of the
  design's verification plan. Task 6 covers the eval/video half,
  explicitly requiring frame-by-frame (not coarse-sample) inspection
  given two prior false positives this session. Task 7 covers ROADMAP
  recording.
- **Scope discipline:** confirmed no task modifies `env_cfg.py`,
  `objects_cfg.py`, `pickplace_env_cfg.py`, `pickplace_mirror_env_cfg.py`,
  or any existing `mdp.py` function.
- **Type/name consistency:** `ik_guided_path_bonus`'s params
  (`robot_cfg, ee_frame_cfg, proximity_std, advance_tolerance,
  ik_joint_std`) match between Task 2's definition and Task 3's
  `RewardsCfg` registration. `gripper_schedule_bonus`'s params
  (`robot_cfg, gripper_joint_names, open_pos, closed_pos`) likewise.
  `_path_waypoints_w`, `_path_waypoint_idx`, `_ik_milestone_max` are the
  same buffer names everywhere they appear across Tasks 1-3.
  `Ar4PickPlaceIkGuidedEnvCfg` is the same class name used in Task 4's
  script-wiring imports as defined in Task 3.
