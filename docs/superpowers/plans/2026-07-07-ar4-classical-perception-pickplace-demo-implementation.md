# Classical Perception-Driven Pick-and-Place Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fully classical (zero RL/learned components) demo: the camera detects the cube live, its position is transformed from camera/world frame into the robot's root frame, a 5-waypoint Cartesian path is planned live from that detection, and a live differential-IK controller executes pregrasp → grasp → lift → transit → place → release → home.

**Architecture:** Generalize the existing sphere-only perception helper to any shape label (Task 1), add a minimal single-cube-plus-camera scene/env cfg with no reward/termination managers (Task 2), add a new demo script with a pure waypoint-planning function and a scripted execution loop reusing the same pursuit-controller math as Experiment 13's `ResidualDifferentialIKAction._compute_base_delta()` (Task 3), then smoke-test and run for real (Task 4).

**Tech Stack:** Isaac Lab `ManagerBasedEnv` (non-RL), `isaaclab.envs.mdp.DifferentialInverseKinematicsActionCfg` (plain, unmodified — this demo has no policy), this repo's existing `perception/` package (`run_perception`, `ObjectTracker`, `find_by_shape`, `draw_detections`), `imageio` for video recording.

## Global Constraints

- Do not modify `env_cfg.py`, `objects_cfg.py`, any `pickplace_*.py` file, `tasks/ar4/mdp.py`, `tasks/ar4/residual_ik_action.py`, `interactive_demo.py`, or `eval_loop.py` — purely additive, per this repo's established per-feature-file convention. `interactive_demo.py`/`eval_loop.py --perception` must keep working unchanged (they call `perceive_sphere`).
- Decided values (verbatim, not placeholders): `_BASE_MAX_STEP = 0.05` (matches `tasks/ar4/residual_ik_action.py`'s constant of the same name — not imported cross-module since it's a private, underscore-prefixed constant there; redefine locally with a comment noting the match), `advance_tolerance = 0.03`, `lift_minimal_height = 0.03`, `pregrasp_hover = 0.05`, `lift_margin = 0.02`, `carry_height = 0.10` (all matching the values `pickplace_taskspace_env_cfg.py`/`pickplace_residual_env_cfg.py` already use), gripper action convention: raw action `>= 0` → open, `< 0` → close (verified directly against Isaac Lab's `BinaryJointAction.process_actions()`, `/home/saps/IsaacLab/source/isaaclab/isaaclab/envs/mdp/actions/binary_joint_actions.py:128-139` — **not** `GRIPPER_OPEN_POS`/`GRIPPER_CLOSED_POS` from `robot_cfg.py`, which are joint-angle radians used only for reading back actual joint state, a different quantity from the ±1 action command).
- Always invoke scripts via `/home/saps/IsaacLab/isaaclab.sh -p scripts/<script>.py` from the repo root, never plain `python`.
- Verify via file evidence (video file existence/size/duration, stdout logs) — this repo has no pytest-based unit tests for Isaac-Sim-dependent code.
- Any subagent dispatched to run a real (non-instant) Isaac Sim script must be told explicitly to block on completion itself (poll in a loop with sleep, or use a long single-call timeout) — do not end a turn expecting an external "background job finished" notification; this exact mistake has recurred multiple times elsewhere in this repo's session history.

---

### Task 1: Generalize `perceive_sphere` into `perceive_object`

**Files:**
- Modify: `scripts/_perception_adapter.py`

**Interfaces:**
- Consumes: `perception.pipeline.run_perception`, `perception.tracker.ObjectTracker`/`find_by_shape` (all pre-existing, unmodified).
- Produces: `perceive_object(env, camera, tracker, ground_z: float, shape_label: str, env_index: int = 0) -> tuple[torch.Tensor | None, list, np.ndarray]` — consumed by Task 3's demo script. `perceive_sphere(env, camera, tracker, ground_z: float, env_index: int = 0)` keeps its exact original signature and return type, now implemented as a thin wrapper — `interactive_demo.py` and `eval_loop.py --perception` (both call `perceive_sphere`) require zero changes.

- [ ] **Step 1: Replace `perceive_sphere` with `perceive_object` + a thin wrapper**

Current content of `scripts/_perception_adapter.py` (lines 41-67):

```python
def perceive_sphere(env, camera, tracker, ground_z: float, env_index: int = 0):
    """Runs perception on `camera`'s current frame for one env, updates `tracker`,
    and returns (sphere_position_in_robot_root_frame_or_None, tracked_objects,
    rgb_frame). `env` must be the raw ManagerBasedRLEnv (e.g. `env.unwrapped`),
    not the rsl_rl-wrapped env. `env_index` selects which parallel env's camera/
    robot data to read (default 0, matching eval_loop.py/interactive_demo.py's
    single-env usage)."""
    depth = camera.data.output["distance_to_image_plane"][env_index, ..., 0].cpu().numpy()
    rgb = camera.data.output["rgb"][env_index, ..., :3].cpu().numpy().astype(np.uint8)
    intrinsics = camera.data.intrinsic_matrices[env_index].cpu().numpy()
    cam_pos = camera.data.pos_w[env_index].cpu().numpy()
    cam_quat_ros = camera.data.quat_w_ros[env_index].cpu().numpy()

    detections = run_perception(depth, intrinsics, cam_pos, cam_quat_ros, ground_z=ground_z)
    tracked = tracker.update(detections)
    sphere = find_by_shape(tracked, "sphere")
    if sphere is None:
        return None, tracked, rgb

    object_pos_w = torch.tensor(sphere.position, dtype=torch.float32, device=env.device).unsqueeze(0)
    robot = env.scene["robot"]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w[env_index : env_index + 1],
        robot.data.root_quat_w[env_index : env_index + 1],
        object_pos_w,
    )
    return object_pos_b, tracked, rgb
```

Replace it with:

```python
def perceive_object(env, camera, tracker, ground_z: float, shape_label: str, env_index: int = 0):
    """Runs perception on `camera`'s current frame for one env, updates `tracker`,
    and returns (object_position_in_robot_root_frame_or_None, tracked_objects,
    rgb_frame) for the first tracked object matching `shape_label`. `env` must
    be the raw ManagerBasedRLEnv (e.g. `env.unwrapped`), not the rsl_rl-wrapped
    env. `env_index` selects which parallel env's camera/robot data to read
    (default 0, matching eval_loop.py/interactive_demo.py/
    classical_pickplace_demo.py's single-env usage)."""
    depth = camera.data.output["distance_to_image_plane"][env_index, ..., 0].cpu().numpy()
    rgb = camera.data.output["rgb"][env_index, ..., :3].cpu().numpy().astype(np.uint8)
    intrinsics = camera.data.intrinsic_matrices[env_index].cpu().numpy()
    cam_pos = camera.data.pos_w[env_index].cpu().numpy()
    cam_quat_ros = camera.data.quat_w_ros[env_index].cpu().numpy()

    detections = run_perception(depth, intrinsics, cam_pos, cam_quat_ros, ground_z=ground_z)
    tracked = tracker.update(detections)
    obj = find_by_shape(tracked, shape_label)
    if obj is None:
        return None, tracked, rgb

    object_pos_w = torch.tensor(obj.position, dtype=torch.float32, device=env.device).unsqueeze(0)
    robot = env.scene["robot"]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w[env_index : env_index + 1],
        robot.data.root_quat_w[env_index : env_index + 1],
        object_pos_w,
    )
    return object_pos_b, tracked, rgb


def perceive_sphere(env, camera, tracker, ground_z: float, env_index: int = 0):
    """Thin wrapper around perceive_object for the sphere-specific call sites
    (eval_loop.py --perception, interactive_demo.py) - neither needs any
    changes as a result of perceive_object's addition."""
    return perceive_object(env, camera, tracker, ground_z, "sphere", env_index)
```

- [ ] **Step 2: Syntax-check the file**

Run: `python3 -c "import ast; ast.parse(open('scripts/_perception_adapter.py').read())"`
Expected: no output.

- [ ] **Step 3: Verify the two existing call sites are unaffected**

Run: `grep -n "perceive_sphere" scripts/interactive_demo.py scripts/eval_loop.py`
Expected: both files still reference `perceive_sphere` with the same call signature as before (`perceive_sphere(env.unwrapped, camera, tracker, GROUND_Z)` style calls) — confirms no caller needs updating.

- [ ] **Step 4: Commit**

```bash
git add scripts/_perception_adapter.py
git commit -m "Generalize perceive_sphere into perceive_object (shape-label parametrized)"
```

---

### Task 2: Scene + env cfg for the classical demo

**Files:**
- Create: `tasks/ar4/classical_demo_env_cfg.py`

**Interfaces:**
- Consumes: `CUBE_CFG` (from `tasks/ar4/objects_cfg.py`, unmodified), `_EE_OFFSET`/`_PERCEPTION_CAMERA_POS`/`_PERCEPTION_CAMERA_QUAT_WORLD` (from `tasks/ar4/pickplace_env_cfg.py`, unmodified), `AR4_MK5_CFG`/`ARM_JOINT_NAMES`/`GRIPPER_JOINT_NAMES`/`GRIPPER_OPEN_POS`/`GRIPPER_CLOSED_POS` (from `tasks/ar4/robot_cfg.py`, unmodified).
- Produces: `Ar4ClassicalDemoEnvCfg` class (a `ManagerBasedEnvCfg`, not the RL variant) — consumed by Task 3's demo script. Scene exposes `env.scene["robot"]`, `env.scene["cube"]`, `env.scene["ee_frame"]`, `env.scene["perception_camera"]`.

- [ ] **Step 1: Write the new file**

```python
# tasks/ar4/classical_demo_env_cfg.py
"""Scene + env cfg for the classical (zero-RL) camera-perception-driven
pick-and-place demo (scripts/classical_pickplace_demo.py). See
docs/superpowers/specs/2026-07-07-ar4-classical-perception-pickplace-demo-design.md.

Additive/parallel to every other tasks/ar4/*.py file: does NOT modify
env_cfg.py, objects_cfg.py, or any pickplace_*.py file. Closely modeled on
Ar4PickPlaceSingleObjectSceneCfg (pickplace_single_object_env_cfg.py),
swapping SPHERE_CFG for CUBE_CFG to match this repo's current single-cube
scope, and on Ar4EnvCfg (env_cfg.py) for the bare ManagerBasedEnvCfg
pattern (no reward/termination managers - this is a scripted demo, not an
RL task).

Import this module only after an Isaac Sim/Isaac Lab AppLauncher has been
created.
"""

import isaaclab.envs.mdp as isaaclab_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils.configclass import configclass

from .objects_cfg import CUBE_CFG
from .pickplace_env_cfg import _EE_OFFSET, _PERCEPTION_CAMERA_POS, _PERCEPTION_CAMERA_QUAT_WORLD
from .robot_cfg import AR4_MK5_CFG, ARM_JOINT_NAMES, GRIPPER_CLOSED_POS, GRIPPER_JOINT_NAMES, GRIPPER_OPEN_POS


@configclass
class Ar4ClassicalDemoSceneCfg(InteractiveSceneCfg):
    """AR4 gripper + a single cube, plus the end-effector FrameTransformer
    sensor and the top-down RGB-D perception camera - both always on, since
    this scene exists specifically for the live-perception classical demo.

    Deliberately duplicates (rather than subclasses) Ar4SceneCfg/
    Ar4PickPlaceSceneCfg/Ar4PickPlaceSingleObjectSceneCfg, since all three
    include either the full four-object set or the sphere instead of the
    cube - same rationale as Ar4PickPlaceSingleObjectSceneCfg's own
    docstring.
    """

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )
    robot: ArticulationCfg = AR4_MK5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cube: RigidObjectCfg = CUBE_CFG

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

    perception_camera: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/PerceptionCamera",
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=40.0, clipping_range=(0.2, 1.0)
        ),
        offset=CameraCfg.OffsetCfg(pos=_PERCEPTION_CAMERA_POS, rot=_PERCEPTION_CAMERA_QUAT_WORLD, convention="world"),
    )


@configclass
class ActionsCfg:
    """Task-space Cartesian action (arm) + binary gripper command - the
    same plain (non-residual) DifferentialInverseKinematicsActionCfg every
    task-space env cfg in this repo uses. This demo has no policy: the
    script itself commands the base pursuit step directly as the raw
    action each step (see scripts/classical_pickplace_demo.py), so there
    is nothing to add a residual on top of."""

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
    """Minimal observation group (joint pos/vel only) - required by
    ManagerBasedEnvCfg's manager machinery, matching Ar4EnvCfg's own
    minimal ObservationsCfg (env_cfg.py). Not used by the demo script's
    own control logic, which reads camera/ee_frame/robot state directly."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=isaaclab_mdp.joint_pos)
        joint_vel = ObsTerm(func=isaaclab_mdp.joint_vel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class Ar4ClassicalDemoEnvCfg(ManagerBasedEnvCfg):
    """Bare (non-RL) env cfg for the classical perception-driven demo - no
    reward/termination managers, matching Ar4EnvCfg's (env_cfg.py)
    precedent for scripted demos. num_envs=1 (single demo instance,
    matching interactive_demo.py's convention)."""

    scene: Ar4ClassicalDemoSceneCfg = Ar4ClassicalDemoSceneCfg(num_envs=1, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()

    def __post_init__(self) -> None:
        self.decimation = 2
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0)
        self.viewer.eye = (1.5, 1.5, 1.2)
        self.viewer.lookat = (0.0, 0.0, 0.4)
```

- [ ] **Step 2: Syntax-check the file**

Run: `python3 -c "import ast; ast.parse(open('tasks/ar4/classical_demo_env_cfg.py').read())"`
Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add tasks/ar4/classical_demo_env_cfg.py
git commit -m "Add Ar4ClassicalDemoEnvCfg: single-cube scene with camera, no RL managers"
```

---

### Task 3: The demo script — waypoint planning + live execution loop

**Files:**
- Create: `scripts/classical_pickplace_demo.py`

**Interfaces:**
- Consumes: `Ar4ClassicalDemoEnvCfg` (Task 2), `perceive_object` (Task 1), `perception.overlay.draw_detections` (pre-existing, unmodified).
- Produces: an executable script writing a video to `logs/videos/ar4_classical_pickplace_demo.mp4` — consumed by Task 4's verification.

- [ ] **Step 1: Write the new file**

```python
# scripts/classical_pickplace_demo.py
"""Fully classical (zero-RL) pick-and-place demo: the camera detects the
cube live, its position is transformed into the robot's root frame, a
5-waypoint Cartesian path is planned live from that detection, and a live
differential-IK controller executes pregrasp -> grasp -> lift -> transit ->
place -> release -> home. No policy, no training, no reward function - see
docs/superpowers/specs/2026-07-07-ar4-classical-perception-pickplace-demo-design.md.

Contrast with the two existing demos: grasp_demo.py uses hardcoded,
offline-precomputed joint waypoints (no camera, no live planning);
interactive_demo.py uses live camera perception but drives a *trained RL
policy*, not a classical planner. This script is the missing "fully
classical, live end-to-end" demonstration.

.. code-block:: bash

    ./isaaclab.sh -p scripts/classical_pickplace_demo.py
"""

import argparse
import os
import subprocess
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Classical (zero-RL) camera-perception-driven pick-and-place demo.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import time

import imageio
import torch

from isaaclab.envs import ManagerBasedEnv  # noqa: E402
from isaaclab.utils.math import subtract_frame_transforms  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _perception_adapter import perceive_object  # noqa: E402
from perception.overlay import draw_detections  # noqa: E402
from perception.tracker import ObjectTracker  # noqa: E402
from tasks.ar4.classical_demo_env_cfg import Ar4ClassicalDemoEnvCfg  # noqa: E402
from tasks.ar4.pickplace_env_cfg import GROUND_Z  # noqa: E402

VIDEO_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "videos", "ar4_classical_pickplace_demo.mp4"
)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Same values as pickplace_taskspace_env_cfg.py/pickplace_residual_env_cfg.py's
# waypoint-planning constants.
_LIFT_MINIMAL_HEIGHT = 0.03
_PREGRASP_HOVER = 0.05
_LIFT_MARGIN = 0.02
_CARRY_HEIGHT = 0.10
_ADVANCE_TOLERANCE = 0.03
_BASE_MAX_STEP = 0.05  # matches tasks/ar4/residual_ik_action.py's _BASE_MAX_STEP
_GOAL_POS_B = (0.0, 0.35, 0.02)  # fixed demo goal, robot root frame
_GRIPPER_OPEN_ACTION = 1.0  # BinaryJointAction convention: raw action >= 0 -> open
_GRIPPER_CLOSE_ACTION = -1.0  # raw action < 0 -> close
_TIMEOUT_S = 15.0  # generous fixed budget - no RL episode-length constraint here


def _raise_window_in_background(title_substr: str, duration_s: float) -> None:
    """Best-effort: keep the sim window raised on desktops that don't auto-focus new windows."""
    try:
        subprocess.Popen(
            ["python3", os.path.join(SCRIPT_DIR, "_raise_window.py"), title_substr, str(duration_s)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        pass  # not fatal if this desktop doesn't support it


def plan_waypoints(
    cube_pos_b: torch.Tensor,
    goal_pos_b: torch.Tensor,
    lift_minimal_height: float,
    pregrasp_hover: float,
    lift_margin: float,
    carry_height: float,
) -> torch.Tensor:
    """Same 5-waypoint geometry as tasks/ar4/mdp.py's compute_path_waypoints
    (pregrasp/grasp/lift/transit/place), adapted to take explicit
    already-perceived positions instead of reading privileged env state -
    see the design spec's "Path planning" section for why this is a
    standalone pure function rather than a refactor of the Isaac Lab
    EventTerm. cube_pos_b/goal_pos_b: shape (1, 3), robot root frame.
    Returns shape (5, 3)."""
    pregrasp = cube_pos_b.clone()
    pregrasp[:, 2] += pregrasp_hover

    grasp = cube_pos_b.clone()

    lift = cube_pos_b.clone()
    lift[:, 2] = lift_minimal_height + lift_margin

    transit = torch.zeros_like(cube_pos_b)
    transit[:, 0] = (cube_pos_b[:, 0] + goal_pos_b[:, 0]) / 2.0
    transit[:, 1] = (cube_pos_b[:, 1] + goal_pos_b[:, 1]) / 2.0
    transit[:, 2] = carry_height

    place = goal_pos_b.clone()

    return torch.cat([pregrasp, grasp, lift, transit, place], dim=0)


def _gripper_action_for_waypoint(waypoint_idx: int) -> float:
    """Same schedule as tasks/ar4/mdp.py's gripper_schedule_bonus: open
    through waypoints 0-1 (pre-grasp, grasp-approach), closed from
    waypoint 2 onward (lift, transit, place)."""
    return _GRIPPER_OPEN_ACTION if waypoint_idx < 2 else _GRIPPER_CLOSE_ACTION


def main() -> None:
    env_cfg = Ar4ClassicalDemoEnvCfg()
    env_cfg.sim.device = args_cli.device
    env = ManagerBasedEnv(cfg=env_cfg)

    _raise_window_in_background("Isaac Sim Python", _TIMEOUT_S + 15.0)

    camera = env.scene["perception_camera"]
    ee_frame = env.scene["ee_frame"]
    robot = env.scene["robot"]
    tracker = ObjectTracker()
    video_writer = imageio.get_writer(VIDEO_PATH, fps=int(1.0 / env.step_dt), codec="libx264")

    goal_pos_b = torch.tensor([_GOAL_POS_B], device=env.device)
    timeout_steps = int(_TIMEOUT_S / env.step_dt)

    waypoints = None
    waypoint_idx = 0
    reached_final = False

    with torch.inference_mode():
        env.reset()
        for step in range(timeout_steps):
            cube_pos_b, tracked, rgb = perceive_object(env, camera, tracker, GROUND_Z, "cube")
            video_writer.append_data(draw_detections(rgb, tracked))

            if waypoints is None:
                if cube_pos_b is None:
                    # No detection yet - hold position, don't plan or move.
                    action = torch.zeros(env.num_envs, 4, device=env.device)
                    action[:, 3] = _GRIPPER_OPEN_ACTION
                    env.step(action)
                    continue
                waypoints = plan_waypoints(
                    cube_pos_b, goal_pos_b, _LIFT_MINIMAL_HEIGHT, _PREGRASP_HOVER, _LIFT_MARGIN, _CARRY_HEIGHT
                )
                print(f"[INFO] Cube detected at {cube_pos_b.tolist()}. Plan computed, executing.")

            ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
            root_pos_w = robot.data.root_pos_w
            root_quat_w = robot.data.root_quat_w
            ee_pos_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w)

            current_waypoint = waypoints[waypoint_idx].unsqueeze(0)
            direction = current_waypoint - ee_pos_b
            dist = torch.norm(direction, dim=-1, keepdim=True)

            if dist.item() < _ADVANCE_TOLERANCE and waypoint_idx < 4:
                waypoint_idx += 1
                current_waypoint = waypoints[waypoint_idx].unsqueeze(0)
                direction = current_waypoint - ee_pos_b
                dist = torch.norm(direction, dim=-1, keepdim=True)

            if waypoint_idx == 4 and dist.item() < _ADVANCE_TOLERANCE:
                reached_final = True

            step_mag = torch.clamp(dist, max=_BASE_MAX_STEP)
            pursuit_delta = direction / (dist + 1e-8) * step_mag

            action = torch.zeros(env.num_envs, 4, device=env.device)
            action[:, 0:3] = pursuit_delta
            action[:, 3] = _gripper_action_for_waypoint(waypoint_idx)
            env.step(action)

            if reached_final:
                print("[INFO] Place waypoint reached. Releasing and holding.")
                for _ in range(60):
                    cube_pos_b, tracked, rgb = perceive_object(env, camera, tracker, GROUND_Z, "cube")
                    video_writer.append_data(draw_detections(rgb, tracked))
                    release_action = torch.zeros(env.num_envs, 4, device=env.device)
                    release_action[:, 3] = _GRIPPER_OPEN_ACTION
                    env.step(release_action)
                break
        else:
            print(f"[INFO] Timed out after {_TIMEOUT_S}s without reaching the place waypoint (idx={waypoint_idx}).")

    video_writer.close()
    print("Done. Holding window open for a few seconds before closing...")
    time.sleep(5.0)
    env.close()
    print(f"Demo video written to: {VIDEO_PATH}")


if __name__ == "__main__":
    main()
    simulation_app.close()
```

- [ ] **Step 2: Syntax-check the file**

Run: `python3 -c "import ast; ast.parse(open('scripts/classical_pickplace_demo.py').read())"`
Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add scripts/classical_pickplace_demo.py
git commit -m "Add classical_pickplace_demo.py: live camera-planned, zero-RL pick-and-place"
```

---

### Task 4: Smoke test + real run + video verification

**Files:**
- None created — this task produces a video file and a short verification note appended to `.superpowers/sdd/task-4-report.md` (git-ignored scratch, not a committed deliverable).

**Interfaces:**
- Consumes: Tasks 1-3's complete script.
- Produces: `logs/videos/ar4_classical_pickplace_demo.mp4`, verified to exist with reasonable size/duration — the controller (not this task) will personally inspect the video's actual content afterward, matching this session's established practice for decisive/diagnostic evidence.

- [ ] **Step 1: Headless smoke test**

Run (from repo root, foreground, allow up to 5 minutes — a `timeout`/nonzero exit code alone is not proof of failure in this repo, Isaac Sim's shutdown sequence sometimes hangs after genuinely finishing; verify via the video file instead):

```bash
rm -f /tmp/classical_demo_smoke_stdout.log
timeout 280 /home/saps/IsaacLab/isaaclab.sh -p scripts/classical_pickplace_demo.py --headless > /tmp/classical_demo_smoke_stdout.log 2>&1
echo "exit_code=$?"
```

Check for a real traceback (not just startup/shutdown noise):
```bash
grep -i "error\|exception\|traceback" /tmp/classical_demo_smoke_stdout.log
```

If a real exception appears, read the context and fix the underlying bug before proceeding — this is the first time this task's code actually runs inside Isaac Sim; a bug here is genuine, useful information, not something to route around. Likely culprits if something fails: a prim-path mismatch in `Ar4ClassicalDemoSceneCfg` (cross-check against `Ar4PickPlaceSingleObjectSceneCfg`'s exact structure if this happens), or a shape mismatch in the action tensor (`action` must be `(num_envs, 4)` — 3 for the Cartesian arm delta + 1 for the gripper scalar, matching `grasp_demo.py`'s precedent of `action[:, num_joints] = gripper_cmd`).

Check the video file was created:
```bash
ls -la logs/videos/ar4_classical_pickplace_demo.mp4
```
Expected: file exists, non-trivial size (more than a few KB — an empty/near-empty file suggests the loop exited almost immediately, worth investigating even if no exception was thrown).

- [ ] **Step 2: Real (non-headless) run**

```bash
/home/saps/IsaacLab/isaaclab.sh -p scripts/classical_pickplace_demo.py
```

This opens a visible Isaac Sim window and runs for up to `_TIMEOUT_S` (15s) plus setup/teardown time — let it run to completion (or timeout) rather than killing it early. Confirm the video file was overwritten with a fresh, non-trivial-sized `.mp4`:

```bash
ls -la logs/videos/ar4_classical_pickplace_demo.mp4
```

- [ ] **Step 3: Report — do not judge success, just confirm the artifact exists**

Write a short note to `.superpowers/sdd/task-4-report.md`: the video file's path, size, and (via `ffprobe` or `python3 -c "import imageio; r = imageio.get_reader('logs/videos/ar4_classical_pickplace_demo.mp4'); print(r.count_frames(), r.get_meta_data())"`) its frame count and duration. Report the stdout log's own printed status line (`"[INFO] Place waypoint reached..."` or `"[INFO] Timed out..."`) verbatim, since that's a real signal from the run itself — but do not editorialize about whether the demo "succeeded" beyond that literal printed line. The actual judgment (does the video show a real pick-and-place) is made separately by watching the frames, not by this task.

Do not commit anything in this task — `logs/videos/*.mp4` is a runtime artifact, not source, and this repo's other demo scripts' videos aren't committed either.
