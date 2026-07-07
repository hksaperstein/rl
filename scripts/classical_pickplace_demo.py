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
