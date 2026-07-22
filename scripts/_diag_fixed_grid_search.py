"""Diagnostic v4 (2026-07-22): re-run the GRASP-waypoint grid search with the
bug found this session fixed - the original grid_search_then_polish (raster
order: i=j2 outer, k=j3 inner) causes a ~2.46rad DISCONTINUOUS jump in j3
every time the outer i-loop advances (j3 wraps from j3_max back to j3_min),
and only allows 15 settle steps to recover - nowhere near enough for the arm
to actually converge from such a large jump. This produces transient,
mid-swing "distance" readings that can look artificially good (this
session's own smoking gun: q=[0.6233,1.1868,-1.4508,0,0,0] was reported at
0.033m by the original grid loop, but independently verified via TWO
different methods - teleport+hold and servo-from-reset - to genuinely
converge to 0.42m away).

Fix tested here: traverse j3 in a SNAKE/boustrophedon order (increasing for
even i, decreasing for odd i) so consecutive candidates are always adjacent
in joint space (no big jumps), plus more settle steps (30 instead of 15).
Prints the best distance found AND re-verifies it with a genuine long
(100-step) settle from a teleport, to confirm the reported number this time
is real, not transient.

.. code-block:: bash

    PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/_diag_fixed_grid_search.py
"""

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Re-run AR4 grasp-waypoint grid search with snake ordering + more settle.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

from isaaclab.envs import ManagerBasedEnv  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.utils.math import subtract_frame_transforms  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from tasks.ar4.grasp_verify_env_cfg import Ar4GraspVerifyEnvCfg  # noqa: E402
from tasks.ar4.robot_cfg import ARM_JOINT_NAMES  # noqa: E402

CUBE_POS_W = (0.20, 0.28, 0.009)
GRASP_AT_HEIGHT = 0.009
GRIPPER_OPEN = 1.0
CALIBRATION_C = -1.5677
GRID_N = 25
GRID_SETTLE_STEPS = 30  # up from 15


def main() -> None:
    env_cfg = Ar4GraspVerifyEnvCfg()
    env_cfg.sim.device = args_cli.device
    env_cfg.scene.robot.actuators["arm"].stiffness = 4000.0
    env_cfg.scene.robot.actuators["arm"].damping = 200.0

    env = ManagerBasedEnv(cfg=env_cfg)
    robot = env.scene["robot"]
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES, body_names=["link_6"])
    robot_entity_cfg.resolve(env.scene)
    num_arm_joints = len(ARM_JOINT_NAMES)

    joint_pos_limits = robot.data.joint_pos_limits[:, robot_entity_cfg.joint_ids]
    j2_min, j2_max = joint_pos_limits[0, 1, 0].item(), joint_pos_limits[0, 1, 1].item()
    j3_min, j3_max = joint_pos_limits[0, 2, 0].item(), joint_pos_limits[0, 2, 1].item()

    with torch.inference_mode():
        env.reset()

        root_pos_w = robot.data.root_pos_w.clone()
        root_quat_w = robot.data.root_quat_w.clone()
        cube_pos_w = torch.tensor([CUBE_POS_W], device=env.device)
        cube_pos_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, cube_pos_w)
        target_bearing = math.atan2(cube_pos_b[0, 1].item(), cube_pos_b[0, 0].item())
        seed_j1 = -(target_bearing - CALIBRATION_C)
        target_pos_b = cube_pos_b.clone()
        target_pos_b[:, 2] = GRASP_AT_HEIGHT
        print(f"[INFO] target_pos_b={target_pos_b[0].tolist()} seed_j1={seed_j1:.4f}")

        best_dist = float("inf")
        best_q = None
        for i in range(GRID_N):
            j2 = j2_min + (j2_max - j2_min) * i / (GRID_N - 1)
            k_range = range(GRID_N) if i % 2 == 0 else range(GRID_N - 1, -1, -1)  # SNAKE ORDER FIX
            for k in k_range:
                j3 = j3_min + (j3_max - j3_min) * k / (GRID_N - 1)
                q = [seed_j1, j2, j3, 0.0, 0.0, 0.0]
                for _ in range(GRID_SETTLE_STEPS):
                    action = torch.zeros(env.num_envs, num_arm_joints + 1, device=env.device)
                    for j in range(num_arm_joints):
                        action[:, j] = q[j]
                    action[:, num_arm_joints] = GRIPPER_OPEN
                    env.step(action)
                ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
                ee_pos_b, _ = subtract_frame_transforms(
                    robot.data.root_pose_w[:, 0:3], robot.data.root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
                )
                dist = torch.norm(ee_pos_b[0] - target_pos_b[0]).item()
                if dist < best_dist:
                    best_dist = dist
                    best_q = list(q)
            print(f"  [row i={i:2d}] j2={j2:.4f} running_best={best_dist:.5f} q={best_q}")

        print(f"[GRID-FIXED] best dist: {best_dist:.5f}m, config: {best_q}")

        # Re-verify with a genuine long settle from a clean teleport.
        seed_q = torch.tensor([best_q], device=env.device)
        robot.write_joint_position_to_sim(seed_q, joint_ids=robot_entity_cfg.joint_ids, env_ids=torch.tensor([0], device=env.device))
        zero_vel = torch.zeros((1, num_arm_joints), device=env.device)
        robot.write_joint_velocity_to_sim(zero_vel, joint_ids=robot_entity_cfg.joint_ids, env_ids=torch.tensor([0], device=env.device))
        for _ in range(100):
            action = torch.zeros(env.num_envs, num_arm_joints + 1, device=env.device)
            action[:, :num_arm_joints] = seed_q
            action[:, num_arm_joints] = GRIPPER_OPEN
            env.step(action)
        ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
        ee_pos_b, _ = subtract_frame_transforms(
            robot.data.root_pose_w[:, 0:3], robot.data.root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        verify_dist = torch.norm(ee_pos_b[0] - target_pos_b[0]).item()
        print(f"[VERIFY after 100-step clean-teleport settle] dist={verify_dist:.5f}m")

        env.reset()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
