"""Direct forward-kinematics grid search over joint_2/joint_3 (with the
now-calibrated joint_1 aimed at the cube's bearing) to find the closest-
matching configuration to the grasp target - no iterative DLS solve
involved at all, so no local-minimum/damping pathology is possible.

ik_seeded_start.py showed that seeding from the "maximum reach" joint_2/
joint_3 combination (elbow at its hard limit) lands in the right
direction but overshoots the target's actual 0.344m distance (that combo
reaches 0.538m) and the DLS solve barely corrects from there. This script
instead densely samples joint_2/joint_3 directly and picks whichever
sampled point's actual (measured, not linearized) forward-kinematics
position is closest to the true target - a simple, robust method that
can't get stuck in a local minimum the way gradient-based iteration can.

.. code-block:: bash

    PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/ik_grid_search.py --headless
"""

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Forward-kinematics grid search for AR4 grasp target.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

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
CALIBRATION_C = -1.5677  # empirically measured joint_1 -> EE-azimuth offset, see ik_seeded_start.py


def main() -> None:
    env_cfg = Ar4GraspVerifyEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device

    env = ManagerBasedEnv(cfg=env_cfg)
    robot = env.scene["robot"]
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES, body_names=["link_6"])
    robot_entity_cfg.resolve(env.scene)

    joint_pos_limits = robot.data.joint_pos_limits[:, robot_entity_cfg.joint_ids]
    j2_min, j2_max = joint_pos_limits[0, 1, 0].item(), joint_pos_limits[0, 1, 1].item()
    j3_min, j3_max = joint_pos_limits[0, 2, 0].item(), joint_pos_limits[0, 2, 1].item()

    with torch.inference_mode():
        env.reset()

        root_pos_w = robot.data.root_pos_w.clone()
        root_quat_w = robot.data.root_quat_w.clone()
        cube_pos_w = torch.tensor([CUBE_POS_W], device=env.device)
        cube_pos_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, cube_pos_w)
        target_pos_b = cube_pos_b.clone()
        target_pos_b[:, 2] = GRASP_AT_HEIGHT
        target_bearing = math.atan2(cube_pos_b[0, 1].item(), cube_pos_b[0, 0].item())
        seed_j1 = -(target_bearing - CALIBRATION_C)
        print(f"[INFO] Target (robot frame): {target_pos_b[0].tolist()}")
        print(f"[INFO] Calibrated joint_1: {seed_j1:.4f} rad")
        print(f"[INFO] joint_2 range: [{j2_min:.4f}, {j2_max:.4f}], joint_3 range: [{j3_min:.4f}, {j3_max:.4f}]")

        N = 25  # 25x25 = 625 grid points
        settle_steps = 15

        best_dist = float("inf")
        best_q = None
        best_ee = None
        num_arm_joints = len(ARM_JOINT_NAMES)

        print(f"\n[INFO] Sweeping {N}x{N}={N*N} joint_2/joint_3 grid points...")
        count = 0
        for i in range(N):
            j2 = j2_min + (j2_max - j2_min) * i / (N - 1)
            for k in range(N):
                j3 = j3_min + (j3_max - j3_min) * k / (N - 1)
                q = [seed_j1, j2, j3, 0.0, 0.0, 0.0]

                for _ in range(settle_steps):
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
                count += 1
                if dist < best_dist:
                    best_dist = dist
                    best_q = list(q)
                    best_ee = ee_pos_b[0].tolist()
                    print(f"  [{count}/{N*N}] NEW BEST: j2={j2:.4f} j3={j3:.4f} -> ee={['%.4f' % x for x in best_ee]} dist={dist:.4f}m")

        print("\n" + "=" * 80)
        print("GRID SEARCH RESULT")
        print("=" * 80)
        print(f"Best joint config found: {best_q}")
        print(f"Best EE position: {best_ee}")
        print(f"Best distance to target: {best_dist:.5f}m")
        print(f"Target: {target_pos_b[0].tolist()}")

        # Now settle at the best config for longer and re-measure, to confirm
        # this isn't a transient/still-moving reading.
        print("\n[INFO] Settling at best config for 60 more steps to confirm...")
        for _ in range(60):
            action = torch.zeros(env.num_envs, num_arm_joints + 1, device=env.device)
            for j in range(num_arm_joints):
                action[:, j] = best_q[j]
            action[:, num_arm_joints] = GRIPPER_OPEN
            env.step(action)
        ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
        ee_pos_b, _ = subtract_frame_transforms(
            robot.data.root_pose_w[:, 0:3], robot.data.root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        final_dist = torch.norm(ee_pos_b[0] - target_pos_b[0]).item()
        print(f"[INFO] Confirmed EE position after extended settle: {ee_pos_b[0].tolist()}")
        print(f"[INFO] Confirmed distance to target: {final_dist:.5f}m")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
