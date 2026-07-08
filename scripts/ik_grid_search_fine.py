"""Fine local grid search around ik_grid_search.py's / ik_polish_from_grid.py's
best-found region, to close the remaining ~3.6cm gap further - the cube is
only 1.8cm per side, so sub-centimeter precision is likely needed for an
actual grasp attempt.

.. code-block:: bash

    PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/ik_grid_search_fine.py --headless
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Fine local grid search around the best-known AR4 grasp config.")
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

# Center the fine search on ik_polish_from_grid.py's best result:
# [0.6054, 1.3090, 0.2917, -0.0143, 1.4968, 0.0013], residual 0.03648m
CENTER_J1 = 0.6054
CENTER_J2 = 1.3090
CENTER_J3 = 0.2917
SEARCH_HALF_WIDTH = 0.15  # rad, search +/- this much around the center on each of j1/j2/j3


def main() -> None:
    env_cfg = Ar4GraspVerifyEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device

    env = ManagerBasedEnv(cfg=env_cfg)
    robot = env.scene["robot"]
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES, body_names=["link_6"])
    robot_entity_cfg.resolve(env.scene)

    with torch.inference_mode():
        env.reset()

        root_pos_w = robot.data.root_pos_w.clone()
        root_quat_w = robot.data.root_quat_w.clone()
        cube_pos_w = torch.tensor([CUBE_POS_W], device=env.device)
        cube_pos_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, cube_pos_w)
        target_pos_b = cube_pos_b.clone()
        target_pos_b[:, 2] = GRASP_AT_HEIGHT
        print(f"[INFO] Target (robot frame): {target_pos_b[0].tolist()}")

        N = 11  # 11x11x11 = 1331 grid points, fine local search
        settle_steps = 15
        num_arm_joints = len(ARM_JOINT_NAMES)

        best_dist = float("inf")
        best_q = None
        best_ee = None

        print(f"\n[INFO] Sweeping {N}x{N}x{N}={N*N*N} fine local grid points (+/-{SEARCH_HALF_WIDTH} rad around center)...")
        count = 0
        for i in range(N):
            j1 = CENTER_J1 - SEARCH_HALF_WIDTH + (2 * SEARCH_HALF_WIDTH) * i / (N - 1)
            for j in range(N):
                j2 = CENTER_J2 - SEARCH_HALF_WIDTH + (2 * SEARCH_HALF_WIDTH) * j / (N - 1)
                for k in range(N):
                    j3 = CENTER_J3 - SEARCH_HALF_WIDTH + (2 * SEARCH_HALF_WIDTH) * k / (N - 1)
                    q = [j1, j2, j3, 0.0, 0.0, 0.0]

                    for _ in range(settle_steps):
                        action = torch.zeros(env.num_envs, num_arm_joints + 1, device=env.device)
                        for jj in range(num_arm_joints):
                            action[:, jj] = q[jj]
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
                        print(f"  [{count}/{N*N*N}] NEW BEST: q=[{j1:.4f},{j2:.4f},{j3:.4f}] -> ee={['%.4f' % x for x in best_ee]} dist={dist:.4f}m")

        print("\n" + "=" * 80)
        print("FINE GRID SEARCH RESULT")
        print("=" * 80)
        print(f"Best joint config found: {best_q}")
        print(f"Best EE position: {best_ee}")
        print(f"Best distance to target: {best_dist:.5f}m")
        print(f"Cube size: 0.018m per side - is this within grasping tolerance? {'YES' if best_dist < 0.015 else 'MARGINAL/NO'}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
