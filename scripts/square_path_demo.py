"""3-DOF demo: move the AR4 end-effector around a square path parallel to
the ground plane (fixed height, robot-frame XY square), using only
joint_1/2/3 for position - joints 4-6 and the gripper are held "limp"
(commanded to their own current position / a fixed open state), same
convention as scripts/grasp_demo_v2.py.

Reuses this session's validated solving method (empirically-calibrated
joint_1-to-azimuth mapping + forward-kinematics grid search over
joint_2/joint_3 + a bounded-step DLS polish) instead of a live per-step
reactive IK loop, which repeatedly failed to converge earlier this
session. joint_1 is a pure base-yaw rotation about the vertical axis, so
the calibration generalizes to any target direction, not just the one
point it was originally measured against.

.. code-block:: bash

    PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/square_path_demo.py --headless
"""

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="3-DOF square-path demo, parallel to the ground plane.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import imageio  # noqa: E402
import torch  # noqa: E402

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: E402
from isaaclab.envs import ManagerBasedEnv  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.utils.math import subtract_frame_transforms  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from tasks.ar4.grasp_verify_env_cfg import Ar4GraspVerifyEnvCfg  # noqa: E402
from tasks.ar4.robot_cfg import ARM_JOINT_NAMES  # noqa: E402

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
VIDEO_PATH = os.path.join(LOG_DIR, "videos", "ar4_square_path_demo.mp4")

GRIPPER_OPEN = 1.0
HOME_Q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
CALIBRATION_C = -1.5677  # empirically measured joint_1 -> EE-azimuth offset (grasp_demo_v2.py investigation)

# Square path: robot-frame, fixed height (parallel to ground), 4 corners +
# 4 edge midpoints for a smoother traced outline. Kept the same
# well-converged center/height (0.25, 0.0, 0.08) from the previous,
# working iteration - just enlarged from side=0.10 to side=0.16. Corner
# distances from base: 0.166m-0.339m, still comfortably inside the arm's
# measured 0.538m reach envelope.
SQUARE_Z = 0.08
SQUARE_POINTS_B = [
    (0.17, -0.08, SQUARE_Z),  # corner 1
    (0.17, 0.00, SQUARE_Z),  # mid 1-2
    (0.17, 0.08, SQUARE_Z),  # corner 2
    (0.25, 0.08, SQUARE_Z),  # mid 2-3
    (0.33, 0.08, SQUARE_Z),  # corner 3
    (0.33, 0.00, SQUARE_Z),  # mid 3-4
    (0.33, -0.08, SQUARE_Z),  # corner 4
    (0.25, -0.08, SQUARE_Z),  # mid 4-1
]

GRID_N = 15
GRID_SETTLE_STEPS = 10
POLISH_STEP_MAX = 0.05
POLISH_ROUNDS = 12
POLISH_SETTLE_STEPS = 25
HOLD_STEPS_PER_POINT = 70


def solve_point(env, ik_controller, robot_entity_cfg, ik_jacobi_idx, target_pos_b, j2_min, j2_max, j3_min, j3_max):
    robot = env.scene["robot"]
    num_arm_joints = len(ARM_JOINT_NAMES)

    target_bearing = math.atan2(target_pos_b[1], target_pos_b[0])
    seed_j1 = -(target_bearing - CALIBRATION_C)

    best_dist = float("inf")
    best_q = None
    target_t = torch.tensor(target_pos_b, device=env.device)
    for i in range(GRID_N):
        j2 = j2_min + (j2_max - j2_min) * i / (GRID_N - 1)
        for k in range(GRID_N):
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
            dist = torch.norm(ee_pos_b[0] - target_t).item()
            if dist < best_dist:
                best_dist = dist
                best_q = list(q)

    seed_q = torch.tensor([best_q], device=env.device)
    robot.write_joint_position_to_sim(seed_q, joint_ids=robot_entity_cfg.joint_ids, env_ids=torch.tensor([0], device=env.device))
    for _ in range(POLISH_SETTLE_STEPS * 2):
        action = torch.zeros(env.num_envs, num_arm_joints + 1, device=env.device)
        action[:, :num_arm_joints] = seed_q
        action[:, num_arm_joints] = GRIPPER_OPEN
        env.step(action)

    final_residual = best_dist
    for _ in range(POLISH_ROUNDS):
        current_joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids].clone()
        ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            robot.data.root_pose_w[:, 0:3], robot.data.root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        direction = target_t.unsqueeze(0) - ee_pos_b
        dist = torch.norm(direction, dim=-1, keepdim=True)
        step_target_b = ee_pos_b + direction / (dist + 1e-8) * torch.clamp(dist, max=POLISH_STEP_MAX)
        jacobian = robot.root_physx_view.get_jacobians()[:, ik_jacobi_idx, :, robot_entity_cfg.joint_ids]
        ik_controller.set_command(step_target_b, ee_pos=ee_pos_b, ee_quat=ee_quat_b)
        joint_pos_des = ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, current_joint_pos)
        # Lock joint_1 to its calibrated seed value during polish - letting
        # DLS freely adjust all 3 joints let it drag joint_1 far from a
        # good grid-search seed on 2/8 square-path points this session,
        # landing much worse (0.24m/0.30m residual) despite the grid
        # search itself having found a good starting point. Restricting
        # polish to the 2 DOF it actually needs (matching the calibrated
        # bearing already established for joint_1) avoids that failure mode.
        joint_pos_des[:, 0] = seed_j1
        for _ in range(POLISH_SETTLE_STEPS):
            action = torch.zeros(env.num_envs, num_arm_joints + 1, device=env.device)
            for j in range(num_arm_joints):
                action[:, j] = joint_pos_des[:, j]
            action[:, num_arm_joints] = GRIPPER_OPEN
            env.step(action)
        ee_pose_w_now = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
        ee_pos_b_now, _ = subtract_frame_transforms(
            robot.data.root_pose_w[:, 0:3], robot.data.root_pose_w[:, 3:7], ee_pose_w_now[:, 0:3], ee_pose_w_now[:, 3:7]
        )
        final_residual = torch.norm(ee_pos_b_now[0] - target_t).item()
        if final_residual < 0.01:
            break

    return robot.data.joint_pos[0, robot_entity_cfg.joint_ids].tolist(), final_residual


def main() -> None:
    env_cfg = Ar4GraspVerifyEnvCfg()
    env_cfg.sim.device = args_cli.device
    env = ManagerBasedEnv(cfg=env_cfg)

    robot = env.scene["robot"]
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES, body_names=["link_6"])
    robot_entity_cfg.resolve(env.scene)
    ik_jacobi_idx = robot_entity_cfg.body_ids[0] - 1 if robot.is_fixed_base else robot_entity_cfg.body_ids[0]
    num_arm_joints = len(ARM_JOINT_NAMES)

    joint_pos_limits = robot.data.joint_pos_limits[:, robot_entity_cfg.joint_ids]
    j2_min, j2_max = joint_pos_limits[0, 1, 0].item(), joint_pos_limits[0, 1, 1].item()
    j3_min, j3_max = joint_pos_limits[0, 2, 0].item(), joint_pos_limits[0, 2, 1].item()

    ik_cfg = DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls")
    ik_controller = DifferentialIKController(ik_cfg, num_envs=env.num_envs, device=env.device)

    os.makedirs(os.path.dirname(VIDEO_PATH), exist_ok=True)
    video_writer = imageio.get_writer(VIDEO_PATH, fps=int(1.0 / env.step_dt), codec="libx264")
    camera = env.scene["demo_camera"]

    with torch.inference_mode():
        env.reset()

        print(f"\n[INFO] Solving {len(SQUARE_POINTS_B)} square-path waypoints (grid search + DLS polish each)...")
        solved_qs = []
        for idx, pt in enumerate(SQUARE_POINTS_B):
            print(f"\n[INFO] Solving point {idx}: {pt}")
            q, residual = solve_point(env, ik_controller, robot_entity_cfg, ik_jacobi_idx, pt, j2_min, j2_max, j3_min, j3_max)
            print(f"  -> residual {residual:.5f}m, q={['%.4f' % x for x in q]}")
            solved_qs.append(q)

        print("\n[INFO] All waypoints solved. Executing square path (looping wrist/gripper limp, arm off, then square, then home)...\n")

        def hold_wrist_limp_and_move(target_q, steps):
            for _ in range(steps):
                wrist_now = robot.data.joint_pos[0, robot_entity_cfg.joint_ids][3:6].tolist()
                q = list(target_q[:3]) + wrist_now
                action = torch.zeros(env.num_envs, num_arm_joints + 1, device=env.device)
                for j in range(num_arm_joints):
                    action[:, j] = q[j]
                action[:, num_arm_joints] = GRIPPER_OPEN
                env.step(action)
                rgb = camera.data.output["rgb"][0].cpu().numpy()
                video_writer.append_data(rgb[:, :, :3].astype("uint8"))

        # Move from home to the first square point, then trace the loop
        # (returning to the first point at the end to close the square),
        # then back home.
        hold_wrist_limp_and_move(HOME_Q, 60)
        for idx, q in enumerate(solved_qs):
            print(f"[EXEC] moving to point {idx}")
            hold_wrist_limp_and_move(q, HOLD_STEPS_PER_POINT)
        print("[EXEC] closing the loop back to point 0")
        hold_wrist_limp_and_move(solved_qs[0], HOLD_STEPS_PER_POINT)
        print("[EXEC] returning home")
        hold_wrist_limp_and_move(HOME_Q, 90)

        env.reset()

    video_writer.close()
    env.close()
    print(f"\nVideo recorded to: {VIDEO_PATH}")


if __name__ == "__main__":
    main()
    simulation_app.close()
