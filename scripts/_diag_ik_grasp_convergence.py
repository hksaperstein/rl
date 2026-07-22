"""Diagnostic (2026-07-22, ar4-grasp-ik-precision task): instrument the
grasp_demo_v2.py grid-search-then-DLS-polish loop for the GRASP waypoint
specifically (the one that plateaus at ~3.3cm residual; the PREGRASP
waypoint 5cm higher converges fine to ~7mm) - to see WHY it plateaus:
oscillating near a singularity, hitting joint limits, or just needs more
rounds/smaller steps.

Not part of the actual fix - throwaway instrumentation script, run once
live to gather data before deciding the fix.

.. code-block:: bash

    PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/_diag_ik_grasp_convergence.py
"""

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Diagnose AR4 grasp-waypoint IK polish convergence.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch  # noqa: E402

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: E402
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
GRID_SETTLE_STEPS = 15
POLISH_STEP_MAX = 0.05
POLISH_ROUNDS = 40
POLISH_SETTLE_STEPS = 30


def main() -> None:
    env_cfg = Ar4GraspVerifyEnvCfg()
    env_cfg.sim.device = args_cli.device
    # Same test-local actuator boost as grasp_demo_v2.py, so this diagnostic
    # isn't confounded by the already-diagnosed weak-arm-actuator issue.
    env_cfg.scene.robot.actuators["arm"].stiffness = 4000.0
    env_cfg.scene.robot.actuators["arm"].damping = 200.0

    env = ManagerBasedEnv(cfg=env_cfg)
    robot = env.scene["robot"]
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES, body_names=["link_6"])
    robot_entity_cfg.resolve(env.scene)
    ik_jacobi_idx = robot_entity_cfg.body_ids[0] - 1 if robot.is_fixed_base else robot_entity_cfg.body_ids[0]
    num_arm_joints = len(ARM_JOINT_NAMES)

    joint_pos_limits = robot.data.joint_pos_limits[:, robot_entity_cfg.joint_ids]
    print(f"[LIMITS] {joint_pos_limits[0].tolist()}")
    j2_min, j2_max = joint_pos_limits[0, 1, 0].item(), joint_pos_limits[0, 1, 1].item()
    j3_min, j3_max = joint_pos_limits[0, 2, 0].item(), joint_pos_limits[0, 2, 1].item()

    ik_cfg = DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls")
    ik_controller = DifferentialIKController(ik_cfg, num_envs=env.num_envs, device=env.device)

    with torch.inference_mode():
        env.reset()

        root_pos_w = robot.data.root_pos_w.clone()
        root_quat_w = robot.data.root_quat_w.clone()
        cube_pos_w = torch.tensor([CUBE_POS_W], device=env.device)
        cube_pos_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, cube_pos_w)
        target_bearing = math.atan2(cube_pos_b[0, 1].item(), cube_pos_b[0, 0].item())
        seed_j1 = -(target_bearing - CALIBRATION_C)
        print(f"[INFO] Cube (robot frame): {cube_pos_b[0].tolist()}, calibrated joint_1: {seed_j1:.4f}")

        target_pos_b = cube_pos_b.clone()
        target_pos_b[:, 2] = GRASP_AT_HEIGHT

        # --- Grid search (identical to grasp_demo_v2.py) ---
        best_dist = float("inf")
        best_q = None
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
                dist = torch.norm(ee_pos_b[0] - target_pos_b[0]).item()
                if dist < best_dist:
                    best_dist = dist
                    best_q = list(q)
        print(f"[GRID] best coarse dist: {best_dist:.5f}m, config: {best_q}")

        seed_q = torch.tensor([best_q], device=env.device)
        robot.write_joint_position_to_sim(seed_q, joint_ids=robot_entity_cfg.joint_ids, env_ids=torch.tensor([0], device=env.device))
        for _ in range(POLISH_STEP_MAX and 60):
            action = torch.zeros(env.num_envs, num_arm_joints + 1, device=env.device)
            action[:, :num_arm_joints] = seed_q
            action[:, num_arm_joints] = GRIPPER_OPEN
            env.step(action)

        # --- Instrumented DLS polish ---
        for round_num in range(POLISH_ROUNDS):
            current_joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids].clone()
            ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                robot.data.root_pose_w[:, 0:3], robot.data.root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            direction = target_pos_b - ee_pos_b
            dist = torch.norm(direction, dim=-1, keepdim=True)
            step_target_b = ee_pos_b + direction / (dist + 1e-8) * torch.clamp(dist, max=POLISH_STEP_MAX)

            jacobian = robot.root_physx_view.get_jacobians()[:, ik_jacobi_idx, :, robot_entity_cfg.joint_ids]
            jacobian_pos = jacobian[:, 0:3]
            svals = torch.linalg.svdvals(jacobian_pos[0])
            cond = (svals.max() / svals.min()).item() if svals.min().item() > 1e-9 else float("inf")

            ik_controller.set_command(step_target_b, ee_pos=ee_pos_b, ee_quat=ee_quat_b)
            joint_pos_des = ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, current_joint_pos)

            # Check for joint-limit clamping in the DESIRED command (before settling).
            des = joint_pos_des[0]
            clip_flags = []
            for jj in range(num_arm_joints):
                lo, hi = joint_pos_limits[0, jj, 0].item(), joint_pos_limits[0, jj, 1].item()
                if des[jj].item() < lo or des[jj].item() > hi:
                    clip_flags.append(jj)

            delta_norm = torch.norm(joint_pos_des - current_joint_pos).item()

            for _ in range(POLISH_SETTLE_STEPS):
                action = torch.zeros(env.num_envs, num_arm_joints + 1, device=env.device)
                for j in range(num_arm_joints):
                    action[:, j] = joint_pos_des[:, j]
                action[:, num_arm_joints] = GRIPPER_OPEN
                env.step(action)

            achieved_q = robot.data.joint_pos[0, robot_entity_cfg.joint_ids]
            track_err = torch.norm(achieved_q - joint_pos_des[0]).item()

            ee_pose_w_now = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
            ee_pos_b_now, _ = subtract_frame_transforms(
                robot.data.root_pose_w[:, 0:3], robot.data.root_pose_w[:, 3:7], ee_pose_w_now[:, 0:3], ee_pose_w_now[:, 3:7]
            )
            new_dist = torch.norm(ee_pos_b_now[0] - target_pos_b[0]).item()

            print(
                f"[R{round_num:02d}] pre_dist={dist.item():.5f} step_max={POLISH_STEP_MAX:.3f} "
                f"cond#={cond:.1f} min_sval={svals.min().item():.5f} delta_q_norm={delta_norm:.4f} "
                f"limit_clip={clip_flags} track_err={track_err:.5f} POST_dist={new_dist:.5f}"
            )
            print(f"       q_achieved={['%.4f' % v for v in achieved_q.tolist()]}")

        env.reset()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
