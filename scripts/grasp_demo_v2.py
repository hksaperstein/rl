"""Classical (non-RL) pick-and-place demo, v2: uses the solving method this
session's investigation actually validated - a coarse forward-kinematics
grid search (no iteration, can't get stuck in a local minimum) followed by
a bounded-step DLS polish - instead of grasp_demo.py's original
from-HOME_Q live DLS solve, which reliably plateaued ~0.33m short of the
target across every variant tried this session.

See the conversation record for the full investigation: measure_reach_envelope.py
proved the target is within the arm's reach envelope via pure forward
kinematics; ik_seeded_start.py showed DLS barely improves even from a
directionally-correct seed; ik_grid_search.py + ik_polish_from_grid.py
found a real solution within ~3.6cm using grid-search-then-polish. This
script applies that same method to both the pregrasp and grasp waypoints
and runs the actual phased pick/lift/hold/release sequence to test
whether that precision is enough for a genuine grasp.

.. code-block:: bash

    PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/grasp_demo_v2.py --headless
"""

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Classical pick-and-place demo v2 (grid-search + DLS polish).")
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
VIDEO_PATH = os.path.join(LOG_DIR, "videos", "ar4_grasp_demo_v2.mp4")

CUBE_POS_W = (0.20, 0.28, 0.009)
PREGRASP_HOVER = 0.05
GRASP_AT_HEIGHT = 0.009
GRIPPER_OPEN = 1.0
GRIPPER_CLOSE = -1.0
HOME_Q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

CALIBRATION_C = -1.5677  # empirically measured joint_1 -> EE-azimuth offset
GRID_N = 25
GRID_SETTLE_STEPS = 15
POLISH_STEP_MAX = 0.05
POLISH_ROUNDS = 15
POLISH_SETTLE_STEPS = 30


def grid_search_then_polish(env, ik_controller, robot_entity_cfg, ik_jacobi_idx, target_pos_b, j2_min, j2_max, j3_min, j3_max, seed_j1):
    robot = env.scene["robot"]
    num_arm_joints = len(ARM_JOINT_NAMES)

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

    print(f"  [GRID] best coarse dist: {best_dist:.5f}m, config: {best_q}")

    # Seed the sim at the best grid config before polishing.
    seed_q = torch.tensor([best_q], device=env.device)
    robot.write_joint_position_to_sim(seed_q, joint_ids=robot_entity_cfg.joint_ids, env_ids=torch.tensor([0], device=env.device))
    for _ in range(POLISH_SETTLE_STEPS * 2):
        action = torch.zeros(env.num_envs, num_arm_joints + 1, device=env.device)
        action[:, :num_arm_joints] = seed_q
        action[:, num_arm_joints] = GRIPPER_OPEN
        env.step(action)

    # DLS polish.
    final_residual = best_dist
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
        ik_controller.set_command(step_target_b, ee_pos=ee_pos_b, ee_quat=ee_quat_b)
        joint_pos_des = ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, current_joint_pos)

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
        final_residual = torch.norm(ee_pos_b_now - target_pos_b[0], dim=-1).item()
        if final_residual < 0.01:
            break

    print(f"  [POLISH] final residual: {final_residual:.5f}m")
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
    camera = env.scene["perception_camera"]

    with torch.inference_mode():
        env.reset()

        root_pos_w = robot.data.root_pos_w.clone()
        root_quat_w = robot.data.root_quat_w.clone()
        cube_pos_w = torch.tensor([CUBE_POS_W], device=env.device)
        cube_pos_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, cube_pos_w)
        target_bearing = math.atan2(cube_pos_b[0, 1].item(), cube_pos_b[0, 0].item())
        seed_j1 = -(target_bearing - CALIBRATION_C)
        print(f"[INFO] Cube (robot frame): {cube_pos_b[0].tolist()}, calibrated joint_1: {seed_j1:.4f}")

        grasp_pos_b = cube_pos_b.clone()
        grasp_pos_b[:, 2] = GRASP_AT_HEIGHT
        pregrasp_pos_b = cube_pos_b.clone()
        pregrasp_pos_b[:, 2] = GRASP_AT_HEIGHT + PREGRASP_HOVER

        print("\n[INFO] Solving GRASP waypoint (grid search + DLS polish)...")
        grasp_q, grasp_residual = grid_search_then_polish(
            env, ik_controller, robot_entity_cfg, ik_jacobi_idx, grasp_pos_b, j2_min, j2_max, j3_min, j3_max, seed_j1
        )

        print("\n[INFO] Solving PREGRASP waypoint (grid search + DLS polish)...")
        pregrasp_q, pregrasp_residual = grid_search_then_polish(
            env, ik_controller, robot_entity_cfg, ik_jacobi_idx, pregrasp_pos_b, j2_min, j2_max, j3_min, j3_max, seed_j1
        )

        print(f"\n[SUMMARY] grasp_residual={grasp_residual:.5f}m pregrasp_residual={pregrasp_residual:.5f}m")
        print(f"[SUMMARY] grasp_q={grasp_q}")
        print(f"[SUMMARY] pregrasp_q={pregrasp_q}")

        PHASES = [
            (60, HOME_Q, GRIPPER_OPEN),
            (150, pregrasp_q, GRIPPER_OPEN),
            (90, grasp_q, GRIPPER_OPEN),
            (60, grasp_q, GRIPPER_CLOSE),
            (90, pregrasp_q, GRIPPER_CLOSE),
            (120, pregrasp_q, GRIPPER_CLOSE),
            (60, pregrasp_q, GRIPPER_OPEN),
            (150, HOME_Q, GRIPPER_OPEN),
        ]

        print("\n[INFO] Starting phased execution...\n")
        for phase_idx, (duration, target_q, gripper_cmd) in enumerate(PHASES):
            # Command the phase's target directly (not a ramped interpolation)
            # and let the PD controller converge over the phase's duration.
            # Tried ramped interpolation from the actual current position first
            # (the "correct-looking" fix for the stale-prev_q bug) - it was
            # much WORSE (final errors grew to 2.6+ rad) than the original
            # buggy version, which accidentally commanded the fixed target
            # directly from step 1 of each phase (since it always ramped
            # FROM the wrong, stale prev_q TO the same target, producing a
            # near-constant commanded value rather than a real ramp). This
            # arm's modest actuator stiffness appears to track a fixed target
            # better than a continuously-moving one.
            start_q = robot.data.joint_pos[0, robot_entity_cfg.joint_ids].tolist()
            print(f"[PHASE {phase_idx}] duration={duration} gripper={'OPEN' if gripper_cmd > 0 else 'CLOSE'} start_q={['%.4f' % x for x in start_q]}")
            for i in range(duration):
                action = torch.zeros(env.num_envs, num_arm_joints + 1, device=env.device)
                for j in range(num_arm_joints):
                    action[:, j] = target_q[j]
                action[:, num_arm_joints] = gripper_cmd
                env.step(action)

                rgb = camera.data.output["rgb"][0].cpu().numpy()
                video_writer.append_data(rgb[:, :, :3].astype("uint8"))

                if phase_idx in (3, 4, 5) and i % 20 == 0:
                    cube_z = env.scene["cube"].data.root_pos_w[0, 2].item()
                    cube_xy = env.scene["cube"].data.root_pos_w[0, :2].tolist()
                    print(f"  [PHASE {phase_idx} step {i:3d}] cube z={cube_z:.4f}m xy={['%.4f' % x for x in cube_xy]}")

            achieved_q = robot.data.joint_pos[0, robot_entity_cfg.joint_ids].tolist()
            max_err = max(abs(a - t) for a, t in zip(achieved_q, target_q))
            print(f"  [PHASE {phase_idx} END] max joint error: {max_err:.5f} rad")

        env.reset()

    video_writer.close()
    env.close()
    print(f"\nVideo recorded to: {VIDEO_PATH}")


if __name__ == "__main__":
    main()
    simulation_app.close()
