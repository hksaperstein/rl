"""Diagnostic v6 (2026-07-22) - THE FIX: root-caused via test_operational_space.py's own
reference pattern - robot.root_physx_view.get_jacobians() returns the Jacobian in the
WORLD frame, but every AR4 classical demo script (grasp_demo.py, grasp_demo_v2.py,
oracle_rollout.py, interactive_joint_demo.py) feeds this raw world-frame Jacobian directly
into DifferentialIKController alongside ROOT-FRAME ee_pos_b/ee_quat_b (via
subtract_frame_transforms) - correct ONLY if the robot base has identity orientation in
world frame. AR4's base is rotated 180deg about Z (tasks/ar4/robot_cfg.py's
init_state rot=(0,0,0,1)), so this is a genuine, silent frame-mismatch bug: every previous
polish/DLS-solve round was applying a corrective delta computed against a MIRRORED
(X,Y negated) sensitivity direction relative to the true root-frame position error - this
fully explains this session's own finding that DLS polish monotonically DIVERGES (moves
away from target) round after round instead of converging, and explains the
joint-slamming-to-limits/oscillation signatures seen in every earlier diagnostic this
session.

Fix (matching Isaac Lab's own test_operational_space.py _update_states reference code):
rotate the raw world-frame Jacobian into the root frame via
jacobian_b[:, :3] = R_root_inv @ jacobian_w[:, :3] (position rows only, since
command_type="position") before feeding it to the DLS controller.

Starts from the same properly-verified 0.14086m seed as _diag_polish_from_fixed_seed.py,
so the ONLY difference from that run is this one fix - a clean A/B test.

.. code-block:: bash

    PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/_diag_polish_jacobian_frame_fix.py
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test the world-to-root Jacobian frame fix for AR4 DLS polish.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: E402
from isaaclab.envs import ManagerBasedEnv  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.utils.math import matrix_from_quat, quat_inv, subtract_frame_transforms  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from tasks.ar4.grasp_verify_env_cfg import Ar4GraspVerifyEnvCfg  # noqa: E402
from tasks.ar4.robot_cfg import ARM_JOINT_NAMES  # noqa: E402

GRIPPER_OPEN = 1.0
SEED_Q = [0.623345817811894, 1.5707963705062866, 0.7024949540694556, 0.0, 0.0, 0.0]
TARGET_POS_B = [-0.2, -0.28, 0.009]

POLISH_STEP_MAX = 0.03
POLISH_ROUNDS = 40
POLISH_SETTLE_STEPS = 30
LAMBDA_VAL = 0.02
CONVERGENCE_THRESHOLD = 0.003


def main() -> None:
    env_cfg = Ar4GraspVerifyEnvCfg()
    env_cfg.sim.device = args_cli.device
    env_cfg.scene.robot.actuators["arm"].stiffness = 4000.0
    env_cfg.scene.robot.actuators["arm"].damping = 200.0

    env = ManagerBasedEnv(cfg=env_cfg)
    robot = env.scene["robot"]
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES, body_names=["link_6"])
    robot_entity_cfg.resolve(env.scene)
    ik_jacobi_idx = robot_entity_cfg.body_ids[0] - 1 if robot.is_fixed_base else robot_entity_cfg.body_ids[0]
    num_arm_joints = len(ARM_JOINT_NAMES)

    joint_pos_limits = robot.data.joint_pos_limits[:, robot_entity_cfg.joint_ids]

    ik_cfg = DifferentialIKControllerCfg(
        command_type="position", use_relative_mode=False, ik_method="dls", ik_params={"lambda_val": LAMBDA_VAL}
    )
    ik_controller = DifferentialIKController(ik_cfg, num_envs=env.num_envs, device=env.device)

    with torch.inference_mode():
        env.reset()

        target_pos_b = torch.tensor([TARGET_POS_B], device=env.device)
        seed_q = torch.tensor([SEED_Q], device=env.device)
        robot.write_joint_position_to_sim(seed_q, joint_ids=robot_entity_cfg.joint_ids, env_ids=torch.tensor([0], device=env.device))
        zero_vel = torch.zeros((1, num_arm_joints), device=env.device)
        robot.write_joint_velocity_to_sim(zero_vel, joint_ids=robot_entity_cfg.joint_ids, env_ids=torch.tensor([0], device=env.device))
        for _ in range(60):
            action = torch.zeros(env.num_envs, num_arm_joints + 1, device=env.device)
            action[:, :num_arm_joints] = seed_q
            action[:, num_arm_joints] = GRIPPER_OPEN
            env.step(action)

        ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
        ee_pos_b0, _ = subtract_frame_transforms(
            robot.data.root_pose_w[:, 0:3], robot.data.root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        print(f"[SEED SETTLED] dist={torch.norm(ee_pos_b0[0] - target_pos_b[0]).item():.5f}m")

        best_dist = torch.norm(ee_pos_b0[0] - target_pos_b[0]).item()
        best_q = seed_q[0].tolist()

        for round_num in range(POLISH_ROUNDS):
            current_joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids].clone()
            ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                robot.data.root_pose_w[:, 0:3], robot.data.root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            direction = target_pos_b - ee_pos_b
            dist = torch.norm(direction, dim=-1, keepdim=True)
            step_target_b = ee_pos_b + direction / (dist + 1e-8) * torch.clamp(dist, max=POLISH_STEP_MAX)

            # --- THE FIX: rotate the world-frame Jacobian into the root frame ---
            jacobian_w = robot.root_physx_view.get_jacobians()[:, ik_jacobi_idx, :, robot_entity_cfg.joint_ids]
            root_rot_matrix = matrix_from_quat(quat_inv(robot.data.root_quat_w))
            jacobian_b = jacobian_w.clone()
            jacobian_b[:, 0:3, :] = torch.bmm(root_rot_matrix, jacobian_b[:, 0:3, :])
            # (position-only command_type="position", so rows 3:6 (angular) unused/not corrected)

            ik_controller.set_command(step_target_b, ee_pos=ee_pos_b, ee_quat=ee_quat_b)
            joint_pos_des = ik_controller.compute(ee_pos_b, ee_quat_b, jacobian_b, current_joint_pos)

            lo = joint_pos_limits[:, :, 0]
            hi = joint_pos_limits[:, :, 1]
            joint_pos_des = torch.clamp(joint_pos_des, min=lo, max=hi)

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
            new_dist = torch.norm(ee_pos_b_now[0] - target_pos_b[0]).item()
            achieved_q = robot.data.joint_pos[0, robot_entity_cfg.joint_ids].tolist()
            print(f"[R{round_num:02d}] dist={new_dist:.5f} q={['%.4f' % v for v in achieved_q]}")

            if new_dist < best_dist:
                best_dist = new_dist
                best_q = achieved_q
            if new_dist < CONVERGENCE_THRESHOLD:
                print(f"[CONVERGED] round {round_num}, dist={new_dist:.5f}")
                break

        print(f"[FINAL] best_dist={best_dist:.5f}m best_q={best_q}")
        env.reset()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
