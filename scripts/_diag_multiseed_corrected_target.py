"""Diagnostic v8 (2026-07-22): multi-seed search for the CORRECTED cube
target (0.0, 0.275, 0.009) - CANDIDATE_SEEDS in grasp_demo_v2.py was tuned
against the OLD, wrong CUBE_POS_W=(0.20,0.28,0.009); now that the real bug
(scene recenters the cube to the workspace midpoint) is fixed, the bearing
angle is very different (seed_j1 ~0 instead of ~0.62) and the old seeds
don't land in as good a basin (21mm/26mm instead of 15mm/0.2mm). This finds
a better seed set for the corrected target, for both GRASP and PREGRASP,
using the same fixed-Jacobian + EE-offset-aware polish as the real script.

.. code-block:: bash

    PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/_diag_multiseed_corrected_target.py
"""

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Multi-seed AR4 IK polish search for corrected cube target.")
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

CUBE_POS_W = (0.0, 0.275, 0.009)
GRASP_AT_HEIGHT = 0.009
PREGRASP_HOVER = 0.05
GRIPPER_OPEN = 1.0
CALIBRATION_C = -1.5677
_EE_OFFSET = (0.0, 0.0, 0.036)

POLISH_STEP_MAX = 0.03
POLISH_ROUNDS = 60
POLISH_SETTLE_STEPS = 30
LAMBDA_VAL = 0.02
CONVERGENCE_THRESHOLD = 0.002

# Wider spread of candidate (j2, j3, j5) seeds, sampling both "elbow up" and
# "elbow down" configurations across the joint ranges, now that the bearing
# is ~straight ahead (seed_j1 ~0) instead of the old ~0.62 offset bearing.
CANDIDATE_SEEDS = [
    (1.0, -0.5, -0.8),
    (1.2, 0.0, -1.2),
    (0.6, 0.3, 0.5),
    (1.3, -0.8, 0.9),
    (0.9, -0.9, -1.5),
    (1.4, 0.4, -0.3),
    (0.8, -0.85, -1.55),
    (1.3, 0.6, -1.7),
    (1.1, 0.9, -1.3),
    (0.5, -0.3, 0.8),
    (1.5, 0.2, 0.2),
    (0.7, 0.7, -0.9),
]


def _ee_point_pos_and_jacobian(ee_pos_b, ee_quat_b, jacobian_b):
    offset_local = torch.tensor([_EE_OFFSET], device=ee_pos_b.device).expand(ee_pos_b.shape[0], 3)
    rot = matrix_from_quat(ee_quat_b)
    world_offset = torch.bmm(rot, offset_local.unsqueeze(-1)).squeeze(-1)
    point_pos_b = ee_pos_b + world_offset
    skew = torch.zeros(ee_pos_b.shape[0], 3, 3, device=ee_pos_b.device)
    skew[:, 0, 1], skew[:, 0, 2] = -world_offset[:, 2], world_offset[:, 1]
    skew[:, 1, 0], skew[:, 1, 2] = world_offset[:, 2], -world_offset[:, 0]
    skew[:, 2, 0], skew[:, 2, 1] = -world_offset[:, 1], world_offset[:, 0]
    jac_ang = jacobian_b[:, 3:6, :]
    point_jac_pos = jacobian_b[:, 0:3, :] - torch.bmm(skew, jac_ang)
    return point_pos_b, point_jac_pos


def _measure_dist(robot, robot_entity_cfg, target_pos_b) -> float:
    ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        robot.data.root_pose_w[:, 0:3], robot.data.root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )
    offset_local = torch.tensor([_EE_OFFSET], device=ee_pos_b.device).expand(ee_pos_b.shape[0], 3)
    rot = matrix_from_quat(ee_quat_b)
    world_offset = torch.bmm(rot, offset_local.unsqueeze(-1)).squeeze(-1)
    point_pos_b = ee_pos_b + world_offset
    return torch.norm(point_pos_b[0] - target_pos_b[0]).item()


def polish(env, robot, ik_controller, robot_entity_cfg, ik_jacobi_idx, joint_pos_limits, num_arm_joints, target_pos_b, seed_q_list, label):
    seed_q = torch.tensor([seed_q_list], device=env.device)
    robot.write_joint_position_to_sim(seed_q, joint_ids=robot_entity_cfg.joint_ids, env_ids=torch.tensor([0], device=env.device))
    zero_vel = torch.zeros((1, num_arm_joints), device=env.device)
    robot.write_joint_velocity_to_sim(zero_vel, joint_ids=robot_entity_cfg.joint_ids, env_ids=torch.tensor([0], device=env.device))
    for _ in range(60):
        action = torch.zeros(env.num_envs, num_arm_joints + 1, device=env.device)
        action[:, :num_arm_joints] = seed_q
        action[:, num_arm_joints] = GRIPPER_OPEN
        env.step(action)

    best_dist = _measure_dist(robot, robot_entity_cfg, target_pos_b)
    best_q = seed_q[0].tolist()

    for round_num in range(POLISH_ROUNDS):
        current_joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids].clone()
        ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            robot.data.root_pose_w[:, 0:3], robot.data.root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )

        jacobian_w = robot.root_physx_view.get_jacobians()[:, ik_jacobi_idx, :, robot_entity_cfg.joint_ids]
        root_rot_matrix = matrix_from_quat(quat_inv(robot.data.root_quat_w))
        jacobian_b = jacobian_w.clone()
        jacobian_b[:, 0:3, :] = torch.bmm(root_rot_matrix, jacobian_b[:, 0:3, :])
        jacobian_b[:, 3:6, :] = torch.bmm(root_rot_matrix, jacobian_b[:, 3:6, :])

        point_pos_b, point_jac_pos = _ee_point_pos_and_jacobian(ee_pos_b, ee_quat_b, jacobian_b)

        direction = target_pos_b - point_pos_b
        dist = torch.norm(direction, dim=-1, keepdim=True)
        step_target_b = point_pos_b + direction / (dist + 1e-8) * torch.clamp(dist, max=POLISH_STEP_MAX)

        ik_controller.set_command(step_target_b, ee_pos=point_pos_b, ee_quat=ee_quat_b)
        joint_pos_des = ik_controller.compute(point_pos_b, ee_quat_b, point_jac_pos, current_joint_pos)

        lo = joint_pos_limits[:, :, 0]
        hi = joint_pos_limits[:, :, 1]
        joint_pos_des = torch.clamp(joint_pos_des, min=lo, max=hi)

        for _ in range(POLISH_SETTLE_STEPS):
            action = torch.zeros(env.num_envs, num_arm_joints + 1, device=env.device)
            for j in range(num_arm_joints):
                action[:, j] = joint_pos_des[:, j]
            action[:, num_arm_joints] = GRIPPER_OPEN
            env.step(action)

        new_dist = _measure_dist(robot, robot_entity_cfg, target_pos_b)
        achieved_q = robot.data.joint_pos[0, robot_entity_cfg.joint_ids].tolist()
        if new_dist < best_dist:
            best_dist = new_dist
            best_q = achieved_q
        if new_dist < CONVERGENCE_THRESHOLD:
            break

    print(f"  [{label}] best_dist={best_dist:.5f}m best_q={['%.4f' % v for v in best_q]}")
    return best_dist, best_q


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

        root_pos_w = robot.data.root_pos_w.clone()
        root_quat_w = robot.data.root_quat_w.clone()
        cube_pos_w = torch.tensor([CUBE_POS_W], device=env.device)
        cube_pos_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, cube_pos_w)
        target_bearing = math.atan2(cube_pos_b[0, 1].item(), cube_pos_b[0, 0].item())
        seed_j1 = -(target_bearing - CALIBRATION_C)

        grasp_pos_b = cube_pos_b.clone()
        grasp_pos_b[:, 2] = GRASP_AT_HEIGHT
        pregrasp_pos_b = cube_pos_b.clone()
        pregrasp_pos_b[:, 2] = GRASP_AT_HEIGHT + PREGRASP_HOVER

        print(f"[INFO] seed_j1={seed_j1:.4f} grasp_pos_b={grasp_pos_b[0].tolist()} pregrasp_pos_b={pregrasp_pos_b[0].tolist()}")

        results_grasp = []
        results_pregrasp = []
        for idx, (j2, j3, j5) in enumerate(CANDIDATE_SEEDS):
            seed_q_list = [seed_j1, j2, j3, 0.0, j5, 0.0]
            print(f"[SEED {idx}] q0={seed_q_list}")
            d, q = polish(env, robot, ik_controller, robot_entity_cfg, ik_jacobi_idx, joint_pos_limits, num_arm_joints, grasp_pos_b, seed_q_list, f"GRASP seed{idx}")
            results_grasp.append((d, q, (j2, j3, j5)))
            d, q = polish(env, robot, ik_controller, robot_entity_cfg, ik_jacobi_idx, joint_pos_limits, num_arm_joints, pregrasp_pos_b, seed_q_list, f"PREGRASP seed{idx}")
            results_pregrasp.append((d, q, (j2, j3, j5)))

        best_grasp = min(results_grasp, key=lambda x: x[0])
        best_pregrasp = min(results_pregrasp, key=lambda x: x[0])
        print(f"[BEST GRASP] dist={best_grasp[0]:.5f}m q={best_grasp[1]} seed={best_grasp[2]}")
        print(f"[BEST PREGRASP] dist={best_pregrasp[0]:.5f}m q={best_pregrasp[1]} seed={best_pregrasp[2]}")

        env.reset()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
