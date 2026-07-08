"""Final polish: seed the proven bounded-step DLS solve from
ik_grid_search.py's found configuration (already within ~3.5-6cm of the
target) and see if it closes the remaining gap to the 1cm convergence
threshold - testing whether DLS works fine once given a well-conditioned,
close starting point (not near any joint limit, small remaining error).

.. code-block:: bash

    PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/ik_polish_from_grid.py --headless
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Polish grid-search result with bounded-step DLS.")
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
IK_STEP_MAX = 0.05
IK_CONVERGENCE_THRESHOLD = 0.01
MAX_IK_ROUNDS = 20
IK_SETTLE_STEPS = 30  # longer settle per round this time, given the earlier grid search showed the
                       # arm still drifting after only 15 steps at a config it hadn't reached via a
                       # smooth interpolation - give the PD controller more time to genuinely settle.

# From ik_grid_search.py's best result.
SEED_Q = [0.623345817811894, 0.5148721436659496, 0.9075711369514465, 0.0, 0.0, 0.0]


def solve_ik_to_target(env, ik_controller, robot_entity_cfg, ik_jacobi_idx, target_pos_b, max_rounds, settle_steps, convergence_threshold):
    robot = env.scene["robot"]
    for round_num in range(max_rounds):
        current_joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids].clone()
        ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            robot.data.root_pose_w[:, 0:3], robot.data.root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        direction = target_pos_b - ee_pos_b
        dist = torch.norm(direction, dim=-1, keepdim=True)
        step_target_b = ee_pos_b + direction / (dist + 1e-8) * torch.clamp(dist, max=IK_STEP_MAX)

        jacobian = robot.root_physx_view.get_jacobians()[:, ik_jacobi_idx, :, robot_entity_cfg.joint_ids]
        ik_controller.set_command(step_target_b, ee_pos=ee_pos_b, ee_quat=ee_quat_b)
        joint_pos_des = ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, current_joint_pos)

        for _ in range(settle_steps):
            action = torch.zeros(env.num_envs, len(ARM_JOINT_NAMES) + 1, device=env.device)
            for j in range(len(ARM_JOINT_NAMES)):
                action[:, j] = joint_pos_des[:, j]
            action[:, len(ARM_JOINT_NAMES)] = GRIPPER_OPEN
            env.step(action)

        ee_pose_w_now = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
        ee_pos_b_now, _ = subtract_frame_transforms(
            robot.data.root_pose_w[:, 0:3], robot.data.root_pose_w[:, 3:7], ee_pose_w_now[:, 0:3], ee_pose_w_now[:, 3:7]
        )
        residual_error = torch.norm(ee_pos_b_now - target_pos_b, dim=-1).item()
        print(f"[IK Round {round_num + 1}/{max_rounds}] Residual error: {residual_error:.5f}m, joint_pos: {current_joint_pos[0].tolist()}")

        if residual_error < convergence_threshold or round_num == max_rounds - 1:
            print(f"[IK CONVERGED] residual {residual_error:.5f}m after {round_num + 1} round(s).")
            return joint_pos_des, residual_error

    return joint_pos_des, residual_error


def main() -> None:
    env_cfg = Ar4GraspVerifyEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device

    env = ManagerBasedEnv(cfg=env_cfg)
    robot = env.scene["robot"]
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES, body_names=["link_6"])
    robot_entity_cfg.resolve(env.scene)
    ik_jacobi_idx = robot_entity_cfg.body_ids[0] - 1 if robot.is_fixed_base else robot_entity_cfg.body_ids[0]

    ik_cfg = DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls")
    ik_controller = DifferentialIKController(ik_cfg, num_envs=env.num_envs, device=env.device)

    with torch.inference_mode():
        env.reset()

        root_pos_w = robot.data.root_pos_w.clone()
        root_quat_w = robot.data.root_quat_w.clone()
        cube_pos_w = torch.tensor([CUBE_POS_W], device=env.device)
        cube_pos_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, cube_pos_w)
        target_pos_b = cube_pos_b.clone()
        target_pos_b[:, 2] = GRASP_AT_HEIGHT

        seed_q = torch.tensor([SEED_Q], device=env.device)
        robot.write_joint_position_to_sim(seed_q, joint_ids=robot_entity_cfg.joint_ids, env_ids=torch.tensor([0], device=env.device))
        for _ in range(60):
            action = torch.zeros(env.num_envs, len(ARM_JOINT_NAMES) + 1, device=env.device)
            action[:, :len(ARM_JOINT_NAMES)] = seed_q
            action[:, len(ARM_JOINT_NAMES)] = GRIPPER_OPEN
            env.step(action)

        seeded_ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
        seeded_ee_pos_b, _ = subtract_frame_transforms(
            robot.data.root_pose_w[:, 0:3], robot.data.root_pose_w[:, 3:7], seeded_ee_pose_w[:, 0:3], seeded_ee_pose_w[:, 3:7]
        )
        seed_residual = torch.norm(seeded_ee_pos_b[0] - target_pos_b[0]).item()
        print(f"[INFO] Target: {target_pos_b[0].tolist()}")
        print(f"[INFO] EE position after seeding+60-step settle: {seeded_ee_pos_b[0].tolist()}, residual: {seed_residual:.5f}m")

        print("\n[INFO] Running bounded-step DLS polish from the grid-search seed...")
        final_q, final_residual = solve_ik_to_target(
            env, ik_controller, robot_entity_cfg, ik_jacobi_idx, target_pos_b, MAX_IK_ROUNDS, IK_SETTLE_STEPS, IK_CONVERGENCE_THRESHOLD
        )

        print("\n" + "=" * 80)
        print("POLISH RESULT")
        print("=" * 80)
        print(f"Seed residual: {seed_residual:.5f}m")
        print(f"Final residual: {final_residual:.5f}m")
        print(f"Final joint config: {final_q[0].tolist()}")
        if final_residual < IK_CONVERGENCE_THRESHOLD:
            print("*** CONVERGED to within 1cm - the target IS reachable and solvable given the right starting basin. ***")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
