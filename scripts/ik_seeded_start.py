"""Test whether seeding the proven bounded-step DLS solve from a
direction-aware starting configuration (rather than always starting cold
from HOME_Q) actually converges - testing the local-minimum-trap
hypothesis directly.

scripts/measure_reach_envelope.py already proved (via pure forward
kinematics, no IK solver) that the cube target (0.20, 0.28, 0.009) is
comfortably within the arm's reach envelope (0.538m max reach vs. 0.344m
needed) and found a joint_2/joint_3 combination that achieves that reach.
This script seeds joint_1 to point directly at the cube's bearing
(atan2, not the arbitrary yaw the reach sweep happened to use) combined
with that same joint_2/joint_3 combination, then runs grasp_demo.py's
exact proven bounded-step solve_ik_to_target logic from there.

.. code-block:: bash

    PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/ik_seeded_start.py --headless
"""

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Seeded-start IK convergence test.")
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
MAX_IK_ROUNDS = 40
IK_SETTLE_STEPS = 20

# From measure_reach_envelope.py's widest-reach sweep result: joint_2/joint_3
# combo that achieved 0.538m reach (joint_1 there was an arbitrary sweep
# value, not aimed at the cube - replaced below with the correct bearing).
SEED_J2 = 0.7853981852531433
SEED_J3 = -1.5533430576324463


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
        print(f"[IK Round {round_num + 1}/{max_rounds}] Residual error: {residual_error:.5f}m")

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
        grasp_pos_b = cube_pos_b.clone()
        grasp_pos_b[:, 2] = GRASP_AT_HEIGHT

        # Empirically calibrated: joint_1's effect on the EE's azimuthal angle
        # is ee_angle = -joint_1 + C (C ~ -1.567 rad), NOT a direct atan2 match -
        # measured from two data points (measure_reach_envelope.py's sweep vs.
        # this script's first run, which used a naive un-calibrated atan2 seed
        # and landed nowhere near the target as a result). Solving for the
        # joint_1 that points this reach-configuration at the cube's actual
        # bearing:
        target_bearing = math.atan2(cube_pos_b[0, 1].item(), cube_pos_b[0, 0].item())
        CALIBRATION_C = -1.5677  # average of two measured data points, see conversation record
        seed_j1 = -(target_bearing - CALIBRATION_C)
        print(f"[INFO] Cube target (robot frame): {grasp_pos_b[0].tolist()}")
        print(f"[INFO] Target bearing (raw atan2): {target_bearing:.4f} rad")
        print(f"[INFO] Seed joint_1 (calibrated for this arm's actual joint_1->azimuth convention): {seed_j1:.4f} rad")
        print(f"[INFO] Seed joint_2/joint_3 (from measure_reach_envelope.py's widest-reach sweep): {SEED_J2:.4f}, {SEED_J3:.4f}")

        seed_q = torch.tensor([[seed_j1, SEED_J2, SEED_J3, 0.0, 0.0, 0.0]], device=env.device)
        robot.write_joint_position_to_sim(seed_q, joint_ids=robot_entity_cfg.joint_ids, env_ids=torch.tensor([0], device=env.device))
        # Let the seeded pose settle physically before starting the IK solve.
        for _ in range(30):
            action = torch.zeros(env.num_envs, len(ARM_JOINT_NAMES) + 1, device=env.device)
            action[:, :len(ARM_JOINT_NAMES)] = seed_q
            action[:, len(ARM_JOINT_NAMES)] = GRIPPER_OPEN
            env.step(action)

        seeded_ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
        seeded_ee_pos_b, _ = subtract_frame_transforms(
            robot.data.root_pose_w[:, 0:3], robot.data.root_pose_w[:, 3:7], seeded_ee_pose_w[:, 0:3], seeded_ee_pose_w[:, 3:7]
        )
        seed_residual = torch.norm(seeded_ee_pos_b[0] - grasp_pos_b[0]).item()
        print(f"[INFO] EE position after seeding+settling: {seeded_ee_pos_b[0].tolist()}, residual to target: {seed_residual:.4f}m")

        print("\n[INFO] Running bounded-step DLS solve from the SEEDED start (not HOME_Q)...")
        final_q, final_residual = solve_ik_to_target(
            env, ik_controller, robot_entity_cfg, ik_jacobi_idx, grasp_pos_b, MAX_IK_ROUNDS, IK_SETTLE_STEPS, IK_CONVERGENCE_THRESHOLD
        )

        print("\n" + "=" * 80)
        print("SEEDED-START RESULT")
        print("=" * 80)
        print(f"Seed residual (before any IK solve rounds): {seed_residual:.5f}m")
        print(f"Final residual after solve: {final_residual:.5f}m")
        print(f"Final joint config: {final_q[0].tolist()}")
        print(f"Comparison: HOME_Q-started 6-DOF baseline plateaued at ~0.33m; 3-DOF-only baseline plateaued at ~0.32m.")
        if final_residual < IK_CONVERGENCE_THRESHOLD:
            print("*** CONVERGED - local-minimum-trap hypothesis CONFIRMED: a better starting point solves it. ***")
        elif final_residual < seed_residual * 0.5:
            print("*** MEANINGFUL IMPROVEMENT over the seed itself, but did not fully converge. ***")
        else:
            print("*** NO MEANINGFUL IMPROVEMENT even from a geometrically-aimed seed - points away from a simple local-minimum-trap explanation. ***")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
