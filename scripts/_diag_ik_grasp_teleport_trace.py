"""Diagnostic v3 (2026-07-22): trace exactly what happens, step by step, when
the arm is teleported to the grid-search's own reported best GRASP-waypoint
config (dist=0.03337m at the moment the grid loop measured it) and then held
there. Two prior diagnostics this session found this "0.033m" config balloons
to ~0.42m of residual once genuinely settled from a clean teleport, with
joint_2/joint_3 slamming to their exact hard limits - the classic signature of
a violent PhysX depenetration reaction, not a controlled PD settle. This
script prints per-step EE and gripper-jaw-link world Z (vs GROUND_Z=0.0) and
joint velocity magnitude for the first 40 steps after teleport, to catch the
divergence in the act.

.. code-block:: bash

    PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/_diag_ik_grasp_teleport_trace.py
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Trace AR4 grasp-waypoint teleport-and-settle divergence.")
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

GRIPPER_OPEN = 1.0
SEED_Q = [0.623345817811894, 1.1868239243825276, -1.4508049661914508, 0.0, 0.0, 0.0]
TARGET_POS_B = [-0.2, -0.28, 0.009]


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

    # Resolve body indices for gripper jaw links and link_5 (wrist) too.
    all_body_names = robot.data.body_names
    print(f"[BODIES] {all_body_names}")

    with torch.inference_mode():
        env.reset()

        target_pos_b = torch.tensor([TARGET_POS_B], device=env.device)
        seed_q = torch.tensor([SEED_Q], device=env.device)
        robot.write_joint_position_to_sim(
            seed_q, joint_ids=robot_entity_cfg.joint_ids, env_ids=torch.tensor([0], device=env.device)
        )
        # Explicitly zero velocity too, in case stale velocity from a prior
        # sequence is what's causing the kick.
        zero_vel = torch.zeros((1, num_arm_joints), device=env.device)
        robot.write_joint_velocity_to_sim(
            zero_vel, joint_ids=robot_entity_cfg.joint_ids, env_ids=torch.tensor([0], device=env.device)
        )

        for step in range(60):
            action = torch.zeros(env.num_envs, num_arm_joints + 1, device=env.device)
            action[:, :num_arm_joints] = seed_q
            action[:, num_arm_joints] = GRIPPER_OPEN
            env.step(action)

            if step < 40 or step % 5 == 0:
                ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
                ee_pos_b, _ = subtract_frame_transforms(
                    robot.data.root_pose_w[:, 0:3], robot.data.root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
                )
                dist = torch.norm(ee_pos_b[0] - target_pos_b[0]).item()
                joint_vel = robot.data.joint_vel[0, robot_entity_cfg.joint_ids]
                vel_norm = torch.norm(joint_vel).item()
                joint_pos = robot.data.joint_pos[0, robot_entity_cfg.joint_ids].tolist()
                ee_z_w = ee_pose_w[0, 2].item()
                print(
                    f"[step {step:2d}] dist={dist:.5f} ee_z_w={ee_z_w:.5f} vel_norm={vel_norm:.4f} "
                    f"q={['%.4f' % v for v in joint_pos]}"
                )

        env.reset()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
