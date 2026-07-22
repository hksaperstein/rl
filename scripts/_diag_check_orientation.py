"""Quick diagnostic (2026-07-22): what orientation does link_6 (and hence the
gripper) actually have at the verified-good grasp_q / pregrasp_q configs?
Position-only DLS gives the solver zero incentive to pick a sensible pinch
orientation - this checks whether the found configs happen to be reasonable
(jaws roughly horizontal/vertical, pointing at the cube) or arbitrary, to
decide whether orientation is the real remaining blocker for a full grasp.

.. code-block:: bash

    PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/_diag_check_orientation.py
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Check AR4 gripper orientation at verified grasp configs.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

from isaaclab.envs import ManagerBasedEnv  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from tasks.ar4.grasp_verify_env_cfg import Ar4GraspVerifyEnvCfg  # noqa: E402
from tasks.ar4.robot_cfg import ARM_JOINT_NAMES  # noqa: E402

GRASP_Q = [-0.01482450682669878, 1.2443578243255615, 0.3348628580570221, -0.08845815807580948, 1.2164382934570312, 0.008290650323033333]
PREGRASP_Q = [0.0001284122554352507, 0.9664260745048523, 0.9023779034614563, -0.0006572315469384193, -0.6306304931640625, 0.00851550791412592]
HOME_Q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def check(env, robot, robot_entity_cfg, num_arm_joints, q_list, label):
    q = torch.tensor([q_list], device=env.device)
    robot.write_joint_position_to_sim(q, joint_ids=robot_entity_cfg.joint_ids, env_ids=torch.tensor([0], device=env.device))
    zero_vel = torch.zeros((1, num_arm_joints), device=env.device)
    robot.write_joint_velocity_to_sim(zero_vel, joint_ids=robot_entity_cfg.joint_ids, env_ids=torch.tensor([0], device=env.device))
    for _ in range(30):
        action = torch.zeros(env.num_envs, num_arm_joints + 1, device=env.device)
        action[:, :num_arm_joints] = q
        action[:, num_arm_joints] = 1.0
        env.step(action)

    ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        robot.data.root_pose_w[:, 0:3], robot.data.root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )
    rot = matrix_from_quat(ee_quat_b)[0]  # 3x3, local->root
    local_x = rot[:, 0].tolist()
    local_y = rot[:, 1].tolist()
    local_z = rot[:, 2].tolist()
    print(f"[{label}] ee_pos_b={ee_pos_b[0].tolist()}")
    print(f"  local +X (jaw slide axis) in root frame: {['%.3f'%v for v in local_x]}")
    print(f"  local +Y                  in root frame: {['%.3f'%v for v in local_y]}")
    print(f"  local +Z (EE_OFFSET axis) in root frame: {['%.3f'%v for v in local_z]}")


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

    with torch.inference_mode():
        env.reset()
        check(env, robot, robot_entity_cfg, num_arm_joints, HOME_Q, "HOME (reference)")
        check(env, robot, robot_entity_cfg, num_arm_joints, GRASP_Q, "GRASP_Q")
        check(env, robot, robot_entity_cfg, num_arm_joints, PREGRASP_Q, "PREGRASP_Q")
        env.reset()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
