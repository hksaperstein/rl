# scripts/smoke_test_graspgoal_env.py
"""Headless smoke test for Ar4PickPlaceGraspGoalEnvCfg (Experiment 26):
builds the env, steps it a fixed number of times with all-zero actions
across a few envs, and prints observation shapes, reward values, and
grasp/lift latch state - real evidence the env cfg actually builds and
runs, not just that it imports. Same pattern as
scripts/smoke_test_touchgoal_env.py (Experiment 25).

.. code-block:: bash

    /home/saps/IsaacLab/isaaclab.sh -p scripts/smoke_test_graspgoal_env.py
"""

import sys

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser(description="Smoke test for the grasp-goal env cfg.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os  # noqa: E402

import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from tasks.ar4.pickplace_graspgoal_env_cfg import Ar4PickPlaceGraspGoalEnvCfg  # noqa: E402


def main() -> None:
    env_cfg = Ar4PickPlaceGraspGoalEnvCfg()
    env_cfg.scene.num_envs = 4
    env_cfg.sim.device = args_cli.device
    env = ManagerBasedRLEnv(cfg=env_cfg)

    with torch.inference_mode():
        obs, _ = env.reset()
        # Observation: joint_pos(8) + joint_vel(8) + cube_position(3) +
        # goal_position(3) + grasp_state(2) + actions(7) = 31. The
        # "actions" term mirrors action_manager.action, whose true
        # dimension is 7 (not 8) - see the action-space comment below.
        print(f"[SMOKE] observation shape: {obs['policy'].shape} (expect (4, 31))")
        # gripper_position is a MirroredGripperActionCfg, subclassing
        # isaaclab's BinaryJointAction, whose action_dim is hardcoded to
        # 1 regardless of how many joints its open/close command dicts
        # span (isaaclab/envs/mdp/actions/binary_joint_actions.py) - one
        # binary command is expanded in software to both gripper joints,
        # it is not a 2-dim per-joint command. So total action dim is
        # arm 6 + gripper 1 = 7, matching every other binary-gripper
        # experiment in this repo (21/22/jawmirror), not 8.
        print(f"[SMOKE] action space shape: {env.action_manager.action.shape} (expect (4, 7) - arm 6 + gripper 1)")

        for step in range(50):
            actions = torch.zeros(4, env.action_manager.total_action_dim, device=env.device)
            obs, rew, terminated, truncated, info = env.step(actions)
            if step % 10 == 0:
                print(
                    f"[SMOKE] step {step}: reward={rew.cpu().tolist()}, "
                    f"terminated={terminated.cpu().tolist()}, truncated={truncated.cpu().tolist()}"
                )

        cube = env.scene["cube"]
        ee_frame = env.scene["ee_frame"]
        print(f"[SMOKE] cube position (env 0): {cube.data.root_pos_w[0].cpu().tolist()}")
        print(f"[SMOKE] ee_frame target position (env 0): {ee_frame.data.target_pos_w[0, 0].cpu().tolist()}")
        print(f"[SMOKE] env._grasped: {env._grasped.cpu().tolist()}")
        print(f"[SMOKE] env._lifted: {env._lifted.cpu().tolist()}")
        print(f"[SMOKE] env._grasp_goal_milestone_max: {env._grasp_goal_milestone_max.cpu().tolist()}")

    env.close()
    print("[SMOKE] PASS: env built, stepped 50 times, no exceptions.")


if __name__ == "__main__":
    main()
    simulation_app.close()
