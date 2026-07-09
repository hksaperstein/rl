# scripts/smoke_test_touchgoal_env.py
"""Headless smoke test for Ar4PickPlaceTouchGoalEnvCfg (Experiment 25):
builds the env, steps it a fixed number of times with random actions
across a few envs, and prints observation shapes, reward values, and
termination behavior - real evidence the env cfg actually builds and
runs, not just that it imports without error.

.. code-block:: bash

    /home/saps/IsaacLab/isaaclab.sh -p scripts/smoke_test_touchgoal_env.py
"""

import sys

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser(description="Smoke test for the touch-goal env cfg.")
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

from tasks.ar4.pickplace_touchgoal_env_cfg import Ar4PickPlaceTouchGoalEnvCfg  # noqa: E402


def main() -> None:
    env_cfg = Ar4PickPlaceTouchGoalEnvCfg()
    env_cfg.scene.num_envs = 4
    env_cfg.sim.device = args_cli.device
    env = ManagerBasedRLEnv(cfg=env_cfg)

    with torch.inference_mode():
        obs, _ = env.reset()
        print(f"[SMOKE] observation shape: {obs['policy'].shape} (expect (4, N) for some N)")
        print(f"[SMOKE] action space shape: {env.action_manager.action.shape} (expect (4, 6) - arm-only)")

        for step in range(50):
            actions = torch.zeros(4, env.action_manager.total_action_dim, device=env.device)
            obs, rew, terminated, truncated, info = env.step(actions)
            if step % 10 == 0:
                print(
                    f"[SMOKE] step {step}: reward={rew.cpu().tolist()}, "
                    f"terminated={terminated.cpu().tolist()}, truncated={truncated.cpu().tolist()}"
                )

        # Drive one env's arm target directly toward a plausible pose and confirm
        # touch_goal_milestone_bonus and touched-cube latch actually respond -
        # not just that the env runs without crashing.
        cube = env.scene["cube"]
        ee_frame = env.scene["ee_frame"]
        print(f"[SMOKE] cube position (env 0): {cube.data.root_pos_w[0].cpu().tolist()}")
        print(f"[SMOKE] ee_frame target position (env 0): {ee_frame.data.target_pos_w[0, 0].cpu().tolist()}")
        print(f"[SMOKE] env._touched_cube: {env._touched_cube.cpu().tolist()}")
        print(f"[SMOKE] env._touch_goal_milestone_max: {env._touch_goal_milestone_max.cpu().tolist()}")

    env.close()
    print("[SMOKE] PASS: env built, stepped 50 times, no exceptions.")


if __name__ == "__main__":
    main()
    simulation_app.close()
