# rl/scripts/eval_loop.py
"""Run a trained AR4 pick-and-place PPO policy for a fixed number of episodes,
recording each one as an mp4 to rl/logs/videos/.

.. code-block:: bash

    ./isaaclab.sh -p rl/scripts/eval_loop.py --checkpoint rl/logs/train/<run>/model_1500.pt --episodes 10
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run a trained AR4 pick-and-place policy and record video.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the rsl_rl checkpoint (.pt) to load.")
parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # required for video recording

if not os.path.isfile(args_cli.checkpoint):
    sys.exit(f"Checkpoint not found: {args_cli.checkpoint}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tasks.ar4.agents.rsl_rl_ppo_cfg import Ar4PickPlacePPORunnerCfg  # noqa: E402
from tasks.ar4.pickplace_env_cfg import Ar4PickPlaceEnvCfg  # noqa: E402

VIDEO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "videos")


def main() -> None:
    env_cfg = Ar4PickPlaceEnvCfg()
    env_cfg.scene.num_envs = 1

    agent_cfg = Ar4PickPlacePPORunnerCfg()

    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=VIDEO_DIR,
        step_trigger=lambda step: step % 250 == 0,
        video_length=250,
        name_prefix="ar4_pickplace",
        disable_logger=True,
    )
    env = RslRlVecEnvWrapper(env, clip_actions=None)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs = env.get_observations()
    completed_episodes = 0
    with torch.inference_mode():
        while completed_episodes < args_cli.episodes and simulation_app.is_running():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if bool(dones[0]):
                completed_episodes += 1
                print(f"[INFO] Completed episode {completed_episodes}/{args_cli.episodes}")

    env.close()
    print(f"Videos written to: {VIDEO_DIR}")


if __name__ == "__main__":
    main()
    simulation_app.close()
