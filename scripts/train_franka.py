"""Train a PPO policy (rsl_rl) for the Franka Panda cube-lift task.

This script trains on the stock-recipe Franka Panda environment with relative-IK
action space.

.. code-block:: bash

    cd ~/projects/rl
    /home/saps/IsaacLab/isaaclab.sh -p scripts/train_franka.py --num_envs 4096
    # bounded probe (real GUI window expected per current instruction, not --headless):
    /home/saps/IsaacLab/isaaclab.sh -p scripts/train_franka.py --num_envs 64 --max_iterations 300
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train the Franka Panda cube-lift policy with PPO (rsl_rl).")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of parallel environments.")
parser.add_argument("--max_iterations", type=int, default=None, help="Override the agent config's max_iterations.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos periodically during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of each recorded video (steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Steps between recorded videos.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import sys
from datetime import datetime

import gymnasium as gym

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tasks.franka.agents.rsl_rl_ppo_cfg import FrankaLiftPPORunnerCfg  # noqa: E402
from tasks.franka.lift_env_cfg import FrankaLiftEnvCfg  # noqa: E402

LOG_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "train_franka")


def main() -> None:
    env_cfg = FrankaLiftEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    agent_cfg = FrankaLiftPPORunnerCfg()
    agent_cfg.device = args_cli.device
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations

    env_cfg.seed = agent_cfg.seed

    log_dir = os.path.join(LOG_ROOT, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)

    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "name_prefix": "franka_lift_train",
            "disable_logger": True,
        }
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()
    print(f"Training complete. Checkpoints and logs written to: {log_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
