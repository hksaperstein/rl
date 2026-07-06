"""Continue AR4 sphere pick-and-place PPO training from an existing
checkpoint with a bumped, fixed learning rate - an SA-PPO-style
intervention (Li et al., Sensors 2025, DOI 10.3390/s25175253) to escape
the "reach, grip, freeze" local optimum every prior experiment this
session has hit. See
docs/superpowers/specs/2026-07-06-ar4-sphere-lift-lr-bump-design.md.

Does NOT change the reward function - reuses whatever checkpoint is
passed in as-is, testing the learning-rate bump in isolation.

.. code-block:: bash

    /home/saps/IsaacLab/isaaclab.sh -p scripts/train_lr_bump.py \\
        --checkpoint logs/train/2026-07-06_12-24-08/model_700.pt \\
        --num_envs 4096 --max_iterations 1500 --headless

    # smoke test:
    /home/saps/IsaacLab/isaaclab.sh -p scripts/train_lr_bump.py \\
        --checkpoint logs/train/2026-07-06_12-24-08/model_700.pt \\
        --num_envs 16 --max_iterations 2 --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Continue AR4 pick-and-place PPO training with a bumped learning rate.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the rsl_rl checkpoint (.pt) to resume from.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of parallel environments.")
parser.add_argument(
    "--max_iterations", type=int, default=1500, help="Additional learning iterations beyond the checkpoint's own iteration count."
)
parser.add_argument("--learning_rate", type=float, default=1.0e-3, help="Bumped, fixed learning rate for this phase.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import sys
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tasks.ar4.agents.rsl_rl_ppo_cfg import Ar4PickPlacePPORunnerCfg  # noqa: E402
from tasks.ar4.pickplace_env_cfg import Ar4PickPlaceEnvCfg  # noqa: E402

LOG_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "train")


def main() -> None:
    env_cfg = Ar4PickPlaceEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    agent_cfg = Ar4PickPlacePPORunnerCfg()
    agent_cfg.device = args_cli.device
    # The bump: fixed (not adaptive) schedule so it isn't corrected back
    # down by KL-divergence feedback from the already-converged policy.
    agent_cfg.algorithm.learning_rate = args_cli.learning_rate
    agent_cfg.algorithm.schedule = "fixed"
    env_cfg.seed = agent_cfg.seed

    log_dir = os.path.join(LOG_ROOT, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)

    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    print(f"Resumed from {args_cli.checkpoint} at iteration {runner.current_learning_iteration}")

    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    runner.learn(num_learning_iterations=args_cli.max_iterations, init_at_random_ep_len=True)

    env.close()
    print(f"Training complete. Checkpoints and logs written to: {log_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
