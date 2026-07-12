"""Train a PPO policy (rsl_rl) for the Franka Panda cube-lift task.

This script trains on the stock-recipe Franka Panda environment with relative-IK
action space.

.. code-block:: bash

    cd ~/projects/rl
    /home/saps/IsaacLab/isaaclab.sh -p scripts/train_franka.py --num_envs 4096
    # bounded probe (real GUI window expected per current instruction, not --headless):
    /home/saps/IsaacLab/isaaclab.sh -p scripts/train_franka.py --num_envs 64 --max_iterations 300
    # resume a previously-interrupted run from its last checkpoint, continuing on to the SAME
    # absolute --max_iterations target (not +max_iterations more iterations on top of it):
    /home/saps/IsaacLab/isaaclab.sh -p scripts/train_franka.py --num_envs 4096 --max_iterations 5000 \
        --checkpoint logs/train_franka/2026-07-09_22-05-51/model_800.pt
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train the Franka Panda cube-lift policy with PPO (rsl_rl).")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of parallel environments.")
parser.add_argument("--max_iterations", type=int, default=None, help="Override the agent config's max_iterations.")
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help=(
        "Path to an rsl_rl checkpoint (.pt) to resume from - restores model + optimizer state and the "
        "checkpoint's own recorded iteration count via rsl_rl.OnPolicyRunner.load(), so training continues "
        "instead of restarting from scratch. --max_iterations is the ABSOLUTE target iteration count when "
        "resuming (e.g. resuming a checkpoint saved at iteration 800 with --max_iterations 5000 runs "
        "iterations 800->5000, not 800->5800 - OnPolicyRunner.learn()'s own num_learning_iterations argument "
        "is iterations-from-here, not an absolute target, so this script converts between the two). Writes "
        "to a NEW timestamped log_dir regardless (the checkpoint's weights/optimizer/iteration are what "
        "carry over - its previous TensorBoard event file is not appended to)."
    ),
)
parser.add_argument("--video", action="store_true", default=False, help="Record videos periodically during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of each recorded video (steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Steps between recorded videos.")
parser.add_argument(
    "--variant",
    choices=["ik-cube", "joint-die", "joint-cube", "joint-die-heavy", "joint-die-big"],
    default="ik-cube",
    help=(
        "ik-cube: the existing stock-recipe cube-lift with relative-IK actions (default, unchanged). "
        "joint-die: d20-die lift with direct joint-position actions (no IK) - see "
        "docs/superpowers/specs/2026-07-11-joint-space-die-lift-design.md. "
        "joint-cube: the spec's fallback rung - joint-position actions with the recipe's own DexCube "
        "(asset-vs-recipe isolation). "
        "joint-die-heavy: asset-bisect rung 1 - the d20 at DexCube's measured 0.216kg mass "
        "(docs/superpowers/specs/2026-07-12-asset-bisect-design.md). "
        "joint-die-big: asset-bisect rung 2 - the d20 scaled to DexCube's measured 48.0mm size, "
        "mass pinned at 0.216kg (docs/superpowers/specs/2026-07-12-asset-bisect-design.md)."
    ),
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Override the PPO runner cfg's seed (asset-bisect 3-seed protocol; default: keep agent cfg's own).",
)

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
    if args_cli.variant == "joint-die":
        from tasks.franka.dice_lift_joint_env_cfg import FrankaDieLiftJointEnvCfg

        env_cfg = FrankaDieLiftJointEnvCfg()
    elif args_cli.variant == "joint-cube":
        from tasks.franka.dice_lift_joint_env_cfg import FrankaCubeLiftJointEnvCfg

        env_cfg = FrankaCubeLiftJointEnvCfg()
    elif args_cli.variant == "joint-die-heavy":
        from tasks.franka.dice_lift_joint_env_cfg import FrankaDieLiftJointHeavyEnvCfg

        env_cfg = FrankaDieLiftJointHeavyEnvCfg()
    elif args_cli.variant == "joint-die-big":
        from tasks.franka.dice_lift_joint_env_cfg import FrankaDieLiftJointBigEnvCfg

        env_cfg = FrankaDieLiftJointBigEnvCfg()
    else:
        env_cfg = FrankaLiftEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    agent_cfg = FrankaLiftPPORunnerCfg()
    agent_cfg.device = args_cli.device
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations

    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed

    env_cfg.seed = agent_cfg.seed

    _log_suffix = {
        "ik-cube": "",
        "joint-die": "_jointdie",
        "joint-cube": "_jointcube",
        "joint-die-heavy": "_jointdieheavy",
        "joint-die-big": "_jointdiebig",
    }[args_cli.variant]
    log_dir = os.path.join(
        LOG_ROOT + _log_suffix,
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )
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

    num_learning_iterations = agent_cfg.max_iterations
    if args_cli.checkpoint is not None:
        runner.load(args_cli.checkpoint)
        resumed_at = runner.current_learning_iteration
        num_learning_iterations = max(agent_cfg.max_iterations - resumed_at, 0)
        print(
            f"Resumed from {args_cli.checkpoint} at iteration {resumed_at}; running "
            f"{num_learning_iterations} more iteration(s) to reach the absolute target "
            f"{agent_cfg.max_iterations}."
        )

    runner.learn(num_learning_iterations=num_learning_iterations, init_at_random_ep_len=True)

    env.close()
    print(f"Training complete. Checkpoints and logs written to: {log_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
