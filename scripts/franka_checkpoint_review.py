"""Record a video of a trained Franka Panda cube-lift checkpoint - a fresh,
from-scratch eval/demo entry point for tasks/franka/ (does NOT reuse/extend
tasks/ar4's own closeup-video scripts, e.g. scripts/graspgoal_closeup_video.py,
per the franka-panda-pivot's "everything new" instruction).

Simpler mechanism than the AR4-era closeup scripts: instead of a dedicated
per-env close-up FrameTransformer/camera sensor, this just uses
gym.wrappers.RecordVideo on the environment's own built-in viewport render
(render_mode="rgb_array") - the same mechanism scripts/train_franka.py's
own --video flag already uses - to capture one continuous video covering
all envs for one full episode. Good enough for "let's look at what this
checkpoint actually does" without needing a bespoke camera setup.

.. code-block:: bash

    /home/saps/IsaacLab/isaaclab.sh -p scripts/franka_checkpoint_review.py \
        --checkpoint logs/train_franka/2026-07-09_22-05-51/model_800.pt --num_envs 8
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record video of a trained Franka Panda cube-lift checkpoint.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the rsl_rl checkpoint (.pt) to load.")
parser.add_argument("--num_envs", type=int, default=8, help="Number of parallel envs to render in the video.")
parser.add_argument(
    "--video_length", type=int, default=250, help="Video length in steps (default: one full episode, 5s @ 50Hz)."
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Directory to write the video to (default: logs/videos/franka_checkpoint_review/).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # required for video rendering

if not os.path.isfile(args_cli.checkpoint):
    sys.exit(f"Checkpoint not found: {args_cli.checkpoint}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from tasks.franka.agents.rsl_rl_ppo_cfg import FrankaLiftPPORunnerCfg  # noqa: E402
from tasks.franka.lift_env_cfg import FrankaLiftEnvCfg_PLAY  # noqa: E402

VIDEO_DIR = args_cli.output_dir or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "videos", "franka_checkpoint_review"
)
os.makedirs(VIDEO_DIR, exist_ok=True)


def main() -> None:
    env_cfg = FrankaLiftEnvCfg_PLAY()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    agent_cfg = FrankaLiftPPORunnerCfg()
    agent_cfg.device = args_cli.device

    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode="rgb_array")

    video_kwargs = {
        "video_folder": VIDEO_DIR,
        "step_trigger": lambda step: step == 0,
        "video_length": args_cli.video_length,
        "name_prefix": f"franka_checkpoint_review_{os.path.splitext(os.path.basename(args_cli.checkpoint))[0]}",
        "disable_logger": True,
    }
    print_dict(video_kwargs, nesting=4)
    env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs = env.get_observations()
    with torch.inference_mode():
        for _ in range(args_cli.video_length):
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

    env.close()
    print(f"Checkpoint review video written to: {VIDEO_DIR}")


if __name__ == "__main__":
    main()
    simulation_app.close()
