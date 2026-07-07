# scripts/eval_loop.py
"""Run a trained AR4 pick-and-place PPO policy for a fixed number of episodes,
recording each one as an mp4 to logs/videos/.

.. code-block:: bash

    cd ~/projects/rl
    /home/saps/IsaacLab/isaaclab.sh -p scripts/eval_loop.py --checkpoint logs/train/<run>/model_1500.pt --episodes 10
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run a trained AR4 pick-and-place policy and record video.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the rsl_rl checkpoint (.pt) to load.")
parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run.")
parser.add_argument(
    "--perception",
    action="store_true",
    default=False,
    help="Use the real camera-based perception pipeline instead of privileged simulation state for the sphere's observed position.",
)
parser.add_argument(
    "--mirror",
    action="store_true",
    default=False,
    help="Evaluate the mirror-goal scene (see scripts/train.py --mirror) instead of the four-object scene.",
)
parser.add_argument(
    "--ik_guided",
    action="store_true",
    default=False,
    help="Evaluate the classical-IK-guided scene (see scripts/train.py --ik_guided) instead of the four-object scene.",
)
parser.add_argument(
    "--taskspace",
    action="store_true",
    default=False,
    help="Evaluate the task-space IK-driven-action scene (see scripts/train.py --taskspace) instead of the four-object scene.",
)
parser.add_argument(
    "--residual",
    action="store_true",
    default=False,
    help="Evaluate the residual-action scene (see scripts/train.py --residual) instead of the four-object scene.",
)
parser.add_argument(
    "--reachskip",
    action="store_true",
    default=False,
    help="Evaluate the reach-skip curriculum scene (see scripts/train.py --reachskip) instead of the four-object scene.",
)
parser.add_argument(
    "--baseproximity",
    action="store_true",
    default=False,
    help="Evaluate the reward-shaping scene (see scripts/train.py --baseproximity) instead of the four-object scene.",
)
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

import imageio  # noqa: E402

from _perception_adapter import perceive_sphere, sphere_position_obs_slice  # noqa: E402
from perception.overlay import draw_detections  # noqa: E402
from perception.tracker import ObjectTracker  # noqa: E402
from tasks.ar4.agents.rsl_rl_ppo_cfg import Ar4PickPlacePPORunnerCfg  # noqa: E402
from tasks.ar4.pickplace_env_cfg import GROUND_Z, Ar4PickPlaceEnvCfg, Ar4PickPlacePerceptionEnvCfg  # noqa: E402
from tasks.ar4.pickplace_ik_guided_env_cfg import Ar4PickPlaceIkGuidedEnvCfg  # noqa: E402
from tasks.ar4.pickplace_mirror_env_cfg import Ar4PickPlaceMirrorEnvCfg  # noqa: E402
from tasks.ar4.pickplace_baseproximity_env_cfg import Ar4PickPlaceBaseProximityEnvCfg  # noqa: E402
from tasks.ar4.pickplace_reachskip_env_cfg import Ar4PickPlaceReachskipEnvCfg  # noqa: E402
from tasks.ar4.pickplace_residual_env_cfg import Ar4PickPlaceResidualEnvCfg  # noqa: E402
from tasks.ar4.pickplace_taskspace_env_cfg import (  # noqa: E402
    Ar4PickPlaceTaskspaceEnvCfg,
    Ar4PickPlaceTaskspacePPORunnerCfg,
)

VIDEO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "videos")


def main() -> None:
    if args_cli.baseproximity:
        env_cfg_cls = Ar4PickPlaceBaseProximityEnvCfg
    elif args_cli.reachskip:
        env_cfg_cls = Ar4PickPlaceReachskipEnvCfg
    elif args_cli.residual:
        env_cfg_cls = Ar4PickPlaceResidualEnvCfg
    elif args_cli.taskspace:
        env_cfg_cls = Ar4PickPlaceTaskspaceEnvCfg
    elif args_cli.ik_guided:
        env_cfg_cls = Ar4PickPlaceIkGuidedEnvCfg
    elif args_cli.mirror:
        env_cfg_cls = Ar4PickPlaceMirrorEnvCfg
    elif args_cli.perception:
        env_cfg_cls = Ar4PickPlacePerceptionEnvCfg
    else:
        env_cfg_cls = Ar4PickPlaceEnvCfg
    env_cfg = env_cfg_cls()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device

    if args_cli.taskspace or args_cli.residual or args_cli.reachskip or args_cli.baseproximity:
        agent_cfg = Ar4PickPlaceTaskspacePPORunnerCfg()
    else:
        agent_cfg = Ar4PickPlacePPORunnerCfg()
    agent_cfg.device = args_cli.device

    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode="rgb_array")

    tracker = None
    perception_writer = None
    if args_cli.perception:
        tracker = ObjectTracker()
        perception_writer = imageio.get_writer(
            os.path.join(VIDEO_DIR, "ar4_pickplace_perception.mp4"), fps=int(1.0 / env.step_dt), codec="libx264"
        )
    else:
        # step_trigger, not episode_trigger: RslRlVecEnvWrapper resets the env exactly
        # once at construction and never again, so episode_trigger's episode_id never
        # advances and silently merges every episode into one video. 250 = one episode's
        # worth of steps (episode_length_s=5.0 / step_dt=decimation*sim.dt=2*0.01=0.02s),
        # from Ar4PickPlaceEnvCfg - update this if that config changes.
        if args_cli.baseproximity:
            name_prefix = "ar4_pickplace_baseproximity"
        elif args_cli.reachskip:
            name_prefix = "ar4_pickplace_reachskip"
        elif args_cli.residual:
            name_prefix = "ar4_pickplace_residual"
        elif args_cli.taskspace:
            name_prefix = "ar4_pickplace_taskspace"
        elif args_cli.ik_guided:
            name_prefix = "ar4_pickplace_ik_guided"
        elif args_cli.mirror:
            name_prefix = "ar4_pickplace_mirror"
        else:
            name_prefix = "ar4_pickplace"
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=VIDEO_DIR,
            step_trigger=lambda step: step % 250 == 0,
            video_length=250,
            name_prefix=name_prefix,
            disable_logger=True,
        )

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    camera = env.unwrapped.scene["perception_camera"] if args_cli.perception else None
    sphere_slice = sphere_position_obs_slice(env.unwrapped) if args_cli.perception else None

    obs = env.get_observations()
    completed_episodes = 0
    with torch.inference_mode():
        while completed_episodes < args_cli.episodes and simulation_app.is_running():
            if args_cli.perception:
                sphere_pos_b, tracked, rgb = perceive_sphere(env.unwrapped, camera, tracker, GROUND_Z)
                if sphere_pos_b is not None:
                    col_start, col_end = sphere_slice
                    obs[:, col_start:col_end] = sphere_pos_b
                perception_writer.append_data(draw_detections(rgb, tracked))

            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if bool(dones[0]):
                completed_episodes += 1
                print(f"[INFO] Completed episode {completed_episodes}/{args_cli.episodes}")

    if perception_writer is not None:
        perception_writer.close()
    env.close()
    print(f"Videos written to: {VIDEO_DIR}")


if __name__ == "__main__":
    main()
    simulation_app.close()
