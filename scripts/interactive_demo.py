# scripts/interactive_demo.py
"""Interactive AR4 pick-and-place demo: drag the sphere anywhere in the Isaac Sim
GUI viewport (native drag gizmo), and once it settles the trained policy picks
it up and places it in the fixed target region on the other side - using the
real camera-based perception pipeline the whole time, exactly as
eval_loop.py --perception does at inference time.

An out-of-view or out-of-the-workspace sphere position never triggers an
attempt - the arm just keeps watching and waiting.

A single "armed" flag guards each trigger: once a pick-and-place attempt
fires, the demo disarms itself so the sphere sitting still at the goal
position right after placement can't immediately re-trigger another
attempt. It only re-arms once the sphere is observed to have changed state
(dragged away, gone stale/out of view, or moved past the stability
tolerance) - i.e. real evidence of a fresh human drag.

.. code-block:: bash

    cd ~/projects/rl
    /home/saps/IsaacLab/isaaclab.sh -p scripts/interactive_demo.py --checkpoint logs/train/<run>/model_1500.pt
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run the interactive AR4 pick-and-place demo.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the rsl_rl checkpoint (.pt) to load.")
parser.add_argument(
    "--stable_seconds", type=float, default=1.0, help="How long the sphere must stay put before the robot acts."
)
parser.add_argument("--stable_tolerance", type=float, default=0.005, help="Max drift (m) still considered 'stable'.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

if not os.path.isfile(args_cli.checkpoint):
    sys.exit(f"Checkpoint not found: {args_cli.checkpoint}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import imageio
import numpy as np
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _perception_adapter import perceive_sphere, sphere_position_obs_slice  # noqa: E402
from perception.overlay import draw_detections  # noqa: E402
from perception.tracker import ObjectTracker, find_by_shape  # noqa: E402
from tasks.ar4.agents.rsl_rl_ppo_cfg import Ar4PickPlacePPORunnerCfg  # noqa: E402
from tasks.ar4.pickplace_env_cfg import GROUND_Z, WORKSPACE_BOUNDS, Ar4PickPlaceDemoEnvCfg  # noqa: E402

VIDEO_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "videos", "ar4_interactive_demo.mp4"
)


def _in_workspace_bounds(position: np.ndarray) -> bool:
    x, y, z = position
    return (
        WORKSPACE_BOUNDS["x"][0] <= x <= WORKSPACE_BOUNDS["x"][1]
        and WORKSPACE_BOUNDS["y"][0] <= y <= WORKSPACE_BOUNDS["y"][1]
        and WORKSPACE_BOUNDS["z"][0] <= z <= WORKSPACE_BOUNDS["z"][1]
    )


def main() -> None:
    env_cfg = Ar4PickPlaceDemoEnvCfg()
    env_cfg.scene.num_envs = 1

    agent_cfg = Ar4PickPlacePPORunnerCfg()

    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(env, clip_actions=None)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    camera = env.unwrapped.scene["perception_camera"]
    sphere_slice = sphere_position_obs_slice(env.unwrapped)
    tracker = ObjectTracker()
    video_writer = imageio.get_writer(VIDEO_PATH, fps=int(1.0 / env.unwrapped.step_dt), codec="libx264")

    stable_steps_needed = int(args_cli.stable_seconds / env.unwrapped.step_dt)
    last_stable_position = None
    stable_count = 0
    # Guards against re-triggering on the sphere the robot itself just placed: a trigger disarms
    # itself, and only re-arms once the sphere is observed to have changed (dragged, gone stale/
    # out of view, or moved past the stability tolerance) since the last placement.
    armed = True

    obs = env.get_observations()
    print("[INFO] Watching for the sphere to be placed and settled. Drag it in the viewport.")
    with torch.inference_mode():
        while simulation_app.is_running():
            sphere_pos_b, tracked, rgb = perceive_sphere(env.unwrapped, camera, tracker, GROUND_Z)
            sphere = find_by_shape(tracked, "sphere")

            ready_to_act = False
            if sphere is not None and not sphere.is_stale and _in_workspace_bounds(sphere.position):
                if last_stable_position is None:
                    # No baseline yet (e.g. right after a trigger reset the bookkeeping below) -
                    # this alone is not evidence of a human drag, so leave `armed` untouched.
                    stable_count = 0
                elif np.linalg.norm(sphere.position - last_stable_position) <= args_cli.stable_tolerance:
                    stable_count += 1
                else:
                    # The sphere actually moved since we last looked - real evidence of a drag.
                    stable_count = 0
                    armed = True
                last_stable_position = sphere.position
                ready_to_act = armed and stable_count >= stable_steps_needed
            else:
                stable_count = 0
                last_stable_position = None
                armed = True

            video_writer.append_data(draw_detections(rgb, tracked))

            if not ready_to_act:
                action_dim = env.unwrapped.action_manager.total_action_dim
                actions = torch.zeros(env.unwrapped.num_envs, action_dim, device=env.unwrapped.device)
                obs, _, _, _ = env.step(actions)
                continue

            print("[INFO] Sphere settled - picking it up.")
            armed = False
            stable_count = 0
            last_stable_position = None
            episode_done = False
            for _ in range(env.unwrapped.max_episode_length):
                if sphere_pos_b is not None:
                    col_start, col_end = sphere_slice
                    obs[:, col_start:col_end] = sphere_pos_b
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                sphere_pos_b, tracked, rgb = perceive_sphere(env.unwrapped, camera, tracker, GROUND_Z)
                video_writer.append_data(draw_detections(rgb, tracked))
                if bool(dones[0]):
                    episode_done = True
                    break
            print(f"[INFO] Pick-and-place {'succeeded' if episode_done else 'timed out'}. Watching for the next drag.")

    video_writer.close()
    env.close()
    print(f"Demo video written to: {VIDEO_PATH}")


if __name__ == "__main__":
    main()
    simulation_app.close()
