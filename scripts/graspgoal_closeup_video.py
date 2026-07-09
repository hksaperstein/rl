# scripts/graspgoal_closeup_video.py
"""Close-up, individual-env video/frame capture for Experiment 26
(Ar4PickPlaceGraspGoalEnvCfg / Ar4GraspGoalDemoEnvCfg): records the trained
policy from a fixed close-up, front/head-on demo_camera
(tasks/ar4/graspgoal_democam_env_cfg.py), one env at a time (not the wide
multi-env training grid), for genuine visual confirmation of what the policy
does. Reads camera.data.output["rgb"] directly per env, same pattern as
scripts/touchgoal_closeup_video.py (Experiment 25's own version of this tool).

Alongside a full per-env mp4, saves PNG snapshots at key moments per env:
start, first-grasp (if any), first-lift (if any), goal-reached (if any), and
the episode's final frame (success or timeout).

This is a post-hoc verification tool; it does not modify
tasks/ar4/pickplace_graspgoal_env_cfg.py, tasks/ar4/mdp.py, or
tasks/ar4/grasp_goal_reward.py.

Usage:

.. code-block:: bash

    /home/saps/IsaacLab/isaaclab.sh -p scripts/graspgoal_closeup_video.py \
        --checkpoint logs/train/2026-07-09_15-18-06/model_1499.pt --num_envs 4
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record close-up per-env video of a trained AR4 grasp-goal policy.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the rsl_rl checkpoint (.pt) to load.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel envs to record (one video/set each).")
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Directory to write videos/frames to (default: logs/videos/graspgoal_closeup/).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # required for camera-sensor rendering

if not os.path.isfile(args_cli.checkpoint):
    sys.exit(f"Checkpoint not found: {args_cli.checkpoint}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import imageio  # noqa: E402
import torch  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from tasks.ar4.agents.rsl_rl_ppo_cfg import Ar4PickPlacePPORunnerCfg  # noqa: E402
from tasks.ar4.graspgoal_democam_env_cfg import Ar4GraspGoalDemoEnvCfg  # noqa: E402
from tasks.ar4.pickplace_graspgoal_env_cfg import GOAL_TOLERANCE  # noqa: E402

VIDEO_DIR = args_cli.output_dir or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "videos", "graspgoal_closeup"
)
os.makedirs(VIDEO_DIR, exist_ok=True)


def main() -> None:
    env_cfg = Ar4GraspGoalDemoEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    agent_cfg = Ar4PickPlacePPORunnerCfg()
    agent_cfg.device = args_cli.device

    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode=None)
    num_envs = env.num_envs
    device = env.device
    max_episode_length = int(env.max_episode_length)
    step_dt = env.step_dt

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    camera = env.unwrapped.scene["demo_camera"]

    writers = [
        imageio.get_writer(os.path.join(VIDEO_DIR, f"env_{i}.mp4"), fps=int(1.0 / step_dt), codec="libx264")
        for i in range(num_envs)
    ]
    done_recording = [False] * num_envs  # True once env i's first episode has ended
    grasped_seen = [False] * num_envs
    lifted_seen = [False] * num_envs
    reached = [False] * num_envs
    snapshots_saved = {i: {} for i in range(num_envs)}  # env_id -> {moment_name: step_idx}

    def save_snapshot(env_id: int, moment: str, step_idx: int, frame) -> None:
        path = os.path.join(VIDEO_DIR, f"env_{env_id}_{moment}_step{step_idx}.png")
        imageio.imwrite(path, frame)
        snapshots_saved[env_id][moment] = step_idx
        print(f"[INFO] env {env_id}: saved '{moment}' snapshot at step {step_idx} -> {path}")

    obs = env.get_observations()
    max_steps = max_episode_length + 20  # small safety margin
    with torch.inference_mode():
        for step_idx in range(1, max_steps + 1):
            if all(done_recording):
                break
            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)

            cube_pos_w = env.unwrapped.scene["cube"].data.root_pos_w
            goal_pos_w = env.unwrapped._cube_goal_pos_w
            goal_dist = torch.norm(cube_pos_w - goal_pos_w, dim=-1)

            # camera.data.output["rgb"] comes back row-0-at-bottom (OpenGL
            # framebuffer convention, matching this camera's convention="opengl"
            # offset) - flip vertically (axis=1, the per-env row axis) so saved
            # frames are right-side-up. Confirmed empirically: an unflipped
            # render showed the ground grid/robot base at the TOP of frame and
            # sky at the bottom - flipping put the ground correctly at the
            # bottom, base resting on it, arm reaching up.
            rgb_all = camera.data.output["rgb"][:, ..., :3].cpu().numpy()[:, ::-1, :, :]
            dones_bool = dones.bool()
            time_outs = extras.get("time_outs")

            for i in range(num_envs):
                if done_recording[i]:
                    continue
                frame = rgb_all[i].astype("uint8")
                writers[i].append_data(frame)

                if step_idx == 1:
                    save_snapshot(i, "start", step_idx, frame)

                if not grasped_seen[i] and bool(env.unwrapped._grasped[i].item()):
                    grasped_seen[i] = True
                    save_snapshot(i, "grasped", step_idx, frame)

                if not lifted_seen[i] and bool(env.unwrapped._lifted[i].item()):
                    lifted_seen[i] = True
                    save_snapshot(i, "lifted", step_idx, frame)

                if lifted_seen[i] and not reached[i] and goal_dist[i].item() < GOAL_TOLERANCE:
                    reached[i] = True
                    save_snapshot(i, "goal_reached", step_idx, frame)

                if bool(dones_bool[i].item()):
                    is_timeout = bool(time_outs[i].item()) if time_outs is not None else True
                    moment = "timeout" if is_timeout else "final_success"
                    save_snapshot(i, moment, step_idx, frame)
                    done_recording[i] = True

            if step_idx % 200 == 0:
                print(f"[INFO] step {step_idx}/{max_steps}, envs done recording: {sum(done_recording)}/{num_envs}")

    for w in writers:
        w.close()
    env.close()

    print("\n" + "=" * 70)
    for i in range(num_envs):
        print(
            f"env {i}: grasped={grasped_seen[i]}, lifted={lifted_seen[i]}, reached={reached[i]}, "
            f"snapshots={snapshots_saved[i]}"
        )
    print(f"Videos/frames written to: {VIDEO_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
    simulation_app.close()
