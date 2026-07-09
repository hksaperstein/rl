"""Direct resolution check: does the trained Experiment 26 checkpoint's
deterministic policy actually move the arm toward the cube early in the
episode and then hold, or does it never move at all?

Prints EE-to-cube distance (reach_dist) every 25 steps (0.5s) across a full
episode, for a handful of envs, using the exact deterministic inference
policy (get_inference_policy) - the same policy scripts/graspgoal_closeup_video.py
uses. Resolves a direct disagreement between an earlier visual frame-sampling
review (which suggested the arm never moves at all) and an instrumented
rollout from a separate investigation (which found the arm reaches within
~2.4cm of the cube). This script settles it with a full per-step distance
trace, not sparse visual sampling or a single before/after check.

.. code-block:: bash

    /home/saps/IsaacLab/isaaclab.sh -p scripts/graspgoal_reach_trajectory_check.py \
        --checkpoint logs/train/2026-07-09_15-18-06/model_1499.pt --num_envs 4
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Trace reach_dist over a full episode for the trained grasp-goal policy.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the rsl_rl checkpoint (.pt) to load.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel envs to trace.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

if not os.path.isfile(args_cli.checkpoint):
    sys.exit(f"Checkpoint not found: {args_cli.checkpoint}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from tasks.ar4.agents.rsl_rl_ppo_cfg import Ar4PickPlacePPORunnerCfg  # noqa: E402
from tasks.ar4.pickplace_graspgoal_env_cfg import Ar4PickPlaceGraspGoalEnvCfg  # noqa: E402


def main() -> None:
    env_cfg = Ar4PickPlaceGraspGoalEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    agent_cfg = Ar4PickPlacePPORunnerCfg()
    agent_cfg.device = args_cli.device

    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode=None)
    max_episode_length = int(env.max_episode_length)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs = env.get_observations()

    def reach_dist():
        cube = env.unwrapped.scene["cube"]
        ee = env.unwrapped.scene["ee_frame"].data.target_pos_w[:, 0, :]
        return torch.norm(ee - cube.data.root_pos_w, dim=-1)

    print("=" * 70)
    print(f"Checkpoint: {args_cli.checkpoint}, {args_cli.num_envs} envs, {max_episode_length} steps/episode")
    print("=" * 70)
    rd0 = reach_dist()
    print(f"step 0 (reset): reach_dist per env = {[round(x, 4) for x in rd0.cpu().tolist()]}")

    with torch.inference_mode():
        for step_idx in range(1, max_episode_length + 1):
            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)
            if step_idx <= 100 and step_idx % 10 == 0 or step_idx % 100 == 0:
                rd = reach_dist()
                grasped = env.unwrapped._grasped.cpu().tolist()
                lifted = env.unwrapped._lifted.cpu().tolist()
                print(
                    f"step {step_idx:4d} ({step_idx * env.unwrapped.step_dt:5.2f}s): "
                    f"reach_dist = {[round(x, 4) for x in rd.cpu().tolist()]}  "
                    f"grasped={grasped} lifted={lifted}"
                )

    env.close()
    print("[DONE]")


if __name__ == "__main__":
    main()
    simulation_app.close()
