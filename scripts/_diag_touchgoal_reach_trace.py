"""Diagnostic: reach_dist trace for Ar4PickPlaceTouchGoalEnvCfg policy.

Traces reach_dist (EE-to-cube distance) across a full episode using the
checkpoint from 2026-07-09_13-07-51, num_envs=8. Used to verify whether
the touchgoal policy actually reaches the cube and at which step, and to
check normalization behavior in the inference policy.

.. code-block:: bash

    DISPLAY=:1 flock /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/_diag_touchgoal_reach_trace.py --num_envs 8"
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Trace reach_dist over a full episode for the trained touchgoal policy.")
parser.add_argument("--checkpoint", type=str, default="logs/train/2026-07-09_13-07-51/model_1499.pt",
                    help="Path to the rsl_rl checkpoint (.pt) to load.")
parser.add_argument("--num_envs", type=int, default=8, help="Number of parallel envs to trace.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

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
from tasks.ar4.pickplace_touchgoal_env_cfg import Ar4PickPlaceTouchGoalEnvCfg  # noqa: E402


def main() -> None:
    env_cfg = Ar4PickPlaceTouchGoalEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    agent_cfg = Ar4PickPlacePPORunnerCfg()
    agent_cfg.device = args_cli.device

    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode=None)
    max_episode_length = int(env.max_episode_length)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print("[INFO] Loading checkpoint...")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    print("[INFO] Getting initial observations...")
    obs = env.get_observations()

    def reach_dist():
        cube = env.unwrapped.scene["cube"]
        ee = env.unwrapped.scene["ee_frame"].data.target_pos_w[:, 0, :]
        return torch.norm(ee - cube.data.root_pos_w, dim=-1)

    print("=" * 80)
    print(f"Checkpoint: {args_cli.checkpoint}")
    print(f"Num envs: {args_cli.num_envs}, Episode length: {max_episode_length} steps")
    print(f"empirical_normalization: {agent_cfg.empirical_normalization}")
    print("=" * 80)
    rd0 = reach_dist()
    print(f"step 0 (reset): reach_dist per env = {[round(x, 4) for x in rd0.cpu().tolist()]}")

    min_reach_dist = rd0.cpu().tolist()
    min_reach_step = [0] * args_cli.num_envs

    with torch.inference_mode():
        for step_idx in range(1, max_episode_length + 1):
            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)

            rd = reach_dist()
            rd_list = rd.cpu().tolist()

            # Track minimum reach distance per env
            for i, dist in enumerate(rd_list):
                if dist < min_reach_dist[i]:
                    min_reach_dist[i] = dist
                    min_reach_step[i] = step_idx

            # Print every 50 steps, or more frequently early on
            if step_idx <= 200 and step_idx % 20 == 0 or step_idx % 100 == 0:
                print(
                    f"step {step_idx:4d} ({step_idx * env.unwrapped.step_dt:6.2f}s): "
                    f"reach_dist = {[round(x, 4) for x in rd_list]}"
                )

    env.close()

    print("=" * 80)
    print("SUMMARY:")
    for i in range(args_cli.num_envs):
        drops_below_threshold = min_reach_dist[i] < 0.05
        print(f"  env {i}: min_reach_dist={round(min_reach_dist[i], 4)}m at step {min_reach_step[i]}, "
              f"drops_below_0.05m={drops_below_threshold}")
    print("=" * 80)
    print("[DONE]")


if __name__ == "__main__":
    main()
    simulation_app.close()
