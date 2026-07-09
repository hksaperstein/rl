# scripts/touchgoal_instrumented_rollout.py
"""Instrumented numeric rollout for Experiment 25 (Ar4PickPlaceTouchGoalEnvCfg):
loads a trained checkpoint and runs it for a fixed number of episodes per env,
directly recomputing the touch/goal criteria from raw simulated state (end-
effector position vs. the cube's live touch point, end-effector position vs.
the fixed snapshotted goal point) rather than trusting the training-time scalar
metrics alone. This is a post-hoc verification tool, not part of the training
pipeline; it does not modify tasks/ar4/pickplace_touchgoal_env_cfg.py,
tasks/ar4/mdp.py, or tasks/ar4/touch_goal_reward.py.

Two independent success signals are tracked and cross-checked per episode:
  1. "own" recomputation: touch_dist = || ee_pos_w - (cube_pos_w + [0,0,cube_half_size]) ||
     latched true once < touch_tolerance; goal_dist = || ee_pos_w - env._touch_goal_pos_w ||,
     "reached" latched true once touched AND goal_dist < goal_tolerance at the same step.
     This mirrors tasks/ar4/mdp.py's touch_then_goal_reached formula exactly, but is
     computed independently in this script from raw scene/asset state, not by reading
     the env's own internal _touched_cube/_touch_goal_pos_w latch buffers (which get
     zeroed by the reset_touch_goal_milestone event term as part of the same step() call
     that ends an episode, before this script would get a chance to read them - a race
     this script avoids entirely by never depending on those buffers post-step).
  2. "mechanism" signal: the actual terminated/truncated outcome the trained env itself
     produced this rollout, taken from RslRlVecEnvWrapper's returned `dones` and
     `extras["time_outs"]` (dones=terminated|truncated, time_outs=truncated) - this is
     the *actual* touch_then_goal_reached DoneTerm firing during this exact rollout,
     not a training-time aggregate. success_this_step = dones & ~time_outs.

Usage:

.. code-block:: bash

    /home/saps/IsaacLab/isaaclab.sh -p scripts/touchgoal_instrumented_rollout.py \
        --checkpoint logs/train/2026-07-09_13-07-51/model_1499.pt --num_envs 16 --episodes_per_env 2
"""

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Instrumented rollout of a trained AR4 touch-goal policy.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the rsl_rl checkpoint (.pt) to load.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel envs to roll out.")
parser.add_argument("--episodes_per_env", type=int, default=2, help="Number of completed episodes to collect per env.")
parser.add_argument(
    "--output", type=str, default=None, help="Optional path to write the raw per-episode results as JSON."
)
parser.add_argument(
    "--stochastic",
    action="store_true",
    default=False,
    help=(
        "Sample actions from the policy's own action distribution (rsl_rl ActorCritic.act, the same"
        " sampling used during training rollouts) instead of the deterministic mean action"
        " (ActorCritic.act_inference, what runner.get_inference_policy()/eval_loop.py's pattern returns)."
        " Added specifically to check whether training-time stochastic exploration noise, not just the"
        " policy's learned mean behavior, accounts for the training-time termination rate."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = False  # purely numeric rollout, no rendering needed

if not os.path.isfile(args_cli.checkpoint):
    sys.exit(f"Checkpoint not found: {args_cli.checkpoint}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tasks.ar4.agents.rsl_rl_ppo_cfg import Ar4PickPlacePPORunnerCfg  # noqa: E402
from tasks.ar4.pickplace_touchgoal_env_cfg import (  # noqa: E402
    CUBE_HALF_SIZE,
    GOAL_TOLERANCE,
    TOUCH_TOLERANCE,
    Ar4PickPlaceTouchGoalEnvCfg,
)


def main() -> None:
    env_cfg = Ar4PickPlaceTouchGoalEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    agent_cfg = Ar4PickPlacePPORunnerCfg()
    agent_cfg.device = args_cli.device

    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode=None)
    num_envs = env.num_envs
    device = env.device
    max_episode_length = env.max_episode_length  # steps per episode (1000 for episode_length_s=20.0, step_dt=0.02s)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    if args_cli.stochastic:
        # get_inference_policy() above already switched the policy to PyTorch eval mode (via
        # nn.Module.eval() - disables dropout, NOT Python's eval() builtin) and moved it to
        # `device`; only swap out which bound method we call each step (act = sample from the
        # actor's Gaussian, the same call rsl_rl's own rollout loop uses during training -
        # vs. act_inference = deterministic mean action, what eval_loop.py's established
        # pattern and the default path of this script use).
        policy = runner.alg.policy.act
        print("[INFO] Using STOCHASTIC action sampling (ActorCritic.act), matching training-time rollout behavior.")

    cube_half_size_t = torch.tensor([0.0, 0.0, CUBE_HALF_SIZE], device=device)

    # Per-env running state for the CURRENT (not-yet-completed) episode.
    own_touched = torch.zeros(num_envs, dtype=torch.bool, device=device)
    own_reached = torch.zeros(num_envs, dtype=torch.bool, device=device)
    min_touch_dist = torch.full((num_envs,), float("inf"), device=device)
    min_goal_dist_after_touch = torch.full((num_envs,), float("inf"), device=device)
    first_touch_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    reach_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    ep_start_step = torch.zeros(num_envs, dtype=torch.long, device=device)
    episode_count = torch.zeros(num_envs, dtype=torch.long, device=device)

    results = []  # list of dicts, one per completed (env, episode)

    max_steps = args_cli.episodes_per_env * int(max_episode_length) + 200  # safety margin
    step_idx = 0

    obs = env.get_observations()
    with torch.inference_mode():
        while step_idx < max_steps and int(episode_count.min().item()) < args_cli.episodes_per_env:
            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)
            step_idx += 1

            ee_pos_w = env.unwrapped.scene["ee_frame"].data.target_pos_w[:, 0, :]
            cube_pos_w = env.unwrapped.scene["cube"].data.root_pos_w
            touch_point_w = cube_pos_w + cube_half_size_t
            touch_dist = torch.norm(ee_pos_w - touch_point_w, dim=-1)
            goal_pos_w = env.unwrapped._touch_goal_pos_w
            goal_dist = torch.norm(ee_pos_w - goal_pos_w, dim=-1)

            newly_touched = (touch_dist < TOUCH_TOLERANCE) & (~own_touched)
            first_touch_step[newly_touched] = step_idx
            own_touched |= touch_dist < TOUCH_TOLERANCE

            min_touch_dist = torch.minimum(min_touch_dist, touch_dist)
            post_touch_mask = own_touched
            min_goal_dist_after_touch = torch.where(
                post_touch_mask, torch.minimum(min_goal_dist_after_touch, goal_dist), min_goal_dist_after_touch
            )

            newly_reached = own_touched & (goal_dist < GOAL_TOLERANCE) & (~own_reached)
            reach_step[newly_reached] = step_idx
            own_reached |= own_touched & (goal_dist < GOAL_TOLERANCE)

            time_outs = extras.get("time_outs")
            dones_bool = dones.bool()
            mechanism_success = dones_bool & (~time_outs.bool() if time_outs is not None else torch.zeros_like(dones_bool))

            done_env_ids = torch.nonzero(dones_bool, as_tuple=False).flatten().tolist()
            for env_id in done_env_ids:
                if int(episode_count[env_id].item()) >= args_cli.episodes_per_env:
                    continue  # already collected enough episodes for this env, ignore further ones
                ep_len = step_idx - int(ep_start_step[env_id].item())
                results.append(
                    {
                        "env_id": env_id,
                        "episode_idx": int(episode_count[env_id].item()),
                        "episode_len_steps": ep_len,
                        "own_touched": bool(own_touched[env_id].item()),
                        "own_reached": bool(own_reached[env_id].item()),
                        "mechanism_success": bool(mechanism_success[env_id].item()),
                        "mechanism_timeout": bool(time_outs[env_id].item()) if time_outs is not None else None,
                        "min_touch_dist": float(min_touch_dist[env_id].item()),
                        "min_goal_dist_after_touch": (
                            float(min_goal_dist_after_touch[env_id].item())
                            if own_touched[env_id].item()
                            else None
                        ),
                        "first_touch_step": int(first_touch_step[env_id].item()),
                        "reach_step": int(reach_step[env_id].item()),
                    }
                )
                episode_count[env_id] += 1
                # reset this env's running trackers for its next episode
                own_touched[env_id] = False
                own_reached[env_id] = False
                min_touch_dist[env_id] = float("inf")
                min_goal_dist_after_touch[env_id] = float("inf")
                first_touch_step[env_id] = -1
                reach_step[env_id] = -1
                ep_start_step[env_id] = step_idx

            if step_idx % 200 == 0:
                print(f"[INFO] step {step_idx}/{max_steps}, episode_count min/max: "
                      f"{int(episode_count.min().item())}/{int(episode_count.max().item())}")

    env.close()

    n_episodes = len(results)
    n_own_reached = sum(1 for r in results if r["own_reached"])
    n_mechanism_success = sum(1 for r in results if r["mechanism_success"])
    n_touched = sum(1 for r in results if r["own_touched"])
    n_agree = sum(1 for r in results if r["own_reached"] == r["mechanism_success"])

    print("\n" + "=" * 70)
    print(f"Total completed episodes collected: {n_episodes} (target {args_cli.num_envs * args_cli.episodes_per_env})")
    print(f"Touched cube at some point:         {n_touched}/{n_episodes} ({n_touched / max(n_episodes,1):.4f})")
    print(f"'own' recompute reached goal:       {n_own_reached}/{n_episodes} ({n_own_reached / max(n_episodes,1):.4f})")
    print(f"mechanism (actual DoneTerm) success: {n_mechanism_success}/{n_episodes} "
          f"({n_mechanism_success / max(n_episodes,1):.4f})")
    print(f"own vs. mechanism signal agreement: {n_agree}/{n_episodes} ({n_agree / max(n_episodes,1):.4f})")
    print("=" * 70)

    for r in results:
        print(r)

    if args_cli.output:
        with open(args_cli.output, "w") as f:
            json.dump(
                {
                    "checkpoint": args_cli.checkpoint,
                    "num_envs": args_cli.num_envs,
                    "episodes_per_env": args_cli.episodes_per_env,
                    "n_episodes": n_episodes,
                    "n_touched": n_touched,
                    "n_own_reached": n_own_reached,
                    "n_mechanism_success": n_mechanism_success,
                    "n_agree": n_agree,
                    "episodes": results,
                },
                f,
                indent=2,
            )
        print(f"[INFO] Wrote raw results to {args_cli.output}")


if __name__ == "__main__":
    main()
    simulation_app.close()
