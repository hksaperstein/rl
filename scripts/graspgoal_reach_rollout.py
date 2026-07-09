# scripts/graspgoal_reach_rollout.py
"""Decisive behavioral check for Experiment 26's freeze: loads a trained
graspgoal checkpoint and rolls it out BOTH deterministically (mean action)
and stochastically (sampled from the policy's Gaussian, matching training-
time exploration), measuring the actual min EE-to-cube distance and grasp/
lift counts reached per episode. Distinguishes "frozen even under training
noise" from "reaches the cube under noise but the learned mean is frozen".

    /home/saps/IsaacLab/isaaclab.sh -p scripts/graspgoal_reach_rollout.py \
        --checkpoint logs/train/2026-07-09_16-48-57/model_299.pt --num_envs 16
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Reach/grasp rollout of a trained AR4 grasp-goal policy.")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=1500, help="steps per mode (~1 episode at 1500).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = False

if not os.path.isfile(args_cli.checkpoint):
    sys.exit(f"Checkpoint not found: {args_cli.checkpoint}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from tasks.ar4.agents.rsl_rl_ppo_cfg import Ar4PickPlacePPORunnerCfg  # noqa: E402
from tasks.ar4.pickplace_graspgoal_env_cfg import Ar4PickPlaceGraspGoalEnvCfg  # noqa: E402


def run_mode(env, policy_fn, label, steps):
    device = env.unwrapped.device
    n = env.num_envs
    min_rd = torch.full((n,), float("inf"), device=device)
    obs = env.get_observations()
    with torch.inference_mode():
        for _ in range(steps):
            actions = policy_fn(obs)
            obs, _, dones, extras = env.step(actions)
            ee = env.unwrapped.scene["ee_frame"].data.target_pos_w[:, 0, :]
            cube = env.unwrapped.scene["cube"].data.root_pos_w
            rd = torch.norm(ee - cube, dim=-1)
            min_rd = torch.minimum(min_rd, rd)
    grasped = env.unwrapped._grasped.sum().item() if hasattr(env.unwrapped, "_grasped") else -1
    lifted = env.unwrapped._lifted.sum().item() if hasattr(env.unwrapped, "_lifted") else -1
    rl = min_rd.cpu().tolist()
    print(f"\n==== MODE: {label} ({steps} steps, {n} envs) ====")
    print(f"  min reach_dist per env: {[round(x, 4) for x in rl]}")
    print(f"  min={min(rl):.4f}  median={sorted(rl)[n // 2]:.4f}  max={max(rl):.4f}")
    print(f"  envs reaching <0.10m: {sum(1 for x in rl if x < 0.10)}/{n}")
    print(f"  envs reaching <0.05m: {sum(1 for x in rl if x < 0.05)}/{n}")
    print(f"  cumulative _grasped envs: {grasped}/{n}, _lifted envs: {lifted}/{n}")


def main() -> None:
    env_cfg = Ar4PickPlaceGraspGoalEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    agent_cfg = Ar4PickPlacePPORunnerCfg()
    agent_cfg.device = args_cli.device

    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)

    det_policy = runner.get_inference_policy(device=env.unwrapped.device)  # mean action
    run_mode(env, det_policy, "DETERMINISTIC (mean action)", args_cli.steps)

    env.close()
    print("\n[ROLLOUT] done.")


if __name__ == "__main__":
    main()
    simulation_app.close()
