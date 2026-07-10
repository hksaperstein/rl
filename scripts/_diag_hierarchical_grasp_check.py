"""Diagnostic: resolve whether the hierarchical policy checkpoint actually
achieved an antipodal grasp at any point during training, or if the tiny
nonzero bumps in Episode_Reward/grasp_lift_goal_milestone_bonus were spurious.

Loads the exact checkpoint and runs a full episode rollout, checking the
env._grasped and env._lifted latched booleans - which are set by tasks/ar4/mdp.py's
_grasp_lift_state helper using the antipodal_grasp_bonus contact-force check.
Since _grasped is latched (ORed, not reset), checking it at the final episode
step is equivalent to checking "was it ever True during the entire episode".

.. code-block:: bash

    /home/saps/IsaacLab/isaaclab.sh -p scripts/_diag_hierarchical_grasp_check.py \
        --checkpoint logs/train/2026-07-09_21-15-02/model_149.pt --num_envs 64
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Check if hierarchical policy actually achieved grasps.")
parser.add_argument("--checkpoint", type=str, default="logs/train/2026-07-09_21-15-02/model_149.pt",
                    help="Path to the rsl_rl checkpoint (.pt) to load.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel envs to trace.")
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
from tasks.ar4.pickplace_hierarchical_env_cfg import Ar4PickPlaceHierarchicalEnvCfg  # noqa: E402


def main() -> None:
    env_cfg = Ar4PickPlaceHierarchicalEnvCfg()
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

    print("=" * 80)
    print(f"Checkpoint: {args_cli.checkpoint}")
    print(f"Num envs: {args_cli.num_envs}, Max episode length: {max_episode_length} steps")
    print("=" * 80)
    rd0 = reach_dist()
    print(f"step 0 (reset): reach_dist per env = {[round(x, 4) for x in rd0.cpu().tolist()]}")

    with torch.inference_mode():
        for step_idx in range(1, max_episode_length + 1):
            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)

            if step_idx % 25 == 0:
                rd = reach_dist()
                grasped = env.unwrapped._grasped.cpu().tolist() if hasattr(env.unwrapped, '_grasped') else None
                lifted = env.unwrapped._lifted.cpu().tolist() if hasattr(env.unwrapped, '_lifted') else None
                print(
                    f"step {step_idx:4d} ({step_idx * env.unwrapped.step_dt:6.2f}s): "
                    f"reach_dist = {[round(x, 4) for x in rd.cpu().tolist()]} | "
                    f"grasped={grasped} lifted={lifted}"
                )

    # Final check: how many envs ever achieved _grasped == True?
    grasped_final = env.unwrapped._grasped.cpu() if hasattr(env.unwrapped, '_grasped') else torch.zeros(args_cli.num_envs, dtype=torch.bool)
    lifted_final = env.unwrapped._lifted.cpu() if hasattr(env.unwrapped, '_lifted') else torch.zeros(args_cli.num_envs, dtype=torch.bool)

    num_grasped = grasped_final.sum().item()
    num_lifted = lifted_final.sum().item()
    grasped_env_ids = torch.where(grasped_final)[0].tolist()
    lifted_env_ids = torch.where(lifted_final)[0].tolist()

    env.close()

    print()
    print("=" * 80)
    print("FINAL SUMMARY:")
    print(f"  Envs with _grasped == True at final step: {num_grasped}/{args_cli.num_envs} (env indices: {grasped_env_ids})")
    print(f"  Envs with _lifted == True at final step: {num_lifted}/{args_cli.num_envs} (env indices: {lifted_env_ids})")
    print("  Note: _grasped and _lifted are latched (ORed) within each episode, so final=True")
    print("        is equivalent to 'was True at any point during the entire episode'.")
    print("=" * 80)
    print("[DONE]")


if __name__ == "__main__":
    main()
    simulation_app.close()
