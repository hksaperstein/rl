"""Harvests real (physically-simulated) arm joint_pos/joint_vel states from
a rollout of the trained Experiment 25 touch-goal reach policy, at the
point EACH ENV's OWN reach_dist first drops below --threshold.

Revision note: an earlier version of this script snapshotted a single fixed
step (default 40) across all envs. A diagnostic trace
(scripts/_diag_touchgoal_reach_trace.py) found this checkpoint's
convergence timing varies hugely per env - only 2/8 traced envs reached
their reach_dist minimum near step ~23, the rest only got there between
step 176 and 786 of the 1000-step/20s episode - so a fixed-step snapshot
mostly captured envs that hadn't gotten anywhere near the cube yet (observed
reach_dist ~0.32m at step 40, not the ~0.024m this checkpoint is capable
of). This version captures each env independently, the first time ITS OWN
reach_dist crosses below threshold, discarding envs that never cross within
--max_steps, so every saved state is a genuine, individually-verified
post-reach state - not a fixed-step guess.

Saves the bank to a .pt file used by reset_arm_from_handoff_bank
(tasks/ar4/mdp.py) to reset a hierarchical grasp-only sub-policy's episodes
from a real reach-converged state, instead of a scripted one-shot IK
teleport (Experiment 14) or a fixed home pose.

Research-thread feasibility prototype, not a numbered production experiment
yet.

.. code-block:: bash

    DISPLAY=:1 flock /tmp/rl_isaac_sim.lock -c "/home/saps/IsaacLab/isaaclab.sh -p scripts/harvest_reach_handoff_states.py \\
        --checkpoint logs/train/2026-07-09_13-07-51/model_1499.pt --num_envs 64 --threshold 0.05 --max_steps 900 \\
        --out logs/reach_handoff_states.pt --headless"
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Harvest real reach-converged arm states from a trained touch-goal checkpoint.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the rsl_rl checkpoint (.pt) to load.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel envs to roll out and harvest from.")
parser.add_argument(
    "--threshold", type=float, default=0.05, help="Per-env reach_dist threshold (meters) that triggers capture."
)
parser.add_argument(
    "--max_steps", type=int, default=900, help="Max steps to roll out (episode is 1000 steps/20s) before giving up."
)
parser.add_argument("--out", type=str, default="logs/reach_handoff_states.pt", help="Output path for the saved bank.")
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
from isaaclab.managers import SceneEntityCfg  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from tasks.ar4.agents.rsl_rl_ppo_cfg import Ar4PickPlacePPORunnerCfg  # noqa: E402
from tasks.ar4.pickplace_touchgoal_env_cfg import Ar4PickPlaceTouchGoalEnvCfg  # noqa: E402
from tasks.ar4.robot_cfg import ARM_JOINT_NAMES  # noqa: E402


def main() -> None:
    env_cfg = Ar4PickPlaceTouchGoalEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    agent_cfg = Ar4PickPlacePPORunnerCfg()
    agent_cfg.device = args_cli.device

    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs = env.get_observations()

    robot = env.unwrapped.scene["robot"]
    cube = env.unwrapped.scene["cube"]
    ee_frame = env.unwrapped.scene["ee_frame"]

    robot_entity_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)
    robot_entity_cfg.resolve(env.unwrapped.scene)
    arm_joint_ids = robot_entity_cfg.joint_ids

    num_envs = args_cli.num_envs
    device = env.unwrapped.device

    captured = torch.zeros(num_envs, dtype=torch.bool, device=device)
    captured_joint_pos = torch.zeros(num_envs, len(arm_joint_ids), device=device)
    captured_joint_vel = torch.zeros(num_envs, len(arm_joint_ids), device=device)
    captured_reach_dist = torch.zeros(num_envs, device=device)
    captured_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)

    def reach_dist():
        ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
        return torch.norm(ee_pos_w - cube.data.root_pos_w, dim=-1)

    print("=" * 70)
    print(
        f"Checkpoint: {args_cli.checkpoint}, {num_envs} envs, per-env adaptive capture at "
        f"reach_dist < {args_cli.threshold}, max {args_cli.max_steps} steps"
    )

    with torch.inference_mode():
        for step in range(args_cli.max_steps):
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

            rd = reach_dist()
            newly = (~captured) & (rd < args_cli.threshold)
            if newly.any():
                idx = newly.nonzero(as_tuple=True)[0]
                captured_joint_pos[idx] = robot.data.joint_pos[idx][:, arm_joint_ids]
                captured_joint_vel[idx] = robot.data.joint_vel[idx][:, arm_joint_ids]
                captured_reach_dist[idx] = rd[idx]
                captured_step[idx] = step
                captured[idx] = True
                print(f"step {step}: captured {int(newly.sum())} new envs (total captured {int(captured.sum())}/{num_envs})")

            if captured.all():
                print(f"All {num_envs} envs captured by step {step}.")
                break

    n_captured = int(captured.sum().item())
    print(f"Captured {n_captured}/{num_envs} envs within {args_cli.max_steps} steps.")
    if n_captured < num_envs:
        uncaptured_idx = (~captured).nonzero(as_tuple=True)[0].cpu().tolist()
        print(f"Envs that never crossed threshold: {uncaptured_idx}")

    if n_captured == 0:
        sys.exit("No envs captured a sub-threshold reach_dist within max_steps - bank would be empty, aborting save.")

    keep_idx = captured.nonzero(as_tuple=True)[0]
    joint_pos = captured_joint_pos[keep_idx].cpu()
    joint_vel = captured_joint_vel[keep_idx].cpu()
    reach_dist_saved = captured_reach_dist[keep_idx].cpu()
    step_saved = captured_step[keep_idx].cpu()

    print(
        f"Saved reach_dist stats: min={reach_dist_saved.min():.4f} max={reach_dist_saved.max():.4f} "
        f"mean={reach_dist_saved.mean():.4f}"
    )
    print(
        f"Captured at steps: min={step_saved.min().item()} max={step_saved.max().item()} "
        f"mean={step_saved.float().mean():.1f}"
    )
    print(f"joint_pos shape: {tuple(joint_pos.shape)}, joint_vel shape: {tuple(joint_vel.shape)}")

    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    torch.save(
        {"joint_pos": joint_pos, "joint_vel": joint_vel, "reach_dist": reach_dist_saved, "step": step_saved},
        args_cli.out,
    )
    print(f"Saved {joint_pos.shape[0]} handoff states to {args_cli.out}")
    print("[DONE]")


if __name__ == "__main__":
    main()
    simulation_app.close()
