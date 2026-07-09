# scripts/graspgoal_freeze_diagnostic.py
"""Root-cause diagnostic for Experiment 26's total-freeze failure.

Runs, in one env build, all five checks from the investigation brief:
  1. Does the env respond to large deterministic actions? (arm moves?)
  2. reach_dist at reset vs REACH_DIST_NORM=0.3 (reach reward dead zone?)
  3. NaN/Inf/absurd obs values (grasp_state, goal_position)?
  4. Does an early-PPO-like random policy (N(0,1) raw actions) ever get
     the EE into the reach reward's active zone / touch the cube?
  5. Action-manager per-term slicing (arm 6 dims vs gripper 1 dim).

Launch:
    PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/graspgoal_freeze_diagnostic.py
"""

import sys

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser(description="Freeze diagnostic for the grasp-goal env cfg.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os  # noqa: E402

import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from tasks.ar4.pickplace_graspgoal_env_cfg import (  # noqa: E402
    Ar4PickPlaceGraspGoalEnvCfg,
    REACH_DIST_NORM,
)


def reach_dist(env):
    cube = env.scene["cube"]
    ee = env.scene["ee_frame"].data.target_pos_w[:, 0, :]
    return torch.norm(ee - cube.data.root_pos_w, dim=-1)


def main() -> None:
    env_cfg = Ar4PickPlaceGraspGoalEnvCfg()
    N = 8
    env_cfg.scene.num_envs = N
    env_cfg.sim.device = args_cli.device
    env = ManagerBasedRLEnv(cfg=env_cfg)
    dev = env.device

    print("=" * 70)
    print("CHECK 5: action-manager per-term slicing")
    print("=" * 70)
    am = env.action_manager
    print(f"total_action_dim = {am.total_action_dim}")
    for name, term in am._terms.items():
        print(f"  term '{name}': action_dim={term.action_dim}")
    print(f"action tensor shape: {am.action.shape}")

    with torch.inference_mode():
        obs, _ = env.reset()
        p = obs["policy"]
        print()
        print("=" * 70)
        print("CHECK 3: observation sanity at reset")
        print("=" * 70)
        print(f"obs['policy'] shape: {p.shape}")
        print(f"  any NaN: {torch.isnan(p).any().item()}, any Inf: {torch.isinf(p).any().item()}")
        print(f"  min={p.min().item():.4f}, max={p.max().item():.4f}")
        # layout: joint_pos(8)+joint_vel(8)+cube_pos(3)+goal_pos(3)+grasp_state(2)+actions(7)
        print(f"  joint_pos[0]   = {p[0, 0:8].cpu().tolist()}")
        print(f"  cube_pos[0]    = {p[0, 16:19].cpu().tolist()}")
        print(f"  goal_pos[0]    = {p[0, 19:22].cpu().tolist()}")
        print(f"  grasp_state[0] = {p[0, 22:24].cpu().tolist()}  (expect [0,0])")

        print()
        print("=" * 70)
        print("CHECK 2: reach_dist at reset vs REACH_DIST_NORM")
        print("=" * 70)
        rd = reach_dist(env)
        print(f"REACH_DIST_NORM = {REACH_DIST_NORM}")
        print(f"reach_dist per env at reset: {[round(x, 4) for x in rd.cpu().tolist()]}")
        print(f"reach_dist mean = {rd.mean().item():.4f}")
        rp = torch.clamp(1.0 - rd / REACH_DIST_NORM, min=0.0, max=1.0)
        print(f"reach_progress at reset (clamped): {[round(x, 5) for x in rp.cpu().tolist()]}")
        cube = env.scene["cube"]
        ee = env.scene["ee_frame"].data.target_pos_w[:, 0, :]
        print(f"cube_pos_w[0] = {cube.data.root_pos_w[0].cpu().tolist()}")
        print(f"ee_pos_w[0]   = {ee[0].cpu().tolist()}")

        # ---- CHECK 1: large deterministic actions ----
        print()
        print("=" * 70)
        print("CHECK 1: env response to large deterministic actions")
        print("=" * 70)
        adim = am.total_action_dim
        # try several distinct arm-joint directions of magnitude ~1.0
        directions = [
            [1, 1, 1, 0, 0, 0],
            [-1, -1, -1, 0, 0, 0],
            [0, 1, -1, 1, 0, 0],
            [1, 0, 1, 0, 1, 0],
        ]
        for d in directions:
            env.reset()
            jp0 = env.scene["robot"].data.joint_pos.clone()
            ee0 = env.scene["ee_frame"].data.target_pos_w[:, 0, :].clone()
            rd0 = reach_dist(env)
            act = torch.zeros(N, adim, device=dev)
            for j in range(6):
                act[:, j] = d[j] * 1.0
            for _ in range(40):
                env.step(act)
            jp1 = env.scene["robot"].data.joint_pos
            ee1 = env.scene["ee_frame"].data.target_pos_w[:, 0, :]
            rd1 = reach_dist(env)
            djp = (jp1[0, 0:6] - jp0[0, 0:6]).abs()
            dee = torch.norm(ee1[0] - ee0[0]).item()
            print(
                f"dir {d}: max|dJointPos|={djp.max().item():.4f}, "
                f"EE moved {dee:.4f} m, reach_dist {rd0[0].item():.3f}->{rd1[0].item():.3f}"
            )

        # ---- CHECK 4: early-PPO-like random policy ----
        print()
        print("=" * 70)
        print("CHECK 4: random policy (N(0,1) raw actions, matches early PPO std~1.0)")
        print("=" * 70)
        torch.manual_seed(0)
        for trial in range(3):
            env.reset()
            min_rd = reach_dist(env).clone()
            n_steps = 600  # ~ full 30s episode at decimation-level control
            for _ in range(n_steps):
                # arm dims ~ N(0,1); gripper dim ~ N(0,1) too (binary sign)
                act = torch.randn(N, adim, device=dev)
                env.step(act)
                min_rd = torch.minimum(min_rd, reach_dist(env))
            entered = (min_rd < REACH_DIST_NORM)
            touched = (min_rd < 0.02)
            print(
                f"trial {trial}: over {n_steps} steps x {N} envs, "
                f"min reach_dist per env = {[round(x, 4) for x in min_rd.cpu().tolist()]}"
            )
            print(
                f"           envs that entered reach zone (<{REACH_DIST_NORM}): "
                f"{entered.sum().item()}/{N}; envs that ~touched (<0.02): {touched.sum().item()}/{N}"
            )

    env.close()
    print()
    print("[DIAG] done.")


if __name__ == "__main__":
    main()
    simulation_app.close()
