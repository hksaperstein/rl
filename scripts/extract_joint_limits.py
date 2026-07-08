"""Extract AR4 joint limits and compare against stuck configuration.

This is a fast diagnostic that doesn't run IK solving.
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Extract AR4 joint limits.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # Required by Ar4GraspVerifyEnvCfg

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

from tasks.ar4.grasp_verify_env_cfg import Ar4GraspVerifyEnvCfg
from tasks.ar4.robot_cfg import ARM_JOINT_NAMES

# The stuck configuration from grasp_demo.py's single-restart run
STUCK_CONFIG = torch.tensor([-0.4140, 1.3328, -0.1425, -0.9181, 1.6214, -0.3007])


def main() -> None:
    # Create minimal environment
    env_cfg = Ar4GraspVerifyEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device

    print("[INFO] Creating environment...")
    env = ManagerBasedEnv(cfg=env_cfg)

    robot = env.scene["robot"]
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES, body_names=["link_6"])
    robot_entity_cfg.resolve(env.scene)

    print("\n" + "=" * 90)
    print("AR4 JOINT LIMITS AND STUCK CONFIGURATION ANALYSIS")
    print("=" * 90)

    # Get joint limits from the robot's data
    joint_pos_limits = robot.data.joint_pos_limits[:, robot_entity_cfg.joint_ids]

    print(f"\nJoint limits shape: {joint_pos_limits.shape}")
    print(f"\nStuck configuration from grasp_demo.py: {STUCK_CONFIG.tolist()}")

    print("\nPer-joint analysis:")
    print(f"{'Joint':<10} {'Min (rad)':<15} {'Max (rad)':<15} {'Range (rad)':<15} {'Stuck Val':<15} {'Min Margin':<15} {'Min Margin %':<15}")
    print("-" * 100)

    all_margins_pct = []

    for j, joint_name in enumerate(ARM_JOINT_NAMES):
        min_limit = joint_pos_limits[0, j, 0].item()
        max_limit = joint_pos_limits[0, j, 1].item()
        range_val = max_limit - min_limit
        stuck_val = STUCK_CONFIG[j].item()

        # Margin to each limit
        margin_to_min = stuck_val - min_limit
        margin_to_max = max_limit - stuck_val
        min_margin = min(margin_to_min, margin_to_max)
        min_margin_pct = (min_margin / range_val * 100) if range_val > 0 else 0

        all_margins_pct.append(min_margin_pct)

        print(f"{joint_name:<10} {min_limit:<15.5f} {max_limit:<15.5f} {range_val:<15.5f} {stuck_val:<15.5f} {min_margin:<15.5f} {min_margin_pct:<15.1f}%")

    print("\n" + "=" * 90)
    print("ANALYSIS")
    print("=" * 90)

    min_margin_overall = min(all_margins_pct)

    print(f"\nStuck configuration margins from nearest joint limit:")
    print(f"  Minimum margin across all joints: {min_margin_overall:.1f}%")

    if min_margin_overall < 5:
        print(f"\n✗ CRITICAL: At least one joint is within {min_margin_overall:.1f}% of its limit!")
        print(f"  This is strong evidence of a joint-limit-driven obstruction.")
        print(f"  The stuck configuration cannot move further in that direction without")
        print(f"  hitting a hard joint limit, making the target position unreachable from HOME_Q.")
    elif min_margin_overall < 10:
        print(f"\n⚠ WARNING: At least one joint is within {min_margin_overall:.1f}% of its limit!")
        print(f"  Joint limits may be contributing to the reachability problem.")
    else:
        print(f"\nNone of the joints are critically close to their limits (min margin: {min_margin_overall:.1f}%).")
        print(f"If this configuration cannot reach the target, it's likely a Jacobian singularity")
        print(f"or other workspace geometry issue, not a joint-limit constraint.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
