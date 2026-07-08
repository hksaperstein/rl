"""Multi-restart IK diagnostic: test whether the AR4 grasp target is reachable
from different starting joint configurations.

This script:
1. Creates Ar4GraspVerifyEnvCfg with num_envs=16
2. For each env, writes a DIFFERENT random initial arm joint configuration via
   robot.write_joint_position_to_sim()
3. From each starting pose, runs the same bounded-step DLS iteration from
   grasp_demo.py's solve_ik_to_target targeting the same grasp position
4. Logs the final residual error per env
5. Prints joint limits and compares against the stuck configuration

Usage:
    /home/saps/IsaacLab/isaaclab.sh -p scripts/ik_multistart_diagnostic.py
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Multi-restart IK diagnostic for AR4 grasp target reachability.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # Required by Ar4GraspVerifyEnvCfg (which includes a camera)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

from tasks.ar4.grasp_verify_env_cfg import Ar4GraspVerifyEnvCfg
from tasks.ar4.robot_cfg import ARM_JOINT_NAMES

# IK solver constants - must match grasp_demo.py exactly
IK_CONVERGENCE_THRESHOLD = 0.01  # meters (~1cm)
IK_STEP_MAX = 0.05  # meters - bounded per-round Cartesian step
MAX_IK_ROUNDS = 20  # reduced from 40 for faster execution (20 * IK_STEP_MAX = 1.0m reach still sufficient)
IK_SETTLE_STEPS = 10  # reduced from 20 for faster execution

# Cube spawn position from tasks/ar4/objects_cfg.py
CUBE_POS_W = (0.20, 0.28, 0.009)
GRASP_AT_HEIGHT = 0.009  # same z as cube spawn
GRIPPER_OPEN = 1.0

# The stuck configuration from grasp_demo.py's single-restart run
STUCK_CONFIG = torch.tensor([-0.4140, 1.3328, -0.1425, -0.9181, 1.6214, -0.3007])


def solve_ik_to_target_batched(
    env: ManagerBasedEnv,
    ik_controller: DifferentialIKController,
    robot_entity_cfg: SceneEntityCfg,
    ik_jacobi_idx: int,
    target_pos_b: torch.Tensor,  # shape: (1, 3) or (num_envs, 3)
    max_rounds: int = MAX_IK_ROUNDS,
    settle_steps: int = IK_SETTLE_STEPS,
    convergence_threshold: float = IK_CONVERGENCE_THRESHOLD,
) -> torch.Tensor:
    """Vectorized IK solver: solve for all num_envs envs in parallel.

    Returns:
        residual_errors: (num_envs,) tensor of final residual errors
    """
    robot = env.scene["robot"]
    residuals_per_env = torch.zeros(env.num_envs, device=env.device)

    # If target is shape (1, 3), expand to (num_envs, 3)
    if target_pos_b.shape[0] == 1:
        target_pos_b = target_pos_b.expand(env.num_envs, 3)

    for round_num in range(max_rounds):
        # Read current joint state from ALL envs
        current_joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids].clone()

        # Get current end-effector pose for all envs
        ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            robot.data.root_pose_w[:, 0:3], robot.data.root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )

        # Bound per-round Cartesian step for all envs
        direction = target_pos_b - ee_pos_b
        dist = torch.norm(direction, dim=-1, keepdim=True)
        step_target_b = ee_pos_b + direction / (dist + 1e-8) * torch.clamp(dist, max=IK_STEP_MAX)

        # Solve IK for all envs simultaneously
        jacobian = robot.root_physx_view.get_jacobians()[:, ik_jacobi_idx, :, robot_entity_cfg.joint_ids]
        ik_controller.set_command(step_target_b, ee_pos=ee_pos_b, ee_quat=ee_quat_b)
        joint_pos_des = ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, current_joint_pos)

        # Step toward the target for settle_steps
        for step in range(settle_steps):
            action = torch.zeros(env.num_envs, len(ARM_JOINT_NAMES) + 1, device=env.device)
            for j in range(len(ARM_JOINT_NAMES)):
                action[:, j] = joint_pos_des[:, j]
            action[:, len(ARM_JOINT_NAMES)] = GRIPPER_OPEN  # Keep gripper open
            env.step(action)

        # Check convergence: read EE position via forward kinematics
        ee_pose_w_now = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
        ee_pos_b_now, _ = subtract_frame_transforms(
            robot.data.root_pose_w[:, 0:3],
            robot.data.root_pose_w[:, 3:7],
            ee_pose_w_now[:, 0:3],
            ee_pose_w_now[:, 3:7],
        )

        residual_error = torch.norm(ee_pos_b_now - target_pos_b, dim=-1)

        # Check which envs have converged
        converged = residual_error < convergence_threshold

        if round_num == 0 or (round_num % 10 == 9):
            print(f"[IK Round {round_num + 1}/{max_rounds}] Residuals (meters):")
            for env_idx in range(min(env.num_envs, 8)):  # Print first 8 envs
                status = "CONVERGED" if converged[env_idx].item() else ""
                print(f"  Env {env_idx}: {residual_error[env_idx].item():.5f}m {status}")
            if env.num_envs > 8:
                print(f"  ... ({env.num_envs - 8} more envs)")

        # Store final residuals
        residuals_per_env = residual_error.clone()

        # Early exit if all envs have converged
        if converged.all():
            print(f"[IK CONVERGED] All envs reached target at round {round_num + 1}")
            break

    return residuals_per_env


def print_joint_limits_and_margins(robot, robot_entity_cfg, stuck_config):
    """Print the joint limits and compare against the stuck configuration."""
    print("\n" + "=" * 80)
    print("JOINT LIMITS AND MARGINS")
    print("=" * 80)

    # Get joint limits from the robot's data
    joint_pos_limits = robot.data.joint_pos_limits[:, robot_entity_cfg.joint_ids]

    print(f"Joint limits shape: {joint_pos_limits.shape}")
    print("\nPer-joint analysis:")
    print(f"{'Joint':<10} {'Min (rad)':<12} {'Max (rad)':<12} {'Range (rad)':<12} {'Stuck Val':<12} {'Min Margin':<12} {'Min Margin %':<12}")
    print("-" * 88)

    for j, joint_name in enumerate(ARM_JOINT_NAMES):
        min_limit = joint_pos_limits[0, j, 0].item()
        max_limit = joint_pos_limits[0, j, 1].item()
        range_val = max_limit - min_limit
        stuck_val = stuck_config[j].item()

        # Margin to each limit
        margin_to_min = stuck_val - min_limit
        margin_to_max = max_limit - stuck_val
        min_margin = min(margin_to_min, margin_to_max)
        min_margin_pct = (min_margin / range_val * 100) if range_val > 0 else 0

        print(f"{joint_name:<10} {min_limit:<12.5f} {max_limit:<12.5f} {range_val:<12.5f} {stuck_val:<12.5f} {min_margin:<12.5f} {min_margin_pct:<12.1f}%")


def main() -> None:
    # Create environment with multiple parallel envs
    num_envs = 8  # reduced from 16 for faster execution
    env_cfg = Ar4GraspVerifyEnvCfg()
    env_cfg.scene.num_envs = num_envs
    env_cfg.sim.device = args_cli.device

    env = ManagerBasedEnv(cfg=env_cfg)

    robot = env.scene["robot"]
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES, body_names=["link_6"])
    robot_entity_cfg.resolve(env.scene)
    ik_jacobi_idx = robot_entity_cfg.body_ids[0] - 1 if robot.is_fixed_base else robot_entity_cfg.body_ids[0]

    print(f"\n[INFO] Created environment with {env.num_envs} parallel envs")

    # Print joint limits and margins immediately
    print_joint_limits_and_margins(robot, robot_entity_cfg, STUCK_CONFIG)

    # Set up IK controller
    ik_cfg = DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls")
    ik_controller = DifferentialIKController(ik_cfg, num_envs=env.num_envs, device=env.device)

    with torch.inference_mode():
        env.reset()

        print("\n[INFO] Computing IK target in robot frame...")

        # Convert world-frame cube target to robot-frame
        cube_pos_w = torch.tensor([CUBE_POS_W], device=env.device)
        root_pos_w = robot.data.root_pos_w
        root_quat_w = robot.data.root_quat_w
        cube_pos_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, cube_pos_w)

        grasp_pos_b = cube_pos_b.clone()
        grasp_pos_b[:, 2] = GRASP_AT_HEIGHT

        print(f"[INFO] Cube position (world): {CUBE_POS_W}")
        print(f"[INFO] Grasp target (robot frame): {grasp_pos_b[0].tolist()}")

        # Generate random initial joint configurations for each env
        print(f"\n[INFO] Generating {num_envs} random initial joint configurations...")

        # Get joint position limits for sampling
        joint_pos_limits = robot.data.joint_pos_limits[:, robot_entity_cfg.joint_ids]
        min_limits = joint_pos_limits[0, :, 0]  # shape: (6,)
        max_limits = joint_pos_limits[0, :, 1]  # shape: (6,)

        # Generate random joint positions
        random_joint_positions = torch.zeros(num_envs, 6, device=env.device)
        for j in range(6):
            random_joint_positions[:, j] = torch.rand(num_envs, device=env.device) * (max_limits[j] - min_limits[j]) + min_limits[j]

        # Write each env's random starting configuration to sim
        for env_idx in range(num_envs):
            robot.write_joint_position_to_sim(
                random_joint_positions[env_idx:env_idx+1],
                joint_ids=robot_entity_cfg.joint_ids,
                env_ids=torch.tensor([env_idx], device=env.device)
            )

        print(f"[INFO] Random initial configurations written to sim")
        print(f"First 4 envs' starting configs (radians):")
        for env_idx in range(min(4, num_envs)):
            config_str = " ".join([f"{v:.4f}" for v in random_joint_positions[env_idx].tolist()])
            print(f"  Env {env_idx}: [{config_str}]")

        # Run IK solver from each starting pose
        print("\n[INFO] Running IK solver from diverse starting configurations...\n")
        residuals = solve_ik_to_target_batched(
            env, ik_controller, robot_entity_cfg, ik_jacobi_idx, grasp_pos_b[0].unsqueeze(0)
        )

        # Sort and report results
        print("\n" + "=" * 80)
        print("FINAL RESULTS: IK Residuals Across All Envs (sorted best to worst)")
        print("=" * 80)

        sorted_residuals, sorted_indices = torch.sort(residuals)

        print(f"\nBest (smallest) residual: {sorted_residuals[0].item():.5f}m")
        print(f"Worst (largest) residual: {sorted_residuals[-1].item():.5f}m")
        print(f"Mean residual: {residuals.mean().item():.5f}m")
        print(f"Std dev: {residuals.std().item():.5f}m")

        print(f"\nConverged (< {IK_CONVERGENCE_THRESHOLD}m) count: {(residuals < IK_CONVERGENCE_THRESHOLD).sum().item()} / {num_envs}")
        print(f"Under 0.15m count: {(residuals < 0.15).sum().item()} / {num_envs}")
        print(f"Under 0.20m count: {(residuals < 0.20).sum().item()} / {num_envs}")

        print("\nAll residuals (best to worst):")
        for rank, (residual, env_idx) in enumerate(zip(sorted_residuals, sorted_indices)):
            status = "CONVERGED" if residual < IK_CONVERGENCE_THRESHOLD else ""
            config_str = " ".join([f"{v:.4f}" for v in random_joint_positions[env_idx].tolist()])
            print(f"  {rank+1:2d}. Env {env_idx:2d}: {residual.item():8.5f}m {status:<10} start=[{config_str}]")

        print("\n" + "=" * 80)
        print("INTERPRETATION")
        print("=" * 80)

        best_residual = sorted_residuals[0].item()
        baseline_residual = 0.33  # from grasp_demo.py's single-restart run

        if best_residual < IK_CONVERGENCE_THRESHOLD:
            print(f"✓ SUCCESS: At least one starting config converged to < {IK_CONVERGENCE_THRESHOLD}m!")
            print(f"  => This suggests a LOCAL MINIMUM TRAP (fixable with different starting pose)")
        elif best_residual < baseline_residual * 0.9:  # Significantly better than baseline
            print(f"✓ PARTIAL: Best residual ({best_residual:.5f}m) is meaningfully better than baseline ({baseline_residual:.2f}m)")
            print(f"  => Some starting configs help, but still doesn't fully converge")
            print(f"  => Possible joint-limit obstruction or singularity near target")
        else:
            print(f"✗ CONSISTENT FAILURE: All starting configs plateau at ~{best_residual:.2f}m (similar to baseline {baseline_residual:.2f}m)")
            print(f"  => Strong evidence of GENUINE UNREACHABILITY")
            print(f"  => Either a joint-limit constraint (check margins above) or")
            print(f"     the target pose geometry is genuinely outside the workspace")


if __name__ == "__main__":
    main()
    simulation_app.close()
