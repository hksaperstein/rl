"""Test 3-DOF position-only IK reaching: restrict IK solver to first 3 arm joints
(base yaw, shoulder, elbow) solving for end-effector POSITION only, while holding
the wrist (joints 4-6) "limp" — each step, command them to their own current
actual joint position so they don't fight the solve.

This is an exactly-determined 3-joint/3D-position system instead of the full 6-DOF
system that struggled with a ~0.33m residual in grasp_demo.py. The wrist being limp
(commanded to stay at its own live position each step) is the closest practical proxy
for passive/uncontrolled joints achievable within this env's JointPositionActionCfg
action space; true zero-effort dynamics would require changing the actuator model
itself (out of scope).

Target: cube at world position (0.20, 0.28, 0.009).

.. code-block:: bash

    ./isaaclab.sh -p scripts/grasp_demo_3dof.py
"""

import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="3-DOF position-only IK reach test.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # Required by Ar4GraspVerifyEnvCfg which includes a perception camera

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

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
DIAG_PATH = os.path.join(LOG_DIR, "grasp_demo_3dof_diag.txt")

# Open diagnostics file for writing
diag_file = open(DIAG_PATH, "w")

# Cube's known spawn position (from tasks/ar4/objects_cfg.py)
CUBE_POS_W = (0.20, 0.28, 0.009)

# Grasp geometry
GRASP_AT_HEIGHT = 0.009  # same z as cube spawn

HOME_Q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
GRIPPER_OPEN = 1.0

# IK solver constants
IK_CONVERGENCE_THRESHOLD = 0.01  # meters (~1cm)
IK_STEP_MAX = 0.05  # meters - bounded per-round Cartesian step
MAX_IK_ROUNDS = 40  # sized so 40 * IK_STEP_MAX = 2.0m of total reach
IK_SETTLE_STEPS = 20  # steps to hold at each per-round target before checking convergence


def _log(msg: str) -> None:
    """Print to both stdout and diagnostic file."""
    print(msg, flush=True)
    diag_file.write(msg + "\n")
    diag_file.flush()


def solve_ik_to_target_3dof(
    env: ManagerBasedEnv,
    ik_controller: DifferentialIKController,
    robot_entity_cfg: SceneEntityCfg,
    ik_jacobi_idx: int,
    target_pos_b: torch.Tensor,
    wrist_joint_ids: list,
    max_rounds: int = MAX_IK_ROUNDS,
    settle_steps: int = IK_SETTLE_STEPS,
    convergence_threshold: float = IK_CONVERGENCE_THRESHOLD,
) -> torch.Tensor:
    """Solve 3-DOF position-only IK to a target position (robot root frame):
    1. Solve IK from current joint state (3-DOF only: joints 1-3)
    2. Apply/step the sim toward that target for settle_steps, holding wrist
       joints limp (commanded to their current position each step)
    3. Check residual error (actual EE pos vs target)
    4. If error < convergence_threshold, done; else resolve from new state
    5. Max max_rounds iterations to avoid infinite loops

    The wrist_joint_ids are held at their current live position each step,
    making them passive/uncontrolled within the position-control action space.

    Returns the final joint configuration for ALL 6 arm joints
    (3-DOF solution + last wrist position).
    Logs residual error each round to console.
    """
    robot = env.scene["robot"]

    for round_num in range(max_rounds):
        # Re-read the ACTUAL current joint state fresh every round
        current_joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids].clone()
        current_wrist_pos = robot.data.joint_pos[:, wrist_joint_ids].clone()

        # Get current end-effector pose
        ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            robot.data.root_pose_w[:, 0:3], robot.data.root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )

        # Bound the per-round Cartesian step
        direction = target_pos_b - ee_pos_b
        dist = torch.norm(direction, dim=-1, keepdim=True)
        step_target_b = ee_pos_b + direction / (dist + 1e-8) * torch.clamp(dist, max=IK_STEP_MAX)

        # Solve IK to the bounded step target in robot frame (3-DOF only)
        jacobian = robot.root_physx_view.get_jacobians()[:, ik_jacobi_idx, :, robot_entity_cfg.joint_ids]
        ik_controller.set_command(step_target_b, ee_pos=ee_pos_b, ee_quat=ee_quat_b)
        joint_pos_des_3dof = ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, current_joint_pos)

        # Step toward the target for settle_steps, holding wrist limp
        for step in range(settle_steps):
            # Read fresh wrist position each step (limp: stays at current value)
            fresh_wrist_pos = robot.data.joint_pos[:, wrist_joint_ids].clone()

            action = torch.zeros(env.num_envs, len(ARM_JOINT_NAMES) + 1, device=env.device)
            # Joints 1-3: command the IK solution
            for j in range(3):
                action[:, j] = joint_pos_des_3dof[:, j]
            # Joints 4-6: command to their own current position (limp)
            for j, wrist_idx in enumerate(wrist_joint_ids):
                # Map to action vector indices 3, 4, 5
                action[:, 3 + j] = fresh_wrist_pos[:, j]
            # Gripper: keep open
            action[:, len(ARM_JOINT_NAMES)] = GRIPPER_OPEN
            env.step(action)

        # Check convergence: read EE position via forward kinematics
        ee_pose_w_now = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
        ee_pos_b_now, _ = subtract_frame_transforms(
            robot.data.root_pose_w[:, 0:3],
            robot.data.root_pose_w[:, 3:7],
            ee_pose_w_now[:, 0:3],
            ee_pose_w_now[:, 3:7],
        )

        residual_error = torch.norm(ee_pos_b_now - target_pos_b, dim=-1).item()
        _log(f"[IK Round {round_num + 1}/{max_rounds}] Residual error: {residual_error:.5f}m")

        if residual_error < convergence_threshold or round_num == max_rounds - 1:
            _log(f"[IK CONVERGED] Target achieved with residual error {residual_error:.5f}m after {round_num + 1} round(s).")
            # Return full 6-DOF config: 3-DOF solution + final wrist position
            final_wrist_pos = robot.data.joint_pos[:, wrist_joint_ids].clone()
            final_q = torch.cat([joint_pos_des_3dof[:, :3], final_wrist_pos], dim=1)
            return final_q

    # Fallback: return what we have
    final_wrist_pos = robot.data.joint_pos[:, wrist_joint_ids].clone()
    final_q = torch.cat([joint_pos_des_3dof[:, :3], final_wrist_pos], dim=1)
    return final_q


def main() -> None:
    # Create environment
    env_cfg = Ar4GraspVerifyEnvCfg()
    env_cfg.sim.device = args_cli.device
    env_cfg.recorders.dataset_export_dir_path = LOG_DIR

    env = ManagerBasedEnv(cfg=env_cfg)

    # Ensure logs directory exists
    os.makedirs(LOG_DIR, exist_ok=True)

    # Set up IK controller (position-only, same as grasp_demo.py)
    ik_cfg = DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls")
    ik_controller = DifferentialIKController(ik_cfg, num_envs=env.num_envs, device=env.device)

    robot = env.scene["robot"]

    # 3-DOF IK: only first 3 arm joints
    robot_entity_cfg_3dof = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES[:3], body_names=["link_6"])
    robot_entity_cfg_3dof.resolve(env.scene)
    ik_jacobi_idx = robot_entity_cfg_3dof.body_ids[0] - 1 if robot.is_fixed_base else robot_entity_cfg_3dof.body_ids[0]

    # Wrist joint IDs (joints 4-6)
    robot_entity_cfg_all = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES, body_names=["link_6"])
    robot_entity_cfg_all.resolve(env.scene)
    wrist_joint_ids = list(robot_entity_cfg_all.joint_ids[3:6])

    _log("[INFO] 3-DOF Position-Only IK Reachability Test")
    _log("=" * 70)
    _log(f"[INFO] Cube position (world): {CUBE_POS_W}")
    _log(f"[INFO] Solving IK with joints 1-3 only (base yaw, shoulder, elbow)")
    _log(f"[INFO] Wrist (joints 4-6) will be held limp (at current position each step)")
    _log(f"[INFO] Convergence threshold: {IK_CONVERGENCE_THRESHOLD}m, Max rounds: {MAX_IK_ROUNDS}")

    # Joint limit reporting for 3-DOF only
    joint_pos_limits_3dof = robot.data.joint_pos_limits[:, robot_entity_cfg_3dof.joint_ids]

    _log("\n[INFO] Joint limits for 3-DOF solve (joints 1-3):")
    _log(f"{'Joint':<12} {'Min (rad)':<15} {'Max (rad)':<15} {'Range (rad)':<15}")
    _log("-" * 57)

    for j, joint_name in enumerate(ARM_JOINT_NAMES[:3]):
        min_limit = joint_pos_limits_3dof[0, j, 0].item()
        max_limit = joint_pos_limits_3dof[0, j, 1].item()
        range_val = max_limit - min_limit
        _log(f"{joint_name:<12} {min_limit:<15.5f} {max_limit:<15.5f} {range_val:<15.5f}")

    with torch.inference_mode():
        env.reset()

        # Convert world-frame target to robot-frame
        cube_pos_w = torch.tensor([CUBE_POS_W], device=env.device)
        root_pos_w = robot.data.root_pos_w
        root_quat_w = robot.data.root_quat_w
        cube_pos_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, cube_pos_w)

        grasp_pos_b = cube_pos_b.clone()
        grasp_pos_b[:, 2] = GRASP_AT_HEIGHT

        _log(f"\n[INFO] Cube position (robot frame): {cube_pos_b.tolist()}")
        _log(f"[INFO] Grasp target (robot frame): {grasp_pos_b.tolist()}")

        _log("\n[INFO] Solving 3-DOF IK for GRASP waypoint...")
        grasp_q = solve_ik_to_target_3dof(
            env,
            ik_controller,
            robot_entity_cfg_3dof,
            ik_jacobi_idx,
            grasp_pos_b[0].unsqueeze(0),
            wrist_joint_ids,
        )

        # Extract and report results
        final_q_list = grasp_q[0].tolist() if grasp_q.dim() > 1 else grasp_q.tolist()

        _log("\n[INFO] FINAL JOINT CONFIGURATION (all 6 joints):")
        _log("-" * 57)
        for j, joint_name in enumerate(ARM_JOINT_NAMES):
            _log(f"{joint_name:<12}: {final_q_list[j]:.6f} rad")

        # Forward-kinematics check: where did we actually end up?
        ee_pose_w_final = robot.data.body_pose_w[:, robot_entity_cfg_3dof.body_ids[0]]
        ee_pos_b_final, _ = subtract_frame_transforms(
            robot.data.root_pose_w[:, 0:3],
            robot.data.root_pose_w[:, 3:7],
            ee_pose_w_final[:, 0:3],
            ee_pose_w_final[:, 3:7],
        )

        final_residual = torch.norm(ee_pos_b_final - grasp_pos_b, dim=-1).item()

        _log("\n[INFO] FINAL RESIDUAL ERROR:")
        _log("-" * 57)
        _log(f"Final residual error: {final_residual:.5f}m ({final_residual*1000:.2f}mm)")

        _log("\n[INFO] COMPARISON TO BASELINE:")
        _log("-" * 57)
        _log(f"6-DOF full-IK baseline (grasp_demo.py): ~0.33000m")
        _log(f"3-DOF position-only result:            {final_residual:.5f}m")

        if final_residual < 0.33:
            improvement_pct = ((0.33 - final_residual) / 0.33) * 100
            _log(f"✓ IMPROVEMENT: {improvement_pct:.1f}% better than 6-DOF baseline")
        elif final_residual == 0.33:
            _log(f"≈ COMPARABLE to 6-DOF baseline")
        else:
            degradation_pct = ((final_residual - 0.33) / 0.33) * 100
            _log(f"✗ WORSE: {degradation_pct:.1f}% worse than 6-DOF baseline")

        # Report 3-DOF joint limits with final values and margins
        _log("\n[INFO] 3-DOF Joint Margins at Final Configuration:")
        _log("-" * 80)
        _log(f"{'Joint':<12} {'Min (rad)':<15} {'Max (rad)':<15} {'Final Val':<15} {'Min Margin':<15}")
        _log("-" * 80)

        for j, joint_name in enumerate(ARM_JOINT_NAMES[:3]):
            min_limit = joint_pos_limits_3dof[0, j, 0].item()
            max_limit = joint_pos_limits_3dof[0, j, 1].item()
            final_val = final_q_list[j]

            margin_to_min = final_val - min_limit
            margin_to_max = max_limit - final_val
            min_margin = min(margin_to_min, margin_to_max)

            _log(f"{joint_name:<12} {min_limit:<15.5f} {max_limit:<15.5f} {final_val:<15.6f} {min_margin:<15.5f}")

    _log("\nTest complete.")
    env.close()
    _log(f"Diagnostics recorded to: {DIAG_PATH}")
    diag_file.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
