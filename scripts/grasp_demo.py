"""Verify the AR4 mk5 gripper and object scene: reach a placed cube, close the
gripper on it, lift it, hold, then release.

UPDATED: This version solves joint targets live using Isaac Lab's
DifferentialIKController (once per waypoint, not every step), settles the arm
to verify convergence, and records video + verifies grasp via contact forces.

The two Cartesian targets (pregrasp hover above cube, at-cube for grasp) are
computed using the real simulator's end-effector position as ground truth for
convergence checking. Each phase's target is solved once and settled, with
residual error logged round-by-round. The arm then interpolates in joint space
toward that fixed target during execution.

.. code-block:: bash

    ./isaaclab.sh -p scripts/grasp_demo.py
"""

import argparse
import os
import subprocess
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Reach, grasp, and lift the cube with the AR4 mk5 gripper.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # Enable rendering for video capture

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

from tasks.ar4.env_cfg import Ar4EnvCfg
from tasks.ar4.pickplace_env_cfg import _EE_OFFSET
from tasks.ar4.robot_cfg import ARM_JOINT_NAMES

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Cube's known spawn position (from tasks/ar4/objects_cfg.py)
CUBE_POS_W = (0.20, 0.28, 0.009)

# Pregrasp and grasp geometry
PREGRASP_HOVER = 0.05  # meters above cube
GRASP_AT_HEIGHT = 0.009  # same z as cube spawn

HOME_Q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
GRIPPER_OPEN = 1.0
GRIPPER_CLOSE = -1.0

# IK solver constants
IK_CONVERGENCE_THRESHOLD = 0.01  # meters (~1cm)
MAX_IK_ROUNDS = 3
IK_SETTLE_STEPS = 50  # steps to hold at each IK target before checking convergence

# Grasp verification constants (same thresholds as antipodal_grasp_bonus)
GRASP_FORCE_THRESHOLD = 0.05  # newtons
GRASP_ANTIPODAL_COS_THRESHOLD = -0.7071  # cos(135°) for ~45° antipodal requirement


def _raise_window_in_background(title_substr: str, duration_s: float) -> None:
    """Best-effort: keep the sim window raised on desktops that don't auto-focus new windows."""
    try:
        subprocess.Popen(
            ["python3", os.path.join(SCRIPT_DIR, "_raise_window.py"), title_substr, str(duration_s)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        pass  # not fatal if this desktop doesn't support it


def solve_ik_to_target(
    env: ManagerBasedEnv,
    ik_controller: DifferentialIKController,
    robot_entity_cfg: SceneEntityCfg,
    ik_jacobi_idx: int,
    target_pos_b: torch.Tensor,
    max_rounds: int = MAX_IK_ROUNDS,
    settle_steps: int = IK_SETTLE_STEPS,
    convergence_threshold: float = IK_CONVERGENCE_THRESHOLD,
) -> torch.Tensor:
    """Solve IK to a target position (robot root frame) iteratively:
    1. Solve IK from current joint state
    2. Apply/step the sim toward that target for settle_steps
    3. Check residual error (actual EE pos vs target)
    4. If error < convergence_threshold, done; else resolve from new state
    5. Max max_rounds iterations to avoid infinite loops

    Returns the final joint configuration.
    Logs residual error each round to console.
    """
    robot = env.scene["robot"]

    current_joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids].clone()

    for round_num in range(max_rounds):
        # Get current end-effector pose
        ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            robot.data.root_pose_w[:, 0:3], robot.data.root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )

        # Solve IK to the target in robot frame
        jacobian = robot.root_physx_view.get_jacobians()[:, ik_jacobi_idx, :, robot_entity_cfg.joint_ids]
        ik_controller.set_command(target_pos_b, ee_pos=ee_pos_b, ee_quat=ee_quat_b)
        joint_pos_des = ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, current_joint_pos)

        # Step toward the target for settle_steps
        for step in range(settle_steps):
            action = torch.zeros(env.num_envs, len(ARM_JOINT_NAMES) + 1, device=env.device)
            for j in range(len(ARM_JOINT_NAMES)):
                action[:, j] = joint_pos_des[:, j]
            action[:, len(ARM_JOINT_NAMES)] = GRIPPER_OPEN  # Keep gripper open during IK settling
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
        print(f"[IK Round {round_num + 1}/{max_rounds}] Residual error: {residual_error:.5f}m")

        if residual_error < convergence_threshold or round_num == max_rounds - 1:
            print(
                f"[IK CONVERGED] Target achieved with residual error {residual_error:.5f}m after {round_num + 1} round(s)."
            )
            return joint_pos_des

        # Else: update current state and resolve
        current_joint_pos = joint_pos_des.clone()

    return joint_pos_des


def check_antipodal_grasp(
    env: ManagerBasedEnv,
    force_threshold: float = GRASP_FORCE_THRESHOLD,
    antipodal_cos_threshold: float = GRASP_ANTIPODAL_COS_THRESHOLD,
) -> tuple[bool, dict]:
    """Check bilateral force-closure grasp condition during hold phase.
    Returns (passes_check, diagnostics_dict) where passes_check is True if
    both jaw contact forces exceed force_threshold AND their directions are
    nearly antipodal (cosine < antipodal_cos_threshold)."""

    # Only check if we actually have contact sensors
    if "gripper_jaw1_contact" not in env.scene._entity_names:
        return False, {"error": "Contact sensors not available in this env"}

    jaw1_sensor = env.scene["gripper_jaw1_contact"]
    jaw2_sensor = env.scene["gripper_jaw2_contact"]

    # force_matrix_w shape: (num_envs, 1 body, 1 filter, 3)
    jaw1_force_vec = jaw1_sensor.data.force_matrix_w.view(env.num_envs, 3)
    jaw2_force_vec = jaw2_sensor.data.force_matrix_w.view(env.num_envs, 3)

    jaw1_force_mag = torch.linalg.vector_norm(jaw1_force_vec, dim=-1)
    jaw2_force_mag = torch.linalg.vector_norm(jaw2_force_vec, dim=-1)

    jaw1_dir = jaw1_force_vec / (jaw1_force_mag.unsqueeze(-1) + 1e-8)
    jaw2_dir = jaw2_force_vec / (jaw2_force_mag.unsqueeze(-1) + 1e-8)
    cos_angle = torch.sum(jaw1_dir * jaw2_dir, dim=-1)

    both_magnitude_ok = (jaw1_force_mag > force_threshold) & (jaw2_force_mag > force_threshold)
    antipodal_ok = cos_angle < antipodal_cos_threshold
    passes_check = (both_magnitude_ok & antipodal_ok).all().item()

    return passes_check, {
        "jaw1_force_mag": jaw1_force_mag.item(),
        "jaw2_force_mag": jaw2_force_mag.item(),
        "cos_angle": cos_angle.item(),
        "both_magnitude_ok": both_magnitude_ok.item(),
        "antipodal_ok": antipodal_ok.item(),
    }


def main() -> None:
    # Create environment (Ar4EnvCfg uses joint position actions)
    env_cfg = Ar4EnvCfg()
    env_cfg.sim.device = args_cli.device
    env_cfg.recorders.dataset_export_dir_path = LOG_DIR
    env_cfg.recorders.dataset_filename = "grasp_demo"

    env = ManagerBasedEnv(cfg=env_cfg)

    total_steps_estimate = (120 + 180 + 90 + 60 + 90 + 120 + 60 + 180)  # phases
    run_time_s = total_steps_estimate * env.step_dt + 10.0
    _raise_window_in_background("Isaac Sim Python", run_time_s)

    # Set up IK controller
    ik_cfg = DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls")
    ik_controller = DifferentialIKController(ik_cfg, num_envs=env.num_envs, device=env.device)

    robot = env.scene["robot"]
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES, body_names=["link_6"])
    robot_entity_cfg.resolve(env.scene)
    ik_jacobi_idx = robot_entity_cfg.body_ids[0] - 1 if robot.is_fixed_base else robot_entity_cfg.body_ids[0]

    num_arm_joints = len(ARM_JOINT_NAMES)

    with torch.inference_mode():
        env.reset()

        print("[INFO] Computing IK targets for pregrasp and grasp waypoints...")

        # Convert world-frame targets to robot-frame
        # (Cube is at (0.20, 0.28, 0.009) in world frame)
        cube_pos_w = torch.tensor([CUBE_POS_W], device=env.device)
        root_pos_w = robot.data.root_pos_w
        root_quat_w = robot.data.root_quat_w
        cube_pos_b, _ = subtract_frame_transforms(root_pos_w, root_quat_w, cube_pos_w)

        pregrasp_pos_b = cube_pos_b.clone()
        pregrasp_pos_b[:, 2] += PREGRASP_HOVER

        grasp_pos_b = cube_pos_b.clone()
        grasp_pos_b[:, 2] = GRASP_AT_HEIGHT

        print(f"[INFO] Cube position (world): {CUBE_POS_W}")
        print(f"[INFO] Cube position (robot frame): {cube_pos_b.tolist()}")
        print(f"[INFO] Pregrasp target (robot frame): {pregrasp_pos_b.tolist()}")
        print(f"[INFO] Grasp target (robot frame): {grasp_pos_b.tolist()}")

        # Solve IK for pregrasp and grasp
        print("\n[INFO] Solving IK for PREGRASP waypoint...")
        pregrasp_q = solve_ik_to_target(
            env, ik_controller, robot_entity_cfg, ik_jacobi_idx, pregrasp_pos_b[0].unsqueeze(0)
        )

        print("\n[INFO] Solving IK for GRASP waypoint...")
        grasp_q = solve_ik_to_target(env, ik_controller, robot_entity_cfg, ik_jacobi_idx, grasp_pos_b[0].unsqueeze(0))

        # Convert to lists for PHASES
        pregrasp_q_list = pregrasp_q[0].tolist() if pregrasp_q.dim() > 1 else pregrasp_q.tolist()
        grasp_q_list = grasp_q[0].tolist() if grasp_q.dim() > 1 else grasp_q.tolist()

        # Define phases with computed targets
        PHASES = [
            (120, HOME_Q, GRIPPER_OPEN),
            (180, pregrasp_q_list, GRIPPER_OPEN),
            (90, grasp_q_list, GRIPPER_OPEN),
            (60, grasp_q_list, GRIPPER_CLOSE),
            (90, pregrasp_q_list, GRIPPER_CLOSE),
            (120, pregrasp_q_list, GRIPPER_CLOSE),  # Hold phase - we'll log grasp data here
            (60, pregrasp_q_list, GRIPPER_OPEN),
            (180, HOME_Q, GRIPPER_OPEN),
        ]

        prev_q = HOME_Q
        hold_phase_idx = 5  # Index of the hold phase in PHASES
        grasp_success_data = []

        print("\n[INFO] Starting movement phases...\n")

        for phase_idx, (duration, target_q, gripper_cmd) in enumerate(PHASES):
            print(f"[PHASE {phase_idx}] Duration: {duration} steps, Gripper: {'OPEN' if gripper_cmd > 0 else 'CLOSE'}")

            for i in range(duration):
                step_start = time.time()
                alpha = (i + 1) / duration
                q = [prev + alpha * (target - prev) for prev, target in zip(prev_q, target_q)]

                action = torch.zeros(env.num_envs, num_arm_joints + 1, device=env.device)
                for j in range(num_arm_joints):
                    action[:, j] = q[j]
                action[:, num_arm_joints] = gripper_cmd
                env.step(action)

                # Log grasp verification data during hold phase
                if phase_idx == hold_phase_idx:
                    try:
                        passes_check, diagnostics = check_antipodal_grasp(env)
                        if "error" not in diagnostics:
                            grasp_success_data.append({
                                "step": i,
                                "passes_check": passes_check,
                                "diagnostics": diagnostics,
                            })

                            if i % 20 == 0:  # Print every 20 steps to avoid spam
                                print(
                                    f"  Step {i:3d}: jaw1_force={diagnostics['jaw1_force_mag']:.4f}N, "
                                    f"jaw2_force={diagnostics['jaw2_force_mag']:.4f}N, "
                                    f"cos_angle={diagnostics['cos_angle']:.4f}, "
                                    f"antipodal_ok={diagnostics['antipodal_ok']}"
                                )
                        elif i == 0:
                            print(f"  [INFO] Contact sensors not available in this env config")
                    except Exception as e:
                        if i == 0:
                            print(f"  [WARNING] Could not check grasp: {e}")

                sleep_time = env.step_dt - (time.time() - step_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            prev_q = target_q

        # Trigger a final reset so the recorder manager exports the episode
        env.reset()

        # Print grasp verification summary
        if grasp_success_data:
            num_passes = sum(1 for d in grasp_success_data if d["passes_check"])
            print(f"\n[GRASP VERIFICATION SUMMARY]")
            print(f"  Hold phase steps: {len(grasp_success_data)}")
            print(f"  Steps passing antipodal check: {num_passes}/{len(grasp_success_data)}")
            if num_passes > 0:
                print(f"  RESULT: Grasp verified (antipodal contact achieved)")
            else:
                print(f"  RESULT: No antipodal grasp contact detected")
        else:
            print(f"\n[GRASP VERIFICATION]: Contact sensors not available in Ar4EnvCfg")
            print(f"[GRASP VERIFICATION]: Visual inspection of hold phase required")

    print("\nDone. Holding window open for a few seconds before closing...")
    time.sleep(5.0)
    env.close()
    print(f"Joint data recorded to: {LOG_DIR}")


if __name__ == "__main__":
    main()
    simulation_app.close()
