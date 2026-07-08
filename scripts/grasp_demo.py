"""Verify the AR4 mk5 gripper and object scene: reach a placed cube, close the
gripper on it, lift it, hold, then release.

UPDATED: This version solves joint targets live using Isaac Lab's
DifferentialIKController (once per waypoint, not every step), settles the arm
to verify convergence, and records video. Verification is joint-space/
Cartesian convergence only - the gripper is treated as "dumb" (open/closed
command only, no contact-force verification), matching the real AR4
hardware, which has no gripper contact/force sensors either.

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

import imageio
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

from tasks.ar4.grasp_verify_env_cfg import Ar4GraspVerifyEnvCfg
from tasks.ar4.pickplace_env_cfg import _EE_OFFSET
from tasks.ar4.robot_cfg import ARM_JOINT_NAMES

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
VIDEO_PATH = os.path.join(LOG_DIR, "videos", "ar4_grasp_demo.mp4")
DIAG_PATH = os.path.join(LOG_DIR, "grasp_demo_diag.txt")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Open diagnostics file for writing
diag_file = open(DIAG_PATH, "w")

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
IK_STEP_MAX = 0.05  # meters - bounded per-round Cartesian step (matches oracle_rollout.py's IK_PURSUIT_MAX_STEP)
MAX_IK_ROUNDS = 40  # sized so 40 * IK_STEP_MAX = 2.0m of total reach, well over this arm's workspace
IK_SETTLE_STEPS = 20  # steps to hold at each (now-small) per-round target before checking convergence


def _log(msg: str) -> None:
    """Print to both stdout and diagnostic file."""
    print(msg, flush=True)
    diag_file.write(msg + "\n")
    diag_file.flush()


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

    for round_num in range(max_rounds):
        # Re-read the ACTUAL current joint state fresh every round (not the
        # previous round's commanded target) - using a stale commanded value
        # here as the Newton-step baseline, while the Jacobian/ee_pos below
        # are read from the real (possibly-not-fully-settled) sim state,
        # caused the residual to grow round-over-round instead of shrinking
        # in an earlier version of this function.
        current_joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids].clone()

        # Get current end-effector pose
        ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            robot.data.root_pose_w[:, 0:3], robot.data.root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )

        # Bound the per-round Cartesian step fed to the DLS solve. Commanding
        # the FULL remaining distance (which can be tens of cm) breaks the
        # differential-IK linearization's local-validity assumption and
        # produces wildly unrealistic joint deltas - the exact "unbounded IK
        # Cartesian jump" bug already found and fixed in this repo's
        # scripts/oracle_rollout.py (see IK_PURSUIT_MAX_STEP there). Bounding
        # here means this function needs multiple rounds to close a large
        # initial gap, which is fine - MAX_IK_ROUNDS is sized for that.
        direction = target_pos_b - ee_pos_b
        dist = torch.norm(direction, dim=-1, keepdim=True)
        step_target_b = ee_pos_b + direction / (dist + 1e-8) * torch.clamp(dist, max=IK_STEP_MAX)

        # Solve IK to the bounded step target in robot frame
        jacobian = robot.root_physx_view.get_jacobians()[:, ik_jacobi_idx, :, robot_entity_cfg.joint_ids]
        ik_controller.set_command(step_target_b, ee_pos=ee_pos_b, ee_quat=ee_quat_b)
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

    return joint_pos_des


def main() -> None:
    # Create environment (Ar4GraspVerifyEnvCfg uses joint position actions + contact sensors)
    env_cfg = Ar4GraspVerifyEnvCfg()
    env_cfg.sim.device = args_cli.device
    env_cfg.recorders.dataset_export_dir_path = LOG_DIR
    env_cfg.recorders.dataset_filename = "grasp_demo"

    env = ManagerBasedEnv(cfg=env_cfg)

    # Ensure logs/videos directory exists
    os.makedirs(os.path.dirname(VIDEO_PATH), exist_ok=True)

    # Set up video writer
    video_writer = imageio.get_writer(VIDEO_PATH, fps=int(1.0 / env.step_dt), codec="libx264")
    camera = env.scene["perception_camera"]

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

        _log("[INFO] Computing IK targets for pregrasp and grasp waypoints...")

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

        _log(f"[INFO] Cube position (world): {CUBE_POS_W}")
        _log(f"[INFO] Cube position (robot frame): {cube_pos_b.tolist()}")
        _log(f"[INFO] Pregrasp target (robot frame): {pregrasp_pos_b.tolist()}")
        _log(f"[INFO] Grasp target (robot frame): {grasp_pos_b.tolist()}")

        # Solve IK for pregrasp and grasp
        _log("\n[INFO] Solving IK for PREGRASP waypoint...")
        pregrasp_q = solve_ik_to_target(
            env, ik_controller, robot_entity_cfg, ik_jacobi_idx, pregrasp_pos_b[0].unsqueeze(0)
        )

        _log("\n[INFO] Solving IK for GRASP waypoint...")
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
            (120, pregrasp_q_list, GRIPPER_CLOSE),  # Hold phase - cube-height telemetry logged here
            (60, pregrasp_q_list, GRIPPER_OPEN),
            (180, HOME_Q, GRIPPER_OPEN),
        ]

        prev_q = HOME_Q

        _log("\n[INFO] Starting movement phases...\n")

        for phase_idx, (duration, target_q, gripper_cmd) in enumerate(PHASES):
            _log(f"[PHASE {phase_idx}] Duration: {duration} steps, Gripper: {'OPEN' if gripper_cmd > 0 else 'CLOSE'}")

            for i in range(duration):
                step_start = time.time()
                alpha = (i + 1) / duration
                q = [prev + alpha * (target - prev) for prev, target in zip(prev_q, target_q)]

                action = torch.zeros(env.num_envs, num_arm_joints + 1, device=env.device)
                for j in range(num_arm_joints):
                    action[:, j] = q[j]
                action[:, num_arm_joints] = gripper_cmd
                env.step(action)

                # Capture and record frame
                rgb = camera.data.output["rgb"][0].cpu().numpy()
                # Convert from (H, W, 4) RGBA to (H, W, 3) RGB for video
                video_writer.append_data(rgb[:, :, :3].astype("uint8"))

                # Track cube height during the lift/hold/transit phases (indices 4-5:
                # arm should be at PRE_GRASP_Q with gripper closed) as a cheap, purely
                # kinematic (no contact sensor needed) signal of whether the cube is
                # moving with the arm - not a substitute for a real grasp-quality
                # check, just directional evidence for this joint-driving diagnostic.
                if phase_idx in (4, 5) and i % 30 == 0:
                    cube_z = env.scene["cube"].data.root_pos_w[0, 2].item()
                    _log(f"  [PHASE {phase_idx} step {i:3d}] cube height (world z): {cube_z:.4f}m")

                sleep_time = env.step_dt - (time.time() - step_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            # End-of-phase joint convergence check: how close did the arm actually
            # get to this phase's commanded target (the whole point of this script).
            achieved_q = robot.data.joint_pos[0, robot_entity_cfg.joint_ids].tolist()
            joint_error = [abs(a - t) for a, t in zip(achieved_q, target_q)]
            max_joint_error = max(joint_error)
            _log(
                f"  [PHASE {phase_idx} END] max joint error vs. target: {max_joint_error:.5f} rad "
                f"(achieved={['%.4f' % x for x in achieved_q]})"
            )

            prev_q = target_q

        # Trigger a final reset so the recorder manager exports the episode
        env.reset()

    _log("\nDone. Holding window open for a few seconds before closing...")
    time.sleep(5.0)
    video_writer.close()
    env.close()
    _log(f"Joint data recorded to: {LOG_DIR}")
    _log(f"Video recorded to: {VIDEO_PATH}")
    _log(f"Diagnostics recorded to: {DIAG_PATH}")
    diag_file.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
