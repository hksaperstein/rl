"""3-DOF demo: move the AR4 end-effector around a square path parallel to
the ground plane (fixed height, robot-frame XY square), using only
joint_1/2/3 for position - joints 4-6 and the gripper are held "limp"
(commanded to their own current position / a fixed open state), same
convention as scripts/grasp_demo_v2.py.

Solves each waypoint using closed-form 3-DOF IK derived from the AR4's URDF
kinematics (see docs/superpowers/plans/2026-07-08-square-path-closed-form-ik.md).
joint_1 is a pure base-yaw rotation about the vertical axis, and joints 2-3
form a 2-link planar arm in the rotated plane.

.. code-block:: bash

    PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/square_path_demo.py --headless
"""

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="3-DOF square-path demo, parallel to the ground plane.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import imageio  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import ManagerBasedEnv  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.utils.math import subtract_frame_transforms  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from tasks.ar4.grasp_verify_env_cfg import Ar4GraspVerifyEnvCfg  # noqa: E402
from tasks.ar4.robot_cfg import ARM_JOINT_NAMES  # noqa: E402

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
VIDEO_PATH = os.path.join(LOG_DIR, "videos", "ar4_square_path_demo.mp4")

GRIPPER_OPEN = 1.0
HOME_Q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Closed-form 3-DOF IK constants (derived from ar_macro.xacro URDF)
L2 = 0.305
L3 = 0.263940
SHOULDER_U = 0.06415
SHOULDER_V = 0.16978

# Square path: robot-frame, fixed height (parallel to ground), 4 corners +
# 4 edge midpoints for a smoother traced outline. Shifted geometry to stay
# within joint limits (near edge x=0.25, far edge x=0.41, margin >=3.7deg on q3).
SQUARE_Z = 0.08
SQUARE_POINTS_B = [
    (0.25, -0.08, SQUARE_Z),  # corner 1
    (0.25, 0.00, SQUARE_Z),   # mid 1-2
    (0.25, 0.08, SQUARE_Z),   # corner 2
    (0.33, 0.08, SQUARE_Z),   # mid 2-3
    (0.41, 0.08, SQUARE_Z),   # corner 3
    (0.41, 0.00, SQUARE_Z),   # mid 3-4
    (0.41, -0.08, SQUARE_Z),  # corner 4
    (0.33, -0.08, SQUARE_Z),  # mid 4-1
]

HOLD_STEPS_PER_POINT = 70
IK_SETTLE_STEPS = 100  # Settle time per correction round (gravity droop correction loop handles residuals)
IK_CORRECTION_ROUNDS = 5  # Max fixed-point correction iterations
IK_CORRECTION_TOLERANCE = 0.005  # 5mm tolerance for convergence


def solve_ik3(x, y, z):
    """Closed-form 3-DOF IK (joints 1-3) for the AR4, wrist held at
    q4=q5=q6=0. Derived from ar_macro.xacro joint origins; see
    docs/superpowers/plans/2026-07-08-square-path-closed-form-ik.md."""
    bearing = math.atan2(y, x)
    q1 = -(bearing + math.pi / 2)
    u_t, v_t = math.hypot(x, y), z
    du, dv = u_t - SHOULDER_U, v_t - SHOULDER_V
    d = math.hypot(du, dv)
    cos_delta = max(-1.0, min(1.0, (d**2 - L2**2 - L3**2) / (2 * L2 * L3)))
    delta = -math.acos(cos_delta)
    phi1 = math.atan2(dv, du) - math.atan2(L3 * math.sin(delta), L2 + L3 * math.cos(delta))
    q2 = math.pi / 2 - phi1
    q3 = -math.pi / 2 - delta
    return q1, q2, q3


def solve_waypoint_with_correction(env, robot, robot_entity_cfg, target_xyz, j2_min, j2_max, j3_min, j3_max,
                                    num_arm_joints, settle_steps=IK_SETTLE_STEPS, max_rounds=IK_CORRECTION_ROUNDS,
                                    tol=IK_CORRECTION_TOLERANCE):
    """Solve a waypoint using iterative target-nudging to correct for gravity droop.

    Uses fixed-point iteration: solve IK for a "corrected" target, settle, measure error,
    nudge the corrected target toward where it needs to be, and re-solve (repeat up to max_rounds).
    This compensates for steady-state gravity droop in the PD-controlled arm without using
    Jacobian-based iterative IK (closed-form solve per round, just target adjustment).
    """
    corrected_target = list(target_xyz)
    q_target = None
    residual = None

    for round_idx in range(max_rounds):
        # Solve IK for the corrected target
        q1, q2, q3 = solve_ik3(corrected_target[0], corrected_target[1], corrected_target[2])

        # Clip q2, q3 to joint limits
        q2 = max(j2_min, min(j2_max, q2))
        q3 = max(j3_min, min(j3_max, q3))
        q_target = [q1, q2, q3, 0.0, 0.0, 0.0]

        # Command and settle
        for _ in range(settle_steps):
            action = torch.zeros(env.num_envs, num_arm_joints + 1, device=env.device)
            for j in range(num_arm_joints):
                action[:, j] = q_target[j]
            action[:, num_arm_joints] = GRIPPER_OPEN
            env.step(action)

        # Measure actual EE position and residual
        ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
        ee_pos_b, _ = subtract_frame_transforms(
            robot.data.root_pose_w[:, 0:3], robot.data.root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        achieved = ee_pos_b[0].cpu().numpy()
        target_t = torch.tensor(target_xyz, device=env.device)
        residual = torch.norm(ee_pos_b[0] - target_t).item()

        # Diagnostic: show achieved vs commanded joint angles
        achieved_q = robot.data.joint_pos[0, robot_entity_cfg.joint_ids][:3].cpu().numpy()
        q2_droop = q_target[1] - achieved_q[1]
        q3_droop = q_target[2] - achieved_q[2]

        print(f"    round {round_idx}: residual={residual:.5f}m")
        if round_idx == 0:  # Print diagnostic details for first round only
            print(f"      cmd q2/q3: {q_target[1]:.4f} {q_target[2]:.4f}")
            print(f"      got q2/q3: {achieved_q[1]:.4f} {achieved_q[2]:.4f}")
            print(f"      droop:     {q2_droop:.4f} {q3_droop:.4f} rad")

        # Check convergence
        if residual < tol:
            print(f"    -> Converged (round {round_idx})")
            break

        # Compute error and nudge the corrected target for next round
        error = [target_xyz[i] - achieved[i] for i in range(3)]
        corrected_target = [corrected_target[i] + error[i] for i in range(3)]

    return q_target, residual


def main() -> None:
    env_cfg = Ar4GraspVerifyEnvCfg()
    env_cfg.sim.device = args_cli.device
    env = ManagerBasedEnv(cfg=env_cfg)

    robot = env.scene["robot"]
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES, body_names=["link_6"])
    robot_entity_cfg.resolve(env.scene)
    num_arm_joints = len(ARM_JOINT_NAMES)

    joint_pos_limits = robot.data.joint_pos_limits[:, robot_entity_cfg.joint_ids]
    j2_min, j2_max = joint_pos_limits[0, 1, 0].item(), joint_pos_limits[0, 1, 1].item()
    j3_min, j3_max = joint_pos_limits[0, 2, 0].item(), joint_pos_limits[0, 2, 1].item()

    os.makedirs(os.path.dirname(VIDEO_PATH), exist_ok=True)
    video_writer = imageio.get_writer(VIDEO_PATH, fps=int(1.0 / env.step_dt), codec="libx264")
    camera = env.scene["demo_camera"]

    with torch.inference_mode():
        env.reset()

        # Startup self-check: verify closed-form IK constants match the actual USD asset kinematics
        # Measure EE position at reset (home position with all joints at 0)
        ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
        ee_pos_b, _ = subtract_frame_transforms(
            robot.data.root_pose_w[:, 0:3], robot.data.root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        actual_home_pos_b = ee_pos_b[0].cpu().numpy().tolist()

        # Compute expected home position from forward kinematics at q1=q2=q3=0:
        # link2 direction = pi/2 - q2 = pi/2 (at q2=0)
        # link3 direction = pi/2 + (-pi/2 - q3) = 0 (at q3=0)
        # u = SHOULDER_U + L2*cos(pi/2) + L3*cos(0) = SHOULDER_U + L3
        # v = SHOULDER_V + L2*sin(pi/2) + L3*sin(0) = SHOULDER_V + L2
        # bearing = -pi/2, so (x,y,z) = (0, -(SHOULDER_U + L3), SHOULDER_V + L2)
        predicted_home_pos_b = [0.0, -(SHOULDER_U + L3), SHOULDER_V + L2]

        residual_home = math.sqrt(sum((a - b) ** 2 for a, b in zip(actual_home_pos_b, predicted_home_pos_b)))
        print(f"\n[STARTUP CHECK] Home pose IK verification:")
        print(f"  Predicted: {[f'{x:.5f}' for x in predicted_home_pos_b]}")
        print(f"  Actual:    {[f'{x:.5f}' for x in actual_home_pos_b]}")
        print(f"  Residual:  {residual_home:.5f}m")

        if residual_home > 0.005:
            raise RuntimeError(
                f"Home pose IK verification FAILED: residual {residual_home:.5f}m exceeds 5mm threshold. "
                f"The built USD asset's kinematics do not match the assumed URDF-derived constants "
                f"(L2={L2}, L3={L3}, SHOULDER_U={SHOULDER_U}, SHOULDER_V={SHOULDER_V}). "
                f"Expected {predicted_home_pos_b}, got {actual_home_pos_b}."
            )
        print(f"  -> Check PASSED\n")

        print(f"[INFO] Solving {len(SQUARE_POINTS_B)} square-path waypoints (closed-form IK with gravity-droop correction)...")
        solved_qs = []
        for idx, pt in enumerate(SQUARE_POINTS_B):
            print(f"\n[INFO] Solving point {idx}: {pt}")
            q_target, residual = solve_waypoint_with_correction(
                env, robot, robot_entity_cfg, pt, j2_min, j2_max, j3_min, j3_max, num_arm_joints
            )
            print(f"  -> final residual {residual:.5f}m, q={['%.4f' % x for x in q_target[:3]]}")
            solved_qs.append(q_target)

        print("\n[INFO] All waypoints solved. Executing square path (looping wrist/gripper limp, arm off, then square, then home)...\n")

        def hold_wrist_limp_and_move(target_q, steps):
            for _ in range(steps):
                wrist_now = robot.data.joint_pos[0, robot_entity_cfg.joint_ids][3:6].tolist()
                q = list(target_q[:3]) + wrist_now
                action = torch.zeros(env.num_envs, num_arm_joints + 1, device=env.device)
                for j in range(num_arm_joints):
                    action[:, j] = q[j]
                action[:, num_arm_joints] = GRIPPER_OPEN
                env.step(action)
                rgb = camera.data.output["rgb"][0].cpu().numpy()
                video_writer.append_data(rgb[:, :, :3].astype("uint8"))

        # Move from home to the first square point, then trace the loop
        # (returning to the first point at the end to close the square),
        # then back home.
        hold_wrist_limp_and_move(HOME_Q, 60)
        for idx, q in enumerate(solved_qs):
            print(f"[EXEC] moving to point {idx}")
            hold_wrist_limp_and_move(q, HOLD_STEPS_PER_POINT)
        print("[EXEC] closing the loop back to point 0")
        hold_wrist_limp_and_move(solved_qs[0], HOLD_STEPS_PER_POINT)
        print("[EXEC] returning home")
        hold_wrist_limp_and_move(HOME_Q, 90)

        env.reset()

    video_writer.close()
    env.close()
    print(f"\nVideo recorded to: {VIDEO_PATH}")


if __name__ == "__main__":
    main()
    simulation_app.close()
