"""Classical grasp contact verification: 3-DOF IK pick cycle with contact
force measurement on the gripper jaws.

This script reuses the closed-form 3-DOF IK approach from
scripts/interactive_joint_demo.py (approach → grasp → lift → lower → release)
and adds real contact-sensor instrumentation after each grasp to verify
whether bilateral antipodal contact is achieved (both jaws touching the
cube) or if the grasp misses (jaws close on empty air).

After each grasp-close phase, it reads the gripper jaw contact sensors' force
magnitudes and checks whether the cube's height actually rises during lift
(genuine grasp working) or stays near the ground (grasp missed).

Runs for exactly 3 cycles then exits cleanly.

Runs with a GUI window by default (a display is available on this machine).
Pass --headless explicitly if you want no UI window.

.. code-block:: bash

    /home/saps/IsaacLab/isaaclab.sh -p scripts/classical_grasp_contact_check.py
"""

import math
import os
import sys

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser(description="Classical 3-DOF IK grasp with contact verification.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # Environment config requires cameras

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import ManagerBasedEnv  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, subtract_frame_transforms  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from tasks.ar4.grasp_verify_env_cfg import Ar4GraspVerifyEnvCfg  # noqa: E402
from tasks.ar4.robot_cfg import GRIPPER_CLOSED_POS, GRIPPER_JOINT_NAMES, GRIPPER_OPEN_POS, ARM_JOINT_NAMES  # noqa: E402

# Same actuator tuning as interactive_joint_demo.py
ARM_STIFFNESS = 2500.0
ARM_DAMPING = 45.0

PATH_STEPS = 25
PATH_SETTLE_TIME_S = 40 / 120
GRASP_SETTLE_TIME_S = 60 / 120
RELEASE_SETTLE_TIME_S = 60 / 120
LIFT_HOLD_TIME_S = 90 / 120
LIFT_HEIGHT = 0.12
LIFT_STEPS = 8

REFINE_MAX_ROUNDS = 4
REFINE_SETTLE_TIME_S = 60 / 120
REFINE_TOL = 0.0025

CUBE_HALF_SIZE = 0.006
GRASP_CLEARANCE = 0.003
GRASP_HEIGHT_OFFSET = CUBE_HALF_SIZE + GRASP_CLEARANCE

# Closed-form 3-DOF IK constants (same as interactive_joint_demo.py)
L2 = 0.305
L3 = 0.299940
SHOULDER_U = 0.06415
SHOULDER_V = 0.16978
GRIPPER_LOCAL_OFFSET = (0.0, 0.0, 0.036)

# Contact verification thresholds
FORCE_THRESHOLD = 0.05  # Newtons - minimum force to count as "touching"
BILATERAL_CONTACT_THRESHOLD = 0.05  # Both jaws must exceed this
CUBE_HEIGHT_RISE_THRESHOLD = 0.01  # Cube must rise at least 1cm to count as lifted


def solve_ik3(x, y, z):
    """Closed-form 3-DOF IK solve."""
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


def main() -> None:
    env_cfg = Ar4GraspVerifyEnvCfg()
    env_cfg.sim.device = args_cli.device
    # Keep contact sensors enabled (unlike interactive_joint_demo.py which disables them)
    env = ManagerBasedEnv(cfg=env_cfg)

    global PATH_SETTLE_STEPS, GRASP_SETTLE_STEPS, RELEASE_SETTLE_STEPS, REFINE_SETTLE_STEPS, LIFT_HOLD_STEPS
    PATH_SETTLE_STEPS = max(1, round(PATH_SETTLE_TIME_S / env.physics_dt))
    GRASP_SETTLE_STEPS = max(1, round(GRASP_SETTLE_TIME_S / env.physics_dt))
    RELEASE_SETTLE_STEPS = max(1, round(RELEASE_SETTLE_TIME_S / env.physics_dt))
    REFINE_SETTLE_STEPS = max(1, round(REFINE_SETTLE_TIME_S / env.physics_dt))
    LIFT_HOLD_STEPS = max(1, round(LIFT_HOLD_TIME_S / env.physics_dt))

    robot = env.scene["robot"]
    arm_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES, body_names=["link_6"])
    arm_cfg.resolve(env.scene)
    gripper_cfg = SceneEntityCfg("robot", joint_names=GRIPPER_JOINT_NAMES)
    gripper_cfg.resolve(env.scene)

    # Set up actuator tuning
    stiff_t = torch.full((1, len(arm_cfg.joint_ids)), ARM_STIFFNESS, device=env.device)
    damp_t = torch.full((1, len(arm_cfg.joint_ids)), ARM_DAMPING, device=env.device)
    robot.write_joint_stiffness_to_sim(stiff_t, joint_ids=arm_cfg.joint_ids)
    robot.write_joint_damping_to_sim(damp_t, joint_ids=arm_cfg.joint_ids)

    with torch.inference_mode():
        env.reset()

    joint_pos_limits = robot.data.joint_pos_limits[:, arm_cfg.joint_ids]
    j2_min, j2_max = joint_pos_limits[0, 1, 0].item(), joint_pos_limits[0, 1, 1].item()
    j3_min, j3_max = joint_pos_limits[0, 2, 0].item(), joint_pos_limits[0, 2, 1].item()

    gripper_offset_t = torch.tensor([GRIPPER_LOCAL_OFFSET], device=env.device)

    # Access contact sensors (they are now enabled)
    jaw1_sensor = env.scene["gripper_jaw1_contact"]
    jaw2_sensor = env.scene["gripper_jaw2_contact"]
    cube = env.scene["cube"]

    def ee_pos_b():
        """Real gripper grasp point position in base frame."""
        ee_pose_w = robot.data.body_pose_w[:, arm_cfg.body_ids[0]]
        link6_pos_b, link6_quat_b = subtract_frame_transforms(
            robot.data.root_pose_w[:, 0:3], robot.data.root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        grasp_pos_b = link6_pos_b + quat_apply(link6_quat_b, gripper_offset_t)
        return grasp_pos_b[0].cpu().numpy()

    def debug_pinch_axis_and_cube():
        """DIAGNOSTIC: link_6 orientation (as euler) and the world-frame
        direction of the gripper's jaw-closing axis (link_6 local +/-X, per
        ar_gripper_macro.xacro: both jaw prismatic joints translate along
        the gripper_base_link's local X, which itself equals link_6's local
        X since gripper_base_joint's -pi/2 rotation is about X). If this
        axis has a large vertical (Z) component, the jaws are not closing
        in a horizontal plane around the cube - they'd be presenting at an
        angle instead of straddling it, which could explain a positional
        near-miss still producing zero contact force."""
        ee_pose_w = robot.data.body_pose_w[:, arm_cfg.body_ids[0]]
        _, link6_quat_b = subtract_frame_transforms(
            robot.data.root_pose_w[:, 0:3], robot.data.root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        roll, pitch, yaw = euler_xyz_from_quat(link6_quat_b)
        pinch_axis_b = quat_apply(link6_quat_b, torch.tensor([[1.0, 0.0, 0.0]], device=env.device))
        cube_vel = cube.data.root_vel_w[0, :3].cpu().numpy() if hasattr(cube.data, "root_vel_w") else None
        print(f"[DEBUG] link6 rpy (rad): ({roll.item():.4f}, {pitch.item():.4f}, {yaw.item():.4f})")
        print(f"[DEBUG] pinch axis (base frame, link6 local X in world-ish base frame): {pinch_axis_b[0].cpu().numpy()}")
        if cube_vel is not None:
            print(f"[DEBUG] cube linear velocity (world): {cube_vel}, |v|={float(np.linalg.norm(cube_vel)):.5f} m/s")

    def command_and_settle(q1, q2, q3, gripper_target, settle_steps):
        q2 = max(j2_min, min(j2_max, q2))
        q3 = max(j3_min, min(j3_max, q3))
        arm_target = torch.tensor([[q1, q2, q3, 0.0, 0.0, 0.0]], device=env.device)
        for _ in range(settle_steps):
            robot.set_joint_position_target(arm_target, joint_ids=arm_cfg.joint_ids)
            robot.set_joint_position_target(gripper_target, joint_ids=gripper_cfg.joint_ids)
            robot.write_data_to_sim()
            env.sim.step(render=not args_cli.headless)
            robot.update(env.physics_dt)
        return (q1, q2, q3)

    open_target = torch.tensor([[GRIPPER_OPEN_POS, GRIPPER_OPEN_POS]], device=env.device)
    closed_target = torch.tensor([[GRIPPER_CLOSED_POS, GRIPPER_CLOSED_POS]], device=env.device)

    def follow_path(start, end, num_steps, gripper_target, tag):
        last_q = None
        for i in range(num_steps):
            t = (i + 1) / num_steps
            waypoint = start + t * (end - start)
            q1, q2, q3 = solve_ik3(*waypoint)
            last_q = command_and_settle(q1, q2, q3, gripper_target, PATH_SETTLE_STEPS)
            residual = float(torch.norm(torch.tensor(ee_pos_b() - waypoint)))
            print(f"[{tag}] step {i + 1}/{num_steps} -> residual {residual:.5f}m")
        return last_q

    def refine_to_target(get_target, gripper_target, tag):
        """Closed-loop refinement with fixed-point error compensation."""
        compensation = np.zeros(3)
        last_q = None
        for round_idx in range(REFINE_MAX_ROUNDS):
            target = get_target()
            corrected = target + compensation
            q1, q2, q3 = solve_ik3(*corrected)
            last_q = command_and_settle(q1, q2, q3, gripper_target, REFINE_SETTLE_STEPS)
            achieved = ee_pos_b()
            residual = float(torch.norm(torch.tensor(achieved - target)))
            per_axis = achieved - target
            print(f"[{tag}-REFINE] round {round_idx + 1}/{REFINE_MAX_ROUNDS}: residual {residual:.5f}m "
                  f"(dx={per_axis[0]:.5f}, dy={per_axis[1]:.5f}, dz={per_axis[2]:.5f}) "
                  f"cube_now={read_cube_pos_b()}")
            if residual < REFINE_TOL:
                break
            compensation = compensation + (target - achieved)
        return last_q, target

    def hold(q_tuple, gripper_target, steps):
        target = torch.tensor([q_tuple + (0.0, 0.0, 0.0)], device=env.device)
        for _ in range(steps):
            robot.set_joint_position_target(target, joint_ids=arm_cfg.joint_ids)
            robot.set_joint_position_target(gripper_target, joint_ids=gripper_cfg.joint_ids)
            robot.write_data_to_sim()
            env.sim.step(render=not args_cli.headless)
            robot.update(env.physics_dt)

    def read_cube_pos_b():
        cube_data = env.scene["cube"]
        pos_b, _ = subtract_frame_transforms(
            robot.data.root_pose_w[:, 0:3], robot.data.root_pose_w[:, 3:7], cube_data.data.root_pos_w, cube_data.data.root_quat_w
        )
        return pos_b[0].cpu().numpy()

    def read_contact_forces():
        """Read gripper jaw contact forces (magnitude in Newtons)."""
        jaw1_force_vec = jaw1_sensor.data.force_matrix_w.view(1, 3)[0]
        jaw2_force_vec = jaw2_sensor.data.force_matrix_w.view(1, 3)[0]
        jaw1_mag = torch.linalg.vector_norm(jaw1_force_vec).item()
        jaw2_mag = torch.linalg.vector_norm(jaw2_force_vec).item()
        return jaw1_mag, jaw2_mag

    # Run exactly 3 cycles
    for cycle in range(1, 4):
        print(f"\n========== CYCLE {cycle} ==========")

        # Phase 1: Approach
        cube_pos_b = read_cube_pos_b()
        grasp_target_b = cube_pos_b + (0.0, 0.0, GRASP_HEIGHT_OFFSET)
        start_pos_b = ee_pos_b()
        print(f"[INFO] EE start: {start_pos_b}")
        print(f"[INFO] Cube position (base frame): {cube_pos_b}")
        print(f"[INFO] Grasp target: {grasp_target_b}")
        print(f"[INFO] Approaching over {PATH_STEPS} steps, gripper open...")
        follow_path(start_pos_b, grasp_target_b, PATH_STEPS, open_target, "APPROACH")

        def live_grasp_target():
            return read_cube_pos_b() + (0.0, 0.0, GRASP_HEIGHT_OFFSET)

        last_q, grasp_target_b = refine_to_target(live_grasp_target, open_target, "APPROACH")
        residual_at_grasp = float(torch.norm(torch.tensor(ee_pos_b() - grasp_target_b)))
        cube_drift = read_cube_pos_b() - cube_pos_b
        print(f"[INFO] At grasp target (residual: {residual_at_grasp:.5f}m).")
        print(f"[DEBUG] cube drift since cycle start (approach+refine): {cube_drift}, "
              f"|drift|={float(np.linalg.norm(cube_drift)):.5f}m")

        # Phase 2: Grasp - close gripper and measure contact forces
        debug_pinch_axis_and_cube()
        print("[INFO] Closing gripper...")
        cube_z_before_lift = read_cube_pos_b()[2]
        hold(last_q, closed_target, GRASP_SETTLE_STEPS)
        jaw_pos_after_close = robot.data.joint_pos[0, gripper_cfg.joint_ids].cpu().tolist()
        print(f"[DEBUG] jaw positions after closing: {jaw_pos_after_close} (GRIPPER_CLOSED_POS={GRIPPER_CLOSED_POS} "
              f"- if these match closely, jaws met with nothing between them)")

        # Read contact forces immediately after grasp
        jaw1_force, jaw2_force = read_contact_forces()
        bilateral_contact = (jaw1_force > BILATERAL_CONTACT_THRESHOLD) and (jaw2_force > BILATERAL_CONTACT_THRESHOLD)

        print(f"[CONTACT] jaw1_force: {jaw1_force:.4f} N")
        print(f"[CONTACT] jaw2_force: {jaw2_force:.4f} N")
        print(f"[CONTACT] bilateral: {bilateral_contact}")
        print(f"[CONTACT] residual_at_grasp: {residual_at_grasp:.5f} m")

        # Phase 3: Lift
        lift_target_b = grasp_target_b + (0.0, 0.0, LIFT_HEIGHT)
        print(f"[INFO] Lifting {LIFT_HEIGHT}m over {LIFT_STEPS} steps, gripper closed...")
        last_q = follow_path(ee_pos_b(), lift_target_b, LIFT_STEPS, closed_target, "LIFT")
        print(f"[INFO] Lifted (residual: {float(torch.norm(torch.tensor(ee_pos_b() - lift_target_b))):.5f}m). Holding.")
        hold(last_q, closed_target, LIFT_HOLD_STEPS)

        # Check if cube actually rose
        cube_z_after_lift = cube.data.root_pos_w[0, 2].item()
        cube_height_rise = cube_z_after_lift - cube_z_before_lift
        cube_actually_lifted = cube_height_rise > CUBE_HEIGHT_RISE_THRESHOLD

        print(f"[LIFT_CHECK] cube z before lift: {cube_z_before_lift:.6f} m")
        print(f"[LIFT_CHECK] cube z after lift: {cube_z_after_lift:.6f} m")
        print(f"[LIFT_CHECK] cube height rise: {cube_height_rise:.6f} m")
        print(f"[LIFT_CHECK] cube_actually_lifted: {cube_actually_lifted}")

        # Phase 4: Lower and release
        print(f"[INFO] Lowering back down over {LIFT_STEPS} steps, gripper closed...")
        last_q = follow_path(ee_pos_b(), grasp_target_b, LIFT_STEPS, closed_target, "LOWER")
        print("[INFO] Releasing...")
        hold(last_q, open_target, RELEASE_SETTLE_STEPS)

        print(f"[SUMMARY CYCLE {cycle}]")
        print(f"  Jaw contact forces (N): jaw1={jaw1_force:.4f}, jaw2={jaw2_force:.4f}")
        print(f"  Bilateral contact: {bilateral_contact}")
        print(f"  Residual at grasp: {residual_at_grasp:.5f} m")
        print(f"  Cube lifted: {cube_actually_lifted} (rise={cube_height_rise:.6f}m)")

        if bilateral_contact and cube_actually_lifted:
            print(f"  Result: GRASP SUCCEEDED (bilateral contact + cube lifted)")
        elif not bilateral_contact:
            print(f"  Result: GRASP MISSED (no bilateral contact - jaws closed on empty space)")
        elif not cube_actually_lifted:
            print(f"  Result: GRASP INCOMPLETE (contact but cube did not lift)")
        else:
            print(f"  Result: UNKNOWN")

        print(f"[INFO] Cycle {cycle} complete.\n")

    print("========== ALL 3 CYCLES COMPLETE ==========")
    env.close()
    print("Environment closed cleanly.")


if __name__ == "__main__":
    main()
    simulation_app.close()
    print("Simulation app closed.")
