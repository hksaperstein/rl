"""Open the AR4 in the Isaac Sim GUI and repeatedly cycle a full 3-DOF pick
sequence on the cube at its actual live position:

1. Approach: a smooth PATH_STEPS-waypoint straight-line Cartesian path from
   the current end-effector position to the cube, gripper held open the
   whole way (closed-form 3-DOF IK per waypoint - see solve_ik3() below),
   followed by a few closed-loop refinement rounds to close the residual
   gap the open-loop path alone leaves (see REFINE_* below).
2. Grasp: once at the cube, close the gripper.
3. Lift: a smooth LIFT_STEPS-waypoint straight-up path, gripper held closed.
4. Lower back down and release, then repeat from step 1 (the cube's live
   position is re-read each cycle, in case it moved after release).

Both paths treat the wrist (joints 4-6) as a static, rigid extension of the
forearm - held at q4=q5=q6=0 the whole time, exactly the same
simplification scripts/square_path_demo.py uses - so the "end effector" is
the fixed point at the tip of the gripper that results from that rigid
extension, not a true 6-DOF wrist-orientation solve.

This script never calls the manager-based env's own step/action pipeline -
it drives PhysX directly (write_data_to_sim + sim.step) and explicitly
refreshes robot.data.* after every step, since Isaac Lab doesn't do that
automatically outside of env.step().

.. code-block:: bash

    /home/saps/IsaacLab/isaaclab.sh -p scripts/interactive_joint_demo.py
"""

import os
import sys

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser(description="Interactive AR4 joint-angle GUI demo.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
if args_cli.headless:
    sys.exit("This demo is for live GUI interaction - run without --headless.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math  # noqa: E402

import numpy as np  # noqa: E402
import omni.ui as ui  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import ManagerBasedEnv  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.utils.math import quat_apply, subtract_frame_transforms  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from tasks.ar4.grasp_verify_env_cfg import Ar4GraspVerifyEnvCfg  # noqa: E402
from tasks.ar4.robot_cfg import GRIPPER_CLOSED_POS, GRIPPER_JOINT_NAMES, GRIPPER_OPEN_POS, ARM_JOINT_NAMES  # noqa: E402

# Same reasoning as square_path_demo.py: the default arm PD gains
# (stiffness=40, damping=4) let the arm sag noticeably under gravity at a
# held target. Raised here (this env instance only - shared robot_cfg.py
# untouched) so joints you set via the GUI actually hold where you put them.
ARM_STIFFNESS = 2500.0
ARM_DAMPING = 45.0

PATH_STEPS = 25  # straight-line waypoints from current EE position to the cube
# Settle budgets are real-world durations, not raw substep counts - this env's
# sim.dt has already changed once (120Hz -> 240Hz, tasks/ar4/grasp_verify_env_cfg.py,
# 2026-07-09 physics-fidelity pass) and a raw step count silently halves in
# real time whenever dt is tuned again. Values below are converted to actual
# step counts in main() via env.physics_dt once the env exists, preserving
# the original (dt=1/120) real-world durations these were tuned against.
PATH_SETTLE_TIME_S = 40 / 120  # settle time at each waypoint - smaller per-waypoint hops need less
GRASP_SETTLE_TIME_S = 60 / 120  # settle time to let the gripper close at the cube
RELEASE_SETTLE_TIME_S = 60 / 120  # settle time to let the gripper open at release
LIFT_HOLD_TIME_S = 90 / 120  # settle time to hold at full lift height before lowering back down
LIFT_HEIGHT = 0.12  # meters, straight up in z
LIFT_STEPS = 8  # straight-line waypoints for the lift/lower, same settle budget as the approach

# The 25-step approach alone plateaus around 5-6cm residual at this cube's
# low, extended pose (gravity droop the open-loop closed-form solve can't
# see) - more than 3x the cube's 12mm size (tasks/ar4/objects_cfg.py), so
# on its own it's not tight enough to actually grasp, only to arrive near
# the cube. A few closed-loop refinement rounds close that last gap.
REFINE_MAX_ROUNDS = 4
REFINE_SETTLE_TIME_S = 60 / 120
REFINE_TOL = 0.0025  # meters - comfortably under the cube's 6mm half-size

# Grasp a bit above the cube's center rather than driving straight at it -
# extra clearance margin on top of the L3 fix above, so a few mm of residual
# doesn't turn back into a ground/cube collision.
CUBE_HALF_SIZE = 0.006  # meters (12mm cube, tasks/ar4/objects_cfg.py)
GRASP_CLEARANCE = 0.003
GRASP_HEIGHT_OFFSET = CUBE_HALF_SIZE + GRASP_CLEARANCE

# Closed-form 3-DOF IK constants. joint_1 is a pure base-yaw; joints 2-3
# form a 2-link planar arm in the rotated plane; wrist (4-6) held at 0.
# L2/SHOULDER_* match scripts/square_path_demo.py (derived from the AR4
# URDF - see docs/superpowers/plans/2026-07-08-square-path-closed-form-ik.md).
# L3 does NOT match that script: this is elbow-to-actual-gripper-grasp-point,
# not elbow-to-link_6. annin_ar4_description/urdf/ar_gripper_macro.xacro
# mounts the gripper on link_6 via gripper_base_joint (origin (0,0,0),
# rpy -pi/2 about x) then gripper_jaw1_joint (origin (0,-0.036,0)) - a fixed
# 0.036m extension past link_6, collinear with the existing L3 direction
# (verified numerically: the same ang3=-q2-q3 relation holds exactly with
# this extension included). square_path_demo.py never grasped anything so
# this discrepancy was harmless there; it isn't here - driving "ee"=link_6
# to the cube's own position was actually driving the real fingertips
# 3.6cm past it, into the ground plane (the wrist deflection under contact
# force, and the approach-refinement residual plateauing around 5-6cm no
# matter how many rounds, were both this - not a modeling inaccuracy that
# more correction rounds could fix).
L2 = 0.305
L3 = 0.299940
SHOULDER_U = 0.06415
SHOULDER_V = 0.16978
GRIPPER_LOCAL_OFFSET = (0.0, 0.0, 0.036)  # link_6-frame offset to the true grasp point


def solve_ik3(x, y, z):
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


class JointReadout:
    """Floating on-screen window (part of the Isaac Sim app, not a separate
    OS window) listing live joint angles - call update() once per frame."""

    def __init__(self, joint_names: list[str]):
        self.joint_names = joint_names
        self._window = ui.Window("AR4 Joint Angles", width=280, height=40 + 24 * len(joint_names))
        self._labels = []
        with self._window.frame:
            with ui.VStack(spacing=4, style={"font_size": 16}):
                for name in joint_names:
                    self._labels.append(ui.Label(f"{name}: --"))

    def update(self, positions_rad):
        for label, name, val in zip(self._labels, self.joint_names, positions_rad):
            label.text = f"{name:16s} {val:+.4f} rad  ({math.degrees(val):+6.1f} deg)"


def main() -> None:
    env_cfg = Ar4GraspVerifyEnvCfg()
    env_cfg.sim.device = args_cli.device
    # Contact sensors are read-only (report force, don't affect physics) so
    # this can't be why the gripper appears to clip the cube - but disabling
    # them here, scoped to just this demo's own env instance, is harmless
    # and doesn't touch the shared scene config other RL environments
    # (pickplace_graspgated_env_cfg.py and others) depend on these for.
    env_cfg.scene.gripper_jaw1_contact = None
    env_cfg.scene.gripper_jaw2_contact = None
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
    readout_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES + GRIPPER_JOINT_NAMES)
    readout_cfg.resolve(env.scene)

    stiff_t = torch.full((1, len(arm_cfg.joint_ids)), ARM_STIFFNESS, device=env.device)
    damp_t = torch.full((1, len(arm_cfg.joint_ids)), ARM_DAMPING, device=env.device)
    robot.write_joint_stiffness_to_sim(stiff_t, joint_ids=arm_cfg.joint_ids)
    robot.write_joint_damping_to_sim(damp_t, joint_ids=arm_cfg.joint_ids)

    with torch.inference_mode():
        env.reset()

    joint_pos_limits = robot.data.joint_pos_limits[:, arm_cfg.joint_ids]
    j2_min, j2_max = joint_pos_limits[0, 1, 0].item(), joint_pos_limits[0, 1, 1].item()
    j3_min, j3_max = joint_pos_limits[0, 2, 0].item(), joint_pos_limits[0, 2, 1].item()

    readout = JointReadout(ARM_JOINT_NAMES + GRIPPER_JOINT_NAMES)

    gripper_offset_t = torch.tensor([GRIPPER_LOCAL_OFFSET], device=env.device)

    def ee_pos_b():
        """Real (physics-measured) position of the true gripper grasp point:
        link_6's actual body pose, rotated/translated by the gripper's fixed
        extension past it (see GRIPPER_LOCAL_OFFSET) - not link_6 itself."""
        ee_pose_w = robot.data.body_pose_w[:, arm_cfg.body_ids[0]]
        link6_pos_b, link6_quat_b = subtract_frame_transforms(
            robot.data.root_pose_w[:, 0:3], robot.data.root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        grasp_pos_b = link6_pos_b + quat_apply(link6_quat_b, gripper_offset_t)
        return grasp_pos_b[0].cpu().numpy()

    def command_and_settle(q1, q2, q3, gripper_target, settle_steps):
        q2 = max(j2_min, min(j2_max, q2))
        q3 = max(j3_min, min(j3_max, q3))
        arm_target = torch.tensor([[q1, q2, q3, 0.0, 0.0, 0.0]], device=env.device)
        for _ in range(settle_steps):
            robot.set_joint_position_target(arm_target, joint_ids=arm_cfg.joint_ids)
            robot.set_joint_position_target(gripper_target, joint_ids=gripper_cfg.joint_ids)
            robot.write_data_to_sim()
            env.sim.step(render=True)
            # env.step() normally calls scene.update(dt) after every physics
            # substep to refresh robot.data.* (joint_pos, body_pose_w, ...)
            # from PhysX - this custom loop bypasses env.step() entirely (to
            # avoid its action-manager overwriting GUI drags), so that
            # refresh has to happen here explicitly. Without it, every
            # robot.data.* read below is frozen at its post-reset value even
            # though PhysX and the rendered viewport are genuinely moving.
            robot.update(env.physics_dt)
            positions = robot.data.joint_pos[0, readout_cfg.joint_ids].cpu().tolist()
            readout.update(positions)
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
            print(f"[{tag}] step {i + 1}/{num_steps} -> waypoint {waypoint}, residual {residual:.5f}m")
        return last_q

    def refine_to_target(get_target, gripper_target, tag):
        """Closed-loop correction combining two things that both turned out
        to matter: (1) fixed-point error compensation for persistent gravity
        droop (each round nudges by the previous round's measured error,
        same idea validated in square_path_demo.py - no Jacobian/DLS), and
        (2) re-reading the true target live every round rather than once.
        A version with only (1) diverged (residual grew round over round,
        1.5cm -> 1.75cm) - consistent with the still-open gripper nudging a
        lightweight (0.01kg) cube during approach, so a cached target was
        silently going stale mid-refinement while compensation kept
        accumulating against it.

        get_target is a callable (e.g. re-reads the cube's live pose +
        offset), called fresh each round."""
        compensation = np.zeros(3)
        last_q = None
        target = get_target()
        for round_idx in range(REFINE_MAX_ROUNDS):
            target = get_target()
            corrected = target + compensation
            q1, q2, q3 = solve_ik3(*corrected)
            last_q = command_and_settle(q1, q2, q3, gripper_target, REFINE_SETTLE_STEPS)
            achieved = ee_pos_b()
            residual = float(torch.norm(torch.tensor(achieved - target)))
            print(f"[{tag}-REFINE] round {round_idx + 1}/{REFINE_MAX_ROUNDS}: "
                  f"target {target}, corrected {corrected}, residual {residual:.5f}m")
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
            env.sim.step(render=True)
            robot.update(env.physics_dt)
            readout.update(robot.data.joint_pos[0, readout_cfg.joint_ids].cpu().tolist())

    def read_cube_pos_b():
        cube = env.scene["cube"]
        pos_b, _ = subtract_frame_transforms(
            robot.data.root_pose_w[:, 0:3], robot.data.root_pose_w[:, 3:7], cube.data.root_pos_w, cube.data.root_quat_w
        )
        return pos_b[0].cpu().numpy()

    cycle = 0
    while simulation_app.is_running():
        cycle += 1
        print(f"\n========== CYCLE {cycle} ==========")

        # --- 1. Approach: smooth straight-line path to just above the cube
        # (re-read live, in case a previous cycle's release moved it),
        # gripper open, then a closed-loop refinement to grasp-tight
        # accuracy. Targeting GRASP_HEIGHT_OFFSET above the cube's center
        # (not the center itself) is the extra safety margin on top of the
        # L3 fix above - your "stop at the top of the cube" idea. ---
        cube_pos_b = read_cube_pos_b()
        grasp_target_b = cube_pos_b + (0.0, 0.0, GRASP_HEIGHT_OFFSET)
        start_pos_b = ee_pos_b()
        print(f"[INFO] EE start: {start_pos_b}, cube: {cube_pos_b}, grasp target: {grasp_target_b}")
        print(f"[INFO] Approaching over {PATH_STEPS} steps, gripper open...\n")
        follow_path(start_pos_b, grasp_target_b, PATH_STEPS, open_target, "APPROACH")

        def live_grasp_target():
            return read_cube_pos_b() + (0.0, 0.0, GRASP_HEIGHT_OFFSET)

        last_q, grasp_target_b = refine_to_target(live_grasp_target, open_target, "APPROACH")
        print(f"\n[INFO] At the cube (residual: {float(torch.norm(torch.tensor(ee_pos_b() - grasp_target_b))):.5f}m).\n")

        # --- 2. Grasp: close the gripper, arm holds still. ---
        print("[INFO] Closing gripper...\n")
        hold(last_q, closed_target, GRASP_SETTLE_STEPS)
        jaw_pos = robot.data.joint_pos[0, gripper_cfg.joint_ids].cpu().tolist()
        print(f"[INFO] Gripper jaw positions after closing: {jaw_pos} "
              f"(GRIPPER_CLOSED_POS={GRIPPER_CLOSED_POS} - if these match closely, the jaws met with "
              f"nothing between them, i.e. the grasp missed)\n")

        # --- 3. Lift: smooth straight-up path, gripper stays closed. ---
        lift_target_b = grasp_target_b + (0.0, 0.0, LIFT_HEIGHT)
        print(f"[INFO] Lifting {LIFT_HEIGHT}m over {LIFT_STEPS} steps, gripper closed...\n")
        last_q = follow_path(ee_pos_b(), lift_target_b, LIFT_STEPS, closed_target, "LIFT")
        print(f"\n[INFO] Lifted (residual: {float(torch.norm(torch.tensor(ee_pos_b() - lift_target_b))):.5f}m). Holding.\n")
        hold(last_q, closed_target, LIFT_HOLD_STEPS)

        # --- 4. Lower back down and release, so the next cycle has a fresh
        # approach to watch. ---
        print(f"[INFO] Lowering back down over {LIFT_STEPS} steps, gripper closed...\n")
        last_q = follow_path(ee_pos_b(), grasp_target_b, LIFT_STEPS, closed_target, "LOWER")
        print("[INFO] Releasing...\n")
        hold(last_q, open_target, RELEASE_SETTLE_STEPS)

        print(f"[INFO] Cycle {cycle} complete.\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
