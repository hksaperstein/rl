"""Cartesian-corrected closing grasp+lift attempt for the AR4 multi-week
investigation (2026-07-28, ar4-cartesian-fingertip-correction task) - direct
continuation of the 2026-07-28 "ar4-pedestal-ground-clearance-fix task",
which got the ground/pedestal-collision failure signature to genuinely
disappear (pinch-point discrepancy shrank from ~21mm to ~6-7mm) but still
found no real lift: contact force while nominally OPEN persisted (51-57N),
and dropped to exactly 0.000N the instant PHASE4-LIFT-CLOSE began at every
point - the cube was nudged/contacted, not gripped.

Hypothesis this task tests: `settle_to_joint_pose` (tasks/ar4/joint_tracking.py)
only nulls error in JOINT space (already confirmed sub-1deg per joint in most
cases) - but a small per-joint residual, compounded through AR4's ~0.5m
serial chain via each joint's own lever arm, can still leave several mm of
Cartesian error at the fingertip even when every individual joint looks
converged. This is a DIFFERENT, independent mechanism from the roll/heading-
misalignment explanation the pedestal-fix task's own "Honest verdict"
section flagged as the more likely cause (ROLL_TOL_DEG=12 bounding but not
zeroing jaw-heading error) - this task's job is to genuinely close the
POSITION residual via a Cartesian outer-loop correction
(`tasks/ar4/joint_tracking.py`'s new `settle_to_cartesian_pose`, added this
same task) and see whether that alone is enough, or whether (per the
dispatch's own explicit anticipation) the grasp still fails for the
already-flagged, separate orientation reason - in which case this task's
job is to report that honestly, not force a positive.

Directly reuses (does not re-derive):
  - tasks/ar4/joint_tracking.py's settle_to_joint_pose (unchanged) AND the
    new settle_to_cartesian_pose (added this task, see its own docstring for
    the DLS/Jacobian derivation - the same _ee_point_pos_and_jacobian
    identity scripts/grasp_demo_v2.py's DLS polish already uses, reimplemented
    in pure torch to keep tasks/ar4/joint_tracking.py isaaclab-import-free).
  - scripts/ar4_pedestal_grasp_confirm.py's own VALIDATION_POINTS (the same
    3 pedestal-corrected points, copied verbatim - not re-derived), PHASE0-6
    structure, and open_gripper_clean/both_jaws_contacted/real_lift/
    held_through_retreat verdict criteria - this task's own scope is adding
    ONE new phase (Cartesian correction, between GRASP-OPEN-SETTLE and
    GRASP-CLOSE) and reporting the before/after fingertip residual, not
    re-deriving anything else already validated.
  - tasks/ar4/pickplace_graspgoal_env_cfg.py's Ar4PickPlaceGraspGoalEnvCfg
    (pedestal + repositioned cube spawn, unchanged) - no camera (numeric/
    fast variant only, matching every prior confirm script's own "don't pay
    render cost on an exploratory run that might not even succeed"
    rationale); a separate camera-enabled script produces the demo video
    only if THIS script confirms a real grasp+lift.

.. code-block:: bash

    flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 /workspace/isaaclab/isaaclab.sh -p scripts/ar4_cartesian_grasp_confirm.py --headless"
"""

import math
import os
import sys

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser(description="Cartesian-corrected AR4 grasp+lift attempt (pedestal + fingertip Cartesian correction).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.utils.math import matrix_from_quat  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from tasks.ar4.pickplace_graspgoal_env_cfg import Ar4PickPlaceGraspGoalEnvCfg  # noqa: E402
from tasks.ar4.robot_cfg import ARM_JOINT_NAMES, GRIPPER_JOINT_NAMES  # noqa: E402
from tasks.ar4.fk_verification import compute_link_pose_from_joint_values  # noqa: E402
from tasks.ar4.joint_tracking import settle_to_joint_pose, settle_to_cartesian_pose  # noqa: E402

# ----------------------------------------------------------------------
# Same 3 pedestal-corrected validation points as
# scripts/ar4_pedestal_grasp_confirm.py - copied verbatim, not re-derived
# (real fingertip clearance above the pedestal top: 9.81/9.81/10.50mm
# respectively, per that script's own provenance comment).
# ----------------------------------------------------------------------
VALIDATION_POINTS = {
    "Q0_bearing95": {
        "cube_xy": (-0.03690307221845676, 0.3902115233016254),
        "grasp_q_deg": [-6.486502296738718, 53.23822236377881, 14.982846754337578, -14.92769805402622, 21.951674171245376, 114.65851419966206],
        "pregrasp_q_deg": [-6.486502296738718, 45.26345050158713, 13.541632691134666, -8.35706089858409, 34.462683836708806, 112.67242887258548],
    },
    "Q1_bearing108": {
        "cube_xy": (-0.12356049880846685, 0.3719673005252888),
        "grasp_q_deg": [-19.45950689021618, 53.23822236377881, 14.982846754337578, -14.92769805402622, 21.951674171245376, 114.65851419966206],
        "pregrasp_q_deg": [-19.45950689021618, 44.40811556086022, 15.481953442347919, -9.280799797359624, 28.688841368903837, 126.93076282947612],
    },
    "Q2_bearing80": {
        "cube_xy": (0.05695223525355342, 0.3931739774682811),
        "grasp_q_deg": [8.648669728984965, 53.613872356501105, 13.204438863134575, 5.288396455851214, 23.404666660546397, -115.23964540657674],
        "pregrasp_q_deg": [8.648669728984965, 44.83718769224915, 13.335498537014379, 2.5021438396328812, 32.386382376666894, -97.13224456010194],
    },
}

HOME_Q_DEG = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
HOME_Q = [math.radians(d) for d in HOME_Q_DEG]

GRIPPER_OPEN_POS = 0.014
GRIPPER_CLOSED_POS = 0.0
GRIPPER_OPEN_EXPR = {"gripper_jaw1_joint": GRIPPER_OPEN_POS, "gripper_jaw2_joint": GRIPPER_OPEN_POS}
GRIPPER_CLOSED_EXPR = {"gripper_jaw1_joint": GRIPPER_CLOSED_POS, "gripper_jaw2_joint": GRIPPER_CLOSED_POS}

EE_OFFSET_LOCAL_LIST = [0.0, 0.0, 0.036]  # matches grasp_demo_v2.py/_EE_OFFSET

# Baseline gain already validated (2026-07-28 ar4-joint-tracking-closed-loop-
# fix task) as safe/effective whenever the target is genuinely reachable -
# not re-swept here, that tracking-gap question is already closed.
STIFFNESS = 4000.0
DAMPING = 200.0
EFFORT_LIMIT = 20.0

# Cartesian-correction convergence bar per this task's own dispatch brief:
# drive the real fingertip within ~1-2mm of the FK-predicted pinch target
# before attempting the grasp (vs. the ~6-7mm the joint-only settle leaves).
CARTESIAN_TOL_M = 0.002


def base_to_world(p_base):
    import numpy as np
    out = np.array(p_base, dtype=float).copy()
    out[..., 0] *= -1.0
    out[..., 1] *= -1.0
    return out


def _quat_wxyz_to_matrix(quat_wxyz):
    import numpy as np
    w, x, y, z = quat_wxyz
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def fk_predicted_pinch_point_world(joint_values_rad: dict):
    pos_b, quat_b = compute_link_pose_from_joint_values(joint_values_rad, "link_6")
    rot_b = _quat_wxyz_to_matrix(quat_b)
    import numpy as np
    pinch_b = pos_b + rot_b @ np.array(EE_OFFSET_LOCAL_LIST)
    return base_to_world(pinch_b)


def _dist_mm(a, b):
    return math.dist(a, b) * 1000.0


def _achieved_pinch_world(robot, link6_body_id):
    link6_pose_w = robot.data.body_pose_w[0, link6_body_id]
    link6_pos_w, link6_quat_w = link6_pose_w[0:3], link6_pose_w[3:7]
    rot_w = matrix_from_quat(link6_quat_w.unsqueeze(0))[0]
    offset_t = torch.tensor(EE_OFFSET_LOCAL_LIST, device=link6_pos_w.device)
    return (link6_pos_w + rot_w @ offset_t).tolist()


def run_grasp_point(env, robot, arm_cfg, gripper_cfg, contact_sensors, cube, link6_body_id, point_label, point):
    print("\n" + "#" * 70)
    print(f"# CARTESIAN-CORRECTED GRASP+LIFT ATTEMPT: {point_label}")
    print("#" * 70)

    grasp_q = [math.radians(d) for d in point["grasp_q_deg"]]
    pregrasp_q = [math.radians(d) for d in point["pregrasp_q_deg"]]
    cube_xy = point["cube_xy"]

    joint_values_commanded = {name: grasp_q[i] for i, name in enumerate(ARM_JOINT_NAMES)}
    fk_pred_pinch_w = fk_predicted_pinch_point_world(joint_values_commanded).tolist()

    env.reset()
    override_z = cube.data.root_pos_w[0, 2].item()
    override_pos = torch.tensor([[cube_xy[0], cube_xy[1], override_z]], device=env.device)
    override_quat = cube.data.root_quat_w[0:1].clone()
    cube.write_root_pose_to_sim(torch.cat([override_pos, override_quat], dim=-1), env_ids=torch.tensor([0], device=env.device))
    cube.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device), env_ids=torch.tensor([0], device=env.device))

    home_t = torch.tensor([HOME_Q], device=env.device)
    open_t = torch.tensor([[GRIPPER_OPEN_EXPR[n] for n in GRIPPER_JOINT_NAMES]], device=env.device)
    for _ in range(30):
        robot.set_joint_position_target(home_t, joint_ids=arm_cfg.joint_ids)
        robot.set_joint_position_target(open_t, joint_ids=gripper_cfg.joint_ids)
        robot.write_data_to_sim()
        env.sim.step(render=False)
        robot.update(env.physics_dt)
    cube_z_on_pedestal = cube.data.root_pos_w[0, 2].item()
    print(f"[INFO] Cube teleported to: {override_pos[0].tolist()}, settled resting height on pedestal={cube_z_on_pedestal:.4f}m")

    n_arm = len(arm_cfg.joint_ids)
    robot.write_joint_stiffness_to_sim(torch.full((1, n_arm), STIFFNESS, device=env.device), joint_ids=arm_cfg.joint_ids)
    robot.write_joint_damping_to_sim(torch.full((1, n_arm), DAMPING, device=env.device), joint_ids=arm_cfg.joint_ids)
    robot.write_joint_effort_limit_to_sim(torch.full((1, n_arm), EFFORT_LIMIT, device=env.device), joint_ids=arm_cfg.joint_ids)

    force_tracker = {"jaw1_max": 0.0, "jaw2_max": 0.0, "open_gripper_max_force": 0.0}
    step_counter = {"n": 0}

    def _track_forces(is_pre_close: bool):
        for sensor in contact_sensors:
            sensor.update(env.physics_dt, force_recompute=True)
        jaw1_force = contact_sensors[0].data.net_forces_w[0, 0].norm().item()
        jaw2_force = contact_sensors[1].data.net_forces_w[0, 0].norm().item()
        force_tracker["jaw1_max"] = max(force_tracker["jaw1_max"], jaw1_force)
        force_tracker["jaw2_max"] = max(force_tracker["jaw2_max"], jaw2_force)
        if is_pre_close:
            force_tracker["open_gripper_max_force"] = max(force_tracker["open_gripper_max_force"], jaw1_force, jaw2_force)
        return jaw1_force, jaw2_force

    def _drive_naive(target_q, gripper_expr, duration, label):
        target_t = torch.tensor([target_q], device=env.device)
        g_t = torch.tensor([[gripper_expr[n] for n in GRIPPER_JOINT_NAMES]], device=env.device)
        for i in range(duration):
            robot.set_joint_position_target(target_t, joint_ids=arm_cfg.joint_ids)
            robot.set_joint_position_target(g_t, joint_ids=gripper_cfg.joint_ids)
            robot.write_data_to_sim()
            env.sim.step(render=False)
            robot.update(env.physics_dt)
            j1, j2 = _track_forces(is_pre_close=False)
            if i % 20 == 0 or i == duration - 1:
                cube_z = cube.data.root_pos_w[0, 2].item()
                print(f"  [{label} step {i:3d}] cube_z={cube_z:.4f}m jaw1={j1:.3f}N jaw2={j2:.3f}N")
        return cube.data.root_pos_w[0, 2].item()

    def _settle_tracked(desired_q, gripper_expr, label):
        def on_step(outer, i):
            j1, j2 = _track_forces(is_pre_close=True)
            step_counter["n"] += 1
            if step_counter["n"] % 30 == 0:
                cube_z = cube.data.root_pos_w[0, 2].item()
                print(f"    [{label} outer={outer} step={i}] cube_z={cube_z:.4f}m jaw1={j1:.3f}N jaw2={j2:.3f}N")

        gripper_target = [gripper_expr[n] for n in GRIPPER_JOINT_NAMES]
        result = settle_to_joint_pose(
            env, robot, arm_cfg.joint_ids, desired_q,
            tol_rad=math.radians(0.15), max_outer_iters=8, inner_settle_steps=150,
            integral_gain=1.0, integral_clamp=0.5,
            gripper_joint_ids=gripper_cfg.joint_ids, gripper_target=gripper_target,
            on_step=on_step, label=label,
        )
        return result

    def _cartesian_settle_tracked(desired_pinch_w, initial_q, gripper_expr, label):
        def on_step(outer, i):
            j1, j2 = _track_forces(is_pre_close=True)
            step_counter["n"] += 1
            if step_counter["n"] % 30 == 0:
                cube_z = cube.data.root_pos_w[0, 2].item()
                print(f"    [{label} outer={outer} step={i}] cube_z={cube_z:.4f}m jaw1={j1:.3f}N jaw2={j2:.3f}N")

        gripper_target = [gripper_expr[n] for n in GRIPPER_JOINT_NAMES]
        result = settle_to_cartesian_pose(
            env, robot, arm_cfg.joint_ids, link6_body_id, EE_OFFSET_LOCAL_LIST,
            desired_pinch_w, initial_q,
            tol_m=CARTESIAN_TOL_M, max_outer_iters=8, inner_settle_steps=150,
            dls_lambda=0.02, max_joint_delta_rad=0.15,
            gripper_joint_ids=gripper_cfg.joint_ids, gripper_target=gripper_target,
            on_step=on_step, label=label,
        )
        return result

    cube_z_by_phase = {}
    cube_z_by_phase["PHASE0-HOME-OPEN"] = cube.data.root_pos_w[0, 2].item()

    pregrasp_result = _settle_tracked(pregrasp_q, GRIPPER_OPEN_EXPR, "PHASE1-PREGRASP-OPEN-SETTLE")
    cube_z_by_phase["PHASE1-PREGRASP-OPEN"] = cube.data.root_pos_w[0, 2].item()
    corrected_pregrasp_target = [d + c for d, c in zip(pregrasp_q, pregrasp_result["correction"])]
    print(f"[INFO] PREGRASP settle: converged={pregrasp_result['converged']} iters={pregrasp_result['n_outer_iters']} max_err_deg={pregrasp_result['max_err_deg']:.4f}")

    grasp_result = _settle_tracked(grasp_q, GRIPPER_OPEN_EXPR, "PHASE2-GRASP-OPEN-SETTLE")
    cube_z_by_phase["PHASE2-GRASP-OPEN"] = cube.data.root_pos_w[0, 2].item()
    joint_corrected_grasp_target = [d + c for d, c in zip(grasp_q, grasp_result["correction"])]
    print(f"[INFO] GRASP joint-settle: converged={grasp_result['converged']} iters={grasp_result['n_outer_iters']} max_err_deg={grasp_result['max_err_deg']:.4f}")

    achieved_pinch_w_before = _achieved_pinch_world(robot, link6_body_id)
    pinch_disc_mm_before = _dist_mm(achieved_pinch_w_before, fk_pred_pinch_w)
    print(f"[INFO] BEFORE Cartesian correction: pinch-point discrepancy vs FK prediction = {pinch_disc_mm_before:.3f}mm")

    cartesian_result = _cartesian_settle_tracked(
        fk_pred_pinch_w, joint_corrected_grasp_target, GRIPPER_OPEN_EXPR, "PHASE2b-GRASP-CARTESIAN-CORRECT"
    )
    corrected_grasp_target = cartesian_result["final_q"]
    pinch_disc_mm_after = cartesian_result["pinch_error_mm"]
    print(
        f"[INFO] AFTER Cartesian correction: pinch-point discrepancy = {pinch_disc_mm_after:.3f}mm "
        f"(converged={cartesian_result['converged']}, iters={cartesian_result['n_outer_iters']})"
    )
    cube_z_by_phase["PHASE2b-GRASP-CARTESIAN-CORRECT"] = cube.data.root_pos_w[0, 2].item()
    print(f"[INFO] open_gripper_max_force so far (PHASE0-2b, pre-CLOSE): {force_tracker['open_gripper_max_force']:.4f}N")

    cube_z_by_phase["PHASE3-GRASP-CLOSE"] = _drive_naive(corrected_grasp_target, GRIPPER_CLOSED_EXPR, 90, "PHASE3-GRASP-CLOSE")
    cube_z_by_phase["PHASE4-LIFT-CLOSE"] = _drive_naive(corrected_pregrasp_target, GRIPPER_CLOSED_EXPR, 120, "PHASE4-LIFT-CLOSE")
    cube_z_by_phase["PHASE5-HOLD-CLOSE"] = _drive_naive(corrected_pregrasp_target, GRIPPER_CLOSED_EXPR, 120, "PHASE5-HOLD-CLOSE")
    cube_z_by_phase["PHASE6-RETREAT-CLOSE"] = _drive_naive(HOME_Q, GRIPPER_CLOSED_EXPR, 150, "PHASE6-RETREAT-CLOSE")

    print("\n" + "=" * 70)
    print(f"SUMMARY: {point_label}")
    print("=" * 70)
    for label, z in cube_z_by_phase.items():
        print(f"  cube_z at end of {label}: {z:.4f}m")
    height_gain = max(cube_z_by_phase.values()) - cube_z_on_pedestal
    final_z = cube_z_by_phase["PHASE6-RETREAT-CLOSE"]
    held_through_retreat = final_z > cube_z_on_pedestal + 0.01
    real_lift = height_gain > 0.01
    both_jaws_contacted = force_tracker["jaw1_max"] > 0.001 and force_tracker["jaw2_max"] > 0.001
    open_gripper_clean = force_tracker["open_gripper_max_force"] < 1.0
    print(f"Cube resting height (on pedestal): {cube_z_on_pedestal:.4f}m")
    print(f"Max cube height reached: {max(cube_z_by_phase.values()):.4f}m (gain={height_gain*1000:.2f}mm)")
    print(f"Final cube height (end of RETREAT): {final_z:.4f}m")
    print(f"Fingertip Cartesian residual: BEFORE={pinch_disc_mm_before:.3f}mm AFTER={pinch_disc_mm_after:.3f}mm")
    print(f"Max jaw-cube contact force WHILE GRIPPER STILL OPEN (pre-CLOSE): {force_tracker['open_gripper_max_force']:.4f}N")
    print(f"Max jaw1/jaw2 contact force (any phase): {force_tracker['jaw1_max']:.4f}N / {force_tracker['jaw2_max']:.4f}N")
    print(f"BOTH jaws registered real contact force (post-close): {both_jaws_contacted}")
    print(f"Real height gain (>1cm): {real_lift}")
    print(f"Held through retreat (>1cm above pedestal-resting height): {held_through_retreat}")
    print(f"Open-gripper collision-free (<1N while nominally open): {open_gripper_clean}")
    verdict = (
        "GRASP+LIFT CONFIRMED" if (open_gripper_clean and both_jaws_contacted and real_lift and held_through_retreat)
        else "GRASP+LIFT NOT CONFIRMED"
    )
    print(f"VERDICT [{point_label}]: {verdict}")
    print("=" * 70)
    return {
        "verdict": verdict, "open_gripper_max_force": force_tracker["open_gripper_max_force"],
        "jaw1_max": force_tracker["jaw1_max"], "jaw2_max": force_tracker["jaw2_max"],
        "height_gain_mm": height_gain * 1000, "held_through_retreat": held_through_retreat,
        "pinch_discrepancy_before_mm": pinch_disc_mm_before, "pinch_discrepancy_after_mm": pinch_disc_mm_after,
        "cartesian_converged": cartesian_result["converged"],
        "pregrasp_settle_iters": pregrasp_result["n_outer_iters"], "grasp_settle_iters": grasp_result["n_outer_iters"],
    }


def main() -> None:
    env_cfg = Ar4PickPlaceGraspGoalEnvCfg()
    env_cfg.sim.device = args_cli.device
    env_cfg.scene.num_envs = 1

    env = ManagerBasedRLEnv(cfg=env_cfg)

    robot = env.scene["robot"]
    arm_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)
    arm_cfg.resolve(env.scene)
    gripper_cfg = SceneEntityCfg("robot", joint_names=GRIPPER_JOINT_NAMES)
    gripper_cfg.resolve(env.scene)
    contact_sensors = [env.scene["gripper_jaw1_contact"], env.scene["gripper_jaw2_contact"]]
    cube = env.scene["cube"]

    link6_body_ids, link6_names_found = robot.find_bodies(["link_6"])
    print(f"[INFO] link_6 body id: {link6_names_found} -> {link6_body_ids}")
    link6_body_id = link6_body_ids[0]

    with torch.inference_mode():
        results = {}
        for point_label, point in VALIDATION_POINTS.items():
            results[point_label] = run_grasp_point(env, robot, arm_cfg, gripper_cfg, contact_sensors, cube, link6_body_id, point_label, point)

        print("\n" + "%" * 70)
        print("FINAL MULTI-POINT SUMMARY (Cartesian-corrected grasp confirm)")
        print("%" * 70)
        for point_label, r in results.items():
            print(
                f"  {point_label}: {r['verdict']}  (open_gripper_max_force={r['open_gripper_max_force']:.4f}N, "
                f"height_gain={r['height_gain_mm']:.2f}mm, held_through_retreat={r['held_through_retreat']}, "
                f"pinch_before={r['pinch_discrepancy_before_mm']:.3f}mm, pinch_after={r['pinch_discrepancy_after_mm']:.3f}mm, "
                f"cartesian_converged={r['cartesian_converged']}, "
                f"settle_iters(pregrasp/grasp)={r['pregrasp_settle_iters']}/{r['grasp_settle_iters']})"
            )
        all_confirmed = all(r["verdict"] == "GRASP+LIFT CONFIRMED" for r in results.values())
        print(f"ALL POINTS CONFIRMED: {all_confirmed}")
        print("%" * 70)

    simulation_app.close()


if __name__ == "__main__":
    main()
