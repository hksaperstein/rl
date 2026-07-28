"""Closing grasp+lift attempt for the AR4 multi-week investigation
(2026-07-28, ar4-pedestal-ground-clearance-fix task) - re-attempts a real
grasp+lift now that the cube rests on a raised pedestal
(tasks/ar4/objects_cfg.py's PEDESTAL_CFG/PEDESTAL_HEIGHT_M), fixing the
2026-07-28 "ar4-joint2-ground-clearance-fix task" finding that a cube
resting flush on the ground (z=0) is genuinely ungraspable anywhere in AR4's
real workspace (best-case real fingertip clearance -3.55mm - always below
the floor).

Directly reuses (does not re-derive):
  - tasks/ar4/joint_tracking.py's settle_to_joint_pose, the closed-loop
    outer-integral-correction primitive validated by the 2026-07-28
    "ar4-joint-tracking-closed-loop-fix task" (sub-0.1deg convergence
    whenever the target is genuinely reachable - which GRASP_Q now is, at
    joint_2~55.7deg, safely under both the 90deg soft limit AND the
    ~59.0-59.1deg ground-collision wall that task found, since the pedestal
    removes that wall's own cause entirely).
  - scripts/ar4_tracking_fix_confirm.py's own PART 3 structure/verdict
    logic (HOME-OPEN -> PREGRASP-OPEN-SETTLE -> GRASP-OPEN-SETTLE ->
    GRASP-CLOSE -> LIFT-CLOSE -> HOLD-CLOSE -> RETREAT-CLOSE, contact-force
    tracking throughout, the same open_gripper_clean/both_jaws_contacted/
    real_lift/held_through_retreat verdict criteria) - this task's own
    scope is confirming the PEDESTAL fix, not re-deriving the grasp
    sequence or tracking-gap fix, both already validated.
  - tasks/ar4/pickplace_graspgoal_env_cfg.py's Ar4PickPlaceGraspGoalEnvCfg,
    now itself carrying the pedestal + repositioned cube spawn (2026-07-28
    edit, same task) - no camera (numeric/fast variant only; a separate
    camera-enabled script produces the demo video only if THIS script
    confirms a real grasp+lift, mirroring ar4_tracking_fix_confirm.py's own
    "don't pay render cost on an exploratory run that might not even
    succeed" rationale).

3 validation points (scripts/_ar4_pedestal_select_grasp_points.py's own
output, this task): genuinely distinct Stage-A FK survivors at 3 different
bearings (~95/108/80deg, not just different joint_1 sweeps of the identical
underlying config), all satisfying the full filter set (height/tilt/
margin/roll/ground-clearance) at the pedestal-corrected GRASP_AT_HEIGHT
(0.0505m = 0.040m pedestal + 0.0105m original convention).

.. code-block:: bash

    flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 /workspace/isaaclab/isaaclab.sh -p scripts/ar4_pedestal_grasp_confirm.py --headless"
"""

import math
import os
import sys

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser(description="Confirm AR4 grasp+lift now that the cube rests on a raised pedestal.")
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
from tasks.ar4.joint_tracking import settle_to_joint_pose  # noqa: E402

# ----------------------------------------------------------------------
# 3 validation points - copied verbatim from
# scripts/_ar4_pedestal_select_grasp_points.py's own printed JSON output
# (this task). Do not re-derive.
# ----------------------------------------------------------------------
VALIDATION_POINTS = {
    "Q0_bearing95": {
        "cube_xy": (-0.03268395855178342, 0.3736940930857331),
        "grasp_q_deg": [-6.486502296738718, 55.70023063033951, 17.653652481102057, -20.108691036880195, 21.589804778653495, 35.46511508237428],
        "pregrasp_q_deg": [-6.486502296738718, 45.81794492857074, 20.660026164077877, -17.35915293735889, 22.343296203133704, 28.24200378228704],
    },
    "Q1_bearing108": {
        "cube_xy": (-0.11574104570008845, 0.35681861876033505),
        "grasp_q_deg": [-19.45950689021618, 55.70023063033951, 17.653652481102057, -20.108691036880195, 21.589804778653495, 35.46511508237428],
        "pregrasp_q_deg": [-19.45950689021618, 45.85773669517647, 20.684872348477526, -15.720456929141802, 22.10412265564435, 38.91001080248476],
    },
    "Q2_bearing80": {
        "cube_xy": (0.06336711870675883, 0.37520332616782015),
        "grasp_q_deg": [10.810837161231213, 55.72469437225485, 17.23085927375185, 19.09519541084618, 18.83806156989572, -18.83102464322388],
        "pregrasp_q_deg": [10.810837161231213, 46.73117460993635, 17.930914264230953, 11.56407455024493, 26.947068307538117, -12.34071498788734],
    },
}

HOME_Q_DEG = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
HOME_Q = [math.radians(d) for d in HOME_Q_DEG]

GRIPPER_OPEN_POS = 0.014
GRIPPER_CLOSED_POS = 0.0
GRIPPER_OPEN_EXPR = {"gripper_jaw1_joint": GRIPPER_OPEN_POS, "gripper_jaw2_joint": GRIPPER_OPEN_POS}
GRIPPER_CLOSED_EXPR = {"gripper_jaw1_joint": GRIPPER_CLOSED_POS, "gripper_jaw2_joint": GRIPPER_CLOSED_POS}

EE_OFFSET_LOCAL_LIST = [0.0, 0.0, 0.036]  # matches grasp_demo_v2.py/_EE_OFFSET

# Baseline gain already validated by the 2026-07-28 ar4-joint-tracking-
# closed-loop-fix task as safe/effective whenever the target is genuinely
# reachable (which every GRASP_Q here now is, per this task's own pedestal
# fix) - not re-swept here, that tracking-gap question is already closed.
STIFFNESS = 4000.0
DAMPING = 200.0
EFFORT_LIMIT = 20.0


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


def run_grasp_point(env, robot, arm_cfg, gripper_cfg, contact_sensors, cube, point_label, point):
    print("\n" + "#" * 70)
    print(f"# GRASP+LIFT ATTEMPT (pedestal): {point_label}")
    print("#" * 70)

    grasp_q = [math.radians(d) for d in point["grasp_q_deg"]]
    pregrasp_q = [math.radians(d) for d in point["pregrasp_q_deg"]]
    cube_xy = point["cube_xy"]

    joint_values_commanded = {name: grasp_q[i] for i, name in enumerate(ARM_JOINT_NAMES)}
    fk_pred_pinch_w = fk_predicted_pinch_point_world(joint_values_commanded).tolist()

    env.reset()
    # Teleport cube to this point's xy, keeping its CURRENT (physically
    # settled, post-reset) resting z - same pattern as
    # ar4_tracking_fix_confirm.py, now resting on the pedestal rather than
    # the ground (Ar4PickPlaceGraspGoalEnvCfg's own scene cfg already places
    # both pedestal and cube's default spawn correctly - see
    # tasks/ar4/pickplace_graspgoal_env_cfg.py's 2026-07-28 edit).
    override_z = cube.data.root_pos_w[0, 2].item()
    override_pos = torch.tensor([[cube_xy[0], cube_xy[1], override_z]], device=env.device)
    override_quat = cube.data.root_quat_w[0:1].clone()
    cube.write_root_pose_to_sim(torch.cat([override_pos, override_quat], dim=-1), env_ids=torch.tensor([0], device=env.device))
    cube.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device), env_ids=torch.tensor([0], device=env.device))

    # Let the cube physically settle onto the pedestal before starting the
    # approach (a few free steps, gripper open, arm held at HOME) - confirms
    # the cube's real resting height on the pedestal directly rather than
    # assuming PEDESTAL_HEIGHT_M + 0.0075 exactly.
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

    cube_z_by_phase = {}
    cube_z_by_phase["PHASE0-HOME-OPEN"] = cube.data.root_pos_w[0, 2].item()

    pregrasp_result = _settle_tracked(pregrasp_q, GRIPPER_OPEN_EXPR, "PHASE1-PREGRASP-OPEN-SETTLE")
    cube_z_by_phase["PHASE1-PREGRASP-OPEN"] = cube.data.root_pos_w[0, 2].item()
    corrected_pregrasp_target = [d + c for d, c in zip(pregrasp_q, pregrasp_result["correction"])]
    print(f"[INFO] PREGRASP settle: converged={pregrasp_result['converged']} iters={pregrasp_result['n_outer_iters']} max_err_deg={pregrasp_result['max_err_deg']:.4f}")

    grasp_result = _settle_tracked(grasp_q, GRIPPER_OPEN_EXPR, "PHASE2-GRASP-OPEN-SETTLE")
    cube_z_by_phase["PHASE2-GRASP-OPEN"] = cube.data.root_pos_w[0, 2].item()
    corrected_grasp_target = [d + c for d, c in zip(grasp_q, grasp_result["correction"])]
    print(f"[INFO] GRASP settle: converged={grasp_result['converged']} iters={grasp_result['n_outer_iters']} max_err_deg={grasp_result['max_err_deg']:.4f}")
    print(f"[INFO] open_gripper_max_force so far (PHASE0-2, pre-CLOSE): {force_tracker['open_gripper_max_force']:.4f}N")

    achieved_pinch_w = _achieved_pinch_world(robot, robot.find_bodies(["link_6"])[0][0])
    pinch_disc_mm = _dist_mm(achieved_pinch_w, fk_pred_pinch_w)
    print(f"[INFO] GRASP pinch-point discrepancy vs FK prediction: {pinch_disc_mm:.3f}mm")

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
        "pinch_discrepancy_mm": pinch_disc_mm,
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

    with torch.inference_mode():
        results = {}
        for point_label, point in VALIDATION_POINTS.items():
            results[point_label] = run_grasp_point(env, robot, arm_cfg, gripper_cfg, contact_sensors, cube, point_label, point)

        print("\n" + "%" * 70)
        print("FINAL MULTI-POINT SUMMARY (pedestal grasp confirm)")
        print("%" * 70)
        for point_label, r in results.items():
            print(f"  {point_label}: {r['verdict']}  (open_gripper_max_force={r['open_gripper_max_force']:.4f}N, "
                  f"height_gain={r['height_gain_mm']:.2f}mm, held_through_retreat={r['held_through_retreat']}, "
                  f"pinch_discrepancy={r['pinch_discrepancy_mm']:.3f}mm, "
                  f"settle_iters(pregrasp/grasp)={r['pregrasp_settle_iters']}/{r['grasp_settle_iters']})")
        all_confirmed = all(r["verdict"] == "GRASP+LIFT CONFIRMED" for r in results.values())
        print(f"ALL POINTS CONFIRMED: {all_confirmed}")
        print("%" * 70)

    simulation_app.close()


if __name__ == "__main__":
    main()
