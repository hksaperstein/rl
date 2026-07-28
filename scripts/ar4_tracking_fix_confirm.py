"""Close the AR4 arm's commanded-vs-achieved joint-tracking gap, then
re-attempt the grasp+lift with the arm ACTUALLY reaching the pose
(2026-07-27, ar4-joint-tracking-closed-loop-fix task).

Direct continuation of the ar4-joint-tracking-diagnostic task
(kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md's matching
UPDATE), which confirmed and quantified a real root cause underlying this
whole investigation's history: the physics arm does not actually reach a
commanded joint pose. Even with the "boosted" arm actuator gains
(stiffness=4000, damping=200) every confirm/grasp script in this
investigation already uses, the arm settles ~19.65mm short of the
FK-computed pinch point at the P0 grasp configuration - larger than the
15mm cube. This script:

  PART 1 - STIFFNESS SWEEP (no cube, isolates pure tracking from any
  contact confound - same method as ar4_joint_tracking_diagnostic.py,
  reused/extended here): sweeps stiffness upward (4000 -> 20000 -> 80000,
  damping scaled ~sqrt(stiffness) for a roughly-constant damping RATIO,
  i.e. damping = 200 * sqrt(stiffness/4000)) commanding P0's exact
  FK-computed GRASP_Q with the cube parked 3m away, measuring per-joint
  tracking error and achieved-vs-FK-predicted pinch-point discrepancy at
  each gain level. Also writes a 10x boosted effort_limit_sim (20 -> 200)
  at the highest stiffness tested, to directly check whether
  robot_cfg.py's shipped effort_limit_sim=20.0 is itself a binding
  saturation ceiling independent of stiffness (a real possibility: PD
  torque = stiffness * error, so a fixed torque ceiling means ever-higher
  stiffness alone eventually stops helping once torque saturates) -
  robot.data.applied_torque is also printed at each regime to make this
  directly checkable rather than merely inferred.

  PART 2 - CLOSED-LOOP SETTLE PRIMITIVE (no cube): tests
  tasks/ar4/joint_tracking.py's settle_to_joint_pose - a bounded outer
  integral-correction loop that measures achieved-vs-desired error after
  each settle and re-commands with an accumulated correction, the
  standard fix for a finite-gain PD controller's steady-state error
  under a persistent load (here: gravity), regardless of the underlying
  cause - at the SAME baseline boosted gain (4000/200) every prior script
  in this investigation already uses, to see whether the primitive alone
  (without needing extreme/possibly-unstable gains) closes the gap.

  DECISION (programmatic, logged not asked - this is a single unattended
  cloud run): if the best swept HIGH-STIFFNESS regime achieves a pinch-
  point discrepancy under 3mm AND produced no NaN/instability, that
  regime is used as the fix for Part 3. Otherwise the CLOSED-LOOP
  primitive (at the safe, already-proven-stable 4000/200 baseline gain)
  is used instead. Either way, the winning method's PREGRASP-OPEN/
  GRASP-OPEN phases in Part 3 use settle_to_joint_pose to guarantee
  genuine convergence before checking collision-free-ness - the whole
  point of this task.

  PART 3 - FULL GRASP+LIFT RE-ATTEMPT (cube present) at all 3 of the
  roll-constraint task's own validation points (P0/P1/P2 -
  scripts/ar4_graspable_workspace_confirm_numeric.py's own
  VALIDATION_POINTS, reused verbatim, not re-derived): HOME (naive
  drive) -> PREGRASP-OPEN (settle_to_joint_pose) -> GRASP-OPEN
  (settle_to_joint_pose, tracking BOTH jaws' contact force throughout via
  an on_step callback - any nonzero force here while nominally OPEN is
  exactly the "collides while open" bug this whole task exists to fix)
  -> GRASP-CLOSE (naive drive, holding the ALREADY-CONVERGED corrected
  GRASP target - no need to re-run the settle loop, the correction found
  during GRASP-OPEN already compensates gravity droop at that exact pose)
  -> LIFT-CLOSE / HOLD-CLOSE (naive drive, holding the already-converged
  corrected PREGRASP target) -> RETREAT-CLOSE (naive drive to HOME,
  uncorrected - precision doesn't matter once retreating away).

Reuses tasks/ar4/pickplace_graspgoal_env_cfg.py's Ar4PickPlaceGraspGoalEnvCfg
(contact sensors + cube, no cameras - this is the numeric/fast variant;
a separate camera-enabled script produces the demo video only if this one
confirms a real grasp+lift, to avoid paying camera-render cost/risk on an
exploratory run that might not even succeed).

.. code-block:: bash

    flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 /workspace/isaaclab/isaaclab.sh -p scripts/ar4_tracking_fix_confirm.py --headless"
"""

import math
import os
import sys

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser(description="Close the AR4 joint-tracking gap (stiffness sweep + closed-loop settle primitive), then re-attempt grasp+lift.")
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
# P0/P1/P2 - copied verbatim from
# scripts/ar4_graspable_workspace_confirm_numeric.py's own VALIDATION_POINTS
# (do not re-derive - this task is about actually REACHING these
# already-computed configs, not recomputing them).
# ----------------------------------------------------------------------
VALIDATION_POINTS = {
    "P0_recommended_bearing94": {
        "cube_xy": (-0.023809635667085667, 0.3436444906384511),
        "grasp_q_deg": [-4.32433486449247, 62.32407343477408, 24.213659245578118, -15.360723852639964, 6.104608002441614, 99.52404978549994],
        "pregrasp_q_deg": [-4.32433486449247, 52.434178877076384, 27.87529831816333, -2.2493188573178595, 8.860705727687604, 100.52702739544871],
    },
    "P1_bearing108": {
        "cube_xy": (-0.11232492, 0.33524543),
        "grasp_q_deg": [-19.45950689021618, 62.35207729321578, 23.688773948380668, -75.37680902627864, 4.445222138626336, -86.48635560483278],
        "pregrasp_q_deg": [-19.45950689021618, 52.372107173615724, 28.140503691604852, -60.985542108444285, 3.49827472818995, -91.03832753992285],
    },
    "P2_bearing98": {
        "cube_xy": (-0.05084859, 0.36347686),
        "grasp_q_deg": [-8.648669728984965, 62.53091830515946, 21.273536133775046, 91.73174064021492, -3.2675925657736515, -3.90406760984636],
        "pregrasp_q_deg": [-8.648669728984965, 52.83970843010327, 25.023756746773174, 119.70508232798234, -4.401980769256796, -26.572406591495163],
    },
}

CUBE_XY_PARK_FAR_AWAY = (3.0, 3.0)
HOME_Q_DEG = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
HOME_Q = [math.radians(d) for d in HOME_Q_DEG]

GRIPPER_OPEN_POS = 0.014
GRIPPER_CLOSED_POS = 0.0
GRIPPER_OPEN_EXPR = {"gripper_jaw1_joint": GRIPPER_OPEN_POS, "gripper_jaw2_joint": GRIPPER_OPEN_POS}
GRIPPER_CLOSED_EXPR = {"gripper_jaw1_joint": GRIPPER_CLOSED_POS, "gripper_jaw2_joint": GRIPPER_CLOSED_POS}

EE_OFFSET_LOCAL_LIST = [0.0, 0.0, 0.036]  # matches grasp_demo_v2.py/_EE_OFFSET


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


# ======================================================================
# PART 1: STIFFNESS SWEEP (no cube)
# ======================================================================
def run_stiffness_sweep(env, robot, arm_cfg, gripper_cfg, link6_body_id, cube, grasp_q_deg):
    print("\n" + "#" * 70)
    print("# PART 1: STIFFNESS SWEEP (no cube, P0 GRASP_Q)")
    print("#" * 70)

    grasp_q = [math.radians(d) for d in grasp_q_deg]
    joint_values_commanded = {name: grasp_q[i] for i, name in enumerate(ARM_JOINT_NAMES)}
    fk_pred_pinch_w = fk_predicted_pinch_point_world(joint_values_commanded).tolist()

    regimes = [
        ("stiffness4000", 4000.0, 200.0, 20.0),
        ("stiffness20000", 20000.0, 200.0 * math.sqrt(20000.0 / 4000.0), 20.0),
        ("stiffness80000", 80000.0, 200.0 * math.sqrt(80000.0 / 4000.0), 20.0),
        ("stiffness80000_effort200", 80000.0, 200.0 * math.sqrt(80000.0 / 4000.0), 200.0),
    ]

    results = []
    for label, stiffness, damping, effort_limit in regimes:
        print(f"\n--- regime {label}: stiffness={stiffness:.1f} damping={damping:.2f} effort_limit_sim={effort_limit:.1f} ---")
        env.reset()
        override_z = cube.data.root_pos_w[0, 2].item()
        override_pos = torch.tensor([[CUBE_XY_PARK_FAR_AWAY[0], CUBE_XY_PARK_FAR_AWAY[1], override_z]], device=env.device)
        override_quat = cube.data.root_quat_w[0:1].clone()
        cube.write_root_pose_to_sim(torch.cat([override_pos, override_quat], dim=-1), env_ids=torch.tensor([0], device=env.device))
        cube.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device), env_ids=torch.tensor([0], device=env.device))

        n_arm = len(arm_cfg.joint_ids)
        robot.write_joint_stiffness_to_sim(torch.full((1, n_arm), stiffness, device=env.device), joint_ids=arm_cfg.joint_ids)
        robot.write_joint_damping_to_sim(torch.full((1, n_arm), damping, device=env.device), joint_ids=arm_cfg.joint_ids)
        robot.write_joint_effort_limit_to_sim(torch.full((1, n_arm), effort_limit, device=env.device), joint_ids=arm_cfg.joint_ids)

        gripper_target_t = torch.tensor([[GRIPPER_OPEN_EXPR[n] for n in GRIPPER_JOINT_NAMES]], device=env.device)

        def _drive(target_q, duration):
            target_t = torch.tensor([target_q], device=env.device)
            for i in range(duration):
                robot.set_joint_position_target(target_t, joint_ids=arm_cfg.joint_ids)
                robot.set_joint_position_target(gripper_target_t, joint_ids=gripper_cfg.joint_ids)
                robot.write_data_to_sim()
                env.sim.step(render=False)
                robot.update(env.physics_dt)

        _drive(HOME_Q, 60)
        _drive(grasp_q, 90)
        _drive(grasp_q, 200)  # extra-settle, watching for genuine convergence

        achieved_q = robot.data.joint_pos[0, arm_cfg.joint_ids].tolist()
        per_joint_err_deg = [math.degrees(a - b) for a, b in zip(achieved_q, grasp_q)]
        max_err_deg = max(abs(e) for e in per_joint_err_deg)
        achieved_pinch_w = _achieved_pinch_world(robot, link6_body_id)
        pinch_disc_mm = _dist_mm(achieved_pinch_w, fk_pred_pinch_w)
        applied_torque = robot.data.applied_torque[0, arm_cfg.joint_ids].tolist()
        is_nan = any(v != v for v in achieved_q)  # NaN check (NaN != NaN)

        print(f"  achieved_q(deg)={[round(math.degrees(v), 3) for v in achieved_q]}")
        print(f"  per_joint_err_deg={[round(e, 4) for e in per_joint_err_deg]}  MAX={max_err_deg:.4f}deg")
        print(f"  applied_torque(Nm)={[round(t, 3) for t in applied_torque]}")
        print(f"  pinch_point_discrepancy_mm={pinch_disc_mm:.3f}")
        print(f"  NaN detected: {is_nan}")

        results.append({
            "label": label, "stiffness": stiffness, "damping": damping, "effort_limit": effort_limit,
            "max_err_deg": max_err_deg, "pinch_discrepancy_mm": pinch_disc_mm,
            "applied_torque": applied_torque, "is_nan": is_nan,
        })

    print("\n=== PART 1 SUMMARY ===")
    for r in results:
        print(f"  {r['label']}: max_joint_err={r['max_err_deg']:.4f}deg pinch_discrepancy={r['pinch_discrepancy_mm']:.3f}mm NaN={r['is_nan']}")
    return results


# ======================================================================
# PART 2: CLOSED-LOOP SETTLE PRIMITIVE TEST (no cube)
# ======================================================================
def run_closed_loop_test(env, robot, arm_cfg, gripper_cfg, link6_body_id, cube, grasp_q_deg):
    print("\n" + "#" * 70)
    print("# PART 2: CLOSED-LOOP settle_to_joint_pose TEST (no cube, P0 GRASP_Q, baseline stiffness=4000/damping=200)")
    print("#" * 70)

    grasp_q = [math.radians(d) for d in grasp_q_deg]
    joint_values_commanded = {name: grasp_q[i] for i, name in enumerate(ARM_JOINT_NAMES)}
    fk_pred_pinch_w = fk_predicted_pinch_point_world(joint_values_commanded).tolist()

    env.reset()
    override_z = cube.data.root_pos_w[0, 2].item()
    override_pos = torch.tensor([[CUBE_XY_PARK_FAR_AWAY[0], CUBE_XY_PARK_FAR_AWAY[1], override_z]], device=env.device)
    override_quat = cube.data.root_quat_w[0:1].clone()
    cube.write_root_pose_to_sim(torch.cat([override_pos, override_quat], dim=-1), env_ids=torch.tensor([0], device=env.device))
    cube.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device), env_ids=torch.tensor([0], device=env.device))

    n_arm = len(arm_cfg.joint_ids)
    robot.write_joint_stiffness_to_sim(torch.full((1, n_arm), 4000.0, device=env.device), joint_ids=arm_cfg.joint_ids)
    robot.write_joint_damping_to_sim(torch.full((1, n_arm), 200.0, device=env.device), joint_ids=arm_cfg.joint_ids)
    robot.write_joint_effort_limit_to_sim(torch.full((1, n_arm), 20.0, device=env.device), joint_ids=arm_cfg.joint_ids)

    gripper_target_t = [GRIPPER_OPEN_EXPR[n] for n in GRIPPER_JOINT_NAMES]

    def _drive(target_q, duration):
        target_t = torch.tensor([target_q], device=env.device)
        g_t = torch.tensor([gripper_target_t], device=env.device)
        for i in range(duration):
            robot.set_joint_position_target(target_t, joint_ids=arm_cfg.joint_ids)
            robot.set_joint_position_target(g_t, joint_ids=gripper_cfg.joint_ids)
            robot.write_data_to_sim()
            env.sim.step(render=False)
            robot.update(env.physics_dt)

    _drive(HOME_Q, 60)

    result = settle_to_joint_pose(
        env, robot, arm_cfg.joint_ids, grasp_q,
        tol_rad=math.radians(0.15),  # ~0.15deg/joint - comfortably tighter than the ~1-2mm pinch target
        max_outer_iters=8, inner_settle_steps=150,
        integral_gain=1.0, integral_clamp=0.5,
        gripper_joint_ids=gripper_cfg.joint_ids, gripper_target=gripper_target_t,
        label="PART2-CLOSED-LOOP",
    )
    achieved_pinch_w = _achieved_pinch_world(robot, link6_body_id)
    pinch_disc_mm = _dist_mm(achieved_pinch_w, fk_pred_pinch_w)

    print("\n=== PART 2 SUMMARY ===")
    print(f"  converged={result['converged']} n_outer_iters={result['n_outer_iters']} max_err_deg={result['max_err_deg']:.4f}")
    print(f"  final correction (deg): {[round(math.degrees(c), 3) for c in result['correction']]}")
    print(f"  pinch_point_discrepancy_mm={pinch_disc_mm:.3f}")
    result["pinch_discrepancy_mm"] = pinch_disc_mm
    return result


# ======================================================================
# PART 3: FULL GRASP+LIFT RE-ATTEMPT (cube present) using the winning method
# ======================================================================
def run_grasp_point(env, robot, arm_cfg, gripper_cfg, contact_sensors, cube, point_label, point, stiffness, damping, effort_limit):
    print("\n" + "#" * 70)
    print(f"# PART 3: GRASP+LIFT RE-ATTEMPT: {point_label} (stiffness={stiffness} damping={damping:.1f} effort_limit={effort_limit})")
    print("#" * 70)

    grasp_q = [math.radians(d) for d in point["grasp_q_deg"]]
    pregrasp_q = [math.radians(d) for d in point["pregrasp_q_deg"]]
    cube_xy = point["cube_xy"]

    env.reset()
    override_z = cube.data.root_pos_w[0, 2].item()
    override_pos = torch.tensor([[cube_xy[0], cube_xy[1], override_z]], device=env.device)
    override_quat = cube.data.root_quat_w[0:1].clone()
    cube.write_root_pose_to_sim(torch.cat([override_pos, override_quat], dim=-1), env_ids=torch.tensor([0], device=env.device))
    cube.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device), env_ids=torch.tensor([0], device=env.device))
    cube_z_on_table = cube.data.root_pos_w[0, 2].item()
    print(f"[INFO] Cube teleported to: {override_pos[0].tolist()}, resting height={cube_z_on_table:.4f}m")

    n_arm = len(arm_cfg.joint_ids)
    robot.write_joint_stiffness_to_sim(torch.full((1, n_arm), stiffness, device=env.device), joint_ids=arm_cfg.joint_ids)
    robot.write_joint_damping_to_sim(torch.full((1, n_arm), damping, device=env.device), joint_ids=arm_cfg.joint_ids)
    robot.write_joint_effort_limit_to_sim(torch.full((1, n_arm), effort_limit, device=env.device), joint_ids=arm_cfg.joint_ids)

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

    def _drive_naive(target_q, gripper_expr, duration, label, is_pre_close=False):
        target_t = torch.tensor([target_q], device=env.device)
        g_t = torch.tensor([[gripper_expr[n] for n in GRIPPER_JOINT_NAMES]], device=env.device)
        for i in range(duration):
            robot.set_joint_position_target(target_t, joint_ids=arm_cfg.joint_ids)
            robot.set_joint_position_target(g_t, joint_ids=gripper_cfg.joint_ids)
            robot.write_data_to_sim()
            env.sim.step(render=False)
            robot.update(env.physics_dt)
            j1, j2 = _track_forces(is_pre_close)
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

    _drive_naive(HOME_Q, GRIPPER_OPEN_EXPR, 60, "PHASE0-HOME-OPEN", is_pre_close=True)
    cube_z_by_phase["PHASE0-HOME-OPEN"] = cube.data.root_pos_w[0, 2].item()

    pregrasp_result = _settle_tracked(pregrasp_q, GRIPPER_OPEN_EXPR, "PHASE1-PREGRASP-OPEN-SETTLE")
    cube_z_by_phase["PHASE1-PREGRASP-OPEN"] = cube.data.root_pos_w[0, 2].item()
    corrected_pregrasp_target = [d + c for d, c in zip(pregrasp_q, pregrasp_result["correction"])]
    print(f"[INFO] PREGRASP settle: converged={pregrasp_result['converged']} iters={pregrasp_result['n_outer_iters']} max_err_deg={pregrasp_result['max_err_deg']:.4f}")

    grasp_result = _settle_tracked(grasp_q, GRIPPER_OPEN_EXPR, "PHASE2-GRASP-OPEN-SETTLE")
    cube_z_by_phase["PHASE2-GRASP-OPEN"] = cube.data.root_pos_w[0, 2].item()
    corrected_grasp_target = [d + c for d, c in zip(grasp_q, grasp_result["correction"])]
    print(f"[INFO] GRASP settle: converged={grasp_result['converged']} iters={grasp_result['n_outer_iters']} max_err_deg={grasp_result['max_err_deg']:.4f}")
    print(f"[INFO] open_gripper_max_force so far (PHASE0-2, pre-CLOSE): {force_tracker['open_gripper_max_force']:.4f}N (pre-fix bug was 45-61N)")

    cube_z_by_phase["PHASE3-GRASP-CLOSE"] = _drive_naive(corrected_grasp_target, GRIPPER_CLOSED_EXPR, 90, "PHASE3-GRASP-CLOSE")
    cube_z_by_phase["PHASE4-LIFT-CLOSE"] = _drive_naive(corrected_pregrasp_target, GRIPPER_CLOSED_EXPR, 120, "PHASE4-LIFT-CLOSE")
    cube_z_by_phase["PHASE5-HOLD-CLOSE"] = _drive_naive(corrected_pregrasp_target, GRIPPER_CLOSED_EXPR, 120, "PHASE5-HOLD-CLOSE")
    cube_z_by_phase["PHASE6-RETREAT-CLOSE"] = _drive_naive(HOME_Q, GRIPPER_CLOSED_EXPR, 150, "PHASE6-RETREAT-CLOSE")

    print("\n" + "=" * 70)
    print(f"SUMMARY: {point_label}")
    print("=" * 70)
    for label, z in cube_z_by_phase.items():
        print(f"  cube_z at end of {label}: {z:.4f}m")
    height_gain = max(cube_z_by_phase.values()) - cube_z_on_table
    final_z = cube_z_by_phase["PHASE6-RETREAT-CLOSE"]
    held_through_retreat = final_z > cube_z_on_table + 0.01
    real_lift = height_gain > 0.01
    both_jaws_contacted = force_tracker["jaw1_max"] > 0.001 and force_tracker["jaw2_max"] > 0.001
    open_gripper_clean = force_tracker["open_gripper_max_force"] < 1.0
    print(f"Cube resting height (table): {cube_z_on_table:.4f}m")
    print(f"Max cube height reached: {max(cube_z_by_phase.values()):.4f}m (gain={height_gain*1000:.2f}mm)")
    print(f"Final cube height (end of RETREAT): {final_z:.4f}m")
    print(f"Max jaw-cube contact force WHILE GRIPPER STILL OPEN (pre-CLOSE): {force_tracker['open_gripper_max_force']:.4f}N")
    print(f"Max jaw1/jaw2 contact force (any phase): {force_tracker['jaw1_max']:.4f}N / {force_tracker['jaw2_max']:.4f}N")
    print(f"BOTH jaws registered real contact force (post-close): {both_jaws_contacted}")
    print(f"Real height gain (>1cm): {real_lift}")
    print(f"Held through retreat (>1cm above table): {held_through_retreat}")
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

    p0_grasp_q_deg = VALIDATION_POINTS["P0_recommended_bearing94"]["grasp_q_deg"]

    with torch.inference_mode():
        sweep_results = run_stiffness_sweep(env, robot, arm_cfg, gripper_cfg, link6_body_id, cube, p0_grasp_q_deg)
        closed_loop_result = run_closed_loop_test(env, robot, arm_cfg, gripper_cfg, link6_body_id, cube, p0_grasp_q_deg)

        # ---- Programmatic decision (no human in the loop this run) ----
        # Only consider the plain sweep regimes for "high stiffness alone"
        # (exclude the effort-limit variant, which is a distinct diagnostic,
        # not a candidate fix on its own per this task's own dispatch
        # instructions).
        plain_sweep = [r for r in sweep_results if r["label"] != "stiffness80000_effort200"]
        best_stiffness_regime = min(plain_sweep, key=lambda r: r["pinch_discrepancy_mm"])
        print("\n" + "=" * 70)
        print("DECISION")
        print("=" * 70)
        print(f"Best swept high-stiffness regime: {best_stiffness_regime['label']} "
              f"(pinch_discrepancy={best_stiffness_regime['pinch_discrepancy_mm']:.3f}mm, NaN={best_stiffness_regime['is_nan']})")
        print(f"Closed-loop primitive (baseline 4000/200): converged={closed_loop_result['converged']} "
              f"pinch_discrepancy={closed_loop_result['pinch_discrepancy_mm']:.3f}mm")

        if (not best_stiffness_regime["is_nan"]) and best_stiffness_regime["pinch_discrepancy_mm"] < 3.0:
            winning_method = "HIGH_STIFFNESS"
            winning_stiffness = best_stiffness_regime["stiffness"]
            winning_damping = best_stiffness_regime["damping"]
            winning_effort_limit = 20.0
            print(f"DECISION: HIGH_STIFFNESS alone reaches <3mm cleanly ({best_stiffness_regime['pinch_discrepancy_mm']:.3f}mm) -> using {best_stiffness_regime['label']} for Part 3.")
        else:
            winning_method = "CLOSED_LOOP"
            winning_stiffness = 4000.0
            winning_damping = 200.0
            winning_effort_limit = 20.0
            print(f"DECISION: HIGH_STIFFNESS alone does NOT cleanly reach <3mm (best={best_stiffness_regime['pinch_discrepancy_mm']:.3f}mm) "
                  f"-> using CLOSED_LOOP settle_to_joint_pose at baseline 4000/200 for Part 3.")
        print("=" * 70)

        # ---- PART 3: full grasp+lift re-attempt at all 3 points ----
        results = {}
        for point_label, point in VALIDATION_POINTS.items():
            results[point_label] = run_grasp_point(
                env, robot, arm_cfg, gripper_cfg, contact_sensors, cube, point_label, point,
                winning_stiffness, winning_damping, winning_effort_limit,
            )

        print("\n" + "%" * 70)
        print("FINAL MULTI-POINT SUMMARY")
        print("%" * 70)
        print(f"WINNING METHOD: {winning_method} (stiffness={winning_stiffness}, damping={winning_damping:.1f})")
        for point_label, r in results.items():
            print(f"  {point_label}: {r['verdict']}  (open_gripper_max_force={r['open_gripper_max_force']:.4f}N, "
                  f"height_gain={r['height_gain_mm']:.2f}mm, held_through_retreat={r['held_through_retreat']}, "
                  f"settle_iters(pregrasp/grasp)={r['pregrasp_settle_iters']}/{r['grasp_settle_iters']})")
        all_confirmed = all(r["verdict"] == "GRASP+LIFT CONFIRMED" for r in results.values())
        print(f"ALL POINTS CONFIRMED: {all_confirmed}")
        print("%" * 70)

    simulation_app.close()


if __name__ == "__main__":
    main()
