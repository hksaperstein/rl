"""Numeric-only (no camera/video) variant of ar4_graspable_workspace_confirm.py.

Added 2026-07-27 after the camera-enabled version genuinely hung on a fresh
GCP instance: PHASE 4 (LIFT) never produced its own first log line, the
output video files' mtimes stopped advancing entirely, and GPU utilization
stayed at a real, nonzero ~60% for 16+ minutes with zero forward progress
in either the log or the video files - a genuine stall, not "slow but
progressing" (this project's own documented "first Isaac-Sim-rendering-
touching docker run can take 10+ minutes" gap does not, on its own, explain
an indefinite stall with zero incremental output). Isolating whether the
stall is specifically the camera/render pipeline (most likely, given the
Vulkan/GL history this same dispatch already fought) by dropping cameras
entirely and driving joints directly (same low-level pattern as
scripts/_verify_gripper_mirror_fix.py: robot.set_joint_position_target +
write_data_to_sim + env.sim.step(render=False), bypassing both the camera
render pipeline AND the action-manager tensor interface) rather than
reusing ar4_graspable_workspace_confirm.py's ManagerBasedEnv/env.step(action)
pattern, which implicitly renders every scene camera each step regardless of
whether the script's own code reads camera.data.output.

Uses tasks/ar4/pickplace_graspgoal_env_cfg.py's Ar4PickPlaceGraspGoalEnvCfg
(contact sensors + cube, NO cameras - already proven camera-free by
_verify_gripper_mirror_fix.py) instead of grasp_verify_env_cfg.py's
Ar4GraspVerifyEnvCfg used by the camera version. This script does not read
or care about that cfg's own reward/termination terms (a ManagerBasedRLEnv
is still used only because pickplace_graspgoal_env_cfg.py is written as
one, matching _verify_gripper_mirror_fix.py's own precedent) - joints are
driven directly, not via the manager's action tensor.

2026-07-27 UPDATE (ar4-graspable-roll-constraint task): now validates
MULTIPLE points (VALIDATION_POINTS below) in a single Isaac Sim launch (one
env.reset() + cube re-teleport per point, no relaunch) - proves the
roll-constrained workspace fix generalizes across genuinely distinct
Stage-A survivors, not just one lucky point, without paying cloud
provisioning/startup cost 3x. Also explicitly tracks
``open_gripper_max_force`` (max jaw-cube contact force during PHASE0-2,
i.e. BEFORE the CLOSE command) separately from the overall max - this is
the exact quantity the prior (roll-unconstrained) sweep's bug showed as
52-61N even though the gripper was nominally open; this task's whole point
is confirming that number is now ~0.

.. code-block:: bash

    flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 /workspace/isaaclab/isaaclab.sh -p scripts/ar4_graspable_workspace_confirm_numeric.py --headless"
"""

import math
import os
import sys

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser(description="Numeric-only live grasp+lift confirmation at the FK-chosen graspable workspace point(s).")
parser.add_argument(
    "--cube-xy", type=float, nargs=2, default=None, metavar=("X", "Y"),
    help="Override the recommended cube world (x,y) - if set, runs ONLY this single point instead of the full VALIDATION_POINTS list.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from tasks.ar4.pickplace_graspgoal_env_cfg import Ar4PickPlaceGraspGoalEnvCfg  # noqa: E402
from tasks.ar4.robot_cfg import ARM_JOINT_NAMES, GRIPPER_JOINT_NAMES  # noqa: E402

HOME_Q_DEG = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# 2026-07-27 UPDATE (ar4-graspable-roll-constraint task): three genuinely
# DISTINCT Stage-A survivors (different joint_2-6, not just different
# joint_1 rotations of the same underlying config) from the roll-constrained
# re-sweep of scripts/ar4_graspable_workspace.py (ROLL_TOL_DEG=12deg from
# parallel to world X/Y), all within the scene's existing bearing~90deg
# "straight ahead" window. P0 is the primary recommended point (best overall
# margin); P1/P2 are independent points chosen to prove the fix generalizes
# across the region rather than being a single lucky point. GRASP_Q from
# scripts/ar4_graspable_workspace.py's own sweep output; PREGRASP_Q from
# scripts/_ar4_pregrasp_search_roll_constrained.py's local-perturbation
# search (same method as the original, roll-unconstrained PREGRASP_Q
# search, now also filtered by the same roll criterion).
VALIDATION_POINTS = {
    "P0_recommended_bearing94": {
        "cube_xy": (-0.023809635667085667, 0.3436444906384511),
        "grasp_q_deg": [-4.32433486449247, 62.32407343477408, 24.213659245578118, -15.360723852639964, 6.104608002441614, 99.52404978549994],
        "pregrasp_q_deg": [-4.32433486449247, 52.434178877076384, 27.87529831816333, -2.2493188573178595, 8.860705727687604, 100.52702739544871],
        "tilt_deg": 2.91, "roll_offset_deg": 10.08, "min_margin_deg": 27.68,
    },
    "P1_bearing108": {
        "cube_xy": (-0.11232492, 0.33524543),  # world xy of P1's own GRASP_Q FK pinch point (see report)
        "grasp_q_deg": [-19.45950689021618, 62.35207729321578, 23.688773948380668, -75.37680902627864, 4.445222138626336, -86.48635560483278],
        "pregrasp_q_deg": [-19.45950689021618, 52.372107173615724, 28.140503691604852, -60.985542108444285, 3.49827472818995, -91.03832753992285],
        "tilt_deg": 5.15, "roll_offset_deg": 1.06, "min_margin_deg": 27.65,
    },
    "P2_bearing98": {
        "cube_xy": (-0.05084859, 0.36347686),  # world xy of P2's own GRASP_Q FK pinch point (see report)
        "grasp_q_deg": [-8.648669728984965, 62.53091830515946, 21.273536133775046, 91.73174064021492, -3.2675925657736515, -3.90406760984636],
        "pregrasp_q_deg": [-8.648669728984965, 52.83970843010327, 25.023756746773174, 119.70508232798234, -4.401980769256796, -26.572406591495163],
        "tilt_deg": 6.91, "roll_offset_deg": 10.83, "min_margin_deg": 27.47,
    },
}

HOME_Q = [math.radians(d) for d in HOME_Q_DEG]

GRIPPER_OPEN_POS = 0.014
GRIPPER_CLOSED_POS = 0.0
GRIPPER_OPEN_EXPR = {"gripper_jaw1_joint": GRIPPER_OPEN_POS, "gripper_jaw2_joint": GRIPPER_OPEN_POS}
GRIPPER_CLOSED_EXPR = {"gripper_jaw1_joint": GRIPPER_CLOSED_POS, "gripper_jaw2_joint": GRIPPER_CLOSED_POS}

# Phases that occur BEFORE the CLOSE command - any nonzero contact force
# here is the exact "collides while nominally OPEN" bug this task is
# checking has actually been fixed.
_PRE_CLOSE_PHASE_LABELS = {"PHASE0-HOME-OPEN", "PHASE1-PREGRASP-OPEN", "PHASE2-GRASP-OPEN"}


def _drive(env, robot, arm_cfg, gripper_cfg, contact_sensors, cube, arm_target, gripper_target_expr, duration, label, force_tracker):
    arm_target_t = torch.tensor([arm_target], device=env.device)
    gripper_target_t = torch.tensor(
        [[gripper_target_expr[n] for n in GRIPPER_JOINT_NAMES]], device=env.device
    )
    is_pre_close = label in _PRE_CLOSE_PHASE_LABELS
    for i in range(duration):
        robot.set_joint_position_target(arm_target_t, joint_ids=arm_cfg.joint_ids)
        robot.set_joint_position_target(gripper_target_t, joint_ids=gripper_cfg.joint_ids)
        robot.write_data_to_sim()
        env.sim.step(render=False)
        robot.update(env.physics_dt)
        for sensor in contact_sensors:
            sensor.update(env.physics_dt, force_recompute=True)
        jaw1_force = contact_sensors[0].data.net_forces_w[0, 0].norm().item()
        jaw2_force = contact_sensors[1].data.net_forces_w[0, 0].norm().item()
        force_tracker["jaw1_max"] = max(force_tracker["jaw1_max"], jaw1_force)
        force_tracker["jaw2_max"] = max(force_tracker["jaw2_max"], jaw2_force)
        if is_pre_close:
            force_tracker["open_gripper_max_force"] = max(
                force_tracker["open_gripper_max_force"], jaw1_force, jaw2_force
            )
        if i % 20 == 0 or i == duration - 1:
            cube_z = cube.data.root_pos_w[0, 2].item()
            cube_xy = cube.data.root_pos_w[0, :2].tolist()
            live_arm_q = robot.data.joint_pos[0, arm_cfg.joint_ids].tolist()
            max_track_err = max(abs(a - b) for a, b in zip(live_arm_q, arm_target))
            gripper_q = robot.data.joint_pos[0, gripper_cfg.joint_ids].tolist()
            print(
                f"  [{label} step {i:3d}] cube_z={cube_z:.4f}m cube_xy={['%.4f' % x for x in cube_xy]} "
                f"jaw1_cube_force={jaw1_force:.4f}N jaw2_cube_force={jaw2_force:.4f}N "
                f"gripper_q={['%.5f' % v for v in gripper_q]} max_arm_track_err={max_track_err:.4f}rad"
            )
    return cube.data.root_pos_w[0, 2].item()


def _run_one_point(env, robot, arm_cfg, gripper_cfg, contact_sensors, cube, point_label, point):
    print("\n" + "#" * 70)
    print(f"# VALIDATION POINT: {point_label}")
    print("#" * 70)

    grasp_q = [math.radians(d) for d in point["grasp_q_deg"]]
    pregrasp_q = [math.radians(d) for d in point["pregrasp_q_deg"]]
    cube_xy = point["cube_xy"]

    env.reset()
    override_z = cube.data.root_pos_w[0, 2].item()
    override_pos = torch.tensor([[cube_xy[0], cube_xy[1], override_z]], device=env.device)
    override_quat = cube.data.root_quat_w[0:1].clone()
    cube.write_root_pose_to_sim(
        torch.cat([override_pos, override_quat], dim=-1), env_ids=torch.tensor([0], device=env.device)
    )
    cube.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device), env_ids=torch.tensor([0], device=env.device))
    print(f"[INFO] Cube teleported to: {override_pos[0].tolist()}")
    cube_z_on_table = cube.data.root_pos_w[0, 2].item()
    print(f"[INFO] Cube resting height (table): {cube_z_on_table:.4f}m")
    print(f"[INFO] GRASP_Q (deg): {point['grasp_q_deg']}")
    print(f"[INFO] PREGRASP_Q (deg): {point['pregrasp_q_deg']}")
    print(f"[INFO] Pre-computed (FK sweep): tilt={point['tilt_deg']}deg roll_offset={point['roll_offset_deg']}deg min_margin={point['min_margin_deg']}deg")

    PHASES = [
        (60, HOME_Q, GRIPPER_OPEN_EXPR, "PHASE0-HOME-OPEN"),
        (150, pregrasp_q, GRIPPER_OPEN_EXPR, "PHASE1-PREGRASP-OPEN"),
        (90, grasp_q, GRIPPER_OPEN_EXPR, "PHASE2-GRASP-OPEN"),
        (90, grasp_q, GRIPPER_CLOSED_EXPR, "PHASE3-GRASP-CLOSE"),
        (120, pregrasp_q, GRIPPER_CLOSED_EXPR, "PHASE4-LIFT-CLOSE"),
        (120, pregrasp_q, GRIPPER_CLOSED_EXPR, "PHASE5-HOLD-CLOSE"),
        (150, HOME_Q, GRIPPER_CLOSED_EXPR, "PHASE6-RETREAT-CLOSE"),
    ]

    cube_z_by_phase = {}
    force_tracker = {"jaw1_max": 0.0, "jaw2_max": 0.0, "open_gripper_max_force": 0.0}
    for duration, target_q, gripper_expr, label in PHASES:
        print(f"\n[{point_label}/{label}] duration={duration} target_q(deg)={[round(math.degrees(v), 2) for v in target_q]}")
        final_z = _drive(env, robot, arm_cfg, gripper_cfg, contact_sensors, cube, target_q, gripper_expr, duration, label, force_tracker)
        cube_z_by_phase[label] = final_z

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
    open_gripper_clean = force_tracker["open_gripper_max_force"] < 1.0  # newtons - was 52-61N pre-fix
    print(f"Cube resting height (table): {cube_z_on_table:.4f}m")
    print(f"Max cube height reached (any phase-end): {max(cube_z_by_phase.values()):.4f}m (gain={height_gain*1000:.2f}mm)")
    print(f"Final cube height (end of RETREAT): {final_z:.4f}m")
    print(f"Max jaw-cube contact force WHILE GRIPPER STILL OPEN (phases 0-2, pre-CLOSE): {force_tracker['open_gripper_max_force']:.4f}N (pre-fix bug was 52-61N)")
    print(f"Max jaw1-cube contact force observed (any step, any phase): {force_tracker['jaw1_max']:.4f}N")
    print(f"Max jaw2-cube contact force observed (any step, any phase): {force_tracker['jaw2_max']:.4f}N")
    print(f"BOTH jaws registered real contact force (post-close): {both_jaws_contacted}")
    print(f"Real height gain (>1cm) at some phase end: {real_lift}")
    print(f"Cube still elevated at end of RETREAT (>1cm above table): {held_through_retreat}")
    print(f"Open-gripper collision-free (<1N while nominally open): {open_gripper_clean}")
    verdict = (
        "GRASP+LIFT CONFIRMED" if (open_gripper_clean and both_jaws_contacted and real_lift and held_through_retreat)
        else "GRASP+LIFT NOT CONFIRMED"
    )
    print(f"VERDICT [{point_label}]: {verdict}")
    print("=" * 70)
    return {
        "verdict": verdict,
        "open_gripper_max_force": force_tracker["open_gripper_max_force"],
        "jaw1_max": force_tracker["jaw1_max"],
        "jaw2_max": force_tracker["jaw2_max"],
        "height_gain_mm": height_gain * 1000,
        "held_through_retreat": held_through_retreat,
    }


def main() -> None:
    env_cfg = Ar4PickPlaceGraspGoalEnvCfg()
    env_cfg.sim.device = args_cli.device
    env_cfg.scene.num_envs = 1

    # Test-local actuator-gain boost (NOT touching the shared
    # tasks/ar4/robot_cfg.py) - established necessary by this investigation
    # for the arm to actually track a commanded multi-joint pose under
    # gravity (see kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md's
    # 2026-07-22 "later, same day" UPDATE).
    env_cfg.scene.robot.actuators["arm"].stiffness = 4000.0
    env_cfg.scene.robot.actuators["arm"].damping = 200.0

    env = ManagerBasedRLEnv(cfg=env_cfg)

    robot = env.scene["robot"]
    arm_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)
    arm_cfg.resolve(env.scene)
    gripper_cfg = SceneEntityCfg("robot", joint_names=GRIPPER_JOINT_NAMES)
    gripper_cfg.resolve(env.scene)
    contact_sensors = [env.scene["gripper_jaw1_contact"], env.scene["gripper_jaw2_contact"]]
    cube = env.scene["cube"]

    with torch.inference_mode():
        if args_cli.cube_xy is not None:
            # Single-point override mode (backwards-compatible with the
            # original single-point CLI usage) - build an ad hoc point using
            # P0's own joint configs (only cube_xy overridden).
            point = dict(VALIDATION_POINTS["P0_recommended_bearing94"])
            point["cube_xy"] = tuple(args_cli.cube_xy)
            results = {"override": _run_one_point(env, robot, arm_cfg, gripper_cfg, contact_sensors, cube, "override", point)}
        else:
            results = {}
            for point_label, point in VALIDATION_POINTS.items():
                results[point_label] = _run_one_point(env, robot, arm_cfg, gripper_cfg, contact_sensors, cube, point_label, point)

        print("\n" + "%" * 70)
        print("FINAL MULTI-POINT SUMMARY")
        print("%" * 70)
        for point_label, r in results.items():
            print(f"  {point_label}: {r['verdict']}  (open_gripper_max_force={r['open_gripper_max_force']:.4f}N, "
                  f"height_gain={r['height_gain_mm']:.2f}mm, held_through_retreat={r['held_through_retreat']})")
        all_confirmed = all(r["verdict"] == "GRASP+LIFT CONFIRMED" for r in results.values())
        print(f"ALL POINTS CONFIRMED: {all_confirmed}")
        print("%" * 70)

    simulation_app.close()


if __name__ == "__main__":
    main()
