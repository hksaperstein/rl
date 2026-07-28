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

Same GRASP_Q/PREGRASP_Q FK-forward-sampled joint configs and cube position
as the camera version - see that script's own module docstring for the full
provenance (scripts/ar4_graspable_workspace.py's 8M-sample sweep).

.. code-block:: bash

    flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 /workspace/isaaclab/isaaclab.sh -p scripts/ar4_graspable_workspace_confirm_numeric.py --headless"
"""

import math
import os
import sys

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser(description="Numeric-only live grasp+lift confirmation at the FK-chosen graspable workspace point.")
parser.add_argument(
    "--cube-xy", type=float, nargs=2, default=None, metavar=("X", "Y"),
    help="Override the recommended cube world (x,y) - defaults to the FK sweep's own recommended point.",
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

# Same FK-recommended point/joint configs as ar4_graspable_workspace_confirm.py
# - see that script's own module docstring for full provenance.
RECOMMENDED_CUBE_XY = (-0.11270833064477809, 0.3255077063604289)
HOME_Q_DEG = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
GRASP_Q_DEG = [-19.45950689021618, 62.32407343477408, 24.213659245578118, -15.360723852639964, 6.104608002441614, 99.52404978549994]
PREGRASP_Q_DEG = [-19.45950689021618, 52.360982999717486, 28.492202965999496, -14.980745756538473, 7.161794376970106, 132.48857602752358]

HOME_Q = [math.radians(d) for d in HOME_Q_DEG]
GRASP_Q = [math.radians(d) for d in GRASP_Q_DEG]
PREGRASP_Q = [math.radians(d) for d in PREGRASP_Q_DEG]

GRIPPER_OPEN_POS = 0.014
GRIPPER_CLOSED_POS = 0.0
GRIPPER_OPEN_EXPR = {"gripper_jaw1_joint": GRIPPER_OPEN_POS, "gripper_jaw2_joint": GRIPPER_OPEN_POS}
GRIPPER_CLOSED_EXPR = {"gripper_jaw1_joint": GRIPPER_CLOSED_POS, "gripper_jaw2_joint": GRIPPER_CLOSED_POS}


def _drive(env, robot, arm_cfg, gripper_cfg, contact_sensors, jaw_body_ids, cube, arm_target, gripper_target_expr, duration, label, force_tracker):
    arm_target_t = torch.tensor([arm_target], device=env.device)
    gripper_target_t = torch.tensor(
        [[gripper_target_expr[n] for n in GRIPPER_JOINT_NAMES]], device=env.device
    )
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
    jaw_body_ids = [robot.data.body_names.index(n) for n in ["gripper_jaw1_link", "gripper_jaw2_link"]]
    contact_sensors = [env.scene["gripper_jaw1_contact"], env.scene["gripper_jaw2_contact"]]
    cube = env.scene["cube"]

    with torch.inference_mode():
        env.reset()

        cube_xy = args_cli.cube_xy if args_cli.cube_xy is not None else RECOMMENDED_CUBE_XY
        override_z = cube.data.root_pos_w[0, 2].item()
        override_pos = torch.tensor([[cube_xy[0], cube_xy[1], override_z]], device=env.device)
        override_quat = cube.data.root_quat_w[0:1].clone()
        cube.write_root_pose_to_sim(
            torch.cat([override_pos, override_quat], dim=-1), env_ids=torch.tensor([0], device=env.device)
        )
        cube.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device), env_ids=torch.tensor([0], device=env.device))
        print(f"[INFO] Cube teleported to FK-recommended world position: {override_pos[0].tolist()}")
        cube_z_on_table = cube.data.root_pos_w[0, 2].item()
        print(f"[INFO] Cube resting height (table): {cube_z_on_table:.4f}m")
        print(f"[INFO] GRASP_Q (deg): {GRASP_Q_DEG}")
        print(f"[INFO] PREGRASP_Q (deg): {PREGRASP_Q_DEG}")

        PHASES = [
            (60, HOME_Q, GRIPPER_OPEN_EXPR, "PHASE0-HOME-OPEN"),
            (150, PREGRASP_Q, GRIPPER_OPEN_EXPR, "PHASE1-PREGRASP-OPEN"),
            (90, GRASP_Q, GRIPPER_OPEN_EXPR, "PHASE2-GRASP-OPEN"),
            (90, GRASP_Q, GRIPPER_CLOSED_EXPR, "PHASE3-GRASP-CLOSE"),
            (120, PREGRASP_Q, GRIPPER_CLOSED_EXPR, "PHASE4-LIFT-CLOSE"),
            (120, PREGRASP_Q, GRIPPER_CLOSED_EXPR, "PHASE5-HOLD-CLOSE"),
            (150, HOME_Q, GRIPPER_CLOSED_EXPR, "PHASE6-RETREAT-CLOSE"),
        ]

        cube_z_by_phase = {}
        force_tracker = {"jaw1_max": 0.0, "jaw2_max": 0.0}
        for duration, target_q, gripper_expr, label in PHASES:
            print(f"\n[{label}] duration={duration} target_q(deg)={[round(math.degrees(v), 2) for v in target_q]}")
            final_z = _drive(env, robot, arm_cfg, gripper_cfg, contact_sensors, jaw_body_ids, cube, target_q, gripper_expr, duration, label, force_tracker)
            cube_z_by_phase[label] = final_z

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        for label, z in cube_z_by_phase.items():
            print(f"  cube_z at end of {label}: {z:.4f}m")
        height_gain = max(cube_z_by_phase.values()) - cube_z_on_table
        final_z = cube_z_by_phase["PHASE6-RETREAT-CLOSE"]
        held_through_retreat = final_z > cube_z_on_table + 0.01
        real_lift = height_gain > 0.01
        both_jaws_contacted = force_tracker["jaw1_max"] > 0.001 and force_tracker["jaw2_max"] > 0.001
        print(f"Cube resting height (table): {cube_z_on_table:.4f}m")
        print(f"Max cube height reached (any phase-end): {max(cube_z_by_phase.values()):.4f}m (gain={height_gain*1000:.2f}mm)")
        print(f"Final cube height (end of RETREAT): {final_z:.4f}m")
        print(f"Max jaw1-cube contact force observed (any step, any phase): {force_tracker['jaw1_max']:.4f}N")
        print(f"Max jaw2-cube contact force observed (any step, any phase): {force_tracker['jaw2_max']:.4f}N")
        print(f"BOTH jaws registered real contact force: {both_jaws_contacted}")
        print(f"Real height gain (>1cm) at some phase end: {real_lift}")
        print(f"Cube still elevated at end of RETREAT (>1cm above table): {held_through_retreat}")
        verdict = "GRASP+LIFT CONFIRMED" if (both_jaws_contacted and real_lift and held_through_retreat) else "GRASP+LIFT NOT CONFIRMED"
        print(f"VERDICT: {verdict}")
        print("=" * 70)

    simulation_app.close()


if __name__ == "__main__":
    main()
