# scripts/_verify_gripper_fk_integration.py
"""One-off, non-permanent integration check (fk-verification-framework task,
2026-07-23): confirms tasks/ar4/fk_verification.py's standing Layer 1/Layer 2
FK framework against the ACTUAL, LIVE, already-fixed Isaac Sim AR4 asset -
not just the pure-numpy unit tests in tests/test_ar4_fk_verification.py.

The gripper jaw-mirroring fix (tasks/ar4/robot_cfg.py's
GRIPPER_OPEN_COMMAND_EXPR/GRIPPER_CLOSED_COMMAND_EXPR, plus the jaw2 PhysX
drive fix, scripts/build_asset.py's _add_gripper_jaw2_drive) has already
landed (confirmed via git log before writing this script - commits
f1f79f0/928af41/2576e94/c87f90a already present on both the Pi checkout and
this desktop checkout at HEAD c87f90a). This script re-runs the framework's
own Layer 1 (assert_link_pose_matches_vendor_fk) and Layer 2
(assert_gripper_separation) checks directly against real
env.reset()/env.step() state, reusing scripts/_verify_gripper_mirror_fix.py's
already-validated scene-construction/settle pattern (same env cfg, same
arm-actuator-stiffness boost to avoid the known gravity-sag confound
documented there).

Runs non-headless per this repo's standing convention (CLAUDE.md
"Environment conventions") - requires DISPLAY=:1.

.. code-block:: bash

    flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 DISPLAY=:1 /home/saps/IsaacLab/isaaclab.sh -p scripts/_verify_gripper_fk_integration.py"
"""

import os
import sys

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser(description="FK-verification-framework live integration check.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.utils.math import subtract_frame_transforms  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from tasks.ar4.pickplace_graspgoal_env_cfg import Ar4PickPlaceGraspGoalEnvCfg  # noqa: E402
from tasks.ar4.robot_cfg import (  # noqa: E402
    ARM_JOINT_NAMES,
    GRIPPER_JOINT_NAMES,
    GRIPPER_OPEN_COMMAND_EXPR,
)
from tasks.ar4.fk_verification import (  # noqa: E402
    assert_gripper_separation,
    assert_link_pose_matches_vendor_fk,
)


def _settle(env, robot, gripper_cfg, arm_cfg, arm_hold_target, target_expr, n_steps=60):
    target = torch.tensor([[target_expr[name] for name in GRIPPER_JOINT_NAMES]], device=env.device)
    for _ in range(n_steps):
        robot.set_joint_position_target(arm_hold_target, joint_ids=arm_cfg.joint_ids)
        robot.set_joint_position_target(target, joint_ids=gripper_cfg.joint_ids)
        robot.write_data_to_sim()
        env.sim.step(render=False)
        robot.update(env.physics_dt)


def main() -> None:
    env_cfg = Ar4PickPlaceGraspGoalEnvCfg()
    env_cfg.sim.device = args_cli.device
    env_cfg.scene.num_envs = 1
    # Same test-local-only arm-actuator-stiffness boost as
    # scripts/_verify_gripper_mirror_fix.py, for the same documented reason
    # (the arm's own real actuator gains are too weak to hold its pose
    # statically against gravity in this static single-target diagnostic -
    # NOT touching the shared robot_cfg.py).
    env_cfg.scene.robot.actuators["arm"].stiffness = 4000.0
    env_cfg.scene.robot.actuators["arm"].damping = 200.0

    env = ManagerBasedRLEnv(cfg=env_cfg)
    robot = env.scene["robot"]

    print("=" * 70)
    print(f"robot.data.body_names = {robot.data.body_names}")
    print("=" * 70)

    gripper_cfg = SceneEntityCfg("robot", joint_names=GRIPPER_JOINT_NAMES)
    gripper_cfg.resolve(env.scene)
    arm_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)
    arm_cfg.resolve(env.scene)

    body_names_needed = ["base_link", "link_1", "link_6", "gripper_jaw1_link", "gripper_jaw2_link"]
    body_ids = {}
    for n in body_names_needed:
        if n in robot.data.body_names:
            body_ids[n] = robot.data.body_names.index(n)
        else:
            print(f"WARNING: body {n!r} not found in robot.data.body_names - skipping checks that need it")

    with torch.inference_mode():
        env.reset()

    arm_hold_target = robot.data.joint_pos[0, arm_cfg.joint_ids].unsqueeze(0).clone()

    with torch.inference_mode():
        _settle(env, robot, gripper_cfg, arm_cfg, arm_hold_target, GRIPPER_OPEN_COMMAND_EXPR, n_steps=60)

    # Build the live joint_values dict fk_verification.py's API expects,
    # from the actual, real, live articulation state (not a synthetic
    # value) - all arm joints + both gripper joints.
    with torch.inference_mode():
        joint_values = {}
        for name in ARM_JOINT_NAMES:
            idx = robot.data.joint_names.index(name)
            joint_values[name] = robot.data.joint_pos[0, idx].item()
        for name in GRIPPER_JOINT_NAMES:
            idx = robot.data.joint_names.index(name)
            joint_values[name] = robot.data.joint_pos[0, idx].item()

    print("=" * 70)
    print(f"LIVE joint_values (post-settle, commanded OPEN) = {joint_values}")
    print("=" * 70)

    # ---- Layer 2: real jaw separation, from the real settled sim state ----
    if "gripper_jaw1_link" in body_ids and "gripper_jaw2_link" in body_ids:
        with torch.inference_mode():
            p1w = robot.data.body_pos_w[0, body_ids["gripper_jaw1_link"]].cpu().numpy()
            p2w = robot.data.body_pos_w[0, body_ids["gripper_jaw2_link"]].cpu().numpy()
        import numpy as np

        real_separation_mm = float(np.linalg.norm(p1w - p2w) * 1000.0)
        print(f"REAL (Isaac world-frame) measured jaw separation = {real_separation_mm:.3f}mm")

        try:
            predicted_mm = assert_gripper_separation(joint_values, min_mm=20.0, max_mm=36.0)
            print(f"Layer 2 assert_gripper_separation: PASS - FK-predicted separation = {predicted_mm:.3f}mm "
                  f"(real measured = {real_separation_mm:.3f}mm)")
        except AssertionError as exc:
            print(f"Layer 2 assert_gripper_separation: FAIL - {exc}")

    # ---- Layer 1: FK-predicted link pose (in base_link frame) vs. live ----
    if "base_link" in body_ids:
        with torch.inference_mode():
            base_pos_w = robot.data.body_pos_w[0, body_ids["base_link"]]
            base_quat_w = robot.data.body_quat_w[0, body_ids["base_link"]]

        for link_name in ["link_1", "link_6", "gripper_jaw1_link", "gripper_jaw2_link"]:
            if link_name not in body_ids:
                continue
            with torch.inference_mode():
                link_pos_w = robot.data.body_pos_w[0, body_ids[link_name]]
                link_quat_w = robot.data.body_quat_w[0, body_ids[link_name]]
                pos_b, quat_b = subtract_frame_transforms(
                    base_pos_w.unsqueeze(0), base_quat_w.unsqueeze(0),
                    link_pos_w.unsqueeze(0), link_quat_w.unsqueeze(0),
                )
                pos_b = pos_b[0].cpu().numpy()
                quat_b = quat_b[0].cpu().numpy()

            try:
                result = assert_link_pose_matches_vendor_fk(
                    pos_b, quat_b, joint_values, link_name, tolerance_mm=5.0
                )
                print(
                    f"Layer 1 assert_link_pose_matches_vendor_fk({link_name}): PASS - "
                    f"discrepancy={result.pos_discrepancy_mm:.3f}mm (tolerance=5.0mm), "
                    f"rot_discrepancy={result.rot_discrepancy_rad:.4f}rad"
                )
            except AssertionError as exc:
                print(f"Layer 1 assert_link_pose_matches_vendor_fk({link_name}): FAIL - {exc}")
    else:
        print("Skipping Layer 1 checks - no 'base_link' body found (see WARNING above).")

    print("=" * 70)
    simulation_app.close()


if __name__ == "__main__":
    main()
