# scripts/_verify_arm_chain_fk_integration.py
"""One-off, non-permanent integration check (ar4-arm-chain-fk-check task,
2026-07-24): extends tasks/ar4/fk_verification.py's Layer 1
(assert_link_pose_matches_vendor_fk) across EVERY AR4 arm link (link_1
through link_6 - NOT just the gripper jaws, which
scripts/_verify_gripper_fk_integration.py already covers) at multiple joint
configurations, to directly check whether the built USD asset's own arm
kinematic chain (joint origins/axes for joints 1-6, as actually imported by
scripts/build_asset.py) matches the vendor's own raw URDF/xacro FK
prediction.

Motivation: with the gripper geometry (jaw-mimic limits, jaw2 drive, jaw2
collision-mesh asymmetry) and the contact-sensing pipeline both independently
verified correct in prior sessions (see kb/wiki/concepts/
ar4-vs-franka-root-cause-comparison.md's 2026-07-21/22/24 UPDATEs), the
entire remaining blocker is a stubborn ~9-10mm position / ~4-7deg rotation
residual at the best-known classical-IK grasp configuration that has
resisted both tighter solver convergence (2026-07-24 ar4-grasp-ik-
convergence-tightening task: 15x the iteration budget and a 3-5x tighter
convergence bound found only a genuine, stable, WORSE local optimum
immediately adjacent to the already-converged point - not a slow
improvement cut short by budget) and a nearby-configuration neighborhood
search (same task: tilt/reach swept across a 64-68deg/0.32-0.36m
neighborhood, found a shallow ~1.5%-flat plateau, not a sharper nearby
optimum). Every previous debugging pass in this investigation assumed the
ARM's own kinematic chain geometry (as distinct from the gripper, already
verified) was correct without ever directly checking it against the vendor
source - only the elbow joint's LIMIT (a scalar range) was ever verified
against the vendor spec (2026-07-22 ar4-tilt-fix task, Part A). A wrong
joint origin/axis at an EARLY joint (e.g. joint_2 or joint_3) would compound
significantly by link_6, 3-4 links later - exactly the kind of defect that
could produce a residual no amount of IK-solver tuning could ever fix,
since the solver would be correctly solving for the WRONG kinematic model.

Checks, for each of several joint-value configurations (see CONFIGS below):
for every arm link (link_1..link_6), read the live Isaac-Sim body pose,
convert it into the robot's own base_link frame (matching Layer 1's own
documented frame convention), and compare against
tasks/ar4/fk_verification.py's independent vendor-URDF FK prediction for
the SAME joint values. Reports EXACT discrepancy numbers (not just
pass/fail) for every link at every configuration, using a tight 1.0mm
default tolerance (vs. the existing gripper integration check's looser
5.0mm - this task specifically wants to catch a "few mm at an early joint"
defect, so the bar needs to be tighter than what was sufficient to confirm
the already-known-correct gripper jaw geometry).

Runs non-headless for local/desktop use per this repo's standing convention
(CLAUDE.md "Environment conventions") - requires DISPLAY=:1. When dispatched
to a headless cloud instance instead (this repo's standing, confirmed
exception for cloud dispatch - see docs/cloud/dispatch-checklist.md), pass
--headless explicitly.

.. code-block:: bash

    flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 DISPLAY=:1 /home/saps/IsaacLab/isaaclab.sh -p scripts/_verify_arm_chain_fk_integration.py"
"""

import os
import sys

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser(description="Full AR4 arm-chain (link_1..link_6) FK-verification-framework live integration check.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import ManagerBasedEnv  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.utils.math import subtract_frame_transforms  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from tasks.ar4.grasp_verify_env_cfg import Ar4GraspVerifyEnvCfg  # noqa: E402
from tasks.ar4.robot_cfg import ARM_JOINT_NAMES  # noqa: E402
from tasks.ar4.fk_verification import assert_link_pose_matches_vendor_fk  # noqa: E402

ARM_LINK_NAMES = ["link_1", "link_2", "link_3", "link_4", "link_5", "link_6"]

# ----------------------------------------------------------------------
# Configurations to check - see module docstring for why each is included.
# ----------------------------------------------------------------------
HOME_Q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# The best-known classical-IK grasp configuration this investigation has
# found (65deg tilt, reach=0.36m - the middle of the confirmed-flat
# 0.30-0.36m plateau), taken directly from the 2026-07-24
# ar4-grasp-ik-convergence-tightening task's own [SUMMARY] line
# (logs/cloud_runs/grasp_convtighten_run2.log) - the actual live-converged
# joint config underlying the ~9.43mm/6.65deg residual this task is trying
# to root-cause, not a hand-picked or idealized value.
GRASP_Q = [-0.2032243013381958, 1.2570910453796387, 0.2663007080554962, 1.7793933153152466, -1.277768850326538, -0.04975356534123421]
PREGRASP_Q = [-0.19154343008995056, 1.100957989692688, 0.339489609003067, 1.8290504217147827, -1.1507694721221924, -0.15697519481182098]

# A synthetic "stress" configuration with a non-trivial, non-near-zero
# value on EVERY joint (GRASP_Q/PREGRASP_Q happen to have small values on
# some joints, e.g. joint_6, which would mask an origin/axis defect on
# that joint's own downstream link if it existed) - chosen well within
# every joint's own range, including joint_3's tightest-known limit
# ([-1.5533, +0.9076] rad, see kb/wiki/concepts/
# ar4-vs-franka-root-cause-comparison.md's 2026-07-22 Part A finding).
STRESS_Q = [0.4, 0.5, -0.3, 0.5, -0.5, 0.5]

CONFIGS = [
    ("HOME_Q (all-zero)", HOME_Q),
    ("PREGRASP_Q (best-known, converged)", PREGRASP_Q),
    ("GRASP_Q (best-known, converged, 65deg/0.36m)", GRASP_Q),
    ("STRESS_Q (synthetic, non-trivial on every joint)", STRESS_Q),
]

TOLERANCE_MM = 1.0
SETTLE_STEPS = 90


def _settle(env, robot, arm_cfg, target_q, n_steps=SETTLE_STEPS):
    target = torch.tensor([target_q], device=env.device)
    for _ in range(n_steps):
        robot.set_joint_position_target(target, joint_ids=arm_cfg.joint_ids)
        robot.write_data_to_sim()
        env.sim.step(render=False)
        robot.update(env.physics_dt)


def main() -> None:
    env_cfg = Ar4GraspVerifyEnvCfg()
    env_cfg.sim.device = args_cli.device

    # Same test-local-only arm-actuator-stiffness boost as
    # scripts/_verify_gripper_fk_integration.py / grasp_demo_v2.py, for the
    # same documented reason (the arm's own real actuator gains are too
    # weak to hold/reach a commanded pose statically against gravity in a
    # static single-target diagnostic like this one - NOT touching the
    # shared tasks/ar4/robot_cfg.py).
    env_cfg.scene.robot.actuators["arm"].stiffness = 4000.0
    env_cfg.scene.robot.actuators["arm"].damping = 200.0

    # tasks/ar4/grasp_verify_env_cfg.py's Ar4GraspVerifyEnvCfg scene adds two
    # cameras (for scripts/grasp_demo_v2.py's own video recording) - this
    # script needs neither their output nor --enable_cameras (which forces a
    # slow real RTX render-pipeline warmup/shader-compile on first
    # env.reset(), observed live to take 10+ minutes on a fresh cloud
    # instance with no benefit here). Drop them from the scene entirely
    # before env creation - the same pattern scripts/plot_arm_skeleton.py
    # already uses for an analogous camera-free diagnostic.
    env_cfg.scene.perception_camera = None
    env_cfg.scene.demo_camera = None

    env = ManagerBasedEnv(cfg=env_cfg)
    robot = env.scene["robot"]

    print("=" * 78)
    print(f"robot.data.body_names = {robot.data.body_names}")
    print("=" * 78)

    arm_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)
    arm_cfg.resolve(env.scene)

    body_ids = {}
    for n in ["base_link"] + ARM_LINK_NAMES:
        if n in robot.data.body_names:
            body_ids[n] = robot.data.body_names.index(n)
        else:
            print(f"WARNING: body {n!r} not found in robot.data.body_names - skipping checks that need it")

    with torch.inference_mode():
        env.reset()

    all_results = []  # (config_label, link_name, pos_mm, rot_rad, passed)
    max_pos_mm = -1.0
    max_pos_link = None
    max_pos_config = None

    for config_label, target_q in CONFIGS:
        print("=" * 78)
        print(f"CONFIG: {config_label}  q={['%.5f' % v for v in target_q]}")
        with torch.inference_mode():
            _settle(env, robot, arm_cfg, target_q)

        with torch.inference_mode():
            joint_values = {}
            for name in ARM_JOINT_NAMES:
                idx = robot.data.joint_names.index(name)
                joint_values[name] = robot.data.joint_pos[0, idx].item()
        print(f"  LIVE settled joint_values = {joint_values}")

        if "base_link" not in body_ids:
            print("  Skipping - no 'base_link' body found.")
            continue

        with torch.inference_mode():
            base_pos_w = robot.data.body_pos_w[0, body_ids["base_link"]]
            base_quat_w = robot.data.body_quat_w[0, body_ids["base_link"]]

        for link_name in ARM_LINK_NAMES:
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
                    pos_b, quat_b, joint_values, link_name, tolerance_mm=TOLERANCE_MM
                )
                passed = True
                pos_mm = result.pos_discrepancy_mm
                rot_rad = result.rot_discrepancy_rad
                print(
                    f"  [{link_name}] PASS - pos_discrepancy={pos_mm:.4f}mm "
                    f"(tolerance={TOLERANCE_MM}mm) rot_discrepancy={rot_rad:.5f}rad "
                    f"live_pos_b={np.array2string(pos_b, precision=5)} "
                )
            except AssertionError as exc:
                passed = False
                msg = str(exc)
                # Pull the discrepancy back out for the summary table even
                # on failure (the exception message already contains it,
                # but recompute cleanly for a machine-readable number).
                import re

                m = re.search(r"discrepancy=([0-9.]+)mm", msg)
                pos_mm = float(m.group(1)) if m else float("nan")
                m2 = re.search(r"rotation discrepancy=([0-9.]+)rad", msg)
                rot_rad = float(m2.group(1)) if m2 else float("nan")
                print(f"  [{link_name}] *** FAIL *** - {msg}")

            all_results.append((config_label, link_name, pos_mm, rot_rad, passed))
            if pos_mm > max_pos_mm:
                max_pos_mm = pos_mm
                max_pos_link = link_name
                max_pos_config = config_label

    print("=" * 78)
    print("SUMMARY TABLE (config, link, pos_discrepancy_mm, rot_discrepancy_rad, PASS/FAIL)")
    print("=" * 78)
    for config_label, link_name, pos_mm, rot_rad, passed in all_results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status:4s}  {config_label:48s}  {link_name:8s}  pos={pos_mm:8.4f}mm  rot={rot_rad:8.5f}rad")

    print("=" * 78)
    any_fail = any(not passed for (_, _, _, _, passed) in all_results)
    if any_fail:
        print(f"VERDICT: FAIL - at least one link/config exceeded the {TOLERANCE_MM}mm tolerance.")
    else:
        print(f"VERDICT: PASS - every arm link (link_1..link_6) matches the vendor-URDF FK prediction "
              f"to within {TOLERANCE_MM}mm at every tested configuration.")
    print(f"LARGEST single discrepancy observed: {max_pos_mm:.4f}mm at link={max_pos_link!r}, config={max_pos_config!r}")
    print("=" * 78)

    simulation_app.close()


if __name__ == "__main__":
    main()
