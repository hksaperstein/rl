"""Direct root-cause check for the joint_2 ~59deg physical wall found by the
2026-07-28 ar4-joint-tracking-closed-loop-fix task
(kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md's matching UPDATE).

That task's stiffness-sweep/closed-loop data strongly inferred (via a
constant ~2.3N.m applied_torque that did NOT scale with a 20x stiffness
increase, and did not respond to a +26deg closed-loop overshoot command
either) that joint_2 is pinned at a genuine hard PhysX position-limit
constraint at ~59deg, NOT a finite-gain PD droop - but that ~59deg number
was never directly read off the built USD/live joint-limit data itself.
This script closes that gap with THREE independent, directly-measured
readbacks in one launch (no cube, no phases - fast, ~seconds of real
compute once the app is up):

  1. RAW USD prim inspection (pxr/UsdPhysics) of all 6 arm joints'
     physics:lowerLimit/physics:upperLimit attributes, read directly off
     the built ar4_mk5.usd stage via its own usd_path.txt manifest - the
     same direct-USD-introspection pattern
     docs/superpowers/specs/research/2026-07-21-ar4-usd-asset-debugging.md
     and scripts/_inspect_jaw_axis_math.py already established for the
     gripper joints, applied here to the arm joints instead.
  2. Isaac Lab's own live articulation-level joint_pos_limits/
     soft_joint_pos_limits (same fields scripts/_diag_ar4_joint_limits_readback.py
     already reads, reused verbatim here rather than re-derived, so this
     script also completes that still-open diagnostic in the same run).
  3. tasks/ar4/robot_cfg.py's own AR4_MK5_CFG confirms
     enabled_self_collisions=False and soft_joint_pos_limit_factor=1.0 -
     printed here as a reminder that self-collision and soft-limit
     narrowing are BOTH already ruled out as confounds by this project's
     own existing config (self-collision is disabled outright; soft
     limits equal hard limits) - so if the raw USD limit itself reads
     ~90deg (matching vendor), the ~59deg wall must have some THIRD
     mechanism not yet identified, worth flagging explicitly rather than
     silently assuming the joint-limit reading is the whole story.

All 6 joints checked (not just joint_2) per this task's own brief:
scripts/ar4_graspable_workspace.py may have other wrong hardcoded ranges
beyond joint_2.

.. code-block:: bash

    flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 /workspace/isaaclab/isaaclab.sh -p scripts/_diag_ar4_joint2_limit_root_cause.py --headless"
"""

import math
import os
import sys

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser(description="Root-cause joint_2's ~59deg physical wall: raw USD limits vs. Isaac Lab live limits vs. vendor spec.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch  # noqa: E402
from pxr import Usd  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402

_RL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _RL_ROOT)  # noqa: E402

from tasks.ar4.pickplace_graspgoal_env_cfg import Ar4PickPlaceGraspGoalEnvCfg  # noqa: E402
from tasks.ar4.robot_cfg import AR4_MK5_CFG, ARM_JOINT_NAMES  # noqa: E402

# Vendor spec (config/mk3.yaml, fetched directly from
# github.com/ycheng517/ar4_ros_driver 2026-07-28, cross-checked against
# urdf/ar_macro.xacro's own <joint>/<limit> tags which pull these exact
# same robot_parameters keys - i.e. these numbers ARE the vendor's raw
# URDF joint limits, not a separate/derived source). This project's own
# prior work (2026-07-22 "ar4-tilt-fix task" Part A) already established
# these limits are identical across all 5 shipped model variants
# (mk1-mk5).
VENDOR_JOINT_LIMITS_DEG = {
    "joint_1": (-170.0, 170.0),
    "joint_2": (-42.0, 90.0),
    "joint_3": (-89.0, 52.0),
    "joint_4": (-180.0, 180.0),
    "joint_5": (-105.0, 105.0),
    "joint_6": (-180.0, 180.0),
}

# scripts/ar4_graspable_workspace.py's own currently-assumed table (as of
# this task's start) - copied verbatim for direct 3-way comparison.
WORKSPACE_TOOL_ASSUMED_DEG = {
    "joint_1": (-170.0, 170.0),
    "joint_2": (-42.0, 90.0),
    "joint_3": (-89.0, 52.0),
    "joint_4": (-180.0, 180.0),
    "joint_5": (-105.0, 105.0),
    "joint_6": (-180.0, 180.0),
}


def main() -> None:
    print("\n" + "=" * 78)
    print("PART 0: robot_cfg.py's own self-collision / soft-limit config (context)")
    print("=" * 78)
    spawn_props = AR4_MK5_CFG.spawn.articulation_props
    print(f"  enabled_self_collisions = {spawn_props.enabled_self_collisions}")
    print(f"  soft_joint_pos_limit_factor = {AR4_MK5_CFG.soft_joint_pos_limit_factor}")

    print("\n" + "=" * 78)
    print("PART 1: RAW USD physics:lowerLimit/upperLimit (direct pxr/UsdPhysics read)")
    print("=" * 78)
    usd_manifest = os.path.join(_RL_ROOT, "assets", "ar4_mk5", "usd_path.txt")
    with open(usd_manifest) as f:
        usd_path = f.read().strip()
    print(f"  USD path: {usd_path}")
    stage = Usd.Stage.Open(usd_path)
    raw_usd_deg = {}
    for name in ARM_JOINT_NAMES:
        prim = stage.GetPrimAtPath(f"/mk5/root_joint/joints/{name}")
        if not prim.IsValid():
            print(f"  {name}: PRIM NOT FOUND at /mk5/root_joint/joints/{name}")
            continue
        lower = prim.GetAttribute("physics:lowerLimit").Get()
        upper = prim.GetAttribute("physics:upperLimit").Get()
        lower_deg = math.degrees(lower) if lower is not None else None
        upper_deg = math.degrees(upper) if upper is not None else None
        raw_usd_deg[name] = (lower_deg, upper_deg)
        vendor_lo, vendor_hi = VENDOR_JOINT_LIMITS_DEG[name]
        mismatch = (
            lower_deg is None
            or upper_deg is None
            or abs(lower_deg - vendor_lo) > 0.5
            or abs(upper_deg - vendor_hi) > 0.5
        )
        print(
            f"  {name}: RAW USD=[{lower_deg}, {upper_deg}]deg  VENDOR=[{vendor_lo}, {vendor_hi}]deg  "
            f"{'*** MISMATCH ***' if mismatch else 'match'}"
        )

    print("\n" + "=" * 78)
    print("PART 2: Isaac Lab live robot.data.joint_pos_limits / soft_joint_pos_limits")
    print("=" * 78)
    env_cfg = Ar4PickPlaceGraspGoalEnvCfg()
    env_cfg.sim.device = args_cli.device
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRLEnv(cfg=env_cfg)
    robot = env.scene["robot"]
    arm_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)
    arm_cfg.resolve(env.scene)

    with torch.inference_mode():
        env.reset()
        hard_limits = robot.data.joint_pos_limits[0, arm_cfg.joint_ids].tolist()
        soft_limits = robot.data.soft_joint_pos_limits[0, arm_cfg.joint_ids].tolist()
        default_q = robot.data.default_joint_pos[0, arm_cfg.joint_ids].tolist()

        for i, name in enumerate(ARM_JOINT_NAMES):
            hard_lo_deg = math.degrees(hard_limits[i][0])
            hard_hi_deg = math.degrees(hard_limits[i][1])
            soft_lo_deg = math.degrees(soft_limits[i][0])
            soft_hi_deg = math.degrees(soft_limits[i][1])
            raw_lo, raw_hi = raw_usd_deg.get(name, (None, None))
            agree_with_raw = (
                raw_lo is not None
                and abs(hard_lo_deg - raw_lo) < 0.5
                and abs(hard_hi_deg - raw_hi) < 0.5
            )
            print(
                f"  {name}: LIVE hard=[{hard_lo_deg:8.3f}, {hard_hi_deg:8.3f}]deg  "
                f"LIVE soft=[{soft_lo_deg:8.3f}, {soft_hi_deg:8.3f}]deg  "
                f"default_q={math.degrees(default_q[i]):7.3f}deg  "
                f"{'agrees with raw USD read' if agree_with_raw else '*** DISAGREES WITH RAW USD READ ***'}"
            )

    print("\n" + "=" * 78)
    print("PART 3: workspace-tool assumed table vs. vendor (all 6 joints)")
    print("=" * 78)
    for name in ARM_JOINT_NAMES:
        tool_lo, tool_hi = WORKSPACE_TOOL_ASSUMED_DEG[name]
        vendor_lo, vendor_hi = VENDOR_JOINT_LIMITS_DEG[name]
        mismatch = abs(tool_lo - vendor_lo) > 0.5 or abs(tool_hi - vendor_hi) > 0.5
        print(
            f"  {name}: TOOL=[{tool_lo}, {tool_hi}]deg  VENDOR=[{vendor_lo}, {vendor_hi}]deg  "
            f"{'*** MISMATCH ***' if mismatch else 'match'}"
        )

    print("\n" + "=" * 78)
    print("SUMMARY: does the ~59deg observed wall match the RAW USD upper limit for joint_2?")
    print("=" * 78)
    j2_raw_lo, j2_raw_hi = raw_usd_deg.get("joint_2", (None, None))
    if j2_raw_hi is not None:
        print(f"  joint_2 RAW USD upper limit = {j2_raw_hi:.3f}deg (observed physical wall ~59.0-59.1deg)")
        if abs(j2_raw_hi - 59.0) < 2.0:
            print("  -> CONFIRMED: the ~59deg wall IS the raw USD hard limit (an asset-import bug vs. the 90deg vendor spec).")
        elif abs(j2_raw_hi - 90.0) < 2.0:
            print("  -> The raw USD limit matches the 90deg vendor spec. The ~59deg wall is NOT the joint's own hard limit -")
            print("     self-collision is disabled (Part 0) and soft limits equal hard limits (factor=1.0), so a THIRD")
            print("     mechanism (not yet identified) must be responsible. Flagged for further investigation.")
        else:
            print(f"  -> Raw USD limit ({j2_raw_hi:.3f}deg) matches NEITHER the ~59deg wall NOR the 90deg vendor spec - genuinely unexplained, flag for further investigation.")

    simulation_app.close()


if __name__ == "__main__":
    main()
