"""Minimal, lightweight variant of _diag_ar4_joint2_limit_root_cause.py: ONLY
the raw pxr/UsdPhysics prim read (Part 1 of that script), using the SAME
lightweight `isaacsim.SimulationApp` bootstrap (no AppLauncher, no
ManagerBasedRLEnv/robot articulation construction) that
scripts/_inspect_jaw_axis_math.py already used successfully for the gripper
joints in the 2026-07-21/23 asset-debugging sessions - no report of a hang
in that prior work, unlike the full-env construction path.

Added 2026-07-28 after _diag_ar4_joint2_limit_root_cause.py hit the same
documented "CPU pinned ~110%, GPU 0%, log stale" cold-start stall THREE
times in a row on nvcr.io/nvidia/isaac-lab:2.3.1 + g2-standard-4/nvidia-l4,
every time before printing even its own first PART-0 line - i.e. the hang
is somewhere in ManagerBasedRLEnv/articulation-view construction (or,
possibly, coincidentally timed AppLauncher extension loading), not in
this script's own diagnostic logic. This variant sidesteps the heavier
construction path entirely to at least answer the raw-USD-vs-vendor
question, even if the live Isaac-Lab-level joint_pos_limits readback (Part
2 of the other script) still needs a working full-env launch separately.

.. code-block:: bash

    flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 /workspace/isaaclab/isaaclab.sh -p scripts/_diag_ar4_joint2_limit_raw_usd_only.py"
"""

import math
import os

_RL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
usd_manifest = os.path.join(_RL_ROOT, "assets", "ar4_mk5", "usd_path.txt")
with open(usd_manifest) as f:
    usd_path = f.read().strip()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": True})

from pxr import Usd  # noqa: E402

ARM_JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]

# Vendor spec (config/mk3.yaml, fetched directly from
# github.com/ycheng517/ar4_ros_driver 2026-07-28; cross-checked against
# urdf/ar_macro.xacro's own <joint>/<limit> tags, which pull these exact
# same robot_parameters keys - i.e. these numbers ARE the vendor's raw
# URDF joint limits). This project's own prior work (2026-07-22
# "ar4-tilt-fix task" Part A) already established these limits are
# identical across all 5 shipped model variants (mk1-mk5).
VENDOR_JOINT_LIMITS_DEG = {
    "joint_1": (-170.0, 170.0),
    "joint_2": (-42.0, 90.0),
    "joint_3": (-89.0, 52.0),
    "joint_4": (-180.0, 180.0),
    "joint_5": (-105.0, 105.0),
    "joint_6": (-180.0, 180.0),
}

print("\n" + "=" * 78)
print(f"RAW USD physics:lowerLimit/upperLimit read from {usd_path}")
print("=" * 78)
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
print("SUMMARY: does the ~59deg observed physical wall match the RAW USD upper limit for joint_2?")
print("=" * 78)
j2_raw_lo, j2_raw_hi = raw_usd_deg.get("joint_2", (None, None))
if j2_raw_hi is not None:
    print(f"  joint_2 RAW USD upper limit = {j2_raw_hi:.3f}deg (observed physical wall ~59.0-59.1deg)")
    if abs(j2_raw_hi - 59.0) < 2.0:
        print("  -> CONFIRMED: the ~59deg wall IS the raw USD hard limit (an asset-import bug vs. the 90deg vendor spec).")
    elif abs(j2_raw_hi - 90.0) < 2.0:
        print("  -> The raw USD limit matches the 90deg vendor spec. The ~59deg wall is NOT the joint's own hard limit -")
        print("     self-collision is disabled and soft limits equal hard limits per robot_cfg.py, so a THIRD")
        print("     mechanism (not yet identified) must be responsible. Flagged for further investigation.")
    else:
        print(f"  -> Raw USD limit ({j2_raw_hi:.3f}deg) matches NEITHER the ~59deg wall NOR the 90deg vendor spec - genuinely unexplained, flag for further investigation.")

print("\nDONE.")
simulation_app.close()
