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
import sys
import traceback

_RL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
usd_manifest = os.path.join(_RL_ROOT, "assets", "ar4_mk5", "usd_path.txt")
with open(usd_manifest) as f:
    usd_path = f.read().strip()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": True})

from pxr import Usd  # noqa: E402

ARM_JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]

# Belt-and-suspenders result capture: a prior attempt at this exact script
# produced ZERO of its own print() output in the tee'd cloud log (jumped
# straight from Kit's own "Simulation App Startup Complete" to "Simulation
# App Shutting Down" with no gap) - plausibly stdout being fully buffered
# (non-TTY, piped through tee) and never flushed before simulation_app's own
# native process teardown, the same class of issue this repo's kb already
# flagged once before ("Python's own block-buffered stdout... delaying the
# visible symptom"). Every print below now also goes to this plain results
# file (opened unbuffered, closed explicitly at the very end) so the result
# survives even if stdout capture is lost again, and every print() call
# passes flush=True as a second, independent mitigation.
_RESULTS_PATH = os.path.join(_RL_ROOT, "joint2_limit_raw_usd_only_result.txt")
_results_f = open(_RESULTS_PATH, "w", buffering=1)


def log(msg: str = "") -> None:
    print(msg, flush=True)
    _results_f.write(msg + "\n")
    _results_f.flush()

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

try:
    log("\n" + "=" * 78)
    log(f"RAW USD physics:lowerLimit/upperLimit read from {usd_path}")
    log("=" * 78)
    stage = Usd.Stage.Open(usd_path)
    raw_usd_deg = {}
    for name in ARM_JOINT_NAMES:
        prim = stage.GetPrimAtPath(f"/mk5/root_joint/joints/{name}")
        if not prim.IsValid():
            log(f"  {name}: PRIM NOT FOUND at /mk5/root_joint/joints/{name}")
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
        log(
            f"  {name}: RAW USD=[{lower_deg}, {upper_deg}]deg  VENDOR=[{vendor_lo}, {vendor_hi}]deg  "
            f"{'*** MISMATCH ***' if mismatch else 'match'}"
        )

    log("\n" + "=" * 78)
    log("SUMMARY: does the ~59deg observed physical wall match the RAW USD upper limit for joint_2?")
    log("=" * 78)
    j2_raw_lo, j2_raw_hi = raw_usd_deg.get("joint_2", (None, None))
    if j2_raw_hi is not None:
        log(f"  joint_2 RAW USD upper limit = {j2_raw_hi:.3f}deg (observed physical wall ~59.0-59.1deg)")
        if abs(j2_raw_hi - 59.0) < 2.0:
            log("  -> CONFIRMED: the ~59deg wall IS the raw USD hard limit (an asset-import bug vs. the 90deg vendor spec).")
        elif abs(j2_raw_hi - 90.0) < 2.0:
            log("  -> The raw USD limit matches the 90deg vendor spec. The ~59deg wall is NOT the joint's own hard limit -")
            log("     self-collision is disabled and soft limits equal hard limits per robot_cfg.py, so a THIRD")
            log("     mechanism (not yet identified) must be responsible. Flagged for further investigation.")
        else:
            log(f"  -> Raw USD limit ({j2_raw_hi:.3f}deg) matches NEITHER the ~59deg wall NOR the 90deg vendor spec - genuinely unexplained, flag for further investigation.")

    log("\nDONE.")
except Exception:
    log("\n!!! EXCEPTION during diagnostic !!!")
    log(traceback.format_exc())
    _results_f.close()
    simulation_app.close()
    sys.exit(1)

_results_f.close()
simulation_app.close()
