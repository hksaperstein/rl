# scripts/_diag_d8d10d12_standard_scale_check.py
"""One-off diagnostic (Task 0 of the unified multi-die specialist-distillation
plan, docs/superpowers/plans/2026-07-16-unified-multi-die-specialist-
distillation.md): extends scripts/_diag_d20_standard_scale_check.py's exact
no-physics headless-SimulationApp bounding-box-read pattern (same asset-
inspection method, same tolerance) from a single hardcoded d20 candidate
scale to a loop over the three newly-baked d8/d10/d12 assets
(assets/dice/{d8,d10,d12}_physics.usd).

Unlike the d20 script, this one does NOT assume a shared scale-per-mm ratio
transfers across shapes (the plan's Task 0 brief explicitly warns not to
assume this) - each shape's native (unscaled) mesh bounding box is measured
directly from its own baked USD first, and a fresh scale is derived from
THAT measurement:

    scale = target_mm / (native_max_dim_stage_units * 1000.0)

This is the same relationship implied by FrankaDieLiftJointStandardEnvCfg's
own scale-per-mm derivation (tasks/franka/dice_lift_joint_env_cfg.py) - that
class's 3.302305e-5/mm ratio is simply 1 / (d20's own native max dim in
stage units * 1000), i.e. a d20-specific constant, not a universal one.
Confirmed by measurement below: d8/d10/d12's native bounding boxes are each
their own distinct size (dice_sets_v1 dice are not all molded to the same
raw stage-unit scale), so each shape gets its own freshly-fit scale rather
than reusing 3.302305e-5.

Real commercial "standard" target sizes (Task 0 Step 1 web research, see
.superpowers/sdd/task-0-report.md for full sourcing): d8 ~16mm, d10 ~16mm,
d12 ~18mm face-to-face, consensus of multiple independent tabletop-gaming
retailer/blog size guides (brycesdice.com, dicegamedepot.com,
runerollers.com), cross-checked against Chessex's own "18mm d12" product
naming convention. Less clean-cut than d20's single citable "Twenty Sided
Store Jumbo 30mm D20" listing - flagged explicitly here and in the task
report rather than presented as equally authoritative.

.. code-block:: bash

    flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/_diag_d8d10d12_standard_scale_check.py"
"""

import os

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Target real "standard" commercial sizes in mm, per Task 0 Step 1 web
# research (see .superpowers/sdd/task-0-report.md for exact sources/quotes).
TARGETS_MM = {
    "d8": 16.0,
    "d10": 16.0,
    "d12": 18.0,
}
TOLERANCE_MM = 0.3


def _measure_native_bbox(usd_path: str):
    """Same mesh-traversal/bound-computation pattern as
    _diag_d20_standard_scale_check.py's main(), but returns the RAW
    (unscaled) per-axis stage-unit dims instead of baking in a candidate
    scale - this script derives its own per-shape scale from this
    measurement rather than assuming one."""
    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError(f"FAILED to open {usd_path}")
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    mesh_count = 0
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh_count += 1
        bound = UsdGeom.Mesh(prim).ComputeWorldBound(Usd.TimeCode.Default(), "default")
        box = bound.ComputeAlignedRange()
        bmin, bmax = box.GetMin(), box.GetMax()
        for i in range(3):
            lo[i] = min(lo[i], bmin[i])
            hi[i] = max(hi[i], bmax[i])
    dims = [hi[i] - lo[i] for i in range(3)]
    return dims, mpu, mesh_count


def main() -> None:
    all_pass = True
    for die, target_mm in TARGETS_MM.items():
        usd_path = os.path.join(REPO_ROOT, "assets", "dice", f"{die}_physics.usd")
        dims, mpu, mesh_count = _measure_native_bbox(usd_path)
        native_max_dim = max(dims)

        # Fresh per-shape scale-per-mm derivation (do NOT reuse d20's
        # 3.302305e-5 - see module docstring).
        raw_scale = target_mm / (native_max_dim * 1000.0)
        # Round to this file's established 6-decimal convention
        # (tasks/franka/dice_lift_joint_env_cfg.py's own scale constants).
        candidate_scale = round(raw_scale, 6)

        eff_mm = [d * candidate_scale * 1000.0 for d in dims]
        max_eff_mm = max(eff_mm)
        delta = max_eff_mm - target_mm
        ok = abs(delta) <= TOLERANCE_MM
        all_pass = all_pass and ok

        print(f"\n[STANDARD SCALE CHECK] die={die} {usd_path}")
        print(f"  metersPerUnit={mpu} mesh_count={mesh_count}")
        print(f"  native (unscaled) bbox dims (stage units): {dims[0]:.4f} x {dims[1]:.4f} x {dims[2]:.4f}")
        print(f"  native_max_dim={native_max_dim:.4f}  raw_scale={raw_scale:.8e}  candidate_scale={candidate_scale}")
        print(f"  EFFECTIVE in-scene dims (mm) at candidate scale: {eff_mm[0]:.3f} x {eff_mm[1]:.3f} x {eff_mm[2]:.3f}")
        print(
            f"  max dim (mm) = {max_eff_mm:.3f}  target = {target_mm}mm  delta = {delta:.3f}mm  "
            f"{'PASS' if ok else 'FAIL'} (tolerance {TOLERANCE_MM}mm)"
        )

    print(f"\n[SUMMARY] all shapes within tolerance: {all_pass}")
    print("\n[DONE]")


if __name__ == "__main__":
    main()
    simulation_app.close()
