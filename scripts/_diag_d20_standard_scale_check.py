# scripts/_diag_d20_standard_scale_check.py
"""One-off diagnostic (2026-07-15 standard-vs-jumbo d20 size correction):
opens this repo's baked d20 USD (assets/dice/d20_physics.usd) directly (no
scene spawn, no physics) and prints the live bounding-box size at the
candidate `FrankaDieLiftJointStandardEnvCfg` scale (derived in
tasks/franka/dice_lift_joint_env_cfg.py from this file's own existing rung
data), to confirm the corrected 22mm standard-d20 target is actually hit
before trusting the arithmetic alone. Same headless-SimulationApp asset-
inspection pattern as scripts/_diag_dexcube_scale_check.py and
scripts/_diag_die_scale_check.py (committed precedent for this script
class; nothing is rendered, no scene spawn).

.. code-block:: bash

    flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/_diag_d20_standard_scale_check.py"
"""

import os

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_D20_USD = os.path.join(REPO_ROOT, "assets", "dice", "d20_physics.usd")

# Candidate scale for FrankaDieLiftJointStandardEnvCfg, derived from this
# file's own existing rung constants (see the class docstring in
# tasks/franka/dice_lift_joint_env_cfg.py for the full derivation):
# average scale-per-mm ratio 3.302305e-5 * 22mm = 0.000727.
CANDIDATE_SCALE = 0.000727
TARGET_MM = 22.0


def main() -> None:
    stage = Usd.Stage.Open(_D20_USD)
    if stage is None:
        print(f"FAILED to open {_D20_USD}")
        return
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    mesh_count = 0
    # TraverseInstanceProxies mirrors _diag_dexcube_scale_check.py - safe
    # even if this asset isn't instanceable, and matches existing precedent.
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
    eff_mm = [d * CANDIDATE_SCALE * 1000.0 for d in dims]

    print(f"\n[STANDARD SCALE CHECK] {_D20_USD}")
    print(f"  metersPerUnit={mpu} mesh_count={mesh_count}")
    print(f"  authored bbox dims (stage units): {dims[0]:.4f} x {dims[1]:.4f} x {dims[2]:.4f}")
    print(f"  candidate scale: {CANDIDATE_SCALE}")
    print(
        f"  EFFECTIVE in-scene dims (mm) at candidate scale: "
        f"{eff_mm[0]:.3f} x {eff_mm[1]:.3f} x {eff_mm[2]:.3f}"
    )
    max_dim_mm = max(eff_mm)
    print(f"  max dim (mm) = {max_dim_mm:.3f}  target = {TARGET_MM}mm  delta = {max_dim_mm - TARGET_MM:.3f}mm")
    print("\n[DONE]")


if __name__ == "__main__":
    main()
    simulation_app.close()
