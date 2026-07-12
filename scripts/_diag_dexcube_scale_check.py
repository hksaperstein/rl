# scripts/_diag_dexcube_scale_check.py
"""One-off diagnostic (asset-bisect spec prep, 2026-07-12): opens the lift
recipe's DexCube USD (Nucleus path) and this repo's baked d20 USD directly
(no scene spawn, no physics) and prints each stage's metersPerUnit +
UsdGeom.Mesh point-cloud bounding box, then the EFFECTIVE in-scene size
after each config's spawn scale (DexCube x0.8 per
tasks/franka/lift_env_cfg.py; d20 x0.001 per
tasks/franka/dice_lift_joint_env_cfg.py). The dice-demo task-1 report
explicitly declined to guess the DexCube's absolute size — the bisect
spec's rung-1 scale factor must come from this measurement, not an
assumption. Same headless-SimulationApp asset-inspection pattern as
scripts/_diag_die_scale_check.py (committed precedent for this script
class; nothing is rendered).

.. code-block:: bash

    flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/_diag_dexcube_scale_check.py"
"""

import os

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Resolved URL taken from the joint-cube training run's own saved config
# (logs/train_franka_jointcube/2026-07-12_07-31-58/params/env.yaml) —
# isaaclab.utils.assets.ISAAC_NUCLEUS_DIR resolves to None under a bare
# SimulationApp (no isaaclab experience/asset-root carb settings loaded),
# so the f-string form silently produces an unopenable "None/..." path.
_DEXCUBE_URL = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
    "/Assets/Isaac/5.1/Isaac/Props/Blocks/DexCube/dex_cube_instanceable.usd"
)

TARGETS = [
    (
        "DexCube (lift recipe object)",
        _DEXCUBE_URL,
        0.8,
    ),
    (
        "baked d20 (joint-die object)",
        os.path.join(REPO_ROOT, "assets", "dice", "d20_physics.usd"),
        0.001,
    ),
]


def measure(label: str, usd_path: str, spawn_scale: float) -> None:
    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        print(f"[{label}] FAILED to open {usd_path}")
        return
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    mesh_count = 0
    # TraverseInstanceProxies: instanceable assets (like the DexCube) hide
    # their meshes inside instances, which a plain Traverse() skips.
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh_count += 1
        # Author-time points in local units; accumulate the world-ish bbox
        # via ComputeWorldBound (includes any xformOps baked in the asset).
        bound = UsdGeom.Mesh(prim).ComputeWorldBound(Usd.TimeCode.Default(), "default")
        box = bound.ComputeAlignedRange()
        bmin, bmax = box.GetMin(), box.GetMax()
        for i in range(3):
            lo[i] = min(lo[i], bmin[i])
            hi[i] = max(hi[i], bmax[i])
    dims = [hi[i] - lo[i] for i in range(3)]
    eff = [d * spawn_scale for d in dims]
    print(f"\n[{label}] {usd_path}")
    print(f"  metersPerUnit={mpu} mesh_count={mesh_count}")
    print(f"  authored bbox dims (stage units): {dims[0]:.4f} x {dims[1]:.4f} x {dims[2]:.4f}")
    print(f"  spawn scale: {spawn_scale}")
    print(
        f"  EFFECTIVE in-scene dims (m, assuming stage units interpreted as meters at import): "
        f"{eff[0]:.4f} x {eff[1]:.4f} x {eff[2]:.4f}"
    )


def main() -> None:
    for label, path, scale in TARGETS:
        measure(label, path, scale)
    print("\n[DONE]")


if __name__ == "__main__":
    main()
    simulation_app.close()
