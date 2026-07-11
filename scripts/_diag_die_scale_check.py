# scripts/_diag_die_scale_check.py
"""One-off diagnostic (not a Gate A component): opens each dice_sets_v1
set_00000 die USD directly (no scene spawn, no physics) and prints its
UsdGeom.Mesh point-cloud bounding box + the stage's metersPerUnit, to
measure whether the raw asset is authored at real dice scale (~1-2cm) or is
mis-scaled (e.g. mm-as-m, ~1000x too large) - evidence for Gate A's exploded-
dice failure, per .superpowers/sdd/dice-demo-task1-brief.md's explicit
instruction to report measured numbers rather than inventing a rescale.

.. code-block:: bash

    flock /tmp/rl_isaac_sim.lock -c "/home/saps/IsaacLab/isaaclab.sh -p scripts/_diag_die_scale_check.py"
"""

import os

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom  # noqa: E402

DIE_TYPES = ["d4", "d8", "d10", "d12", "d20"]
DICE_SET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vision", "data", "raw", "dice_sets_v1"
)


def main() -> None:
    for die_type in DIE_TYPES:
        usd_path = os.path.join(DICE_SET_DIR, f"set_00000_{die_type}.usd")
        stage = Usd.Stage.Open(usd_path)
        meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)

        min_pt = [float("inf")] * 3
        max_pt = [float("-inf")] * 3
        mesh_count = 0
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                mesh_count += 1
                mesh = UsdGeom.Mesh(prim)
                points = mesh.GetPointsAttr().Get()
                if points is None:
                    continue
                xform_cache = UsdGeom.XformCache()
                world_xform = xform_cache.GetLocalToWorldTransform(prim)
                for p in points:
                    wp = world_xform.Transform(p)
                    for i in range(3):
                        min_pt[i] = min(min_pt[i], wp[i])
                        max_pt[i] = max(max_pt[i], wp[i])

        extent = [max_pt[i] - min_pt[i] for i in range(3)]
        print(
            f"[SCALE CHECK] {die_type}: mesh_count={mesh_count} "
            f"metersPerUnit={meters_per_unit} "
            f"bbox_extent_stage_units={tuple(round(e, 6) for e in extent)} "
            f"bbox_extent_meters={tuple(round(e * meters_per_unit, 6) for e in extent)} "
            f"min={tuple(round(v, 6) for v in min_pt)} max={tuple(round(v, 6) for v in max_pt)}"
        )


if __name__ == "__main__":
    main()
    simulation_app.close()
