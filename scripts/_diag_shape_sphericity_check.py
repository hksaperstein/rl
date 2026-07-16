"""One-off diagnostic (Task 1 of the unified multi-die specialist-distillation
plan, docs/superpowers/plans/2026-07-16-unified-multi-die-specialist-
distillation.md): measures a real, mesh-derived sphericity value for each of
the four baked dice shapes (assets/dice/{d8,d10,d12,d20}_physics.usd), used
to hardcode tasks/franka/shape_observations.py's SHAPE_GEOMETRY_DESCRIPTORS
constants. Same headless-SimulationApp/pxr mesh-inspection pattern as
scripts/_diag_d8d10d12_standard_scale_check.py (no physics, no video -
batch asset inspection, not a watchable run).

Sphericity (Wadell, H. 1935, "Volume, shape, and roundness of quartz
particles", Journal of Geology 43(3):250-280 - the standard particle-shape
descriptor, still in routine use across geology/powder-metallurgy/granular-
mechanics literature):

    psi = pi^(1/3) * (6*V)^(2/3) / A

where V and A are the volume and surface area of the particle's convex hull.
psi = 1.0 for a perfect sphere and decreases as a shape deviates from
spherical. This ratio is scale-invariant (V ~ L^3, A ~ L^2, so
(6V)^(2/3)/A ~ L^2/L^2 = dimensionless) - so it can be computed directly off
each mesh's own native (unscaled) point cloud, independent of the
spawn-time scale each env cfg applies.

Method: open each baked USD, collect ALL mesh points (world-space, i.e. the
mesh's own local/native coordinates since no transform is applied above the
mesh in these bakes), take their 3-D convex hull via
scipy.spatial.ConvexHull (real geometric measurement of the physical
dice shape, not a canonical/idealized Platonic-solid formula - these are
real bevelled/manufacturing-style meshes per
tasks/franka/antipodal_edge_grasp.py's own d4 mesh-inspection precedent,
244 points for a 4-vertex shape), and read off ConvexHull.volume/.area
directly (scipy's ConvexHull computes both for a 3-D point set).

.. code-block:: bash

    flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/_diag_shape_sphericity_check.py"

Hit CLAUDE.md's documented "hangs during Kit/extension shutdown teardown
after the script's actual work is already done" failure mode on its first
run (2026-07-16): the four result lines and the "Python dict" block printed
fine at [7.918s], then the process sat idle in "Simulation App Shutting
Down" for 20+ minutes with near-zero CPU - `kill -TERM` on the still-live
process was safe (real output already flushed to the piped log) and
released the flock lock immediately, exactly as documented.
"""

import os

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402
from scipy.spatial import ConvexHull  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHAPES = ("d8", "d10", "d12", "d20")


def _collect_mesh_points(usd_path: str) -> np.ndarray:
    """All UsdGeom.Mesh points in the stage, in each mesh's own local
    coordinates (no world transform applied - matches
    _diag_d8d10d12_standard_scale_check.py's own note that these bakes have
    no transform authored above the mesh, so local == native == the
    coordinate frame the spawn-time `scale` is later applied to)."""
    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError(f"FAILED to open {usd_path}")
    all_points = []
    mesh_count = 0
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh_count += 1
        pts_attr = UsdGeom.Mesh(prim).GetPointsAttr()
        pts = pts_attr.Get()
        if pts is None:
            continue
        all_points.extend([(p[0], p[1], p[2]) for p in pts])
    if mesh_count == 0:
        raise RuntimeError(f"no UsdGeom.Mesh prims found in {usd_path}")
    return np.asarray(all_points, dtype=float)


def main() -> None:
    print(f"{'shape':<6} {'n_points':>9} {'volume':>14} {'area':>14} {'sphericity':>12}")
    results = {}
    for shape in SHAPES:
        usd_path = os.path.join(REPO_ROOT, "assets", "dice", f"{shape}_physics.usd")
        points = _collect_mesh_points(usd_path)
        hull = ConvexHull(points)
        volume = hull.volume
        area = hull.area
        sphericity = (np.pi ** (1.0 / 3.0)) * (6.0 * volume) ** (2.0 / 3.0) / area
        results[shape] = sphericity
        print(f"{shape:<6} {points.shape[0]:>9d} {volume:>14.6f} {area:>14.6f} {sphericity:>12.6f}")

    print()
    print("Python dict for tasks/franka/shape_observations.py:")
    print("SHAPE_GEOMETRY_DESCRIPTORS = {")
    for shape in SHAPES:
        print(f'    "{shape}": {round(results[shape], 6)},')
    print("}")


if __name__ == "__main__":
    main()
    simulation_app.close()
