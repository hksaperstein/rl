"""Verify the AR4 gripper collision meshes now carry the convexDecomposition
(not convexHull) approximation, so PhysX preserves the fingers' concave
gripping notch instead of filling it solid.

Companion to scripts/build_asset.py's _set_gripper_collision_convex_decomposition
(2026-08-03, ar4-gripper-collision-approximation-fix task) and to
scripts/_analyze_gripper_collision_hull_vs_mesh.py (which quantified the
convexHull inflation: hull/mesh volume 2.38x/2.30x/1.54x on jaw1/jaw2/base).

Pure USD inspection -- opens the asset with pxr (no SimulationApp / RTX
init needed) and reports, for each of the three gripper collision meshes:
  - the authored UsdPhysics.MeshCollisionAPI approximation attr
  - the PhysxConvexDecompositionCollisionAPI params (if present)
  - the raw collision-mesh AABB extent + point/face counts (the geometry
    PhysX's approximation is computed from)

Run inside the Isaac container (its python has pxr):
    /isaac-sim/python.sh scripts/_verify_gripper_collision_approx.py [usd_path]
"""

import os
import sys

from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema

MESH_PATHS = [
    "/mk5/root_joint/gripper_jaw1_link/collisions/gripper_jaw1_link/node_STL_BINARY_/mesh",
    "/mk5/root_joint/gripper_jaw2_link/collisions/gripper_jaw2_link/node_STL_BINARY_/mesh",
    "/mk5/root_joint/gripper_base_link/collisions/gripper_base_link/node_STL_BINARY_/mesh",
]


def _resolve_usd_path() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    manifest = os.path.join(repo_root, "assets", "ar4_mk5", "usd_path.txt")
    if os.path.isfile(manifest):
        with open(manifest) as f:
            p = f.read().strip()
        # The manifest may hold a container-side path; fall back to the
        # local repo copy if that literal path doesn't exist here.
        if os.path.isfile(p):
            return p
    return os.path.join(repo_root, "assets", "ar4_mk5", "ar4_mk5.usd")


def main() -> None:
    usd_path = _resolve_usd_path()
    print(f"[verify] opening: {usd_path}")
    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        sys.exit(f"[verify] FATAL: could not open stage at {usd_path}")

    all_decomp = True
    for mesh_path in MESH_PATHS:
        link = mesh_path.split("/")[3]
        # Traverse instance proxies so we read through instanceable ancestors.
        prim = stage.GetPrimAtPath(mesh_path)
        print(f"\n=== {link} ===")
        print(f"  path: {mesh_path}")
        if not prim.IsValid():
            # Retry via an instance-proxy-aware lookup.
            for p in stage.Traverse(Usd.TraverseInstanceProxies()):
                if str(p.GetPath()) == mesh_path:
                    prim = p
                    break
        if not prim.IsValid():
            print("  WARNING: prim not found (invalid) -- cannot verify")
            all_decomp = False
            continue

        coll = UsdPhysics.MeshCollisionAPI(prim)
        approx = coll.GetApproximationAttr().Get() if coll else None
        print(f"  MeshCollisionAPI.approximation = {approx!r}")
        if approx != "convexDecomposition":
            all_decomp = False

        decomp = PhysxSchema.PhysxConvexDecompositionCollisionAPI(prim)
        if decomp and prim.HasAPI(PhysxSchema.PhysxConvexDecompositionCollisionAPI):
            vr = decomp.GetVoxelResolutionAttr().Get()
            mh = decomp.GetMaxConvexHullsAttr().Get()
            hv = decomp.GetHullVertexLimitAttr().Get()
            ep = decomp.GetErrorPercentageAttr().Get()
            sw = decomp.GetShrinkWrapAttr().Get()
            print(f"  PhysxConvexDecomposition: voxelRes={vr} maxHulls={mh} hullVertLimit={hv} errPct={ep} shrinkWrap={sw}")
        else:
            print("  PhysxConvexDecompositionCollisionAPI: (not applied)")

        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        faces = mesh.GetFaceVertexCountsAttr().Get()
        if pts:
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; zs = [p[2] for p in pts]
            ext = (max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
            print(f"  collision mesh: {len(pts)} points / {len(faces) if faces else 0} faces")
            print(f"  raw mesh AABB extent (mm): X={ext[0]*1000:.1f} Y={ext[1]*1000:.1f} Z={ext[2]*1000:.1f}")
        else:
            print("  collision mesh: (no points)")

    print("\n" + "=" * 60)
    print(f"VERIFY VERDICT: all three gripper collision meshes use convexDecomposition = {all_decomp}")
    print("=" * 60)
    sys.exit(0 if all_decomp else 2)


if __name__ == "__main__":
    main()
