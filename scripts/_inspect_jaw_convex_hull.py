"""One-off direct USD inspection + offline geometry check: is there a real,
previously-never-quantified ASYMMETRY between gripper_jaw1_link's and
gripper_jaw2_link's own collision mesh geometry (raw mesh point/face counts,
convex-hull vertex/face counts, local bounding-box extents) - the concrete
follow-up this project's own 2026-07-21 USD-asset-debugging doc flagged and
never did ("pull gripper_jaw1_link's render-mesh points ... and compare its
real convex hull's face count against the original mesh's face count ...
follow-up: same for jaw2, then compare jaw1 vs jaw2 directly").

Both `UsdPhysics.MeshCollisionAPI.approximation == "convexHull"` on jaw1 and
jaw2 was already confirmed real and symmetric at the SCHEMA level (both jaws
carry the identical attribute) by the 2026-07-21 doc - what was never
checked is whether the underlying MESH GEOMETRY those hulls are computed
from is itself actually symmetric between the two jaws, independent of the
schema being present on both. This is exactly the kind of "looks symmetric
at the config level, isn't at the data level" bug class this project has
hit repeatedly (see e.g. the jaw2-joint-limit-mismatch and jaw2-missing-
DriveAPI findings, both invisible from schema presence alone).

Method:
  1. Instance-proxy stage traversal (Usd.TraverseInstanceProxies() - the
     default predicate does NOT descend into the instanceable prototype
     references the URDF importer uses for every mesh, per the 2026-07-21
     doc's own finding) to find every Mesh prim under gripper_jaw1_link and
     gripper_jaw2_link that has UsdPhysics.CollisionAPI applied - this is
     the actual geometry PhysX's convexHull approximation is computed from,
     whatever the mesh's own name/nesting happens to be (not assumed to be
     the same prim as the RENDER/visual mesh, even though prior sessions
     found the same "node_STL_BINARY_" prim name for both APIs on this
     asset - confirmed fresh here rather than assumed).
  2. Raw point/face counts for each jaw's collision mesh, in the jaw
     LINK's own local frame (UsdGeom.XformCache.ComputeRelativeTransform,
     the same pattern build_asset.py's _add_substitute_link_collision
     already uses for link_5/link_6).
  3. scipy.spatial.ConvexHull computed independently on each jaw's raw
     points - vertex count, face count, volume - the actual quantitative
     "how much does convexHull distort/differ from the raw mesh, and does
     it differ between jaw1 and jaw2" check the 2026-07-21 doc flagged and
     never performed.
  4. A local-frame bounding-box comparison (size only, not position - jaw1
     and jaw2 have different mounting orientations by design) as a second,
     independent symmetry signal alongside the hull numbers.

Run via: /path/to/isaaclab.sh -p scripts/_inspect_jaw_convex_hull.py
Exits 0 always (this is a diagnostic, not a pass/fail gate) - reads its
own printed output for the actual verdict.
"""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
usd_manifest = os.path.join(REPO_ROOT, "assets", "ar4_mk5", "usd_path.txt")
with open(usd_manifest) as f:
    usd_path = f.read().strip()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

try:
    from scipy.spatial import ConvexHull
except ImportError:
    ConvexHull = None
    print("[WARN] scipy not importable - convex-hull vertex/face/volume numbers will be skipped "
          "(raw mesh point/face counts and bounding-box comparison still run). "
          "Install via: pip install scipy")

print(f"[INFO] Opening USD: {usd_path}")
stage = Usd.Stage.Open(usd_path)


def find_collision_meshes(link_name: str):
    """Return [(prim_path, points_in_link_local_frame)] for every Mesh prim
    under /mk5/root_joint/<link_name>/collisions that represents actual
    collision geometry, using instance-proxy traversal so instanceable-
    prototype mesh references are actually visited (plain Usd.PrimRange does
    not descend into them - confirmed 2026-07-21).

    2026-07-24 correction (found live on this task's own first cloud run,
    which returned zero matches everywhere): UsdPhysics.CollisionAPI /
    MeshCollisionAPI are applied to the WRAPPER Xform prim
    (.../collisions/<link_name>/node_STL_BINARY_), not to its child Mesh
    prim (.../node_STL_BINARY_/mesh) that actually holds the geometry -
    confirmed via a direct prim-by-prim dump (scripts/_debug_jaw_prim_dump.py).
    Requiring BOTH UsdGeom.Mesh typing AND CollisionAPI on the SAME prim (the
    original approach) therefore never matches anything. Fixed by walking
    the dedicated .../<link_name>/collisions scope specifically (the URDF
    importer's own <collision>-vs-<visual> scope separation, the same
    convention scripts/build_asset.py's _add_substitute_link_collision
    already relies on for the "visuals" scope) and taking every Mesh prim
    found there directly, rather than re-checking CollisionAPI on each one -
    anything under a link's own "collisions" scope is collision geometry by
    construction, regardless of which specific ancestor prim happens to
    carry the schema instance.
    """
    link_prim = stage.GetPrimAtPath(f"/mk5/root_joint/{link_name}")
    if not link_prim.IsValid():
        print(f"[WARN] {link_name} prim not found at /mk5/root_joint/{link_name}")
        return []
    collisions_prim = link_prim.GetChild("collisions")
    if not collisions_prim or not collisions_prim.IsValid():
        print(f"[WARN] {link_name}/collisions prim not found")
        return []

    collisions_path_str = str(collisions_prim.GetPath())
    xf_cache = UsdGeom.XformCache()
    results = []
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        prim_path_str = str(prim.GetPath())
        if not prim_path_str.startswith(collisions_path_str):
            continue
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        if not pts:
            continue
        local_to_link = xf_cache.ComputeRelativeTransform(prim, link_prim)[0]
        pts_local = np.array([list(local_to_link.Transform(p)) for p in pts])
        face_counts = mesh.GetFaceVertexCountsAttr().Get()
        n_faces = len(face_counts) if face_counts else 0
        results.append((prim_path_str, pts_local, n_faces))
    return results


def describe_hull(name: str, pts: np.ndarray):
    print(f"\n--- {name} convex hull ---")
    if ConvexHull is None or len(pts) < 4:
        print("  [SKIP] scipy unavailable or too few points")
        return None
    hull = ConvexHull(pts)
    print(f"  input points: {len(pts)}")
    print(f"  hull vertices: {len(hull.vertices)}")
    print(f"  hull faces (simplices): {len(hull.simplices)}")
    print(f"  hull volume: {hull.volume:.8f} m^3")
    print(f"  hull surface area: {hull.area:.8f} m^2")
    return hull


all_jaw_data = {}
for link_name in ("gripper_jaw1_link", "gripper_jaw2_link"):
    meshes = find_collision_meshes(link_name)
    print(f"\n=== {link_name}: {len(meshes)} collision mesh prim(s) found ===")
    for prim_path, pts, n_faces in meshes:
        print(f"  {prim_path}: {len(pts)} points, {n_faces} raw mesh faces")
        bbox_lo = pts.min(axis=0)
        bbox_hi = pts.max(axis=0)
        bbox_size = bbox_hi - bbox_lo
        print(f"    local-frame bbox lo={bbox_lo.tolist()} hi={bbox_hi.tolist()} size={bbox_size.tolist()}")
        hull = describe_hull(f"{link_name} ({prim_path})", pts)
        all_jaw_data[link_name] = {
            "prim_path": prim_path,
            "points": pts,
            "n_raw_faces": n_faces,
            "bbox_size": bbox_size,
            "hull": hull,
        }

print("\n\n========== DIRECT jaw1 vs jaw2 COMPARISON ==========")
if "gripper_jaw1_link" in all_jaw_data and "gripper_jaw2_link" in all_jaw_data:
    d1 = all_jaw_data["gripper_jaw1_link"]
    d2 = all_jaw_data["gripper_jaw2_link"]
    print(f"raw point count:  jaw1={len(d1['points'])}  jaw2={len(d2['points'])}  "
          f"{'MATCH' if len(d1['points']) == len(d2['points']) else 'MISMATCH'}")
    print(f"raw face count:   jaw1={d1['n_raw_faces']}  jaw2={d2['n_raw_faces']}  "
          f"{'MATCH' if d1['n_raw_faces'] == d2['n_raw_faces'] else 'MISMATCH'}")
    print(f"local bbox size:  jaw1={d1['bbox_size'].tolist()}  jaw2={d2['bbox_size'].tolist()}")
    bbox_diff = np.abs(np.sort(d1["bbox_size"]) - np.sort(d2["bbox_size"]))
    print(f"  sorted-bbox-size abs diff (mirror-invariant check): {bbox_diff.tolist()} "
          f"(max={bbox_diff.max():.6f}m)")
    if d1["hull"] is not None and d2["hull"] is not None:
        print(f"hull vertex count: jaw1={len(d1['hull'].vertices)}  jaw2={len(d2['hull'].vertices)}  "
              f"{'MATCH' if len(d1['hull'].vertices) == len(d2['hull'].vertices) else 'MISMATCH'}")
        print(f"hull face count:   jaw1={len(d1['hull'].simplices)}  jaw2={len(d2['hull'].simplices)}  "
              f"{'MATCH' if len(d1['hull'].simplices) == len(d2['hull'].simplices) else 'MISMATCH'}")
        vol_diff_pct = 100.0 * abs(d1["hull"].volume - d2["hull"].volume) / max(d1["hull"].volume, 1e-12)
        area_diff_pct = 100.0 * abs(d1["hull"].area - d2["hull"].area) / max(d1["hull"].area, 1e-12)
        print(f"hull volume:       jaw1={d1['hull'].volume:.8f}  jaw2={d2['hull'].volume:.8f}  "
              f"diff={vol_diff_pct:.3f}%")
        print(f"hull surface area: jaw1={d1['hull'].area:.8f}  jaw2={d2['hull'].area:.8f}  "
              f"diff={area_diff_pct:.3f}%")
        print(
            "\n[VERDICT] "
            + ("SYMMETRIC (raw + hull geometry match within float noise) - Hypothesis A (jaw collision "
               "geometry asymmetry) is NOT supported by this data."
               if len(d1["points"]) == len(d2["points"])
               and d1["n_raw_faces"] == d2["n_raw_faces"]
               and len(d1["hull"].vertices) == len(d2["hull"].vertices)
               and len(d1["hull"].simplices) == len(d2["hull"].simplices)
               and vol_diff_pct < 1.0
               and area_diff_pct < 1.0
               else "ASYMMETRIC - a real difference exists between jaw1's and jaw2's own collision "
               "mesh/hull geometry. This IS a candidate root cause for the jaw1-contact/jaw2-zero "
               "signature - investigate further (which specific prim/mesh file diverges).")
        )
else:
    print("[FAIL] Could not find collision mesh data for both jaws - see WARN lines above.")

simulation_app.close()
