"""Quick diagnostic: dump every prim under gripper_jaw1_link/gripper_jaw2_link
(instance-proxy traversal) with type + CollisionAPI/MeshCollisionAPI status,
to debug why _inspect_jaw_convex_hull.py's own filter found zero matches.
Run via: /path/to/isaaclab.sh -p scripts/_debug_jaw_prim_dump.py
"""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
usd_manifest = os.path.join(REPO_ROOT, "assets", "ar4_mk5", "usd_path.txt")
with open(usd_manifest) as f:
    usd_path = f.read().strip()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

stage = Usd.Stage.Open(usd_path)

for link_name in ("gripper_jaw1_link", "gripper_jaw2_link"):
    link_prim = stage.GetPrimAtPath(f"/mk5/root_joint/{link_name}")
    print(f"\n=== {link_name} valid={link_prim.IsValid()} path={link_prim.GetPath()} ===")
    link_path_str = str(link_prim.GetPath())
    count = 0
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        p = str(prim.GetPath())
        if not p.startswith(link_path_str):
            continue
        count += 1
        is_mesh = prim.IsA(UsdGeom.Mesh)
        has_coll = prim.HasAPI(UsdPhysics.CollisionAPI)
        has_meshcoll = prim.HasAPI(UsdPhysics.MeshCollisionAPI)
        print(f"  {p}  type={prim.GetTypeName()}  IsA(Mesh)={is_mesh}  HasAPI(CollisionAPI)={has_coll}  HasAPI(MeshCollisionAPI)={has_meshcoll}  IsInstanceProxy={prim.IsInstanceProxy()}")
    print(f"  ({count} descendant prims total under this link)")

simulation_app.close()
