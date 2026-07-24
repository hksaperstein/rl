"""Debug: after _fix_jaw2_collision_mesh_asymmetry() claims success, why does
a fresh Usd.Stage.Open still show jaw2's OLD point count? Check (a) whether
the root layer itself has a local prim spec with a points opinion for jaw2's
mesh, (b) the property value resolution stack (which layer wins), (c) file
mtime.
"""
import os
import subprocess

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
usd_manifest = os.path.join(REPO_ROOT, "assets", "ar4_mk5", "usd_path.txt")
with open(usd_manifest) as f:
    usd_path = f.read().strip()

print("[INFO] usd_path:", usd_path)
print("[INFO] mtime:", subprocess.run(["stat", "-c", "%y", usd_path], capture_output=True, text=True).stdout)

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": True})

from pxr import Sdf, Usd, UsdGeom  # noqa: E402

stage = Usd.Stage.Open(usd_path)
root_layer = stage.GetRootLayer()
print("[INFO] root layer identifier:", root_layer.identifier)
print("[INFO] root layer subLayerPaths:", root_layer.subLayerPaths)

jaw2_mesh_path = "/mk5/root_joint/gripper_jaw2_link/collisions/gripper_jaw2_link/node_STL_BINARY_/mesh"
prim = stage.GetPrimAtPath(jaw2_mesh_path)
print("[INFO] prim valid:", prim.IsValid(), "IsInstanceProxy:", prim.IsInstanceProxy() if prim.IsValid() else None)

pts_attr = UsdGeom.Mesh(prim).GetPointsAttr()
print("[INFO] composed points count:", len(pts_attr.Get() or []))

print("[INFO] property stack (strongest first):")
for spec in pts_attr.GetPropertyStack():
    layer = spec.layer
    val = spec.default
    print(f"    layer={layer.identifier}  hasDefault={spec.HasDefaultValue()}  "
          f"count={len(val) if val else 'N/A'}")

# Does the ROOT layer itself have a local prim spec at this path at all?
root_prim_spec = root_layer.GetPrimAtPath(jaw2_mesh_path)
print("[INFO] root layer has local prim spec at this path:", root_prim_spec is not None)
if root_prim_spec is not None:
    print("[INFO] root layer prim spec properties:", list(root_prim_spec.properties.keys()) if hasattr(root_prim_spec, "properties") else "N/A")

simulation_app.close()
