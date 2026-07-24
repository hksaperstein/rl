"""Debug round 2: is /mk5/root_joint/gripper_jaw2_link/collisions/.../mesh
directly (non-instanced) defined inside configuration/ar4_mk5_base.usd when
THAT file is opened as its own stage, rather than composed through the
top-level ar4_mk5.usd (where it resolves as an instance proxy)?
"""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
usd_manifest = os.path.join(REPO_ROOT, "assets", "ar4_mk5", "usd_path.txt")
with open(usd_manifest) as f:
    usd_path = f.read().strip()
usd_out_dir = os.path.dirname(usd_path)

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom  # noqa: E402

import sys  # noqa: E402
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
from build_asset import _locate_base_layer  # noqa: E402

base_layer_path = _locate_base_layer(usd_out_dir)
print("[INFO] base layer path:", base_layer_path)

stage = Usd.Stage.Open(base_layer_path)
mesh_path = "/mk5/root_joint/gripper_jaw2_link/collisions/gripper_jaw2_link/node_STL_BINARY_/mesh"
prim = stage.GetPrimAtPath(mesh_path)
print("[INFO] prim valid:", prim.IsValid())
if prim.IsValid():
    print("[INFO] IsInstanceProxy:", prim.IsInstanceProxy())
    pts = UsdGeom.Mesh(prim).GetPointsAttr().Get()
    print("[INFO] points count:", len(pts) if pts else 0)

# Also check jaw1 for comparison and instance-root ancestor search from here.
jaw2_mesh = prim
instance_root = jaw2_mesh
depth = 0
while instance_root and instance_root.IsValid() and not instance_root.IsInstance() and depth < 20:
    instance_root = instance_root.GetParent()
    depth += 1
print("[INFO] instance_root from this file's own stage:", instance_root.GetPath() if instance_root and instance_root.IsValid() else None,
      "IsInstance:", instance_root.IsInstance() if instance_root and instance_root.IsValid() else None)

simulation_app.close()
