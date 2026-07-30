"""Quick pxr-only (NO SimulationApp) inspection of the AR4 gripper jaw
collision geometry: where do the actual pad/collision prims sit relative to
each jaw LINK origin, in the link's own LOCAL frame? This resolves the true
pad-offset vector needed to aim a grasp at the real contact surfaces (the
standing JAW1_LOWER_EXTENT scalar assumed a world-Z offset, invalid once the
gripper is tilted). Run in the isaac-lab container: /isaac-sim/python.sh."""
import sys
from isaacsim import SimulationApp
sys.argv += [
    "--/plugins/carb.tasking.plugin/threadCount=16",
    "--/app/asyncRendering=false",
]
_app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics, Gf  # noqa: E402

USD = "/workspace/rl/assets/ar4_mk5/ar4_mk5.usd"
stage = Usd.Stage.Open(USD)
xc = UsdGeom.XformCache(Usd.TimeCode.Default())


def find(name):
    for p in stage.Traverse():
        if p.GetName() == name:
            return p
    return None


for linkname in ["gripper_jaw1_link", "gripper_jaw2_link"]:
    link = find(linkname)
    if link is None:
        print(f"{linkname}: NOT FOUND")
        continue
    Lworld = xc.GetLocalToWorldTransform(link)
    Lworld_inv = Lworld.GetInverse()
    lt = Lworld.ExtractTranslation()
    print(f"\n=== {linkname} @ {link.GetPath()}")
    print(f"  link world translation = ({lt[0]:.5f}, {lt[1]:.5f}, {lt[2]:.5f})")
    # collect collision / mesh / geometry prims under the link
    for c in Usd.PrimRange(link):
        is_coll = c.HasAPI(UsdPhysics.CollisionAPI)
        img = UsdGeom.Imageable(c)
        bnd = UsdGeom.Boundable(c)
        if not (is_coll or c.GetTypeName() in ("Mesh", "Cube", "Capsule", "Cylinder", "Sphere")):
            continue
        try:
            # world-aligned bbox of this prim
            bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide])
            box = bb.ComputeWorldBound(c).ComputeAlignedRange()
            wmin = box.GetMin(); wmax = box.GetMax()
            wc = (np.array([wmin[0], wmin[1], wmin[2]]) + np.array([wmax[0], wmax[1], wmax[2]])) / 2.0
            # world center -> link-local
            lc = Lworld_inv.Transform(Gf.Vec3d(float(wc[0]), float(wc[1]), float(wc[2])))
            size = np.array([wmax[0]-wmin[0], wmax[1]-wmin[1], wmax[2]-wmin[2]])
            print(f"  prim {c.GetName()} type={c.GetTypeName()} coll={is_coll}")
            print(f"    world_center=({wc[0]:.5f},{wc[1]:.5f},{wc[2]:.5f}) size_mm=({size[0]*1000:.1f},{size[1]*1000:.1f},{size[2]*1000:.1f})")
            print(f"    center in LINK-LOCAL frame (m) = ({lc[0]:.5f}, {lc[1]:.5f}, {lc[2]:.5f})  |offset|_mm={np.linalg.norm([lc[0],lc[1],lc[2]])*1000:.2f}")
        except Exception as e:
            print(f"  prim {c.GetName()}: bbox failed: {e}")
print("\nDONE")
