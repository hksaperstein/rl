"""One-off asset bake: copy a dice_sets_v1 die USD and write physics schemas
into the copy, so RL env cfgs can spawn it without runtime patching.

Headless SimulationApp is correct here (batch asset processing, not a
watchable run - same precedent as scripts/build_asset.py). Run under flock:

    flock /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 \
      /home/saps/IsaacLab/isaaclab.sh -p scripts/bake_die_asset.py \
      --die d20 2>&1 | tee /tmp/bake_die_asset.log"

The output USD keeps the source's mm-as-m geometry units; spawn-time
scale=(0.001,)*3 is the consumer's job (dice-demo-validated convention).
The default prim is renamed to 'Object' so the stock lift recipe's
SceneEntityCfg("object", body_names="Object") terms match unchanged.
"""

import argparse
import os
import shutil

from isaacsim import SimulationApp

parser = argparse.ArgumentParser(description="Bake physics schemas into a die USD copy.")
parser.add_argument("--die", default="d20", choices=["d4", "d8", "d10", "d12", "d20"])
parser.add_argument("--set", dest="set_name", default="set_00000")
parser.add_argument("--mass", type=float, default=0.01, help="kg, dice-demo value")
args = parser.parse_args()

simulation_app = SimulationApp({"headless": True})

from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "vision", "data", "raw", "dice_sets_v1", f"{args.set_name}_{args.die}.usd")
OUT_DIR = os.path.join(REPO, "assets", "dice")
OUT = os.path.join(OUT_DIR, f"{args.die}_physics.usd")


def main() -> None:
    if not os.path.isfile(SRC):
        raise FileNotFoundError(SRC)
    os.makedirs(OUT_DIR, exist_ok=True)
    shutil.copyfile(SRC, OUT)

    stage = Usd.Stage.Open(OUT)
    old_root = stage.GetDefaultPrim()
    if not old_root:
        raise RuntimeError(f"{OUT} has no default prim")

    # Rename default prim to 'Object' (stock lift recipe's expected body name).
    if old_root.GetName() != "Object":
        # Usd has no in-place rename; re-parent via a new Xform and move children.
        # Simpler robust approach: define /Object, move the old root's children
        # is error-prone - instead use Sdf-level namespace edit.
        from pxr import Sdf

        layer = stage.GetRootLayer()
        edit = Sdf.BatchNamespaceEdit()
        edit.Add(Sdf.NamespaceEdit.Rename(old_root.GetPath(), "Object"))
        if not layer.Apply(edit):
            raise RuntimeError("prim rename failed")
        stage.SetDefaultPrim(stage.GetPrimAtPath("/Object"))

    root = stage.GetDefaultPrim()
    UsdPhysics.RigidBodyAPI.Apply(root)
    PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    mass_api = UsdPhysics.MassAPI.Apply(root)
    mass_api.CreateMassAttr(args.mass)

    mesh_count = 0
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdGeom.Mesh):
            UsdPhysics.CollisionAPI.Apply(prim)
            mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_api.CreateApproximationAttr("convexHull")
            mesh_count += 1
    if mesh_count == 0:
        raise RuntimeError("no UsdGeom.Mesh prims found - nothing baked")

    stage.GetRootLayer().Save()

    # Verify by re-opening fresh.
    check = Usd.Stage.Open(OUT)
    croot = check.GetDefaultPrim()
    assert croot.GetName() == "Object", croot.GetName()
    assert UsdPhysics.RigidBodyAPI(croot), "RigidBodyAPI missing after bake"
    assert UsdPhysics.MassAPI(croot), "MassAPI missing after bake"
    n = sum(
        1
        for p in Usd.PrimRange(croot)
        if p.IsA(UsdGeom.Mesh)
        and UsdPhysics.CollisionAPI(p)
        and UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get() == "convexHull"
    )
    assert n == mesh_count, (n, mesh_count)
    print(f"[BAKE] OK: {OUT} root='Object' meshes_with_convex_hull={n} mass={args.mass}kg")


main()
simulation_app.close()
