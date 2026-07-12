"""One-off asset bake: copy a dice_sets_v1 die USD and write physics schemas
into the copy, so RL env cfgs can spawn it without runtime patching. Also
supports authoring a fresh flat-faced cube through the SAME physics-schema
code path (asset-bisect rung 3, docs/superpowers/specs/2026-07-12-asset-
bisect-design.md) - isolating rolling-geometry/shape from pipeline
provenance.

Headless SimulationApp is correct here (batch asset processing, not a
watchable run - same precedent as scripts/build_asset.py). Run under flock:

    flock /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 \
      /home/saps/IsaacLab/isaaclab.sh -p scripts/bake_die_asset.py \
      --die d20 2>&1 | tee /tmp/bake_die_asset.log"

    flock /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 \
      /home/saps/IsaacLab/isaaclab.sh -p scripts/bake_die_asset.py \
      --shape cube --size 48.0 --mass 0.216 2>&1 | tee /tmp/bake_cube48.log"

The output USD keeps the source's mm-as-m geometry units; spawn-time
scale=(0.001,)*3 is the consumer's job (dice-demo-validated convention).
The default prim is renamed to 'Object' (die path) / authored directly as
'Object' (cube path) so the stock lift recipe's
SceneEntityCfg("object", body_names="Object") terms match unchanged.

Cube mode (--shape cube) authors a fresh 8-vertex/12-triangle UsdGeom.Mesh
cube (NOT a UsdGeom.Cube gprim - parity with the die assets, which are
Mesh), of edge length --size stage units, centered at the origin. It reads
metersPerUnit/upAxis off the --die source stage (read-only, never
modified) so the cube's unit convention matches the die bakes' byte for
byte, rather than falling back to USD's own defaults - the "source's
mm-as-m convention" the module docstring above refers to.
"""

import argparse
import os
import shutil

from isaacsim import SimulationApp

parser = argparse.ArgumentParser(description="Bake physics schemas into a die or cube USD.")
parser.add_argument(
    "--shape",
    default="die",
    choices=["die", "cube"],
    help="die: existing dice_sets_v1 copy path (default, unchanged). cube: author a fresh Mesh cube.",
)
parser.add_argument("--die", default="d20", choices=["d4", "d8", "d10", "d12", "d20"])
parser.add_argument("--set", dest="set_name", default="set_00000")
parser.add_argument("--mass", type=float, default=0.01, help="kg, dice-demo value")
parser.add_argument(
    "--size",
    type=float,
    default=48.0,
    help=(
        "cube mode only: edge length in stage units, source's mm-as-m convention "
        "(e.g. 48.0 -> 48mm at the consumer's 0.001 spawn scale)."
    ),
)
args = parser.parse_args()

simulation_app = SimulationApp({"headless": True})

from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "vision", "data", "raw", "dice_sets_v1", f"{args.set_name}_{args.die}.usd")
OUT_DIR = os.path.join(REPO, "assets", "dice")
if args.shape == "cube":
    OUT = os.path.join(OUT_DIR, f"cube{int(args.size)}_physics.usd")
else:
    OUT = os.path.join(OUT_DIR, f"{args.die}_physics.usd")


def apply_physics_schema(stage: Usd.Stage) -> int:
    """Apply RigidBodyAPI/MassAPI to the stage's default prim, and
    CollisionAPI + convexHull MeshCollisionAPI to every UsdGeom.Mesh prim
    at or beneath it. Shared by the die and cube bake paths - THE SAME
    physics-schema code for both, per the asset-bisect rung-3 requirement
    that shape be the only isolated variable. Returns the mesh count."""
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
    return mesh_count


def build_cube_mesh(stage: Usd.Stage, size: float) -> Usd.Prim:
    """Author a fresh cube: an 'Object' Xform default prim (the
    RigidBodyAPI/MassAPI target) containing one nested 'geometry' Mesh
    child (8 points, 12 triangles, CCW winding viewed from outside each
    face) of edge length `size` stage units, centered at the origin. This
    mirrors the die bakes' own structure EXACTLY (confirmed by direct
    inspection of assets/dice/d20_physics.usd: default prim is an Xform,
    the actual collision Mesh is a nested child two levels down at
    /Object/d20_die/d20_mesh) - NOT a coincidence, but load-bearing: an
    initial single-prim version of this function (Mesh as both the
    RigidBodyAPI root AND the CollisionAPI target) baked and mass-checked
    fine but produced an object with NO working collision at all - it
    free-fell straight through the table under zero action every episode
    (root cause: Isaac Lab's spawner composes an Xform-typed prim at the
    scene prim_path, and referencing a bare-Mesh-typed default prim into
    it produces a typeName mismatch that silently drops PhysX's mesh-
    collider dispatch, even though schema APIs like RigidBodyAPI/MassAPI -
    which aren't typeName-gated - still read back fine). Returns the
    Xform root prim."""
    h = size / 2.0
    points = [
        (-h, -h, -h),  # 0
        (h, -h, -h),  # 1
        (h, h, -h),  # 2
        (-h, h, -h),  # 3
        (-h, -h, h),  # 4
        (h, -h, h),  # 5
        (h, h, h),  # 6
        (-h, h, h),  # 7
    ]
    tris = [
        (4, 5, 6), (4, 6, 7),  # +Z
        (0, 3, 2), (0, 2, 1),  # -Z
        (1, 2, 6), (1, 6, 5),  # +X
        (0, 4, 7), (0, 7, 3),  # -X
        (3, 7, 6), (3, 6, 2),  # +Y
        (0, 1, 5), (0, 5, 4),  # -Y
    ]
    root_xform = UsdGeom.Xform.Define(stage, "/Object")
    stage.SetDefaultPrim(root_xform.GetPrim())

    mesh = UsdGeom.Mesh.Define(stage, "/Object/geometry")
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
    mesh.CreateFaceVertexCountsAttr([3] * len(tris))
    mesh.CreateFaceVertexIndicesAttr([idx for tri in tris for idx in tri])
    mesh.CreateExtentAttr([Gf.Vec3f(-h, -h, -h), Gf.Vec3f(h, h, h)])
    return root_xform.GetPrim()


def bake_die() -> int:
    """Existing behavior, byte-identical: copy the dice_sets_v1 source,
    rename its default prim to 'Object', apply physics schemas."""
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
        layer = stage.GetRootLayer()
        edit = Sdf.BatchNamespaceEdit()
        edit.Add(Sdf.NamespaceEdit.Rename(old_root.GetPath(), "Object"))
        if not layer.Apply(edit):
            raise RuntimeError("prim rename failed")
        stage.SetDefaultPrim(stage.GetPrimAtPath("/Object"))

    mesh_count = apply_physics_schema(stage)
    stage.GetRootLayer().Save()
    return mesh_count


def bake_cube() -> int:
    """Author a fresh stage containing a Mesh cube, matching the --die
    source's metersPerUnit/upAxis convention (read-only reference, never
    modified), then apply the SAME physics-schema function as bake_die()."""
    if not os.path.isfile(SRC):
        raise FileNotFoundError(f"--die source needed for unit-convention reference: {SRC}")
    os.makedirs(OUT_DIR, exist_ok=True)

    src_stage = Usd.Stage.Open(SRC)
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(src_stage)
    up_axis = UsdGeom.GetStageUpAxis(src_stage)
    del src_stage

    if os.path.exists(OUT):
        os.remove(OUT)
    stage = Usd.Stage.CreateNew(OUT)
    UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
    UsdGeom.SetStageUpAxis(stage, up_axis)

    build_cube_mesh(stage, args.size)
    mesh_count = apply_physics_schema(stage)
    stage.GetRootLayer().Save()
    return mesh_count


def main() -> None:
    mesh_count = bake_cube() if args.shape == "cube" else bake_die()

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
    extra = ""
    if args.shape == "cube":
        geom = UsdGeom.Mesh(check.GetPrimAtPath("/Object/geometry"))
        pts = geom.GetPointsAttr().Get()
        tris = geom.GetFaceVertexCountsAttr().Get()
        extra = f" shape=cube size={args.size} n_points={len(pts)} n_triangles={len(tris)}"
    print(f"[BAKE] OK: {OUT} root='Object' meshes_with_convex_hull={n} mass={args.mass}kg{extra}")


main()
simulation_app.close()
