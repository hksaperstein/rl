# scripts/build_notch_fixture_asset.py
"""Authors the d4 rung-1 V-notch fingertip fixture (see
docs/superpowers/specs/2026-07-15-d4-rung1-pad-geometry-design.md and
.superpowers/sdd/task-1-brief.md) as a small standalone USD asset,
`assets/shapes/notch_fixture.usd` - the SAME asset file is referenced by
BOTH the left and right fingertip attachments (see
tasks/franka/dice_scene_cfg.py); the right-side attachment applies a
180-degree fixed-joint rotation at runtime to mirror it (see
tasks/franka/notch_fixture.py's `joint_local_rot1_wxyz` docstring).

Deliberately does NOT boot `isaacsim.SimulationApp`/`AppLauncher` (this
task's brief: "no sim launch, no GPU") - reuses the EXACT same plain-`pxr`,
no-Kit-boot technique
scripts/_diag_franka_fingertip_geometry.py established for Task 0 (Kit's
`pxr` bindings are ordinary compiled packages under the `omni.usd.libs`
extension's own `pxr/` directory, importable by adding that directory to
`sys.path` - no Kit boot, no renderer, no GPU context). Also structurally
identical in spirit to scripts/build_asset.py's own `_generate_wedge_usd`
(the established convention this repo already uses for a small
custom-geometry USD asset with no built-in Isaac Lab primitive) - same
Xform-root/Mesh-child/RigidBodyAPI+MassAPI-on-root/CollisionAPI-on-mesh
pattern, generalized to 2 mesh prims (the notch's two walls) instead of 1,
and using this task's own `tasks/franka/notch_fixture.py` pure-geometry
module for every vertex/face computation rather than inlining trig here.

.. code-block:: bash

    LD_LIBRARY_PATH="/home/saps/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311/bin:/home/saps/isaacsim/extscache/omni.usd.schema.physx-107.3.26+107.3.3.lx64.r.cp311.u353/bin" \\
        /home/saps/IsaacLab/_isaac_sim/kit/python/bin/python3 scripts/build_notch_fixture_asset.py

LD_LIBRARY_PATH must be set in the process environment *before* launch (not
via os.environ inside the script) - see
scripts/_diag_franka_fingertip_geometry.py's own docstring for why (the
dynamic linker resolves libusd_tf.so/libphysxSchema.so and friends via
dlopen at import time, too early for an in-script os.environ.setdefault).
`PXR_PLUGINPATH_NAME` (below) is different - USD's plugin registry reads it
lazily at schema-registration time, which happens when `PhysxSchema` is
first imported, so THAT one CAN be (and is) set via os.environ inside this
script, confirmed by hand before settling on this split.

Output: assets/shapes/notch_fixture.usd (gitignored, matching
scripts/build_asset.py's assets/shapes/wedge.usd convention - `assets/` as
a whole is gitignored per this repo's public-repo git conventions; rerun
this script once per machine/checkout before running
scripts/dice_pick_demo.py, same as the wedge asset).
"""

import os
import sys

_OMNI_USD_LIBS_ROOT = (
    "/home/saps/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311"
)
# PhysxSchema (UsdPhysics's own core Usd/UsdGeom/Sdf/Gf/Tf/... comes bundled
# in omni.usd.libs above, same as scripts/_diag_franka_fingertip_geometry.py
# already established) lives in a SEPARATE extension,
# omni.usd.schema.physx - not needed by that Task 0 script (it never applied
# any Physx-specific schema), but needed here for
# `PhysxSchema.PhysxRigidBodyAPI` (mirrors scripts/build_asset.py's
# `_generate_wedge_usd`, which gets this for free since it runs inside a
# fully-booted SimulationApp with every extension already loaded - this
# script instead adds just the one extra extension root it needs, same
# "no Kit boot, no GPU, just sys.path" technique).
_OMNI_USD_SCHEMA_PHYSX_ROOT = (
    "/home/saps/isaacsim/extscache/omni.usd.schema.physx-107.3.26+107.3.3.lx64.r.cp311.u353"
)
for _root in (_OMNI_USD_LIBS_ROOT, _OMNI_USD_SCHEMA_PHYSX_ROOT):
    if _root not in sys.path:
        sys.path.insert(0, _root)

# USD's plugin registry needs to find PhysxSchema's plugInfo.json (normally
# supplied by Kit's own extension-loading system, which this no-Kit-boot
# script bypasses entirely) - without this, `from pxr import PhysxSchema`
# imports fine but APPLYING any PhysxSchema API (e.g. PhysxRigidBodyAPI,
# used below) raises `Failed to find plugin for schema type ...` (confirmed
# by hand: this exact error was hit before adding this block). Each
# extension's plugins/<Name>/resources dir (containing its own
# plugInfo.json) is added, not just the top-level plugins/ dir - matches
# how Kit's own extension loader registers each plugin individually.
os.environ["PXR_PLUGINPATH_NAME"] = os.pathsep.join(
    filter(None, [
        os.environ.get("PXR_PLUGINPATH_NAME", ""),
        f"{_OMNI_USD_SCHEMA_PHYSX_ROOT}/plugins/PhysxSchema/resources",
        f"{_OMNI_USD_SCHEMA_PHYSX_ROOT}/plugins/PhysxSchemaAddition/resources",
        f"{_OMNI_USD_SCHEMA_PHYSX_ROOT}/plugins/OmniUsdPhysicsDeformableSchema/resources",
    ])
)

from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from tasks.franka.notch_fixture import (  # noqa: E402
    FIXTURE_MASS_KG,
    mirror_profile_x,
    notch_wall_profile_xy,
    wall_prism_points,
    WALL_PRISM_FACE_VERTEX_COUNTS,
    WALL_PRISM_FACE_VERTEX_INDICES,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHAPES_OUT_DIR = os.path.join(REPO_ROOT, "assets", "shapes")
NOTCH_FIXTURE_USD_PATH = os.path.join(SHAPES_OUT_DIR, "notch_fixture.usd")


def _define_wall_mesh(stage: Usd.Stage, mesh_path: str, profile_xy) -> None:
    """Defines one wall's convex collision mesh prim (a 6-vertex prism, see
    tasks/franka/notch_fixture.py's `wall_prism_points`), with
    CollisionAPI + MeshCollisionAPI(convexHull) applied - matches
    scripts/build_asset.py's `_generate_wedge_usd` schema-application
    pattern exactly. Two of these (one per wall, +X and -X) live under one
    shared rigid-body root (see `main()`) - PhysX supports multiple
    collision shapes under a single rigid body natively (a "compound"
    collider), which is how this module represents the notch's inherently
    CONCAVE functional shape (a V-groove) using only individually CONVEX
    per-piece collision meshes (this task's brief requires "a rigid convex
    collision mesh" - satisfied per-piece; the concave groove function
    emerges from the GAP between the two convex wall pieces, not from
    either piece's own shape - a standard PhysX compound-collider
    technique, not a new one invented here)."""
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    points = wall_prism_points(profile_xy)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(WALL_PRISM_FACE_VERTEX_COUNTS)
    mesh.CreateFaceVertexIndicesAttr(WALL_PRISM_FACE_VERTEX_INDICES)
    mesh.CreateSubdivisionSchemeAttr("none")

    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")


def build_notch_fixture_usd(out_path: str = NOTCH_FIXTURE_USD_PATH) -> str:
    """Authors the notch fixture USD (root Xform with RigidBodyAPI+MassAPI,
    2 child wall meshes each with CollisionAPI+MeshCollisionAPI(convexHull))
    and returns the written path. Canonical orientation: protrudes in local
    -Y (matches the LEFT finger's own convention, Task 0 measurement) - the
    RIGHT finger's fixed joint mirrors it at runtime via a 180-degree
    rotation about local Z (see tasks/franka/notch_fixture.py's
    `joint_local_rot1_wxyz`), so this ONE asset file serves both fingers -
    no separate mirrored asset is authored."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    stage = Usd.Stage.CreateNew(out_path)
    root = UsdGeom.Xform.Define(stage, "/notch_fixture")

    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    PhysxSchema.PhysxRigidBodyAPI.Apply(root.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass_api.CreateMassAttr(FIXTURE_MASS_KG)

    plus_x_profile = notch_wall_profile_xy()
    minus_x_profile = mirror_profile_x(plus_x_profile)

    _define_wall_mesh(stage, "/notch_fixture/geometry/wall_plus_x", plus_x_profile)
    _define_wall_mesh(stage, "/notch_fixture/geometry/wall_minus_x", minus_x_profile)

    stage.SetDefaultPrim(stage.GetPrimAtPath("/notch_fixture"))
    stage.GetRootLayer().Save()
    return out_path


def main() -> None:
    out_path = build_notch_fixture_usd()
    print(f"[build_notch_fixture_asset] wrote {out_path}")

    # Re-open and print a quick summary (point counts, mass) as a basic
    # sanity check this ran to completion and produced the expected
    # structure - mirrors scripts/_diag_franka_fingertip_geometry.py's own
    # "measure, don't just trust exit code" verification habit.
    stage = Usd.Stage.Open(out_path)
    for mesh_path in ["/notch_fixture/geometry/wall_plus_x", "/notch_fixture/geometry/wall_minus_x"]:
        prim = stage.GetPrimAtPath(mesh_path)
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        print(f"  {mesh_path}: {len(points)} points = {[tuple(round(c, 6) for c in p) for p in points]}")
    root_prim = stage.GetPrimAtPath("/notch_fixture")
    mass_attr = UsdPhysics.MassAPI(root_prim).GetMassAttr()
    print(f"  /notch_fixture mass={mass_attr.Get()}kg rigid_body={root_prim.HasAPI(UsdPhysics.RigidBodyAPI)}")
    print("[DONE]")


if __name__ == "__main__":
    main()
