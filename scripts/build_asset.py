"""Convert the AR4 mk5 arm URDF (from an external ar4_ros_driver checkout) to a
USD asset that Isaac Lab tasks can load, and generate the small shape assets
(e.g. the triangular prism, which has no built-in Isaac Lab primitive) used
by the object scene.

Run once per machine (rerun if the URDF/meshes change). Requires the
AR4_DESCRIPTION_PATH environment variable to point at the annin_ar4_description
package (the directory containing urdf/ and meshes/), e.g.:

    export AR4_DESCRIPTION_PATH=/home/saps/projects/annin_ws/src/ar4_ros_driver/annin_ar4_description
    ./isaaclab.sh -p rl/scripts/build_asset.py

Output is written to rl/assets/ar4_mk5/ar4_mk5.usd and rl/assets/shapes/wedge.usd
(gitignored).
"""

import os
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USD_OUT_DIR = os.path.join(REPO_ROOT, "assets", "ar4_mk5")
SHAPES_OUT_DIR = os.path.join(REPO_ROOT, "assets", "shapes")
WEDGE_USD_PATH = os.path.join(SHAPES_OUT_DIR, "wedge.usd")


def _resolve_description_path() -> str:
    path = os.environ.get("AR4_DESCRIPTION_PATH")
    if not path:
        sys.exit(
            "AR4_DESCRIPTION_PATH is not set. Point it at the annin_ar4_description "
            "package (the directory containing urdf/ and meshes/), e.g.:\n"
            "  export AR4_DESCRIPTION_PATH=/home/saps/projects/annin_ws/src/ar4_ros_driver/annin_ar4_description"
        )
    if not os.path.isdir(path):
        sys.exit(f"AR4_DESCRIPTION_PATH does not exist: {path}")
    return path


def _generate_plain_urdf(description_path: str, out_path: str) -> None:
    xacro_file = os.path.join(description_path, "urdf", "ar.urdf.xacro")
    if not os.path.isfile(xacro_file):
        sys.exit(f"Expected xacro file not found: {xacro_file}")

    # xacro's shebang is a fixed /usr/bin/python3, but it still inherits
    # PYTHONPATH from this process's environment. isaaclab.sh points
    # PYTHONPATH at Isaac Sim's bundled Python 3.11 stdlib, which is binary-
    # incompatible with the system python3 xacro actually runs under
    # ("SRE module mismatch"). Strip Isaac Sim's own entries (keeping the
    # ROS ones, which xacro needs to find its package metadata).
    xacro_env = {**os.environ, "ROS_PACKAGE_PATH": os.path.dirname(description_path)}
    filtered_pythonpath = [
        entry for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep) if "_isaac_sim" not in entry
    ]
    xacro_env["PYTHONPATH"] = os.pathsep.join(filtered_pythonpath)

    subprocess.run(
        [
            "xacro",
            xacro_file,
            "ar_model:=mk5",
            "tf_prefix:=",
            "include_gripper:=true",
            "-o",
            out_path,
        ],
        check=True,
        env=xacro_env,
    )

    # The arm's own meshes are still referenced via package://annin_ar4_description/...
    # (only the gripper macro pre-resolves its own meshes to file:// via xacro's
    # $(find) at expansion time). The USD importer's package:// resolution
    # against ROS_PACKAGE_PATH silently drops any mesh it can't resolve rather
    # than erroring, which was producing a robot with fully invisible arm
    # links. Rewrite package:// URIs to absolute file:// paths ourselves so
    # resolution can't depend on the importer's ROS package-path handling.
    with open(out_path) as f:
        urdf_text = f.read()
    urdf_text = urdf_text.replace("package://annin_ar4_description/", f"file://{description_path}/")
    with open(out_path, "w") as f:
        f.write(urdf_text)


def _generate_wedge_usd(out_path: str, radius: float = 0.011, height: float = 0.018) -> None:
    """Author a small triangular-prism (wedge) mesh to USD.

    Isaac Lab has no built-in wedge/prism shape spawner (only cuboid, sphere,
    cylinder, capsule, cone), so this authors the geometry directly.

    RigidObjectCfg's rigid_props/collision_props/mass_props only *modify*
    existing PhysX schemas on a referenced USD file (unlike the built-in
    shape spawners, which apply the schema themselves on freshly created
    prims) - so the RigidBodyAPI/CollisionAPI/MassAPI schemas must already
    be present on this authored asset, or RigidObjectCfg silently finds
    nothing to modify and the object never becomes a rigid body.
    """
    import math

    from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    stage = Usd.Stage.CreateNew(out_path)
    # Must be an explicit Xform (not just an implicit ancestor of the mesh
    # prim below), since RigidBodyAPI is applied to this root prim and
    # requires it to be Xformable.
    root = UsdGeom.Xform.Define(stage, "/wedge")
    mesh = UsdGeom.Mesh.Define(stage, "/wedge/geometry/mesh")

    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    PhysxSchema.PhysxRigidBodyAPI.Apply(root.GetPrim())
    UsdPhysics.MassAPI.Apply(root.GetPrim())
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")

    half_h = height / 2.0
    tri_xy = [
        (radius, 0.0),
        (-radius / 2.0, radius * math.sqrt(3) / 2.0),
        (-radius / 2.0, -radius * math.sqrt(3) / 2.0),
    ]
    points = [(x, y, -half_h) for x, y in tri_xy] + [(x, y, half_h) for x, y in tri_xy]

    face_vertex_counts = [3, 3, 4, 4, 4]
    face_vertex_indices = [
        0, 2, 1,  # bottom (normal -z)
        3, 4, 5,  # top (normal +z)
        0, 1, 4, 3,  # side 0-1
        1, 2, 5, 4,  # side 1-2
        2, 0, 3, 5,  # side 2-0
    ]

    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(face_vertex_counts)
    mesh.CreateFaceVertexIndicesAttr(face_vertex_indices)
    mesh.CreateSubdivisionSchemeAttr("none")

    stage.SetDefaultPrim(stage.GetPrimAtPath("/wedge"))
    stage.GetRootLayer().Save()


def main() -> None:
    description_path = _resolve_description_path()

    with tempfile.TemporaryDirectory(prefix="ar4_urdf_") as tmp_dir:
        urdf_path = os.path.join(tmp_dir, "ar4_mk5.urdf")
        _generate_plain_urdf(description_path, urdf_path)
        print(f"Generated plain URDF: {urdf_path}")

        # The generated URDF still references meshes via package://
        # annin_ar4_description/... URIs (xacro's own $(find) substitution
        # is only used inside the gripper macro, not the main arm macro).
        # The URDF importer resolves those against ROS_PACKAGE_PATH.
        os.environ["ROS_PACKAGE_PATH"] = os.path.dirname(description_path)

        from isaacsim import SimulationApp

        simulation_app = SimulationApp({"headless": True})

        import omni.kit.app

        ext_manager = omni.kit.app.get_app().get_extension_manager()
        ext_manager.set_extension_enabled_immediate("omni.scene.optimizer.core", True)
        ext_manager.set_extension_enabled_immediate("isaacsim.robot.schema", True)

        import omni.kit.commands

        os.makedirs(USD_OUT_DIR, exist_ok=True)
        output_usd = os.path.join(USD_OUT_DIR, "ar4_mk5.usd")

        _, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
        import_config.fix_base = True
        import_config.merge_fixed_joints = True
        import_config.parse_mimic = True
        # Without this, the exported USD has no default prim, so referencing
        # the bare file path (as robot_cfg.py's UsdFileCfg does) fails to
        # resolve at load time ("Unresolved reference ... <defaultPrim>").
        import_config.make_default_prim = True

        _, urdf_robot = omni.kit.commands.execute(
            "URDFParseFile", urdf_path=urdf_path, import_config=import_config
        )
        result_path = omni.kit.commands.execute(
            "URDFImportRobot",
            urdf_path=urdf_path,
            urdf_robot=urdf_robot,
            import_config=import_config,
            dest_path=output_usd,
            get_articulation_root=True,
        )

        if not result_path or not os.path.isfile(output_usd):
            simulation_app.close()
            sys.exit("URDF import failed: no USD output produced.")

        manifest_path = os.path.join(USD_OUT_DIR, "usd_path.txt")
        with open(manifest_path, "w") as f:
            f.write(output_usd)

        print(f"AR4 mk5 USD asset written to: {output_usd}")

        _generate_wedge_usd(WEDGE_USD_PATH)
        print(f"Wedge (triangular prism) USD asset written to: {WEDGE_USD_PATH}")

        simulation_app.close()


if __name__ == "__main__":
    main()
