"""Convert the AR4 mk5 arm URDF (from an external ar4_ros_driver checkout) to a
USD asset that Isaac Lab tasks can load, and generate the small shape assets
(e.g. the triangular prism, which has no built-in Isaac Lab primitive) used
by the object scene.

Run once per machine (rerun if the URDF/meshes change). Requires the
AR4_DESCRIPTION_PATH environment variable to point at the annin_ar4_description
package (the directory containing urdf/ and meshes/), e.g.:

    export AR4_DESCRIPTION_PATH=/home/saps/projects/annin_ws/src/ar4_ros_driver/annin_ar4_description
    ./isaaclab.sh -p scripts/build_asset.py

Output is written to assets/ar4_mk5/ar4_mk5.usd and assets/shapes/wedge.usd
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


def _build_urdf_color_map(urdf_path: str) -> dict:
    """Parse a URDF and return {visual_mesh_basename -> (r, g, b)} from each
    ``<visual>``'s ``<material><color rgba>``.

    The installed URDF importer (isaacsim.asset.importer.urdf) authors a white
    ``DefaultMaterial`` (``diffuse_color_constant = (1, 1, 1)``) for every
    imported mesh and discards the URDF's per-visual ``<material><color>``
    values entirely - there is no ImportConfig flag controlling this. This map
    is used in a post-import pass to write the authored colors back onto the
    generated USD's shaders, so the arm renders with its real aluminum / dark-
    motor / enclosure scheme instead of a flat white silhouette.

    URDF allows a material to be defined once (with a color) and referenced by
    name later without repeating the color, so named colors are collected
    globally first and then resolved per visual.
    """
    import xml.etree.ElementTree as ET

    def _parse_rgba(color_el):
        rgba = color_el.get("rgba", "").split()
        if len(rgba) < 3:
            return None
        return (float(rgba[0]), float(rgba[1]), float(rgba[2]))

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Global name -> color table (any <material name=...><color/></material>).
    named_colors: dict = {}
    for mat in root.iter("material"):
        name = mat.get("name")
        color_el = mat.find("color")
        if name and color_el is not None:
            rgb = _parse_rgba(color_el)
            if rgb is not None:
                named_colors[name] = rgb

    color_map: dict = {}
    for visual in root.iter("visual"):
        mesh_el = visual.find("geometry/mesh")
        mat_el = visual.find("material")
        if mesh_el is None or mat_el is None:
            continue
        filename = mesh_el.get("filename", "")
        if not filename:
            continue
        basename = os.path.splitext(os.path.basename(filename))[0]
        # Prefer an inline <color>; fall back to a named-material reference.
        color_el = mat_el.find("color")
        rgb = _parse_rgba(color_el) if color_el is not None else named_colors.get(mat_el.get("name"))
        if rgb is not None:
            color_map[basename] = rgb
    return color_map


def _apply_visual_colors(base_usd_path: str, color_map: dict) -> None:
    """Write the URDF's per-visual colors onto the imported USD's shaders.

    The importer nests each visual under ``.../visuals/<MeshName>/...`` with a
    bound ``DefaultMaterial`` whose MDL shader carries a
    ``diffuse_color_constant`` input (default white). ``<MeshName>`` is the STL
    file's basename, which is exactly the key produced by
    :func:`_build_urdf_color_map`. Both the shader's ``diffuse_color_constant``
    (what the RTX renderer uses) and the mesh's ``displayColor`` primvar (a
    fallback for non-RTX/Hydra-Storm viewports) are set.

    Authored directly on the base configuration layer (the layer that *defines*
    the geometry and materials), so the change composes into the referenced
    asset without running into instanceable-proxy edit restrictions.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

    stage = Usd.Stage.Open(base_usd_path)

    def _mesh_name_from_path(path_str: str):
        # In the base layer the importer authors each visual under
        #   /visuals/<link>/<STLName>/node_STL_BINARY_/{mesh,Looks/DefaultMaterial/...}
        # so the STL basename (the color_map key) is the component immediately
        # *before* the node_STL_* marker - not the component after "visuals"
        # (that is the link name). Collision geometry lives under a separate
        # scope and never carries a node_STL_* visual marker here.
        parts = path_str.split("/")
        marker_idx = next((i for i, p in enumerate(parts) if p.startswith("node_STL")), None)
        if marker_idx is None or marker_idx == 0:
            return None
        if "collisions" in parts:
            return None  # never recolor collision meshes
        return parts[marker_idx - 1]

    shaders_set = 0
    meshes_set = 0
    unmatched = set()

    for prim in stage.Traverse():
        tname = prim.GetTypeName()
        if tname == "Shader":
            name = _mesh_name_from_path(prim.GetPath().pathString)
            if name is None:
                continue
            rgb = color_map.get(name)
            if rgb is None:
                unmatched.add(name)
                continue
            shader = UsdShade.Shader(prim)
            vec = Gf.Vec3f(*rgb)
            inp = shader.GetInput("diffuse_color_constant")
            if not inp:
                inp = shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f)
            inp.Set(vec)
            shaders_set += 1
        elif tname == "Mesh":
            name = _mesh_name_from_path(prim.GetPath().pathString)
            if name is None:
                continue
            rgb = color_map.get(name)
            if rgb is None:
                continue
            mesh = UsdGeom.Mesh(prim)
            mesh.CreateDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*rgb)]))
            meshes_set += 1

    stage.GetRootLayer().Save()
    print(
        f"[colors] applied URDF visual colors: {shaders_set} shaders, "
        f"{meshes_set} mesh displayColors set (base layer: {os.path.basename(base_usd_path)})"
    )
    if unmatched:
        print(f"[colors] WARNING: {len(unmatched)} visual mesh name(s) had no URDF color match: {sorted(unmatched)}")


def _locate_base_layer(usd_out_dir: str) -> str:
    """Return the configuration sub-layer that defines the imported geometry
    and materials (``*_base.usd``; falls back to the largest .usd)."""
    config_dir = os.path.join(usd_out_dir, "configuration")
    candidates = [f for f in os.listdir(config_dir) if f.endswith(".usd")]
    base = [f for f in candidates if f.endswith("_base.usd")]
    chosen = base[0] if base else max(candidates, key=lambda f: os.path.getsize(os.path.join(config_dir, f)))
    return os.path.join(config_dir, chosen)


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

        # The importer discards the URDF's per-visual <material><color> values
        # (every mesh gets a white DefaultMaterial). Write them back onto the
        # generated USD so the arm renders in its real color scheme.
        color_map = _build_urdf_color_map(urdf_path)
        base_layer = _locate_base_layer(USD_OUT_DIR)
        _apply_visual_colors(base_layer, color_map)

        manifest_path = os.path.join(USD_OUT_DIR, "usd_path.txt")
        with open(manifest_path, "w") as f:
            f.write(output_usd)

        print(f"AR4 mk5 USD asset written to: {output_usd}")

        _generate_wedge_usd(WEDGE_USD_PATH)
        print(f"Wedge (triangular prism) USD asset written to: {WEDGE_USD_PATH}")

        simulation_app.close()


if __name__ == "__main__":
    main()
