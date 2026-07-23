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


def _remove_gripper_jaw2_mimic_constraint(output_usd: str) -> None:
    """Remove gripper_jaw2_joint's PhysxMimicJointAPI entirely and set its
    hard physics limits to mirror gripper_jaw1_joint's own limits directly.

    UPDATE 2026-07-21 (later, ar4-grasp-fix task): this supersedes the
    original ``_fix_gripper_jaw2_mimic_limits`` fix above/below in history.
    That fix corrected jaw2's hard limits to be mathematically consistent
    with its authored ``PhysxMimicJointAPI`` (referenceJoint=gripper_jaw1_joint,
    gearing=-1.0, offset=0.0), but a live dynamic rollout after that fix
    (``scripts/_verify_gripper_mirror_fix.py``, see
    kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md's 2026-07-21
    "later" UPDATE) found jaw2 still does not track its own commanded PD
    target at all under real physics - it gets pinned near one of its two
    hard limits regardless of target. The identified candidate mechanism:
    the PhysxMimicJointAPI spring constraint and the independent
    ImplicitActuatorCfg PD actuator (tasks/ar4/robot_cfg.py) are BOTH
    trying to drive gripper_jaw2_joint simultaneously, and something in
    that interaction dominates and drives jaw2 into a hard limit rather
    than either mechanism's own target in isolation.

    Fix: remove the PhysxMimicJointAPI constraint entirely, removing the
    physics-level tug-of-war outright, and rely purely on the
    software-level mirrored command (GRIPPER_OPEN_COMMAND_EXPR /
    GRIPPER_CLOSED_COMMAND_EXPR in tasks/ar4/robot_cfg.py) driving BOTH
    jaws as independent ImplicitActuatorCfg PD targets. jaw2's own hard
    limits are still re-derived from jaw1's own limits under the
    known mirror geometry so its physical range of motion is still
    correct - only the physics-level spring constraint is removed, not
    the limit correction from the original fix.

    UPDATE 2026-07-23: the gearing=-1.0 value below (read off the URDF's
    own authored PhysxMimicJointAPI) was empirically DISPROVEN as the
    correct command-to-world mapping by a direct live measurement
    (scripts/_sweep_jaw2_symmetry.py - see tasks/ar4/robot_cfg.py's own
    2026-07-23 UPDATE comment and kb/wiki/concepts/
    ar4-vs-franka-root-cause-comparison.md for the full writeup): with
    jaw1 held at its OPEN target and jaw2's commanded joint value swept
    directly, jaw2's actual world-frame position came back as exactly
    -1.0 * (jaw2's own commanded value) - meaning jaw2's local-to-world
    mapping (a consequence of its 180-degree-rotated joint frame, not of
    the mimic's gearing attribute) ALREADY contains the sign flip. Using
    the URDF-authored gearing=-1.0 AGAIN here to map jaw1's limits onto
    jaw2 double-negates and produces limits ([-0.014, 0.000]) that reject
    the one command value that actually produces a real mirrored-open
    pincer (+0.014, the same signed value as jaw1). The corrected,
    empirically-confirmed gearing for this derivation is +1.0 (jaw2's
    hard limits should equal jaw1's own [0.000, 0.014] directly, not
    negated) - hardcoded below rather than trusting the mimic API's own
    authored attribute, since that attribute describes the raw URDF joint
    kinematic relationship, not the corrected command convention this
    session's live measurement established.
    """
    from pxr import PhysxSchema, Usd

    stage = Usd.Stage.Open(output_usd)
    jaw1 = stage.GetPrimAtPath("/mk5/root_joint/joints/gripper_jaw1_joint")
    jaw2 = stage.GetPrimAtPath("/mk5/root_joint/joints/gripper_jaw2_joint")
    if not jaw1.IsValid() or not jaw2.IsValid():
        print("[mimic-removal] WARNING: gripper jaw joint prims not found at the expected paths - skipping fix")
        return

    lower1 = jaw1.GetAttribute("physics:lowerLimit").Get()
    upper1 = jaw1.GetAttribute("physics:upperLimit").Get()
    if lower1 is None or upper1 is None:
        print("[mimic-removal] WARNING: gripper_jaw1_joint limits unreadable - skipping fix")
        return

    # gearing=+1.0 is the EMPIRICALLY-CORRECTED value (2026-07-23, see
    # docstring UPDATE above) - deliberately NOT read from the mimic API's
    # own authored gearing attribute (URDF-authored, -1.0), which was
    # shown to produce the wrong (collapsed-jaw) limits/command
    # convention. The mimic API is still stripped below (removes the
    # physics-level spring constraint per the 2026-07-21 fix this
    # function is named for) - only the NUMERIC gearing used for the
    # limit computation is overridden.
    gearing, offset = 1.0, 0.0
    mimic = PhysxSchema.PhysxMimicJointAPI.Get(jaw2, "rotX")
    if mimic:
        urdf_gearing = mimic.GetGearingAttr().Get()
        urdf_offset = mimic.GetOffsetAttr().Get()
        removed = jaw2.RemoveAPI(PhysxSchema.PhysxMimicJointAPI, "rotX")
        print(
            f"[mimic-removal] PhysxMimicJointAPI:rotX removed from gripper_jaw2_joint (success={removed}); "
            f"URDF-authored gearing={urdf_gearing}/offset={urdf_offset} intentionally NOT used for the limit "
            f"derivation below (empirically disproven 2026-07-23 - using corrected gearing={gearing}/offset={offset} instead)"
        )
    else:
        print(
            "[mimic-removal] WARNING: no PhysxMimicJointAPI:rotX found on gripper_jaw2_joint to remove "
            f"(nothing to strip; proceeding to set limits from the corrected gearing={gearing}, offset={offset} anyway)"
        )

    mapped_a = gearing * lower1 + offset
    mapped_b = gearing * upper1 + offset
    new_lower, new_upper = min(mapped_a, mapped_b), max(mapped_a, mapped_b)

    old_lower = jaw2.GetAttribute("physics:lowerLimit").Get()
    old_upper = jaw2.GetAttribute("physics:upperLimit").Get()
    jaw2.GetAttribute("physics:lowerLimit").Set(new_lower)
    jaw2.GetAttribute("physics:upperLimit").Set(new_upper)
    stage.GetRootLayer().Save()
    print(
        f"[mimic-removal] gripper_jaw2_joint limits set independently of any mimic constraint: "
        f"[{old_lower:.4f}, {old_upper:.4f}] -> [{new_lower:.4f}, {new_upper:.4f}] (mirrors "
        f"gripper_jaw1_joint's own [{lower1:.4f}, {upper1:.4f}] under gearing={gearing}, offset={offset}; "
        f"jaw2 is now driven purely by its own independent ImplicitActuatorCfg PD actuator + the "
        f"software-level mirrored command, with no competing physics-level constraint)"
    )


def _add_gripper_jaw2_drive(output_usd: str) -> None:
    """Author a UsdPhysics.DriveAPI:linear on gripper_jaw2_joint, mirroring
    gripper_jaw1_joint's own authored drive.

    Root cause found live-testing the mimic-removal fix above (2026-07-22,
    ar4-vs-franka-root-cause-comparison.md's own UPDATE for that date): with
    the arm base genuinely held fixed (a separate confound - see that doc -
    the arm's own default actuator gains are too weak to hold its pose
    against gravity, which was masking this joint's real behavior under an
    apparent "opposite-end"/Coriolis-coupling signature in two earlier live
    tests), gripper_jaw2_joint turned out to be COMPLETELY unresponsive to
    any commanded PD target at all, in any of 3 tested target values (0,
    -0.014, -0.007) - not mistracking, not sign-inverted, just inert.

    Direct USD inspection (`prim.GetAppliedSchemas()`) explains why:
    gripper_jaw1_joint carries `PhysicsDriveAPI:linear` (type=acceleration,
    stiffness=625.0, damping=0.0, maxForce=3.4e38) from the URDF importer;
    gripper_jaw2_joint carries NO DriveAPI schema instance at all (only
    PhysicsJointStateAPI:linear, PhysxJointAPI, IsaacJointAPI). This makes
    sense in hindsight: before this fix, jaw2 was a URDF mimic joint (see
    the removed PhysxMimicJointAPI docs above) - importers generally only
    author an independent DriveAPI on a mimic joint's REFERENCE joint
    (jaw1), since the mimic's own gearing constraint was meant to be jaw2's
    sole actuation mechanism. `_remove_gripper_jaw2_mimic_constraint`
    (above) stripped that mimic constraint but never gave jaw2 an
    independent drive to replace it - Isaac Lab's ImplicitActuatorCfg
    happily calls `set_dof_stiffnesses`/`set_dof_dampings` on jaw2's DOF
    with no error or warning (confirmed by reading
    isaaclab/assets/articulation/articulation.py's actuator-processing
    path - no DriveAPI/HasAPI check anywhere in that call chain), but with
    no PhysX drive object ever created for this axis in the first place,
    those writes are an apparent silent no-op at the PhysX level - jaw2
    just sits wherever it starts, never actually driven toward any target.

    Fix: author the missing DriveAPI directly, using jaw1's own authored
    values as the starting point (Isaac Lab's ImplicitActuatorCfg
    overwrites the numeric stiffness/damping at scene-creation time
    regardless, per the same source reading above - these authored values
    only matter for giving PhysX a real drive object to attach the
    ImplicitActuatorCfg's own gains to, not as the actual runtime gains).
    """
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(output_usd)
    jaw1 = stage.GetPrimAtPath("/mk5/root_joint/joints/gripper_jaw1_joint")
    jaw2 = stage.GetPrimAtPath("/mk5/root_joint/joints/gripper_jaw2_joint")
    if not jaw1.IsValid() or not jaw2.IsValid():
        print("[jaw2-drive-fix] WARNING: gripper jaw joint prims not found at the expected paths - skipping fix")
        return

    jaw1_drive = UsdPhysics.DriveAPI.Get(jaw1, "linear")
    if not jaw1_drive:
        print("[jaw2-drive-fix] WARNING: gripper_jaw1_joint has no linear DriveAPI to mirror - skipping fix")
        return
    if UsdPhysics.DriveAPI.Get(jaw2, "linear"):
        print("[jaw2-drive-fix] gripper_jaw2_joint already has a linear DriveAPI - nothing to do")
        return

    jaw2_drive = UsdPhysics.DriveAPI.Apply(jaw2, "linear")
    jaw2_drive.CreateTypeAttr(jaw1_drive.GetTypeAttr().Get())
    jaw2_drive.CreateStiffnessAttr(jaw1_drive.GetStiffnessAttr().Get())
    jaw2_drive.CreateDampingAttr(jaw1_drive.GetDampingAttr().Get())
    jaw2_drive.CreateMaxForceAttr(jaw1_drive.GetMaxForceAttr().Get())
    jaw2_drive.CreateTargetPositionAttr(0.0)
    stage.GetRootLayer().Save()
    print(
        "[jaw2-drive-fix] gripper_jaw2_joint: authored a new linear DriveAPI mirroring gripper_jaw1_joint's own "
        f"(type={jaw1_drive.GetTypeAttr().Get()}, stiffness={jaw1_drive.GetStiffnessAttr().Get()}, "
        f"damping={jaw1_drive.GetDampingAttr().Get()}, maxForce={jaw1_drive.GetMaxForceAttr().Get()}) - "
        "jaw2 now has a real PhysX drive object for Isaac Lab's ImplicitActuatorCfg to write runtime gains into."
    )


def _add_substitute_link_collision(output_usd: str, link_name: str) -> None:
    """Add a simple box collider to a link that has no collision geometry
    at all in the current upstream mesh checkout.

    ``Link_5_Col.STL``/``Link_6_Col.STL`` genuinely do not exist in the
    ``annin_ar4_description`` package's meshes directory as of the pinned
    upstream commit (confirmed by direct USD inspection - both links'
    ``collisions`` scopes import as empty instanceable prototypes with zero
    mesh children, not merely an unresolved-reference warning; see
    docs/superpowers/specs/research/2026-07-21-ar4-usd-asset-debugging.md).
    This leaves link_5/link_6 (the wrist links immediately upstream of the
    gripper) with a genuine blind spot in any ground/self-collision safety
    check. Fix: derive an axis-aligned box from the link's own (correctly
    resolved) VISUAL sub-meshes' combined bounding box, in the link's local
    frame, and author it as a plain ``UsdGeom.Cube`` collider (an analytic
    PhysX shape, no mesh-approximation risk) - computed fresh from whatever
    visual geometry actually resolved, not a hardcoded guess, so this stays
    correct if the upstream checkout's meshes ever change.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(output_usd)
    link_prim = stage.GetPrimAtPath(f"/mk5/root_joint/{link_name}")
    if not link_prim.IsValid():
        print(f"[collision-fix] WARNING: {link_name} prim not found - skipping")
        return
    visuals_prim = link_prim.GetChild("visuals")
    if not visuals_prim:
        print(f"[collision-fix] WARNING: {link_name}/visuals not found - skipping")
        return

    xf_cache = UsdGeom.XformCache()
    pts_local = []
    visuals_path_str = str(visuals_prim.GetPath())
    for sub in stage.Traverse(Usd.TraverseInstanceProxies()):
        if str(sub.GetPath()).startswith(visuals_path_str) and sub.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(sub)
            pts = mesh.GetPointsAttr().Get()
            if not pts:
                continue
            local_to_link = xf_cache.ComputeRelativeTransform(sub, link_prim)[0]
            for pt in pts:
                pts_local.append(local_to_link.Transform(pt))

    if not pts_local:
        print(f"[collision-fix] WARNING: no resolved visual mesh points found under {link_name}/visuals - skipping")
        return

    xs = [p[0] for p in pts_local]
    ys = [p[1] for p in pts_local]
    zs = [p[2] for p in pts_local]
    lo = Gf.Vec3d(min(xs), min(ys), min(zs))
    hi = Gf.Vec3d(max(xs), max(ys), max(zs))
    center = (lo + hi) * 0.5
    size = hi - lo

    box_path = link_prim.GetPath().AppendChild("substitute_collision_box")
    cube = UsdGeom.Cube.Define(stage, box_path)
    cube.CreateSizeAttr(1.0)
    # "guide" purpose: a collision-only proxy, not meant to be rendered
    # (this link's own visual meshes already render normally and are
    # untouched by this fix).
    cube.CreatePurposeAttr("guide")
    cube.AddTranslateOp().Set(center)
    cube.AddScaleOp().Set(Gf.Vec3f(size[0], size[1], size[2]))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

    stage.GetRootLayer().Save()
    print(
        f"[collision-fix] added substitute box collider for {link_name}: "
        f"center={tuple(center)} size={tuple(size)} (from {len(pts_local)} resolved visual mesh "
        "points; the source STL collision mesh for this link is genuinely missing from the upstream repo)"
    )


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

        # Two direct USD-level asset fixes, see docs/superpowers/specs/
        # research/2026-07-21-ar4-usd-asset-debugging.md for the full
        # findings each is grounded in.
        _remove_gripper_jaw2_mimic_constraint(output_usd)
        _add_gripper_jaw2_drive(output_usd)
        _add_substitute_link_collision(output_usd, "link_5")
        _add_substitute_link_collision(output_usd, "link_6")

        manifest_path = os.path.join(USD_OUT_DIR, "usd_path.txt")
        with open(manifest_path, "w") as f:
            f.write(output_usd)

        print(f"AR4 mk5 USD asset written to: {output_usd}")

        _generate_wedge_usd(WEDGE_USD_PATH)
        print(f"Wedge (triangular prism) USD asset written to: {WEDGE_USD_PATH}")

        simulation_app.close()


if __name__ == "__main__":
    main()
