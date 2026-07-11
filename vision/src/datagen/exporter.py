"""
Bakes the non-destructive edge fillet, saves the finished model as a
standalone .blend FIRST (before any export), then exports the die as
USD/STL, renders a thumbnail for visual spot-checking, and writes the
per-asset JSON manifest.

The .blend save happens before USD/STL/thumbnail so it always captures the
fully-finished model as the definitive source state everything else is
derived from -- and, as a side effect, so the thumbnail render's own
temporary camera/light objects never exist yet at .blend-save time (they're
created and removed afterward), which would otherwise leak into the saved
.blend the same way Blender's default startup Cube/Light/Camera would (see
_save_blend_copy).

The Bevel modifier's segments=8 (rather than the default 1) produces a
smooth rounded fillet on the die's structural edges/corners instead of a
single flat chamfer facet. limit_method='WEIGHT' (not 'ANGLE') rounds
only the edges geometry.build_die_base_mesh marked with full bevel
weight on the pristine polyhedron before any engraving ever ran --
'ANGLE' was tried first and rejected: it can't distinguish those
structural edges from the many similarly-steep-angled edges an engraved
numeral's recess introduces, so it also rounded recess geometry,
producing catastrophic degenerate output (confirmed on a real batch:
42/49 engraved assets affected, e.g. 57,353 degenerate faces on one
d20/cjk_numerals asset -- see test_exporter.py's regression test).
"""
import json
import os

import bmesh
import bpy


def export_asset(die_obj, manifest_record, outdir, bevel_fraction, size_mm):
    os.makedirs(outdir, exist_ok=True)
    asset_id = manifest_record["asset_id"]

    bpy.context.view_layer.objects.active = die_obj

    mod = die_obj.modifiers.new(name="Bevel", type='BEVEL')
    mod.width = size_mm * bevel_fraction
    mod.segments = 8
    mod.limit_method = 'WEIGHT'
    bpy.ops.object.modifier_apply(modifier=mod.name)

    # Soften every edge by SHADING, not geometry: smooth-shade all
    # polygons, then apply face-area-weighted normals so the die's large
    # flat faces dominate their own vertices' normals (staying visually
    # flat) while small faces -- structural bevel segments and engraving
    # recess rims/walls -- blend softly into their surroundings. This is
    # what makes engraved edges read as "softened": at realistic
    # engraving depths the recess is far too small for any geometric
    # rounding to be visible, and both geometric micro-bevel attempts
    # measurably corrupted the mesh (absolute width: thousands of
    # degenerate sliver faces; percent width: rim vertices smeared
    # millimeters along the big face polygon's long edges, shredding
    # numerals into spike artifacts -- both confirmed on real dice).
    # Plain shade-smooth WITHOUT weighted normals is also wrong: the big
    # n-gon faces render pillowy because their boundary vertices average
    # in recess-wall and bevel normals.
    die_obj.data.polygons.foreach_set(
        "use_smooth", [True] * len(die_obj.data.polygons)
    )
    die_obj.data.update()
    wn = die_obj.modifiers.new(name="SoftNormals", type='WEIGHTED_NORMAL')
    wn.mode = 'FACE_AREA'
    wn.weight = 100
    wn.keep_sharp = False
    bpy.ops.object.modifier_apply(modifier=wn.name)

    bpy.ops.object.select_all(action='DESELECT')
    die_obj.select_set(True)
    bpy.context.view_layer.objects.active = die_obj

    quality_warning = _mesh_quality_warning(die_obj)
    manifest_record["mesh_quality_warnings"] = [quality_warning] if quality_warning else []
    if quality_warning:
        print(f"WARNING: {quality_warning}")

    blend_path = os.path.join(outdir, f"{asset_id}.blend")
    _save_blend_copy(blend_path)

    usd_path = os.path.join(outdir, f"{asset_id}.usd")
    bpy.ops.wm.usd_export(filepath=usd_path, selected_objects_only=True)

    stl_path = os.path.join(outdir, f"{asset_id}.stl")
    bpy.ops.wm.stl_export(filepath=stl_path, export_selected_objects=True)

    thumb_path = os.path.join(outdir, f"{asset_id}_thumb.png")
    _render_thumbnail(die_obj, thumb_path, size_mm)

    manifest_record["usd_path"] = f"{asset_id}.usd"
    manifest_record["stl_path"] = f"{asset_id}.stl"
    manifest_record["blend_path"] = f"{asset_id}.blend"
    manifest_record["thumbnail_path"] = f"{asset_id}_thumb.png"
    manifest_path = os.path.join(outdir, f"{asset_id}.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest_record, f, indent=2)

    return manifest_path


def _mesh_quality_warning(die_obj):
    """
    Scans the final (post-bevel) mesh for non-manifold junctions (edges with
    3+ linked faces), open boundary edges (edges with only 1 linked face),
    and zero-area faces -- smaller, separate classes of defects from the
    catastrophic angle-based-bevel-on-recess-edges bug (see
    test_export_asset_bevel_does_not_runaway_tessellate_an_engraved_die)
    that can still occur when bevel geometry sits very close to an
    engraved recess wall, confirmed empirically even with the
    weight-based bevel fix applied. Rather than attempt to eliminate
    this rarer case here (its root cause -- bevel/recess geometric
    proximity -- is distinct and harder), give it the same visibility
    every other known imperfection class in this pipeline already gets:
    a warning string for the manifest, so validate_dice_assets.py
    surfaces it instead of a possibly-defective mesh shipping silently.

    1e-9 matches the scale used throughout this session's investigation:
    legitimate die faces are many orders of magnitude larger than this
    at any realistic size_mm, so it separates true degenerate slivers
    from small-but-real geometry without needing a per-die-size-relative
    threshold.
    """
    bm = bmesh.new()
    bm.from_mesh(die_obj.data)
    non_manifold_junctions = sum(1 for e in bm.edges if len(e.link_faces) > 2)
    boundary_edges = sum(1 for e in bm.edges if len(e.link_faces) == 1)
    zero_area = sum(1 for f in bm.faces if f.calc_area() < 1e-9)
    bm.free()

    if non_manifold_junctions == 0 and boundary_edges == 0 and zero_area == 0:
        return None

    return (
        f"{die_obj.name}: {non_manifold_junctions} non-manifold junction edge(s), "
        f"{boundary_edges} open boundary edge(s), and "
        f"{zero_area} zero-area/degenerate face(s) found in the final "
        f"exported mesh."
    )


def _save_blend_copy(blend_path):
    """
    Saves the current .blend state as a standalone copy the user can open
    directly in Blender, with color/texture/material immediately visible.
    Three things have to be handled explicitly here that usd_export/
    stl_export don't need to worry about, since save_as_mainfile has no
    "selected objects only" option -- it always saves Blender's entire
    current file:

    1. Blender's own default-startup scene (present in every
       `blender --background --python ...` session that doesn't load an
       explicit .blend file) links a "Cube", "Light", and "Camera" object
       into the scene. Nothing in this pipeline uses them, but they'd
       otherwise silently end up saved into every single asset's .blend
       alongside the actual die.
    2. This function runs once per asset inside one long-running batch
       session (see orchestrator.generate_batch/generate_set_batch). Each
       previous asset's die object is removed via
       bpy.data.objects.remove(..., do_unlink=True) at the end of its own
       iteration, which unlinks it from the scene but leaves its
       mesh/material/image data-blocks resident in bpy.data with zero
       users. Without purging these first, every later asset's saved
       .blend would accumulate every earlier asset's orphaned data too
       (confirmed empirically: mesh/material counts and file size grew on
       every iteration of a save loop without this purge, and stayed flat
       once it was added).
    3. Blender's default viewport shading mode ("Solid") doesn't evaluate
       the shader node graph at all -- it shows a material's separate
       diffuse_color property instead (see materials.py). Setting every
       VIEW_3D viewport's shading to Material Preview means opening this
       .blend shows the die's real color/texture/material immediately, on
       whichever workspace tab (Layout, Modeling, Shading, etc.) happens
       to be active, with no manual shading-mode switch needed. Confirmed
       feasible even in --background mode: every one of Blender's default
       workspace screens has a real, settable VIEW_3D area.

    copy=True is required so saving this per-asset .blend never changes
    the long-running batch session's own "current file" identity.

    relative_remap=False is required for a 4th reason, found by actually
    reloading shipped .blend files fresh (rather than only inspecting the
    live in-memory session, which never surfaces this): this batch
    session never calls plain save_as_mainfile (only copy=True saves), so
    bpy.data.filepath stays empty/unset for the entire run. With the
    default relative_remap=True, Blender tries to remap already-loaded
    image paths (e.g. printed_decal's composited textures, loaded via
    bpy.data.images.load with a real absolute path) to be "//"-relative
    to that undefined current-file location, producing garbage like
    "//../../../../../../../data/raw/dice_assets/asset_00004_face0_composited.png"
    -- confirmed on every one of a real batch's printed_decal assets
    (51/51 affected), and confirmed fixed by explicitly setting
    relative_remap=False across a multi-iteration save loop at the same
    directory depth as this project, which leaves each image's already-
    correct absolute path untouched and resolvable after a fresh reload.
    """
    for name in ("Cube", "Light", "Camera"):
        stray = bpy.data.objects.get(name)
        if stray is not None:
            bpy.data.objects.remove(stray, do_unlink=True)

    bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)

    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type != 'VIEW_3D':
                continue
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = 'MATERIAL'

    bpy.ops.wm.save_as_mainfile(
        filepath=blend_path, copy=True, check_existing=False, relative_remap=False,
    )


def _render_thumbnail(die_obj, thumb_path, size_mm, resolution=512):
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.film_transparent = True

    cam_data = bpy.data.cameras.new(f"{die_obj.name}_cam")
    # Widen the field of view (default lens is 50mm) so the die comfortably
    # fits the frame even at the generous distance below.
    cam_data.lens = 35
    cam_obj = bpy.data.objects.new(f"{die_obj.name}_cam", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    # The die's own vertices extend up to roughly size_mm in radius from the
    # origin (e.g. a d20's base vertices reach ~0.95 * size_mm). Placing the
    # camera at (dist, -dist, dist) puts it sqrt(3) * dist from the origin,
    # so dist must be well above size_mm to sit clearly outside the die's
    # geometry with headroom to frame the whole object.
    dist = size_mm * 2.2
    cam_obj.location = (dist, -dist, dist)
    direction = die_obj.location - cam_obj.location
    cam_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    scene.camera = cam_obj

    light_data = bpy.data.lights.new(f"{die_obj.name}_light", type='SUN')
    light_data.energy = 3.0
    light_obj = bpy.data.objects.new(f"{die_obj.name}_light", light_data)
    light_obj.location = (dist, dist, dist * 1.5)
    bpy.context.collection.objects.link(light_obj)

    scene.render.filepath = thumb_path
    bpy.ops.render.render(write_still=True)

    bpy.data.objects.remove(cam_obj, do_unlink=True)
    bpy.data.objects.remove(light_obj, do_unlink=True)
    bpy.data.cameras.remove(cam_data)
    bpy.data.lights.remove(light_data)
