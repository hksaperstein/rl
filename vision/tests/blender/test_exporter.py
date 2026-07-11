import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from _harness import run_and_report


def test_export_asset_writes_usd_manifest_and_thumbnail():
    import bpy
    from datagen.domains.dice import geometry
    from datagen import materials, exporter

    obj = geometry.build_die_base_mesh("d6", size_mm=16.0)
    mat = materials.build_material("d6", "opaque", {"hue": 0.3, "saturation": 0.7, "value": 0.6, "roughness": 0.4})
    materials.apply_material(obj, mat)

    with tempfile.TemporaryDirectory() as outdir:
        record = {"asset_id": "test_d6", "die_type": "d6"}
        manifest_path = exporter.export_asset(obj, record, outdir, bevel_fraction=0.04, size_mm=16.0)

        usd_path = os.path.join(outdir, "test_d6.usd")
        thumb_path = os.path.join(outdir, "test_d6_thumb.png")
        stl_path = os.path.join(outdir, "test_d6.stl")
        blend_path = os.path.join(outdir, "test_d6.blend")

        assert os.path.exists(usd_path) and os.path.getsize(usd_path) > 0
        assert os.path.exists(thumb_path) and os.path.getsize(thumb_path) > 0
        assert os.path.exists(stl_path) and os.path.getsize(stl_path) > 0
        assert os.path.exists(blend_path) and os.path.getsize(blend_path) > 0
        assert os.path.exists(manifest_path)

        with open(manifest_path) as f:
            loaded = json.load(f)
        assert loaded["usd_path"] == "test_d6.usd"
        assert loaded["thumbnail_path"] == "test_d6_thumb.png"
        assert loaded["stl_path"] == "test_d6.stl"
        assert loaded["blend_path"] == "test_d6.blend"

    bpy.data.objects.remove(obj, do_unlink=True)


def test_export_asset_blend_file_contains_only_the_die_object():
    """
    export_asset's saved .blend must contain exactly the die object, not
    Blender's own default-startup "Cube"/"Light"/"Camera" objects that a
    `blender --background --python ...` session links into the scene by
    default whenever no explicit .blend is loaded (confirmed empirically
    during planning: a fresh headless session's scene contains exactly
    those three objects). Unlike usd_export/stl_export (both take a
    selected-objects-only flag), save_as_mainfile has no such option and
    always saves the entire current file, so this guarantee has to be
    established explicitly in export_asset itself.
    """
    import bpy
    from datagen.domains.dice import geometry
    from datagen import materials, exporter

    obj = geometry.build_die_base_mesh("d6", size_mm=16.0)
    mat = materials.build_material("d6", "opaque", {"hue": 0.3, "saturation": 0.7, "value": 0.6, "roughness": 0.4})
    materials.apply_material(obj, mat)
    obj_name = obj.name

    with tempfile.TemporaryDirectory() as outdir:
        record = {"asset_id": "test_single", "die_type": "d6"}
        exporter.export_asset(obj, record, outdir, bevel_fraction=0.04, size_mm=16.0)

        blend_path = os.path.join(outdir, "test_single.blend")
        with bpy.data.libraries.load(blend_path) as (data_from, _data_to):
            objects_in_file = list(data_from.objects)

        assert objects_in_file == [obj_name], (
            f"expected the saved .blend to contain only the die object "
            f"{obj_name!r} and nothing else (no default Cube/Light/Camera), "
            f"got {objects_in_file}"
        )

    bpy.data.objects.remove(obj, do_unlink=True)


def test_export_asset_blend_files_do_not_accumulate_across_multiple_exports():
    """
    Regression test for orphaned-data-block accumulation across a batch:
    in one long-running `blender --background` session, each previous
    asset's die object is removed via bpy.data.objects.remove(...,
    do_unlink=True) (see orchestrator._generate_from_params) which unlinks
    it from the scene but leaves its mesh/material data resident in
    bpy.data with zero users. Confirmed empirically during planning that
    without purging these before each save_as_mainfile call, every later
    asset's .blend silently accumulates every earlier asset's orphaned
    mesh/material data too (mesh/material counts and file size grew on
    every iteration of a throwaway repro loop; purging before each save
    kept both flat). This test exports two different dice in the same
    session and asserts each one's saved .blend contains only its own
    object/mesh/material data, not the other's.
    """
    import bpy
    from datagen.domains.dice import geometry
    from datagen import materials, exporter

    obj1 = geometry.build_die_base_mesh("d6", size_mm=16.0)
    mat1 = materials.build_material("d6", "opaque", {"hue": 0.3, "saturation": 0.7, "value": 0.6, "roughness": 0.4})
    materials.apply_material(obj1, mat1)
    obj1_name = obj1.name
    mesh1_name = obj1.data.name
    mat1_name = mat1.name

    with tempfile.TemporaryDirectory() as outdir:
        record1 = {"asset_id": "test_first", "die_type": "d6"}
        exporter.export_asset(obj1, record1, outdir, bevel_fraction=0.04, size_mm=16.0)
        bpy.data.objects.remove(obj1, do_unlink=True)

        obj2 = geometry.build_die_base_mesh("d4", size_mm=14.0)
        mat2 = materials.build_material("d4", "opaque", {"hue": 0.6, "saturation": 0.5, "value": 0.5, "roughness": 0.3})
        materials.apply_material(obj2, mat2)
        obj2_name = obj2.name
        mesh2_name = obj2.data.name
        mat2_name = mat2.name

        record2 = {"asset_id": "test_second", "die_type": "d4"}
        exporter.export_asset(obj2, record2, outdir, bevel_fraction=0.04, size_mm=14.0)

        blend1_path = os.path.join(outdir, "test_first.blend")
        blend2_path = os.path.join(outdir, "test_second.blend")

        with bpy.data.libraries.load(blend1_path) as (data_from, _data_to):
            objects_in_first = list(data_from.objects)
            meshes_in_first = list(data_from.meshes)
            materials_in_first = list(data_from.materials)
        with bpy.data.libraries.load(blend2_path) as (data_from, _data_to):
            objects_in_second = list(data_from.objects)
            meshes_in_second = list(data_from.meshes)
            materials_in_second = list(data_from.materials)

        assert objects_in_first == [obj1_name], (
            f"the first asset's .blend should contain only its own die "
            f"({obj1_name!r}); obj1 was already removed from the live "
            f"scene before this check, so this reads back what was "
            f"actually written to disk, got {objects_in_first}"
        )
        assert objects_in_second == [obj2_name], (
            f"the second asset's .blend should contain only its own die "
            f"({obj2_name!r}), not any orphaned data left over from the "
            f"first asset, got {objects_in_second}"
        )

        # Mesh data for the first asset
        assert meshes_in_first == [mesh1_name], (
            f"the first asset's .blend should contain only its own mesh data "
            f"({mesh1_name!r}), got {meshes_in_first}"
        )
        # Material data for the first asset
        assert materials_in_first == [mat1_name], (
            f"the first asset's .blend should contain only its own material "
            f"({mat1_name!r}), got {materials_in_first}"
        )

        # Mesh data for the second asset must not include the first die's mesh
        assert mesh1_name not in meshes_in_second, (
            f"the second asset's .blend should NOT contain the first asset's mesh "
            f"({mesh1_name!r}), which would indicate orphaned data accumulation; "
            f"got meshes: {meshes_in_second}"
        )
        assert meshes_in_second == [mesh2_name], (
            f"the second asset's .blend should contain only its own mesh data "
            f"({mesh2_name!r}), got {meshes_in_second}"
        )

        # Material data for the second asset must not include the first die's material
        assert mat1_name not in materials_in_second, (
            f"the second asset's .blend should NOT contain the first asset's material "
            f"({mat1_name!r}), which would indicate orphaned data accumulation; "
            f"got materials: {materials_in_second}"
        )
        assert materials_in_second == [mat2_name], (
            f"the second asset's .blend should contain only its own material "
            f"({mat2_name!r}), got {materials_in_second}"
        )

        bpy.data.objects.remove(obj2, do_unlink=True)


def test_export_asset_saves_blend_before_usd_and_stl_export():
    """
    Regression test for the ordering requirement: the .blend must be saved
    right after the model is finished (bevel/fillet applied), BEFORE any
    of USD/STL/thumbnail export, so the .blend always represents the
    single definitive source state everything else is derived from -- and
    so the thumbnail render's own temporary camera/light objects (created
    and removed AFTER this point) never exist yet at .blend-save time,
    avoiding the same kind of leak _save_blend_copy already guards against
    for Blender's default startup Cube/Light/Camera.

    Verified by wrapping the real _save_blend_copy and checking, at the
    exact moment it's called, that the USD and STL files it must precede
    don't exist on disk yet.
    """
    import bpy
    from datagen.domains.dice import geometry
    from datagen import materials, exporter

    obj = geometry.build_die_base_mesh("d6", size_mm=16.0)
    mat = materials.build_material("d6", "opaque", {"hue": 0.3, "saturation": 0.7, "value": 0.6, "roughness": 0.4})
    materials.apply_material(obj, mat)

    with tempfile.TemporaryDirectory() as outdir:
        usd_path = os.path.join(outdir, "test_order.usd")
        stl_path = os.path.join(outdir, "test_order.stl")

        real_save_blend_copy = exporter._save_blend_copy
        observed = {}

        def spy_save_blend_copy(blend_path):
            observed["usd_existed"] = os.path.exists(usd_path)
            observed["stl_existed"] = os.path.exists(stl_path)
            return real_save_blend_copy(blend_path)

        exporter._save_blend_copy = spy_save_blend_copy
        try:
            record = {"asset_id": "test_order", "die_type": "d6"}
            exporter.export_asset(obj, record, outdir, bevel_fraction=0.04, size_mm=16.0)
        finally:
            exporter._save_blend_copy = real_save_blend_copy

        assert observed["usd_existed"] is False, (
            "the USD file already existed when _save_blend_copy ran -- "
            ".blend must be saved BEFORE usd_export, not after"
        )
        assert observed["stl_existed"] is False, (
            "the STL file already existed when _save_blend_copy ran -- "
            ".blend must be saved BEFORE stl_export, not after"
        )

    bpy.data.objects.remove(obj, do_unlink=True)


def test_export_asset_uses_fillet_segments_not_flat_chamfer():
    """
    Regression test: export_asset's Bevel modifier must use segments=8 (a
    smooth rounded fillet) rather than the default segments=1 (a single
    flat chamfer facet). Verified by independently building a second,
    untouched die with the exact bevel/limit settings this test expects
    and comparing its resulting vertex/face count against what
    export_asset actually produced on the first die -- rather than
    hardcoding an expected vertex/face count as a magic number disconnected
    from the modifier settings themselves. Also confirms the result does
    NOT match a plain 1-segment chamfer's count, so a regression back to
    the old default wouldn't slip through by coincidence.
    """
    import bpy
    import math
    from datagen.domains.dice import geometry
    from datagen import materials, exporter

    size_mm = 16.0
    bevel_fraction = 0.04

    obj = geometry.build_die_base_mesh("d6", size_mm=size_mm)
    mat = materials.build_material("d6", "opaque", {"hue": 0.3, "saturation": 0.7, "value": 0.6, "roughness": 0.4})
    materials.apply_material(obj, mat)

    with tempfile.TemporaryDirectory() as outdir:
        record = {"asset_id": "test_fillet", "die_type": "d6"}
        exporter.export_asset(obj, record, outdir, bevel_fraction=bevel_fraction, size_mm=size_mm)

    actual_verts = len(obj.data.vertices)
    actual_polys = len(obj.data.polygons)

    reference_obj = geometry.build_die_base_mesh("d6", size_mm=size_mm)
    ref_mod = reference_obj.modifiers.new(name="Bevel", type='BEVEL')
    ref_mod.width = size_mm * bevel_fraction
    ref_mod.segments = 8
    ref_mod.limit_method = 'ANGLE'
    ref_mod.angle_limit = math.radians(35)
    bpy.context.view_layer.objects.active = reference_obj
    bpy.ops.object.modifier_apply(modifier=ref_mod.name)

    expected_verts = len(reference_obj.data.vertices)
    expected_polys = len(reference_obj.data.polygons)

    assert actual_verts == expected_verts, (
        f"expected {expected_verts} verts (8-segment fillet reference), "
        f"got {actual_verts} -- if this matches a 1-segment chamfer's "
        f"count instead, segments=8 was not applied"
    )
    assert actual_polys == expected_polys, (
        f"expected {expected_polys} polys (8-segment fillet reference), "
        f"got {actual_polys}"
    )

    single_segment_obj = geometry.build_die_base_mesh("d6", size_mm=size_mm)
    chamfer_mod = single_segment_obj.modifiers.new(name="Bevel", type='BEVEL')
    chamfer_mod.width = size_mm * bevel_fraction
    chamfer_mod.limit_method = 'ANGLE'
    chamfer_mod.angle_limit = math.radians(35)
    bpy.context.view_layer.objects.active = single_segment_obj
    bpy.ops.object.modifier_apply(modifier=chamfer_mod.name)
    chamfer_polys = len(single_segment_obj.data.polygons)

    assert actual_polys != chamfer_polys, (
        f"exported die's face count ({actual_polys}) matches a plain "
        f"1-segment chamfer's count ({chamfer_polys}) -- segments=8 fillet "
        f"was not actually applied"
    )

    bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.objects.remove(reference_obj, do_unlink=True)
    bpy.data.objects.remove(single_segment_obj, do_unlink=True)


def test_export_asset_bevel_does_not_runaway_tessellate_an_engraved_die():
    """
    Regression test for a confirmed real bug: the Bevel modifier's OLD
    limit_method='ANGLE' (35 degrees) could not distinguish a die's large
    structural edges from the many small, similarly-steep-angled edges
    bounding an engraved numeral's recess, so 8-segment rounding got
    applied inside tiny recess geometry too -- producing catastrophic
    degenerate output (confirmed on a real 100-asset batch: e.g. a single
    d20/cjk_numerals asset had 57,353 degenerate faces after export,
    traced to this specific step via isolated synthetic reproduction: a
    single engraved cube went from 1,314 to 17,754 faces and gained a
    zero-area face purely from the OLD angle-based bevel step). The fix
    (limit_method='WEIGHT', paired with geometry.build_die_base_mesh
    marking only the pristine polyhedron's edges -- see test_geometry.py)
    keeps bevel's face-count growth modest and bounded regardless of
    glyph complexity, since it now only ever rounds the die's original
    structural edges.
    """
    import bmesh
    import bpy
    from datagen.domains.dice import geometry, glyphs
    from datagen import materials, exporter

    size_mm = 19.68719216365438
    obj = geometry.build_die_base_mesh("d8", size_mm=size_mm)
    face_pairs = geometry.compute_opposite_face_pairs(obj)
    assignment = {}
    for i, (a, b) in enumerate(face_pairs):
        assignment[a] = i + 1
        assignment[b] = 9 - (i + 1)

    glyphs.apply_engraved_glyphs(
        obj, "d8", assignment, "arabic_numerals", "painted",
        "font_serif_regular", size_mm,
    )
    mat = materials.build_material("d8", "metallic", {
        "hue": 0.06, "saturation": 0.66, "value": 0.23, "roughness": 0.07,
    })
    materials.apply_material(obj, mat)

    bm_pre = bmesh.new()
    bm_pre.from_mesh(obj.data)
    faces_pre = len(bm_pre.faces)
    bm_pre.free()

    with tempfile.TemporaryDirectory() as outdir:
        record = {"asset_id": "test_bevel_d8", "die_type": "d8"}
        exporter.export_asset(obj, record, outdir, bevel_fraction=0.0358, size_mm=size_mm)

    bm_post = bmesh.new()
    bm_post.from_mesh(obj.data)
    faces_post = len(bm_post.faces)
    bm_post.free()

    # The RecessSoften geometric pass that briefly relaxed this bound to
    # 5x was removed (it shredded numerals into spike artifacts --
    # softening is now done via shading normals, which adds no faces),
    # so the original 2x bound is restored: structural-edge rounding is
    # the only intentional face growth again.
    assert faces_pre < faces_post < faces_pre * 2, (
        f"bevel grew face count from {faces_pre} to {faces_post} -- expected "
        f"some growth from rounding structural edges (faces_pre < faces_post), "
        f"but not the runaway growth that matches the confirmed angle-based "
        f"tessellation bug (faces_post < faces_pre * 2)"
    )

    assert all(p.use_smooth for p in obj.data.polygons), (
        "export_asset must smooth-shade every polygon (the weighted-normal "
        "softening pass) -- flat-shaded output means the soften-by-shading "
        "step was skipped"
    )

    bpy.data.objects.remove(obj, do_unlink=True)


def test_save_blend_copy_sets_every_view3d_to_material_preview_shading():
    """
    _save_blend_copy must set every VIEW_3D viewport's shading to Material
    Preview before saving, so opening the resulting .blend immediately
    shows full color/texture/material regardless of which of Blender's
    default workspace tabs (Layout, Modeling, Shading, etc.) is active --
    without this, Blender's default "Solid" shading mode doesn't evaluate
    the shader node graph at all (materials.py's diffuse_color fix handles
    Solid mode's own separate fallback color; this handles making Material
    Preview the default instead, so textures show too).
    """
    import bpy
    from datagen.domains.dice import geometry
    from datagen import materials, exporter

    obj = geometry.build_die_base_mesh("d6", size_mm=16.0)
    mat = materials.build_material("d6", "opaque", {"hue": 0.3, "saturation": 0.7, "value": 0.6, "roughness": 0.4})
    materials.apply_material(obj, mat)

    with tempfile.TemporaryDirectory() as outdir:
        record = {"asset_id": "test_shading", "die_type": "d6"}
        exporter.export_asset(obj, record, outdir, bevel_fraction=0.04, size_mm=16.0)

    view3d_spaces = [
        space
        for screen in bpy.data.screens
        for area in screen.areas
        if area.type == 'VIEW_3D'
        for space in area.spaces
        if space.type == 'VIEW_3D'
    ]
    assert view3d_spaces, "expected at least one VIEW_3D space to check"
    for space in view3d_spaces:
        assert space.shading.type == 'MATERIAL', (
            f"expected every VIEW_3D viewport's shading.type to be "
            f"'MATERIAL' after export_asset, got {space.shading.type!r}"
        )

    bpy.data.objects.remove(obj, do_unlink=True)


def test_mesh_quality_warning_flags_a_degenerate_mesh():
    import bmesh
    import bpy
    from datagen.domains.dice import geometry
    from datagen import exporter

    obj = geometry.build_die_base_mesh("d6", size_mm=16.0)

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    v = bm.verts.new((0.0, 0.0, 0.0))
    v2 = bm.verts.new((0.0, 0.0, 0.0))
    v3 = bm.verts.new((0.0, 0.0, 0.0))
    bm.faces.new((v, v2, v3))
    bm.to_mesh(obj.data)
    obj.data.update()
    bm.free()

    warning = exporter._mesh_quality_warning(obj)
    assert warning is not None
    assert "degenerate" in warning.lower() or "zero-area" in warning.lower()

    bpy.data.objects.remove(obj, do_unlink=True)


def test_mesh_quality_warning_is_none_for_a_clean_die():
    import bpy
    from datagen.domains.dice import geometry
    from datagen import exporter

    obj = geometry.build_die_base_mesh("d6", size_mm=16.0)
    warning = exporter._mesh_quality_warning(obj)
    assert warning is None

    bpy.data.objects.remove(obj, do_unlink=True)


def test_export_asset_sets_mesh_quality_warnings_key():
    import bpy
    from datagen.domains.dice import geometry
    from datagen import materials, exporter

    obj = geometry.build_die_base_mesh("d6", size_mm=16.0)
    mat = materials.build_material("d6", "opaque", {"hue": 0.3, "saturation": 0.7, "value": 0.6, "roughness": 0.4})
    materials.apply_material(obj, mat)

    with tempfile.TemporaryDirectory() as outdir:
        record = {"asset_id": "test_d6_quality", "die_type": "d6"}
        exporter.export_asset(obj, record, outdir, bevel_fraction=0.04, size_mm=16.0)
        assert record["mesh_quality_warnings"] == []

    bpy.data.objects.remove(obj, do_unlink=True)


def test_export_asset_blend_image_textures_resolve_after_fresh_reload():
    """
    Regression test for a bug only visible by actually reloading a saved
    .blend fresh, not by inspecting the live in-memory session (which is
    why every earlier check in this codebase missed it -- thumbnails and
    manual verification all rendered from the live session's already-
    loaded image data, never re-reading the saved file's own stored path).

    _save_blend_copy runs in a long-running batch session that never
    calls plain save_as_mainfile (only copy=True saves), so
    bpy.data.filepath stays empty/unset for the whole run. With the
    default relative_remap=True, Blender tried to remap already-loaded
    image paths (e.g. printed_decal's composited textures, loaded via
    bpy.data.images.load with a real absolute path) to be "//"-relative
    to that undefined current-file location, producing garbage paths
    like "//../../../../../../../data/raw/dice_assets/asset_00004_face0_composited.png"
    -- confirmed on every one of a real 51-asset printed_decal batch.
    Fixed by passing relative_remap=False.

    This test builds a die with a material that has a real image texture
    (mimicking apply_decal_glyphs's pattern: bpy.data.images.load with an
    absolute path, wired into a ShaderNodeTexImage), exports it, then
    RELOADS the saved .blend fresh via bpy.ops.wm.open_mainfile and
    checks every image datablock's stored path still resolves to an
    existing file -- the only way to catch this class of bug.
    """
    import bpy
    from datagen.domains.dice import geometry
    from datagen import materials, exporter

    with tempfile.TemporaryDirectory() as outdir:
        obj = geometry.build_die_base_mesh("d6", size_mm=16.0)
        base_mat = materials.build_material("d6", "opaque", {"hue": 0.3, "saturation": 0.7, "value": 0.6, "roughness": 0.4})
        materials.apply_material(obj, base_mat)

        # Mimic apply_decal_glyphs: a real image file, loaded via an
        # absolute path, wired into a material's Base Color.
        decal_image_path = os.path.join(outdir, "decal_texture.png")
        tmp_img = bpy.data.images.new("decal_texture", 8, 8)
        tmp_img.filepath_raw = decal_image_path
        tmp_img.file_format = 'PNG'
        tmp_img.save()
        bpy.data.images.remove(tmp_img)

        decal_mat = base_mat.copy()
        decal_mat.name = "d6_decal"
        nt = decal_mat.node_tree
        tex_node = nt.nodes.new("ShaderNodeTexImage")
        tex_node.image = bpy.data.images.load(decal_image_path)
        nt.links.new(tex_node.outputs["Color"], nt.nodes["Principled BSDF"].inputs["Base Color"])
        obj.data.materials.append(decal_mat)
        obj.data.polygons[0].material_index = len(obj.data.materials) - 1

        record = {"asset_id": "test_decal_reload", "die_type": "d6"}
        exporter.export_asset(obj, record, outdir, bevel_fraction=0.04, size_mm=16.0)

        blend_path = os.path.join(outdir, "test_decal_reload.blend")
        bpy.ops.wm.open_mainfile(filepath=blend_path)

        checked_any = False
        for img in bpy.data.images:
            if not img.filepath:
                continue
            checked_any = True
            real_path = bpy.path.abspath(img.filepath)
            assert os.path.exists(real_path), (
                f"image {img.name!r} has filepath {img.filepath!r} which "
                f"does not resolve to a real file after a fresh reload of "
                f"the saved .blend -- got {real_path!r}"
            )
        assert checked_any, "expected at least one image datablock with a filepath to check"


def run():
    test_export_asset_writes_usd_manifest_and_thumbnail()
    test_export_asset_blend_file_contains_only_the_die_object()
    test_export_asset_blend_files_do_not_accumulate_across_multiple_exports()
    test_export_asset_saves_blend_before_usd_and_stl_export()
    test_export_asset_uses_fillet_segments_not_flat_chamfer()
    test_export_asset_bevel_does_not_runaway_tessellate_an_engraved_die()
    test_mesh_quality_warning_flags_a_degenerate_mesh()
    test_mesh_quality_warning_is_none_for_a_clean_die()
    test_export_asset_sets_mesh_quality_warnings_key()
    test_save_blend_copy_sets_every_view3d_to_material_preview_shading()
    # Runs last: bpy.ops.wm.open_mainfile() fully replaces the session's
    # bpy.data, which would invalidate any objects/state earlier tests
    # still held references to.
    test_export_asset_blend_image_textures_resolve_after_fresh_reload()


run_and_report(run)
