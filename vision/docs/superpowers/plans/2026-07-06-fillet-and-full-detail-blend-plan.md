# Fillet Edges and Full-Detail .blend Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix two issues found when opening generated `.blend` files directly in Blender: dice appeared as blank grey polygons (materials never set the legacy `diffuse_color` property Blender's default "Solid" viewport shading reads), and edges/corners showed a flat single-facet chamfer instead of a smooth rounded fillet. Also reorder `export_asset` so the `.blend` is saved right after the model is finished (bevel/fillet applied), before any of USD/STL/thumbnail export, and make every viewport default to Material Preview shading so color/texture/material are visible immediately on open.

**Architecture:** Two independent, small changes to already-existing files: `src/dice_gen/materials.py` gains a `diffuse_color` assignment in each of its two material-building functions; `src/dice_gen/exporter.py` gets its Bevel modifier's `segments` raised from the default 1 to 8, its `_save_blend_copy` call moved earlier in `export_asset`'s sequence, and a new viewport-shading-to-Material-Preview step added inside `_save_blend_copy`. No new files, no new dependencies.

**Tech Stack:** Same as the rest of `src/dice_gen/` — Blender 5.1.2's bundled `bpy`, no external pip packages. Every fact below (fillet vertex/face counts, `VIEW_3D` space availability in `--background` mode, `diffuse_color`'s default value) was confirmed by running standalone scripts through `blender --background --python` during planning.

## Global Constraints

- Fillet is applied **uniformly to every die** — no per-asset randomization, no hard-edge option. Only `mod.segments` changes (1 → 8); `mod.width` (still `size_mm * bevel_fraction`, `bevel_fraction` still sampled 0.02-0.06 in `sampler.py`, unchanged), `mod.limit_method`, and `mod.angle_limit` are untouched.
- `materials.py`'s `diffuse_color` fix must mirror the SAME representative HSV-derived color already used for the Principled BSDF's Base Color — not a new/different color, and not skipped for procedural categories (marbled/speckled/glitter) just because their Base Color INPUT gets overridden by a node link afterward.
- `export_asset`'s new order: apply Bevel modifier (with fillet segments) → save `.blend` → export USD → export STL → render thumbnail. The `.blend` save must happen strictly before USD export, STL export, and the thumbnail render (which creates its own temporary camera/light objects that must not exist yet at `.blend`-save time).
- Every `VIEW_3D` viewport's `shading.type` must be set to `'MATERIAL'` before saving the `.blend`, across every workspace screen (confirmed feasible in `--background` mode: all 10 default workspaces have real, settable `VIEW_3D` areas even headlessly).
- No change to USD/STL export logic, engrave/decal glyph pipelines, or the shader node graphs themselves — `diffuse_color` is a supplementary flat-color property for Solid-mode display only.

---

## File Structure

- Modify: `src/dice_gen/materials.py` — `build_material`, `build_fill_material` each set `mat.diffuse_color`.
- Modify: `src/dice_gen/exporter.py` — `export_asset`'s call order; Bevel modifier gains `segments = 8`; `_save_blend_copy` gains the viewport-shading step.
- Test: `tests/blender/test_materials.py` — 2 new tests.
- Test: `tests/blender/test_exporter.py` — 3 new tests.

---

### Task 1: `materials.py` sets `diffuse_color` for Solid-shading display

**Files:**
- Modify: `src/dice_gen/materials.py:20-73` (`build_material`), `:83-91` (`build_fill_material`)
- Test: `tests/blender/test_materials.py`

**Interfaces:**
- Consumes: nothing new — same `_hsv_to_rgba(h, s, v, a=1.0)` helper already in this file.
- Produces: nothing new consumed by other tasks — this task is self-contained and independent of Task 2.

- [ ] **Step 1: Write the failing tests**

Add these two tests to `tests/blender/test_materials.py`, right after `test_build_fill_material_returns_valid_material` (i.e. right before the `run()` function):

```python
def test_build_material_sets_diffuse_color_for_solid_shading_across_all_categories():
    """
    materials.py builds every material via shader nodes, but Blender's
    default "Solid" viewport shading mode reads the separate legacy
    material.diffuse_color property instead of evaluating the node graph.
    Confirmed empirically (opening a real generated .blend headlessly)
    that diffuse_color was left at Blender's own default (0.8, 0.8, 0.8,
    1.0) grey while the Principled BSDF's Base Color held the correct
    color -- meaning every die appeared as a blank grey polygon in Solid
    shading despite having fully correct material data underneath. This
    checks diffuse_color mirrors the same representative HSV-derived
    color used for Base Color, across every material category --
    including "marbled"/"speckled"/"glitter", where the Base Color INPUT
    itself gets overridden by a procedural node link afterward, since
    diffuse_color should still reflect the original flat representative
    color regardless.
    """
    from dice_gen import materials

    params = {
        "hue": 0.5, "saturation": 0.7, "value": 0.6, "roughness": 0.3,
        "ior": 1.45, "transmission": 0.9, "noise_scale": 5.0,
        "secondary_hue": 0.1, "sparkle_density": 40.0, "speckle_density": 60.0,
    }
    expected = materials._hsv_to_rgba(params["hue"], params["saturation"], params["value"])

    for category in materials.MATERIAL_CATEGORIES:
        mat = materials.build_material("test_die", category, params)
        actual = tuple(mat.diffuse_color)
        assert all(abs(a - e) < 1e-5 for a, e in zip(actual, expected)), (
            f"{category}: expected diffuse_color close to {expected}, got {actual}"
        )


def test_build_fill_material_sets_diffuse_color_to_match_fill_hue():
    from dice_gen import materials

    params = {"hue": 0.2, "saturation": 0.8, "value": 0.5, "roughness": 0.4}
    fill_hue = (params["hue"] + 0.5) % 1.0
    expected = materials._hsv_to_rgba(fill_hue, 0.8, 0.9)

    mat = materials.build_fill_material("test_die", params)
    actual = tuple(mat.diffuse_color)
    assert all(abs(a - e) < 1e-5 for a, e in zip(actual, expected)), (
        f"expected diffuse_color close to {expected}, got {actual}"
    )
```

Add both function names to `run()` at the bottom of `tests/blender/test_materials.py` (currently lines 83-89), right after `test_build_fill_material_returns_valid_material()`:

```python
    test_build_material_sets_diffuse_color_for_solid_shading_across_all_categories()
    test_build_fill_material_sets_diffuse_color_to_match_fill_hue()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/saps/projects/Dice-Detection && blender --background --python tests/blender/test_materials.py; echo "exit=$?"`
Expected: FAIL (non-zero exit). Both new tests fail because `mat.diffuse_color` is still Blender's default `(0.8, 0.8, 0.8, 1.0)`, not close to the expected HSV-derived color.

- [ ] **Step 3: Implement the diffuse_color assignment**

In `src/dice_gen/materials.py`, in `build_material`, add `mat.diffuse_color = base_color` right after the existing `base_color = _hsv_to_rgba(...)` line and the two `bsdf.inputs[...]` assignments that follow it, so the top of the function reads:

```python
def build_material(die_name, category, params):
    mat = bpy.data.materials.new(name=f"{die_name}_{category}")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]

    base_color = _hsv_to_rgba(params["hue"], params["saturation"], params["value"])
    bsdf.inputs["Base Color"].default_value = base_color
    bsdf.inputs["Roughness"].default_value = params["roughness"]
    mat.diffuse_color = base_color

    if category == "opaque":
```

(Everything from `if category == "opaque":` onward is unchanged.)

In `build_fill_material`, add `mat.diffuse_color = fill_color` using a new `fill_color` variable so the function reads:

```python
def build_fill_material(die_name, params):
    """Plain-color material for painted glyph fill (material slot 1)."""
    fill_hue = (params["hue"] + 0.5) % 1.0
    fill_color = _hsv_to_rgba(fill_hue, 0.8, 0.9)
    mat = bpy.data.materials.new(name=f"{die_name}_fill")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = fill_color
    bsdf.inputs["Roughness"].default_value = 0.4
    mat.diffuse_color = fill_color
    return mat
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/saps/projects/Dice-Detection && blender --background --python tests/blender/test_materials.py; echo "exit=$?"`
Expected: `ALL TESTS PASSED` and `exit=0`.

- [ ] **Step 5: Commit**

```bash
git add src/dice_gen/materials.py tests/blender/test_materials.py
git commit -m "feat: set diffuse_color so materials show correctly in Solid viewport shading"
```

---

### Task 2: `exporter.py` fillets edges, reorders export, defaults to Material Preview shading

**Files:**
- Modify: `src/dice_gen/exporter.py` (module docstring, `export_asset`, `_save_blend_copy`)
- Test: `tests/blender/test_exporter.py`

**Interfaces:**
- Consumes: nothing new from Task 1 — independent of it (both tasks touch different files and can land in either order; this plan just lists materials.py first).
- Produces: nothing new consumed by later tasks — Task 3 is verification-only.

- [ ] **Step 1: Write the failing tests**

Add these three tests to `tests/blender/test_exporter.py`, right after `test_export_asset_blend_files_do_not_accumulate_across_multiple_exports` (i.e. right before the `run()` function):

```python
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
    from dice_gen import geometry, materials, exporter

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
    from dice_gen import geometry, materials, exporter

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
    from dice_gen import geometry, materials, exporter

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
```

Add all three function names to the `run()` function at the bottom of `tests/blender/test_exporter.py`, right after `test_export_asset_blend_files_do_not_accumulate_across_multiple_exports()`:

```python
    test_export_asset_saves_blend_before_usd_and_stl_export()
    test_export_asset_uses_fillet_segments_not_flat_chamfer()
    test_save_blend_copy_sets_every_view3d_to_material_preview_shading()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/saps/projects/Dice-Detection && blender --background --python tests/blender/test_exporter.py; echo "exit=$?"`
Expected: FAIL (non-zero exit). `test_export_asset_saves_blend_before_usd_and_stl_export` fails because `.blend` is currently saved AFTER `usd_export`/`stl_export`, so both files already exist when `_save_blend_copy` runs. `test_export_asset_uses_fillet_segments_not_flat_chamfer` fails because the shipped Bevel modifier still defaults to `segments=1`, so `actual_polys` matches the 1-segment chamfer reference instead of the 8-segment fillet reference. `test_save_blend_copy_sets_every_view3d_to_material_preview_shading` fails because `_save_blend_copy` doesn't touch viewport shading yet, so it's still Blender's default `'SOLID'`.

- [ ] **Step 3: Implement the exporter.py changes**

Replace the entire contents of `src/dice_gen/exporter.py` with:

```python
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
single flat chamfer facet. limit_method='ANGLE' (not 'NONE') ensures it
only rounds those structural edges (e.g. a cube's ~90 degree edges) while
leaving shallow engraved-numeral recesses (much shallower angle deltas)
crisp.
"""
import json
import math
import os

import bpy


def export_asset(die_obj, manifest_record, outdir, bevel_fraction, size_mm):
    os.makedirs(outdir, exist_ok=True)
    asset_id = manifest_record["asset_id"]

    mod = die_obj.modifiers.new(name="Bevel", type='BEVEL')
    mod.width = size_mm * bevel_fraction
    mod.segments = 8
    mod.limit_method = 'ANGLE'
    mod.angle_limit = math.radians(35)
    bpy.context.view_layer.objects.active = die_obj
    bpy.ops.object.modifier_apply(modifier=mod.name)

    bpy.ops.object.select_all(action='DESELECT')
    die_obj.select_set(True)
    bpy.context.view_layer.objects.active = die_obj

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

    bpy.ops.wm.save_as_mainfile(filepath=blend_path, copy=True, check_existing=False)


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
```

(Only the module docstring, `export_asset`'s body, and `_save_blend_copy`'s body changed; `_render_thumbnail` is reproduced verbatim/unchanged so the file replacement above is complete and correct.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/saps/projects/Dice-Detection && blender --background --python tests/blender/test_exporter.py; echo "exit=$?"`
Expected: `ALL TESTS PASSED` and `exit=0`.

Also re-run the two other Blender-dependent suites that call `export_asset` indirectly or directly, to confirm no regressions:

Run: `cd /home/saps/projects/Dice-Detection && blender --background --python tests/blender/test_orchestrator.py; echo "exit=$?"`
Expected: `ALL TESTS PASSED` and `exit=0`.

Run: `cd /home/saps/projects/Dice-Detection && blender --background --python tests/blender/test_glyphs.py; echo "exit=$?"`
Expected: `ALL TESTS PASSED` and `exit=0`.

- [ ] **Step 5: Commit**

```bash
git add src/dice_gen/exporter.py tests/blender/test_exporter.py
git commit -m "feat: fillet edges, save .blend before export, default to Material Preview shading"
```

---

### Task 3: Regenerate the existing batch and verify

**Files:** none modified — verification only.

**Interfaces:** none — this task consumes Tasks 1-2's finished code as-is.

- [ ] **Step 1: Regenerate the existing 100-asset batch in place**

Run: `cd /home/saps/projects/Dice-Detection && blender --background --python scripts/generate_dice_assets.py -- --count 100 --seed 42 --outdir data/raw/dice_assets`

Expected: completes without crashing, printing `Generated: 100, Failed: 0` (or matching whatever count the batch already had before this plan — this plan doesn't change engrave correctness).

- [ ] **Step 2: Validate the regenerated batch**

Run: `cd /home/saps/projects/Dice-Detection && python3 scripts/validate_dice_assets.py data/raw/dice_assets`

Expected: same result as before this plan (the one known `engraving_warnings` entry from asset_00064, unrelated to this plan's scope, is acceptable — this plan doesn't touch engrave detection). If NEW errors appear (e.g. missing `.blend`/`.stl`/USD files), investigate as a regression.

- [ ] **Step 3: Visually confirm the fix by reading a thumbnail image**

Use the Read tool to view `data/raw/dice_assets/asset_00000_thumb.png` directly (this doesn't test the fillet/diffuse_color/shading fixes themselves, since thumbnails were already rendering correctly before this plan via EEVEE with real lighting — but it's a fast sanity check that regeneration didn't break the render path).

- [ ] **Step 4: Confirm the .blend contains a fillet, not a chamfer, and opens with Material Preview shading**

Run: `cd /home/saps/projects/Dice-Detection && blender --background data/raw/dice_assets/asset_00001.blend --python-expr "import bpy; obj = [o for o in bpy.data.objects if o.type == 'MESH'][0]; print('VERTS:', len(obj.data.vertices)); print('SHADING:', [s.shading.type for scr in bpy.data.screens for a in scr.areas if a.type == 'VIEW_3D' for s in a.spaces if s.type == 'VIEW_3D'])"`

Expected: `VERTS:` prints a number consistent with an 8-segment fillet having been applied (i.e. clearly more than the ~24-vertex range a 1-segment chamfer would produce on a similarly simple die type, though exact counts vary by die type and how much engraving each asset has), and `SHADING:` prints a list containing only `'MATERIAL'` entries.

No commit for this task — it's a verification pass over regenerated (gitignored) data, not a code change.
