# Blend and STL Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every asset the pipeline generates already gets a `.usd` file (for downstream/Isaac tooling) and a PNG thumbnail (for spot-checking). This plan adds a `.blend` file (openable directly in Blender's own UI) and a `.stl` file (for slicers/mesh tools) alongside them for every asset, as a permanent part of `scripts/generate_dice_assets.py`'s export step — not a one-off script. It also regenerates the existing verified 100-asset batch at `data/raw/dice_assets/` (seed=42) in place so those files actually exist on disk, and extends `validate_dice_assets.py` to check for them.

**Architecture:** `src/dice_gen/exporter.py`'s `export_asset` already exports USD and renders a thumbnail for a single die object per call, inside one long-running `blender --background` process that loops over every asset in a batch. This plan adds two more exports to that same function: `bpy.ops.wm.stl_export` (which, like `usd_export`, takes a selected-objects-only flag, so it naturally only ever contains the current die) and `bpy.ops.wm.save_as_mainfile(..., copy=True)` (which has NO such flag — it always saves Blender's entire current file). That second fact means a new helper, `_save_blend_copy`, has to explicitly guarantee the saved `.blend` contains only the current die: it removes Blender's own default-startup "Cube"/"Light"/"Camera" objects (present in every `blender --background --python ...` session that doesn't load an explicit `.blend`, confirmed empirically below) and purges orphaned data-blocks left over from every previously-exported die in the same batch session (also confirmed empirically to otherwise accumulate: mesh/material counts and per-file size grew on every iteration in a throwaway repro script). Both fixes are idempotent, so calling them on every single `export_asset` call (not just once per batch) keeps the function self-contained and correct even when called in isolation, e.g. from a test.

**Tech Stack:** Same as the rest of `src/dice_gen/` — Blender 5.1.2's bundled `bpy`, no external pip packages. `bpy.ops.wm.stl_export` and `bpy.ops.outliner.orphans_purge` were confirmed to exist with the exact signatures used below by running small standalone scripts through `blender --background --python` during planning.

## Global Constraints

- `.blend`/`.stl` export becomes a **permanent** part of `export_asset` — every future `generate_dice_assets.py` run produces them, not just this one batch.
- The saved `.blend` for a given asset must contain **only that asset's own die object** — no Blender default-startup objects (`Cube`/`Light`/`Camera`), and no orphaned mesh/material/image data left over from any other asset generated earlier in the same batch session. This was empirically verified to be a real risk, not a hypothetical: a throwaway repro script showed mesh/material counts and `.blend` file size growing on every iteration of a save-loop when this wasn't handled, and dropping to a flat, constant size once orphans were purged before each save.
- Use `bpy.ops.wm.save_as_mainfile(filepath=..., copy=True, check_existing=False)` — `copy=True` is required so saving a per-asset `.blend` never changes the long-running batch session's own "current file" identity.
- `validate_dice_assets.py`'s new `blend_path`/`stl_path` checks must use `record.get(...)` (not direct `record[...]` indexing) and skip the check entirely when the key is absent, exactly like the existing `engraving_warnings` check — this keeps every existing test/manifest in `tests/test_validate_dice_assets.py` that doesn't set these keys passing unchanged, and tolerates validating a manifest generated before this plan.
- Known, accepted limitation (do not try to fix as part of this plan): decal-method dice reference their face-texture PNGs by absolute filesystem path (already true today, unrelated to this plan) rather than packing them into the `.blend`. As long as the `.blend` and its sibling PNGs in the same output directory are kept together, textures resolve correctly; moving just the `.blend` file elsewhere on its own would show missing textures for decal-method dice. This is a pre-existing characteristic of the pipeline, not a regression introduced here, and packing images into `.blend` files is out of scope.
- Isaac Sim/Isaac Lab integration remains out of scope.

---

## File Structure

- Modify: `src/dice_gen/exporter.py` — `export_asset` gains STL + `.blend` export; new private helper `_save_blend_copy`.
- Modify: `scripts/validate_dice_assets.py` — `validate()` gains `blend_path`/`stl_path` existence/non-empty checks.
- Test: `tests/blender/test_exporter.py` — extended + 2 new tests.
- Test: `tests/test_validate_dice_assets.py` — 3 new tests.

---

### Task 1: `exporter.py` exports `.stl` and `.blend` alongside USD

**Files:**
- Modify: `src/dice_gen/exporter.py:16-42` (`export_asset`)
- Test: `tests/blender/test_exporter.py`

**Interfaces:**
- Consumes: nothing new — same `die_obj`, `manifest_record`, `outdir`, `bevel_fraction`, `size_mm` args `export_asset` already takes.
- Produces: `manifest_record["stl_path"]` and `manifest_record["blend_path"]` (both `f"{asset_id}.<ext>"`, written unconditionally, same convention as the existing `usd_path`/`thumbnail_path`) — consumed by Task 2's `validate_dice_assets.py` check.

- [ ] **Step 1: Write the failing tests**

Replace the entire contents of `tests/blender/test_exporter.py` with:

```python
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from _harness import run_and_report


def test_export_asset_writes_usd_manifest_and_thumbnail():
    import bpy
    from dice_gen import geometry, materials, exporter

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
    from dice_gen import geometry, materials, exporter

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
    object, not the other's.
    """
    import bpy
    from dice_gen import geometry, materials, exporter

    obj1 = geometry.build_die_base_mesh("d6", size_mm=16.0)
    mat1 = materials.build_material("d6", "opaque", {"hue": 0.3, "saturation": 0.7, "value": 0.6, "roughness": 0.4})
    materials.apply_material(obj1, mat1)
    obj1_name = obj1.name

    with tempfile.TemporaryDirectory() as outdir:
        record1 = {"asset_id": "test_first", "die_type": "d6"}
        exporter.export_asset(obj1, record1, outdir, bevel_fraction=0.04, size_mm=16.0)
        bpy.data.objects.remove(obj1, do_unlink=True)

        obj2 = geometry.build_die_base_mesh("d4", size_mm=14.0)
        mat2 = materials.build_material("d4", "opaque", {"hue": 0.6, "saturation": 0.5, "value": 0.5, "roughness": 0.3})
        materials.apply_material(obj2, mat2)
        obj2_name = obj2.name

        record2 = {"asset_id": "test_second", "die_type": "d4"}
        exporter.export_asset(obj2, record2, outdir, bevel_fraction=0.04, size_mm=14.0)

        blend1_path = os.path.join(outdir, "test_first.blend")
        blend2_path = os.path.join(outdir, "test_second.blend")

        with bpy.data.libraries.load(blend1_path) as (data_from, _data_to):
            objects_in_first = list(data_from.objects)
        with bpy.data.libraries.load(blend2_path) as (data_from, _data_to):
            objects_in_second = list(data_from.objects)

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

        bpy.data.objects.remove(obj2, do_unlink=True)


def run():
    test_export_asset_writes_usd_manifest_and_thumbnail()
    test_export_asset_blend_file_contains_only_the_die_object()
    test_export_asset_blend_files_do_not_accumulate_across_multiple_exports()


run_and_report(run)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/saps/projects/Dice-Detection && blender --background --python tests/blender/test_exporter.py; echo "exit=$?"`
Expected: FAIL (non-zero exit). The first test fails on the new `assert os.path.exists(stl_path)` (or the `blend_path`/manifest-key assertions right after) since `export_asset` doesn't write those yet. The second and third tests fail with a `FileNotFoundError` from `bpy.data.libraries.load` since no `.blend` file exists yet at all.

- [ ] **Step 3: Implement the export changes**

Replace the entire contents of `src/dice_gen/exporter.py` with:

```python
"""
Bakes the non-destructive edge bevel, exports the die as USD/STL/blend,
renders a thumbnail for visual spot-checking, and writes the per-asset JSON
manifest.

Bevel uses limit_method='ANGLE' (not 'NONE') so it only rounds the die's
structural edges (e.g. a cube's ~90 degree edges) while leaving shallow
engraved-numeral recesses (much shallower angle deltas) crisp.
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
    mod.limit_method = 'ANGLE'
    mod.angle_limit = math.radians(35)
    bpy.context.view_layer.objects.active = die_obj
    bpy.ops.object.modifier_apply(modifier=mod.name)

    bpy.ops.object.select_all(action='DESELECT')
    die_obj.select_set(True)
    bpy.context.view_layer.objects.active = die_obj

    usd_path = os.path.join(outdir, f"{asset_id}.usd")
    bpy.ops.wm.usd_export(filepath=usd_path, selected_objects_only=True)

    stl_path = os.path.join(outdir, f"{asset_id}.stl")
    bpy.ops.wm.stl_export(filepath=stl_path, export_selected_objects=True)

    blend_path = os.path.join(outdir, f"{asset_id}.blend")
    _save_blend_copy(blend_path)

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
    directly in Blender. Two things have to be handled explicitly here that
    usd_export/stl_export don't need to worry about, since save_as_mainfile
    has no "selected objects only" option -- it always saves Blender's
    entire current file:

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

    copy=True is required so saving this per-asset .blend never changes
    the long-running batch session's own "current file" identity.
    """
    for name in ("Cube", "Light", "Camera"):
        stray = bpy.data.objects.get(name)
        if stray is not None:
            bpy.data.objects.remove(stray, do_unlink=True)

    bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
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

(Only `export_asset` and the new `_save_blend_copy` changed; `_render_thumbnail` is reproduced verbatim/unchanged so the file replacement above is complete and correct.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/saps/projects/Dice-Detection && blender --background --python tests/blender/test_exporter.py; echo "exit=$?"`
Expected: `ALL TESTS PASSED` and `exit=0`.

Also re-run the two other Blender-dependent suites to confirm no regressions (both call `export_asset` indirectly or directly):

Run: `cd /home/saps/projects/Dice-Detection && blender --background --python tests/blender/test_orchestrator.py; echo "exit=$?"`
Expected: `ALL TESTS PASSED` and `exit=0`.

Run: `cd /home/saps/projects/Dice-Detection && blender --background --python tests/blender/test_glyphs.py; echo "exit=$?"`
Expected: `ALL TESTS PASSED` and `exit=0`.

- [ ] **Step 5: Commit**

```bash
git add src/dice_gen/exporter.py tests/blender/test_exporter.py
git commit -m "feat: export .stl and .blend alongside USD for every asset"
```

---

### Task 2: `validate_dice_assets.py` checks `blend_path`/`stl_path`

**Files:**
- Modify: `scripts/validate_dice_assets.py:20-42` (inside `validate()`'s `for record in manifest:` loop)
- Test: `tests/test_validate_dice_assets.py`

**Interfaces:**
- Consumes: `record.get("blend_path")` / `record.get("stl_path")` (from Task 1; may be absent on a manifest generated before this plan — must not raise `KeyError`).
- Produces: additional entries in `validate()`'s returned `errors` list. No signature change.

- [ ] **Step 1: Write the failing tests**

Add these three tests to `tests/test_validate_dice_assets.py`, right after `test_validate_does_not_flag_empty_engraving_warnings` (i.e. right before `test_validate_passes_for_well_formed_manifest`):

```python
def test_validate_flags_missing_blend_file(tmp_path):
    _write_manifest(tmp_path, [{
        "asset_id": "a1", "die_type": "d6", "num_sides": 6,
        "usd_path": "a1.usd", "thumbnail_path": "a1_thumb.png",
        "blend_path": "a1.blend", "stl_path": "a1.stl",
    }])
    open(os.path.join(tmp_path, "a1.usd"), "w").write("x")
    open(os.path.join(tmp_path, "a1_thumb.png"), "w").close()
    open(os.path.join(tmp_path, "a1.stl"), "w").write("x")
    # a1.blend deliberately not created

    errors = validate(str(tmp_path))
    assert any("missing .blend file" in e for e in errors), errors


def test_validate_flags_missing_stl_file(tmp_path):
    _write_manifest(tmp_path, [{
        "asset_id": "a1", "die_type": "d6", "num_sides": 6,
        "usd_path": "a1.usd", "thumbnail_path": "a1_thumb.png",
        "blend_path": "a1.blend", "stl_path": "a1.stl",
    }])
    open(os.path.join(tmp_path, "a1.usd"), "w").write("x")
    open(os.path.join(tmp_path, "a1_thumb.png"), "w").close()
    open(os.path.join(tmp_path, "a1.blend"), "w").write("x")
    # a1.stl deliberately not created

    errors = validate(str(tmp_path))
    assert any("missing STL file" in e for e in errors), errors


def test_validate_passes_when_blend_and_stl_present(tmp_path):
    _write_manifest(tmp_path, [{
        "asset_id": "a1", "die_type": "d6", "num_sides": 6,
        "usd_path": "a1.usd", "thumbnail_path": "a1_thumb.png",
        "blend_path": "a1.blend", "stl_path": "a1.stl",
    }])
    open(os.path.join(tmp_path, "a1.usd"), "w").write("x")
    open(os.path.join(tmp_path, "a1_thumb.png"), "w").close()
    open(os.path.join(tmp_path, "a1.blend"), "w").write("x")
    open(os.path.join(tmp_path, "a1.stl"), "w").write("x")

    errors = validate(str(tmp_path))
    assert errors == []
```

Every existing test in this file (e.g. `test_validate_passes_for_well_formed_manifest`, `test_validate_passes_for_complete_set`) deliberately keeps NOT setting `blend_path`/`stl_path` at all — do not modify them. Task 2's Step 3 implementation must make the new checks a no-op whenever those keys are absent, exactly like the existing `engraving_warnings` check, specifically so none of those pre-existing tests need to change.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/saps/projects/Dice-Detection && python3 -m pytest tests/test_validate_dice_assets.py -v`
Expected: `test_validate_flags_missing_blend_file` and `test_validate_flags_missing_stl_file` FAIL (no error message contains "missing .blend file"/"missing STL file" yet, since `validate()` doesn't check these fields). `test_validate_passes_when_blend_and_stl_present` should already PASS (nothing currently flags anything for this record) — that's fine, it's a regression guard for Step 3.

- [ ] **Step 3: Implement the checks**

In `scripts/validate_dice_assets.py`, add this block inside the `for record in manifest:` loop, right after the existing `for warning in record.get("engraving_warnings") or []:` block:

```python
        blend_rel_path = record.get("blend_path")
        if blend_rel_path:
            blend_path = os.path.join(outdir, blend_rel_path)
            if not os.path.exists(blend_path):
                errors.append(f"{asset_id}: missing .blend file {blend_path}")
            elif os.path.getsize(blend_path) == 0:
                errors.append(f"{asset_id}: empty .blend file {blend_path}")

        stl_rel_path = record.get("stl_path")
        if stl_rel_path:
            stl_path = os.path.join(outdir, stl_rel_path)
            if not os.path.exists(stl_path):
                errors.append(f"{asset_id}: missing STL file {stl_path}")
            elif os.path.getsize(stl_path) == 0:
                errors.append(f"{asset_id}: empty STL file {stl_path}")
```

The full loop body (for reference, showing where the new block fits) should read:

```python
    for record in manifest:
        asset_id = record["asset_id"]
        usd_path = os.path.join(outdir, record["usd_path"])
        thumb_path = os.path.join(outdir, record["thumbnail_path"])

        if not os.path.exists(usd_path):
            errors.append(f"{asset_id}: missing USD file {usd_path}")
        elif os.path.getsize(usd_path) == 0:
            errors.append(f"{asset_id}: empty USD file {usd_path}")

        if not os.path.exists(thumb_path):
            errors.append(f"{asset_id}: missing thumbnail {thumb_path}")

        die_type = record["die_type"]
        expected_sides = len(numbering.get_values(die_type))
        if record["num_sides"] != expected_sides:
            errors.append(
                f"{asset_id}: num_sides {record['num_sides']} != expected "
                f"{expected_sides} for {die_type}"
            )

        for warning in record.get("engraving_warnings") or []:
            errors.append(f"{asset_id}: {warning}")

        blend_rel_path = record.get("blend_path")
        if blend_rel_path:
            blend_path = os.path.join(outdir, blend_rel_path)
            if not os.path.exists(blend_path):
                errors.append(f"{asset_id}: missing .blend file {blend_path}")
            elif os.path.getsize(blend_path) == 0:
                errors.append(f"{asset_id}: empty .blend file {blend_path}")

        stl_rel_path = record.get("stl_path")
        if stl_rel_path:
            stl_path = os.path.join(outdir, stl_rel_path)
            if not os.path.exists(stl_path):
                errors.append(f"{asset_id}: missing STL file {stl_path}")
            elif os.path.getsize(stl_path) == 0:
                errors.append(f"{asset_id}: empty STL file {stl_path}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/saps/projects/Dice-Detection && python3 -m pytest tests/test_validate_dice_assets.py -v`
Expected: all tests PASS, including the three new ones.

Also run the full pure-Python suite to confirm no regressions:

Run: `cd /home/saps/projects/Dice-Detection && python3 -m pytest tests/ -v --ignore=tests/blender`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/validate_dice_assets.py tests/test_validate_dice_assets.py
git commit -m "feat: check blend_path/stl_path in validate_dice_assets.py"
```

---

### Task 3: Regenerate the existing batch and verify

**Files:** none modified — verification only.

**Interfaces:** none — this task consumes Tasks 1-2's finished code as-is.

- [ ] **Step 1: Regenerate the existing 100-asset batch in place**

The batch at `data/raw/dice_assets/` was generated with `seed=42` (per project history) and is gitignored/regeneratable. Regenerate it with the same seed so it now also contains `.blend`/`.stl` files per asset:

Run: `cd /home/saps/projects/Dice-Detection && blender --background --python scripts/generate_dice_assets.py -- --count 100 --seed 42 --outdir data/raw/dice_assets`

Expected: completes without crashing, printing `Generated: 100, Failed: 0` (or close to it — some non-zero `Failed` count is acceptable if it matches what the previous verified batch already had; the point of this step is producing the new file types, not re-litigating engrave correctness, which was already verified separately).

- [ ] **Step 2: Validate the regenerated batch**

Run: `cd /home/saps/projects/Dice-Detection && python3 scripts/validate_dice_assets.py data/raw/dice_assets`

Expected: exits 0 (no errors). If it exits non-zero, inspect whether the errors are new (a regression from this plan's changes — investigate and fix) or match the same engraving warnings the batch already had before this plan (acceptable, unrelated to this plan's scope).

- [ ] **Step 3: Confirm the new files exist and are openable**

Run: `cd /home/saps/projects/Dice-Detection && ls data/raw/dice_assets/*.blend | wc -l && ls data/raw/dice_assets/*.stl | wc -l`

Expected: both print `100` (one `.blend` and one `.stl` per asset).

Run: `cd /home/saps/projects/Dice-Detection && blender --background data/raw/dice_assets/asset_00000.blend --python-expr "import bpy; print('OBJECTS:', [o.name for o in bpy.data.objects])"`

Expected: prints `OBJECTS:` followed by a list containing exactly one object (the die), confirming the saved `.blend` opens cleanly and contains only the expected die geometry.

No commit for this task — it's a verification pass over regenerated (gitignored) data, not a code change.
