# Bevel Recess-Edge-Selection Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the export pipeline's Bevel modifier from rounding the internal edges of engraved-numeral recesses, which currently produces catastrophic degenerate/non-manifold geometry (confirmed on a real 100-asset batch: 42/49 engraved assets affected, with degenerate-face counts up to 57,353 on a single d20), and add manifest-level visibility for whatever smaller residual mesh-quality defects remain.

**Architecture:** `exporter.export_asset` currently selects which edges to bevel via `limit_method='ANGLE'` (35 degrees). This cannot distinguish the die's large structural edges (e.g. a cube's ~90 degree edges) from the many small, steep-angled edges bounding an engraved numeral's recess floor/walls — both exceed the 35 degree threshold, so 8-segment rounding gets applied inside tiny recess geometry too, which is the confirmed root cause of the degenerate output (validated empirically: reproduced on a synthetic cube, isolated to the Bevel step specifically, and fixed by switching selection strategy). The fix marks every edge of the *pristine* polyhedron (before any engraving cut ever runs) with Blender's per-edge `bevel_weight_edge` attribute in `geometry.build_die_base_mesh`, then switches the modifier to `limit_method='WEIGHT'` in `exporter.export_asset`. Boolean DIFFERENCE operations don't rebuild untouched edges away from a cut, so this attribute survives every engraving cut intact (confirmed empirically across 8 sequential cuts). A second, smaller class of defect can still occur (bevel geometry self-intersecting with a nearby recess wall) — this plan adds a post-bevel mesh-quality scan that records a warning in the manifest (mirroring the existing `engraving_warnings` pattern) rather than silently shipping a broken asset.

**Tech Stack:** Blender 5.1 bpy/bmesh, Python, existing `_harness.run_and_report` test runner (`tests/blender/*`).

## Global Constraints

- Match existing code style: no comments explaining *what* code does, only non-obvious *why* (this file's existing docstrings are a deliberate, heavily-precedented exception for documenting empirically-discovered pathologies — new additions should follow that same standard: explain the empirical finding and why the fix works, not restate the code).
- `bevel_weight_edge` is the exact Blender 5.1 attribute name for per-edge bevel weight (confirmed via `bm.edges.layers.float.new('bevel_weight_edge')` / `mesh.attributes`) — do not use the deprecated `edge.bevel_weight` bmesh property, it does not exist in this Blender version.
- Degenerate-face threshold for mesh-quality checks: `face.calc_area() < 1e-9` (matches this session's diagnostic scripts and the scale of real geometry here, where legitimate faces are many orders of magnitude larger).
- Do not change `ENGRAVE_DEPTH_FRACTION`, `bevel_fraction` sampling, or `segments=8` — those are unrelated, already-tuned parameters; only the edge-selection *method* changes.

---

### Task 1: Mark structural edges with bevel weight at base-mesh build time

**Files:**
- Modify: `src/dice_gen/geometry.py:97-125` (`build_die_base_mesh`)
- Test: `tests/blender/test_geometry.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: every mesh returned by `build_die_base_mesh` now carries a `bevel_weight_edge` float-attribute layer with every edge set to `1.0`. `exporter.export_asset` (Task 2) relies on this attribute existing and being `1.0` on all original edges.

- [ ] **Step 1: Write the failing test**

Add to `tests/blender/test_geometry.py`:

```python
def test_build_die_base_mesh_marks_all_edges_with_full_bevel_weight():
    """
    exporter.export_asset's Bevel modifier (see test_exporter.py) selects
    edges to round via limit_method='WEIGHT', not 'ANGLE' -- ANGLE cannot
    tell a die's large structural edges apart from the many small,
    similarly-steep-angled edges bounding an engraved numeral's recess,
    which caused catastrophic degenerate geometry (confirmed on a real
    batch: 42/49 engraved assets affected). This only works if every edge
    of the pristine (pre-engrave) polyhedron is marked with full bevel
    weight before any cut ever runs, since boolean DIFFERENCE cuts don't
    rebuild -- and so don't re-mark -- edges away from the cut region.
    """
    import bmesh
    import bpy
    from dice_gen import geometry

    for die_type, spec in geometry.DIE_SPECS.items():
        obj = geometry.build_die_base_mesh(die_type, size_mm=16.0)

        bm = bmesh.new()
        bm.from_mesh(obj.data)
        layer = bm.edges.layers.float.get('bevel_weight_edge')
        assert layer is not None, f"{die_type}: missing bevel_weight_edge layer"
        weights = [e[layer] for e in bm.edges]
        assert weights == [1.0] * len(bm.edges), (
            f"{die_type}: expected every edge weighted 1.0, got {weights}"
        )
        bm.free()

        bpy.data.objects.remove(obj, do_unlink=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `blender --background --python tests/blender/test_geometry.py 2>&1 | tail -30`
Expected: FAIL — `AssertionError: d4: missing bevel_weight_edge layer` (or similar; the attribute does not exist yet).

- [ ] **Step 3: Write minimal implementation**

In `src/dice_gen/geometry.py`, modify `build_die_base_mesh` (replace the body between the `dissolve_limit` check and the `mesh = bpy.data.meshes.new(...)` line):

```python
def build_die_base_mesh(die_type, size_mm):
    spec = DIE_SPECS[die_type]
    scale = size_mm / 2.0

    bm = bmesh.new()
    bmverts = [bm.verts.new((x * scale, y * scale, z * scale)) for (x, y, z) in spec["base_vertices"]]
    bmesh.ops.convex_hull(bm, input=bmverts)
    bmesh.ops.dissolve_limit(
        bm, angle_limit=math.radians(DISSOLVE_ANGLE_DEG), verts=bm.verts, edges=bm.edges
    )
    bm.faces.ensure_lookup_table()
    bm.normal_update()

    if len(bm.faces) != spec["expected_faces"] or len(bm.verts) != spec["expected_verts"]:
        n_faces, n_verts = len(bm.faces), len(bm.verts)
        bm.free()
        raise GeometryBuildError(
            f"{die_type}: expected {spec['expected_faces']} faces / {spec['expected_verts']} verts, "
            f"got {n_faces} faces / {n_verts} verts"
        )

    # Mark every structural edge of the pristine polyhedron before any
    # engraving cut ever runs -- see exporter.export_asset's Bevel modifier
    # (limit_method='WEIGHT') for why this must happen here rather than
    # right before bevel: boolean DIFFERENCE cuts don't rebuild untouched
    # edges away from the cut, so this weight survives every cut intact,
    # letting the eventual bevel round only the die's real structural
    # edges and never the many similarly-steep-angled edges an engraved
    # numeral's recess introduces.
    bevel_layer = bm.edges.layers.float.new('bevel_weight_edge')
    for e in bm.edges:
        e[bevel_layer] = 1.0

    mesh = bpy.data.meshes.new(f"{die_type}_mesh")
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    obj = bpy.data.objects.new(f"{die_type}_die", mesh)
    bpy.context.collection.objects.link(obj)
    return obj
```

- [ ] **Step 4: Run test to verify it passes**

Run: `blender --background --python tests/blender/test_geometry.py 2>&1 | tail -30`
Expected: PASS for all six die types.

- [ ] **Step 5: Run the full geometry test file to check no regressions**

Run: `blender --background --python tests/blender/test_geometry.py 2>&1 | tail -30`
Expected: all tests in the file PASS (this file's existing `test_all_six_dice_build_with_correct_topology` and others must be unaffected).

- [ ] **Step 6: Commit**

```bash
git add src/dice_gen/geometry.py tests/blender/test_geometry.py
git commit -m "fix: mark structural edges with bevel weight at base-mesh build time"
```

---

### Task 2: Switch the export Bevel modifier from angle-based to weight-based edge selection

**Files:**
- Modify: `src/dice_gen/exporter.py:29-39` (`export_asset`)
- Test: `tests/blender/test_exporter.py`

**Interfaces:**
- Consumes: `die_obj.data`'s `bevel_weight_edge` attribute layer from Task 1 (must be present and `1.0` on structural edges before `export_asset` is called with an engraved die).
- Produces: no change to `export_asset`'s signature or return value.

- [ ] **Step 1: Write the failing test**

Add to `tests/blender/test_exporter.py`:

```python
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
    from dice_gen import geometry, glyphs, materials, exporter

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

    assert faces_post < faces_pre * 2, (
        f"bevel grew face count from {faces_pre} to {faces_post} -- "
        f"this magnitude of growth matches the confirmed angle-based "
        f"runaway-tessellation bug, not normal structural rounding"
    )

    bpy.data.objects.remove(obj, do_unlink=True)
```

Add `import tempfile` to the top of `tests/blender/test_exporter.py` if not already present (it already is, per the existing `test_export_asset_writes_usd_manifest_and_thumbnail`).

- [ ] **Step 2: Run test to verify it fails**

Run: `blender --background --python tests/blender/test_exporter.py 2>&1 | tail -40`
Expected: FAIL — `faces_post` is many times larger than `faces_pre * 2` (the modifier still uses `limit_method='ANGLE'` at this point).

- [ ] **Step 3: Write minimal implementation**

In `src/dice_gen/exporter.py`, replace the bevel setup in `export_asset`:

```python
    mod = die_obj.modifiers.new(name="Bevel", type='BEVEL')
    mod.width = size_mm * bevel_fraction
    mod.segments = 8
    mod.limit_method = 'WEIGHT'
    bpy.context.view_layer.objects.active = die_obj
    bpy.ops.object.modifier_apply(modifier=mod.name)
```

(This removes the `mod.limit_method = 'ANGLE'` and `mod.angle_limit = math.radians(35)` lines. `import math` in this file becomes unused by this change alone but stays required by `_render_thumbnail`'s `math.radians` calls further down — do not remove the import.)

Update the module docstring's bevel-behavior paragraph (currently describing `limit_method='ANGLE'`) to match:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `blender --background --python tests/blender/test_exporter.py 2>&1 | tail -40`
Expected: PASS.

- [ ] **Step 5: Run the full exporter test file to check no regressions**

Run: `blender --background --python tests/blender/test_exporter.py 2>&1 | tail -60`
Expected: every test in the file PASSes, including the pre-existing `test_export_asset_blend_image_textures_resolve_after_fresh_reload` (must stay last in the file's `run()` list, per its existing placement).

- [ ] **Step 6: Commit**

```bash
git add src/dice_gen/exporter.py tests/blender/test_exporter.py
git commit -m "fix: bevel structural edges by weight instead of angle to stop rounding engraved recesses"
```

---

### Task 3: Detect and record residual mesh-quality defects in the manifest

**Files:**
- Modify: `src/dice_gen/exporter.py` (add `_mesh_quality_warning`, call it from `export_asset`)
- Modify: `scripts/validate_dice_assets.py:41-42` area (surface the new warnings the same way `engraving_warnings` already are)
- Test: `tests/blender/test_exporter.py`

**Interfaces:**
- Consumes: `die_obj.data` after the Bevel modifier has been applied.
- Produces: `export_asset` now always sets `manifest_record["mesh_quality_warnings"]` to a list (empty if the mesh is clean). `scripts/validate_dice_assets.py` reports each entry as a validation error, matching how it already reports `engraving_warnings`.

Task 2's weight-based bevel fix eliminates the *catastrophic* degenerate-geometry case, but a smaller, separate class of defect can still occur where bevel geometry sits very close to a recess wall (confirmed empirically: even with the Task 1+2 fix applied, a synthetic d20/cjk_numerals case still showed 1,356 non-manifold edges post-bevel, and a synthetic d4/arabic_numerals case showed 1,251 — both far smaller than the pre-fix tens-of-thousands, but not zero). Rather than attempt to eliminate this rarer residual case in this plan (its root cause — bevel/recess geometric proximity — is a distinct, harder problem from the one just fixed), give it the same visibility this codebase already gives every other known imperfection class (`engraving_warnings`): record it in the manifest so `validate_dice_assets.py` surfaces it, instead of silently shipping a possibly-defective mesh.

- [ ] **Step 1: Write the failing test**

Add to `tests/blender/test_exporter.py`:

```python
def test_mesh_quality_warning_flags_a_degenerate_mesh():
    import bmesh
    import bpy
    from dice_gen import geometry, exporter

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
    from dice_gen import geometry, exporter

    obj = geometry.build_die_base_mesh("d6", size_mm=16.0)
    warning = exporter._mesh_quality_warning(obj)
    assert warning is None

    bpy.data.objects.remove(obj, do_unlink=True)


def test_export_asset_sets_mesh_quality_warnings_key():
    import bpy
    from dice_gen import geometry, materials, exporter

    obj = geometry.build_die_base_mesh("d6", size_mm=16.0)
    mat = materials.build_material("d6", "opaque", {"hue": 0.3, "saturation": 0.7, "value": 0.6, "roughness": 0.4})
    materials.apply_material(obj, mat)

    with tempfile.TemporaryDirectory() as outdir:
        record = {"asset_id": "test_d6_quality", "die_type": "d6"}
        exporter.export_asset(obj, record, outdir, bevel_fraction=0.04, size_mm=16.0)
        assert record["mesh_quality_warnings"] == []

    bpy.data.objects.remove(obj, do_unlink=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `blender --background --python tests/blender/test_exporter.py 2>&1 | tail -40`
Expected: FAIL — `AttributeError`/`module 'dice_gen.exporter' has no attribute '_mesh_quality_warning'`, and the manifest-key test fails with a `KeyError`.

- [ ] **Step 3: Write minimal implementation**

In `src/dice_gen/exporter.py`, add this function (near `_save_blend_copy`, before it):

```python
def _mesh_quality_warning(die_obj):
    """
    Scans the final (post-bevel) mesh for non-manifold edges and
    zero-area faces -- a smaller, separate class of defect from the
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
    bm.faces.ensure_lookup_table()
    non_manifold = sum(1 for e in bm.edges if not e.is_manifold)
    zero_area = sum(1 for f in bm.faces if f.calc_area() < 1e-9)
    bm.free()

    if non_manifold == 0 and zero_area == 0:
        return None

    return (
        f"{die_obj.name}: {non_manifold} non-manifold edge(s) and "
        f"{zero_area} zero-area/degenerate face(s) found in the final "
        f"exported mesh."
    )
```

Then in `export_asset`, after the Bevel `modifier_apply` call and before `_save_blend_copy`:

```python
    quality_warning = _mesh_quality_warning(die_obj)
    manifest_record["mesh_quality_warnings"] = [quality_warning] if quality_warning else []
    if quality_warning:
        print(f"WARNING: {quality_warning}")
```

`import bmesh` needs to be added to the top of `src/dice_gen/exporter.py` alongside the existing `import bpy`.

- [ ] **Step 4: Run test to verify it passes**

Run: `blender --background --python tests/blender/test_exporter.py 2>&1 | tail -40`
Expected: PASS.

- [ ] **Step 5: Wire the new warnings into validate_dice_assets.py**

In `scripts/validate_dice_assets.py`, immediately after the existing block:

```python
        for warning in record.get("engraving_warnings") or []:
            errors.append(f"{asset_id}: {warning}")
```

add:

```python
        for warning in record.get("mesh_quality_warnings") or []:
            errors.append(f"{asset_id}: {warning}")
```

- [ ] **Step 6: Run the full exporter test file to check no regressions**

Run: `blender --background --python tests/blender/test_exporter.py 2>&1 | tail -60`
Expected: every test in the file PASSes, including `test_export_asset_blend_image_textures_resolve_after_fresh_reload` staying last.

- [ ] **Step 7: Commit**

```bash
git add src/dice_gen/exporter.py scripts/validate_dice_assets.py tests/blender/test_exporter.py
git commit -m "feat: surface residual mesh-quality defects in the manifest"
```

---

### Task 4: Regenerate the batch and quantify the fix

**Files:** none changed — this task runs the existing pipeline and existing validator, no new code.

**Interfaces:** none.

- [ ] **Step 1: Regenerate the full 100-asset batch in place**

Run (from repo root, matching this session's established regeneration command — check `scripts/` or prior batch-generation invocation for the exact CLI entry point used earlier in this session, e.g. a `generate_batch.py`-style script with `count=100, seed=7`):

```bash
find . -maxdepth 2 -iname "*generate*batch*" -o -iname "*generate*dice*"
```

Then run that script with the same `count=100, seed=7, outdir=data/raw/dice_assets` parameters used for every prior regeneration this session, overwriting the existing batch in place.

- [ ] **Step 2: Run the existing manifest validator**

Run: `python3 scripts/validate_dice_assets.py data/raw/dice_assets`
Expected: 0 errors (or a dramatically smaller, documented set of `mesh_quality_warnings`-only entries versus the pre-fix 42/49 engraved-asset defect rate).

- [ ] **Step 3: Re-run the fresh-reload STL/blend mesh-quality check from this session's investigation**

Reuse the same bmesh-based check pattern from Task 3's `_mesh_quality_warning` (non-manifold edges + zero-area faces) across every asset's freshly-reloaded `.blend`, and report the aggregate before/after comparison versus the pre-fix baseline (75/100 STL files flagged, with degenerate-face counts up to 51,353 and non-manifold-edge counts up to 22,780 per asset).

- [ ] **Step 4: Report the quantified before/after result**

No commit for this task (no code changes) — report the final defect-count comparison.
