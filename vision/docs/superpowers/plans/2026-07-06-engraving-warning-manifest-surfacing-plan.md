# Engraving Warning Manifest Surfacing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface the engrave pipeline's existing per-cut and end-of-loop failure warnings (currently only `print()`ed to the console during Blender generation) into each asset's `manifest.json` record, and make `validate_dice_assets.py` treat a non-empty warning list as a validation error — so a batch with a silently-skipped or debris-discarded numeral fails automated validation instead of requiring someone to grep Blender's console output or manually spot-check thumbnails.

**Architecture:** No new detection logic and no new dependencies. `src/dice_gen/glyphs.py` already detects every known engrave failure mode (`_boolean_diff_apply`'s EXACT/FLOAT retry-and-give-up path, `_discard_non_body_closed_debris`'s end-of-loop backstop) — it just discards the result after printing it. This plan changes those two functions to *return* the warning message (`None` on success) instead of only printing it, has `apply_engraved_glyphs` collect all of them into a list it returns, has `orchestrator.py` write that list into the manifest record as `engraving_warnings`, and has `validate_dice_assets.py` flag any record where that list is non-empty.

**Tech Stack:** Same as the rest of `src/dice_gen/` — Blender 5.1.2's bundled `bpy`/`bmesh`, no external pip packages. Blender-dependent tests run via `blender --background --python tests/blender/test_<module>.py`. Pure-Python tests (`validate_dice_assets.py`, `orchestrator.py`'s manifest shape via the Blender-harness suite) use the existing conventions in `tests/`.

## Global Constraints

- Do not change any existing detection heuristic or threshold (e.g. the `volume_result < volume_before * 0.5` collapse check, the debris delta comparison). This plan only changes how an *already-detected* warning is communicated — see [[project_dice_asset_pipeline_status]] and the standing rule that new heuristics/thresholds must be validated against a real batch before being trusted. This plan introduces no new heuristic, so that re-validation is not required here.
- Do not add any post-export/USD-level validation step or new Python dependency (e.g. `trimesh`, `open3d`). Verification stays inside the Blender-side generation pipeline, per explicit user direction: validate while the dice are being created/designed in Blender, before export.
- Isaac Sim/Isaac Lab integration is out of scope — not touched by this plan.
- Preserve every existing `print()` call (console visibility during a live `blender --background` batch run is still useful) — warnings are *additionally* returned, not print-only-replaced-by-return.
- Follow existing test conventions exactly: Blender-API-dependent test scripts (`tests/blender/*.py`) end with `run_and_report(run)` from `tests/blender/_harness.py` and are invoked via `blender --background --python tests/blender/test_<module>.py; echo "exit=$?"`. Pure-Python tests use plain `pytest`.

---

## File Structure

- Modify: `src/dice_gen/glyphs.py` — `_boolean_diff_apply`, `_discard_non_body_closed_debris`, `apply_engraved_glyphs` change return contracts.
- Modify: `src/dice_gen/orchestrator.py` — `_generate_from_params` captures the returned warnings into the manifest record.
- Modify: `scripts/validate_dice_assets.py` — `validate()` flags non-empty `engraving_warnings`.
- Test: `tests/blender/test_glyphs.py` — new tests for the return-value contracts.
- Test: `tests/blender/test_orchestrator.py` — extend existing manifest-shape assertions.
- Test: `tests/test_validate_dice_assets.py` — new tests for the warning-flagging check.

---

### Task 1: `glyphs.py` returns engrave warnings instead of only printing them

**Files:**
- Modify: `src/dice_gen/glyphs.py:172-217` (`_discard_non_body_closed_debris`), `src/dice_gen/glyphs.py:220-378` (`_boolean_diff_apply`), `src/dice_gen/glyphs.py:381-430` (`apply_engraved_glyphs`)
- Test: `tests/blender/test_glyphs.py`

**Interfaces:**
- Consumes: existing `_connected_components(bm)`, `_non_body_closed_component_count(bm)` (unchanged).
- Produces:
  - `_boolean_diff_apply(die_obj, cutter_obj) -> str | None` — `None` if the cut succeeded (on EXACT or after FLOAT retry), or a warning string if both solvers produced a collapsed/debris-laden result and the cut was skipped entirely. (Previously returned nothing.)
  - `_discard_non_body_closed_debris(die_obj) -> str | None` — `None` if no un-subtracted debris was found, or a warning string describing what was discarded. (Previously returned nothing.)
  - `apply_engraved_glyphs(...) -> list[str]` — every warning collected during the cut loop and the final backstop, in the order encountered; `[]` if the die engraved cleanly. (Previously returned `None` implicitly.) This is consumed by `orchestrator.py` in Task 2.

- [ ] **Step 1: Write the failing tests**

Add these four tests to `tests/blender/test_glyphs.py`, inserted after `test_engraved_arabic_numerals_d4_does_not_leave_undetected_debris` (i.e. right before `test_decal_glyphs_survive_usd_export_roundtrip_without_black_faces`, currently at line 608):

```python
def test_apply_engraved_glyphs_returns_empty_list_for_clean_die():
    """
    Contract test for the manifest-warning-surfacing feature: a die that
    engraves cleanly (the common case -- see the ~1.8% baseline warning
    rate noted in _boolean_diff_apply's docstring) must return an empty
    list from apply_engraved_glyphs, not None and not a list with entries,
    so orchestrator.py can write `engraving_warnings: []` straight into a
    clean asset's manifest record.
    """
    import bpy
    from dice_gen import geometry, numbering, glyphs

    die_type = "d6"
    obj = geometry.build_die_base_mesh(die_type, size_mm=16.0)
    pairs = geometry.compute_opposite_face_pairs(obj)
    assignment = numbering.assign_values_to_opposite_pairs(die_type, pairs)

    warnings = glyphs.apply_engraved_glyphs(
        obj, die_type, assignment,
        glyph_style="arabic_numerals", glyph_fill="blank",
        font_id="font_sans_bold", size_mm=16.0,
    )

    assert warnings == [], f"expected no warnings for a clean d6 engrave, got {warnings}"

    bpy.data.objects.remove(obj, do_unlink=True)


def test_apply_engraved_glyphs_aggregates_forced_cut_warnings():
    """
    Verifies apply_engraved_glyphs actually collects and returns whatever
    _boolean_diff_apply reports, in order, rather than swallowing it.
    Forces every cut to report a warning (via a monkeypatched
    _boolean_diff_apply, the same spy technique already used by
    test_engraved_glyphs_use_pristine_face_orientations_not_reindexed_mid_loop)
    so the aggregation logic is exercised deterministically, independent of
    whether any real Blender boolean happens to fail on this run.
    """
    import bpy
    from dice_gen import geometry, numbering, glyphs

    die_type = "d6"
    obj = geometry.build_die_base_mesh(die_type, size_mm=16.0)
    pairs = geometry.compute_opposite_face_pairs(obj)
    assignment = numbering.assign_values_to_opposite_pairs(die_type, pairs)
    assert len(assignment) == 6, "d6 should have 6 faces assigned"

    real_boolean_apply_fn = glyphs._boolean_diff_apply
    call_count = [0]

    def fake_boolean_apply(die_obj_arg, cutter_obj):
        call_count[0] += 1
        bpy.data.objects.remove(cutter_obj, do_unlink=True)
        return f"forced test warning #{call_count[0]}"

    glyphs._boolean_diff_apply = fake_boolean_apply
    try:
        warnings = glyphs.apply_engraved_glyphs(
            obj, die_type, assignment,
            glyph_style="arabic_numerals", glyph_fill="blank",
            font_id="font_sans_bold", size_mm=16.0,
        )
    finally:
        glyphs._boolean_diff_apply = real_boolean_apply_fn

    assert len(warnings) == 6, f"expected one forced warning per face cut, got {warnings}"
    assert warnings == [f"forced test warning #{i}" for i in range(1, 7)], warnings

    bpy.data.objects.remove(obj, do_unlink=True)


def test_boolean_diff_apply_returns_warning_when_both_solvers_fail():
    """
    Directly exercises _boolean_diff_apply's "both EXACT and FLOAT produced
    a collapsed or debris-laden result" branch, which used to only print a
    WARNING and now must also return that message. Forces the branch
    deterministically by monkeypatching _non_body_closed_component_count
    (called inside _boolean_diff_apply's local _apply_and_measure) to always
    report more debris than before the cut, regardless of what the real cut
    did -- this makes both the EXACT and FLOAT attempts look "bad" without
    depending on finding a genuinely pathological glyph/size combination.
    """
    import bpy
    from dice_gen import geometry, glyphs

    die_type = "d6"
    obj = geometry.build_die_base_mesh(die_type, size_mm=16.0)

    import bmesh
    bm_pristine = bmesh.new()
    bm_pristine.from_mesh(obj.data)
    volume_pristine = bm_pristine.calc_volume()
    faces_pristine = len(bm_pristine.faces)
    bm_pristine.free()

    bpy.ops.mesh.primitive_uv_sphere_add(radius=16.0 * 0.05)
    cutter = bpy.context.active_object
    cutter.location = (0, 0, 0)

    real_component_count_fn = glyphs._non_body_closed_component_count
    glyphs._non_body_closed_component_count = lambda bm: 999
    try:
        warning = glyphs._boolean_diff_apply(obj, cutter)
    finally:
        glyphs._non_body_closed_component_count = real_component_count_fn

    assert warning is not None, "expected a warning string when both solvers look bad"
    assert obj.name in warning

    bm_after = bmesh.new()
    bm_after.from_mesh(obj.data)
    volume_after = bm_after.calc_volume()
    faces_after = len(bm_after.faces)
    bm_after.free()

    assert faces_after == faces_pristine, (
        "a skipped cut must leave the die's mesh exactly as it was before "
        "the cut was attempted"
    )
    assert abs(volume_after - volume_pristine) < 1e-6

    bpy.data.objects.remove(obj, do_unlink=True)


def test_discard_non_body_closed_debris_returns_warning_for_manual_debris():
    """
    Directly exercises _discard_non_body_closed_debris's "found un-subtracted
    debris" branch by manually constructing a die mesh with an artificial
    disconnected closed shell, rather than depending on a genuine boolean
    failure to produce one. A small closed cube, disjoint from the die
    body, is exactly what real un-subtracted cutter debris looks like to
    this function (see _connected_components' bbox_diag_sq discussion).
    """
    import bpy
    import bmesh
    from dice_gen import geometry, glyphs

    die_type = "d6"
    obj = geometry.build_die_base_mesh(die_type, size_mm=16.0)

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    faces_before = len(bm.faces)
    debris_verts = bmesh.ops.create_cube(bm, size=1.0)["verts"]
    for v in debris_verts:
        v.co += (100, 100, 100)  # place far from the die body, fully disjoint
    bm.to_mesh(obj.data)
    obj.data.update()
    bm.free()

    warning = glyphs._discard_non_body_closed_debris(obj)

    assert warning is not None, "expected a warning string when debris is present"
    assert obj.name in warning

    bm2 = bmesh.new()
    bm2.from_mesh(obj.data)
    faces_after = len(bm2.faces)
    bm2.free()

    assert faces_after == faces_before, (
        "the manually added debris cube should have been discarded, leaving "
        "the die's original face count"
    )

    bpy.data.objects.remove(obj, do_unlink=True)
```

Also add these four function names to the `run()` function at the bottom of `tests/blender/test_glyphs.py` (currently lines 854-865), inserted right after `test_engraved_arabic_numerals_d4_does_not_leave_undetected_debris()`:

```python
    test_apply_engraved_glyphs_returns_empty_list_for_clean_die()
    test_apply_engraved_glyphs_aggregates_forced_cut_warnings()
    test_boolean_diff_apply_returns_warning_when_both_solvers_fail()
    test_discard_non_body_closed_debris_returns_warning_for_manual_debris()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/saps/projects/Dice-Detection && blender --background --python tests/blender/test_glyphs.py; echo "exit=$?"`
Expected: FAIL (non-zero exit). `test_apply_engraved_glyphs_returns_empty_list_for_clean_die` fails because `apply_engraved_glyphs` currently returns `None`, not `[]`. `test_apply_engraved_glyphs_aggregates_forced_cut_warnings` fails the same way. `test_boolean_diff_apply_returns_warning_when_both_solvers_fail` fails because `_boolean_diff_apply` currently returns `None` always. `test_discard_non_body_closed_debris_returns_warning_for_manual_debris` fails because `_discard_non_body_closed_debris` currently returns `None` always.

- [ ] **Step 3: Implement the return-value changes**

In `src/dice_gen/glyphs.py`, replace `_discard_non_body_closed_debris` (lines 172-217) with:

```python
def _discard_non_body_closed_debris(die_obj):
    """
    Final backstop, run ONCE after the entire cut loop in apply_engraved_glyphs
    finishes (not per-cut -- no need to re-scan/delete mid-loop when every cut
    already got its own EXACT->FLOAT retry chance via _boolean_diff_apply).
    Even that per-cut retry is not guaranteed to fully merge every glyph
    cutter into the die on every degenerate input -- four rounds of
    progressively subtler EXACT-solver pathologies have been found
    empirically on this codebase (afb1af5's Alpha-glyph volume collapse,
    cd7b268's total silent no-op, the debris-outweighs-body face-count
    masking, and the closed-component-count/absolute-threshold blind spot --
    see _boolean_diff_apply and _non_body_closed_component_count for the
    full history). Rather than assume this is finally the last one,
    guarantee the SHIPPED asset is always clean: delete any remaining closed
    shell other than the largest-bbox-diagonal one (the real body; any other
    OPEN pieces, e.g. asset_00091-style harmless seam splits, are left
    alone), so no exported asset ever contains stray un-subtracted cutter
    geometry. This can leave a single numeral missing from one face in the
    rare worst case, which is a far smaller defect than shipping floating
    garbage polygons in training data.

    Returns the warning message (also printed, as before) if debris was
    found and discarded, or None if the die was already clean -- callers
    (apply_engraved_glyphs) collect this into the asset's manifest record so
    a batch-level validation pass can flag it without depending on anyone
    reading the console output of the Blender generation run.
    """
    bm = bmesh.new()
    bm.from_mesh(die_obj.data)
    components = _connected_components(bm)

    warning = None
    if components:
        body = max(components, key=lambda c: c["bbox_diag_sq"])
        debris = [c for c in components if c is not body and not c["has_boundary"]]

        if debris:
            debris_face_count = sum(len(c["faces"]) for c in debris)
            for extra in debris:
                bmesh.ops.delete(bm, geom=extra["faces"], context='FACES')
            bm.to_mesh(die_obj.data)
            die_obj.data.update()
            warning = (
                f"{die_obj.name}: {len(debris)} un-subtracted closed debris "
                f"shell(s) ({debris_face_count} faces total) survived every "
                f"per-cut EXACT->FLOAT retry and were discarded at the end "
                f"of the cut loop to keep the shipped asset clean; this "
                f"likely means at least one numeral/pip cut failed to "
                f"engrave on this die."
            )
            print(f"WARNING: {warning}")

    bm.free()
    return warning
```

Then, in `_boolean_diff_apply` (lines 220-378), leave the entire docstring and the body up through the `bad = _apply_and_measure('EXACT')` line unchanged, and replace only the code from `if bad:` (line 356) through the end of the function (line 378) with:

```python
    warning = None

    if bad:
        bm_before.to_mesh(die_obj.data)
        die_obj.data.update()
        bad = _apply_and_measure('FLOAT')

        if bad:
            # Both solvers produced a collapsed or still-debris-laden result
            # -- give up on this specific cut rather than ship a broken
            # mesh. The die ends up exactly as it was before this glyph was
            # ever attempted; the missing numeral is the same class of rare,
            # tracked exception apply_engraved_glyphs' end-of-loop backstop
            # already accepts for the debris case.
            bm_before.to_mesh(die_obj.data)
            die_obj.data.update()
            warning = (
                f"{die_obj.name}: a glyph cut was skipped entirely -- both "
                f"EXACT and FLOAT solvers produced a collapsed or "
                f"debris-laden result for this cutter; this die is missing "
                f"one numeral/pip as a result."
            )
            print(f"WARNING: {warning}")

    bm_before.free()
    bpy.data.objects.remove(cutter_obj, do_unlink=True)
    return warning
```

Add one sentence to `_boolean_diff_apply`'s docstring (append to the end of the existing docstring, right before the closing `"""` at line 328) noting the new return contract:

```
    Returns None if the cut succeeded (on EXACT or after a FLOAT retry), or
    the warning message (also printed, as before) if both solvers failed and
    the cut was skipped -- collected by apply_engraved_glyphs into the
    asset's manifest record.
```

Finally, in `apply_engraved_glyphs` (lines 381-430), add a `warnings = []` list right after the existing `depth`/`glyph_font_size` computation, append to it at both call sites of `_boolean_diff_apply`, append the `_discard_non_body_closed_debris` result, and `return warnings` at the end:

```python
def apply_engraved_glyphs(die_obj, die_type, assignment, glyph_style, glyph_fill, font_id, size_mm):
    depth = size_mm * ENGRAVE_DEPTH_FRACTION
    glyph_font_size = size_mm * 0.18
    warnings = []

    # Phase 1: compute every cut's (value, orientation) against the PRISTINE
    # mesh, entirely before any boolean modifier is applied. Each
    # bpy.ops.object.modifier_apply call below rebuilds die_obj.data's
    # topology (reindexing/reordering polygons), so face_index values from
    # `assignment` (captured once upfront by geometry.compute_opposite_face_pairs)
    # must never be re-resolved against die_obj.data.polygons after a cut.
    planned_cuts = []
    for face_index, value in assignment.items():
        face = die_obj.data.polygons[face_index]
        orient = _face_orientation_matrix(face, die_obj.matrix_world)
        planned_cuts.append((value, orient))

    # Phase 2: build and apply each cutter using only the precomputed
    # orientation matrices — no further indexing into die_obj.data.polygons.
    for value, orient in planned_cuts:
        if glyph_style == "pips":
            for (ox, oy) in PIP_VALUE_LAYOUTS.get(value, [(0, 0)]):
                bpy.ops.mesh.primitive_uv_sphere_add(radius=size_mm * 0.05)
                pip = bpy.context.active_object
                pip.location = orient @ Vector(
                    (ox * size_mm * 0.4, oy * size_mm * 0.4, -depth * 0.5)
                )
                cut_warning = _boolean_diff_apply(die_obj, pip)
                if cut_warning is not None:
                    warnings.append(cut_warning)
        else:
            label = glyph_label(value, glyph_style)
            bpy.ops.object.text_add()
            txt_obj = bpy.context.active_object
            txt_obj.data.body = label
            txt_obj.data.align_x = 'CENTER'
            txt_obj.data.align_y = 'CENTER'
            txt_obj.data.size = glyph_font_size
            txt_obj.data.extrude = depth
            bpy.context.view_layer.objects.active = txt_obj
            bpy.ops.object.convert(target='MESH')
            _weld_cutter_mesh(txt_obj)
            txt_obj.matrix_world = orient @ Matrix.Translation((0, 0, -depth))
            cut_warning = _boolean_diff_apply(die_obj, txt_obj)
            if cut_warning is not None:
                warnings.append(cut_warning)

    # Final backstop, run once after every cut has had its own per-cut retry
    # chance: discard any un-subtracted closed debris shell still left over
    # (see _discard_non_body_closed_debris) so the shipped die is guaranteed
    # free of stray cutter geometry.
    debris_warning = _discard_non_body_closed_debris(die_obj)
    if debris_warning is not None:
        warnings.append(debris_warning)

    if glyph_fill == "painted":
        _assign_fill_material_to_recessed_faces(die_obj)

    return warnings
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/saps/projects/Dice-Detection && blender --background --python tests/blender/test_glyphs.py; echo "exit=$?"`
Expected: `ALL TESTS PASSED` and `exit=0`.

- [ ] **Step 5: Commit**

```bash
git add src/dice_gen/glyphs.py tests/blender/test_glyphs.py
git commit -m "feat: surface engrave-failure warnings as return values from glyphs.py"
```

---

### Task 2: `orchestrator.py` writes `engraving_warnings` into the manifest record

**Files:**
- Modify: `src/dice_gen/orchestrator.py:82-114`
- Test: `tests/blender/test_orchestrator.py`

**Interfaces:**
- Consumes: `glyphs.apply_engraved_glyphs(...) -> list[str]` from Task 1.
- Produces: `manifest_record["engraving_warnings"]` (a `list[str]`, always present, `[]` for decal-method assets and for cleanly-engraved assets) — consumed by Task 3's `validate_dice_assets.py` check.

- [ ] **Step 1: Write the failing test**

In `tests/blender/test_orchestrator.py`, add this assertion inside the existing `for record in manifest:` loop in `test_generate_batch_produces_manifest_and_assets` (currently lines 27-32), so the loop body becomes:

```python
        for record in manifest:
            usd_path = os.path.join(outdir, record["usd_path"])
            thumb_path = os.path.join(outdir, record["thumbnail_path"])
            assert os.path.exists(usd_path)
            assert os.path.exists(thumb_path)
            assert record["die_type"] in ("d4", "d6", "d8", "d10", "d12", "d20")
            assert isinstance(record.get("engraving_warnings"), list), (
                f"{record['asset_id']}: expected an engraving_warnings list "
                f"in every manifest record (empty for decal-method or "
                f"cleanly-engraved dice), got {record.get('engraving_warnings')!r}"
            )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/saps/projects/Dice-Detection && blender --background --python tests/blender/test_orchestrator.py; echo "exit=$?"`
Expected: FAIL (non-zero exit) — `record.get("engraving_warnings")` is `None` today, so `isinstance(None, list)` is `False`.

- [ ] **Step 3: Implement the manifest field**

In `src/dice_gen/orchestrator.py`, replace lines 82-98 (the `if params.glyph_method == "engraved": ... else: ...` block) with:

```python
    if params.glyph_method == "engraved":
        engraving_warnings = glyphs.apply_engraved_glyphs(
            die_obj, params.die_type, assignment, params.glyph_style,
            params.glyph_fill, params.font_or_style_id, params.size_mm,
        )
        mat = materials.build_material(die_obj.name, params.material_category, params.material_params)
        materials.apply_material(die_obj, mat, slot_index=0)
        if params.glyph_fill == "painted":
            fill_mat = materials.build_fill_material(die_obj.name, params.material_params)
            materials.apply_material(die_obj, fill_mat, slot_index=1)
    else:
        engraving_warnings = []
        mat = materials.build_material(die_obj.name, params.material_category, params.material_params)
        materials.apply_material(die_obj, mat, slot_index=0)
        glyphs.apply_decal_glyphs(
            die_obj, params.die_type, assignment, params.glyph_style,
            params.font_or_style_id, params.size_mm, asset_id, outdir,
        )
```

Then add `"engraving_warnings": engraving_warnings,` as the last entry of the `manifest_record` dict literal (currently lines 100-114), right after `"seed": params.seed,`:

```python
    manifest_record = {
        "asset_id": asset_id,
        "die_type": params.die_type,
        "num_sides": len(numbering.get_values(params.die_type)),
        "size_mm": params.size_mm,
        "bevel_fraction": params.bevel_fraction,
        "numbering_scheme": params.numbering_scheme,
        "glyph_style": params.glyph_style,
        "glyph_method": params.glyph_method,
        "glyph_fill": params.glyph_fill,
        "font_or_style_id": params.font_or_style_id,
        "material_category": params.material_category,
        "material_params": params.material_params,
        "seed": params.seed,
        "engraving_warnings": engraving_warnings,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/saps/projects/Dice-Detection && blender --background --python tests/blender/test_orchestrator.py; echo "exit=$?"`
Expected: `ALL TESTS PASSED` and `exit=0`.

Also re-run Task 1's suite to confirm nothing regressed:

Run: `cd /home/saps/projects/Dice-Detection && blender --background --python tests/blender/test_glyphs.py; echo "exit=$?"`
Expected: `ALL TESTS PASSED` and `exit=0`.

- [ ] **Step 5: Commit**

```bash
git add src/dice_gen/orchestrator.py tests/blender/test_orchestrator.py
git commit -m "feat: write engraving_warnings into every manifest record"
```

---

### Task 3: `validate_dice_assets.py` flags non-empty `engraving_warnings`

**Files:**
- Modify: `scripts/validate_dice_assets.py:20-39`
- Test: `tests/test_validate_dice_assets.py`

**Interfaces:**
- Consumes: `record["engraving_warnings"]` (a `list[str]`, from Task 2; use `record.get("engraving_warnings") or []` for backward compatibility with any manifest generated before this plan, so validating an old batch doesn't crash with a `KeyError`).
- Produces: additional entries in `validate()`'s returned `errors` list — no change to `validate()`'s signature or the existing error-string conventions (`f"{asset_id}: ..."`).

- [ ] **Step 1: Write the failing tests**

`tests/test_validate_dice_assets.py` already has `import json`/`import os`/`import sys` at module scope and a `_write_manifest(tmp_path, records)` helper (writes `records` to `tmp_path/manifest.json`) used by every existing test. Add these two tests after `test_validate_reports_wrong_num_sides` (currently ending at line 38), following the exact same `tmp_path`-fixture style as `test_validate_reports_missing_usd_file` (lines 16-24):

```python
def test_validate_flags_non_empty_engraving_warnings(tmp_path):
    _write_manifest(tmp_path, [{
        "asset_id": "a1", "die_type": "d6", "num_sides": 6,
        "usd_path": "a1.usd", "thumbnail_path": "a1_thumb.png",
        "engraving_warnings": [
            "a1_d6_die: a glyph cut was skipped entirely -- both EXACT and "
            "FLOAT solvers produced a collapsed or debris-laden result for "
            "this cutter; this die is missing one numeral/pip as a result."
        ],
    }])
    open(os.path.join(tmp_path, "a1.usd"), "w").write("x")
    open(os.path.join(tmp_path, "a1_thumb.png"), "w").close()

    errors = validate(str(tmp_path))
    assert any("a1" in e and "glyph cut was skipped" in e for e in errors), errors


def test_validate_does_not_flag_empty_engraving_warnings(tmp_path):
    _write_manifest(tmp_path, [{
        "asset_id": "a1", "die_type": "d6", "num_sides": 6,
        "usd_path": "a1.usd", "thumbnail_path": "a1_thumb.png",
        "engraving_warnings": [],
    }])
    open(os.path.join(tmp_path, "a1.usd"), "w").write("x")
    open(os.path.join(tmp_path, "a1_thumb.png"), "w").close()

    errors = validate(str(tmp_path))
    assert errors == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/saps/projects/Dice-Detection && python3 -m pytest tests/test_validate_dice_assets.py -v`
Expected: `test_validate_flags_non_empty_engraving_warnings` FAILS (no error mentions "glyph cut was skipped" since `validate()` doesn't look at `engraving_warnings` yet). `test_validate_does_not_flag_empty_engraving_warnings` should already PASS (nothing currently flags anything for this record) — that's fine, it's a regression guard for Step 3.

- [ ] **Step 3: Implement the check**

In `scripts/validate_dice_assets.py`, inside the `for record in manifest:` loop in `validate()` (lines 20-39), add this block right after the existing `num_sides` check (after line 39, still inside the loop, before the loop ends):

```python
        for warning in record.get("engraving_warnings") or []:
            errors.append(f"{asset_id}: {warning}")
```

The full loop body should read:

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/saps/projects/Dice-Detection && python3 -m pytest tests/test_validate_dice_assets.py -v`
Expected: all tests PASS, including both new ones.

Also run the full pure-Python test suite to confirm no regressions. Note `tests/blender/` must be explicitly excluded — its test scripts are designed to run only via `blender --background --python`, and executing `run_and_report(...)` at import time under plain pytest raises `ModuleNotFoundError: No module named 'bpy'` during collection:

Run: `cd /home/saps/projects/Dice-Detection && python3 -m pytest tests/ -v --ignore=tests/blender`
Expected: all tests PASS (`test_numbering.py`, `test_sampler.py`, `test_validate_dice_assets.py`).

- [ ] **Step 5: Commit**

```bash
git add scripts/validate_dice_assets.py tests/test_validate_dice_assets.py
git commit -m "feat: flag non-empty engraving_warnings in validate_dice_assets.py"
```

---

### Task 4: End-to-end sanity check against a real generated batch

**Files:** none modified — verification only.

**Interfaces:** none — this task consumes Tasks 1-3's finished code as-is.

- [ ] **Step 1: Regenerate a small real batch**

Run: `cd /home/saps/projects/Dice-Detection && blender --background --python scripts/generate_dice_assets.py -- --count 30 --seed 42 --outdir /tmp/dice_gen_warning_check`

(`scripts/generate_dice_assets.py` reads `sys.argv[sys.argv.index("--") + 1:]` — args after `--` are its own argparse args, everything before is Blender's; this is the correct invocation, no need to re-derive it.)

Expected: completes without crashing, printing `Generated: N, Failed: M`.

- [ ] **Step 2: Validate the batch and inspect for warnings**

Run: `cd /home/saps/projects/Dice-Detection && python3 scripts/validate_dice_assets.py /tmp/dice_gen_warning_check`

Expected: exits 0 (no errors) on a clean batch, OR exits 1 with specific `engraving_warnings`-derived error lines if any of the 30 assets genuinely hit a skipped-cut or discarded-debris case — either outcome is an acceptable pass for this task, since the goal is confirming the new field round-trips correctly end-to-end, not that this particular batch is defect-free. If it exits 1, manually inspect the flagged asset's warning text and confirm it's a real, previously-invisible-outside-the-console signal (i.e., the mechanism works), not a bug in the new code.

- [ ] **Step 3: Inspect the raw manifest**

Run: `cd /home/saps/projects/Dice-Detection && python3 -c "import json; m = json.load(open('/tmp/dice_gen_warning_check/manifest.json')); print(sum(1 for r in m if r.get('engraving_warnings')), 'of', len(m), 'records have at least one engraving warning'); [print(r['asset_id'], r['engraving_warnings']) for r in m if r.get('engraving_warnings')]"`

Expected: every record has an `engraving_warnings` key (confirms Task 2 landed correctly across the whole batch, not just the single-asset test case), and the count matches whatever `validate_dice_assets.py` reported as errors in Step 2.

- [ ] **Step 4: Clean up**

Run: `rm -rf /tmp/dice_gen_warning_check`

No commit for this task — it's a verification pass, not a code change.
