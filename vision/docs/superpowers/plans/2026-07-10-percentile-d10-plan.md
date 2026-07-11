# Percentile d10 (`d10_pct`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a 7th die type, `d10_pct` (the percentile/tens die), to the dice-generation pipeline so `generate_set_batch` produces true 7-piece matched D&D sets, following the design in `docs/superpowers/specs/2026-07-10-percentile-d10-design.md`.

**Architecture:** `d10_pct` reuses the existing d10 mesh (identical pentagonal trapezohedron geometry) and reuses d10's own hemisphere-parity-aware value assignment algorithm (scaled ×10 afterward, since d10_pct's own values are all even and can't drive that algorithm's odd/even split directly). Rendering is arabic-numerals-only, zero-padded ("00" not "0"), even inside a matched set that samples a different shared style for its other 6 dice — no real-world percentile-die convention exists for roman/greek/CJK digits, so none is invented.

**Tech Stack:** Same as the rest of `src/dice_gen/` — Blender 5.1.2's bundled `bpy`/`bmesh`, no external pip packages. Pure-Python modules (`numbering.py`, `sampler.py`, `validate_dice_assets.py`) are tested with plain `pytest`; Blender-API-dependent modules (`geometry.py`, `glyphs.py`, `orchestrator.py`) are tested via `blender --background --python tests/blender/test_<module>.py`, which ends with `run_and_report(run)` from `tests/blender/_harness.py` (exit code 1 + traceback on failure, "ALL TESTS PASSED" + exit 0 on success — plain asserts do NOT fail the process in Blender's `--background` mode, so every Blender test script must use this harness).

## Global Constraints

- Face values for `d10_pct`: exactly `{0, 10, 20, 30, 40, 50, 60, 70, 80, 90}` — verified real-world convention, hard invariant.
- `opposite_sum: 90` is enforced as a hard invariant (`raise ValueError` on violation, identical to every other die type) — not a soft/manifest-flagged property. It's our own generation-time design choice, mathematically guaranteed satisfiable (same combinatorics as d10 scaled ×10), not a claim about all physical dice.
- `d10_pct` glyph rendering is **arabic_numerals only, always** — even inside a matched set whose other 6 dice share a different sampled `glyph_style`. Never invent roman/greek/CJK percentile-numeral conventions.
- Zero face renders as `"00"`, not `"0"`.
- No new manifest fields, no new soft-validation machinery — every existing mechanism (`verify_opposite_sum`, `engraving_warnings`, `mesh_quality_warnings`) applies to `d10_pct` unchanged.

---

### Task 1: Numbering — add `d10_pct` scheme with scaled hemisphere-aware assignment

**Files:**
- Modify: `src/dice_gen/numbering.py`
- Test: `tests/test_numbering.py`

**Interfaces:**
- Produces: `numbering.NUMBERING_SCHEMES["d10_pct"] = {"values": [0,10,20,...,90], "opposite_sum": 90}`. `numbering.assign_values_to_opposite_pairs("d10_pct", face_pairs, hemisphere_of_face=None)` returns `{face_index: value}` where `value` is d10's own assignment for that pairing, scaled ×10. `numbering.verify_opposite_sum("d10_pct", face_pairs, assignment)` and `numbering.get_values("d10_pct")` work unchanged (generic, reading the new scheme entry).
- Consumes: nothing from other tasks.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_numbering.py`:

```python
def test_get_values_d10_pct_has_10_unique_values_multiples_of_ten():
    values = numbering.get_values("d10_pct")
    assert len(values) == 10
    assert set(values) == {0, 10, 20, 30, 40, 50, 60, 70, 80, 90}


def test_d10_pct_opposite_faces_sum_to_90():
    face_pairs = [(i, i + 5) for i in range(5)]
    assignment = numbering.assign_values_to_opposite_pairs("d10_pct", face_pairs)
    assert numbering.verify_opposite_sum("d10_pct", face_pairs, assignment)
    assert set(assignment.values()) == {0, 10, 20, 30, 40, 50, 60, 70, 80, 90}


def test_d10_pct_assignment_is_d10_assignment_scaled_by_ten():
    face_pairs = [(i, i + 5) for i in range(5)]
    d10_assignment = numbering.assign_values_to_opposite_pairs("d10", face_pairs)
    pct_assignment = numbering.assign_values_to_opposite_pairs("d10_pct", face_pairs)
    assert pct_assignment == {face: value * 10 for face, value in d10_assignment.items()}


def test_d10_pct_assignment_respects_hemisphere_parity_split():
    face_pairs = [(i, i + 5) for i in range(5)]
    hemisphere_of_face = {
        0: "top", 1: "top", 2: "bottom", 3: "bottom", 4: "top",
        5: "bottom", 6: "bottom", 7: "top", 8: "bottom", 9: "top",
    }
    d10_assignment = numbering.assign_values_to_opposite_pairs(
        "d10", face_pairs, hemisphere_of_face=hemisphere_of_face,
    )
    pct_assignment = numbering.assign_values_to_opposite_pairs(
        "d10_pct", face_pairs, hemisphere_of_face=hemisphere_of_face,
    )
    assert pct_assignment == {face: value * 10 for face, value in d10_assignment.items()}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/saps/projects/Dice-Detection && python3 -m pytest tests/test_numbering.py -v`
Expected: the 4 new tests FAIL with `KeyError: 'd10_pct'` (not yet in `NUMBERING_SCHEMES`).

- [ ] **Step 3: Implement**

In `src/dice_gen/numbering.py`, update the module docstring and `NUMBERING_SCHEMES`:

```python
"""
Standard real-world face-numbering conventions for each die type.

d4 (tetrahedron) has no face-to-face antipodal relationship (its faces are
opposite a *vertex*, not another face), so it has no opposite_sum rule --
values are just assigned once each. All other die types are centrally
symmetric and follow their standard convention:
  d6:      opposite faces sum to 7
  d8:      opposite faces sum to 9
  d10:     opposite faces sum to 9 (values 0-9, pairing k with 9-k)
  d10_pct: opposite faces sum to 90 (percentile/tens die, values
           0,10,...,90 -- same physical mold as d10, different digits
           printed on each face; see assign_values_to_opposite_pairs'
           d10_pct special case for why this scales d10's own assignment
           rather than running through the generic scheme below)
  d12:     opposite faces sum to 13
  d20:     opposite faces sum to 21
"""

NUMBERING_SCHEMES = {
    "d4": {"values": [1, 2, 3, 4], "opposite_sum": None},
    "d6": {"values": [1, 2, 3, 4, 5, 6], "opposite_sum": 7},
    "d8": {"values": [1, 2, 3, 4, 5, 6, 7, 8], "opposite_sum": 9},
    "d10": {"values": list(range(0, 10)), "opposite_sum": 9},
    "d10_pct": {"values": [v * 10 for v in range(0, 10)], "opposite_sum": 90},
    "d12": {"values": list(range(1, 13)), "opposite_sum": 13},
    "d20": {"values": list(range(1, 21)), "opposite_sum": 21},
}
```

Add a special case at the top of `assign_values_to_opposite_pairs` (before the `scheme = NUMBERING_SCHEMES[die_type]` line):

```python
def assign_values_to_opposite_pairs(die_type, face_pairs, hemisphere_of_face=None):
    """
    ...(existing docstring unchanged)...
    """
    if die_type == "d10_pct":
        # Same physical mold as d10, different digits printed on each
        # face. d10_pct's own values (0,10,...,90) are all even, so
        # running them directly through the generic even/odd hemisphere
        # split below would find no parity variance to split on. Instead,
        # reuse d10's own assignment (which already solves the real
        # hemisphere-parity problem on the underlying 0-9 digit) and
        # scale every value x10 for display.
        base_assignment = assign_values_to_opposite_pairs(
            "d10", face_pairs, hemisphere_of_face=hemisphere_of_face,
        )
        return {face: value * 10 for face, value in base_assignment.items()}

    scheme = NUMBERING_SCHEMES[die_type]
    values = scheme["values"]
    opposite_sum = scheme["opposite_sum"]
    # ...(rest of function unchanged)...
```

`verify_opposite_sum` and `get_values` need no changes — both are generic over `NUMBERING_SCHEMES`, and `verify_opposite_sum`'s arithmetic check (`assignment[a] + assignment[b] == opposite_sum`) works correctly on the already-scaled values.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/saps/projects/Dice-Detection && python3 -m pytest tests/test_numbering.py -v`
Expected: all tests PASS (including the pre-existing ones — confirms no regression to d4/d6/d8/d10/d12/d20).

- [ ] **Step 5: Commit**

```bash
git add src/dice_gen/numbering.py tests/test_numbering.py
git commit -m "feat: add d10_pct numbering scheme (percentile die, scaled d10 assignment)"
```

---

### Task 2: Geometry — reuse d10's mesh and pole detection for `d10_pct`

**Files:**
- Modify: `src/dice_gen/geometry.py`
- Test: `tests/blender/test_geometry.py`

**Interfaces:**
- Consumes: nothing from Task 1 directly (geometry is independent of numbering), but this task's new hemisphere test consumes `numbering.assign_values_to_opposite_pairs("d10_pct", ...)` from Task 1.
- Produces: `geometry.DIE_SPECS["d10_pct"]` (same shape as `"d10"`'s entry). `geometry.compute_face_poles(obj, "d10_pct")` returns a pole mapping (not `None`), identical in structure to `"d10"`'s.

- [ ] **Step 1: Write the failing tests**

In `tests/blender/test_geometry.py`, rename `test_all_six_dice_build_with_correct_topology` to `test_all_seven_dice_build_with_correct_topology` (function name only — the body already loops `geometry.DIE_SPECS.items()` generically, so it needs no other change) and update its call in `run()` to match.

Change the loop in `test_compute_face_poles_maps_every_d8_and_d10_face_to_exactly_one_pole`:

```python
    for die_type in ("d8", "d10", "d10_pct"):
```

Change the loop in `test_assign_values_to_opposite_pairs_splits_d8_and_d10_hemispheres_by_parity` — **leave this one as `("d8", "d10")` unchanged**. `d10_pct`'s values are all even, so the raw `assignment[f] % 2` parity check this test uses would trivially fail for it (both hemispheres would show parity `0`) — that's expected, not a bug, and adding `d10_pct` to this loop would break the test for the wrong reason. Instead, add a new, separate test right after it:

```python
def test_assign_values_to_opposite_pairs_splits_d10_pct_hemispheres_by_tens_digit_parity():
    """
    d10_pct's own values (0,10,...,90) are all even, so the raw-value
    parity check used for d8/d10 above doesn't apply directly -- see
    numbering.py's d10_pct special case, which reuses d10's parity-aware
    assignment on the underlying 0-9 digit and scales by 10 afterward.
    This checks the real invariant that scaling is supposed to preserve:
    the underlying (pre-scale) digit's parity is still split cleanly by
    hemisphere, exactly as a real d10's is.
    """
    import bpy
    from dice_gen import geometry, numbering

    obj = geometry.build_die_base_mesh("d10_pct", size_mm=18.0)
    poles = geometry.compute_face_poles(obj, "d10_pct")
    top_pole = max((p.z, tuple(p)) for p in poles.values())[1]

    hemisphere_of_face = {
        face_idx: ("top" if tuple(pole) == top_pole else "bottom")
        for face_idx, pole in poles.items()
    }

    face_pairs = geometry.compute_opposite_face_pairs(obj)
    assignment = numbering.assign_values_to_opposite_pairs(
        "d10_pct", face_pairs, hemisphere_of_face=hemisphere_of_face,
    )
    assert numbering.verify_opposite_sum("d10_pct", face_pairs, assignment)

    top_digit_parities = {(assignment[f] // 10) % 2 for f, h in hemisphere_of_face.items() if h == "top"}
    bottom_digit_parities = {(assignment[f] // 10) % 2 for f, h in hemisphere_of_face.items() if h == "bottom"}
    assert len(top_digit_parities) == 1, f"d10_pct: top hemisphere has mixed tens-digit parities {top_digit_parities}"
    assert len(bottom_digit_parities) == 1, f"d10_pct: bottom hemisphere has mixed tens-digit parities {bottom_digit_parities}"
    assert top_digit_parities != bottom_digit_parities, "d10_pct: top and bottom ended up with the same tens-digit parity"

    bpy.data.objects.remove(obj, do_unlink=True)
```

Leave `test_compute_face_poles_returns_none_for_non_bipyramid_dice` (`("d4", "d6", "d12", "d20")`) and `test_assign_values_to_opposite_pairs_without_hemisphere_arg_is_unchanged` (`("d4", "d6", "d12", "d20")`) unchanged — `d10_pct` is a bipyramid (correctly excluded from the "returns None" test) and is new special-cased behavior (correctly excluded from the "unchanged legacy behavior" test).

Add the new test's call to `run()`, right after `test_assign_values_to_opposite_pairs_splits_d8_and_d10_hemispheres_by_parity()`:

```python
    test_assign_values_to_opposite_pairs_splits_d10_pct_hemispheres_by_tens_digit_parity()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/saps/projects/Dice-Detection && blender --background --python tests/blender/test_geometry.py 2>&1 | tail -40`
Expected: FAIL (traceback, exit code 1) — `KeyError: 'd10_pct'` from `geometry.DIE_SPECS['d10_pct']` inside `build_die_base_mesh`.

- [ ] **Step 3: Implement**

In `src/dice_gen/geometry.py`, add a `"d10_pct"` entry to `DIE_SPECS`, right after `"d10"`:

```python
    "d10_pct": {
        "num_sides": 10,
        "base_vertices": _d10_base_vertices(),
        "expected_faces": 10,
        "expected_verts": 12,
        "expected_edges": 20,
    },
```

Update `compute_face_poles`'s docstring and guard clause:

```python
def compute_face_poles(obj, die_type):
    """
    d8, d10, and d10_pct are all built (see DIE_SPECS / _d10_base_vertices)
    as bipyramids: exactly two pole vertices at the extremal local-Z
    positions, plus a ring of equatorial vertices. d10_pct shares d10's
    exact mesh (same base_vertices) -- only the face labels differ, so it
    has the identical pole structure. Every face touches exactly one pole
    (confirmed empirically this session via direct vertex-index inspection
    on both die types). Real dice orient each face's numeral relative to
    its OWN pole, not one global up-vector -- see glyphs.py's
    _tangent_bitangent for the orientation fix this enables, and
    numbering.py's assign_values_to_opposite_pairs for the matching
    hemisphere-consistent value assignment.

    Returns None for die types without this two-pole structure (d4, d6,
    d12, d20) -- those keep their existing single-global-up-vector
    orientation convention unchanged.
    """
    if die_type not in ("d8", "d10", "d10_pct"):
        return None
    # ...(rest of function unchanged)...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/saps/projects/Dice-Detection && blender --background --python tests/blender/test_geometry.py 2>&1 | tail -40`
Expected: `ALL TESTS PASSED`, exit code 0.

- [ ] **Step 5: Commit**

```bash
git add src/dice_gen/geometry.py tests/blender/test_geometry.py
git commit -m "feat: add d10_pct geometry (reuses d10 mesh and pole detection)"
```

---

### Task 3: Glyphs — arabic-only, zero-padded rendering for `d10_pct`

**Files:**
- Modify: `src/dice_gen/glyphs.py`
- Test: `tests/blender/test_glyphs.py`

**Interfaces:**
- Consumes: `geometry.build_die_base_mesh("d10_pct", ...)`, `geometry.compute_opposite_face_pairs`, `numbering.assign_values_to_opposite_pairs("d10_pct", ...)` from Tasks 1-2.
- Produces: `glyphs.glyph_label(value, glyph_style, die_type=None)` — new optional third parameter; when `die_type == "d10_pct"` and `glyph_style == "arabic_numerals"`, returns a zero-padded 2-digit string (`f"{value:02d}"`); all other combinations are unchanged from before. `apply_engraved_glyphs` and `apply_decal_glyphs` correctly render `d10_pct` labels (both already receive `die_type` as a parameter).

- [ ] **Step 1: Write the failing tests**

In `tests/blender/test_glyphs.py`, add after `test_glyph_label_formats`:

```python
def test_glyph_label_formats_d10_pct_as_zero_padded_two_digit():
    from dice_gen import glyphs

    assert glyphs.glyph_label(0, "arabic_numerals", die_type="d10_pct") == "00"
    assert glyphs.glyph_label(10, "arabic_numerals", die_type="d10_pct") == "10"
    assert glyphs.glyph_label(90, "arabic_numerals", die_type="d10_pct") == "90"


def test_glyph_label_arabic_unchanged_for_non_percentile_die_types():
    from dice_gen import glyphs

    assert glyphs.glyph_label(0, "arabic_numerals", die_type="d10") == "0"
    assert glyphs.glyph_label(6, "arabic_numerals") == "6"
    assert glyphs.glyph_label(6, "arabic_numerals", die_type="d6") == "6"


def test_engraved_glyphs_reduce_solid_volume_for_d10_pct():
    """
    Mirrors test_engraved_glyphs_reduce_solid_volume (d6) but for d10_pct
    specifically -- exercises all 10 faces with real 2-character
    zero-padded labels ("00".."90") through the full engrave/boolean-cut
    path, catching any font-size or cutter regression the 2-digit label
    might introduce that a pure glyph_label unit test can't see.
    """
    import bpy
    from dice_gen import geometry, numbering, glyphs

    die_type = "d10_pct"
    obj = geometry.build_die_base_mesh(die_type, size_mm=18.0)
    pairs = geometry.compute_opposite_face_pairs(obj)
    poles = geometry.compute_face_poles(obj, die_type)
    top_pole_z = max(p.z for p in poles.values())
    hemisphere_of_face = {
        face_idx: ("top" if pole.z == top_pole_z else "bottom")
        for face_idx, pole in poles.items()
    }
    assignment = numbering.assign_values_to_opposite_pairs(
        die_type, pairs, hemisphere_of_face=hemisphere_of_face,
    )
    assert numbering.verify_opposite_sum(die_type, pairs, assignment)

    import bmesh
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    volume_before = bm.calc_volume()
    bm.free()

    warnings = glyphs.apply_engraved_glyphs(
        obj, die_type, assignment,
        glyph_style="arabic_numerals", glyph_fill="painted",
        font_id="font_sans_bold", size_mm=18.0,
    )

    bm2 = bmesh.new()
    bm2.from_mesh(obj.data)
    volume_after = bm2.calc_volume()
    bm2.free()

    assert volume_after < volume_before, "engraving should remove material"
    assert len(obj.data.materials) >= 2, "painted fill should add a second material slot"
    assert warnings == [], f"expected no engraving warnings on a clean d10_pct cut, got {warnings}"

    bpy.data.objects.remove(obj, do_unlink=True)
```

Add all three new test calls to `run()`, after `test_glyph_label_formats()`:

```python
    test_glyph_label_formats_d10_pct_as_zero_padded_two_digit()
    test_glyph_label_arabic_unchanged_for_non_percentile_die_types()
    test_engraved_glyphs_reduce_solid_volume_for_d10_pct()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/saps/projects/Dice-Detection && blender --background --python tests/blender/test_glyphs.py 2>&1 | tail -40`
Expected: FAIL — `glyphs.glyph_label(0, "arabic_numerals", die_type="d10_pct")` raises `TypeError: glyph_label() got an unexpected keyword argument 'die_type'` (parameter doesn't exist yet), and/or `KeyError`/`GeometryBuildError` from `d10_pct` not existing in `DIE_SPECS` if run before Task 2 — confirm Tasks 1-2 are committed first.

- [ ] **Step 3: Implement**

In `src/dice_gen/glyphs.py`, update `glyph_label`:

```python
def glyph_label(value, glyph_style, die_type=None):
    if glyph_style == "arabic_numerals":
        if die_type == "d10_pct":
            return f"{value:02d}"
        return str(value)
    if glyph_style == "roman_numerals":
        return ROMAN_NUMERALS.get(value, str(value))
    if glyph_style == "greek_numerals":
        return GREEK_NUMERALS.get(value, str(value))
    if glyph_style == "cjk_numerals":
        return CJK_NUMERALS.get(value, str(value))
    raise ValueError(f"glyph_label not applicable to style {glyph_style!r}")
```

Update all four call sites to pass `die_type` (already in scope as a parameter of the enclosing `apply_engraved_glyphs`/`apply_decal_glyphs` function at every site):

- `glyphs.py` inside `apply_engraved_glyphs`, Phase 1 loop (`label = glyph_label(value, glyph_style)` computing `font_size`):
  ```python
  label = glyph_label(value, glyph_style, die_type)
  ```
- `glyphs.py` inside `apply_engraved_glyphs`, Phase 2 non-pips branch (`label = glyph_label(value, glyph_style)` before `bpy.ops.object.text_add()`):
  ```python
  label = glyph_label(value, glyph_style, die_type)
  ```
- `glyphs.py` inside `apply_decal_glyphs`, d4 vertex-numerals branch (`label = glyph_label(value, glyph_style)` before the 3-corner loop):
  ```python
  label = glyph_label(value, glyph_style, die_type)
  ```
- `glyphs.py` inside `apply_decal_glyphs`, single-centered-label branch (`label = glyph_label(value, glyph_style)` in the `else:`):
  ```python
  label = glyph_label(value, glyph_style, die_type)
  ```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/saps/projects/Dice-Detection && blender --background --python tests/blender/test_glyphs.py 2>&1 | tail -40`
Expected: `ALL TESTS PASSED`, exit code 0.

- [ ] **Step 5: Commit**

```bash
git add src/dice_gen/glyphs.py tests/blender/test_glyphs.py
git commit -m "feat: render d10_pct labels as zero-padded arabic numerals"
```

---

### Task 4: Sampler wiring + full-pipeline integration proof

**Files:**
- Modify: `src/dice_gen/sampler.py`
- Modify: `tests/test_sampler.py`
- Modify: `tests/blender/test_orchestrator.py`

**Interfaces:**
- Consumes: `geometry.DIE_SPECS["d10_pct"]`, `numbering.NUMBERING_SCHEMES["d10_pct"]`, `glyphs.glyph_label(..., die_type=...)` from Tasks 1-3 (transitively, via `orchestrator._generate_from_params`, unchanged in this task).
- Produces: `sampler.DIE_TYPES` includes `"d10_pct"`. `sampler.SIZE_RANGES_MM["d10_pct"] = (14.0, 20.0)` (same class as units d10). `sample_variant`/`sample_set` force `glyph_style = "arabic_numerals"` whenever `die_type == "d10_pct"`, overriding the otherwise-shared/sampled style. `orchestrator.generate_set_batch` (unchanged code, but now produces 7 assets per set since it loops `sampler.DIE_TYPES`).

- [ ] **Step 1: Write the failing tests**

In `tests/test_sampler.py`, update `test_sample_set_has_exactly_the_expected_die_type_keys`:

```python
def test_sample_set_has_exactly_the_expected_die_type_keys():
    for seed in range(10):
        variants = sampler.sample_set(seed)
        assert set(variants.keys()) == set(sampler.DIE_TYPES)
        assert set(variants.keys()) == {"d4", "d6", "d8", "d10", "d10_pct", "d12", "d20"}
```

Add two new tests at the end of the file:

```python
def test_sample_variant_d10_pct_glyph_style_is_always_arabic_numerals():
    seen_d10_pct = False
    for seed in range(300):
        v = sampler.sample_variant(seed)
        if v.die_type == "d10_pct":
            seen_d10_pct = True
            assert v.glyph_style == "arabic_numerals"
    assert seen_d10_pct, "expected at least one d10_pct sample across 300 seeds"


def test_sample_set_d10_pct_glyph_style_is_always_arabic_numerals():
    for seed in range(50):
        variants = sampler.sample_set(seed)
        assert variants["d10_pct"].glyph_style == "arabic_numerals"
```

In `tests/blender/test_orchestrator.py`, update `test_generate_batch_produces_manifest_and_assets`'s die_type assertion:

```python
            assert record["die_type"] in ("d4", "d6", "d8", "d10", "d10_pct", "d12", "d20")
```

Update `test_generate_set_batch_produces_matching_set` in full:

```python
def test_generate_set_batch_produces_matching_set():
    from dice_gen import orchestrator

    with tempfile.TemporaryDirectory() as outdir:
        generated, failed = orchestrator.generate_set_batch(num_sets=1, seed=2000, outdir=outdir)

        assert generated + failed == 7

        manifest_path = os.path.join(outdir, "manifest.json")
        with open(manifest_path) as f:
            manifest = json.load(f)
        assert len(manifest) == generated

        set_ids = {record["set_id"] for record in manifest}
        assert len(set_ids) == 1

        die_types = {record["die_type"] for record in manifest}
        assert die_types == {"d4", "d6", "d8", "d10", "d10_pct", "d12", "d20"}

        material_categories = {record["material_category"] for record in manifest}
        font_ids = {record["font_or_style_id"] for record in manifest}
        assert len(material_categories) == 1
        assert len(font_ids) == 1

        non_pct_glyph_styles = {
            record["glyph_style"] for record in manifest if record["die_type"] != "d10_pct"
        }
        assert len(non_pct_glyph_styles) == 1, (
            f"expected the 6 non-percentile dice to share one glyph_style, got {non_pct_glyph_styles}"
        )
        pct_record = next(r for r in manifest if r["die_type"] == "d10_pct")
        assert pct_record["glyph_style"] == "arabic_numerals"
```

- [ ] **Step 2: Run the pure-Python tests to verify they fail**

Run: `cd /home/saps/projects/Dice-Detection && python3 -m pytest tests/test_sampler.py -v`
Expected: FAIL — `KeyError: 'd10_pct'` from `sampler.SIZE_RANGES_MM['d10_pct']`, and the die-type-keys test fails since `"d10_pct"` isn't in `sampler.DIE_TYPES` yet.

- [ ] **Step 3: Implement**

In `src/dice_gen/sampler.py`:

```python
DIE_TYPES = ["d4", "d6", "d8", "d10", "d10_pct", "d12", "d20"]

SIZE_RANGES_MM = {
    "d4": (14.0, 20.0),
    "d6": (12.0, 20.0),
    "d8": (14.0, 20.0),
    "d10": (14.0, 20.0),
    "d10_pct": (14.0, 20.0),
    "d12": (16.0, 22.0),
    "d20": (16.0, 24.0),
}
```

In `sample_variant`, change the glyph_style branch:

```python
    if die_type in ("d6", "d4"):
        glyph_style = rng.choice(["arabic_numerals", "pips"])
    elif die_type == "d10_pct":
        glyph_style = "arabic_numerals"
    else:
        glyph_style = rng.choice([s for s in GLYPH_STYLES if s != "pips"])
```

In `sample_set`, override per-die-type inside the loop (the set-wide `glyph_style` sampled earlier in the function stays as the shared style for the other 6 dice):

```python
    variants = {}
    for die_type in DIE_TYPES:
        lo, hi = SIZE_RANGES_MM[die_type]
        size_mm = rng.uniform(lo, hi)
        d4_placement = rng.choice(D4_PLACEMENT_STYLES) if die_type == "d4" else None
        die_glyph_style = "arabic_numerals" if die_type == "d10_pct" else glyph_style

        variants[die_type] = DiceVariantParams(
            die_type=die_type,
            size_mm=size_mm,
            bevel_fraction=bevel_fraction,
            numbering_scheme=f"standard_{die_type}",
            glyph_style=die_glyph_style,
            glyph_method=glyph_method,
            glyph_fill=glyph_fill,
            font_or_style_id=font_or_style_id,
            material_category=material_category,
            material_params=material_params,
            d4_placement=d4_placement,
            seed=seed,
        )

    return variants
```

- [ ] **Step 4: Run the pure-Python tests to verify they pass**

Run: `cd /home/saps/projects/Dice-Detection && python3 -m pytest tests/test_sampler.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Run the full-pipeline integration test to verify it fails, then passes**

Run: `cd /home/saps/projects/Dice-Detection && blender --background --python tests/blender/test_orchestrator.py 2>&1 | tail -60`

Before Step 3's `sampler.py` changes, this would already be broken by `sampler.DIE_TYPES` lacking `"d10_pct"` in combination with the test file changes above (e.g. `die_types == {..., "d10_pct"}` failing). Since Step 3 is already applied by this point, expect this run to `ALL TESTS PASSED` directly — if it doesn't, the failure is real and must be fixed before proceeding (this test exercises Tasks 1-4 together end-to-end via `generate_set_batch`, the first true integration check of the whole feature).

- [ ] **Step 6: Commit**

```bash
git add src/dice_gen/sampler.py tests/test_sampler.py tests/blender/test_orchestrator.py
git commit -m "feat: wire d10_pct into sampler and matched-set generation"
```

---

### Task 5: Validator — recognize `d10_pct` in expected 7-die sets

**Files:**
- Modify: `scripts/validate_dice_assets.py`
- Modify: `tests/test_validate_dice_assets.py`

**Interfaces:**
- Consumes: `numbering.get_values("d10_pct")` from Task 1 (already generic, no change needed there).
- Produces: `scripts.validate_dice_assets.EXPECTED_SET_DIE_TYPES` includes `"d10_pct"`, so a set missing the percentile die is now flagged as incomplete, and a complete 7-die set validates cleanly.

- [ ] **Step 1: Write the failing tests**

In `tests/test_validate_dice_assets.py`, update `_set_record`'s `num_sides` lookup:

```python
def _set_record(tmp_path, asset_id, die_type, set_id):
    usd_name = f"{asset_id}.usd"
    thumb_name = f"{asset_id}_thumb.png"
    open(os.path.join(tmp_path, usd_name), "w").write("x")
    open(os.path.join(tmp_path, thumb_name), "w").close()
    num_sides = {"d4": 4, "d6": 6, "d8": 8, "d10": 10, "d10_pct": 10, "d12": 12, "d20": 20}[die_type]
    return {
        "asset_id": asset_id, "die_type": die_type, "num_sides": num_sides,
        "usd_path": usd_name, "thumbnail_path": thumb_name, "set_id": set_id,
    }
```

Update `test_validate_passes_for_complete_set`:

```python
def test_validate_passes_for_complete_set():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_path:
        die_types = ["d4", "d6", "d8", "d10", "d10_pct", "d12", "d20"]
        records = [
            _set_record(tmp_path, f"set_00000_{dt}", dt, "set_00000")
            for dt in die_types
        ]
        _write_manifest(tmp_path, records)

        errors = validate(tmp_path)
        assert errors == []
```

Add a new test after `test_validate_reports_missing_die_type_in_set` (that existing test, which omits `d20` from a 5-die set, still passes unchanged once `EXPECTED_SET_DIE_TYPES` grows to 7 — the set is now missing both `d20` and `d10_pct`, and the test only asserts `d20` appears in the single combined error message, which remains true):

```python
def test_validate_reports_missing_percentile_die_in_set():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_path:
        die_types = ["d4", "d6", "d8", "d10", "d12", "d20"]  # missing d10_pct
        records = [
            _set_record(tmp_path, f"set_00000_{dt}", dt, "set_00000")
            for dt in die_types
        ]
        _write_manifest(tmp_path, records)

        errors = validate(tmp_path)
        set_errors = [e for e in errors if "set_00000" in e]
        assert len(set_errors) == 1
        assert "missing die types" in set_errors[0]
        assert "d10_pct" in set_errors[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/saps/projects/Dice-Detection && python3 -m pytest tests/test_validate_dice_assets.py -v`
Expected: `test_validate_passes_for_complete_set` FAILs (`d10_pct` reported as missing, since `EXPECTED_SET_DIE_TYPES` doesn't include it yet). `test_validate_reports_missing_percentile_die_in_set` FAILs too (message doesn't mention `d10_pct` since it isn't expected yet, so nothing is reported missing).

- [ ] **Step 3: Implement**

In `scripts/validate_dice_assets.py`:

```python
EXPECTED_SET_DIE_TYPES = {"d4", "d6", "d8", "d10", "d10_pct", "d12", "d20"}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/saps/projects/Dice-Detection && python3 -m pytest tests/test_validate_dice_assets.py -v`
Expected: all tests PASS.

Also run the full pure-Python suite to confirm no regressions:

Run: `cd /home/saps/projects/Dice-Detection && python3 -m pytest tests/ -v --ignore=tests/blender`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/validate_dice_assets.py tests/test_validate_dice_assets.py
git commit -m "feat: recognize d10_pct in expected 7-die matched sets"
```

---

### Task 6: Generate a real batch and verify end-to-end

**Files:** none modified — verification only, per this project's standing practice of verifying exported files fresh rather than trusting in-memory/generation-time state.

**Interfaces:** none — this task consumes Tasks 1-5's finished code as-is.

- [ ] **Step 1: Run the full Blender test suite once more, back to back**

Run:
```bash
cd /home/saps/projects/Dice-Detection && \
  blender --background --python tests/blender/test_geometry.py 2>&1 | tail -10 && \
  blender --background --python tests/blender/test_glyphs.py 2>&1 | tail -10 && \
  blender --background --python tests/blender/test_materials.py 2>&1 | tail -10 && \
  blender --background --python tests/blender/test_exporter.py 2>&1 | tail -10 && \
  blender --background --python tests/blender/test_orchestrator.py 2>&1 | tail -10
```
Expected: every file prints `ALL TESTS PASSED`.

- [ ] **Step 2: Generate a small real batch of matched sets**

Run: `cd /home/saps/projects/Dice-Detection && blender --background --python scripts/generate_dice_assets.py -- --sets 5 --seed 500 --outdir data/raw/dice_percentile_smoke`

Expected: completes without crashing, prints `Generated: 35, Failed: 0` (5 sets × 7 dice). A non-zero `Failed` count would mean some asset's generation raised (including, notably, `numbering.verify_opposite_sum` failing for a `d10_pct` asset) — if that happens, inspect `data/raw/dice_percentile_smoke/failures.json` and treat any `d10_pct`-related failure as a real regression to fix before proceeding, not something to route around.

- [ ] **Step 3: Validate the batch**

Run: `cd /home/saps/projects/Dice-Detection && python3 scripts/validate_dice_assets.py data/raw/dice_percentile_smoke`

Expected: exits 0, `0 error(s)`. This confirms the manifest-level structural checks pass, including the new `EXPECTED_SET_DIE_TYPES` check (all 5 sets have all 7 die types, no duplicates) and `num_sides` correctness for `d10_pct` (10 sides).

- [ ] **Step 4: Confirm zero d10_pct generation failures**

Run: `cd /home/saps/projects/Dice-Detection && python3 -c "import json; f = json.load(open('data/raw/dice_percentile_smoke/failures.json')); pct = [x for x in f if 'd10_pct' in x['asset_id']]; print('d10_pct failures:', len(pct)); print(pct)"`

Expected: `d10_pct failures: 0`. Since `orchestrator._generate_from_params` raises `ValueError` and routes to `failures.json` on any `verify_opposite_sum` violation, zero `d10_pct` failures across all 5 generated assets is direct proof the opposite-sum-90 invariant held for every one — this is the practical verification for that invariant (the assignment/scale logic doesn't have a way to silently produce a wrong-but-passing result at the manifest level, since a violation always raises).

- [ ] **Step 5: Fresh-reload spot check on one `d10_pct` asset**

Find one d10_pct asset ID from the manifest:

Run: `cd /home/saps/projects/Dice-Detection && python3 -c "import json; m = json.load(open('data/raw/dice_percentile_smoke/manifest.json')); r = next(x for x in m if x['die_type'] == 'd10_pct'); print(r['asset_id']); print(r['blend_path']); print(r['thumbnail_path'])"`

Then reload the saved `.blend` fresh (not the in-memory session that generated it) and inspect the mesh:

Run: `cd /home/saps/projects/Dice-Detection && blender --background data/raw/dice_percentile_smoke/<blend_path from above> --python-expr "import bpy; obj = [o for o in bpy.data.objects if o.type == 'MESH'][0]; print('OBJECTS:', [o.name for o in bpy.data.objects]); print('VERTS:', len(obj.data.vertices)); print('MATERIALS:', [m.name for m in obj.data.materials])"`

Expected: `OBJECTS:` lists exactly one mesh object (the die). `MATERIALS:` lists 1 or 2 materials (2 if the sampled `glyph_fill` was `"painted"`), confirming the file reloads cleanly and isn't corrupted or empty.

- [ ] **Step 6: Visual spot check on the thumbnail**

Use the Read tool on `data/raw/dice_percentile_smoke/<thumbnail_path from Step 5>` and confirm the rendered die shows a plausible two-digit numeral ending in 0 (e.g. "30", "70") on its visible face — not a corrupted/collapsed glyph, not a single unpadded digit. This is a human/agent-eyeball check, not scriptable — it's the same category of check this project's `CLAUDE.md` already calls for (thumbnail-based spot check), and mirrors the acknowledged limitation in `docs/ROADMAP.md` item 1 that structural checks alone (`mesh_quality_warnings`/`engraving_warnings`) can't confirm a numeral is actually visually present and correct.

No commit for this task — it's a verification pass over freshly generated (gitignored) data, not a code change. Report the actual `Generated`/`Failed` counts and anything found in Step 6 honestly, per this project's standing practice — a clean-sounding "done" that skips a real observation is worse than an honest partial result.
