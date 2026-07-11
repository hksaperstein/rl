# Glyph Orientation, Decal Visibility, and Font Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three real, empirically-confirmed bugs in the glyph pipeline — painted/decal numerals often invisible (shared-UV-atlas bug), inconsistent numeral rotation between adjacent faces (orientation heuristic discontinuity), and `font_or_style_id` being sampled but never actually applied to any rendered glyph — plus two small tuning changes (shallower engraving depth, a fresh seed for the next regeneration).

**Architecture:** All code changes are contained in `src/dice_gen/glyphs.py`. A new shared `_tangent_bitangent(normal)` helper replaces the inline up-hint logic in `_face_orientation_matrix` with a corrected threshold, and is reused by a new `_unwrap_faces_to_full_square` function that replaces `bpy.ops.uv.smart_project` in `apply_decal_glyphs`. A new `_load_font(font_id, glyph_style)` helper maps `font_or_style_id` to real installed font files and is wired into both `apply_engraved_glyphs` and `_render_label_to_image` (which gains a `font_id` parameter). No other files change.

**Tech Stack:** Same as the rest of `src/dice_gen/` — Blender 5.1.2's bundled `bpy`, no external pip packages. Every fact below (UV island bounds, orientation discontinuity location, font glyph coverage) was confirmed by running standalone scripts through `blender --background --python` during planning; see the design doc for the raw evidence.

## Global Constraints

- The shared tangent/bitangent basis must be used by BOTH the engrave cutter placement (`_face_orientation_matrix`, world-space normal) and the new decal UV unwrap (local-space normal) — one consistent convention, not two independent ones.
- The up-hint fallback threshold is `abs(normal.z) < 0.999` (not `0.9`) — it must only fire for genuinely axis-aligned normals (where the Z-based cross product is truly degenerate), never for merely-steep faces.
- `_unwrap_faces_to_full_square` must give each face a UV island that (a) spans at least `1.0 - 2*margin` in its larger axis, and (b) contains the `(0.5, 0.5)` center point — this is what guarantees the glyph (always centered in its own dedicated per-face texture) actually lands somewhere visible on the face.
- Font mapping is exactly: `font_sans_bold` → `/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf`, `font_serif_regular` → `/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf`, `font_display_condensed` → `/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Bold.ttf`. These exact file paths were confirmed present on this system during planning.
- `glyph_style == "cjk_numerals"` must ALWAYS use Blender's default font, regardless of `font_or_style_id` — confirmed empirically that the mapped Liberation fonts have no CJK glyph coverage (render as an empty placeholder box) while Blender's default font renders CJK correctly.
- A font file must be loaded once and reused across a batch (check `bpy.data.fonts` for an already-loaded font with the matching filepath before calling `bpy.data.fonts.load` again).
- `ENGRAVE_DEPTH_FRACTION` changes from `0.04` to `0.025`. No other constant/threshold in the engrave-failure-detection logic changes.
- No change to `_composite_alpha_over`, `_render_material_swatch`, the USD/STL/`.blend` export pipeline, `materials.py`, `orchestrator.py`, or `sampler.py`.
- The earlier resting-pose request is explicitly out of scope — the die's object origin stays at its geometric center.

---

## File Structure

- Modify: `src/dice_gen/glyphs.py` — new `_tangent_bitangent`, `_face_orientation_matrix` refactor, new `_unwrap_faces_to_full_square`, `apply_decal_glyphs` uses it instead of `smart_project`, new `FONT_FILES` + `_load_font`, `apply_engraved_glyphs` and `_render_label_to_image` (gains a `font_id` parameter) use `_load_font`, `ENGRAVE_DEPTH_FRACTION` changes.
- Test: `tests/blender/test_glyphs.py` — new tests per task, inserted before `run()`.

---

### Task 1: Shared tangent/bitangent helper with corrected threshold

**Files:**
- Modify: `src/dice_gen/glyphs.py:60-68` (`_face_orientation_matrix`)
- Test: `tests/blender/test_glyphs.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `_tangent_bitangent(normal, threshold=0.999) -> (Vector, Vector)` — a module-level function in `glyphs.py` consumed by Task 2's `_unwrap_faces_to_full_square`.

- [ ] **Step 1: Write the failing tests**

Add these two tests to `tests/blender/test_glyphs.py`, right before the `run()` function at the end of the file:

```python
def test_tangent_bitangent_only_falls_back_to_y_for_truly_vertical_normals():
    """
    Regression test for the orientation-discontinuity bug: the old
    abs(normal.z) < 0.9 threshold triggered the Y-axis up-hint fallback
    for a d20's near-pole faces (normal.z ~= +/-0.9342), which are tilted
    but NOT axis-aligned, creating an abrupt jump between that ring and
    its neighboring ring -- confirmed both numerically (dumping every
    d20 face's computed bitangent: the old threshold produced a flat
    (0, 1, 0) exactly at that ring while neighboring rings varied
    smoothly) and visually (one face's engraved numeral reading upright,
    the adjacent face's numeral at a distinctly different angle). The
    fixed threshold (0.999) must mean every one of a d20's 20 faces
    (none of which has a normal exactly axis-aligned to Z) uses the
    smoothly-varying Z-projection, never the flat Y fallback.
    """
    import bpy
    from dice_gen import geometry, glyphs

    obj = geometry.build_die_base_mesh("d20", size_mm=20.0)
    for face in obj.data.polygons:
        normal = face.normal
        assert abs(normal.z) < 0.999, (
            f"test assumption violated: d20 face {face.index} has "
            f"normal.z={normal.z:.4f}, expected all d20 faces to be "
            f"non-axis-aligned"
        )
        tangent, bitangent = glyphs._tangent_bitangent(normal)
        is_flat_y_fallback = (
            abs(bitangent.x) < 1e-6
            and abs(bitangent.z) < 1e-6
            and abs(abs(bitangent.y) - 1.0) < 1e-6
        )
        assert not is_flat_y_fallback, (
            f"face {face.index} (normal.z={normal.z:.4f}, not axis-aligned) "
            f"incorrectly used the flat Y-axis fallback bitangent {tuple(bitangent)}"
        )
    bpy.data.objects.remove(obj, do_unlink=True)


def test_tangent_bitangent_falls_back_to_y_for_axis_aligned_normals():
    """
    d6 has faces with normal EXACTLY axis-aligned to global Z (e.g. a
    cube's top/bottom faces). For these, up_hint=Z and normal=Z are
    parallel, making up_hint.cross(normal) the zero vector -- an
    undefined/degenerate tangent. The Y-axis fallback must still trigger
    for these so tangent/bitangent stay well-defined (non-zero-length),
    confirming the fixed 0.999 threshold didn't accidentally remove the
    fallback for the case it's actually needed for.
    """
    import bpy
    from dice_gen import geometry, glyphs

    obj = geometry.build_die_base_mesh("d6", size_mm=16.0)
    found_axis_aligned_face = False
    for face in obj.data.polygons:
        normal = face.normal
        if abs(normal.z) > 0.999:
            found_axis_aligned_face = True
            tangent, bitangent = glyphs._tangent_bitangent(normal)
            assert tangent.length > 0.5, (
                f"degenerate (near-zero-length) tangent for axis-aligned "
                f"face {face.index}: {tuple(tangent)}"
            )
            assert bitangent.length > 0.5, (
                f"degenerate (near-zero-length) bitangent for axis-aligned "
                f"face {face.index}: {tuple(bitangent)}"
            )
    assert found_axis_aligned_face, (
        "expected d6 to have at least one Z-axis-aligned face (a cube's "
        "top or bottom face) -- test assumption violated"
    )
    bpy.data.objects.remove(obj, do_unlink=True)
```

Add both function names to `run()` at the bottom of the file, right after `test_discard_non_body_closed_debris_returns_warning_for_manual_debris()`:

```python
    test_tangent_bitangent_only_falls_back_to_y_for_truly_vertical_normals()
    test_tangent_bitangent_falls_back_to_y_for_axis_aligned_normals()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/saps/projects/Dice-Detection && blender --background --python tests/blender/test_glyphs.py; echo "exit=$?"`
Expected: FAIL (non-zero exit). Both new tests fail with `AttributeError: module 'dice_gen.glyphs' has no attribute '_tangent_bitangent'`, since the function doesn't exist yet.

- [ ] **Step 3: Implement `_tangent_bitangent` and refactor `_face_orientation_matrix`**

In `src/dice_gen/glyphs.py`, replace the existing `_face_orientation_matrix` function (lines 60-68) with:

```python
def _tangent_bitangent(normal, threshold=0.999):
    """
    Given a (normalized) face normal, returns a consistent (tangent,
    bitangent) in-plane basis by projecting a global "up" reference
    direction onto the face's plane. Global +Z is used as the up
    reference for every face EXCEPT when normal is itself (near-)parallel
    to +/-Z, where the projection is undefined (up_hint.cross(normal)
    would be the zero vector) -- global +Y is used instead for that
    narrow case only.

    The threshold (normal.z's absolute value) for switching to the Y
    fallback must stay very close to 1.0. An earlier version used 0.9,
    which also caught merely-steep-but-not-vertical faces (e.g. a d20's
    near-pole ring, normal.z ~= +/-0.9342): those faces got the flat
    (0, 1, 0) fallback while their immediate neighbors (normal.z ~= +/-0.577)
    got the smoothly-varying Z-projection, producing an abrupt rotation
    jump between adjacent faces -- confirmed both numerically (dumping
    every d20 face's computed bitangent) and visually (one face's engraved
    numeral reading upright, the adjacent face's numeral at a distinctly
    different angle). 0.999 only catches genuinely axis-aligned normals
    (e.g. d6/d8's exactly-vertical top/bottom faces), where the fallback
    is actually required to avoid a degenerate zero-length tangent.

    Shared by _face_orientation_matrix (engraved cutter placement,
    world-space normal) and _unwrap_faces_to_full_square (decal UV
    unwrap, local-space normal) so both glyph methods use one consistent
    orientation convention instead of two independently-behaving ones.
    """
    up_hint = Vector((0, 0, 1)) if abs(normal.z) < threshold else Vector((0, 1, 0))
    tangent = up_hint.cross(normal).normalized()
    bitangent = normal.cross(tangent).normalized()
    return tangent, bitangent


def _face_orientation_matrix(face, obj_matrix):
    center = obj_matrix @ face.center
    normal = (obj_matrix.to_3x3() @ face.normal).normalized()
    tangent, bitangent = _tangent_bitangent(normal)
    rot = Matrix((tangent, bitangent, normal)).transposed().to_4x4()
    rot.translation = center
    return rot
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/saps/projects/Dice-Detection && blender --background --python tests/blender/test_glyphs.py; echo "exit=$?"`
Expected: `ALL TESTS PASSED` and `exit=0`.

- [ ] **Step 5: Commit**

```bash
git add src/dice_gen/glyphs.py tests/blender/test_glyphs.py
git commit -m "fix: use one consistent up-hint threshold for glyph orientation"
```

---

### Task 2: Decal glyphs get a per-face full-square UV unwrap instead of a shared atlas

**Files:**
- Modify: `src/dice_gen/glyphs.py:628-649` (`apply_decal_glyphs`'s unwrap step)
- Test: `tests/blender/test_glyphs.py`

**Interfaces:**
- Consumes: `_tangent_bitangent(normal, threshold=0.999) -> (Vector, Vector)` from Task 1.
- Produces: `_unwrap_faces_to_full_square(die_obj, margin=0.1) -> None` (mutates `die_obj.data`'s active UV layer in place) — no other task depends on this function directly.

- [ ] **Step 1: Write the failing test**

Add this test to `tests/blender/test_glyphs.py`, right before `run()`:

```python
def test_unwrap_faces_to_full_square_covers_full_uv_range_per_face():
    """
    Regression test for the decal-glyph-invisibility bug: apply_decal_glyphs
    used to call bpy.ops.uv.smart_project across the whole die, which packs
    every face into one shared UV atlas -- confirmed empirically on a d8,
    each face's island only covered roughly a 0.27x0.31 patch of the 0-1
    space (e.g. u=[0.013,0.279], v=[0.013,0.320]), never the full square.
    Since each face has its OWN dedicated texture (the composited
    swatch+glyph PNG, glyph centered at (0.5, 0.5)), most faces ended up
    sampling only a background-colored corner of their own image, missing
    the glyph entirely.

    _unwrap_faces_to_full_square must give each face's UV island a span
    of at least 1.0 - 2*margin in its larger axis, AND must have that
    island contain the (0.5, 0.5) center point where the glyph's ink
    actually lives -- proving the glyph will land somewhere visible on
    the face, not just that SOME UV coordinates exist. Checked across
    every die type, since face shape (triangle, quad, kite, pentagon)
    differs by type.
    """
    import bpy
    from dice_gen import geometry, glyphs

    margin = 0.1
    for die_type in ("d4", "d6", "d8", "d10", "d12", "d20"):
        obj = geometry.build_die_base_mesh(die_type, size_mm=16.0)
        glyphs._unwrap_faces_to_full_square(obj, margin=margin)

        uv_layer = obj.data.uv_layers.active.data
        for poly in obj.data.polygons:
            us = [uv_layer[li].uv.x for li in poly.loop_indices]
            vs = [uv_layer[li].uv.y for li in poly.loop_indices]
            u_span = max(us) - min(us)
            v_span = max(vs) - min(vs)

            assert max(u_span, v_span) >= (1.0 - 2 * margin) - 1e-6, (
                f"{die_type} face {poly.index}: UV span too small "
                f"(u_span={u_span:.3f}, v_span={v_span:.3f}), expected "
                f"at least {1.0 - 2 * margin:.3f} in one axis"
            )
            assert min(us) <= 0.5 <= max(us), (
                f"{die_type} face {poly.index}: UV u-range "
                f"{min(us):.3f}-{max(us):.3f} does not contain the image "
                f"center (0.5) where the glyph's ink lives"
            )
            assert min(vs) <= 0.5 <= max(vs), (
                f"{die_type} face {poly.index}: UV v-range "
                f"{min(vs):.3f}-{max(vs):.3f} does not contain the image "
                f"center (0.5) where the glyph's ink lives"
            )

        bpy.data.objects.remove(obj, do_unlink=True)
```

Add the function name to `run()`, right after `test_tangent_bitangent_falls_back_to_y_for_axis_aligned_normals()`:

```python
    test_unwrap_faces_to_full_square_covers_full_uv_range_per_face()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/saps/projects/Dice-Detection && blender --background --python tests/blender/test_glyphs.py; echo "exit=$?"`
Expected: FAIL (non-zero exit) with `AttributeError: module 'dice_gen.glyphs' has no attribute '_unwrap_faces_to_full_square'`.

- [ ] **Step 3: Implement `_unwrap_faces_to_full_square` and use it in `apply_decal_glyphs`**

In `src/dice_gen/glyphs.py`, add this new function directly above `apply_decal_glyphs`:

```python
def _unwrap_faces_to_full_square(die_obj, margin=0.1):
    """
    Gives every face its OWN UV island filling the full 0-1 square,
    instead of bpy.ops.uv.smart_project's shared-atlas packing (which
    only gives each face a small fraction of the 0-1 space -- confirmed
    empirically on a d8, each face's island only covered roughly a
    0.27x0.31 patch). apply_decal_glyphs gives each face its own
    DEDICATED texture image (the glyph centered at (0.5, 0.5)), so an
    atlas-style shared unwrap is the wrong tool: it leaves most faces
    sampling only a background-colored corner of their own image,
    missing the centered glyph entirely.

    Projects each face's vertices into its own (tangent, bitangent) frame
    (see _tangent_bitangent) relative to the face center, then scales so
    the larger of the two axis spans fits into 1.0 - 2*margin, centered
    at (0.5, 0.5). Verified empirically to produce full per-face coverage
    across d6/d8/d10/d12/d20's differently-shaped faces (triangle, quad,
    kite, pentagon).
    """
    mesh = die_obj.data
    if mesh.uv_layers.active is None:
        mesh.uv_layers.new(name="decal_uv")
    uv_layer = mesh.uv_layers.active.data

    for poly in mesh.polygons:
        tangent, bitangent = _tangent_bitangent(poly.normal)
        center = poly.center

        local_coords = []
        for loop_index in poly.loop_indices:
            vertex_index = mesh.loops[loop_index].vertex_index
            rel = mesh.vertices[vertex_index].co - center
            local_coords.append((rel.dot(tangent), rel.dot(bitangent)))

        us = [c[0] for c in local_coords]
        vs = [c[1] for c in local_coords]
        span = max(max(us) - min(us), max(vs) - min(vs))
        scale = (1.0 - 2 * margin) / span if span > 0 else 1.0

        for loop_index, (u, v) in zip(poly.loop_indices, local_coords):
            uv_layer[loop_index].uv = (0.5 + u * scale, 0.5 + v * scale)
```

Then, in `apply_decal_glyphs`, replace this block (currently lines 645-649):

```python
    bpy.context.view_layer.objects.active = die_obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(island_margin=0.05)
    bpy.ops.object.mode_set(mode='OBJECT')
```

with:

```python
    _unwrap_faces_to_full_square(die_obj)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/saps/projects/Dice-Detection && blender --background --python tests/blender/test_glyphs.py; echo "exit=$?"`
Expected: `ALL TESTS PASSED` and `exit=0`.

- [ ] **Step 5: Commit**

```bash
git add src/dice_gen/glyphs.py tests/blender/test_glyphs.py
git commit -m "fix: give every decal face its own full-square UV island instead of a shared atlas"
```

---

### Task 3: Wire `font_or_style_id` to real fonts

**Files:**
- Modify: `src/dice_gen/glyphs.py` — new `FONT_FILES` + `_load_font`; `apply_engraved_glyphs` (text-cutter creation) and `_render_label_to_image` (gains a `font_id` parameter) both call it; `apply_decal_glyphs`'s call to `_render_label_to_image` passes `font_id` through.
- Test: `tests/blender/test_glyphs.py`

**Interfaces:**
- Consumes: nothing new from Tasks 1-2.
- Produces: `FONT_FILES: dict[str, str]` and `_load_font(font_id, glyph_style) -> bpy.types.VectorFont | None` — not consumed by any later task.

- [ ] **Step 1: Write the failing tests**

Add these five tests to `tests/blender/test_glyphs.py`, right before `run()`:

```python
def test_load_font_maps_font_ids_to_distinct_installed_fonts():
    """
    Regression test for the font_or_style_id-is-sampled-but-never-applied
    gap: sampler.py samples one of FONT_IDS ("font_sans_bold",
    "font_serif_regular", "font_display_condensed") per die and stores it
    in the manifest, but neither apply_engraved_glyphs nor
    _render_label_to_image ever read it -- every die used Blender's single
    default font regardless. This checks _load_font maps each of the 3
    IDs to a real, distinct font file (confirmed installed on this system
    during planning), and that the SAME font_id returns the SAME font
    datablock on a second call (no redundant reload).
    """
    from dice_gen import glyphs

    seen_filepaths = set()
    for font_id, expected_path in glyphs.FONT_FILES.items():
        font = glyphs._load_font(font_id, glyph_style="arabic_numerals")
        assert font is not None, f"{font_id}: expected a loaded font, got None"
        assert font.filepath == expected_path, (
            f"{font_id}: expected filepath {expected_path}, got {font.filepath}"
        )
        assert font.filepath not in seen_filepaths, (
            f"{font_id}: filepath {font.filepath} was already used by another "
            f"font_id -- font_or_style_id values must map to genuinely distinct fonts"
        )
        seen_filepaths.add(font.filepath)

        font_again = glyphs._load_font(font_id, glyph_style="arabic_numerals")
        assert font_again is font, (
            f"{font_id}: calling _load_font twice should reuse the same "
            f"loaded font datablock, not create a duplicate"
        )


def test_load_font_returns_none_for_cjk_numerals_regardless_of_font_id():
    """
    Liberation Sans/Serif/Sans-Narrow (this project's FONT_FILES) have no
    CJK glyph coverage -- confirmed by rendering a CJK character with
    Liberation Sans Bold during planning, which produced an empty
    placeholder rectangle instead of the correct character, while
    Blender's own default bundled font renders the same character
    correctly. _load_font must return None for glyph_style ==
    "cjk_numerals" for every font_id, so the caller leaves
    txt_obj.data.font at Blender's default rather than swapping to a font
    that can't render the requested characters.
    """
    from dice_gen import glyphs

    for font_id in glyphs.FONT_FILES:
        font = glyphs._load_font(font_id, glyph_style="cjk_numerals")
        assert font is None, (
            f"{font_id}: expected None for cjk_numerals glyph_style, got {font}"
        )


def test_load_font_returns_none_for_unrecognized_font_id():
    from dice_gen import glyphs

    font = glyphs._load_font("not_a_real_font_id", glyph_style="arabic_numerals")
    assert font is None


def test_apply_engraved_glyphs_uses_load_font_with_correct_glyph_style():
    """
    Confirms apply_engraved_glyphs actually calls _load_font with this
    call's own font_id/glyph_style (not a stale/hardcoded value), via a
    spy on the real function -- since the cutter text object is destroyed
    by the boolean cut, this is the only way to prove the wiring happened
    without re-deriving font state from the final baked mesh.
    """
    import bpy
    from dice_gen import geometry, numbering, glyphs

    die_type = "d6"
    size_mm = 16.0
    obj = geometry.build_die_base_mesh(die_type, size_mm=size_mm)
    pairs = geometry.compute_opposite_face_pairs(obj)
    assignment = numbering.assign_values_to_opposite_pairs(die_type, pairs)
    assert len(assignment) == 6, "d6 should have 6 faces assigned"

    real_load_font = glyphs._load_font
    calls = []

    def spy_load_font(font_id, glyph_style):
        calls.append((font_id, glyph_style))
        return real_load_font(font_id, glyph_style)

    glyphs._load_font = spy_load_font
    try:
        glyphs.apply_engraved_glyphs(
            obj, die_type, assignment,
            glyph_style="roman_numerals", glyph_fill="blank",
            font_id="font_serif_regular", size_mm=size_mm,
        )
    finally:
        glyphs._load_font = real_load_font

    assert len(calls) == 6, f"expected one _load_font call per face cut, got {len(calls)}"
    assert all(c == ("font_serif_regular", "roman_numerals") for c in calls), calls

    bpy.data.objects.remove(obj, do_unlink=True)


def test_apply_decal_glyphs_uses_load_font_with_correct_glyph_style():
    import bpy
    from dice_gen import geometry, numbering, glyphs, materials

    die_type = "d6"
    size_mm = 16.0
    obj = geometry.build_die_base_mesh(die_type, size_mm=size_mm)
    pairs = geometry.compute_opposite_face_pairs(obj)
    assignment = numbering.assign_values_to_opposite_pairs(die_type, pairs)
    mat = materials.build_material(obj.name, "opaque", {"hue": 0.3, "saturation": 0.7, "value": 0.6, "roughness": 0.4})
    materials.apply_material(obj, mat)

    real_load_font = glyphs._load_font
    calls = []

    def spy_load_font(font_id, glyph_style):
        calls.append((font_id, glyph_style))
        return real_load_font(font_id, glyph_style)

    glyphs._load_font = spy_load_font
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            glyphs.apply_decal_glyphs(
                obj, die_type, assignment,
                glyph_style="greek_numerals", font_id="font_display_condensed",
                size_mm=size_mm, asset_id="test_font_spy", tmp_dir=tmp_dir,
            )
    finally:
        glyphs._load_font = real_load_font

    assert len(calls) == 6, f"expected one _load_font call per face, got {len(calls)}"
    assert all(c == ("font_display_condensed", "greek_numerals") for c in calls), calls

    bpy.data.objects.remove(obj, do_unlink=True)
```

Add all five function names to `run()`, right after `test_unwrap_faces_to_full_square_covers_full_uv_range_per_face()`:

```python
    test_load_font_maps_font_ids_to_distinct_installed_fonts()
    test_load_font_returns_none_for_cjk_numerals_regardless_of_font_id()
    test_load_font_returns_none_for_unrecognized_font_id()
    test_apply_engraved_glyphs_uses_load_font_with_correct_glyph_style()
    test_apply_decal_glyphs_uses_load_font_with_correct_glyph_style()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/saps/projects/Dice-Detection && blender --background --python tests/blender/test_glyphs.py; echo "exit=$?"`
Expected: FAIL (non-zero exit) with `AttributeError: module 'dice_gen.glyphs' has no attribute 'FONT_FILES'` (or `_load_font`), and the two spy-based tests fail because `_render_label_to_image` doesn't yet accept a `font_id` argument.

- [ ] **Step 3: Implement `FONT_FILES`, `_load_font`, and wire both call sites**

In `src/dice_gen/glyphs.py`, add this near the top of the file, right after the `ENGRAVE_DEPTH_FRACTION` line:

```python
FONT_FILES = {
    "font_sans_bold": "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "font_serif_regular": "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "font_display_condensed": "/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Bold.ttf",
}


def _load_font(font_id, glyph_style):
    """
    Maps a sampled font_or_style_id to a real, distinct installed font,
    loaded once and reused (bpy.data.fonts.load creates a new VectorFont
    datablock each call unless one with a matching filepath is already
    loaded, so this checks first to avoid redundant loads across a
    batch).

    Returns None for glyph_style == "cjk_numerals" regardless of
    font_id -- confirmed empirically that none of FONT_FILES' fonts have
    CJK glyph coverage (rendering a CJK character with Liberation Sans
    Bold produces an empty placeholder rectangle, not the correct
    character), while Blender's own default bundled font already renders
    CJK correctly. Returning None means the caller leaves
    txt_obj.data.font unset, i.e. Blender's default font.
    """
    if glyph_style == "cjk_numerals":
        return None
    font_path = FONT_FILES.get(font_id)
    if font_path is None:
        return None
    for font in bpy.data.fonts:
        if font.filepath == font_path:
            return font
    return bpy.data.fonts.load(font_path)
```

In `apply_engraved_glyphs`, in the text-cutter branch (the `else:` branch that builds a numeral/pip label cutter), add the font assignment right after `txt_obj.data.body = label`:

```python
            label = glyph_label(value, glyph_style)
            bpy.ops.object.text_add()
            txt_obj = bpy.context.active_object
            txt_obj.data.body = label
            font = _load_font(font_id, glyph_style)
            if font is not None:
                txt_obj.data.font = font
            txt_obj.data.align_x = 'CENTER'
            txt_obj.data.align_y = 'CENTER'
            txt_obj.data.size = glyph_font_size
            txt_obj.data.extrude = depth
```

In `_render_label_to_image`, change the function signature to accept `font_id`:

```python
def _render_label_to_image(value, glyph_style, font_id, image_path, resolution=256):
```

and in its `else:` branch (the non-pips text-label branch), add the font assignment right after `txt_obj.data.body = label`:

```python
        label = glyph_label(value, glyph_style)
        bpy.ops.object.text_add(location=(0, 0, 0))
        txt_obj = bpy.context.active_object
        txt_obj.data.body = label
        font = _load_font(font_id, glyph_style)
        if font is not None:
            txt_obj.data.font = font
        txt_obj.data.align_x = 'CENTER'
        txt_obj.data.align_y = 'CENTER'
        txt_obj.data.size = 1.0
```

Finally, in `apply_decal_glyphs`, update the call site to pass `font_id` through:

```python
    for face_index, value in assignment.items():
        image_path = os.path.join(tmp_dir, f"{asset_id}_face{face_index}.png")
        _render_label_to_image(value, glyph_style, font_id, image_path, resolution=resolution)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/saps/projects/Dice-Detection && blender --background --python tests/blender/test_glyphs.py; echo "exit=$?"`
Expected: `ALL TESTS PASSED` and `exit=0`.

Also re-run the two other Blender-dependent suites that call into `glyphs.py`, to confirm no regressions:

Run: `cd /home/saps/projects/Dice-Detection && blender --background --python tests/blender/test_orchestrator.py; echo "exit=$?"`
Expected: `ALL TESTS PASSED` and `exit=0`.

Run: `cd /home/saps/projects/Dice-Detection && blender --background --python tests/blender/test_exporter.py; echo "exit=$?"`
Expected: `ALL TESTS PASSED` and `exit=0`.

- [ ] **Step 5: Commit**

```bash
git add src/dice_gen/glyphs.py tests/blender/test_glyphs.py
git commit -m "feat: wire font_or_style_id to real installed fonts"
```

---

### Task 4: Shallower engraving depth

**Files:**
- Modify: `src/dice_gen/glyphs.py:14` (`ENGRAVE_DEPTH_FRACTION`)
- Test: `tests/blender/test_glyphs.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: nothing consumed by another task.

- [ ] **Step 1: Write the failing test**

Add this test to `tests/blender/test_glyphs.py`, right before `run()`:

```python
def test_engrave_depth_fraction_is_shallower_than_before():
    from dice_gen import glyphs

    assert glyphs.ENGRAVE_DEPTH_FRACTION == 0.025, (
        f"expected ENGRAVE_DEPTH_FRACTION == 0.025, got "
        f"{glyphs.ENGRAVE_DEPTH_FRACTION}"
    )
```

Add the function name to `run()`, right after `test_apply_decal_glyphs_uses_load_font_with_correct_glyph_style()`:

```python
    test_engrave_depth_fraction_is_shallower_than_before()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/saps/projects/Dice-Detection && blender --background --python tests/blender/test_glyphs.py; echo "exit=$?"`
Expected: FAIL (non-zero exit) — `ENGRAVE_DEPTH_FRACTION` is still `0.04`.

- [ ] **Step 3: Implement the constant change**

In `src/dice_gen/glyphs.py`, change:

```python
ENGRAVE_DEPTH_FRACTION = 0.04
```

to:

```python
ENGRAVE_DEPTH_FRACTION = 0.025
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/saps/projects/Dice-Detection && blender --background --python tests/blender/test_glyphs.py; echo "exit=$?"`
Expected: `ALL TESTS PASSED` and `exit=0`.

- [ ] **Step 5: Commit**

```bash
git add src/dice_gen/glyphs.py tests/blender/test_glyphs.py
git commit -m "fix: reduce engraving depth for a shallower cut"
```

---

### Task 5: Regenerate the batch with a new seed and verify

**Files:** none modified — verification only.

**Interfaces:** none — this task consumes Tasks 1-4's finished code as-is.

- [ ] **Step 1: Regenerate the batch with a new seed**

Run: `cd /home/saps/projects/Dice-Detection && blender --background --python scripts/generate_dice_assets.py -- --count 100 --seed 7 --outdir data/raw/dice_assets`

Expected: completes without crashing, printing `Generated: 100, Failed: 0` (or close to it).

- [ ] **Step 2: Validate the regenerated batch**

The new run uses the same `--count 100` and the same `asset_00000`-`asset_00099` filenames as the previous seed=42 batch, so every file is overwritten in place — no leftover files from the old seed to clean up (unlike the earlier 500-vs-100 mismatch from a prior session, where the asset count itself had changed).

Run: `cd /home/saps/projects/Dice-Detection && python3 scripts/validate_dice_assets.py data/raw/dice_assets`

Expected: exits 0, or if non-zero, inspect whether the errors are new (investigate) or an unrelated known engraving warning (acceptable, out of this plan's scope).

- [ ] **Step 3: Visually confirm decal glyph visibility**

Use the Read tool to view several `_thumb.png` files from the new batch, prioritizing `printed_decal`-method assets (check `manifest.json` for `"glyph_method": "printed_decal"` records) — confirm the numeral/pip is now visible, not just a flat swatch color.

- [ ] **Step 4: Visually confirm orientation consistency**

Use the Read tool to view a few `engraved`-method thumbnails, especially d12/d20 (die types with many differently-tilted faces) — confirm adjacent visible faces' numerals no longer show the stark rotation mismatch seen before this plan.

- [ ] **Step 5: Visually confirm font variety**

Cross-reference `manifest.json` for a few assets with different `font_or_style_id` values and the same `glyph_style`, and view their thumbnails side by side — confirm the numeral shapes visibly differ (e.g. serif vs. sans vs. condensed), not identical across all three.

No commit for this task — it's a verification pass over regenerated (gitignored) data, not a code change.
