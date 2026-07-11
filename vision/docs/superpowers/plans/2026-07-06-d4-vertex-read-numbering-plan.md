# D4 Vertex-Read Numbering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Match real commercial d4 dice (standard tetrahedra, the shape this pipeline builds): show the same numeral 3 times per face — once near each corner, oriented radially outward — instead of one centered numeral, for both the engraved and printed-decal glyph paths. Pips and every other die type (d6/d8/d10/d12/d20) are unaffected.

**Architecture:** A new `_face_vertex_orientations` helper (engraved path) mirrors the existing `_face_orientation_matrix` but computes one orientation per face vertex, with each copy's "up" direction pointing radially outward toward that vertex, instead of a single global-up-hint-based orientation. `apply_engraved_glyphs` branches on `die_type == "d4" and glyph_style != "pips"` to plan 3 cuts per face instead of 1. `_render_label_to_image` (decal path) gains a `die_type` parameter and an analogous 3-corner rendering branch, using a fixed canonical equilateral-triangle layout (valid since every d4 face is congruent, and all 3 copies show the identical value).

**Tech Stack:** Same as the rest of `src/dice_gen/` — Blender 5.1.2's bundled `bpy`/`bmesh`/`mathutils`, no external pip packages. All geometry/placement values below (inset ratio, corner angles, font sizes, pixel-region boundaries for the decal test) were confirmed by running standalone scripts through `blender --background --python` during planning — see `docs/superpowers/specs/2026-07-06-d4-vertex-read-numbering-design.md` for the research and evidence.

## Global Constraints

- Only d4's **numeral** glyph styles (`arabic_numerals`, `roman_numerals`, `greek_numerals`, `cjk_numerals`) get the 3-corner treatment. `pips` on d4 is unaffected (no researched real-world vertex convention for pip-style d4s). Every other die type is unaffected regardless of glyph_style.
- The engraved path's Phase 1 (compute all cut orientations against the pristine mesh) / Phase 2 (apply cuts) split must be preserved exactly — recomputing `face.vertices`/`face.normal` against `die_obj.data.polygons` mid-loop, after an earlier cut has already rebuilt the mesh topology, causes a hard Blender crash (segfault), confirmed by hitting this exact bug during planning.
- Engraved corner inset: `0.55` (fraction of the way from face center to vertex). Engraved corner font size: `size_mm * 0.13` (vs. the existing `size_mm * 0.18` for single-centered numerals).
- Decal corner layout: angles `90, 210, 330` degrees, radius `0.5`, text size `0.42` (all confirmed via render during planning to produce three non-overlapping, correctly-outward-rotated copies within the existing camera framing).
- No change to `geometry.py`, `numbering.py`, `sampler.py`, `orchestrator.py`, `materials.py`, `exporter.py`, or the shared `_tangent_bitangent`/`_face_orientation_matrix` convention used by every other die type.

---

## File Structure

- Modify: `src/dice_gen/glyphs.py` — new `D4_CORNER_GLYPH_FONT_SIZE_FRACTION` constant and `_face_vertex_orientations` helper; `apply_engraved_glyphs`'s Phase 1 planning branches for d4 numerals; `_render_label_to_image` gains `die_type` and a d4-specific 3-corner branch; `apply_decal_glyphs`'s call site passes `die_type` through; `import math` added.
- Test: `tests/blender/test_glyphs.py` — new tests per task, inserted before `run()`.

---

### Task 1: Engraved d4 numerals cut 3 corners per face instead of 1 centered cut

**Files:**
- Modify: `src/dice_gen/glyphs.py` (new constant near `ENGRAVE_DEPTH_FRACTION`; new `_face_vertex_orientations` function; `apply_engraved_glyphs`'s Phase 1 loop and font-size setup)
- Test: `tests/blender/test_glyphs.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `_face_vertex_orientations(mesh, face, obj_matrix, inset=0.55) -> list[Matrix]` — not consumed by any other task (Task 2 is the decal-side analog, implemented independently with its own inline logic, not by calling this engrave-specific helper).

- [ ] **Step 1: Write the failing tests**

Add these four tests to `tests/blender/test_glyphs.py`, right before the `run()` function at the end of the file:

```python
def test_face_vertex_orientations_returns_one_outward_pointing_matrix_per_vertex():
    """
    Direct geometric check of _face_vertex_orientations: for a d4 face
    (triangle), it must return exactly 3 orientation matrices (one per
    vertex), each positioned between the face center and that vertex (at
    the `inset` fraction), with its bitangent (the numeral's "up"
    direction) pointing toward that same vertex -- i.e. each corner copy
    points radially outward toward its own corner, matching how real
    vertex-read d4 dice arrange their three per-face numerals.
    """
    import bpy
    from dice_gen import geometry, glyphs

    obj = geometry.build_die_base_mesh("d4", size_mm=16.0)
    face = obj.data.polygons[0]
    assert len(face.vertices) == 3, "d4 faces should be triangles"

    orientations = glyphs._face_vertex_orientations(obj.data, face, obj.matrix_world)
    assert len(orientations) == 3

    center = obj.matrix_world @ face.center
    for vertex_index, orient in zip(face.vertices, orientations):
        vertex_world = obj.matrix_world @ obj.data.vertices[vertex_index].co
        expected_radial = (vertex_world - center).normalized()

        bitangent = orient.to_3x3().col[1].normalized()
        alignment = bitangent.dot(expected_radial)
        assert alignment > 0.9, (
            f"vertex {vertex_index}: bitangent {tuple(bitangent)} should "
            f"point toward the vertex (radial {tuple(expected_radial)}), "
            f"got alignment {alignment:.3f}"
        )

        position = orient.translation
        dist_center_to_pos = (position - center).length
        dist_center_to_vertex = (vertex_world - center).length
        assert 0 < dist_center_to_pos < dist_center_to_vertex, (
            f"vertex {vertex_index}: cut position should sit strictly "
            f"between the face center and the vertex"
        )

    bpy.data.objects.remove(obj, do_unlink=True)


def test_apply_engraved_glyphs_cuts_three_corners_per_face_for_d4_numerals():
    """
    Real commercial d4 dice (standard tetrahedra, the shape this pipeline
    builds) show the same digit three times per face, once per corner,
    rather than the single centered numeral every other die type uses
    (see docs/superpowers/specs/2026-07-06-d4-vertex-read-numbering-design.md
    for the research). Confirms apply_engraved_glyphs performs 3 cuts per
    face (12 total for d4's 4 faces) for a numeral glyph_style, via a spy
    on the real _boolean_diff_apply, matching the spy technique already
    used elsewhere in this file.
    """
    import bpy
    from dice_gen import geometry, numbering, glyphs

    die_type = "d4"
    size_mm = 16.0
    obj = geometry.build_die_base_mesh(die_type, size_mm=size_mm)
    pairs = geometry.compute_opposite_face_pairs(obj)
    assignment = numbering.assign_values_to_opposite_pairs(die_type, pairs)
    assert len(assignment) == 4, "d4 should have 4 faces assigned"

    real_boolean_apply_fn = glyphs._boolean_diff_apply
    call_count = [0]

    def spy_boolean_apply(die_obj_arg, cutter_obj):
        call_count[0] += 1
        return real_boolean_apply_fn(die_obj_arg, cutter_obj)

    glyphs._boolean_diff_apply = spy_boolean_apply
    try:
        glyphs.apply_engraved_glyphs(
            obj, die_type, assignment,
            glyph_style="arabic_numerals", glyph_fill="blank",
            font_id="font_sans_bold", size_mm=size_mm,
        )
    finally:
        glyphs._boolean_diff_apply = real_boolean_apply_fn

    assert call_count[0] == 12, (
        f"expected 3 cuts per face x 4 faces = 12 total cuts for d4 "
        f"numerals, got {call_count[0]}"
    )

    bpy.data.objects.remove(obj, do_unlink=True)


def test_apply_engraved_glyphs_does_not_triple_pips_for_d4():
    """
    The vertex-read tripling only applies to numeral glyph_styles, not
    pips (no researched real-world vertex convention for pip-style d4s).
    Confirms pip cuts on a d4 are unaffected -- still one
    _boolean_diff_apply call per pip in PIP_VALUE_LAYOUTS[value], not 3x.
    """
    import bpy
    from dice_gen import geometry, numbering, glyphs

    die_type = "d4"
    size_mm = 16.0
    obj = geometry.build_die_base_mesh(die_type, size_mm=size_mm)
    pairs = geometry.compute_opposite_face_pairs(obj)
    assignment = numbering.assign_values_to_opposite_pairs(die_type, pairs)

    real_boolean_apply_fn = glyphs._boolean_diff_apply
    call_count = [0]

    def spy_boolean_apply(die_obj_arg, cutter_obj):
        call_count[0] += 1
        return real_boolean_apply_fn(die_obj_arg, cutter_obj)

    glyphs._boolean_diff_apply = spy_boolean_apply
    try:
        glyphs.apply_engraved_glyphs(
            obj, die_type, assignment,
            glyph_style="pips", glyph_fill="blank",
            font_id="font_sans_bold", size_mm=size_mm,
        )
    finally:
        glyphs._boolean_diff_apply = real_boolean_apply_fn

    expected_calls = sum(
        len(glyphs.PIP_VALUE_LAYOUTS.get(v, [(0, 0)])) for v in assignment.values()
    )
    assert call_count[0] == expected_calls, (
        f"expected {expected_calls} pip cuts (unaffected by d4 vertex-read "
        f"tripling), got {call_count[0]}"
    )

    bpy.data.objects.remove(obj, do_unlink=True)


def test_apply_engraved_glyphs_does_not_triple_numerals_for_non_d4_dice():
    """
    Regression guard: the vertex-read tripling is d4-only. A d6 with a
    numeral glyph_style must still cut exactly 1 numeral per face (6
    total), not 3 per face.
    """
    import bpy
    from dice_gen import geometry, numbering, glyphs

    die_type = "d6"
    size_mm = 16.0
    obj = geometry.build_die_base_mesh(die_type, size_mm=size_mm)
    pairs = geometry.compute_opposite_face_pairs(obj)
    assignment = numbering.assign_values_to_opposite_pairs(die_type, pairs)
    assert len(assignment) == 6

    real_boolean_apply_fn = glyphs._boolean_diff_apply
    call_count = [0]

    def spy_boolean_apply(die_obj_arg, cutter_obj):
        call_count[0] += 1
        return real_boolean_apply_fn(die_obj_arg, cutter_obj)

    glyphs._boolean_diff_apply = spy_boolean_apply
    try:
        glyphs.apply_engraved_glyphs(
            obj, die_type, assignment,
            glyph_style="arabic_numerals", glyph_fill="blank",
            font_id="font_sans_bold", size_mm=size_mm,
        )
    finally:
        glyphs._boolean_diff_apply = real_boolean_apply_fn

    assert call_count[0] == 6, (
        f"expected 1 cut per face x 6 faces = 6 for d6 (vertex-read "
        f"tripling is d4-only), got {call_count[0]}"
    )

    bpy.data.objects.remove(obj, do_unlink=True)
```

Add all four function names to `run()` at the bottom of the file, right after the last existing test call:

```python
    test_face_vertex_orientations_returns_one_outward_pointing_matrix_per_vertex()
    test_apply_engraved_glyphs_cuts_three_corners_per_face_for_d4_numerals()
    test_apply_engraved_glyphs_does_not_triple_pips_for_d4()
    test_apply_engraved_glyphs_does_not_triple_numerals_for_non_d4_dice()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/saps/projects/Dice-Detection && blender --background --python tests/blender/test_glyphs.py; echo "exit=$?"`
Expected: FAIL (non-zero exit). `test_face_vertex_orientations_returns_one_outward_pointing_matrix_per_vertex` fails with `AttributeError: module 'dice_gen.glyphs' has no attribute '_face_vertex_orientations'`. `test_apply_engraved_glyphs_cuts_three_corners_per_face_for_d4_numerals` fails with `assert 4 == 12` (still 1 cut per face). The other two tests should already pass (nothing has tripled yet) — that's fine, they're regression guards for after Step 3.

- [ ] **Step 3: Implement `_face_vertex_orientations` and wire it into `apply_engraved_glyphs`**

In `src/dice_gen/glyphs.py`, add this constant right after `ENGRAVE_DEPTH_FRACTION = 0.03`:

```python
D4_CORNER_GLYPH_FONT_SIZE_FRACTION = 0.13
```

Add this new function directly above `apply_engraved_glyphs`:

```python
def _face_vertex_orientations(mesh, face, obj_matrix, inset=0.55):
    """
    Returns one orientation matrix per vertex of `face`, for d4's
    vertex-read numeral convention: real commercial d4 dice (standard
    tetrahedra -- confirmed this is the shape geometry.py builds) show
    the same digit three times per face, once near each corner, oriented
    so whichever corner points up when the die rests on the opposite
    face reads correctly -- unlike every other die type, which uses a
    single centered numeral via _face_orientation_matrix's global
    up-hint convention.

    For each vertex, "up" (bitangent) is the direction from the face
    center toward that vertex, projected into the face plane -- i.e.
    each copy points radially outward toward its own corner, matching
    the 120-degree-apart rotational pattern real vertex-read d4s show.
    `inset` places each copy 55% of the way from the face center to the
    vertex (tested empirically: keeps the numeral clear of both the
    face center and the beveled edge).
    """
    center = obj_matrix @ face.center
    normal = (obj_matrix.to_3x3() @ face.normal).normalized()

    orientations = []
    for vertex_index in face.vertices:
        vertex_world = obj_matrix @ mesh.vertices[vertex_index].co
        radial = vertex_world - center
        radial = (radial - radial.dot(normal) * normal).normalized()
        tangent = radial.cross(normal).normalized()
        bitangent = normal.cross(tangent).normalized()
        rot = Matrix((tangent, bitangent, normal)).transposed().to_4x4()
        rot.translation = center + (vertex_world - center) * inset
        orientations.append(rot)
    return orientations
```

Then, in `apply_engraved_glyphs`, replace the first 4 lines of the function body and the Phase 1 loop. The function currently starts:

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
```

Replace it with:

```python
def apply_engraved_glyphs(die_obj, die_type, assignment, glyph_style, glyph_fill, font_id, size_mm):
    depth = size_mm * ENGRAVE_DEPTH_FRACTION
    is_d4_vertex_numerals = die_type == "d4" and glyph_style != "pips"
    glyph_font_size = (
        size_mm * D4_CORNER_GLYPH_FONT_SIZE_FRACTION if is_d4_vertex_numerals
        else size_mm * 0.18
    )
    warnings = []

    # Phase 1: compute every cut's (value, orientation) against the PRISTINE
    # mesh, entirely before any boolean modifier is applied. Each
    # bpy.ops.object.modifier_apply call below rebuilds die_obj.data's
    # topology (reindexing/reordering polygons), so face_index values from
    # `assignment` (captured once upfront by geometry.compute_opposite_face_pairs)
    # must never be re-resolved against die_obj.data.polygons after a cut.
    # Real commercial d4 dice show the same numeral 3 times per face (once
    # per corner, vertex-read) rather than the single centered numeral every
    # other die type uses -- see _face_vertex_orientations. This branch must
    # stay inside Phase 1 (computed entirely against the pristine mesh):
    # recomputing face.vertices/face.normal mid-loop, after an earlier cut
    # has already rebuilt the mesh topology, causes a Blender crash.
    planned_cuts = []
    for face_index, value in assignment.items():
        face = die_obj.data.polygons[face_index]
        if is_d4_vertex_numerals:
            for orient in _face_vertex_orientations(die_obj.data, face, die_obj.matrix_world):
                planned_cuts.append((value, orient))
        else:
            orient = _face_orientation_matrix(face, die_obj.matrix_world)
            planned_cuts.append((value, orient))
```

Everything from `# Phase 2: build and apply each cutter...` onward is unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/saps/projects/Dice-Detection && blender --background --python tests/blender/test_glyphs.py; echo "exit=$?"`
Expected: `ALL TESTS PASSED` and `exit=0`.

- [ ] **Step 5: Commit**

```bash
git add src/dice_gen/glyphs.py tests/blender/test_glyphs.py
git commit -m "feat: engrave d4 numerals at 3 corners per face, matching real dice"
```

---

### Task 2: Decal d4 numerals render 3 corner copies per face texture instead of 1 centered copy

**Files:**
- Modify: `src/dice_gen/glyphs.py` (add `import math`; `_render_label_to_image` gains `die_type` parameter and a d4 branch; `apply_decal_glyphs`'s call site)
- Test: `tests/blender/test_glyphs.py`

**Interfaces:**
- Consumes: nothing new from Task 1 (independent implementation, no shared helper — this task's 3-corner layout is a fixed canonical arrangement, not derived from `_face_vertex_orientations`).
- Produces: `_render_label_to_image(value, glyph_style, font_id, die_type, image_path, resolution=256) -> None` (signature gains `die_type` as its 4th positional parameter) — not consumed by any other task.

- [ ] **Step 1: Write the failing test**

Add this test to `tests/blender/test_glyphs.py`, right before `run()`:

```python
def test_render_label_to_image_renders_three_corner_copies_for_d4():
    """
    Real commercial d4 dice show the same digit at all three corners of
    each face (see the vertex-read design doc). For die_type == "d4" and
    a numeral glyph_style (not pips), _render_label_to_image must render
    3 copies of the label near the image's three corners instead of 1
    centered copy. Verified by checking for non-transparent ("ink")
    pixels in three separate regions of the rendered image -- near the
    top, bottom-left, and bottom-right -- and confirming there is NO ink
    in the exact center (where a single centered copy would put it),
    proving this is genuinely a 3-corner layout, not a coincidentally
    large single copy that happens to touch all these regions. Region
    boundaries were confirmed empirically during planning by rendering
    this exact layout and inspecting the raw pixel buffer.
    """
    import bpy
    import numpy as np
    from dice_gen import glyphs

    resolution = 256
    with tempfile.TemporaryDirectory() as tmp_dir:
        image_path = os.path.join(tmp_dir, "test_d4_corners.png")
        glyphs._render_label_to_image(
            3, "arabic_numerals", "font_sans_bold", "d4", image_path, resolution=resolution,
        )

        img = bpy.data.images.load(image_path)
        pixels = np.empty(resolution * resolution * 4, dtype=np.float32)
        img.pixels.foreach_get(pixels)
        alpha = pixels.reshape(resolution, resolution, 4)[:, :, 3]
        bpy.data.images.remove(img)

        def region_has_ink(y0, y1, x0, x1):
            return bool((alpha[y0:y1, x0:x1] > 0.05).any())

        assert not region_has_ink(108, 148, 108, 148), (
            "expected NO ink in the exact center -- a 3-corner layout "
            "should leave the center empty"
        )
        assert region_has_ink(190, 256, 90, 166), "expected ink near the top corner"
        assert region_has_ink(0, 90, 10, 86), "expected ink near the bottom-left corner"
        assert region_has_ink(0, 90, 170, 246), "expected ink near the bottom-right corner"
```

Add the function name to `run()`, right after `test_apply_engraved_glyphs_does_not_triple_numerals_for_non_d4_dice()`:

```python
    test_render_label_to_image_renders_three_corner_copies_for_d4()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/saps/projects/Dice-Detection && blender --background --python tests/blender/test_glyphs.py; echo "exit=$?"`
Expected: FAIL (non-zero exit) with `TypeError: _render_label_to_image() takes from 4 to 5 positional arguments but 6 were given` (or similar), since `_render_label_to_image` doesn't accept a `die_type` parameter yet.

- [ ] **Step 3: Implement the `die_type` parameter and d4 3-corner branch**

In `src/dice_gen/glyphs.py`, add `import math` at the top of the file, right after `import os`:

```python
import math
import os
```

Change `_render_label_to_image`'s signature from:

```python
def _render_label_to_image(value, glyph_style, font_id, image_path, resolution=256):
```

to:

```python
def _render_label_to_image(value, glyph_style, font_id, die_type, image_path, resolution=256):
```

Replace the function's `else:` branch (the non-pips text-label branch, currently):

```python
    else:
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
        bpy.context.collection.objects.unlink(txt_obj)
        scene.collection.objects.link(txt_obj)
        glyph_objs.append(txt_obj)
```

with:

```python
    elif die_type == "d4":
        # Real commercial d4 dice (standard tetrahedra) show the same
        # digit at all three corners of each face, oriented so whichever
        # corner is "up" reads correctly -- see the vertex-read design
        # doc. Every d4 face is a congruent equilateral triangle, so a
        # fixed canonical 3-corner layout (not tied to this face's real
        # 3D vertex positions) is sufficient: all three copies show the
        # identical value, so exact per-vertex correspondence doesn't
        # matter, only that each corner gets one correctly-outward-
        # rotated copy. Angles/radius/size confirmed via render during
        # planning to produce three non-overlapping, correctly-rotated
        # copies within this function's existing camera framing.
        label = glyph_label(value, glyph_style)
        for angle_deg in (90, 210, 330):
            angle = math.radians(angle_deg)
            ox, oy = 0.5 * math.cos(angle), 0.5 * math.sin(angle)
            bpy.ops.object.text_add(location=(ox, oy, 0))
            txt_obj = bpy.context.active_object
            txt_obj.data.body = label
            font = _load_font(font_id, glyph_style)
            if font is not None:
                txt_obj.data.font = font
            txt_obj.data.align_x = 'CENTER'
            txt_obj.data.align_y = 'CENTER'
            txt_obj.data.size = 0.42
            # Rotate so this copy's "up" points radially outward toward
            # its own corner (the top corner, angle_deg=90, needs zero
            # rotation since text already reads "up" by default).
            txt_obj.rotation_euler = (0, 0, angle - math.pi / 2)
            bpy.context.collection.objects.unlink(txt_obj)
            scene.collection.objects.link(txt_obj)
            glyph_objs.append(txt_obj)
    else:
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
        bpy.context.collection.objects.unlink(txt_obj)
        scene.collection.objects.link(txt_obj)
        glyph_objs.append(txt_obj)
```

(The existing `if glyph_style == "pips":` branch immediately above stays exactly as-is; this changes only what follows it.)

Finally, in `apply_decal_glyphs`, update the call site from:

```python
        _render_label_to_image(value, glyph_style, font_id, image_path, resolution=resolution)
```

to:

```python
        _render_label_to_image(value, glyph_style, font_id, die_type, image_path, resolution=resolution)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/saps/projects/Dice-Detection && blender --background --python tests/blender/test_glyphs.py; echo "exit=$?"`
Expected: `ALL TESTS PASSED` and `exit=0`.

Also re-run the two other Blender-dependent suites that exercise the glyph functions, to confirm no regressions:

Run: `cd /home/saps/projects/Dice-Detection && blender --background --python tests/blender/test_orchestrator.py; echo "exit=$?"`
Expected: `ALL TESTS PASSED` and `exit=0`.

Run: `cd /home/saps/projects/Dice-Detection && blender --background --python tests/blender/test_exporter.py; echo "exit=$?"`
Expected: `ALL TESTS PASSED` and `exit=0`.

- [ ] **Step 5: Commit**

```bash
git add src/dice_gen/glyphs.py tests/blender/test_glyphs.py
git commit -m "feat: render d4 decal numerals at 3 corners per face texture, matching real dice"
```

---

### Task 3: Regenerate the batch and verify

**Files:** none modified — verification only.

**Interfaces:** none — this task consumes Tasks 1-2's finished code as-is.

- [ ] **Step 1: Regenerate the batch**

Run: `cd /home/saps/projects/Dice-Detection && blender --background --python scripts/generate_dice_assets.py -- --count 100 --seed 7 --outdir data/raw/dice_assets`

Expected: completes without crashing (in particular, no segfault — this exact class of bug was hit and fixed during planning), printing `Generated: 100, Failed: 0` (or close to it).

- [ ] **Step 2: Validate the regenerated batch**

Run: `cd /home/saps/projects/Dice-Detection && python3 scripts/validate_dice_assets.py data/raw/dice_assets`

Expected: same result as before this plan (1 known `engraving_warnings` entry, unrelated to this plan's scope, is acceptable). If NEW errors appear, investigate as a regression.

- [ ] **Step 3: Visually confirm 3-corner numerals on both glyph methods**

Use the Read tool to view several d4 asset thumbnails (`data/raw/dice_assets/asset_*_thumb.png` for `die_type == "d4"` records in `manifest.json` — check both `glyph_method: engraved` and `glyph_method: printed_decal`, and confirm `glyph_style != "pips"` for the ones you check, since pips are intentionally unaffected). Confirm each visible face shows the numeral near multiple corners (not one centered number), and that pip-style d4 assets are unaffected (still one centered pip cluster).

No commit for this task — it's a verification pass over regenerated (gitignored) data, not a code change.
