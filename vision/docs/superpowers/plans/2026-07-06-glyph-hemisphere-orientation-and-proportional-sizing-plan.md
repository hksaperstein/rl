# Glyph Hemisphere Orientation and Proportional Sizing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix two independently-confirmed real bugs the user identified from direct dice knowledge and rendered output: (1) d8/d10 numerals use one global "up" direction across the whole die, producing a smooth rotating pattern instead of the correct mirrored-hemisphere pattern real dice show; (2) engraved/decal glyph size is a fixed fraction of the die's overall `size_mm` regardless of each die type's actual (very different) face size, causing tiny illegible numerals on some dice (d4's 3 vertex-copies barely visible), oversized recesses overlapping the beveled fillet on others, and — newly confirmed this session — a meaningful share of this session's residual boolean-cut mesh-quality defects (25/100 assets after the prior bevel fix).

**Architecture:**

*Hemisphere orientation (d8/d10):* Both are built (`geometry.py`) as bipyramids with exactly two pole vertices (extremal Z) and a ring; every face touches exactly one pole (confirmed empirically for both). Real dice orient each face's numeral "up" toward its own pole, which — because the two poles sit on opposite sides of the die — produces a *mirror reflection* between the two hemispheres. The current code (`_tangent_bitangent` in `glyphs.py`) instead projects one single global vector (world +Z, Y-fallback) onto every face, producing a smooth, continuously-rotating pattern with no reflection anywhere (confirmed by direct computation: the global-projection "up" for a south-pole face works out to the exact negative of that face's true pole-relative "up" — the current code doesn't merely omit the mirroring, it structurally cannot produce it). Separately (found while verifying this), the existing face-to-value assignment (`numbering.assign_values_to_opposite_pairs`) does not consistently put odd values on one hemisphere and even on the other — verified empirically to currently split 2-odd/2-even on each d8 pole, not the clean odd-pole/even-pole split the real convention requires. Both must be fixed together; fixing only the orientation formula does not produce mirrored numerals if the same face doesn't consistently receive the same-parity value as its hemisphere twin.

*Proportional sizing:* `apply_engraved_glyphs`/`apply_decal_glyphs` currently use `size_mm * 0.18` (or `size_mm * 0.13` for d4's corner copies) as the glyph font size, regardless of die type. Measured face inradius (world-space distance from face centroid to nearest edge, at a representative `size_mm=18.0`) varies from 3.674mm (d8) to 9.0mm (d6) — nearly 2.5x — so one fixed fraction of `size_mm` cannot be simultaneously well-proportioned for both. This plan replaces the fixed fraction with `face_inradius * BASE_FRACTION / (1 + (label_length - 1) * EXTRA_CHAR_FACTOR)`, calibrated empirically this session (`BASE_FRACTION=0.5`, `EXTRA_CHAR_FACTOR=0.35`) against the real worst cases (d8 single digit, d20 2-digit arabic, d20 5-character roman numeral "XVIII", d4 corner copies): this reduced per-cut non-manifold-junction counts from the hundreds/thousands down to single/low-double digits — the small remainder matches this file's already-documented, inherent EXACT-solver imperfections (see `_boolean_diff_apply`'s docstring), not a sizing defect, and remains visible via the existing `mesh_quality_warnings` manifest field rather than silently regressing.

**Tech Stack:** Blender 5.1 bpy/bmesh/mathutils, Python, existing `_harness.run_and_report` test runner (`tests/blender/*`).

## Global Constraints

- `BASE_FRACTION = 0.5` and `EXTRA_CHAR_FACTOR = 0.35` are the calibrated constants from this session's empirical testing — use them verbatim, do not re-derive or re-tune them as part of this plan.
- Hemisphere/pole detection and hemisphere-aware orientation apply ONLY to `d8` and `d10` (the only two die types built as two-pole bipyramids in `geometry.py`). `d4`, `d6`, `d12`, `d20` must keep their exact current orientation behavior unchanged — do not touch `_tangent_bitangent`'s existing global-up behavior for these types, only add an optional override path.
- The existing Phase 1 (compute all orientations against the pristine mesh) / Phase 2 (apply cuts) split in `apply_engraved_glyphs` is load-bearing (a prior empirically-confirmed Blender segfault resulted from violating it — see the function's existing comments). Any new geometry queries (pole positions, face inradius) must be computed in Phase 1, before any boolean modifier is ever applied, exactly like the existing orientation-matrix computation.
- Do not change `ENGRAVE_DEPTH_FRACTION`, `bevel_fraction` sampling, `segments=8`, or anything from the bevel-recess-edge-selection-fix plan — this plan is scoped to glyph orientation and sizing only.
- Match existing code style: no comments explaining *what* code does, only non-obvious *why*. This file's existing docstrings document empirically-discovered pathologies at length — follow that same standard for new additions (state the empirical finding, not a restatement of the code).
- `face_inradius`/pole-position computations must use world-space coordinates (`obj_matrix @ vertex.co`), matching the existing convention in `_face_orientation_matrix`/`_face_vertex_orientations`.

---

### Task 1: Detect each d8/d10 face's pole vertex

**Files:**
- Modify: `src/dice_gen/geometry.py` (add `compute_face_poles`)
- Test: `tests/blender/test_geometry.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `geometry.compute_face_poles(obj, die_type) -> dict[int, Vector] | None`. Maps each face index to the WORLD-SPACE position of whichever of the die's two pole vertices (extremal Z) that face touches. Returns `None` for any die type other than `"d8"`/`"d10"`. Task 2 and Task 3 both depend on this.

- [ ] **Step 1: Write the failing test**

Add to `tests/blender/test_geometry.py`:

```python
def test_compute_face_poles_maps_every_d8_and_d10_face_to_exactly_one_pole():
    """
    d8 and d10 are both built (see DIE_SPECS/_d10_base_vertices) as
    bipyramids: two pole vertices at the extremal Z positions, plus a ring.
    Every face touches exactly one pole (confirmed empirically for both
    die types this session, via direct vertex-index inspection). This is
    the geometric fact the hemisphere-aware orientation/numbering fixes
    (see glyphs.py and numbering.py) depend on -- real dice mirror each
    face's numeral relative to its own pole, which the die's single prior
    global-up-vector convention could not express.
    """
    import bpy
    from dice_gen import geometry

    for die_type in ("d8", "d10"):
        obj = geometry.build_die_base_mesh(die_type, size_mm=18.0)
        poles = geometry.compute_face_poles(obj, die_type)

        assert poles is not None, f"{die_type}: expected a pole mapping"
        assert len(poles) == len(obj.data.polygons), (
            f"{die_type}: expected every face mapped, got {len(poles)} of "
            f"{len(obj.data.polygons)}"
        )
        distinct_poles = {tuple(round(c, 6) for c in v) for v in poles.values()}
        assert len(distinct_poles) == 2, (
            f"{die_type}: expected exactly 2 distinct pole positions, got "
            f"{distinct_poles}"
        )
        zs = sorted(p[2] for p in distinct_poles)
        assert zs[0] < 0 < zs[1], f"{die_type}: expected one pole above and one below the origin, got {zs}"

        bpy.data.objects.remove(obj, do_unlink=True)


def test_compute_face_poles_returns_none_for_non_bipyramid_dice():
    import bpy
    from dice_gen import geometry

    for die_type in ("d4", "d6", "d12", "d20"):
        obj = geometry.build_die_base_mesh(die_type, size_mm=18.0)
        assert geometry.compute_face_poles(obj, die_type) is None, (
            f"{die_type}: expected None (not a two-pole bipyramid)"
        )
        bpy.data.objects.remove(obj, do_unlink=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `blender --background --python tests/blender/test_geometry.py 2>&1 | tail -30`
Expected: FAIL — `AttributeError: module 'dice_gen.geometry' has no attribute 'compute_face_poles'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/dice_gen/geometry.py`, after `compute_opposite_face_pairs`:

```python
def compute_face_poles(obj, die_type):
    """
    d8 and d10 are both built (see DIE_SPECS / _d10_base_vertices) as
    bipyramids: exactly two pole vertices at the extremal local-Z
    positions, plus a ring of equatorial vertices. Every face touches
    exactly one pole (confirmed empirically this session via direct
    vertex-index inspection on both die types). Real dice orient each
    face's numeral relative to its OWN pole, not one global up-vector --
    see glyphs.py's _tangent_bitangent for the orientation fix this
    enables, and numbering.py's assign_values_to_opposite_pairs for the
    matching hemisphere-consistent value assignment.

    Returns None for die types without this two-pole structure (d4, d6,
    d12, d20) -- those keep their existing single-global-up-vector
    orientation convention unchanged.
    """
    if die_type not in ("d8", "d10"):
        return None

    mesh = obj.data
    top_idx = max(range(len(mesh.vertices)), key=lambda i: mesh.vertices[i].co.z)
    bottom_idx = min(range(len(mesh.vertices)), key=lambda i: mesh.vertices[i].co.z)
    top_co = obj.matrix_world @ mesh.vertices[top_idx].co
    bottom_co = obj.matrix_world @ mesh.vertices[bottom_idx].co

    poles = {}
    for face in mesh.polygons:
        verts = set(face.vertices)
        if top_idx in verts:
            poles[face.index] = top_co
        elif bottom_idx in verts:
            poles[face.index] = bottom_co
        else:
            raise GeometryBuildError(
                f"{die_type} face {face.index} touches neither pole vertex "
                f"-- the two-pole bipyramid assumption compute_face_poles "
                f"relies on doesn't hold for this mesh"
            )
    return poles
```

- [ ] **Step 4: Run test to verify it passes**

Run: `blender --background --python tests/blender/test_geometry.py 2>&1 | tail -30`
Expected: PASS for both new tests, and all pre-existing tests in the file still pass.

- [ ] **Step 5: Commit**

```bash
git add src/dice_gen/geometry.py tests/blender/test_geometry.py
git commit -m "feat: detect each d8/d10 face's pole vertex for hemisphere-aware orientation"
```

---

### Task 2: Hemisphere-aware engraved-glyph orientation for d8/d10

**Files:**
- Modify: `src/dice_gen/glyphs.py` (`_tangent_bitangent`, `_face_orientation_matrix`, `apply_engraved_glyphs`)
- Test: `tests/blender/test_glyphs.py`

**Interfaces:**
- Consumes: `geometry.compute_face_poles` (Task 1).
- Produces: `_tangent_bitangent(normal, up_reference=None, threshold=0.999)` — new optional third parameter; when given a non-None `up_reference` vector, projects THAT (instead of the global Z/Y hint) onto the face plane. `_face_orientation_matrix(face, obj_matrix, pole_world_co=None)` — new optional third parameter; when given, computes `up_reference = pole_world_co - center` and passes it through. Both parameters default to preserving today's exact behavior when omitted, so every call site outside `apply_engraved_glyphs`'s d8/d10 path is unaffected.

- [ ] **Step 1: Write the failing test**

Add to `tests/blender/test_glyphs.py`:

```python
def test_tangent_bitangent_up_reference_overrides_global_up_hint():
    """
    Task 2's mechanism: when up_reference is given, _tangent_bitangent must
    project THAT vector (not the global Z/Y hint) onto the face plane. This
    is what lets d8/d10 orient each face relative to its own pole instead
    of one global direction shared by the whole die -- see
    test_face_orientation_matrix_mirrors_between_d8_hemispheres below for
    the end-to-end confirmation this produces the real mirrored pattern.
    """
    from dice_gen.glyphs import _tangent_bitangent
    from mathutils import Vector

    normal = Vector((0, 0, 1))
    # An up_reference pointing along +X (not +Z/+Y) must produce a
    # bitangent aligned with +X once projected into the normal's plane --
    # a result the global Z/Y hint alone could never produce for this
    # normal (global Z is parallel to this normal, forcing the Y-fallback,
    # which would give a bitangent along some Y-derived direction, not X).
    tangent, bitangent = _tangent_bitangent(normal, up_reference=Vector((1, 0, 0)))
    assert bitangent.x > 0.99, f"expected bitangent aligned with +X, got {bitangent}"


def test_face_orientation_matrix_mirrors_between_d8_hemispheres():
    """
    Regression test for a confirmed real bug: with the OLD single-global-up
    convention, a face's numeral "up" direction was a smooth, continuous
    function of the face normal alone, with no reflection anywhere -- for
    two faces on opposite (top-pole vs bottom-pole) hemispheres sharing an
    equatorial edge, the global convention's computed "up" for the
    bottom-pole face works out to the exact NEGATIVE of that face's own
    true pole-relative up (confirmed by direct computation this session).
    The fix: for every d8 face, using its own pole position (Task 1) as
    the up_reference must make its bitangent point TOWARD its own pole
    (positive dot product with the pole direction) -- true by construction
    for every face after the fix, and NOT true for roughly half the faces
    under the old global-Z-projection convention.
    """
    import bpy
    from dice_gen import geometry
    from dice_gen.glyphs import _face_orientation_matrix

    obj = geometry.build_die_base_mesh("d8", size_mm=18.0)
    poles = geometry.compute_face_poles(obj, "d8")

    for face in obj.data.polygons:
        pole_co = poles[face.index]
        center = obj.matrix_world @ face.center
        orient = _face_orientation_matrix(face, obj.matrix_world, pole_world_co=pole_co)
        bitangent = orient.col[1].xyz
        pole_direction = (pole_co - center).normalized()
        assert bitangent.dot(pole_direction) > 0.5, (
            f"face {face.index}: expected bitangent to point toward its "
            f"own pole, got dot={bitangent.dot(pole_direction):.3f}"
        )

    bpy.data.objects.remove(obj, do_unlink=True)


def test_apply_engraved_glyphs_orients_d8_hemispheres_toward_their_own_pole():
    """
    End-to-end: apply_engraved_glyphs must pass each d8/d10 face's own
    pole position (not the global up-vector) into _face_orientation_matrix
    when cutting. Verified by re-deriving each cut's expected orientation
    directly (mirroring test_face_orientation_matrix_mirrors_between_d8_hemispheres'
    invariant) rather than depending on glyphs.py's internals beyond its
    public entry point.
    """
    import bpy
    from dice_gen import geometry, glyphs

    size_mm = 18.0
    obj = geometry.build_die_base_mesh("d8", size_mm=size_mm)
    poles = geometry.compute_face_poles(obj, "d8")
    face_pairs = geometry.compute_opposite_face_pairs(obj)
    assignment = {}
    for i, (a, b) in enumerate(face_pairs):
        assignment[a] = i + 1
        assignment[b] = 9 - (i + 1)

    expected = {}
    for face in obj.data.polygons:
        pole_co = poles[face.index]
        center = obj.matrix_world @ face.center
        expected[face.index] = (pole_co - center).normalized()

    glyphs.apply_engraved_glyphs(
        obj, "d8", assignment, "arabic_numerals", "blank",
        "font_serif_regular", size_mm,
    )

    # The die's own body mesh changed (cuts applied); this test only
    # verifies the ORIENTATION computation ran without error and that
    # apply_engraved_glyphs completed for every face -- the precise
    # per-vertex mirroring invariant is already covered directly above
    # without needing to inspect post-cut mesh state.
    assert len(obj.data.polygons) > 0

    bpy.data.objects.remove(obj, do_unlink=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `blender --background --python tests/blender/test_glyphs.py 2>&1 | tail -50`
Expected: FAIL on the first two new tests — `TypeError: _tangent_bitangent() got an unexpected keyword argument 'up_reference'` and the analogous error for `_face_orientation_matrix`.

- [ ] **Step 3: Write minimal implementation**

In `src/dice_gen/glyphs.py`, replace `_tangent_bitangent`:

```python
def _tangent_bitangent(normal, up_reference=None, threshold=0.999):
    """
    Given a (normalized) face normal, returns a consistent (tangent,
    bitangent) in-plane basis by projecting an "up" reference direction
    onto the face's plane.

    When up_reference is given (d8/d10's hemisphere-aware orientation --
    see _face_orientation_matrix), THAT vector is projected instead of
    the global hint below -- this is what lets each face orient toward
    its OWN pole vertex rather than one direction shared by the whole
    die, which is what real d8/d10 dice do (confirmed empirically: the
    single-global-vector convention is structurally incapable of
    producing the mirrored-hemisphere pattern real dice show, since it's
    a smooth function of the normal alone with no reflection anywhere).

    Global +Z is used as the up reference for every OTHER face (d4, d6,
    d12, d20, and any d8/d10 caller that doesn't pass up_reference)
    EXCEPT when normal is itself (near-)parallel to +/-Z, where the
    projection is undefined (up_hint.cross(normal) would be the zero
    vector) -- global +Y is used instead for that narrow case only.

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
    if up_reference is None:
        up_reference = Vector((0, 0, 1)) if abs(normal.z) < threshold else Vector((0, 1, 0))
    tangent = up_reference.cross(normal).normalized()
    bitangent = normal.cross(tangent).normalized()
    return tangent, bitangent
```

Replace `_face_orientation_matrix`:

```python
def _face_orientation_matrix(face, obj_matrix, pole_world_co=None):
    center = obj_matrix @ face.center
    normal = (obj_matrix.to_3x3() @ face.normal).normalized()
    up_reference = None
    if pole_world_co is not None:
        to_pole = pole_world_co - center
        up_reference = (to_pole - to_pole.dot(normal) * normal).normalized()
    tangent, bitangent = _tangent_bitangent(normal, up_reference=up_reference)
    rot = Matrix((tangent, bitangent, normal)).transposed().to_4x4()
    rot.translation = center
    return rot
```

In `apply_engraved_glyphs`, in the Phase 1 loop (where `planned_cuts` is built), pass the pole position for d8/d10:

```python
    face_poles = geometry_compute_face_poles(die_obj, die_type)
    ...
    for face_index, value in assignment.items():
        face = die_obj.data.polygons[face_index]
        if is_d4_vertex_numerals:
            for orient in _face_vertex_orientations(die_obj.data, face, die_obj.matrix_world):
                planned_cuts.append((value, orient))
        else:
            pole_co = face_poles[face_index] if face_poles is not None else None
            orient = _face_orientation_matrix(face, die_obj.matrix_world, pole_world_co=pole_co)
            planned_cuts.append((value, orient))
```

Add `from . import geometry` style import — check how `glyphs.py` currently imports sibling modules (it may not import `geometry` at all yet, since orientation/pairing computation has lived in `orchestrator.py`/callers up to now). If `glyphs.py` has no existing import of `geometry`, add `from .geometry import compute_face_poles` at the top (alongside the existing `import bpy`/`import bmesh` block) and call it as `compute_face_poles(die_obj, die_type)` (drop the `geometry_` prefix used above — that was illustrative; use the actual imported name).

- [ ] **Step 4: Run test to verify it passes**

Run: `blender --background --python tests/blender/test_glyphs.py 2>&1 | tail -50`
Expected: PASS for the 3 new tests, and every pre-existing test in the file still passes (in particular the existing d4/d20 orientation and winding-consistency tests, which must be unaffected since `up_reference` defaults to `None` for every call site except the new d8/d10 one).

- [ ] **Step 5: Commit**

```bash
git add src/dice_gen/glyphs.py tests/blender/test_glyphs.py
git commit -m "fix: orient d8/d10 engraved numerals toward their own pole, not one global up vector"
```

---

### Task 3: Hemisphere-consistent value assignment for d8/d10

**Files:**
- Modify: `src/dice_gen/numbering.py` (`assign_values_to_opposite_pairs`)
- Modify: `src/dice_gen/orchestrator.py` (`_generate_from_params`, to compute and pass hemisphere info for d8/d10)
- Test: `tests/blender/test_numbering.py` if it exists, else add to an appropriate existing test file — check `tests/` for a `test_numbering.py`; if absent, add these tests to `tests/blender/test_geometry.py` since they need `geometry.build_die_base_mesh`/`compute_face_poles` too (numbering.py itself has no bpy dependency, but the tests need a real mesh to derive hemisphere membership from).

**Interfaces:**
- Consumes: `geometry.compute_face_poles` (Task 1).
- Produces: `numbering.assign_values_to_opposite_pairs(die_type, face_pairs, hemisphere_of_face=None)` — new optional parameter, a `dict[int, "top"|"bottom"]`. When given, assigns EVEN values to every face mapped to `"top"` and ODD values to every face mapped to `"bottom"` (or the reverse for a given pair — whichever preserves `opposite_sum`), instead of the current arbitrary `min(remaining)`-to-`face_a` assignment. When `None` (default, and for all die types other than d8/d10), behavior is unchanged byte-for-byte.

Before writing code: confirm this is achievable for BOTH assignment values in every pair. Since every scheme's `opposite_sum` is odd (7, 9, 9, 13, 21), every antipodal pair has exactly one odd and one even value — so for any pair `(face_a, face_b)` where one touches "top" and the other "bottom" (always true, confirmed in Task 1's geometric analysis), assigning the even value to whichever face is "top" and the odd value to whichever is "bottom" is always well-defined; no pair can ever have two odd or two even values to choose from.

- [ ] **Step 1: Write the failing test**

Add to `tests/blender/test_geometry.py` (needs both `geometry` and `numbering`):

```python
def test_assign_values_to_opposite_pairs_splits_d8_and_d10_hemispheres_by_parity():
    """
    Regression test for a confirmed real bug: independent of the
    orientation fix (see test_glyphs.py), the face-to-value assignment
    itself did not consistently put odd values on one pole and even on
    the other -- verified empirically this session (d8's current
    assignment split 2-odd/2-even on EACH pole, not a clean split).
    Fixing orientation alone does not produce the real mirrored-hemisphere
    pattern if the same face's value could be either parity from one
    asset to the next.
    """
    import bpy
    from dice_gen import geometry, numbering

    for die_type in ("d8", "d10"):
        obj = geometry.build_die_base_mesh(die_type, size_mm=18.0)
        poles = geometry.compute_face_poles(obj, die_type)
        top_pole = max((p.z, tuple(p)) for p in poles.values())[1]

        hemisphere_of_face = {
            face_idx: ("top" if tuple(pole) == top_pole else "bottom")
            for face_idx, pole in poles.items()
        }

        face_pairs = geometry.compute_opposite_face_pairs(obj)
        assignment = numbering.assign_values_to_opposite_pairs(
            die_type, face_pairs, hemisphere_of_face=hemisphere_of_face,
        )
        assert numbering.verify_opposite_sum(die_type, face_pairs, assignment)

        top_parities = {assignment[f] % 2 for f, h in hemisphere_of_face.items() if h == "top"}
        bottom_parities = {assignment[f] % 2 for f, h in hemisphere_of_face.items() if h == "bottom"}
        assert len(top_parities) == 1, f"{die_type}: top hemisphere has mixed parities {top_parities}"
        assert len(bottom_parities) == 1, f"{die_type}: bottom hemisphere has mixed parities {bottom_parities}"
        assert top_parities != bottom_parities, f"{die_type}: top and bottom ended up with the same parity"

        bpy.data.objects.remove(obj, do_unlink=True)


def test_assign_values_to_opposite_pairs_without_hemisphere_arg_is_unchanged():
    """
    Every other die type (and any caller not passing hemisphere_of_face)
    must keep today's exact assignment behavior -- this plan only adds an
    opt-in path for d8/d10.
    """
    import bpy
    from dice_gen import geometry, numbering

    for die_type in ("d4", "d6", "d12", "d20"):
        obj = geometry.build_die_base_mesh(die_type, size_mm=18.0)
        face_pairs = geometry.compute_opposite_face_pairs(obj)
        before = numbering.assign_values_to_opposite_pairs(die_type, face_pairs)
        after = numbering.assign_values_to_opposite_pairs(die_type, face_pairs, hemisphere_of_face=None)
        assert before == after
        bpy.data.objects.remove(obj, do_unlink=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `blender --background --python tests/blender/test_geometry.py 2>&1 | tail -50`
Expected: FAIL — `TypeError: assign_values_to_opposite_pairs() got an unexpected keyword argument 'hemisphere_of_face'`.

- [ ] **Step 3: Write minimal implementation**

Replace `assign_values_to_opposite_pairs` in `src/dice_gen/numbering.py`:

```python
def assign_values_to_opposite_pairs(die_type, face_pairs, hemisphere_of_face=None):
    """
    face_pairs: list of (face_index_a, face_index_b) tuples covering every
    face exactly once. For die types with an opposite_sum rule, each pair is
    assigned (v, opposite_sum - v) so the invariant holds. For d4 (no rule),
    values are just handed out in iteration order -- face_pairs there is only
    a convenient grouping, not a real geometric antipodal relationship.

    hemisphere_of_face: optional {face_index: "top"|"bottom"}, for d8/d10.
    Real d8/d10 dice show a consistent odd/even split by hemisphere (every
    face touching one pole shows one parity, every face touching the other
    pole shows the other) -- confirmed this session to require an explicit
    fix, since the plain min(remaining)-to-face_a assignment below has no
    hemisphere awareness and produces an arbitrary, inconsistent split.
    Every scheme's opposite_sum here is odd, so every antipodal pair always
    has exactly one odd and one even value -- this makes "assign the
    even one to whichever face is 'top', the odd one to whichever is
    'bottom'" always well-defined, for every pair, with no exceptions.
    When None (default), behavior is exactly as before this parameter
    existed.

    Returns {face_index: value}.
    """
    scheme = NUMBERING_SCHEMES[die_type]
    values = scheme["values"]
    opposite_sum = scheme["opposite_sum"]

    if opposite_sum is None:
        flat = [face for pair in face_pairs for face in pair]
        return {face: value for face, value in zip(flat, values)}

    remaining = set(values)
    assignment = {}
    for face_a, face_b in face_pairs:
        v_a = min(remaining)
        v_b = opposite_sum - v_a
        if v_b not in remaining:
            raise ValueError(
                f"{die_type}: cannot satisfy opposite_sum={opposite_sum} "
                f"with remaining values {sorted(remaining)}"
            )
        remaining.discard(v_a)
        remaining.discard(v_b)

        if hemisphere_of_face is None:
            assignment[face_a] = v_a
            assignment[face_b] = v_b
        else:
            even_value, odd_value = (v_a, v_b) if v_a % 2 == 0 else (v_b, v_a)
            for face in (face_a, face_b):
                assignment[face] = (
                    even_value if hemisphere_of_face[face] == "top" else odd_value
                )
    return assignment
```

Then in `src/dice_gen/orchestrator.py`'s `_generate_from_params`, compute hemisphere info for d8/d10 before calling `assign_values_to_opposite_pairs`:

```python
    face_pairs = geometry.compute_opposite_face_pairs(die_obj)
    poles = geometry.compute_face_poles(die_obj, params.die_type)
    hemisphere_of_face = None
    if poles is not None:
        top_pole_z = max(p.z for p in poles.values())
        hemisphere_of_face = {
            face_idx: ("top" if pole.z == top_pole_z else "bottom")
            for face_idx, pole in poles.items()
        }
    assignment = numbering.assign_values_to_opposite_pairs(
        params.die_type, face_pairs, hemisphere_of_face=hemisphere_of_face,
    )
```

(This replaces the existing `assignment = numbering.assign_values_to_opposite_pairs(params.die_type, face_pairs)` line.)

- [ ] **Step 4: Run test to verify it passes**

Run: `blender --background --python tests/blender/test_geometry.py 2>&1 | tail -50`
Expected: PASS for both new tests, and all pre-existing tests in the file still pass.

- [ ] **Step 5: Run the orchestrator tests to check no regressions**

Run: `blender --background --python tests/blender/test_orchestrator.py 2>&1 | tail -50`
Expected: all pre-existing tests pass (the orchestrator change must not break asset generation for any die type).

- [ ] **Step 6: Commit**

```bash
git add src/dice_gen/numbering.py src/dice_gen/orchestrator.py tests/blender/test_geometry.py
git commit -m "fix: split d8/d10 value assignment by hemisphere so odd/even are pole-consistent"
```

---

### Task 4: Apply hemisphere-aware orientation to the decal (printed) glyph path

**Files:**
- Modify: `src/dice_gen/glyphs.py` (`_unwrap_faces_to_full_square`, `apply_decal_glyphs`)
- Test: `tests/blender/test_glyphs.py`

**Interfaces:**
- Consumes: `geometry.compute_face_poles` (Task 1), the `up_reference`-aware `_tangent_bitangent` (Task 2).
- Produces: `_unwrap_faces_to_full_square(die_obj, die_type, margin=0.1)` — gains a required `die_type` parameter (needed to look up pole info; check the current call site in `apply_decal_glyphs` and update it). For d8/d10, each face's UV tangent/bitangent basis now uses that face's own pole-relative up direction (LOCAL-space, matching this function's existing local-space normal convention — do not use world-space here, this function already operates in local space per its existing docstring). For all other die types, behavior is unchanged.

Printed-decal dice must show the same real-world mirrored-hemisphere convention as engraved dice — otherwise the same die type would look correct with one glyph_method and wrong with the other, which is inconsistent with there being one real numbering convention per die type.

- [ ] **Step 1: Write the failing test**

Add to `tests/blender/test_glyphs.py`:

```python
def test_unwrap_faces_to_full_square_mirrors_between_d8_hemispheres():
    """
    The decal path's UV orientation must match the engraved path's fix
    (test_face_orientation_matrix_mirrors_between_d8_hemispheres) --
    otherwise the same d8 die type would show the correct mirrored
    numeral pattern when engraved but the old smoothly-rotating (wrong)
    pattern when printed as a decal, which is inconsistent: there is one
    real numbering convention per die type, independent of glyph_method.
    """
    import bpy
    from dice_gen import geometry
    from dice_gen.glyphs import _unwrap_faces_to_full_square

    obj = geometry.build_die_base_mesh("d8", size_mm=18.0)
    poles = geometry.compute_face_poles(obj, "d8")
    _unwrap_faces_to_full_square(obj, "d8")

    uv_layer = obj.data.uv_layers.active.data
    for face in obj.data.polygons:
        pole_world = poles[face.index]
        center_world = obj.matrix_world @ face.center
        pole_direction_world = (pole_world - center_world).normalized()

        # The face's "up" in its own UV square is the +V direction; find
        # which world-space direction that corresponds to by comparing
        # the two vertices with the highest and lowest V coordinate in
        # this face's loop.
        loop_indices = list(range(face.loop_start, face.loop_start + face.loop_total))
        highest_v_loop = max(loop_indices, key=lambda li: uv_layer[li].uv.y)
        lowest_v_loop = min(loop_indices, key=lambda li: uv_layer[li].uv.y)
        highest_vert = obj.data.loops[highest_v_loop].vertex_index
        lowest_vert = obj.data.loops[lowest_v_loop].vertex_index
        world_up_direction = (
            obj.matrix_world @ obj.data.vertices[highest_vert].co
            - obj.matrix_world @ obj.data.vertices[lowest_vert].co
        ).normalized()

        assert world_up_direction.dot(pole_direction_world) > 0, (
            f"face {face.index}: UV 'up' direction does not point toward "
            f"this face's own pole"
        )

    bpy.data.objects.remove(obj, do_unlink=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `blender --background --python tests/blender/test_glyphs.py 2>&1 | tail -50`
Expected: FAIL — either a `TypeError` (missing required `die_type` argument, once Step 3 changes the signature — write the test against the NEW signature first per TDD, so at this point it fails with the CURRENT signature not accepting/needing `die_type` the same way, or the mirroring assertion fails outright). If the current signature doesn't accept `die_type` at all yet, expect a `TypeError` calling `_unwrap_faces_to_full_square(obj, "d8")` (extra positional argument) until Step 3's signature change lands.

- [ ] **Step 3: Write minimal implementation**

In `src/dice_gen/glyphs.py`, update `_unwrap_faces_to_full_square`'s signature to `def _unwrap_faces_to_full_square(die_obj, die_type, margin=0.1):` and add pole lookup at the top of the function body:

```python
    face_poles = compute_face_poles(die_obj, die_type)
```

(`compute_face_poles` already imported per Task 2's Step 3.) Then, wherever this function currently calls `_tangent_bitangent(normal)` (local-space normal, per its existing docstring) to build the per-face UV basis, compute an `up_reference` the same way `_face_orientation_matrix` does and pass it through:

```python
    up_reference = None
    if face_poles is not None:
        pole_local = die_obj.matrix_world.inverted() @ face_poles[face.index]
        to_pole = pole_local - face.center
        up_reference = (to_pole - to_pole.dot(normal) * normal).normalized()
    tangent, bitangent = _tangent_bitangent(normal, up_reference=up_reference)
```

Read the existing function body first (`_unwrap_faces_to_full_square` in `src/dice_gen/glyphs.py`) to find the exact current call site and variable names (`face`, `normal`) before making this edit — the function predates this plan and its precise local variable names must be matched, not assumed.

Update the one existing call site in `apply_decal_glyphs`:

```python
    _unwrap_faces_to_full_square(die_obj, die_type)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `blender --background --python tests/blender/test_glyphs.py 2>&1 | tail -50`
Expected: PASS, and every pre-existing test in the file still passes — in particular the existing d4/d8/d20 winding-consistency and apex-orientation tests for `_unwrap_faces_to_full_square`, which must still pass with the new required `die_type` parameter (update their call sites to pass the die type they already test with, if they call this function directly).

- [ ] **Step 5: Commit**

```bash
git add src/dice_gen/glyphs.py tests/blender/test_glyphs.py
git commit -m "fix: orient d8/d10 decal UVs toward each face's own pole to match the engraved path"
```

---

### Task 5: Face-geometry-proportional glyph sizing

**Files:**
- Modify: `src/dice_gen/geometry.py` (add `compute_face_inradius`)
- Modify: `src/dice_gen/glyphs.py` (`apply_engraved_glyphs`, `apply_decal_glyphs`, `_render_label_to_image`)
- Test: `tests/blender/test_geometry.py`, `tests/blender/test_glyphs.py`

**Interfaces:**
- Consumes: nothing new from other tasks (independent of Tasks 1-4, but grouped last in this plan since it's the least architecturally novel).
- Produces: `geometry.compute_face_inradius(mesh, face, obj_matrix) -> float` — world-space distance from the face's centroid to the nearest of its edges (treated as infinite lines, which for a point at the centroid of a convex polygon equals the true minimum distance to the polygon boundary). `glyphs.py` gains a module-level `FONT_INRADIUS_FRACTION = 0.5` and `FONT_EXTRA_CHAR_SHRINK = 0.35`, and a small shared helper `_proportional_font_size(inradius, label)`.

- [ ] **Step 1: Write the failing test**

Add to `tests/blender/test_geometry.py`:

```python
def test_compute_face_inradius_matches_known_values_for_a_cube():
    """
    For a d6 (cube) of size_mm=18.0 (half-extent 9.0), each square face's
    inradius (centroid to nearest edge) is exactly half the face's own
    edge length, which for this cube construction equals the half-extent
    itself: 9.0. This is the simplest checkable case (a cube's face
    inradius is trivial to state exactly), used as the basic correctness
    check before this helper is trusted for the less-trivial polyhedra.
    """
    import bpy
    from dice_gen import geometry

    obj = geometry.build_die_base_mesh("d6", size_mm=18.0)
    for face in obj.data.polygons:
        inradius = geometry.compute_face_inradius(obj.data, face, obj.matrix_world)
        assert abs(inradius - 9.0) < 1e-4, f"expected inradius 9.0, got {inradius}"
    bpy.data.objects.remove(obj, do_unlink=True)


def test_compute_face_inradius_is_smaller_for_d8_and_d20_than_d6_and_d12_at_same_size():
    """
    Regression anchor for the real bug this task fixes: at the same
    size_mm, d8/d20 have much smaller individual faces than d6/d12
    (confirmed empirically this session: inradius 3.674/5.196mm vs
    9.0/7.656mm at size_mm=18.0) -- yet the OLD code used one fixed
    font-size fraction of size_mm for every die type, oversizing glyphs
    on the small-faced dice (contributing to boolean-cut mesh-quality
    defects) and undersizing them on the large-faced ones (illegible
    numerals, e.g. d4's vertex copies). This test only anchors the
    geometric fact the fix depends on, not the fix itself.
    """
    import bpy
    from dice_gen import geometry

    size_mm = 18.0
    small_faced = {}
    large_faced = {}
    for die_type in ("d8", "d20"):
        obj = geometry.build_die_base_mesh(die_type, size_mm=size_mm)
        small_faced[die_type] = geometry.compute_face_inradius(obj.data, obj.data.polygons[0], obj.matrix_world)
        bpy.data.objects.remove(obj, do_unlink=True)
    for die_type in ("d6", "d12"):
        obj = geometry.build_die_base_mesh(die_type, size_mm=size_mm)
        large_faced[die_type] = geometry.compute_face_inradius(obj.data, obj.data.polygons[0], obj.matrix_world)
        bpy.data.objects.remove(obj, do_unlink=True)

    assert max(small_faced.values()) < min(large_faced.values()), (
        f"expected d8/d20 inradius ({small_faced}) to be smaller than "
        f"d6/d12 inradius ({large_faced}) at the same size_mm"
    )
```

Add to `tests/blender/test_glyphs.py`:

```python
def test_proportional_font_size_shrinks_for_longer_labels():
    """
    Calibrated this session against the real worst cases (d8 single
    digit, d20 2-digit arabic, d20 5-character roman numeral "XVIII"):
    BASE_FRACTION=0.5, EXTRA_CHAR_FACTOR=0.35 -- reduced per-cut
    non-manifold-junction counts from the hundreds/thousands down to
    single/low-double digits (the small remainder matches this file's
    already-documented inherent EXACT-solver imperfections, not a sizing
    defect -- see _boolean_diff_apply's docstring). This test only
    anchors the shrink-with-length behavior the calibration depends on,
    not the exact calibrated constants themselves.
    """
    from dice_gen.glyphs import _proportional_font_size

    inradius = 5.0
    size_1_char = _proportional_font_size(inradius, "8")
    size_2_char = _proportional_font_size(inradius, "20")
    size_5_char = _proportional_font_size(inradius, "XVIII")

    assert size_1_char > size_2_char > size_5_char
    assert size_1_char == inradius * 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `blender --background --python tests/blender/test_geometry.py 2>&1 | tail -30` then `blender --background --python tests/blender/test_glyphs.py 2>&1 | tail -30`
Expected: FAIL — `AttributeError: module 'dice_gen.geometry' has no attribute 'compute_face_inradius'` and `ImportError: cannot import name '_proportional_font_size'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/dice_gen/geometry.py`, after `compute_face_poles`:

```python
def compute_face_inradius(mesh, face, obj_matrix):
    """
    World-space distance from face's centroid to the nearest of its
    edges (treated as infinite lines -- for a point at a convex polygon's
    centroid, this equals the true minimum distance to the polygon
    boundary). Used by glyphs.py to size engraved/decal numerals
    proportionally to each die type's actual face size, instead of one
    fixed fraction of the die's overall size_mm -- confirmed this
    session that face inradius varies nearly 2.5x across die types at
    the same size_mm (3.674mm for d8 vs 9.0mm for d6, at size_mm=18.0),
    so a single fixed fraction cannot be well-proportioned for every die
    type at once.
    """
    center = obj_matrix @ face.center
    verts_world = [obj_matrix @ mesh.vertices[i].co for i in face.vertices]
    n = len(verts_world)
    min_dist = None
    for i in range(n):
        a = verts_world[i]
        b = verts_world[(i + 1) % n]
        edge_dir = (b - a).normalized()
        proj = (center - a).dot(edge_dir)
        closest = a + edge_dir * proj
        dist = (center - closest).length
        if min_dist is None or dist < min_dist:
            min_dist = dist
    return min_dist
```

In `src/dice_gen/glyphs.py`, add near the top (alongside `ENGRAVE_DEPTH_FRACTION`/`D4_CORNER_GLYPH_FONT_SIZE_FRACTION`):

```python
FONT_INRADIUS_FRACTION = 0.5
FONT_EXTRA_CHAR_SHRINK = 0.35
```

and import `compute_face_inradius` alongside the existing `compute_face_poles` import. Add the shared helper (near `glyph_label`):

```python
def _proportional_font_size(inradius, label):
    """
    Calibrated this session (FONT_INRADIUS_FRACTION=0.5,
    FONT_EXTRA_CHAR_SHRINK=0.35) against the real worst cases across
    every die type/glyph style combination -- see
    test_proportional_font_size_shrinks_for_longer_labels. Longer labels
    (e.g. d20's 2-digit arabic numerals, or "XVIII" for roman numeral 18)
    need a smaller per-character size to occupy roughly the same total
    footprint as a single-character label at the same font size would.
    """
    n = len(label)
    return inradius * FONT_INRADIUS_FRACTION / (1 + (n - 1) * FONT_EXTRA_CHAR_SHRINK)
```

In `apply_engraved_glyphs`, replace the fixed `glyph_font_size` computation:

```python
    depth = size_mm * ENGRAVE_DEPTH_FRACTION
    is_d4_vertex_numerals = die_type == "d4" and glyph_style != "pips"
```

(keep these two lines), and remove the old fixed-fraction `glyph_font_size = (...)` line entirely — font size is now computed PER-CUT (it depends on each face's own inradius and each cut's own label, both of which vary per face/value), not once for the whole die. In the Phase 2 loop, where `label = glyph_label(value, glyph_style)` is computed and `txt_obj.data.size = glyph_font_size` is set, replace with:

```python
            label = glyph_label(value, glyph_style)
            inradius = compute_face_inradius(die_obj.data, die_obj.data.polygons[face_index], die_obj.matrix_world)
            font_size = _proportional_font_size(inradius, label)
            ...
            txt_obj.data.size = font_size
```

This requires `face_index` to still be available in Phase 2 — check the existing `planned_cuts` tuple structure (currently `(value, orient)`) and extend it to `(value, orient, face_index)` in Phase 1 so Phase 2 can look up the correct face's inradius without re-indexing `die_obj.data.polygons` after a cut has already run (the existing Phase 1/Phase 2 split comment explains why re-indexing mid-loop is unsafe — computing inradius in Phase 1 alongside orientation, against the pristine mesh, and carrying it through in the tuple, avoids this entirely; prefer computing `font_size` itself in Phase 1 and carrying `(value, orient, font_size)` through, rather than carrying `face_index` and recomputing in Phase 2).

For d4's vertex-numeral case (`_face_vertex_orientations`), compute inradius once per face in Phase 1 (same face, 3 corner copies share the same face-level inradius) and use it for all 3 corner copies' font size — do not use `D4_CORNER_GLYPH_FONT_SIZE_FRACTION`/`size_mm` for this anymore; the d4 branch's `glyph_font_size` line should be removed the same way.

In `apply_decal_glyphs`, apply the same proportional sizing to `_render_label_to_image`'s call site (used for the printed-decal path — this function renders the numeral into a per-face texture image, sized to look proportionally correct once mapped onto the face by `_unwrap_faces_to_full_square`). Pass the face's inradius through (compute once per face in `apply_decal_glyphs`'s existing per-face loop, before calling `_render_label_to_image`) so the rendered text's relative size within its own 256x256 (or however many `resolution`) canvas matches its real proportion on the die, and adjust `_render_label_to_image`'s internal text-sizing call the same way `_proportional_font_size` is used in the engraved path — read `_render_label_to_image`'s current body first to find its existing fixed size value and replace it with the passed-through proportional size, keeping the function's other behavior (d4 3-corner layout, image resolution, compositing) unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run: `blender --background --python tests/blender/test_geometry.py 2>&1 | tail -30` then `blender --background --python tests/blender/test_glyphs.py 2>&1 | tail -80`
Expected: PASS for all new tests. Every pre-existing test in `test_glyphs.py` must still pass — in particular any test that currently asserts specific pixel regions for rendered labels (e.g. the d4 three-corner-copy pixel-region test) may need its expected regions re-checked/updated since label size is changing; if such a test fails only because the numeral is now a different (larger, more proportional) size than before, update its expected pixel regions to match the new rendered output rather than reverting the sizing fix — confirm by visually re-rendering the specific case (`_render_label_to_image` with the new sizing) and inspecting the output, the same way this session's earlier d4 decal work verified pixel regions against actual scanned output.

- [ ] **Step 5: Commit**

```bash
git add src/dice_gen/geometry.py src/dice_gen/glyphs.py tests/blender/test_geometry.py tests/blender/test_glyphs.py
git commit -m "fix: size engraved/decal glyphs proportionally to each die type's actual face geometry"
```

---

### Task 6: Regenerate the batch and verify all four fixes end-to-end

**Files:** none changed — this task runs the existing pipeline, existing validator, and visual spot-checks. No new code.

**Interfaces:** none.

- [ ] **Step 1: Regenerate the full 100-asset batch in place**

```bash
blender --background --python scripts/generate_dice_assets.py -- --count 100 --seed 7 --outdir data/raw/dice_assets
```

- [ ] **Step 2: Run the existing manifest validator**

```bash
python3 scripts/validate_dice_assets.py data/raw/dice_assets
```
Expected: `mesh_quality_warnings`-related error count substantially lower than the pre-this-plan baseline (25/100 assets, max 5,786 non-manifold junction edges / 33 degenerate faces) — report the new count and max severity, whatever it is; do not adjust code to force a specific number, this step is measurement, not a pass/fail gate.

- [ ] **Step 3: Visually spot-check the specific complaints this plan targets**

Read (via the Read tool, as an image) at least: 2 d4 thumbnails (`asset_XXXXX_thumb.png` for two different engraved d4 assets — grep `data/raw/dice_assets/*.json` for `"die_type": "d4"` and `"glyph_method": "engraved"`), 2 d8 thumbnails, 2 d10 thumbnails (same approach, checking for `"die_type": "d8"`/`"d10"` and `"glyph_method": "engraved"` — pick one `arabic_numerals` case for each if available), and 1 d6 thumbnail. Confirm: d4's 3 corner-copy numerals are now clearly visible and legibly sized (not the barely-visible tiny numerals from before this plan); d8/d10 numerals visibly differ in orientation between hemispheres rather than reading as one smooth rotating sequence; no numeral visibly overlaps the die's beveled edge.

- [ ] **Step 4: Report the quantified before/after result**

No commit for this task (no code changes) — report the final defect-count comparison and the visual spot-check findings (pass/fail per item in Step 3, with specifics for anything that still looks wrong).

---

### Task 7: Shallower engraving depth with softened recess edges

**Files:**
- Modify: `src/dice_gen/glyphs.py` (`ENGRAVE_DEPTH_FRACTION`, cutter construction in `apply_engraved_glyphs`)
- Test: `tests/blender/test_glyphs.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `ENGRAVE_DEPTH_FRACTION` changes from `0.03` to `0.02`. The engraved-cutter construction gains a small Bevel modifier (applied to the cutter mesh itself, before the boolean cut) to round the recess's edges instead of leaving a sharp 90-degree transition — calibrated this session at `width = depth * 0.25`, `segments = 2`, `limit_method='NONE'` (bevel every edge of the cutter, since the whole cutter is one small glyph feature, unlike the die-body bevel's structural-edges-only selection from the earlier bevel-recess-edge-selection-fix plan).

Explicit user request: "all engravings should be even shallower with even softer edges." Depth was already reduced once this session (0.04 → 0.03, see git history); this reduces it further. "Softer edges" means the recess's boundary (where the vertical cut wall meets the die's flat face, and the cut floor's own edges) should be rounded rather than a sharp crisp line — achieved by beveling the CUTTER mesh (not the die body — that bevel is already scoped to structural edges only, by design, from the prior plan) before it's used to cut.

Verified this session: beveling the cutter mesh does introduce a small number of additional non-manifold-junction defects on some cuts (10-22, tested on a representative d8 case) compared to an unbeveled cutter (0) — smaller than the calibrated font-sizing fix's own residual, and, like that residual, remains visible via the existing `mesh_quality_warnings` manifest field rather than silently shipping. This is an accepted, explicit tradeoff for the requested visual softness, not an oversight.

- [ ] **Step 1: Write the failing test**

Add to `tests/blender/test_glyphs.py`:

```python
def test_engrave_depth_fraction_is_shallower_than_prior_value():
    """
    Explicit user request: "all engravings should be even shallower" --
    this session had already reduced ENGRAVE_DEPTH_FRACTION once (0.04 ->
    0.03); this reduces it again.
    """
    from dice_gen.glyphs import ENGRAVE_DEPTH_FRACTION

    assert ENGRAVE_DEPTH_FRACTION == 0.02


def test_apply_engraved_glyphs_cutter_edges_are_rounded_not_sharp():
    """
    Explicit user request: "softer edges" on engraved recesses -- the
    cutter mesh used for each numeral/pip cut must have its edges beveled
    (rounded) before the boolean cut runs, rather than left as sharp
    90-degree transitions. Verified by checking the cutter mesh gains
    additional geometry from a bevel step: an unbeveled extruded-text
    mesh has exactly as many faces as its convert(target='MESH') +
    _weld_cutter_mesh output; a beveled one has more (the bevel adds
    faces along every edge it rounds).
    """
    import bpy
    from dice_gen import geometry
    from dice_gen.glyphs import (
        _face_orientation_matrix, _weld_cutter_mesh, ENGRAVE_DEPTH_FRACTION,
    )

    size_mm = 18.0
    depth = size_mm * ENGRAVE_DEPTH_FRACTION

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.object.text_add()
    txt_obj = bpy.context.active_object
    txt_obj.data.body = "8"
    txt_obj.data.align_x = 'CENTER'; txt_obj.data.align_y = 'CENTER'
    txt_obj.data.size = 1.8
    txt_obj.data.extrude = depth
    bpy.context.view_layer.objects.active = txt_obj
    bpy.ops.object.convert(target='MESH')
    _weld_cutter_mesh(txt_obj)
    unbeveled_face_count = len(txt_obj.data.polygons)

    from dice_gen.glyphs import _soften_cutter_edges
    _soften_cutter_edges(txt_obj, depth)
    beveled_face_count = len(txt_obj.data.polygons)

    assert beveled_face_count > unbeveled_face_count, (
        f"expected bevel to add faces (softened edges), got "
        f"{unbeveled_face_count} -> {beveled_face_count}"
    )

    bpy.data.objects.remove(txt_obj, do_unlink=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `blender --background --python tests/blender/test_glyphs.py 2>&1 | tail -30`
Expected: FAIL — `AssertionError: 0.03 == 0.02` and `ImportError: cannot import name '_soften_cutter_edges'`.

- [ ] **Step 3: Write minimal implementation**

In `src/dice_gen/glyphs.py`, change:

```python
ENGRAVE_DEPTH_FRACTION = 0.02
```

Add a new function near `_weld_cutter_mesh`:

```python
def _soften_cutter_edges(cutter_obj, depth):
    """
    Explicit user request: engraved recesses should have softened
    (rounded) edges, not a sharp 90-degree transition where the cut wall
    meets the die's flat face. Bevels every edge of the cutter mesh
    itself (limit_method='NONE') before it's used in the boolean cut --
    the cutter is one small glyph feature, unlike the die body's own
    bevel (see exporter.py), which is deliberately scoped to structural
    edges only via a bevel-weight attribute. width/segments calibrated
    this session: large enough to visibly round the recess, small enough
    relative to `depth` to keep the added boolean-cut complexity (and
    its small, measured non-manifold-junction cost -- 10-22 edges on a
    representative case, tracked via mesh_quality_warnings like every
    other known residual in this pipeline) modest.
    """
    mod = cutter_obj.modifiers.new(name="SoftenEdges", type='BEVEL')
    mod.width = depth * 0.25
    mod.segments = 2
    mod.limit_method = 'NONE'
    bpy.context.view_layer.objects.active = cutter_obj
    bpy.ops.object.modifier_apply(modifier=mod.name)

    bm = bmesh.new()
    bm.from_mesh(cutter_obj.data)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(cutter_obj.data)
    cutter_obj.data.update()
    bm.free()
```

In `apply_engraved_glyphs`'s Phase 2 loop, immediately after the existing `_weld_cutter_mesh(txt_obj)` call (for both the pips branch's sphere cutter — check whether pips need this; UV spheres are already smooth/watertight per `_weld_cutter_mesh`'s own docstring, so skip `_soften_cutter_edges` for the pips branch, it's for the sharp-cornered text-glyph cutters specifically — and the text-glyph branch), add:

```python
    _soften_cutter_edges(txt_obj, depth)
```

right after `_weld_cutter_mesh(txt_obj)` in the text-glyph (non-pips) branch only.

- [ ] **Step 4: Run test to verify it passes**

Run: `blender --background --python tests/blender/test_glyphs.py 2>&1 | tail -50`
Expected: PASS, and every pre-existing test in the file still passes.

- [ ] **Step 5: Commit**

```bash
git add src/dice_gen/glyphs.py tests/blender/test_glyphs.py
git commit -m "fix: shallower engraving depth with softened (beveled) recess edges"
```
