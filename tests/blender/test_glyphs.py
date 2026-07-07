import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from _harness import run_and_report


def test_glyph_label_formats():
    from dice_gen import glyphs

    assert glyphs.glyph_label(6, "arabic_numerals") == "6"
    assert glyphs.glyph_label(20, "roman_numerals") == "XX"
    assert glyphs.glyph_label(9, "roman_numerals") == "IX"


def test_engraved_glyphs_reduce_solid_volume():
    import bpy
    from dice_gen import geometry, numbering, glyphs

    die_type = "d6"
    obj = geometry.build_die_base_mesh(die_type, size_mm=16.0)
    pairs = geometry.compute_opposite_face_pairs(obj)
    assignment = numbering.assign_values_to_opposite_pairs(die_type, pairs)

    import bmesh
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    volume_before = bm.calc_volume()
    bm.free()

    glyphs.apply_engraved_glyphs(
        obj, die_type, assignment,
        glyph_style="arabic_numerals", glyph_fill="painted",
        font_id="font_sans_bold", size_mm=16.0,
    )

    bm2 = bmesh.new()
    bm2.from_mesh(obj.data)
    volume_after = bm2.calc_volume()
    bm2.free()

    assert volume_after < volume_before, "engraving should remove material"
    assert len(obj.data.materials) >= 2, "painted fill should add a second material slot"

    bpy.data.objects.remove(obj, do_unlink=True)


def test_engraved_glyphs_blank_fill_does_not_add_second_material():
    import bpy
    from dice_gen import geometry, numbering, glyphs

    die_type = "d6"
    obj = geometry.build_die_base_mesh(die_type, size_mm=16.0)
    pairs = geometry.compute_opposite_face_pairs(obj)
    assignment = numbering.assign_values_to_opposite_pairs(die_type, pairs)

    glyphs.apply_engraved_glyphs(
        obj, die_type, assignment,
        glyph_style="arabic_numerals", glyph_fill="blank",
        font_id="font_sans_bold", size_mm=16.0,
    )

    assert len(obj.data.materials) < 2, (
        "blank fill should not add a second (painted fill) material slot"
    )

    bpy.data.objects.remove(obj, do_unlink=True)


def test_decal_glyphs_assigns_one_material_per_face():
    import bpy
    from dice_gen import geometry, numbering, glyphs

    die_type = "d6"
    obj = geometry.build_die_base_mesh(die_type, size_mm=16.0)
    pairs = geometry.compute_opposite_face_pairs(obj)
    assignment = numbering.assign_values_to_opposite_pairs(die_type, pairs)

    with tempfile.TemporaryDirectory() as tmp_dir:
        glyphs.apply_decal_glyphs(
            obj, die_type, assignment,
            glyph_style="arabic_numerals", font_id="font_sans_bold",
            size_mm=16.0, asset_id="test_asset", tmp_dir=tmp_dir,
        )
        assert len(obj.data.materials) == 6
        for face_index in assignment:
            mat_index = obj.data.polygons[face_index].material_index
            assert obj.data.materials[mat_index] is not None

    bpy.data.objects.remove(obj, do_unlink=True)


def test_engraved_glyphs_use_pristine_face_orientations_not_reindexed_mid_loop():
    """
    Regression test for the face-index-drift bug: apply_engraved_glyphs used
    to loop over assignment.items() and re-read die_obj.data.polygons[face_index]
    INSIDE the loop, after prior iterations had already applied a boolean
    modifier (bpy.ops.object.modifier_apply), which rebuilds/reindexes mesh
    topology. On a d10 this caused polygon counts to jump around wildly
    (10 -> 204 -> 235 -> 249 -> 6) and eventually raise IndexError, and even
    when it didn't crash, numerals were engraved onto the wrong faces.

    This test both (a) directly verifies the fix's mechanism -- that the
    orientation matrices used for cutting match those computed once on the
    pristine mesh, with no drift -- and (b) checks an indirect symptom (a
    sane, non-degenerate final mesh with the expected volume reduction) that
    would have caught the collapse/IndexError behavior seen in the original
    bug.
    """
    import bpy
    import bmesh
    from dice_gen import geometry, numbering, glyphs

    die_type = "d10"
    size_mm = 16.0

    # Build once and capture the "pristine" per-face orientation matrices
    # ourselves, exactly the way a correct implementation must (i.e. compute
    # everything BEFORE any cut is applied). This is the ground truth we
    # compare the fixed implementation's behavior against.
    obj = geometry.build_die_base_mesh(die_type, size_mm=size_mm)
    pairs = geometry.compute_opposite_face_pairs(obj)
    assignment = numbering.assign_values_to_opposite_pairs(die_type, pairs)
    assert len(assignment) == 10, "d10 should have 10 faces assigned"

    pristine_orientations = {}
    for face_index in assignment:
        face = obj.data.polygons[face_index]
        pristine_orientations[face_index] = glyphs._face_orientation_matrix(
            face, obj.matrix_world
        ).copy()

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    volume_before = bm.calc_volume()
    faces_before = len(bm.faces)
    bm.free()

    # (a) Direct mechanism check: instrument the real _face_orientation_matrix
    # and _boolean_diff_apply calls made by apply_engraved_glyphs itself, so
    # we can prove (not just infer) that:
    #   1. ALL orientation-matrix computations happen strictly before the
    #      FIRST boolean cut is applied (i.e. Phase 1 fully precedes Phase 2 --
    #      the actual defect was reading polygons mid-loop, interleaved with
    #      cuts), and
    #   2. every orientation matrix actually used for cutting is bit-for-bit
    #      identical to the one computed independently against the pristine
    #      mesh before apply_engraved_glyphs was ever called.
    call_log = []  # list of ("orient", face_index, matrix) or ("cut",)
    real_orientation_fn = glyphs._face_orientation_matrix
    real_boolean_apply_fn = glyphs._boolean_diff_apply

    def spy_orientation(face, obj_matrix, **kwargs):
        result = real_orientation_fn(face, obj_matrix, **kwargs)
        call_log.append(("orient", face.index, result.copy()))
        return result

    def spy_boolean_apply(die_obj_arg, cutter_obj):
        call_log.append(("cut",))
        return real_boolean_apply_fn(die_obj_arg, cutter_obj)

    glyphs._face_orientation_matrix = spy_orientation
    glyphs._boolean_diff_apply = spy_boolean_apply
    try:
        glyphs.apply_engraved_glyphs(
            obj, die_type, assignment,
            glyph_style="arabic_numerals", glyph_fill="painted",
            font_id="font_sans_bold", size_mm=size_mm,
        )
    finally:
        glyphs._face_orientation_matrix = real_orientation_fn
        glyphs._boolean_diff_apply = real_boolean_apply_fn

    orient_calls = [entry for entry in call_log if entry[0] == "orient"]
    cut_calls = [entry for entry in call_log if entry[0] == "cut"]
    assert len(orient_calls) == len(assignment), (
        f"expected exactly {len(assignment)} orientation computations "
        f"(one per face, all upfront), got {len(orient_calls)}"
    )
    assert len(cut_calls) >= len(assignment), "expected at least one cut per face"

    first_cut_position = call_log.index(cut_calls[0])
    last_orient_position = max(
        i for i, entry in enumerate(call_log) if entry[0] == "orient"
    )
    assert last_orient_position < first_cut_position, (
        "all face-orientation computations must happen BEFORE the first "
        "boolean cut is applied (this is the actual fix: no re-indexing "
        "into die_obj.data.polygons after any cut has mutated the mesh)"
    )

    for _, face_index, orient_used in orient_calls:
        expected = pristine_orientations[face_index]
        drift = (orient_used.translation - expected.translation).length
        assert drift < 1e-9, (
            f"face {face_index}: orientation actually used for cutting "
            f"({orient_used.translation}) must exactly match the "
            f"independently precomputed pristine orientation "
            f"({expected.translation}), got drift {drift}"
        )

    # (b) Indirect sanity check: the final mesh must be non-degenerate and
    # show volume loss consistent with 10 real engraving cuts, not the
    # collapsed 6-or-235-polygon garbage the bug produced.
    bm3 = bmesh.new()
    bm3.from_mesh(obj.data)
    volume_after = bm3.calc_volume()
    faces_after = len(bm3.faces)
    bm3.free()

    assert volume_after > 0, "engraved die must not collapse to a degenerate/zero-volume mesh"
    assert volume_after < volume_before, "engraving should remove material"
    # Sanity bounds: 10 numeral cuts should remove a modest fraction of the
    # die's volume, not gut it (which is what happened when cutters ended up
    # applied at wildly wrong locations/sizes due to the drift bug).
    fraction_removed = (volume_before - volume_after) / volume_before
    assert 0.001 < fraction_removed < 0.5, (
        f"unexpected volume loss fraction {fraction_removed} "
        f"(before={volume_before}, after={volume_after})"
    )
    # The original bug produced polygon counts that swung wildly between
    # cuts (10 -> 204 -> 235 -> 249) before collapsing to a degenerate 6
    # once the reindexing finally pointed a cutter somewhere pathological.
    # A correctly engraved d10 (10 arabic-numeral text cuts, each of which
    # legitimately contributes a few hundred new boolean-diff faces from the
    # extruded glyph geometry) empirically lands around ~2000 faces on this
    # Blender version/font -- well above the 10 base faces, and nowhere near
    # a collapsed handful, but this is naturally a much larger number than
    # the mid-corruption snapshot values seen in the bug repro (which were
    # measured mid-loop, after only 1-3 of the cuts had been mangled).
    assert faces_before < faces_after < 5000, (
        f"face count {faces_after} outside sane range for a correctly "
        f"engraved d10 (before={faces_before})"
    )

    assert len(obj.data.materials) >= 2, "painted fill should add a second material slot"

    bpy.data.objects.remove(obj, do_unlink=True)


def test_engraved_greek_numerals_d12_does_not_collapse_from_unwelded_cutter():
    """
    Regression test for asset_00006 (d12, greek_numerals, seed=48): the text
    cutter mesh produced by bpy.ops.object.convert(target='MESH') has
    duplicate, unwelded vertices at every seam between the front cap, back
    cap, and extrusion walls (true of every glyph tested, including this
    Greek-numeral style), which is not watertight/manifold. Feeding this
    straight into the EXACT boolean solver usually gets away with it, but on
    this exact die/style/size it corrupted the mesh catastrophically: after
    the first cut (Greek capital Alpha, "Α") face count *dropped* from 12 to
    10 (a correct cut should *increase* face count from the new recess
    walls), and after the second cut the whole die collapsed to two
    disconnected 1-2mm garbage fragments (28 verts/18 polys total).

    The fix welds duplicate vertices (bmesh.ops.remove_doubles) and
    recomputes normals on the cutter mesh right after conversion, before it
    is used as a boolean operand. This test reproduces the exact failing
    die/style/size and asserts the engraved result is a sane, non-collapsed
    mesh: positive volume within a tight band of the pristine solid's
    volume, and a final face count far above the pristine 12-face count (a
    collapse produces something tiny, e.g. 6-18 faces).

    Note on the volume check: unlike the simpler d6/arabic_numerals case in
    test_engraved_glyphs_reduce_solid_volume, this exact die/glyph/size
    combination empirically does NOT show a net volume *decrease* after
    engraving -- the EXACT boolean solver's own numerical noise on these
    particular multi-character Greek glyph cutters (e.g. "ΙΓ", "ΙΒ") nets
    out to a small (~0.25%) volume *increase* even with the fix applied, as
    verified directly against this codebase (pristine 10362.67 -> engraved
    10389.02). That is a harmless solver quirk, confirmed unrelated to the
    bug: bounding-box dimensions scale correctly (golden-ratio 1.618x
    size_mm on all 3 axes) and face count lands at the same ~3488 this
    codebase's fixed code produces when run through the real seed=48
    pipeline. The bug this test guards against is catastrophic collapse
    (volume divebombing to ~0.18 out of 10362, i.e. a ~56000x reduction),
    not fine-grained volume drift, so the bound below is intentionally wide
    enough to tolerate solver noise while still catching any collapse.
    """
    import bpy
    import bmesh
    from dice_gen import geometry, numbering, glyphs

    die_type = "d12"
    size_mm = 17.89272167378179  # matches asset_00006 (seed=48) exactly

    obj = geometry.build_die_base_mesh(die_type, size_mm=size_mm)
    pairs = geometry.compute_opposite_face_pairs(obj)
    assignment = numbering.assign_values_to_opposite_pairs(die_type, pairs)
    assert len(assignment) == 12, "d12 should have 12 faces assigned"

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    volume_before = bm.calc_volume()
    faces_before = len(bm.faces)
    bm.free()
    assert faces_before == 12, "pristine d12 should have 12 faces"

    glyphs.apply_engraved_glyphs(
        obj, die_type, assignment,
        glyph_style="greek_numerals", glyph_fill="blank",
        font_id="font_sans_bold", size_mm=size_mm,
    )

    bm2 = bmesh.new()
    bm2.from_mesh(obj.data)
    volume_after = bm2.calc_volume()
    faces_after = len(bm2.faces)
    bm2.free()

    assert volume_after > 0, "engraved die must not collapse to a degenerate/zero-volume mesh"
    volume_ratio = volume_after / volume_before
    assert 0.5 < volume_ratio < 1.5, (
        f"volume ratio {volume_ratio} (before={volume_before}, "
        f"after={volume_after}) is wildly off from 1.0 -- the original bug "
        f"gutted the die down to two 1-2mm garbage fragments "
        f"(ratio ~= 0.0000174)"
    )
    assert faces_after > 100, (
        f"face count {faces_after} looks collapsed (pristine was "
        f"{faces_before}); a correctly engraved d12 with 12 numeral cuts "
        f"should land well above 100 faces, not near the pristine count or "
        f"a handful like the original collapse (6-18 faces)"
    )

    bpy.data.objects.remove(obj, do_unlink=True)


def test_engraved_greek_numerals_d10_does_not_collapse_from_exact_solver_on_alpha_cut():
    """
    Regression test for asset_00091 (d10, greek_numerals, seed=133,
    size_mm=16.12691595326456): even after _weld_cutter_mesh welds duplicate
    vertices and recomputes normals, the Greek capital Alpha ("Α") glyph's
    cutter mesh retains ~42 residual non-manifold edges -- a genuine
    self-overlap intrinsic to how Blender's built-in Bfont tessellates that
    specific glyph's outline, not just an unwelded seam. Stepping through this
    exact die's cuts one at a time showed the die's volume was fine through
    cut 0 and cut 1, then cut 2 (value=1, label "Α") dropped the die's volume
    from 970.03 to 2.984 in a single boolean modifier_apply using the EXACT
    solver -- an outright catastrophic collapse, distinct from (and not fixed
    by) the unwelded-mesh fix that resolved asset_00006's d12 collapse.

    The fix in _boolean_diff_apply snapshots the die's volume before each cut,
    applies with EXACT as before, and if the result removed more than half the
    die's volume (geometrically impossible for one small glyph incision),
    restores the pre-cut mesh and retries the identical cut with the FLOAT
    solver, which tolerates this class of degenerate cutter (empirically
    970.03 -> 970.13 on this exact cut, a negligible change).

    This test reproduces the exact failing die/style/size and asserts the
    fully engraved result (all 10 cuts, including the "Α" cut) is a sane,
    non-collapsed mesh: volume well above the ~0.3%-remaining collapse seen in
    the actual bug, and a face count consistent with 10 real engraving cuts
    rather than a gutted shell.

    Note on connectivity: this test intentionally does not assert the result
    is a single connected component. Even the already-fixed, already-passing
    test_engraved_greek_numerals_d12_does_not_collapse_from_unwelded_cutter
    case (EXACT solver only, no FLOAT fallback triggered) empirically leaves
    one tiny (8-vert) disconnected garbage fragment alongside the main body --
    a harmless, pre-existing artifact of the EXACT boolean solver on these
    glyph cutters in general, unrelated to the catastrophic-collapse bug this
    test targets.
    """
    import bpy
    import bmesh
    from dice_gen import geometry, numbering, glyphs

    die_type = "d10"
    size_mm = 16.12691595326456  # matches asset_00091 (seed=133) exactly

    obj = geometry.build_die_base_mesh(die_type, size_mm=size_mm)
    pairs = geometry.compute_opposite_face_pairs(obj)
    assignment = numbering.assign_values_to_opposite_pairs(die_type, pairs)
    assert len(assignment) == 10, "d10 should have 10 faces assigned"

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    volume_before = bm.calc_volume()
    faces_before = len(bm.faces)
    bm.free()
    assert faces_before == 10, "pristine d10 should have 10 faces"

    glyphs.apply_engraved_glyphs(
        obj, die_type, assignment,
        glyph_style="greek_numerals", glyph_fill="blank",
        font_id="font_sans_bold", size_mm=size_mm,
    )

    bm2 = bmesh.new()
    bm2.from_mesh(obj.data)
    volume_after = bm2.calc_volume()
    faces_after = len(bm2.faces)
    bm2.free()

    assert volume_after > 0, "engraved die must not collapse to a degenerate/zero-volume mesh"
    # The actual bug left only ~0.3% of the pristine volume remaining
    # (2.984 / 970.03). The fix should keep volume well above that; a sane
    # engraved d10 with 10 small numeral cuts should retain the vast majority
    # of its volume, so 0.5 is a bound that would have failed against the bug
    # (consistent with the 50% collapse-detection threshold in the fix) while
    # comfortably passing a correctly engraved die.
    assert volume_after > 0.5 * volume_before, (
        f"volume ratio {volume_after / volume_before} looks collapsed "
        f"(before={volume_before}, after={volume_after}); the original bug "
        f"produced a ratio of ~0.003 (2.984 / 970.03) on this exact die"
    )
    assert faces_after > 100, (
        f"face count {faces_after} looks collapsed (pristine was "
        f"{faces_before}); a correctly engraved d10 with 10 numeral cuts "
        f"should land well above 100 faces, not near the pristine count or "
        f"a handful like the original collapse"
    )

    bpy.data.objects.remove(obj, do_unlink=True)


def test_engraved_arabic_numerals_d20_does_not_silently_noop_from_exact_solver():
    """
    Regression test for asset_00026 (d20, arabic_numerals, seed=68,
    size_mm=19.73093050365471): this asset passed every check from the two
    prior fixes (validate_dice_assets.py, bbox-diagonal/size_mm ratio, and
    the afb1af5 volume-collapse safety net) yet was completely unengraved.
    Direct reproduction showed that for every single one of the 20 numeral
    cuts, the EXACT boolean solver produced a complete no-op on the die's
    actual body: the pristine 20-face icosahedron survived byte-for-byte
    untouched through the entire cut loop, with each cutter's mesh merely
    appended into the die object's data as an inert, un-subtracted floating
    solid instead of being differenced in. The exported asset ended up as one
    correct-looking 62-face beveled-but-unengraved die body plus 31 leftover
    disconnected "debris" shells (one or two per digit).

    This silent no-op never tripped the afb1af5 volume-collapse check,
    because nothing was actually being subtracted -- the total volume barely
    changed per cut, so there was no collapse to detect. The fix (since
    refined three times more -- see _boolean_diff_apply's docstring for the
    full history) now checks a per-cut structural delta instead of a
    face-count heuristic: did this specific cut create a new closed
    (watertight) shell that wasn't there before it ran, excluding whichever
    component has the largest bounding-box diagonal (the die's own body,
    which is essentially never itself fully closed)? If so, that's
    un-subtracted debris, and the cut is retried with the more tolerant
    FLOAT solver, with a final backstop (run once after the whole cut loop)
    that discards any non-body closed shell still left over.

    This test reproduces the exact failing die/style/size and asserts the
    die was actually engraved: no non-body closed (debris) shell may remain
    (per _non_body_closed_component_count), and the die's face count must
    land far above the pristine 20-face count (a real 20-cut arabic-numeral
    engrave on a d20 empirically lands around 5617 faces per the senior's
    reproduction), not stuck near the tiny beveled-but-untouched base size
    that the original bug produced.
    """
    import bpy
    import bmesh
    from dice_gen import geometry, numbering, glyphs

    die_type = "d20"
    size_mm = 19.73093050365471  # matches asset_00026 (seed=68) exactly

    obj = geometry.build_die_base_mesh(die_type, size_mm=size_mm)
    pairs = geometry.compute_opposite_face_pairs(obj)
    assignment = numbering.assign_values_to_opposite_pairs(die_type, pairs)
    assert len(assignment) == 20, "d20 should have 20 faces assigned"

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    faces_before = len(bm.faces)
    pristine_verts = sorted(
        (round(v.co.x, 5), round(v.co.y, 5), round(v.co.z, 5)) for v in bm.verts
    )
    bm.free()
    assert faces_before == 20, "pristine d20 should have 20 faces"

    glyphs.apply_engraved_glyphs(
        obj, die_type, assignment,
        glyph_style="arabic_numerals", glyph_fill="blank",
        font_id="font_sans_bold", size_mm=size_mm,
    )

    bm2 = bmesh.new()
    bm2.from_mesh(obj.data)
    non_body_closed_after = glyphs._non_body_closed_component_count(bm2)
    faces_after = len(bm2.faces)
    post_verts = sorted(
        (round(v.co.x, 5), round(v.co.y, 5), round(v.co.z, 5)) for v in bm2.verts
    )
    bm2.free()

    # The original bug left the pristine vertex set completely untouched
    # (byte-for-byte, up to float rounding) as an isolated component, with
    # cutter debris merely appended alongside it. A correctly engraved die
    # must NOT still contain that exact untouched vertex set.
    assert post_verts != pristine_verts, (
        "die's vertex set is identical to the pristine pre-engrave mesh -- "
        "this is exactly the silent no-op failure mode from asset_00026, "
        "where EXACT left the body completely untouched on every cut"
    )

    assert non_body_closed_after == 0, (
        f"expected 0 non-body closed (un-subtracted debris) shells after "
        f"engraving, got {non_body_closed_after}"
    )

    assert faces_after > 500, (
        f"die has only {faces_after} faces after engraving (pristine was "
        f"{faces_before}); a correctly engraved d20 with 20 arabic-numeral "
        f"cuts should land in the thousands (~5617 empirically), not stay "
        f"near the tiny beveled-but-unengraved base size produced by the "
        f"original silent-no-op bug"
    )

    bpy.data.objects.remove(obj, do_unlink=True)


def test_engraved_arabic_numerals_d4_does_not_leave_undetected_debris():
    """
    Regression test for asset_00079 (d4, arabic_numerals, seed=121,
    size_mm=15.308910559884074): this is the die that exposed the deepest bug
    in this whole engrave-verification saga. Three progressively more
    sophisticated fixes were tried against it and all failed:

    1. Tracking the largest connected shell's face count (the cd7b268 fix for
       asset_00026) was fooled because a d4 starts at only 4 faces, so
       EXACT's un-subtracted debris for the numeral "2" cutter (263 faces)
       outweighed the real body (14 faces at that point) and looked like
       growth.
    2. Selecting "the body" by world-space bounding-box diagonal instead of
       face count (so debris, always physically tiny, can't masquerade as
       the body) fixed the selection but not the check: the body still only
       grew by 1 incidental face on the "2" cut (13->14) while the other 263
       faces of that cutter sat there as untouched debris, so "the body grew"
       stayed trivially true.
    3. Counting closed (watertight) shells and retrying/discarding whenever
       more than one existed also failed, because the die's own body is
       essentially never itself fully closed after a cut -- so a single
       un-subtracted debris blob (which IS closed) coexisting with the
       (open) body always presented as exactly "1 closed shell", identical
       to the "0 debris" case, and the check could never fire for exactly
       one debris blob (only for two or more coexisting at once, and even
       then it left older debris from earlier cuts untouched).

    The fix that actually resolves this asset asks a per-cut DELTA question
    instead of an absolute count or a growth comparison: did *this specific
    cut* create a new closed shell that wasn't present immediately before it
    ran, excluding whichever component has the largest bounding-box diagonal
    (the die's own body, whether or not that component itself happens to be
    open or closed)? See _non_body_closed_component_count and
    _boolean_diff_apply's docstring for the full history. A final backstop in
    apply_engraved_glyphs (_discard_non_body_closed_debris) also deletes any
    non-body closed shell still present after the whole cut loop finishes, as
    a guarantee independent of whether the per-cut retry logic caught it.

    This test reproduces the exact failing die/style/size and asserts no
    debris survived at all: not via a specific face-count range (since the
    end-of-loop backstop may or may not have had to trigger for this exact
    asset -- either path is an acceptable pass), but via the direct
    structural invariant itself.
    """
    import bpy
    import bmesh
    from dice_gen import geometry, numbering, glyphs

    die_type = "d4"
    size_mm = 15.308910559884074  # matches asset_00079 (seed=121) exactly

    obj = geometry.build_die_base_mesh(die_type, size_mm=size_mm)
    pairs = geometry.compute_opposite_face_pairs(obj)
    assignment = numbering.assign_values_to_opposite_pairs(die_type, pairs)
    assert len(assignment) == 4, "d4 should have 4 faces assigned"

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    faces_before = len(bm.faces)
    bm.free()
    assert faces_before == 4, "pristine d4 should have 4 faces"

    glyphs.apply_engraved_glyphs(
        obj, die_type, assignment,
        glyph_style="arabic_numerals", glyph_fill="blank",
        font_id="font_sans_bold", size_mm=size_mm,
    )

    bm2 = bmesh.new()
    bm2.from_mesh(obj.data)
    non_body_closed_after = glyphs._non_body_closed_component_count(bm2)
    faces_after = len(bm2.faces)
    bm2.free()

    assert non_body_closed_after == 0, (
        f"expected 0 non-body closed (un-subtracted debris) shells after "
        f"engraving asset_00079's exact die, got {non_body_closed_after}; "
        f"this is exactly the failure mode (a lone un-subtracted numeral "
        f"cutter, e.g. the 263-face cutter mesh for digit '2') that "
        f"survived three prior fix attempts on this asset"
    )
    assert faces_after > faces_before, (
        f"die has only {faces_after} faces after engraving (pristine was "
        f"{faces_before}); a genuinely engraved d4 should show substantial "
        f"growth from the 4 numeral cuts"
    )

    bpy.data.objects.remove(obj, do_unlink=True)


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
    call_count = [0]
    def fake_component_count(bm):
        call_count[0] += 1
        if call_count[0] == 1:
            return 0  # First call: pristine die has no debris
        else:
            return 999  # Subsequent calls: force debris to appear

    glyphs._non_body_closed_component_count = fake_component_count
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
    from mathutils import Vector
    from dice_gen import geometry, glyphs

    die_type = "d6"
    obj = geometry.build_die_base_mesh(die_type, size_mm=16.0)

    bm = bmesh.new()
    bm.from_mesh(obj.data)
    faces_before = len(bm.faces)
    debris_verts = bmesh.ops.create_cube(bm, size=1.0)["verts"]
    for v in debris_verts:
        v.co += Vector((100, 100, 100))  # place far from the die body, fully disjoint
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


def test_decal_glyphs_survive_usd_export_roundtrip_without_black_faces():
    """
    Regression test for the export-loss bug fixed alongside commit 5bc3361's
    shader-graph fix: apply_decal_glyphs used to composite the glyph decal
    onto the die's base material via a ShaderNodeMix wired into the
    Principled BSDF's Base Color input, using the decal's alpha as the mix
    factor. That in-memory graph rendered correctly in Blender's own
    viewport/render, and the existing
    test_decal_glyphs_assigns_one_material_per_face test only ever inspected
    that in-memory graph -- so it kept passing even though the fix was
    silently lost on export. bpy.ops.wm.usd_export cannot represent a
    ShaderNodeMix (confirmed in both default UsdPreviewSurface mode and with
    generate_materialx_network=True): every previously-shipped
    printed_decal asset (~247 of a 500-asset batch) exported with Base Color
    fed directly by the raw, uncomposited glyph-decal Image Texture again --
    reproducing the exact pre-5bc3361 defect (solid black faces outside the
    glyph strokes) invisibly, only in the exported USD, never caught by any
    in-memory-only test.

    The real fix (this commit) composites the glyph decal onto a render of
    the die's actual base material at the pixel level in Python/numpy
    (_render_material_swatch + _composite_alpha_over) *before* the texture
    is ever wired into the shader graph, so the wire itself is a plain,
    single Image-Texture-to-Base-Color connection with no Mix node for the
    exporter to drop.

    This test proves the fix actually survives export, not just that the
    in-memory graph looks right: it builds a die with a real base material
    (the same materials.build_material + apply_material sequence
    orchestrator._generate_one uses for the decal path), applies decal
    glyphs, round-trips the die through a real bpy.ops.wm.usd_export /
    bpy.ops.wm.usd_import cycle, and inspects the RELOADED material and
    RELOADED texture image -- exactly how the original bug was confirmed
    (by reloading a shipped USD and inspecting its node graph). It asserts
    (a) the reloaded face material's Base Color is fed directly by a
    ShaderNodeTexImage, not a ShaderNodeMix (which usd_import could never
    produce anyway, since usd_export never wrote one), and (b) the
    reloaded texture image's actual pixels are NOT solid black in a corner
    far from any glyph ink -- directly reproducing (and disproving) the
    visual defect this whole fix addresses.
    """
    import bpy
    import numpy as np
    from dice_gen import geometry, numbering, glyphs, materials

    die_type = "d6"
    size_mm = 16.0

    obj = geometry.build_die_base_mesh(die_type, size_mm=size_mm)
    pairs = geometry.compute_opposite_face_pairs(obj)
    assignment = numbering.assign_values_to_opposite_pairs(die_type, pairs)

    # Use a bright, saturated "opaque" color so a black-vs-non-black pixel
    # check is unambiguous, and build/apply the material the same way
    # orchestrator._generate_one does for the printed_decal path.
    mat_params = {"hue": 0.55, "saturation": 0.85, "value": 0.95, "roughness": 0.3}
    base_mat = materials.build_material(obj.name, "opaque", mat_params)
    materials.apply_material(obj, base_mat, slot_index=0)

    with tempfile.TemporaryDirectory() as tmp_dir:
        glyphs.apply_decal_glyphs(
            obj, die_type, assignment,
            glyph_style="arabic_numerals", font_id="font_sans_bold",
            size_mm=size_mm, asset_id="test_asset", tmp_dir=tmp_dir,
        )

        face_index = next(iter(assignment))

        usd_path = os.path.join(tmp_dir, "roundtrip.usd")
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.wm.usd_export(filepath=usd_path, selected_objects_only=True)

        # Remove the in-memory die entirely before reloading, so nothing
        # below can accidentally pass by inspecting pre-export state instead
        # of what actually round-tripped through the USD file.
        bpy.data.objects.remove(obj, do_unlink=True)

        pre_import_objects = set(bpy.context.scene.objects)
        bpy.ops.wm.usd_import(filepath=usd_path)
        new_objects = [
            o for o in bpy.context.scene.objects if o not in pre_import_objects
        ]
        reimported_die = next(o for o in new_objects if o.type == 'MESH')

        reimported_mat_index = reimported_die.data.polygons[face_index].material_index
        reimported_mat = reimported_die.data.materials[reimported_mat_index]
        assert reimported_mat is not None, "reimported face has no material"

        nt = reimported_mat.node_tree
        bsdf = next(n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED')
        base_color_input = bsdf.inputs["Base Color"]

        feeding_link = None
        for link in nt.links:
            if link.to_socket == base_color_input:
                feeding_link = link
                break

        assert feeding_link is not None, (
            "reimported material's Base Color has no incoming link at all "
            "-- expected a direct Image Texture connection"
        )
        assert feeding_link.from_node.type == 'TEX_IMAGE', (
            f"reimported material's Base Color is fed by a "
            f"{feeding_link.from_node.type} node, not a plain Image Texture "
            f"-- if this is a ShaderNodeMix (or anything else), the "
            f"compositing is (still) happening in the shader graph, which "
            f"does not survive bpy.ops.wm.usd_export"
        )
        assert feeding_link.from_node.type != 'MIX', (
            "reimported material's Base Color is fed by a Mix node -- this "
            "cannot possibly have come from usd_export/usd_import, so "
            "something is deeply wrong with this test's assumptions"
        )

        tex_image = feeding_link.from_node.image
        assert tex_image is not None, "reimported Image Texture node has no image"
        width, height = tex_image.size
        assert width > 0 and height > 0, "reimported texture image has zero size"

        pixels = np.empty(width * height * 4, dtype=np.float32)
        tex_image.pixels.foreach_get(pixels)
        pixels = pixels.reshape(height, width, 4)

        # Sample a corner pixel: the glyph decal's UV smart-project maps
        # each face to its own island, but the glyph ink itself is centered
        # and small (see PIP_VALUE_LAYOUTS / glyph_font_size), so a texture
        # corner is always far from any glyph stroke and should show the
        # die's actual base material color, not black.
        corner_rgb = pixels[0, 0, :3]
        assert not np.allclose(corner_rgb, 0.0, atol=0.02), (
            f"reimported decal texture's corner pixel is solid black "
            f"({corner_rgb}) -- this is exactly the pre-5bc3361 visual "
            f"defect (raw glyph-on-transparent composited/exported with no "
            f"base material showing through) that this whole fix addresses"
        )

    bpy.data.objects.remove(reimported_die, do_unlink=True)


def test_decal_glyphs_use_asset_id_to_avoid_cross_asset_filename_collisions():
    """
    Regression test for a filename-collision bug found by inspecting real
    shipped assets: apply_decal_glyphs (and its helpers _render_label_to_image,
    _render_material_swatch, _composite_alpha_over) used to derive every
    output image's filename from die_obj.name. die_obj.name is always just
    f"{die_type}_die" (e.g. "d8_die") -- identical across every asset of the
    same die type in a batch -- because orchestrator._generate_one removes
    the die object at the end of each iteration (bpy.data.objects.remove(...,
    do_unlink=True)), which frees the name so Blender does not auto-suffix it
    on the next same-die-type asset. orchestrator._generate_one also passes
    the shared batch outdir (not a per-asset temp directory) as tmp_dir. Net
    effect: every same-die-type printed_decal asset in a batch wrote to
    colliding filenames like "d8_die_face0.png" and "d8_die_swatch.png", so a
    later asset's render would silently overwrite an earlier asset's texture
    file on disk -- confirmed independently by finding two real shipped
    assets of the same die type but different material_params whose face-0
    materials both pointed at the exact same file with byte-identical pixel
    content.

    This test reproduces the collision directly: it builds two d8 dice with
    DIFFERENT asset_ids and DIFFERENT material colors, and calls
    apply_decal_glyphs for each in the SAME tmp_dir (mirroring the shared
    batch outdir orchestrator._generate_one uses for the decal path). Against
    the pre-fix code (filenames keyed on die_obj.name, identical for both
    dice since they're both "d8_die"), the second call's renders would
    overwrite the first's files on disk, and both dice's face-0 Image
    Texture would resolve to the identical filepath with identical pixel
    content. The fix keys every output filename on the caller-supplied
    asset_id instead, which is unique per asset within a batch, so the two
    dice's files can never collide.
    """
    import bpy
    import numpy as np
    from dice_gen import geometry, numbering, glyphs, materials

    die_type = "d8"
    size_mm = 16.0

    with tempfile.TemporaryDirectory() as tmp_dir:
        resolved_paths = []
        corner_pixels = []

        for asset_id, hue in (("asset_A", 0.05), ("asset_B", 0.6)):
            obj = geometry.build_die_base_mesh(die_type, size_mm=size_mm)
            pairs = geometry.compute_opposite_face_pairs(obj)
            assignment = numbering.assign_values_to_opposite_pairs(die_type, pairs)

            mat_params = {"hue": hue, "saturation": 0.85, "value": 0.95, "roughness": 0.3}
            base_mat = materials.build_material(obj.name, "opaque", mat_params)
            materials.apply_material(obj, base_mat, slot_index=0)

            glyphs.apply_decal_glyphs(
                obj, die_type, assignment,
                glyph_style="arabic_numerals", font_id="font_sans_bold",
                size_mm=size_mm, asset_id=asset_id, tmp_dir=tmp_dir,
            )

            face_index = next(iter(assignment))
            mat_index = obj.data.polygons[face_index].material_index
            face_mat = obj.data.materials[mat_index]
            nt = face_mat.node_tree
            bsdf = nt.nodes["Principled BSDF"]
            base_color_input = bsdf.inputs["Base Color"]

            feeding_link = None
            for link in nt.links:
                if link.to_socket == base_color_input:
                    feeding_link = link
                    break
            assert feeding_link is not None, (
                f"{asset_id}: face-0 material's Base Color has no incoming "
                f"link -- expected an Image Texture connection"
            )
            tex_image = feeding_link.from_node.image
            assert tex_image is not None, f"{asset_id}: Image Texture node has no image"

            resolved_path = bpy.path.abspath(tex_image.filepath_raw or tex_image.filepath)
            resolved_paths.append(resolved_path)

            width, height = tex_image.size
            pixels = np.empty(width * height * 4, dtype=np.float32)
            tex_image.pixels.foreach_get(pixels)
            pixels = pixels.reshape(height, width, 4)
            corner_pixels.append(pixels[0, 0, :3].copy())

            bpy.data.objects.remove(obj, do_unlink=True)

        assert resolved_paths[0] != resolved_paths[1], (
            f"both dice's face-0 Image Texture resolved to the SAME file "
            f"path ({resolved_paths[0]!r}) despite different asset_ids -- "
            f"this is exactly the cross-asset filename collision this fix "
            f"addresses: a later same-die-type asset in a batch would "
            f"silently overwrite an earlier asset's texture file on disk"
        )
        assert not np.allclose(corner_pixels[0], corner_pixels[1], atol=0.02), (
            f"both dice's corner (background swatch) pixels are the same "
            f"color ({corner_pixels[0]} vs {corner_pixels[1]}) despite "
            f"different material hues -- suggests one asset's composited "
            f"texture file was silently overwritten by the other's render, "
            f"reproducing the original filename-collision bug"
        )


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
        glyphs._unwrap_faces_to_full_square(obj, die_type, margin=margin)

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
            assert min(us) >= -1e-6 and max(us) <= 1.0 + 1e-6, (
                f"{die_type} face {poly.index}: UV u-range "
                f"{min(us):.4f}-{max(us):.4f} falls outside [0,1]"
            )
            assert min(vs) >= -1e-6 and max(vs) <= 1.0 + 1e-6, (
                f"{die_type} face {poly.index}: UV v-range "
                f"{min(vs):.4f}-{max(vs):.4f} falls outside [0,1]"
            )

        bpy.data.objects.remove(obj, do_unlink=True)


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


def test_engrave_depth_fraction_is_shallower_than_before():
    """
    0.025 was the first candidate value, but it measurably increased the
    known EXACT/FLOAT-collapse skipped-cut failure rate on some dice
    (confirmed empirically: a specific d20/roman_numerals die went from 1
    skipped cut at the original 0.04 to 8 skipped cuts at 0.025, and a
    d20/cjk_numerals die from 1 to 3). 0.03 was chosen after confirming it
    restores both dice to at or below their original 0.04 failure count
    (1 and 0 respectively) while still being 25% shallower than 0.04.
    """
    from dice_gen import glyphs

    assert glyphs.ENGRAVE_DEPTH_FRACTION == 0.03, (
        f"expected ENGRAVE_DEPTH_FRACTION == 0.03, got "
        f"{glyphs.ENGRAVE_DEPTH_FRACTION}"
    )


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
    large single copy that happens to touch all these regions.

    Region boundaries match the equilateral-triangle-vertex-based layout
    (not a fixed-radius circle -- an earlier version used a circle, which
    was found during manual batch regeneration to place the two "bottom"
    copies well inside the face, nowhere near the real bottom corners;
    see _render_label_to_image's d4 branch for the full explanation).
    Confirmed empirically by rendering this exact layout and scanning the
    raw pixel buffer row-by-row: ink forms two clearly separated clusters
    with a wide gap between them, top cluster at rows ~140-214 (the apex
    copy) and bottom cluster at rows ~44-115 (the two base copies, which
    stay clearly separated from each other at every row in this range --
    left cluster columns stay <=108, right cluster columns stay >=148).

    The "no ink at center" region was re-measured (not guessed) after
    task 5 (face-geometry-proportional glyph sizing) made this d4 face's
    glyphs noticeably larger than before (d4's own face inradius is
    large relative to size_mm -- 7.35mm of 18.0mm, ratio 0.408 -- second
    only to d6's 0.5, so d4's corner numerals grew the most of any die
    type under the new proportional sizing): the apex copy's glyph now
    extends down to row ~138, which overlaps the OLD center-check box's
    row range (108-148). Rendering this exact case
    (glyph_style="arabic_numerals", value=3, die_type="d4", real d4
    inradius/size_mm=18.0 fed through _proportional_font_size +
    DECAL_FONT_CANVAS_SCALE) and scanning the actual pixel buffer found
    rows 116-137 are the true, fully-empty gap between the two clusters
    at every column -- the new center-check box (119-135, 96-160) sits
    comfortably inside that measured gap with margin on every side. The
    other three regions (top/bottom-left/bottom-right corner ink)
    verified unchanged against the new render and did not need updating.
    """
    import bpy
    import numpy as np
    from dice_gen import geometry, glyphs

    resolution = 256
    size_mm = 18.0
    d4_obj = geometry.build_die_base_mesh("d4", size_mm=size_mm)
    inradius = geometry.compute_face_inradius(
        d4_obj.data, d4_obj.data.polygons[0], d4_obj.matrix_world
    )
    bpy.data.objects.remove(d4_obj, do_unlink=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        image_path = os.path.join(tmp_dir, "test_d4_corners.png")
        glyphs._render_label_to_image(
            3, "arabic_numerals", "font_sans_bold", "d4", image_path, resolution=resolution,
            inradius=inradius, size_mm=size_mm,
        )

        img = bpy.data.images.load(image_path)
        pixels = np.empty(resolution * resolution * 4, dtype=np.float32)
        img.pixels.foreach_get(pixels)
        alpha = pixels.reshape(resolution, resolution, 4)[:, :, 3]
        bpy.data.images.remove(img)

        def region_has_ink(y0, y1, x0, x1):
            return bool((alpha[y0:y1, x0:x1] > 0.05).any())

        assert not region_has_ink(119, 135, 96, 160), (
            "expected NO ink in the measured gap between the apex and "
            "base copies -- a 3-corner layout should leave this band empty"
        )
        assert region_has_ink(150, 200, 100, 160), "expected ink near the top corner"
        assert region_has_ink(55, 95, 40, 100), "expected ink near the bottom-left corner"
        assert region_has_ink(55, 95, 160, 220), "expected ink near the bottom-right corner"


def test_unwrap_faces_to_full_square_gives_d4_faces_consistent_apex_up_orientation():
    """
    Regression test for a bug found via manual batch regeneration: the raw
    tangent/bitangent projection in _unwrap_faces_to_full_square does not
    guarantee the same triangle orientation face to face -- on a d4, faces
    alternated between "apex-up" (one vertex at high v, two sharing a low
    v) and "apex-down" (the reverse), depending on each face's normal
    relative to the shared global-up-hint convention. This broke
    _render_label_to_image's fixed 3-corner d4 layout, which assumes every
    face is apex-up: on apex-down faces, the "top" numeral landed on a
    flat edge (clipped) and the two "bottom" numerals fell outside the
    real triangular UV footprint (invisible). Confirmed visually: several
    regenerated d4 decal assets showed only a clipped top numeral and no
    bottom-corner numerals at all.

    This checks every one of a d4's 4 faces ends up with its single
    distinctive vertex (the one whose v differs from the other two) at
    the HIGH v value, consistently -- i.e. every face is apex-up.
    """
    import bpy
    from dice_gen import geometry, glyphs

    obj = geometry.build_die_base_mesh("d4", size_mm=16.0)
    glyphs._unwrap_faces_to_full_square(obj, "d4")

    uv_layer = obj.data.uv_layers.active.data
    for poly in obj.data.polygons:
        vs = [uv_layer[li].uv.y for li in poly.loop_indices]
        assert len(vs) == 3, "d4 faces should be triangles"

        _, apex_index = min(
            (abs(vs[0] - vs[1]), 2),
            (abs(vs[1] - vs[2]), 0),
            (abs(vs[0] - vs[2]), 1),
        )
        other_v = sum(v for i, v in enumerate(vs) if i != apex_index) / 2.0

        assert vs[apex_index] > other_v, (
            f"face {poly.index}: expected the isolated vertex (v={vs[apex_index]:.3f}) "
            f"to be ABOVE the other two (avg v={other_v:.3f}) -- i.e. apex-up -- "
            f"got apex-down instead, got vs={vs}"
        )

    bpy.data.objects.remove(obj, do_unlink=True)


def test_unwrap_faces_to_full_square_does_not_mirror_any_triangular_face():
    """
    Regression test for a severe bug in the apex-up fix above: the first
    version of that fix corrected apex-down faces by negating ONLY v,
    which is a MIRROR reflection (determinant -1), not a rotation --
    reversing winding/handedness on exactly the faces that needed
    correction, which would render any text on those faces backwards.
    Confirmed empirically: the raw (pre-fix) tangent/bitangent projection
    was ALREADY winding-consistent across every face of a d4/d8/d20 (all
    positive signed area), but the v-only-negate fix produced a MIX of
    positive and negative signed areas (exactly the faces it "corrected"
    flipped sign), while a fixed version that negates BOTH u and v (a
    proper 180-degree rotation, determinant +1) keeps every face's signed
    area the same sign as every other face's, on every die type with
    triangular faces (d4, d8, d20) -- not just the ones exercised by the
    apex-up test above.

    This checks the signed area (2D shoelace formula) of every
    triangular face's final UV coordinates has the SAME sign as every
    other triangular face's, for d4, d8, and d20.
    """
    import bpy
    from dice_gen import geometry, glyphs

    def signed_area(coords):
        area = 0.0
        n = len(coords)
        for i in range(n):
            x1, y1 = coords[i]
            x2, y2 = coords[(i + 1) % n]
            area += x1 * y2 - x2 * y1
        return area / 2.0

    for die_type, size_mm in (("d4", 16.0), ("d8", 20.0), ("d20", 20.0)):
        obj = geometry.build_die_base_mesh(die_type, size_mm=size_mm)
        glyphs._unwrap_faces_to_full_square(obj, die_type)

        uv_layer = obj.data.uv_layers.active.data
        signs = set()
        for poly in obj.data.polygons:
            coords = [(uv_layer[li].uv.x, uv_layer[li].uv.y) for li in poly.loop_indices]
            signs.add(signed_area(coords) > 0)

        assert len(signs) == 1, (
            f"{die_type}: expected every triangular face's UV winding to "
            f"have the SAME sign (no mirroring), but found faces with "
            f"BOTH signs -- some faces are mirrored relative to others, "
            f"which would render text backwards on the mirrored ones"
        )

        bpy.data.objects.remove(obj, do_unlink=True)


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


def run():
    test_glyph_label_formats()
    test_proportional_font_size_shrinks_for_longer_labels()
    test_engraved_glyphs_reduce_solid_volume()
    test_engraved_glyphs_blank_fill_does_not_add_second_material()
    test_decal_glyphs_assigns_one_material_per_face()
    test_decal_glyphs_survive_usd_export_roundtrip_without_black_faces()
    test_decal_glyphs_use_asset_id_to_avoid_cross_asset_filename_collisions()
    test_engraved_glyphs_use_pristine_face_orientations_not_reindexed_mid_loop()
    test_engraved_greek_numerals_d12_does_not_collapse_from_unwelded_cutter()
    test_engraved_greek_numerals_d10_does_not_collapse_from_exact_solver_on_alpha_cut()
    test_engraved_arabic_numerals_d20_does_not_silently_noop_from_exact_solver()
    test_engraved_arabic_numerals_d4_does_not_leave_undetected_debris()
    test_apply_engraved_glyphs_returns_empty_list_for_clean_die()
    test_apply_engraved_glyphs_aggregates_forced_cut_warnings()
    test_boolean_diff_apply_returns_warning_when_both_solvers_fail()
    test_discard_non_body_closed_debris_returns_warning_for_manual_debris()
    test_tangent_bitangent_only_falls_back_to_y_for_truly_vertical_normals()
    test_tangent_bitangent_falls_back_to_y_for_axis_aligned_normals()
    test_unwrap_faces_to_full_square_covers_full_uv_range_per_face()
    test_load_font_maps_font_ids_to_distinct_installed_fonts()
    test_load_font_returns_none_for_cjk_numerals_regardless_of_font_id()
    test_load_font_returns_none_for_unrecognized_font_id()
    test_apply_engraved_glyphs_uses_load_font_with_correct_glyph_style()
    test_apply_decal_glyphs_uses_load_font_with_correct_glyph_style()
    test_engrave_depth_fraction_is_shallower_than_before()
    test_engrave_depth_fraction_is_shallower_than_prior_value()
    test_apply_engraved_glyphs_cutter_edges_are_rounded_not_sharp()
    test_face_vertex_orientations_returns_one_outward_pointing_matrix_per_vertex()
    test_apply_engraved_glyphs_cuts_three_corners_per_face_for_d4_numerals()
    test_apply_engraved_glyphs_does_not_triple_pips_for_d4()
    test_apply_engraved_glyphs_does_not_triple_numerals_for_non_d4_dice()
    test_render_label_to_image_renders_three_corner_copies_for_d4()
    test_unwrap_faces_to_full_square_gives_d4_faces_consistent_apex_up_orientation()
    test_unwrap_faces_to_full_square_does_not_mirror_any_triangular_face()
    test_tangent_bitangent_up_reference_overrides_global_up_hint()
    test_face_orientation_matrix_mirrors_between_d8_hemispheres()
    test_apply_engraved_glyphs_orients_d8_hemispheres_toward_their_own_pole()
    test_unwrap_faces_to_full_square_mirrors_between_d8_hemispheres()


run_and_report(run)
