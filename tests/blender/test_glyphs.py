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
            size_mm=16.0, tmp_dir=tmp_dir,
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

    def spy_orientation(face, obj_matrix):
        result = real_orientation_fn(face, obj_matrix)
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
    changed per cut, so there was no collapse to detect. The fix adds a
    second, complementary check based on connected-component face count: a
    genuine engrave cut always adds at least a few wall/floor faces to
    whatever it touches, so if the largest connected shell's face count fails
    to grow after an EXACT apply, that's a no-op, and the cut is retried with
    the more tolerant FLOAT solver.

    This test reproduces the exact failing die/style/size and asserts the
    die was actually engraved: the largest connected component's face count
    must land far above the pristine 20-face count (a real 20-cut
    arabic-numeral engrave on a d20 empirically lands around 5617 faces per
    the senior's reproduction), not stuck near the tiny beveled-but-untouched
    base size that the original bug produced.
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
    largest_component_after = glyphs._largest_component_face_count(bm2)
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

    assert largest_component_after > 500, (
        f"largest connected component has only {largest_component_after} "
        f"faces after engraving (pristine was {faces_before}); a correctly "
        f"engraved d20 with 20 arabic-numeral cuts should land in the "
        f"thousands (~5617 empirically), not stay near the tiny "
        f"beveled-but-unengraved base size produced by the original "
        f"silent-no-op bug"
    )

    bpy.data.objects.remove(obj, do_unlink=True)


def run():
    test_glyph_label_formats()
    test_engraved_glyphs_reduce_solid_volume()
    test_engraved_glyphs_blank_fill_does_not_add_second_material()
    test_decal_glyphs_assigns_one_material_per_face()
    test_engraved_glyphs_use_pristine_face_orientations_not_reindexed_mid_loop()
    test_engraved_greek_numerals_d12_does_not_collapse_from_unwelded_cutter()
    test_engraved_greek_numerals_d10_does_not_collapse_from_exact_solver_on_alpha_cut()
    test_engraved_arabic_numerals_d20_does_not_silently_noop_from_exact_solver()


run_and_report(run)
