import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from _harness import run_and_report


def test_all_seven_dice_build_with_correct_topology():
    import bpy
    from datagen.domains.dice import geometry

    for die_type, spec in geometry.DIE_SPECS.items():
        obj = geometry.build_die_base_mesh(die_type, size_mm=16.0)
        n_faces = len(obj.data.polygons)
        n_verts = len(obj.data.vertices)
        n_edges = len(obj.data.edges)

        assert n_faces == spec["expected_faces"], (
            f"{die_type}: expected {spec['expected_faces']} faces, got {n_faces}"
        )
        assert n_verts == spec["expected_verts"], (
            f"{die_type}: expected {spec['expected_verts']} verts, got {n_verts}"
        )
        assert n_edges == spec["expected_edges"], (
            f"{die_type}: expected {spec['expected_edges']} edges, got {n_edges}"
        )
        bpy.data.objects.remove(obj, do_unlink=True)


def test_opposite_face_pairs_are_geometrically_antiparallel_for_d6():
    import bpy
    from datagen.domains.dice import geometry

    obj = geometry.build_die_base_mesh("d6", size_mm=16.0)
    pairs = geometry.compute_opposite_face_pairs(obj)
    assert len(pairs) == 3

    obj.data.polygons.foreach_set  # ensure normals accessible
    for a, b in pairs:
        na = obj.data.polygons[a].normal
        nb = obj.data.polygons[b].normal
        dot = na.dot(nb)
        assert dot < -0.99, f"faces {a},{b} not antiparallel (dot={dot})"

    bpy.data.objects.remove(obj, do_unlink=True)


def test_d4_opposite_face_pairs_returns_two_pairs_covering_all_faces():
    import bpy
    from datagen.domains.dice import geometry

    obj = geometry.build_die_base_mesh("d4", size_mm=16.0)
    pairs = geometry.compute_opposite_face_pairs(obj)
    flat = sorted(f for pair in pairs for f in pair)
    assert flat == [0, 1, 2, 3]
    bpy.data.objects.remove(obj, do_unlink=True)


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
    from datagen.domains.dice import geometry

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
    from datagen.domains.dice import geometry

    for die_type in ("d8", "d10", "d10_pct"):
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
    from datagen.domains.dice import geometry

    for die_type in ("d4", "d6", "d12", "d20"):
        obj = geometry.build_die_base_mesh(die_type, size_mm=18.0)
        assert geometry.compute_face_poles(obj, die_type) is None, (
            f"{die_type}: expected None (not a two-pole bipyramid)"
        )
        bpy.data.objects.remove(obj, do_unlink=True)


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
    from datagen.domains.dice import geometry, numbering

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
    from datagen.domains.dice import geometry, numbering

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


def test_assign_values_to_opposite_pairs_without_hemisphere_arg_is_unchanged():
    """
    Every other die type (and any caller not passing hemisphere_of_face)
    must keep today's exact assignment behavior -- this plan only adds an
    opt-in path for d8/d10.
    """
    import bpy
    from datagen.domains.dice import geometry, numbering

    for die_type in ("d4", "d6", "d12", "d20"):
        obj = geometry.build_die_base_mesh(die_type, size_mm=18.0)
        face_pairs = geometry.compute_opposite_face_pairs(obj)
        before = numbering.assign_values_to_opposite_pairs(die_type, face_pairs)
        after = numbering.assign_values_to_opposite_pairs(die_type, face_pairs, hemisphere_of_face=None)
        assert before == after
        bpy.data.objects.remove(obj, do_unlink=True)


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
    from datagen.domains.dice import geometry

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
    from datagen.domains.dice import geometry

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


def run():
    test_all_seven_dice_build_with_correct_topology()
    test_opposite_face_pairs_are_geometrically_antiparallel_for_d6()
    test_d4_opposite_face_pairs_returns_two_pairs_covering_all_faces()
    test_build_die_base_mesh_marks_all_edges_with_full_bevel_weight()
    test_compute_face_poles_maps_every_d8_and_d10_face_to_exactly_one_pole()
    test_compute_face_poles_returns_none_for_non_bipyramid_dice()
    test_assign_values_to_opposite_pairs_splits_d8_and_d10_hemispheres_by_parity()
    test_assign_values_to_opposite_pairs_splits_d10_pct_hemispheres_by_tens_digit_parity()
    test_assign_values_to_opposite_pairs_without_hemisphere_arg_is_unchanged()
    test_compute_face_inradius_matches_known_values_for_a_cube()
    test_compute_face_inradius_is_smaller_for_d8_and_d20_than_d6_and_d12_at_same_size()


run_and_report(run)
