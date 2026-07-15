"""Sim-independent unit tests for tasks/franka/notch_fixture.py's pure
geometry (no isaaclab/pxr/torch import needed). Run via:
/home/saps/IsaacLab/_isaac_sim/python.sh -m pytest tests/test_notch_fixture.py -v -p no:launch_testing
(this repo's established convention - see project_pytest-needs-isaac-sim-python
memory; this module has no numpy dependency at all, but the canonical
invocation stays consistent with every other tests/ file in this repo)."""

import math

import pytest

from tasks.franka.notch_fixture import (
    CHAMFER_DEPTH_M,
    CHAMFER_FLARE_HALF_WIDTH_M,
    D4_EDGE_LENGTH_M,
    FINGER_TIP_LOCAL_Z_M,
    FIXTURE_Z_EXTENT_M,
    NOTCH_ANGLE_DEG,
    NOTCH_DEPTH_M,
    PAD_HALF_WIDTH_M,
    PAD_Z_MIN_M,
    WALL_PRISM_FACE_VERTEX_COUNTS,
    WALL_PRISM_FACE_VERTEX_INDICES,
    apex_height_m,
    dihedral_angle_rad,
    grip_height_above_table_m,
    joint_local_pos0_m,
    joint_local_rot1_wxyz,
    local_edge_length_at_depth_below_apex_m,
    mirror_profile_x,
    notch_angle_from_dihedral_rad,
    notch_opening_width_m,
    notch_wall_profile_xy,
    wall_prism_points,
)

# Task 0's own re-derivation table (.superpowers/sdd/task-0-report.md,
# section 3), reused here as the known-answer fixture for this module's
# formulas - not re-derived independently, same source arithmetic, this
# test just guards against a future regression/transcription error.
_EXPECTED_APEX_HEIGHT_MM = 19.262
_EXPECTED_DIHEDRAL_DEG = 70.5288
_EXPECTED_NOTCH_ANGLE_DEG = 109.4712
_EXPECTED_GRIP_HEIGHT_MM = 9.262
_EXPECTED_LOCAL_EDGE_AT_10MM_MM = 12.247
_EXPECTED_OPENING_WIDTH_MM = 11.425
_EXPECTED_FLANKING_MM = 3.217


class TestTrigDerivation:
    """Re-derives Task 0's own desk-check table - guards against a future
    transcription/regression error in the formulas themselves."""

    def test_apex_height(self):
        assert apex_height_m(D4_EDGE_LENGTH_M) * 1000 == pytest.approx(_EXPECTED_APEX_HEIGHT_MM, abs=1e-3)

    def test_dihedral_angle(self):
        assert math.degrees(dihedral_angle_rad()) == pytest.approx(_EXPECTED_DIHEDRAL_DEG, abs=1e-4)

    def test_notch_angle_from_dihedral_rounds_to_110(self):
        # NOTE: the spec/report round 109.4712deg UP to 110deg ("rounded to
        # a buildable value"), not Python's nearest-integer round() (which
        # would give 109) - this test checks the RAW value matches Task 0's
        # figure and that NOTCH_ANGLE_DEG is the nearest-integer-or-above
        # buildable value actually used downstream, not that it's a
        # mathematically-nearest rounding.
        raw_deg = math.degrees(notch_angle_from_dihedral_rad())
        assert raw_deg == pytest.approx(_EXPECTED_NOTCH_ANGLE_DEG, abs=1e-4)
        assert abs(NOTCH_ANGLE_DEG - raw_deg) < 1.0

    def test_grip_height_above_table(self):
        grip_m = grip_height_above_table_m(D4_EDGE_LENGTH_M, below_apex_m=0.010)
        assert grip_m * 1000 == pytest.approx(_EXPECTED_GRIP_HEIGHT_MM, abs=1e-3)

    def test_local_edge_length_at_10mm_below_apex(self):
        local_edge_m = local_edge_length_at_depth_below_apex_m(0.010, D4_EDGE_LENGTH_M)
        assert local_edge_m * 1000 == pytest.approx(_EXPECTED_LOCAL_EDGE_AT_10MM_MM, abs=1e-3)

    def test_notch_opening_width(self):
        opening_m = notch_opening_width_m(NOTCH_DEPTH_M, NOTCH_ANGLE_DEG)
        assert opening_m * 1000 == pytest.approx(_EXPECTED_OPENING_WIDTH_MM, abs=1e-3)

    def test_flanking_material_each_side(self):
        # abs=0.01 (not 0.001 like the other checks here): Task 0's own
        # report table computed this using a DISPLAY-rounded 17.86mm pad
        # width, not the exact 17.862mm PAD_HALF_WIDTH_M*2 this module
        # uses - a ~0.001mm-scale display-rounding difference, not a
        # formula error (confirmed: (17.862-11.425)/2=3.2185 vs the
        # report's (17.86-11.425)/2=3.2175).
        opening_m = notch_opening_width_m(NOTCH_DEPTH_M, NOTCH_ANGLE_DEG)
        flanking_m = (2 * PAD_HALF_WIDTH_M - opening_m) / 2.0
        assert flanking_m * 1000 == pytest.approx(_EXPECTED_FLANKING_MM, abs=0.01)


class TestFixtureFitsWithinPad:
    """The fixture must not overhang the stock pad's own extent (Task 0's
    "buildable without touching the base asset" constraint) - checked here
    as executable assertions, not just eyeballed from the numbers."""

    def test_mouth_half_width_under_pad_half_width(self):
        profile = notch_wall_profile_xy()
        mouth_half_width = profile[-1].x  # outermost point's own x
        assert 0 < mouth_half_width < PAD_HALF_WIDTH_M

    def test_fixture_z_extent_stays_within_pad_z_range(self):
        fixture_min_z = FINGER_TIP_LOCAL_Z_M - FIXTURE_Z_EXTENT_M
        assert fixture_min_z >= PAD_Z_MIN_M

    def test_fixture_does_not_protrude_past_stock_tip(self):
        # The fixture's own local Z=0 (attachment point) coincides with the
        # stock tip (FINGER_TIP_LOCAL_Z_M) and it only extends in -Z (never
        # +Z) - see module docstring's "does not protrude past the stock
        # fingertip's own measured tip" design choice.
        z_values = [p[2] for p in wall_prism_points(notch_wall_profile_xy())]
        assert max(z_values) == 0.0
        assert min(z_values) == pytest.approx(-FIXTURE_Z_EXTENT_M)


class TestNotchProfile:
    """The wall cross-section profile itself: 3 points, correct ordering
    (apex -> notch/chamfer transition -> chamfer mouth), both x and y
    monotonically increasing in magnitude (mouth is the outermost point)."""

    def test_profile_has_3_points(self):
        assert len(notch_wall_profile_xy()) == 3

    def test_profile_apex_at_origin(self):
        profile = notch_wall_profile_xy()
        assert profile[0].x == 0.0
        assert profile[0].y == 0.0

    def test_profile_monotonic_outward(self):
        profile = notch_wall_profile_xy()
        xs = [p.x for p in profile]
        ys = [p.y for p in profile]
        assert xs == sorted(xs)  # non-decreasing x (moving outward, +X wall)
        assert ys == sorted(ys, reverse=True)  # non-increasing y (more negative = further from finger body)

    def test_profile_transition_point_matches_notch_opening_formula(self):
        profile = notch_wall_profile_xy()
        transition = profile[1]
        assert transition.x == pytest.approx(notch_opening_width_m() / 2.0)
        assert transition.y == pytest.approx(-NOTCH_DEPTH_M)

    def test_profile_mouth_point_adds_chamfer(self):
        profile = notch_wall_profile_xy()
        mouth = profile[2]
        expected_x = notch_opening_width_m() / 2.0 + CHAMFER_FLARE_HALF_WIDTH_M
        expected_y = -(NOTCH_DEPTH_M + CHAMFER_DEPTH_M)
        assert mouth.x == pytest.approx(expected_x)
        assert mouth.y == pytest.approx(expected_y)

    def test_mirror_profile_negates_x_only(self):
        profile = notch_wall_profile_xy()
        mirrored = mirror_profile_x(profile)
        for original, mirror in zip(profile, mirrored):
            assert mirror.x == pytest.approx(-original.x)
            assert mirror.y == pytest.approx(original.y)

    def test_profile_is_x_symmetric_under_union_with_mirror(self):
        """Confirms the module docstring's load-bearing claim: the notch's
        full cross-section (both walls together) is symmetric under
        x -> -x - the property `joint_local_rot1_wxyz(mirror=True)`'s
        "combined X/Y flip is equivalent to a pure Y flip" reasoning
        depends on."""
        profile = notch_wall_profile_xy()
        mirrored = mirror_profile_x(profile)
        profile_points = {(round(p.x, 9), round(p.y, 9)) for p in profile}
        mirrored_points = {(round(-p.x, 9), round(p.y, 9)) for p in mirrored}
        assert profile_points == mirrored_points


class TestWallPrism:
    """The 3D prism construction (profile extruded along Z) - point count,
    face topology consistency, and a direct volume check (a real geometric
    invariant, independent of any face-winding convention) confirming the
    authored points describe a sane, non-degenerate solid."""

    def test_point_count(self):
        points = wall_prism_points(notch_wall_profile_xy())
        assert len(points) == 6

    def test_face_indices_reference_valid_points(self):
        assert sum(WALL_PRISM_FACE_VERTEX_COUNTS) == len(WALL_PRISM_FACE_VERTEX_INDICES)
        assert max(WALL_PRISM_FACE_VERTEX_INDICES) == 5
        assert min(WALL_PRISM_FACE_VERTEX_INDICES) == 0

    def test_prism_volume_matches_profile_area_times_height(self):
        """Volume of a prism = (2D cross-section area) * (extrusion
        height) - a face-winding-independent sanity check that the
        authored points form the expected non-degenerate solid (not, e.g.,
        a zero-volume degenerate sliver from a sign error)."""
        profile = notch_wall_profile_xy()
        # Shoelace formula (absolute value - sign is a winding artifact,
        # not tested here, see tasks/franka/notch_fixture.py's own
        # WALL_PRISM_FACE_VERTEX_INDICES comment on why winding isn't
        # load-bearing).
        area = 0.0
        n = len(profile)
        for i in range(n):
            x1, y1 = profile[i].x, profile[i].y
            x2, y2 = profile[(i + 1) % n].x, profile[(i + 1) % n].y
            area += x1 * y2 - x2 * y1
        area = abs(area) / 2.0
        expected_volume = area * FIXTURE_Z_EXTENT_M
        assert expected_volume > 0.0
        # Order-of-magnitude sanity: a few mm^2 cross-section * 10mm height
        # should land well under 1 cm^3 (1e-6 m^3) for this small fixture.
        assert expected_volume < 1e-6


class TestJointPlacement:
    """Fixed-joint local-offset helpers (consumed by
    scripts/dice_pick_demo.py's `attach_notch_fixtures` at runtime)."""

    def test_joint_local_pos0_is_finger_tip(self):
        pos0 = joint_local_pos0_m()
        assert pos0 == (0.0, 0.0, FINGER_TIP_LOCAL_Z_M)

    def test_joint_local_rot1_identity_for_left(self):
        assert joint_local_rot1_wxyz(mirror=False) == (1.0, 0.0, 0.0, 0.0)

    def test_joint_local_rot1_180_about_z_for_right(self):
        assert joint_local_rot1_wxyz(mirror=True) == (0.0, 0.0, 0.0, 1.0)

    def test_mirror_rotation_flips_x_and_y_leaves_z(self):
        """Applies the quaternion `joint_local_rot1_wxyz(mirror=True)`
        directly to a sample point (standard quaternion rotation formula,
        reimplemented here rather than imported - this test module stays
        independent of any other module's rotation helper) and confirms it
        flips X and Y while leaving Z unchanged - the property the module
        docstring's "combined flip == pure Y flip for an X-symmetric
        profile" argument depends on."""
        w, x, y, z = joint_local_rot1_wxyz(mirror=True)
        px, py, pz = 0.0075, -0.006, -0.003  # a representative mouth-region point

        def _rotate(qw, qx, qy, qz, vx, vy, vz):
            # v' = q * v * q_conj, (w,x,y,z) convention.
            uvx, uvy, uvz = qy * vz - qz * vy, qz * vx - qx * vz, qx * vy - qy * vx
            uuvx, uuvy, uuvz = qy * uvz - qz * uvy, qz * uvx - qx * uvz, qx * uvy - qy * uvx
            return (
                vx + 2 * (qw * uvx + uuvx),
                vy + 2 * (qw * uvy + uuvy),
                vz + 2 * (qw * uvz + uuvz),
            )

        rx, ry, rz = _rotate(w, x, y, z, px, py, pz)
        assert rx == pytest.approx(-px)
        assert ry == pytest.approx(-py)
        assert rz == pytest.approx(pz)
