"""Sim-independent unit tests for tasks/franka/antipodal_edge_grasp.py's pure
geometry (no isaaclab/pxr import needed - pure numpy/scipy). Run via:
/home/saps/IsaacLab/_isaac_sim/python.sh -m pytest tests/test_antipodal_edge_grasp.py -v -p no:launch_testing
(plain python3/pytest lacks numpy/scipy consistently across this repo's
python installs - see project_pytest-needs-isaac-sim-python memory; this
module happens to be pure numpy/scipy so it WOULD also run under a bare
python3 with those installed, but the canonical invocation stays consistent
with every other tests/ file in this repo)."""

import numpy as np
import pytest

from tasks.franka.antipodal_edge_grasp import (
    CANONICAL_DOWN_QUAT_WXYZ,
    PANDA_HAND_LOCAL_SQUEEZE_AXIS,
    GraspAxis,
    axis_angle_to_quat,
    best_reachable_pair,
    classify_down_face,
    edge_pair_grasp_axes,
    extract_polyhedron_vertices,
    quat_mul,
    quat_to_rot_matrix,
    regular_tetrahedron_vertices,
    stage_waypoints_world,
)

# Spec's own desk-check edge length (docs/superpowers/specs/2026-07-13-d4-edge
# -grasp-rung0-design.md's "Desk check" section: a=30.3mm -> span a/sqrt(2)
# ~21.4mm, tilt arctan(1/sqrt(2))~35.26deg) - used here as the KNOWN-ANSWER
# test fixture per the plan's own Task 1 bullet ("known resting quats ->
# expected 35.26 deg tilt, 21.4mm span"). NOTE (Task 0 report): the actual
# d4 die mesh's OWN measured edge length is ~23.59mm, not 30.3mm - a real,
# flagged discrepancy from the spec's assumed value (see report). This
# constant is a pure geometry fixture (any edge length gives the same
# 35.26deg tilt - that's a universal regular-tetrahedron constant - and the
# same a/sqrt(2) span relationship), independent of that discrepancy.
_SPEC_EDGE_M = 0.0303
_EXPECTED_TILT_DEG = 35.264389682754654  # arctan(1/sqrt(2))
_EXPECTED_SPAN_M = _SPEC_EDGE_M / np.sqrt(2.0)  # ~0.021425


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(w,x,y,z) quat rotating unit vector a onto unit vector b - test-only
    helper for constructing "face X is down" fixture quaternions."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot > 1.0 - 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if dot < -1.0 + 1e-9:
        # 180 degrees - pick any perpendicular axis
        perp = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(a, np.array([0.0, 1.0, 0.0]))
        return axis_angle_to_quat(perp, np.pi)
    axis = np.cross(a, b)
    angle = float(np.arccos(dot))
    return axis_angle_to_quat(axis, angle)


class TestKnownAnswerGeometry:
    """35.26deg tilt / 21.4mm span for the spec's own a=30.3mm fixture,
    resting face-down at identity orientation (the construction in
    `regular_tetrahedron_vertices` already places face (0,1,2) in the z=0
    plane, apex (vertex 3) up)."""

    def setup_method(self):
        self.vertices = regular_tetrahedron_vertices(_SPEC_EDGE_M)
        # Identity orientation - the die's resting quat, NOT the gripper's
        # canonical quat (unrelated frames) - `regular_tetrahedron_vertices`
        # already places face (0,1,2) in the z=0 plane at identity.
        self.axes = edge_pair_grasp_axes(self.vertices, np.array([1.0, 0.0, 0.0, 0.0]))

    def test_three_pairs_returned(self):
        assert len(self.axes) == 3

    def test_tilt_matches_arctan_one_over_sqrt2(self):
        for axis in self.axes:
            tilt_deg = np.degrees(axis.tilt_from_horizontal_rad)
            assert abs(tilt_deg - _EXPECTED_TILT_DEG) < 0.01, (
                f"pair {axis.pair_id}: expected tilt {_EXPECTED_TILT_DEG:.3f} deg, got {tilt_deg:.3f} deg"
            )

    def test_span_matches_a_over_sqrt2(self):
        for axis in self.axes:
            assert abs(axis.span_m - _EXPECTED_SPAN_M) < 1e-6, (
                f"pair {axis.pair_id}: expected span {_EXPECTED_SPAN_M * 1000:.3f}mm, "
                f"got {axis.span_m * 1000:.3f}mm"
            )

    def test_opposite_edge_is_apex_plus_remaining_face_vertex(self):
        """Each bottom edge (i, j) of the down face should pair with
        (k, apex) where k is the down face's third vertex and apex=3 (the
        vertex outside the down face, per `regular_tetrahedron_vertices`'
        own construction)."""
        for axis in self.axes:
            i, j = axis.bottom_edge
            k, apex = axis.opposite_edge
            assert apex == 3
            assert {i, j, k} == {0, 1, 2}

    def test_wrist_yaws_are_120_degrees_apart(self):
        """The 3 pairs are related by the tetrahedron's own 3-fold symmetry
        about its vertical axis - their required wrist yaws should be
        evenly spaced 120deg apart (mod sign/wrap)."""
        yaws_deg = sorted(np.degrees(a.wrist_yaw_rad) for a in self.axes)
        assert len(yaws_deg) == 3
        diffs = [yaws_deg[1] - yaws_deg[0], yaws_deg[2] - yaws_deg[1]]
        for d in diffs:
            assert abs(d - 60.0) < 0.1 or abs(d - 120.0) < 0.1, f"unexpected yaw spacing: {yaws_deg}"


class TestSqueezeAxisAlignment:
    """The self-verification assertion inside edge_pair_grasp_axes already
    guards this at call time (raises AssertionError on failure) - these
    tests independently re-derive the same check from the OUTSIDE, so a
    change that silently weakens/removes that internal guard would still be
    caught here."""

    def test_squeeze_axis_matches_computed_axis_dir(self):
        vertices = regular_tetrahedron_vertices(_SPEC_EDGE_M)
        axes = edge_pair_grasp_axes(vertices, np.array([1.0, 0.0, 0.0, 0.0]))
        for axis in axes:
            rot = quat_to_rot_matrix(axis.grasp_quat_wxyz)
            squeeze_world = rot @ PANDA_HAND_LOCAL_SQUEEZE_AXIS
            alignment = abs(float(np.dot(squeeze_world, axis.axis_dir)))
            assert alignment > np.cos(np.radians(1.0)), (
                f"pair {axis.pair_id}: squeeze axis {squeeze_world} vs axis_dir {axis.axis_dir}"
            )

    def test_grasp_quat_reduces_to_canonical_at_zero_tilt_zero_yaw(self):
        """Sanity check on the quaternion composition itself (not the
        tetrahedron geometry): yaw=0, tilt=0 about any axis should return
        exactly the canonical straight-down quat."""
        q_yaw = axis_angle_to_quat(np.array([0.0, 0.0, 1.0]), 0.0)
        q_after_yaw = quat_mul(q_yaw, CANONICAL_DOWN_QUAT_WXYZ)
        q_tilt = axis_angle_to_quat(np.array([1.0, 0.0, 0.0]), 0.0)
        q_final = quat_mul(q_tilt, q_after_yaw)
        assert np.allclose(np.abs(q_final), np.abs(CANONICAL_DOWN_QUAT_WXYZ), atol=1e-9)


class TestAllFourRestingFaces:
    """Coverage for all 4 possible resting faces of the tetrahedron (research
    doc confound 3 / plan Task 1 bullet: "all-4-resting-faces coverage") -
    not just the identity-orientation case where face (0,1,2) happens to
    already be down."""

    @pytest.mark.parametrize("down_face", [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])
    def test_each_face_classified_correctly_when_resting(self, down_face):
        vertices = regular_tetrahedron_vertices(_SPEC_EDGE_M)
        centroid = vertices.mean(axis=0)
        p0, p1, p2 = vertices[list(down_face)]
        normal = np.cross(p1 - p0, p2 - p0)
        normal = normal / np.linalg.norm(normal)
        face_centroid = (p0 + p1 + p2) / 3.0
        if np.dot(normal, face_centroid - centroid) < 0:
            normal = -normal  # outward

        # Rotate so this face's outward normal maps to world -z (resting on it).
        quat = _rotation_between(normal, np.array([0.0, 0.0, -1.0]))

        classified_face, _classified_normal = classify_down_face(
            vertices @ quat_to_rot_matrix(quat).T, [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
        )
        assert set(classified_face) == set(down_face), f"expected {down_face}, classified {classified_face}"

        # And the full pipeline should succeed end-to-end (3 axes, correct
        # tilt) for every one of the 4 resting orientations, not just the
        # identity-quat case.
        axes = edge_pair_grasp_axes(vertices, quat)
        assert len(axes) == 3
        for axis in axes:
            assert abs(np.degrees(axis.tilt_from_horizontal_rad) - _EXPECTED_TILT_DEG) < 0.01
            assert abs(axis.span_m - _EXPECTED_SPAN_M) < 1e-6


class TestNonSettledRejection:
    """A pose where no face is close to resting flat (balanced on an edge,
    or an arbitrary tumbled orientation) must raise ValueError, not silently
    pick the closest face."""

    def test_edge_balanced_pose_rejected(self):
        """Rotate the identity pose (face (0,1,2) down) by 60deg about a
        horizontal axis - adjacent face normals are ~109.47deg apart
        (regular tetrahedron dihedral geometry), so a 60deg tilt lands
        solidly outside the classifier's 10deg tolerance for every face."""
        vertices = regular_tetrahedron_vertices(_SPEC_EDGE_M)
        quat = axis_angle_to_quat(np.array([1.0, 0.0, 0.0]), np.radians(60.0))
        with pytest.raises(ValueError, match="not resting flat"):
            edge_pair_grasp_axes(vertices, quat)

    def test_vertex_up_balanced_pose_rejected(self):
        """A pose exactly balanced on a single vertex (the down-face's
        antipode) - not really achievable in real physics but a clean
        analytic non-settled case: rotate 180deg so the ORIGINAL apex
        points straight down instead of up (no face's normal is anywhere
        near straight down in this orientation)."""
        vertices = regular_tetrahedron_vertices(_SPEC_EDGE_M)
        quat = axis_angle_to_quat(np.array([1.0, 0.0, 0.0]), np.pi)
        with pytest.raises(ValueError, match="not resting flat"):
            edge_pair_grasp_axes(vertices, quat)


class TestNotImplementedForNonTetrahedra:
    def test_five_vertex_polyhedron_raises_not_implemented(self):
        """A degenerate 5-point convex "polyhedron" (down face + 2 remaining
        vertices) should raise NotImplementedError, not silently guess an
        opposite-edge pairing (module docstring's honest generality scope)."""
        vertices = np.array(
            [
                [0.0, 1.0, 0.0],
                [-1.0, -0.5, 0.0],
                [1.0, -0.5, 0.0],
                [0.3, 0.0, 1.0],
                [-0.3, 0.2, 1.2],
            ]
        )
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        with pytest.raises(NotImplementedError):
            edge_pair_grasp_axes(vertices, quat)


class TestExtractPolyhedronVertices:
    """extract_polyhedron_vertices' k-means clustering, exercised against a
    synthetic bevelled-mesh-like point cloud (small clusters of noisy points
    around 4 known corners) rather than a real USD fixture (this module has
    no pxr dependency - see Task 0 report for the real-mesh cross-check,
    done separately with actual pxr USD data, not repeated here)."""

    def test_recovers_known_corners_from_noisy_clusters(self):
        rng = np.random.default_rng(42)
        true_corners = regular_tetrahedron_vertices(0.030)
        noisy_points = []
        for corner in true_corners:
            noisy_points.append(corner + rng.normal(scale=0.0005, size=(30, 3)))
        noisy_points = np.concatenate(noisy_points, axis=0)

        recovered = extract_polyhedron_vertices(noisy_points, num_vertices=4, seed=0)

        # Each recovered center should match exactly one true corner within
        # the injected noise's own scale.
        for center in recovered:
            dists = np.linalg.norm(true_corners - center, axis=1)
            assert dists.min() < 0.002, f"recovered center {center} not close to any true corner"


class TestStageWaypoints:
    def test_grasp_waypoint_backs_off_along_approach_axis_for_standoff(self):
        vertices = regular_tetrahedron_vertices(_SPEC_EDGE_M)
        axes = edge_pair_grasp_axes(vertices, np.array([1.0, 0.0, 0.0, 0.0]))
        best = best_reachable_pair(axes, current_wrist_yaw_rad=0.0)
        object_pos = np.array([0.5, 0.1, 0.02])
        standoff = 0.05
        approach_wp, grasp_wp = stage_waypoints_world(
            best, object_pos, hand_to_fingertip_offset_m=0.1034, standoff_m=standoff
        )
        # approach waypoint should be exactly `standoff` further from the
        # grasp waypoint along the (negative) approach direction.
        dist = np.linalg.norm(approach_wp - grasp_wp)
        assert abs(dist - standoff) < 1e-9

    def test_best_reachable_pair_picks_smallest_yaw_delta(self):
        vertices = regular_tetrahedron_vertices(_SPEC_EDGE_M)
        axes = edge_pair_grasp_axes(vertices, np.array([1.0, 0.0, 0.0, 0.0]))
        target_yaw = axes[0].wrist_yaw_rad + 0.01  # nudge toward pair 0
        best = best_reachable_pair(axes, current_wrist_yaw_rad=target_yaw)
        assert best.pair_id == axes[0].pair_id
