# tasks/franka/antipodal_edge_grasp.py
"""Shape-general "antipodal edge-pair axis from mesh + pose" geometry for the
d4 edge-grasp rung-0 scripted pick (see
docs/superpowers/specs/2026-07-13-d4-edge-grasp-rung0-design.md and
docs/superpowers/plans/2026-07-13-d4-edge-grasp-rung0.md's Task 1).

Pure numpy/scipy module - NO isaaclab/pxr/torch imports, no sim dependency,
so it (and its unit tests) run standalone via plain pytest. This module is
imported BY scripts/dice_pick_demo.py, never the reverse - it has no
knowledge of the demo script's own staged-IK machinery beyond the geometric
primitives (grasp orientation quaternion, waypoint positions) a caller needs
to drive it.

Design (see Task 0's report, `.superpowers/sdd/task-d4-rung0-tasks01-report.md`,
for the full derivation/verification of every constant and sign convention
below - this docstring summarizes, doesn't re-derive):

  1. `classify_down_face` — given a convex polyhedron's vertices already
     rotated into world-ALIGNED orientation (not translated) and its face
     list, finds whichever face's outward normal is closest to straight
     down ("resting flat on this face"); raises if none is close enough
     (non-settled rejection - balanced on an edge/vertex).
  2. `edge_pair_grasp_axes` — for a polyhedron whose down-face has exactly
     one vertex "left over" (the tetrahedron/d4 case: 3 face vertices + 1
     apex = 4 total), enumerates the 3 (bottom-edge, opposite-edge) pairs,
     computes each pair's common-perpendicular axis, and derives the
     panda_hand grasp orientation that aligns the gripper's own jaw-closing
     axis with it (grasp_quat_wxyz) plus a signed wrist yaw (for
     reachability scoring across the 3 candidates - see
     `best_reachable_pair`).
  3. `stage_waypoints_world` — turns one `GraspAxis` (plus the object's
     world translation) into the two staged panda_hand targets a caller
     drives to: an approach standoff and the actual grasp-height target
     (already correcting panda_hand-origin -> fingertip-pinch-point offset,
     the same correction scripts/dice_pick_demo.py's canonical straight-down
     path already applies via its own `_VALIDATED_HAND_TO_PINCH_POINT_Z`).

Generality honestly scoped (not overclaimed): steps 1 and 3, and the
per-pair axis/orientation math inside step 2, are written for ANY convex
polyhedron (arbitrary vertex/face count) - nothing here is d4-specific by
name. The one place this module is NOT fully N-vertex-general is the
opposite-edge-PAIRING rule itself: "the bottom edge (v_i, v_j)'s opposite
edge is (v_k, apex)" only has an unambiguous meaning when the resting face's
complement is a single vertex (exactly the tetrahedron's own structure - 4
vertices total). `edge_pair_grasp_axes` raises `NotImplementedError` rather
than guess for any polyhedron where that doesn't hold; extending to shapes
with a genuine edge-adjacency graph (d8/d10/d12/d20) is future work, not
attempted here (this rung's only caller is the d4).
"""

import dataclasses
from itertools import combinations

import numpy as np

# ---------------------------------------------------------------------------
# Quaternion / rotation utilities (w, x, y, z convention throughout, matching
# scripts/dice_pick_demo.py's own `_quat_to_rot_matrix` convention).
# Reimplemented here rather than imported - tasks/ must stay free of any
# scripts/ dependency (tasks/ is imported BY scripts/, never the reverse).
# ---------------------------------------------------------------------------


def quat_to_rot_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix for a (w, x, y, z) quaternion (body -> world)."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
        ]
    )


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return q / np.linalg.norm(q)


def axis_angle_to_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    """(w, x, y, z) quaternion for a right-handed rotation of `angle`
    radians about `axis` (need not be pre-normalized)."""
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = axis / norm
    half = angle / 2.0
    return np.array([np.cos(half), *(axis * np.sin(half))])


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 (x) q2, (w, x, y, z). When both represent
    body->world rotations, R(q1 (x) q2) = R(q1) @ R(q2) - i.e. q1 is applied
    ON TOP OF (after, in world frame) q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


# ---------------------------------------------------------------------------
# Franka-hand-specific constants (measured, not guessed - see report).
# ---------------------------------------------------------------------------

CANONICAL_DOWN_QUAT_WXYZ = np.array([0.0, 1.0, 0.0, 0.0])
"""Canonical straight-down panda_hand orientation - matches
scripts/dice_pick_demo.py's own `canonical_down_quat_w`. Local +Z (finger-
pointing/approach axis) -> world [0, 0, -1]; R = diag(1, -1, -1)."""

PANDA_HAND_LOCAL_SQUEEZE_AXIS = np.array([0.0, 1.0, 0.0])
"""panda_hand's own local +Y is the jaw-closing (squeeze) axis - derived
(not guessed) from the Franka Panda USD's own joint kinematics (Task 0,
`panda_instanceable.usd` fetched directly from the Isaac Nucleus/S3 asset
store and inspected via plain pxr, no SimulationApp): panda_finger_joint1's
`physics:axis=X` is defined in the JOINT's own local frame, reached from
panda_hand's body frame via `physics:localRot0=(0.7071,0,0,0.7071)` (w,x,y,z)
- a +90deg rotation about the shared hand/finger body frame's local Z (the
fingers' own `xformOp:orient` was measured IDENTICAL to panda_hand's in the
asset's rest pose, i.e. zero relative rotation between hand and finger
bodies). Rotating the joint-local +X axis back through that 90deg-about-Z
rotation gives local +Y in panda_hand's own frame. `panda_finger_joint2`'s
`localRot0=(0.7071,0,0,-0.7071)` gives local -Y, confirming the two fingers
open symmetrically along +-Y (not +-X, which is a common but wrong
assumption for this asset). Under CANONICAL_DOWN_QUAT_WXYZ, local +Y ->
world [0,-1,0] - i.e. every existing straight-down grasp in this demo
already closes along world Y; this constant lets `edge_pair_grasp_axes`
compute the world-frame yaw needed to instead align the squeeze axis with a
tilted edge-pair axis."""

PANDA_HAND_LOCAL_APPROACH_AXIS = np.array([0.0, 0.0, 1.0])
"""panda_hand's local +Z, the finger-pointing/approach axis (matches
scripts/dice_pick_demo.py's own `_print_live_hand_orientation` usage)."""

_DOWN_FACE_NORMAL_TOL_RAD = np.radians(10.0)
"""How far a face's outward normal may sit from world -Z and still count as
"resting flat on this face" - generous relative to PhysX settling jitter,
while still cleanly rejecting a genuinely non-flat/balanced pose (which for
a regular tetrahedron differs from every face normal by tens of degrees at
minimum - see the non-settled-rejection unit test)."""


@dataclasses.dataclass
class GraspAxis:
    """One candidate opposite-edge-pair grasp for a resting convex
    polyhedron. All point/vector fields are in the WORLD-ALIGNED frame
    (the polyhedron's local vertices rotated by `resting_quat`) but NOT
    translated by the object's own world position - callers add that
    translation themselves (keeps this module position-agnostic, matching
    the plan's `edge_pair_grasp_axes(mesh_vertices, resting_quat)`
    signature, which takes no translation input)."""

    pair_id: int
    bottom_edge: tuple[int, int]
    opposite_edge: tuple[int, int]
    lower_point: np.ndarray  # midpoint of bottom_edge (world-aligned, origin-relative)
    upper_point: np.ndarray  # midpoint of opposite_edge
    axis_dir: np.ndarray  # unit vector, lower_point -> upper_point
    span_m: float  # |upper_point - lower_point|
    tilt_from_horizontal_rad: float
    grasp_quat_wxyz: np.ndarray  # world-aligned tilted panda_hand orientation
    wrist_yaw_rad: float  # signed yaw (about world Z, from canonical) needed to reach this pair


def extract_polyhedron_vertices(mesh_points: np.ndarray, num_vertices: int, seed: int = 0) -> np.ndarray:
    """k-means-clusters a (possibly bevelled/dense, duplicate-corner) mesh
    point cloud down to its `num_vertices` true corners. Needed because a
    real exported die mesh (see report - the d4's own USD has 244 points for
    a 4-vertex shape, from small manufacturing-style bevels/chamfers) is not
    already a clean N-vertex list. Returns an (num_vertices, 3) array of
    cluster centroids. Callers should sanity-check the resulting edge
    lengths are consistent (see this module's own d4 caller / report) -
    this function doesn't itself validate the input is actually close to a
    regular/consistent shape."""
    from scipy.cluster.vq import kmeans2

    pts = np.asarray(mesh_points, dtype=float)
    centers, _labels = kmeans2(pts, num_vertices, minit="++", seed=seed)
    return centers


def _all_triangle_faces(num_vertices: int) -> list[tuple[int, int, int]]:
    """Every 3-vertex combination. For a tetrahedron (num_vertices==4) this
    IS exactly the true face list (4 faces, each the complement of one
    vertex) - not a general convex-hull face-finder (see module docstring's
    generality note)."""
    return list(combinations(range(num_vertices), 3))


def classify_down_face(
    vertices_world_aligned: np.ndarray, faces: list[tuple[int, int, int]]
) -> tuple[tuple[int, int, int], np.ndarray]:
    """Returns (face_indices, outward_normal) for whichever face's outward
    normal is closest to world -Z ("resting flat on this face"). Raises
    ValueError if the closest face is still more than
    `_DOWN_FACE_NORMAL_TOL_RAD` from straight down (non-settled rejection)."""
    vertices_world_aligned = np.asarray(vertices_world_aligned, dtype=float)
    centroid = vertices_world_aligned.mean(axis=0)
    down = np.array([0.0, 0.0, -1.0])

    best_angle = None
    best_face = None
    best_normal = None
    for face in faces:
        p0, p1, p2 = vertices_world_aligned[list(face)]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue  # degenerate (collinear) triple - not a real face
        normal = normal / norm
        face_centroid = (p0 + p1 + p2) / 3.0
        if np.dot(normal, face_centroid - centroid) < 0:
            normal = -normal  # orient outward (away from the polyhedron's own centroid)
        angle = float(np.arccos(np.clip(np.dot(normal, down), -1.0, 1.0)))
        if best_angle is None or angle < best_angle:
            best_angle, best_face, best_normal = angle, face, normal

    if best_face is None or best_angle > _DOWN_FACE_NORMAL_TOL_RAD:
        closest_deg = np.degrees(best_angle) if best_angle is not None else float("nan")
        raise ValueError(
            f"No face's outward normal is within {np.degrees(_DOWN_FACE_NORMAL_TOL_RAD):.1f} deg "
            f"of straight down (closest: {closest_deg:.1f} deg) - die is not resting flat on any "
            "face (balanced on an edge/vertex, or resting_quat is stale/wrong)."
        )
    return best_face, best_normal


def edge_pair_grasp_axes(mesh_vertices_local: np.ndarray, resting_quat_wxyz: np.ndarray) -> list[GraspAxis]:
    """Shape-general opposite-edge-pair grasp-axis computation (see module
    docstring for scope). `mesh_vertices_local`: (N, 3) array of the
    polyhedron's TRUE (already-deduplicated - see `extract_polyhedron_vertices`)
    vertices in its own local/body frame. `resting_quat_wxyz`: the object's
    current world orientation (w, x, y, z), e.g. read from sim ground truth.

    Returns one `GraspAxis` per bottom edge of whichever face is resting on
    the table (3 for a tetrahedron), each carrying a self-verified
    `grasp_quat_wxyz` (see the alignment assertion below - this recomputes
    and checks the squeeze axis against `axis_dir` every call, not just in
    a unit test, given how easy a sign/composition bug is to get wrong
    here - see report). Raises `ValueError` if the object isn't resting
    flat on any face, `NotImplementedError` if the resting face's
    complement isn't a single vertex (see module docstring)."""
    vertices_local = np.asarray(mesh_vertices_local, dtype=float)
    num_vertices = len(vertices_local)

    rot = quat_to_rot_matrix(quat_normalize(np.asarray(resting_quat_wxyz, dtype=float)))
    vertices_world_aligned = vertices_local @ rot.T

    faces = _all_triangle_faces(num_vertices)
    down_face, _down_normal = classify_down_face(vertices_world_aligned, faces)

    remaining = [i for i in range(num_vertices) if i not in down_face]
    if len(remaining) != 1:
        raise NotImplementedError(
            f"edge_pair_grasp_axes' opposite-edge-pair rule needs exactly 1 vertex outside the "
            f"resting face (the tetrahedron/d4 case); got {len(remaining)} remaining vertices for "
            f"a {num_vertices}-vertex polyhedron. Not implemented for other shapes - see module "
            "docstring's generality note."
        )
    apex = remaining[0]

    bottom_edges = [
        (down_face[0], down_face[1]),
        (down_face[1], down_face[2]),
        (down_face[2], down_face[0]),
    ]

    squeeze_canonical_world = quat_to_rot_matrix(CANONICAL_DOWN_QUAT_WXYZ) @ PANDA_HAND_LOCAL_SQUEEZE_AXIS

    axes: list[GraspAxis] = []
    for pair_id, (i, j) in enumerate(bottom_edges):
        k = next(v for v in down_face if v not in (i, j))
        opposite_edge = (k, apex)

        lower = (vertices_world_aligned[i] + vertices_world_aligned[j]) / 2.0
        upper = (vertices_world_aligned[k] + vertices_world_aligned[apex]) / 2.0
        delta = upper - lower
        span = float(np.linalg.norm(delta))
        if span < 1e-9:
            raise ValueError(f"pair {pair_id}: degenerate zero-length axis (bottom_edge={i, j}).")
        axis_dir = delta / span
        if axis_dir[2] < 0:
            # Shouldn't happen for a genuinely resting-flat tetrahedron (the
            # opposite edge sits on the apex side, elevated above the table)
            # - flip defensively rather than silently produce a
            # downward-pointing axis.
            axis_dir = -axis_dir

        tilt = float(np.arcsin(np.clip(axis_dir[2], -1.0, 1.0)))

        horiz = axis_dir.copy()
        horiz[2] = 0.0
        h_norm = float(np.linalg.norm(horiz))
        if h_norm < 1e-9:
            raise ValueError(f"pair {pair_id}: grasp axis has ~zero horizontal component (near-vertical axis).")
        h_hat = horiz / h_norm

        # Bottom-edge world direction, sign-resolved so that rotating h_hat
        # by +tilt about it moves toward +z (Rodrigues' formula: need
        # (edge_dir x h_hat).z > 0 - see report's derivation).
        edge_dir = vertices_world_aligned[j] - vertices_world_aligned[i]
        edge_dir = edge_dir / np.linalg.norm(edge_dir)
        if np.cross(edge_dir, h_hat)[2] < 0:
            edge_dir = -edge_dir

        # Yaw: rotate the canonical (fixed, world-horizontal) squeeze axis
        # onto +-h_hat about world Z, picking whichever sign needs less
        # rotation (the squeeze axis is a line, not a directed vector - the
        # gripper doesn't care which physical jaw ends up on which side).
        def _signed_yaw(target_h: np.ndarray) -> float:
            cross_z = squeeze_canonical_world[0] * target_h[1] - squeeze_canonical_world[1] * target_h[0]
            dot = squeeze_canonical_world[0] * target_h[0] + squeeze_canonical_world[1] * target_h[1]
            return float(np.arctan2(cross_z, dot))

        yaw_pos = _signed_yaw(h_hat)
        yaw_neg = _signed_yaw(-h_hat)
        yaw = yaw_pos if abs(yaw_pos) <= abs(yaw_neg) else yaw_neg

        q_yaw = axis_angle_to_quat(np.array([0.0, 0.0, 1.0]), yaw)
        q_after_yaw = quat_normalize(quat_mul(q_yaw, CANONICAL_DOWN_QUAT_WXYZ))
        q_tilt = axis_angle_to_quat(edge_dir, tilt)
        q_final = quat_normalize(quat_mul(q_tilt, q_after_yaw))

        # Self-verification (cheap, always-on - not just a unit test): the
        # computed grasp_quat's own squeeze axis must reproduce +-axis_dir.
        # Deliberately a hard runtime check given how error-prone this
        # quaternion composition is to get right (see report) - a future
        # edit that breaks it should fail loudly immediately, not just when
        # someone happens to run the unit tests.
        squeeze_world = quat_to_rot_matrix(q_final) @ PANDA_HAND_LOCAL_SQUEEZE_AXIS
        alignment = abs(float(np.dot(squeeze_world, axis_dir)))
        if alignment < np.cos(np.radians(1.0)):
            raise AssertionError(
                f"pair {pair_id}: computed grasp_quat's squeeze axis {squeeze_world} does not align "
                f"with axis_dir {axis_dir} (|dot|={alignment:.5f}, need > "
                f"{np.cos(np.radians(1.0)):.5f}) - quaternion composition bug."
            )

        axes.append(
            GraspAxis(
                pair_id=pair_id,
                bottom_edge=(i, j),
                opposite_edge=opposite_edge,
                lower_point=lower,
                upper_point=upper,
                axis_dir=axis_dir,
                span_m=span,
                tilt_from_horizontal_rad=tilt,
                grasp_quat_wxyz=q_final,
                wrist_yaw_rad=yaw,
            )
        )

    return axes


def best_reachable_pair(axes: list[GraspAxis], current_wrist_yaw_rad: float = 0.0) -> GraspAxis:
    """Picks whichever of `axes` needs the smallest wrist-yaw change from
    `current_wrist_yaw_rad` (default 0.0 - the canonical/ready-to-descend
    configuration's own reference yaw). A simple, defensible proxy for the
    plan's "reachability-best pair" (full IK-reachability scoring would need
    live robot state, out of scope for this pure-geometry module - see
    report)."""
    return min(axes, key=lambda a: abs(a.wrist_yaw_rad - current_wrist_yaw_rad))


def approach_direction_world(grasp_quat_wxyz: np.ndarray) -> np.ndarray:
    """World-frame direction the panda_hand's local approach axis (+Z,
    finger-pointing) points under `grasp_quat_wxyz` - analogous to
    `[0, 0, -1]` for the canonical straight-down case."""
    return quat_to_rot_matrix(quat_normalize(grasp_quat_wxyz)) @ PANDA_HAND_LOCAL_APPROACH_AXIS


def stage_waypoints_world(
    axis: GraspAxis,
    object_world_pos: np.ndarray,
    hand_to_fingertip_offset_m: float,
    standoff_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Turns one `GraspAxis` (world-ALIGNED, origin-relative) plus the
    object's own world translation into the two staged panda_hand targets:
    (approach_waypoint, grasp_waypoint), both world-frame positions for the
    IK target's own reference point (panda_hand, NOT the fingertip pinch
    point - `hand_to_fingertip_offset_m` applies the same correction
    scripts/dice_pick_demo.py's canonical path already makes via its
    `_VALIDATED_HAND_TO_PINCH_POINT_Z`).

    approach_waypoint = grasp_waypoint backed off by `standoff_m` along
    -approach_direction (mirrors the canonical path's stage1: start outside/
    above the target, then descend in along the approach axis for stage2)."""
    object_world_pos = np.asarray(object_world_pos, dtype=float)
    grasp_center_world = object_world_pos + (axis.lower_point + axis.upper_point) / 2.0

    approach_dir = approach_direction_world(axis.grasp_quat_wxyz)
    grasp_waypoint = grasp_center_world - hand_to_fingertip_offset_m * approach_dir
    approach_waypoint = grasp_waypoint - standoff_m * approach_dir

    return approach_waypoint, grasp_waypoint


def regular_tetrahedron_vertices(edge_length_m: float) -> np.ndarray:
    """Closed-form vertices (local frame, shape (4, 3)) of a regular
    tetrahedron with the given edge length, matching the standard
    circumradius-centered construction used in this task's desk-check
    (Task 0 report) - handy as a caller-side convenience/test fixture; NOT
    used internally by `edge_pair_grasp_axes` (which takes vertices as an
    input, not this specific shape)."""
    a = float(edge_length_m)
    r = a / np.sqrt(3.0)
    h = a * np.sqrt(2.0 / 3.0)
    return np.array(
        [
            [0.0, r, 0.0],
            [-a / 2.0, -r / 2.0, 0.0],
            [a / 2.0, -r / 2.0, 0.0],
            [0.0, 0.0, h],
        ]
    )
