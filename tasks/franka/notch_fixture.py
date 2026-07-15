# tasks/franka/notch_fixture.py
"""Pure geometry for the d4 rung-1 V-notch fingertip fixture (see
docs/superpowers/specs/2026-07-15-d4-rung1-pad-geometry-design.md and
.superpowers/sdd/task-1-brief.md). NO isaaclab/pxr/torch imports, no sim
dependency - runs standalone via plain pytest, matching this repo's existing
convention for shape-general geometry modules
(tasks/franka/antipodal_edge_grasp.py, rung 0's own pure-geometry module).

This module owns:
  1. The tetrahedron-apex / grip-height trig (re-derived here exactly as
     Task 0's desk check computed it - `apex_height_m`, `grip_height_above_table_m`,
     `local_edge_length_at_height_below_apex`, `notch_opening_width_m`).
  2. The notch fixture's own cross-section profile (`notch_wall_profile_xy` +
     its mirror) and 3D prism point/face construction
     (`wall_prism_points`/`WALL_PRISM_FACE_VERTEX_COUNTS`/
     `WALL_PRISM_FACE_VERTEX_INDICES`) - consumed by
     scripts/build_notch_fixture_asset.py to author the actual USD mesh via
     plain pxr (no SimulationApp), same "no sim launch, no GPU" technique
     Task 0's own diagnostic script established.
  3. The fixed-joint placement constants/helpers
     (`FINGER_TIP_LOCAL_Z_M`, `FIXTURE_Z_EXTENT_M`, `joint_local_pos0_m`,
     `joint_local_rot1_wxyz`) consumed by scripts/dice_pick_demo.py's
     `attach_notch_fixtures` at scene-spawn runtime.

Coordinate-frame conventions (all in the FIXTURE's own local frame, which is
defined so its origin IS the fixed-joint attachment point):
  - Local Z mirrors the FINGER's own local Z (both point the same way in
    world - "down the finger, toward the tip" - for both left and right
    fingers, since a straight-down grasp holds both fingers at the same
    world orientation). The fixture occupies Z in [-FIXTURE_Z_EXTENT_M, 0]
    (Z=0 at the attachment point, extending BACKWARD toward the finger's
    existing rubber pad - i.e. it does NOT protrude past the stock
    fingertip's own measured tip (local Z=53.9mm, Task 0's measurement),
    which keeps the already-validated `_VALIDATED_HAND_TO_PINCH_POINT_Z`
    hand->fingertip-pinch-point offset (tasks/franka/dice_pick_demo.py)
    valid UNCHANGED for every die type - a deliberate design choice to
    minimize non-d4 regression risk, see this task's report).
  - Local Y is the gripper's own closing/squeeze axis. Y=0 is flush with the
    ORIGINAL stock pad's own contact plane (Task 0's measured `Y≈0` pad
    boundary); the notch fixture protrudes further OUT (toward the
    approaching die) in -Y, reaching its outermost "mouth" at
    Y = -(NOTCH_DEPTH_M + CHAMFER_DEPTH_M).
  - Local X is the pad's own width axis (unchanged meaning from Task 0's
    measurement), notch centered at X=0 (bilaterally symmetric - "both
    fingertips get an identical notch", spec's North Star call, not the
    asymmetric edge/face-ramp refinement).

Both fingers use the SAME authored mesh (one USD file, not a left/right
mirror pair) - the right finger's own Y-convention is mirrored relative to
the left's (Task 0's measurement: left pad Y in [~0, +10.9mm], right pad Y
in [-10.9mm, ~0]), so the right-side fixed joint applies a 180-degree
rotation about local Z (`joint_local_rot1_wxyz(mirror=True)`) - which flips
local X and Y but leaves Z unchanged. Because the notch's own cross-section
is X-symmetric (mirroring X of a symmetric shape reproduces the same
shape), a combined (X, Y) flip is geometrically equivalent to the intended
Y-only flip for this specific shape. This equivalence is exercised directly
in this module's own unit tests, not just asserted here.
"""

import dataclasses
import math

# ---------------------------------------------------------------------------
# Tetrahedron / grip-height trig (Task 0's desk-check re-derivation, against
# the already-double-verified real d4 mesh edge length - see
# .superpowers/sdd/task-0-report.md section 3 and
# docs/superpowers/specs/2026-07-13-d4-edge-grasp-rung0-design.md's own
# measurement. Matches tasks/franka/antipodal_edge_grasp.py's
# `regular_tetrahedron_vertices` use of the same `a*sqrt(2/3)` apex-height
# formula - not re-derived independently, same source fact.)
# ---------------------------------------------------------------------------

D4_EDGE_LENGTH_M = 0.023591
"""Measured real d4 mesh edge length (rung-0 Task 0/1, double-verified;
reused here verbatim per this task's brief - "reuse the existing edge-length
measurement... do not re-run the mesh k-means a third time")."""

NOTCH_ANGLE_DEG = 110.0
"""Internal notch angle - the tetrahedron's own dihedral-angle supplement
(`arccos(-1/3) = 109.47 deg`) rounded to a buildable value (spec's Design
section)."""

NOTCH_DEPTH_M = 0.004
"""How far (along local -Y, into the fixture) the true 110-degree notch's
own vertex sits behind its own opening plane."""

CHAMFER_DEPTH_M = 0.002
"""Additional local -Y extent of the chamfered lead-in, OUTSIDE (further
from the finger body than) the true notch's own opening - i.e. the die
meets the wider chamfer FIRST, before reaching the tighter 110-degree
notch. Total fixture -Y extent is NOTCH_DEPTH_M + CHAMFER_DEPTH_M."""

CHAMFER_FLARE_HALF_WIDTH_M = 0.002
"""Extra HALF-width (each side, symmetric) the chamfer's own outer mouth
flares beyond the true notch's own opening half-width, at the chamfer's
outermost point. Judgment call, not a spec-given closed-form (unlike
NOTCH_DEPTH_M/NOTCH_ANGLE_DEG's `opening = 2*depth*tan(angle/2)` formula) -
the spec only says "~2mm outward flare"; this module implements that as a
constant linear extra half-width at the mouth, tapering to zero extra width
at the notch/chamfer transition. See this task's report for why this
specific interpretation was chosen (no closed-form was given in the spec to
derive it from)."""


def apex_height_m(edge_length_m: float = D4_EDGE_LENGTH_M) -> float:
    """Height of a regular tetrahedron's apex above its resting base face,
    `a*sqrt(2/3)` (Task 0 report: 19.262mm for the measured d4)."""
    return edge_length_m * math.sqrt(2.0 / 3.0)


def dihedral_angle_rad() -> float:
    """A regular tetrahedron's dihedral angle along any edge, `arccos(1/3)`
    (~70.53 deg) - independent of edge length."""
    return math.acos(1.0 / 3.0)


def notch_angle_from_dihedral_rad() -> float:
    """The dihedral angle's supplement, `arccos(-1/3)` (~109.47 deg) - the
    un-rounded source value NOTCH_ANGLE_DEG (110 deg) is rounded from."""
    return math.acos(-1.0 / 3.0)


def grip_height_above_table_m(
    edge_length_m: float = D4_EDGE_LENGTH_M, below_apex_m: float = 0.010
) -> float:
    """World-frame height above the table at which the notch should grip
    the d4 - `apex_height - below_apex` (spec: ~9.3mm for the default
    10mm-below-apex target, Task 0 report re-derivation: 9.262mm)."""
    return apex_height_m(edge_length_m) - below_apex_m


def local_edge_length_at_depth_below_apex_m(
    depth_below_apex_m: float, edge_length_m: float = D4_EDGE_LENGTH_M
) -> float:
    """The tetrahedron's own local (triangular cross-section) edge length at
    a given depth below its apex - similar-triangles scaling,
    `a * depth_below_apex / apex_height` (spec: ~12.2mm at 10mm below apex,
    Task 0 report: 12.247mm)."""
    return edge_length_m * depth_below_apex_m / apex_height_m(edge_length_m)


def notch_opening_width_m(depth_m: float = NOTCH_DEPTH_M, angle_deg: float = NOTCH_ANGLE_DEG) -> float:
    """The true 110-degree notch's own opening width (at its transition from
    the chamfer, NOT including the chamfer's own extra flare) -
    `2 * depth * tan(angle/2)` (spec's own formula; Task 0 report: 11.425mm
    for depth=4mm, angle=110deg)."""
    return 2.0 * depth_m * math.tan(math.radians(angle_deg / 2.0))


# ---------------------------------------------------------------------------
# Franka fingertip pad geometry (Task 0 measurement, see
# .superpowers/sdd/task-0-report.md and scripts/_diag_franka_fingertip_geometry.py
# - RubberGray GeomSubset, local finger frame: mount origin at Z=0).
# ---------------------------------------------------------------------------

FINGER_TIP_LOCAL_Z_M = 0.0539
"""Finger-local Z of the fingertip's very tip (whole-finger mesh bbox max,
Task 0 measurement: 53.9mm from the mount origin)."""

PAD_Z_MIN_M = 0.03609
"""Finger-local Z where the RubberGray pad subset begins (Task 0
measurement: 36.09mm) - the fixture's own Z-extent must stay within
[PAD_Z_MIN_M, FINGER_TIP_LOCAL_Z_M] to avoid protruding past the stock
tip (see module docstring)."""

PAD_HALF_WIDTH_M = 0.008931
"""Half the RubberGray pad's measured local-X width (17.862mm / 2 = 8.931mm,
Task 0 measurement) - the notch's own mouth half-width (including chamfer
flare) must stay under this so the fixture doesn't overhang the stock pad's
own width."""

FIXTURE_Z_EXTENT_M = 0.010
"""How far (along local -Z, from the attachment point at Z=0) the fixture's
own wall geometry extends - chosen so `FINGER_TIP_LOCAL_Z_M - FIXTURE_Z_EXTENT_M
= 43.9mm` stays comfortably within the pad's own Z range (>= PAD_Z_MIN_M =
36.09mm), i.e. the fixture sits entirely within the region the stock pad
already occupied lengthwise - it does not extend the finger's own reach in
Z, only adds new material outward in Y (see module docstring)."""

FIXTURE_MASS_KG = 0.002
"""Small placeholder rigid-body mass (2g) for the fixture - a thin plastic/
metal wedge piece this size plausibly masses a few grams; not a
precision-critical value (same "reasonable placeholder for grasp dynamics,
not measured against a real reference" caveat as
tasks/franka/dice_scene_cfg.py's own `_DICE_MASS`), dwarfed by the finger's
own mass either way."""


# ---------------------------------------------------------------------------
# Notch cross-section profile + 3D prism construction.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class NotchProfilePoint:
    """One point of a wall's (x, y) cross-section profile, local frame."""

    x: float
    y: float


def notch_wall_profile_xy(
    notch_depth_m: float = NOTCH_DEPTH_M,
    notch_angle_deg: float = NOTCH_ANGLE_DEG,
    chamfer_depth_m: float = CHAMFER_DEPTH_M,
    chamfer_flare_half_width_m: float = CHAMFER_FLARE_HALF_WIDTH_M,
) -> list[NotchProfilePoint]:
    """The +X wall's own 3-point (x, y) cross-section profile, innermost
    (apex, shared with the -X wall) to outermost (chamfer mouth):

      P0 = apex           = (0, 0)
      P1 = notch/chamfer transition = (+half_notch_width, -notch_depth_m)
      P2 = chamfer mouth  = (+half_notch_width + chamfer_flare_half_width_m,
                             -(notch_depth_m + chamfer_depth_m))

    A 3-point polyline is trivially convex (only 4+ points could fail
    convexity), so extruding this profile along Z (`wall_prism_points`)
    always yields a valid convex prism - no separate convexity check is
    needed, but this module's own tests verify the specific numeric
    monotonicity (both x and y move outward from P0 to P2) this docstring's
    claim depends on."""
    half_notch_width = notch_opening_width_m(notch_depth_m, notch_angle_deg) / 2.0
    return [
        NotchProfilePoint(0.0, 0.0),
        NotchProfilePoint(half_notch_width, -notch_depth_m),
        NotchProfilePoint(half_notch_width + chamfer_flare_half_width_m, -(notch_depth_m + chamfer_depth_m)),
    ]


def mirror_profile_x(profile: list[NotchProfilePoint]) -> list[NotchProfilePoint]:
    """The -X wall's profile - same y values, negated x (bilateral symmetry
    about local X=0, per the spec's "both fingertips get an identical
    notch" design)."""
    return [NotchProfilePoint(-p.x, p.y) for p in profile]


def wall_prism_points(
    profile_xy: list[NotchProfilePoint], z_lo: float = -FIXTURE_Z_EXTENT_M, z_hi: float = 0.0
) -> list[tuple[float, float, float]]:
    """Extrudes a (x, y) profile along Z into a prism's 3D point list: the
    profile's own points at z_hi, then again at z_lo (matching
    scripts/build_asset.py's `_generate_wedge_usd` point-ordering
    convention exactly, so `WALL_PRISM_FACE_VERTEX_INDICES` below can mirror
    its face-winding pattern one-for-one)."""
    top = [(p.x, p.y, z_hi) for p in profile_xy]
    bottom = [(p.x, p.y, z_lo) for p in profile_xy]
    return top + bottom


# Face topology for a 3-point-profile prism (6 points: indices 0-2 = top
# ring at z_hi, 3-5 = bottom ring at z_lo) - same "cap / cap / 3 side quads"
# structure as scripts/build_asset.py's `_generate_wedge_usd`, generalized to
# this module's own (kinked, not equilateral) 3-point profile. This
# profile's own point order (apex, transition, mouth) has NEGATIVE signed
# area (CW when viewed from +Z looking toward -Z) - the opposite rotational
# sense from the wedge asset's equilateral profile (CCW from +Z) - so the
# reversed/unreversed cap assignment below is swapped relative to that
# function's own comment (top uses the reversed order, bottom the
# unreversed order, to keep each cap's outward normal pointing away from
# the solid).
#
# Winding/normal-direction correctness below is best-effort, NOT physics-
# load-bearing: PhysX's `convexHull` collision approximation (used by both
# this fixture and scripts/build_asset.py's wedge) computes the hull purely
# from the mesh's point positions, ignoring face winding/normals entirely.
# The load-bearing correctness check is the POINT positions themselves -
# this module's own tests verify the profile geometry/volume, not mesh
# winding.
WALL_PRISM_FACE_VERTEX_COUNTS = [3, 3, 4, 4, 4]
WALL_PRISM_FACE_VERTEX_INDICES = [
    0, 2, 1,  # top (z_hi, normal +z)
    3, 4, 5,  # bottom (z_lo, normal -z)
    0, 3, 4, 1,  # side 0-1
    1, 4, 5, 2,  # side 1-2
    2, 5, 3, 0,  # side 2-0
]


def joint_local_pos0_m() -> tuple[float, float, float]:
    """Fixed-joint `localPos0` (finger-side offset, in the finger's own
    local frame) - the attachment point is the finger's own tip center,
    (X=0, Y=0, Z=FINGER_TIP_LOCAL_Z_M)."""
    return (0.0, 0.0, FINGER_TIP_LOCAL_Z_M)


def joint_local_rot1_wxyz(mirror: bool) -> tuple[float, float, float, float]:
    """Fixed-joint `localRot1` (fixture-side rotation) - identity for the
    left finger (canonical mesh already protrudes in -Y, matching left's own
    "die is toward -Y" convention, Task 0 measurement), or a 180-degree
    rotation about local Z for the right finger (`mirror=True`) - flips
    local X and Y, which for this module's X-symmetric notch profile is
    geometrically equivalent to a pure Y-flip (see module docstring; this
    equivalence is exercised in this module's own tests, not just
    asserted)."""
    return (0.0, 0.0, 0.0, 1.0) if mirror else (1.0, 0.0, 0.0, 0.0)
