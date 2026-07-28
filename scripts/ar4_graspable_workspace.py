"""Compute AR4's graspable workspace directly via forward kinematics (no IK,
no physics, no Isaac Sim) and recommend a well-margined cube placement.

Background / motivation (2026-07-27, ar4-graspable-workspace-from-fk task):
this project's multi-week AR4 grasp investigation
(kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md) repeatedly found a
~9-25mm classical-IK positioning shortfall at the cube's DEFAULT position
(world (0.0, 0.275, ~0.0075) in the recentered scene grasp_demo_v2.py uses),
traced to joint_3 (the elbow, real vendor limit -89/+52 degrees) running out
of comfortable range before a vertical-wrist grasp can reach the true grasp
height there - a genuine multi-joint reachability-envelope effect, not a
solver bug (see that doc's 2026-07-27 "ar4-moveit-vs-dls-root-cause" update:
the vendor's own MoveIt config uses the same DLS/Jacobian algorithm family
this project's own IK does, decoupled from physics - not a "smarter solver").

Rather than continue tuning/searching IK at a cube position that may simply
be outside AR4's comfortable graspable envelope, this script inverts the
approach: forward-sample the arm's own 6-joint configuration space (pure FK,
zero solver risk - a config either lands somewhere or it doesn't, no local
minima possible) and directly compute where a genuinely graspable pinch pose
(near-vertical approach, correct grasp height for the current 15mm cube,
comfortable joint-limit margin on every joint) actually exists. The cube can
then be placed INSIDE that computed envelope instead of fighting IK to reach
a position chosen without regard for reachability.

Reuses (does not re-derive) tasks/ar4/fk_verification.py's vendor-URDF joint
table (DEFAULT_JOINT_TABLE) and scripts/grasp_demo_v2.py's own established
pinch-point/approach-axis convention (_EE_OFFSET=(0,0,0.036) along link_6's
local +Z, the same axis the canonical straight-down target aligns to world
-Z). Pure numpy - no isaaclab/torch import, runs on plain python3 (this
repo's toy_env/.venv has numpy+matplotlib and is used to run this script;
see module-level comment near the bottom for the exact invocation).

Two-stage sampling strategy (exploits a real kinematic decoupling, but
Stage B recomputes full 6-joint FK directly rather than trusting the
argument algebraically, to avoid a silent sign/derivation bug):

  Stage A: sample joint_2..joint_6 (5D) with joint_1 FIXED at 0.0. joint_1
  is a pure rotation about the vertical (Z) axis at the very base of the
  chain (its <origin> rotates the axis vector, but the axis is still a line
  through a point ON the Z axis - see JointSpec table) - rotating the whole
  downstream chain about Z changes a candidate's (x,y) position and azimuth
  but does NOT change its height (z), its tilt-from-vertical (the angle a
  vector makes with the Z axis is invariant under further rotation about
  that same axis), or joints 2-6's own margins. Filtering for
  height/tilt/margin at joint_1=0 is therefore representative of every
  joint_1 value - this lets Stage A search a 5D space instead of 6D.

  Stage B: for every Stage-A survivor, sweep joint_1 itself across its own
  valid (margin-respecting) range and recompute the FULL 6-joint FK at each
  step - not an analytic rotation applied after the fact. This directly
  confirms height/tilt are unchanged (rather than assuming the invariance
  argument above is bug-free) and adds joint_1's own margin at each swept
  angle, producing the full graspable (x,y) region in the base frame.

The world-frame transform is then just the robot's own known base pose
(tasks/ar4/robot_cfg.py's AR4_MK5_CFG.init_state: rot=(0,0,0,1) i.e. a
180-degree rotation about Z, no translation - ArticulationCfg's default
init_state pos is the origin and robot_cfg.py does not override it) applied
to each surviving base-frame point: world = (-x_base, -y_base, z_base).
"""

from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

_RL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _RL_ROOT)

from tasks.ar4.fk_verification import (  # noqa: E402
    DEFAULT_JOINT_TABLE,
    compute_link_pose_from_joint_values,
)

# ----------------------------------------------------------------------
# Constants (all reused from established project conventions - see
# module docstring above for exact provenance of each).
# ----------------------------------------------------------------------

# Vendor joint limits (degrees), joint_1..joint_6.
#
# Sourced from AR4 vendor annin_ar4_description's config/mk3.yaml (fetched
# 2026-07-27 directly from github.com/ycheng517/ar4_ros_driver, main branch,
# via the GitHub API + raw file content - both cross-checked). NOTE:
# mk5.yaml no longer exists in that upstream repo as of this fetch (confirmed
# via a GitHub API directory listing of annin_ar4_description/config/: only
# mk1.yaml/mk2.yaml/mk3.yaml are present) - this project's own asset was
# built from a separate desktop-local clone that apparently had mk1-mk5
# variants (see kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md's
# 2026-07-22 "ar4-tilt-fix task" Part A, which read joint_3's limit from that
# desktop-local mk5.yaml directly). mk3's joint_3 value used here
# ([-89, +52] degrees) matches that already-independently-verified number
# to 4 decimal places once converted to radians ([-1.5533, +0.9076]), and
# that same 2026-07-22 investigation explicitly checked "all 5 shipped model
# variants (mk1-mk5) - identical -89/52 limit in every one" - i.e. this
# project's own prior work already established these limits don't vary by
# model variant, so mk3's numbers for the other 5 joints are used here as
# the representative real vendor spec, not a downgrade to a different robot.
JOINT_LIMITS_DEG = {
    "joint_1": (-170.0, 170.0),
    "joint_2": (-42.0, 90.0),
    "joint_3": (-89.0, 52.0),
    "joint_4": (-180.0, 180.0),
    "joint_5": (-105.0, 105.0),
    "joint_6": (-180.0, 180.0),
}
JOINT_LIMITS_RAD = {
    name: (math.radians(lo), math.radians(hi)) for name, (lo, hi) in JOINT_LIMITS_DEG.items()
}
ARM_JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]

# Matches scripts/grasp_demo_v2.py's _EE_OFFSET exactly (link_6 -> gripper
# jaw pinch point, measured along link_6's own local +Z axis).
_EE_OFFSET_LOCAL = np.array([0.0, 0.0, 0.036])

# Matches scripts/grasp_demo_v2.py's current GRASP_AT_HEIGHT default
# (0.0105 = 3mm above the 15mm cube's resting center height of 0.0075,
# tasks/ar4/objects_cfg.py's CUBE_CFG) - the true grasp-point convention
# already established for the current cube size, not re-derived here.
GRASP_AT_HEIGHT = 0.0105

# Filter tolerances (2026-07-27 task brief's own specified ranges).
HEIGHT_TOL_M = 0.002  # +/- 2mm band around GRASP_AT_HEIGHT
TILT_TOL_DEG = 12.0  # within the brief's "<=10-15 degrees of straight-down"
MARGIN_MIN_RAD = 0.25  # ~14.3 degrees; within the brief's "0.2-0.3rad" range

# ROLL/HEADING tolerance (2026-07-27 UPDATE, ar4-graspable-roll-constraint
# task). The original sweep above constrained ONLY the approach axis's tilt
# from vertical (TILT_TOL_DEG) - a single DOF - and left the OTHER
# in-plane rotational DOF (heading/roll about that same approach axis)
# completely unconstrained. The live confirmation this omission fed into
# (kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md's 2026-07-27
# "ar4-graspable-workspace-from-fk task" UPDATE) found the gripper colliding
# with the cube at ~52-61N even while nominally OPEN (28mm aperture vs a
# 15mm cube) - the jaw-slide axis was pointed at some arbitrary in-plane
# heading, not straddling the cube's own square footprint. This constant
# bounds that heading error (see _jaw_heading_offset_deg below for the exact
# quantity) to within a tight tolerance of being parallel to either world X
# or world Y - the only two headings that let a flat-jaw gripper close on an
# axis-aligned, non-rotated cube (tasks/ar4/objects_cfg.py's CUBE_CFG) without
# clipping a corner. Chosen equal to TILT_TOL_DEG (not looser): the tilt
# tolerance was already established as "tight enough to still be a vertical
# approach" for this same grasp geometry, and roll misalignment is at least
# as sensitive (a jaw-slide axis off-heading by 12 degrees on a 15mm cube
# grazes a corner well before an equivalent tilt error would); if the
# post-constraint sweep turns out empty or too sparse at this value, WIDEN
# it deliberately and note the widening here, rather than silently loosening
# without comment.
ROLL_TOL_DEG = 12.0

# GROUND CLEARANCE constraint (2026-07-28, ar4-joint2-ground-clearance-fix
# task). Root cause of the "joint_2 physically cannot be driven past
# ~59.0-59.1deg" wall found by the 2026-07-28 joint-tracking-closed-loop-fix
# task: direct pxr/UsdPhysics inspection of the built USD (this same task,
# scripts/_diag_ar4_joint2_limit_raw_usd_only.py) confirmed joint_2's own
# authored physics:lowerLimit/upperLimit is -42/+90deg - EXACTLY matching
# the vendor spec (config/mk3.yaml) to within float32 rounding - so it is
# NOT a joint-limit asset-import bug. A local (Pi, pure-FK, no GPU) sweep
# of gripper_jaw1_link's own world-frame height as joint_2 increases toward
# GRASP_Q (all 3 of P0/P1/P2's own GRASP_Q configs, which independently
# share joint_2~62.3-62.5deg) shows a clean, monotonically DECREASING
# height that is nearly IDENTICAL across all three (different joint_3-6)
# configs - a purely geometric, joint_2-driven effect, not an idiosyncratic
# per-point issue - reaching only ~28-31mm above the z=0 ground plane
# (tasks/ar4/pickplace_graspgoal_env_cfg.py's GroundPlaneCfg, no
# init_state override, top surface at z=0) right at the empirically
# observed ~59deg wall. Combined with the 2026-07-27 joint-tracking-
# diagnostic task's own no-cube-obstruction control test (cube parked 3m
# away, wall still hit identically) - which already ruled out cube contact
# as the cause - and jaw1's own previously-measured collision-mesh extent
# (scripts/build_asset.py's _fix_jaw2_collision_mesh_asymmetry docstring:
# jaw1 spans [-0.018475, +0.015825] along its own local axis, i.e. ~18.5mm
# below its own link origin toward the object), this is conclusively a
# GROUND-PLANE COLLISION by the gripper's real physical geometry - a
# mechanism this sweep's original filter set never modeled at all (it only
# ever checked the abstract _EE_OFFSET-based PINCH POINT's height, never
# any real link's proximity to the ground). GROUND_CLEARANCE_MIN_M below
# requires gripper_jaw1_link's own world-frame z (held at the physically-
# correct OPEN position during approach, batch_fk_gripper_jaw1's own
# _GRIPPER_OPEN_POS_M default) to stay above 0.030m (30mm) - chosen to
# match the DIRECTLY-OBSERVED live-physics stall point (~26-31mm origin
# height at the empirically-confirmed ~59.0-59.1deg wall, not merely the
# theoretical ~18.5mm mesh-extent number, which is itself only a LOWER
# bound on the true required clearance since it ignores PhysX contact-
# generation margins and any tilt beyond the nominal value at the point of
# contact) - applied in ADDITION to (not instead of) the existing
# height/tilt/margin/roll filters. IMPORTANT ROBUSTNESS NOTE: the
# conclusion below (empty workspace) does NOT depend sensitively on this
# exact value - a sweep of this constant from 25mm down to 15mm (i.e. even
# BELOW the theoretical mesh-extent number, a deliberately loose/permissive
# choice) still finds ZERO survivors; only at 12mm and below does the
# filter start passing any configs at all. Directly computing the
# ACHIEVABLE gripper_jaw1_link ground-clearance ceiling across the full
# height/tilt/margin-satisfying population (ignoring roll/joint_1, n=235
# survivors out of a 2M-sample probe) gives max=14.93mm - i.e. even in the
# best case found anywhere in this population, the real fingertip (mesh
# extends ~18.475mm below the origin) would sit at 14.93-18.475=-3.55mm,
# STILL BELOW GROUND. The workspace is empty at this GRASP_AT_HEIGHT
# convention regardless of exactly which defensible clearance threshold is
# used.
GROUND_CLEARANCE_MIN_M = 0.030

# World-frame base transform (tasks/ar4/robot_cfg.py's AR4_MK5_CFG.init_state:
# rot=(0,0,0,1) wxyz = a 180-degree rotation about Z, pos defaults to origin
# since robot_cfg.py does not override ArticulationCfg's own default).
def base_to_world(p_base: np.ndarray) -> np.ndarray:
    """p_base: (..., 3) array of base_link-frame points -> world-frame points."""
    out = p_base.copy()
    out[..., 0] *= -1.0
    out[..., 1] *= -1.0
    return out


# The base->world transform above is a pure 180-degree-about-Z rotation, no
# translation - so it applies identically to FREE VECTORS (directions, e.g. a
# jaw-slide axis) as it does to POSITIONS. Reused directly (not re-derived)
# for exactly that purpose in _jaw_heading_offset_deg below.
_axis_base_to_world = base_to_world


def _jaw_heading_offset_deg(jaw_axis_world_xy: np.ndarray) -> np.ndarray:
    """2026-07-27 roll/heading constraint (see ROLL_TOL_DEG above for full
    motivation). Given the gripper's jaw-slide axis's WORLD-frame horizontal
    (x, y) component - shape (..., 2) - returns, in degrees, how far that
    in-plane heading is from being parallel to EITHER world X or world Y
    (the only two headings compatible with straddling a face of the
    world-axis-aligned, non-rotated cube: tasks/ar4/objects_cfg.py's
    CUBE_CFG). 0deg = exactly parallel to world X or world Y. 45deg (the
    worst possible case) = exactly diagonal, equidistant from both and
    therefore misaligned with both.

    Derivation: any heading and its "straddles a face" quality repeats every
    90 degrees (a square cube face grasped jaw-parallel to X is equally as
    valid as one grasped jaw-parallel to Y, and heading+180deg is the same
    physical jaw orientation as heading, since an open/close axis has no
    intrinsic direction) - so the raw heading angle is folded into [0, 90)
    via modulo, then folded again into [0, 45] by taking the smaller distance
    to either fold boundary (0 or 90), which are exactly the "parallel to
    world X" / "parallel to world Y" headings.
    """
    heading_deg = np.degrees(np.arctan2(jaw_axis_world_xy[..., 1], jaw_axis_world_xy[..., 0]))
    folded_0_90 = np.mod(heading_deg, 90.0)
    return np.minimum(folded_0_90, 90.0 - folded_0_90)


def _sanity_check_roll_criterion() -> None:
    """Numerically confirm the roll criterion accepts scripts/grasp_demo_v2.py's
    own established-good jaw-slide heading BEFORE trusting it for the real
    sweep (2026-07-27 task brief's own explicit requirement). That script's
    canonical grasp orientation (`_CANONICAL_X_AXIS_W = (0.0, 1.0, 0.0)`,
    i.e. link_6 local +X - the jaw-slide axis, same convention as this
    script's `_EE_OFFSET`/approach-axis local +Z) is defined directly as a
    WORLD-frame basis vector, so this checks _jaw_heading_offset_deg against
    that exact world-frame vector - no FK/base-frame conversion needed for
    this particular check, since the vector is already given in world frame.
    Also checks the orthogonal (0,1,0)->world +Y case is equally accepted
    (by symmetry, since a jaw-slide axis has no intrinsic +/- direction and
    world X/Y are both valid cube-straddling headings), and that a genuinely
    BAD 45-degree-diagonal heading is correctly REJECTED at ROLL_TOL_DEG, to
    confirm the check isn't vacuously permissive."""
    canonical_jaw_axis_world_xy = np.array(_CANONICAL_X_AXIS_W_FOR_SANITY_CHECK[:2])
    offset_canonical = float(_jaw_heading_offset_deg(canonical_jaw_axis_world_xy))
    offset_world_x = float(_jaw_heading_offset_deg(np.array([1.0, 0.0])))
    offset_diagonal = float(_jaw_heading_offset_deg(np.array([1.0, 1.0])))
    print(
        f"[sanity-check] grasp_demo_v2.py canonical jaw axis (0,1,0) -> heading offset "
        f"{offset_canonical:.4f} deg (must be ~0, i.e. ACCEPTED at ROLL_TOL_DEG={ROLL_TOL_DEG})"
    )
    print(f"[sanity-check] world +X jaw axis (1,0,0) -> heading offset {offset_world_x:.4f} deg (must be ~0)")
    print(
        f"[sanity-check] worst-case 45deg-diagonal jaw axis (1,1) -> heading offset "
        f"{offset_diagonal:.4f} deg (must be ~45, i.e. REJECTED at ROLL_TOL_DEG={ROLL_TOL_DEG})"
    )
    assert offset_canonical < 1e-6, "roll criterion does NOT accept grasp_demo_v2.py's own known-good jaw heading"
    assert offset_world_x < 1e-6, "roll criterion does NOT accept world +X heading"
    assert abs(offset_diagonal - 45.0) < 1e-6, "roll criterion does not correctly identify the worst-case diagonal heading"
    assert offset_diagonal > ROLL_TOL_DEG, "ROLL_TOL_DEG is set so loose it would accept the worst-case diagonal heading"


# Matches scripts/grasp_demo_v2.py's own _CANONICAL_X_AXIS_W exactly (see that
# module's own docstring, "2026-07-22, ar4-grasp-orientation-fix task", for
# the full derivation/justification of this specific heading choice) - the
# established-good jaw-slide-axis convention this sweep's own roll criterion
# is sanity-checked against above. Duplicated here as a plain constant
# (rather than importing scripts/grasp_demo_v2.py, which has its own
# AppLauncher/Isaac-Sim-touching import-time side effects unsuitable for this
# pure-numpy script) with an explicit cross-reference so the two never
# silently drift apart unnoticed.
_CANONICAL_X_AXIS_W_FOR_SANITY_CHECK = (0.0, 1.0, 0.0)


# ----------------------------------------------------------------------
# Vectorized batch FK for link_6, built from DEFAULT_JOINT_TABLE (the SAME
# vendor-URDF-derived joint parameters tasks/ar4/fk_verification.py's own
# scalar compute_link_pose_from_joint_values uses - not re-typed/re-derived
# here, just batched over a leading sample dimension for speed). Verified
# against the scalar function below before being trusted for the full sweep.
# ----------------------------------------------------------------------


def _rpy_to_matrix(rpy: tuple[float, float, float]) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    cp, sp = np.cos(pitch), np.sin(pitch)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    cy, sy = np.cos(yaw), np.sin(yaw)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _axis_angle_to_matrix_batch(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """axis: (3,) unit-ish vector (normalized here). angle: (N,). Returns (N,3,3)."""
    a = axis / np.linalg.norm(axis)
    n = angle.shape[0]
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    k = np.broadcast_to(k, (n, 3, 3))
    kk = np.einsum("nij,njk->nik", k, k)
    eye = np.broadcast_to(np.eye(3), (n, 3, 3))
    sin_a = np.sin(angle)[:, None, None]
    cos_a = (1 - np.cos(angle))[:, None, None]
    return eye + sin_a * k + cos_a * kk


# Build the base_link -> link_6 chain (arm joints only) once, reusing the
# exact same joint table + chain-walk logic compute_link_pose_from_joint_values
# uses internally (walk link_6 back to base_link, then reverse).
_child_to_joint = {j.child: j for j in DEFAULT_JOINT_TABLE}
_chain = []
_cur = "link_6"
while _cur != "base_link":
    _j = _child_to_joint[_cur]
    _chain.append(_j)
    _cur = _j.parent
_chain.reverse()

# 2026-07-28 (ar4-joint2-ground-clearance-fix task) addition: the base_link
# -> gripper_jaw1_link chain (arm joints + the fixed gripper_base_joint +
# the prismatic gripper_jaw1_joint), used ONLY to add a real ground-
# clearance filter to the sweep below - see GROUND_CLEARANCE_MIN_M's own
# comment for the full motivation (this sweep's original filter set only
# ever checked the abstract _EE_OFFSET-based pinch point's height, never
# any REAL link's proximity to the ground plane, which turned out to be
# the actual mechanism stopping joint_2 at its observed ~59deg physical
# wall - see kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md's
# 2026-07-28 "ar4-joint2-ground-clearance-fix task" UPDATE).
_cur = "gripper_jaw1_link"
_extra_gripper_joints = []
while _cur != "link_6":
    _j = _child_to_joint[_cur]
    _extra_gripper_joints.append(_j)
    _cur = _j.parent
_chain_to_jaw1 = _chain + list(reversed(_extra_gripper_joints))

# gripper_jaw1_joint's own commanded position during the PREGRASP/GRASP-OPEN
# approach phases every grasp script in this investigation uses (see
# tasks/ar4/robot_cfg.py's GRIPPER_OPEN_POS) - the gripper is OPEN, not
# closed, during the exact descent phase where ground-clipping risk is
# highest (this is also the configuration the 2026-07-27 roll-constraint
# task's own "open-gripper collision" finding was measured at), so this is
# the physically-correct value to hold the jaw at for this ground-clearance
# check, not an arbitrary choice.
_GRIPPER_OPEN_POS_M = 0.014


def _batch_transform_chain(chain: list, joint_values: dict[str, np.ndarray], n: int) -> tuple[np.ndarray, np.ndarray]:
    """Shared batched FK core: compose `chain` (a list of JointSpec, base ->
    tip order) for `n` samples. Handles fixed, revolute (existing), and
    prismatic (new, needed for the gripper_jaw1_joint ground-clearance
    chain) joint types, mirroring fk_verification.py's own scalar
    `_joint_transform`'s literal-URDF semantics (translate along `axis` in
    the joint's own post-origin-rotation frame) - not re-derived, just
    batched over a leading sample dimension."""
    r_cum = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
    p_cum = np.zeros((n, 3))
    for j in chain:
        r_origin = _rpy_to_matrix(j.origin_rpy)
        p_origin = np.array(j.origin_xyz, dtype=float)
        if j.joint_type == "fixed":
            r_j = np.broadcast_to(r_origin, (n, 3, 3))
            p_j = np.broadcast_to(p_origin, (n, 3))
        elif j.joint_type == "revolute":
            q = joint_values.get(j.name, np.zeros(n))
            r_motion = _axis_angle_to_matrix_batch(np.array(j.axis, dtype=float), q)
            r_j = np.einsum("ij,njk->nik", r_origin, r_motion)
            p_j = np.broadcast_to(p_origin, (n, 3))
        elif j.joint_type == "prismatic":
            # Plain literal URDF semantics (matches fk_verification.py's
            # scalar _joint_transform exactly): p = p_origin + r_origin @
            # (axis * q). r_origin @ axis is a fixed (3,) vector for this
            # joint; batched over q (n,) via outer-product broadcasting.
            q = joint_values.get(j.name, np.zeros(n))
            axis = np.array(j.axis, dtype=float)
            r_j = np.broadcast_to(r_origin, (n, 3, 3))
            p_j = p_origin[None, :] + q[:, None] * (r_origin @ axis)[None, :]
        else:
            raise ValueError(f"_batch_transform_chain only handles fixed/revolute/prismatic joints, got {j.joint_type}")
        p_cum = p_cum + np.einsum("nij,nj->ni", r_cum, p_j)
        r_cum = np.einsum("nij,njk->nik", r_cum, r_j)
    return p_cum, r_cum


def batch_fk_link6(joint_values: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """joint_values: dict of {joint_name: (N,) array}. Missing joints default
    to an all-zeros (N,) array inferred from the first provided array's
    length. Returns (pos (N,3), rot (N,3,3)) for link_6 in base_link frame."""
    n = len(next(iter(joint_values.values())))
    return _batch_transform_chain(_chain, joint_values, n)


def batch_fk_gripper_jaw1(joint_values: dict[str, np.ndarray]) -> np.ndarray:
    """Like batch_fk_link6, but for gripper_jaw1_link (base_link frame),
    with gripper_jaw1_joint's own value defaulted to _GRIPPER_OPEN_POS_M
    (not 0.0) unless explicitly provided - the physically-correct "gripper
    open during approach" state this ground-clearance check is meant to
    model. Returns pos only (N,3); rotation isn't needed by the clearance
    check."""
    n = len(next(iter(joint_values.values())))
    jv = dict(joint_values)
    jv.setdefault("gripper_jaw1_joint", np.full(n, _GRIPPER_OPEN_POS_M))
    pos, _ = _batch_transform_chain(_chain_to_jaw1, jv, n)
    return pos


def _cross_check(n_samples: int = 30, seed: int = 0) -> None:
    """Confirm batch_fk_link6 agrees with the already-established scalar
    compute_link_pose_from_joint_values to floating-point precision, before
    trusting the vectorized version for the full sweep."""
    rng = np.random.default_rng(seed)
    max_pos_err = 0.0
    max_rot_err = 0.0
    for _ in range(n_samples):
        jv = {name: rng.uniform(lo, hi) for name, (lo, hi) in JOINT_LIMITS_RAD.items()}
        pos_scalar, quat_scalar = compute_link_pose_from_joint_values(jv, "link_6")
        jv_batch = {name: np.array([val]) for name, val in jv.items()}
        pos_batch, rot_batch = batch_fk_link6(jv_batch)
        pos_err = np.linalg.norm(pos_batch[0] - pos_scalar)
        # Reconstruct rotation matrix from the scalar quat (w,x,y,z) to compare.
        w, x, y, z = quat_scalar
        rot_scalar = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        rot_err = np.linalg.norm(rot_batch[0] - rot_scalar)
        max_pos_err = max(max_pos_err, pos_err)
        max_rot_err = max(max_rot_err, rot_err)
    print(f"[cross-check] {n_samples} random configs: max pos err={max_pos_err:.3e} m, max rot err={max_rot_err:.3e}")
    assert max_pos_err < 1e-9, "batch FK disagrees with scalar FK on position - do not trust the sweep"
    assert max_rot_err < 1e-9, "batch FK disagrees with scalar FK on rotation - do not trust the sweep"


def _cross_check_gripper_jaw1(n_samples: int = 30, seed: int = 1) -> None:
    """Same cross-check as _cross_check above, but for the NEW
    batch_fk_gripper_jaw1 (2026-07-28 ground-clearance addition) - confirms
    it agrees with the scalar compute_link_pose_from_joint_values for
    gripper_jaw1_link (including the prismatic-joint batching path
    _cross_check above never exercises) before trusting it for the
    ground-clearance filter."""
    rng = np.random.default_rng(seed)
    max_pos_err = 0.0
    for _ in range(n_samples):
        jv = {name: rng.uniform(lo, hi) for name, (lo, hi) in JOINT_LIMITS_RAD.items()}
        jv["gripper_jaw1_joint"] = _GRIPPER_OPEN_POS_M
        pos_scalar, _ = compute_link_pose_from_joint_values(jv, "gripper_jaw1_link")
        jv_batch = {name: np.array([val]) for name, val in jv.items()}
        pos_batch = batch_fk_gripper_jaw1(jv_batch)
        pos_err = np.linalg.norm(pos_batch[0] - pos_scalar)
        max_pos_err = max(max_pos_err, pos_err)
    print(f"[cross-check] gripper_jaw1_link, {n_samples} random configs: max pos err={max_pos_err:.3e} m")
    assert max_pos_err < 1e-9, "batch_fk_gripper_jaw1 disagrees with scalar FK - do not trust the ground-clearance filter"


# ----------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------


def _evaluate(joint_values: dict[str, np.ndarray]):
    """Given batched joint values (all 6 joints), return a dict of derived
    per-sample arrays: pinch_pos (N,3) in base frame, tilt_deg (N,),
    margin_rad (N,6) (per-joint margin, columns in ARM_JOINT_NAMES order),
    roll_offset_deg (N,) - the 2026-07-27 addition, see ROLL_TOL_DEG's own
    docstring above for what this quantity means and why it's needed - and
    ground_clearance_m (N,) - the 2026-07-28 addition, see
    GROUND_CLEARANCE_MIN_M's own docstring above."""
    pos, rot = batch_fk_link6(joint_values)
    pinch_pos = pos + np.einsum("nij,j->ni", rot, _EE_OFFSET_LOCAL)
    approach_axis = np.einsum("nij,j->ni", rot, np.array([0.0, 0.0, 1.0]))
    cos_tilt = np.clip(-approach_axis[:, 2], -1.0, 1.0)
    tilt_deg = np.degrees(np.arccos(cos_tilt))
    n = pos.shape[0]
    margins = np.zeros((n, len(ARM_JOINT_NAMES)))
    for i, name in enumerate(ARM_JOINT_NAMES):
        lo, hi = JOINT_LIMITS_RAD[name]
        q = joint_values.get(name, np.zeros(n))
        margins[:, i] = np.minimum(q - lo, hi - q)
    # Jaw-slide axis = link_6 local +X (rot's first column, SAME convention as
    # scripts/grasp_demo_v2.py's _CANONICAL_X_AXIS_W / this file's own
    # approach_axis-as-local-+Z above), converted from base_link frame to
    # world frame via the same pure-rotation transform base_to_world already
    # applies to positions (valid for free vectors too - see
    # _axis_base_to_world's own comment).
    jaw_axis_base = rot[:, :, 0]
    jaw_axis_world = _axis_base_to_world(jaw_axis_base)
    roll_offset_deg = _jaw_heading_offset_deg(jaw_axis_world[:, :2])
    # Ground clearance: gripper_jaw1_link's own (base_link-frame, hence also
    # world-frame - z is unaffected by the base->world 180-about-Z rotation)
    # height, held at the physically-correct OPEN position during approach.
    # base_link frame z IS world z here (base_to_world only negates x/y).
    jaw1_pos = batch_fk_gripper_jaw1(joint_values)
    ground_clearance_m = jaw1_pos[:, 2]
    return pinch_pos, tilt_deg, margins, roll_offset_deg, ground_clearance_m


def run_sweep(n_stage_a: int, n_joint1_steps: int, seed: int = 0, chunk_size: int = 200_000):
    """Memory-bounded: this Pi has only ~1.8GB RAM (confirmed via `free -h`
    during development - far less than a typical desktop), so both stages
    process n_stage_a/n_joint1_steps in bounded-size chunks rather than
    materializing all N samples' worth of (N,3,3) rotation-matrix arrays at
    once (an earlier attempt at 5M samples in one shot was OOM-killed,
    exit 137)."""
    rng = np.random.default_rng(seed)

    # ---- Stage A, chunked ----
    t0 = time.time()
    survivors_2to6 = {name: [] for name in ["joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]}
    n_done = 0
    while n_done < n_stage_a:
        n = min(chunk_size, n_stage_a - n_done)
        joint_values_a = {"joint_1": np.zeros(n)}
        for name in ["joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]:
            lo, hi = JOINT_LIMITS_RAD[name]
            joint_values_a[name] = rng.uniform(lo, hi, size=n)

        # NOTE: roll_offset_deg is DELIBERATELY NOT filtered here in Stage A,
        # unlike height/tilt/margin_2to6. Those three quantities are provably
        # joint_1-invariant (see this function's own module-docstring
        # argument: joint_1 rotates the downstream chain about the vertical
        # axis, which cannot change a point's height or a vector's angle from
        # vertical, nor joints 2-6's own margins) - but a jaw-slide axis's
        # WORLD-frame HEADING is exactly the kind of quantity joint_1 DOES
        # change (the same way it changes a point's world-frame bearing, the
        # very reason Stage B exists to sweep joint_1 and recompute bearing
        # per step). Pre-filtering roll at joint_1=0 would silently discard
        # Stage-A survivors whose roll only becomes acceptable at some OTHER
        # joint_1 value - roll is therefore evaluated correctly per-step in
        # Stage B below, alongside the full-FK recompute already happening
        # there for exactly this kind of joint_1-dependent quantity.
        pinch_pos_a, tilt_deg_a, margins_a, _roll_offset_deg_a_unused, ground_clearance_a = _evaluate(joint_values_a)
        height_err_a = pinch_pos_a[:, 2] - GRASP_AT_HEIGHT
        margin_2to6_a = margins_a[:, 1:].min(axis=1)  # exclude joint_1's own column (always 0 here)

        # ground_clearance_a is joint_1-invariant (a z-coordinate, unaffected
        # by rotation about the vertical axis - same argument as
        # height/tilt/margin above), so it is safe to filter here in Stage A
        # too, unlike roll (see roll_offset_deg's own Stage-A comment above).
        mask_a = (
            (np.abs(height_err_a) <= HEIGHT_TOL_M)
            & (tilt_deg_a <= TILT_TOL_DEG)
            & (margin_2to6_a >= MARGIN_MIN_RAD)
            & (ground_clearance_a >= GROUND_CLEARANCE_MIN_M)
        )
        for name in survivors_2to6:
            survivors_2to6[name].append(joint_values_a[name][mask_a])
        n_done += n

    survivors_2to6 = {name: np.concatenate(arrs) for name, arrs in survivors_2to6.items()}
    n_survivors_a = len(survivors_2to6["joint_2"])
    print(
        f"[stage A] {n_stage_a} samples (joint_1=0), {time.time() - t0:.1f}s -> "
        f"{n_survivors_a} survivors ({100.0 * n_survivors_a / n_stage_a:.4f}%)"
    )
    if n_survivors_a == 0:
        return None

    # ---- Stage B, chunked over joint_1 steps ----
    lo1, hi1 = JOINT_LIMITS_RAD["joint_1"]
    j1_lo_margined = lo1 + MARGIN_MIN_RAD
    j1_hi_margined = hi1 - MARGIN_MIN_RAD
    j1_steps = np.linspace(j1_lo_margined, j1_hi_margined, n_joint1_steps)

    s = n_survivors_a
    m = n_joint1_steps
    t1 = time.time()
    steps_per_chunk = max(1, chunk_size // max(s, 1))

    world_xy_chunks, world_z_chunks, tilt_chunks, height_err_chunks = [], [], [], []
    margins_chunks, margin_min_chunks, roll_offset_chunks, ground_clearance_chunks = [], [], [], []
    joint_values_chunks = {name: [] for name in ARM_JOINT_NAMES}

    for start in range(0, m, steps_per_chunk):
        j1_chunk = j1_steps[start : start + steps_per_chunk]
        mc = len(j1_chunk)
        joint_values_b = {
            name: np.tile(survivors_2to6[name], mc) for name in ["joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
        }
        joint_values_b["joint_1"] = np.repeat(j1_chunk, s)

        pinch_pos_b, tilt_deg_b, margins_b, roll_offset_deg_b, ground_clearance_b = _evaluate(joint_values_b)
        height_err_b = pinch_pos_b[:, 2] - GRASP_AT_HEIGHT
        margin_all_b = margins_b.min(axis=1)

        mask_b = (
            (np.abs(height_err_b) <= HEIGHT_TOL_M)
            & (tilt_deg_b <= TILT_TOL_DEG)
            & (margin_all_b >= MARGIN_MIN_RAD)
            & (roll_offset_deg_b <= ROLL_TOL_DEG)
            & (ground_clearance_b >= GROUND_CLEARANCE_MIN_M)
        )
        if mask_b.sum() == 0:
            continue

        world_xy_chunks.append(base_to_world(pinch_pos_b[mask_b])[:, :2])
        world_z_chunks.append(pinch_pos_b[mask_b, 2])
        tilt_chunks.append(tilt_deg_b[mask_b])
        height_err_chunks.append(height_err_b[mask_b])
        margins_chunks.append(margins_b[mask_b])
        margin_min_chunks.append(margin_all_b[mask_b])
        roll_offset_chunks.append(roll_offset_deg_b[mask_b])
        ground_clearance_chunks.append(ground_clearance_b[mask_b])
        for name in ARM_JOINT_NAMES:
            joint_values_chunks[name].append(joint_values_b[name][mask_b])

    n_total_b = s * m
    if not world_xy_chunks:
        print(f"[stage B] {s} survivors x {m} joint_1 steps = {n_total_b} samples, {time.time() - t1:.1f}s -> 0 final survivors")
        return None

    world_xy = np.concatenate(world_xy_chunks)
    world_z = np.concatenate(world_z_chunks)
    tilt_final = np.concatenate(tilt_chunks)
    height_err_final = np.concatenate(height_err_chunks)
    margins_final = np.concatenate(margins_chunks)
    margin_min_final = np.concatenate(margin_min_chunks)
    roll_offset_final = np.concatenate(roll_offset_chunks)
    ground_clearance_final = np.concatenate(ground_clearance_chunks)
    joint_values_final = {name: np.concatenate(joint_values_chunks[name]) for name in ARM_JOINT_NAMES}

    n_survivors_b = world_xy.shape[0]
    print(
        f"[stage B] {s} survivors x {m} joint_1 steps = {n_total_b} samples, {time.time() - t1:.1f}s -> "
        f"{n_survivors_b} final survivors ({100.0 * n_survivors_b / n_total_b:.4f}%)"
    )

    return {
        "world_xy": world_xy,
        "world_z": world_z,
        "tilt_deg": tilt_final,
        "height_err": height_err_final,
        "margins": margins_final,
        "margin_min": margin_min_final,
        "roll_offset_deg": roll_offset_final,
        "ground_clearance_m": ground_clearance_final,
        "joint_values": joint_values_final,
    }


def summarize_and_plot(result, out_png: str):
    world_xy = result["world_xy"]
    margin_min = result["margin_min"]
    radii = np.linalg.norm(world_xy, axis=1)
    bearing_deg = np.degrees(np.arctan2(world_xy[:, 1], world_xy[:, 0]))

    best_idx = int(np.argmax(margin_min))
    best_xy = world_xy[best_idx]

    # NOTE on why this is NOT a naive xy centroid over "high-margin" points:
    # joint_1 is swept across nearly its FULL +/-170deg range in Stage B, so
    # the surviving set forms an approximate ANNULUS (many azimuths at
    # similar radius/margin), not one localized 2D blob - a plain xy mean
    # over a wide azimuthal spread cancels toward the origin and would be a
    # meaningless "recommended point" (verified during development: it
    # landed near radius ~0.09m, nowhere near any actual surviving point's
    # own margin). Instead: characterize the workspace in (radius, bearing)
    # terms, and choose a practical recommended point near bearing=90deg
    # (world +Y, "straight ahead") - the SAME approach direction
    # tasks/ar4/objects_cfg.py's CUBE_CFG and grasp_demo_v2.py's recentered
    # scene default already use - so the recommendation is a same-direction,
    # different-radius change from the current setup, not an unrelated new
    # heading, and stays directly comparable to the existing scene/camera
    # framing. A genuine local xy centroid is then computed only within a
    # tight neighborhood of that specific chosen point (not the whole
    # annulus).
    bearing_window_deg = 20.0
    near_90_mask = np.abs(((bearing_deg - 90.0 + 180.0) % 360.0) - 180.0) <= bearing_window_deg
    if near_90_mask.sum() > 0:
        rec_idx_local = int(np.argmax(margin_min[near_90_mask]))
        rec_idx = np.flatnonzero(near_90_mask)[rec_idx_local]
    else:
        rec_idx = best_idx  # fallback, shouldn't happen given near-full azimuthal coverage
    rec_xy = world_xy[rec_idx]

    # Local (genuinely spatial, not azimuthally-mixed) cluster centroid: only
    # points within 3cm radius AND 10 degrees bearing of the recommended point.
    local_mask = (
        (np.abs(radii - radii[rec_idx]) <= 0.03)
        & (np.abs(((bearing_deg - bearing_deg[rec_idx] + 180.0) % 360.0) - 180.0) <= 10.0)
    )
    local_centroid = world_xy[local_mask].mean(axis=0)

    def _report_point(label, idx):
        margins = result["margins"][idx]
        tilt = result["tilt_deg"][idx]
        height_err = result["height_err"][idx]
        roll_offset = result["roll_offset_deg"][idx]
        ground_clearance = result["ground_clearance_m"][idx]
        joints = {name: float(result["joint_values"][name][idx]) for name in ARM_JOINT_NAMES}
        xy = world_xy[idx]
        print(f"\n{label}: world (x, y) = ({xy[0]:.4f}, {xy[1]:.4f})  radius={radii[idx]:.4f}m  bearing={bearing_deg[idx]:.1f}deg")
        print(f"  height error: {height_err * 1000:.3f} mm (target z={GRASP_AT_HEIGHT:.4f})")
        print(f"  tilt from vertical: {tilt:.2f} deg")
        print(f"  jaw-slide-axis roll/heading offset from world X/Y: {roll_offset:.2f} deg (tol={ROLL_TOL_DEG:.1f} deg)")
        print(
            f"  gripper_jaw1_link ground clearance: {ground_clearance * 1000:.2f} mm "
            f"(min={GROUND_CLEARANCE_MIN_M * 1000:.1f} mm, 2026-07-28 ground-collision fix)"
        )
        for i, name in enumerate(ARM_JOINT_NAMES):
            lo_deg, hi_deg = JOINT_LIMITS_DEG[name]
            q_deg = math.degrees(joints[name])
            print(
                f"  {name}: q={q_deg:7.2f} deg (limits [{lo_deg:.1f}, {hi_deg:.1f}]), "
                f"margin={math.degrees(margins[i]):.2f} deg"
            )
        return joints, margins, roll_offset, ground_clearance

    print("\n=== Graspable workspace summary ===")
    print(f"Total graspable configs found: {world_xy.shape[0]}")
    print(f"World-frame X extent: [{world_xy[:, 0].min():.4f}, {world_xy[:, 0].max():.4f}] m")
    print(f"World-frame Y extent: [{world_xy[:, 1].min():.4f}, {world_xy[:, 1].max():.4f}] m")
    print(f"Radius from base extent: [{radii.min():.4f}, {radii.max():.4f}] m")
    print(f"Bearing extent: [{bearing_deg.min():.1f}, {bearing_deg.max():.1f}] deg (near-full circle minus joint_1's own +/-170deg-limited gap)")

    best_joints, best_margins, best_roll_offset, best_ground_clearance = _report_point(
        "GLOBAL best (max min-joint-margin, any azimuth)", best_idx
    )
    rec_joints, rec_margins, rec_roll_offset, rec_ground_clearance = _report_point(
        f"RECOMMENDED point (best margin within +/-{bearing_window_deg:.0f}deg of bearing=90deg, "
        "the scene's existing straight-ahead approach direction)",
        rec_idx,
    )
    print(f"\nLocal spatial cluster centroid around recommended point (+/-3cm radius, +/-10deg bearing): ({local_centroid[0]:.4f}, {local_centroid[1]:.4f})")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 8))
        sc = ax.scatter(world_xy[:, 0], world_xy[:, 1], c=margin_min, cmap="viridis", s=2, alpha=0.5)
        plt.colorbar(sc, label="min joint margin (rad)")
        ax.scatter([best_xy[0]], [best_xy[1]], c="blue", marker="*", s=250, edgecolors="black", label="global best (any azimuth)")
        ax.scatter([rec_xy[0]], [rec_xy[1]], c="red", marker="*", s=400, edgecolors="black", label="RECOMMENDED point (bearing~90deg)")
        ax.scatter([0.20], [0.28], c="orange", marker="x", s=150, label="old cube default (0.20, 0.28)")
        ax.scatter([0.0], [0.275], c="magenta", marker="x", s=150, label="recentered cube default (0.0, 0.275)")
        ax.scatter([0.0], [0.0], c="black", marker="s", s=80, label="arm base (world origin)")
        ax.set_xlabel("world x (m)")
        ax.set_ylabel("world y (m)")
        ax.set_title(
            f"AR4 graspable workspace at z={GRASP_AT_HEIGHT:.4f}m (15mm cube grasp height)\n"
            f"vertical approach <= {TILT_TOL_DEG} deg, joint margin >= {MARGIN_MIN_RAD:.2f}rad, "
            f"jaw-heading offset <= {ROLL_TOL_DEG:.0f} deg (2026-07-27 roll constraint),\n"
            f"gripper ground clearance >= {GROUND_CLEARANCE_MIN_M * 1000:.0f}mm (2026-07-28 ground-collision fix)"
        )
        ax.legend(loc="upper right", fontsize=8)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        fig.savefig(out_png, dpi=150)
        print(f"\nSaved visualization to {out_png}")
    except ImportError:
        print("\nmatplotlib not available - skipped visualization")

    return {
        "global_best_xy": tuple(best_xy.tolist()),
        "global_best_joints_deg": {name: math.degrees(v) for name, v in best_joints.items()},
        "global_best_margins_deg": {name: math.degrees(best_margins[i]) for i, name in enumerate(ARM_JOINT_NAMES)},
        "global_best_roll_offset_deg": float(best_roll_offset),
        "global_best_ground_clearance_mm": float(best_ground_clearance) * 1000,
        "recommended_xy": tuple(rec_xy.tolist()),
        "recommended_joints_deg": {name: math.degrees(v) for name, v in rec_joints.items()},
        "recommended_margins_deg": {name: math.degrees(rec_margins[i]) for i, name in enumerate(ARM_JOINT_NAMES)},
        "recommended_roll_offset_deg": float(rec_roll_offset),
        "recommended_ground_clearance_mm": float(rec_ground_clearance) * 1000,
        "recommended_local_cluster_centroid_xy": tuple(local_centroid.tolist()),
    }


_JAW1_MESH_LOWER_EXTENT_M = 0.018475
"""Known jaw1 collision-mesh lower extent along its own local axis (below
the gripper_jaw1_link joint origin, toward the object), from
scripts/build_asset.py's `_fix_jaw2_collision_mesh_asymmetry` docstring
(2026-07-24 finding: jaw1 spans [-0.018475, +0.015825] in its own local
frame). Used only by diagnose_empty_workspace below to translate an
ORIGIN-height ceiling into a real-FINGERTIP-height ceiling."""


def diagnose_empty_workspace(n_probe: int = 2_000_000, chunk_size: int = 200_000, seed: int = 0) -> None:
    """Run when run_sweep finds zero survivors (2026-07-28 ground-clearance
    fix): quantifies WHY, rather than leaving a bare "0 survivors" message.
    Computes the population of configs satisfying height/tilt/margin ALONE
    (joint_1=0, no ground-clearance or roll filter - the same Stage-A-style
    sample this module already draws elsewhere) and reports the achievable
    gripper_jaw1_link ground-clearance CEILING across that population,
    translated into a real-fingertip-height ceiling via
    _JAW1_MESH_LOWER_EXTENT_M. Also saves a histogram PNG (the meaningful
    substitute for the usual scatter-plot visualization, which has nothing
    to plot when the workspace itself is empty)."""
    rng = np.random.default_rng(seed)
    gc_chunks = []
    n_survivors = 0
    for start in range(0, n_probe, chunk_size):
        n = min(chunk_size, n_probe - start)
        jv = {"joint_1": np.zeros(n)}
        for name in ["joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]:
            lo, hi = JOINT_LIMITS_RAD[name]
            jv[name] = rng.uniform(lo, hi, size=n)
        pinch_pos, tilt_deg, margins, _roll_unused, ground_clearance = _evaluate(jv)
        height_err = pinch_pos[:, 2] - GRASP_AT_HEIGHT
        margin_2to6 = margins[:, 1:].min(axis=1)
        mask = (np.abs(height_err) <= HEIGHT_TOL_M) & (tilt_deg <= TILT_TOL_DEG) & (margin_2to6 >= MARGIN_MIN_RAD)
        n_survivors += int(mask.sum())
        if mask.sum() > 0:
            gc_chunks.append(ground_clearance[mask] * 1000.0)

    print("\n" + "=" * 78)
    print("DIAGNOSIS: why is the ground-clearance-corrected workspace empty?")
    print("=" * 78)
    print(
        f"Configs satisfying height (+/-{HEIGHT_TOL_M * 1000:.0f}mm of GRASP_AT_HEIGHT={GRASP_AT_HEIGHT * 1000:.1f}mm), "
        f"tilt (<={TILT_TOL_DEG:.0f}deg), and joint-2..6 margin (>={MARGIN_MIN_RAD:.2f}rad) ALONE "
        f"(no ground-clearance or roll filter yet): {n_survivors} / {n_probe} samples"
    )
    if not gc_chunks:
        print("No configs satisfy even height/tilt/margin alone - workspace is empty for a reason upstream of ground clearance.")
        return
    gc = np.concatenate(gc_chunks)
    fingertip_ceiling_mm = gc.max() - _JAW1_MESH_LOWER_EXTENT_M * 1000.0
    print(f"gripper_jaw1_link origin ground-clearance among these: min={gc.min():.2f}mm max={gc.max():.2f}mm mean={gc.mean():.2f}mm median={np.median(gc):.2f}mm")
    print(
        f"Real fingertip height ceiling (best case anywhere in this population) = "
        f"{gc.max():.2f}mm - {_JAW1_MESH_LOWER_EXTENT_M * 1000:.2f}mm (known jaw1 mesh extent) = {fingertip_ceiling_mm:.2f}mm"
    )
    if fingertip_ceiling_mm < 0:
        print(
            f"-> NEGATIVE: even in the BEST case found anywhere in this population, the real gripper fingertip "
            f"would sit {abs(fingertip_ceiling_mm):.2f}mm BELOW the z=0 ground plane. This is not a filter-tuning "
            "artifact - the workspace is genuinely empty at this GRASP_AT_HEIGHT for a near-vertical approach."
        )
    else:
        print(f"-> POSITIVE: {fingertip_ceiling_mm:.2f}mm of real fingertip clearance is achievable somewhere - re-check GROUND_CLEARANCE_MIN_M's own value against this ceiling.")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(gc, bins=40, color="steelblue", edgecolor="black", alpha=0.8)
        ax.axvline(_JAW1_MESH_LOWER_EXTENT_M * 1000.0, color="red", linestyle="--", linewidth=2,
                   label=f"known jaw1 mesh extent ({_JAW1_MESH_LOWER_EXTENT_M * 1000:.1f}mm) = zero real-fingertip clearance")
        ax.axvline(GROUND_CLEARANCE_MIN_M * 1000.0, color="orange", linestyle=":", linewidth=2,
                   label=f"shipped GROUND_CLEARANCE_MIN_M ({GROUND_CLEARANCE_MIN_M * 1000:.0f}mm)")
        ax.set_xlabel("gripper_jaw1_link origin ground clearance (mm)")
        ax.set_ylabel("count (height/tilt/margin-satisfying samples)")
        ax.set_title(
            f"AR4 empty-workspace diagnosis: achievable ground clearance never reaches\n"
            f"the real fingertip's own required clearance, at GRASP_AT_HEIGHT={GRASP_AT_HEIGHT * 1000:.1f}mm "
            f"(2026-07-28 finding)"
        )
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        out_png = os.path.join(_RL_ROOT, "outputs", "ar4_graspable_workspace", "empty_workspace_ground_clearance_diagnosis.png")
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        fig.savefig(out_png, dpi=150)
        print(f"\nSaved diagnosis visualization to {out_png}")
    except ImportError:
        print("\nmatplotlib not available - skipped visualization")


if __name__ == "__main__":
    print("Cross-checking vectorized batch FK against the established scalar FK...")
    _cross_check()
    _cross_check_gripper_jaw1()

    print("\nSanity-checking the roll/heading criterion against grasp_demo_v2.py's own known-good jaw heading...")
    _sanity_check_roll_criterion()

    print("\nRunning forward-sampling sweep...")
    result = run_sweep(n_stage_a=8_000_000, n_joint1_steps=145, seed=0, chunk_size=200_000)
    if result is None:
        print(
            "No graspable configs found at all with the current filter thresholds "
            f"(including the 2026-07-28 GROUND_CLEARANCE_MIN_M={GROUND_CLEARANCE_MIN_M * 1000:.0f}mm constraint)."
        )
        diagnose_empty_workspace()
        sys.exit(1)

    out_png = os.path.join(_RL_ROOT, "outputs", "ar4_graspable_workspace", "graspable_workspace.png")
    summary = summarize_and_plot(result, out_png)
    print("\n=== JSON summary ===")
    import json

    print(json.dumps(summary, indent=2))
