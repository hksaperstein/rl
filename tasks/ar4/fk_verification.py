"""Standing, reusable AR4 kinematic verification framework (Layer 1 + Layer 2).

Built 2026-07-23 after a string of real AR4 defects (a missing gripper
physics drive, 4 classical-IK positioning bugs, a wrist-orientation bug, and
a gripper jaw-mirroring bug where commanding "open" made both jaws converge
to the same point instead of spreading apart) were each found by ad hoc,
one-off diagnostic scripts or by the user directly watching the simulation.
See kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md's 2026-07-21
"UPDATE (later, ar4-franka-fixes-transfer plan, Task 5)" and 2026-07-22
UPDATEs for the full incident history this module is meant to catch
automatically going forward, as a standing test rather than a one-off script.

Pure numpy, NO isaaclab/torch import - runs on plain python3 (this repo's
established "pure-torch/numpy testable without Isaac Sim" convention, see
tasks/franka/antipodal_grasp_reward.py's precedent; numpy specifically here
since this module is scalar/dict-based kinematics, not batched tensor math).

======================================================================
Layer 1 - asset-geometry check (independent forward kinematics)
======================================================================

AR4 is a simple serial chain (6 revolute arm joints + 2 prismatic gripper
jaw joints, no closed loops), so a straightforward homogeneous-transform FK
chain is implemented directly below rather than pulling in a new dependency
(pytorch_kinematics was checked and is NOT installed anywhere in this
project's environments as of 2026-07-23 - confirmed via `python3 -c "import
pytorch_kinematics"` on the Pi; a hand-rolled ~10-joint serial chain is also
simple enough that a dependency wouldn't meaningfully reduce code size).

The joint table below is an independently-sourced, provenance-commented
snapshot of the VENDOR's own raw URDF/xacro source - read directly via `ssh
desktop` on 2026-07-23 from
AR4_DESCRIPTION_PATH=/home/saps/projects/annin_ws/src/ar4_ros_driver/annin_ar4_description
(not reachable from the Pi, and deliberately not vendored as a runtime
dependency - see module docstring below for why hardcoding is preferred
here). Specifically:

  - urdf/ar_macro.xacro: joints 1-6 (arm) + the fixed ee_joint at the end of
    link_6. Every origin/axis/rpy value below is copied verbatim from that
    file's <joint> tags.
  - urdf/ar_gripper_macro.xacro: gripper_base_joint (fixed, mounts the
    gripper to ee_link) + gripper_jaw1_joint/gripper_jaw2_joint (prismatic).
  - config/mk5.yaml: confirms `l6_length: 0.041` - this project uses the mk5
    variant exclusively (tasks/ar4/robot_cfg.py's AR4_MK5_CFG,
    assets/ar4_mk5/) - and ar_macro.xacro's joint_6 <origin> already
    hardcodes `xyz="0 0 0.041"` (its own comment: "copy exact origin from
    your previous working joint_6"), so no further mk5-specific
    substitution is needed for joint_6.

This chain is DELIBERATELY re-derived from the raw XML text, not imported
from or cross-checked against scripts/build_asset.py's own URDF-import
pipeline - the whole point of Layer 1 is to catch bugs baked into that
pipeline (or into the built USD asset), not silently reproduce them.

`compute_link_pose_from_joint_values` returns pose relative to `base_link`
(the arm's own root, i.e. BEFORE the fixed `base_joint` that mounts it to
"world" in the raw URDF, and before whatever world-frame placement
tasks/ar4/robot_cfg.py's own ArticulationCfg.init_state applies in Isaac
Sim). To compare against a live Isaac Sim link pose (which is reported in
Isaac's own WORLD frame), the caller must first transform the live pose
into the robot's own base_link frame (e.g. via
isaaclab.utils.math.subtract_frame_transforms against the articulation's
own live root pose - the same pattern already used throughout
scripts/grasp_demo_v2.py) - this module has no isaaclab import, so it does
not do that conversion itself.

======================================================================
IMPORTANT: gripper_jaw2_joint's sign convention - history and correction
======================================================================

gripper_jaw2_joint's raw URDF origin is
`<origin xyz="0.0 -0.036 0" rpy="0.0 -pi 0.0"/>` with `<axis xyz="-1 0 0"/>`
(identical origin xyz and identical axis vector to gripper_jaw1_joint - only
the origin's rpy differs, flipping jaw2's mounting by pi about Y). Applying
the URDF spec literally (translate along `axis` in the joint's OWN frame,
i.e. AFTER rotating by the origin's rpy) gives an effective world-frame
translation axis of `R_origin(rpy) @ axis = (+1, 0, 0)` for jaw2, vs.
`(-1, 0, 0)` for jaw1 (jaw1's origin rpy is identity) - i.e. a plain,
literal reading of the raw URDF predicts jaw1/jaw2 should be commanded
with the SAME sign to mirror correctly (matching the raw URDF's own
`<mimic ... multiplier="1"/>` tag on jaw2, also present in the raw file).
This module's joint table below implements exactly that plain literal
reading, with NO special-cased sign correction for either jaw joint.

This project's own history briefly disagreed with that plain reading, then
came back around to it. A 2026-07-21 investigation (jaw-mimic-vs-actuator
dynamics conflict, kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md's
"UPDATE (later, ar4-franka-fixes-transfer plan, Task 5)") found a live
`env.reset()` on that day's build showing correct ~28mm mirrored-open
separation only with OPPOSITE-signed jaw1/jaw2 commands, and
`tasks/ar4/robot_cfg.py` was fixed accordingly (commit `928af41`). A first
draft of this module (2026-07-23, same day, earlier in this file's own
history) matched that opposite-sign convention via a
`translate_axis_in_parent_frame` special case for jaw2.

That opposite-sign convention was then ITSELF independently found wrong
the same day (2026-07-23), by the concurrent gripper-fix task
(`scripts/_sweep_jaw2_symmetry.py`, commit `d59595a`,
`tasks/ar4/robot_cfg.py`'s own current "UPDATE 2026-07-23" comment): a
direct sweep holding jaw1 fixed and varying jaw2's commanded value found
jaw2's world-frame position is exactly `-1.0 * (jaw2's own commanded
value)` - i.e. jaw2's local-to-world mapping already contains the sign
flip, so commanding jaw2 to `-1.0 * jaw1` (the 2026-07-21 fix) double-
negates and collapses both jaws onto the same point. The correct command,
re-confirmed live, is the SAME signed value for both joints - exactly what
a plain reading of the raw URDF already said. This module's own live
integration check
(`scripts/_verify_gripper_fk_integration.py`, run 2026-07-23 on a fresh
GCP cloud build of the current asset - see
kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md's matching update
for the full run) independently reproduced this: a live `env.reset()`
open state read `gripper_jaw1_joint=+0.01400`,
`gripper_jaw2_joint=+0.01400` (same sign) with a REAL measured world-frame
jaw separation of `27.996mm` - and this module's plain-URDF FK model
(after removing the now-obsolete `translate_axis_in_parent_frame` special
case) predicted that same live pose to within `0.000mm`.

The lesson this module's own history is a direct example of: an
independent FK check grounded in the raw vendor source is more durable
than calibrating a bug-verification tool against a single historical
"already-empirically-confirmed" belief, which can itself later turn out
to be wrong. This module no longer special-cases gripper_jaw2_joint at
all - the plain, literal URDF joint table is now confirmed, via a live
Isaac Sim measurement on the current asset, to be correct as written.

======================================================================
Layer 2 - control-intent/task-invariant check
======================================================================

`assert_gripper_separation` uses Layer 1's now-calibrated FK to check that
a COMMANDED joint_values dict produces the intended real-world jaw
separation - not just "did each joint individually reach its own target."
This is the exact check that would have caught the 2026-07-21 bug (every
AR4 env cfg commanded gripper_jaw2_joint to the IDENTICAL signed value as
gripper_jaw1_joint instead of jaw1's negation - see
tests/test_ar4_fk_verification.py's TestJawMirroringRegression for the
concrete demonstration: the OLD buggy same-sign convention FAILS this
check (near-zero separation) and the corrected convention PASSES (~28mm,
matching tasks/ar4/objects_cfg.py's own documented "~28mm max aperture").
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

# ----------------------------------------------------------------------
# Joint table (see module docstring for full provenance)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class JointSpec:
    """One joint in the AR4 serial chain, as read from the raw vendor URDF.

    origin_xyz/origin_rpy: the joint's <origin> tag (translation, then
    roll-pitch-yaw rotation, both expressed in the PARENT link's frame).
    axis: the joint's <axis> tag (a unit vector; None for fixed joints),
    applied via the plain, literal URDF semantics (translate/rotate along
    `axis` in the joint's own post-origin-rotation frame) with no special
    casing for any joint - see the module docstring's "IMPORTANT" section
    above for why gripper_jaw2_joint in particular no longer needs one.
    """

    name: str
    parent: str
    child: str
    joint_type: str  # "revolute" | "prismatic" | "fixed"
    origin_xyz: tuple[float, float, float]
    origin_rpy: tuple[float, float, float]
    axis: tuple[float, float, float] | None = None


_PI = np.pi

# Arm chain: urdf/ar_macro.xacro's own <joint> tags, verbatim.
_ARM_JOINTS: list[JointSpec] = [
    JointSpec(
        name="joint_1", parent="base_link", child="link_1", joint_type="revolute",
        origin_xyz=(0.0, 0.0, 0.092), origin_rpy=(_PI, 0.0, 0.0), axis=(0.0, 0.0, 1.0),
    ),
    JointSpec(
        name="joint_2", parent="link_1", child="link_2", joint_type="revolute",
        origin_xyz=(0.0, 0.06415, -0.07778), origin_rpy=(1.5708, 0.0, -1.5708), axis=(0.0, 0.0, -1.0),
    ),
    JointSpec(
        name="joint_3", parent="link_2", child="link_3", joint_type="revolute",
        origin_xyz=(0.0, -0.305, 0.0), origin_rpy=(0.0, 0.0, 3.1416), axis=(0.0, 0.0, -1.0),
    ),
    JointSpec(
        name="joint_4", parent="link_3", child="link_4", joint_type="revolute",
        origin_xyz=(0.0, 0.0, 0.0), origin_rpy=(1.5708, 0.0, -1.5708), axis=(0.0, 0.0, -1.0),
    ),
    JointSpec(
        name="joint_5", parent="link_4", child="link_5", joint_type="revolute",
        origin_xyz=(0.0, 0.0, -0.22294), origin_rpy=(_PI, 0.0, -1.5708), axis=(1.0, 0.0, 0.0),
    ),
    JointSpec(
        name="joint_6", parent="link_5", child="link_6", joint_type="revolute",
        # xacro hardcodes xyz="0 0 0.041" directly (its own comment: "copy
        # exact origin from your previous working joint_6") - matches
        # config/mk5.yaml's l6_length: 0.041, the variant this project uses.
        origin_xyz=(0.0, 0.0, 0.041), origin_rpy=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0),
    ),
    JointSpec(
        name="ee_joint", parent="link_6", child="ee_link", joint_type="fixed",
        origin_xyz=(0.0, 0.0, 0.0), origin_rpy=(0.0, 0.0, 0.0),
    ),
]

# Gripper: urdf/ar_gripper_macro.xacro's own <joint> tags, verbatim - plain
# literal semantics for both jaws, no sign special-casing (see the big
# module-docstring "IMPORTANT" note above for why gripper_jaw2_joint does
# NOT need one, despite this project's own history briefly believing
# otherwise).
_GRIPPER_JOINTS: list[JointSpec] = [
    JointSpec(
        name="gripper_base_joint", parent="ee_link", child="gripper_base_link", joint_type="fixed",
        origin_xyz=(0.0, 0.0, 0.0), origin_rpy=(-1.5708, 0.0, 0.0),
    ),
    JointSpec(
        name="gripper_jaw1_joint", parent="gripper_base_link", child="gripper_jaw1_link", joint_type="prismatic",
        origin_xyz=(0.0, -0.036, 0.0), origin_rpy=(0.0, 0.0, 0.0), axis=(-1.0, 0.0, 0.0),
    ),
    JointSpec(
        name="gripper_jaw2_joint", parent="gripper_base_link", child="gripper_jaw2_link", joint_type="prismatic",
        origin_xyz=(0.0, -0.036, 0.0), origin_rpy=(0.0, -_PI, 0.0), axis=(-1.0, 0.0, 0.0),
    ),
]

DEFAULT_JOINT_TABLE: tuple[JointSpec, ...] = tuple(_ARM_JOINTS + _GRIPPER_JOINTS)
"""The vendor-URDF-derived joint chain used by default. Exposed (rather than
kept module-private) so tests can pass a deliberately-corrupted copy into
`compute_link_pose_from_joint_values`/`assert_link_pose_matches_vendor_fk`'s
`joint_table` argument to prove Layer 1 actually catches an import-style
asset-geometry defect - see
tests/test_ar4_fk_verification.py::TestCorruptedOriginIsCaught."""


def with_corrupted_origin(joint_table: tuple[JointSpec, ...], joint_name: str, delta_xyz: tuple[float, float, float]) -> tuple[JointSpec, ...]:
    """Return a copy of joint_table with joint_name's origin_xyz perturbed
    by delta_xyz - a deliberately-corrupted asset-geometry defect, for
    testing that Layer 1 actually catches this whole bug class."""
    out = []
    for j in joint_table:
        if j.name == joint_name:
            new_xyz = tuple(a + b for a, b in zip(j.origin_xyz, delta_xyz))
            j = replace(j, origin_xyz=new_xyz)
        out.append(j)
    return tuple(out)


# ----------------------------------------------------------------------
# Rotation math (pure numpy - no scipy dependency)
# ----------------------------------------------------------------------


def _rpy_to_matrix(rpy: tuple[float, float, float]) -> np.ndarray:
    """URDF rpy convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    roll, pitch, yaw = rpy

    cr, sr = np.cos(roll), np.sin(roll)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])

    cp, sp = np.cos(pitch), np.sin(pitch)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])

    cy, sy = np.cos(yaw), np.sin(yaw)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])

    return rz @ ry @ rx


def _axis_angle_to_matrix(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    """Rodrigues' rotation formula. `axis` need not be pre-normalized."""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def _matrix_to_quat_wxyz(r: np.ndarray) -> np.ndarray:
    """Rotation matrix -> quaternion in (w, x, y, z) order, matching Isaac
    Lab's own quaternion convention (see e.g. tasks/ar4/robot_cfg.py's
    `rot=(0.0, 0.0, 0.0, 1.0)` init_state comment: "(w, x, y, z)")."""
    trace = np.trace(r)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def _joint_transform(joint: JointSpec, q: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (R, p): the joint's own child-in-parent rotation matrix and
    translation, for the given scalar joint value q (ignored for fixed
    joints)."""
    r_origin = _rpy_to_matrix(joint.origin_rpy)
    p_origin = np.array(joint.origin_xyz, dtype=float)

    if joint.joint_type == "fixed":
        return r_origin, p_origin
    elif joint.joint_type == "revolute":
        r_motion = _axis_angle_to_matrix(joint.axis, q)
        return r_origin @ r_motion, p_origin
    elif joint.joint_type == "prismatic":
        axis = np.asarray(joint.axis, dtype=float)
        # Plain literal URDF semantics: translate along axis in the
        # joint's own post-origin-rotation frame.
        p = p_origin + r_origin @ (axis * q)
        return r_origin, p
    else:
        raise ValueError(f"Unknown joint_type {joint.joint_type!r} for joint {joint.name!r}")


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def compute_link_pose_from_joint_values(
    joint_values: dict[str, float],
    link_name: str,
    joint_table: tuple[JointSpec, ...] = DEFAULT_JOINT_TABLE,
) -> tuple[np.ndarray, np.ndarray]:
    """Pure kinematic FK: given a dict of {joint_name: value} (missing
    joints default to 0.0; fixed joints' values are ignored), return
    (pos, quat_wxyz) for `link_name`, relative to `base_link` (see module
    docstring for the base_link-vs-world-frame caveat).

    No Isaac Sim dependency - pure numpy, unit-testable directly.
    """
    if link_name == "base_link":
        return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])

    child_to_joint = {j.child: j for j in joint_table}
    if link_name not in child_to_joint:
        raise ValueError(f"Unknown link_name {link_name!r} - not a child of any joint in this chain")

    # Walk from link_name back up to base_link, then compose forward.
    chain: list[JointSpec] = []
    cur = link_name
    while cur != "base_link":
        j = child_to_joint[cur]
        chain.append(j)
        cur = j.parent
    chain.reverse()

    r_cum = np.eye(3)
    p_cum = np.zeros(3)
    for j in chain:
        q = 0.0 if j.joint_type == "fixed" else joint_values.get(j.name, 0.0)
        r_j, p_j = _joint_transform(j, q)
        p_cum = p_cum + r_cum @ p_j
        r_cum = r_cum @ r_j

    return p_cum, _matrix_to_quat_wxyz(r_cum)


@dataclass(frozen=True)
class FKCheckResult:
    passed: bool
    link_name: str
    expected_pos: np.ndarray
    actual_pos: np.ndarray
    pos_discrepancy_mm: float
    rot_discrepancy_rad: float


def assert_link_pose_matches_vendor_fk(
    live_pos: np.ndarray,
    live_quat: np.ndarray,
    joint_values: dict[str, float],
    link_name: str,
    tolerance_mm: float = 1.0,
    joint_table: tuple[JointSpec, ...] = DEFAULT_JOINT_TABLE,
) -> FKCheckResult:
    """Compare a live (Isaac-Sim-reported) link pose, already expressed in
    the robot's own base_link frame (see module docstring), against this
    module's independent vendor-URDF FK prediction for the same
    joint_values. Raises AssertionError with the measured discrepancy on
    mismatch; returns an FKCheckResult on success.
    """
    expected_pos, expected_quat = compute_link_pose_from_joint_values(joint_values, link_name, joint_table)
    live_pos = np.asarray(live_pos, dtype=float)
    live_quat = np.asarray(live_quat, dtype=float)

    pos_discrepancy_mm = float(np.linalg.norm(live_pos - expected_pos) * 1000.0)
    # Quaternion double-cover: q and -q represent the same rotation.
    quat_dot = float(np.clip(abs(np.dot(live_quat, expected_quat)), -1.0, 1.0))
    rot_discrepancy_rad = float(2.0 * np.arccos(quat_dot))

    passed = pos_discrepancy_mm <= tolerance_mm
    result = FKCheckResult(
        passed=passed,
        link_name=link_name,
        expected_pos=expected_pos,
        actual_pos=live_pos,
        pos_discrepancy_mm=pos_discrepancy_mm,
        rot_discrepancy_rad=rot_discrepancy_rad,
    )
    if not passed:
        raise AssertionError(
            f"FK mismatch for link {link_name!r}: expected pos={expected_pos}, got live pos={live_pos} "
            f"(discrepancy={pos_discrepancy_mm:.3f}mm, tolerance={tolerance_mm}mm); "
            f"rotation discrepancy={rot_discrepancy_rad:.4f}rad for joint_values={joint_values}"
        )
    return result


def assert_gripper_separation(
    joint_values: dict[str, float],
    min_mm: float,
    max_mm: float | None = None,
    joint_table: tuple[JointSpec, ...] = DEFAULT_JOINT_TABLE,
) -> float:
    """Layer 2: compute both jaw links' FK-predicted positions (relative to
    base_link - the common upstream arm-chain transform cancels out of the
    separation distance regardless of arm pose) and assert their real
    3D separation falls in [min_mm, max_mm]. Raises AssertionError with the
    measured separation on failure; returns the measured separation (mm) on
    success.
    """
    p1, _ = compute_link_pose_from_joint_values(joint_values, "gripper_jaw1_link", joint_table)
    p2, _ = compute_link_pose_from_joint_values(joint_values, "gripper_jaw2_link", joint_table)
    separation_mm = float(np.linalg.norm(p1 - p2) * 1000.0)

    too_small = separation_mm < min_mm
    too_large = max_mm is not None and separation_mm > max_mm
    if too_small or too_large:
        bound = f"[{min_mm}, {max_mm}]" if max_mm is not None else f">= {min_mm}"
        raise AssertionError(
            f"Gripper jaw separation {separation_mm:.3f}mm outside expected range {bound} "
            f"for joint_values={joint_values}"
        )
    return separation_mm
