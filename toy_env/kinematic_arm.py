"""Pure-kinematics N-link arm model: forward kinematics + a numerical Jacobian.

**Scope note (read this before trusting anything here for a real decision):**
This module is part of `toy_env/`, a CPU-only, physics-free proxy environment
for early-stage RL prototyping. It has NO dynamics (no mass/inertia/torque),
NO contact/collision, and NO friction. It exists to let algorithm/action-space
questions be iterated in seconds on a laptop/Pi before spending real Isaac Sim
GPU time on anything promising. A result produced using this module is a
hypothesis generator, not a conclusion — anything interesting found here still
needs re-verification in the real Isaac Lab simulator. See
`kb/wiki/concepts/toy-kinematic-proxy-env.md` for the full writeup of what this
is/isn't good for, and `kb/wiki/experiments/d8-antipodal-grasp-quality.md`
(its "Root cause investigation" section) for the real Isaac-Lab finding this
environment is trying to let us reproduce cheaply: that **absolute
joint-position control** (action semantics that map to a configuration-
independent *target*, so a given action's effect on the end effector depends
heavily on the arm's current pose) causes a training collapse — a policy
transiently discovers how to approach an object, then abandons that behavior
over training — while **task-space/Cartesian control** (where the action's
effect on the end effector is ~consistent regardless of current configuration)
does not have this problem, or has it less severely.

Design
------
A 7-joint revolute chain (7 DOF, matching Franka Panda's own joint count so
this toy arm is "informative about Franka's own configuration-dependent
behavior" per the design brief, without attempting to match Franka's exact
DH parameters/link geometry — that precision isn't needed for what this
environment is used for). Joint rotation axes alternate Z/Y
(`AXES = ['z','y','z','y','z','y','z']`), a common simplification for giving
an arm chain reach in all three dimensions without needing a full DH-parameter
table. A fixed, non-actuated final offset (`GRIPPER_OFFSET`) represents the
gripper/end-effector length past the last joint.

Forward kinematics is computed by walking the chain and, at each joint,
applying a fixed translation (`LINK_OFFSETS[i]`, in the parent frame) to reach
that joint's pivot, then a rotation about that joint's axis. This is
deliberately not a DH-parameter implementation — it's the simplest scheme that
still produces a well-defined, testable 3D chain.

The Jacobian (`jacobian_position`) is purely numerical (central finite
differences of end-effector position with respect to each joint angle) rather
than a closed-form/analytical Jacobian. This is a deliberate simplification
(the module docstring for `arm_reach_env.py` describes it as "a basic Jacobian
pseudo-inverse solve is fine, doesn't need to be sophisticated" per the design
brief) — it is slower than a closed form but trivially correct for any chain
shape, which matters more here than speed at this arm's tiny scale (7 finite
differences per call, negligible compared to an RL rollout's own cost).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

N_JOINTS = 7

# Alternating rotation axes per joint, base to wrist.
AXES = ["z", "y", "z", "y", "z", "y", "z"]

# Fixed translation (in the parent/previous-joint frame) applied to reach each
# joint's pivot point, walking base -> wrist. Chosen to give an overall reach
# (~0.8-1.0m fully extended) in the same rough scale as a real Franka Panda's
# ~0.85m reach, without claiming to match its actual link geometry.
LINK_OFFSETS = np.array(
    [
        [0.0, 0.0, 0.10],  # joint 0 (base yaw, Z) pivot height above base
        [0.0, 0.0, 0.05],  # joint 1 (shoulder pitch, Y)
        [0.0, 0.0, 0.28],  # joint 2 (Z) - upper-arm-length segment
        [0.0, 0.0, 0.10],  # joint 3 (Y) - elbow-ish
        [0.0, 0.0, 0.24],  # joint 4 (Z) - forearm-length segment
        [0.0, 0.0, 0.08],  # joint 5 (Y) - wrist pitch
        [0.0, 0.0, 0.10],  # joint 6 (Z) - wrist roll
    ]
)

# Fixed, non-actuated offset from the last joint to the end-effector /
# "gripper tip" point used for reward/observation purposes.
GRIPPER_OFFSET = np.array([0.0, 0.0, 0.06])

# Symmetric joint limits (radians), same bound for every joint for simplicity.
JOINT_LIMIT = 2.9  # ~166 degrees


def _rot_matrix(axis: str, theta: float) -> np.ndarray:
    """3x3 rotation matrix about the world-aligned local 'z' or 'y' axis."""
    c, s = np.cos(theta), np.sin(theta)
    if axis == "z":
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    if axis == "y":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    raise ValueError(f"unsupported axis {axis!r}, only 'z'/'y' implemented")


@dataclass
class FKResult:
    """Result of a forward-kinematics evaluation."""

    joint_positions: np.ndarray  # (N_JOINTS + 2, 3): base origin, each joint pivot, EE tip
    ee_pos: np.ndarray = field(init=False)  # (3,)

    def __post_init__(self) -> None:
        self.ee_pos = self.joint_positions[-1].copy()


def forward_kinematics(theta: np.ndarray) -> FKResult:
    """Compute every joint-pivot position and the end-effector tip position.

    Args:
        theta: (N_JOINTS,) joint angles in radians, base to wrist.

    Returns:
        FKResult with `joint_positions` shape (N_JOINTS + 2, 3) — index 0 is
        the fixed base origin, indices 1..N_JOINTS are each joint's pivot
        point, and the last index is the end-effector/gripper-tip position —
        and `ee_pos`, an alias for that last row.
    """
    theta = np.asarray(theta, dtype=np.float64)
    assert theta.shape == (N_JOINTS,), f"expected shape ({N_JOINTS},), got {theta.shape}"

    T = np.eye(4)
    positions = [T[:3, 3].copy()]  # base origin

    for i in range(N_JOINTS):
        t_offset = np.eye(4)
        t_offset[:3, 3] = LINK_OFFSETS[i]
        T = T @ t_offset
        positions.append(T[:3, 3].copy())  # this joint's pivot point

        r = np.eye(4)
        r[:3, :3] = _rot_matrix(AXES[i], theta[i])
        T = T @ r

    t_gripper = np.eye(4)
    t_gripper[:3, 3] = GRIPPER_OFFSET
    T = T @ t_gripper
    positions.append(T[:3, 3].copy())  # end-effector tip

    return FKResult(joint_positions=np.stack(positions, axis=0))


def jacobian_position(theta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Numerical (central-difference) position Jacobian, shape (3, N_JOINTS).

    d(ee_pos)/d(theta_i) for each joint i, holding all other joints fixed.
    Used by the task-space action mode to convert a desired Cartesian
    end-effector velocity into a joint-space velocity via the Moore-Penrose
    pseudo-inverse (`np.linalg.pinv`).
    """
    theta = np.asarray(theta, dtype=np.float64)
    J = np.zeros((3, N_JOINTS))
    for i in range(N_JOINTS):
        d = np.zeros(N_JOINTS)
        d[i] = eps
        ee_plus = forward_kinematics(theta + d).ee_pos
        ee_minus = forward_kinematics(theta - d).ee_pos
        J[:, i] = (ee_plus - ee_minus) / (2 * eps)
    return J


def max_reach() -> float:
    """Sum of all link-offset magnitudes plus the gripper offset — an upper
    bound on how far the end effector can possibly be from the base origin
    (only achieved in a fully-extended, generally unreachable-in-practice
    configuration given the alternating-axis chain)."""
    return float(np.linalg.norm(LINK_OFFSETS, axis=1).sum() + np.linalg.norm(GRIPPER_OFFSET))
