"""Reusable primitive for closing the "commanded joint pose vs. physically
achieved joint pose" gap on the AR4 arm's implicit PD actuator.

Background (2026-07-27, ar4-joint-tracking-closed-loop-fix task, direct
continuation of the ar4-joint-tracking-diagnostic task): that task confirmed
and quantified a real tracking gap - even with the "boosted" arm actuator
gains (stiffness=4000, damping=200) every confirm/grasp script in this
investigation already uses, the physics arm settles ~19.65mm short of the
FK-commanded pinch point at the P0 grasp configuration, larger than the
15mm cube itself. Mechanism: ``isaaclab.actuators.ImplicitActuatorCfg`` is a
pure proportional+derivative (PD) controller with no integral or gravity-
compensation term, so *some* nonzero steady-state position error under a
persistent load (the arm's own weight) is inherent to finite-gain PD
control - raising gains shrinks it, never eliminates it, and may plateau
before shrinking below the cube's own size (e.g. if
``effort_limit_sim`` starts clipping the actuator's torque output).

``settle_to_joint_pose`` implements the standard, gain-independent fix: an
outer iterated integral-correction loop. Command the desired pose, let it
settle, measure the achieved pose, compute the per-joint error
``e = desired - achieved``, then re-command ``target = desired +
accumulated_correction`` where the correction integrates a bounded
(anti-windup-clamped) fraction of the error, and repeat until the achieved
pose is within tolerance. This is the joint-space analog of a mechanism
this repo has ALREADY validated in Cartesian/task space:
``scripts/oracle_rollout.py``'s ``ik_pursuit_action`` maintains "a per-env
Cartesian integral-error accumulator... provides the missing integral
action that proportional-only control lacks" to fix the exact same
"textbook P-only droop" signature (nonzero computed_torque at zero
velocity with frozen joint_pos) - this function generalizes that same idea
into a small, reusable, joint-space primitive other grasp scripts can call
directly instead of re-deriving the same fix.

Deliberately duck-types on ``robot``/``env`` (no isaaclab import at module
level) so this module itself has no Isaac-Sim import-time side effects and
can be imported freely by any script, matching this repo's
``tasks/ar4/fk_verification.py`` precedent for keeping reusable math/logic
separate from any one script's own AppLauncher-construction order
constraints - the difference here is this function DOES need a live,
already-constructed Isaac Sim env/robot to call (unlike fk_verification.py,
which is pure numpy), so it is not standalone-testable outside Isaac Sim;
only its import must stay side-effect-free.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

import torch


# ----------------------------------------------------------------------
# Cartesian closed-loop correction (2026-07-28, ar4-cartesian-fingertip-
# correction task, direct continuation of the pedestal-fix task above).
#
# Background: the 2026-07-28 "ar4-pedestal-ground-clearance-fix task"
# UPDATE (kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md) found
# that even after `settle_to_joint_pose` above converges every arm joint to
# sub-1deg of its FK-computed GRASP_Q target, the real fingertip (link_6's
# own `_EE_OFFSET`-based pinch point, in the SAME convention
# `scripts/grasp_demo_v2.py`/`scripts/ar4_pedestal_grasp_confirm.py` already
# use) still lands ~6-7mm from the FK-PREDICTED pinch point for that exact
# commanded joint pose - not zero, because nulling error independently
# per-joint does not null the compounded, lever-arm-amplified Cartesian
# error at the end of a ~0.5m serial chain (a small residual on an
# upstream joint moves the fingertip far more than the same residual on
# joint_6). `settle_to_joint_pose` cannot fix this by construction: it only
# ever measures/corrects joint-space error, never the Cartesian quantity
# that actually determines whether the jaws land on the cube.
#
# `settle_to_cartesian_pose` fixes this the standard way: null the error in
# the space that actually matters. Each outer iteration (1) holds the
# current joint target for `inner_settle_steps` (reusing the exact same
# manual physics-driving pattern as `settle_to_joint_pose` above - no
# `env.step(action)`, so callers must not also drive these joints through
# the action manager), (2) measures the REAL achieved pinch-point world
# position via `robot.data.body_pose_w` for the actual end-effector body
# (matches the geometry-diagnostic convention already established by
# `scripts/ar4_pedestal_grasp_confirm.py`'s own `_achieved_pinch_world`),
# (3) computes the Cartesian error against the desired world target, and
# (4) maps that 3D error to a joint-space delta via the SAME verified
# Jacobian machinery `scripts/grasp_demo_v2.py`'s DLS polish already uses
# (`root_physx_view.get_jacobians()` + the point-Jacobian correction for a
# rigidly-offset pinch point, i.e. `_ee_point_pos_and_jacobian`'s own
# `J_pos - skew(offset) @ J_ang` identity) - a damped-least-squares (DLS)
# solve restricted to the 3 position rows only (this primitive corrects
# fingertip POSITION; grasp ORIENTATION is established upstream by the
# caller's own choice of GRASP_Q and is not touched here).
#
# Deliberately reimplements the tiny quat->matrix and skew helpers here in
# pure torch (see `_quat_wxyz_to_matrix_torch`/`_skew_torch` below) instead
# of importing `isaaclab.utils.math.matrix_from_quat`/
# `_world_jacobian_to_root_frame` from `grasp_demo_v2.py` directly - this
# module's own module docstring requires staying import-side-effect-free
# (no isaaclab import at module level), and `grasp_demo_v2.py` is a script
# whose top-level `argparse.parse_args()` would fire on import, exactly the
# reason `tasks/ar4/objects_cfg.py`'s `PEDESTAL_HEIGHT_M` comment gives for
# duplicating rather than importing a small piece of another script.
# Works entirely in WORLD frame throughout (unlike grasp_demo_v2.py's own
# DLS polish, which works in the robot's ROOT frame because its target is
# given in root frame) - since both the desired target and the live
# Jacobian read here are naturally in world frame already, no
# world-to-root rotation is needed at all, avoiding a second frame
# convention in this file for no benefit.
# ----------------------------------------------------------------------


def _quat_wxyz_to_matrix_torch(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Single (non-batched) wxyz quaternion (shape (4,)) -> (3,3) rotation
    matrix. Same formula as `scripts/ar4_pedestal_grasp_confirm.py`'s numpy
    `_quat_wxyz_to_matrix` / `isaaclab.utils.math.matrix_from_quat`, just
    reimplemented in plain torch to keep this module isaaclab-import-free."""
    w, x, y, z = quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]
    row0 = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)])
    row1 = torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)])
    row2 = torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)])
    return torch.stack([row0, row1, row2])


def _skew_torch(v: torch.Tensor) -> torch.Tensor:
    """(3,) -> (3,3) skew-symmetric matrix, so `skew(v) @ w == cross(v, w)`."""
    zero = torch.zeros((), dtype=v.dtype, device=v.device)
    row0 = torch.stack([zero, -v[2], v[1]])
    row1 = torch.stack([v[2], zero, -v[0]])
    row2 = torch.stack([-v[1], v[0], zero])
    return torch.stack([row0, row1, row2])


def settle_to_cartesian_pose(
    env,
    robot,
    arm_joint_ids: Sequence[int],
    ee_body_id: int,
    ee_offset_local: Sequence[float],
    desired_pinch_pos_w: Sequence[float],
    initial_q: Sequence[float],
    *,
    tol_m: float = 0.002,
    max_outer_iters: int = 8,
    inner_settle_steps: int = 150,
    dls_lambda: float = 0.02,
    max_joint_delta_rad: float = 0.15,
    gripper_joint_ids: Optional[Sequence[int]] = None,
    gripper_target: Optional[Sequence[float]] = None,
    render: bool = False,
    on_step: Optional[Callable[[int, int], None]] = None,
    label: str = "CART_SETTLE",
    verbose: bool = True,
) -> dict:
    """Iteratively drive the REAL fingertip/pinch point (``ee_body_id``'s
    world pose + the rigid local offset ``ee_offset_local``) to
    ``desired_pinch_pos_w`` (world-frame meters), starting from the joint
    target ``initial_q`` (radians) - typically the caller's own already
    joint-settled ``settle_to_joint_pose`` result, so this function only
    needs to correct the residual Cartesian gap that primitive cannot see.

    Each outer iteration: hold ``current_q`` for ``inner_settle_steps``,
    measure the achieved pinch point, compute the world-frame position
    error, and - unless already within ``tol_m`` - map that error through a
    damped-least-squares solve of the pinch point's own position Jacobian
    (``J_pos - skew(R @ offset) @ J_ang``, the same identity
    ``scripts/grasp_demo_v2.py``'s ``_ee_point_pos_and_jacobian`` uses) to
    get a joint-space correction, clamped to +/-``max_joint_delta_rad`` per
    joint per outer iteration (anti-divergence, same purpose as
    ``settle_to_joint_pose``'s ``integral_clamp``), and add it directly to
    ``current_q`` (Gauss-Newton style - no separate integral accumulator,
    since each outer iteration already re-linearizes at the newly-settled
    state).

    Returns a dict: ``achieved_pinch_w`` (list[float], meters),
    ``pinch_error_mm`` (float, final |desired-achieved| in mm),
    ``final_q`` (list[float], radians - the joint target that produced
    ``achieved_pinch_w``), ``n_outer_iters`` (int), ``converged`` (bool,
    whether ``tol_m`` was reached before ``max_outer_iters``)."""
    device = robot.data.joint_pos.device
    n = len(arm_joint_ids)
    desired_t = torch.tensor(list(desired_pinch_pos_w), dtype=torch.float32, device=device)
    offset_t = torch.tensor(list(ee_offset_local), dtype=torch.float32, device=device)
    current_q = torch.tensor(list(initial_q), dtype=torch.float32, device=device)

    gripper_target_t = None
    if gripper_joint_ids is not None and gripper_target is not None:
        gripper_target_t = torch.tensor([list(gripper_target)], dtype=torch.float32, device=device)

    ee_jacobi_idx = (ee_body_id - 1) if robot.is_fixed_base else ee_body_id

    achieved_pinch_w = None
    converged = False
    outer_used = 0
    err_mm = float("nan")

    for outer in range(max_outer_iters):
        outer_used = outer + 1
        target_t = current_q.unsqueeze(0)
        for i in range(inner_settle_steps):
            robot.set_joint_position_target(target_t, joint_ids=arm_joint_ids)
            if gripper_target_t is not None:
                robot.set_joint_position_target(gripper_target_t, joint_ids=gripper_joint_ids)
            robot.write_data_to_sim()
            env.sim.step(render=render)
            robot.update(env.physics_dt)
            if on_step is not None:
                on_step(outer, i)

        ee_pose_w = robot.data.body_pose_w[0, ee_body_id]
        ee_pos_w, ee_quat_w = ee_pose_w[0:3], ee_pose_w[3:7]
        rot_w = _quat_wxyz_to_matrix_torch(ee_quat_w)
        world_offset = rot_w @ offset_t
        achieved_pinch_t = ee_pos_w + world_offset
        achieved_pinch_w = achieved_pinch_t.tolist()

        error_t = desired_t - achieved_pinch_t
        err_mm = torch.norm(error_t).item() * 1000.0
        if verbose:
            print(f"  [{label} outer={outer}] achieved_pinch={achieved_pinch_w} err={err_mm:.4f}mm")
        if err_mm < tol_m * 1000.0:
            converged = True
            break

        # Same indexing convention as scripts/grasp_demo_v2.py's own DLS
        # polish (jacobian_w = get_jacobians()[:, ik_jacobi_idx, :, joint_ids]),
        # then drop the leading env-batch dim (single env only, [0]).
        jacobian_w = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm_joint_ids][0]
        jac_pos = jacobian_w[0:3, :]
        jac_ang = jacobian_w[3:6, :]
        skew = _skew_torch(world_offset)
        point_jac_pos = jac_pos - skew @ jac_ang  # (3, n)

        jjt = point_jac_pos @ point_jac_pos.T
        lam2_i = (dls_lambda ** 2) * torch.eye(3, dtype=torch.float32, device=device)
        delta_q = point_jac_pos.T @ torch.linalg.solve(jjt + lam2_i, error_t)
        delta_q = torch.clamp(delta_q, -max_joint_delta_rad, max_joint_delta_rad)
        current_q = current_q + delta_q

    return {
        "achieved_pinch_w": achieved_pinch_w,
        "pinch_error_mm": err_mm,
        "final_q": current_q.tolist(),
        "n_outer_iters": outer_used,
        "converged": converged,
    }


def settle_to_joint_pose(
    env,
    robot,
    arm_joint_ids: Sequence[int],
    desired_q: Sequence[float],
    *,
    tol_rad: float = 0.005,
    max_outer_iters: int = 6,
    inner_settle_steps: int = 150,
    integral_gain: float = 1.0,
    integral_clamp: float = 0.5,
    gripper_joint_ids: Optional[Sequence[int]] = None,
    gripper_target: Optional[Sequence[float]] = None,
    render: bool = False,
    on_step: Optional[Callable[[int, int], None]] = None,
    label: str = "SETTLE",
    verbose: bool = True,
) -> dict:
    """Iteratively drive ``arm_joint_ids`` to genuinely ACHIEVE
    ``desired_q`` (radians), not merely command it once and hope.

    Each outer iteration commands ``target = desired_q + correction``,
    steps the sim ``inner_settle_steps`` times (holding that fixed target,
    same "hold constant for the whole phase" convention this repo's own
    grasp scripts already established as working better than a ramped
    interpolation - see ``grasp_demo_v2.py``'s PHASES loop comment), then
    measures the achieved joint position and updates
    ``correction += integral_gain * (desired_q - achieved)``, clamped to
    +/-``integral_clamp`` radians per joint (anti-windup, same purpose as
    ``oracle_rollout.py``'s own ``INTEGRAL_CLAMP``). Stops early once the
    max per-joint error is under ``tol_rad``.

    ``env.sim.step(render=render)`` / ``robot.update(env.physics_dt)`` is
    the same low-level manual-driving pattern already established by
    ``scripts/_verify_gripper_mirror_fix.py`` / ``scripts/
    ar4_graspable_workspace_confirm_numeric.py`` (bypasses the action-
    manager tensor interface entirely) - this function does not call
    ``env.step(action)``, so the caller must NOT also be driving the same
    joints through the action manager concurrently.

    ``on_step(outer_iter, inner_step)`` is an optional callback invoked
    after every single inner physics step (e.g. to update/record a camera
    or contact sensor that isn't otherwise touched by this function) -
    kept as a generic hook rather than hardcoding camera/sensor logic into
    this shared primitive.

    Returns a dict: ``achieved_q`` (list[float], radians), ``correction``
    (list[float], radians, final accumulated correction), ``per_joint_err_deg``
    (list[float], final |desired-achieved| in degrees), ``max_err_deg``
    (float), ``n_outer_iters`` (int, how many outer iterations actually ran),
    ``converged`` (bool, whether tol_rad was reached before max_outer_iters).
    """
    n = len(arm_joint_ids)
    device = robot.data.joint_pos.device
    desired_t = torch.tensor(list(desired_q), dtype=torch.float32, device=device)
    correction = torch.zeros(n, dtype=torch.float32, device=device)

    gripper_target_t = None
    if gripper_joint_ids is not None and gripper_target is not None:
        gripper_target_t = torch.tensor([list(gripper_target)], dtype=torch.float32, device=device)

    achieved_t = robot.data.joint_pos[0, arm_joint_ids].clone()
    converged = False
    outer_used = 0

    for outer in range(max_outer_iters):
        outer_used = outer + 1
        target_t = (desired_t + correction).unsqueeze(0)
        for i in range(inner_settle_steps):
            robot.set_joint_position_target(target_t, joint_ids=arm_joint_ids)
            if gripper_target_t is not None:
                robot.set_joint_position_target(gripper_target_t, joint_ids=gripper_joint_ids)
            robot.write_data_to_sim()
            env.sim.step(render=render)
            robot.update(env.physics_dt)
            if on_step is not None:
                on_step(outer, i)

        achieved_t = robot.data.joint_pos[0, arm_joint_ids].clone()
        error_t = desired_t - achieved_t
        max_err_deg = torch.max(torch.abs(error_t)).item() * 180.0 / math.pi
        if verbose:
            print(
                f"  [{label} outer={outer}] target=desired+correction max_err_after_settle={max_err_deg:.4f}deg "
                f"correction_deg={[round(math.degrees(c), 3) for c in correction.tolist()]}"
            )
        if max_err_deg < math.degrees(tol_rad):
            converged = True
            break

        correction = torch.clamp(correction + integral_gain * error_t, -integral_clamp, integral_clamp)

    per_joint_err_deg = (torch.abs(desired_t - achieved_t) * 180.0 / math.pi).tolist()
    return {
        "achieved_q": achieved_t.tolist(),
        "correction": correction.tolist(),
        "per_joint_err_deg": per_joint_err_deg,
        "max_err_deg": max(per_joint_err_deg),
        "n_outer_iters": outer_used,
        "converged": converged,
    }
