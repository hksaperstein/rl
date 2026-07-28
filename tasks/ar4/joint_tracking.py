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
