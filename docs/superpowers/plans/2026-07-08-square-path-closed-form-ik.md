# Plan: Replace square-path demo's grid-search IK with closed-form 3-DOF IK

## Context

`scripts/square_path_demo.py` traces a square with the AR4's joints 1-3
(joints 4-6/gripper held "limp"). It currently solves each waypoint via a
15x15 forward-kinematics grid search (225 candidates x 10 settle steps) plus
a bounded-step DLS "polish" with joint_1 locked — a documented workaround for
DLS getting trapped in local minima (see commit history / ROADMAP). It's slow
and, per its own comments, silently failed to converge well on 2/8 points in
one prior run.

This is not an RL experiment (no reward/training/action-space change) — it's
a classical-kinematics correctness fix to a diagnostic demo script. The
Tier-1 hypothesis-gate process in CLAUDE.md does not apply; this plan is the
appropriate weight of process for it.

## Derivation (already done and numerically verified — implementer should not
re-derive, just transcribe and verify against the running sim)

Source: `/home/saps/projects/annin_ws/src/ar4_ros_driver/annin_ar4_description/urdf/ar_macro.xacro`
(AR4 mk5 URDF — the same hardware description this repo's USD asset was
built from).

Forward-kinematics analysis (done with a standalone numpy FK chain built
from the URDF's joint origins/axes, cross-checked against this session's
existing empirical `CALIBRATION_C` constant, which it reproduces to 0.2°)
shows:

1. **Joint 1 is an exact, decoupled base yaw.** For any fixed q2,q3, the
   end-effector's cylindrical radius from the base's z-axis and its height
   are *invariant* to q1 — q1 only rotates the "arm plane" about z. Exactly:
   ```
   bearing = atan2(target_y, target_x)
   q1 = -(bearing + pi/2)
   ```
2. **Joints 2 and 3 form an exact 2-link planar arm** operating in that
   rotated plane, in coordinates `u = radial distance from z-axis`,
   `v = height (z)`:
   - Shoulder pivot (invariant to q2, q3): `U_S = 0.06415`, `V_S = 0.16978`
   - Upper-arm length (shoulder→elbow, rigid, invariant to q2): `L2 = 0.305`
   - Forearm+wrist length (elbow→EE with joints 4/5/6 held at 0, rigid,
     invariant to q3): `L3 = 0.263940`
   - Link angle relations (exact): direction of link2 = `pi/2 - q2`;
     direction of link3 relative to link2 = `-pi/2 - q3`.

   Standard law-of-cosines 2-link solve, target `(u_t, v_t) = (hypot(x,y), z)`:
   ```
   du, dv = u_t - U_S, v_t - V_S
   d = hypot(du, dv)
   cos_delta = clip((d**2 - L2**2 - L3**2) / (2*L2*L3), -1, 1)
   delta = -acos(cos_delta)          # elbow-sign branch verified below
   phi1 = atan2(dv, du) - atan2(L3*sin(delta), L2 + L3*cos(delta))
   q2 = pi/2 - phi1
   q3 = -pi/2 - delta
   ```
   The `delta = -acos(...)` branch (not `+acos`) is the one that stays
   within the AR4 mk5's real joint limits (j2: -42..90deg, j3: -89..52deg) —
   the `+acos` branch requires q2/q3 far outside range. Do not add branch
   selection logic; always use `-acos`.

3. **Verified exact** (residual ~1e-6m) against the same numpy FK chain for
   all 8 planned waypoints below. This is real math, not a fit — no need to
   re-derive, but DO add a runtime self-check (see Task, step 4) since the
   installed USD asset could in principle differ subtly from this URDF.

## Reachability finding — square geometry must change

The *current* square's near edge (x=0.17, y in [-0.08,0.08], z=0.08)
requires q3 up to 63deg at some points — past the real 52deg limit. This is
exactly the failure mode the old grid search was masking (landing near, not
on, those points). Verified fix: shift the near edge from x=0.17 to x=0.25
(keep far edge at x=0.33->0.41, same 0.16m side length, same z=0.08,
same y=[-0.08,0.08]). This keeps every waypoint within joint limits with
>=3.7 degrees of margin on the tightest joint (q3), confirmed by direct
computation over all 8 points.

New `SQUARE_POINTS_B` (replaces the existing list, same 8-point corner+
midpoint layout, x shifted +0.08 throughout):
```python
SQUARE_Z = 0.08
SQUARE_POINTS_B = [
    (0.25, -0.08, SQUARE_Z),  # corner 1
    (0.25, 0.00, SQUARE_Z),   # mid 1-2
    (0.25, 0.08, SQUARE_Z),   # corner 2
    (0.33, 0.08, SQUARE_Z),   # mid 2-3
    (0.41, 0.08, SQUARE_Z),   # corner 3
    (0.41, 0.00, SQUARE_Z),   # mid 3-4
    (0.41, -0.08, SQUARE_Z),  # corner 4
    (0.33, -0.08, SQUARE_Z),  # mid 4-1
]
```

## Task 1: Implement closed-form IK in square_path_demo.py

Read `scripts/square_path_demo.py` in full first.

1. Delete the grid-search machinery: `GRID_N`, `GRID_SETTLE_STEPS`,
   `POLISH_STEP_MAX`, `POLISH_ROUNDS`, `POLISH_SETTLE_STEPS` constants, the
   entire body of `solve_point()`, and the `DifferentialIKController`/
   `DifferentialIKControllerCfg` import and instantiation in `main()` (no
   longer used — this is a closed-form solve, not iterative).
2. Add the closed-form solver (module-level constants + a small pure
   function), using exactly the formulas and constants in this plan's
   Derivation section. Suggested shape:
   ```python
   L2 = 0.305
   L3 = 0.263940
   SHOULDER_U = 0.06415
   SHOULDER_V = 0.16978

   def solve_ik3(x, y, z):
       """Closed-form 3-DOF IK (joints 1-3) for the AR4, wrist held at
       q4=q5=q6=0. Derived from ar_macro.xacro joint origins; see
       docs/superpowers/plans/2026-07-08-square-path-closed-form-ik.md."""
       bearing = math.atan2(y, x)
       q1 = -(bearing + math.pi / 2)
       u_t, v_t = math.hypot(x, y), z
       du, dv = u_t - SHOULDER_U, v_t - SHOULDER_V
       d = math.hypot(du, dv)
       cos_delta = max(-1.0, min(1.0, (d**2 - L2**2 - L3**2) / (2 * L2 * L3)))
       delta = -math.acos(cos_delta)
       phi1 = math.atan2(dv, du) - math.atan2(L3 * math.sin(delta), L2 + L3 * math.cos(delta))
       q2 = math.pi / 2 - phi1
       q3 = -math.pi / 2 - delta
       return q1, q2, q3
   ```
3. Replace `SQUARE_POINTS_B` with the shifted list from this plan (near
   edge x=0.25, far edge x=0.41).
4. Add a startup self-check, once, right after `env.reset()` in `main()`,
   before solving any waypoints: with the arm at its home pose (all joints
   0 — which is the reset state), read the actual `link_6` position in the
   robot frame the same way `solve_point`/the rest of the script already
   does (`subtract_frame_transforms` against `robot.data.body_pose_w` /
   `root_pose_w`), and compare it to `solve_ik3`'s own forward-kinematics
   prediction at q=0 — i.e. compute what `solve_ik3` *implies* the home-pose
   EE position should be (you can get this by picking any reachable target
   and checking the round-trip, or more directly by evaluating the known
   closed-form relations at q1=q2=q3=0: the model predicts EE at
   `u = SHOULDER_U + L2*cos(pi/2) + L3*cos(-pi/2) = SHOULDER_U`,
   `v = SHOULDER_V + L2*sin(pi/2) + L3*sin(-pi/2) = SHOULDER_V + L2 - L3`,
   i.e. predicted home position is `(x,y,z) = (0, -SHOULDER_U, SHOULDER_V + L2 - L3)`
   in the robot's base frame, since bearing=-pi/2 at the home orientation).
   Compare this predicted point to the actual measured home-pose EE
   position; if they differ by more than 5mm, raise a clear `RuntimeError`
   explaining the built USD asset's kinematics don't match the assumed
   URDF-derived constants (do not silently proceed) — this follows this
   repo's `isaac-lab-manipulator-research` skill discipline of verifying
   sim artifacts directly rather than trusting derived math blindly. If it
   matches, print the residual and proceed.
5. In the per-waypoint solve loop (replacing the old `solve_point()` call
   site), for each target: call `solve_ik3`, clip q2/q3 to the actual
   runtime joint limits already read from `robot.data.joint_pos_limits`
   (keep that limit-reading code — it's a real safety net even though the
   chosen geometry already has margin), command that joint target directly
   (same `env.step()` action-writing pattern already used elsewhere in the
   file, gripper open, wrist held at whatever it currently is / limp), run
   enough settle steps for the PD actuators to converge (60-80 steps is a
   reasonable starting point — implementer should watch the printed
   residual and increase if it hasn't converged to sub-cm), then log the
   achieved residual (measured EE position vs. target) the same way the old
   code did, for comparison. Expect residuals far smaller than the old grid
   search's (no iterative solver error, only actuator settle-time
   error) — if residuals come out worse than roughly 1cm on any waypoint,
   that is a real bug (check the joint-limit clip isn't firing, and check
   sign conventions), not something to paper over by adding back DLS.
6. Everything else in the file (camera setup, video recording, the square
   loop / `hold_wrist_limp_and_move`, home position handling) stays as-is.

## Verification (implementer does this as part of the task, not a separate
task)

Run:
```
PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/square_path_demo.py --headless
```
Confirm in the printed output: the startup self-check passes (no
RuntimeError), and all 8 waypoint residuals are small (target: sub-cm;
flag in the report if any exceed 1cm). Then actually watch
`logs/videos/ar4_square_path_demo.mp4` (per this repo's verification
standard — do not rely on exit code / printed numbers alone) and confirm
the arm traces a clean, complete square with no visible stalls, jumps, or
corners cut short. Report frame-level observations in the task report, not
just "it ran."

## Global constraints

- Do not reintroduce `DifferentialIKController`/iterative DLS anywhere in
  this file — the whole point of this change is that the closed form
  replaces it.
- Use the exact constants/formulas from this plan (`L2=0.305`,
  `L3=0.263940`, `SHOULDER_U=0.06415`, `SHOULDER_V=0.16978`, the
  `delta = -acos(...)` branch, the shifted `SQUARE_POINTS_B` list) — do not
  re-derive or re-fit them.
- Commit the change when done (private solo repo, no PR workflow, commit
  straight to `main`).
