# Experiment 24 Gate 1 Report: Scripted-Oracle Viability — FAIL

**Verdict: FAIL.** The pure, non-learned, reactive differential-IK oracle
specified by the Gate 1 plan cannot reliably follow this repo's existing
5-waypoint geometric path. It stalls — end-effector position frozen
bit-identical for 50+ consecutive steps — well before reaching the grasp
waypoint, in the overwhelming majority of sampled episodes. Per the plan's
own hard gate ("if fewer than 30/50 episodes pass the grasp+lift gate... do
NOT proceed to writing Gate 2's implementation plan"), this is a clean
FAIL. Task 4's formal 50-episode run was not executed, because the
mechanism-level failure documented below makes that run's outcome
foregone: the vast majority of episodes never advance past the very first
waypoint, let alone traverse all five waypoints and achieve a genuine
antipodal grasp+lift.

## What was built (Tasks 1-3 of the original plan, all complete)

`scripts/oracle_rollout.py`: drives the AR4 arm via live differential IK
toward `compute_path_waypoints`' 5-waypoint path, with a hand-coded
gripper open/close schedule, reusing `Ar4PickPlaceMirrorEnvCfg` completely
unmodified (as required, so any future demonstration trajectories would
have matched the RL policy's exact action/observation format). Grasp+lift
gate scoring (`compute_gate_fires`, reproducing `antipodal_grasp_bonus`'s
force-closure criteria) and per-episode trajectory recording to
`demonstrations/oracle/*.npz` were both implemented and smoke-tested
successfully — the plumbing works. The blocker is entirely in the
oracle's ability to actually reach the waypoints in the first place.

## The stall: discovery and root-cause investigation

**Original discovery** (`.superpowers/sdd/task-1-report.md`): a
fine-grained diagnostic showed the classic signature of a fixed point —
distance to the active waypoint genuinely decreases for the first ~15-20
steps of an episode (real progress, ~0.05m/step matching
`IK_PURSUIT_MAX_STEP`), then reverses, then plateaus at a stable pose well
short of the target, frozen for the rest of the episode. Combined sample
across smoke-test runs: **1 of 24 sampled envs** advanced past waypoint 0.

**Three architecturally distinct fix attempts, all failed:**

1. **Cartesian-space stall-detection + escape-perturbation** (ported
   directly from this repo's `classical_pickplace_demo.py`, same proven
   constants). Verified the mechanism itself fires correctly
   (`escape_steps_remaining` cycles down and re-triggers exactly as
   designed), but produced **zero measurable effect** on distance-to-target
   (~0.3mm drift over 196 steps and 7 escape triggers). Root cause: the
   perturbation was added to the Cartesian pursuit target *before* the IK
   solve, and got absorbed by the same ill-conditioned Jacobian direction
   causing the stall in the first place.
2. **Joint-space perturbation** (inject the escape noise directly into
   `joint_pos_des` *after* the IK solve, bypassing the Jacobian entirely).
   Verified the perturbation genuinely reached the commanded joint target
   (raw actions swung by up to 0.6 rad step-to-step during escape), but
   **`joint_pos` stayed frozen bit-identical anyway**, at two different
   perturbation scales (0.05 rad, then 0.15 rad — tripling the magnitude
   produced zero additional movement).
3. **Integral (accumulated-error) correction** on the receding-horizon
   pursuit target, motivated by a mechanistic reading of the diagnostic
   evidence (see below): at `INTEGRAL_GAIN=0.02`, no measurable effect,
   all 16 sampled envs still stuck at waypoint 0. At `INTEGRAL_GAIN=0.05`
   (the one permitted follow-up attempt), **envs actively diverged away
   from the target** (e.g. one env's distance grew 0.46m → 0.65m) — a
   materially worse outcome than doing nothing, and a strong signal that
   near this pose, a larger step in the "linearized correction" direction
   makes real (nonlinear) progress *worse*, not better.

**Root-cause diagnostics (all independently re-verified by a second
subagent before being trusted, not just accepted from the first report):**

- **Not actuator torque saturation.** `computed_torque` (unclamped PD
  output) exactly equals `applied_torque` (post-clip) at every single
  logged step across two independent runs — the effort-limit clip never
  engages. Peak torque observed anywhere, even during genuine motion, was
  14.1 N·m, comfortably under the 20.0 N·m `effort_limit_sim`. During the
  stall itself, steady torques sit around 1-8 N·m per joint — 5-36% of the
  ceiling.
- **Not a hard joint-limit hit.** Closest any joint gets to its
  configured position limit during the frozen window is ~13° of margin.
- **Not contact/collision.** Both gripper jaw contact sensors read exactly
  0.0 N throughout the entire stall window (this waypoint is well before
  any grasp attempt, so this was mostly a sanity check, but it rules out
  an unexpected early collision).
- **Not a classic rank-deficient Jacobian singularity.** The Jacobian's
  smallest singular value plateaus at ~0.1515 during the stall — nowhere
  near zero. Manipulability (`sqrt(det(J·Jᵀ))`) plateaus at ~0.016 —
  small, but not degenerate in the textbook sense.
- **Consistent instead with a fixed point of the receding-horizon control
  loop in a poorly-conditioned kinematic region**: nonzero, non-saturating
  torque at zero joint velocity is the textbook signature of a genuine
  physical steady state (not a bug reading stale data — independently
  reconfirmed with fresh instrumentation). The pursuit target is redrawn
  every step from wherever the arm currently is, so if a step's actual
  displacement falls short of its intended `IK_PURSUIT_MAX_STEP`, the next
  step's target simply gets recomputed from the same undershot position —
  a self-consistent equilibrium, not progress. The integral-correction
  result (worse, not better, when strengthened) suggests this isn't a
  simple linear droop fixable by more of the same corrective push; the
  local linearization itself appears to stop reliably pointing toward the
  goal in this region.

## Why this isn't "just needs one more tuning pass"

Per this project's systematic-debugging discipline: three fixes across
three different mechanisms (Cartesian-space perturbation, joint-space
perturbation, integral accumulation) have now failed, with the third
producing a *worse* outcome under a stronger setting — the textbook
pattern for "question the architecture," not "try fix #4." This also
independently echoes this repo's own prior finding from Experiment 20
(`docs/superpowers/plans/2026-07-07-ar4-experiment20-report.md`): a
damping sweep across six values, multiple target formulations, and
multiple orientation targets all converged on the same conclusion —
**hard-locking pose via a single-Newton-step-per-env-step differential IK
solve is not a stable mechanism for this specific arm**, independent of
target orientation or damping tuning. That finding was about sustained
6-DOF pose-holding; this investigation reached a structurally analogous
conclusion via a completely different path (waypoint-pursuit stalling
rather than orientation drift), reinforcing rather than merely repeating
it.

## Explicit Gate 1 verdict

**FAIL.** Both required conditions fail:
- `N >= 30` of 50 episodes passing the grasp+lift gate: not measured via
  the formal Task 4 run, because the precondition (reliably traversing
  even the *first* waypoint) is not met — combined evidence across all
  diagnostic runs this investigation touched (dozens of sampled envs
  across the original smoke tests and three fix-attempt verification
  runs) shows advancement past waypoint 0 in only a small minority of
  cases, with no fix found that improves this rate.
- Video-confirmed genuine grasps: not applicable — the oracle does not
  reliably reach the grasp waypoint at all.

Per the plan's own scope note, Gate 2's implementation plan is **not**
written as a result of this report. This falsifies the premise that a
demonstrable successful trajectory is straightforwardly obtainable from
this repo's current waypoint-following + live-differential-IK mechanism,
as implemented. Per the plan's own FAIL-branch guidance, this points to a
mechanism-level problem (the reactive IK control law), not an
exploration-only problem — and the plan's built-in "one bounded
refinement attempt" escape valve has been used (in fact exceeded: three
attempts, not one, given the ambiguity of what counts as "the" mechanism
fix going in).

## Recommendation for what comes next

Do not attempt a fourth patch on the reactive single-Newton-step IK
control law — the evidence (especially the integral-correction reversal)
suggests the problem is in the fundamental mechanism, not a missing gain
or a missing perturbation trick. Two structurally different paths forward,
not mutually exclusive:

1. **A genuinely different oracle mechanism**: multi-iteration IK
   convergence *before* ever calling `env.step()` (proper Newton-Raphson
   refinement toward each waypoint using repeated local linearizations
   with re-evaluated forward kinematics, rather than one linearized step
   per physical env step), or a joint-space path planner that doesn't rely
   on per-step Cartesian re-linearization at all. This is real new
   engineering, not a quick fix — appropriately scoped as its own Tier 1
   decision if pursued.
2. **Reconsider whether a scripted, non-human oracle is the right
   demonstration source at all** for Experiment 24's underlying goal
   (bootstrapping BC pretraining). A human teleoperator doesn't get stuck
   in a differential-IK local fixed point the way an open-loop reactive
   controller does — this was exactly the constraint Gate 1 was designed
   to avoid needing, but the difficulty encountered here is evidence that
   avoiding it costs more than assumed.

This is a genuine fork in Experiment 24's direction, not a mechanical next
step — it should go through this repo's normal Tier 1
brainstorm/research/spec/plan process rather than being decided inside a
debugging report.

## Files changed this investigation

- `scripts/oracle_rollout.py`: core oracle (Task 1-3), three fix attempts
  (2 reverted-in-spirit by later commits removing the dead code, 1 left
  in place — the integral mechanism is currently present but proven
  ineffective; left as-is rather than reverted, since it's harmless
  dead-weight-but-documented code, not actively wrong).
- This report.
- `ROADMAP.md` follow-up entry (this session).

No push to origin from subagents during this investigation — final commit
of this report and the ROADMAP update will be pushed by the controller.
