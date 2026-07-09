# Experiment 25: Touch-cube-then-reach-goal, grasp removed entirely

**Object:** cube. Direct user structural decision, not a spec-writer's own
next-lever choice, made after two blocking findings surfaced during
pre-training review of `pickplace_mirror_env_cfg.py`. This is the first new
experiment article compiled since
[[experiment-14-reach-skip-curriculum]] (2026-07-07) — Experiments 15
through 24 are not yet compiled into their own articles (the same
acknowledged gap `index.md`'s coverage-boundary note describes for
ROADMAP.md items 6-9) and this article does not attempt to backfill them; it
follows directly from ROADMAP.md items 9-10, not from Experiment 14 itself.

## Hypothesis

Not a new mechanism hypothesis in the usual sense — the actual claim tested
here is a scope-reduction one: if grasp and lift are dropped from the task
entirely and replaced with a two-stage sequential end-effector reach (touch
the cube's top, then reach a fixed goal point), the policy should converge
reliably, because reach is the one sub-behavior that has converged
(~0.92-0.95) across nearly every experiment in this project's history,
independent of reward design or action-space choice — unlike grasp/lift,
which has never been reliably closed despite 24 experiments' worth of
attempts.

## Background: why grasp/lift was dropped rather than fixed again

Before training `pickplace_mirror_env_cfg.py` "from scratch" as originally
planned (the direct next step after ROADMAP.md item 9's physics-fidelity
pass), two separate findings closed that off as a bad bet:

- **(a)** Six consecutive prior experiments (17-22) had each targeted a
  different angle on the same underlying mechanical defect — the gripper's
  two jaws are not actually mechanically coupled (the source URDF's `mimic`
  constraint is confirmed unenforced by Isaac Sim's USD import). Both a
  physics-level fix (Experiment 19, two configurations tried) and a
  software-level fix (Experiment 22, which introduced a new "reactive lag"
  failure mode) made the problem worse, not better, rather than resolving
  it.
- **(b)** `pickplace_mirror_env_cfg.py`'s own production reward
  (`staged_milestone_bonus`, built on `_raw_lift_progress_mirrored`) turned
  out to still combine reach/grasp/lift/goal as a plain **ungated** weighted
  sum — precisely the exploitable shape Experiment 16 already diagnosed. (In
  Experiment 16, a policy that scored well on a lift-shaped reward was later
  found, only after the user directly challenged the controller's own
  video read and a fresh instrumented rollout was run, to never actually
  grip the cube with its fingers at all — the cube was wedged against the
  wrist the entire "held" period, with both jaw contact sensors reading
  exactly 0.0000 at all 250 logged steps. See [[sim-physics-fidelity]]'s
  discipline of verifying visual/behavioral claims with real sensor
  instrumentation rather than eyeballed video frames — the same discipline
  this finding motivated.) `pickplace_mirror_env_cfg.py` never inherited
  Experiment 17's grasp-gating fix for this exact shape, which lives only in
  a separate env-cfg lineage (`pickplace_graspgated_env_cfg.py`).

This was flagged to the user rather than trained blind against those two
known risks. The user's direct decision: stop attempting a seventh fix to
the same jaw-coupling defect, and stop reusing a reward shape already known
to be exploitable — instead, **drop grasp/lift from the task entirely**.

## What changed

New `Ar4PickPlaceTouchGoalEnvCfg` (`tasks/ar4/pickplace_touchgoal_env_cfg.py`):
fixed cube position `(0.20, 0.28, 0.006)`, fixed goal position
`(-0.20, 0.28, 0.15)` (~0.42m apart), **arm-only action space** — no gripper
action term at all (gripper joints stay physically present but unactuated),
reusing the already-`_EE_OFFSET`-corrected `ee_frame` as the shared
touch/goal reference point. Built via subagent-driven-development: design
spec, implementation plan, 4 plan tasks each independently task-reviewed
clean.

## Pre-training review finding: running-max is unsound for spatially-opposed stages

The final whole-branch review (dispatched on the most capable available
model, this project's own convention for architecture-level review) caught
a real Critical defect **before any training run** — the kind of finding
this project's Tier-1 verification standard exists to catch early rather
than after a wasted 1500-iteration run.

The reused mechanism (`staged_milestone_bonus`'s running-max pattern) is
valid for the lift task it was built for, because that task's stages
(reach, grasp, lift, carry-to-goal) are **co-satisfiable along one
continuous trajectory** — progress toward a later stage does not require
moving away from wherever an earlier stage's potential peaked. Here it is
not: the touch point and the goal point are ~0.42m apart. Two independent
narrow `tanh` bumps (one centered at touch, one at the goal), summed, dip
from ~0.3 at touch to ~0.02 partway to goal before recovering to ~0.7 at the
goal — and under running-max, once 0.3 is banked at touch, the tracked
reward stays at **exactly zero** until the raw combined value re-exceeds
0.3 again, which does not happen until roughly 93% of the way to the goal.
This would very likely have produced "touch-and-freeze" — a failure
signature qualitatively indistinguishable from the sphere era's original
"reach, grip, freeze," and easily misread as "even reduced-to-reaching
fails" rather than what it actually would have been: a reward-mechanism
defect inherited from a task whose stage geometry doesn't transfer. The
defect was independently re-derived (not just trusted) by the Principal
before dispatching a fix, and again by a second reviewer after the fix
landed.

Also flagged in the same review: the goal was being read from the cube's
**live** position every step, but the cube is a dynamic (non-kinematic)
`RigidObject` — an incidental touch-contact nudge could silently move the
"fixed" goal mid-episode.

**Fix** (commit `7170b6b`, plus a trivial follow-up constant-drift fix):
extracted the reward math into a new Isaac-Lab-free module
(`tasks/ar4/touch_goal_reward.py`) with a genuinely monotonic post-touch
potential — `0.3 + 0.7·clamp(1 - goal_dist/touch_to_goal_dist, 0, 1)`,
linear in distance and provably non-decreasing along the straight
touch→goal line, so running-max can never stall once touch registers.
Added `set_touch_goal_position`, a reset-time event snapshotting the goal
once from the cube's position at reset (decoupling it from any later cube
displacement). Added `tests/test_touch_goal_reward.py` — a genuine
sim-independent pytest suite (3/3 passing, no Isaac Sim launch needed)
directly proving the monotonicity property the original formula lacked, a
new pattern for this project (its first non-perception pure-math unit
test). Independently re-reviewed and confirmed correct by re-deriving the
math from scratch, not by re-reading the fix's own claims.

## Verdict

**Touch is solved; reach-to-goal converges but the deployed (deterministic)
policy stalls just outside the 2cm goal tolerance, consistently, not
randomly.**

Two training runs. Run 1 (`episode_length_s=5.0`, copied from
`pickplace_mirror_env_cfg.py`) showed `goal_reached` peak at ~0.37-0.39
then decline to ~0.01-0.03, with episodes running to the timeout almost
every time — stopped before completion. Run 2 (`episode_length_s=20.0`,
re-derived from Isaac Lab's own reference-task episode-length conventions
— see [[hyperparameter-registry]]) completed cleanly: `goal_reached`
climbed and held at ~0.55-0.65, finishing at 0.5987, with
`Loss/value_function` small and bounded throughout.

An instrumented rollout of the trained checkpoint found the training-time
rate reflects PPO's own exploration-noise sampling, not the deployed
policy's reliability: deterministic action (`ActorCritic.act_inference`,
what a deployed policy actually uses) touched the cube in 32/32 rollout
episodes but reached the goal in only 2/32 (6.25%) — the misses cluster
tightly at 0.0175-0.0285m past the touch point, just outside the 0.02m
tolerance, not scattered. Stochastic action (`ActorCritic.act`, the same
sampling used during training) reached the goal in 29/32 (90.6%).
Independent recomputation from raw end-effector/cube/goal state agreed
with the actual termination signal on 100% of episodes in both
conditions, ruling out an instrumentation bug. A close-up single-env
camera (built for this check — the wide multi-env training camera proved
too low-resolution to confirm behavior, twice) shows the same shape
directly: the arm curls onto the cube by step ~22 (0.44s in), extends
toward the goal, and stops short of it at timeout.

This is also a process result, not just a training result: a
structurally unsound reward mechanism (two independent tanh proximity
bumps summed under a running-max mechanism, leaving a reward-free dead
zone across most of the touch-to-goal traverse) was caught and fixed
*before* it could produce a misleading null result, by the same
research-both-directions discipline (re-derive the defect independently,
don't trust a self-report) this project applies elsewhere. See
[[staged-reward-co-satisfiability]] for the generalized lesson, and
[[hyperparameter-registry]] for the episode-length derivation.

Superseded by a direct user decision to reintroduce the gripper next
(grasp/lift back in scope) rather than continue narrowing the goal
tolerance on this reduced task.

## Related concepts

[[reach-grasp-lift-gap]] — this experiment is the pivot point recorded in
that article's newest closing section: for the first time, the response to
"grasp/lift still doesn't work" was to remove the requirement rather than
attempt another mechanism fix. [[staged-reward-co-satisfiability]] — the
running-max dead-zone defect this experiment's pre-training review caught,
generalized as a methodology lesson for any future staged reward.
[[sim-physics-fidelity]] — the verify-with-instrumentation discipline
(Experiment 16's wedging finding) that directly motivated flagging risk (b)
rather than training blind.

## Sources

`docs/superpowers/specs/2026-07-09-ar4-experiment25-touch-goal-reach-design.md`,
`docs/superpowers/plans/2026-07-09-ar4-experiment25-touch-goal-reach-implementation.md`,
`ROADMAP.md` items 9-10.
