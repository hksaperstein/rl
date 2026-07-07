# Experiment 13: Residual RL over a classical waypoint-seeking base controller

**Object:** cube. A structurally different pivot, not another reward tweak,
per this project's mandate to escalate after [[experiment-12-stillness-reward-rate]]'s
inconclusive result rather than keep tuning the same paradigm.

## Hypothesis

Rather than commanding the raw task-space delta directly, have the policy
command only a small correction on top of a classical proportional ("seek")
controller that pursues the already-computed active waypoint — additive
superposition, per Silver et al. 2018 "Residual Policy Learning"
(arXiv:1812.06298) and Johannink et al. 2019 "Residual Reinforcement
Learning for Robot Control" (ICRA), both verified directly.

## What changed

New action term `ResidualDifferentialIKAction`
(`tasks/ar4/residual_ik_action.py`), new `Ar4PickPlaceResidualEnvCfg`
(`tasks/ar4/pickplace_residual_env_cfg.py`) reusing Experiment 12's exact
reward weights unchanged.

Implementation review caught a real, separate bug before training: the Cfg
subclass was originally missing `@configclass`, which — per Isaac Lab's
dataclass-based config machinery — would have silently kept the *parent*
class's `class_type` default (plain, non-residual
`DifferentialInverseKinematicsAction`), meaning the entire residual
mechanism would never have run, with no exception. Fixed and independently
re-verified (reproduced the exact dataclass-default failure/fix standalone)
before any training happened; a post-fix smoke test (`params/env.yaml`)
confirmed the fixed action term actually got selected and constructed
correctly inside Isaac Sim.

## Quantitative result

Diagnostic (300 iter) and full (1500 iter) runs both showed
`Loss/value_function` staying bounded (max 0.17, actually healthier than
Experiment 12's own reference run) — no critic divergence, ruling out the
specific failure class Experiment 11 hit under a different new action term.

Scalar comparison against Experiment 12 is the largest, most one-sided
shift of any prior experiment-to-experiment comparison this session:
`antipodal_grasp_bonus` collapsed -98.9% (0.012777 → 0.000140, though still
nonzero on 80.4% of iterations, down from 93.2%); `stillness_penalty` got
*worse*, not better (-12.8%, more stagnant time than Experiment 12, the
opposite direction from a "shorter, more efficient holds" explanation);
`path_proximity_bonus` improved marginally (+5.3%); `cube_reached_goal`
dropped -44.2%.

## Qualitative video finding

3 of 10 recorded episodes, personally inspected (cropped, dense-frame review
of episode 1's final ~15% for signs of genuine settling vs. continued
motion): a new, materially worse failure signature not seen in Experiments
11–12. Episode 1: the arm never settles — frame-by-frame comparison of the
final four sampled frames shows the arm still visibly, continuously
folding/collapsing right up to the end, ending in a compact, collapsed heap
near its own base — ongoing instability, not a static bad pose. Episode 2:
reaches an elevated diagonal pose that does look stable from partway
through onward, cube occluded from this camera angle, genuinely ambiguous.
Episode 3: a clean repeat of Experiment 12's episode-3 signature — the arm
settles into a static pose that never reaches the cube at all. None of the
3 episodes show a lift.

## Verdict

**Genuine regression, not an inconclusive result** — unlike Experiment 12,
the scalar picture (stillness_penalty moving the wrong direction) and the
video evidence (a new, actively-unstable failure mode) corroborate each
other rather than pointing in different directions. Root-cause hypothesis,
not yet tested: the base controller's pursuit step is unconditional —
it fires every step toward whichever waypoint is currently active,
regardless of the policy's residual, with no gating on grasp state. Both
cited papers explicitly warm-start the residual to avoid exactly this kind
of early-training conflict (Johannink et al. hold the residual fixed at
zero for an initial period while training only the value function). **This
experiment's design did not implement that warm-start** — actor and critic
trained jointly from iteration 0, the residual summed onto the base
controller's step every step from the start. Recommended next step: either
(a) retry with the literature's warm-start technique properly implemented,
or (b) per the project's mandate to pivot after a second non-improving
result in a row, move to the still-undone episode-length/staged-
decomposition direction instead — both flagged as legitimate, this is a
real decision point, not a default.

## Related concepts

[[action-space-design]] — a second task-space variant (residual-over-
classical-base) tested after Experiment 11's plain task-space IK; the
residual formulation itself, not task-space control generally, is
implicated in the regression. [[ppo-critic-divergence]] — the second
new-action-mechanism instability bug this session, but this one is a
config bug caught before training, and the critic itself stayed healthy;
contrast with Experiment 11's actual critic explosion. [[reach-grasp-lift-gap]]
— a regression in the through-line, not progress.

## Sources

`docs/superpowers/specs/2026-07-07-ar4-residual-ik-action-design.md`,
`docs/superpowers/plans/2026-07-07-ar4-residual-ik-action-implementation.md`,
`docs/superpowers/plans/2026-07-07-ar4-experiment13-report.md`
