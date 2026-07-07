# PPO / critic-divergence failure modes

## The pattern

Twice this session, introducing a *new action-term mechanism* (not a reward
change) destabilized or risked destabilizing PPO training in a way no prior
reward-only or curriculum-only change ever did across Experiments 1–10.
Both cases involved `DifferentialInverseKinematicsActionCfg`-based control.

## Experiment 11: real critic divergence, caught by the controller

[[experiment-11-taskspace-ik]]'s first full run's PPO critic diverged: `Mean
value_function loss` exploded from ~0.0000 to ~1.56 to ~4047 to ~3.2M
between iterations 66–69/1500, reaching ~5.2e23 by the final iteration and
never recovering. ~95% of the run's policy updates were driven by a
diverged critic. This was independently traced by the controller directly
from `/tmp/exp11_train_stdout.log` — **the implementer's own status report
had called it "Non-Critical,"** an instance where a subagent's severity
judgment did not hold up under independent verification. Root cause: an
outlier raw policy action, previously harmless under
`JointPositionActionCfg` (which saturates cleanly at joint limits), likely
drove the new `DifferentialInverseKinematicsActionCfg` term's IK solve into
a discontinuous joint-space jump that destabilized PhysX for one env/step,
producing an extreme observation the critic couldn't fit. Fix:
`clip_actions=5.0` (~3.4x the observed action-noise std of 1.46), scoped
only to the taskspace experiment's own PPO runner config. Verified on both
a 300-iteration diagnostic and the full 1500-iteration re-run —
`Loss/value_function` stayed bounded (max 7.88) for the entire corrected
run.

## Experiment 13: a config bug that would have silently disabled the mechanism

[[experiment-13-residual-rl]]'s `Ar4PickPlaceResidualEnvCfg`-adjacent action
Cfg subclass was originally missing the `@configclass` decorator. Per Isaac
Lab's dataclass-based config machinery, this would have silently kept the
*parent* class's `class_type` default (plain, non-residual
`DifferentialInverseKinematicsAction`) — meaning the entire residual
mechanism would never have run, **with no exception raised**. This was
caught in implementation review before training, and independently
re-verified by reproducing the exact dataclass-default failure/fix
standalone rather than trusting inspection alone. Unlike Experiment 11, this
experiment's actual `Loss/value_function` stayed healthy throughout both the
diagnostic and full runs (max 0.17) — the eventual behavioral regression in
this experiment was real, but it was not a critic-stability problem; it was
attributed to a missing warm-start period for the residual (see
[[action-space-design]]).

## The general lesson

In this project's data so far, **new action-term mechanisms are the
recurring source of critic instability, not reward-function changes
themselves** — every reward/curriculum/optimizer-only change across
Experiments 1–10 and 12/14 left `Loss/value_function` well-behaved; both
critic-risk events (one realized, one caught before it could happen)
coincide exactly with the two experiments that changed the action term.
This argues for extra scrutiny (smoke tests, a short diagnostic run with
explicit loss-curve inspection, not just exit-code/"did it run" checks) any
time a new action space or action-generating mechanism is introduced,
independent of whether the reward function changed at all.

## Related concepts

[[action-space-design]] — the action-space changes that triggered both
incidents. [[reward-rate-arithmetic]] — a different failure axis (reward
arithmetic, not optimizer stability) that happens to co-occur with these
same two experiments' aftermath.

## Related experiments

[[experiment-11-taskspace-ik]], [[experiment-13-residual-rl]]
