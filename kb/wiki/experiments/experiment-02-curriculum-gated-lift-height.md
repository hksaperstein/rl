# Experiment 2: Curriculum-gated dense lift-height reward

**Object:** sphere. Direct follow-up to [[experiment-01-contact-sensor-grasp-reward]]:
grip is now reliably achieved, but the arm never subsequently attempts to
lift.

## Hypothesis

Now that grip is reliably achieved, staging lift as an explicit next phase
(via a curriculum that only rewards lift progress once grasp is already
well-established) may work where an always-competing lift signal wouldn't
have before grip was solved.

## What changed

Added a new dense, `tanh`-shaped `lift_height_progress` reward term, inert
(`weight=0.0`) during phase-1 reach+grip training and switched on
(`weight=15.0`) at iteration 700 via Isaac Lab's own `modify_reward_weight`
curriculum term — the same mechanism the Franka lift task this repo's
rewards were adapted from uses for its own curriculum. The switch point
(iteration 700) was timed to this run's own TensorBoard data, where
`grasp_contact` plateaus by iteration ~600–750.

## Quantitative result

The curriculum mechanism itself fired exactly as designed —
`lift_height_progress` reads `0.0` at iteration 699, nonzero at 701,
confirmed directly from the raw TensorBoard scalars. But its real-world
magnitude was negligible: the logged `Episode_Reward` max of `0.0065` is
`weight(15.0) × mean per-step tanh value`, so the real per-step `tanh` is
~`0.00043`, corresponding to roughly **0.0043mm** of real height gain — many
orders of magnitude short of the 21mm `lifting_sphere` actually requires.
`lifting_sphere` itself never rose above noise (max `0.0027`, comparable to
the ContactSensor baseline's own transient blip), including after the
curriculum switch.

## Qualitative video finding

10 episodes, frame-extracted, all inspected directly: **0/10 show any real
lift** — the same "reach, grip, freeze" static-pose signature as
[[experiment-01-contact-sensor-grasp-reward]]. Two episodes showed the
sphere briefly vanish from a single sampled frame each; adjacent-frame
inspection confirmed the sphere reappearing at the identical ground-level
position next to the gripper in both cases — a viewing-angle occlusion
artifact, not a lift.

## Verdict

**Falsified.** The curriculum window opened too late and/or too weak
relative to how deeply the static-grip behavior had already converged by
iteration 700 (`grasp_contact` was already at ~17.8/20, essentially its
plateau, at the switch point). This rules out "the sparse `lifting_sphere`
signal was the only problem" as a complete explanation; entrenchment of the
static-grip optimum itself looks like the more likely bottleneck. Not
extended with further unilateral tuning — flagged back as a decision point.

## Related concepts

[[reach-grasp-lift-gap]] — an explicit, deliberate attempt to close the gap
by staging lift as its own phase; doesn't move the needle. [[reward-hacking-and-sparse-discoverability]]
— a correctly-designed, non-hackable dense term that's simply too weak,
relative to an already-entrenched competing optimum, to be discovered.

## Sources

`docs/superpowers/specs/2026-07-06-ar4-sphere-lift-curriculum-design.md`,
`docs/superpowers/plans/2026-07-06-ar4-sphere-lift-curriculum-implementation.md`,
`docs/superpowers/plans/2026-07-06-ar4-sphere-lift-curriculum-report.md`
