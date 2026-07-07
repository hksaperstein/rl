# Experiment 14: Reach-skip curriculum

**Object:** cube. A structurally different pivot — the third non-improving
result in a row (12, 13) triggers the project's "escalate, don't keep
tuning" mandate — built on [[experiment-12-stillness-reward-rate]]'s clean
baseline, deliberately not on [[experiment-13-residual-rl]]'s unresolved
regression.

## Hypothesis

Three experiments running (11–13) all showed reliable grasp-contact but
never lift+carry+place, with `path_proximity_bonus`/`antipodal_grasp_bonus`
consistently indicating reach and grasp are the well-learned parts — so
remove reach from what the policy has to (re-)discover every episode,
reallocating the full step/exploration budget to the actually-unsolved
grasp→lift→carry→place sub-problem.

## What changed

New one-shot reset `EventTerm` `reset_arm_to_pregrasp_pose`
(`tasks/ar4/mdp.py`) computes a pregrasp joint configuration via a single
`DifferentialIKController` solve against each episode's just-randomized
cube position and writes it directly via `write_joint_position_to_sim`/
`set_joint_position_target`, run once at reset between `reset_cube_position`
and `randomize_goal`. New `Ar4PickPlaceReachskipEnvCfg`
(`tasks/ar4/pickplace_reachskip_env_cfg.py`) reuses Experiment 12's exact
reward weights and plain (non-residual) action term unchanged — isolating
the starting-state variable alone.

## Quantitative result

Diagnostic (300 iter) and full (1500 iter) runs both showed
`Loss/value_function` staying bounded (max 0.024436, matching the
diagnostic's own max almost exactly) — the cleanest, healthiest value-
function behavior of any new-mechanism gate this session (vs. Experiment
13's full-run max of 0.17), confirming the new one-shot direct-joint-state-
write reset event does not itself destabilize the critic.

Scalar comparison against Experiment 12 is mixed, deliberately not used
alone to call a verdict. `cube_reached_goal` improved modestly at the
final-iteration snapshot (+5.8%, 0.010773 → 0.011393), with a mid-run peak
of 0.022308 near iteration 750 — more than double the final value,
suggesting the final snapshot understates the effect mid-run.
`path_proximity_bonus` declined slightly (-8.4%). `antipodal_grasp_bonus`
dropped sharply (-94.4%, 0.012777 → 0.000709) but remained nonzero on 89.9%
of iterations, consistent with the same "less static holding time"
ambiguity flagged in Experiment 11→12's transition rather than a clean
regression signal on its own. `stillness_penalty` got 133% more negative
(-0.001857 → -0.004328) — plausibly explained by the changed episode/reset
structure (episodes now start mid-task, so per-episode accounting for these
terms isn't directly comparable to a full-reach-included episode) rather
than by worse behavior, but this is a hypothesis, not a confirmed
explanation.

## Qualitative video finding

3 of 10 recorded episodes, 25 frames each, personally inspected: no lift in
any episode, and a new failure signature in 2 of 3. Episode 1: the arm
reaches down to a low pose near the cube by ~frame 5 of 25 and holds a
static position near it for the rest of the episode — the same "reach and
freeze near the cube" signature seen in Experiments 11–13, just reached
faster (consistent with the reach sub-problem being skipped). Episode 2:
the arm does *not* settle into a stable hold — starting from an elevated
pose near the cube, it progressively folds into an increasingly compact
configuration close to its own base over the course of the episode
(visible ongoing changes in arm angle across frames 17, 21, 25), ending in
a low, contorted fold. Episode 3: the arm folds into a tight, contorted
pose near its own base almost immediately (already collapsed by frame 5)
and stays there; the cube — clearly visible, untouched — is never
approached at all. **None of the 3 episodes show the cube leaving the
ground, and no episode reaches waypoint index ≥2 (lift).**

## Verdict

**No improvement on the stated success criterion (reaching waypoint index
≥2 and/or genuine lift-off-the-ground in a meaningfully larger fraction of
episodes than the ~0/3 seen in Experiments 12–13), plus a new,
partially-explained failure mode — do not extend this exact mechanism with
further tuning.** The spec's own success bar was not met: 0/3 on lift, no
improvement. The spec also called out that a null result would itself be
informative by ruling out "the reach sub-problem is eating all the
exploration budget" as the explanation — that ruling-out did happen, but
with an unanticipated finding: 2 of 3 episodes show a *new* failure mode
(folding toward the robot's own base) not present in Experiments 12–13's
samples, where the more common failure was freezing near the cube or
disengaging from it, not actively collapsing toward the base. Root-cause
hypothesis, not yet tested: the design spec itself flagged that the
one-shot IK reset lands the arm "near," not exactly at, the computed
pregrasp target — depending on which IK solution the one-shot solve lands
on (elbow-up/down ambiguity, proximity to joint limits), some fraction of
episodes may start from an awkward or self-occluding joint configuration
the policy hasn't learned to recover from. This is the third experiment in
a row (12, 13, 14) that does not resolve "grasp/lift never emerges"; per
this project's mandate, the next experiment should be a genuinely different
lever, not a fourth variation in the same reward/action/curriculum family.
The base-collapse pattern seen here directly motivates (beyond the original
request) the cube-near-robot-base penalty term already directed for
Experiment 15, alongside wiring in the existing `ground_penalty` function
and raising `antipodal_grasp_bonus`'s weight.

## Related concepts

[[reach-grasp-lift-gap]] — third consecutive non-resolving experiment in
the through-line (12, 13, 14); still the project's central open problem at
the end of this pass's coverage. [[reward-rate-arithmetic]] — the
antipodal-drop-vs-stillness-penalty-worsening pattern recurs here in a
third variant.

## Sources

`docs/superpowers/specs/2026-07-07-ar4-reachskip-curriculum-design.md`,
`docs/superpowers/plans/2026-07-07-ar4-reachskip-curriculum-implementation.md`,
`docs/superpowers/plans/2026-07-07-ar4-experiment14-report.md`
