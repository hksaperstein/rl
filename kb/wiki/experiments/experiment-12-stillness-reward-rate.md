# Experiment 12: Stillness-penalty reward-rate fix

**Object:** cube. Direct arithmetic follow-up on
[[experiment-11-taskspace-ik]]'s "reach, grasp, freeze" video signature.

## Hypothesis

A direct arithmetic check of Experiment 11's reward weights found that
holding a grasp without further progress netted **+1.0/step**
(`antipodal_grasp_bonus`'s continuous +3.0/step, only partly offset by
`stillness_penalty`'s -2.0/step once its 25-step patience window elapsed) —
a real, previously-unverified bug directly matching Experiment 11's
observed freeze behavior. Raising `stillness_penalty`'s weight to restore a
net -2.0/step penalty for stagnation should remove the incentive to freeze.

## What changed

`stillness_penalty`'s weight raised 2.0 → 5.0 in
`pickplace_taskspace_env_cfg.py` only (same net -2.0/step target spelled
out in the design doc) — nothing else changed from Experiment 11.

## Quantitative result

**Genuinely mixed, not a clean win or loss.** `antipodal_grasp_bonus`'s
final value dropped 32% versus Experiment 11 (0.018815 → 0.012777) — but
its *nonzero rate* rose (91.6% → 93.2% of iterations), and
`stillness_penalty` became *less* negative despite its weight increasing
2.5x (-0.002533 → -0.001857, meaning the "grasped and stagnant" condition
fired *less*, i.e. less frozen time) while both outcome-oriented metrics
improved (`path_proximity_bonus` +8%, `cube_reached_goal` +5.4%). The
implementer's own report initially misread the antipodal drop alone as
proof the fix failed; the controller rejected that verdict as premature —
the drop is exactly what a proxy term that pays for *static* holding would
do if the policy is holding *less* statically, which is not distinguishable
from failure using scalars alone.

## Qualitative video finding

3 of 10 recorded episodes, 25 frames each at 5fps, personally inspected
with cropped/upscaled close-ups of the gripper region — **does not resolve
the ambiguity.** Episode 1: arm settles into a low pose near the cube's
spawn area by ~1s and holds it for the rest of the episode, materially the
same signature as Experiment 11's video. Episode 2: arm folds into a
compact pose close to its own base; a small reddish sliver is visible at
the fold but not clearly identifiable as a held cube from this camera
distance. Episode 3: unambiguous — a distinct red cube sits stationary on
the ground for the entire episode while the arm folds down near its own
base, never engaging it at all — a specific failure signature not seen in
Experiment 11's single inspected episode. **None of the 3 episodes show the
cube leaving the ground.** Given training's own `cube_reached_goal` rate is
only ~1%, a 3-episode sample is not powered to distinguish "no improvement"
from "same low success rate, different per-episode failure mode by
chance."

## Verdict

**Inconclusive, not negative.** The reward-rate bug this experiment fixed
was real and independently verified (both by the arithmetic and by
`stillness_penalty`'s reduced firing rate in the actual run) — but neither
the scalars nor the video sample are strong enough evidence to say whether
fixing it changed observable pick-and-place behavior. "Pick up and move" as
a whole remains unachieved. This result doesn't on its own justify either
doubling down on reward-rate tuning or abandoning the direction — the
still-undone episode-length/staged-decomposition ideas remain the most
likely candidates to produce an unambiguous behavioral change, since they
address a complementary hypothesis (episode too short/unguided for
lift+carry+place to be discoverable, independent of the freeze incentive
fixed here).

## Related concepts

[[reward-rate-arithmetic]] — this experiment's central subject; the fix is
directly analogous to Experiment 6's earlier stillness-penalty sign bug.
[[reach-grasp-lift-gap]] — an ambiguous data point in the through-line, not
a resolution.

## Sources

`docs/superpowers/specs/2026-07-07-ar4-experiment12-stillness-reward-rate-design.md`,
`docs/superpowers/plans/2026-07-07-ar4-experiment12-stillness-reward-rate-implementation.md`,
`docs/superpowers/plans/2026-07-07-ar4-experiment12-report.md`
