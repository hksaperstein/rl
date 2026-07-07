# Experiment 5: Monotonic potential-based reward shaping

**Object:** sphere. Second of the two literature-backed candidates flagged
by [[experiment-03-always-on-lift-height]], run regardless of
[[experiment-04-sa-ppo-lr-bump]]'s outcome per explicit user instruction.

## Hypothesis

Ng, Harada & Russell's potential-based reward shaping (ICML 1999, "Policy
Invariance Under Reward Transformations") offers a genuine theoretical
guarantee that decomposing reach→grasp→lift as a potential-function chain
doesn't change what the optimal policy is — a principled alternative to the
ad-hoc curriculum-timing approach already tried twice.

## What changed

Replaced six independent additive reward terms with a single running-max
potential-based term: `gamma * new_potential - prev_potential`, where the
potential Φ tracks the best-ever stage reached (reach → grasp → lift →
carry → place).

## Quantitative result

`Episode_Reward/staged_potential_progress` **declined to -0.109** over
training instead of growing.

## Qualitative video finding

**0/10 real eval episodes show any lift** — identical "reach, grip, freeze"
signature to every prior experiment; markers in this run never even show
real grasping/lifting attempts, just approach-and-hover.

## Verdict

**Falsified — and a genuine formula bug found, not just another null
result.** The term's docstring claimed `gamma * new_potential - prev_potential`
is "always >= 0" — false whenever the agent merely *holds* its best-ever
potential without improving further: `new_potential == prev_potential == Φ`
gives reward `Φ * (gamma - 1)`, which is **negative** for any `gamma < 1`
(here `gamma=0.98`). Over a ~225-step episode, reaching the object and
holding there (`Φ ≈ 0.1`) costs roughly `0.1 * (-0.02) * 225 ≈ -0.45` total
reward — *worse* than never approaching the object at all (`Φ` stays 0 the
whole episode, reward stays exactly 0). The policy that minimizes this cost
is to never reach for the sphere, exactly what the eval showed. This is a
bug in the shaping formula's discount handling, not evidence against
potential-based shaping as an approach. Fifth real falsified attempt on the
reward/optimization axis for this sub-problem; per the systematic-debugging
Phase 4.5 mandate this would normally be flagged back rather than attempt a
sixth unilateral tweak — here the user independently raised two related,
concrete ideas in parallel (a grasp-gated stillness penalty and full-
workspace randomization with a mirrored goal), making the sixth attempt
user-directed (see [[experiment-06-mirror-scene-stillness-penalty]]).

## Related concepts

[[reward-rate-arithmetic]] — the earliest instance of a discount/sign bug
that inverts a reward term's intended incentive, a bug class recurring
later in Experiment 6's own stillness-penalty sign error and Experiment 12's
stillness-penalty weight-rate imbalance. [[reach-grasp-lift-gap]] — fifth
attempt, still falsified.

## Sources

`docs/superpowers/specs/2026-07-06-ar4-sphere-lift-potential-shaping-design.md`,
`docs/superpowers/plans/2026-07-06-ar4-sphere-lift-potential-shaping-report.md`
