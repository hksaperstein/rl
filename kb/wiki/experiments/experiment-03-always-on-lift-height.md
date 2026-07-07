# Experiment 3: Always-on dense lift-height reward

**Object:** sphere. Direct follow-up to [[experiment-02-curriculum-gated-lift-height]],
which left open the question of whether the curriculum's iteration-700
switch point was simply too late.

## Hypothesis

Removing the curriculum gate entirely — making `lift_height_progress`
active from iteration 0 — isolates whether curriculum *timing* was the
problem, independent of the term's magnitude.

## What changed

`lift_height_progress` active from iteration 0 at `weight=25.0` (matching
`lifting_sphere`'s own weight), per the design doc's own "Revision" section
— no other change from Experiment 2.

## Quantitative result

`lift_height_progress` reached a measurably larger real value than the
curriculum experiment (~0.0141mm of real height gain vs. ~0.0043mm,
weight-normalized — a genuine ~3.3x increase), with an early-training bump
that faded as `grasp_contact` converged. Both figures remain many orders of
magnitude short of the 21mm `lifting_sphere` requires, and `lifting_sphere`
itself never moved off zero.

## Qualitative video finding

**0/10 real eval episodes show any lift** — the same "reach, grip, freeze"
static-pose signature as both prior experiments.

## Verdict

**Falsified.** Rules out "the curriculum switch came too late" as the sole
explanation, since removing the gate entirely produced the same outcome.
Citation-verified literature research (junior + independent senior review;
several claims from the junior's first pass — including a fabricated "2-3x
safety factor" number — were caught and struck) converged on: the likely
mechanism is **PPO entropy collapse**, not a specific grasp/lift reward
conflict — once a safe, reward-sufficient behavior is found, policy entropy
drops and exploration of riskier alternatives (like lifting) effectively
stops, even with a dense term nudging toward it (Li et al., *Sensors* 2025,
25(17):5253, DOI 10.3390/s25175253, targets exactly this "local optimum
trap" via a simulated-annealing+PPO hybrid). Grip force was independently
checked and found very unlikely to be the physical bottleneck (this repo's
measured ~20-30N contact force vs. the sphere's 0.098N weight). Two
literature-backed candidates not yet tried were flagged: SA-PPO-style
dynamic learning-rate adjustment once `grasp_contact` saturates (tested next,
see [[experiment-04-sa-ppo-lr-bump]]), and potential-based reward shaping
(see [[experiment-05-potential-based-reward-shaping]]).

## Related concepts

[[reach-grasp-lift-gap]] — a second, stronger attempt at the same fix,
still no lift. [[citation-verification-practice]] — a fabricated safety-
factor citation caught here, one of several such catches across the
session.

## Sources

`docs/superpowers/specs/2026-07-06-ar4-sphere-lift-curriculum-design.md`
(Revision section), `docs/superpowers/plans/2026-07-06-ar4-sphere-lift-always-on-implementation.md`,
`docs/superpowers/plans/2026-07-06-ar4-sphere-lift-always-on-report.md`,
`docs/superpowers/specs/research/2026-07-06-lift-reward-literature-junior.md`,
`docs/superpowers/specs/research/2026-07-06-lift-reward-literature-senior-review.md`
