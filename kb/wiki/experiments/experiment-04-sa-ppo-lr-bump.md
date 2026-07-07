# Experiment 4: SA-PPO-style dynamic learning-rate bump

**Object:** sphere. First of two literature-backed candidates flagged by
[[experiment-03-always-on-lift-height]]'s research pass, tested in
isolation against the identical starting policy.

## Hypothesis

If policy entropy has collapsed around a safe, static-grip local optimum
(the leading explanation from Experiment 3's literature review), then a
deliberate learning-rate bump at the point that optimum has converged
(SA-PPO's simulated-annealing-style mechanism) may reopen exploration enough
to discover lift.

## What changed

Resumed training from `model_700.pt` (Experiment 3's own "grip converged,
exploration about to collapse" checkpoint — the identical starting point)
with `learning_rate` bumped `1e-4` → `1e-3` and `schedule` switched
`"adaptive"` → `"fixed"`. No reward-function changes — this isolates the
learning-rate intervention alone against the exact same starting point as
Experiment 3.

A correction was made during final review: an earlier draft of this
experiment's own rationale claimed the adaptive schedule would "claw the
bump back down" given the converged policy's low KL divergence. Checking
`rsl_rl/algorithms/ppo.py:281-284` directly showed the opposite — low KL
divergence **increases** the adaptive learning rate, not decreases it. This
didn't invalidate the experiment (`schedule="fixed"` is still correct for a
controlled test — it guarantees the rate stays at exactly the intended
value rather than drifting under `"adaptive"`'s own dynamics), only the
stated reason for needing it was wrong and was corrected in the record.

## Quantitative result

The learning rate was confirmed to hold at `0.001` across the entire
1500-iteration continuation (no decay back toward baseline) — the
experiment genuinely tested its premise. `lifting_sphere`'s downsampled
trajectory reads exactly `0.0` at every sample, with a run `max` of `0.0027`
— the same order of magnitude as Experiment 3's own transient blips, not
more null and not less.

## Qualitative video finding

**0/10 real eval episodes show any lift** — same "reach, grip, freeze"
signature as every prior experiment.

## Verdict

**Falsified.** A substantial, sustained, correctly-held optimizer-level
perturbation, injected at precisely the point the literature identified as
critical, produced no measurable improvement. This argues against
"insufficient exploration pressure at the right moment" being a sufficient
explanation on its own, at least via this specific lever. Per the user's
explicit "try both" instruction, this did not gate the second candidate
([[experiment-05-potential-based-reward-shaping]]), which proceeded
regardless of this result.

## Related concepts

[[reach-grasp-lift-gap]] — fourth attempt on the same sub-problem, still
falsified. [[ppo-critic-divergence]] — a related but distinct PPO-dynamics
intervention; unlike Experiments 11/13's action-mechanism-driven critic
instability, this experiment deliberately and safely perturbed the
optimizer without destabilizing it.

## Sources

`docs/superpowers/specs/2026-07-06-ar4-sphere-lift-lr-bump-design.md`,
`docs/superpowers/plans/2026-07-06-ar4-sphere-lift-lr-bump-implementation.md`,
`docs/superpowers/plans/2026-07-06-ar4-sphere-lift-lr-bump-report.md`
