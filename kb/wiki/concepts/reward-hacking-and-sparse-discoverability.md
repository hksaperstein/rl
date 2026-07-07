# Reward hacking vs. sparse-signal discoverability

## The tradeoff

Two failure modes recur on opposite ends of the same axis whenever this
project adds a new dense-shaping term to encourage an under-explored
behavior: a term loose enough to be *discoverable* by random exploration is
often loose enough to be satisfied without the real target behavior
(reward hacking); a term tightened to be geometrically *correct* is often
tightened past the point unguided exploration can ever stumble into it
(discoverability failure). Both failure modes appear in this project's
sphere-era precursor experiments, before the numbered Experiment-1-through-
14 sequence begins, and the tension between them motivates several of the
numbered experiments' own design choices.

## The two clearest instances (unnumbered sphere-era precursors)

- **Dense proximity+closure grasp bonus (reward-hacked).** A static dense
  reward term (`grasp_sphere`, adapted from Isaac Lab's own
  `manipulation/cabinet` task's `grasp_handle` pattern) rewarded closing the
  gripper within 4cm of the sphere. The term was fully learned and
  saturated near its theoretical max well before training ended — but real
  eval video showed 0/10 episodes with a genuine grasp: the gripper closed
  *beside* the sphere, not around it. The reward only checked EE-to-object
  distance + gripper closure, with no check that the object was actually
  enclosed between the fingers — exactly the kind of loose-but-hackable
  term this tradeoff predicts. This is qualitatively worse than a no-op
  failure (a later concept, [[reward-rate-arithmetic]], catches the same
  "worse than useless" character in other terms) because a trivially-
  satisfiable dense term risks entrenching a fake-grasp local optimum
  against future fixes — the code change was reverted, not merged, unlike
  harmless no-op changes kept forward.
  (`docs/superpowers/specs/2026-07-05-ar4-sphere-grasp-bonus-design.md`,
  `docs/superpowers/plans/2026-07-05-ar4-sphere-grasp-bonus-report.md`)
- **Multiplicatively-gated alignment bonus (too sparse to discover).** The
  direct fix to the above — per GRIT's verbatim-confirmed `r_h·α_h`
  multiplicative pattern (arXiv:2604.04138) — required true centering (1cm
  `centering_std` window) between the sphere and the fingertip midpoint,
  multiplicatively gating the closure reward by this alignment score. The
  gate was structurally un-hackable (no partial credit for a near-miss) but
  stayed at noise level (max 0.00207, ~0.7% of its theoretical max) for the
  entire 1500-iteration run — a sharp contrast with the reward-hacked
  term's near-saturation. The policy's exploration noise essentially never
  produced the joint (position, orientation, closure) combination needed
  to get any nonzero signal — tight relative to the sphere's own ~9mm
  radius and the gripper's small travel range.
  (`docs/superpowers/specs/2026-07-05-ar4-sphere-grasp-alignment-design.md`,
  `docs/superpowers/plans/2026-07-05-ar4-sphere-grasp-alignment-report.md`)

## How later, numbered experiments relate to this tradeoff

[[experiment-01-contact-sensor-grasp-reward]] sidesteps the tradeoff
entirely for the grasp sub-problem by using ground-truth contact sensing
rather than a geometric proxy — neither too loose nor too tight, because
it measures the real physical event directly rather than approximating it.
But the same tension resurfaces one level up, for the *lift* sub-problem:
[[experiment-02-curriculum-gated-lift-height]] and
[[experiment-03-always-on-lift-height]] both introduce dense,
non-hackable `tanh`-shaped lift-height terms that are never reward-hacked
(their formulas are directly inspectable and correct) but also never
produce meaningful real height gain (0.0043mm and 0.0141mm respectively,
against a 21mm requirement) — closer to the discoverability-failure end of
the spectrum, though the leading explanation there shifts to PPO entropy
collapse around an already-converged, safe static-grip optimum (see
Experiment 3's citation-verified literature finding, Li et al. *Sensors*
2025) rather than pure signal sparsity.

## Related concepts

[[reach-grasp-lift-gap]] — the broader through-line these two failure
modes are instances of. [[citation-verification-practice]] — the GRIT
citation backing the alignment-gate design was itself verbatim-confirmed
during this same research pass, alongside two misapplied citations that
were caught and struck.

## Related experiments

[[experiment-01-contact-sensor-grasp-reward]], [[experiment-02-curriculum-gated-lift-height]],
[[experiment-03-always-on-lift-height]]
