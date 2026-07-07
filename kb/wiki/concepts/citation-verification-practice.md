# Citation-verification practice

## The pattern

Nearly every literature-research pass commissioned during this project's
research arc followed the same two-tier structure: a junior researcher
produces a first-pass literature review, and an independent senior review
re-verifies every citation against primary sources before any claim is
allowed to inform a design decision. This concept article isn't about any
single experiment's mechanism — it's about a recurring, load-bearing
methodological practice that shows up across nearly every reward/action-
space decision in this project, and about how often it actually catches
real problems (not a formality).

## What the senior review has caught, across this project's history

- **Grasp-alignment research** (behind
  [[experiment-01-contact-sensor-grasp-reward]]'s precursor, the
  multiplicatively-gated alignment bonus — see
  [[reward-hacking-and-sparse-discoverability]]): a fabricated verbatim
  quote and a fabricated `std=0.02m` claim, plus two misapplied citations,
  caught before the design was finalized.
- **Grasp-scale/PD-gain research**: 11 citations, no fabrications this
  round (a first) — but the headline recommendation was found overstated:
  the cited paper's own three-way split says stiff/overdamped gains mainly
  harm *sim-to-real transfer*, not pure in-sim PPO convergence, which is
  all this repo does. The specific rescaled-gain target numbers were the
  junior's own extrapolation, not literature-backed. The eventual empirical
  null result (gains had no effect) was noted as *consistent with* this
  correction, not contradicting it — a case where catching an overstated
  citation correctly predicted a null result before the experiment ran.
- **Lift-reward literature research** (behind
  [[experiment-03-always-on-lift-height]] and
  [[experiment-04-sa-ppo-lr-bump]]): the junior's first pass cited two
  real-but-off-topic multi-objective-RL papers as if they specifically
  documented a "grasp-reward-vs-lift-motion conflict," and included a
  fabricated "2-3x safety factor" number — both caught independently by
  the senior review and the junior's own Google-Scholar-first re-pass (a
  convergent signal, not just one reviewer's opinion). What survived
  verification (PPO entropy collapse as the likely mechanism, Li et al.
  *Sensors* 2025; potential-based reward shaping's real theoretical
  guarantee, Ng/Harada/Russell ICML 1999) directly shaped
  [[experiment-04-sa-ppo-lr-bump]] and
  [[experiment-05-potential-based-reward-shaping]].
- **RL-manipulation and classical-manipulation research** (behind
  [[experiment-09-antipodal-grasp-bonus]]): two senior-tier passes
  commissioned in parallel, both independently converging on real,
  actionable, externally-verified findings (the 118:1 reward-rate
  imbalance cross-checked against three external reference implementations
  read directly, and the magnitude-vs-antipodal distinction cross-checked
  against five classical grasp-mechanics sources) — no fabrications caught
  this round, but the convergent-yet-independent structure of the two
  passes is itself part of why [[experiment-09-antipodal-grasp-bonus]]'s
  redesign was trusted enough to bundle both fixes into one change.
- **Perception/sensing literature research** (out of this pass's scope —
  perception work is not covered per `kb/README.md` — but the same pattern
  recurred there too: a fabricated verbatim quote and a fabricated
  "3-6mm RMS depth noise" figure built on garbled arithmetic, echoing the
  same fabricated-precision-number pattern seen in the grasp-reward
  research).

## Why this matters for the wiki itself

`kb/README.md` names a directly analogous risk for this wiki: an LLM
linting a wiki it wrote itself can be confidently wrong in a way that's
internally consistent, propagating its own error through backlinks it also
authored. The citation-verification practice documented here is the
project's existing answer to exactly that class of risk in the research
process — health-checks for this wiki should apply the same principle
(re-derive claims from `ROADMAP.md`/specs/plans directly, not just check
the wiki's internal consistency).

## Related experiments

[[experiment-01-contact-sensor-grasp-reward]], [[experiment-03-always-on-lift-height]],
[[experiment-04-sa-ppo-lr-bump]], [[experiment-05-potential-based-reward-shaping]],
[[experiment-09-antipodal-grasp-bonus]]
