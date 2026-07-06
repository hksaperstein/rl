# Senior citation-verification review: lift-reward literature research

Reviews `2026-07-06-lift-reward-literature-junior.md`. Independent
verification (WebSearch/WebFetch against arXiv abstracts/PDFs) of every
citation used to answer the junior's 4 research questions on why the AR4
sphere policy grips reliably but never lifts.

## Question 1: reward conflict (grasp_contact vs. lift motion)

- **arXiv:2509.14816** ("Scalable Multi-Objective Robot RL through Gradient
  Conflict Resolution", Munn et al.) — resolves, real, evaluated on
  IsaacLab benchmarks generically. **No text found supporting the specific
  claim** that contact-maintenance rewards suppress exploratory lift
  motion in a grasp task. **Verdict: Misapplied/Overstated.**
- **arXiv:2403.00282** ("Conflict-Averse Gradient Aggregation for
  Constrained Multi-Objective RL", Kim et al.) — resolves, real, but is a
  general **safe/constrained RL** paper unrelated to manipulation or
  grasping. **Verdict: Misapplied** (off-topic paper cited as if
  domain-specific).
- Platt's contact-relative-motion grasp-synthesis report — real,
  topic-adjacent, but the specific "penalizing contact loss traps the
  policy in static hold" claim isn't supported in retrievable text.
  **Verdict: Misapplied/Unverifiable.**

**Conclusion: the central "gradient conflict" claim is not literature-
validated.** It's the junior's own plausible-sounding extrapolation,
citation-dressed. Treat "grasp_contact vs. lift is a documented reward-
conflict pattern" as an untested hypothesis, not a finding.

## Question 2: alternatives to additive contact+lift rewards

- **Ng, Harada, Russell, ICML 1999** ("Policy Invariance Under Reward
  Transformations") — **verbatim-confirmed.** Theorem and formula
  (`F(s,s') = γΦ(s') − Φ(s)`) match the real paper exactly. The
  application to this repo's reach→grasp→lift potential chain is the
  junior's own (reasonable) extension, not something the paper itself
  proposes — correctly presented as an application, not overclaimed.
- **arXiv:2407.09986** ("Curriculum Is More Influential Than Haptic
  Information...") — resolves, real, genuinely about curriculum vs.
  haptic-feedback tradeoffs in a multi-fingered-hand ball-lift task (the
  closest topical match in the whole report). The specific "freeze reach,
  then enable grasp, then enable lift" procedural recipe attributed to it
  is **more specific than verifiable** in the retrievable text.
  **Verdict: Overstated** (real, on-topic, but over-specified).

**Conclusion: potential-based reward shaping is the one genuinely solid,
correctly-characterized citation in the entire report.**

## Question 3: PPO stuck in static behavior (entropy collapse)

- **arXiv:2603.06793** ("Optimistic Policy Regularization") — resolves,
  real (Mar 2026 submission — initially mis-flagged as suspicious by a
  fetch that didn't know the actual current date, confirmed real on
  follow-up). Mechanism (buffer of successful trajectories +
  reward-shaping + BC) roughly matches the paraphrase; domain (Atari +
  cyber-defense, not manipulation) and the "backtrack from static grasp"
  framing are the junior's own extrapolation. **Verdict: Overstated but
  not fabricated.**
- **arXiv:2010.07986** ("Empowerment-based Solution to Robotic
  Manipulation Tasks with Sparse Rewards") — resolves, real, genuinely
  about empowerment/curiosity intrinsic motivation for sparse-reward
  manipulation. **Verdict: Verbatim-confirmed**, best-matched citation
  for this question.
- **arXiv:2101.04882** (OpenAI asymmetric self-play) — resolves, real;
  core mechanism accurately characterized, specific "0.01m/0.02m
  milestone" framing is invented illustrative detail. **Verdict:
  Overstated on specifics, core characterization accurate.**

## Question 4: grip-force safety factor

- **arXiv:1607.06620** ("Multi-Fingered Robotic Grasping: A Primer") —
  resolves, real. **The specific "2-3x safety factor" claim — the
  numerical linchpin of the report's "130-200x" headline conclusion —
  could not be found anywhere in the paper's abstract or extractable PDF
  text, despite multiple targeted searches. Verdict: Fabricated/
  Misapplied.**
- **arXiv:2103.06252** ("Grasp Stability Analysis with Passive
  Reactions") — resolves, real, but is a theoretical/numerical-methods
  paper unrelated to the claimed "light objects, forgiving friction
  cones" point. **Verdict: Misapplied.**
- The formula `F_min = m(g+a_lift)/(2μ)` is correct, derivable statics
  (not literally sourced from either citation) — should be treated as the
  junior's own derivation, not a literature citation.
- The "130-200x" comparison is arithmetically self-consistent but rests
  on an unverified safety-factor baseline.

**Conclusion: the grip-force-is-not-the-bottleneck conclusion is directionally
plausible (also independently supported by this repo's own real measured
contact force, ~20-30N against a 0.098N object weight, from the
ContactSensor experiment's calibration script) but not literature-backed
the way this report claims.**

## Overall assessment

Of the report's 3 priority recommendations:

1. **Multiplicative gating (contact-gates-lift):** not literature-
   supported — the citations don't show this specific pattern. Reasonable
   engineering idea on its own merits, but should be evaluated as an
   untested hypothesis, not a validated fix.
2. **Curriculum-based learning-rate scaling:** partially supported — real,
   topically-close paper (arXiv:2407.09986), but attributed procedural
   specifics go beyond what's verifiable.
3. **Potential-based reward shaping:** the best-supported recommendation —
   Ng/Harada/Russell 1999 is real, correctly characterized, and offers a
   genuine theoretical guarantee (policy invariance) for combining
   reach/grasp/lift sub-goals without the ad-hoc curriculum-timing
   problems this session has now hit twice.

**Recommendation:** treat this report's engineering instincts (try gating,
try potential-based shaping, don't blame gripper force) as reasonable
hypotheses to test empirically — not as literature-validated conclusions.
Roughly half its citations are real-but-off-topic or carry unverifiable
specific numbers. Do not cite arXiv:2509.14816, arXiv:2403.00282, or the
"2-3x safety factor" claim as literature evidence in any future record;
the Ng/Harada/Russell 1999 potential-based-shaping citation can be trusted
at face value.
