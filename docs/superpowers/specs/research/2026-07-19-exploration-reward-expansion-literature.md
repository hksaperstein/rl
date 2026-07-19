# Research: expanding PPO exploration mechanisms and rewards for gripper-closure discoverability

**Date:** 2026-07-19
**Author:** Senior research thread (delegated by Principal)
**Purpose:** Tier 1 hypothesis-gate research, triggered by a direct user
instruction after watching the clutter-detection experiment's Stage SO eval
video (`target-selection-clutter/eval-artifacts/joint-die-target-selection-so/`,
0/8 both shapes — the arm reaches down and hovers directly over the die but
never closes the gripper): *"if u are failing to explore and find the cube,
work on expanding your exploration metrics and rewards."* This document
surveys what is actually available beyond bare `entropy_coef`/
`init_noise_std` tuning, since this project's own prior research
(`docs/superpowers/specs/research/2026-07-19-d8-d10-grasp-discoverability-literature.md`,
hereafter "the d8/d10 doc") already ranked bare exploration-coefficient
tuning as the *weakest* lever for a related problem — that ranking is
re-examined fresh here rather than assumed to settle this broader question.
**This is research only — no env cfg code, no reward-term design, no Isaac
Sim launches.**

---

## 0. Relationship to the in-flight Stage SO warm-start fix — not a duplicate

The controller has a separate fix already in flight for Stage SO
specifically: `BACKLOG.md`'s "Clutter experiment Stage SO gate" entry and
commits `ee2ee7d`/`a589a49` (a first-layer weight-surgery warm start from
the already-proven `model_2998.pt` checkpoint, provably lossless at Stage SO
since the two new observation dims — `distractor_distance_summary` — are
hard-zeroed there). That fix targets a specific confound: Stage SO's 0/8
result cannot currently distinguish "the new scene/observation wiring broke
something" from "plain from-scratch PPO cold-start on this die population is
itself hard" (Stage SO is, per `2054380`'s own commit message, the first
from-scratch, non-warm-started PPO run of the d12/d20-mixed population in
this project's history — every prior success went through a BC/DAgger
distillation bootstrap). The warm start isolates the first question by
sidestepping the second.

**This document is deliberately orthogonal, not a second attempt at the same
fix.** Everything surveyed below — intrinsic-motivation bonuses, entropy/
noise scheduling, exploration-oriented reward shaping — is a change to the
PPO *exploration mechanism itself*, applicable regardless of whether a given
run is warm-started or cold-started, and regardless of whether the confound
Stage SO hit is present. It is relevant to: (a) Stage SO's own later stages
(D1/D2, which reintroduce live distractors and could plausibly re-open a
discovery problem even after the warm-start fix resolves Stage SO itself),
(b) any future from-scratch PPO run this project runs without a
distillation bootstrap available, and (c) this project's standing,
repeatedly-observed "reach solved, grasp/closure never discovered" pattern
generally (`kb/wiki/concepts/reach-grasp-lift-gap.md` — recurs from the
unnumbered AR4-era sphere precursors through Experiment 26). Nothing here
proposes touching Stage SO's own training run or the warm-start script.

---

## 1. Re-examining the d8/d10 doc's "weakest lever" ranking

The d8/d10 doc ranked exploration-noise/entropy tuning last among three
candidates (behind demonstration-augmented warm-start and geometry-ordered
checkpoint warm-start) for the d8/d10 shape-discoverability problem, with
this reasoning (quoted from that document, §3c): *"`entropy_coef=0.006`/
`init_noise_std=1.0` is the identical setting used for every shape in this
arc, including d12/d20, which did discover grasps at the same 48mm-parity
anchor. The same global exploration budget that was sufficient for d12/d20
not producing discovery for d8/d10 is corroborating... evidence that this is
a geometry-specific affordance problem rather than a generic PPO-exploration-
budget shortfall."*

**That reasoning is sound for the comparison it was making, and does not
transfer to this document's question, for two independent reasons:**

1. **It is a same-recipe-different-shape comparison, not a same-recipe-
   different-task-structure comparison.** The d8/d10 doc's logic isolates
   *shape* as the one variable that changed while everything else (PPO
   config, reward, scene, observation dimensionality) was held fixed. Stage
   SO's problem is structurally different: it is the first from-scratch PPO
   run of this population *at all*, on a wider observation schema (43-dim
   vs. the 41-dim baseline), in a scene with (inert, but present)
   distractor-object infrastructure. There is no existing "worked at these
   settings" control run in this specific regime the way d12/d20 was a
   control for d8/d10 — the argument "the same budget worked elsewhere" has
   a weaker analogous case to point to here.
2. **The d8/d10 doc only ever evaluated a *fixed* coefficient tried at one
   value.** It did not consider — and its own reasoning does not address —
   *scheduled* entropy/noise (§3 below), which is a qualitatively different
   mechanism (state-of-training-adaptive rather than pre-committed), nor
   intrinsic-motivation bonuses (§2 below), which are not an
   "exploration-budget" lever at all but a structurally different reward
   signal. "The fixed value already used elsewhere didn't help a harder
   case" is not evidence against a scheduled or intrinsic-motivation
   mechanism the prior document never examined.

**Conclusion: the d8/d10 doc's ranking stands, correctly, for its own
narrow question (bare fixed-coefficient tuning, isolating shape as the
variable) and should not be read as having evaluated — let alone ruled out
— the mechanisms this document covers.**

---

## 2. Intrinsic/curiosity-driven exploration bonuses

Every citation in this section was independently verified live this session
via the arXiv API (`export.arxiv.org/api/query?id_list=...`, `https`, not
the redirecting plain-`http` endpoint) — title match confirmed for each ID
listed, not taken from a sub-researcher's report on faith, per this
project's own `kb/wiki/concepts/citation-verification-practice.md`.

### 2a. The two foundational mechanisms, and their PPO-compatibility

- **Burda, Edwards, Storkey, Klimov, "Exploration by Random Network
  Distillation," ICLR 2019, arXiv:1810.12894.** Verified (title match).
  Intrinsic reward = prediction error between a fixed random target network
  and a trained predictor network on observations. **The paper's own
  reference implementation is built directly on PPO** — the single closest
  algorithmic match to this project's `rsl_rl` PPO setup among the
  candidates surveyed. First superhuman Montezuma's Revenge result without
  demonstrations — an Atari/navigation result, not manipulation (see §2b).
- **Pathak, Agrawal, Efros, Darrell, "Curiosity-driven Exploration by
  Self-supervised Prediction," ICML 2017, arXiv:1705.05363.** Verified
  (title match; the paper's own arXiv comment field states "In ICML 2017").
  Curiosity = forward-model prediction error in a learned inverse-dynamics
  feature space (explicitly designed for partial resistance to unpredictable/
  uncontrollable "noisy-TV" distractors). **The original paper uses A3C, not
  PPO** — a real algorithmic gap; PPO+ICM combinations exist in practice
  (community implementations) but are not the paper's own validated
  configuration.
- **Count-based bonuses:** Bellemare et al., "Unifying Count-Based
  Exploration and Intrinsic Motivation," NeurIPS 2016, arXiv:1606.01868
  (verified); Tang et al., "#Exploration: A Study of Count-Based
  Exploration for Deep Reinforcement Learning," NeurIPS 2017,
  arXiv:1611.04717 (verified). Both are Atari/general-MuJoCo, state-novelty
  bonuses with no bias toward a specific rare action — the weaker-fit
  candidates of this subsection for a problem defined by one specific rare
  *action* (gripper closure) rather than general state coverage.

### 2b. Manipulation-specific evidence — the decisive check this project's
own framing asks for

This is the load-bearing question: do these mechanisms actually help a
manipulation/grasping task with a sparse-discovery problem, not just Atari?

- **Dai, Xu, Hofmann, Williams, "An Empowerment-based Solution to Robotic
  Manipulation Tasks with Sparse Rewards," RSS 2021, arXiv:2010.07986.**
  Verified (title match; arXiv comment confirms RSS 2021 acceptance). Uses
  **PPO**, on box/sphere/cylinder lifting and pick-and-place tasks —
  directly grasping-adjacent. Runs **ICM alone as an explicit baseline**
  and reports their empowerment+curiosity hybrid outperforms it. **This is
  a mixed/negative result for ICM in isolation on a lifting task**, not a
  clean success story — the exact caveat this project's citation-discipline
  practice requires surfacing rather than only citing the positive framing.
  Exact numeric gap not independently re-extracted here (PDF/text
  extraction was reported as unreliable by the sub-researcher); flag as
  directionally, not quantitatively, confirmed.
- **Vulin, Christen, Stevsic, Hilliges, "Improved Learning of Robot
  Manipulation Tasks via Tactile Intrinsic Motivation," IEEE RA-L 2021,
  DOI 10.1109/LRA.2021.3061308 (arXiv:2102.11051).** Verified independently
  via both CrossRef (`api.crossref.org/works/10.1109/LRA.2021.3061308`,
  title match, IEEE RA-L, April 2021) and arXiv (title match). Uses a
  **tactile/contact-force-based** intrinsic signal (not RND/ICM's
  prediction-error mechanism) targeted directly at sparse-reward
  manipulation exploration, reporting outperformance of prior SOTA on
  manipulation benchmarks — a real, verified alternative intrinsic-reward
  *design*, more directly aimed at contact/grasp events specifically than
  generic curiosity.
- **Liu et al., "ContactExplorer: Contact Coverage-Guided Exploration for
  General-Purpose Dexterous Manipulation," arXiv:2603.10971 (2026).**
  Verified (title match). A count-based intrinsic reward on **contact-event
  coverage** specifically — structurally the closest match found to this
  project's exact failure signature (a rare *contact* event, gripper
  closure near/on the object, essentially never sampled). Not RND/ICM
  (different mechanism family: contact-coverage counting, not prediction
  error), but the most directly on-point framing among everything surveyed.
- **Han, Peng, Liu, Tang, Yu, Zhou, "Learning robotic manipulation skills
  with multiple semantic goals by conservative curiosity-motivated
  exploration," Frontiers in Neurorobotics, 2023 (PMC10028088).** Verified
  independently via NCBI's own e-utilities API
  (`eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pmc&id=10028088`,
  confirms source "Front Neurorobot," pubdate 2023, author list matches). A
  genuine caution flag, not just supporting evidence: naive curiosity-driven
  exploration can push a manipulation policy into "dangerous"/unproductive
  states, motivating a "conservative" (mutual-information-gated) variant —
  worth weighing against an unconditional recommendation to just add
  curiosity.
- **Rajeswar et al., "Touch-based Curiosity for Sparse-Reward Tasks," CoRL
  2021, arXiv:2104.00442.** Verified (title match). Cross-modal vision/touch
  curiosity for pushing/door-opening — grasping-adjacent, not grasping
  itself; included for completeness, not a strong direct precedent.

### 2c. A distinct, mechanistically important finding: exploration noise may
never reach the gripper at all

**Neunert, Abdolmaleki, Wulfmeier, Lampe, Springenberg, Hafner, Romano,
Buchli, Heess, Riedmiller (DeepMind), "Continuous-Discrete Reinforcement
Learning for Hybrid Control in Robotics," CoRL 2019, arXiv:2001.00449.**
Verified (title match). This is not an intrinsic-motivation paper, but it
surfaced a mechanistic account worth flagging prominently because it bears
directly on *why* a bare noise/entropy increase (or even an intrinsic bonus
layered on top of unchanged action dynamics) might fail regardless of which
exploration mechanism is chosen. Quoted directly from the paper's own
motivation: **"The gripper fingers are slow and thus they effectively act as
a low pass filter that filters out most of the policy's (zero mean
initialized) Gaussian exploration."** Their fix is an action-space
redesign (discretizing gripper velocity to {−1, 1}, or a repeated/meta-
action formulation), not an exploration-mechanism change at all.

**This is flagged, not adopted, for two reasons.** First, it is a real,
citable, independently-plausible alternative explanation for this project's
own recurring "reach solved, gripper closure never discovered" pattern that
no exploration-mechanism change (bonus, schedule, or reward shaping) can
fix if it is the operative cause here — worth checking directly against
this project's own gripper action formulation
(`tasks/franka/lift_env_cfg.py`'s `panda_finger_.*` joint-position action
scale) before or alongside any exploration-mechanism experiment. Second,
per this repo's own Tier 1 workflow gate, an action-space change is exactly
the kind of cross-cutting/architectural decision this document is not
authorized to propose or design — flagged for whoever scopes the next spec,
not resolved here.

---

## 3. Exploration-noise/entropy scheduling (not a fixed value)

### 3a. PPO-native scheduling literature — real, peer-reviewed, general
continuous control

- **Li, Li, Li, "Adaptive Exploration Proximal Policy Optimization for
  Efficient Robotic Continuous Control" (AE-PPO), *Symmetry* 18(5):717,
  MDPI, 2026, DOI 10.3390/sym18050717.** Verified via CrossRef
  (`api.crossref.org/works/10.3390/sym18050717`, title match, published
  2026-04-24). Schedules the entropy *weight* via a target-entropy-style
  feedback rule — raising it when measured policy entropy drops below a
  threshold, lowering it when above — combined with adaptive clip-range
  adjustment. Reports faster convergence and lower variance on
  "challenging high-dimensional" continuous-control tasks. **Caveat: could
  not confirm manipulation/grasping is among its benchmarks** (likely
  locomotion/generic control) — cite as a mechanism-level precedent for
  "PPO entropy scheduling beats a fixed value," not as manipulation-
  validated.
- **"Entropy adjustment by interpolation for exploration in Proximal Policy
  Optimization (PPO)," *Engineering Applications of Artificial
  Intelligence*, Elsevier, 2024, DOI 10.1016/j.engappai.2024.108401.**
  Verified via CrossRef (title match, published 2024-07). Replaces PPO's
  fixed entropy coefficient with an interpolation-derived schedule,
  explicitly framed as reducing exploration over training rather than
  holding it fixed. Same caveat as above: general continuous control, not
  manipulation-specific.
- **PPO-BR, arXiv:2505.17714 (2025), single-author preprint.** Verified
  real via arXiv (title match: "Dual-Signal Entropy-Reward Adaptation for
  Trust Region Policy Optimization"), but flagged as **low-confidence
  evidence**: single-author, reported by the sub-researcher as still under
  review at IEEE TNNLS per a TechRxiv listing, not yet peer-reviewed.
  Mechanism (entropy-driven trust-region clip-range adjustment, not the
  entropy coefficient itself) is adjacent, not identical, to the ask. Do
  not treat as validated — included only because it is real and
  transparently caveated, not omitted silently.

### 3b. SAC-style automatic temperature tuning as the principled mechanism
behind scheduling

Both foundational SAC papers verified via arXiv: **Haarnoja, Zhou, Abbeel,
Levine, "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a
Stochastic Actor," arXiv:1801.01290 (ICML 2018)**, and **Haarnoja, Zhou,
Hartikainen, et al., "Soft Actor-Critic Algorithms and Applications,"
arXiv:1812.05905 (2018)** — the latter introduces automatic temperature
tuning: the entropy temperature α is learned via gradient descent against a
dual objective that penalizes deviation of realized policy entropy from a
**target entropy** (conventionally `-dim(action_space)`), rather than being
hand-fixed or hand-annealed. This is the principled "why" behind AE-PPO's
own threshold-feedback rule (§3a) — a real idea that has been *adapted*
toward PPO in the literature above via simpler feedback rules, but **no
source found performs a literal transplant of SAC's dual-gradient-descent
α-update into PPO** — flag this as an adaptation, not a direct port, if
cited in a future spec.

### 3c. The specific sub-question — narrow rare-behavior discovery via
scheduling, in manipulation, under PPO — is a genuine literature gap

No source was found that is simultaneously (a) PPO, (b) manipulation/
grasping, and (c) an entropy/noise *schedule* explicitly motivated by
narrow rare-behavior discovery (as opposed to general convergence-speed or
variance reduction). This is stated plainly as a gap, not stretched to fit:

- Hollenstein, Auddy, Saveriano, Renaudo, Piater, "Action Noise in
  Off-Policy Deep Reinforcement Learning," arXiv:2206.03787 — verified,
  supports "decreasing noise schedule beats fixed" generally, but
  off-policy algorithms (DDPG/TD3/SAC) only, no PPO, no manipulation task.
- Plappert et al., "Parameter Space Noise for Exploration," arXiv:1706.01905
  (ICLR 2018) — verified, a genuinely different mechanism (parameter-space
  rather than action-space noise) built explicitly for sparse-reward tasks,
  but tested on DQN/DDPG/TRPO, not PPO.

**Honest synthesis: the strongest defensible claim combines three separate,
individually-real pieces of evidence rather than one direct citation** — (a)
scheduled entropy measurably beats fixed entropy in PPO on general
continuous control (§3a), (b) SAC's target-entropy framing is the
principled reason a *schedule* (rather than a fixed value) should help a
narrow-behavior problem specifically, since it keeps stochasticity high
until the target entropy is actually reached rather than decaying on a
fixed timetable regardless of whether the rare behavior has been found yet
(§3b), and (c) sparse contact-event discovery is a real, separately
recognized problem class in manipulation RL (ContactExplorer, §2b) — while
being explicit that the specific combination (all three at once) is not
directly literature-validated and would be this project's own contribution
if pursued.

### 3d. `desired_kl`/adaptive-LR interaction — quick check, not a citation

This project's PPO config (`tasks/franka/agents/rsl_rl_ppo_cfg.py:41-53`)
already uses `schedule="adaptive"` + `desired_kl=0.01`, an adaptive-*
learning-rate* schedule keyed to measured KL divergence, tracing to
Schulman, Wolski, Dhariwal, Radford, Klimov, "Proximal Policy Optimization
Algorithms," arXiv:1707.06347 (verified) — its adaptive-KL-penalty variant.
This controls **step size** (how far each update moves the policy), not
**stochasticity** (how random rollouts are) — an orthogonal axis to
entropy/noise scheduling. No literature found studying their interaction
directly; this project's own assessment (not a citation) is that the two
are not redundant and could plausibly be synergistic (higher early entropy
→ more diverse rollouts → the adaptive-KL mechanism paces the resulting
updates safely) — stated as inference, not a verified finding, per this
project's citation-discipline practice.

---

## 4. Reward-shaping for exploration/discoverability, distinct from reward hacking

This project already has a load-bearing internal framework for this exact
distinction — `kb/wiki/concepts/reward-hacking-and-sparse-discoverability.md`
— and direct prior experience applying Ng, Harada & Russell's potential-
based reward shaping (ICML 1999) to a staged reach→grasp→lift reward
(`kb/wiki/experiments/experiment-05-potential-based-reward-shaping.md`),
where a real formula bug was found: a running-max potential's "always ≥ 0"
claim is false whenever the agent merely *holds* its best-ever potential
without improving further, under any discount `gamma < 1` — this produced a
*negative* reward for holding position and likely explains that
experiment's own total non-approach failure. This section does not
re-derive Ng/Harada/Russell itself (already covered by this project's own
prior work) — it looks for what's available *beyond* it.

### 4a. Extensions directly relevant to this project's own known bug class

- **Forbes, Villalobos-Arias, Wang, Jhala, Roberts, "Potential-Based
  Intrinsic Motivation: Preserving Optimality With Complex, Non-Markovian
  Shaping Rewards" (PBIM/GRM), arXiv:2410.12197.** Verified (title match).
  Classical PBRS assumes a Markovian potential Φ(s); this paper extends the
  policy-invariance guarantee to **history-dependent (non-Markovian)**
  shaping functions. **This is the single most directly applicable finding
  in this section**: a running-max potential is, by construction,
  non-Markovian (it depends on the episode's history of visited states, not
  just the current one) — this project's own Experiment 5 bug is a concrete
  instance of exactly the failure mode this paper's framework is built to
  handle correctly instead of accidentally violating "always ≥ 0." Worth a
  direct read before any future staged/running-max shaping design, not just
  a citation.
- **Forbes, Wang, Villalobos-Arias, Jhala, Roberts, "Action-Dependent
  Optimality-Preserving Reward Shaping" (ADOPS), arXiv:2505.12611.**
  Verified (title match; AAMAS 2025 + ICML 2025 per arXiv metadata).
  Standard PBRS requires the shaping term's cumulative discounted return to
  be *action-independent* to preserve the optimal policy — ADOPS removes
  that restriction, allowing a shaping term to depend on the action taken,
  correcting on-the-fly only when doing so would flip which action is
  preferred versus extrinsic reward alone. **Directly relevant to the
  user's own proposed pattern** ("a small proximity-gated exploration bonus
  for attempting gripper closure near the object") — an attempt bonus is
  inherently a function of the action taken (closing the gripper), which
  vanilla state-only PBRS structurally cannot express without either
  losing the policy-invariance guarantee or requiring an awkward
  state-augmentation workaround. Caveat: relies on learned separate
  extrinsic/intrinsic critic estimates and a policy-stability assumption —
  a design pattern to study, not necessarily a drop-in mechanism for
  `rsl_rl`'s stock PPO implementation without added infrastructure.
- **Okudo, Yamada, "Subgoal-based Reward Shaping to Improve Efficiency in
  Reinforcement Learning," arXiv:2104.06411.** Verified (title match).
  Uses human-specified representative subgoal *states* directly as
  potential landmarks rather than a hand-designed potential function —
  addresses the general "potentials are unintuitive to design for staged
  tasks" problem. Does not appear to address this project's own specific
  "later stage requires moving away from an earlier stage's peak"
  co-satisfiability pitfall
  (`kb/wiki/concepts/staged-reward-co-satisfiability.md`) — that finding
  remains, as far as this search determined, this project's own
  contribution rather than one already covered in the literature.

### 4b. Rewarding the attempt, not the success — precedent and a genuine alternative-in-kind

- **Andrychowicz et al., "Hindsight Experience Replay" (HER),
  arXiv:1707.01495 (2017).** Verified (title match). Not a shaping bonus at
  all — sidesteps the fakeable-proxy problem entirely by relabeling failed
  trajectories against alternative achieved goals, using only the original
  binary sparse-success signal. A genuine alternative-in-kind worth flagging
  even though it is not what the user asked for specifically: **caveat, not
  glossed over — HER's standard formulation is built for off-policy,
  goal-conditioned algorithms (originally DQN/DDPG), and porting it to an
  on-policy PPO/`rsl_rl` setup is a materially bigger structural change than
  anything else in this document** (it requires a replay buffer and
  goal-relabeling machinery PPO's on-policy rollout structure doesn't
  natively have) — flagged for completeness, not recommended as this
  document's primary candidate.
- **Rajeswaran et al., "Learning Complex Dexterous Manipulation with Deep
  RL and Demonstrations" (DAPG), arXiv:1709.10087 (RSS 2018).** Verified
  (title match) — already this project's own H1 candidate for d8/d10 (the
  d8/d10 doc's §3a/§4). Relevant here as the reward-shaping-adjacent
  instance of "structurally hard to fake": a demonstration-cloning auxiliary
  term can't be satisfied by a gripper-closes-beside-not-around-the-object
  fake grasp the way this project's own AR4-era `grasp_sphere` dense bonus
  was (`kb/wiki/concepts/reward-hacking-and-sparse-discoverability.md`),
  because a fake attempt simply won't match the demonstrated action
  sequence. Not re-derived in full here since it's already covered
  elsewhere in this project's own research — cited only to connect it to
  this document's "structurally hard to hack" framing.
- **No paper was found that specifically formalizes a count-based bonus
  biased toward one particular action** (e.g., "count gripper-closure
  attempts near the object specifically," rather than general state
  novelty). The general state-action count formula from Tang et al.
  (§2a, `r+(s,a) = beta/sqrt(n(s,a))`) generalizes cleanly to this by
  substitution, but this is this document's own extrapolation from general
  theory, not a directly-precedented technique — flagged as such rather
  than presented as literature-validated.

### 4c. Multiplicative/AND-gated design — a quick check, not the main focus

This project already has a working example of this pattern
(`grasp_contact`/`antipodal_grasp_bonus`, requiring both proximity AND
force-direction alignment, per
`kb/wiki/concepts/grasp-mechanics-antipodal-vs-magnitude.md`) and a
verbatim-confirmed literature citation for a related multiplicative pattern
(GRIT, arXiv:2604.04138 — re-verified live this session via the arXiv API,
title confirmed: "Learning Dexterous Grasping from Sparse Taxonomy
Guidance"; the unusual-looking `2604` prefix is simply the standard
YYMM arXiv numbering for April 2026, not a fabrication artifact). One
additional, genuinely new finding worth adding to this project's own
framing: **Skalse, Howe, Krasheninnikov, Krueger, "Defining and
Characterizing Reward Hacking," NeurIPS 2022, arXiv:2209.13085.** Verified
(title match). Formally proves that "unhackability" of a proxy reward is a
very strong condition — for stochastic policies, two reward functions can
only be mutually unhackable if one is constant. **This is a useful
correction to over-trusting AND-gating alone**: a multiplicative/
conjunctive gate (GRIT's own pattern, or this project's antipodal check)
raises the bar for hacking empirically but is not proven policy-invariant
the way genuine PBRS is — worth stating explicitly in any future spec that
proposes an AND-gated exploration bonus, so it isn't mistaken for carrying
Ng/Harada/Russell's formal guarantee.

---

## 5. Proposed hypothesis and methodology (for a future spec to cite — NOT a spec itself)

Three candidate interventions, ranked by strength of grounding for this
project's specific problem (a manipulation task where reach converges but
gripper closure is essentially never sampled), each with an explicit
falsification condition. Consistent with this project's Tier 1 workflow,
none of this is proposed for immediate implementation — it is the grounding
a future spec would cite.

**H1 (primary): a non-Markovian-aware, action-dependent potential-based
exploration bonus for gripper-closure attempts near the object**, built on
the PBIM/GRM (arXiv:2410.12197) and/or ADOPS (arXiv:2505.12611) framework
rather than an ad hoc running-max sum — directly answering the user's own
"rewards" framing while fixing, by construction, the exact bug class this
project already hit in Experiment 5 (a running-max potential silently
violating its own policy-invariance claim). Ranked first because it is
simultaneously the most direct response to the user's literal instruction
and the only candidate that comes with a genuine theoretical non-hackability
argument (subject to §4c's caveat that AND-gating alone is not equivalently
strong) rather than an empirical "seems to help" result from an unrelated
domain.

*Falsification:* if a correctly-implemented (verified via unit test against
the "always non-negative"/policy-invariance property this project's own
Experiment 5 review would now know to check) attempt-bonus, applied at
Stage SO or in a controlled from-scratch single-object rerun, still shows
0/8 sustained-lift across 3 seeds, that falsifies "a structurally
non-hackable attempt-reward is sufficient to unlock discovery" and would
point toward the intrinsic-motivation (H2) or actuator-filtering (§2c)
explanations instead.

**H2 (secondary): a contact-coverage-style count-based intrinsic bonus**,
following ContactExplorer's framing (arXiv:2603.10971) of counting rare
*contact events* rather than general state novelty, implemented via RND
(arXiv:1810.12894) given its native PPO compatibility — reusing an
established, PPO-validated architecture rather than porting ICM's
A3C-native design. Ranked second because manipulation-specific evidence for
curiosity-family bonuses is real but mixed (Dai et al. 2021 found plain ICM
insufficient alone on a lifting task, only outperforming when combined with
an empowerment term; Han et al. 2023 found naive curiosity can be actively
counterproductive in manipulation) — a real, literature-grounded lever, but
with documented failure modes this project should design around rather than
assume away.

*Falsification:* if an RND-style contact-coverage bonus, added on top of
the existing reward without other changes, still shows 0/8 across 3 seeds
and/or exhibits the "curiosity pushes into unproductive states" failure
mode Han et al. describe (visibly erratic, non-goal-directed exploration
increasing without any corresponding rise in grasp-attempt rate), that
falsifies "intrinsic motivation alone is sufficient" and corroborates
Han et al.'s own conservative/gated variant as the next escalation, not a
fourth from-scratch attempt at plain curiosity.

**H3 (tertiary, weakest fit among the three, but stronger than the d8/d10
doc's already-tried fixed-value tuning): target-entropy-style scheduled
exploration**, adapting AE-PPO's (DOI 10.3390/sym18050717) threshold-
feedback entropy scheduling, principled by SAC's automatic-temperature
mechanism (arXiv:1812.05905) — raising entropy/noise dynamically until a
target policy entropy is reached, rather than holding `entropy_coef=0.006`/
`init_noise_std=1.0` fixed for the entire run. Ranked third, not dismissed:
per §1, this is a materially different mechanism than what the d8/d10 doc
already found insufficient (a fixed value), and per §3c no manipulation-
specific narrow-behavior evidence exists either for or against it — it is
the least-precedented-for-this-exact-problem candidate, not a
disproven one.

*Falsification:* if scheduled entropy/noise targeting, evaluated the same
way, still shows 0/8 across 3 seeds, that would be a real, useful result
distinguishing "the global exploration budget, even adaptively managed, was
never the bottleneck" from H1/H2's more targeted mechanisms — closing off
the entire fixed-vs-scheduled entropy axis for this problem, not just one
value on it.

**A flagged, out-of-scope prerequisite check, not one of the three ranked
hypotheses:** before investing in any of H1-H3, verify directly whether
this project's Franka gripper action formulation exhibits the low-pass-
filtering failure mode Neunert et al. describe (§2c) — i.e., whether the
existing `panda_finger_.*` joint-position action scale and the gripper's
own actuator dynamics could structurally prevent a full-closure command
from ever being produced by Gaussian exploration noise, regardless of its
magnitude or schedule. If so, none of H1-H3 would be expected to help until
addressed — but per this project's own Tier 1 workflow, an action-space
change is a cross-cutting/architectural decision, not something this
research document is authorized to design; flagging it for whoever scopes
the next spec to check first (a cheap logging/histogram check of realized
gripper-action magnitudes across a short rollout, not a full experiment) is
this document's only recommendation on that axis.

---

## 6. Open risks / gaps, stated plainly

- **No source found tests any of RND/ICM/count-based/entropy-scheduling
  specifically on a rigid convex-polyhedron pick task with a parallel-jaw
  gripper and PPO** — the closest manipulation evidence (Dai et al. 2021,
  Vulin et al. 2021, ContactExplorer) spans lifting/pushing/dexterous-hand
  tasks, not this project's exact die/cube-and-parallel-jaw setting. Every
  hypothesis in §5 is grounded in adjacent, not identical, precedent.
- **The Neunert et al. low-pass-filter finding (§2c) is not independently
  re-verified against this project's own action-scale numbers in this
  document** — it is flagged as a real, plausible, checkable risk, not
  confirmed to actually apply here. Treat the "flagged prerequisite check"
  in §5 as genuinely unresolved, not a formality.
- **PBIM/GRM and ADOPS (§4a) are both 2024/2025 papers without, as far as
  this search determined, a public reference implementation this project
  could directly reuse** — unlike DAPG/RND, which have widely-used
  reference code. H1's implementation cost is real and not yet scoped; a
  future spec should treat "build this from the paper's formulas" as a
  genuine cost, not assume a drop-in library exists.
- **Han et al.'s "conservative curiosity" caution (§2b) was read from an
  esummary/metadata check plus the sub-researcher's direct fetch, not
  independently re-read by this document's author against the full paper
  text** — the specific claim (naive curiosity can push manipulation
  policies into unproductive/dangerous states) should be treated as
  reported-and-plausible, one tier below the citations this document
  independently re-fetched and title-matched directly via arXiv/CrossRef.
- **This document does not resolve whether H1/H2/H3 should be tried
  together or in the strict ranked sequence proposed** — a future spec
  should make that sequencing call explicitly (this project's own
  systematic-debugging convention favors one falsifiable change at a time,
  which argues for the sequence in §5, but that is a spec-time judgment
  call, not re-derived here).

---

## Addendum (2026-07-19): the flagged Neunert et al. low-pass-filter prerequisite check — resolved, does NOT apply here

Per §2c/§5's flagged, unresolved prerequisite ("verify directly whether this
project's Franka gripper action formulation exhibits the low-pass-filtering
failure mode Neunert et al. describe... before investing in any of H1-H3"),
a Senior thread checked this directly (diagnostic only, no training run, no
env cfg changes) and got a clear answer: **no, it does not apply.**

**1. Config-level finding.** This project's gripper action
(`tasks/franka/lift_env_cfg.py`'s `gripper_action = mdp.
BinaryJointPositionActionCfg(...)`, inherited unchanged by every joint-die
variant including the target-selection-clutter Stage SO env) is mapped by
Isaac Lab's own `BinaryJointAction.process_actions` (`isaaclab/envs/mdp/
actions/binary_joint_actions.py`, read directly from the installed package
source) via a **hard sign threshold**: `binary_mask = actions < 0`, then
`processed_actions = where(binary_mask, close_command, open_command)`. Any
raw policy action, however small in magnitude, produces the FULL
open/close joint-position target — there is no proportional/scaled mapping
from action magnitude to a partial command. This is structurally different
from the setup Neunert et al.'s finding describes: their quoted low-pass-
filter sentence is about their **baseline**, which controls the gripper in
**continuous velocity mode** — small-magnitude, zero-mean-initialized
Gaussian noise there produces small velocity commands that slow finger
actuator dynamics can attenuate before enough motion accumulates to reach a
meaningfully closed position. Neunert et al.'s own *fix* for this is to
discretize the gripper action to `{-1, 1}` (full speed open/close) —
i.e., a binary/threshold gripper action, which is structurally what this
project already has, not what its failure-mode baseline used. This point
alone was enough to make the low-pass-filter mechanism implausible on
priors, before any rollout.

**2. Direct rollout evidence, on the actual failing checkpoint.** Per this
project's verification standard (real evidence over config-reading alone),
a short instrumented rollout was run against the exact Stage SO checkpoint
this document's opening cited (`gs://rl-manipulation-hks-runs/target-
selection-clutter/joint-die-target-selection-so/seed42/2026-07-19_21-25-52/
model_1499.pt`, d20-pinned-target eval variant, 8 envs, one full 250-step
episode, deterministic inference policy, new diagnostic script
`scripts/_diag_gripper_lowpass_check.py`), logging the raw gripper action,
the processed joint-position command, and the REALIZED finger joint
position every step. Result: **the raw gripper action was positive
(commanding "open") for 100% of steps, in all 8 envs, with no exceptions**
— `frac_steps_raw_action_negative = 0.000` for every env.  Critically, the
raw action values were not small/borderline (which might still suggest a
noise-magnitude story): they ranged from +0.48 to +7.77 across envs/steps —
a confidently, strongly positive "stay open" signal, not a near-zero value
teetering on the sign threshold. The policy never once, in any of the 8
sampled episodes, produced a negative (attempt-close) raw action.

**Conclusion: this is a reward/exploration-discovery problem, not an
actuator/action-space attenuation problem.** The converged Stage SO policy
has confidently learned to keep the gripper open throughout the episode —
consistent with the video-review finding ("the gripper reaches down and
hovers directly over the die but never closes around it") but for a
different underlying reason than Neunert et al.'s mechanism would predict.
Nothing here is being filtered out by slow actuator dynamics; the policy's
own output signal is unambiguous and never attempts closure. This
corroborates, rather than undercuts, this document's own H1/H2/H3
research track (exploration-mechanism/reward changes remain the right
class of intervention) and closes the flagged prerequisite from §5 as
resolved-negative — no action-space/actuator-config change is indicated
before pursuing H1-H3.

**Caveats, stated plainly:** (a) this rollout used the deterministic
inference policy (`runner.get_inference_policy`, the standard rsl_rl mean-
action path, no sampling noise), so it directly characterizes the
*converged* Stage SO policy's behavior, not the stochastic action
distribution actually sampled during early training — but the config-level
structural argument (§1 above: a hard sign-threshold action mapping cannot
attenuate small-magnitude noise regardless of when in training one looks)
already rules out the mechanism independent of this timing distinction. (b)
One incidental, minor, unexplained observation not central to this check:
env 0 showed `frac_steps_joint_actually_20pct_closed=0.372` despite never
being commanded closed — plausibly the fingers physically contacting the
die while trying to reach the commanded-open target (a passive contact
effect, not a commanded closure); not investigated further as it doesn't
bear on the low-pass-filter question. (c) Only Stage SO's d20-pinned-target
eval variant was checked (not d12, and not D1/D2); given the mechanism
being tested is about the *action-space plumbing* (shared, unchanged,
across every joint-die variant in this codebase) rather than anything
d12/d20/distractor-count-specific, this is not expected to change the
answer for those, but was not independently re-verified.

Full raw per-step arrays and JSON summary (not committed, `logs/` is
gitignored): `logs/diag_gripper_lowpass/summary_d20.json` and
`raw_arrays_d20.npz`, produced by
`scripts/_diag_gripper_lowpass_check.py --checkpoint <Stage SO
model_1499.pt> --eval_target_shape d20 --num_envs 8 --num_steps 250` (run
on the desktop GPU, isolated `git worktree` at commit `7ed07a2` to avoid
touching the concurrent d8/d10 demo-warm-start workstream's dirty checkout
on that same machine).

## Related

[[reward-hacking-and-sparse-discoverability]] (the general framework this
document's §4 specializes toward exploration-boosting shaping specifically),
[[reach-grasp-lift-gap]] (the standing "reach solved, grasp never
discovered" pattern this document's problem is the latest instance of),
[[staged-reward-co-satisfiability]] and
[[experiment-05-potential-based-reward-shaping]] (source of the running-max
policy-invariance bug this document's §4a directly addresses),
[[grasp-mechanics-antipodal-vs-magnitude]] (source of this project's own
AND-gated grasp-check precedent, §4c),
`docs/superpowers/specs/research/2026-07-19-d8-d10-grasp-discoverability-literature.md`
(the prior document whose exploration-tuning ranking this document
re-examines in §1, and source of the DAPG/demonstration-augmentation
citation reused in §4b), `kb/wiki/concepts/citation-verification-practice.md`
(the standing practice this document's citation checks follow — every
citation above was independently re-verified live via the arXiv API or
CrossRef by this document's own author, not taken from a sub-researcher's
report on faith).
