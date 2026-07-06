# Senior Review: Citation Verification of Grasp-Alignment Literature Report

**Reviewer:** senior review pass (independent citation verification)
**Date:** 2026-07-05
**Subject:** `2026-07-05-grasp-alignment-literature-junior.md`
**Method:** Every arXiv ID was fetched directly from the arXiv API
(`export.arxiv.org/api/query`, ground truth, not LLM-summarized) to confirm
existence/title/date. Full text was then checked via ar5iv HTML (cross-checked
with raw `curl | grep` against the underlying HTML source, not just an LLM's
paraphrase of it, wherever a claim was numerically specific) to confirm the
junior's specific formulas/numbers/claims are actually present in each paper.

## Headline finding

**None of the 10 arXiv IDs are fabricated.** All resolve to real papers with
titles matching the junior's citations, including the two IDs that looked
suspicious at first glance (`2604.xxxxx`, `2606.xxxxx` — these imply
April/June 2026, which is in the past relative to this session's simulated
current date of 2026-07-05, and both are confirmed real preprints via the
arXiv API with matching publication dates). So the "fabricated near-future
ID" failure mode did **not** occur here — good sign for the junior's basic
citation hygiene.

However, on the substance: **two of the four core numbered claims in the
prompt are overstated or fabricated**, most importantly the specific
`std=0.02m` recommendation, which has **no basis in any cited paper**. One
citation (AsymDex) is applied to a problem it doesn't actually address
(single-gripper fingertip alignment vs. bimanual hand coordination). One
citation (Factory) is credited with a mechanism (bilateral contact-force
reward gating) that is **not in the paper at all**. Details below.

Note on tooling: WebFetch's underlying summarizer model gave **contradictory
answers about the same paper** across different calls (e.g. calling
`2606.21148` "fictitious" in one pass while another pass extracted detailed,
verifiably-correct content from it). Do not trust a single WebFetch summary
verdict about a paper's authenticity — always cross-check against the raw
arXiv API and, for numeric claims, raw HTML/grep.

---

## Per-citation verdicts

### [2604.04138] "Learning Dexterous Grasping from Sparse Taxonomy Guidance" (GRIT)
**Verdict: CONFIRMED REAL AND ACCURATE.**
- Real paper (arXiv API confirms, published 2026-04-05, revised 2026-06-30).
  Title and topic (taxonomy-conditioned dexterous grasping policy, GRIT
  framework, 87.9% success rate) match.
- The specific multiplicative reward formula the junior quotes is **verbatim
  correct**. Confirmed directly from the paper's HTML (Eq. 3):
  `r = r_h·α_h + r_o·α_o − r_pen`, with the paper's own text: "α_h and
  α_o ∈ [0,1] are multiplicative constraint coefficients that attenuate
  rewards under undesirable behaviors." This is a genuine, well-verified,
  strong citation for the multiplicative-gating recommendation.

### [2606.21148] "Pose-Agnostic Robotic Functional Grasping via Observation-Action Canonicalization" (AnyMug)
**Verdict: REAL BUT OVERSTATED / mischaracterized on the specific closure-gating claim.**
- Real paper (arXiv API confirms, published 2026-06-19). Topic
  (object-centric/canonicalized observation-action frame for mug-handle
  grasping, 93%/80% success) matches.
- The junior's Section 1 claim — "closure reward explicitly gated on
  prerequisite orientation alignment being satisfied" / "gripper approaches
  with correct orientation first, then closes only after fingers are
  positioned on opposite sides" — is **not accurate**. The paper's actual
  reward (its Eq. 4.2) is **additive**, not a staged/gated closure:
  `r_t = w_p·r_pos + w_R·r_rot + w_o·r_opp − w_a·r_act`. The only internal
  gating is inside the finger-opposition sub-term, which is gated by
  fingertip *proximity* to the handle, not by orientation alignment as
  claimed. Moreover the paper explicitly states gripper closure is a
  **continuous policy output that is not staged/hard-gated at all**: "This
  term encourages a grasp-ready finger configuration without prescribing a
  hard-coded closure time." So this citation does NOT actually support the
  "gate closure on alignment" pattern the junior uses it for in Tier 1 item 1
  — it supports something more specific and different (an internally-gated
  opposition *reward term*, with an additive top-level combination, and no
  staged closure at all).
- The canonicalization claim itself (Option A, Tier 2 item 4) is accurately
  represented — that part of the paper checks out.

### [2411.13020] "AsymDex: Asymmetry and Relative Coordinates for RL-based Bimanual Dexterity"
**Verdict: REAL BUT MISAPPLIED — the citation doesn't support the use case it's cited for.**
- Real paper (arXiv API confirms, Nov 2024, matches title/authors).
- The reward formula the junior quotes, `R_rel_pos = (α − ‖x_obj −
  x_initial‖)·β`, does appear in the paper — but "relative" in AsymDex means
  **relative pose between the two robot hands** (the dominant hand's frame
  relative to what the facilitating hand is holding), for **bimanual**
  coordination. It is not about a single parallel-jaw gripper's fingertip
  distances to a grasped object, which is the problem the junior's Section 2/
  Tier 1 item 3 actually needs to solve. The paper's own confirmation: "the
  paper uses relative coordinates between two robot hands, not between
  fingertips and a single object... 'Relative' here emphasizes inter-hand
  coordination, not fingertip-to-object metrics." This is a real paper being
  cited for a claim it does not make — the junior extrapolated a
  superficially similar word ("relative") across an unrelated problem
  setting (single-gripper vs. bimanual). This significantly weakens the
  evidence basis for "fingertip-relative distance reward" (Tier 1 item 3).

### [2205.03532] "Factory: Fast Contact for Robotic Assembly"
**Verdict: REAL BUT THE SPECIFIC CLAIM IS FABRICATED.**
- Real, well-known NVIDIA/UW paper (RSS 2022), title/authors/abstract all
  confirmed.
- The junior's specific claim — "Uses bilateral contact detection: reward
  gates progress on 'both fingers registering contact force > threshold'" —
  is **not in the paper**. Direct text search of the full paper (raw HTML
  grep, not LLM paraphrase) found **zero occurrences of the word
  "bilateral"** anywhere in the text. The paper's actual grasping/assembly
  reward for the Pick/Place/Screw tasks is a **dense keypoint-distance
  reward** ("sum of the keypoint distance [between nut and base of bolt] and
  [between end-effector and nut]"), not a contact-force-threshold gate. The
  paper's "Contact Forces" section (V-F) is about *validating simulated
  contact forces against real-world measured values* for physics-accuracy
  purposes — not about using contact force as a bilateral reward gate. The
  "1024 parallel environments" detail is accurate (1024 is the actual
  standard scene size used throughout the paper, though the abstract itself
  says "1000" as a round-number simplification), but the reward-gating
  mechanism attributed to this paper is invented.
- The Isaac Lab / IsaacGymEnvs `factory.md` claim about "0.1–1.0 N per finger
  contact thresholds gating success criteria" across five Factory tasks was
  not independently verified and should be treated as unverified pending a
  direct check of the actual IsaacGymEnvs docs/source, not assumed true by
  association with the arXiv paper.

### [2206.13966] "Dext-Gen: Dexterous Grasping in Sparse Reward Environments with Full Orientation Control"
**Verdict: CONFIRMED REAL AND ACCURATE.**
- Real paper (arXiv API confirms, June 2022).
- The specific curriculum claim is **verbatim correct**, confirmed from the
  paper's text: "starting with an orientation goal tolerance of π radians
  and iteratively reducing the tolerance to a minimum of 0.2 radians when the
  agent achieves a 0.75 success rate." This is a genuinely well-verified
  citation for progressive tolerance-tightening.
- Important caveat: this is an **orientation tolerance** (radians) for a
  6-DoF pose-matching task, not a **position/distance kernel std** in
  meters. It supports the general "progressive tightening" pattern but does
  **not** support any specific meters-scale std number. The junior's use of
  this citation as partial "citation basis" for "std ≈ 0.02m" (Tier 1 item 2)
  is an overreach — the paper never discusses linear distance kernel std at
  all.

### [2011.08458] "Learning Dense Rewards for Contact-Rich Manipulation Tasks"
**Verdict: REAL BUT OVERSTATED — does not support the specific claim it's cited for.**
- Real paper (arXiv API confirms, Nov 2020). Topic (self-supervised dense
  reward learning from images/tactile, peg-in-hole and USB insertion) is
  correctly identified.
- The junior's claim — "precision thresholds must be set relative to object
  size and gripper aperture... visibility threshold δ is determined
  empirically" — mischaracterizes the paper's content. Direct check found
  only task-specific hardware clearances (peg-in-hole 2.4mm, USB insertion
  1.0mm) with **no general statement or framework** about scaling reward
  thresholds to object size/gripper aperture. This paper does not actually
  support the "std should scale to object size" claim it's cited for.

### [2307.16752] "Dexterous Pre-grasp Manipulation for Human-like Functional Categorical Grasping"
**Verdict: FABRICATED SPECIFIC FORMULA — the claimed reward equation is not in the paper.**
- Real paper (arXiv API confirms, July 2023). Topic (dense multi-component
  reward, hand-centric r_h + object-centric r_o decomposition, learned
  pre-grasp manipulation) is correctly identified in general terms.
- However, the junior's specific claim — "reaching reward (1-tanh(10.0·d))...
  subgoal rewards (0.25) kick in only when d < 0.05 m" — was checked directly
  against the paper's full text and **no tanh-based reward formula appears
  anywhere in this paper at all**, nor a 0.25-magnitude subgoal reward at a
  0.05m threshold. This specific formula appears to be fabricated or
  conflated from a different source. (Notably, a `tanh(10·d)`-style formula
  *does* show up when checking a **different** cited paper, 2003.02740 — see
  below — suggesting the junior may have cross-contaminated details between
  two different papers it cited in the same section.)
- Note also: this paper uses the exact same `r_h` / `r_o` notation the junior
  attributed to a *different* paper (2604.04138/GRIT) for its multiplicative
  formula. That notation genuinely occurs in GRIT (verified above); whether
  2307.16752 uses hand-centric/object-centric decomposition too is plausible
  from its abstract, but the specific formula-level claims about it should
  not be trusted without further direct verification.

### [2003.02740] "Dense2Sparse Reward Shaping for Robot Manipulation" (actual title: "Balance Between Efficient and Effective Learning: Dense2Sparse Reward Shaping for Robot Manipulation with Environment Uncertainty")
**Verdict: REAL BUT THE KEY NUMERIC CLAIM IS FABRICATED.**
- Real paper (arXiv API confirms, March 2020). General dense-to-sparse
  curriculum concept is correctly represented.
- The junior's specific claim — "for small-object grasping, final-stage
  kernel std is typically 0.05–0.1 m" — was checked directly and **is not
  present**. The paper's own reaching-task reward does use a `tanh(10·d)`-
  style scaling with a completion threshold at **d ≤ 0.03 m** (not a "kernel
  std" and not the 0.05–0.1m range claimed), and this is a *task-completion
  threshold*, not a std parameter. This paper does not support the
  "std=0.05-0.1m" number attributed to it.

### [2511.04831] "Isaac Lab: A GPU-Accelerated Simulation Framework for Multi-Modal Robot Learning"
**Verdict: CONFIRMED REAL; infrastructure claim independently verified against actual installed Isaac Lab source, not just the paper.**
- Real paper (arXiv API confirms, Nov 2025). General framework description
  matches.
- Rather than trust the paper's text for the specific API claim, I checked
  the actual Isaac Lab installation in this environment
  (`/home/saps/IsaacLab`). `ContactSensor` is a real, first-class sensor
  class (`isaaclab/sensors/contact_sensor/contact_sensor.py`). A
  `contact_forces` reward function genuinely exists in
  `isaaclab/envs/mdp/rewards.py:281`, docstring: "Penalize contact forces as
  the amount of violations of the net contact force" — this matches the
  junior's paraphrase reasonably well (the junior's literal code snippet
  `penalize(net_contact_force_violations)` is a paraphrase, not the actual
  function signature, but the underlying capability claim is accurate).
  `undesired_contacts` and `desired_contacts` reward functions also exist.
  **This is the one claim in the report backed by primary-source
  verification stronger than the arXiv paper itself, and it holds up.**

### [2510.13694] "Information-Theoretic Reward Modeling for Stable RLHF"
**Verdict: REAL but essentially irrelevant / decorative citation.**
- Real paper (arXiv API confirms, Oct 2025) — but it is an **RLHF reward
  model paper** (language model preference reward hacking), not a
  manipulation-RL or robotics paper. It's cited only as generic "reward
  hacking is documented" flavor text alongside a Lilian Weng blog post, not
  for any specific technical claim about grasping. Harmless but not load-
  bearing; shouldn't be counted as robotics-domain evidence.

---

## Verdict on the specific `std=0.02m` claim (explicitly requested scrutiny)

**FABRICATED / unsupported extrapolation, not a literature-derived number.**

The junior's report states: "Recommended std ≈ 0.01–0.02 m (10–20 mm) for
18mm sphere" and lists citation basis as [2206.13966], [2011.08458],
[2003.02740]. Having checked all three papers directly:

- [2206.13966] (Dext-Gen) only discusses **orientation tolerance in radians**
  (π → 0.2 rad), never a linear-distance kernel std in meters.
- [2011.08458] only discusses **task-specific hardware clearances** (2.4mm
  peg, 1.0mm USB) as experimental setup facts, with no general
  size-to-threshold scaling claim.
- [2003.02740] only discusses a **0.03m task-completion threshold** for its
  own reaching task, not a 0.05–0.1m std range, and not a size-scaling rule.

None of the three cited papers state or imply a std of 0.02m, or even the
"0.05–0.1m" figure the junior claims is the literature norm before halving it
again to reach 0.02m. The 0.02m number is the junior's own back-of-envelope
geometric reasoning (18mm sphere, ~28mm gripper aperture → "14mm per side" →
round down) dressed up with citations that don't actually contain it. The
underlying geometric intuition (a 0.1m/100mm std is very loose relative to an
18mm object) is directionally reasonable engineering judgment, but it should
be labeled as such — an estimate, not a literature-backed number — and the
specific "1000x too loose" framing and "std ≈ 0.02m" target should be treated
as a plausible starting hypothesis to test empirically, not a validated
finding.

---

## Overall verdict on the Tier 1/2/3 recommendation

**The recommendation needs revision, not wholesale rejection.** The
underlying engineering diagnosis (reward-hacking via independently-satisfiable
terms; need alignment-aware gating; kernel too loose for object size) is
sound and matches genuine literature patterns. But two of three Tier-1 items
lost their strongest cited backing on inspection:

- **Tier 1 #1 (multiplicative gating):** Still solid. [2604.04138]/GRIT is a
  strong, precisely-verified citation for exactly this pattern. Keep as
  Tier 1. Drop [2606.21148] as support for "closure gating" specifically
  (it doesn't do that) — it can still be cited for canonicalization
  (Tier 2 #4), just not for staged/gated closure.
- **Tier 1 #2 (tighten kernel std):** The *direction* (tighten it) is good
  engineering judgment given the 18mm object vs. 100mm std mismatch, but
  **the specific target of 0.02m has no literature basis** — treat it as a
  hypothesis to sweep empirically (e.g., try 0.02m, 0.03m, 0.05m and measure
  grasp-success / training-stability trade-offs) rather than a number to
  adopt on citation authority.
- **Tier 1 #3 (fingertip-relative reward):** The citation basis is weaker
  than presented — [2411.13020]/AsymDex is about bimanual inter-hand
  relative coordinates, not single-gripper fingertip-to-object distance.
  [2307.16752]'s specific formula claim is also unverified/likely
  fabricated. The idea itself (reward symmetric fingertip-to-object distance
  to prevent one-sided contact) is reasonable and probably worth trying, but
  it should be presented as an engineering proposal informed by the general
  concept of relative/canonicalized observations, not as something these two
  papers specifically validated.
- **Tier 2 (canonicalization, curriculum):** Canonicalization via
  [2606.21148] holds up on inspection (that part of the paper's claim is
  accurate). Curriculum staging via [2206.13966] holds up (verbatim
  confirmed). [2606.31377] (Stage-Transition Dense Reward) is a real paper
  with a real "grasping regulation module... to prevent reward hacking" —
  relevant in spirit but I did not verify its "gripper-local measurements"
  claim to the same depth; treat as directionally supportive, not verbatim
  confirmed.
- **Tier 3 (ContactSensor fallback):** Infrastructure claim independently
  verified as real and available in this repo's actual Isaac Lab install —
  this is actually the best-verified claim in the whole report, ironically
  via primary source rather than the arXiv citation. However, drop
  [2205.03532]/Factory as support for "bilateral contact force gating
  reward" — that specific mechanism is not in the Factory paper; the
  citation should be replaced with "this is a standard, available Isaac Lab
  capability" (verified directly against source) rather than attributed to
  Factory's methodology.

**Bottom line:** proceed with the escalation structure (reward
redesign → observation canonicalization → contact sensors) — it's reasonable
engineering strategy — but strip out the specific fabricated numeric claim
(std=0.02m as a "literature value") and the two misapplied/fabricated
citations (AsymDex for fingertip-relative reward, Factory for bilateral
contact gating), and don't lean on [2606.21148] for the "gated closure"
narrative since the paper actually describes an additive, ungated-closure
reward instead.
