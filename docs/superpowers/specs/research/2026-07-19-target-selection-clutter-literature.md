# Research: target selection in clutter — grounding for the multi-die-with-distractors follow-on

**Date:** 2026-07-19
**Author:** Senior research thread (delegated by Principal)
**Purpose:** Tier 1 hypothesis-gate research for a future spec extending the
completed unified d12/d20 specialist policy
(`kb/wiki/experiments/unified-multi-die-specialist-distillation.md`) to find
and grasp a commanded die when OTHER (distractor) dice are also present in
the scene — explicitly deferred at that experiment's spec time, gated on one
citation (DexSinGrasp, arXiv:2504.04516) that was never independently
re-verified against its own primary text. Per CLAUDE.md's scientific-method
gate, this document must exist and be cited before that follow-on spec is
written. **This is research only — no env cfg code, no reward-term design,
no Isaac Sim launches.**

---

## 1. Verifying the DexSinGrasp citation

**The paper is real.** Confirmed via the arXiv API
(`export.arxiv.org/api/query?id_list=2504.04516`, live HTTP fetch, not taken
on faith): Xu, Liu, Gui, Guo, Jiang, Zhang, Xu, Gao, Shao, **"DexSinGrasp:
Learning a Unified Policy for Dexterous Object Singulation and Grasping in
Densely Cluttered Environments,"** submitted 2025-04-06, latest revision
v3 2025-10-25, cs.RO. Full text (not just abstract) fetched via
`ar5iv.labs.arxiv.org/html/2504.04516` and read directly.

**What it actually claims — checked against the primary text, not the prior
spec's paraphrase:**

- The paper's own no-curriculum ablation ("Training from scratch," Table
  IV) reports **exactly 0% success rate on R-8** (random arrangement, 8
  surrounding distractor objects) — a real, quantified collapse to zero,
  not a qualitative claim.
- **This does not generalize to every clutter condition tested in the same
  table**, and the prior spec's paraphrase ("uncurriculated multi-object
  clutter can collapse RL discovery") should be read with this caveat, not
  as a blanket claim: the same no-curriculum run scores **97% on D-8**
  (dense arrangement, 8 distractors) — i.e. training from scratch directly
  at the hardest distractor count did NOT collapse in the dense-arrangement
  condition, only in the random-arrangement one. The correct, precise
  reading is *"training directly at high distractor counts with no
  curriculum can collapse success to zero in some (not all) clutter
  arrangements — specifically the paper's harder, more randomly-scattered
  condition"* — a real and still strongly relevant finding for this
  project, but narrower than the prior spec's unqualified sentence implies.
- A second, independent data point in the same paper supports the general
  direction without the curriculum mechanism at all: the paper's
  **GraspReward-Only baseline** (pure grasp reward, no singulation
  mechanism, trained directly on cluttered scenes) degrades monotonically
  with distractor count — 66%→40%→10% (D-4→D-6→D-8) and 73%→61%→33%
  (R-4→R-6→R-8), Tables II/III. This is a *different* comparison (reward
  design, not curriculum) but corroborates the same underlying phenomenon:
  more distractors makes bare-grasp-reward RL discovery harder, monotonically,
  even before considering curriculum at all.
- **Verdict: the citation is real and accurately characterizes a genuine,
  quantified risk, but the prior spec's one-line paraphrase overstated its
  universality.** The revised, precise citation to carry into the future
  spec: *"DexSinGrasp (Xu et al. 2025, arXiv:2504.04516) found that training
  a unified singulation-and-grasping RL policy directly at 8 random
  distractors with no curriculum collapses success rate to exactly 0%
  (Table IV), while the identical no-curriculum recipe at 8 densely-packed
  distractors still reaches 97% — collapse is real but conditional on
  clutter arrangement/difficulty, not universal to 'clutter' as a category."*

No fabrication or misattribution found in the prior spec's citation (unlike
several precedents in `kb/wiki/concepts/citation-verification-practice.md`)
— the paper and its curriculum finding are real; only the paraphrase's
precision needed correcting.

---

## 2. How target selection in clutter is actually solved — survey

Four verified sources, spanning curriculum, observation-space, and
architectural approaches. Each fetched and checked directly (arXiv API for
existence, full text or abstract for claims), not cited from memory or a
secondary source.

### 2a. Curriculum: distractor-count annealing (DexSinGrasp itself, + its own foundational citation)

DexSinGrasp's own "Clutter Arrangement Curriculum Learning" (§III-B, read in
full) is the concrete, validated mechanism, and its starting point is
directly relevant: the curriculum literally **begins from a single-object
grasping policy** (their own words: *"we begin by training a grasping
policy designed exclusively for single-object scenarios... Based on this
initial policy, we continuously follow the curriculum and train on
increasingly complex singulation and grasping tasks"* — stages labeled
SO [Single-Object] → D-4 → D-6 → D-8, then transferred to R-4 → R-6 → R-8),
then anneals up distractor count. **This is the exact starting point this
project is already at** — a working single-object d12/d20 policy — which
makes DexSinGrasp's specific curriculum shape (not just "curriculum
learning" in the abstract) a close structural match, not just a generic
precedent. Their own Table IV shows the dense-to-random curriculum
direction (98/92/97% on D-4/6/8, 96/96/94% on R-4/6/8) beats both the
opposite order and no curriculum at every cell.

Curriculum learning as a general technique traces to Bengio, Louradour,
Collobert, Weston, **"Curriculum Learning,"** ICML 2009 — cited by
DexSinGrasp itself as its own foundational reference (their ref [25]); not
independently re-verified here beyond confirming DexSinGrasp cites it
correctly, since it's well-established and not the load-bearing claim.

### 2b. Observation-space: a dedicated target-object state slot + a fixed-size aggregate distractor-distance feature (DexSinGrasp's actual mechanism, read from its equations)

This is the single most directly transferable finding, and it's more
specific than "add a target-indicator one-hot." Reading DexSinGrasp's own
observation-space equation (§III-A, Eq. 1) directly: the state-based
teacher's observation is `s_t = [s_t^R, a_{t-1}, s_t^O, d_t^{HO}, T_t,
d_t^S]`, where:

- `s_t^O ∈ ℝ^16` is **only the target object's own state** (position,
  quaternion, linear/angular velocity, object-hand position difference) —
  distractor objects do **not** each get their own individual state vector
  in the observation at all.
- `d_t^S ∈ ℝ^8` is a **fixed-size, zero-padded aggregate** — "the
  distances between the target object and surrounding objects, indicating
  the level of enclosure," padded with zeros if the true distractor count
  is below 8.

In other words, DexSinGrasp's real mechanism for distinguishing the
commanded object from distractors is **not** "attention/pointer over a
variable-count object set" (§2d below) and **not** "broadcast a one-hot
flag across every spawned object's own full state" — it's architecturally
simpler: the target gets a full, privileged, individually-identified state
slot (exactly analogous to what this project already computes per env as
"the object"), and distractors are compressed into one small, fixed-size,
padded summary vector, never individually distinguished from each other.

### 2c. Target selection via a pre-specified target identity, when the arm actually has to disambiguate visually — Danielczuk et al. and Zeng et al.

Two additional verified real papers cover the case where you cannot assume
ground-truth state and must select the target visually:

- Danielczuk, Kurenkov, Balakrishna, Matl, Wang, Martín-Martín, Garg,
  Savarese, Goldberg, **"Mechanical Search: Multi-Step Retrieval of a
  Target Object Occluded by Clutter,"** ICRA 2019 (arXiv:1903.01588,
  verified via arXiv API). Confirmed via abstract: formalizes "Mechanical
  Search" as retrieving a **known target object** from a heap of
  distractors using an RGBD perception system plus push/suction/grasp
  action policies, evaluated over 15,000 simulated and 300 physical
  trials, >95% success. The target's identity is given a priori (a known
  target, not discovered); the paper's problem is exposing/extracting it
  from occlusion, not identifying which candidate is the target from
  scratch — directly analogous to this project's setup (the commanded
  shape is already known at episode start, exactly like Mechanical
  Search's "known target").
- Zeng, Song, Nagarajan, Yuan, Yin, et al. (MIT-Princeton team),
  **"Robotic Pick-and-Place of Novel Objects in Clutter with
  Multi-Affordance Grasping and Cross-Domain Image Matching,"** ICRA 2018
  (arXiv:1710.01330, verified) — 1st place, 2017 Amazon Robotics Challenge
  stowing task. Confirmed via abstract: after grasping *some* object from
  clutter, a separate cross-domain image-matching recognizer identifies
  *which* object was actually picked, by matching against reference
  product images. This is a real precedent for "target identity resolved
  by a recognition step external to the grasp policy itself" — relevant
  context for this project's own eventual Phase I (vision-detector-driven
  target identity), but a materially different mechanism than DexSinGrasp's
  ground-truth privileged-state approach, and not what this project would
  use while it still trains without a detector in the RL loop.

### 2d. Architectural approaches for variable-count object sets — considered, not needed here

For completeness (the task brief asked specifically about attention/pointer
mechanisms): Vinyals, Fortunato, Jaitly, **"Pointer Networks,"** NeurIPS
2015 (arXiv:1506.03134, verified) and Zaheer, Kottur, Ravanbakhsh, Poczos,
Salakhutdinov, Smola, **"Deep Sets,"** NeurIPS 2017 (arXiv:1703.06114,
verified) are the real foundational references for permutation-invariant/
variable-cardinality set representations, the general architecture family
that would be needed if the number of distractors were unbounded or
unknown at observation-construction time. **DexSinGrasp itself does not use
this family** — it sidesteps the variable-count problem with a fixed
maximum (8) and zero-padding (§2b). Given this project's own dice supply is
similarly small and bounded (this project's own die pool is 2-4 shapes,
single-digit distractor counts at most, matching DexSinGrasp's own 0-8
range), the fixed-size-padded approach is the better-precedented and
simpler starting point; Deep Sets/Pointer-Network-style architectures are
flagged as a real, literature-grounded fallback only if a fixed-size
padded feature turns out insufficient, not a first-choice recommendation.

### 2e. What this project's own existing foundational method (UniDexGrasp/UniDexGrasp++) does *not* cover — a real gap, not an oversight

Worth stating explicitly since this project's completed distillation
experiment is built directly on UniDexGrasp++'s GiGSL pattern: re-reading
both papers' abstracts directly (Wan et al., **"UniDexGrasp++,"** ICCV
2023, arXiv:2304.00464, verified; Xu et al., **"UniDexGrasp,"** CVPR 2023,
arXiv:2303.00938, verified) confirms both are explicitly **single-object,
table-top** formulations — "goal-conditioned" in both papers refers to a
target grasp *pose*, not object identity among simultaneously-present
distractors, and neither paper's setting includes other objects in the
scene at all. **This project's own foundational grasping method has never
been tested against simultaneous multi-object clutter** — DexSinGrasp is a
genuinely separate lineage (dexterous-hand clutter-singulation, not
UniDexGrasp's proposal-generation-plus-goal-conditioned-execution split),
and the follow-on experiment cannot assume GiGSL's own curriculum
mechanism (geometry-aware object-shape curriculum) transfers to a
distractor-count curriculum without treating it as a new, separate
mechanism to validate — which is exactly why DexSinGrasp, not
UniDexGrasp/UniDexGrasp++, is this section's primary source.

---

## 3. Grounding in this project's specific setup

- **No vision detector in the RL training loop yet.** Confirmed directly:
  `kb/wiki/experiments/dice-pick-demo.md` states *"Phase I (detector-derived
  state inside a trained policy) remains open"* — the completed dice-pick
  demo's detector runs inside a **scripted** DiffIK controller, not inside
  an RL policy, and the RL side (the unified specialist/distillation
  experiment) has never run with anything but ground-truth simulator state.
  This means §2c's vision-matching mechanisms (Zeng et al., and the
  detection half of Mechanical Search) are **not** the right mechanism for
  the *next* experiment specifically — they're the right mechanism for
  whatever eventually closes Phase I. The next experiment should use
  ground-truth ("which spawned die *is* the commanded one, read directly
  from sim state") target identification, matching DexSinGrasp's own
  state-based-teacher formulation (§2b) and this project's own existing
  practice of ground-truth privileged observations elsewhere (e.g. this
  project's `object_shape_class_onehot` term, also ground-truth/config-time,
  never detector-derived).
- **The existing `object_shape_class_onehot`/`object_geometry_descriptor`
  terms (`tasks/franka/mdp.py:124-167`, `tasks/franka/shape_observations.py`)
  answer "what shape is this die," not "which of several spawned dice is
  the commanded one."** Read directly: these are **config-time-static,
  broadcast-per-env** properties (`env.cfg.die_shape_class`, or for the
  mixed-population env, a deterministic `env_index % len(shapes)` function)
  — they say what shape a given env's object *is*, not which object among
  several *co-present* objects in the *same* env should be grasped. A
  target-selection experiment needs a structurally new observation concept
  (per-object identity among several simultaneously spawned objects in one
  env), not a natural extension of these terms — though the underlying
  *mechanism* (a ground-truth, config/state-derived per-env feature, not a
  learned or perception-derived one) is exactly the same pattern already
  established and validated by this project's own Task 1 for shape.
- **Naming collision to flag for the future spec:** this project's existing
  `target_object_position` observation term
  (`tasks/franka/lift_env_cfg.py:208`, `mdp.generated_commands(command_name=
  "object_pose")`) is the commanded **goal location** to carry the object
  to — it has nothing to do with target *object identity* despite the
  similar name. A distractor-selection observation term should be named to
  avoid this collision (e.g. `commanded_die_index` / `target_die_state`,
  not anything containing "target_object" unqualified).
- **DexSinGrasp's `s_t^O`-plus-`d_t^S` split (§2b) maps cleanly onto this
  project's existing schema.** This project already computes a per-env
  "the object's" position/pose as its own dedicated observation term
  (`object_position`, `tasks/franka/mdp.py:69`,
  `object_position_in_robot_root_frame`) — in a multi-die env, this term
  would become "the *commanded* die's position" (privileged, dedicated,
  matching DexSinGrasp's `s_t^O`), and a **new** term would be needed for
  the distractor dice specifically (a fixed-size, zero-padded
  distance-or-position summary, matching DexSinGrasp's `d_t^S`) — additive
  to the existing schema, not a redesign of it, the same "additive, not a
  redesign" pattern this project's own Task 5
  (`shape_class_onehot_per_env`/`geometry_descriptor_per_env`) already used
  successfully for per-env shape.
- **Action space (joint-space PPO via `rsl_rl`) is not implicated by any
  source found in this survey.** DexSinGrasp's own curriculum/observation
  findings are stated action-space-agnostically (their action space is a
  22-dim dexterous-hand palm+finger space, not directly comparable to this
  project's 8-dim arm+gripper space in dimensionality, but the curriculum
  and observation-space mechanisms are not described as depending on a
  particular action space) — no literature found here argues the Franka
  joint-space action formulation itself needs to change for target
  selection in clutter to work; this is consistent with this project's
  own `2026-07-11-joint-space-lift-research.md` finding that action space
  and reward/observation design are treated as orthogonal axes both in
  Isaac Lab's own shipped tasks and in this literature.

---

## 4. Proposed candidate hypothesis and methodology (for a future spec to cite — NOT a spec itself)

**Hypothesis:** Starting from the already-checkpointed, 8/8-discovery,
single-object d12/d20 unified policy
(`gs://rl-manipulation-hks-runs/unified-multi-die-specialists/joint-die-d12-d20-mixed/seed42/2026-07-19_12-53-35/model_2998.pt`)
and continuing training with (a) a new ground-truth, ADDITIVE observation
pair — a dedicated "commanded die" state slot (replacing the existing
single-object `object_position` semantics with "the commanded die's
position," unchanged in form) plus a new fixed-size, zero-padded
distractor-distance/position summary term (DexSinGrasp's `d_t^S` pattern,
§2b) — and (b) a distractor-count curriculum starting at 0 distractors
(the policy's own current, already-solved single-object state) and
annealing upward (DexSinGrasp's SO→D-n pattern, §2a), will preserve most of
the existing 8/8 discovery rate for the commanded shape, substantially
outperforming the same distractor-count target trained directly with no
curriculum and no distractor-identity observation term.

**Falsification condition:** if this two-part mechanism (additive
observation term + count curriculum) still collapses discovery
substantially at the target distractor count — i.e. a drop comparable in
kind to DexSinGrasp's own reported no-curriculum R-8 collapse (0%,
Table IV, §1) even with both mechanisms in place — that falsifies "these
two additions are sufficient," and the honest escalation per §2d is toward
a genuinely different mechanism (a Deep-Sets/attention-style architecture
over distractor state, rather than a fixed-size padded summary), not a
parameter retune within the same mechanism.

**What this research does NOT resolve (left for the spec/plan):** the exact
distractor-count ladder and arrangement (dense vs. random analog for a
flat tabletop of dice, not a heap); the exact padded feature width/shape;
whether distractor dice should be drawn from the same or different shape
classes than the commanded one; reward-term design for
inadvertent-distractor-disturbance (DexSinGrasp's own reward design
explicitly rewards singulation motion, which may or may not be the right
model for dice on a flat table rather than a heap) — all Tier 1 spec-time
decisions, not research-time ones.

---

## 5. Open risks / gaps, stated plainly rather than papered over

- **No source found isolates "PPO, ground-truth state (no vision), flat
  tabletop small-rigid-object multi-object target selection" as its own
  studied variable.** DexSinGrasp's own setting is a dexterous hand over a
  heaped/piled clutter arrangement (objects genuinely occluding each
  other), materially different from dice resting separately on an open
  tabletop (which this project's dice are, per every prior experiment in
  this arc) — DexSinGrasp's singulation mechanism specifically exists to
  solve *occlusion*, which may not even be a real problem for flat,
  non-overlapping dice placement. This is a real gap: the curriculum and
  observation-space findings (§2a/§2b) are the best available grounding,
  but whether *singulation* (actively displacing distractors) is even a
  relevant sub-problem for this project's likely tabletop (non-heaped)
  arrangement is unresolved and should be an explicit scoping decision in
  the future spec, not assumed either way.
  - **However, a real, project-adjacent counter-data-point already exists:
    `dice-pick-demo.md`'s five-die tabletop scene** (open, non-heaped
    arrangement, closest analog this project has to what the RL follow-on
    would use) already required real disambiguation work even without
    occlusion — its own documented history includes a same-location
    higher-confidence wrong-class candidate displacing the correct
    detection (the d4/d10 confusion at seed 123, §"d4 may be a
    systematically weak detection class") and a table-hole false positive
    needing an explicit geometric-plausibility filter. That's a
    perception-layer finding, not an RL-target-selection one — but it's
    evidence that "flat, non-heaped, no-occlusion" does not imply
    "trivial to disambiguate," even before considering whether an RL
    policy's *motor* behavior (as opposed to a detector's *classification*)
    would misdirect toward the wrong same-shape-class die at close range.
- **DexSinGrasp's collapse finding is for a dexterous multi-finger hand,
  not a Franka parallel-jaw gripper** — the paper's own singulation
  mechanism (*"finger flickering, palm rubbing, finger-palm vibration"*,
  §IV-D, read directly) exploits high-DoF fingers Franka's parallel jaw
  does not have. If the future spec leans on singulation specifically
  (rather than just curriculum/observation, which are the parts actually
  argued as transferable above), that dependency should be re-examined,
  not assumed to transfer.
- **No literature found that isolates "distractor identity via a
  fixed-size padded feature" vs. "distractor identity via a per-object
  one-hot flag on every spawned object" as a controlled comparison** —
  §2b's recommendation rests on DexSinGrasp being the one directly
  verified precedent that made this exact design choice and reported it
  working, not on a comparison study showing it beats alternatives.
- **This document does not verify whether the curriculum's stage-transition
  criterion (DexSinGrasp's own `Σ‖p^target − p_i‖₂/n > 0.16` distance-based
  gate) is meaningful at this project's own object/table scale** — die
  sizes (~16-48mm, per the completed specialist experiment) vs.
  DexSinGrasp's own object scale were not cross-checked; a future spec
  should treat the curriculum's exact gating threshold as a parameter to
  re-derive for this project's own geometry, not copy verbatim.

---

## Related

[[unified-multi-die-specialist-distillation]] (the experiment this
follow-on extends), [[dice-pick-demo]] (this project's only existing
multi-object/five-die scene, scripted not RL — perception-layer
disambiguation evidence in §5),
`kb/wiki/concepts/citation-verification-practice.md` (the standing practice
this document's §1 follows).
