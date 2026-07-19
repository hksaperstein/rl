# Target selection among distractor dice: extending the unified d12/d20 policy into clutter (Experiment 2 of the multi-die RL arc)

## Context

`kb/wiki/experiments/unified-multi-die-specialist-distillation.md` (spec:
`docs/superpowers/specs/2026-07-16-unified-multi-die-specialist-distillation-design.md`)
finished 2026-07-19: a single unified PPO policy
(`FrankaDieLiftJointD12D20MixedEnvCfg`,
`gs://rl-manipulation-hks-runs/unified-multi-die-specialists/joint-die-d12-d20-mixed/seed42/2026-07-19_12-53-35/model_2998.pt`)
reliably grasps-and-lifts a commanded d12 or d20 die (8/8 sustained-lift
envs for each shape, `num_envs=8`, undiluted 48mm, `franka_checkpoint_review.py`'s
instrumented lift-threshold convention) **when exactly one die is present
in the scene.** That spec explicitly deferred distractor dice / target
selection to a follow-on ("Experiment 2"), gated on one citation
(DexSinGrasp, arXiv:2504.04516) that had never been independently
re-verified against its own primary text — see that spec's own Scope
section, "Distractor dice / target-selection among co-present objects."

That gate is now closed:
`docs/superpowers/specs/research/2026-07-19-target-selection-clutter-literature.md`
independently re-verified the DexSinGrasp citation (real, but the prior
spec's paraphrase overstated its universality — see that doc's §1) and
surveyed real curriculum/observation-space/architecture precedent for
target selection in clutter. This spec is the follow-on that document
was written to ground, per CLAUDE.md's Tier 1 gate ("this document must
exist and be cited before that follow-on spec is written").

**Starting point:** the finished unified checkpoint above. **Question:**
can that policy be extended to find and grasp a commanded die among
distractor dice, in the same scene, without collapsing its current 8/8
per-shape discovery rate?

## Research grounding

Full research doc:
`docs/superpowers/specs/research/2026-07-19-target-selection-clutter-literature.md`
(citation re-verification + a 4-part survey: curriculum, observation-space,
vision-based target selection, variable-cardinality architectures + a
grounding pass against this project's own codebase). Load-bearing
citations for this spec's design choices:

- **Curriculum mechanism**: **DexSinGrasp (Xu, Liu, Gui, Guo, Jiang,
  Zhang, Xu, Gao, Shao, "DexSinGrasp: Learning a Unified Policy for
  Dexterous Object Singulation and Grasping in Densely Cluttered
  Environments," arXiv:2504.04516)**, §III-B "Clutter Arrangement
  Curriculum Learning," independently re-verified against its own primary
  text (research doc §1, §2a). Its own curriculum **begins from a
  single-object grasping policy** ("we begin by training a grasping
  policy designed exclusively for single-object scenarios... Based on
  this initial policy, we continuously follow the curriculum and train on
  increasingly complex singulation and grasping tasks," stages SO → D-4 →
  D-6 → D-8 → R-4 → R-6 → R-8) — a close structural match to this
  project's own starting point (a working single-object d12/d20 policy),
  not just generic "curriculum learning" precedent. Its own no-curriculum
  ablation (Table IV) is the precise, re-verified risk this curriculum
  guards against: training directly at 8 random distractors with no
  curriculum collapses to **exactly 0%** (R-8), while the identical
  no-curriculum recipe at 8 densely-packed distractors still reaches
  **97%** (D-8) — collapse is real but conditional on clutter
  arrangement/difficulty, not universal to "clutter" as a category
  (research doc §1). A second, curriculum-independent data point
  (GraspReward-Only baseline) corroborates the general direction: bare
  grasp reward with no singulation mechanism degrades monotonically with
  distractor count (66%→40%→10% D-4→D-6→D-8, 73%→61%→33% R-4→R-6→R-8,
  Tables II/III).
- **Observation-space mechanism**: DexSinGrasp's own state-based-teacher
  observation equation (§III-A, Eq. 1, read directly), not a paraphrase:
  `s_t = [s_t^R, a_{t-1}, s_t^O, d_t^{HO}, T_t, d_t^S]`, where `s_t^O ∈
  ℝ^16` is **only the target object's own state** (distractors get no
  individual state vector at all) and `d_t^S ∈ ℝ^8` is a **fixed-size,
  zero-padded aggregate** of target-to-surrounding-object distances,
  "padded with zeros if the true distractor count is below 8" (research
  doc §2b). This is the single most directly transferable finding, and it
  is architecturally simpler than a per-object one-hot flag or an
  attention/pointer mechanism over a variable-count object set — it
  sidesteps the variable-cardinality problem with a fixed maximum and
  zero-padding, not a permutation-invariant encoder (Deep Sets/Pointer
  Networks, both real, verified, and considered — research doc §2d — but
  flagged as a fallback only if the fixed-size approach proves
  insufficient, not a first choice, since DexSinGrasp itself doesn't use
  that family and this project's own distractor counts are similarly
  small and bounded).
- **This project's own foundational grasping method (UniDexGrasp/
  UniDexGrasp++, GiGSL — the basis of the just-finished specialist/
  distillation experiment) has never been tested against simultaneous
  multi-object clutter** — both are explicitly single-object, table-top
  formulations (research doc §2e, verified against both papers'
  abstracts directly). DexSinGrasp is a genuinely separate lineage; this
  spec cannot assume GiGSL's own curriculum mechanism (a geometry-aware
  shape curriculum, unrelated to distractor count) transfers to a
  distractor-count curriculum without treating it as its own, separately
  validated mechanism — which is why DexSinGrasp, not UniDexGrasp++, is
  this spec's primary curriculum/observation source.
- **No vision detector in this experiment.** Ground-truth,
  simulator-read object state only — matching DexSinGrasp's own
  state-based-teacher formulation and this project's own existing
  practice for `object_shape_class_onehot` (also ground-truth,
  config-time-derived, never detector-derived). Vision-based target
  selection under occlusion (Danielczuk et al., "Mechanical Search," ICRA
  2019; Zeng et al., ICRA 2018 — both real, verified, research doc §2c)
  is the right mechanism for this project's eventual Phase I
  (detector-derived state inside a trained policy, still open per
  `kb/wiki/experiments/dice-pick-demo.md`), not for this experiment.

## Grounding in this project's actual codebase (what's new vs. reused)

This section is the concrete bridge from the research doc's §3 (which
identified the relevant gaps but did not resolve them) to an actual
design.

### Scene topology — genuinely new, no RL precedent in this repo

Every RL env cfg in `tasks/franka/dice_lift_joint_env_cfg.py` (including
`FrankaDieLiftJointD12D20MixedEnvCfg`, this spec's starting point) has
exactly **one** `scene.object` slot per env — even the mixed-shape env
varies which asset lands in that one slot per env (via
`MultiAssetSpawnerCfg(random_choice=False)`'s deterministic round-robin),
never two simultaneous objects in the same env. There is no
`ManagerBasedRLEnv` precedent in this codebase for multiple simultaneous
`RigidObjectCfg` entities in one scene.

There IS a real, working precedent for multiple simultaneous dice in one
`InteractiveSceneCfg` — just not wired into a `ManagerBasedRLEnv`:
`tasks/franka/dice_scene_cfg.py`'s `DiceSceneCfg` (the scripted five-die
`dice_pick_demo.py` scene) declares five independent `RigidObjectCfg`
fields (`die_d4`, `die_d8`, `die_d10`, `die_d12`, `die_d20`), each its own
`prim_path="{ENV_REGEX_NS}/Die_<type>"` and its own `UsdFileCfg` spawn,
all as sibling fields on one `InteractiveSceneCfg` subclass. `InteractiveSceneCfg`
is the same base class `FrankaLiftSceneCfg`/`FrankaDieLiftJointD12D20MixedEnvCfg`'s
scene already use — this pattern is directly reusable inside a
`ManagerBasedRLEnv`, it has simply never been exercised there before.

**Design decision: extend `FrankaDieLiftJointD12D20MixedEnvCfg`'s scene
with two new sibling `RigidObjectCfg` fields**, `distractor_1` and
`distractor_2` (prim paths `{ENV_REGEX_NS}/Distractor1`/`Distractor2`),
following `DiceSceneCfg`'s exact pattern — new scene fields, not a new
mechanism. `scene.object` remains the sole target/commanded die slot,
unchanged in meaning.

### Target identity — NOT a new observation flag; resolved by scene topology itself

The research doc's proposed hypothesis (§4) describes "a dedicated
commanded-die observation slot" as something to add. Re-reading it
against this codebase's actual constraints resolves that more precisely:
**no new target-identity signal is needed.** In this project's scene
design, `scene["object"]` is *structurally* always the commanded die (the
role a scene field plays is fixed at cfg-construction time, matching
every other per-env-cfg-constant convention already in this codebase —
`die_shape_class`, `die_shape_classes_per_env`; Isaac Lab has no
mechanism for reassigning which named scene entity is "the target" at
runtime, the same constraint that makes `MultiAssetSpawnerCfg`'s
per-env-fixed-at-spawn-time assignment the norm, not an exception). The
already-existing `object_position` observation term (unchanged in form)
**is** DexSinGrasp's `s_t^O` — a full, privileged, individually-identified
state slot for the target, exactly the paper's own mechanism, already
present in this codebase, needing no redesign. This resolves the research
doc's open question cleanly and avoids the `target_object_position`
naming collision it flagged (§3): the commanded-die identity was never
missing, only real *co-present other objects* were.

### The one genuinely new observation term: a fixed-size, zero-padded distractor-distance summary

Matching DexSinGrasp's `d_t^S` (§2b) as literally as this project's own
schema conventions allow: a new observation term, **`distractor_distance_summary`**
(K=2, one scalar per distractor slot — Euclidean distance between
`scene["object"]`'s root position and each distractor's own root
position, in world frame; frame choice doesn't affect a scalar distance).
Slot `i` is **hard-zeroed** (not the distractor's real, possibly-large,
parked-off-table distance) whenever curriculum stage makes that slot
inactive — see "Curriculum" below — mirroring the paper's own literal
zero-padding convention ("padded with zeros if the true distractor count
is below" the max), not a stand-in "far away" sentinel value. This is
implemented as a new pure function (in `tasks/franka/shape_observations.py`
or a new sibling module) following the exact idiom
`shape_class_onehot_per_env`/`geometry_descriptor_per_env` already
established: a per-env-cfg constant (`active_distractor_count`, new) read
at cfg-construction time, not live USD/spawner-state introspection.

This is additive to the existing 41-dim schema (Task 4's verified
observation layout: `joint_pos_rel(9) + joint_vel_rel(9) +
object_position(3) + target_object_position(7) + last_action(8) +
shape_class_onehot(4) + geometry_descriptor(1)`), growing it by exactly
K=2 to 43 dims. No existing term changes shape or meaning.

### Distractor shape population — reuses proven infra, deliberately excludes d8/d10

Each distractor slot gets its own `MultiAssetSpawnerCfg(assets_cfg=[d12_cfg,
d20_cfg], random_choice=False)`, drawing from **{d12, d20} only** — the
same two already-proven, already-48mm-parity-scaled assets
`FrankaDieLiftJointD12D20MixedEnvCfg` already uses for the target, at
their existing verified scale constants (d12: 0.001476, d20: 0.001585).
No re-derivation, no new size research. A given env's distractor(s) may
land on the same shape as that env's own target (e.g. target=d12,
distractor=d12 — the harder, near-identical-appearance case) or the other
shape (target=d12, distractor=d20 — the easier, already-shape-distinguishable
case); both regimes are present in one pooled training population rather
than choosing one, since neither the research nor this project's own
prior evidence motivates excluding either regime, and splitting same-shape
vs. cross-shape into separate controlled arms would multiply this
experiment's run count for a finer-grained question ("does shape
similarity change distractor-rejection difficulty") that is a natural
follow-on ablation, not this experiment's own hypothesis.

**d8/d10 are deliberately excluded from the distractor population**,
even though they're visually/physically valid dice: both are genuinely
null shapes for this project's own grasp task (0/9 real training
attempts across two independent size regimes, `kb/wiki/experiments/
unified-multi-die-specialist-distillation.md`'s Task 2 + Task 3.5) —
their behavior as a *co-present, never-to-be-grasped* object has never
been tested at all (no confirmed-working physics/collision behavior in
any multi-object scene, RL or scripted), so including them here would
confound any failure between "the policy can't reject this distractor"
and "this untested shape's mere presence in the scene misbehaves for
reasons unrelated to target selection." Restricting to the two shapes
this project has already validated end-to-end (48mm parity, real 8/8
discovery, real physics behavior under load) isolates the one new
variable this experiment actually tests.

### Distractor placement — reuses the existing reset-event mechanism, no new sampler

The existing `reset_object_position` event
(`mdp.reset_root_state_uniform`, `tasks/franka/lift_env_cfg.py`'s
`EventCfg`) already randomizes the target's position within a bounded
range at every reset. **Design decision: reuse this same mechanism
per-entity with disjoint position ranges** — one `EventTermCfg` per
scene entity (`object`, `distractor_1`, `distractor_2`), each a
`reset_root_state_uniform` call with its own non-overlapping `pose_range`
sub-region of the table, rather than building a new minimum-spacing
rejection sampler (`dice_pick_demo.py`'s `sample_dice_layout` does exist
as a precedent for that approach, but is new, more complex code this
experiment doesn't need if disjoint ranges are sufficient). This is
flagged as a real spec-time design choice, not deferred blindly to the
implementing task: the implementing task must verify empirically
(spawn-and-settle check, no training needed) that the chosen disjoint
sub-regions (a) don't push any entity off the table or out of the arm's
reachable workspace, and (b) leave no possibility of table-edge overlap
between adjacent regions given each die's own real footprint — reusing
this project's own established "verify a scale/spacing choice via a
`_diag_*` script before trusting it in training" convention.

An **inactive** distractor slot (see "Curriculum" below) reuses the same
`reset_root_state_uniform` mechanism with a degenerate (zero-width) range
at a fixed off-workspace parking position (outside the arm's reachable
volume and outside any camera/collision path relevant to the target) —
again a parameter choice on an existing event term, not new event-handling
code.

### Reward and termination — deliberately unchanged

**No new reward term for distractor avoidance/disturbance is added in
this experiment**, even though the research doc's §4 flags this as an
open, unresolved design question. This is a deliberate scope choice, not
an oversight: DexSinGrasp's own singulation-specific reward mechanism
("finger flickering, palm rubbing, finger-palm vibration," §IV-D) is
explicitly a high-DoF dexterous-hand technique that does not transfer to
a Franka parallel-jaw gripper (research doc §5) — leaning on it here
would import an unvalidated mechanism. The paper's own GraspReward-Only
baseline (pure grasp reward, no singulation mechanism at all) still
achieves real, if degrading, success under clutter (§ above) — evidence
that curriculum + observation conditioning alone is a reasonable first
mechanism to test before adding a new reward term. Just as importantly,
adding a new reward term in the SAME experiment as the curriculum and
observation changes would make any failure impossible to attribute to a
specific cause — the same reasoning the prior spec used to justify NOT
combining shape-unification, target-selection, and full clutter in one
experiment. `RewardsCfg`, `TerminationsCfg` (`object_dropping` scoped to
`scene["object"]` only, unchanged) are inherited byte-identical from
`FrankaDieLiftJointD12D20MixedEnvCfg`. A knocked-off-table distractor is
accepted as an untreated edge case for this first pass (monitored via eval
video, not engineered around preemptively without evidence it's a real
problem).

### Curriculum — separate env cfg classes + checkpoint-resume, matching this project's own staged-anneal precedent

DexSinGrasp's curriculum (SO → D-n, §III-B) is a *training-time*
mechanism in their own codebase; this project has no equivalent runtime
distractor-count curriculum, but it has direct, already-proven precedent
for the same *effect* via a different, already-established mechanism:
`FrankaDieLiftJointMidEnvCfg`/`...HeavyEnvCfg`'s staged size-anneal
(48→39.1→30.3mm, each stage its own env cfg class, checkpoint-resumed via
`train_franka.py --checkpoint`). This experiment reuses that exact
pattern for distractor count instead of size:

- **Stage SO** (0 active distractors): a NEW env cfg
  (`FrankaDieLiftJointD12D20TargetSelectionSOEnvCfg`, illustrative name)
  — the full 3-die scene topology from `FrankaDieLiftJointD12D20MixedEnvCfg`
  plus `distractor_1`/`distractor_2` (both **parked**,
  `active_distractor_count=0`, `distractor_distance_summary` fully
  zero-padded). **Trained from scratch, NOT resumed from `model_2998.pt`.**
  This is a deliberate, precedented choice, not an oversight: `model_2998.pt`
  was trained under the 41-dim schema; this stage's schema is 43-dim
  (the new `distractor_distance_summary` term). A checkpoint resume
  requires matching observation/action dimensionality (rsl_rl's
  `ActorCritic` network shape is fixed at construction from the
  checkpoint's own dims) — there is no cross-dimensionality warm-start
  mechanism in this codebase, and building one would itself be new,
  unvalidated architecture, out of a spec's scope to invent. This
  project's own precedent for exactly this situation is Task 1→Task 2 of
  the prior experiment: when `object_shape_class_onehot`/
  `object_geometry_descriptor` were added as new observation dims, every
  downstream specialist was trained fresh under the new schema, never
  resumed across the schema change. Stage SO is this experiment's
  equivalent — and doubles as a real internal check that the schema
  extension itself (two new, always-inert scene bodies + two new
  always-zero observation dims) doesn't by itself regress discovery,
  independent of whether real distractor pressure does.
- **Stage D1** (1 active distractor, `distractor_1` active with a real
  MultiAssetSpawnerCfg population, `distractor_2` still parked): resumes
  from Stage SO's own checkpoint via `train_franka.py --checkpoint`
  (schema now stable across stages — a normal, already-precedented
  resume).
- **Stage D2** (2 active distractors, both `distractor_1`/`distractor_2`
  real): resumes from Stage D1's own checkpoint.

Each stage: 1500 iterations (this project's established from-scratch/
per-stage default), `num_envs=4096` split via the existing d12/d20
round-robin (unaffected by this change), single seed (seed42, matching
the checkpoint this experiment starts from) — multi-seed replication is
explicitly deferred (see Global Constraints below).

### Eval tooling — reused with one required extension

`franka_checkpoint_review.py`'s instrumented lift-threshold mechanism
(`resting_z`/`max_height_gain`/`max_consecutive_lifted_steps` off
`scene["object"].data.root_pos_w[:, 2]`, the same convention every eval
in this arc already uses) is reused **unchanged** for the actual
pass/fail measurement — success is still scored purely on whether the
*target* die (`scene["object"]`, still a single, unambiguous entity) gets
lifted and sustained, exactly as today. The one required extension: the
script's `--variant` dispatch (`if/elif` chain, per-variant `_PLAY` cfg
import/construction) needs new variant strings
(`joint-die-target-selection-so`/`-d1`/`-d2`, illustrative names) mapped
to this spec's new `_PLAY` env cfg classes — the same mechanical addition
every prior variant in this file already required, not a redesign of the
script. Recording distractor positions/whether they were disturbed is a
genuinely nice-to-have instrumentation addition, not required for this
experiment's own pass/fail criterion — left to the implementing task's
judgment, not mandated here (avoiding scope creep into engineering a
diagnostic this hypothesis doesn't need).

### Execution backend

Per CLAUDE.md's desktop-first GPU routing: dispatch each stage via
`scripts/check_gpu_availability.sh` → desktop (`run_on_desktop_gpu.sh`) if
AVAILABLE, cloud fallback (`docs/cloud/dispatch-checklist.md`'s recipe)
otherwise, matching every recent task in this arc (Tasks 5/6 both ran
desktop-only, $0 cloud spend). Non-headless when local/desktop
(CLAUDE.md's standing instruction), headless only if it falls back to
cloud.

## Scope

### Global Constraints (in scope for this experiment)

- **Two shapes only: d12 and d20**, for both target and distractor roles
  — the two shapes this project has actually validated end-to-end. Do not
  introduce d8/d10/d4 as distractors or targets in this experiment (see
  "Distractor shape population" above for why).
- **Distractor count: 0 → 1 → 2, staged.** Do not train or evaluate a
  3+-distractor configuration in this experiment.
- **Flat, non-overlapping tabletop placement only** (disjoint reset-range
  regions per entity). Do not build a heap/piled/occluding arrangement, or
  any singulation-specific reward/action mechanism (finger-flicking,
  palm-rubbing, or any other DexSinGrasp-specific dexterous-hand
  mechanism) — this project's dice sit separately on an open table in
  every prior experiment in this arc, and DexSinGrasp's own singulation
  mechanism exists specifically to solve *occlusion*, which this
  project's flat placement does not obviously have (research doc §5,
  explicitly unresolved — treated here as a reason to NOT assume
  singulation is needed, not as license to add it speculatively).
- **Ground-truth object-state observations only** — no vision-detector
  integration in this experiment (matches every experiment in this arc so
  far).
- **Lift-only task horizon** — no new carry-to-goal criterion beyond what
  `FrankaDieLiftJointD12D20MixedEnvCfg` already inherits.
- **Reward function unchanged** — no new distractor-avoidance/disturbance
  reward term in this experiment (see "Reward and termination" above for
  the reasoning; this is a hypothesis to test in a follow-on only if the
  curriculum+observation mechanism alone is falsified, not a parallel
  mechanism to test simultaneously).
- **Single seed (seed42) per stage** — multi-seed replication is
  explicitly deferred to a follow-on if this experiment's own single-seed
  result is positive and warrants confirming its robustness; this
  experiment establishes whether the mechanism works at all before
  spending the additional GPU budget multi-seed replication would cost.
- **Scene topology: exactly 3 simultaneous dice per env at most** (1
  target + 2 distractors) — do not extend the scene to more entities in
  this experiment.

### Explicitly out of scope, deferred to a further follow-on (not this spec)

- **d8/d10 as distractors or targets** — remains gated on those shapes
  ever achieving real single-object discovery first (unresolved, per the
  prior experiment's FINAL VERDICT).
- **3+ distractors, heaped/occluding arrangements, or any singulation
  mechanism** — DexSinGrasp's own R-8/D-8 configurations are a
  dexterous-multi-finger-hand heap-clutter setting explicitly
  distinguished from this project's flat tabletop (research doc §5); if
  this experiment succeeds at 2 distractors, a later follow-on should
  decide whether to push further based on this experiment's own real
  result, not assume it now.
- **A distractor-avoidance/disturbance reward term** — only if this
  experiment's curriculum+observation-only mechanism is falsified (see
  Falsification condition below).
- **Same-shape-only vs. cross-shape-only distractor ablation** — this
  experiment pools both regimes in one training population (see
  "Distractor shape population" above); isolating them is a finer-grained
  follow-on question.
- **Multi-seed replication.**
- **Vision-detector-derived target identity** — this project's eventual
  Phase I, unrelated to this experiment's ground-truth mechanism.

## Falsifiable hypothesis

> Starting from the checkpointed, 8/8-discovery, single-object d12/d20
> unified policy (`model_2998.pt`), extending it with (a) two new,
> always-present scene entities (`distractor_1`/`distractor_2`, drawn
> from {d12, d20}) plus one new, additive observation term
> (`distractor_distance_summary`, a fixed-size K=2 zero-padded
> target-to-distractor distance vector — DexSinGrasp's own `d_t^S`
> mechanism, arXiv:2504.04516 §III-A Eq. 1) and (b) a 3-stage
> distractor-count curriculum (0 → 1 → 2 active distractors, each stage
> checkpoint-resumed from the prior stage — DexSinGrasp's own SO→D-n
> curriculum shape, §III-B), while leaving the reward function and
> target-identification mechanism (the existing, unchanged `object_position`
> term) untouched, will preserve most of the existing single-object 8/8
> discovery rate for BOTH d12 and d20 at the final (2-active-distractor)
> curriculum stage.

**Falsification condition (numeric, pre-registered):**

- **Primary bar:** at Stage D2 (2 active distractors, the target
  configuration), if EITHER shape's discovery rate — evaluated with
  `franka_checkpoint_review.py`'s existing instrumented lift-threshold
  convention, `num_envs=8`, the full 3-die scene active, target shape
  fixed per eval run exactly as every specialist eval in this arc already
  does — falls **below 6/8 (75%)**, that shape's result falsifies "this
  two-part mechanism (curriculum + observation, reward unchanged) is
  sufficient to preserve discovery under 2-distractor clutter." 6/8 is
  chosen as a bar that sits clearly above both (i) DexSinGrasp's own
  reported no-curriculum-collapse floor (an exact 0%, Table IV — the
  literature's own definition of genuine collapse) and (ii) this
  project's own historical "partial/fragile" discovery signature (e.g.
  the original d20 asset-bisect's 1/3-seed pattern, or Task 5's own
  BC/DAgger regression to 4/8 and 1/8) — while still explicitly allowing
  some real degradation from the clean single-object 8/8, consistent with
  DexSinGrasp's own curriculum-equipped results never reaching a literal
  100% either (92–98% D-8, 94–96% R-8, Table IV).
- **Internal sanity gate (Stage SO, before D1/D2 are attempted):** if
  Stage SO's own freshly-trained (not resumed) checkpoint does not reach
  at least 7/8 for both shapes in the SAME 3-die-topology, 0-active-distractor
  scene, that specifically falsifies "the new scene entities + new
  always-zero observation dims are inert" — a materially different,
  earlier failure than a real distractor-pressure collapse, and D1/D2
  should not proceed until this is understood (per this project's own
  "Phase 1 gate before Phase 2" precedent, e.g. the prior experiment's
  Task 3.5 gate before Task 4).
- **Escalation on falsification:** per the research doc's own §4/§2d, if
  Stage D2 falsifies the primary bar for either shape while Stage SO
  clears its own gate (i.e. the schema extension itself is fine, but real
  distractor pressure specifically collapses discovery), the honest
  escalation is toward a genuinely different mechanism — most directly, a
  Deep-Sets/attention-style architecture over distractor state (research
  doc §2d) or a new distractor-avoidance reward term (deliberately not
  attempted in this experiment, see Scope above) — not a parameter retune
  within the same two mechanisms (more curriculum stages, a wider/narrower
  padded feature) unless there's a specific, evidenced reason to believe a
  parameter was simply mis-set rather than the mechanism itself being
  insufficient.

## Explicit known-weak points (not smoothed over)

- **No source found isolates this project's exact setting** — PPO,
  ground-truth state, flat non-heaped tabletop, parallel-jaw gripper,
  small rigid dice — as its own studied variable (research doc §5).
  DexSinGrasp's curriculum/observation findings are the best available
  grounding, transplanted from a materially different setting (dexterous
  hand, heaped/occluding clutter). Whether *singulation* is even a
  relevant sub-problem here (as opposed to pure target selection with no
  occlusion) is explicitly untested by this spec's own design — Stage
  D1/D2's real result is this project's first real data point on that
  question, not assumed either way going in.
- **A real, project-adjacent counter-data-point already exists and cuts
  the other way**: `dice-pick-demo.md`'s five-die tabletop scene (open,
  non-heaped — closest existing analog to this experiment's own scene)
  already required real disambiguation work even without occlusion (a
  same-location higher-confidence wrong-class detection at seed 123, a
  table-hole false positive needing a geometric-plausibility filter) —
  evidence that "flat, non-heaped, no-occlusion" does not imply "trivial"
  even before considering whether an RL policy's *motor* behavior (not a
  detector's *classification*) can misdirect toward a nearby distractor.
  This is a perception-layer finding from a different mechanism
  (scripted IK + a real detector), not a controlled prediction for this
  RL experiment's own ground-truth-state setting, but it's a reason not
  to be surprised by real difficulty here.
- **No literature found that controls "fixed-size padded distance
  summary" against "per-object one-hot flag" or an attention/pointer
  architecture** — §2b's recommendation rests on DexSinGrasp being the
  one directly verified precedent that made this exact design choice and
  reported it working, not a comparison study.
- **The disjoint-reset-range placement mechanism (see "Distractor
  placement" above) is a spec-time design choice, not yet empirically
  verified against this project's actual table/workspace geometry** — the
  implementing task must run a spawn-and-settle check (no training) before
  trusting it, exactly as flagged above.
- **Whether multiple independent `MultiAssetSpawnerCfg` fields (one per
  scene entity: target, `distractor_1`, `distractor_2`) coexist correctly
  in one `InteractiveSceneCfg` with `scene.replicate_physics = False`** is
  structurally expected to work (each field's spawner is independent) but
  has never been directly exercised in this codebase — an implementation-task
  risk to verify early (a live diagnostic check, matching this project's
  own established `_diag_*` script convention), not assumed silently.

## Success/failure reporting

Per this project's verification standard: real eval videos (not exit
codes) for both shapes at Stage D2 at minimum, instrumented
`max_height_gain`/`max_consecutive_lifted_steps` numbers per shape (not
just the summary discovery-rate fraction, matching this project's own
repeated settle-detection-bug discipline — verify past the JSON, not just
trust it), and explicit before/after comparison against the single-object
8/8 baseline (`model_2998.pt`, this experiment's own starting point) at
every stage, not just the final one. Report Stage SO's own gate result
explicitly even if it passes cleanly (per this project's "report exactly
as observed" convention, not just report the final headline number).

## Related

[[unified-multi-die-specialist-distillation]] (the experiment this spec
extends), the research doc this spec cites throughout
(`docs/superpowers/specs/research/2026-07-19-target-selection-clutter-literature.md`),
[[dice-pick-demo]] (source of the `DiceSceneCfg` multi-object-scene
precedent and the perception-layer disambiguation counter-data-point in
"Explicit known-weak-points").
