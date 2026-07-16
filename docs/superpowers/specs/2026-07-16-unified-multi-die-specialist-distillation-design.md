# Unified multi-shape grasp policy: specialist training + distillation (Experiment 1 of the multi-die RL arc)

## Context

This project's scripted `dice_pick_demo.py` already picks 4/5 die shapes
correctly off a 5-die table (d8/d10/d12/d20; d4 fails) using classical IK,
not RL. On the RL side, `tasks/franka/dice_lift_joint_env_cfg.py`'s
asset-bisect ladder isolated that grasp **discovery** itself (not lift
execution) is gated by object **shape**: a same-scale/mass baked cube
trains reliably (3/3 seeds), a d20 (near-spherical) only discovers a
grasp in 1/3 seeds at the same 48mm/0.216kg, and 0/4 at its real 30.3mm
size. Two independent scale-based fixes — a mixed-size DR curriculum
(`FrankaDieLiftJointMixedEnvCfg`) and a staged size-anneal curriculum
(`FrankaDieLiftJointMidEnvCfg`/`...HeavyEnvCfg`, checkpoint-resumed
48→39.1→30.3mm) — were both independently **FALSIFIED** (0/3 and 1/3
seeds respectively; see `ROADMAP.md`'s 2026-07-13 entries). The
staged-anneal's own failure mode was diagnosed precisely: the *transfer
mechanism* worked (a seed that discovered a grasp at 48mm carried it
undegraded through both anneal stages), but the *base discovery rate* at
48mm (1/3) was the actual bottleneck — no curriculum variant *creates*
new discovery, it only propagates discovery that already happened. Per
that verdict's own next-step ("shape itself needs a new spec/research
pass rather than further object-scale curriculum variants"), this spec is
that pass.

This is also the first RL attack on a **unified multi-shape policy** —
one policy across {d8, d10, d12, d20} — rather than one policy per shape.
d4 is explicitly excluded: it has two independently falsified grasp
mechanisms of its own (opposite-edge grasp blocked at IK reachability;
V-notch fingertip fixture sweeps the die aside with zero engagement) and
no working mechanism exists to build on yet.

## Research grounding

Full research doc: `.superpowers/sdd/research-multi-die-unified-policy.md`
(four parallel literature passes + an independent citation-verification
pass; 8/10 load-bearing citations confirmed exact, one number corrected,
one previously-uncertain citation resolved — see that doc's Review
section). Load-bearing citations for this spec's design choices:

- **Observation encoding**: no paper tests this project's exact scheme
  (flatten + one-hot shape-class at N≤5) directly — a confirmed gap.
  Isaac Lab's own first-party `Stack` task
  (`isaaclab_tasks/manager_based/manipulation/stack/`) is real, shipped,
  first-party precedent that flat per-slot pose concatenation is a
  validated encoding at this engine/scale (though `Stack` has no
  shape-class or target-selection axis, since it manipulates 3 identical
  cubes). Recommendation to use flat concatenation over a
  permutation-invariant/entity encoder here is **by elimination**: every
  citation motivating permutation-invariant encoders (Zambaldi et al.
  2018, Karch et al. 2020, Li et al. ICRA 2020, Haramati et al. ICLR
  2024) targets **variable/growing object-count generalization**, an
  axis this fixed-4/5-shape experiment does not need.
- **Shape-generalization**: this is the weakest-grounded of the research
  doc's four questions — stated plainly there and restated here rather
  than smoothed over. The top-ranked technique, a continuous
  geometry-descriptor observation feature (**Mosbach & Behnke, "Efficient
  Representations of Object Geometry for RL of Interactive Grasping
  Policies," arXiv:2211.10957**, independently verified real), is real
  precedent for the *general idea* that explicit geometric information
  changes an RL policy's ability to explore/discover grasps across
  geometrically distinct objects — it is **not** literature-validated on
  this project's exact binary discovery-rate metric. Using it here is
  this spec's own extrapolation, made explicit.
- **Specialist-then-distill architecture**: **Wan, Geng, Liu, Shan, Yang,
  Yi, Wang, "UniDexGrasp++," ICCV 2023, arXiv:2304.00464**
  (independently re-verified: author order, venue, and its
  generalist-specialist iterative distillation pattern, GiGSL, all
  confirmed against the paper's own text). This is the most directly
  analogous **published, working system** for "one policy, many object
  shapes" found across the whole literature survey (dexterous grasping
  across thousands of shapes, ICCV 2023). This spec adopts GiGSL's
  specialist→distill→iterate structure, not its own GeoCurriculum
  diversity-expansion mechanism (a different, not-adopted part of that
  paper).
- **d20's specific retry — per-episode size-domain-randomization +
  geometry feature**: this deliberately **re-runs a mechanism this
  project already falsified** (`FrankaDieLiftJointMixedEnvCfg`'s mixed-size
  DR, 0/3 seeds) with exactly one new ingredient the original attempt
  never had (the geometry-descriptor feature above). This is a direct
  user decision, made explicitly aware of the precedent — see "Falsifiable
  hypothesis" below for why this isn't blind repetition: the original
  failure's diagnosed mechanism (population dilution with no way for the
  policy to condition on which size it currently faces) is structurally
  different from a policy that receives continuous shape/geometry
  information to condition on.

## Scope

**In scope (this experiment):**
- Shapes: d8, d10, d12, d20. One die per episode (no distractors).
- Ground-truth object-state observations (not vision-detector-derived).
- Lift-only task horizon (no carry-to-goal).
- Three ordered phases: per-shape specialists → distillation → RL
  fine-tune, detailed below.

**Explicitly out of scope, deferred to a future spec (not this one):**
- d4 (no working grasp mechanism exists yet).
- Distractor dice / target-selection among co-present objects
  ("Experiment 2" of this arc) — gated on this experiment succeeding.
  The research doc's strongest single finding (**Xu, Liu, Gui, Guo,
  Jiang, Zhang, Xu, Gao, Shao, "DexSinGrasp," arXiv:2504.04516**,
  independently re-verified exact Table IV numbers: 0% success training
  from scratch with full clutter vs. 94% with a staged single-object-first
  curriculum, at their hardest 8-object configuration) is why: combining
  shape-unification, target-selection, AND full clutter in one experiment
  would make any failure impossible to attribute to a specific cause.
- Vision-detector-derived observations (a separate, later integration
  step once ground-truth discovery itself works).
- Full pick-and-place / carry-to-goal.

## Design

### Observation schema (all phases)

Extends the existing `ObservationsCfg.PolicyCfg` (`tasks/franka/lift_env_cfg.py`)
object-position term with, per present die:

- Position (3) + quaternion (4) — as today.
- One-hot shape-class (4 dims: d8/d10/d12/d20).
- A continuous geometry-descriptor feature (dims TBD by the implementing
  task — e.g. a sphericity/curvature proxy computed from each baked
  asset's known mesh, not a learned encoder; the research doc's Mosbach &
  Behnke citation studies multiple concrete representations and the
  implementing task should pick the simplest one that's actually
  computable from this project's existing baked-mesh pipeline).

No target-flag or distractor-relative-vector terms are needed in this
experiment (single die per episode) — those are Experiment 2's concern,
noted here only so the schema doesn't need to be redesigned when that
experiment starts (the shape one-hot and geometry feature carry forward
unchanged; only a target-flag and distractor terms get added later).

### Phase 1 — per-shape specialists

Train 4 separate PPO policies (`FrankaLiftPPORunnerCfg`-derived, same
`rsl_rl` recipe as every existing lift variant in this file), one per
shape, each at its own real target size:

- **d8, d10, d12**: first-time RL training for these three shapes — they
  have only ever been used in the scripted demo, never trained via PPO.
  Treated as genuinely new territory, not assumed to work by
  cube-shape-analogy. Requires baking physics assets via the existing
  `scripts/bake_die_asset.py --die {d8,d10,d12}` (the script already
  supports all 5 die choices; only `d20_physics.usd` and
  `cube48_physics.usd` exist on disk today) at each shape's real
  commercial "standard" size — derived via the same web-research method
  already used to correct d20's own target size (30.3mm "jumbo" →
  ~20-22mm "standard"; see `ROADMAP.md`'s 2026-07-15 entry) rather than
  assumed. This size-derivation is an explicit Task 0 sub-step, not a
  number this spec invents.
- **d20**: per-episode size-domain-randomized training (reusing
  `MultiAssetSpawnerCfg` with `random_choice=True` this time — note the
  falsified `FrankaDieLiftJointMixedEnvCfg` used `random_choice=False`,
  a deterministic per-env round-robin assignment, not true per-episode
  resampling; the implementing task must verify whether Isaac Lab's
  `MultiAssetSpawnerCfg` actually supports per-*episode*-reset
  resampling or only per-env-at-spawn-time assignment, and report back if
  the mechanism can't do what this spec assumes — flag to Principal
  rather than silently substituting the deterministic mechanism), over a
  range spanning at least 22mm (real standard) to 48mm (the already-tested
  cube-parity size), combined with the geometry-descriptor feature above.

### Phase 1 falsification criterion (d20 specifically)

The d20 retry is falsified if its discovery rate does not clear the
original size-curriculum's 0/3 floor by a meaningful margin — i.e., if
adding the geometry feature to the same population-dilution mechanism
produces the same or worse result, the hypothesis that "missing
shape/geometry conditioning" (not "mixing sizes" per se) explains the
original failure is rejected, and this experiment's Phase 2/3 proceed
without a d20 specialist (or with whatever partial discovery rate Phase 1
actually achieved).

### Phase 2 — distillation

Once each of the 4 specialists reliably discovers+lifts its own shape,
distill them into ONE unified policy (UniDexGrasp++'s GiGSL pattern:
train per-shape specialists, then distill into a single generalist,
iterating) that:
- Takes single-die-per-episode observations with shape randomized across
  resets (the die's *shape* varies episode-to-episode; still zero
  distractors).
- Is conditioned on the same one-hot shape-class + geometry-descriptor
  features as the specialists.
- Learns via imitation/behavior-cloning (or a KL-regularized distillation
  loss — implementing task's choice, per GiGSL's own published mechanism)
  from the 4 frozen specialist "teachers."

### Phase 3 — RL fine-tune

Iterate distillation ↔ PPO fine-tuning on the unified policy (per GiGSL)
until the unified policy's per-shape discovery rate is not meaningfully
below its corresponding specialist's own rate for that shape.

## Falsifiable hypothesis

> (1) A per-episode size-domain-randomized d20 population, conditioned on
> a continuous geometry-descriptor observation feature, will achieve a
> grasp-discovery rate at or above the asset-bisect baseline (≥1/3 seeds
> at 48mm-equivalent difficulty) — succeeding where the same
> population-mixing mechanism without geometry conditioning failed (0/3),
> because the original failure's diagnosed cause (a policy with no way to
> condition its strategy on the object it currently faces) is directly
> addressed by adding that exact conditioning signal, not by the mixing
> itself. (2) A unified policy distilled from 4 per-shape specialists
> (GiGSL, UniDexGrasp++, ICCV 2023 — a published, working system for
> "one policy, many shapes") will, after RL fine-tuning, achieve a
> per-shape discovery rate not meaningfully below each specialist's own
> rate.
>
> Falsified if: (i) the d20 retry's discovery rate does not clear 0/3 by
> a meaningful margin (rejects the geometry-conditioning hypothesis, not
> just the mixing mechanism already known to fail); or (ii) the
> distilled+fine-tuned unified policy's discovery rate for any shape is
> meaningfully below that shape's own specialist rate (rejects
> distillation as sufficient to preserve per-shape discovery in a unified
> policy at this project's scale).

## Explicit known-weak points (not smoothed over)

- The geometry-descriptor-feature choice for Q3 (shape-generalization) is
  this project's own extrapolation from adjacent-but-different-axis
  literature — the research doc rates this as the weakest-grounded of its
  four questions. If Phase 1's d20 retry fails, that failure does not by
  itself indict the geometry-feature idea in general (see hypothesis
  above) — it specifically fails to rescue a population-dilution
  mechanism this project already knows is fragile.
- `MultiAssetSpawnerCfg`'s actual resampling semantics (per-env-fixed vs.
  per-episode-resampled) are not yet confirmed for `random_choice=True`
  in this codebase — this is a concrete implementation-task risk, called
  out explicitly above, not assumed away.
- d8/d10/d12 specialist training is entirely unprecedented in this
  project — no assumption of "cube-like, so it'll just work" should be
  carried into the implementation; each is its own real discovery-rate
  question.

## Success/failure reporting

Per this project's verification standard: real eval videos (not exit
codes), instrumented z-gain/discovery-rate numbers per shape per seed, and
explicit before/after comparison against the asset-bisect baseline (cube
3/3, d20 1/3 at 48mm / 0/4 at 30.3mm) and the falsified size-curriculum
baseline (0/3) for the d20-specific claim.
