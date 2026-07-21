# Target-selection clutter, Stage E1: scaling 2→3 distractors (d12/d20 only)

## Context

[[target-selection-clutter]] (spec:
`docs/superpowers/specs/2026-07-19-target-selection-clutter-design.md`)
finished 2026-07-19: a single unified PPO policy grasps-and-lifts a
commanded d12 or d20 die when 2 other dice (independently drawn from
{d12,d20}) are simultaneously present, at **8/8 for both shapes**
(checkpoint `model_5096.pt`, `gs://rl-manipulation-hks-runs/
target-selection-clutter/joint-die-target-selection-d2/seed42/
2026-07-19_21-08-07/model_5096.pt`), preserving the single-object 8/8
baseline exactly, via a 3-stage distractor-count curriculum (SO: 0 → D1:
1 → D2: 2, each checkpoint-resumed from the prior stage) plus one new,
additive, fixed-size zero-padded observation term
(`distractor_distance_summary`, DexSinGrasp's own `d_t^S` mechanism,
arXiv:2504.04516 §III-A). That experiment's own scope explicitly capped
distractor count at 2 and deferred "3+ distractors... whether the same
curriculum+observation mechanism scales further" to a follow-on.

`docs/superpowers/specs/research/2026-07-21-multi-shape-clutter-extension-
literature.md` (this spec's required Tier-1 research gate, completed
2026-07-21) surveyed extending this toward a 5-die, 4-distractor,
all-shapes-present scene and recommended, instead of one large leap, a
3-stage sequence — **E1 (2→3 distractors, d12/d20 only) → E2 (3→4, same
population) → S1 (fold in d8/d10 only after count-scaling is validated)**
— each checkpoint-resumed from the prior, mirroring this project's own
"isolate one variable at a time" discipline (the exact lesson the original
experiment's own confounded Stage-SO attempt already taught it). **This
spec designs E1 only.** E2 and S1 are named here as the stages that follow
if E1 passes, and are explicitly not designed beyond that naming — each is
its own future spec, gated on this one's own real result, per the research
doc's own §3a staging rationale and per CLAUDE.md's one-variable-at-a-time
Tier-1 discipline.

## Research grounding

Primary citation for this spec's design choices:
`docs/superpowers/specs/research/2026-07-21-multi-shape-clutter-extension-
literature.md` (§1a for the observation/scene-cfg mechanical-extension
precedent, §1b for the scene-topology packing problem and its own
2D-grid suggestion, §1c for the shape-coverage ledger behind excluding
d8/d10/d4, §3a for the staged E1→E2→S1 sequence and its own falsification
framing). That document itself builds on, and does not re-litigate,
`docs/superpowers/specs/research/2026-07-19-target-selection-clutter-
literature.md`'s independently-re-verified DexSinGrasp citation
(arXiv:2504.04516) — E1 does not introduce any new literature-grounded
mechanism beyond what the finished D2 experiment already validated; it
tests whether that same, already-validated mechanism (curriculum +
fixed-size zero-padded distance observation, reward/target-identification
unchanged) continues to hold at one more distractor.

One correction made independently while grounding this spec directly
against the code (not carried over from the research doc uncorrected,
per this project's citation-verification practice of checking a claim
against its actual source before propagating it): the 2026-07-21 research
doc's §1a states extending "the finished D2 checkpoint's 43-dim schema to
a 45-dim (K=3)... schema." Recomputing directly from
`tasks/franka/lift_env_cfg.py`'s own `ObservationsCfg.PolicyCfg` (41 base
dims: `joint_pos_rel`(9) + `joint_vel_rel`(9) + `object_position`(3) +
`target_object_position`(7) + `last_action`(8) + `shape_class_onehot`(4) +
`geometry_descriptor`(1) = 41) plus exactly **one scalar per distractor
slot** (confirmed directly against `distractor_distance_summary`'s own
code: K=2 → 43 dims, i.e. 41+2, per `TargetSelectionObservationsCfg`'s own
docstring, "grows the observation space by exactly 2 dims (41 -> 43)"),
**K=3 is 44 dims (41+3), not 45.** This spec uses 44 throughout; if a
future E2/S1 spec reuses the research doc's own "47-dim (K=4)" figure
uncorrected, it should independently re-derive it the same way (41+4=45,
not 47) rather than propagate this same off-by-one-per-slot arithmetic
error.

## Falsifiable hypothesis

> Starting from the finished, checkpointed D2 policy (`model_5096.pt`,
> 8/8 both shapes at 2 active distractors), extending it with (a) one new
> distractor scene entity (`distractor_3`, drawn from {d12, d20} exactly
> like `distractor_1`/`distractor_2`), (b) the mechanical K=2→K=3
> extension of `distractor_distance_summary` (41+3=44 dims, one new
> zero-padded distance scalar), and (c) a real placement-topology change
> (the existing 3-lane y-strip is already at its established
> `CommandsCfg` workspace boundary and cannot host a 4th disjoint region —
> see "Scene layout" below), while training with all 3 distractors
> **simultaneously active from the start of E1's own training** (no
> further 0/1/2-active sub-curriculum within E1 itself — the count
> curriculum's 0→1→2 steps are already validated by the finished D2
> experiment; E1 tests the NEXT increment, 2→3, in one step, checkpoint-
> resumed from D2 exactly as D1 resumed from SO and D2 from D1), will
> preserve **most of** Stage D2's own 8/8-both-shapes discovery rate for
> d12 and d20 under 3-distractor clutter — matching the original
> experiment's own D2 falsification bar's own explicit allowance for some
> real degradation from a clean 8/8, not a demand for a literal, unbroken
> 100%.

**Falsification condition (numeric, pre-registered, directly templated on
the original D2 spec's own primary bar):**

- **Primary bar:** evaluated with `franka_checkpoint_review.py`'s existing
  instrumented lift-threshold convention (`num_envs=8`, full 4-entity E1
  scene active, target shape fixed per eval run — a new
  `--variant joint-die-target-selection-e1` mapped to new `_PLAY_D12Target`/
  `_PLAY_D20Target` env cfg classes, the same mechanical addition every
  prior variant in `dice_lift_joint_env_cfg.py`/
  `franka_checkpoint_review.py` already required), if EITHER shape's
  sustained-lift discovery rate falls **below 6/8 (75%)**, that shape's
  result falsifies "the count-curriculum + fixed-size zero-padded
  observation mechanism, already validated through K=2, continues to
  scale to K=3." 6/8 is reused unchanged from the original D2 bar, for the
  same reasons that bar was chosen there (clearly above DexSinGrasp's own
  reported 0% no-curriculum-collapse floor, clearly above this project's
  own historical "partial/fragile" discovery signatures, while still
  allowing real degradation from a clean 8/8).
- **Internal pre-training gate (blocking, before any training spend):**
  the 43→44-dim weight-surgery extension (see "Checkpoint warm-start"
  below) must show **exactly 0.0 max abs diff, both actor and critic
  branches**, via `scripts/extend_checkpoint_observation_dims.py --verify`
  run against the REAL `model_5096.pt`, once locally before any cloud/
  desktop dispatch and once again immediately before training starts on
  the actual training host — matching the original experiment's own
  double-verification protocol for its 41→43-dim extension exactly. This
  is a **mechanical-correctness check** (does the new K=3 pure function's
  first two output columns match the old K=2 function's two columns in
  the same order, and did the surgery script append rather than
  interleave the new column), not a behavioral/discovery-rate check —
  unlike the original SO gate, no training budget needs to be spent to
  answer this, because the "new column always multiplies its weight by
  the value fed to it" linear-algebra property `--verify` checks is
  unconditional (see "Why this gate differs from Stage SO's" below). If
  this does not show exactly 0.0, STOP — do not train — the mechanical
  extension itself has a bug (most likely a column-order mismatch between
  the K=2 and K=3 pure functions) that must be found and fixed first.
- **Escalation on falsification of the primary bar:** per the research
  doc's own §3a framing, if Stage E1 collapses for d12/d20 specifically
  while the internal pre-training gate passed cleanly, that falsifies "the
  existing count-curriculum + fixed-size zero-padded observation mechanism
  scales past K=2" specifically — DexSinGrasp's own Table IV
  (arXiv:2504.04516, already independently re-verified per the 2026-07-19
  research doc) shows an arrangement-dependent, non-monotonic collapse can
  occur even within a single paper's own distractor-count range, so this
  is a real, literature-grounded possible outcome, not just caution. The
  honest escalation path is a richer/architecturally-different observation
  mechanism (Deep Sets/attention over distractor state, per the 2026-07-19
  research doc's §2d), not another parameter retune (more iterations, a
  different placement grid) unless there is a specific, evidenced reason
  to believe a parameter was simply mis-set. E2/S1 should not proceed if
  E1 falsifies — per the research doc's own staging, E1 failing means the
  count-scaling axis itself, not just this specific increment, is in
  doubt.

## Why this gate differs from Stage SO's

Stage SO's original 41→43-dim weight-surgery warm start was verified
**behaviorally lossless at t=0** specifically because Stage SO's own
0-active-distractor topology hard-zeroed both new observation columns —
so the extended network was mathematically guaranteed to reproduce
`model_2998.pt`'s exact original output on every real Stage-SO input, not
just on the synthetic all-zero batch `--verify` uses to check it.

**E1's situation is structurally different, and this spec does not claim
the same t=0 behavioral guarantee for E1.** `distractor_3` is real and
active from the very first step of E1's own training — there is no
"parked" sub-phase for it within E1 (the 0→1→2 count curriculum is
already-validated territory; E1 tests the 2→3 increment directly). This
means:

- The new 44th observation column carries a real, immediately nonzero,
  per-episode-varying value (target-to-distractor_3 distance) from
  iteration 0 onward — its randomly-initialized network weight will
  inject real (if initially uninformed) signal into the very first
  forward passes, unlike Stage SO's provably-inert extension.
- Relocating `distractor_1`/`distractor_2` into the new grid layout (see
  "Scene layout" below) also shifts the **distribution** of the two
  already-learned distance columns away from what `model_5096.pt` was
  actually trained on (D2's y-strip lane centers ~185mm apart vs. E1's
  grid cells' ~70-190mm range) — a second, independent reason this is not
  a zero-cost resume, on top of the new column.

`--verify`'s own mathematical guarantee (if a column is fed exactly 0.0,
its weight contributes exactly 0.0 to that layer's pre-activation,
regardless of what real values the OTHER columns carry) is unconditional
and still holds — it is a pure linear-algebra fact about how a Linear
layer's new column behaves when zeroed, independent of what distribution
of values the network was originally trained under. That is why it is
still the right tool to reuse directly (not redesigned) and still
provides a real, blocking, pre-training correctness check. But it proves
the *mechanical extension* is correct, not that E1 is a free/no-op resume
— which is exactly why E1 gets a real iteration budget (see "Training
approach" below), analogous to how Stage D1's own real, immediately-active
`distractor_1` needed genuine adaptation iterations rather than being
treated as a no-op, even though D1 introduced no new dimensionality at
all.

## Grounding in the actual codebase — what's new vs. reused

### Observation term: mechanical K=2 → K=3 extension, additive not destructive

Read directly (not assumed) from the current code:

- `tasks/franka/distractor_observations.py::distractor_distance_summary`
  (lines 55-77) takes `target_pos`, `distractor_1_pos`, `distractor_2_pos`
  as three named positional args and returns a hardcoded `(num_envs, 2)`
  tensor via two `if active_distractor_count >= {1,2}` branches.
- `tasks/franka/mdp.py::distractor_distance_summary` (lines 176-206) reads
  exactly `env.scene["distractor_1"]`/`env.scene["distractor_2"]` by
  literal string key.
- `tasks/franka/lift_env_cfg.py::TargetSelectionObservationsCfg` (lines
  227-259) is its own class specifically so this term doesn't get added to
  the shared `ObservationsCfg` base and break every other, non-clutter env
  cfg in this repo (its own docstring states this explicitly).

**Design decision: do NOT edit these three existing pieces in place.**
Editing `distractor_distance_summary`'s signature/shape in place would
silently change `TargetSelectionObservationsCfg`'s own dimensionality
(currently 43-dim, used unchanged by the already-finished, already-
checkpointed SO/D1/D2 env cfgs) out from under them — the same "don't
mutate a shared base class whose old callers depend on the old shape"
reasoning `TargetSelectionObservationsCfg`'s own docstring already used to
justify not touching the base `ObservationsCfg`. Instead, add new,
parallel, K=3 siblings, following the exact "distractor_1/distractor_2 are
already near-duplicates of each other" copy-paste pattern the 2026-07-21
research doc's §1a identified as this project's own precedent for this
exact kind of extension:

- **`tasks/franka/distractor_observations.py`**: add
  `distractor_distance_summary_3(target_pos, distractor_1_pos,
  distractor_2_pos, distractor_3_pos, active_distractor_count) ->
  torch.Tensor` — a `(num_envs, 3)` tensor, same hard-zero-per-slot-index
  convention (`if active_distractor_count >= {1,2,3}`), columns 0/1
  computed identically (same Euclidean-norm formula, same argument order)
  to the existing K=2 function's own columns 0/1 — this identity is
  load-bearing for the checkpoint warm-start (see below). The existing
  `distractor_distance_summary` (K=2) is left completely untouched.
- **`tasks/franka/mdp.py`**: add a new wrapper
  `distractor_distance_summary_3(env) -> torch.Tensor` reading
  `env.scene["object"]`/`["distractor_1"]`/`["distractor_2"]`/
  `["distractor_3"]` and `env.cfg.active_distractor_count`, delegating to
  the new pure function above — same idiom as the existing wrapper, one
  more scene-entity lookup. The existing `distractor_distance_summary`
  wrapper is left completely untouched.
- **`tasks/franka/lift_env_cfg.py`**: add
  `TargetSelectionE1ObservationsCfg(ObservationsCfg)` — subclassing the
  41-dim base directly (matching how `TargetSelectionObservationsCfg`
  itself subclasses `ObservationsCfg`, not some hypothetical
  single-distractor intermediate class), with one new field,
  `distractor_distance_summary = ObsTerm(func=mdp.distractor_distance_summary_3)`.
  `@configclass` inheritance places this one new field after all 41
  inherited ones (same reasoning `extend_checkpoint_observation_dims.py`'s
  own docstring already relies on for the K=2 case), giving 41+3=44 dims,
  with columns 0-40 identical in meaning/order to D2's own schema and
  columns 41-42 (target-to-distractor_1/2 distance) computed identically
  to D2's own columns 41-42 — only column 43 (target-to-distractor_3
  distance) is genuinely new. The existing `TargetSelectionObservationsCfg`
  (K=2, 43-dim) is left completely untouched, still used unchanged by
  SO/D1/D2 and their `_PLAY` eval variants.

### Scene cfg: a 3rd `RigidObjectCfg` slot, added without touching the finished stages' own scene cfg

`FrankaDieLiftTargetSelectionSceneCfg` (`tasks/franka/dice_lift_joint_env_
cfg.py:1377-1426`) declares exactly `distractor_1`/`distractor_2` as
sibling `RigidObjectCfg` fields, each defaulting to a PARKED placeholder
(off-workspace, zero-width pose_range) that IS Stage SO's own real
topology. **Design decision:** add a new subclass,
`FrankaDieLiftTargetSelectionE1SceneCfg(FrankaDieLiftTargetSelectionSceneCfg)`,
adding one new field, `distractor_3`, following `distractor_1`/
`distractor_2`'s exact pattern (same `_D12_USD`/scale/rigid_props/mass
placeholder, its own PARKED default position, e.g.
`_PARKED_DISTRACTOR_3_POS = (0.5, 0.0, -0.9)` — distinct x/y from the
existing two parked spots so no two parked bodies spawn exactly
coincident, matching the existing rationale for why
`_PARKED_DISTRACTOR_1_POS`/`_PARKED_DISTRACTOR_2_POS` already differ from
each other). **Do not add `distractor_3` to the base
`FrankaDieLiftTargetSelectionSceneCfg` itself** — that class is still used
unchanged by the finished SO/D1/D2 env cfgs; adding a field there would be
harmless to their physics (an inert extra parked body) but is an
unnecessary, avoidable coupling between a finished, checkpointed
experiment's own scene topology and this new one's, with no benefit.

Similarly, add `TargetSelectionE1EventCfg(TargetSelectionEventCfg)` with
one new field, `reset_distractor_3_position` (an `EventTerm` using
`mdp.reset_root_state_uniform`, same as the two existing
`reset_distractor_{1,2}_position` terms, PARKED zero-width `pose_range`
default, `SceneEntityCfg("distractor_3", body_names="Distractor3")`) —
again, a new subclass, not a mutation of the base `TargetSelectionEventCfg`
still used unchanged by SO/D1/D2.

### Scene layout — a real redesign, needed at K=3 already, not deferred to K=4/K=5

The 2026-07-21 research doc's §1b already established the existing 3-lane
y-strip design (target y∈[-0.05,0.05], distractor_1 y∈[-0.25,-0.12],
distractor_2 y∈[0.12,0.25], all sharing x∈[0.4,0.6]) **already spans the
full `CommandsCfg` `pos_y=(-0.25,0.25)` range edge-to-edge with no slack
left over** — distractor_1's outer edge sits exactly on -0.25, distractor_2's
exactly on +0.25. Re-checking this directly against E1's own specific
need (going from 3 total co-present entities to 4, not yet 5): **there is
no free y-band left to add a 4th 1D lane without shrinking an existing
lane's spread or its buffer gap below the values the original D2 topology
already validated (a 70mm edge gap, empirically measured at 95mm real
minimum separation against a 60mm safety floor).** This means the
2D-grid alternative the research doc flagged as "worth flagging for the
next experiment's own scene-topology task" is not an optional nicety
deferred to E2/S1's larger 5-object case — it is **required at E1
specifically**, since even one additional distractor already exceeds the
existing 1D layout's capacity.

**Design: a 2×2 grid using both the x-dimension (200mm, `pos_x=(0.4,0.6)`,
currently unused for lane differentiation — every existing lane shares the
same x range) and the y-dimension (500mm, `pos_y=(-0.25,0.25)`) —
2 rows split on x, 2 columns split on y, exactly 4 cells for E1's exactly
4 co-present entities (1 target + 3 distractors):**

| Cell | Row (x) | Column (y) | `init_state.pos` center (x, y, z=0.055) | `pose_range` (x, y) | Assigned entity |
|---|---|---|---|---|---|
| A | near | left  | (0.43, -0.125) | x:(-0.03,0.03), y:(-0.09,0.09) | `object` (target) |
| B | near | right | (0.43, +0.125) | x:(-0.03,0.03), y:(-0.09,0.09) | `distractor_1` |
| C | far  | left  | (0.57, -0.125) | x:(-0.03,0.03), y:(-0.09,0.09) | `distractor_2` |
| D | far  | right | (0.57, +0.125) | x:(-0.03,0.03), y:(-0.09,0.09) | `distractor_3` |

(Cell-to-entity assignment is a labeling convenience only — target
identity is structural, via `scene["object"]` being a physically distinct
`RigidObjectCfg`, exactly as the original D2 design already established;
which grid cell each entity's lane center sits at has no effect on which
one is "the target.")

Resulting per-axis ranges: near row x∈[0.40,0.46] (touching the
`CommandsCfg` lower x-bound exactly, mirroring D2's own precedent of
touching a boundary exactly along one axis — its y-lanes did the same);
far row x∈[0.54,0.60] (touching the upper bound exactly); left column
y∈[-0.215,-0.035]; right column y∈[0.035,0.215] (each column's margin to
the ±0.25 y-bound is 35mm).

**Worst-case nominal (reset-range-only) pairwise minimum center-to-center
separation, computed the same way the original diagnostic script does
(raw Euclidean distance between draws from each pair of ranges, not
edge-to-edge lane gap):**

- Same-row pairs (A-B, C-D): differ only in y → min separation = 70mm
  (0.035 - (-0.035)) — an **exact match** to the original D2 design's own
  70mm target↔distractor gap, which that design's own live diagnostic
  measured at 95mm real minimum separation against a 60mm safety floor.
- Same-column pairs (A-C, B-D): differ only in x → min separation = 80mm
  (0.54 - 0.46) — **more generous** than the 70mm case.
- Diagonal pairs (A-D, B-C): differ in both axes → min separation =
  sqrt(80²+70²) ≈ 106mm, strictly larger than either non-diagonal case
  (no new binding constraint).

The binding worst case (70mm) equals the original design's own
already-validated gap, not a narrower one — this grid is at least as
conservative as the design the original live diagnostic already confirmed
safe. **This is still a design-time estimate for a genuinely new
topology, not a re-use of an already-measured one** — per this project's
own established practice (`scripts/_diag_target_selection_clutter_scene_
check.py`), the implementing task must build an equivalent live,
no-training-spend spawn-and-settle diagnostic for this specific 4-cell
grid (extending `_diag_target_selection_clutter_scene_check.py`'s exact
pattern: build the real E1 env at small `num_envs`, step physics to
settle, verify per-env (a) no entity ends up outside the reachable
workspace and (b) no two entities' live post-settle positions fall below
a 60mm minimum separation) **before** any training spend — this design's
70mm/80mm/106mm nominal reset-range separations are the same kind of
design-time arithmetic estimate the original spec's own 70mm-gap number
was, not yet an empirical confirmation.

**Implementation note carried over unchanged from the existing module-
level comment block in `dice_lift_joint_env_cfg.py`:** `pose_range`'s
x/y/z keys are sampled as an OFFSET added to the entity's own
`init_state.pos`, not an absolute world-frame range (confirmed directly
against `isaaclab.envs.mdp.events.reset_root_state_uniform`). Because the
grid moves `object`'s own lane center too (from its pre-existing default
(0.5, 0.0) to Cell A's (0.43, -0.125)), the E1 env cfg's `__post_init__`
must override BOTH `scene.object.init_state.pos` AND
`events.reset_object_position.params["pose_range"]` together (not just
narrow the pose_range as D2 did, since D2 never moved the target's own
lane center) — the same "changing only one half silently produces the
wrong region" trap the existing comment block already warns about for the
distractor lanes, now applying to the target lane too since E1, unlike
SO/D1/D2, relocates it.

### Checkpoint warm-start: reuse `extend_checkpoint_observation_dims.py` directly, not rebuilt

Per the original D2 experiment's own precedent (Task 4, `kb/wiki/
experiments/target-selection-clutter.md`), a checkpoint whose first-layer
weight shape doesn't match the live env's observation width fails a hard,
strict `load_state_dict` — there is no cross-dimensionality resume
mechanism anywhere in this repo's rsl_rl install. `scripts/
extend_checkpoint_observation_dims.py` already solves exactly this: finds
the real `actor.0`/`critic.0` first-Linear-layer keys by pattern, extends
each from `(256, N_old)` to `(256, N_new)` by copying the existing N_old
columns unchanged and appending `N_new - N_old` freshly-initialized
columns (a throwaway `nn.Linear(N_new, 256)`'s own default
`kaiming_uniform_` init, so the new columns' scale matches what a real
from-scratch layer of the new width would get), writes the
`{"model_state_dict", "optimizer_state_dict": {}, "iter", "infos"}`
policy-only-checkpoint format, and has a `--verify` flag that builds
both the old and new networks as plain `nn.Sequential` MLPs and checks
their outputs match on a batch with the new columns zeroed. **This script
is reused verbatim for E1, not redesigned** — its own docstring already
states "not K=2-specific... works for any column-count increase."

Concrete invocation for E1 (illustrative flags, matching the script's own
documented usage):

```bash
python3 scripts/extend_checkpoint_observation_dims.py \
    --input model_5096.pt \
    --output model_5096_44dim.pt \
    --old-obs-dim 43 --new-obs-dim 44 --seed 42 --verify
```

Run once locally (a throwaway CPU-only venv, no Isaac Sim/GPU required,
matching how this same script was already run on the Pi host for the
original SO warm start) before any cloud/desktop dispatch, and once again
on the actual training host immediately before launching
`train_franka.py`, exactly matching the original experiment's own
double-verification protocol. Then:

```bash
train_franka.py --variant joint-die-target-selection-e1 \
    --checkpoint model_5096_44dim.pt --policy_only_checkpoint \
    --max_iterations <5096 + budget>
```

`--policy_only_checkpoint` is required (not optional) here, for the same
reason it was required at the original Stage SO: the first-layer weight
shape changed, so the old run's Adam optimizer moment buffers for that
layer are shape-incompatible too — a fresh PPO optimizer must be built,
exactly matching the SO precedent. `--max_iterations` must be the
ABSOLUTE target (`train_franka.py`'s own convention, preserving
`model_5096.pt`'s real recorded `iter=5096`), not the additional-iteration
count alone.

### Training approach: single resume, real iteration budget (not a no-op)

Per "Why this gate differs from Stage SO's" above, E1's warm start is not
behaviorally free the way SO's was — `distractor_3` is real/active from
iteration 0, and `distractor_1`/`distractor_2`'s own relocated lane
centers shift the distance-value distribution the already-learned network
sees. **Recommended starting budget: 1000 additional iterations**
(`--max_iterations 6096`), matching Stage D2's own reasoning for
introducing "a second simultaneous distractor" (a real, bounded increment
over an already-working base skill, not a new skill from scratch) — E1's
own increment (a genuinely new observation dimension AND a third
simultaneous distractor AND a full topology relocation) is judged
comparable to or somewhat larger than D2's own 2nd-distractor increment,
so D2's own 1000-iteration budget (not D1's smaller 801) is the closer
match. **This is a starting point for the implementing task's own
judgment, not a rigid mandate** — per this project's own established
practice at every prior stage (SO/D1/D2 all had their actual budget
informed by watching the live streamed reward curve, not fixed purely at
spec-time), the implementing task should watch
`Episode_Reward/lifting_object`/`Episode_Reward/object_goal_tracking`/
`Episode_Termination/object_dropping` during the run and extend the budget
if the curve is still climbing meaningfully as 1000 iterations approaches,
exactly as the original experiment's own task write-ups already did.

`num_envs=4096` (matching every prior stage), single seed (`seed42`,
matching the checkpoint this experiment starts from — multi-seed
replication remains explicitly deferred, unchanged from the original
experiment's own deferral, which this spec does not revisit).

### Reward and termination — unchanged, per the original experiment's own reasoning

`RewardsCfg`/`TerminationsCfg` (`object_dropping` scoped to
`scene["object"]` only) are inherited byte-identical, exactly as the
original D2 design left them unchanged from `FrankaDieLiftJointD12D20MixedEnvCfg`.
No new distractor-avoidance/disturbance reward term is added in E1, for
the same reason the original spec gave: adding a new reward term in the
same experiment as a topology/observation change would make any failure
impossible to attribute to a specific cause. A distractor-avoidance reward
term remains the pre-registered fallback only if the curriculum+
observation-only mechanism is falsified (see "Falsifiable hypothesis"
above), not a parallel mechanism tested alongside this one.

### Eval tooling — the same mechanical addition every prior variant required

`franka_checkpoint_review.py`'s `--variant` dispatch needs two new mapped
strings (`joint-die-target-selection-e1-d12`/`-d20`, illustrative), each
constructing a new `_PLAY` env cfg
(`FrankaDieLiftJointD12D20TargetSelectionE1EnvCfg_PLAY_D12Target`/
`_PLAY_D20Target`, `num_envs=8`, target shape pinned, distractor slots
left un-pinned exactly as D1/D2's own `_PLAY` variants already do) — the
same mechanical addition every prior variant already required, not a
redesign of the script. Success is still scored purely on whether the
target (`scene["object"]`) gets lifted and sustained, using the existing
`max_height_gain`/`max_consecutive_lifted_steps` instrumentation
unchanged.

### Execution backend

Per CLAUDE.md's desktop-first/cloud-fallback GPU routing: dispatch via
`scripts/check_gpu_availability.sh` → `run_on_desktop_gpu.sh` if
AVAILABLE, cloud fallback (`docs/cloud/dispatch-checklist.md`'s recipe)
otherwise — matching every stage in this arc. Non-headless when local/
desktop (CLAUDE.md's standing instruction), headless only if it falls
back to cloud. Wrap any Isaac-Sim-touching invocation with
`flock -o /tmp/rl_isaac_sim.lock`.

## Scope

### In scope for E1

- **Two shapes only: d12 and d20**, for target, `distractor_1`,
  `distractor_2`, AND `distractor_3` alike — unchanged from D2's own
  population.
- **Distractor count: exactly 3 active, trained in one resume from D2's
  own 2-active checkpoint.** No further 0/1/2/3 sub-curriculum within E1
  itself (that ground is already validated by the finished SO/D1/D2
  stages) — this is the single new 2→3 increment, checkpoint-resumed
  exactly as each prior stage resumed from the one before it.
- **The 2×2 grid scene-layout redesign** described above, replacing (not
  extending) the existing y-strip layout for all 4 co-present entities
  (target + 3 distractors).
- **The K=2→K=3 mechanical extension** of the observation term, scene cfg,
  and event cfg, as new sibling classes/functions alongside (not replacing)
  the existing K=2 versions still used by the finished SO/D1/D2 stages.
- **The `extend_checkpoint_observation_dims.py` 43→44-dim weight-surgery
  warm start**, reused directly, with its own `--verify` flag as a
  blocking pre-training correctness gate.
- Flat, non-overlapping tabletop placement only (grid cells, not a heap/
  piled/occluding arrangement) — unchanged in kind from the original
  design's own scope.
- Ground-truth object-state observations only — no vision detector,
  unchanged.
- Single seed (seed42).

### Explicitly out of scope for E1 specifically (not just "eventually")

- **d4 — excluded, not deferred for lack of time.** Per the 2026-07-21
  research doc's §2, d4's documented failures (Rung 0 blocked by hard
  PhysX-level geometric infeasibility; Rung 1 blocked first by a
  perception gap, then by a purpose-built V-notch fixture sweeping the die
  aside without ever engaging it even under ground-truth-position
  bypass — "the notch swept the die aside without ever engaging it") are
  **grasp-mechanism/geometric-affordance failures upstream of anything a
  curriculum or distractor-count mechanism can address**, not exploration/
  discovery failures of the kind this project's curriculum/warm-start
  toolkit has repeatedly solved for other shapes. Including d4 in a
  distractor-count-scaling experiment would test nothing about count
  scaling for d4 (it fails before clutter is even relevant) while
  confounding any real d12/d20 result with an unrelated, near-certain
  d4 failure — precisely the Stage-SO-confound mistake this project's own
  history already taught it to avoid.
- **d8/d10 — excluded from E1, reserved for Stage S1 only.** Per the
  research doc's own §1c ledger: d8/d10 have never been tested with ANY
  distractor present, in either role, and their own single-object success
  is itself conditional on a specific, mechanistically different fix
  (cross-shape checkpoint warm-start from a converged d12 specialist, per
  [[d8-d10-demo-warmstart]]'s H2 result) — not curriculum, not
  from-scratch discovery, the mechanism this exact E1/E2 sequence tests.
  Folding d8/d10 into E1 would stack two independently-validated but
  mechanistically distinct fixes in one experiment, making any failure
  impossible to attribute cleanly — the research doc's own staged
  sequence exists specifically to avoid this by isolating count-scaling
  (E1→E2) from shape-introduction (S1).
- **E2 (3→4 distractors, same d12/d20 population) and S1 (fold in d8/d10)
  are named here only as the stages that follow E1 if it passes.** Neither
  is designed in this spec beyond that naming — each is gated on this
  spec's own real E1 result (per the research doc's own §3a staging) and
  is a future spec's job, not this one's.
- **3+ distractors beyond exactly 3, heaped/occluding arrangements, or any
  singulation mechanism** — unchanged from the original design's own
  out-of-scope list; this spec only adds one more distractor to an
  already-flat, non-overlapping topology.
- **A distractor-avoidance/disturbance reward term** — only the
  pre-registered fallback on falsification, not attempted here.
- **Multi-seed replication** — remains deferred, unchanged from the
  original experiment's own deferral.

## Explicit known-weak points (not smoothed over)

- **The 2×2 grid layout is a spec-time design-time estimate for a
  genuinely new topology, not yet empirically verified against real
  physics settle behavior.** Its nominal reset-range separations (70mm/
  80mm/106mm) match or exceed the original D2 design's own already-
  measured 70mm-gap/95mm-actual precedent, but this is reasoning by
  analogy, not a re-use of an actual measurement on this specific 4-cell
  grid. A live spawn-and-settle diagnostic (extending
  `_diag_target_selection_clutter_scene_check.py`'s exact pattern) is
  required before any training spend, per this project's own established
  practice — this spec does not treat the grid numbers as trustworthy
  until that diagnostic passes.
- **E1's checkpoint warm-start is NOT bit-for-bit behaviorally lossless at
  t=0**, unlike Stage SO's — see "Why this gate differs from Stage SO's"
  above for the full reasoning. The `--verify` pre-training gate proves
  the mechanical extension is correct, not that E1 starts from
  unperturbed D2 behavior; real adaptation iterations are required and
  budgeted for, not assumed away.
- **No literature source validates a 2D-grid distractor-placement topology
  specifically** — the 2026-07-21 research doc flagged the idea (using
  the unused x-dimension) but did not resolve or validate specific cell
  geometry; this spec's exact grid numbers are this project's own
  engineering judgment call responding to that suggestion, checked against
  this project's own established 60mm-separation/70mm-gap safety
  conventions, not a literature-grounded claim in themselves. This is
  consistent with the original spec's own similar disjoint-lane placement
  choice also being "a spec-time design choice, not yet empirically
  verified" rather than a literature citation.
- **The corrected 44-dim (not 45-dim) figure** — see "Research grounding"
  above. Verified directly against the code in this pass; flagged
  explicitly so a future E2/S1 spec author re-derives its own K=4 figure
  (45, not the research doc's own uncorrected 47) rather than propagate
  the same arithmetic error forward.
- **Relocating `distractor_1`/`distractor_2`'s own lane centers changes
  the distance-value distribution the D2 checkpoint was actually trained
  on**, on top of the new K=3 column — a second, independent reason E1 is
  a real-adaptation resume, not a low-risk drop-in. Not measured/
  quantified in this spec; the primary falsification bar (trained,
  evaluated discovery rate) is the intended way this risk actually gets
  tested, not a separate analysis.

## Success/failure reporting

Per this project's verification standard: real eval videos (not exit
codes) for both shapes at E1's own final checkpoint, instrumented
`max_height_gain`/`max_consecutive_lifted_steps` numbers per shape (not
just the summary discovery-rate fraction), and an explicit before/after
comparison against Stage D2's own 8/8-both-shapes baseline — report the
internal pre-training `--verify` gate's result explicitly even though it
is expected to pass cleanly (matching this project's "report exactly as
observed" convention), and report the final discovery-rate numbers
exactly as observed, not just the pass/fail verdict against the 6/8 bar.
Explicitly check (per D2's own precedent) for a "grasped the wrong die"
episode across every inspected frame, since this is a new, potentially
newly-possible failure mode each time distractor count increases.

## Cost

No cost estimate exists yet for a 4-entity/K=3 stage specifically (the
original 3-stage SO/D1/D2 sequence cost ≈$1.35 combined, well under its
own $5 cap). **Proposed cap for E1 alone: $2** — a judgment call, roughly
matching the original combined cap divided across its 3 stages
($5/3≈$1.67) rounded up slightly for E1 folding a topology redesign,
dimensionality bump, and 3rd distractor into one single training run
rather than 3 separately-budgeted stages. Report to the controller if this
cap is exceeded; otherwise the existing "well under, no notification
needed" convention applies. Budget dispatch time accounting for this
project's documented recurring frictions (SPOT preemption, the
project-wide `GPUS_ALL_REGIONS=1` cloud quota contention both prior
stages hit) rather than assuming a clean, uncontended run.

## Related

[[target-selection-clutter]] (the finished experiment E1 extends; source
of the K=2 observation term, curriculum precedent, and `model_5096.pt`),
`docs/superpowers/specs/2026-07-19-target-selection-clutter-design.md`
(the original design this spec's format/rigor directly templates),
`docs/superpowers/specs/research/2026-07-21-multi-shape-clutter-extension-
literature.md` (this spec's required Tier-1 research gate, source of the
E1→E2→S1 staging and the d4/d8/d10 exclusion reasoning),
`docs/superpowers/specs/research/2026-07-19-target-selection-clutter-
literature.md` (source of the original, independently-re-verified
DexSinGrasp citation this spec's mechanism ultimately rests on),
[[d8-d10-demo-warmstart]] (source of the H2 cross-shape warm-start
mechanism this spec's own d8/d10 exclusion reasoning cites).
