# Research: extending target-selection-clutter to 5 shapes / 4 distractors — feasibility, scope, and hypothesis for the next follow-on

**Date:** 2026-07-21
**Author:** Senior research thread (delegated by Principal)
**Purpose:** Tier 1 hypothesis-gate research for a future spec that would
extend the completed 2-distractor, 2-shape (d12/d20)
[[target-selection-clutter]] experiment to a 5-die scene (all of
d4/d8/d10/d12/d20 present, 4 active distractors) with sequential commanded
picks. Per CLAUDE.md's scientific-method gate, this document must exist and
be cited before that follow-on spec is written. **This is research only — no
env cfg code, no reward-term design, no Isaac Sim launches, no training.**

---

## 1. What's actually known vs. unknown, checked directly against the code

### 1a. `distractor_distance_summary` — the K=2 observation term does NOT architecturally generalize to K=4 as-is; it needs a real (if mechanical) code extension at three levels

Read directly, not assumed from the design doc's prose:

- `tasks/franka/distractor_observations.py::distractor_distance_summary` takes
  `target_pos`, `distractor_1_pos`, `distractor_2_pos` as three **named
  positional arguments** (not a list/tensor stack of arbitrary length) and
  returns a hardcoded `(num_envs, 2)` tensor built via two `if
  active_distractor_count >= {1,2}` branches. There is no loop over a
  variable-length distractor set anywhere in this function.
- `tasks/franka/mdp.py::distractor_distance_summary` (the thin env-facing
  wrapper, lines 176-206) is equally hardcoded: it reads exactly
  `env.scene["distractor_1"]`/`env.scene["distractor_2"]` by literal string
  key, nothing more general.
- `tasks/franka/dice_lift_joint_env_cfg.py::FrankaDieLiftTargetSelectionSceneCfg`
  declares exactly two `RigidObjectCfg` fields (`distractor_1`,
  `distractor_2`), and `TargetSelectionEventCfg` declares exactly two
  matching `reset_distractor_{1,2}_position` event terms. Both are real
  configclass fields, not something built by iterating a count.

**Conclusion: the underlying *design pattern* (fixed-size, hard-zero-padded
aggregate — DexSinGrasp's `d_t^S`, per
`docs/superpowers/specs/research/2026-07-19-target-selection-clutter-literature.md`
§2b, already verified real and cited there) generalizes cleanly to K=4 in
principle — the paper's own reference setting pads up to 8. But this
project's own *implementation* of that pattern is not parameterized by K at
all; extending to K=4 means literally adding two more named
`RigidObjectCfg`/event-term/pure-function-argument slots (`distractor_3`,
`distractor_4`), following the exact copy-paste pattern already used to go
from 1→2 distractor slots (visible directly in how `distractor_1`/
`distractor_2` are already near-duplicates of each other throughout this
file). This is mechanical, not a redesign — but it is not "just bump a
constant," either, and a future plan should scope it as its own explicit
task, matching this experiment's own Task 1/Task 2 split.**

A second, load-bearing precedent already exists for the resulting
**41→43 dimensionality-mismatch problem this will reproduce at a larger
scale**: the completed experiment hit exactly this when Stage SO's new
43-dim schema was incompatible with the finished single-object
`model_2998.pt` (41-dim), and solved it with
`scripts/extend_checkpoint_observation_dims.py` — a weight-surgery script
that extends a network's first-layer input width by copying existing
columns unchanged and randomly initializing only the new ones, verified
bit-for-bit identical to the source checkpoint's output whenever the new
columns are hard-zeroed. **This script is not K=2-specific** — its mechanism
(copy N existing columns, randomly init the rest) works for any column-count
increase, so extending the finished D2 checkpoint's 43-dim schema to a
45-dim (K=3) or 47-dim (K=4) schema can reuse this exact tool rather than
needing new weight-surgery code. This is a concrete, already-validated
warm-start path for whatever count-curriculum design the next experiment
chooses (§3 below).

### 1b. Scene topology / reachability — does NOT extend cleanly; the existing 3-lane design has already consumed the full established workspace footprint

Read directly against `tasks/franka/lift_env_cfg.py`'s `CommandsCfg`
(`pos_x=(0.4,0.6)`, `pos_y=(-0.25,0.25)`, `pos_z=(0.25,0.5)` — the goal-pose
range this project has treated as its validated reachable-workspace
convention for both goal placement *and* object spawn lanes, per
`dice_lift_joint_env_cfg.py`'s own choice to match its lane x-range to this
exact `pos_x` bound) and the target-selection-clutter scene constants
(`dice_lift_joint_env_cfg.py:1353-1373`):

- **The existing 3-lane arrangement already spans the entire y-range exactly
  to its edges, with no slack left over.** Target lane: center y=0, spread
  ±0.05 → occupies y∈[-0.05, 0.05]. Distractor 1: center y=-0.185, spread
  ±0.065 → y∈[-0.25, -0.12], its outer edge landing EXACTLY on the
  `CommandsCfg` boundary (-0.25). Distractor 2: mirror image, outer edge at
  +0.25. There are 70mm-wide unused "buffer gap" bands between the target
  lane's own edge (±0.05) and each distractor lane's inner edge (±0.12) —
  this buffer is what produced the live diagnostic's measured 95mm
  minimum pairwise separation (vs. a 60mm safety floor established by that
  same diagnostic, `scripts/_diag_target_selection_clutter_scene_check.py`)
  for exactly 3 objects.
- **Every lane shares the identical x pose_range** (`x: (-0.1, 0.1)` for
  target, distractor_1, AND distractor_2 alike) — lanes are differentiated
  by y-offset only. This means the x-dimension of the reachable rectangle
  (200mm wide, 0.4-0.6) is currently completely unused for lane
  differentiation; all 3 current lanes occupy the same x-slice.
- **Naive extension (cram 2 more 1D y-lanes into the same 500mm y-band)
  is a real, non-trivial packing problem, not free headroom.** Back-of-
  envelope: 5 contiguous, equal-width lanes across the same 500mm total span
  works out to ~100mm/lane if packed edge-to-edge with zero buffer — tighter
  than the current 3-lane design's own ~120mm-wide distractor lanes *plus*
  70mm explicit buffer gaps to the target lane. Whether a re-tuned, narrower
  per-lane jitter spread (e.g. shrinking distractor spread from ±65mm to
  something closer to ±20-25mm) still clears the same 60mm minimum-
  separation floor for 5 simultaneously-present ~48mm-parity objects (this
  project's own established training-scale convention — see §1c) is an
  empirical question, not something this document can resolve by arithmetic
  alone. It needs the same kind of live, no-training-spend scene-topology
  diagnostic the original experiment already built and ran
  (`_diag_target_selection_clutter_scene_check.py`) before any training
  budget is committed — this is a genuine open risk, correctly flagged as
  "may become a real constraint" per the task brief, not resolved here.
- **A promising alternative worth flagging for the next experiment's own
  scene-topology task, discovered directly from reading the code (not
  assumed):** since the x-dimension is currently completely unused for lane
  differentiation, a 2D grid arrangement (e.g. a 2-row × 3-column layout
  using both the 200mm x-range and the 500mm y-range) would have
  substantially more usable area than continuing to squeeze every new
  distractor into a single 1D y-strip, and doesn't require widening beyond
  the already-validated reachable rectangle. This is an architectural
  scene-design choice for the next spec to make explicitly, not a
  foregone "just add 2 more y-lanes" continuation of the existing pattern.

### 1c. Shape coverage — a precise per-shape ledger, since the task brief's premise ("more shapes than validated in this clutter setting") needs exact grounding, not just directionally-true framing

| shape | single-object from-scratch PPO | single-object realized discovery | tested with ANY distractors present | tested as a distractor itself |
|---|---|---|---|---|
| d12 | 1/3 seeds (asset-bisect) | 8/8 (via distillation+RL-fine-tune pipeline, [[unified-multi-die-specialist-distillation]]) | YES — target-selection-clutter Stage D2, 8/8 | YES — Stage D1/D2, pooled with d20 |
| d20 | 2/3 seeds (asset-bisect/distillation Task 6) | 8/8 (same pipeline) | YES — Stage D2, 8/8 | YES — Stage D1/D2 |
| d8 | **0/3 seeds — never discovered from scratch** | 3/3 seeds full 8/8, but ONLY via H2 cross-shape checkpoint warm-start from the converged d12 specialist ([[d8-d10-demo-warmstart]]) | **NO — never tested with any distractor present** | **NO** |
| d10 | **0/3 seeds — never discovered from scratch** | 1/3 seeds full 8/8, same H2 mechanism | **NO — never tested with any distractor present** | **NO** |
| d4 | never successfully grasped in ANY setting (see §2) | N/A | **NO** | **NO** |

This confirms the task brief's framing precisely: d12/d20 are the only
shapes with any clutter (distractor-present) validation at all, and that
validation used *only each other* as the distractor pool — never d8/d10.
d8/d10's own single-object success is itself conditional on a specific
warm-start mechanism (cross-shape checkpoint transfer, not curriculum, not
from-scratch), a mechanistically different fix than what solved d12/d20's
clutter extension (distractor-count curriculum + observation term, over an
already-solved from-scratch-adjacent base). Folding d8/d10 into a clutter
population is therefore genuinely stacking two independently-validated but
*mechanistically distinct* fixes, not just "one more shape of the same
kind already proven to work in clutter."

---

## 2. d4 — explicit scope recommendation: **(b) exclude d4**, scope this experiment to {d8, d10, d12, d20}

**Recommendation, stated directly per the task's own "pick one, don't hedge"
instruction: exclude d4 entirely from this next experiment.** Justification,
grounded in d4's own full documented history (`ROADMAP.md` lines
~2799-2994, `kb/wiki/experiments/`), not a generic "d4 seems hard":

- **d4's failures are not exploration/discovery failures — the exact class
  of problem this project's own toolkit (curriculum, warm-start,
  distillation) has repeatedly solved for other shapes.** They are
  *grasp-mechanism/geometric-affordance* failures, upstream of anything a
  training procedure can fix:
  - Rung 0 (tilted opposite-edge grasp, scripted DiffIK controller):
    FALSIFIED at reachability — root-caused via exact waypoint-arithmetic
    reconstruction to the lower jaw needing to go 3-15mm *below the table
    surface* for every plausible finger-opening/geometry combination.
    PhysX blocks it outright; this is a hard geometric fact about stock
    Franka jaws vs. a table-resting tetrahedron's edge-pair axis, not a
    tuning problem.
  - Rung 1 (purpose-built 110° V-notch fingertip fixture, straight-down
    approach): first blocked entirely by the vision detector (0/5 seeds,
    zero `d4`-class detections — a separate, since-diagnosed perception
    weakness, not yet the grasp mechanism at all).
  - **Ground-truth XY-bypass (the closest thing this project has to an
    isolated test of "can this gripper physically grasp a d4 at all,"
    explicitly designed to route around the perception failure and test
    the mechanism directly):** 3/5 seeds genuinely reached the grasp
    mechanism this time. **0/3 succeeded** — lateral ejection 18.8-172.0mm
    (vs. a ≤5mm threshold), confirmed visually frame-by-frame: "gripper
    fully closed at the die's original position, die sitting undisturbed
    several cm away — not a subtle grasp-then-eject, **the notch swept the
    die aside without ever engaging it**." This is the exact phrase the
    task brief quotes, and reading the surrounding ROADMAP context confirms
    it is not a throwaway description — it is this project's own most
    direct, most-isolated grasp-mechanism test for d4, purpose-built
    hardware included, and it still failed to so much as touch the die
    correctly.
  - The original dice-pick-demo convergence milestone (2026-07-11) already
    pre-declared d4 the sole permitted failure of 5 shapes for the same
    underlying reason: "flat-pad closure squeezes it out."
- **d4 has never once been attempted via a learned (RL) policy at all** —
  every attempt above is a *scripted* DiffIK controller, not RL. This is a
  real gap in strict logical terms (RL could in principle discover a
  strategy no scripted controller considered, e.g. tipping the die onto an
  edge first, an asymmetric two-stage approach, etc.) — but it does not
  change the recommendation, because the specific failure mode observed
  (a specially-engineered V-notch fixture, fed ground-truth position,
  cannot even keep contact with the die during closure — it slides it away)
  describes a closure/contact-geometry problem that exists identically
  regardless of what process selects the approach pose. An RL policy
  driving this same rigid gripper geometry has no more physical ability to
  keep the die from being swept aside during jaw closure than the scripted
  controller did; RL changes *how the approach pose and timing are chosen*,
  not the physical contact/friction-cone facts about a flat-faced
  tetrahedron between parallel-jaw pads that rung 0/1 already established.
- **Including d4 anyway would confound the experiment's own falsification
  logic**, repeating a mistake this exact experiment's own predecessor
  already made and had to root-cause: the original Stage SO 0/8 result was
  confounded because it stacked "does the new schema/scene wiring work" with
  a separate, pre-existing "does from-scratch PPO ever discover this grasp"
  difficulty, and took real cloud spend and investigation to disentangle
  ([[target-selection-clutter]], "Task 4 (original, CONFOUNDED)" section).
  Adding d4 — a shape with a *near-certain* near-0% success rate for reasons
  unrelated to clutter/distractor-count at all — into a 5-shape,
  4-distractor experiment that is *already* testing two other genuinely
  novel axes (K=2→4 distractor scaling, d8/d10-in-clutter) would make any
  d4 failure impossible to attribute cleanly, and would not even be a
  meaningful test of the clutter mechanism for d4 specifically, since d4
  fails upstream of clutter entirely.
- **d4 deserves its own dedicated grasp-mechanism research track** (e.g.
  resuming the rung ladder past the V-notch's current 110°/dimensions, a
  genuinely different strategy — tip-to-edge reorientation before grasp,
  a non-antipodal caging strategy, etc.) — orthogonal to and prior to any
  clutter-scaling question, not a stretch goal to bolt onto this experiment.
  This matches CLAUDE.md's own Tier 1 discipline of isolating one structural
  variable per experiment.

---

## 3. Proposed hypothesis and methodology for the next experiment (4 shapes: d8, d10, d12, d20; K=2→4 distractors)

### 3a. Why a staged design, and what the stages should isolate

The completed experiment's own curriculum (SO→D1→D2, each stage
checkpoint-resumed from the prior) is the direct, already-validated
precedent for scaling distractor *count*. The completed d8/d10
demo-warmstart experiment's own H2 result (cross-shape checkpoint
warm-start from the converged d12 specialist, not curriculum, not
from-scratch) is the direct precedent for bringing in a shape that has
never discovered its own grasp cold. **These are two mechanistically
different fixes for two mechanistically different problems**
(§1c's table makes this concrete), and the task brief's own framing
("does a similar staged approach make sense... does the checkpoint
warm-start precedent suggest anything") points at combining them in
sequence rather than fusing them into one untested leap — directly
mirroring how this exact experiment's own Stage-SO confound taught this
project that combining two untested changes at once makes failure
attribution impossible.

**Proposed stage sequence (for the eventual spec to formalize, not decided
here):**

1. **Stage E1 (count-scaling only, d12/d20 population unchanged):**
   resume from the finished D2 checkpoint (`model_5096.pt`, 43-dim, 8/8
   both shapes at K=2), extend the observation schema to K=3
   (`extend_checkpoint_observation_dims.py`, reused not rebuilt, following
   the exact same "copy existing columns, randomly init the new
   hard-zeroed ones" mechanism verified bit-for-bit lossless at Stage SO),
   add a third distractor slot to the scene/event cfg (§1a's mechanical
   extension), and train with 3 active distractors, target/distractor
   population still restricted to {d12, d20}. This isolates "does the
   distractor-count curriculum + observation mechanism itself scale past
   K=2" as its own falsifiable question, with zero shape-coverage risk
   mixed in.
2. **Stage E2 (K=3→4, same d12/d20-only population):** resume from Stage
   E1, extend to K=4 the same way, complete the count-scaling axis in
   isolation before touching shape coverage at all.
3. **Stage S1 (shape introduction, K=4 distractors already validated by
   E2):** fold d8/d10 into the target *and* distractor population for the
   first time, resuming from Stage E2's checkpoint. This isolates "does
   bringing in never-clutter-tested shapes work" as a separate question
   from "does K=4 work," matching the same one-variable-at-a-time
   discipline.

**Falsifiable hypothesis:** starting from the finished D2 checkpoint and
proceeding through this staged sequence (count-scaling first, in isolation;
shape-introduction second, only after count-scaling is confirmed), each
stage checkpoint-resumed from the prior exactly as the original SO→D1→D2
curriculum was, will preserve sustained-lift discovery at or above the
same 6/8 (75%) falsification bar the original experiment used, for d12 and
d20 throughout, and — separately, at Stage S1 only — for d8/d10 as well,
without a repeat of the "reach but never grasp" collapse pattern the
original from-scratch Stage SO attempt hit.

**Falsification condition, split per stage (so a single collapse doesn't
retroactively implicate the wrong mechanism):** if Stage E1/E2 collapses
for d12/d20 specifically, that falsifies "the existing count-curriculum +
fixed-size zero-padded observation mechanism scales past K=2" — the
literature's own DexSinGrasp precedent (§1 of the prior clutter-literature
doc; Table IV) already shows an *arrangement-dependent*, non-monotonic
collapse can occur even within a single paper's own 4→6→8 distractor
range, so this is a real, literature-grounded possible outcome, not just
caution for its own sake — and the honest escalation path is a
richer/architecturally-different observation mechanism (Deep Sets/
attention over distractor state), not another parameter retune. If Stage
S1 collapses specifically for d8/d10 while d12/d20 remain fine, that
would mirror this project's own original Stage-SO confound almost exactly
(a shape that's never proven a from-scratch-adjacent cold-start into a new
schema) — the pre-authorized fallback, directly modeled on H2's own
precedent, is a per-shape checkpoint warm-start (extend Stage S1's own
finished d12/d20-clutter weights using d8/d10's own already-converged
single-object H2 checkpoints as an additional seed, rather than assuming
Stage S1's from-scratch-into-new-shapes introduction works cleanly the
first time) — flagged here as the anticipated fallback, not designed in
full; that is the eventual spec/plan's job.

### 3b. What this research does NOT resolve

- The exact scene-topology fix for K=4 (1D lane-packing retune vs. the 2D
  grid alternative flagged in §1b) — needs its own live diagnostic,
  analogous to `_diag_target_selection_clutter_scene_check.py`, before any
  training spend, exactly as the original experiment's own Task 1 did.
- Whether d8/d10 should be drawn from the SAME `MultiAssetSpawnerCfg`
  population as d12/d20 (`random_choice=True`, matching D1/D2's own
  cross-shape-pairing convention) or something more conservative (e.g.
  d8/d10 introduced as target-only or distractor-only first) — a real
  spec-time scoping decision this document deliberately leaves open.
  A conservative option worth flagging for the spec author: since d8/d10
  have never been tested as a *distractor* at all (only as a target, and
  only via H2 warm-start), Stage S1 could further sub-stage
  "d8/d10-as-target-only" before "d8/d10-as-distractor-too," an even finer
  isolation than the 2-stage split above — the eventual plan should weigh
  this against added cost/iteration overhead, not something this research
  document decides.
- Iteration budgets per stage — the original experiment's own judgment
  calls (300/801/1000 iterations for SO/D1/D2, scaled to how much genuinely
  new adaptation each stage required) are the right *pattern* to reuse, not
  a specific number to copy verbatim for a materially different stage set.
- Cost — not estimated here; the original experiment's whole 3-stage
  curriculum cost ≈$1.35 (well under a $5 cap), but a 4-stage sequence
  (E1/E2/S1, plus any per-shape H2-style fallback) touching 4096-env
  populations across 4 shapes is a materially larger training surface and
  should get its own explicit cost cap in the spec, not inherit the prior
  experiment's cap by assumption.

---

## 4. Open risks / gaps, stated plainly

- **The scene-topology packing question (§1b) is the single largest
  unresolved feasibility risk this document identifies** — it is the one
  place where "does the existing mechanism extend" genuinely might be
  "no, not without a real redesign (2D grid, or a narrower per-lane
  spread)," rather than "yes, mechanically." Every other extension point
  (observation term, weight-surgery warm-start tooling) is mechanical-but-
  real work over an already-validated pattern; this one is not yet known to
  fit within the existing reachable-workspace convention at all for 5
  simultaneous ~48mm-parity objects.
- **No literature source found (in this document or the prior verified
  clutter-literature doc) that isolates "shape-heterogeneous clutter"
  (different object geometries as target vs. distractor, simultaneously)
  as its own studied variable** — DexSinGrasp's own reference setting
  (§2 of the prior literature doc) does not describe shape-diverse
  distractor pools as a distinct condition from same-shape ones; this
  project's own only same-vs-cross-shape data point is the completed
  D1/D2 experiment's own explicit pooling of same-shape and cross-shape
  d12/d20 pairings (both tested together, both passed) — a real, if
  narrow, positive precedent that shape heterogeneity itself is not
  automatically fatal to the mechanism, but with only 2 shapes' worth of
  evidence behind it.
- **This document does not re-verify whether d8/d10's real (non-48mm-
  parity) size difference from d12/d20 matters** — per
  `docs/superpowers/specs/research/2026-07-19-d8-d10-grasp-discoverability-literature.md`
  and the d8-d10-demo-warmstart experiment itself, this project trains all
  shapes at a common "48mm-parity" scale (decoupled from real/standard die
  sizes) specifically so that shape, not size, is the varying factor —
  this convention should hold for the next experiment too, but was not
  independently re-checked against that other document's own reasoning in
  this pass.
- **Cost/quota contention** — both the completed target-selection-clutter
  and d8/d10-demo-warmstart experiments independently hit the project-wide
  `GPUS_ALL_REGIONS=1` cloud quota as a real, recurring cross-workstream
  friction point (documented in both kb articles). A 3-4-stage sequence
  spanning 4 shapes is likely to be a longer-running, more cloud-GPU-hungry
  effort than either predecessor alone; the eventual plan should budget
  dispatch time accordingly, not assume desktop availability.

---

## Related

[[target-selection-clutter]] (the experiment this extends; source of the
K=2 observation term, scene topology, and curriculum precedent),
[[d8-d10-demo-warmstart]] (source of the H2 cross-shape checkpoint
warm-start precedent, and the from-scratch-discoverability ledger in §1c),
[[unified-multi-die-specialist-distillation]] (source of the per-shape
from-scratch discovery rates in §1c and the sphericity-based
nearest-neighbor warm-start reasoning H2 used),
`docs/superpowers/specs/research/2026-07-19-target-selection-clutter-literature.md`
(the DexSinGrasp citation this document reuses directly rather than
re-verifying — already confirmed real, with the precise, non-overstated
paraphrase of its curriculum/observation findings), `ROADMAP.md`'s d4
edge-grasp-rung history (source of §2's grasp-mechanism-failure
justification for excluding d4).
