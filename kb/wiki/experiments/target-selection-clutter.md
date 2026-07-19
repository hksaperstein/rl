# Target-selection-among-distractor-dice experiment (2026-07-19 -> in progress)

**Status: Stage SO internal sanity gate FAILED (0/8 both shapes) — plan
execution STOPPED per its own pre-registered gate discipline, Stage
D1/D2 not started.** This is an interim entry (Task 4 of the plan only);
the closing verdict lands at the plan's Task 7 once the controller
decides how to proceed.

**Goal:** extend [[unified-multi-die-specialist-distillation]]'s finished
single-object d12/d20 policy (`model_2998.pt`, 8/8 both shapes with
exactly one die in the scene) into a 3-die scene (1 commanded target + 2
distractor dice) via a distractor-count curriculum (SO: 0 active -> D1: 1
-> D2: 2), testing whether curriculum + a new fixed-size zero-padded
distractor-distance observation term (DexSinGrasp's own `d_t^S`
mechanism) preserves discovery under clutter. Spec:
`docs/superpowers/specs/2026-07-19-target-selection-clutter-design.md`.
Plan: `docs/superpowers/plans/2026-07-19-target-selection-clutter-
implementation.md`.

## Tasks 1-3: scene topology + observation term + wiring (complete)

Added `FrankaDieLiftTargetSelectionSceneCfg` (two new sibling
`RigidObjectCfg` distractor slots, following `dice_scene_cfg.py`'s
multi-`RigidObjectCfg` pattern), `TargetSelectionEventCfg` (disjoint
per-entity reset lanes, target y in [-0.05,0.05], distractor_1 y in
[-0.25,-0.12], distractor_2 y in [0.12,0.25], all sharing x in
[0.4,0.6]), the new `distractor_distance_summary` observation term (K=2,
hard-zero-padded per curriculum stage, 41->43 dims), and the 3
curriculum-stage env cfgs + 6 pinned-target `_PLAY` eval variants. Live
diagnostic (`scripts/_diag_target_selection_clutter_scene_check.py`,
Stage D2 topology, num_envs=16) confirmed correct shape assignment, live
USD scale, and no cross-entity overlap (min pairwise separation 95mm vs a
60mm safety floor) — no bug found in the scene/observation code itself.
Commits `82a44b8`, `a08335a`, `9ce3a6d`.

## Task 4: Stage SO trained fresh — sanity gate FAILED, 0/8 both shapes (2026-07-19)

Trained `FrankaDieLiftJointD12D20TargetSelectionSOEnvCfg` (0 active
distractors — both distractor slots present but PARKED off-workspace at
z=-0.9, zero-width pose_range) completely FROM SCRATCH (random PPO init,
not resumed — the 43-dim schema is incompatible with the 41-dim
`model_2998.pt`, no cross-dimensionality warm-start mechanism exists in
this codebase), seed 42, 1500 iterations, cloud (`g2-standard-4`+L4 SPOT,
desktop was busy with a concurrent d8/d10 workstream at dispatch time).
Instrumented eval (`franka_checkpoint_review.py --eval_target_shape
{d12,d20}`, num_envs=8, full 3-die topology, both distractors parked):

| shape | envs_with_sustained_lift | gate (>=7/8) | max_height_gain (typical) |
|-------|---------------------------|--------------|----------------------------|
| d12   | **0/8**                   | FAIL         | ~3.3-3.5mm (vs 40mm threshold) |
| d20   | **0/8**                   | FAIL         | ~1.3-2.3mm (vs 40mm threshold) |

Per the plan's own pre-registered gate: **STOP, do not proceed to Stage
D1/D2.** Checkpoint:
`gs://rl-manipulation-hks-runs/target-selection-clutter/joint-die-target-selection-so/seed42/2026-07-19_21-25-52/model_1499.pt`.
Eval artifacts:
`gs://rl-manipulation-hks-runs/target-selection-clutter/eval-artifacts/joint-die-target-selection-so/`.
Cost: ~$0.44 (cloud, ~70min instance existence), full teardown verified
(`scripts/check_cloud_state.sh` clean).

**No code bug found on investigation** (reward/termination config
confirmed byte-identical to the already-proven baseline and scoped only
to `scene["object"]`; parked-distractor positions confirmed physically
inert — 0.9m below the table, zero-width pinned pose_range; the new
observation dims are additive and hard-zeroed at
`active_distractor_count=0`, matching Task 2's own unit tests). The
training curve and eval video instead show a specific, consistent
behavioral pattern: `Episode_Reward/reaching_object` climbs cleanly to
~0.83 (the arm learns to approach the die) while
`Episode_Reward/lifting_object` plateaus at a constant ~0.12 from
iteration ~50 onward and never improves, `object_goal_tracking` stays
near 0, and `Metrics/object_pose/position_error` never decreases across
the whole 1500-iteration run. Video frames confirm this directly: the
gripper reaches down and hovers directly over the die but never closes
around it or lifts it, for the full length of every inspected episode.

**Open question this result does NOT cleanly resolve, flagged for the
controller rather than decided here:** this experiment's own baseline
checkpoint (`model_2998.pt`, 8/8 both shapes) was itself never produced
by a from-scratch PPO run on the d12/d20-mixed population — per
[[unified-multi-die-specialist-distillation]]'s own Task 5/6, plain
BC/DAgger distillation onto this same 2-shape population only reached
4/8 (d20) / 1/8 (d12), and needed 1500 *additional* RL-fine-tune
iterations on top of an already-partially-grasp-capable distilled
checkpoint to reach 8/8. No run of this env's plain from-scratch PPO
(random init, no distillation bootstrap) has ever been attempted or
documented in this project before Stage SO. Stage SO's 0/8-both-shapes
"reach but never grasp" pattern is consistent with — and may simply be
re-confirming — this project's own long-documented d12/d20
grasp-*discovery* difficulty (the same barrier the entire specialist ->
distill -> RL-fine-tune pipeline was built to solve), rather than a new
defect introduced by the parked-distractor/observation-schema additions.
The plan's Task 4 design (train fully from scratch, forced by the
41-vs-43-dim checkpoint-incompatibility) may have inadvertently
re-introduced exactly the cold-start difficulty the distillation
pipeline exists to route around, confounding "did the schema extension
break something" (what Stage SO was designed to isolate) with "does
plain from-scratch PPO ever discover d12/d20 grasps at all" (a
pre-existing, separately-documented difficulty). This is a real
candidate explanation, not a decided one — no additional training run
was performed to test it directly (out of this dispatch's scope), and no
architectural response (e.g. a partial/best-effort weight-transfer warm
start into the 43-dim network from `model_2998.pt`'s 41-dim weights) has
been decided or implemented.

## Open, not yet decided

Whether to: (a) re-run Stage SO with some form of warm start from
`model_2998.pt` despite the dimensionality mismatch (e.g. partial weight
transfer for the shared 41 input dims, random-init the 2 new dims) to
separate the "from-scratch difficulty" confound from the "schema
extension" question the gate was meant to isolate; (b) accept the
from-scratch difficulty as the explanation and treat this as a
non-finding about clutter/target-selection specifically; or (c) some
other structurally different approach. Not decided in this task — see
the plan's Task 4 Step 4 instruction ("STOP and report to the
controller").

## Related

[[unified-multi-die-specialist-distillation]] — this experiment's own
direct predecessor and checkpoint source.
