# Target-selection-among-distractor-dice experiment (2026-07-19, COMPLETE)

**FINAL VERDICT: hypothesis PASSES.** Stage SO gate PASSED (d12 8/8, d20
7/8), Stage D1 (1 active distractor) PASSED cleanly (d12 8/8, d20 8/8),
and Stage D2 — the primary falsification check, 2 active distractors, the
real target configuration — also PASSED cleanly: **d12 8/8, d20 8/8**,
both comfortably above the pre-registered 6/8 (75%) bar, and no
grasped-the-wrong-die episode observed in any inspected env/frame for
either shape. Curriculum + a fixed-size zero-padded distractor-distance
observation term (DexSinGrasp's own `d_t^S` mechanism), with the reward
function and target-identification mechanism left completely unchanged,
is sufficient to preserve the single-object 8/8 discovery rate under
2-distractor clutter for both d12 and d20. See "FINAL VERDICT" section at
the bottom for the full closing writeup across all 3 stages.

The originally-reported Stage SO 0/8-both-shapes from-scratch result
below was confounded (see "Task 4 corrected" section) and was superseded
by a partial-weight warm-started retrain that isolates the question the
gate was actually meant to answer.

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

## Task 4 (original, CONFOUNDED — superseded below): Stage SO trained fresh, 0/8 both shapes (2026-07-19)

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

## Task 4 corrected: partial-weight warm start from `model_2998.pt` — sanity gate PASSED, d12 8/8 / d20 7/8 (2026-07-19)

Resolves the "Open, not yet decided" question below in favor of option
(a). Per `BACKLOG.md`'s "Clutter experiment Stage SO gate: confounded, fix
is a partial-weight warm start" entry: `distractor_distance_summary` (the
2 new obs dims) is a hard-zeroed *constant* at Stage SO
(`active_distractor_count=0`), so a network whose first-layer weight
matrix is extended from 41->43 input columns by copying the 41 existing
columns unchanged and randomly initializing only the 2 new ones is
mathematically guaranteed to produce IDENTICAL output to `model_2998.pt`
at Stage SO specifically — a lossless, verifiable warm start that
isolates the schema-extension question from the separate from-scratch
cold-start difficulty the original Task 4 attempt (above) got confounded
with.

**Mechanism, new script:** `scripts/extend_checkpoint_observation_dims.py`
(pure `torch`, no Isaac Sim/rsl_rl dependency required for the surgery
itself — only needs a real interpreter with `torch` installed, ran fine
on the Pi host itself with a throwaway CPU-only venv). Finds the real
`actor.0`/`critic.0` first-Linear-layer keys by pattern (confirmed live
against a real `model_2998.pt`: `actor.0.weight`/`critic.0.weight` are
both `(256, 41)`, matching `tasks/franka/agents/rsl_rl_ppo_cfg.py`'s
`[256,128,64]` hidden dims), extends each to `(256, 43)` by appending 2
columns freshly initialized via a throwaway `nn.Linear(43, 256)`'s own
default `kaiming_uniform_` init (so the new columns' scale matches what a
real from-scratch layer of the new width would get), and writes the
output in `distillation.py:save_student_checkpoint`'s own
`{"model_state_dict", "optimizer_state_dict": {}, "iter", "infos"}`
format (the pre-existing "policy-only checkpoint" convention
`--policy_only_checkpoint` was built for) — `iter` defaults to preserving
the source's real `2998`, so `train_franka.py --max_iterations` must be
set as an ABSOLUTE target (`2998 + budget`), per that script's own
convention.

**Verification BEFORE any training spend (per this task's own
requirement):** `--verify` builds the original 41-dim and extended 43-dim
networks as plain hand-rolled `nn.Sequential` MLPs (matching rsl_rl's own
Linear/ELU layout, no rsl_rl import needed), feeds the extended network a
batch of random 43-dim observations with the last 2 columns zeroed (the
real Stage-SO condition) and the original network the same batch's first
41 columns, and asserts the outputs match. Run twice against the REAL
`model_2998.pt` (once locally on the Pi with a throwaway CPU venv before
any cloud spend, once again on the cloud instance immediately before
training) — **both passed at exactly 0.0 max abs diff for both actor and
critic branches**, i.e. bit-for-bit identical, not just "within
tolerance." A negative-control check (feeding nonzero values into the new
columns instead of zero) produced a large 0.53 diff, confirming the
verify check is a real discriminating test, not a vacuous pass — the
"always-zero, so it's inert" reasoning holds exactly as predicted, no
hidden bias/normalization-layer wrinkle.

**Retrain:** `train_franka.py --variant joint-die-target-selection-so
--checkpoint <extended checkpoint> --policy_only_checkpoint
--max_iterations 3298` (2998 preserved + 300 new iterations — a small
bounded budget, not the original 1500, since the policy is warm-started
from already-8/8 behavior and needs no exploration/discovery, only to
confirm the new wiring doesn't perturb it once the 2 new dims start
carrying real distractor-distance signal in later stages). Cloud
(`g2-standard-4`+L4 SPOT, desktop busy with a concurrent d8/d10
workstream at dispatch time — confirmed live via
`scripts/check_cloud_state.sh` showing an active `extract_demo_trajectory.py`
process before dispatch). `Episode_Reward/lifting_object` was already
0.40 at the very first logged iteration (vs. the original attempt's
~0.12 permanent plateau) and reached 12.37 by the run's end — confirming
the theoretical prediction directly, not just via the final eval gate.

Instrumented eval (`franka_checkpoint_review.py --eval_target_shape
{d12,d20}`, num_envs=8, full 3-die topology, both distractors parked,
checkpoint `model_3297.pt`):

| shape | envs_with_sustained_lift | gate (>=7/8) | max_height_gain (typical) |
|-------|---------------------------|--------------|----------------------------|
| d12   | **8/8**                   | PASS         | ~376-439mm |
| d20   | **7/8**                   | PASS         | ~363-401mm (1 env: 85mm, below sustained threshold) |

Video + per-step height data both confirm real sustained lifts (213-217
consecutive post-settle steps above the 40mm threshold out of ~239
analysis-window steps, i.e. lifted for the large majority of each
episode) — not a brief height blip of the kind Experiment 16 warned
about.

Checkpoint (extended, pre-training):
`gs://rl-manipulation-hks-runs/target-selection-clutter/joint-die-target-selection-so-warmstart-checkpoint/model_2998_43dim.pt`.
Trained run:
`gs://rl-manipulation-hks-runs/target-selection-clutter-stageso-warmstart/joint-die-target-selection-so/seed42/2026-07-19_22-52-41/`
(final checkpoint `model_3297.pt`). Eval artifacts:
`gs://rl-manipulation-hks-runs/target-selection-clutter/eval-artifacts/joint-die-target-selection-so-warmstart/`.
Cost: ~$0.39 (cloud, ~60min instance existence — cheaper than the
original $0.44/~70min attempt despite the extra weight-surgery step, since
300 iterations trains much faster than 1500), full teardown verified
(`scripts/check_cloud_state.sh` clean, zero instances/disks/snapshots).
Combined Task 4 total (both attempts): ~$0.83 of the plan's $5 cap for
Tasks 4-6 — ~$4.17 remains for Tasks 5/6.

**Verdict: Stage SO's gate now PASSES for both shapes.** This confirms
the "confounded by pre-existing from-scratch cold-start difficulty"
explanation from the original Task 4 attempt (above) was correct — the
scene/observation-schema wiring itself was never broken, exactly as the
"no code bug found on investigation" note above already suspected. Stage
D1/D2 (plan Tasks 5/6) are unblocked to proceed from this corrected
checkpoint.

## Task 5: Stage D1 (1 active distractor), resumed from Stage SO — PASSED, d12 8/8, d20 8/8 (2026-07-19)

Resumed `FrankaDieLiftJointD12D20TargetSelectionD1EnvCfg` (1 active
distractor, drawn per-env from `{d12,d20}` via
`MultiAssetSpawnerCfg(random_choice=True)`; `distractor_2` still parked)
directly from Stage SO's real, passing checkpoint —
`gs://rl-manipulation-hks-runs/target-selection-clutter-stageso-warmstart/joint-die-target-selection-so/seed42/2026-07-19_22-52-41/model_3297.pt`
(the corrected/warm-started run above, NOT the original confounded
from-scratch attempt) — via a normal same-dimensionality PPO resume
(`--checkpoint ... `, no `--policy_only_checkpoint`; both the 43-dim
schema and the optimizer state carry over unchanged from Stage SO to D1).
`train_franka.py` printed `Resumed from .../model_3297.pt at iteration
3297`, confirming the checkpoint's own recorded iteration count directly
rather than assuming it from the filename.

**Iteration budget (judgment call, not the plan's default 1500):** chose
801 additional iterations (`--max_iterations 4098`, final checkpoint
`model_4097.pt`) — a deliberate middle ground between Stage SO's 300 (an
inert, mathematically-guaranteed-identical-output warm start needing no
real exploration) and this project's usual 1500 from-scratch default (for
genuinely novel behavior from random init). Reasoning: Stage D1 is not a
cold start — the base grasp/lift skill is already at 8/8 coming in, and
target identity is still structurally given (`scene["object"]` is always
the commanded die by construction, unchanged from Stage SO) — but it is
also not a no-op like Stage SO's inert zero-padded dims: the distractor
is now a real, physically-present, per-episode-varying nearby object for
the first time, so the policy genuinely has something new to adapt to
(tolerating a real nonzero `distractor_distance_summary` signal,
not colliding with/being distracted by a nearby die during reach). 801
iterations was chosen as enough real adaptation time for that narrower
adaptation without paying for a full from-scratch budget. The streamed
reward curve, watched live during the run, supports this choice after
the fact: `Episode_Reward/lifting_object` rose from ~11.27 (iteration
~3603) to ~12.9 by the run's end, `Episode_Reward/object_goal_tracking`
rose from ~6.5 to ~11.2, `Episode_Reward/reaching_object` stayed flat
near its ceiling (~0.77-0.78) throughout, and `Episode_Termination/
object_dropping` stayed low (~0.7-1.2%) the whole run — genuine,
sustained improvement with no divergence/collapse, and a visible (if not
fully flat) plateau forming by the final ~150 iterations, not a curve cut
off mid-climb.

Instrumented eval (`franka_checkpoint_review.py --eval_target_shape
{d12,d20}`, num_envs=8, full 3-die topology, `distractor_1` real/active,
`distractor_2` still parked, checkpoint `model_4097.pt`):

| shape | envs_with_sustained_lift | max_height_gain (typical) | max_consecutive_lifted_steps |
|-------|---------------------------|----------------------------|-------------------------------|
| d12   | **8/8**                   | ~365-457mm                 | 215-222 (of ~239 post-settle steps) |
| d20   | **8/8**                   | ~409-440mm                 | 217-222 (of ~239 post-settle steps) |

This equals or exceeds the single-object 8/8-both-shapes baseline
(`model_2998.pt`) and exceeds Stage SO's own d20 result (7/8) — Stage D1
shows no discovery degradation at 1 active distractor. Both eval videos
were downloaded and inspected directly (frames extracted via `ffmpeg`,
not just the JSON summary read): every inspected frame shows the gripper
holding the target die clearly elevated above the table while the
distractor die sits undisturbed and ungrasped in its own reset region —
visually confirming the height-based "sustained lift" numbers reflect a
real, clean single-target grasp, not a wedge/contact artifact or an
accidental distractor grab.

Checkpoint: `gs://rl-manipulation-hks-runs/target-selection-clutter/joint-die-target-selection-d1/seed42/2026-07-19_23-58-28/model_4097.pt`.
Eval artifacts: `gs://rl-manipulation-hks-runs/target-selection-clutter/eval-artifacts/joint-die-target-selection-d1/`.
Cost: ~$0.36 (cloud — desktop was busy with a concurrent
`bc_pretrain_demo_warmstart.py` workstream at dispatch time, confirmed
live via `scripts/check_gpu_availability.sh` reporting `TARGET=cloud`/
BUSY before provisioning; `g2-standard-4`+L4 SPOT, ~57min instance
existence), full teardown verified (`scripts/check_cloud_state.sh`
clean, zero instances/disks/snapshots). Combined Task 4+5 total: ~$1.19
of the plan's $5 cap for Tasks 4-6 — ~$3.81 remains for Task 6 (Stage D2,
the primary falsification check).

No code bug found or fixed during this task — Task 1-3's wiring (scene
topology, observation term, `--variant`/`--eval_target_shape` plumbing)
worked exactly as built for the D1 variant with no changes needed.

## Task 6: Stage D2 (2 active distractors), resumed from Stage D1 — PRIMARY FALSIFICATION CHECK PASSED, d12 8/8, d20 8/8 (2026-07-19)

Resumed `FrankaDieLiftJointD12D20TargetSelectionD2EnvCfg` (both
`distractor_1`/`distractor_2` now real/active, each its own independent
`MultiAssetSpawnerCfg(random_choice=True)` population drawn from
`{d12,d20}`) directly from Stage D1's own checkpoint —
`gs://rl-manipulation-hks-runs/target-selection-clutter/joint-die-target-selection-d1/seed42/2026-07-19_23-58-28/model_4097.pt`
— via a normal same-dimensionality PPO resume (`--checkpoint`, no
`--policy_only_checkpoint`). `train_franka.py` printed `Resumed from
.../model_4097.pt at iteration 4097`, confirming the checkpoint's own
recorded iteration count directly (matching the filename, but verified
via the printed message rather than assumed) before committing to an
iteration budget.

**Execution backend (real friction, worth recording):** `check_gpu_availability.sh`
reported the desktop BUSY at dispatch time (a concurrent
`bc_pretrain_demo_warmstart.py` workstream), so this task started on
cloud (`g2-standard-4`+L4 SPOT, `us-central1-a`, instance
`rl-franka-clutter-d2`) per the desktop-first/cloud-fallback policy. The
install completed cleanly, but the instance hit a genuine SPOT
preemption (confirmed via `gcloud compute operations list` showing a
real `compute.instances.preempted` system event) ~19 minutes after
creation, before training started. On restart attempt, the same zone
stocked out (`ZONE_RESOURCE_POOL_EXHAUSTED`); investigating further
surfaced a real, previously-undocumented constraint: **this project's
GCP `GPUS_ALL_REGIONS` quota is 1, project-wide** — and a *different*
concurrent Senior workstream (the exploration-bonus experiment) already
held that single GPU slot with its own live cloud instance
(`rl-franka-exploration-bonus`, `us-east1-c`), so a fresh instance could
not be created in any zone until that other workstream's instance was
released. Rather than wait on or touch another workstream's resource,
this task rechecked `scripts/check_gpu_availability.sh` and found the
desktop had freed up in the meantime — switched to desktop dispatch for
the remainder of the task (a fresh, isolated `git clone` at
`~/projects/rl-target-selection-d2` on the desktop, deliberately NOT the
shared `~/projects/rl` checkout there, which had uncommitted changes
belonging to the concurrent d8/d10 demo-warmstart workstream at the
time). The aborted cloud instance was fully deleted (zero
instances/disks/snapshots left from this task's own cloud usage,
`scripts/check_cloud_state.sh` confirmed clean of this task's own
resources afterward). **Cross-cutting infra note for the controller:** a
project-wide GPU quota of 1 means at most one Senior workstream can use
cloud GPU dispatch at a time; this is a real, currently-undocumented
constraint on this project's "genuinely parallel workstreams" model that
the controller may want to raise (a quota increase request, Console-UI
only, same as the earlier `GPUS_ALL_REGIONS` grant) — not decided or
requested here, out of this task's own scope.

**Iteration budget (judgment call):** chose 1000 additional iterations
(`--max_iterations 5097`, final checkpoint `model_5096.pt`) — somewhat
more than Stage D1's own 801, per this plan's own guidance ("D2
introduces a second simultaneous distractor, a further real step up in
difficulty from D1's one, so probably needs somewhat more than D1's 801
iterations"). Reasoning: Stage D1 already demonstrated the base
grasp/lift skill tolerates one real, physically-present distractor at
8/8; Stage D2 asks the same skill to tolerate two simultaneous
distractors (more collision-avoidance/reach-path complexity, and the
`distractor_distance_summary` observation now carries two genuinely
nonzero, independently-varying values instead of one), a real but
bounded increment, not a new skill from scratch — 1000 iterations gives
real room for that increment without approaching a full from-scratch
budget. The reward curve, watched live during the run, supports this
choice after the fact: at the very first logged iteration (4097),
`Episode_Reward/lifting_object` was already 0.12 (a fresh-resume
transient, matching the same pattern seen at the start of every prior
resume in this experiment) and had already recovered to 12.80 by
iteration 4269 (172 iterations in) — essentially Stage D1's own final
value — then continued a slow, healthy climb to a stable plateau around
13.0 (`Episode_Reward/reaching_object` flat at its ceiling ~0.78-0.79
throughout, `Episode_Reward/object_goal_tracking` climbing from ~11.3 to
~12.2, `Episode_Termination/object_dropping` low and stable at 0.5-1.3%
the whole run). No divergence, no plateau cut off mid-climb — the curve
had clearly leveled off well before the 1000-iteration budget ran out.
Training took ~24.5 minutes wall-clock on the desktop's dedicated
(uncontended) L4.

Instrumented eval (`franka_checkpoint_review.py --eval_target_shape
{d12,d20}`, num_envs=8, full 3-die topology, both distractors
real/active, checkpoint `model_5096.pt`):

| shape | envs_with_sustained_lift | falsification bar (>=6/8) | max_height_gain | max_consecutive_lifted_steps |
|-------|---------------------------|----------------------------|------------------|-------------------------------|
| d12   | **8/8**                   | PASS                       | 308.3-480.8mm    | 216-223 (of 249 analysis-window steps) |
| d20   | **8/8**                   | PASS                       | 334.3-481.2mm    | 217-222 (of 249 analysis-window steps) |

**Primary falsification check (pre-registered in the spec): both shapes
independently clear the 6/8 bar at 8/8 (100%)** — comfortably above the
75% floor, matching (not just barely clearing) the single-object 8/8
baseline and Stage D1's own 8/8/8/8 result. The hypothesis is NOT
falsified for either shape.

**Video inspection — both target-shape eval videos downloaded and
inspected frame-by-frame (`ffmpeg`-extracted stills at 2fps plus zoomed
crops around the gripper, not just the JSON summary), per this project's
verification standard and matching Stage D1's own practice:** across
every inspected timestamp for both the d12-target and d20-target runs,
the two distractor dice remain visibly in their fixed reset positions on
the table, completely undisturbed, while the gripper carries the
(white-on-white, visually hard to distinguish from the gripper itself at
this camera distance, but ground-truth-confirmed via the height
instrumentation) target die through a large elevation arc. **No episode
in any inspected frame showed the gripper grasping or disturbing a
distractor die** — this is a distinct, explicitly-checked-for failure
mode (a "lifted the wrong die" outcome would be a materially different,
more concerning result than a simple discovery-rate shortfall) and it
was not observed. Note also a structural (not just visual) guarantee
worth recording: `franka_checkpoint_review.py`'s height instrumentation
reads `scene["object"]`'s own root position directly — `scene["object"]`
is structurally always the commanded target (a cfg-construction-time
scene-topology property, not a runtime-reassignable flag), a physically
separate rigid body from either distractor slot. A policy that grasped a
distractor instead of the target could not produce an inflated
`max_height_gain` reading for the target — so the 8/8 sustained-lift
metric itself already rules out "lifted a distractor, left the target
behind" as an explanation for these results, independent of the video
check; the video check confirms the complementary case (no distractor
disturbance alongside a correct target grasp), which the height metric
alone cannot see.

**Checkpoint:**
`gs://rl-manipulation-hks-runs/target-selection-clutter/joint-die-target-selection-d2/seed42/2026-07-19_21-08-07/model_5096.pt`.
Eval artifacts (videos + heights JSON, both shapes):
`gs://rl-manipulation-hks-runs/target-selection-clutter/eval-artifacts/joint-die-target-selection-d2/`.

**Cost:** ~$0.16 (the aborted cloud attempt — install completed but no
training ran before the SPOT preemption + quota contention forced a
switch to desktop; `scripts/check_cloud_state.sh` confirmed zero
instances/disks/snapshots left from this task afterward) + $0 (desktop —
both training and both eval runs). Desktop teardown verified:
`nvidia-smi --query-compute-apps` empty, no `tmux` server running,
`systemd-inhibit --list` clear of any `rl-gpu-job` guard,
`scripts/check_gpu_availability.sh` reports AVAILABLE. **Combined Task
4+5+6 total: ≈$1.35 of the plan's $5 cap for Tasks 4-6** — well under,
no controller notification needed per the plan's own cost-cap clause.

No code bug found or fixed during this task — Task 1-3's wiring worked
exactly as built for the D2 variant with no changes needed, matching
Stage D1's own finding.

## FINAL VERDICT — target-selection-among-distractor-dice experiment (Tasks 0-6 / Stages SO/D1/D2, 2026-07-19)

**The pre-registered falsifiable hypothesis PASSES for both shapes at the
primary bar.** Starting from the finished single-object unified d12/d20
policy (`model_2998.pt`, 8/8 both shapes with exactly one die in the
scene), extending it with (a) two new always-present distractor scene
entities plus one new additive `distractor_distance_summary` observation
term (DexSinGrasp's `d_t^S` mechanism, K=2, hard-zero-padded per
curriculum stage) and (b) a 3-stage distractor-count curriculum (SO: 0 ->
D1: 1 -> D2: 2 active distractors, each stage checkpoint-resumed from the
prior stage), while leaving the reward function and target-identification
mechanism (the existing, unchanged `object_position` term) completely
untouched, **preserved the single-object 8/8 discovery rate for both d12
and d20 all the way through the full 2-active-distractor target
configuration** — not just above the pre-registered 6/8 floor, but an
exact, undegraded match to the single-object baseline.

**Per-stage results, reported exactly as observed (not averaged, not
just the final headline number), each shape separately:**

| stage | active distractors | d12 | d20 | gate/bar | verdict |
|---|---|---|---|---|---|
| SO (original, from-scratch) | 0 (parked) | 0/8 | 0/8 | >=7/8 internal sanity gate | FAIL — confounded, see below |
| SO (corrected, warm-started from `model_2998.pt`) | 0 (parked) | 8/8 | 7/8 | >=7/8 internal sanity gate | PASS |
| D1 | 1 (real) | 8/8 | 8/8 | intermediate data point, no formal bar | matches/exceeds single-object 8/8 baseline |
| D2 (primary falsification check) | 2 (real, both slots) | **8/8** | **8/8** | >=6/8 (75%) primary bar | **PASS — hypothesis NOT falsified** |

**The Stage SO from-scratch confound and its resolution is itself a real
finding, not just a false start:** the plan's own design forced a
from-scratch (not resumed) Stage SO, since the new 43-dim observation
schema is incompatible with the 41-dim `model_2998.pt` checkpoint and no
cross-dimensionality warm-start mechanism existed in this codebase before
this experiment. That from-scratch run got 0/8 both shapes — a
"reach but never grasp" pattern indistinguishable from this project's
own long-documented d12/d20 cold-start grasp-*discovery* difficulty (the
exact barrier the specialist -> distill -> RL-fine-tune pipeline in
[[unified-multi-die-specialist-distillation]] exists to route around),
confounding "did the schema extension break something" with "does plain
from-scratch PPO ever discover these grasps at all." The fix — a
weight-surgery script
(`scripts/extend_checkpoint_observation_dims.py`) that extends
`model_2998.pt`'s 41-dim first-layer weights to 43 dims by copying the
41 existing columns unchanged and randomly initializing only the 2 new
(always-zero-at-Stage-SO) columns — is mathematically guaranteed
lossless at Stage SO specifically, and was verified bit-for-bit identical
(0.0 max abs diff) against the real checkpoint before any training spend,
with a negative-control check (nonzero values in the new columns) that
did produce a large diff, confirming the verification wasn't vacuous.
This resolved the confound cleanly: the corrected, warm-started Stage SO
passed cleanly (d12 8/8, d20 7/8), confirming the schema/scene extension
itself was never broken — the original 0/8 result really was the
pre-existing cold-start difficulty, not a new defect.

**Not measurement artifacts — checked past the summary numbers at every
stage, per this project's own repeated settle-detection-bug discipline:**
every stage's eval used the current (post-`977a748`) MIN-over-fixed-
early-window settle-detection logic; `max_height_gain` was large and
decisive at every passing stage (SO: ~363-439mm; D1: ~365-457mm; D2:
308.3-481.2mm — all far above the 40mm lift threshold) and
`max_consecutive_lifted_steps` held for the large majority of each
analysis window (D1: 215-222/239; D2: 216-223/249) at every stage, not a
brief threshold-crossing blip. Eval videos were downloaded and inspected
frame-by-frame at D1 and D2 (not just the JSON) — D1 confirmed a clean
single-target grasp with the distractor undisturbed; **D2 additionally
and explicitly checked for, and did not find, any "grasped the wrong
die" episode** across either target shape's full eval video, the
specific additional failure mode this final stage's real 2-distractor
configuration made newly possible to observe.

**Cost:** ≈$1.35 total across Tasks 4/5/6 (SO's two attempts ≈$0.83 +
D1 ≈$0.36 + D2 ≈$0.16 aborted-cloud/desktop-completed), of the plan's
$5 cap for these three tasks combined — well under, matching this
project's now-consistent pattern of desktop dispatch bringing real cloud
spend close to $0 whenever the desktop is actually available.

**Checkpoints (final, per stage):**
- SO (corrected, warm-started): `gs://rl-manipulation-hks-runs/target-selection-clutter-stageso-warmstart/joint-die-target-selection-so/seed42/2026-07-19_22-52-41/model_3297.pt`
- D1: `gs://rl-manipulation-hks-runs/target-selection-clutter/joint-die-target-selection-d1/seed42/2026-07-19_23-58-28/model_4097.pt`
- D2 (final, this experiment's own end state): `gs://rl-manipulation-hks-runs/target-selection-clutter/joint-die-target-selection-d2/seed42/2026-07-19_21-08-07/model_5096.pt`

**Bottom line:** a single unified policy that grasps-and-lifts a
commanded d12 or d20 die when 2 other dice (independently drawn from the
same 2-shape population, same-shape and cross-shape pairings both pooled)
are simultaneously present in the scene, with no degradation from the
single-object 8/8 baseline and no evidence of the policy ever grasping
the wrong entity, is real and checkpointed. Curriculum + a fixed-size
zero-padded distractor-distance observation term, transplanted from
DexSinGrasp's own state-based-teacher formulation despite that paper's
materially different setting (dexterous multi-finger hand, heaped/
occluding clutter, vs. this project's parallel-jaw gripper, flat
non-occluding tabletop), was sufficient on its own — no
distractor-avoidance reward term was needed, and this experiment
deliberately did not add one, per its own scope. Since the primary bar
passed cleanly (not falsified), the escalation path this project's own
spec pre-registered for a falsification (a Deep-Sets/attention
architecture over distractor state, or a distractor-avoidance reward
term) is not needed and was not attempted.

## Open, not yet decided

- Was: whether to warm-start Stage SO from `model_2998.pt` despite the
  dimensionality mismatch, accept the from-scratch difficulty as the
  explanation, or find some other approach. **Decided and executed above
  (option (a), partial-weight warm start) — nothing open on this question
  anymore.**
- **d8/d10 as distractors or targets** — still gated on those shapes ever
  achieving real single-object discovery first (unresolved, per
  [[unified-multi-die-specialist-distillation]]'s own FINAL VERDICT); out
  of scope for any follow-on to this experiment until that's resolved.
- **3+ distractors, heaped/occluding arrangements, or any singulation
  mechanism** — this experiment only tested flat, non-overlapping,
  at-most-2-distractor clutter; whether the same curriculum+observation
  mechanism scales further, or whether singulation-specific techniques
  become necessary once occlusion is possible, is a genuinely open
  question for a future follow-on, not assumed either way by this
  experiment's own clean pass.
- **Multi-seed replication** — this experiment used a single seed
  (seed42) throughout, matching the checkpoint it started from; the
  spec's own scope explicitly deferred multi-seed replication to a
  follow-on now that the single-seed result is positive and (per the
  spec's own stated condition) warrants confirming its robustness.
- **A project-wide `GPUS_ALL_REGIONS` GCP quota of 1** was discovered
  during Task 6 (see that section above) — a real, previously-
  undocumented constraint on how many Senior workstreams can use cloud
  GPU dispatch simultaneously. Flagged to the controller, not resolved
  here (a Console-UI-only quota increase request, same mechanism as the
  original grant).

## Gripper actuator low-pass-filtering check (2026-07-19) — ruled out

A candidate alternative explanation for the "reach but never grasp"
pattern was checked and ruled out: Neunert et al. (DeepMind, CoRL 2019,
arXiv:2001.00449) report that slow gripper actuators can act as a
low-pass filter that attenuates small-amplitude Gaussian exploration
noise before it ever produces a full-closure command — but only for a
**continuous velocity/position-delta** gripper action space. This
project's gripper action is `mdp.BinaryJointPositionActionCfg` (a hard
sign-threshold mapping any raw action to a FULL open or close joint
target, structurally the same fix Neunert et al. themselves recommend,
not their failure-mode baseline). A direct instrumented rollout of this
Stage SO checkpoint (`model_1499.pt`, d20 target, 8 envs, one full
episode) confirmed the raw gripper action was positive ("open",
magnitude +0.48 to +7.77) for 100% of steps in all 8 envs — never once
negative. The policy has confidently learned to keep the gripper open
throughout the episode; nothing is being filtered out by actuator
dynamics. This is a reward/exploration-discovery problem, consistent
with the exploration-reward research track (H1/H2/H3) in
`docs/superpowers/specs/research/2026-07-19-exploration-reward-expansion-
literature.md`, whose addendum has the full writeup. Diagnostic script:
`scripts/_diag_gripper_lowpass_check.py`.

## Related

[[unified-multi-die-specialist-distillation]] — this experiment's own
direct predecessor and checkpoint source.
