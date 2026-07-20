# Target-selection-among-distractor-dice experiment (2026-07-19 -> in progress)

**Status: Stage SO gate PASSED (d12 8/8, d20 7/8) and Stage D1 (1 active
distractor) also PASSED cleanly — d12 8/8, d20 8/8 (2026-07-19).** The
originally-reported Stage SO 0/8-both-shapes from-scratch result below
was confounded (see "Task 4 corrected" section) and has been superseded
by a partial-weight warm-started retrain that isolates the question the
gate was actually meant to answer. Stage D1, resumed from that corrected
Stage SO checkpoint, cleared its own eval at 8/8 for both shapes — see
"Task 5" section below. Stage D2 (plan Task 6, the primary falsification
check) is next. This is still an interim entry (plan Tasks 4-5 only); the
closing verdict lands at the plan's Task 7.

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

## Open, not yet decided

Was: whether to warm-start Stage SO from `model_2998.pt` despite the
dimensionality mismatch, accept the from-scratch difficulty as the
explanation, or find some other approach. **Decided and executed above
(option (a), partial-weight warm start) — nothing open on this question
anymore.**

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
