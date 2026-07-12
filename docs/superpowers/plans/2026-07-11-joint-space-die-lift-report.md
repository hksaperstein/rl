# Joint-space (no-IK) d20 die-lift — run report

Spec: `docs/superpowers/specs/2026-07-11-joint-space-die-lift-design.md`
Plan: `docs/superpowers/plans/2026-07-11-joint-space-die-lift.md`

## Task 3: 300-iteration diagnostic readout (2026-07-12)

Run: `logs/train_franka_jointdie/2026-07-12_06-39-53/` (4096 envs, 300
iterations, ~80k steps/s, DIAG_EXIT=0, model_299.pt on disk). All numbers
read directly from the TensorBoard event file via
`event_accumulator` (all 300 points, not samples).

### Gate checks (plan Task 3 Step 3)

- `Loss/value_function`: **bounded** — min 1.2e-4, max 0.0239 (at it0),
  0 points above 1e3. No Experiment-11-style divergence.
- `Episode_Reward/reaching_object`: **clearly rose** — 6.1e-5 (it0) →
  0.273 (it75) → 0.687 (it150). Not flat-at-noise. Gate PASSES as
  written; proceeding to Task 4.

### Authoritative success metric (named per plan, BEFORE Task 4)

This env logs **no** explicit success/goal-reached termination scalar
(terminations: `time_out`, `object_dropping` only — verified against the
full scalar tag list). The authoritative metric for the Task 4 verdict is:

**`Metrics/object_pose/position_error`** — mean object-to-commanded-goal
distance (meters), logged by the stock lift env's command manager.
Do-nothing baseline ≈ 0.21–0.22 (it0 value: 0.216, i.e. die resting on
table vs. sampled air goals). SUCCESS = sustained, clearly-trending fall
substantially below that resting baseline (goals are predominantly in the
air, so a policy that never lifts cannot drive this below baseline).
Corroborating (not authoritative): `Episode_Reward/object_goal_tracking`
and `object_goal_tracking_fine_grained`.

### Watch items (not gate failures, recorded for Task 4 interpretation)

1. **Second-half regression**: `Train/mean_reward` 4.09 (it150) → 1.60
   (it299); `reaching_object` 0.687 (it150) → 0.190 (it299). Not
   divergence (VF loss stays ≤ ~1e-3, `Policy/mean_noise_std` stable
   0.84–1.08). Cause unknown; NOT the stock curriculum (both
   `Curriculum/*` weights still at -0.0001 — the switch fires at
   ~iteration 417, beyond this diagnostic).
2. **`position_error` currently WORSE than do-nothing**: 0.216 (it0) →
   ~0.36–0.38 plateau. The policy interacts with/pushes the die but not
   toward goals at this stage. If this hasn't fallen decisively below the
   ~0.21 baseline by iteration 1500, the metric criterion FAILS and the
   spec's pre-authorized DexCube fallback rung fires (asset-vs-recipe
   isolation).
3. **`lifting_object` constant ~0.118–0.126 floor from it0**: artifact,
   not learning signal — the die spawns at z=0.055, above the stock
   `minimal_height=0.04` lifted-threshold, so every episode's first
   settle-steps score as "lifted" regardless of policy. Watch for growth
   ABOVE this floor, not above zero.
4. `Episode_Termination/object_dropping` spiked to ~2–6% mid-run,
   back to 0 by it299 (policy stopped knocking the die off the table).

## Task 4 Step 2: full-run scalar readout (2026-07-12)

Run: `logs/train_franka_jointdie/2026-07-12_06-56-02/` (fresh 1500-iter
run, 4096 envs, 31 checkpoints incl. model_1499.pt, FULL_EXIT=0,
~32 min). All 1500 points read from the event file.

- `Loss/value_function`: bounded all run — min 1.0e-4, max 0.171, 0
  points >1e3. Divergence criterion clean.
- `Episode_Reward/reaching_object`: slow steady climb 6.1e-5 → 0.120
  (it375) → 0.232 (it750) → 0.346 (it1125) → 0.391 (it1499). Notably
  does NOT reproduce the 300-iter diagnostic's fast rise to 0.687 by
  it150 (different seed/rollout trajectory; both runs end well below
  the ik-cube recipe's typical converged reach).
- `Episode_Reward/lifting_object`: **flat at the 0.12 spawn-artifact
  floor for the entire 1500 iterations** (min 0.1175, max 0.148). Lift
  never emerged.
- `object_goal_tracking`: flat ~0.02; `fine_grained`: ≤4.6e-5. Nothing.
- **Authoritative metric `Metrics/object_pose/position_error`: FAILED.**
  it0=0.216 (do-nothing baseline), rises to ~0.35 and stays there;
  last-100 mean 0.331; series min 0.2028 (early noise only). Never
  decisively below baseline — the policy moves the die but not toward
  goals.
- `Train/mean_reward`: 0.70 → ~2.0 by it1499 with a deep transient to
  -10.65 (consistent with the stock curriculum's action-rate/joint-vel
  penalty step-up at ~it417, recovers after).
- `object_dropping`: ~0 essentially all run.

**Verdict so far: metric criterion FAILED → per the spec's verdict
protocol, the pre-authorized DexCube fallback rung fires (identical
joint-space config, object swapped back to the recipe's DexCube) to
isolate asset-vs-recipe. Die-run eval video still recorded (video
criterion documentation + instrumented height check).**

## Task 4 Step 3: die-run eval (10-episode-scale video + instrumentation)

Eval: `franka_checkpoint_review.py --variant joint-die` on model_1499.pt
(8 envs, 250 steps, whole-arm framing per standing instruction). Video:
`logs/videos/franka_checkpoint_review/franka_checkpoint_review_joint-die_model_1499-step-0.mp4`.
Instrumented heights: `heights_joint-die_model_1499.{json,npy}`.

- **Instrumented check: 0/8 envs sustained lift.** Per env: resting z
  0.0114m, max z exactly 0.0550m — the spawn height, i.e. the only
  above-threshold reading is the episode-start settle drop;
  max_consecutive_lifted_steps=1 for every env.
- **Video (controller-inspected frames across the episode):** arm
  reaches down toward the die early, then settles into a static low
  folded pose near the table for the rest of the episode. No grasp
  attempt, no lift, no carry toward the floating goal marker. Whole-arm
  framing confirmed.
- **Video criterion: FAILED** (consistent with the metric criterion).

## Task 4 Step 4: verdict

**Hypothesis (joint-space no-IK action formulation enables die lift):
FALSIFIED for the d20 die — but the fallback rung isolates the failure
to the ASSET, not the recipe.** Same config, only the object swapped
(joint-cube fallback, `logs/train_franka_jointcube/2026-07-12_07-31-58/`):

| | joint-die (d20) | joint-cube (DexCube) |
|---|---|---|
| lifting_object (wt 15) | 0.12 artifact floor, flat | 13.38 sustained |
| object_goal_tracking (wt 16) | ~0.02 noise | 12.29, climbing |
| position_error last-100 | 0.331 (worse than 0.216 baseline) | **0.105** |
| Train/mean_reward | ~2.0 | 138.4 |

The joint-space action recipe itself trains lift+carry decisively on the
recipe's own DexCube. The d20-specific failure candidates (NOT
investigated further per the spec's STOP-after-fallback rule): die size
(measured 2026-07-12 via scripts/_diag_dexcube_scale_check.py: d20
30.3mm vs DexCube 48.0mm effective — a 1.6x gap, smaller than first
assumed), near-spherical convex-hull
geometry (rolls when poked; low pinch stability), baked mass 0.01kg,
friction/material of the baked asset, spawn scale 0.001 pipeline.
Joint-cube eval confirmed (2026-07-12, later same morning): **8/8 envs
sustained lift** (instrumented, `heights_joint-cube_model_1499.json`),
video shows the cube held at the commanded air goals
(controller-inspected frames). Both rungs' evidence complete.
