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
