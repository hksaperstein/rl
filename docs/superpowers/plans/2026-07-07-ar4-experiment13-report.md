# Experiment 13 Training Run Report: Residual Action Term Over Classical Waypoint Controller

**Date:** 2026-07-07
**Log Directory:** `/home/saps/projects/rl/logs/train/2026-07-07_10-22-46`
**Training Status:** COMPLETED (1500 iterations)

## Overview

Experiment 13 tests a residual-action formulation: the policy learns a small
corrective delta on top of a classical waypoint (task-space IK) controller,
rather than the raw task-space delta action used in Experiment 11/12. This is
implemented via `Ar4PickPlaceResidualEnvCfg`
(`tasks/ar4/pickplace_residual_env_cfg.py`) and the `--residual` flag wired
into `scripts/train.py`/`scripts/eval_loop.py`. The 300-iteration diagnostic
(Task 4, `logs/train/2026-07-07_10-16-53/`) passed its gate checks —
`Loss/value_function` max 0.17 across the diagnostic window, healthier than
the reference (Experiment 12) run. This report covers the full 1500-iteration
run, launched with the residual action term in place, no other changes.

## Verification Results

### Checkpoint Integrity
- **Config `save_interval`:** 50
- **Expected checkpoints:** 1500 / 50 + 1 = 31
- **Actual checkpoints:** 31 — confirmed
- **Checkpoint iterations:** 0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1499 — confirmed
- **model_1499.pt exists:** YES (1,178,293 bytes, modified 2026-07-07 10:37) — confirmed

### TensorBoard Event File
- **Event file:** `/home/saps/projects/rl/logs/train/2026-07-07_10-22-46/events.out.tfevents.1783434171.home.15785.0`
- **Size:** 1.9M (1,894,806 bytes)
- **Modification time:** 2026-07-07 10:37
- **model_1499.pt modification time:** 2026-07-07 10:37
- **Status:** Event file mtime matches checkpoint completion time — confirmed. Run started 10:22:46, completed ~10:37 (~15 minutes wall clock), consistent with prior 1500-iteration runs at `num_envs=4096`.

## Critical Check: `Loss/value_function` Across the FULL 1500-Iteration Run

Extracted via TensorBoard's `Loss/value_function` scalar (1500 points):

- **Max value across all 1500 iterations: 0.172533**
- **Min value: 0.0000015**
- First value (iteration 0): 0.002782
- Sampled trajectory (every ~150 iterations): 0.002782, 0.000353, 0.000517, 0.000802, 0.000460, 0.000341, 0.000310, 0.000414, 0.000300, 0.000356, 0.000342 (last, iteration 1499)
- **No sustained exponential growth.** The loss sits in the 0.0003–0.0008 range for the vast majority of the run after an early settling period, with a single max of 0.172533 consistent with (not exceeding) the 300-iteration diagnostic's own max of 0.17.

**Conclusion: `Loss/value_function` stayed bounded for the entire 1500-iteration run.** The full run's max (0.172533) matches the diagnostic run's max, indicating the diagnostic accurately captured the run's worst-case value-function behavior and no new instability emerged in the remaining 1200 iterations.

## Scalar Trajectories (All 1500 Iterations)

### 1. Episode_Reward/path_proximity_bonus
**Summary:**
- Total data points: 1500
- First: iteration=0, value=0.000423
- Last: iteration=1499, value=0.067860
- Min: 0.000151
- Max: 0.068514

**Sample trajectory (every ~150 iterations):**
```
iteration=   0, value=0.000423
iteration= 150, value=0.020245
iteration= 300, value=0.045743
iteration= 450, value=0.056460
iteration= 600, value=0.063410
iteration= 750, value=0.067098
iteration= 900, value=0.064658
iteration=1050, value=0.065323
iteration=1200, value=0.065656
iteration=1350, value=0.065645
iteration=1499, value=0.067860
```

### 2. Episode_Reward/gripper_schedule_bonus
**Summary:**
- Total data points: 1500
- First: iteration=0, value=0.003125
- Last: iteration=1499, value=0.086488
- Min: 0.003125
- Max: 0.099461

**Sample trajectory (every ~150 iterations):**
```
iteration=   0, value=0.003125
iteration= 150, value=0.083752
iteration= 300, value=0.061310
iteration= 450, value=0.063323
iteration= 600, value=0.078086
iteration= 750, value=0.085021
iteration= 900, value=0.083721
iteration=1050, value=0.084366
iteration=1200, value=0.083101
iteration=1350, value=0.083971
iteration=1499, value=0.086488
```

### 3. Episode_Reward/antipodal_grasp_bonus
**Summary:**
- Total data points: 1500
- First: iteration=0, value=0.000000
- Last: iteration=1499, value=0.000140
- Min: 0.000000
- Max: 0.006542
- Non-zero occurrences: 1206 / 1500 (80.4%)

**Sample trajectory (every ~150 iterations):**
```
iteration=   0, value=0.000000
iteration= 150, value=0.000125
iteration= 300, value=0.002850
iteration= 450, value=0.000411
iteration= 600, value=0.000130
iteration= 750, value=0.000020
iteration= 900, value=0.000000
iteration=1050, value=0.000000
iteration=1200, value=0.000156
iteration=1350, value=0.000280
iteration=1499, value=0.000140
```

### 4. Episode_Reward/stillness_penalty
**Summary:**
- Total data points: 1500
- First: iteration=0, value=0.000000
- Last: iteration=1499, value=-0.002095
- Min: -0.022942
- Max: 0.000000
- Non-zero occurrences: 1488 / 1500 (99.2%)

**Sample trajectory (every ~150 iterations):**
```
iteration=   0, value=0.000000
iteration= 150, value=-0.002689
iteration= 300, value=-0.004136
iteration= 450, value=-0.006433
iteration= 600, value=-0.005223
iteration= 750, value=-0.005332
iteration= 900, value=-0.003467
iteration=1050, value=-0.003226
iteration=1200, value=-0.003796
iteration=1350, value=-0.002347
iteration=1499, value=-0.002095
```

### 5. Episode_Termination/cube_reached_goal
**Summary:**
- Total data points: 1500
- First: iteration=0, value=0.002574
- Last: iteration=1499, value=0.006012
- Min: 0.001475
- Max: 0.019826

**Sample trajectory (every ~150 iterations):**
```
iteration=   0, value=0.002574
iteration= 150, value=0.004679
iteration= 300, value=0.009145
iteration= 450, value=0.017131
iteration= 600, value=0.010732
iteration= 750, value=0.007853
iteration= 900, value=0.006663
iteration=1050, value=0.008341
iteration=1200, value=0.010579
iteration=1350, value=0.008341
iteration=1499, value=0.006012
```

## Key Comparison: Experiment 13 vs Experiment 12 (Final Values Only)

Per the established correction protocol (final-iteration snapshot vs.
final-iteration snapshot only — not cumulative or mid-run comparisons).

### Antipodal Grasp Bonus
- **Experiment 12 final value:** 0.012777
- **Experiment 13 final value:** 0.000140
- **Change:** -0.012637 (-98.9%)

### Stillness Penalty
- **Experiment 12 final value:** -0.001857
- **Experiment 13 final value:** -0.002095
- **Change:** -0.000238 (12.8% more negative, i.e. larger penalty incurred)

### Path Proximity Bonus
- **Experiment 12 final value:** 0.064421
- **Experiment 13 final value:** 0.067860
- **Change:** +0.003439 (+5.3%)

### Cube Reached Goal
- **Experiment 12 final value:** 0.010773
- **Experiment 13 final value:** 0.006012
- **Change:** -0.004761 (-44.2%)

## Assessment

The scalar comparison against Experiment 12 is mixed: `path_proximity_bonus`
improved slightly (+5.3%), while `antipodal_grasp_bonus` dropped sharply
(-98.9%, though it remains non-zero on 80.4% of iterations, meaning the
grasp condition is still being reached — just contributing far less
accumulated reward per episode than in Experiment 12), `stillness_penalty`
worsened modestly (12.8% more negative), and `cube_reached_goal` dropped
substantially (-44.2%).

Per this project's established correction protocol (Experiment 12's own
report initially misread a similar `antipodal_grasp_bonus` drop as a
"hypothesis incorrect" conclusion, which the controller subsequently
corrected after finding the underlying behavior had plausibly improved,
not regressed, once read against the other four scalars together) — **this
report does not draw a final success/failure conclusion from the scalars
alone.** The action-space change in Experiment 13 (residual over a classical
waypoint controller, replacing Experiment 12's raw task-space delta action)
is architecturally different enough from Experiment 12 that these proxy
reward terms may not be directly comparable in the same way episode-to-episode
scalars were between Experiments 11 and 12 — e.g. a residual policy could be
producing shorter, more efficient antipodal holds (lower accumulated
`antipodal_grasp_bonus`, similar to the Experiment 11→12 pattern) or could
genuinely be grasping less often. The `cube_reached_goal` drop is a more
directly outcome-oriented metric and is a larger relative change than the
Experiment 11→12 gripper/antipodal shifts were, so it is a real point of
concern to flag — but the same "proxy term can drop while true behavior
improves" hazard makes any success/failure call here premature without
Task 6's video inspection.

The value function loss stayed bounded throughout (max 0.172533, no
exponential divergence; matches the 300-iteration diagnostic's own max of
0.17), confirming the residual action term did not introduce optimizer
instability at full scale.

**Next steps:** Task 6's video inspection is required to determine whether
the residual-action policy is genuinely grasping/lifting/placing the cube
more, less, or differently than Experiment 12's raw task-space policy — the
scalars alone, per the above, are ambiguous.
