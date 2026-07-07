# Experiment 11 Training Run Report: Task-Space IK-Driven Action

**Date:** 2026-07-06
**Log Directory:** `/home/saps/projects/rl/logs/train/2026-07-06_22-57-31`
**Training Status:** COMPLETED (1500 iterations)

## This is the corrected, authoritative Experiment 11 result

An earlier full-length run (`logs/train/2026-07-06_22-28-15/`, recorded in
commit [933de42](https://github.com/hksaperstein/rl/commit/933de42f763ab8d2c890cbd9a319d4d7737b61ad) and re-flagged in
[806a6bf](https://github.com/hksaperstein/rl/commit/806a6bf5500dbdfe0702b44d6e4ce7e733226acf)) diverged: the PPO critic's `Mean
value_function loss` exploded exponentially starting at iteration 67/1500
and reached ~5.2e23 by the final iteration, meaning roughly 95% of that
run's policy updates were driven by garbage advantage signal. The fix —
`clip_actions=5.0` added to the new `Ar4PickPlaceTaskspacePPORunnerCfg`
(`tasks/ar4/pickplace_taskspace_env_cfg.py`) — was committed in
[16fb16a](https://github.com/hksaperstein/rl/commit/16fb16abf838976ae660c4eafc2284caa39a534d) and verified on a 300-iteration diagnostic
run (`logs/train/2026-07-06_22-49-44/`) that showed the value-function loss
staying bounded (max 7.88, an isolated 2-iteration transient spike,
immediate recovery).

This report covers the full 1500-iteration re-run at
`logs/train/2026-07-06_22-57-31/`, launched with the fix in place, and
supersedes the diverged run's conclusions entirely. Full diagnostic detail
for the original divergence lives in git history (commits above) and the
`.superpowers/sdd/progress.md` ledger, not repeated here.

## Verification Results

### Checkpoint Integrity
- **Config `save_interval`:** 50
- **Expected checkpoints:** 1500 / 50 + 1 = 31
- **Actual checkpoints:** 31 — confirmed
- **Checkpoint iterations:** 0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1499 — confirmed
- **model_1499.pt exists:** YES (1,178,293 bytes, modified 2026-07-06 23:13) — confirmed

### TensorBoard Event File
- **Event file:** `/home/saps/projects/rl/logs/train/2026-07-06_22-57-31/events.out.tfevents.1783393056.home.698865.0`
- **Size:** 1.9M
- **Modification time:** 2026-07-06 23:13
- **model_1499.pt modification time:** 2026-07-06 23:13
- **Status:** Event file mtime matches checkpoint completion time — confirmed
- **No Python traceback/exception found in the full stdout log** (`/tmp/exp11_train_v2_stdout.log`)

## Critical Check: `Mean value_function loss` Across the FULL 1500-Iteration Run

This is the single most important check in this task, since the entire point
of this re-run is confirming the `clip_actions=5.0` fix holds for the whole
run, not just the 300-iteration diagnostic.

Extracted via TensorBoard's `Loss/value_function` scalar (1500 points,
authoritative source; cross-checked against a `grep` of stdout for `Mean
value_function loss`, which produced 1499 matching print lines — rsl_rl
appears to skip logging on the very first iteration, a benign off-by-one in
the print statement, not a data gap: the tensorboard scalar series itself
has all 1500 points, steps 0-1499).

- **Max value across all 1500 iterations: 7.8763**, occurring at iteration 47
- First 10 values: 0.0026, 0.0010, 0.00017, 0.00014, 0.00011, 0.00008, 0.00007, 0.00004, 0.00003, 0.00003 (normal, decaying range)
- Last 5 values: 0.00075, 0.00077, 0.00087, 0.00100, 0.00070 (normal range)
- **Only one jump exceeding 50x between consecutive iterations** in the entire run: iteration 45 (0.0000680) -> iteration 46 (1.1522) -> iteration 47 (7.8763), followed immediately by iteration 48 dropping back to 0.0058 and staying in the 0.0003-0.001 range for the rest of the run.
- **No compounding/exponential pattern found anywhere in the 1500-iteration run.** This is exactly the same isolated 2-iteration transient spike (same rough magnitude, same iteration range) observed in the 300-iteration diagnostic, not a recurrence of the original divergence (which grew from 1.5 to 5.2e23 and never recovered over ~1430 consecutive iterations).

**Conclusion: the `clip_actions=5.0` fix holds for the entire 1500-iteration run.** The critic's value-function loss stayed bounded throughout, with a single benign transient spike and immediate recovery — never the sustained exponential blow-up seen in the original diverged run.

## Scalar Trajectories (All 1500 Iterations)

### 1. Episode_Reward/path_proximity_bonus
**Summary:**
- Total data points: 1500
- First: iteration=0, value=0.000366
- Last: iteration=1499, value=0.059629
- Min: 0.000366
- Max: 0.060834

**Sample trajectory (every ~150 iterations):**
```
iteration=   0, value=0.000366
iteration= 150, value=0.008981
iteration= 300, value=0.034842
iteration= 450, value=0.050737
iteration= 600, value=0.058261
iteration= 750, value=0.058082
iteration= 900, value=0.057440
iteration=1050, value=0.059638
iteration=1200, value=0.059459
iteration=1350, value=0.059314
iteration=1499, value=0.059629
```

### 2. Episode_Reward/gripper_schedule_bonus
**Summary:**
- Total data points: 1500
- First: iteration=0, value=0.003133
- Last: iteration=1499, value=0.079407
- Min: 0.003133
- Max: 0.099485

**Sample trajectory (every ~150 iterations):**
```
iteration=   0, value=0.003133
iteration= 150, value=0.092000
iteration= 300, value=0.060175
iteration= 450, value=0.062573
iteration= 600, value=0.076565
iteration= 750, value=0.075823
iteration= 900, value=0.077087
iteration=1050, value=0.080311
iteration=1200, value=0.080199
iteration=1350, value=0.080860
iteration=1499, value=0.079407
```

### 3. Episode_Reward/antipodal_grasp_bonus
**Summary:**
- Total data points: 1500
- First: iteration=0, value=0.000000
- Last: iteration=1499, value=0.018815
- Min: 0.000000
- Max: 0.033995
- Non-zero occurrences: 1374 / 1500 (91.6%)

**Sample trajectory (every ~150 iterations):**
```
iteration=   0, value=0.000000
iteration= 150, value=0.000000
iteration= 300, value=0.000402
iteration= 450, value=0.003659
iteration= 600, value=0.007363
iteration= 750, value=0.008593
iteration= 900, value=0.008265
iteration=1050, value=0.015760
iteration=1200, value=0.016255
iteration=1350, value=0.016963
iteration=1499, value=0.018815
```

### 4. Episode_Reward/stillness_penalty
**Summary:**
- Total data points: 1500
- First: iteration=0, value=0.000000
- Last: iteration=1499, value=-0.002533
- Min: -0.011242
- Max: 0.000000
- Non-zero occurrences: 1470 / 1500 (98.0%)

**Sample trajectory (every ~150 iterations):**
```
iteration=   0, value=0.000000
iteration= 150, value=-0.000044
iteration= 300, value=-0.001577
iteration= 450, value=-0.001808
iteration= 600, value=-0.000494
iteration= 750, value=-0.000506
iteration= 900, value=-0.003785
iteration=1050, value=-0.004317
iteration=1200, value=-0.002705
iteration=1350, value=-0.001566
iteration=1499, value=-0.002533
```

### 5. Episode_Termination/cube_reached_goal
**Summary:**
- Total data points: 1500
- First: iteration=0, value=0.002574
- Last: iteration=1499, value=0.010223
- Min: 0.000804
- Max: 0.018860

**Sample trajectory (every ~150 iterations):**
```
iteration=   0, value=0.002574
iteration= 150, value=0.004252
iteration= 300, value=0.010651
iteration= 450, value=0.014150
iteration= 600, value=0.011108
iteration= 750, value=0.015859
iteration= 900, value=0.009674
iteration=1050, value=0.015198
iteration=1200, value=0.011017
iteration=1350, value=0.017151
iteration=1499, value=0.010223
```

## Key Comparison: Experiment 11 (Corrected) vs Experiment 10 (Final Values Only)

Per the correction protocol already established in this project (a prior
report incorrectly compared a cumulative/summed quantity against a
single-episode value — the comparisons below are strictly final-iteration
snapshot vs. final-iteration snapshot).

### Antipodal Grasp Bonus
- **Experiment 10 final value:** 0.000000
- **Experiment 11 (corrected) final value:** 0.018815
- **Change:** now nonzero. The final-iteration antipodal grasp bonus is no longer exactly zero, unlike both Experiment 10 and the original (diverged) Experiment 11 run. It is also nonzero across 91.6% of all 1500 logged iterations, not a rare/scattered occurrence.

### Cube Reached Goal
- **Experiment 10 final value:** 0.002848
- **Experiment 11 (corrected) final value:** 0.010223
- **Change:** +0.007375 (roughly 3.6x Experiment 10's final value).

Both metrics improved at the final iteration relative to Experiment 10. **No success/failure conclusion about cube-lifting capability is drawn here** — that requires evaluation video inspection, a separate, later task not in scope for this report.

## Assessment

With the `clip_actions=5.0` fix in place, the critic's value-function loss
stayed bounded for the entire 1500-iteration run (max 7.8763, one isolated
transient spike, no exponential compounding), giving a training run whose
policy-gradient updates were driven by well-behaved advantage estimates
throughout — unlike the original run, where roughly 95% of iterations were
compromised. Under this trustworthy run, the task-space IK-driven action
design's final-iteration antipodal grasp bonus and cube-reached-goal
termination rate both improved over Experiment 10, and the antipodal grasp
bonus is no longer identically zero. Whether this translates into the
policy actually lifting the cube in evaluation rollouts is left to the
separate, later video-inspection task.
