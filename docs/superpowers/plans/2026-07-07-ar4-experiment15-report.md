# Experiment 15 Training Run Report: Ground/Base-Proximity Penalties + Higher Grasp Weight

**Date:** 2026-07-07
**Log Directory:** `/home/saps/projects/rl/logs/train/2026-07-07_13-06-23`
**Training Status:** COMPLETED (1500 iterations)

## Overview

Experiment 15 adds two new reward terms — `ground_penalty` (penalizes the
cube resting on the ground) and `base_proximity_penalty` (penalizes the cube
approaching the robot's base in a collapse-adjacent way) — plus a higher
antipodal-grasp weight, via `Ar4PickPlaceBaseProximityEnvCfg`
(`tasks/ar4/pickplace_baseproximity_env_cfg.py`) and the `--baseproximity`
flag wired into `scripts/train.py`/`scripts/eval_loop.py`. The 300-iteration
diagnostic (Task 4, verification-only, `logs/train/2026-07-07_12-55-23/`)
passed its 3 gate checks, including a `Loss/value_function` reading of a
single-iteration spike to 17.66 at step 39 that decayed back to baseline
within ~10-15 iterations and stayed flat/low (0.0003-0.001) for the
remaining ~250 diagnostic iterations with no recurrence. This magnitude was
flagged as unusually large relative to this project's prior diagnostics
(Experiment 13's full-run max was 0.17, Experiment 14's diagnostic max was
0.024) even though the shape matched the pattern this project has always
accepted. This report covers the full 1500-iteration run, launched with the
`--baseproximity` config in place, no other changes, and specifically
addresses whether that spike recurs or grows at full scale.

## Verification Results

### Checkpoint Integrity
- **Config `save_interval`:** 50
- **Expected checkpoints:** 1500 / 50 + 1 = 31
- **Actual checkpoints:** 31 — confirmed
- **Checkpoint iterations:** 0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1499 — confirmed
- **model_1499.pt exists:** YES (1,178,293 bytes, modified 2026-07-07 13:21) — confirmed

### TensorBoard Event File
- **Event file:** `/home/saps/projects/rl/logs/train/2026-07-07_13-06-23/events.out.tfevents.1783443989.home.38409.0`
- **Size:** 2.1M (2,110,556 bytes)
- **Modification time:** 2026-07-07 13:21
- **model_1499.pt modification time:** 2026-07-07 13:21
- **Status:** Event file mtime matches checkpoint completion time — confirmed. Run started 13:06:23, completed ~13:21 (~15 minutes wall clock), consistent with prior 1500-iteration runs at `num_envs=4096`.

## Critical Check: `Loss/value_function` Across the FULL 1500-Iteration Run

Extracted via TensorBoard's `Loss/value_function` scalar (1500 points):

- **Max value across all 1500 iterations: 17.657946**, at step 39
- **Min value: 0.0000097**
- First value (iteration 0): 0.002101
- Sampled trajectory (every ~150 iterations): 0.002101, 0.000171, 0.000414, 0.000501, 0.000651, 0.000641, 0.000551, 0.000826, 0.001005, 0.001427, 0.001553 (last, iteration 1499)

**Detail around the spike (steps 34-54):**
```
step 34: 0.0000139        step 39: 17.657946  <- spike
step 35: 0.0000157        step 40: 1.638131
step 36: 0.0000578        step 41: 0.553891
step 37: 0.0000210        step 42: 0.129577
step 38: 0.0000263        step 43: 0.037903   <- back under 0.05
step 44: 0.046798         step 49: 0.016433
step 45: 0.022411         step 50: 0.017353
step 46: 0.025442         step 51: 0.005955
step 47: 0.024030         step 52: 0.007725
step 48: 0.012186         step 53: 0.014081
```
Only **4 of 1500** logged iterations (steps 39-42) ever exceed 0.1, and only
**2** (steps 39-40) exceed 1.0. No other point in the entire 1500-iteration
run comes close to this magnitude — every other value stays below 0.05, and
the vast majority sit in the 0.0002-0.002 range.

**Direct comparison to the diagnostic's 17.66 spike:** the full run's max
(17.657946 at step 39) is effectively the same event, at the same step, as
the 300-iteration diagnostic's reported spike of 17.66 at step 39 — matching
to 4+ significant figures. This is not a new, larger, or recurring
instability: it is the identical single-iteration transient already seen
in the diagnostic, and the additional ~1200 iterations run here after the
diagnostic's window closed produced **no second occurrence of any spike
above 1.0**, no sustained climb, and no gradual upward trend in the loss's
baseline level (the loss stays in the same low ~0.0002-0.002 band from
iteration 50 through iteration 1499). This confirms — per the same
diagnostic-vs-full-run comparison pattern established in Experiment 14 (diagnostic
max 0.0244 vs. full-run max 0.024436) — that the diagnostic's spike was a
genuine one-off outlier and the full run's worst-case value-function
behavior, not the leading edge of an unresolved instability introduced by
the new `ground_penalty`/`base_proximity_penalty` reward terms.

**Caveat:** unlike Experiment 14 (whose diagnostic and full-run max were
both small, sub-0.03 numbers), this run's worst-case magnitude itself
(17.66) is far larger in absolute terms than any prior diagnostic or full
run in this project (Experiment 13's full-run max: 0.172533; Experiment
14's full-run max: 0.024436). The finding here is narrowly that the full
run does not make that number worse or more frequent than the diagnostic
already showed — it does not establish that a transient of this raw
magnitude is itself benign. That judgment is out of scope for scalar
extraction and is left to the controller's qualitative review.

## Scalar Trajectories (All 1500 Iterations)

### 1. Episode_Reward/path_proximity_bonus
**Summary:**
- Total data points: 1500
- First: iteration=0, value=0.000366
- Last: iteration=1499, value=0.061520
- Min: 0.000366
- Max: 0.063359
- Non-zero occurrences: 1500 / 1500

**Sample trajectory (every ~150 iterations):**
```
iteration=   0, value=0.000366
iteration= 150, value=0.015868
iteration= 300, value=0.031901
iteration= 450, value=0.046698
iteration= 600, value=0.055465
iteration= 750, value=0.055588
iteration= 900, value=0.059299
iteration=1050, value=0.060277
iteration=1200, value=0.059259
iteration=1350, value=0.060380
iteration=1499, value=0.061520
```

### 2. Episode_Reward/gripper_schedule_bonus
**Summary:**
- Total data points: 1500
- First: iteration=0, value=0.003133
- Last: iteration=1499, value=0.077319
- Min: 0.003133
- Max: 0.098947
- Non-zero occurrences: 1500 / 1500

**Sample trajectory (every ~150 iterations):**
```
iteration=   0, value=0.003133
iteration= 150, value=0.084810
iteration= 300, value=0.079098
iteration= 450, value=0.054174
iteration= 600, value=0.051056
iteration= 750, value=0.067533
iteration= 900, value=0.074632
iteration=1050, value=0.079687
iteration=1200, value=0.075882
iteration=1350, value=0.076062
iteration=1499, value=0.077319
```

### 3. Episode_Reward/antipodal_grasp_bonus
**Summary:**
- Total data points: 1500
- First: iteration=0, value=0.000000
- Last: iteration=1499, value=0.033199
- Min: 0.000000
- Max: 0.054559
- Non-zero occurrences: 1348 / 1500 (89.9%)

**Sample trajectory (every ~150 iterations):**
```
iteration=   0, value=0.000000
iteration= 150, value=0.000030
iteration= 300, value=0.000144
iteration= 450, value=0.001039
iteration= 600, value=0.006120
iteration= 750, value=0.007371
iteration= 900, value=0.023793
iteration=1050, value=0.013740
iteration=1200, value=0.018285
iteration=1350, value=0.030257
iteration=1499, value=0.033199
```
Unlike Experiment 14 (whose final antipodal_grasp_bonus dropped sharply to
0.000709), this run shows a clear, largely monotonic climb across the back
half of training, reaching 0.033199 by the final iteration — well above
both Experiment 12's final value (0.012777) and Experiment 14's near-zero
final value.

### 4. Episode_Reward/stillness_penalty
**Summary:**
- Total data points: 1500
- First: iteration=0, value=0.000000
- Last: iteration=1499, value=-0.002727
- Min: -0.015276
- Max: 0.000000
- Non-zero occurrences: 1469 / 1500 (97.9%)

**Sample trajectory (every ~150 iterations):**
```
iteration=   0, value=0.000000
iteration= 150, value=-0.000118
iteration= 300, value=-0.000272
iteration= 450, value=-0.001843
iteration= 600, value=-0.001886
iteration= 750, value=-0.000671
iteration= 900, value=-0.000926
iteration=1050, value=-0.002225
iteration=1200, value=-0.005715
iteration=1350, value=-0.001282
iteration=1499, value=-0.002727
```

### 5. Episode_Reward/ground_penalty (NEW TERM)
**Summary:**
- Total data points: 1500
- First: iteration=0, value=-0.004964
- Last: iteration=1499, value=-0.096263
- Min: -0.100000
- Max: -0.004964
- Non-zero occurrences: 1500 / 1500 (100.0%)

**Sample trajectory (every ~150 iterations):**
```
iteration=   0, value=-0.004964
iteration= 150, value=-0.099105
iteration= 300, value=-0.099490
iteration= 450, value=-0.098726
iteration= 600, value=-0.097679
iteration= 750, value=-0.097287
iteration= 900, value=-0.095821
iteration=1050, value=-0.096868
iteration=1200, value=-0.095090
iteration=1350, value=-0.095522
iteration=1499, value=-0.096263
```

**Nonzero-rate trend (first 150 vs. last 150 iterations), per design spec's success criterion:**
- First 150 iterations (steps 0-149): nonzero = 150/150 (100.0%)
- Last 150 iterations (steps 1350-1499): nonzero = 150/150 (100.0%)
- **The nonzero rate does not trend down — it is saturated at 100% in both windows and in every one of the ten 150-iteration windows across the full run.** This term fires on essentially every logged iteration from iteration 1 onward (iteration 0's smaller magnitude, -0.004964, reflects the initial reset state rather than a change in firing frequency).
- Windowed magnitude average (10 windows of 150 iterations): -0.09616, -0.09916, -0.09851, -0.09785, -0.09785, -0.09721, -0.09650, -0.09612, -0.09606, -0.09530. The magnitude jumps from a near-zero iteration-0 reading to roughly -0.096 to -0.099 within the first ~150 iterations, then stays essentially flat (a shallow ~4% decline from -0.099 to -0.095 across the remaining ~1350 iterations) for the rest of the run.
- **Factual conclusion:** the data does not show the hoped-for signal (a nonzero rate that trends down as the cube spends less time on the ground). The rate is already saturated at 100% almost immediately and never drops; the magnitude shows only a shallow late-run decline off an early plateau, not a continued downward trend.

### 6. Episode_Reward/base_proximity_penalty (NEW TERM)
**Summary:**
- Total data points: 1500
- First: iteration=0, value=0.000000
- Last: iteration=1499, value=-0.000600
- Min: -0.007639
- Max: 0.000000
- Non-zero occurrences: 1306 / 1500 (87.1%)

**Sample trajectory (every ~150 iterations):**
```
iteration=   0, value=0.000000
iteration= 150, value=-0.000240
iteration= 300, value=-0.000212
iteration= 450, value=-0.000507
iteration= 600, value=-0.004219
iteration= 750, value=-0.001048
iteration= 900, value=-0.003351
iteration=1050, value=-0.003516
iteration=1200, value=-0.001133
iteration=1350, value=-0.000903
iteration=1499, value=-0.000600
```

**Overall nonzero-rate, per design spec's success criterion:**
- Overall: 1306 / 1500 = **87.1%** nonzero — this is not a low rate.
- Windowed nonzero rate (10 windows of 150 iterations): 12.0%, 69.3%, 89.3%, 100.0%, 100.0%, 100.0%, 100.0%, 100.0%, 100.0%, 100.0%.
- **Factual conclusion:** the nonzero rate does not stay low across the run — it rises sharply from 12.0% in the first 150 iterations to 100.0% by iteration ~450, then remains saturated at 100.0% for the remaining ~1050 iterations (roughly 70% of the run). The overall 87.1% figure is an average that is pulled down only by the early low-firing window; for the large majority of training this term fires on every single logged iteration. This does not match the design spec's hoped-for pattern of a low, stable rate consistent with firing only on specific base-collapse-adjacent cases — instead, the term is firing on essentially every iteration for most of the run.

### 7. Episode_Termination/cube_reached_goal
**Summary:**
- Total data points: 1500
- First: iteration=0, value=0.002574
- Last: iteration=1499, value=0.017202
- Min: 0.001129
- Max: 0.022583
- Non-zero occurrences: 1500 / 1500

**Sample trajectory (every ~150 iterations):**
```
iteration=   0, value=0.002574
iteration= 150, value=0.006653
iteration= 300, value=0.006938
iteration= 450, value=0.014170
iteration= 600, value=0.016724
iteration= 750, value=0.007426
iteration= 900, value=0.014282
iteration=1050, value=0.017314
iteration=1200, value=0.019704
iteration=1350, value=0.018575
iteration=1499, value=0.017202
```

## Key Comparison: Experiment 15 vs Experiment 12 and Experiment 14 (Final Values Only)

Per the established correction protocol (final-iteration snapshot vs.
final-iteration snapshot only — not cumulative, mid-run, or diagnostic-window
comparisons).

### Antipodal Grasp Bonus (vs. Experiment 12)
- **Experiment 12 final value:** 0.012777
- **Experiment 15 final value:** 0.033199
- **Change:** +0.020422 (+159.8%)

### Stillness Penalty (vs. Experiment 12)
- **Experiment 12 final value:** -0.001857
- **Experiment 15 final value:** -0.002727
- **Change:** -0.000870 (46.8% more negative, i.e. larger penalty incurred)

### Path Proximity Bonus (vs. Experiment 12)
- **Experiment 12 final value:** 0.064421
- **Experiment 15 final value:** 0.061520
- **Change:** -0.002901 (-4.5%)

### Cube Reached Goal (vs. Experiment 12)
- **Experiment 12 final value:** 0.010773
- **Experiment 15 final value:** 0.017202
- **Change:** +0.006429 (+59.7%)

### Cube Reached Goal (vs. Experiment 14)
- **Experiment 14 final value:** 0.011393
- **Experiment 15 final value:** 0.017202
- **Change:** +0.005809 (+51.0%)

## Assessment

The scalar comparison against Experiment 12 is largely positive:
`cube_reached_goal` improved by +59.7% at the final-iteration snapshot (and
by +51.0% against Experiment 14's final value), `antipodal_grasp_bonus`
increased substantially (+159.8%, also showing a monotonic late-training
climb rather than the sharp collapse seen in Experiment 14), while
`stillness_penalty` worsened modestly (46.8% more negative) and
`path_proximity_bonus` declined slightly (-4.5%). Read together, the two
outcome-adjacent metrics (`cube_reached_goal`, `antipodal_grasp_bonus`) both
moved favorably at the final snapshot, which is a more consistent pattern
than Experiment 14's mixed/ambiguous comparison against Experiment 12.

The two new terms did not behave as the design spec's success criteria
hoped for: `ground_penalty`'s nonzero rate never trended down (it is
saturated at 100% in both the first and last 150-iteration windows and every
window in between — see Section 5 above), and `base_proximity_penalty`'s
nonzero rate did not stay low (it rose from 12.0% in the first 150
iterations to a saturated 100.0% for roughly the last 1050 iterations of the
run — see Section 6 above). These are reported factually per the brief;
whether this reflects a genuine behavioral outcome (e.g., the cube legitimately
spending time near the ground/base as an unavoidable consequence of the pick
sequence, so both penalties fire on most steps) versus reward-shaping
friction the policy is fighting is not something scalars alone can
determine.

The value-function loss stayed bounded across the full run in the specific
sense examined here: the diagnostic's isolated spike to 17.66 at step 39
recurred identically (not additionally) in the full run — same step, same
magnitude to 4+ significant figures — with no second occurrence above 1.0
anywhere in the remaining ~1460 iterations and no gradual upward trend in
the loss's baseline (see the Critical Check section above for full detail
and the stated caveat about this spike's raw magnitude being large in
absolute terms relative to this project's other diagnostics/runs).

Per this project's established correction protocol (Experiment 12's
original report misread a scalar drop as failure and had to be corrected by
the controller) — **this report does not draw a final success/failure
conclusion from the scalars alone.** The scalars here are more consistently
positive than Experiment 14's were, and the value-function loss's worst case
did not worsen at full scale versus the diagnostic — but final judgment on
whether the policy is actually grasping/lifting/placing the cube well, and
whether the new ground/base-proximity penalties are shaping useful behavior
or just accumulating unavoidable penalty, requires video inspection. Per
this session's established pattern, that qualitative review is done
separately by the controller outside this plan (no Task 6 in this plan).
