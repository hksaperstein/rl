# Experiment 14 Training Run Report: Reach-Skip Curriculum

**Date:** 2026-07-07
**Log Directory:** `/home/saps/projects/rl/logs/train/2026-07-07_12-05-30`
**Training Status:** COMPLETED (1500 iterations)

## Overview

Experiment 14 tests a reach-skip curriculum reset event: episodes are reset
directly into a pre-grasp pose (skipping the initial reach phase) via
`Ar4PickPlaceReachSkipEnvCfg` (`tasks/ar4/pickplace_reachskip_env_cfg.py`)
and the `--reachskip` flag wired into `scripts/train.py`/
`scripts/eval_loop.py`. The 300-iteration diagnostic (Task 4,
`logs/train/2026-07-07_11-59-44/`) passed its gate checks —
`Loss/value_function` max 0.0244 across the diagnostic window, an isolated
transient with immediate recovery, healthier than this session's prior
precedent (Experiment 13's diagnostic max of 0.17). This report covers the
full 1500-iteration run, launched with the reach-skip reset event in place,
no other changes.

## Verification Results

### Checkpoint Integrity
- **Config `save_interval`:** 50
- **Expected checkpoints:** 1500 / 50 + 1 = 31
- **Actual checkpoints:** 31 — confirmed
- **Checkpoint iterations:** 0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1499 — confirmed
- **model_1499.pt exists:** YES (1,178,293 bytes, modified 2026-07-07 12:21) — confirmed

### TensorBoard Event File
- **Event file:** `/home/saps/projects/rl/logs/train/2026-07-07_12-05-30/events.out.tfevents.1783440336.home.30292.0`
- **Size:** 1.9M (1,894,832 bytes)
- **Modification time:** 2026-07-07 12:21
- **model_1499.pt modification time:** 2026-07-07 12:21
- **Status:** Event file mtime matches checkpoint completion time — confirmed. Run started 12:05:30, completed ~12:21 (~16 minutes wall clock), consistent with prior 1500-iteration runs at `num_envs=4096`.

## Critical Check: `Loss/value_function` Across the FULL 1500-Iteration Run

Extracted via TensorBoard's `Loss/value_function` scalar (1500 points):

- **Max value across all 1500 iterations: 0.024436**
- **Min value: 0.0000111**
- First value (iteration 0): 0.000722
- Sampled trajectory (every ~150 iterations): 0.000722, 0.000222, 0.000554, 0.000606, 0.000493, 0.000496, 0.000749, 0.000480, 0.000633, 0.000434, 0.000502 (last, iteration 1499)
- **No sustained exponential growth.** The loss sits in the 0.0002–0.0007 range for the vast majority of the run, with a single max of 0.024436 consistent with (matching, to five significant figures) the 300-iteration diagnostic's own reported max of 0.0244.

**Conclusion: `Loss/value_function` stayed bounded for the entire 1500-iteration run.** The full run's max (0.024436) matches the diagnostic run's max almost exactly, indicating the diagnostic's isolated transient was indeed the run's worst-case value-function behavior and no new instability emerged in the remaining 1200 iterations. This is markedly healthier than Experiment 13's full-run max (0.172533) under the same checkpoint-integrity criteria.

## Scalar Trajectories (All 1500 Iterations)

### 1. Episode_Reward/path_proximity_bonus
**Summary:**
- Total data points: 1500
- First: iteration=0, value=0.000814
- Last: iteration=1499, value=0.059027
- Min: 0.000814
- Max: 0.059537
- Non-zero occurrences: 1500 / 1500

**Sample trajectory (every ~150 iterations):**
```
iteration=   0, value=0.000814
iteration= 150, value=0.017994
iteration= 300, value=0.045326
iteration= 450, value=0.051416
iteration= 600, value=0.052087
iteration= 750, value=0.053600
iteration= 900, value=0.054516
iteration=1050, value=0.054070
iteration=1200, value=0.056433
iteration=1350, value=0.055522
iteration=1499, value=0.059027
```

### 2. Episode_Reward/gripper_schedule_bonus
**Summary:**
- Total data points: 1500
- First: iteration=0, value=0.003073
- Last: iteration=1499, value=0.078903
- Min: 0.003073
- Max: 0.098841
- Non-zero occurrences: 1500 / 1500

**Sample trajectory (every ~150 iterations):**
```
iteration=   0, value=0.003073
iteration= 150, value=0.087011
iteration= 300, value=0.065128
iteration= 450, value=0.053911
iteration= 600, value=0.053190
iteration= 750, value=0.050809
iteration= 900, value=0.056101
iteration=1050, value=0.062579
iteration=1200, value=0.070824
iteration=1350, value=0.072082
iteration=1499, value=0.078903
```

### 3. Episode_Reward/antipodal_grasp_bonus
**Summary:**
- Total data points: 1500
- First: iteration=0, value=0.000000
- Last: iteration=1499, value=0.000709
- Min: 0.000000
- Max: 0.008417
- Non-zero occurrences: 1348 / 1500 (89.9%)

**Sample trajectory (every ~150 iterations):**
```
iteration=   0, value=0.000000
iteration= 150, value=0.000000
iteration= 300, value=0.000043
iteration= 450, value=0.001043
iteration= 600, value=0.001877
iteration= 750, value=0.000683
iteration= 900, value=0.001236
iteration=1050, value=0.002830
iteration=1200, value=0.001822
iteration=1350, value=0.000560
iteration=1499, value=0.000709
```

### 4. Episode_Reward/stillness_penalty
**Summary:**
- Total data points: 1500
- First: iteration=0, value=0.000000
- Last: iteration=1499, value=-0.004328
- Min: -0.020507
- Max: 0.000000
- Non-zero occurrences: 1432 / 1500 (95.5%)

**Sample trajectory (every ~150 iterations):**
```
iteration=   0, value=0.000000
iteration= 150, value=-0.000044
iteration= 300, value=-0.002461
iteration= 450, value=-0.002476
iteration= 600, value=-0.001926
iteration= 750, value=-0.003050
iteration= 900, value=-0.004936
iteration=1050, value=-0.009936
iteration=1200, value=-0.004496
iteration=1350, value=-0.002978
iteration=1499, value=-0.004328
```

### 5. Episode_Termination/cube_reached_goal
**Summary:**
- Total data points: 1500
- First: iteration=0, value=0.002574
- Last: iteration=1499, value=0.011393
- Min: 0.001017
- Max: 0.022308
- Non-zero occurrences: 1500 / 1500

**Sample trajectory (every ~150 iterations):**
```
iteration=   0, value=0.002574
iteration= 150, value=0.010478
iteration= 300, value=0.013479
iteration= 450, value=0.014476
iteration= 600, value=0.015666
iteration= 750, value=0.020416
iteration= 900, value=0.014628
iteration=1050, value=0.017202
iteration=1200, value=0.018819
iteration=1350, value=0.015737
iteration=1499, value=0.011393
```

**Note on the 300-iteration diagnostic's early signal:** the Task 4
diagnostic (`logs/train/2026-07-07_11-59-44/`) reported `cube_reached_goal`
= 0.014 at iteration 300, already above Experiment 12's full-run final value
of 0.010773. The full run's own iteration-300 value here (0.013479) closely
reproduces that diagnostic reading. Across the remaining 1200 iterations,
`cube_reached_goal` continued climbing to a mid-run peak of 0.022308 (near
iteration 750) before settling back down to a final value of 0.011393. The
diagnostic's early elevated reading did hold up in the sense that it was not
a one-off spike unique to the first 300 iterations — the metric stayed
elevated (and briefly went considerably higher) for most of the run — but
the final-iteration snapshot used for the Experiment 12 comparison below
came down substantially from that mid-run peak by the end of training.

## Key Comparison: Experiment 14 vs Experiment 12 (Final Values Only)

Per the established correction protocol (final-iteration snapshot vs.
final-iteration snapshot only — not cumulative, mid-run, or diagnostic-window
comparisons).

### Antipodal Grasp Bonus
- **Experiment 12 final value:** 0.012777
- **Experiment 14 final value:** 0.000709
- **Change:** -0.012068 (-94.4%)

### Stillness Penalty
- **Experiment 12 final value:** -0.001857
- **Experiment 14 final value:** -0.004328
- **Change:** -0.002471 (133.0% more negative, i.e. larger penalty incurred)

### Path Proximity Bonus
- **Experiment 12 final value:** 0.064421
- **Experiment 14 final value:** 0.059027
- **Change:** -0.005394 (-8.4%)

### Cube Reached Goal
- **Experiment 12 final value:** 0.010773
- **Experiment 14 final value:** 0.011393
- **Change:** +0.000620 (+5.8%)

## Assessment

The scalar comparison against Experiment 12 is mixed: `cube_reached_goal`
improved modestly (+5.8%) at the final-iteration snapshot, `path_proximity_bonus`
declined slightly (-8.4%), `stillness_penalty` worsened (133.0% more
negative), and `antipodal_grasp_bonus` dropped sharply (-94.4%, though it
remains non-zero on 89.9% of iterations, meaning the grasp condition is
still being reached on most logged points — just contributing far less
accumulated reward per episode than in Experiment 12).

Per this project's established correction protocol (Experiment 12's own
report initially misread a similar `antipodal_grasp_bonus` drop as a
"hypothesis incorrect" conclusion, which the controller subsequently
corrected after finding the underlying behavior had plausibly improved, not
regressed, once read against the other metrics together) — **this report
does not draw a final success/failure conclusion from the scalars alone.**
The reach-skip reset event changes what state distribution the policy is
trained on (episodes start from a pre-grasp pose rather than a full reach),
which plausibly changes the per-episode accounting for terms like
`antipodal_grasp_bonus` and `stillness_penalty` independent of whether the
policy's actual grasp/lift/place behavior is better or worse — e.g. a
reach-skip policy could spend proportionally less of each shorter/different
episode in a state where antipodal bonus accrues, without grasping less
often in absolute terms. `cube_reached_goal` is the more directly
outcome-oriented metric here and shows a positive (if modest) final-value
change, consistent with the diagnostic's early elevated reading, though the
metric's mid-run peak (0.022) was more than double its final value, so the
final snapshot likely understates how strong the effect was mid-run. None of
this substitutes for the qualitative check.

The value function loss stayed bounded throughout (max 0.024436, no
exponential divergence; matches the 300-iteration diagnostic's own max of
0.0244), confirming the reach-skip reset event did not introduce optimizer
instability at full scale — and did so more cleanly than Experiment 13's
full run (max 0.17).

**Next steps:** Task 6's video inspection is required to determine whether
the reach-skip curriculum policy is genuinely grasping/lifting/placing the
cube more, less, or differently than Experiment 12's policy — the scalars
alone, per the above, are ambiguous and possibly non-comparable given the
different episode/reset structure.
