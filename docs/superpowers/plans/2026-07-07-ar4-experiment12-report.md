# Experiment 12 Training Run Report: Antipodal/Stillness Reward-Rate Fix

**Date:** 2026-07-07
**Log Directory:** `/home/saps/projects/rl/logs/train/2026-07-07_08-54-34`
**Training Status:** COMPLETED (1500 iterations)

## Overview

Experiment 12 tests a reward-rate fix targeting the imbalanced learning dynamics observed in Experiment 11. **Correction to the hypothesis as originally stated in this section** (the original text here incorrectly said `antipodal_grasp_bonus`'s weight was 0.1 relative to `stillness_penalty`'s 2.0 — 0.1 is `gripper_schedule_bonus`'s weight, not antipodal's): the actual hypothesis, per `docs/superpowers/specs/2026-07-07-ar4-experiment12-stillness-reward-rate-design.md`, is that `antipodal_grasp_bonus` (weight 3.0, paid every step the grasp holds, uncapped and unconditional on progress) combined with the old `stillness_penalty` (weight 2.0, only firing after a 25-step patience window) netted **+1.0/step positive** reward for holding a grasp without further progress — directly rewarding the "reach, grasp, freeze" failure mode Experiment 11 exhibited on video. Fix applied: raise `stillness_penalty` weight from 2.0 to 5.0 in `tasks/ar4/pickplace_taskspace_env_cfg.py` (commit 82d8cc8), making the same situation net **-2.0/step**. The 300-iteration diagnostic (Task 2, `logs/train/2026-07-07_08-49-33/`) passed gate checks (value function bounded). This report covers the full 1500-iteration run, launched with the fix in place.

## Verification Results

### Checkpoint Integrity
- **Config `save_interval`:** 50
- **Expected checkpoints:** 1500 / 50 + 1 = 31
- **Actual checkpoints:** 31 — confirmed
- **Checkpoint iterations:** 0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1499 — confirmed
- **model_1499.pt exists:** YES (1,178,293 bytes, modified 2026-07-07 09:09) — confirmed

### TensorBoard Event File
- **Event file:** `/home/saps/projects/rl/logs/train/2026-07-07_08-54-34/events.out.tfevents.1783428879.home.8196.0`
- **Size:** 1.9M
- **Modification time:** 2026-07-07 09:09
- **model_1499.pt modification time:** 2026-07-07 09:09
- **Status:** Event file mtime matches checkpoint completion time — confirmed

## Critical Check: `Loss/value_function` Across the FULL 1500-Iteration Run

The value function loss stayed bounded throughout the entire run, confirming the fix (and the Experiment 11 fix before it) holds across full-scale training.

Extracted via TensorBoard's `Loss/value_function` scalar (1500 points):

- **Max value across all 1500 iterations: 0.042336**, occurring around iteration 47–48 region
- First 10 values: 0.002588, 0.000473, 0.000542, 0.000622, 0.000899, ..., all in normal range
- Last 5 values: 0.000673, 0.000707, 0.000673, 0.000566 (steady, normal range)
- **No sustained exponential growth.** The loss stays in the range 0.00011–0.00089 for the vast majority of the run, with no recurrence of the original Experiment 11 divergence pattern.

**Conclusion: `Loss/value_function` stayed bounded for the entire 1500-iteration run.** The policy-gradient updates were driven by well-behaved advantage estimates throughout.

## Scalar Trajectories (All 1500 Iterations)

### 1. Episode_Reward/path_proximity_bonus
**Summary:**
- Total data points: 1500
- First: iteration=0, value=0.000366
- Last: iteration=1499, value=0.064421
- Min: 0.000366
- Max: 0.066471

**Sample trajectory (every ~150 iterations):**
```
iteration=   0, value=0.000366
iteration= 150, value=0.035756
iteration= 300, value=0.050812
iteration= 450, value=0.057263
iteration= 600, value=0.058615
iteration= 750, value=0.061194
iteration= 900, value=0.060943
iteration=1050, value=0.062124
iteration=1200, value=0.065478
iteration=1350, value=0.061729
iteration=1499, value=0.064421
```

### 2. Episode_Reward/gripper_schedule_bonus
**Summary:**
- Total data points: 1500
- First: iteration=0, value=0.003133
- Last: iteration=1499, value=0.077216
- Min: 0.003133
- Max: 0.099175

**Sample trajectory (every ~150 iterations):**
```
iteration=   0, value=0.003133
iteration= 150, value=0.075471
iteration= 300, value=0.056663
iteration= 450, value=0.049324
iteration= 600, value=0.071191
iteration= 750, value=0.077492
iteration= 900, value=0.077407
iteration=1050, value=0.076026
iteration=1200, value=0.081414
iteration=1350, value=0.077279
iteration=1499, value=0.077216
```

### 3. Episode_Reward/antipodal_grasp_bonus
**Summary:**
- Total data points: 1500
- First: iteration=0, value=0.000000
- Last: iteration=1499, value=0.012777
- Min: 0.000000
- Max: 0.026318
- Non-zero occurrences: 1398 / 1500 (93.2%)

**Sample trajectory (every ~150 iterations):**
```
iteration=   0, value=0.000000
iteration= 150, value=0.000059
iteration= 300, value=0.000230
iteration= 450, value=0.002542
iteration= 600, value=0.005009
iteration= 750, value=0.005395
iteration= 900, value=0.004326
iteration=1050, value=0.011893
iteration=1200, value=0.009920
iteration=1350, value=0.008807
iteration=1499, value=0.012777
```

### 4. Episode_Reward/stillness_penalty
**Summary:**
- Total data points: 1500
- First: iteration=0, value=0.000000
- Last: iteration=1499, value=-0.001857
- Min: -0.013396
- Max: 0.000000
- Non-zero occurrences: 1485 / 1500 (99.0%)

**Sample trajectory (every ~150 iterations):**
```
iteration=   0, value=0.000000
iteration= 150, value=-0.000924
iteration= 300, value=-0.000738
iteration= 450, value=-0.002873
iteration= 600, value=-0.001847
iteration= 750, value=-0.001248
iteration= 900, value=-0.009301
iteration=1050, value=-0.001424
iteration=1200, value=-0.001284
iteration=1350, value=-0.000819
iteration=1499, value=-0.001857
```

### 5. Episode_Termination/cube_reached_goal
**Summary:**
- Total data points: 1500
- First: iteration=0, value=0.002574
- Last: iteration=1499, value=0.010773
- Min: 0.001088
- Max: 0.020966

**Sample trajectory (every ~150 iterations):**
```
iteration=   0, value=0.002574
iteration= 150, value=0.008606
iteration= 300, value=0.011363
iteration= 450, value=0.020518
iteration= 600, value=0.011241
iteration= 750, value=0.010030
iteration= 900, value=0.007131
iteration=1050, value=0.013407
iteration=1200, value=0.015767
iteration=1350, value=0.011861
iteration=1499, value=0.010773
```

## Key Comparison: Experiment 12 vs Experiment 11 (Final Values Only)

Per the established correction protocol (final-iteration snapshot vs. final-iteration snapshot only).

### Antipodal Grasp Bonus
- **Experiment 11 (corrected) final value:** 0.018815
- **Experiment 12 final value:** 0.012777
- **Change:** -0.006038 (32% decrease from Experiment 11)
- **Assessment:** The reward-rate fix (increasing stillness_penalty weight 2.0 → 5.0) did NOT improve antipodal grasp bonus. In fact, it decreased it. The policy is achieving lower antipodal grasp rewards than Experiment 11.

### Stillness Penalty
- **Experiment 11 (corrected) final value:** -0.002533
- **Experiment 12 final value:** -0.001857
- **Change:** +0.000676 (less negative, i.e., better / less penalty incurred)
- **Assessment:** The increased stillness_penalty weight did not increase the stillness penalty — instead, the policy incurred less penalty. This suggests the policy may be moving less, or the higher weight is actually reducing the magnitude of its negative impact per episode by discouraging movement more strongly.

### Cube Reached Goal
- **Experiment 11 (corrected) final value:** 0.010223
- **Experiment 12 final value:** 0.010773
- **Change:** +0.000550 (5.4% increase)
- **Assessment:** Cube-reached-goal termination rate increased slightly.

### Path Proximity Bonus
- **Experiment 11 (corrected) final value:** 0.059629
- **Experiment 12 final value:** 0.064421
- **Change:** +0.004792 (8% increase)
- **Assessment:** Path proximity bonus improved, suggesting the policy is reaching waypoint indices further along the 5-waypoint trajectory on average. However, this scalar alone cannot definitively indicate whether the policy is reaching waypoint index ≥2 (lift) more than Experiment 11, since it aggregates running-max deltas across all 5 waypoints. Per-waypoint breakdown requires video inspection (Task 4), not available in this report. **Do not draw a definitive success/failure conclusion here** — that requires Task 4's video inspection.

## Assessment

**Controller's independent re-assessment (this section replaces this task's original Assessment, which drew a premature "hypothesis incorrect" conclusion from `antipodal_grasp_bonus` alone — reviewed and rejected as unsupported by the full scalar picture; see reasoning below).**

`antipodal_grasp_bonus`'s final-value drop (0.018815 → 0.012777, -32%) is **not**, on its own, evidence the fix failed, because that term was never the task's success metric — it's an instrumental proxy that pays out every step a stable, low-motion, low-force-noise antipodal condition holds. The fix's entire point was to stop rewarding a *static* hold. Read together with the other four scalars, the drop is consistent with the fix working as intended, not with it failing:

- **`stillness_penalty` got *less* negative** (-0.002533 → -0.001857) despite its weight increasing 2.5x (2.0 → 5.0). The only way a 2.5x larger weight produces a *smaller* accumulated penalty is if the "grasped and stagnant for ≥25 steps" condition is firing less often than in Experiment 11 — i.e., the policy is spending less time frozen post-grasp. This is the mechanism the fix targeted, working.
- **`antipodal_grasp_bonus`'s nonzero rate went *up*, not down** (1374/1500 = 91.6% in Exp 11 → 1398/1500 = 93.2% in Exp 12). Grasp attempts did not become rarer. Only the per-episode *accumulated magnitude* dropped, consistent with briefer, more dynamic holds (moving while grasping naturally interrupts the antipodal condition more than sitting still) rather than fewer or weaker grasps.
- **Both outcome-oriented metrics improved**: `path_proximity_bonus` +8% (more/faster progress along the 5-waypoint path), `cube_reached_goal` +5.4% (higher task-success termination rate).

This pattern — less frozen time, more grasp attempts, more path progress, higher success rate, but less reward from a term that specifically pays for static holding — is at minimum ambiguous on the scalars alone, and plausibly a genuine improvement misread by a metric that was never supposed to be maximized in isolation. It is not the clear regression the original Assessment claimed.

The value function loss stayed bounded throughout (max 0.042336, no exponential divergence), confirming the underlying critic-stability fix (clip_actions=5.0) from Experiment 11 continues to hold.

**Next steps**: No "hypothesis incorrect" conclusion is warranted from scalars alone — this is exactly the situation Task 4's video inspection exists to resolve (does the arm actually lift/carry more than Experiment 11's static hold, or not). Proceeding to Task 4 before deciding on any further reward-shaping direction.
