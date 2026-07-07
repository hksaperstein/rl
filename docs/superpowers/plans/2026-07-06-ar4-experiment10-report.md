# Experiment 10 Training Run Report: Physics-Corrected Threshold + Action Scale + Solver Iterations

**Date:** 2026-07-06  
**Log Directory:** `/home/saps/projects/rl/logs/train/2026-07-06_21-51-00`  
**Training Status:** ✅ COMPLETED (1500 iterations)

## Verification Results

### Check 1-3: Checkpoint Integrity
- **Config `save_interval`:** 50
- **Expected checkpoints:** 1500 / 50 + 1 = 31
- **Actual checkpoints:** 31 ✅
- **Checkpoint iterations:** 0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1499 ✅
- **model_1499.pt exists:** YES (1.2M, modified 2026-07-06 22:04) ✅

### Check 4: TensorBoard Event File
- **Event file:** `/home/saps/projects/rl/logs/train/2026-07-06_21-51-00/events.out.tfevents.1783389066.home.359629.0`
- **Size:** 1.9M
- **Modification time:** 2026-07-06 22:04 ✅
- **Status:** Modified at training completion time ✅

## Scalar Trajectories (All 1500 Iterations)

### 1. Episode_Reward/ik_guided_path_bonus
**Summary:**
- Total data points: 1500
- First: iteration=0, value=0.004369
- Last: iteration=1499, value=0.117441
- Min: 0.004369
- Max: 0.130317

**Sample trajectory (every ~150 iterations):**
```
iteration=   0, value=0.004369
iteration= 150, value=0.122518
iteration= 300, value=0.125662
iteration= 450, value=0.121365
iteration= 600, value=0.117623
iteration= 750, value=0.116245
iteration= 900, value=0.114421
iteration=1050, value=0.114683
iteration=1200, value=0.114415
iteration=1350, value=0.116658
iteration=1499, value=0.117441
```

### 2. Episode_Reward/gripper_schedule_bonus
**Summary:**
- Total data points: 1500
- First: iteration=0, value=0.002995
- Last: iteration=1499, value=0.089251
- Min: 0.002995
- Max: 0.093714

**Sample trajectory (every ~150 iterations):**
```
iteration=   0, value=0.002995
iteration= 150, value=0.074089
iteration= 300, value=0.075358
iteration= 450, value=0.081985
iteration= 600, value=0.088510
iteration= 750, value=0.088829
iteration= 900, value=0.092534
iteration=1050, value=0.089622
iteration=1200, value=0.088198
iteration=1350, value=0.088610
iteration=1499, value=0.089251
```

### 3. Episode_Reward/antipodal_grasp_bonus
**Summary:**
- Total data points: 1500
- First: iteration=0, value=0.000000
- Last: iteration=1499, value=0.000000
- Min: 0.000000
- Max: 0.002000
- Non-zero occurrences: 143 / 1500 (9.5%)

**Sample trajectory (every ~150 iterations):**
```
iteration=   0, value=0.000000
iteration= 150, value=0.000000
iteration= 300, value=0.000000
iteration= 450, value=0.000000
iteration= 600, value=0.000000
iteration= 750, value=0.000000
iteration= 900, value=0.000000
iteration=1050, value=0.000000
iteration=1200, value=0.000000
iteration=1350, value=0.000000
iteration=1499, value=0.000000
```

### 4. Episode_Reward/stillness_penalty
**Summary:**
- Total data points: 1500
- First: iteration=0, value=0.000000
- Last: iteration=1499, value=-0.000033
- Min: -0.001026
- Max: 0.000000

**Sample trajectory (every ~150 iterations):**
```
iteration=   0, value=0.000000
iteration= 150, value=-0.000076
iteration= 300, value=-0.000094
iteration= 450, value=-0.000079
iteration= 600, value=-0.000056
iteration= 750, value=-0.000018
iteration= 900, value=-0.000122
iteration=1050, value=-0.000024
iteration=1200, value=-0.000068
iteration=1350, value=0.000000
iteration=1499, value=-0.000033
```

### 5. Episode_Termination/cube_reached_goal
**Summary:**
- Total data points: 1500
- First: iteration=0, value=0.002574
- Last: iteration=1499, value=0.002848
- Min: 0.001221
- Max: 0.013336

**Sample trajectory (every ~150 iterations):**
```
iteration=   0, value=0.002574
iteration= 150, value=0.010142
iteration= 300, value=0.010661
iteration= 450, value=0.008555
iteration= 600, value=0.005402
iteration= 750, value=0.003927
iteration= 900, value=0.003713
iteration=1050, value=0.003418
iteration=1200, value=0.004964
iteration=1350, value=0.004303
iteration=1499, value=0.002848
```

## Key Comparison: Antipodal Grasp Bonus Analysis

### Experiment 9 (baseline):
- Final antipodal_grasp_bonus: 0.001416
- Final ik_guided_path_bonus: 0.151246
- Cumulative ratio: 1:107 (antipodal fires at ~0.9% cumulative contribution)

### Experiment 10 (this run):
- Final antipodal_grasp_bonus: 0.000000
- Final ik_guided_path_bonus: 0.117441
- Cumulative ratio: ~1:11,250 (antipodal fires at ~0.009% cumulative contribution)
- **Cumulative antipodal total:** 0.015632 across 1500 iterations
- **Cumulative ik_guided total:** 175.784608 across 1500 iterations

### Critical Finding
The antipodal grasp bonus is firing **FAR LESS OFTEN** in Experiment 10 than in Experiment 9, despite the threshold being corrected from -0.85 to -0.7071 (which should make the check less strict).

- Experiment 9: antipodal fires with cumulative value 0.001416 at iteration 1499
- Experiment 10: antipodal fires with cumulative value 0.015632 across all 1500 iterations, but final value is 0.000000

This indicates the threshold fix alone did not solve the antipodal grasp bonus deficit. The bonus was expected to increase meaningfully, but instead it remains nearly zero throughout the run. The 143 non-zero occurrences are concentrated in early iterations (mostly iterations 24-58) and remain extremely small in magnitude (median < 0.0001).

**Conclusion:** The threshold correction did NOT achieve the expected result of making the antipodal grasp bonus fire more often. This suggests a deeper issue beyond just the threshold value, possibly related to:
- Gripper force sensing or geometry
- Cube/gripper contact behavior under the new solver settings
- The interaction between the corrected threshold and the action scale/solver iteration changes

**No success/failure judgment on cube lift capability** — evaluation video analysis (a separate follow-up task) is required to assess whether the policy actually lifts the cube.
