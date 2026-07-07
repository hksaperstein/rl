# Experiment 11 Training Run Report: Task-Space IK-Driven Action

**Date:** 2026-07-06  
**Log Directory:** `/home/saps/projects/rl/logs/train/2026-07-06_22-28-15`  
**Training Status:** ✅ COMPLETED (1500 iterations)

## Verification Results

### Check 1-3: Checkpoint Integrity
- **Config `save_interval`:** 50
- **Expected checkpoints:** 1500 / 50 + 1 = 31
- **Actual checkpoints:** 31 ✅
- **Checkpoint iterations:** 0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1499 ✅
- **model_1499.pt exists:** YES (1.2M, modified 2026-07-06 22:43) ✅

### Check 4: TensorBoard Event File
- **Event file:** `/home/saps/projects/rl/logs/train/2026-07-06_22-28-15/events.out.tfevents.1783391301.home.672869.0`
- **Size:** 1.9M
- **Modification time:** 2026-07-06 22:43 ✅
- **Status:** Modified at training completion time ✅

## Scalar Trajectories (All 1500 Iterations)

### 1. Episode_Reward/path_proximity_bonus
**Summary:**
- Total data points: 1500
- First: iteration=0, value=0.000366
- Last: iteration=1499, value=0.005800
- Min: 0.000366
- Max: 0.006156

**Sample trajectory (every ~150 iterations):**
```
iteration=   0, value=0.000366
iteration= 150, value=0.004796
iteration= 300, value=0.005166
iteration= 450, value=0.005517
iteration= 600, value=0.005910
iteration= 750, value=0.005420
iteration= 900, value=0.005486
iteration=1050, value=0.005216
iteration=1200, value=0.005250
iteration=1350, value=0.005447
iteration=1499, value=0.005800
```

### 2. Episode_Reward/gripper_schedule_bonus
**Summary:**
- Total data points: 1500
- First: iteration=0, value=0.003133
- Last: iteration=1499, value=0.082574
- Min: 0.003133
- Max: 0.086053

**Sample trajectory (every ~150 iterations):**
```
iteration=   0, value=0.003133
iteration= 150, value=0.082418
iteration= 300, value=0.082833
iteration= 450, value=0.085052
iteration= 600, value=0.083309
iteration= 750, value=0.084292
iteration= 900, value=0.083452
iteration=1050, value=0.082151
iteration=1200, value=0.082157
iteration=1350, value=0.082867
iteration=1499, value=0.082574
```

### 3. Episode_Reward/antipodal_grasp_bonus
**Summary:**
- Total data points: 1500
- First: iteration=0, value=0.000000
- Last: iteration=1499, value=0.000000
- Min: 0.000000
- Max: 0.000150
- Non-zero occurrences: 4 / 1500 (0.3%)

**Sample trajectory (every ~150 iterations):**
```
iteration=   0, value=0.000000
iteration= 150, value=0.000025
iteration= 300, value=0.000000
iteration= 450, value=0.000000
iteration= 600, value=0.000000
iteration= 750, value=0.000150
iteration= 900, value=0.000000
iteration=1050, value=0.000031
iteration=1200, value=0.000000
iteration=1350, value=0.000000
iteration=1499, value=0.000000
```

### 4. Episode_Reward/stillness_penalty
**Summary:**
- Total data points: 1500
- First: iteration=0, value=0.000000
- Last: iteration=1499, value=0.000000
- Min: -0.002833
- Max: 0.000000

**Sample trajectory (every ~150 iterations):**
```
iteration=   0, value=0.000000
iteration= 150, value=-0.000053
iteration= 300, value=-0.000038
iteration= 450, value=-0.000040
iteration= 600, value=-0.000067
iteration= 750, value=-0.000067
iteration= 900, value=0.000000
iteration=1050, value=-0.002833
iteration=1200, value=0.000000
iteration=1350, value=-0.000685
iteration=1499, value=0.000000
```

### 5. Episode_Termination/cube_reached_goal
**Summary:**
- Total data points: 1500
- First: iteration=0, value=0.002574
- Last: iteration=1499, value=0.002858
- Min: 0.002574
- Max: 0.007660

**Sample trajectory (every ~150 iterations):**
```
iteration=   0, value=0.002574
iteration= 150, value=0.004354
iteration= 300, value=0.004150
iteration= 450, value=0.005351
iteration= 600, value=0.004252
iteration= 750, value=0.004262
iteration= 900, value=0.005219
iteration=1050, value=0.005117
iteration=1200, value=0.007660
iteration=1350, value=0.006358
iteration=1499, value=0.002858
```

## Key Comparison: Experiment 11 vs Experiment 10 (Final Values Only)

### Antipodal Grasp Bonus — Critical Finding
- **Experiment 10 final value:** 0.000000
- **Experiment 11 final value:** 0.000000
- **Change:** NO IMPROVEMENT — antipodal grasp bonus remains **exactly zero** ✗

Despite the task-space IK-driven action architecture intended to provide finer gripper positioning control, the policy still does not achieve even a single antipodal grasp in the final trained state. Non-zero occurrences are vanishingly rare (4 out of 1500 iterations, 0.3%) and scattered throughout training, not concentrated at the end. **This represents a null result on the experiment's primary hypothesis.**

### Cube Reached Goal — Secondary Metric
- **Experiment 10 final value:** 0.002848
- **Experiment 11 final value:** 0.002858
- **Change:** +0.000010 (0.35% improvement, effectively flat)

Marginal and statistically insignificant improvement in the goal-reaching metric.

### Gripper Schedule Bonus — Reference
- **Experiment 10 final value:** 0.089251
- **Experiment 11 final value:** 0.082574
- **Change:** -0.006677 (7.5% decline)

Gripper scheduling appears slightly worse under task-space action, though both values remain in a similar operational range.

## Assessment

The task-space IK-driven action design (Experiment 11) did not resolve the antipodal grasp bonus regression observed in Experiment 10. The final trained policy achieves zero antipodal contact, identical to Experiment 10's end state. This suggests that **precise gripper positioning/alignment via IK solver alone is not sufficient to achieve reliable bilateral opposition** — or that the current IK solver's kinematic model, precision, or interaction with task-space action filtering is preventing the policy from discovering such a solution within 1500 training iterations.

**No success/failure judgment on cube lift capability** — evaluation video analysis (Task 5, a separate follow-up) is required to assess whether the policy actually lifts the cube despite the antipodal metric showing no progress.
