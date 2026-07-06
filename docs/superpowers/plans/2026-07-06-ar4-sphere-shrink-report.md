# AR4 Sphere-Shrink Experiment: Training Run Results

## Task 1: Full training run

**Log directory:** `/home/saps/projects/rl/logs/train/2026-07-06_16-31-04`

**Model checkpoint:** `model_1499.pt` exists ✓

### TensorBoard Scalars

**Episode_Reward/staged_milestone_bonus:**
- Step    0: 0.010881
- Step   18: 0.010156
- Step  150: 0.019776
- Step  300: 0.023303
- Step  450: 0.027467
- Step  600: 0.030303
- Step  750: 0.031535
- Step  900: 0.033331
- Step 1050: 0.032920
- Step 1200: 0.034513
- Step 1340: 0.037264
- Step 1350: 0.035341
- Step 1499: 0.036173

**Episode_Reward/stillness_penalty:**
- Step    0: 0.000000
- Step  150: -0.000249
- Step  285: -0.000725
- Step  300: -0.000262
- Step  450: -0.000070
- Step  600: -0.000112
- Step  750: -0.000162
- Step  900: -0.000034
- Step 1050: -0.000085
- Step 1200: -0.000217
- Step 1350: -0.000163
- Step 1499: -0.000046

**Episode_Termination/sphere_reached_goal:**
- Step    0: 0.002818
- Step   12: 0.001261
- Step  150: 0.014628
- Step  300: 0.023356
- Step  450: 0.028697
- Step  600: 0.029032
- Step  750: 0.030121
- Step  900: 0.031748
- Step 1050: 0.035441
- Step 1200: 0.042786
- Step 1344: 0.044566
- Step 1350: 0.040009
- Step 1499: 0.038106

**Episode_Reward/action_rate:**
- Step    0: -0.000066
- Step  150: -0.000787
- Step  300: -0.000553
- Step  450: -0.000785
- Step  600: -0.001233
- Step  750: -0.001633
- Step  900: -0.002009
- Step 1050: -0.002536
- Step 1200: -0.002856
- Step 1350: -0.003010
- Step 1494: -0.003308
- Step 1499: -0.003260

**Episode_Reward/joint_vel:**
- Step    0: -0.000074
- Step   50: -0.001553
- Step  150: -0.001039
- Step  300: -0.001244
- Step  450: -0.001180
- Step  600: -0.001144
- Step  750: -0.001029
- Step  900: -0.001016
- Step 1050: -0.001115
- Step 1200: -0.001119
- Step 1350: -0.001188
- Step 1499: -0.001346

### Trajectory Analysis

Both `staged_milestone_bonus` and `sphere_reached_goal` show monotonic growth throughout training, with `staged_milestone_bonus` peaking at 0.037264 (step 1340) and `sphere_reached_goal` peaking at 0.044566 (step 1344). The 12mm sphere produces marginally lower peak milestone bonus (0.037264) compared to the 18mm baseline (approximately 0.04), suggesting the reduced aperture margin does not improve gripper stability in reaching the goal state.
