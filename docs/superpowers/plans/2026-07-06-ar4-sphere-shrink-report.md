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

## Task 2: Real eval + video inspection (decision gate)

**Checkpoint tested:** `logs/train/2026-07-06_16-31-04/model_1499.pt`

**Eval setup:** 10-episode evaluation with `--mirror`, 12mm-diameter sphere (shrunk from 18mm), randomized spawn across the full workspace, goal mirrored to the opposite side of the robot. All 10 episodes personally inspected frame-by-frame by the controller (not delegated), given a prior misjudgment this session where a coarse start/25/50/75/end sample misread an accidental knock as a genuine lift.

**Spawn randomization:** confirmed still working — sphere appears at visibly different relative positions across episodes.

**Per-episode findings:**

| Episode | Result |
|---|---|
| 0 (step-0) | No lift — sphere stays on/near ground throughout, gripper never secures it |
| 1 (step-250) | No lift — arm ends in a collapsed pose, sphere occluded/undetected |
| 2 (step-500) | No lift — arm remains in the same collapsed pose as episode 1's end (see note below) |
| 3 (step-750) | No lift — sphere stays on ground; ambiguous gripper-adjacent frame at end but no elevation evidence |
| 4 (step-1000) | **Accidental knock/launch**, not a grasp — a motion-blur streak is visible trailing upward from the gripper at frame 010, and the sphere is airborne and disconnected from the gripper by frames 020-050 (gripper stays at ground level). Same signature as the false positive found in the prior (18mm) experiment's Episode 5. |
| 5 (step-1250) | No lift — sphere disappears from view near the gripper (occlusion, not elevation) |
| 6 (step-1500) | No lift — sphere stays at/near ground level near the gripper's jaws |
| 7 (step-1750) | No lift — sphere on ground, gripper not co-located with it by episode end |
| 8 (step-2000) | No lift — sphere stays on ground at the gripper's tip |
| 9 (step-2250) | No lift — sphere remains on ground, arm doesn't close the distance to it |

**Note on episode continuity:** frame_001 of the step-500 segment shows the identical collapsed arm pose as frame_050 of the step-250 segment, suggesting the two segments may not represent two independently-reset episodes but could be one continuous rollout viewed through consecutive time windows. This wasn't fully resolved and doesn't change the core finding (no genuine lift in any segment inspected), but is worth investigating if this eval methodology is reused — worth adding an explicit per-episode reset marker (e.g. logging env reset events or overlaying the episode/env step count on the video) in a future eval run rather than inferring reset boundaries from arm pose alone.

### Decision Gate Assessment

**Result: 0/10 episodes show a genuine, controlled grasp-and-lift.**

**Verdict: FALSIFIED.** Shrinking the sphere from 18mm to 12mm diameter (roughly doubling the gripper's per-side clearance margin, from 5mm to 8mm) did not produce any improvement — the gripper still never achieves and holds a bilateral grasp in any of the 10 eval episodes, and the one apparent elevation event (Episode 4) is, like the prior experiment's Episode 5, consistent with an accidental collision-launch rather than a controlled grasp. This is evidence against the gripper-aperture-margin hypothesis specifically: a meaningfully looser aperture margin did not change the outcome, so the failure is likely not primarily about clearance tolerance.

**This is the seventh real attempt on the reward/optimization/physical-setup axis** for this sub-problem (sparse-only, curriculum-gated dense, always-on dense, LR-bump, potential-shaping, mirror-scene+stillness-penalty, sphere-shrink). Recommend the Principal/user weigh in on the remaining candidates before further unilateral iteration: a hierarchical policy (separate reach-to-pregrasp and close-gripper phases, rather than one flat policy learning both simultaneously) or a closer look at whether the joint-position action space itself (rather than object size) is the bottleneck for precise gripper-closure timing.
