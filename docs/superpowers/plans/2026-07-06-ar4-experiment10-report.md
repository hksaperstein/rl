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

### Controller correction: the "cumulative ratio" comparison below compared different quantities

The original version of this section computed "Experiment 9 cumulative
1:107" vs "Experiment 10 cumulative 1:11,250" by summing Experiment 10's
per-iteration logged values across all 1500 training iterations, then
comparing that sum against Experiment 9's single final-iteration value.
These are not comparable quantities (one is a single episode snapshot,
the other integrates a whole training run's worth of snapshots). The
only fair, apples-to-apples comparison is **final value vs. final
value**, both single-episode snapshots at the end of a same-length run:

- Experiment 9 final `antipodal_grasp_bonus`: **0.001416**
- Experiment 10 final `antipodal_grasp_bonus`: **0.000000**

**This is a real regression, not an artifact of the ratio math.**
Experiment 10's antipodal condition is satisfied in exactly **0** of the
envs being averaged by the end of training — worse than Experiment 9's
already-tiny 0.001416. The 143 non-zero occurrences (9.5% of logged
iterations) are concentrated in early iterations (~24-58) and vanish
entirely by the end of training, i.e., **the policy converged away from
ever satisfying even the loosened antipodal condition**, rather than
learning toward it.

### What this suggests

`antipodal_grasp_bonus` is a flat, per-step reward (like the
`contact_grasp_bonus` it replaced) at a comparatively small weight
(3.0, vs. `ik_guided_path_bonus`'s 25.0) — the policy has little
incentive to specifically pursue genuine antipodal contact over
whatever non-antipodal-but-magnitude-satisfying contact pattern is
easier to fall into (which is exactly what `contact_grasp_bonus`
rewarded before, and which the policy clearly could achieve reliably
in Experiments 8-9's ~16.8 cumulative `contact_grasp_bonus` values).
Loosening the geometric threshold (-0.85 -> -0.7071) didn't help
because the bottleneck isn't threshold strictness — it's that achieving
**genuine bilateral opposition** appears to require more precise final
gripper positioning/alignment than the current control scheme reliably
achieves, and there's currently little reward pressure directing the
policy toward that precision specifically.

This is a real, evidence-based argument *for* the next planned
experiment (Experiment 11: task-space IK-driven action, replacing
joint-space actions) — if precise final positioning/alignment is the
actual bottleneck, offloading low-level joint coordination to a
classical IK solver (so the policy only has to specify *where* the
gripper should be, not *how* to move 6 joints to get there) may make
that precision more achievable than it is under direct joint-space
control.

**No success/failure judgment on cube lift capability** — evaluation
video analysis (a separate follow-up task) is required to assess
whether the policy actually lifts the cube.
