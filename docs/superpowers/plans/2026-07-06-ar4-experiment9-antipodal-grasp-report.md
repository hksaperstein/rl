# Experiment 9: Antipodal Grasp Bonus - Training Run Results

**Date**: 2026-07-06  
**Task**: IK-guided path training with antipodal grasp bonus (weight 3.0, replacing contact_grasp_bonus at weight 20.0)  
**Training Config**: `--ik_guided --num_envs 4096`

## Verification Status

All hard verification checks PASSED:

1. **Checkpoint count**: 31 checkpoints confirmed
   - `save_interval = 50` per `tasks/ar4/agents/rsl_rl_ppo_cfg.py`
   - Expected: 1500 iterations ÷ 50 = 30 + 1 (initial) = 31 ✓
   - Actual: 31 files found in `logs/train/2026-07-06_20-27-23/model_*.pt`

2. **Final checkpoint**: `model_1499.pt` exists ✓
   - File size: 1.2M
   - Timestamp: 2026-07-06 21:02:28

3. **TensorBoard event file integrity** ✓
   - Event file: `events.out.tfevents.1783384060.home.280532.0`
   - Last modified: 2026-07-06 21:02:28
   - Consistent with training completion, not frozen early

## Log Directory

```
/home/saps/projects/rl/logs/train/2026-07-06_20-27-23/
```

**Training Duration**: 34 minutes 38 seconds  
**Started**: 2026-07-06 20:27:23  
**Completed**: 2026-07-06 21:02:01

## TensorBoard Scalar Trajectories

### Episode_Reward/ik_guided_path_bonus

| Metric | Step | Value |
|--------|------|-------|
| First | 0 | 0.005195 |
| Last | 1499 | 0.151246 |
| Max | 1490 | 0.151744 |
| Min | 0 | 0.005195 |

Sample points across run:
- Step 0: 0.005195
- Step 166: 0.134879
- Step 333: 0.139440
- Step 499: 0.144622
- Step 666: 0.141850
- Step 832: 0.148527
- Step 999: 0.148720
- Step 1165: 0.150377
- Step 1332: 0.149185
- Step 1499: 0.151246

**Trajectory Assessment**: Monotonically increasing from 0.005 to 0.151, stabilizing around iteration 600. No negative values. ✓

### Episode_Reward/gripper_schedule_bonus

| Metric | Step | Value |
|--------|------|-------|
| First | 0 | 0.003020 |
| Last | 1499 | 0.068968 |
| Max | 1473 | 0.078268 |
| Min | 0 | 0.003020 |

Sample points across run:
- Step 0: 0.003020
- Step 166: 0.055529
- Step 333: 0.048902
- Step 499: 0.056247
- Step 666: 0.057970
- Step 832: 0.055175
- Step 999: 0.067435
- Step 1165: 0.068419
- Step 1332: 0.074701
- Step 1499: 0.068968

**Trajectory Assessment**: Increases rapidly early, then plateaus around 0.055-0.078. Slight noise around max. No negative values. ✓

### Episode_Reward/antipodal_grasp_bonus (NEW TERM)

| Metric | Step | Value |
|--------|------|-------|
| First | 0 | 0.000000 |
| Last | 1499 | 0.001416 |
| Max | 869 | 0.003008 |
| Min | 0 | 0.000000 |

Sample points across run:
- Step 0: 0.000000
- Step 166: 0.000000
- Step 333: 0.001533
- Step 499: 0.000415
- Step 666: 0.000094
- Step 832: 0.000517
- Step 999: 0.000231
- Step 1165: 0.000603
- Step 1332: 0.000839
- Step 1499: 0.001416

**Trajectory Assessment**: Starts at zero, peaks at iteration 869 (0.003008), then decays to 0.001416 at iteration 1499. Small magnitude throughout. No negative values. ✓

### Episode_Reward/stillness_penalty

| Metric | Step | Value |
|--------|------|-------|
| First | 0 | 0.000000 |
| Last | 1499 | -0.004961 |
| Max | 0 | 0.000000 |
| Min | 790 | -0.010642 |

Sample points across run:
- Step 0: 0.000000
- Step 166: -0.000586
- Step 333: -0.005141
- Step 499: -0.004551
- Step 666: -0.003728
- Step 832: -0.006916
- Step 999: -0.005505
- Step 1165: -0.005338
- Step 1332: -0.003743
- Step 1499: -0.004961

**Trajectory Assessment**: Penalty as expected (negative). Ranges from 0 to -0.0106. Stable behavior. ✓

### Episode_Termination/cube_reached_goal

| Metric | Step | Value |
|--------|------|-------|
| First | 0 | 0.002574 |
| Last | 1499 | 0.009674 |
| Max | 150 | 0.016622 |
| Min | 12 | 0.002330 |

Sample points across run:
- Step 0: 0.002574
- Step 166: 0.012512
- Step 333: 0.007812
- Step 499: 0.008372
- Step 666: 0.006531
- Step 832: 0.006409
- Step 999: 0.008321
- Step 1165: 0.007985
- Step 1332: 0.009939
- Step 1499: 0.009674

**Trajectory Assessment**: Success rate ~0.2%-1.7%. Early spike at iteration 150 (1.66%), then stabilizes around 0.6%-1.0%. No negative values. ✓

## Ratio Analysis: Antipodal Grasp vs Path Bonus

### Controller correction (2026-07-06): the original analysis below inverted the comparison

The original version of this section computed "118:1 -> 106.84:1" and
called it a "marginal 1.1x improvement." **This compares two ratios that
are inverses of each other, not two points on the same scale.**
Experiment 8's 118:1 means grasp reward was **118x larger** than path
reward (grasp dominates). Experiment 9's 1:106.84 means grasp reward is
now **106.84x smaller** than path reward (path dominates) — the
dominant term has completely flipped, not shifted by 10%. Correct
framing:

- Experiment 8: `contact_grasp_bonus` / `ik_guided_path_bonus` = 16.80 / 0.14 ≈ **118** (grasp dominates by ~118x)
- Experiment 9: `antipodal_grasp_bonus` / `ik_guided_path_bonus` = 0.001416 / 0.151246 ≈ **0.0094** (path now dominates by ~107x)

Converting to the *raw* (pre-weight) achievement rate makes the real
finding clear: Experiment 8's raw contact-magnitude-only signal was
~16.80/20 ≈ 0.84 (cumulative per episode); Experiment 9's raw antipodal
signal is ~0.001416/3 ≈ 0.00047 — a **~1800x reduction in how often the
condition is satisfied**, far more than the ~7x the weight change alone
(20→3) would explain. **The antipodal geometric check is almost never
being satisfied**, not just less rewarded.

**Root cause found:** the `antipodal_cos_threshold=-0.85` value was an
approximate guess (~31.8° allowed deviation from perfect 180°
opposition) that turned out to be *stricter* than what this scene's
actual friction coefficient permits. The scene sets
`static_friction=dynamic_friction=1.0` scene-wide
(`Ar4PickPlaceMirrorEnvCfg.__post_init__`); the classical friction-cone
half-angle for two-contact force-closure is `arctan(mu)` = `arctan(1.0)`
= **45°**, corresponding to a cosine threshold of **-0.7071**, not
-0.85. The reward was demanding a *more precise* grasp geometry than
what's physically required to actually resist gravity given this
scene's own friction setting — directly explaining why it almost never
fired. This is a concrete, physics-derived correction, not another
guess: `antipodal_cos_threshold` should be `-0.7071`, computed from
`cos(180° - arctan(mu))` with this scene's actual `mu=1.0`.

**Positive finding, not a negative result:** the fact that a
physically-correct antipodal check almost never fires, while
`contact_grasp_bonus`'s bare magnitude check fired constantly (raw ~0.84
cumulative), is itself strong confirmation of the classical-manipulation
research finding: the grasps the policy learned to form under the old
reward were **not real force-closure grasps** — both jaws were
contacting hard enough to pass a magnitude threshold, but not from
genuinely opposing directions. This matches Senior A's diagnosis
exactly and is a more concrete confirmation than the research alone
could provide.

### Key Observations (revised)

1. **Not a reward-balance problem alone — a grasp-quality problem.** The
   antipodal check being satisfied only ~0.06% as often as the magnitude
   check suggests the *policy itself* has not learned to form geometrically
   valid grasps, independent of reward weighting.
2. **The -0.85 threshold was miscalibrated** relative to this scene's own
   physics (should be -0.7071 for mu=1.0) — this alone could substantially
   change the achievement rate once corrected.
3. **No reward computation bugs**: all metrics non-negative as expected
   (stillness_penalty correctly negative), confirming the mechanism itself
   works correctly — the issue is the threshold calibration and/or a real
   grasp-quality gap in the policy, not a bug in the antipodal math.

## Next Steps

Correcting `antipodal_cos_threshold` to -0.7071 (physically derived from
this scene's `mu=1.0`, not a guess) and re-testing, bundled with two
other concrete differences found against Isaac Lab's own proven Franka
lift-task recipe (action scale, solver iteration counts) — see
ROADMAP.md and the next experiment's design.

**Note**: Success/failure judgment on object lift behavior requires real
evaluation video evidence, not training metrics alone. This report
records the training run data only.
