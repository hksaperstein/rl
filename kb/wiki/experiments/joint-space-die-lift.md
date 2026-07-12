# Joint-space (no-IK) d20 die lift — falsified, failure isolated to the asset

**Date:** 2026-07-12 (branch `franka-panda-pivot`)
**Spec:** `docs/superpowers/specs/2026-07-11-joint-space-die-lift-design.md`
**Report:** `docs/superpowers/plans/2026-07-11-joint-space-die-lift-report.md`

## Hypothesis

Isaac Lab's validated Franka lift recipe, with the action space swapped
from task-space differential IK to direct joint-position control (the
recipe's own `joint_pos` variant values: scale 0.5, use_default_offset)
and the object swapped from DexCube to the physics-baked d20
(`assets/dice/d20_physics.usd`, mm-as-m at 0.001 scale, convex hull,
0.01kg), trains a die lift+carry policy.

## Method

Two variables isolated across three runs, everything else pinned
(rewards, observations, commands, PPO cfg byte-inherited from
`FrankaLiftEnvCfg`):

1. 300-iter diagnostic (joint-die) — gate: value-loss boundedness +
   reach trend. Passed.
2. 1500-iter full run (joint-die) — authoritative metric
   `Metrics/object_pose/position_error` (this env logs no success
   termination; do-nothing baseline ≈ 0.216).
3. 1500-iter fallback (joint-cube: identical config, DexCube object) —
   spec-pre-authorized asset-vs-recipe isolation.

## Results

| | joint-die (d20) | joint-cube (DexCube) |
|---|---|---|
| `position_error` last-100 | 0.331 (worse than baseline) | **0.105** |
| `lifting_object` (wt 15) | 0.12 spawn-artifact floor, flat | 13.38 |
| `object_goal_tracking` (wt 16) | ~0.02 noise | 12.29 |
| `Train/mean_reward` | ~2.0 | 138.4 |
| eval (8 envs, instrumented z) | 0/8 sustained lifts; reach-then-settle on video | (video pending; scalars decisive) |

Value loss bounded in all runs (no Experiment-11-style divergence).

## Conclusions

- **The joint-space no-IK action formulation works on this platform** —
  it trains lift+carry on the DexCube decisively within 1500 iterations.
- **The d20 asset is what fails.** Candidate causes, deliberately left
  untested per the spec's stop-after-fallback rule: ~2cm die size vs the
  DexCube's much larger pinch target; near-spherical rolling geometry;
  baked 0.01kg mass; friction/material of the baked asset; the 0.001
  spawn-scale pipeline.
- Echo of the AR4 lesson ([[franka-panda-pivot]]): what presents as an
  RL/reward-design difficulty can be an asset/object property problem.
  This time the isolation experiment was pre-authorized in the spec and
  took one run to prove.

## Gotchas recorded

- The die spawns at z=0.055, above the stock `minimal_height=0.04`
  lifted-threshold → `Episode_Reward/lifting_object` shows a constant
  ~0.12 floor from iteration 0 (settle-steps score as "lifted"). Watch
  for growth above the floor, not above zero.
- This env logs NO success-termination scalar — terminations are
  `time_out`/`object_dropping` only. `Metrics/object_pose/position_error`
  is the honest success metric (its do-nothing baseline is the resting
  die-to-air-goal distance, ≈0.216 here).
- 300-iter diagnostics can mislead in both directions: the diagnostic
  showed a fast reach rise (0.687 by it150) then decline that the full
  run did not reproduce (slow steady climb to 0.39).

## Next (needs its own spec + research grounding)

Bisect the asset gap: joint-space config vs a DexCube-sized die, or the
d20 rescaled to DexCube size — separates size from shape from mass.
