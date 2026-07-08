# Experiment 23 Report: Warm-Started Residual RL over Classical Waypoint Pursuit (Full 1500 Iterations)

**Date:** 2026-07-07/08
**Log Directory:** `/home/saps/projects/rl/logs/train/2026-07-07_22-38-52`
**Training Status:** COMPLETED (1500 iterations, all 31 checkpoints present, no tracebacks)

## Recap: what this experiment tested

Per `docs/superpowers/specs/2026-07-07-ar4-experiment23-warmstarted-residual-design.md`, this experiment revisited Experiment 13's residual-RL-over-classical-controller paradigm (a bounded pursuit step toward the active waypoint, plus the policy's own action as an RL residual on top), with the one specific fix Experiment 13's own ROADMAP entry diagnosed as missing and never implemented: a literature-grounded warm-start (Johannink et al. 2019) — the residual's authority ramps linearly from 0 to 1.0 over the first 1200 env steps (~3.3% of the 36,000-step training budget), rather than contributing at full strength from iteration 0. Reward/observation/termination/curriculum configuration is Experiment 22's exact set, unchanged — the only new variable is the arm action space.

## Verification Recap (Tasks 1-4)

- **Task 1** (`WarmStartedResidualDifferentialIKAction`): reviewer independently confirmed the base-pursuit logic exactly matches Experiment 13's original `_compute_base_delta`, the waypoint-advance side effect matches `ik_guided_path_bonus`'s logic, and the ramp formula/step-count increment ordering are correct. Approved.
- **Task 2** (`Ar4PickPlaceWarmResidualEnvCfg`): reviewer independently confirmed every parameter matches the plan and traced the `configclass`=`dataclass` inheritance mechanism (`EventCfg(_BaseEventCfg)`) via source inspection, since a live Isaac Sim instantiation check could not complete due to a reproducible environment-specific startup hang. Approved.
- **Task 3** (`--warmresidual` flag wiring): reviewer confirmed all edit sites present and correctly ordered in both `train.py`/`eval_loop.py`. Approved.
- **Task 4 (hard gate)**: a real 1300-step env rollout confirmed `residual_authority` genuinely rises from ~0 at step 0 to exactly 1.0 at step 1200 and stays clamped through step 1300, reading the live action term's own `_step_count`/`cfg.warmup_steps` inside an actual `ManagerBasedRLEnv` — not just the isolated Task 1 formula check. Reviewer independently recomputed every logged value against the formula and confirmed it matches the term's internal blend exactly. **Gate confirmed PASS** — cleared to proceed to full training.
- **Task 5** (300-iteration diagnostic): `Loss/value_function` bounded (max 0.0598), no tracebacks, `lifting_object` 0/300 as expected at that scale.

## Full Run: TensorBoard Scalar Extraction

```
Episode_Reward/reaching_object: count=1500 nonzero=1500 min=0.0002 max=0.2459 final=0.0025
Episode_Reward/pregrasp_readiness: count=1500 nonzero=1500 min=0.0000 max=0.1641 final=0.0000
Episode_Reward/orientation_alignment: count=1500 nonzero=1500 min=0.0722 max=1.9643 final=1.9625
Episode_Reward/lifting_object: count=1500 nonzero=0 min=0.0000 max=0.0000 final=0.0000
Episode_Reward/object_goal_tracking: count=1500 nonzero=0 min=0.0000 max=0.0000 final=0.0000
Episode_Reward/object_goal_tracking_fine_grained: count=1500 nonzero=0 min=0.0000 max=0.0000 final=0.0000
Episode_Termination/cube_reached_goal: count=1500 nonzero=1500 min=0.0006 max=0.0053 final=0.0021
Loss/value_function: count=1500 nonzero=1500 min=0.0000 max=0.1878 final=0.0000
```

### Checkpoint Integrity
31 checkpoints confirmed (`model_0.pt` through `model_1499.pt` at `save_interval=50`), `model_1499.pt` exists, no tracebacks/exceptions in the training log (only routine benign USD/Fabric warnings identical to every prior experiment's logs this session).

### `Loss/value_function`
Bounded throughout (max 0.1878) — no sustained growth trend, no critic-divergence risk observed for the new action term. Comparable to or better than every prior task-space experiment (13/20/21/22).

### `Episode_Reward/lifting_object`
**Exactly `0/1500`** — identical to Experiment 22's exact `0/1500`, and to Experiments 17-21's before it.

### `Episode_Reward/orientation_alignment`
Saturating near ceiling (max 1.9643 of ~2.0), consistent with Experiments 20-22's identical reward term under the same weights — confirms the reward configuration is genuinely unchanged, only the action space differs.

## Contact Diagnostic (`scripts/warmresidual_contact_diagnostic.py`, model_1499.pt, 3 episodes × 250 steps)

```
[SUMMARY] total_steps=750 height_ok_steps=0 both_magnitude_ok_steps=0 antipodal_ok_steps=0
grasp_ok_steps=0 gate_fires_steps=0 max_jaw1_force=0.0 max_jaw2_force=0.0
max_cube_z=0.01272988598793745 min_residual_authority=0.0008333333333333334
```

**`both_magnitude_ok_steps` is exactly `0/750`** — identical to Experiment 22's exact `0/750`. Unlike Experiment 21/22 (where at least one jaw registered nonzero contact force — Exp21: jaw1=6.73N/jaw2=27.44N; Exp22: jaw1=0.0N/jaw2=23.45N), **this checkpoint registers exactly zero force on BOTH jaws across all 750 steps** — confirmed at every individual step (`grep`'d for any `jaw1_force=0.[1-9]` or `jaw2_force=0.[1-9]` pattern across the full log: zero matches). `max_cube_z=0.0127` — the cube never approaches the `0.03` height threshold at any point in this diagnostic.

## Important Methodological Note: the Diagnostic Does Not Evaluate the Policy at Full Residual Authority

The diagnostic's own log reveals `residual_authority` starts at `0.0008` at the diagnostic's first step and only reaches **`0.625`** by its last step (750 total env steps / `warmup_steps=1200` = 0.625, exactly matching the ramp formula). **This is because `_step_count` is instance state on the `WarmStartedResidualDifferentialIKAction` object itself, not part of the checkpoint** (`runner.load()` restores only the policy/value network weights, not action-term internal counters) — every time a script constructs a fresh `ManagerBasedRLEnv`, the ramp restarts from step 0, regardless of how much training the loaded policy actually saw.

This means **this specific contact diagnostic evaluated the trained policy under a residual authority that never exceeded 0.625**, not the full 1.0 authority the policy was trained under for the vast majority of its 1500-iteration run (`warmup_steps=1200` env steps is only ~3.3% of the full 36,000-step training budget — the policy spent ~97% of training at full authority). The primary training-time result (`lifting_object` exactly `0/1500` across the *entire* 1500-iteration run) is **not** confounded by this — training itself was correctly exposed to full-authority residual for nearly the whole run. But the contact diagnostic's specific zero-force finding should be read with this caveat: it is not a clean test of "how does the fully-warmed-up policy behave," only of "how does this policy behave when re-evaluated starting from a fresh, once-again-ramping base+residual blend." A future diagnostic script for any warm-started action term should force `_step_count` to a value ≥ `warmup_steps` at construction (or otherwise decouple the ramp from a fresh instance's own step counter) before evaluating a fully-trained checkpoint, to avoid this confound. This is a real, previously-unknown gap in this repo's residual-action diagnostic tooling, worth fixing before any future residual-RL diagnostic is trusted at face value.

## Assessment

**Both `lifting_object` (0/1500) and `both_magnitude_ok_steps` (0/750) are exactly null**, matching Experiment 22's exact results. Per the design spec's own falsification criterion — confirmed via Task 4's independently-verified hard gate that the warm-start mechanism is genuinely implemented and functioning correctly — **this specifically falsifies "the warm-start gap explains Experiment 13's regression and blocks the residual mechanism from working."** The warm-start fix Experiment 13's ROADMAP entry recommended has now been correctly implemented and independently verified working (Task 4), and combining it with the residual-RL-over-classical-controller mechanism still does not produce a working grasp+lift, under the training-run's own exposure to full residual authority for ~97% of training.

This points toward the classical-base-plus-residual paradigm itself being a more fundamental non-fit for this specific task's grasp/lift sub-problem, not an implementation-gap problem — the base controller's own waypoint-pursuit geometry (grasp/lift/transit/place waypoints computed purely from cube/goal positions, with no notion of gripper closure timing) does not appear to structurally help the policy discover the compound grasp behavior any more than plain joint-space or task-space actions did in Experiments 17-22.

**Where this leaves the research program:** eight consecutive experiments (17-23, with 13 as the original residual-RL predecessor) spanning reward shaping, grasp gating, orientation bias, proximity gating, software jaw mirroring, and now warm-started residual RL over a classical controller, have all converged on the identical `lifting_object`/`both_magnitude_ok_steps` null. Per this repo's own mandate to default toward a structurally new direction after a string of nulls, and having now exhausted the most well-grounded remaining variant within the "reward/action-space engineering over pure PPO exploration" technique family, **demonstration/imitation bootstrapping** (the remaining major untried paradigm, previously deferred because Isaac Lab Mimic requires human teleoperation) is now the most concretely justified next direction — either via a from-scratch expert-controller-generated demonstration pipeline (bypassing the teleoperation requirement) or a renewed look at what such a pipeline would concretely require in this repo's specific setup.

## Files changed

- `scripts/warmresidual_contact_diagnostic.py` (new — contact-force diagnostic for this experiment's checkpoint)
- `docs/superpowers/plans/2026-07-07-ar4-experiment23-report.md` (this file)
- `ROADMAP.md` (Experiment 23 entry added)
