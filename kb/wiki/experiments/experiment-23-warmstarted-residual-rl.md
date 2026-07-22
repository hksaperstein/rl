# Experiment 23: Warm-started residual RL

**Object:** cube. A structurally new direction after six consecutive reward/action-space tweaks (17–22) — this experiment implements the literature-grounded warm-start mechanism (Johannink et al. 2019) that [[experiment-13-residual-rl]] diagnosed as missing and hypothesized as blocking the residual-RL paradigm, to isolate whether the implementation gap or the paradigm itself is the non-fit.

## Hypothesis

Experiment 13's residual-RL regression was caused by missing the gradual authority-ramp warm-start step specified in Johannink et al. 2019: adding this mechanism (a `residual_authority` term rising from ~0 at step 0 to 1.0 at step 1200, blending the classical base controller's actions with the learned residual) will restore the paradigm's efficacy and unblock lift emergence.

## What changed

Added a warm-start ramp to the residual action term: `residual_authority` parameter `warmup_steps=1200` gradually raises the residual component's weight from 0 to 1.0 over the first 3.3% of the 36,000-step training budget, forcing early-training reliance on the proven classical base controller while the residual learns incrementally. Reused Experiment 13's base 5-waypoint pursuit controller and reward setup unchanged — isolating the warm-start mechanism alone.

## Quantitative result

Hard gate passed before full-run commitment: a real 1300-step environment rollout of a trained policy confirmed `residual_authority` rises from ~0 at step 0 to exactly 1.0 at step 1200 and remains clamped, reading the live action term's internal state inside an actual `ManagerBasedRLEnv`. Independent verification by a reviewer recomputed every logged value against the ramp formula and confirmed exact match to the term's real internal blend.

Full run (1500 iter) showed `lifting_object` at exactly 0/1500 and `both_magnitude_ok_steps` at exactly 0/750 — null by strict criteria, matching Experiments 17, 18, 20, 21, and 22. The policy spent ~97% of its training run at full authority (`warmup_steps=1200` accounting for only ~3.3% of the 36,000-step budget), so the null result is not a training-duration artifact.

A methodological gap was surfaced and recorded: the contact diagnostic's logged `residual_authority` only reached 0.625 by the diagnostic's final step, not 1.0, because `_step_count` lives on the action term instance rather than the saved checkpoint — a fresh diagnostic ramps from zero regardless of training progress. This does not confound the training-time result itself but reveals a gap in this repo's residual-action diagnostic tooling (worth fixing via forced `_step_count` initialization before trusting future residual-RL diagnostics at face value).

## Qualitative video finding

Not separately recorded; the instrumented-diagnostic evidence (verified warm-start mechanism combined with null lifting outcome) provides the signal.

## Verdict

**Null result with confirmed warm-start mechanism falsifies "the warm-start gap explains Experiment 13's regression and blocks residual RL." The classical-base-plus-residual paradigm itself appears to be a more fundamental non-fit for this task's grasp/lift sub-problem, not an implementation-gap problem.** Eight consecutive experiments (13, 17–23) spanning reward shaping, grasp gating, orientation bias, proximity gating, software jaw mirroring, and now literature-specified warm-started residual RL have all converged on identical null. Having exhausted the most well-grounded remaining variant within the reward/action-space-engineering-over-pure-PPO-exploration family, the next justified direction is demonstration/imitation bootstrapping — either via a from-scratch expert-controller-generated pipeline or a renewed evaluation of what such a pipeline would concretely require in this setup.

## Related concepts

[[reach-grasp-lift-gap]] — eight consecutive null experiments in the reach-grasp-lift throughline; still the project's central open problem. [[experiment-13-residual-rl]] — the direct predecessor whose regression and hypothesized warm-start gap motivated this experiment's design. [[ppo-critic-divergence]] — the broader mechanism-instability concept covering Experiment 13's original regression and new-action divergence patterns, of which this experiment's outcome refines the diagnosis.

## Sources

`docs/superpowers/specs/2026-07-07-ar4-experiment23-warmstarted-residual-design.md`, `docs/superpowers/plans/2026-07-07-ar4-experiment23-report.md`
