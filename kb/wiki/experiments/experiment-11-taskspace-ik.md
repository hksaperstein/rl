# Experiment 11: Task-space IK-driven action

**Object:** cube. User-proposed, structurally different pivot away from
joint-space action refinement after [[experiment-10-antipodal-threshold-action-scale-solver]]
found precision-of-positioning, not reward calibration, was the likely
bottleneck.

## Hypothesis

Instead of the policy outputting joint-angle deltas directly (with IK used
only for reward-shaping in Experiments 8–10), have the policy output
Cartesian end-effector deltas and let Isaac Lab's built-in
`DifferentialInverseKinematicsActionCfg` convert them to joint targets
*inside the control loop* — offloading "how to move 6 joints" to a
classical solver so the policy only has to learn "where to go."

## What changed

New file `tasks/ar4/pickplace_taskspace_env_cfg.py`
(`Ar4PickPlaceTaskspaceEnvCfg`), new simplified `path_proximity_bonus`
reward (drops the now-redundant IK-match sub-signal `ik_guided_path_bonus`
needed when IK was reward-only rather than action-generating).
`antipodal_grasp_bonus`/`gripper_schedule_bonus`/`stillness_penalty` carried
over unchanged from Experiment 10. New `--taskspace` flag on
`scripts/train.py`/`scripts/eval_loop.py`.

## Quantitative result

**First full run diverged.** The controller (not the implementer, whose own
status report had called this "Non-Critical") independently traced
`/tmp/exp11_train_stdout.log` and found the PPO critic's
`Mean value_function loss` exploding from ~0.0000 to ~1.56 to ~4047 to
~3.2M between iterations 66–69/1500, reaching ~5.2e23 by the final
iteration and never recovering — ~95% of the run's policy updates were
driven by a diverged critic. Never seen in Experiments 1–10 (same PPO
config, joint-space action), implicating the new
`DifferentialInverseKinematicsActionCfg` term specifically: an outlier raw
policy action, previously harmless under `JointPositionActionCfg`
(saturates at joint limits), likely drives the IK solve into a
discontinuous joint-space jump that destabilizes PhysX for one env/step,
producing an extreme observation the critic can't fit. **Fix**: new
`Ar4PickPlaceTaskspacePPORunnerCfg(clip_actions=5.0)` (~3.4x the observed
action-noise std of 1.46), scoped to the taskspace experiment only —
`Ar4PickPlacePPORunnerCfg` itself stays unmodified. Verified on a
300-iteration diagnostic, then the full 1500-iteration re-run:
`Loss/value_function` stayed bounded for the entire run (max 7.88, one
isolated 2-iteration transient spike, immediate recovery — independently
re-verified against the raw TensorBoard event file, all 1500 points).

On the corrected run: `antipodal_grasp_bonus` final value **0.018815**,
nonzero in **91.6%** of all 1500 logged iterations — every prior experiment
had this at exactly 0 (Experiment 10) or 0.001416 at best (Experiment 9).
`cube_reached_goal` final 0.010223, ~3.6x Experiment 10's 0.002848.

## Qualitative video finding

25 frames, 5fps, full episode: the arm reaches down toward the cube within
the first ~1s and then holds an almost identical low, near-ground pose for
the remaining ~4s of the episode — the small red cube stays at or near the
gripper tip throughout, consistent with the nonzero antipodal-contact
metric (a real, held bilateral grasp is plausible), but the arm never
visibly lifts the cube to height or carries it toward the goal in this
rollout.

## Verdict

**First positive signal after 11 experiments, but qualified.** Task-space
IK-driven action produced the first genuine, sustained antipodal grasp
contact this project has seen — a real improvement on the specific "grasp
never emerges" sub-problem — but "pick up and move" as a whole is still not
achieved. The next sub-problem is getting the policy from "hold a low
grasp" to "lift and carry," which motivates the staged-decomposition/
episode-length/richer-goal-placement ideas queued as follow-up direction.

## Related concepts

[[action-space-design]] — the central pivot this experiment tests, and its
strongest positive result. [[ppo-critic-divergence]] — the first of two
new-action-mechanism critic-divergence bugs this session (see also
[[experiment-13-residual-rl]]). [[reach-grasp-lift-gap]] — grasp is now
real and sustained; lift is still the unsolved remainder.
[[reward-rate-arithmetic]] — this run's own reward structure is what
Experiment 12 finds a net +1.0/step freeze incentive in.

## Sources

`docs/superpowers/specs/2026-07-06-ar4-taskspace-ik-action-design.md`,
`docs/superpowers/plans/2026-07-06-ar4-taskspace-ik-action-implementation.md`,
`docs/superpowers/plans/2026-07-06-ar4-experiment11-report.md`
