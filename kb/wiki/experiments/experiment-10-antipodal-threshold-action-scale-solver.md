# Experiment 10: Physics-derived antipodal threshold + action scale + solver iterations

**Object:** cube. Bundles the root-cause correction from
[[experiment-09-antipodal-grasp-bonus]] with a systematic comparison against
Isaac Lab's own shipped, proven Franka cube-lift reference recipe.

## Hypothesis

Experiment 9's own analysis showed its antipodal threshold was stricter
than physically justified; correcting it to the friction-cone-derived value
should let more real grasps register. Separately, a direct comparison of
this repo's full PPO config, action space, and object physics against
Isaac Lab's shipped Franka+DexCube lift example (same reward functions/
weights, same `BinaryJointPositionActionCfg` gripper action, already
verified an exact copy) surfaced two remaining physical-scale differences
worth testing: this repo's `scale=1.0` for arm joint-position actions
(double Franka's `scale=0.5`) may specifically hurt the precise final
grasp-closing phase, and this repo's cube used only default PhysX solver
iteration counts where Franka's own cube explicitly boosts them
(`solver_position_iteration_count=16, solver_velocity_iteration_count=1`)
for stable contact resolution during grasping.

## What changed

Three changes bundled together, each independently motivated: (1)
`antipodal_cos_threshold` corrected `-0.85` → `-0.7071` (the physically
correct 45° friction-cone value for `mu=1.0`); (2) new `ActionsCfg`
(`scale=0.5`) scoped to `pickplace_mirror_env_cfg.py`/
`pickplace_ik_guided_env_cfg.py` only — the shared `ActionsCfg` in
`env_cfg.py` (`scale=1.0`) stays unchanged, still used by the original
sphere task, `grasp_demo.py`, `interactive_demo.py`, and perception
scripts; (3) cube solver iteration counts boosted to match Franka's recipe,
scoped via `.replace()` on `CUBE_CFG`'s spawn, not touching the shared
`objects_cfg.py`.

## Quantitative result

`antipodal_grasp_bonus` **regressed to exactly 0.000000** by the end of
training — worse than Experiment 9's already-tiny 0.001416.

## Qualitative video finding

Not separately video-inspected for this checkpoint — the zero-value
scalar result was treated as sufficient evidence on its own.

## Verdict

**Falsified.** This is the ninth real attempt on this sub-problem's
reward/optimization/physics axis, grounded in both literature research and
a direct, systematic comparison against a proven working reference
implementation. Loosening the geometric threshold (making it easier to
satisfy) didn't help — arguing the bottleneck is *precision* of final
gripper positioning/alignment under direct joint-space control, not reward-
threshold calibration. This directly motivates abandoning joint-space
action refinement in favor of a different action-space paradigm, tested
next in [[experiment-11-taskspace-ik]].

## Related concepts

[[action-space-design]] — the action-scale halving tested here (still
joint-space) as the last joint-space-only lever tried before the pivot to
task-space IK. [[grasp-mechanics-antipodal-vs-magnitude]] — the
physically-correct antipodal threshold, still yielding zero real grasps
under joint-space control.

## Sources

`docs/superpowers/plans/2026-07-06-ar4-experiment10-report.md`
