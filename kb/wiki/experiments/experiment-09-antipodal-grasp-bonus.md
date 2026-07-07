# Experiment 9: Antipodal grasp bonus + reward-rate weight reduction

**Object:** cube. Direct response to [[experiment-08-classical-ik-guided-path]]'s
two simultaneously-verified literature findings.

## Hypothesis

Fixing both independently-verified real problems in `contact_grasp_bonus`
together — its ungated 118:1 reward-rate dominance over path progress, and
its magnitude-only (non-directional) contact check — as one evidence-based
redesign, rather than sequential single-variable guesses, should close both
gaps at once.

## What changed

New function `antipodal_grasp_bonus` (`tasks/ar4/mdp.py`) requires jaw1/jaw2
contact-force directions within ~30° of anti-parallel (dot product < -0.85,
the `antipodal_cos_threshold`) in addition to the existing magnitude check —
replaces `contact_grasp_bonus` in `Ar4PickPlaceIkGuidedEnvCfg`'s
`RewardsCfg`, at a substantially reduced weight (20.0 → 3.0) to close the
reward-rate gap rather than relying on `stillness_penalty` alone to outweigh
it. `contact_grasp_bonus` itself was left unchanged in `mdp.py` — still used
by the original sphere-based `pickplace_env_cfg.py` task.

This bundles two independently-verified real problems (reward-rate,
geometric correctness), externally cross-checked against Isaac Lab's own
shipped lift task, IsaacGymEnvs `FrankaCubeStack`, and ManiSkill3
`PickCube-v1` source (read directly) — all three gate downstream reward
behind grasp/lift state; this repo's prior reward was the structural
outlier. Mao et al. 2025 (arXiv:2502.15442) independently names this exact
"reach then freeze" local optimum in unrelated work (quote verified against
primary text). Classical grasp-mechanics literature (Nguyen 1988; Ponce &
Faverjon 1991/93; Ferrari & Canny 1992; GraspIt! 2004; Dex-Net/GPD/
QuickGrasp) treats a geometric/antipodal force-closure check as mandatory,
never substitutable by contact-force magnitude alone.

## Quantitative result

**Reward dominance completely reversed: 118:1 grasp-dominant →
~107:1 path-dominant**, not a modest improvement. The antipodal geometric
check fires ~1800x less often than the old magnitude-only check did — far
more than the 6.67x weight reduction (20→3) alone would explain.

## Qualitative video finding

Not separately video-inspected for this checkpoint — see the root-cause
analysis below.

## Verdict

**Root-caused rather than treated as a straightforward win.**
`antipodal_cos_threshold=-0.85` (~31.8° allowed deviation from perfect
opposition) was an approximate guess, stricter than this scene's actual
physics permits — the classical friction-cone half-angle for this scene's
`mu=1.0` (`static_friction=dynamic_friction=1.0`, scene-wide) is
`arctan(1.0)=45°`, giving a correct threshold of `-0.7071`, not `-0.85`.
That the physically-correct check almost never fires while the
magnitude-only check fired constantly is itself strong confirmation that the
grasps the policy learned under the old reward were not real force-closure
grasps, just coincidentally hard bilateral contact from non-opposing
directions. The threshold correction is tested next in
[[experiment-10-antipodal-threshold-action-scale-solver]].

## Related concepts

[[reward-rate-arithmetic]] — the 118:1→107:1 reversal is this concept's
central before/after data point. [[grasp-mechanics-antipodal-vs-magnitude]]
— the antipodal check itself, and the friction-cone-derived threshold
correction this experiment's own result motivates.

## Sources

`docs/superpowers/plans/2026-07-06-ar4-experiment9-antipodal-grasp-report.md`,
`docs/superpowers/specs/research/2026-07-06-rl-manipulation-senior-b.md`,
`docs/superpowers/specs/research/2026-07-06-classical-manipulation-senior-a.md`
