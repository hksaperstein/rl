# Experiment 17: Antipodal grasp gate

**Object:** cube. Experiment 16 discovered a wedging exploit where the gripper could jam into the cube at a non-antipodal angle and accumulate reward. This experiment gates `lifting_object` and `object_goal_tracking` on genuine bilateral antipodal jaw contact, closing the exploit while providing clear verification of the root cause.

## Hypothesis

Experiment 16's wedging exploit was enabled by the reward structure allowing lift/carry rewards to fire on any close-proximity condition, regardless of whether genuine force-closure contact existed. Requiring real force-closure contact — both jaws exceeding `force_threshold=0.05` and jaw forces within `antipodal_cos_threshold=-0.7071` — before lift/goal-tracking rewards can fire should close this exploit. This is grounded in Xu et al. 2026's "Stage-Transition Dense Reward Modeling" (arXiv:2606.31377), which identifies exactly this failure class as "stage leakage" and shows in ablation that grasp-verification gates reduce convergence-time jitter.

## What changed

New `Ar4PickPlaceGraspGatedEnvCfg` (`tasks/ar4/pickplace_graspgated_env_cfg.py`), identical to Experiment 16 except `lifting_object` and `object_goal_tracking`/`object_goal_tracking_fine_grained` now require `antipodal_grasp_bonus`'s existing force-closure check in addition to height, at identical weights to Experiment 16 — isolating the gate as the only new variable.

## Quantitative result

Training stability metrics passed cleanly. Both 300-iteration diagnostic and full 1500-iteration run showed `Loss/value_function` bounded throughout (full-run max 0.0547, roughly two orders of magnitude below Experiment 16's peak of 4.588). The gate never fired once across all 1500 iterations: `Episode_Reward/lifting_object` and `Episode_Reward/object_goal_tracking` logged exactly 0.0 at all iterations (`nonzero: 0/1500` for both), versus Experiment 16's 81.3% nonzero by iteration 150. `cube_reached_goal` dropped to 0.002360, the lowest of any experiment since Experiment 11, directly reflecting that the only reward gradient the policy had ever found was now gated away.

## Qualitative rollout instrumentation

Rather than casual video review, this experiment's entire purpose was fixing the verification gaps that motivated the move to instrumented diagnostics. A dedicated instrumented investigation across ~1,487 rollout steps (~5.9 episodes) of the final checkpoint disambiguated why the gate never fired. `height_ok` (cube z > 0.03) was true exactly 0 times — the cube's maximum observed height (0.00901) remained indistinguishable from its 0.009 resting height. This confirms exploration difficulty (the policy never discovered genuine grasp+lift coordination), not a gate bug or threshold miscalibration. When contact did occur elsewhere (forces reached 7-20N, 150-400× above the 0.05N threshold), magnitude was never the limiting factor. A real asset defect surfaced: `gripper_jaw1_joint` tracked its commanded `[0, 0.014]` envelope exactly, while `gripper_jaw2_joint` drifted to 0.0168 (20% past its limit) under load — confirming the mimic-joint constraint is unenforced by the USD import. The one substantial contact event (episode 0, 230 consecutive steps) was a non-antipodal wedge (jaw forces: jaw1=13-20N, jaw2=7-10N; cosine angle frozen at -0.6409 vs. required <-0.7071 — five degrees short, never varying) — exactly the failure mode the gate was designed to reject, and it correctly refused to credit it for all 230 steps.

## Verdict

**The fix worked exactly as designed (exploit closed, confirmed by instrumentation), but closing the exploit removed the policy's only reward gradient. Without a replacement learning signal, the compound grasp+lift behavior that should replace the exploit was not discovered in 1500 iterations.** This is a direct, well-precedented cost of closing a reward-hacking exploit without providing a new path toward the real behavior (Xu et al.'s own ablation shows the same tradeoff). Recommended next step: add dense shaping toward correct pre-grasp positioning — cube proximity to the end-effector's pinch-point frame specifically — so the policy has gradient pointing it toward "get near the gripper and close around the object" rather than an all-or-nothing binary gate. The user's refinement — checking whether the cube tracks end-effector co-movement, not just proximity — is stronger and more targeted, and directly tests whether this exploit can recur. The confirmed mimic-joint asset defect warrants separate investigation (Does Isaac Lab support enforcing URDF `mimic` joints? Can symmetric jaw-command workarounds enforce the constraint?). Per this project's scientific-method requirement, the next experiment needs its own hypothesis and background research before a new spec.

## Related concepts

[[reach-grasp-lift-gap]] — the ongoing exploration difficulty; this experiment closes one exploitable path but doesn't solve the underlying problem. [[grasp-mechanics-antipodal-vs-magnitude]] — this experiment directly validates that magnitude is not the barrier; the antipodal constraint and force-closure verification are central to understanding why wedging fails when force alone would seem sufficient.

## Sources

`docs/superpowers/specs/2026-07-07-ar4-experiment17-grasp-gated-lift-design.md`, `docs/superpowers/plans/2026-07-07-ar4-experiment17-report.md`
