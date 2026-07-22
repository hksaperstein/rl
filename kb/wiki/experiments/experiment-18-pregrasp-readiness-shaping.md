# Experiment 18: Pre-grasp readiness shaping

**Object:** cube. A clean falsification of the "missing gradient" hypothesis — strong shaping-term learning coupled to zero lift emergence, narrowing rather than repeating the open question after a third consecutive null result (16, 17, 18).

## Hypothesis

Experiment 17's own Task 6 instrumented finding showed the policy explores "get close" and "close the gripper" independently but never combines them — a dense reward for cube-to-EE proximity multiplied by gripper closedness should give the policy a continuous incentive to combine both halves, without reintroducing a hackable substitute for genuine antipodal contact. Grounded in Xu et al. 2026's "within-stage progress feedback" concept (arXiv:2606.31377, already cited for Experiment 17) and arXiv:1803.04996 — the binary antipodal gate stays untouched as the only path to the large `lifting_object`/`object_goal_tracking` reward.

## What changed

New `pregrasp_readiness_bonus` (`tasks/ar4/mdp.py`), wired into new `Ar4PickPlacePregraspEnvCfg` (`tasks/ar4/pickplace_pregrasp_env_cfg.py`) at weight 2.0, on top of Experiment 17's exact unchanged reward set. The binary antipodal gate and all other reward terms remain untouched.

## Quantitative result

`Episode_Reward/pregrasp_readiness` is nonzero at all 1500/1500 logged iterations (100% in every one of ten 150-iteration windows), growing from 0.000166 at iteration 0 to 1.268935 at the final iteration and stabilizing in the 1.24–1.27 range for the entire second half of training — the shaping mechanism itself works exactly as designed, with the policy actively adopting reaching configurations that register readiness.

`Episode_Reward/lifting_object` remains at exactly 0/1500 — identical to Experiment 17's null result, despite the strong readiness signal. Not "slightly worse" or "noisier" — the same exact zero at every one of 1500 logged iterations. `object_goal_tracking`/`object_goal_tracking_fine_grained` (both gated on the same lift condition) are correspondingly also 0/1500. `Loss/value_function` stayed small and bounded throughout (max 0.148235, comparable to Experiment 17's 0.0547, both roughly two orders of magnitude below Experiment 16's curriculum-driven peak of 4.588) — training itself was stable; this is not a divergence artifact.

## Instrumented finding

Per the design spec's own success criteria, this null result narrows rather than repeats the open question — it specifically falsifies the "missing approach-and-close gradient" hypothesis. The policy can learn to be "ready" (close to cube + gripper closing) as a side effect of pursuing `reaching_object`, without that readiness ever translating into an actual attempted lift.

## Verdict

**The strong readiness learning coupled to zero lift emergence falsifies the experiment's specific hypothesis: readiness and lifting are decoupled, not bottlenecked on a missing gradient.** This is the third consecutive null result (16, 17, 18) terminating in the same root fact — the cube never leaves the ground by any margin. Per this project's mandate to prefer structurally different directions over repeated parameter tweaks after such a string of nulls, the next step is not a fourth reward-shaping variant on the same mechanism.

The result implicates either the confirmed mimic-joint asset defect (Experiment 17's Task 6 finding: `gripper_jaw2_joint` drifts 20% past its own commanded open limit under load, independent of `jaw1_joint`) or a discoverability gap in the lift action itself (vertical motion while maintaining contact) that is categorically different from, and not solved by, better pre-grasp positioning.

Candidate structurally different directions, not yet scoped: (a) directly investigate/fix the confirmed mimic-joint asset defect, since it is a verified, independent, mechanical confound present in every experiment run so far, not a reward-design question; (b) demonstration or curriculum bootstrapping for the lift primitive specifically (e.g. a scripted/classical grasp-lift trajectory as a warm-start or residual-RL base, since three different reward designs across ~4500 combined training iterations have never once produced any vertical lift via pure exploration); (c) the previously-queued staged-decomposition/longer-episode redesign now has stronger direct motivation than when first proposed.

## Related concepts

[[reach-grasp-lift-gap]] — third consecutive non-resolving experiment in the through-line (16, 17, 18); still the project's central open problem at the end of this pass's coverage. [[grasp-mechanics-antipodal-vs-magnitude]] — the shaping term is designed to feed into the antipodal gate, the core mechanism distinguished in that concept article.

## Sources

`docs/superpowers/specs/2026-07-07-ar4-experiment18-pregrasp-readiness-shaping-design.md`, `docs/superpowers/plans/2026-07-07-ar4-experiment18-report.md`
