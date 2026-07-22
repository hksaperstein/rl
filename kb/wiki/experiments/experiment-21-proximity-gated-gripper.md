# Experiment 21: Proximity-gated gripper

**Object:** cube. A direct test of the user's own design contribution —
approaching with the gripper open, then only allowing close once in
position — applied against [[experiment-20-vertical-orientation-lock]]'s
specific finding that one jaw never touched the cube at all. Null on the literal
success criterion (lifting), but a real narrowing result resolving the
asymmetric-contact failure signature.

## Hypothesis

Experiment 20's failure signature (one jaw never registering contact)
suggests an asymmetry in how the two jaws coordinate. Hard-gating the
gripper open during approach, only allowing the policy's own close command
once the cube is within 5cm of the end-effector, should remove the grasp
coordination problem from approach and test whether the asymmetry persists
when the jaws are forced to enter grasp-readiness simultaneously rather than
on the policy's unconstrained timing.

## What changed

New action term `ProximityGatedBinaryJointPositionAction` (`tasks/ar4/actions.py`)
forces the gripper open regardless of the policy's own close command unless
the cube is within a `proximity_threshold=0.05m` of the end-effector. Behavior
verified directly: cube far + close command → forced open; cube 0.02m away +
close command → actually closes. Design grounded in staged approach-then-close
structures observed in learned grasping literature (arXiv:2303.17592).

## Quantitative result

`lifting_object` remains 0/1500, matching Experiments 17, 18, and 20 —
no improvement on the strict success metric. `orientation_alignment` and
`pregrasp_readiness` both remained healthy and consistent with Experiment 20's
values, confirming the new gate did not disrupt existing mechanisms.

Instrumented contact diagnostic (same Task-6-style rollout as Experiment 20):
`max_jaw1_force` went from 0.0N (Experiment 20, jaw1 never registered contact)
to 6.73N; `max_jaw2_force` went from 2.23N to 27.44N. `max_orientation_dot=1.0`
(tighter than Experiment 20's 0.9998) rules out orientation regression.
Strict `both_magnitude_ok_steps` success criterion remains 0/750.

## Instrumented diagnostic finding

Both jaws now register genuine, substantial contact with the cube — resolving
Experiment 20's specific one-jaw-never-touches asymmetry. However, simultaneity
remains missing: both jaws touch, but never at the same timestep during the
same rollout. This narrows the failure mode from "one jaw doesn't touch" to
"both touch, but not in sync."

## Verdict

**A narrowing rather than a null result. Do not extend this exact mechanism,
but treat the finding as concrete evidence for the next lever.** Five
consecutive experiments (17, 18, 19, 20, 21) have each ruled out a different
specific candidate, with instrumented diagnostics progressively narrowing the
failure signature: "cube never leaves the ground" → "one jaw never touches"
→ "both jaws touch, just not simultaneously." The timing/coordination defect
directly implicates the mimic-joint mechanical asymmetry itself
(Experiment 17 Task 6 / Experiment 19: jaw2 tracks its commanded position
20% worse than jaw1 under load). This is exactly what an uncoupled,
asymmetric gripper mechanism would produce. Not fixable via the `PhysxMimicJointAPI`
already ruled out in Experiment 19, but now the most concretely-evidenced
next lever ahead of demonstration/imitation bootstrapping.

## Related concepts

[[reach-grasp-lift-gap]] — the through-line problem that experiments 17–21
are attacking via narrowing instrumented diagnostics rather than broad
mechanism changes. [[grasp-mechanics-antipodal-vs-magnitude]] — the
"both jaws touch, just not simultaneously" finding is squarely about
antipodal/force-closure timing, not raw magnitude.

## Sources

`docs/superpowers/specs/2026-07-07-ar4-experiment21-proximity-gated-gripper-design.md`,
`docs/superpowers/plans/2026-07-07-ar4-experiment21-report.md`
