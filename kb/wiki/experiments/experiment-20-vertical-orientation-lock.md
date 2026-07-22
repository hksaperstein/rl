# Experiment 20: Vertical orientation lock

**Object:** cube. The fourth consecutive non-improving result (Experiments
17–20) in a convergent line of inquiry, now with enough direct evidence to
rule out approach-orientation discovery as the exploration bottleneck. The
experiment pivoted mid-run after independent verification found the original
hard-IK mechanism structurally unstable; the revised soft reward-bias mechanism
worked exactly as designed but still never produced lift.

## Hypothesis

Constraining the gripper's approach orientation toward vertical/top-down via an
IK-based action term would remove orientation as a discovery bottleneck and
enable grasp-to-lift progression, similar to how Experiments 17–19 explored
whether grasp mechanics, contact calibration, or reward shaping were the
blocking factors.

## What changed

**Original design:** a custom absolute-pose differential-IK action term
(`VerticalLockDifferentialIKAction`, `tasks/ar4/actions.py`) locking
orientation exactly each step, leaving only 3D position under policy control.
Code review passed cleanly, but independent re-verification against the live
simulated system (not isolated quaternion math) found the mechanism genuinely
unstable: the real end-effector orientation converged to within ~9–10 degrees
of target by step 30, then diverged to 75–99 degrees off target within the same
episode under zero commanded policy action. Three systematically-tried fixes
were attempted per this repo's debugging discipline (tilting the target
orientation off exact-vertical; adding persistent state to the position target;
sweeping DLS damping from 0.01 to 1.0), but none achieved stability — an
architecture-level finding, not a tuning bug. The mechanism was abandoned as
not viable for sustained pose-holding on this arm.

**Revised mechanism:** a soft dense reward term, `orientation_alignment_bonus`,
layered onto Experiment 18's already-proven joint-space action instead of
introducing a new IK-based action space. This tested the identical underlying
hypothesis without the IK-stability problem class. New env cfg
`Ar4PickPlaceOrientationBiasEnvCfg` (`tasks/ar4/pickplace_orientationbias_env_cfg.py`)
reused Experiment 18's exact action/observation/event/termination/curriculum
configuration plus the one new reward term.

## Quantitative result

The revised mechanism worked exactly as intended — the strongest, most
completely saturated dense signal recorded in this repo's history.
`Episode_Reward/orientation_alignment` reached its effective ceiling (weight
2.0 × a [0,1]-bounded function) by iteration 150 and stayed saturated at
1.92–1.96 for the remaining ~90% of training, more completely saturated than
Experiment 18's `pregrasp_readiness` (which settled around 1.2–1.25 out of a
similar ceiling, not fully saturated). This confirms the policy unambiguously
solved the specific sub-problem this experiment targeted: orienting the
gripper's approach axis vertically.

`Episode_Reward/lifting_object` stayed at exactly 0/1500 regardless — a clean
falsification, not an inconclusive result. Identical to Experiments 17–18's
outcomes. Because the orientation-alignment signal was unambiguously solved
(not merely attempted), this specifically rules out approach-orientation
discovery as the exploration bottleneck, rather than leaving that question open.

## Instrumented follow-up findings

A follow-up Task 6 instrumented rollout on the final checkpoint (750 steps,
3 episodes) revealed a critical asymmetry: `height_ok_steps=0` (cube never left
the ground, `max_cube_z=0.00905` vs. 0.009 resting), but crucially
`both_magnitude_ok_steps=0/750` — `gripper_jaw1_joint`'s contact sensor
registered zero force at every single step (`max_jaw1_force=0.0`), while
`gripper_jaw2_joint` did register contact intermittently (`max_jaw2_force=2.23N`).
This is a genuinely different failure signature from Experiment 17's Task 6,
which found both jaws contacting simultaneously in a non-antipodal wedge
(`both_magnitude_ok_steps=231/750`). `max_orientation_dot=0.9998` confirms the
orientation mechanism achieved near-perfect vertical alignment per-step in the
same rollout, so this is not an orientation regression. The asymmetric, one-jaw-only
contact directly implicates the mimic-joint mechanical asymmetry (Experiment 17
Task 6 / Experiment 19: jaw2 tracks its commanded position 20% worse than jaw1
under load) as a more directly-evidenced blocker.

## Verdict

**No improvement on the stated success criterion (genuine lift-off-the-ground
emerging) despite the orientation-alignment mechanism working perfectly,
plus direct evidence narrowing the remaining blocker to gripper mechanics.**
This cleanly falsifies the orientation-discovery-bottleneck hypothesis,
unlike Experiments 17–18's more ambiguous results. Four consecutive
experiments (17, 18, 19, 20) now converge on the same underlying fact — the
cube never leaves the ground by any margin — after reward shaping, hard and
soft orientation constraints, and a mimic-joint fix attempt have each been
tried and each failed to move it. Per this repo's own mandate to prefer a
structurally new direction over another variant on the same approach class
after a string of nulls, the next research direction should not be a fifth
reward/action-space tweak on pure joint-space RL exploration. The asymmetric
one-jaw-only contact finding directly implicates the underlying gripper
mechanical asymmetry as a more tractable next candidate than further reward
engineering, and motivates returning to root-cause investigation of the
jaw-mimic constraint's enforcement.

## Related concepts

[[reach-grasp-lift-gap]] — fourth consecutive experiment in this through-line
(17, 18, 19, 20) converging on the same unsolved lift problem; the mechanism
now empirically rules out orientation discovery as the bottleneck. [[grasp-mechanics-antipodal-vs-magnitude]]
— the asymmetric one-jaw-only contact signature from the instrumented follow-up
directly concerns force-closure and antipodal mechanics, distinguishing this
experiment's failure mode from prior experiments' more ambiguous contact patterns.

## Sources

`docs/superpowers/specs/2026-07-07-ar4-experiment20-vertical-orientation-lock-design.md`,
`docs/superpowers/plans/2026-07-07-ar4-experiment20-report.md`
