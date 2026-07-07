# Grasp mechanics: antipodal/force-closure vs. magnitude-only contact

## The core distinction

Classical grasp-mechanics literature (Nguyen 1988; Ponce & Faverjon
1991/93; Ferrari & Canny 1992; GraspIt! 2004; modern data-driven planners
Dex-Net/GPD/QuickGrasp) treats a geometric/antipodal force-closure check as
mandatory for a "real" grasp, never substitutable by contact-force
*magnitude* alone. A real bilateral force reading can register from a
non-antipodal, unstable pinch that isn't actually resistant to gravity's
wrench — hard contact on both jaws is necessary but not sufficient for a
stable grasp.

## How this project rediscovered it empirically

- **[[experiment-01-contact-sensor-grasp-reward]]** introduced the first
  ground-truth contact signal (`grasp_contact`, via `ContactSensorCfg`),
  replacing purely geometric EE-distance proxies. It converged to ~92%
  sustained bilateral contact — real progress on "does the gripper close on
  the object" — but this was still a **magnitude-only** check (contact
  force above a threshold on both jaws), with no directional/antipodal
  requirement.
- **[[experiment-08-classical-ik-guided-path]]**'s completed cube run
  produced the data that let a dedicated classical-manipulation literature
  review (run in parallel with an RL-manipulation review) independently
  conclude the magnitude-only check was a real, separate problem from the
  reward-rate imbalance found in the same run — a bilateral force reading
  need not correspond to an antipodal grasp.
- **[[experiment-09-antipodal-grasp-bonus]]** replaced the magnitude-only
  check with `antipodal_grasp_bonus`, requiring jaw contact-force
  directions within ~30° of anti-parallel (dot product < -0.85). The
  antipodal check fired ~1800x less often than the old magnitude-only
  check — strong evidence that most of the "grasps" the policy had
  previously learned under Experiment 1's reward were coincidentally hard
  bilateral contact from non-opposing directions, not real force-closure
  grasps.
- **[[experiment-10-antipodal-threshold-action-scale-solver]]** corrected
  the threshold to the scene's own physically-derived value (`-0.7071`,
  the 45° friction-cone half-angle for `mu=1.0`) rather than the earlier
  approximate guess (`-0.85`) — loosening the geometric requirement to what
  physics actually permits. This made the check *even sparser* in practice
  (antipodal signal regressed to exactly 0), pointing at joint-space
  positioning precision, not threshold calibration, as the remaining
  bottleneck — which directly motivated the pivot to task-space control in
  [[experiment-11-taskspace-ik]] (see [[action-space-design]]).
- **[[experiment-11-taskspace-ik]]**, once precision improved via
  task-space IK control, produced the first genuine sustained antipodal
  signal (0.018815, nonzero 91.6% of iterations) — confirming the antipodal
  check itself is learnable once the action space allows precise-enough
  gripper positioning.

## A separate, purely geometric hypothesis that was ruled out

**[[experiment-07-sphere-shrink]]** tested a different, non-force-related
grasp-mechanics hypothesis: that the gripper's aperture-to-object-size ratio
(28mm aperture vs. 18mm sphere, 5mm per-side clearance) left too little
margin for reliable closure. Doubling the clearance margin (shrinking the
sphere to 12mm) produced no improvement — ruling out object-size tolerance
as the primary bottleneck, distinct from (and prior to) the antipodal-vs-
magnitude distinction above.

## Status

Getting the contact/grasp signal right (Experiment 1, then 9, then 11) has
never, on its own, produced a lift — see [[reach-grasp-lift-gap]]. Grasp
quality and lift-discovery appear to be genuinely separate sub-problems in
this project's data.

## Related experiments

[[experiment-01-contact-sensor-grasp-reward]], [[experiment-07-sphere-shrink]],
[[experiment-08-classical-ik-guided-path]], [[experiment-09-antipodal-grasp-bonus]],
[[experiment-10-antipodal-threshold-action-scale-solver]], [[experiment-11-taskspace-ik]]
