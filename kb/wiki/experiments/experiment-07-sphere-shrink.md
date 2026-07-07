# Experiment 7: Sphere shrink (aperture-margin test)

**Object:** sphere. User-directed follow-up to
[[experiment-06-mirror-scene-stillness-penalty]]'s candidate hypothesis
about gripper-to-object clearance margin.

## Hypothesis

The gripper's ~28mm max aperture vs. the sphere's original 18mm diameter
(5mm per-side clearance) may leave too little margin for the joint-position
action space to reliably converge on a stable bilateral grasp pose.
Roughly doubling that margin should make a real difference if aperture
tolerance is the bottleneck.

## What changed

Reduced the sphere from 18mm to 12mm diameter (roughly doubling the
gripper's per-side clearance margin, 5mm → 8mm), scoped to the mirror-scene
task's own config only — the shared `SPHERE_CFG` in `objects_cfg.py`
(used by `interactive_demo.py`/perception scripts/`grasp_demo.py`) was left
untouched.

## Quantitative result

Not the primary evidence for this experiment — see qualitative finding
below; the decision was made from direct, personally-inspected video rather
than scalar comparison alone, given the prior session's history of
scalar-only misjudgments on this exact sub-problem.

## Qualitative video finding

10-episode real eval, **all 10 personally inspected frame-by-frame by the
controller** (not delegated, given the prior misjudgment on this same
sub-problem): **0/10 episodes show a genuine, controlled grasp-and-lift.**
One episode again showed the accidental-collision-launch signature (a
motion-blur streak trailing upward from the gripper, then the sphere
floating disconnected from a static gripper) rather than a real grasp — the
same false-positive pattern as [[experiment-06-mirror-scene-stillness-penalty]]'s
own Episode 5.

## Verdict

**Falsified.** Evidence *against* the aperture-margin hypothesis
specifically: doubling the clearance margin produced no improvement, so
gripper-to-object size tolerance is likely not the primary bottleneck.
Seventh real attempt on this sub-problem. Remaining candidates flagged: a
hierarchical policy (separate reach-to-pregrasp and close-gripper phases
instead of one flat policy learning both), or examining whether the
joint-position action space itself (rather than object size) limits precise
gripper-closure timing — the latter directly motivates the pivot to
task-space control tested starting at [[experiment-11-taskspace-ik]].

## Related concepts

[[reach-grasp-lift-gap]] — seventh attempt, still falsified; rules out a
purely geometric explanation (object size) for the grasp/lift gap.

## Sources

`docs/superpowers/plans/2026-07-06-ar4-sphere-shrink-report.md` (scoped
within the mirror-scene design, `docs/superpowers/specs/2026-07-06-ar4-sphere-mirror-scene-design.md`;
no separate design spec was written for this narrowly-scoped follow-up).
