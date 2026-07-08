# Experiment 21 Training Run Report: Proximity-Gated Gripper Closing (Full 1500 Iterations)

**Date:** 2026-07-07
**Log Directory:** `/home/saps/projects/rl/logs/train/2026-07-07_20-24-29`
**Training Status:** COMPLETED (1500 iterations)

## Gate Verification Recap

Before training, the `ProximityGatedBinaryJointPositionAction` gate
logic was verified directly (not inferred from downstream training
behavior): with the cube far away (0.55m) and a policy command of
"close" issued every step, the gate forced `_processed_actions` to the
open command (`[0.014, 0.014]`) regardless. With the cube teleported to
0.02m from the end-effector (within the 0.05m threshold) and the same
"close" command, the gate allowed the close command through
(`[0.0, 0.0]`). PASS on both conditions — the gate works exactly as
designed, independent of the training run's own outcome.

## Verification Results

### Checkpoint Integrity
- **Expected checkpoints:** 31 — confirmed.
- **model_1499.pt exists:** YES.
- **No tracebacks/exceptions:** confirmed.

### `Loss/value_function` Sanity Check
First: 0.050161, last: 0.000468, max: 0.190217, min: 0.000255,
nonzero 1500/1500. Small and bounded, declining trend — training stable.

## Reward Trajectories (Full-Run Summary)

| Term | First | Last | Max | Nonzero |
|---|---|---|---|---|
| `reaching_object` | 0.000163 | 0.632525 | 0.636712 | 1500/1500 |
| `pregrasp_readiness` | 0.000022 | 1.213089 | 1.228775 | 1500/1500 |
| `orientation_alignment` | 0.071741 | 1.948481 | 1.962189 | 1500/1500 |
| `lifting_object` | 0.0 | 0.0 | 0.0 | 0/1500 |
| `object_goal_tracking` | 0.0 | 0.0 | 0.0 | 0/1500 |
| `object_goal_tracking_fine_grained` | 0.0 | 0.0 | 0.0 | 0/1500 |

`orientation_alignment` and `pregrasp_readiness` both show healthy,
consistent trajectories closely matching Experiment 20's own values —
confirming the new gripper gate did not disrupt either mechanism.
`Episode_Termination/cube_reached_goal` final value: 0.001872 (nonzero
1500/1500, reported factually per this project's established protocol,
not as an independent verdict).

**`Episode_Reward/lifting_object` stays at exactly `0/1500`** —
identical to Experiments 17, 18, and 20.

## Critical Question: Does the Contact Diagnostic's `both_magnitude_ok_steps` Move Off `0/750`?

Per this experiment's own specific success criteria (a more mechanistic
test than `lifting_object` alone, since this experiment specifically
targets Experiment 20's asymmetric single-jaw-contact finding), ran the
same Task-6-style instrumented rollout against the trained checkpoint
(`model_1499.pt`, 3 episodes, 750 total steps):

```
[SUMMARY] total_steps=750 height_ok_steps=0 both_magnitude_ok_steps=0
antipodal_ok_steps=0 grasp_ok_steps=0 gate_fires_steps=0
max_jaw1_force=6.726553 max_jaw2_force=27.436047 max_cube_z=0.009597
min_orientation_dot=0.013319 max_orientation_dot=1.0
```

**Literal answer: no, `both_magnitude_ok_steps` is still exactly
`0/750`** — both jaws never register significant force *simultaneously*
at any single step. By the letter of this experiment's stated success
criterion, this is a null result.

**But the underlying picture changed in a specific, meaningful way.**
Compare directly against Experiment 20's contact diagnostic
(`docs/superpowers/plans/2026-07-07-ar4-experiment20-report.md`'s
addendum):

| | Experiment 20 (no gate) | Experiment 21 (proximity-gated) |
|---|---|---|
| `max_jaw1_force` | **0.0N** (never registers any contact) | **6.73N** (real, substantial contact) |
| `max_jaw2_force` | 2.23N | 27.44N |
| `both_magnitude_ok_steps` | 0/750 | 0/750 |

Experiment 20's specific failure signature — `gripper_jaw1_joint`
never registering any contact force at all across an entire rollout —
is **resolved**. Both jaws now make real, substantial contact with the
cube at some point during the rollout. What remains missing is
*simultaneity*: both jaws contact the cube, but not at the same step.
`max_orientation_dot=1.0` (a perfect vertical alignment reading, even
tighter than Experiment 20's 0.9998) confirms the orientation mechanism
and the new gate coexist without conflict — this isn't a regression
elsewhere.

**Interpretation:** the proximity gate did what it was designed to do —
it changed *when* the gripper closes, and that changed *which* jaw
makes contact and when, resolving the specific one-jaw-never-touches
asymmetry. It did not, however, produce coordinated *simultaneous*
bilateral contact. This narrows the open question further than
Experiment 20 did: the remaining gap is not "does the gripper ever
touch the cube with real force" (now confirmed yes, both jaws do) but
specifically "do the two jaws close in enough sync to touch at the same
moment" — a timing/coordination question that more directly implicates
the confirmed mimic-joint mechanical asymmetry (Experiment 17 Task 6 /
Experiment 19: `gripper_jaw2_joint` tracks its commanded position 20%
worse than `gripper_jaw1_joint` under load) than either orientation or
*when* closing is initiated.

## Assessment

**Gate mechanism verification:** confirmed working exactly as designed,
both by direct pre-training logic testing and by the post-training
contact diagnostic showing a real behavioral change (jaw1 now contacts
the cube, where it never did before).

**Primary finding:** `lifting_object` remains at exactly `0/1500`, and
`both_magnitude_ok_steps` remains at exactly `0/750` — by the strict
letter of this experiment's success criteria, a null result. Video
inspection was not performed, matching this project's established
practice: `lifting_object` staying at exactly zero with unambiguous
scalar evidence means video adds nothing new.

**But this is a narrowing result, not a repeat of the same null.**
Five consecutive experiments (17, 18, 19, 20, 21) have now each ruled
out a different specific candidate: dense pre-grasp shaping alone (18),
a specific mimic-joint fix mechanism (19), hard and soft
orientation-discovery burden (20), and now premature/imprecise-timing
closing (21) — while each successive instrumented diagnostic has
narrowed what's actually happening at the moment of failure, from "cube
never leaves the ground" (all) to "one jaw never touches at all" (20)
to "both jaws touch, just not simultaneously" (21). The most
directly-implicated remaining candidate is the mimic-joint mechanical
asymmetry itself — not fixable via the specific `PhysxMimicJointAPI`
mechanism Experiment 19 already ruled out, but still a live, physically
real defect this experiment's own evidence points at more precisely
than before.
