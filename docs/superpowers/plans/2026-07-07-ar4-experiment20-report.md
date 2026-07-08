# Experiment 20 Training Run Report: Soft Orientation-Alignment Bias (Full 1500 Iterations)

**Date:** 2026-07-07
**Log Directory:** `/home/saps/projects/rl/logs/train/2026-07-07_19-46-35`
**Training Status:** COMPLETED (1500 iterations)

## Mechanism Revision Recap

Experiment 20 originally designed a hard action-space constraint — a
custom absolute-pose differential-IK action term
(`VerticalLockDifferentialIKAction`, `tasks/ar4/actions.py`) locking the
gripper's orientation to a fixed vertical target every step. Independent
instrumented verification (not the implementer's own unreproduced
claim) found this mechanism structurally unstable: the real simulated
end-effector orientation diverged 75-99 degrees off target within a
single episode under zero policy action, reproducing identically across
episodes. Three genuinely different fixes were tried and instrumented
(tilting the target off exact-vertical, giving the position target
persistent state instead of a self-referential current-position
reference, sweeping DLS damping 0.01-1.0) — none achieved stability. Per
this repo's systematic-debugging discipline, this was treated as an
architecture-level finding, not a bug to keep patching, and the
mechanism was revised: a soft dense reward term
(`orientation_alignment_bonus`, `tasks/ar4/mdp.py`) layered onto the
already-proven joint-space action (Experiment 18's exact `ActionsCfg`),
instead of a new IK-based action space. Full account in
`docs/superpowers/specs/2026-07-07-ar4-experiment20-vertical-orientation-lock-design.md`'s
"Revision" section.

This report covers the revised mechanism's full 1500-iteration run.

## Verification Results

### Checkpoint Integrity
- **Config `save_interval`:** 50
- **Expected checkpoints:** 1500 / 50 + 1 = 31
- **Actual checkpoints:** 31 — confirmed
- **model_1499.pt exists:** YES
- **No tracebacks/exceptions:** confirmed, case-insensitive sweep of the full-run log found zero matches.

### `Loss/value_function` Sanity Check
- First: 0.050627, last: 0.000313, max: 0.200989, min: 0.000238, nonzero 1500/1500.
- Small and bounded throughout, declining trend, no sustained growth or divergence — training was stable.

## Critical Question 1: Does `orientation_alignment_bonus` Show Real, Growing Nonzero Signal — Confirming the Soft-Bias Mechanism Actually Works?

**Full-run scalar extraction:**
```
=== Episode_Reward/orientation_alignment ===
  points: 1500 first: 0.071741 last: 1.946610 max: 1.967313 min: 0.071741 nonzero: 1500 / 1500
```

**Sampled trajectory (every 150 iterations):**
```
iteration=   0, value=0.071741
iteration= 150, value=1.957659
iteration= 300, value=1.946141
iteration= 450, value=1.955047
iteration= 600, value=1.956040
iteration= 750, value=1.926779
iteration= 900, value=1.936081
iteration=1050, value=1.940251
iteration=1200, value=1.934253
iteration=1350, value=1.940619
iteration=1499, value=1.946610
```

**Answer: Yes, decisively.** The term is nonzero at all 1500/1500
logged iterations, reaches essentially its ceiling (weight 2.0 × a
[0,1]-bounded reward function, so 2.0 is the theoretical max) by
iteration 150, and stays saturated in the 1.92-1.96 range for the
entire remaining ~90% of training. This confirms the revised soft-bias
mechanism works exactly as designed — the policy strongly and
consistently learned to orient the gripper's approach axis toward
vertical, without any of the IK-stability problems the abandoned hard-
lock mechanism exhibited. This is the strongest, most cleanly saturated
signal of any dense shaping term across this repo's entire experiment
history (compare Experiment 18's `pregrasp_readiness`, which settled
around 1.2-1.25 out of a similar ~2.0 ceiling, not fully saturated).

## Critical Question 2: Does `Episode_Reward/lifting_object`'s Nonzero Count Move Off Exactly `0/1500`?

**Full-run scalar extraction:**
```
=== Episode_Reward/lifting_object ===
  points: 1500 first: 0.0 last: 0.0 max: 0.0 min: 0.0 nonzero: 0 / 1500
```

Flat zero at every single one of 1500 logged iterations. Sampled
trajectory (every 150 iterations) confirms: 0.000000 at every point
from iteration 0 through 1499. `object_goal_tracking` and
`object_goal_tracking_fine_grained` (both gated on the same lift
condition) are correspondingly also 0/1500.

**Answer: No.** `lifting_object` remains at exactly `0/1500` — identical
to Experiment 17's and Experiment 18's outcomes.

**The specific finding, stated directly:** despite `orientation_alignment`
being the strongest, most saturated dense signal recorded in this
repo's history — meaning the policy unambiguously learned to solve the
orientation problem this experiment specifically targeted — `lifting_object`
does not advance beyond 0/1500. This is a clean, decisive falsification
of the orientation-discovery-bottleneck hypothesis: reducing (in fact,
essentially eliminating) the policy's burden of discovering a good
approach orientation did not enable lift discovery. Whatever is blocking
`lifting_object` from ever firing is not primarily an orientation-
discovery problem.

## Other Reward Terms (Full-Run Summary)

| Term | First | Last | Max | Nonzero |
|---|---|---|---|---|
| `reaching_object` | 0.000163 | 0.624200 | 0.634757 | 1500/1500 |
| `pregrasp_readiness` | 0.000166 | 1.236136 | 1.252569 | 1500/1500 |
| `orientation_alignment` | 0.071741 | 1.946610 | 1.967313 | 1500/1500 |
| `lifting_object` | 0.0 | 0.0 | 0.0 | 0/1500 |
| `object_goal_tracking` | 0.0 | 0.0 | 0.0 | 0/1500 |
| `object_goal_tracking_fine_grained` | 0.0 | 0.0 | 0.0 | 0/1500 |

## Key Comparison: Experiment 20 vs Experiment 18 and Experiment 17 (Final Values Only)

Per the established correction protocol (final-iteration snapshot vs.
final-iteration snapshot only).

- **Experiment 17 final `cube_reached_goal`:** 0.002360
- **Experiment 18 final `cube_reached_goal`:** 0.003499
- **Experiment 20 final `cube_reached_goal`:** 0.002116

Stated factually, not as an independent verdict per this project's
established practice for this single noisy bespoke-termination scalar:
Experiment 20's final value is close to Experiment 17's and somewhat
below Experiment 18's. The `lifting_object` nonzero-rate comparison
(0/1500 in all three) is the primary evidence, not this scalar.

## Assessment

**Mechanism verification:** the revised soft orientation-alignment bias
works exactly as intended — the strongest, most saturated dense
shaping signal recorded in this repo's history. This is not a case of
"the new term failed to provide signal" (unlike, hypothetically, a
term that stayed near zero); the policy definitively solved the
orientation-alignment sub-problem.

**Primary finding:** `Episode_Reward/lifting_object` stays at exactly
`0/1500`, identical to Experiment 17 and Experiment 18. This is a clean
falsification of Experiment 20's specific hypothesis (approach-orientation
discovery is a primary exploration bottleneck for finding an antipodal
grasp). Per the design spec's own success criteria, a null result with a
confirmed-working bias mechanism narrows rather than repeats the open
question — orientation was not the bottleneck.

**Where this leaves the research program:** four consecutive experiments
now (17, 18, 20, plus 19's mimic-joint investigation) have converged on
the same underlying fact — the cube never leaves the ground by any
margin, regardless of reward shaping (dense pre-grasp readiness),
orientation constraint (hard-locked or soft-biased), or a confirmed and
then-reverted mechanical fix attempt. This is a strong, cross-experiment
signal that the blocking factor is not any of: grasp-approach shaping,
orientation discovery, or (per Experiment 19) the specific mimic-joint
asset defect fixable via `PhysxMimicJointAPI`. Candidates not yet tried:
demonstration/imitation bootstrapping for the lift primitive specifically
(since reward shaping and action-space constraints have both been
exhausted without success), or a deeper look at whether antipodal
contact is ever being achieved at all under this orientation bias
(worth a Task-6-style instrumented rollout before deciding the next
experiment, to confirm whether the remaining bottleneck is "reaches
antipodal contact but can't lift" vs. "still never reaches antipodal
contact even with the orientation problem solved").

**Video inspection status:** not performed. Per the design spec's own
guidance and this project's established practice, `lifting_object`
remaining exactly `0/1500` with unambiguous scalar evidence means video
would add nothing new — there is no nonzero occurrence to visually
characterize.

## Addendum: Instrumented Antipodal-Contact Diagnostic

Following this report's own "Net assessment" recommendation, ran a
Task-6-style instrumented rollout of the final checkpoint
(`model_1499.pt`, 3 episodes, 750 total steps) reproducing
`antipodal_grasp_bonus`'s exact sub-condition computation inline, to
determine whether antipodal contact is ever reached at all now that
orientation-alignment is solved.

```
[SUMMARY] total_steps=750 height_ok_steps=0 both_magnitude_ok_steps=0
antipodal_ok_steps=0 grasp_ok_steps=0 gate_fires_steps=0
max_jaw1_force=0.0 max_jaw2_force=2.227085 max_cube_z=0.009049
min_orientation_dot=0.016509 max_orientation_dot=0.999778
```

**`height_ok_steps=0`, `max_cube_z=0.00905`** — consistent with every
prior experiment: the cube never leaves the ground by any margin
(spawn/resting height is 0.009).

**`max_orientation_dot=0.9998`** confirms the policy does achieve
near-perfect vertical alignment at times during this rollout (0.9998 ≈
1 degree off straight down) — direct, per-step confirmation the
mechanism works in practice, not just in the training-time aggregate
reward.

**The failure signature itself has changed from Experiment 17's Task 6
finding.** There, both jaws registered contact force simultaneously
(`both_magnitude_ok_steps=231/750`) in a non-antipodal wedge. Here,
**`both_magnitude_ok_steps=0/750` — `gripper_jaw1_joint`'s contact
sensor never registers any force at all across the entire rollout**
(`max_jaw1_force=0.0`), while `gripper_jaw2_joint` does register contact
intermittently (`max_jaw2_force=2.23N`). This is a genuinely different,
asymmetric failure mode, not a repeat of Experiment 17's wedge signature.

**Interpretation:** with the orientation problem now solved, the
gripper can approach vertically, but the two jaws are not making
symmetric contact with the cube at all — consistent with, though not
independently re-confirmed here, the mimic-joint mechanical defect
Experiment 17's Task 6 and Experiment 19's own investigation already
established as real and unresolved (jaw2 tracks its commanded position
20% worse than jaw1 under load). An asymmetric gripper that closes
unevenly would plausibly produce exactly this pattern: one jaw reaching
the cube, the other consistently falling short or drifting past it.
This reframes the mimic-joint defect's priority — Experiment 19 closed
out the specific `PhysxMimicJointAPI`-based fix as not viable, but the
underlying mechanical asymmetry itself remains a live, and now more
directly implicated, candidate blocker, independent of a `PhysxMimicJointAPI`-specific
solution (e.g. a different coupling mechanism, or hardware-level
verification of the real robot's actual mimic linkage).
