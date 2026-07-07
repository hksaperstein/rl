# Experiment 6: Mirror-scene + grasp-gated stillness penalty

**Object:** sphere. User-directed sixth attempt (not unilateral, per
[[experiment-05-potential-based-reward-shaping]]'s note): the user raised
two concrete ideas in parallel — a grasp-gated movement incentive and a
penalty for staying static within a bound.

## Hypothesis

Full-workspace spawn randomization plus a mirrored opposite-side goal
removes any possible left/right positional degeneracy, and an explicit
penalty for staying still *after* a grasp is achieved directly counteracts
whatever is making "hold a static grip" a locally-optimal terminal state.

## What changed

New single-sphere scene with randomized spawn and a mirrored opposite-side
goal (`docs/superpowers/specs/2026-07-06-ar4-sphere-mirror-scene-design.md`).
A corrected *undiscounted* running-max milestone bonus (drops the `gamma`
decay that caused [[experiment-05-potential-based-reward-shaping]]'s bug),
and a new grasp-gated `stillness_penalty`.

**A sign-convention bug was found during the first training run (before
eval) and fixed before proceeding**: `stillness_penalty`'s own function
body already returns the signed value (`-1.0` when triggered, `0.0`
otherwise), but its `RewardsCfg` registration used `weight=-2.0`.
`RewardManager.compute()` computes `func(...) * weight * dt` — multiplying
two negatives turned the intended *penalty* into a **+2.0*dt reward** for
the exact stay-still-after-grasp behavior this term exists to punish.
Caught by reading the actual TensorBoard data (`Episode_Reward/stillness_penalty`
grew to +1.3 over training, impossible for a true penalty) rather than
trusting the design doc's own stated intent. Fixed to `weight=2.0`; the
first run's data was discarded and Task 4 was re-run with the corrected
weight before any eval/video judgment.

## Quantitative result

On the corrected checkpoint (`logs/train/2026-07-06_16-02-16/model_1499.pt`):
what's newly confirmed working is full-workspace spawn randomization, the
mirrored opposite-side goal mechanism, the stillness-penalty sign fix
(confirmed non-positive throughout training), and the corrected
undiscounted milestone-bonus formula (confirmed non-negative and growing,
no decay bug).

## Qualitative video finding

10-episode real eval: **0/10 episodes show a genuine, controlled grasp-and-
lift.** One episode (5) initially looked like a lift in a coarse
start/25%/50%/75%/end sample, but a full frame-by-frame re-inspection (022
through 050) showed the sphere separating from the gripper and drifting to
a hover disconnected from the arm — the gripper stays static near the
ground throughout, never co-located with the airborne sphere again after
first contact. Far more consistent with the tiny object (0.01kg, 9mm
radius) being knocked/launched by a glancing collision with the arm's body
than with a bilateral grasp (`contact_grasp_bonus` requires simultaneous
force on *both* jaws, which a glancing knock from one link wouldn't
satisfy).

## Verdict

**Falsified.** None of the newly-fixed/confirmed mechanisms changed the core
outcome — the gripper still never achieves and holds a bilateral grasp in
any eval episode. Sixth real attempt on the reward/optimization axis for
this sub-problem. Per the systematic-debugging Phase 4.5 mandate, flagged
back rather than attempting a seventh tweak unilaterally. Candidates
surfaced for consideration: the gripper's ~28mm max aperture vs. the
sphere's 18mm diameter may leave too little margin for the joint-position
action space to reliably converge on a stable grasp pose (tested next, see
[[experiment-07-sphere-shrink]]), or a hierarchical policy separating
reach-to-pregrasp and close-gripper phases.

## Related concepts

[[reward-rate-arithmetic]] — the sign-convention bug here (an already-signed
function combined with a negative weight, silently flipping a penalty into
a reward) is a direct precedent for Experiment 12's later, subtler
reward-rate imbalance in the same `stillness_penalty` term. [[reach-grasp-lift-gap]]
— sixth attempt, still no lift.

## Sources

`docs/superpowers/specs/2026-07-06-ar4-sphere-mirror-scene-design.md`,
`docs/superpowers/plans/2026-07-06-ar4-sphere-mirror-scene-implementation.md`,
`docs/superpowers/plans/2026-07-06-ar4-sphere-mirror-scene-report.md`
