# Experiment 12: fix the antipodal/stillness reward-rate imbalance

## Context

Experiment 11 (task-space differential-IK action,
`docs/superpowers/specs/2026-07-06-ar4-taskspace-ik-action-design.md`)
produced the first genuine, sustained antipodal grasp contact in this
project (`antipodal_grasp_bonus` nonzero in 91.6% of logged iterations).
But real eval video showed the arm reaches down, grasps, and then holds a
static low near-ground pose for the rest of the ~5s episode — no
lift-to-height, no carry-to-goal
(`docs/superpowers/plans/2026-07-06-ar4-experiment11-report.md`, ROADMAP.md).

## Root cause

`tasks/ar4/pickplace_taskspace_env_cfg.py`'s `RewardsCfg` currently pays a
net-*positive* reward for holding a grasp without making further progress:

- `antipodal_grasp_bonus` (weight 3.0): fires every step the bilateral
  antipodal contact condition holds, uncapped, unconditional on further
  progress toward lift/place → **+3.0/step**, continuous.
- `stillness_penalty` (weight 2.0): only fires (-1.0) once the object has
  been within `still_bound` (5mm) of a reference position for more than
  `patience_steps` (25 control steps, 0.5s) → **-2.0/step**, but only
  *after* that grace window.
- `gripper_schedule_bonus` and `path_proximity_bonus`'s milestone deltas
  don't pay out during a hold at the grasp waypoint — they require the
  waypoint index to have already advanced past lift, which requires
  actually moving up.

Net during a hold-after-grasp episode: `+3.0/step` for the first 25 steps,
then `+3.0 − 2.0 = +1.0/step` for the remaining ~200 steps of a 250-step
episode (`episode_length_s=5.0`, control `dt=0.02`) — on the order of
**+250 accumulated reward from freezing alone**, a large, low-variance,
guaranteed payout versus the one-time, riskier (grasp-losing) reward from
actually lifting/carrying/placing.

This is the same reward-rate-imbalance failure class documented for
Experiment 9's `contact_grasp_bonus` (118:1 ratio,
`docs/superpowers/specs/research/2026-07-06-rl-manipulation-senior-b.md`),
resurfacing in the *rebalanced* replacement term: Experiment 9's fix
closed the ratio from ~9:1 to ~3:2 (comment at
`pickplace_ik_guided_env_cfg.py:164-171`) but never verified the sign
actually flipped. It didn't — holding is still profitable, just less
overwhelmingly so. This is a newly-exposed consequence, not a
previously-tried-and-rejected fix: Experiments 9/10 used the same weights
under joint-space action, where `antipodal_grasp_bonus` never fired at all
(stuck at 0.0/0.001416), so this specific incentive was never actually
live until Experiment 11's task-space action finally made grasp
achievable.

## Fix

Raise `stillness_penalty`'s weight from 2.0 → 5.0 in
`tasks/ar4/pickplace_taskspace_env_cfg.py`'s `RewardsCfg` only. Net
post-patience reward for grasp-without-progress becomes
`3.0 − 5.0 = −2.0/step` — clearly negative, not razor's-edge. Nothing else
changes: same `antipodal_grasp_bonus` weight/threshold, same
`patience_steps`/`still_bound`, same action space, same waypoints, same
scene.

Scope: this file only. `pickplace_mirror_env_cfg.py` and
`pickplace_ik_guided_env_cfg.py` (legacy/superseded configs, still used by
other tooling per their own file docstrings) and `tasks/ar4/mdp.py`'s
shared `stillness_penalty`/`antipodal_grasp_bonus` function bodies are
untouched — only the weight parameter passed by this one `RewardsCfg`
changes.

## Why this, and why alone

This repo's established pattern (Experiments 9, 10, 4b) is to isolate one
verified-real bug per run rather than stack unvalidated guesses. This is a
verified arithmetic bug (the weights themselves, not a hypothesis about
what might help), so it's the highest-confidence single change available.
The previously-queued ideas (longer episodes, explicit staged
sub-objectives, richer goal placement) stay queued behind this — worth
revisiting only if flipping this incentive alone doesn't unlock lift/carry
behavior, since a null result here would otherwise be confounded with
those other untested variables.

## What this fix does NOT address

`still_bound`'s "moved" check is any-direction Euclidean displacement, not
progress toward the current waypoint — a policy could in principle
discover a small in-place wiggle that resets the stagnancy counter every
`patience_steps` without net progress, evading the penalty indefinitely.
Not fixed here (would be a second, unverified change) but explicitly
flagged: eval video review should check for a wiggling-in-place signature
in addition to a genuine lift, since this fix's diagnostic run is also the
first time this exact incentive structure will run for 1500 iterations at
all.

## Verification plan

1. Diagnostic run (~300 iterations): confirm `Episode_Reward/stillness_penalty`
   and `Episode_Reward/antipodal_grasp_bonus` move as predicted (stillness
   penalty firing then declining as freezing gets discouraged; antipodal
   bonus still present, not collapsing to 0 the way it did when the
   geometric threshold was miscalibrated in Experiment 10), and that
   `path_proximity_bonus` starts crediting waypoint index ≥2 (lift) more
   than Experiment 11's near-zero baseline for that stage. No critic-
   divergence check needed here (that was tied to the task-space IK action
   term itself, unchanged since Experiment 11's fix; a pure reward-weight
   change doesn't touch the action pipeline).
2. If the diagnostic looks directionally right, full 1500-iteration run.
3. Eval (10 episodes) + real video frame inspection (not just scalar
   trends) — specifically checking for genuine lift-off-the-ground motion,
   and separately checking for the in-place-wiggle evasion pattern flagged
   above.
4. Record result in ROADMAP.md regardless of outcome (positive, null, or a
   new failure signature), per this project's standing practice of keeping
   negative results as part of the research record.

## Success criteria

Not "full pick-and-place solved" in one shot — the bar is genuine forward
progress on the specific symptom Experiment 11 left open: the arm actually
lifting the cube off the ground in at least some fraction of eval
episodes, and `path_proximity_bonus` milestone data showing the policy
reaching waypoint index ≥2 substantially more than Experiment 11's
baseline.
