# Experiment 15: Ground penalty + base proximity penalty

**Object:** cube. Direct user-requested reward-shaping changes (ground and
base contact penalties) built on [[experiment-12-stillness-reward-rate]]'s
clean baseline, after two consecutive non-improving results (Experiments 13,
14) triggered the escalation mandate. Best outcome-metric scalars of the
session, but the new penalties saturated and converged to the opposite
behavior than designed, revealing a recurring cross-experiment pattern.

## Hypothesis

Explicitly penalizing ground contact (existing `ground_penalty` function,
wired in for the first time) and adding a new penalty for the cube being
proximate to the robot's own base (new `base_proximity_penalty`) — while
raising `antipodal_grasp_bonus`'s weight to reward successful grasping — will
improve success rates by deterring failure modes (premature ground contact,
cube collapsing toward the base) without requiring changes to action space
or reset structure.

## What changed

New `base_proximity_penalty` function (`tasks/ar4/mdp.py`) implementing cube
xy-distance to robot root origin, distinct from `ground_penalty`'s z-height
check per explicit user instruction. Previously-unused `ground_penalty` wired
in. Weight adjustments: `antipodal_grasp_bonus` 3.0 → 4.0, `stillness_penalty`
5.0 → 6.0 (preserving the [[reward-rate-arithmetic]] -2.0/step anti-freeze
margin). New `Ar4PickPlaceBaseProximityEnvCfg` (`tasks/ar4/pickplace_baseproximity_env_cfg.py`), reusing Experiment 12's action/observations/events/terminations
unchanged — isolating reward function as the only new variable.

## Quantitative result

Diagnostic (300 iter) showed a single-iteration `Loss/value_function` spike
to 17.66 at step 39 — roughly 100x any prior accepted spike — but isolated,
single-occurrence, and decaying within 10–15 iterations. Full run (1500 iter)
confirmed the identical spike recurring at step 39 (to 4+ significant figures)
with no other peaks above 1.0 and no sustained upward drift — flagged for
scrutiny but not disqualifying.

Scalar comparison is the most consistently positive of the session. `cube_reached_goal`
improved +59.7% versus Experiment 12 (0.010773 → 0.017202) and +51.0% versus
Experiment 14 (0.011393 → 0.017202) — best final-iteration success rate of
any experiment this session. `antipodal_grasp_bonus` rose +159.8% versus
Experiment 12 (0.012777 → 0.033199) with clear monotonic climb mid-run (unlike
Experiment 14's collapse). `stillness_penalty` worsened modestly (-46.8%),
`path_proximity_bonus` declined slightly (-4.5%).

**However, both new penalty terms failed to behave as designed.** `ground_penalty`
never trended down, saturating at 100% across all training windows. `base_proximity_penalty`
rose from 12.0% in first 150 iterations to 100.0% saturation for the last
1050 of 1500 iterations — the opposite of the intended low steady-state.
This indicates the policy actively converged toward base-proximate states as
training progressed, not merely unlucky spawn proximity.

## Qualitative video finding

3 of 10 recorded episodes (25 frames each) inspected. Episode 1: arm reaches
down near cube by frame 5 and holds static pose for remainder — the established
"reach and freeze" signature, no lift. Episode 3: identical pattern. Episode 2:
reproduces Experiment 14's base-collapse failure — arm progressively curls
tighter against its own base over the episode, and cube ends up immediately
adjacent to the base by final frames. None of the 3 show lift or reach waypoint
index ≥2; the design spec's stated success criterion is not met on this sample,
though the improved scalar success rate (1.7% final) suggests larger samples
would likely show occasional successes.

## Verdict

**Best outcome metrics of the session, but the explicit penalty against a
failure mode did not prevent it and moved in the wrong direction — do not
extend this exact `base_proximity_penalty` formulation with further tuning.**
The cross-experiment pattern of base-collapse (seen here, in Experiment 14,
and independently in the classical demo's singularity-stall investigations)
now appears in three structurally different mechanism changes, suggesting a
fundamental attractor in the action space or task geometry rather than a
pure reward-incentive artifact. Adding pressure via weight raises risks
repeating the "tuning without fixing mechanism" pattern already observed
not working here. Recommend a dedicated investigation into the base-collapse
attractor (kinematic singularity, self-collision, or geometric sink) before
further reward tuning in this family, alongside the still-queued episode-
length/staged-decomposition direction from Experiment 11.

## Related concepts

[[reach-grasp-lift-gap]] — fourth data point (Experiments 12, 13, 14, 15) in
the persistent reach-grasp-lift bottleneck, now with a recurrent base-collapse
side effect seen twice. [[reward-rate-arithmetic]] — `antipodal_grasp_bonus`
and `stillness_penalty` weight increase preserves the anti-freeze margin.

## Sources

`docs/superpowers/specs/2026-07-07-ar4-experiment15-reward-shaping-design.md`,
`docs/superpowers/plans/2026-07-07-ar4-experiment15-report.md`
