# Experiment 26: Gripper reintroduction (grasp/lift/carry/goal), 30s episodes

**Object:** cube. Direct continuation of [[experiment-25-touch-goal-reach]]
per direct user instruction: reintroduce the gripper (grasp/lift/carry/goal
back in scope) rather than continue narrowing the goal tolerance on the
reduced touch-goal task. Follows from ROADMAP.md item 11; item 10's own
follow-up entry records the jaw-mirroring retirement referenced below.
Experiments 15 through 24 remain uncompiled (the same acknowledged gap
`index.md`'s coverage-boundary note describes) — this article does not
backfill them.

## Hypothesis

Composing two previously-individually-validated fixes — Experiment 21's
proximity-gated gripper closure and Experiment 17's antipodal grasp gate —
with a 4-stage extension of Experiment 25's monotonic staged-potential
reward (reach → grasp → lift → goal, each stage floor a monotonic function
of a single scalar measure within its own stage, the exact property that
made Experiment 25's post-touch potential immune to the running-max
dead-zone bug — see [[staged-reward-co-satisfiability]]) and a
Stack-task-precedented 30.0s episode length (see [[hyperparameter-registry]]
for the Isaac Lab reference-task episode-length derivation Experiment 25
already established) would produce measurably more reliable grasp discovery
than any prior single-variable attempt, because the compound behavior's
obstacles (premature/asymmetric jaw closing, insufficient episode time
under the corrected physics) are addressed together rather than one at a
time, building on positioning precision Experiment 25 already validated.

A third originally-planned fix — Experiment 22's jaw-mirroring mechanism —
was part of the design's initial three-fix hypothesis but was found, during
final whole-branch review after implementation, to be an inert no-op (see
`tasks/ar4/pickplace_graspgoal_env_cfg.py`'s `ActionsCfg` docstring for the
full account, and ROADMAP.md item 10's own follow-up entry). It was retired
rather than carried forward, amending the hypothesis to rest on the two
remaining composed fixes; jaw asymmetry under contact is treated as a
possibly-genuine physics-level condition for the antipodal gate to tolerate
or reject on its own merits, not a defect this design still claims to fix.

**Falsifiable as:** if the antipodal contact condition (or the new staged
milestone's grasp-stage component) still shows `0` or near-`0` nonzero rate
after a full 1500-iteration run under this composed setup, the hypothesis
is falsified — the remaining bottleneck is not (only) the addressed
obstacles.

## Design

`tasks/ar4/pickplace_graspgoal_env_cfg.py`. Reward/termination/observation
functions added to `tasks/ar4/mdp.py`, extending Experiment 25's
Isaac-Lab-free `touch_goal_reward.py` pattern to four stages (reach floor
0.0, grasp floor 0.25, lift floor 0.50, goal floor 0.75, ceiling 1.0). New
`grasp_state_observation` term (a 2-element latched `[grasped, lifted]`
float pair) added to the observation space. Design:
`docs/superpowers/specs/2026-07-09-ar4-experiment26-gripper-reintroduction-design.md`.
Plan:
`docs/superpowers/plans/2026-07-09-ar4-experiment26-gripper-reintroduction-implementation.md`.

## Result

Trained for the full 1500 iterations.
`Episode_Termination/cube_reached_goal` stayed at exactly `0.0000` for the
entire run — not a single logged point showed any nonzero value.
`Episode_Reward/grasp_goal_milestone_bonus` rose from `0.0001` to `~0.0037`
within the first ~15 iterations, then stayed completely flat at that value
for the remaining ~1485 iterations. `Episode_Termination/time_out` was
`1.0000` throughout — every episode ran the full 1500-step/30s length.
Training itself showed no instability signature (`Loss/value_function`
≈0, `Mean action noise std` ≈1.0-1.1 throughout).

A new front, head-on close-up camera
(`tasks/ar4/graspgoal_democam_env_cfg.py`, built per direct user request,
after two framing corrections during construction — wrong height, then
wrong side of the robot entirely, confirmed by the rendered image showing a
rear access panel instead of the front) was used to check this directly.
A 4-env deterministic rollout of the final checkpoint (`model_1499.pt`)
shows 0/4 envs ever grasping, lifting, or reaching the goal. Frames sampled
at 3-second intervals throughout the 30-second episode are visually
identical from step 1 to episode timeout (step 1500): **the trained policy
holds one completely static pose for the entire episode and never even
attempts to move toward the cube** — not a reach that stalls partway, not
a grasp followed by a freeze, no motion at all from the first logged step.

Building this camera also surfaced and fixed a real, separate bug,
unrelated to the training result itself: both the new
`scripts/graspgoal_closeup_video.py` and the existing
`scripts/touchgoal_closeup_video.py` (sharing the same
`camera.data.output["rgb"]`-reading code) were saving every frame
vertically flipped — an OpenGL framebuffer row-order convention
(row-0-at-bottom) never corrected before writing to PNG/mp4, confirmed
empirically (an unflipped render showed the ground grid at the top of frame
and sky at the bottom). Fixed in both scripts. Experiment 25's own
touch-goal video-review conclusions were based on relative
position/distance state, not pixel interpretation, so are not invalidated
by this — the finding is recorded because the images themselves were
genuinely inverted and this is exactly the kind of silent, easy-to-miss
rendering bug worth flagging for reuse elsewhere (see
[[sim-physics-fidelity]]).

## Verdict

**Falsified, and by a qualitatively more severe failure signature than
anything else in this project's history.** Every prior null result in this
project's whole reach-grasp-lift arc — sphere-era "reach, grip, freeze,"
Experiment 25's touch-goal near-miss — involved at least some genuine
motion: a reach attempt, sometimes a real grasp, before the policy stalled
or froze. Here there is no reach attempt at all. The policy is static from
step 1, for the full 30-second episode, in all 4 inspected envs. This is a
complete freeze, not a partial one — precisely the distinction this article
exists to record, and the reason it is a new, more severe point on the
[[reach-grasp-lift-gap]] throughline rather than a repeat of the "reach,
grip, freeze" pattern.

Since `--touchgoal` (arm-only, 2-stage, no gripper) reliably converges
under the same physics/PPO setup Experiment 25 already validated, the
regression specifically implicates something introduced by the
reintroduced gripper action/observation surface (action dimension 7→8,
adding the gripper's binary action; observation dimension 24→31, adding
the new `grasp_state` term) or the reward's `reach` segment specifically
failing to provide a usable gradient under the new 4-stage formula — not a
general breakdown of PPO or physics under this task family. Not yet
root-caused; flagged as the next investigation, not pursued further in
this pass.

## Related concepts

[[reach-grasp-lift-gap]] — this experiment adds a new, more severe closing
stage to that article's whole arc: complete static freeze from step 1,
distinct from every prior "at least some motion" failure signature.
[[staged-reward-co-satisfiability]] — the 4-stage monotonic potential this
design's reward reuses is a direct extension of Experiment 25's
dead-zone-immune mechanism; the design's hypothesis explicitly credits this
mechanism as the reason this attempt should differ from Experiment 16's
ungated-sum shape and Experiment 17/18's separately-weighted gated terms.
[[hyperparameter-registry]] — the 30.0s episode length is inherited
directly from Experiment 25's own Isaac-Lab-reference-task derivation
(Stack task, 30.0s, the closest structural analog to this 4-stage task);
see that article for the derivation itself.

## Sources

`docs/superpowers/specs/2026-07-09-ar4-experiment26-gripper-reintroduction-design.md`,
`docs/superpowers/plans/2026-07-09-ar4-experiment26-gripper-reintroduction-implementation.md`,
`ROADMAP.md` item 11 (and item 10's own follow-up entry for the
jaw-mirroring retirement referenced above).
