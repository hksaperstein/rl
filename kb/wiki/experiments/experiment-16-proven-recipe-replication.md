# Experiment 16: Proven-recipe replication

**Object:** cube. Direct user request to research actual working RL-manipulation
examples and replicate them on the AR4+cube from scratch. Built on independently-published
Isaac Lab and IsaacGymEnvs references, not on prior in-repo baselines.

## Hypothesis

Isaac Lab's own Franka Cube Lift task and IsaacGymEnvs' FrankaCubeStack both (a) never
reward grasp quality as a standalone term, and (b) multiplicatively gate goal-tracking
reward behind an actual lift condition — structurally unlike Experiments 1–15, which
kept ungated grasp-quality terms. This reward-gating structure, identified in the repo's
own [[reach-grasp-lift-gap]] research, should resolve the lift emergence failure seen
across that prior arc.

## What changed

New `Ar4PickPlaceProvenRecipeEnvCfg` (`tasks/ar4/pickplace_provenrecipe_env_cfg.py`)
with 6 reward terms, replicating both reference implementations exactly: `reaching_object`
and `lifting_object` copied directly from Isaac Lab's installed source (unmodified), plain
binary per-step lift reward (not milestone/running-max bonus), goal-tracking reward gated
on lift, plain joint-space action (reverting from this session's task-space lineage), and
Isaac Lab's curriculum manager (regularization-weight curriculum) — the repo's first use
of it.

## Quantitative result

Diagnostic phase flagged an unusual `Loss/value_function` shape: climbing rather than
spiking-then-recovering. Fully resolved by full run with direct causal explanation: loss
peaked at 4.588 at iteration 417 — the exact step the curriculum mechanism fired,
bumping `action_rate`/`joint_vel` weights 1000x as designed — then declined 93.5% to
0.298 by run end, settling into a structurally elevated (~0.25–0.45) equilibrium.

Scalar picture mixed and precisely stated: `cube_reached_goal` final value (0.008962)
regressed from Experiment 12 (−16.8%) and Experiment 15 (−47.9%). But `lifting_object`
and `object_goal_tracking` grew strongly and monotonically across the full run: both
reached 100% nonzero rate by iteration ~150; `lifting_object`'s per-window average
climbed ~220x (0.05 → 12.1). The two signal families point in opposite directions —
precisely the kind of ambiguity the repo's correction protocol resolves via video, not
scalar alone.

## Qualitative video finding and correction

**Initial observation (INCORRECT):** 3 of 10 recorded episodes, personally inspected,
appeared to show the arm reaching the cube within ~1–2 seconds, grasping it, lifting it
visibly off the ground, and holding it elevated for the episode duration. This looked
qualitatively different from Experiments 1–15, all of which showed either static low
holds or collapses toward the robot base — never sustained elevation.

**Correction (same day, user-prompted and instrumented verification):** The "genuine
grasp/lift" claim is **wrong**. The cube is not gripped by the gripper jaws at any point.
A fresh instrumented rollout of the exact checkpoint logged gripper jaw contact forces
(`force_matrix_w`, the same field the antipodal grasp check reads) and joint positions
across a full episode: both jaw contact sensors read exactly 0.0 at every step, including
the initial approach — the gripper never registers contact. During the "held" period
(steps 80–248 of 250), `gripper_jaw1_joint` sits at ≈0.014 (essentially fully open), and
the cube's distance to the wrist base (link_6/gripper_base_link ≈0.023m) is consistently
smaller than its distance to either jaw (≈0.051–0.056m). **Root cause: the cube is being
wedged and carried via contact with the wrist/gripper-housing body as the arm reorients,
not gripped by the fingers.** This height-only reward function, faithfully matching both
proven references, has no requirement for genuine grasp-force closure; the policy found a
cheaper exploit.

Secondary finding: `gripper_jaw1_joint`/`gripper_jaw2_joint` do not track each other
despite the source URDF's explicit mimic constraint — Isaac Sim's USD import does not
enforce it, so the two jaws behave as independently-actuated rather than coupled.

## Verdict

**This is not "grasp/lift solved" — grasp is being bypassed, not solved.** The experiment
does constitute a qualitatively new behavior relative to every one of Experiments 1–15's
video samples (none ever moved the cube meaningfully off the ground at all), confirming
the reward-gating change altered behavior. However, the corrected interpretation inverts
the next step: genuine grasp-contact requirements must gate the lift/goal-tracking reward
(e.g., requiring bilateral jaw force before `lifting_object` fires) rather than building
on this checkpoint's gap-to-goal behavior as if grasp were already solved. This
preservation of the "grasp earns no direct reward" principle — gating instead of direct
reward — becomes Experiment 17's hypothesis, not a quick tuning patch.

## Related concepts

[[reach-grasp-lift-gap]] — this experiment narrows the qualitative gap (cube actually
leaves the ground, unlike Exps 1–15's stasis) but via a wrist-wedging exploit rather
than real grasp. [[grasp-mechanics-antipodal-vs-magnitude]] — this experiment surfaces
the core structural gap: height-only lift reward has no antipodal or force-closure
requirement at all, enabling the exploit and motivating Experiment 17's grasp-gate.

## Sources

`docs/superpowers/specs/2026-07-07-ar4-experiment16-proven-recipe-replication-design.md`,
`docs/superpowers/plans/2026-07-07-ar4-experiment16-report.md`
