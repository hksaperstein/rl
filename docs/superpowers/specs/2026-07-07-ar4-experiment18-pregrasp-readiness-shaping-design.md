# Experiment 18: dense pre-grasp-readiness shaping

## Hypothesis

**Experiment 17's binary antipodal gate correctly closed Experiment 16's
exploit, but left the policy with zero reward gradient toward the
compound "get near the object AND close the gripper around it"
behavior — confirmed directly by Task 6's instrumented rollout, which
showed the two halves of a grasp being explored independently but never
combined: one episode showed the gripper fully closing, but nowhere
within 5.9cm of the cube; a separate event showed the arm driving within
2.6cm of the cube, but with the gripper pinned open, not closing.**
Adding a dense shaping term that rewards proximity to the object
*specifically when the gripper is closing*, not proximity alone (which
`reaching_object` already provides and evidently isn't sufficient on its
own), should give the policy a continuous incentive to combine both
halves into one coordinated behavior, without reintroducing a
hackable substitute for genuine antipodal contact — the binary gate from
Experiment 17 stays in place, unchanged, as the only path to the large
`lifting_object`/`object_goal_tracking` reward.

Falsifiable: if `Episode_Reward/lifting_object` still shows `nonzero: 0`
across a full run despite this new term, the missing-gradient hypothesis
is wrong or the new term itself fails to provide a usable gradient, and
the exploration bottleneck lies elsewhere (e.g. genuinely in the
mimic-joint asset defect, which this experiment does not fix).

## Background research

**Task 6's own instrumented evidence** (already-confirmed root cause for
this experiment, not a new hypothesis to re-establish —
`docs/superpowers/plans/2026-07-07-ar4-experiment17-report.md`,
`.superpowers/sdd/task-6-report.md`): across 1,487 rollout steps of
Experiment 17's trained checkpoint, `height_ok` was true 0 times — the
cube never left the ground by any margin. The one real contact event
(230 steps) was a static wedge with the gripper pinned *open*
(`jaw1_joint_pos=0.014`, fully open) — the arm got close but did not
close its gripper. A separate episode showed the gripper reaching full
closure, but never within 5.9cm of the cube — the arm closed its
gripper but not near the object. These are complementary failures, not
the same one: the policy has independently explored "get close" and
"close the gripper," never both together.

**Xu et al., "Stage-Transition Dense Reward Modeling for Reinforcement
Learning" (arXiv:2606.31377)** — already verified from source for
Experiment 17's design, re-cited here for its complementary half: their
framework's *other* signal (alongside the grasp-verification gate
already adapted in Experiment 17) is "within-stage progress feedback"
— dense, continuous reward for progress *toward* a stage transition, not
just a binary pass/fail at the transition itself. This experiment adds
exactly that missing half for the pre-grasp stage specifically.

**"Comparing Task Simplifications to Learn Closed-Loop Object Picking
Using Deep Reinforcement Learning" (arXiv:1803.04996)** — abstract
verified directly from source: a real, published comparison of
reward-shaping, curriculum-learning, and warm-start approaches for
RL-based closed-loop object picking with a robotic manipulator. Cited
here only for its confirmed scope (dense reward shaping is a standard,
actively-studied lever for this exact problem class) — a secondary
websearch summary claimed a specific quantitative ablation result from
this paper's body ("removing EE-object distance reward causes drastic
deterioration"), which could not be independently verified from the PDF
extraction and is **not** relied on here, consistent with this project's
citation-verification standard.

## Design

New file `tasks/ar4/pickplace_pregrasp_env_cfg.py`
(`Ar4PickPlacePregraspEnvCfg`) — built directly on Experiment 17's
`Ar4PickPlaceGraspGatedEnvCfg` structure (same scene, action, curriculum,
observations, events, terminations, and — critically — the exact same
grasp-gated `lifting_object`/`object_goal_tracking` terms, unchanged),
with one new reward term added.

**New dense shaping term**, appended to `tasks/ar4/mdp.py`:

```python
def pregrasp_readiness_bonus(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    gripper_joint_names: list[str],
    open_pos: float,
    closed_pos: float,
) -> torch.Tensor:
    """Dense reward for combining proximity AND gripper closure - the two
    halves Task 6's instrumented rollout showed being explored
    independently but never together (Experiment 17: one event showed
    the gripper fully closed nowhere near the cube; another showed the
    arm within 2.6cm of the cube with the gripper pinned open). Reward is
    the product of a proximity term (same tanh-kernel shape as
    reaching_object) and a normalized "closedness" term (1.0 when the
    gripper is fully closed, 0.0 when fully open) - maximized only when
    both are true simultaneously, giving zero credit for closing far
    from the object or approaching without closing. Does NOT reward
    antipodal alignment or contact force - purely a positional/
    configuration signal, kept deliberately weaker/less specific than
    antipodal_grasp_bonus's own force-closure check, which remains the
    only gate for lifting_object/object_goal_tracking, unchanged from
    Experiment 17. See
    docs/superpowers/specs/2026-07-07-ar4-experiment18-pregrasp-readiness-shaping-design.md.
    """
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    dist = torch.norm(object.data.root_pos_w - ee_pos_w, dim=-1)
    proximity_term = 1.0 - torch.tanh(dist / std)

    gripper_joint_ids, _ = robot.find_joints(gripper_joint_names)
    gripper_pos = robot.data.joint_pos[:, gripper_joint_ids].mean(dim=-1)
    closedness_term = torch.clamp((open_pos - gripper_pos) / (open_pos - closed_pos), 0.0, 1.0)

    return proximity_term * closedness_term
```

**`RewardsCfg`** (only one term added relative to Experiment 17; all
other terms — `reaching_object`, `lifting_object`, `object_goal_tracking`,
`object_goal_tracking_fine_grained`, `action_rate`, `joint_vel` —
unchanged, identical weights):

```python
pregrasp_readiness = RewTerm(
    func=ar4_mdp.pregrasp_readiness_bonus,
    weight=2.0,
    params={
        "std": 0.1,
        "object_cfg": SceneEntityCfg("cube"),
        "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        "robot_cfg": SceneEntityCfg("robot"),
        "gripper_joint_names": GRIPPER_JOINT_NAMES,
        "open_pos": GRIPPER_OPEN_POS,
        "closed_pos": GRIPPER_CLOSED_POS,
    },
)
```

`weight=2.0` — twice `reaching_object`'s weight (1.0), since this term is
strictly harder to satisfy (requires both proximity AND closure
simultaneously, where `reaching_object` requires only proximity) and is
meant to meaningfully compete with the policy's existing incentive to
just sit at a comfortable reaching distance without closing. Kept well
below `lifting_object`'s weight (15.0), since this is explicitly a
stepping-stone signal toward the gated reward, not a replacement for it
— per this repo's own established reward-rate-arithmetic discipline
(verify the incentive doesn't dominate or replace the real objective),
`pregrasp_readiness`'s own per-episode ceiling (weight × max raw value ≈
2.0) stays an order of magnitude below `lifting_object`'s.

**Not a test of the user's stronger "cube tracks the EE's motion"
refinement** — that specific idea (checking whether cube-to-EE distance
stays small *while the EE is actually moving*, not just at a single
instant) is a better test for verifying a *held* grasp than for shaping
*approach* behavior, and is earmarked as a candidate refinement to
`genuine_grasp_and_lift`'s own gate (replacing or supplementing the
antipodal check) in a later experiment, contingent on this experiment's
own result and a dedicated look at whether Isaac Lab exposes body
velocity/relative-motion data cleanly enough to compute it as a reward
term. Kept separate here to isolate one new variable at a time.

## What this does NOT change

No modification to `tasks/ar4/pickplace_graspgated_env_cfg.py` or any
existing function in `tasks/ar4/mdp.py` (including `antipodal_grasp_bonus`,
`genuine_grasp_and_lift`, `lifting_object_grasp_gated`,
`mirrored_goal_distance_grasp_gated`, all reused unchanged) — purely
additive (one new function, one new env cfg file). Does not attempt to
fix the confirmed mimic-joint asset defect (a separate, independent
thread, not yet investigated).

## Verification plan

Same sequence as every Tier-1 experiment: smoke test, 300-iteration
diagnostic, full 1500-iteration run + report comparing against
Experiment 17's and Experiment 12's final values. Report must explicitly
state `pregrasp_readiness_bonus`'s own nonzero rate/trend (confirming the
new term actually provides a usable gradient at all) and, critically,
whether `lifting_object`'s nonzero rate moves off zero at any point —
the specific, falsifiable question this experiment exists to answer.
Video inspection only becomes relevant if the gate does start firing;
if `lifting_object` stays at 0/1500 again, video adds nothing new (per
Experiment 17's own finding, there'd be nothing to inspect) and the
report should say so rather than performing a video-review step with no
real content.

## Success criteria

Primary: does `Episode_Reward/lifting_object`'s nonzero rate move off
zero at any point in the full run (any real, non-zero count) — even a
small nonzero rate would be a qualitatively different, informative
result versus Experiment 17's exact zero. Secondary: does
`pregrasp_readiness_bonus` itself show real growth over training
(confirming the new term is being discovered/exploited as intended,
prerequisite evidence for the primary criterion to have a chance). A
null result (still 0/1500 on `lifting_object`) would specifically
implicate the mimic-joint asset defect or a still-too-large a
discoverability gap even with this shaping, rather than "no shaping was
tried" — narrowing, not repeating, the open question.
