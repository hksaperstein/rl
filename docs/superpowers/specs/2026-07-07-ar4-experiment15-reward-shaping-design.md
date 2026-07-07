# Experiment 15: ground contact, base-proximity, and grasp-weight reward shaping

## Context

Direct user instructions for the next experiment (2026-07-07): "negative
reward for contacting the ground. higher reward for the cube being in the
grasp position", then a refinement: "negative reward for the cube contacting
the base of the robot" (specifically x/y proximity to the robot's own base,
not the existing z-height ground check — an explicitly separate term).

Experiment 14 (reach-skip curriculum) is now closed out
(`docs/superpowers/specs/2026-07-07-ar4-reachskip-curriculum-design.md`,
ROADMAP entry): no improvement on its own success criterion (0/3 sampled
eval episodes showed lift), and its video evidence surfaced a new failure
mode not present in Experiments 12-13 — in 2 of 3 episodes the arm folded
into an increasingly compact, contorted pose near its own base rather than
holding a stable position near the cube. That finding independently
motivates the user's base-proximity request, beyond the original ask: if the
arm's own tendency to collapse toward its base is part of what's going wrong,
a term that makes the cube's proximity to the base costly gives the policy a
reason to actively avoid that region rather than drift into it.

**Baseline choice: Experiment 12's clean env cfg
(`pickplace_taskspace_env_cfg.py`), not Experiment 14's reach-skip
mechanism.** Experiment 14 did not demonstrate an improvement to build on,
and its one-shot IK reset event is a separate, still-not-validated
mechanism (design spec's own noted caveat: the reset lands "near," not
exactly at, the pregrasp target, a plausible contributor to the new
base-collapse failure). Layering new reward terms on top of an unvalidated,
possibly-implicated mechanism would confound whether any observed change
comes from the reward shaping or from reach-skip's own side effects.
Building on Experiment 12's full-episode, fixed-home-pose baseline isolates
the three new reward terms as the only new variable — consistent with this
project's practice in every experiment this session (Experiment 13 was also
explicitly built on Experiment 12, not on Experiment 11, for the same
isolation reason).

## Design

Three changes, all reward-side only — no action space, curriculum, or scene
change. New file `tasks/ar4/pickplace_baseproximity_env_cfg.py`
(`Ar4PickPlaceBaseProximityEnvCfg`), modeled directly on
`pickplace_taskspace_env_cfg.py` (same scene, same task-space IK action,
same PPO runner cfg), with only `RewardsCfg` changed.

### 1. Wire in the existing `ground_penalty` function

`ground_penalty` (`tasks/ar4/mdp.py:353-369`) already exists — built for an
earlier task, never activated in any current `RewardsCfg`. Reuse it
unchanged: `-1.0` whenever the cube's world-frame z is below
`ground_height_threshold`, applied every step regardless of grasp state
(unlike `stillness_penalty`, which only fires post-grasp).

`ground_height_threshold = 0.015`: the cube (`objects_cfg.py`, `size=(0.018,
0.018, 0.018)`) rests at `z=0.009` (half its side length) when untouched.
`0.015` sits between that resting height and `_LIFT_MINIMAL_HEIGHT = 0.03`
(the existing lift threshold used throughout `mdp.py`'s waypoint/path
logic) — the cube must clear roughly half the existing lift bar before this
penalty stops firing, giving continuous pressure from the start of episode
rather than only at the moment of a full lift.

**Weight: 0.1.** This term fires almost every step until any lift happens
(unlike the running-max milestone bonuses, which pay out once per
milestone). At weight 1.0, a full 250-step episode with the cube never
leaving the ground accumulates -250 — an order of magnitude larger than
`path_proximity_bonus`'s entire per-episode ceiling (weight 25 × max raw
~1.0 = 25) or `gripper_schedule_bonus`'s per-episode ceiling (weight 0.1 ×
~250 steps ≈ 25), which would make this one term dominate the total reward
signal for every policy early in training (when the cube is on the ground
almost the entire episode, by construction) and drown out the terms that
actually carry gradient toward correct behavior. At weight 0.1, the same
worst-case episode accumulates -25 — the same order of magnitude as
`path_proximity_bonus`'s and `gripper_schedule_bonus`'s own per-episode
ceilings, a background pressure rather than a dominant one.

```python
ground_penalty = RewTerm(
    func=ar4_mdp.ground_penalty,
    weight=0.1,
    params={
        "object_cfg": SceneEntityCfg("cube"),
        "ground_height_threshold": 0.015,
    },
)
```

### 2. New `base_proximity_penalty` function (distinct from `ground_penalty`)

New function in `tasks/ar4/mdp.py`, appended after `reset_arm_to_pregrasp_pose`:

```python
def base_proximity_penalty(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    base_xy_threshold: float,
) -> torch.Tensor:
    """Penalty for the cube being horizontally close to the robot's own
    base, independent of height - unlike ground_penalty (z-height only,
    fires for any low cube position anywhere in the workspace), this
    specifically targets the cube sitting at or sliding into the base
    column, a distinct failure mode from "not yet lifted." Direct user
    request (2026-07-07): "negative reward for the cube contacting the
    base of the robot" - explicitly requested as a new function, separate
    from ground_penalty. x/y distance only (not z): a cube directly above
    the base at carry height should not be penalized by this term, only
    one sitting/sliding into the base footprint itself.
    """
    object: RigidObject = env.scene[object_cfg.name]
    robot: RigidObject = env.scene[robot_cfg.name]
    object_xy = object.data.root_pos_w[:, :2]
    robot_xy = robot.data.root_pos_w[:, :2]
    xy_dist = torch.norm(object_xy - robot_xy, dim=-1)
    too_close = xy_dist < base_xy_threshold
    return -too_close.float()
```

`base_xy_threshold = 0.08` (8cm keep-out radius around the robot's root
origin, in the world-frame xy plane). Reasoned from workspace proportion,
not measured base geometry (Task 1 should confirm against the robot's
actual base-link collision bounds if easily queryable from the USD, but
should not block on it if that turns out to be nontrivial to extract): the
cube's spawn range is `x: (-0.30, 0.30)`, `y: (-0.175, 0.175)` around the
robot's local origin (`pickplace_taskspace_env_cfg.py`'s `EventCfg`) — a
~0.6m × 0.35m = 0.21 m² rectangle. An 8cm-radius circle centered on that
same origin covers ~0.02 m², under 10% of the total spawn area, so the
penalty is active only for a genuinely small fraction of possible cube
positions, not a large swath of the reachable workspace.

**Weight: 0.1** — same reasoning as `ground_penalty`: fires every step the
condition holds, so it must stay in the same order of magnitude as the
other per-episode ceilings, not the sparse milestone-bonus scale.

```python
base_proximity_penalty = RewTerm(
    func=ar4_mdp.base_proximity_penalty,
    weight=0.1,
    params={
        "object_cfg": SceneEntityCfg("cube"),
        "robot_cfg": SceneEntityCfg("robot"),
        "base_xy_threshold": 0.08,
    },
)
```

### 3. Raise `antipodal_grasp_bonus`'s weight, with a matched `stillness_penalty` raise

Direct request: "higher reward for the cube being in the grasp position."
`antipodal_grasp_bonus`'s weight goes `3.0 → 4.0`.

Per this project's own established practice (Experiment 12's entire
purpose was fixing a verified reward-rate incentive: holding a grasp
without further progress must stay net-negative once `stillness_penalty`'s
patience window elapses), raising `antipodal_grasp_bonus` alone would
narrow that margin: at the current `stillness_penalty` weight of 5.0, a
grasped-and-stagnant state would net `4.0 - 5.0 = -1.0/step` instead of the
current `3.0 - 5.0 = -2.0/step` — still negative, so not reopening the exact
bug, but a smaller disincentive than Experiment 12 verified and shipped.
To preserve that exact, already-validated -2.0/step margin rather than
silently weakening it, `stillness_penalty`'s weight is raised in the same
proportion, `5.0 → 6.0`: `4.0 - 6.0 = -2.0/step`, identical margin to the
current baseline.

```python
antipodal_grasp_bonus = RewTerm(
    func=ar4_mdp.antipodal_grasp_bonus,
    weight=4.0,
    params={
        "force_threshold": 0.05,
        "antipodal_cos_threshold": -0.7071,
        "jaw1_contact_cfg": SceneEntityCfg("gripper_jaw1_contact"),
        "jaw2_contact_cfg": SceneEntityCfg("gripper_jaw2_contact"),
    },
)

stillness_penalty = RewTerm(
    func=ar4_mdp.stillness_penalty,
    weight=6.0,
    params={
        "object_cfg": SceneEntityCfg("cube"),
        "jaw1_contact_cfg": SceneEntityCfg("gripper_jaw1_contact"),
        "jaw2_contact_cfg": SceneEntityCfg("gripper_jaw2_contact"),
        "force_threshold": 0.05,
        "still_bound": 0.005,
        "patience_steps": 25,
    },
)
```

All other `RewardsCfg` terms (`path_proximity_bonus`, `gripper_schedule_bonus`,
`action_rate`, `joint_vel`) carry over from `pickplace_taskspace_env_cfg.py`
unchanged. `ActionsCfg`, `ObservationsCfg`, `EventCfg`, `TerminationsCfg`,
and `Ar4PickPlaceTaskspacePPORunnerCfg` are all reused/imported unchanged —
this experiment isolates the reward function alone.

## What this does NOT change

- No modification to `pickplace_taskspace_env_cfg.py`,
  `pickplace_residual_env_cfg.py`, `pickplace_reachskip_env_cfg.py`, or
  their action terms/event terms — purely additive (two new reward
  functions/terms, two weight changes, one new env cfg file).
- Does not resume or extend Experiment 13's residual mechanism or
  Experiment 14's reach-skip mechanism — both remain open, separate
  threads.
- `ground_penalty`'s existing signature/behavior is reused as-is, not
  modified.

## Verification plan

Same sequence as every experiment this session: smoke test (new env cfg
constructs, both reward functions run without exception), 300-iteration
diagnostic (`Loss/value_function` bounded — reward-only changes are lower
risk than a new action/reset mechanism, but the diagnostic gate is cheap
and this project's standing practice applies it uniformly), full
1500-iteration run + report comparing against Experiment 12's exact final
values (the same reference point Experiments 13 and 14 both used, keeping
all three comparable to one common baseline), multi-episode video
inspection (≥3 of 10 recorded episodes, personally reviewed by the
controller — per the standing lesson that one episode is not a
representative sample at this task's success rate), ROADMAP record
regardless of outcome.

## Success criteria

Same bar the last three experiments have used: does the video/scalar
evidence show the policy reaching waypoint index ≥2 (lift) more than prior
experiments, and does `cube_reached_goal` improve past Experiment 14's
final value (0.011393) without a new regression in `path_proximity_bonus`
or `gripper_schedule_bonus`. Additionally, specific to this experiment's own
new terms: does `ground_penalty`'s nonzero rate *decrease* over the course
of training (evidence the cube is spending less time on the ground as
training progresses, not just accumulating a constant penalty the policy
never learns to reduce), and does `base_proximity_penalty`'s nonzero rate
stay low/flat (evidence the term isn't firing so often that it's fighting
the cube's legitimate spawn distribution rather than only the base-collapse
failure mode it targets). A null result on lift, with these two new terms'
rates behaving as expected, would still be informative: it would indicate
the grasp/lift gap is not primarily a reward-incentive problem at all
(three separate reward-shaping attempts — Experiment 12, and now this one's
two new terms plus the grasp-weight raise — having failed to move it), and
would argue for revisiting the queued episode-length/staged-decomposition
ideas or a genuinely different mechanism next, rather than further reward
tuning.
