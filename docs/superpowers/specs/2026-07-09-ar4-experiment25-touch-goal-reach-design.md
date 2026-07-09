# Experiment 25: touch-cube-top-then-reach-goal, grasp removed entirely

## Context

Six consecutive experiments (17-22) each targeted a different specific
mechanism for the same underlying problem — the gripper's two jaws are
not mechanically coupled (the source URDF's `mimic` constraint on
`gripper_jaw2_joint` is confirmed unenforced by Isaac Sim's USD import),
and every attempted fix made jaw synchronization measurably *worse*, not
better: Experiment 19's `PhysxMimicJointAPI` fix (two configurations, both
regressions, reverted to the pre-fix baseline — commit `255b9b2`) and
Experiment 22's software jaw-mirroring fix (a new "reactive lag" failure
mode, not resolved). `lifting_object` stayed at exactly `0/1500` across
Experiments 17, 18, 20, 21, and 22.

Today (2026-07-09), enabling contact sensors on the untracked
`scripts/interactive_joint_demo.py` classical-IK demo for the first time
reproduced the same signature independently: exactly 0.0N bilateral
contact force across every grasp attempt, not a partial miss. Separately,
`pickplace_mirror_env_cfg.py`'s own reward (`staged_milestone_bonus` →
`_raw_lift_progress_mirrored`, `tasks/ar4/mdp.py:254-287`) turned out to
combine reach/grasp/lift/goal as a plain, ungated weighted sum — the
exact reward shape Experiment 16 already found exploitable via wrist-
wedging rather than genuine grasp, without Experiment 17's grasp-gating
fix (which lives only in the separate `pickplace_graspgated_env_cfg.py`
lineage).

Given this — a still-unresolved, twice-attempted-and-failed mechanical
defect, and a training target whose reward shape reintroduces an
already-diagnosed exploit — training `pickplace_mirror_env_cfg.py` from
scratch as-is would very likely reproduce one of two already-expensive,
already-documented null results rather than produce new information.

Direct user decision (2026-07-09): drop grasp/lift entirely for this next
task. Reduce the problem to what this project's entire history shows
reliably works — reaching — expressed as two sequential waypoints: the
end effector touches the cube's top, then reaches a goal point. No
grasping, no lift, no object transport.

## Hypothesis

Removing the grasp/lift/mimic-joint-dependent mechanism entirely and
reducing the task to two-stage sequential end-effector reaching will
reliably converge, because reaching is the one sub-behavior that has
converged (~0.92-0.95 on `reaching_object`/`reaching_sphere`-style terms)
in essentially every experiment this project has run regardless of
reward design or action space (Experiments 1 through 24, sphere and cube
eras alike) — the failure has never once been in reaching, only in
everything conditioned on gripper/contact dynamics downstream of it.

## Grounding

- **This project's own prior verified evidence** (per `CLAUDE.md`'s
  explicit allowance for prior-verified-evidence grounding, not only
  fresh literature): every numbered experiment's `reaching_*` reward term
  has converged reliably and early, independent of which downstream
  mechanism (grasp bonus, antipodal gate, curriculum, residual RL,
  orientation lock) was layered on top of it. This is the single most
  consistent quantitative fact in the entire ROADMAP history.
- **Xu et al. 2026** ("Stage-Transition Dense Reward Modeling,"
  arXiv:2606.31377), already read directly from source and cited in this
  project's Experiments 17 and 18 for staged/stage-transition dense
  reward design — directly applicable methodology grounding for the
  stage-gate mechanism below (goal reward inert until the touch stage is
  achieved), reused rather than re-researched, since it is the same
  mechanism class already verified in this codebase.

## Design

### Scene & action space

New file `tasks/ar4/pickplace_touchgoal_env_cfg.py`. Scene: robot + a
single cube at a **fixed** world position `(0.20, 0.28, 0.006)` — reusing
`env_cfg.py`'s existing `CUBE_CFG` default pose unchanged, already
covered by today's physics-fidelity pass (dt/decimation, collision
offsets). No `rect_prism`/`sphere`/`wedge`, no gripper contact sensors
(no grasp signal needed). Keep the `ee_frame` `FrameTransformerCfg`
(`_EE_OFFSET`-corrected gripper-tip point, re-verified both numerically
and visually earlier today) as the single position reference for both
touch and goal distances.

**Action space is arm-only**: `JointPositionActionCfg` over the 6 arm
joints (`ARM_JOINT_NAMES`), no gripper action term. The gripper joints
remain physically present on the articulation but unactuated — nothing
commands them after the initial reset pose, so they simply stay wherever
they're initialized (open).

Fixed goal point: world `(-0.20, 0.28, 0.15)` — mirrored across the cube
in X, elevated clear of the ground plane so the arm doesn't need to
descend to reach it.

### Reward: two-stage gated running-max milestone

Reuses the exact running-max delta mechanism already proven and debugged
in `staged_milestone_bonus` (`tasks/ar4/mdp.py:290-323` — the
`env._lift_milestone_max`-style monotonic bonus that fixed
`staged_potential_progress`'s gamma-discount bug: reward is `max(new,
prev_max) - prev_max`, never negative, never rewards regressing away
from a best-ever point), reduced to two terms with an explicit stage
gate this time (the current `_raw_lift_progress_mirrored` is *not*
gated — its four terms are just additively summed, which is exactly what
let Experiment 16's wedging exploit satisfy `lifting_object`/goal reward
without genuine grasp).

New function `_raw_touch_goal_progress(env, object_cfg, ee_frame_cfg,
goal_pos_w, touch_std=0.05, touch_tolerance=0.02, goal_std=0.1)` (concrete
defaults: `touch_std` tighter than `_raw_lift_progress_mirrored`'s
`reach_std=0.1` since touching specifically needs closer proximity than
general reaching; `touch_tolerance=0.02` matches this project's existing
2cm goal-tolerance convention (`cube_reached_goal`'s own threshold);
`goal_std=0.1` tighter than that function's `goal_std=0.3` since this
task's goal is a fixed point, not an object subject to physics noise):

```
ee_pos = ee_frame.data.target_pos_w[:, 0, :]
touch_point = object.data.root_pos_w + [0, 0, cube_half_size]   # cube-top point
touch_dist = norm(ee_pos - touch_point)
touch_term = 1.0 - tanh(touch_dist / touch_std)

if not hasattr(env, "_touched_cube"):
    env._touched_cube = zeros(num_envs, dtype=bool)
env._touched_cube |= (touch_dist < touch_tolerance)

goal_dist = norm(ee_pos - goal_pos_w)
goal_term_raw = 1.0 - tanh(goal_dist / goal_std)
goal_term = where(env._touched_cube, goal_term_raw, 0.0)   # stage gate

return 0.3 * touch_term + 0.7 * goal_term
```

`touch_goal_milestone_bonus` wraps this exactly as
`staged_milestone_bonus` wraps `_raw_lift_progress_mirrored` (running-max
delta, same `env._touch_goal_milestone_max` buffer pattern, reset each
episode). Reusing the 0.3/0.7 weight split (touch is the easier,
earlier-achieved sub-goal; goal is the harder, gated final stage) mirrors
the increasing-weight-per-stage convention already used in
`_raw_lift_progress_mirrored`'s own 0.1/0.2/0.3/0.4 split.

`env._touched_cube` is a genuine one-way latch (`|=`, never reset except
at episode start) — once touched, the touch requirement stays satisfied
for the rest of the episode even if the EE moves away again, so the
policy isn't punished for leaving the cube's vicinity once it has
legitimately triggered the gate.

### Termination

- `time_out`: standard, `episode_length_s = 5.0` (matching
  `pickplace_mirror_env_cfg.py`'s convention).
- `goal_reached`: new `DoneTerm` — EE within a tight tolerance (e.g.
  0.02m, matching this project's existing goal-tolerance convention) of
  the goal point **and** `env._touched_cube` true for that env. Checking
  both explicitly (not relying on the reward gate alone) makes success
  unambiguous regardless of any future reward-weight retuning.

### Events

1. `reset_all` (`mdp.reset_scene_to_default`, unchanged pattern).
2. `reset_touch_goal_milestone` — new, zeroes
   `env._touch_goal_milestone_max` and `env._touched_cube` for resetting
   envs (mirrors `reset_lift_milestone`'s existing pattern).

No cube-position or goal randomization this pass (direct user decision —
fixed positions for a first pass, to isolate the new reward mechanism as
the only variable; randomization/mirroring can be reintroduced later
once this converges, reusing `pickplace_mirror_env_cfg.py`'s already-built
mechanism).

### Observations

Same shape as `pickplace_mirror_env_cfg.py`'s `ObservationsCfg` minus
anything gripper-specific: `joint_pos_rel`, `joint_vel_rel` (arm joints
only now, since there's no gripper action), `cube_position` (robot-root-
frame, unchanged — still useful for locating the touch target),
`last_action` (6-dim now, not 8). No `target_object_position` term needed
in the same form — replace with a fixed goal-position-in-robot-root-frame
observation (computed once, constant per env, robot-root-frame per the
existing `object_position_in_robot_root_frame` convention).

### Training

Standard `rsl_rl` PPO wiring, reusing
`Ar4PickPlaceMirrorPPORunnerCfg`'s hyperparameters unchanged (no reason
to retune an already-proven PPO config for a *simpler* task than what it
was tuned for). Full 1500-iteration run, video review before verdict, per
this project's Tier-1 standard — this is a genuinely new reward
mechanism (stage-gated two-term milestone, arm-only action space), not a
parameter tweak, so the full process applies even though the task itself
is simpler than what came before it.

## Success criteria

`Episode_Termination/goal_reached` converging meaningfully above 0 (this
project's established standard: the real termination-rate metric, not a
shaped reward scalar) constitutes success. `touch_progress`/`goal_progress`
components of the reward should both show healthy, non-degenerate
convergence in TensorBoard as a secondary check, but the termination rate
is the decision criterion per this project's own established discipline
(shaped rewards climbing while real behavior doesn't is exactly the
recurring pattern that motivated the Tier-1 verification standard in the
first place).
