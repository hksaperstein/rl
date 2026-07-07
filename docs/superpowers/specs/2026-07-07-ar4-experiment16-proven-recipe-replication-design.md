# Experiment 16: replicate a proven working RL-manipulation recipe from scratch

## Hypothesis

**This repo's entire reward-design lineage (Experiments 1-15) has always
included a standalone, separately-rewarded grasp-quality term
(`contact_grasp_bonus` → `antipodal_grasp_bonus`) and an ungated
milestone/waypoint-proximity bonus (`path_proximity_bonus`) that credits
progress on pre-lift waypoints — a structural difference from proven,
independently-published RL manipulation recipes that never reward grasp
directly and gate the majority of available reward behind an actual lift
condition. This structural difference, not episode length, not action
space per se, and not any of the specific weight values tuned across 15
experiments, is the most likely root cause of this repo's persistent
"grasp achieved, lift never emerges" pattern.** Replacing this repo's
reward structure with a faithful replication of the proven gating
pattern — built from scratch as a new env cfg, not layered incrementally
on the current baseproximity design — should produce genuine lift and
carry where 15 prior iterations did not.

This is falsifiable: if the replicated recipe, run at this repo's own
established diagnostic-then-full-run scale, still shows the same
reach-and-freeze/no-lift pattern in video, the hypothesis is wrong and the
bottleneck is elsewhere (episode length, AR4-specific kinematics, PPO
hyperparameters — though the last is already the least likely candidate,
see Background research below).

## Background research

Two independent, official, widely-used Isaac-ecosystem manipulation RL
tasks were read directly from source (not summarized secondhand) and
found to structurally agree on the same design choice this repo has never
used:

**1. Isaac Lab's own Franka Cube Lift task**
(`isaaclab_tasks/manager_based/manipulation/lift/lift_env_cfg.py`,
`mdp/rewards.py`, `config/franka/joint_pos_env_cfg.py` — read directly
from the installed Isaac Lab source at `/home/saps/IsaacLab`, the same
installation this repo already runs against). Six reward terms:
`reaching_object` (dense tanh-kernel EE-to-object distance, weight 1.0),
`lifting_object` (**plain binary** per-step reward for object height
above a threshold, weight 15.0 — not a milestone/running-max bonus),
`object_goal_tracking` and `object_goal_tracking_fine_grained` (dense
tanh-kernel object-to-goal distance at two different `std` scales,
**multiplicatively gated on the lift condition** — `(height > threshold)
* (1 - tanh(dist/std))` — weights 16.0 and 5.0), plus the same
`action_rate`/`joint_vel` regularizers (-1e-4) this repo already uses.
**No grasp-quality reward term of any kind** — no contact sensing, no
force-closure check. A `CurriculumCfg` bumps `action_rate`/`joint_vel`
from -1e-4 to -1e-1 after 10,000 steps. Action space: plain
`JointPositionActionCfg` (`scale=0.5`), not task-space/IK. Episode
length 5.0s, decimation 2, `sim.dt=0.01` — identical to this repo's
current settings. PPO config
(`config/franka/agents/rsl_rl_ppo_cfg.py`): `num_steps_per_env=24`,
`max_iterations=1500`, `save_interval=50`, identical network
(`[256,128,64]`, ELU) and algorithm hyperparameters
(`clip_param=0.2`, `entropy_coef=0.006`, `num_learning_epochs=5`,
`num_mini_batches=4`, `learning_rate=1e-4` adaptive, `gamma=0.98`,
`lam=0.95`) — **this repo's own `Ar4PickPlacePPORunnerCfg` was already
copied verbatim from this exact file** (confirmed by direct comparison),
so PPO/network/episode-length are already well-aligned; the divergence is
squarely in the reward function and (to a lesser extent) the action space.

**2. IsaacGymEnvs' FrankaCubeStack task**
(`isaacgymenvs/tasks/franka_cube_stack.py`,
github.com/isaac-sim/IsaacGymEnvs, `compute_franka_reward`). Independently
corroborates the same gating pattern: a dense hand-to-cube distance term,
a binary lift indicator, an **alignment reward multiplied by the lift
indicator** (`(1 - tanh(10*d_ab)) * cubeA_lifted`), and a binary stack-
success term — composed with `torch.where(stack_success, stack_reward,
dist+lift+align)` rather than a flat additive sum of independent terms.
Same conclusion as (1) from a structurally different, independently-
maintained proven codebase: **the downstream/goal-tracking reward is
unavailable until the object is actually lifted, and grasp itself is never
directly rewarded.**

**3. Corroborating academic context** (arXiv:2509.13239, "Collaborative
Loco-Manipulation for Pick-and-Place Tasks with Dynamic Reward
Curriculum"): explicitly documents that lift-then-place is a known-hard
transition for a single RL policy and that *re-emphasizing* the lifting
reward via a curriculum is what let their system reliably do both —
independent confirmation that this repo's own struggle with the
lift-to-carry-to-place transition is a recognized, published difficulty,
not evidence that this repo's specific approach is uniquely broken.

**Conclusion drawn from triangulating three independent sources**: the
proven pattern is (a) reward reaching with a small dense term, (b) reward
lift with a plain per-step binary term at a substantial weight, (c) gate
essentially all downstream/goal-tracking reward behind the lift condition
so most of the available reward is structurally unreachable without
lifting, and (d) never reward grasp quality as its own term — let it
remain purely instrumental. This repo has never tried (c) or (d)
together; every experiment so far has kept a standalone grasp reward and
an ungated (or only weakly-staged, via running-max waypoint index)
progression signal.

## Design: replicate the recipe on the AR4+cube scene

New file `tasks/ar4/pickplace_provenrecipe_env_cfg.py`
(`Ar4PickPlaceProvenRecipeEnvCfg`) — built from scratch, not layered on
`pickplace_baseproximity_env_cfg.py`. Reuses `Ar4PickPlaceMirrorSceneCfg`
(same scene: AR4 + single cube, same contact sensors/ee_frame, unchanged)
and `set_mirrored_goal`/`object_reached_mirrored_goal`/
`mirrored_target_position_in_robot_root_frame` (this repo's existing
mirrored-goal mechanism — the reference tasks' `CommandsCfg`/
`UniformPoseCommandCfg` can't express a goal that's a function of the
object's own random spawn, the same reason this repo built the custom
buffer in the first place; keeping it is not a deviation from the
recipe's *reward* structure, only from an unrelated goal-sampling
mechanism this repo already had a documented reason to replace).

**Action space — also reverts to plain joint-space, matching both
references exactly**, rather than this repo's task-space/IK-driven
action (Experiments 11-15): `isaaclab_mdp.JointPositionActionCfg(
asset_name="robot", joint_names=ARM_JOINT_NAMES, scale=0.5)` — this exact
scale value is not invented for this experiment; it's already used by
`pickplace_mirror_env_cfg.py`'s own `ActionsCfg`
(`tasks/ar4/pickplace_mirror_env_cfg.py:55-75`), which already cites the
same Franka lift-task precedent for that value. Isolating the reward
structure as the primary variable would argue for keeping the current
task-space action instead — but the point of "start from scratch, replicate
an actual working example" is to test the *whole* proven recipe with
minimal invented deviation, not to cherry-pick only the reward piece and
leave in this repo's own not-yet-validated task-space innovation. If this
experiment succeeds, a natural follow-up (not part of this experiment) is
re-testing the gated reward structure against the task-space action to
see if the gating alone (independent of action space) is what mattered.

**Reward terms** (`RewardsCfg`, 6 terms total, matching the reference's
count and structure):

```python
reaching_object = RewTerm(
    func=mdp.object_ee_distance,
    weight=1.0,
    params={"std": 0.1, "object_cfg": SceneEntityCfg("cube"), "ee_frame_cfg": SceneEntityCfg("ee_frame")},
)
```
Reused directly from `isaaclab_tasks.manager_based.manipulation.lift.mdp`
(already imported in every env cfg file in this repo as `mdp`) —
fully generic, parameterized by `object_cfg`/`ee_frame_cfg`, no
Franka-specific assumptions. Not reimplemented.

```python
lifting_object = RewTerm(
    func=mdp.object_is_lifted,
    weight=15.0,
    params={"minimal_height": 0.03, "object_cfg": SceneEntityCfg("cube")},
)
```
Also reused directly from the same module. `minimal_height=0.03` (not the
reference's 0.04) — matches this repo's own already-established
`_LIFT_MINIMAL_HEIGHT` constant used throughout `mdp.py`'s waypoint logic,
appropriate since this repo's cube (0.018m) is smaller than the
reference's DexCube (scaled 0.8x from its own base size) — a deliberate,
documented adaptation to this repo's own object scale, not a copy error.

```python
object_goal_tracking = RewTerm(
    func=ar4_mdp.mirrored_goal_distance_gated,
    weight=16.0,
    params={"std": 0.3, "minimal_height": 0.03, "object_cfg": SceneEntityCfg("cube")},
)
object_goal_tracking_fine_grained = RewTerm(
    func=ar4_mdp.mirrored_goal_distance_gated,
    weight=5.0,
    params={"std": 0.05, "minimal_height": 0.03, "object_cfg": SceneEntityCfg("cube")},
)
```
New function, `mirrored_goal_distance_gated`, appended to
`tasks/ar4/mdp.py` — a direct adaptation of
`isaaclab_tasks...lift.mdp.object_goal_distance`'s exact formula
(`(height > minimal_height) * (1 - tanh(distance / std))`), swapping only
the goal source (this repo's `env._target_pos_w` buffer, already
world-frame, instead of the command manager — the one unavoidable
adaptation, for the reason stated above):

```python
def mirrored_goal_distance_gated(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Direct adaptation of isaaclab_tasks.manager_based.manipulation.lift.mdp.object_goal_distance's
    exact tanh-kernel-distance-gated-on-lift formula to this repo's
    mirrored-goal buffer (env._target_pos_w, already world-frame, set by
    set_mirrored_goal) instead of the command manager - see
    docs/superpowers/specs/2026-07-07-ar4-experiment16-proven-recipe-replication-design.md
    for why the command manager can't be used here (this repo's goal is a
    function of the object's own random spawn) and why this is otherwise
    an unmodified replication of the reference formula, not a new design.
    """
    object: RigidObject = env.scene[object_cfg.name]
    if not hasattr(env, "_target_pos_w"):
        env._target_pos_w = torch.zeros(env.num_envs, 3, device=env.device)
    distance = torch.norm(env._target_pos_w - object.data.root_pos_w, dim=-1)
    lifted = (object.data.root_pos_w[:, 2] > minimal_height).float()
    return lifted * (1.0 - torch.tanh(distance / std))
```

```python
action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")})
```
Unchanged — both this repo and both references already agree on these
values.

**Deliberately NOT included, matching the reference exactly:**
`antipodal_grasp_bonus`, `stillness_penalty`, `ground_penalty`,
`base_proximity_penalty`, `gripper_schedule_bonus`,
`path_proximity_bonus` — none of these exist in either proven reference.
Grasp becomes purely instrumental: the policy gets zero direct reward for
grasping, only for the reach/lift/goal-tracking consequences of having
grasped. Gripper open/close timing is likewise unscored — purely an
action the policy must learn to use effectively, not guided by a
schedule-matching bonus.

**Curriculum** (`CurriculumCfg`, new to this repo — no prior experiment
has used Isaac Lab's curriculum manager): replicated from the reference
exactly, using the framework-provided `mdp.modify_reward_weight` (from
`isaaclab_tasks...lift.mdp`, not new code):

```python
action_rate_curr = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000})
joint_vel_curr = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000})
```

**Events**: `reset_all`, `reset_cube_position` (same pose_range as every
prior experiment), `set_mirrored_goal` (reused). **`compute_path_waypoints`
is dropped entirely** — there is no waypoint/milestone system in this
design, matching the reference's much simpler structure.

**Observations, terminations, PPO runner cfg**: unchanged from this
repo's current baseline (`ObservationsCfg` identical structure; `time_out`
+ `cube_reached_goal` terminations, this repo's own success definition,
kept since the reference's tabletop-drop termination doesn't apply to
this repo's ground-level scene geometry; `Ar4PickPlacePPORunnerCfg`, not
`Ar4PickPlaceTaskspacePPORunnerCfg`, since this experiment uses plain
joint-space action and doesn't need the task-space-specific
`clip_actions=5.0` override).

## What this does NOT change

No modification to any existing `pickplace_*.py` file or any existing
function in `tasks/ar4/mdp.py` — purely additive (one new function, one
new env cfg file). Does not touch the classical demo, perception
pipeline, or any prior experiment's artifacts.

## Verification plan

Same sequence as every Tier-1 experiment this session: smoke test, 300-
iteration diagnostic (`Loss/value_function` bounded — a new action space
+ curriculum manager combination not yet exercised together in this
repo), full 1500-iteration run + report comparing `cube_reached_goal`
against Experiment 15's final value (0.017202, the best of the session so
far) and Experiment 12's (0.010773, the original task-space baseline),
multi-episode video inspection (≥3 of 10 episodes, personally reviewed by
the controller), ROADMAP record regardless of outcome.

## Success criteria

Primary: does eval video show genuine sustained lift (waypoint-equivalent
progress this repo has never reliably observed) and, ideally,
carry-toward-goal, in a meaningfully larger fraction of sampled episodes
than every prior experiment's ~0/3. Secondary, scalar: does
`lifting_object`'s own nonzero rate (this experiment's direct per-step
lift indicator, a much more literal signal than any prior experiment's
proxy terms) show real growth over training, and does
`object_goal_tracking`'s nonzero rate track it (confirming the gate is
actually unlocking, not staying at zero because lift never happens). A
null result — still no lift despite faithfully replicating two proven
recipes — would be highly informative: it would specifically implicate
something about the AR4 arm's own kinematics/gripper geometry, this
repo's specific cube/workspace scale, or a scene-level difference not
captured by the reward/action replication, rather than reward-shaping
philosophy in general, since reward-shaping philosophy is exactly what
this experiment isolates and tests directly against two independently-
proven examples.
