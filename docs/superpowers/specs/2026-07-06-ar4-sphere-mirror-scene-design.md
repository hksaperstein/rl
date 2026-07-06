# AR4 sphere mirror-goal scene + grasp-gated stillness penalty — Design

## Goal

Two changes, bundled into one experiment (the sixth on the "grasp/lift
never emerges" sub-problem, this time explicitly user-directed rather than
a unilateral reward retry):

1. **Scene simplification + task redefinition:** train on a scene with
   only the sphere (no cube/rect_prism/wedge), with its spawn position
   randomized across the full workspace, and the goal always on the
   opposite side of the robot from wherever it spawned.
2. **Grasp-gated stillness penalty:** a new reward term that goes negative
   if the object stops moving for too long *after* grasp is achieved —
   directly targeting the "reach, grip, freeze" pattern that all five
   prior experiments this session exhibited.

## Why now

The potential-shaping experiment (Task 2/3) just came back **falsified:
0/10 real eval episodes show any lift** —
`docs/superpowers/plans/2026-07-06-ar4-sphere-lift-potential-shaping-report.md`.
Worse, `Episode_Reward/staged_potential_progress` *declined* to -0.109
over training rather than growing. Root cause, found while designing this
experiment: `staged_potential_progress`'s formula `gamma * new_potential -
prev_potential` was claimed to be "always >= 0" per its docstring, but
that's only true the step a *new* milestone is reached. On every step the
agent merely *holds* its best-ever potential (`new_potential ==
prev_potential == Φ`), the reward is `Φ * (gamma - 1)` — **negative**,
since `gamma=0.98 < 1`. Over a ~225-step episode, reaching the object and
holding there (potential ≈ 0.1) costs roughly `0.1 * (-0.02) * 225 ≈
-0.45` total reward — *worse* than never approaching the object at all
(potential stays 0 the whole episode, reward stays exactly 0). The
"optimal" policy under this formula is to never reach for the sphere,
which matches exactly what the eval video showed ("no actual grasping or
lifting occurs... all markers remain on the ground"). This is a bug in
the shaping formula, not evidence against potential-shaping as an idea —
**fixed in this design** (see Part 1's reward section) by dropping the
discount from the bookkeeping delta.

Separately, the user's own observation ("is there a reason the robot
isn't even trying to rotate the joints... reward actuation... for picking
up", "set a negative reward on static positioning/staying still within a
bound") identifies a second, independent gap: even a *correctly
non-negative* plateaued-potential term still supplies no *positive*
pressure to keep moving once a milestone is reached — "hold the milestone
forever" and "keep trying" become reward-equivalent. Per the now-saved
proactive-reward-adjustment guidance, both fixes are folded into this one
experiment rather than run as two more sequential single-variable
retries.

## Part 1: Scene redesign

**New file:** `tasks/ar4/pickplace_mirror_env_cfg.py` — parallel to
`pickplace_env_cfg.py`, following the existing convention set by
`pickplace_single_object_env_cfg.py`: does **not** touch `Ar4SceneCfg`
(`env_cfg.py`) or `objects_cfg.py`, since `interactive_demo.py`,
perception scripts, and `grasp_demo.py` all depend on all four objects
existing in the shared base.

### Scene

`Ar4PickPlaceMirrorSceneCfg(InteractiveSceneCfg)` — ground, light, robot,
**sphere only** (no cube/rect_prism/wedge), plus the same `ee_frame`
FrameTransformer and `gripper_jaw1_contact`/`gripper_jaw2_contact`
ContactSensors as `Ar4PickPlaceSceneCfg` (byte-identical sensor configs,
copied not imported, since the base class differs).

The sphere is instantiated as `SPHERE_CFG.replace(init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.275, 0.009)))`
— recentering its default (env-local) position to the midpoint of
`WORKSPACE_BOUNDS` (`x: (-0.30, 0.30)`, `y: (0.10, 0.45)`) — SPHERE_CFG
itself (in `objects_cfg.py`) is untouched; `.replace(...)` returns a new
config, doesn't mutate the shared constant.

### Randomized spawn + mirrored goal

Confirmed from `robot_cfg.py`: the robot's `init_state` has no explicit
`pos`, defaulting to `(0, 0, 0)` — the robot base sits at each env's own
local origin. So "local" (env-relative) coordinates are already
robot-relative, and `WORKSPACE_BOUNDS`'s x/y are already defined in this
same local frame (confirmed against `objects_cfg.py`'s existing object
placements). Mirroring "the other side of the robot" is exactly negating
local x.

Two new `EventTerm`s (`mode="reset"`), registered in this order:

1. `reset_sphere_position` — reuses the existing, proven
   `mdp.reset_root_state_uniform` (unchanged function, just wider
   `pose_range`): `{"x": (-0.30, 0.30), "y": (-0.175, 0.175), "z": (0.0, 0.0)}`
   added to the recentered default above, giving an absolute local spawn
   range of exactly `WORKSPACE_BOUNDS`'s x/y.
2. `randomize_goal` (new function in `tasks/ar4/mdp.py`,
   `set_mirrored_goal(env, env_ids, sphere_cfg, goal_y_range,
   goal_z_range)`) — runs *after* Step 1 in the same `mode="reset"` pass
   (Isaac Lab's `EventManager` executes same-mode terms in registration
   order), reads the sphere's now-updated `root_pos_w`, computes
   `goal_local_x = -(sphere_pos_w.x - env_origin.x)`, independently
   resamples `goal_local_y` within `goal_y_range` (also
   `WORKSPACE_BOUNDS`'s y, for two degrees of freedom instead of one) and
   `goal_local_z` within `goal_z_range` (reusing the existing command's
   `pos_z=(0.0, 0.02)` range), and writes the resulting **world-frame**
   position into a new per-env stateful buffer `env._target_pos_w`
   (shape `(num_envs, 3)`, lazy-`hasattr`-initialized, same pattern as
   `env._lift_potential_max`).

This buffer replaces `CommandsCfg`/`UniformPoseCommandCfg` entirely for
this scene — the command manager samples independently at each
resampling interval and has no way to make one term's target a function
of another term's own random draw within the same reset. `CommandsCfg` is
omitted from the new env config (no `commands` field).

### Consumers updated to read `env._target_pos_w`

- **Observation:** new function `mirrored_target_position_in_robot_root_frame(env, robot_cfg, object_cfg)`
  in `tasks/ar4/mdp.py`, mirroring the existing
  `object_position_in_robot_root_frame` pattern exactly but reading
  `env._target_pos_w` via `subtract_frame_transforms` instead of
  `mdp.generated_commands`.
- **Reward:** new functions `_raw_lift_progress_mirrored` and
  `staged_milestone_bonus` — the raw staged signal
  (`_raw_lift_progress_mirrored`) is a copy of the existing
  `_raw_lift_progress` with one simplification: since `env._target_pos_w`
  is already world-frame (no command-frame transform needed), the goal
  sub-term becomes
  `goal_dist = torch.norm(object.data.root_pos_w - env._target_pos_w, dim=-1)`
  directly — no `combine_frame_transforms`, no `robot_cfg`/`command_name`
  params for this sub-term. The wrapper is **not** a copy of
  `staged_potential_progress` — per the "Why now" section's finding, that
  formula's `gamma` discount makes holding a plateaued potential cost
  negative reward every step, which made "never approach the object" the
  reward-minimizing policy. `staged_milestone_bonus` fixes this by
  dropping the discount from the bookkeeping delta entirely:

  ```python
  def staged_milestone_bonus(env, object_cfg, ee_frame_cfg, jaw1_contact_cfg,
                              jaw2_contact_cfg, reach_std, force_threshold,
                              lift_minimal_height, goal_std) -> torch.Tensor:
      if not hasattr(env, "_lift_milestone_max"):
          env._lift_milestone_max = torch.zeros(env.num_envs, device=env.device)
      raw = _raw_lift_progress_mirrored(env, object_cfg, ee_frame_cfg,
                                         jaw1_contact_cfg, jaw2_contact_cfg,
                                         reach_std, force_threshold,
                                         lift_minimal_height, goal_std)
      prev = env._lift_milestone_max.clone()
      env._lift_milestone_max = torch.maximum(env._lift_milestone_max, raw)
      return env._lift_milestone_max - prev  # undiscounted: 0 at a plateau, > 0 on a new milestone, never negative
  ```

  No `gamma` param at all (there is nothing to discount) — this also
  means no more "must match `Ar4PickPlacePPORunnerCfg.algorithm.gamma`
  exactly" constraint, since that constraint was itself part of the bug.
  Uses a new buffer name `_lift_milestone_max` (not `_lift_potential_max`)
  so a stray import of the old scene's env alongside this one can never
  silently share state. New functions throughout (not edits to the
  existing `pickplace_env_cfg.py` ones), which stay in active use
  elsewhere.
- **Termination:** new function `object_reached_mirrored_goal(env,
  threshold, object_cfg)` — same shape as `mdp.object_reached_goal` but
  compares against `env._target_pos_w` instead of `command_manager`.

### Env config

`Ar4PickPlaceMirrorEnvCfg(ManagerBasedRLEnvCfg)` — **`num_envs=4096`**
(matching `Ar4PickPlaceEnvCfg`, not the smaller single-object precedent's
`num_envs=16` default — this is a real training run, not an interactive
demo), same `episode_length_s=5.0`, `decimation=2`, `sim.dt=0.01`, PPO
hyperparameters unchanged (reuses `Ar4PickPlacePPORunnerCfg` as-is;
`staged_milestone_bonus` has no `gamma` param, so there is no discount to
keep in sync this time).

## Part 2: Grasp-gated stillness penalty

New reward function in `tasks/ar4/mdp.py`:

```python
def stillness_penalty(
    env, object_cfg, jaw1_contact_cfg, jaw2_contact_cfg,
    force_threshold, still_bound, patience_steps,
) -> torch.Tensor:
```

Tracks a per-env reference position (`env._still_ref_pos`) and a
stagnant-step counter (`env._still_steps`), both lazy-`hasattr`-init and
explicitly zeroed by a new `reset_stillness_buffers(env, env_ids,
object_cfg)` EventTerm (`mode="reset"`, registered after
`randomize_goal` so the reference reflects the new episode's spawn, not
the prior episode's end state):

- Each step: if the object has moved more than `still_bound` (5mm) since
  the reference was last updated, update the reference to the current
  position and reset the counter to 0; otherwise increment the counter.
- Returns `-1.0` for envs where `contact_grasp_bonus(...) > 0.5` (grasp
  achieved — reusing the existing tested function, not re-deriving the
  jaw-force check) **and** the stagnant counter exceeds `patience_steps`;
  `0.0` otherwise. Pre-grasp stillness (normal reach behavior) is never
  penalized.
- `still_bound = 0.005` (m), `patience_steps = 25` (0.5s at the
  0.02s control step — `decimation=2` × `sim.dt=0.01`), `weight = -2.0`
  in `RewardsCfg` (comparable in magnitude to the reach/grasp stage
  weights inside `staged_milestone_bonus`, so it's not negligible next
  to the -1e-4 `action_rate`/`joint_vel` terms, but far smaller than the
  25.0-weighted main term).

### `RewardsCfg` for the new scene

```python
staged_milestone_bonus = RewTerm(func=ar4_mdp.staged_milestone_bonus, weight=25.0, params={
    "object_cfg": SceneEntityCfg("sphere"),
    "ee_frame_cfg": SceneEntityCfg("ee_frame"),
    "jaw1_contact_cfg": SceneEntityCfg("gripper_jaw1_contact"),
    "jaw2_contact_cfg": SceneEntityCfg("gripper_jaw2_contact"),
    "reach_std": 0.1,
    "force_threshold": 0.05,
    "lift_minimal_height": 0.03,
    "goal_std": 0.3,
})
stillness_penalty = RewTerm(func=ar4_mdp.stillness_penalty, weight=-2.0, params={
    "object_cfg": SceneEntityCfg("sphere"),
    "jaw1_contact_cfg": SceneEntityCfg("gripper_jaw1_contact"),
    "jaw2_contact_cfg": SceneEntityCfg("gripper_jaw2_contact"),
    "force_threshold": 0.05,
    "still_bound": 0.005,
    "patience_steps": 25,
})
action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")})
```

## Verification plan

1. Smoke test (`--num_envs 16 --max_iterations 2`): exits 0, dumped
   `env.yaml` shows exactly 4 reward terms, no exceptions on
   `_target_pos_w`/`_still_ref_pos`/`_still_steps` regardless of
   reward-fn/reset-event ordering.
2. Separately verify the mirroring logic in isolation before the full
   run: a short script (or smoke-test-adjacent check) that reads
   `env._target_pos_w` and the sphere's local position right after a
   reset and confirms `target_local_x` and `sphere_local_x` have
   opposite signs (within floating point) for every env — this is new,
   untested logic and the single easiest place for a sign/frame bug to
   hide.
3. Full 1500-iteration run at `num_envs=4096` (per user's explicit
   instruction), then the same real-eval-video decision gate as every
   prior experiment (10 episodes, extract frames, visually inspect,
   check adjacent frames before concluding "no lift" if the sphere looks
   occluded).
4. Record outcome in `ROADMAP.md` regardless of result.

## Global constraints for the plan

- Do not modify `Ar4SceneCfg` (`env_cfg.py`), `objects_cfg.py`,
  `pickplace_env_cfg.py`, or any of that file's existing `mdp.py`
  functions (`_raw_lift_progress`, `staged_potential_progress`,
  `contact_grasp_bonus`, `reset_lift_potential`) — this is a new, parallel
  scene/task, not a change to the existing one.
- `_EE_OFFSET = (0.0, 0.0, 0.036)` and all `ContactSensorCfg` fields are
  copied verbatim from `pickplace_env_cfg.py`.
- `num_envs=4096` for the real training run (user-specified explicitly,
  overriding the smaller single-object precedent's default).
- PPO hyperparameters (`Ar4PickPlacePPORunnerCfg`) unchanged, reused
  as-is. `staged_milestone_bonus` takes no `gamma` param — do not add one
  or reintroduce a discounted delta; that discount is the exact bug this
  design fixes (see "Why now").
