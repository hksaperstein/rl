# Hyperparameter registry

## What this is

A living reference for every hyperparameter this project actively tunes
or has tuned — current value, where it's set, why it's set that way, and
what changed it last. Unlike the other concept articles (each a
synthesized narrative across experiments), this one is a table-first
reference meant to be edited in place as values change, not rewritten as
prose each time. Update it in the same pass as the ROADMAP.md entry that
changes a value (per this project's own "update kb continuously"
convention) — add a row or edit a cell, don't wait for a batch pass.

Four categories, since they live in different files and change for
different reasons: simulation/physics, PPO/training, actuators, and
task/reward. The last category changes the most often (new task = new
weights); the first three are largely shared infrastructure and change
rarely, on their own explicit justification.

## Simulation / physics

Set per env cfg's `__post_init__` (`tasks/ar4/*_env_cfg.py`). Most
`pickplace_*_env_cfg.py` variants still use the older, coarser values
below (`sim.dt=0.01`) — only `env_cfg.py`, `grasp_verify_env_cfg.py`,
`pickplace_mirror_env_cfg.py`, and `pickplace_touchgoal_env_cfg.py` have
the 2026-07-09 physics-fidelity pass applied. See
[[sim-physics-fidelity]] for the full methodology behind these specific
values.

| Parameter | Current value | Where | Why |
|---|---|---|---|
| `sim.dt` | `1/240` (`env_cfg.py`, `grasp_verify_env_cfg.py`); `0.005` (`pickplace_mirror_env_cfg.py`, `pickplace_touchgoal_env_cfg.py`); `0.01` (all other `pickplace_*_env_cfg.py` variants, not yet updated) | per-file `__post_init__` | Halved 2026-07-09 for finer PhysX substep resolution grasping/manipulating a 12mm object; control period held constant via a matching `decimation` change so no RL policy's MDP interface changed. |
| `decimation` | `4` (updated files above); `2` (older variants) | per-file `__post_init__` | Doubled in lockstep with the `dt` halving above — control period (`decimation × dt`) is unchanged: 1/60s for `env_cfg.py`/`grasp_verify_env_cfg.py`, 0.02s (50Hz) for the mirror/touchgoal lineage. |
| Cube `contact_offset`/`rest_offset` | PhysX auto-compute (`-inf` sentinel in the USD schema) | `tasks/ar4/objects_cfg.py`'s `_COLLISION_PROPS` | Empirically bounded under 0.5mm via a 2400Hz free-fall drop test 2026-07-09 (see [[sim-physics-fidelity]]) — negligible relative to the cube's 6mm half-extent, no explicit override needed. |
| Cube solver iterations | `solver_position_iteration_count=16`, `solver_velocity_iteration_count=1` | `pickplace_mirror_env_cfg.py`'s cube `.replace()` | Experiment 10: matched to Isaac Lab's own Franka cube-lift recipe (well above PhysX defaults) for more stable contact resolution during grasping. |
| Robot solver iterations | `solver_position_iteration_count=8`, `solver_velocity_iteration_count=0` | `robot_cfg.py`'s `AR4_MK5_CFG.articulation_props` | Set at initial asset authoring; not independently re-derived since. |
| `physics_material` (scene-wide) | `static_friction=1.0, dynamic_friction=1.0` | per-env-cfg `__post_init__` | Default friction was too low for the small objects to grip reliably (predates this pass; carried forward unchanged). |
| `_EE_OFFSET` (gripper pinch-point offset from `link_6`) | `(0.0, 0.0, 0.036)` | `pickplace_env_cfg.py`, imported everywhere else | Corrected from a wrong `0.09` (5.4cm off) during the Experiment 21/22 investigation; re-verified 2026-07-09 both numerically and visually (`debug_vis`). This is the single most load-bearing geometric constant in the whole repo — every reach/grasp/touch reward depends on it. |

## PPO / training (`tasks/ar4/agents/rsl_rl_ppo_cfg.py`)

`Ar4PickPlacePPORunnerCfg` is the default for essentially every task
(including Experiment 25/`--touchgoal`); `Ar4PickPlaceTaskspacePPORunnerCfg`
(`pickplace_taskspace_env_cfg.py`) is a separate cfg used only by the
taskspace/residual/reachskip/baseproximity/warmresidual lineage (differs
by `clip_actions=5.0`, everything else identical — verified directly
during Experiment 25's final review via the trained run's own
`params/agent.yaml`). Adapted from Isaac Lab's own Franka cube-lift
example (`isaaclab_tasks/manager_based/manipulation/lift/config/franka/agents/rsl_rl_ppo_cfg.py`).

| Parameter | Current value | Notes |
|---|---|---|
| `num_steps_per_env` | `24` | Rollout length collected per env between PPO updates — independent of `episode_length_s`; see the 2026-07-09 session's own clarification that longer episodes don't make training iterations slower, they just span more iterations per episode. |
| `max_iterations` | `1500` | This project's standard full-run length; overridable via `--max_iterations` for quick smoke/diagnostic runs. |
| `learning_rate` | `1.0e-4` | [[experiment-04-sa-ppo-lr-bump]] tried a scheduled bump at a literature-flagged critical point; no measurable improvement, not kept. |
| `schedule` | `"adaptive"` | KL-adaptive LR schedule (`desired_kl=0.01`), from the Franka recipe. |
| `clip_param` | `0.2` | Franka recipe default, unchanged. |
| `entropy_coef` | `0.006` | Franka recipe default, unchanged. |
| `gamma` / `lam` | `0.98` / `0.95` | Franka recipe default. **Caution**: [[reward-rate-arithmetic]] documents a real bug class where `gamma < 1` interacting with a reward function's own formula (not this file) made holding position actively rewarding — check any new potential-based reward term against this before assuming `gamma` itself is the lever to adjust. |
| `num_learning_epochs` / `num_mini_batches` | `5` / `4` | Franka recipe default. |
| `clip_actions` | `null` (default cfg) / `5.0` (taskspace cfg) | See note above — confirm which runner cfg a task actually uses via its own `params/agent.yaml` at runtime, not just which one the code *should* select (this exact check caught nothing wrong in Experiment 25, but the mechanism for verifying it is worth reusing). |
| Actor/critic MLP | `[256, 128, 64]`, ELU | Franka recipe default; unchanged across every experiment to date. |

## Actuators (`tasks/ar4/robot_cfg.py`)

| Joint group | `stiffness` | `damping` | `armature` | `effort_limit_sim` | `velocity_limit_sim` |
|---|---|---|---|---|---|
| Arm (`joint_1`-`joint_6`) | `40.0` | `4.0` | `1e-3` | `20.0` | `3.0` |
| Gripper (`gripper_jaw1/2_joint`) | `1000.0` | `50.0` | `1e-3` | `20.0` | `1.0` |

The arm's default gains let it sag noticeably under gravity at a held
target — `scripts/square_path_demo.py`/`scripts/interactive_joint_demo.py`
raise `ARM_STIFFNESS`/`ARM_DAMPING` to `2500.0`/`45.0` for their own
scripted-demo env instances only (not this shared config), specifically
so joints hold where commanded during closed-loop IK diagnostics. This
has never been changed in `robot_cfg.py` itself since a change there
would affect every RL task's actuator dynamics, not just demo scripts —
flag this explicitly if a future experiment ever considers raising it
project-wide.

## Task / reward (changes per task — most active category)

This section tracks only the **currently active** task lineage
(Experiment 25, touch-goal) in full; older tasks' reward hyperparameters
are in their own ROADMAP/kb entries, not duplicated here. Add a new
subsection here when a new task becomes the active one, rather than
maintaining every historical task's full parameter set in this file.

### Experiment 25 (`tasks/ar4/pickplace_touchgoal_env_cfg.py`, `tasks/ar4/touch_goal_reward.py`)

| Parameter | Current value | Why |
|---|---|---|
| `episode_length_s` | `20.0` | Was `5.0` (copied from `pickplace_mirror_env_cfg.py`'s single-object lift task). Run 1 (5.0s) showed episodes always hitting the timeout with `goal_reached` peaking then declining; re-derived from Isaac Lab's own reference tasks, which scale episode length with task *structure*: Reach 12.0s, Lift 5.0s, Cabinet 8.0s, Stack (sequential multi-stage, the closest analog) 30.0s. Run 2 (20.0s) converged and held `goal_reached` at ~0.60 — see ROADMAP.md's Experiment 25 entry ("Training run 1"/"Training run 2") for the full comparison. |
| `touch_std` | `0.05` | Tanh-kernel width for the pre-touch proximity shaping term — tighter than general `reach_std=0.1` (used elsewhere) since touching specifically needs closer proximity than general reaching. |
| `touch_tolerance` | `0.02` | Distance at which the touch latch (`env._touched_cube`) flips true. Matches this project's standard 2cm goal-tolerance convention (`cube_reached_goal`'s own threshold). |
| `GOAL_TOLERANCE` | `0.02` | Termination distance-to-goal threshold, same convention. |
| `TOUCH_TO_GOAL_DIST` | `math.sqrt(...)` ≈ `0.4231` | Derived (not hardcoded) from `GOAL_OFFSET`/`CUBE_HALF_SIZE` so it can't silently drift from the geometry it measures — a final-whole-branch-review finding caught exactly this kind of drift once already (a hardcoded test literal vs. the derived config value, 7.8e-6 apart). |
| `touch_goal_milestone_bonus` weight | `25.0` | Matches `staged_milestone_bonus`'s own weight in the sibling mirror task — no independent derivation yet, inherited by precedent. |
| `action_rate` / `joint_vel` weights | `-1e-4` / `-1e-4` | Standard small smoothness penalties, unchanged from every other task in this repo. |
| Action scale | `0.5` | Matches Experiment 10's finding (halved from `env_cfg.py`'s shared `scale=1.0`) for finer joint-position correction during precise final positioning. |

**Formula note**: the reward is *not* two summed tanh proximity bumps
(that shape created a large reward-free dead zone under the running-max
mechanism — see [[staged-reward-co-satisfiability]]) — it's `0.3 * touch_term`
pre-touch, `0.3 + 0.7 * clamp(1 - goal_dist/TOUCH_TO_GOAL_DIST, 0, 1)`
post-touch, monotonic by construction. The `0.3`/`0.7` split itself
(touch vs. goal weighting within the milestone) hasn't been independently
tuned — it mirrors `_raw_lift_progress_mirrored`'s increasing-weight-per-
stage convention without a specific derivation.

### Tier 2: reward-weight hillclimbing

For **weight/threshold tuning within an already-validated mechanism**
(not new terms), this project uses a separate fast, unattended, git-based
loop rather than hand-editing this table per attempt —
`scripts/hillclimb_rewards.py`, design in
`docs/superpowers/specs/2026-07-07-ar4-hillclimb-loop-design.md`, running
results table in `docs/superpowers/plans/2026-07-07-ar4-hillclimb-results.md`
(one row per attempt: parameter, old/new value, proxy metric, outcome,
timestamp — kept, reverted-if-worse). That table is the authoritative
history for anything tuned through it; don't duplicate its rows here,
just link to it.

## Coverage boundary

This first pass (2026-07-09) covers: current physics/PPO/actuator
defaults (verified directly against the installed files, not recalled
from memory), and Experiment 25's task-specific hyperparameters in full.
**Not yet backfilled**: per-experiment hyperparameter deltas for
Experiments 1-24 beyond what's already cited above (e.g. Experiment 10's
exact antipodal-threshold value is cited since it's still load-bearing
`_EE_OFFSET`-adjacent context, but e.g. Experiment 20's DLS damping sweep
values are not) — pull those from `ROADMAP.md`/the relevant
`kb/wiki/experiments/*.md` article on demand rather than duplicating the
whole history here preemptively.

## Related concepts

[[sim-physics-fidelity]], [[staged-reward-co-satisfiability]],
[[reward-rate-arithmetic]], [[action-space-design]],
[[ppo-critic-divergence]]
