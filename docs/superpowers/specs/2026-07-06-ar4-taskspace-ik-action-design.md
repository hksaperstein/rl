# AR4 task-space IK-driven action — Design

## Goal

Experiment 11, the eleventh real attempt on the "grasp/lift never
emerges" sub-problem. User-proposed architectural change: instead of the
policy directly outputting joint positions (with IK used only as a
reward-shaping aid, Experiments 8-10), let the policy generate
**Cartesian path/position commands** and use IK **as part of the actual
control loop** to convert them into joint targets — offloading the "how
to move 6 joints" problem to a classical, non-learned solver, so the
policy only has to learn "where to go and when."

## Why now

Experiment 10's own data supports this directly: after correcting
`antipodal_grasp_bonus`'s geometric threshold (loosening it from -0.85 to
the physically-correct -0.7071), the antipodal condition still regressed
to **exactly 0** by the end of training (worse than Experiment 9's
already-tiny 0.001416) — the policy converged *away* from ever achieving
genuine bilateral opposition, not toward it. Loosening the threshold
didn't help, which argues the bottleneck isn't strictness but **precision**:
achieving true antipodal contact (both jaws pressing from truly opposing
directions on a small 18mm cube) requires more precise final gripper
positioning/alignment than direct joint-space control reliably achieves.
Task-space IK control directly targets that precision gap.

## Architecture

New parallel task file `tasks/ar4/pickplace_taskspace_env_cfg.py`,
reusing `Ar4PickPlaceMirrorSceneCfg` (the cube scene, spawn
randomization, mirrored goal — all already proven working) and
Experiment 10's reward fixes (antipodal threshold, corrected). Only the
**action space** and the **path-tracking reward's internals** change.

### Action space

Replace `JointPositionActionCfg` (6 joint-angle outputs) with Isaac
Lab's built-in `DifferentialInverseKinematicsActionCfg`:

- `command_type="position"` (3D Cartesian, not full 6-DOF pose —
  orientation isn't critical for this task's fixed top-down approach
  geometry, keeping the action space small, consistent with every
  prior experiment's scope).
- `use_relative_mode=True`: the policy outputs incremental Cartesian
  deltas each step (not absolute positions) — smoother for RL
  exploration, and mirrors how the *current* joint-space action already
  works as small deltas from a default offset (`use_default_offset=True`).
- `body_name="link_6"`, `body_offset=OffsetCfg(pos=_EE_OFFSET)`: reuses
  the already-measured-and-verified 0.036m gripper-pinch-point offset
  (the same constant used throughout this session's `ee_frame`
  `FrameTransformerCfg`), so the *controlled point* is the actual
  gripper tip, not `link_6` itself.
- `scale=0.05`: 5cm maximum Cartesian step per unit of policy output,
  reasoned from the workspace's ~0.3-0.5m scale (large enough to cross
  the workspace in a reasonable number of steps at full deflection,
  small enough for fine positioning control).
- `controller=DifferentialIKControllerCfg(command_type="position",
  use_relative_mode=False, ik_method="dls")` — the controller's own
  `use_relative_mode` is a separate flag from the action term's (the
  action term computes the target pose from the incremental input
  itself; the controller then solves IK for that absolute target each
  step), and `ik_method="dls"` matches the damped-least-squares choice
  already validated in Experiments 8-10's reward-side IK usage.
- Gripper action (`BinaryJointPositionActionCfg`) is unchanged.

Isaac Lab's `DifferentialInverseKinematicsAction` implementation
(`isaaclab/envs/mdp/actions/task_space_actions.py`) already handles the
fixed-base Jacobian indexing internally (`self._asset.is_fixed_base`
branch) — the exact same logic this session's `ik_guided_path_bonus`
had to hand-implement for its reward-only IK usage. No new indexing
code needed.

### Reward simplification

`ik_guided_path_bonus`'s "IK-match" sub-signal (compare the policy's
actual joint configuration against what a live IK controller suggests)
becomes redundant once IK is *part of the control loop* — the arm will
track IK's suggestion by construction, so scoring "does it match IK" is
close to tautological. New, simpler `path_proximity_bonus` function:
keeps the waypoint-sequenced Cartesian proximity term (same proven
undiscounted running-max pattern: `env._path_waypoint_idx` advances
monotonically, `env._path_milestone_max` running-max delta), dropping
the Jacobian/IK-controller computation entirely from the reward
function. This also means the reward function no longer needs to
construct its own `DifferentialIKController` instance (previously
lazily instantiated inside `ik_guided_path_bonus`) — that instance now
lives only in the action term, one less redundant object.

`antipodal_grasp_bonus` (with Experiment 10's `-0.7071` threshold),
`gripper_schedule_bonus`, `stillness_penalty`, `action_rate`,
`joint_vel` all carry over unchanged — this experiment isolates the
action-space variable specifically.

### Observations

Unchanged: `joint_pos_rel`, `joint_vel_rel`, `cube_position`,
`target_object_position`, `last_action` are all generic w.r.t. the
underlying action term (they read joint/object state directly, not the
action term's internals) and remain valid without modification.
`last_action` will now report the 3D Cartesian delta + 2D binary
gripper command instead of the previous 6D joint-delta + 2D gripper
command — a smaller observation vector, no code change needed since
it's already generic.

## Verification plan

1. Smoke test (`--num_envs 16 --max_iterations 2`): exits 0 (verified
   via file evidence per this session's established practice — Isaac
   Sim's stdout is unreliable), confirms the expected 5 reward terms
   (`path_proximity_bonus`, `gripper_schedule_bonus`,
   `antipodal_grasp_bonus`, `stillness_penalty`, `action_rate`,
   `joint_vel` — 6 total) active with no exceptions.
2. Full 1500-iteration run at `num_envs=4096`, same TensorBoard-scalar
   verification standard as every prior experiment this session
   (checkpoint count, `model_1499.pt` existence, event-file timing all
   checked before trusting any scalar data).
3. Real eval + frame-by-frame video inspection (not just TensorBoard
   scalars) before any success/failure judgment — this session's
   established standard, reinforced by two prior false positives from
   coarse video sampling.
4. Record outcome in `ROADMAP.md` regardless of result.

## Global constraints for the plan

- Do not modify `env_cfg.py`, `objects_cfg.py`, `pickplace_env_cfg.py`,
  `pickplace_mirror_env_cfg.py`, or `pickplace_ik_guided_env_cfg.py` —
  this is a new, parallel task reusing `Ar4PickPlaceMirrorSceneCfg` by
  import, not modifying it.
- Do not modify any existing `mdp.py` function (`ik_guided_path_bonus`,
  `compute_path_waypoints`, `antipodal_grasp_bonus`, `contact_grasp_bonus`,
  `gripper_schedule_bonus`, `stillness_penalty`, `set_mirrored_goal`,
  etc.) — `path_proximity_bonus` is a new function, and
  `compute_path_waypoints`'s existing waypoint-computation logic is
  reused as-is (it doesn't reference IK at all, purely geometric).
- Decided values, not placeholders: `command_type="position"`,
  `use_relative_mode=True` (action term), `scale=0.05`,
  `body_name="link_6"`, `body_offset=_EE_OFFSET=(0.0, 0.0, 0.036)`,
  controller `use_relative_mode=False`, `ik_method="dls"`.
- `num_envs=4096` for the real training run, matching every prior
  experiment this session.
- Reuses `Ar4PickPlacePPORunnerCfg` unchanged (no new PPO
  hyperparameters — this experiment isolates the action-space variable).
