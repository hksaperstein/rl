# Experiment 26: Gripper Reintroduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `Ar4PickPlaceGraspGoalEnvCfg` — grasp/lift/carry back in
scope after Experiment 25 removed them, composing three
previously-individually-validated fixes (proximity gate, corrected jaw
mirroring, antipodal grasp gate) with a 30s episode and a 4-stage
extension of Experiment 25's validated monotonic reward mechanism.

**Architecture:** Fix a real lag bug in existing shared code
(`tasks/ar4/actions.py`'s `MirroredGripperAction`), add a new
Isaac-Lab-free pure-math reward module (`tasks/ar4/grasp_goal_reward.py`,
same testability pattern as `touch_goal_reward.py`), add
reward/termination/observation wrapper functions to `tasks/ar4/mdp.py`,
build a new env cfg file, smoke-test it, wire it into `scripts/train.py`.

**Tech Stack:** Isaac Lab `ManagerBasedRLEnv`, `rsl_rl` PPO, PyTorch.
Launch everything via `/home/saps/IsaacLab/isaaclab.sh -p <script>`,
never bare `python` for anything touching Isaac Lab.

## Global Constraints

- Fixed cube world position: `(0.20, 0.28, 0.006)`. Fixed goal (now what
  the **cube** must reach, not just the end-effector): `GOAL_OFFSET =
  (-0.40, 0.0, 0.144)` relative to the cube's own position, same as
  Experiment 25.
- `REACH_DIST_NORM=0.3`, `LIFT_MINIMAL_HEIGHT=0.03`,
  `LIFT_TARGET_HEIGHT=0.10`, `CUBE_TO_GOAL_DIST` **derived** via
  `math.sqrt(GOAL_OFFSET[0]**2 + GOAL_OFFSET[1]**2 + GOAL_OFFSET[2]**2)`
  (≈0.4251m) — do not hardcode this as a literal.
- `force_threshold=0.05`, `antipodal_cos_threshold=-0.7071` — reused
  verbatim from Experiments 17/21/22, do not re-derive.
- `proximity_threshold=0.05` — reused verbatim from Experiments 21/22.
- Action space: arm (`JointPositionActionCfg`, `scale=0.5`, matching
  Experiment 25) + gripper (`MirroredGripperActionCfg`, the corrected
  version from Task 1).
- `episode_length_s=30.0`, `decimation=4`, `sim.dt=0.005` (today's
  physics-fidelity values — NOT the older `0.01` most other
  `pickplace_*_env_cfg.py` variants still use).
- `Ar4PickPlacePPORunnerCfg` unchanged (no new PPO cfg this pass).

---

### Task 1: Fix `MirroredGripperAction`'s lag bug

**Files:**
- Modify: `tasks/ar4/actions.py:191-194` (the `MirroredGripperAction.process_actions` method)

**Interfaces:**
- No signature changes — same class, same `MirroredGripperActionCfg`,
  same consumers (`tasks/ar4/pickplace_jawmirror_env_cfg.py` already
  uses this class; this fix changes its runtime behavior, not its
  interface).

- [ ] **Step 1: Read the current method to confirm the exact bug**

Run: `sed -n '191,195p' /home/saps/projects/rl/tasks/ar4/actions.py`

Confirm it reads:
```python
    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        jaw1_actual_pos = self._asset.data.joint_pos[:, self._joint_ids[0]]
        self._processed_actions[:, 1] = jaw1_actual_pos
```

- [ ] **Step 2: Apply the fix**

Replace with:

```python
    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        # Track jaw1's own COMMANDED target (already gate-processed by
        # super().process_actions this same step, zero lag) instead of
        # its physically-settled actual position (one physics step
        # stale under contact load) - Experiment 22's own report
        # identified this exact fix as the concrete next lever after
        # finding jaw2 structurally lags a moving jaw1 target by one
        # control step. See
        # docs/superpowers/specs/2026-07-09-ar4-experiment26-gripper-reintroduction-design.md.
        jaw1_commanded_target = self._processed_actions[:, 0]
        self._processed_actions[:, 1] = jaw1_commanded_target
```

- [ ] **Step 3: Update the class docstring**

The class docstring (`tasks/ar4/actions.py:160-180`) currently says
"gripper_jaw2_joint's commanded target continuously tracks
gripper_jaw1_joint's ACTUAL measured position each step." Update this
sentence to say it tracks jaw1's own **commanded target** (zero-lag),
not its actual measured position, and add one sentence noting this is
the corrected version of the mechanism Experiment 22 originally shipped
(which tracked the actual/settled position and was found to have a
reactive-lag problem under a moving target).

- [ ] **Step 4: Verify nothing else in the repo depends on the old (buggy) behavior**

Run: `grep -rn "MirroredGripperAction\|jaw1_actual_pos" /home/saps/projects/rl/tasks/ /home/saps/projects/rl/scripts/`

Confirm the only other consumer is
`tasks/ar4/pickplace_jawmirror_env_cfg.py` (Experiment 22's own env
cfg, which imports `MirroredGripperActionCfg` for its `ActionsCfg`).
This file doesn't reference `jaw1_actual_pos` directly (it only
constructs the cfg, doesn't touch the action class's internals), so it
is not depending on the specific buggy line — this fix improves its
behavior too, consistent with Experiment 22's own conclusion that the
lag was actively harmful. No changes needed to that file.

- [ ] **Step 5: Verify Python syntax is valid**

Run: `python3 -c "import ast; ast.parse(open('/home/saps/projects/rl/tasks/ar4/actions.py').read())"`
Expected: no output, exit code 0.

- [ ] **Step 6: Commit**

```bash
git add tasks/ar4/actions.py
git commit -m "Fix MirroredGripperAction's reactive-lag bug (Experiment 26)"
```

---

### Task 2: Pure-math 4-stage reward module + unit tests

**Files:**
- Create: `tasks/ar4/grasp_goal_reward.py`
- Create: `tests/test_grasp_goal_reward.py`

**Interfaces:**
- Produces: `grasp_goal_progress(reach_dist, grasped, lifted,
  cube_height_above_ground, goal_dist, reach_dist_norm,
  lift_minimal_height, lift_target_height, cube_to_goal_dist) ->
  torch.Tensor` — pure tensor math, zero Isaac Lab imports.
- Consumes (Task 3 will call this): raw scalar distances/booleans
  computed from live simulated state.

- [ ] **Step 1: Write `tasks/ar4/grasp_goal_reward.py`**

```python
"""Pure-tensor 4-stage (reach/grasp/lift/goal) progress math for
Experiment 26 - no Isaac Lab dependency, so this is testable with plain
pytest+torch (see tests/test_grasp_goal_reward.py), the same pattern
tasks/ar4/touch_goal_reward.py established for Experiment 25.
tasks/ar4/mdp.py reads live simulated state (end-effector position, cube
position, contact forces via antipodal_grasp_bonus's own condition) and
delegates the actual staging formula to grasp_goal_progress() below.
"""

import torch


def grasp_goal_progress(
    reach_dist: torch.Tensor,
    grasped: torch.Tensor,
    lifted: torch.Tensor,
    cube_height_above_ground: torch.Tensor,
    goal_dist: torch.Tensor,
    reach_dist_norm: float,
    lift_minimal_height: float,
    lift_target_height: float,
    cube_to_goal_dist: float,
) -> torch.Tensor:
    """Four equal 0.25-wide stage segments: reach (0.00-0.25, dense tanh-
    free linear proximity ramp, always active), grasp (0.25-0.50, a
    discrete achievement jump - genuine bilateral antipodal contact,
    computed by the caller via antipodal_grasp_bonus's own condition, IS
    the gate, not a shaped sub-metric, per Experiment 18's falsified
    dense-pre-grasp-readiness-shaping hypothesis - no partial credit
    within this segment beyond the reach ceiling already banked), lift
    (0.50-0.75, linear ramp on cube height once grasped), goal
    (0.75-1.00, linear ramp on cube-to-goal distance once lifted).
    Monotonically non-decreasing along any trajectory where reach_dist
    shrinks then grasped/lifted latch true then cube_height/goal_dist
    improve, by the same construction Experiment 25's
    touch_goal_reward.touch_goal_progress() used to avoid the dual-tanh-
    sum dead-zone bug - see
    docs/superpowers/specs/2026-07-09-ar4-experiment25-touch-goal-reach-design.md
    and [[staged-reward-co-satisfiability]] for that lesson generalized.

    `grasped`/`lifted` are latched booleans (once true, stay true for the
    episode) - the caller (tasks/ar4/mdp.py) owns that state, this
    function is a pure function of whatever it's passed each call.
    `lift_minimal_height` is accepted for interface symmetry with
    _raw_lift_progress_mirrored's own param name but is not used in this
    function's own math (the caller uses it to decide when `lifted`
    latches true in the first place, before calling this function)."""
    reach_progress = torch.clamp(1.0 - reach_dist / reach_dist_norm, min=0.0, max=1.0)
    lift_progress = torch.clamp(cube_height_above_ground / lift_target_height, min=0.0, max=1.0)
    goal_progress = torch.clamp(1.0 - goal_dist / cube_to_goal_dist, min=0.0, max=1.0)

    reach_stage = 0.25 * reach_progress
    grasp_stage = 0.50 + 0.25 * lift_progress
    goal_stage = 0.75 + 0.25 * goal_progress

    return torch.where(lifted, goal_stage, torch.where(grasped, grasp_stage, reach_stage))
```

- [ ] **Step 2: Write `tests/test_grasp_goal_reward.py`**

```python
"""Sim-independent unit tests for tasks/ar4/grasp_goal_reward.py's pure
progress math (Experiment 26) - no Isaac Lab import needed. Run via:
/home/saps/IsaacLab/_isaac_sim/python.sh -m pytest tests/test_grasp_goal_reward.py -v -p no:launch_testing
(plain python3/pytest lacks torch in this environment - see
project_pytest-needs-isaac-sim-python memory)."""

import torch

from tasks.ar4.grasp_goal_reward import grasp_goal_progress

REACH_DIST_NORM = 0.3
LIFT_MINIMAL_HEIGHT = 0.03
LIFT_TARGET_HEIGHT = 0.10
CUBE_TO_GOAL_DIST = 0.4251


def test_reach_stage_monotonic_and_bounded():
    """Before grasp, raw progress must rise monotonically as reach_dist
    shrinks, and never exceed the 0.25 reach ceiling."""
    reach_dist = torch.linspace(REACH_DIST_NORM, 0.0, 100)
    grasped = torch.zeros(100, dtype=torch.bool)
    lifted = torch.zeros(100, dtype=torch.bool)
    zeros = torch.zeros(100)

    raw = grasp_goal_progress(
        reach_dist, grasped, lifted, zeros, zeros,
        REACH_DIST_NORM, LIFT_MINIMAL_HEIGHT, LIFT_TARGET_HEIGHT, CUBE_TO_GOAL_DIST,
    )

    deltas = raw[1:] - raw[:-1]
    assert torch.all(deltas >= -1e-6), f"reach stage decreased: min delta {deltas.min().item()}"
    assert torch.all(raw <= 0.25 + 1e-6), f"reach stage exceeded 0.25 ceiling: max {raw.max().item()}"
    assert abs(raw[-1].item() - 0.25) < 1e-5, f"reach stage should reach ~0.25 at reach_dist=0, got {raw[-1].item()}"


def test_grasp_jump_is_at_least_a_quarter():
    """Achieving grasp (grasped flips true) must produce a raw-progress
    jump of at least 0.25 relative to the reach ceiling, regardless of
    reach_dist at that instant."""
    reach_dist = torch.tensor([0.0, 0.0])
    grasped = torch.tensor([False, True])
    lifted = torch.tensor([False, False])
    cube_height = torch.tensor([0.0, 0.0])
    goal_dist = torch.tensor([CUBE_TO_GOAL_DIST, CUBE_TO_GOAL_DIST])

    raw = grasp_goal_progress(
        reach_dist, grasped, lifted, cube_height, goal_dist,
        REACH_DIST_NORM, LIFT_MINIMAL_HEIGHT, LIFT_TARGET_HEIGHT, CUBE_TO_GOAL_DIST,
    )

    assert raw[1].item() - raw[0].item() >= 0.25 - 1e-6, f"grasp jump too small: {(raw[1] - raw[0]).item()}"
    assert abs(raw[1].item() - 0.50) < 1e-5, f"grasp stage floor should be exactly 0.50, got {raw[1].item()}"


def test_lift_and_goal_stages_monotonic_and_bounded():
    """Once grasped (not yet lifted), raw progress ramps 0.50->0.75 with
    cube height. Once lifted, raw progress ramps 0.75->1.00 with
    cube-to-goal distance. Neither stage ever exceeds its ceiling."""
    n = 100
    cube_height = torch.linspace(0.0, LIFT_TARGET_HEIGHT, n)
    grasped = torch.ones(n, dtype=torch.bool)
    lifted = torch.zeros(n, dtype=torch.bool)
    zeros = torch.zeros(n)

    lift_raw = grasp_goal_progress(
        zeros, grasped, lifted, cube_height, zeros,
        REACH_DIST_NORM, LIFT_MINIMAL_HEIGHT, LIFT_TARGET_HEIGHT, CUBE_TO_GOAL_DIST,
    )
    lift_deltas = lift_raw[1:] - lift_raw[:-1]
    assert torch.all(lift_deltas >= -1e-6), f"lift stage decreased: min delta {lift_deltas.min().item()}"
    assert torch.all(lift_raw <= 0.75 + 1e-6), f"lift stage exceeded 0.75 ceiling: max {lift_raw.max().item()}"

    goal_dist = torch.linspace(CUBE_TO_GOAL_DIST, 0.0, n)
    lifted_all = torch.ones(n, dtype=torch.bool)

    goal_raw = grasp_goal_progress(
        zeros, grasped, lifted_all, zeros, goal_dist,
        REACH_DIST_NORM, LIFT_MINIMAL_HEIGHT, LIFT_TARGET_HEIGHT, CUBE_TO_GOAL_DIST,
    )
    goal_deltas = goal_raw[1:] - goal_raw[:-1]
    assert torch.all(goal_deltas >= -1e-6), f"goal stage decreased: min delta {goal_deltas.min().item()}"
    assert abs(goal_raw[-1].item() - 1.0) < 1e-5, f"goal stage should reach ~1.0 at goal_dist=0, got {goal_raw[-1].item()}"
```

- [ ] **Step 3: Run the tests**

Run: `/home/saps/IsaacLab/_isaac_sim/python.sh -m pytest tests/test_grasp_goal_reward.py -v -p no:launch_testing`
Expected: 3 passed.

- [ ] **Step 4: Commit**

```bash
git add tasks/ar4/grasp_goal_reward.py tests/test_grasp_goal_reward.py
git commit -m "Add pure-math 4-stage grasp/lift/goal reward module + unit tests"
```

---

### Task 3: `tasks/ar4/mdp.py` wrapper functions

**Files:**
- Modify: `tasks/ar4/mdp.py` (append after Experiment 25's touch-goal
  functions — find the exact insertion point via `grep -n
  "def touch_goal_position_in_robot_root_frame" tasks/ar4/mdp.py` and
  insert after that function)

**Interfaces:**
- Consumes: `grasp_goal_progress` (Task 2), `antipodal_grasp_bonus`
  (existing, `tasks/ar4/mdp.py:733-772` as of this plan's writing —
  re-verify the line number via `grep` since other work may have shifted
  it).
- Produces: `set_cube_goal_position(env, env_ids, object_cfg,
  goal_offset) -> None`, `grasp_goal_milestone_bonus(env, object_cfg,
  ee_frame_cfg, jaw1_contact_cfg, jaw2_contact_cfg, reach_dist_norm=0.3,
  lift_minimal_height=0.03, lift_target_height=0.10,
  force_threshold=0.05, antipodal_cos_threshold=-0.7071,
  cube_to_goal_dist=0.4251) -> torch.Tensor`,
  `reset_grasp_goal_milestone(env, env_ids) -> None`,
  `cube_reached_goal_after_lift(env, threshold, object_cfg,
  ee_frame_cfg, jaw1_contact_cfg, jaw2_contact_cfg, reach_dist_norm=0.3,
  lift_minimal_height=0.03, force_threshold=0.05,
  antipodal_cos_threshold=-0.7071) -> torch.Tensor`,
  `grasp_state_observation(env, object_cfg, jaw1_contact_cfg,
  jaw2_contact_cfg, force_threshold=0.05, antipodal_cos_threshold=-0.7071,
  lift_minimal_height=0.03) -> torch.Tensor` (shape `(num_envs, 2)` —
  `[grasped_float, lifted_float]`), `cube_goal_position_in_robot_root_frame(env,
  robot_cfg) -> torch.Tensor`.

- [ ] **Step 1: Confirm the exact current line number of `antipodal_grasp_bonus` and the insertion point**

Run: `grep -n "^def antipodal_grasp_bonus\|^def touch_goal_position_in_robot_root_frame" /home/saps/projects/rl/tasks/ar4/mdp.py`

- [ ] **Step 2: Add the import**

Near the top of `tasks/ar4/mdp.py`, alongside the existing `from
.touch_goal_reward import touch_goal_progress` line (added in
Experiment 25 — find it via `grep -n "from .touch_goal_reward"
tasks/ar4/mdp.py`), add immediately after it:

```python
from .grasp_goal_reward import grasp_goal_progress
```

- [ ] **Step 3: Add the functions**

Insert after `touch_goal_position_in_robot_root_frame`:

```python
def set_cube_goal_position(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    object_cfg: SceneEntityCfg,
    goal_offset: tuple[float, float, float],
) -> None:
    """Event term (mode="reset"): snapshot the goal position once, from
    the cube's position at reset time - same decoupling rationale as
    Experiment 25's set_touch_goal_position (this file), now measuring
    where the CUBE itself must end up (carried there by the arm), not
    an end-effector waypoint."""
    object: RigidObject = env.scene[object_cfg.name]
    if not hasattr(env, "_cube_goal_pos_w"):
        env._cube_goal_pos_w = torch.zeros(env.num_envs, 3, device=env.device)
    goal_offset_t = torch.tensor(goal_offset, device=env.device)
    env._cube_goal_pos_w[env_ids] = object.data.root_pos_w[env_ids] + goal_offset_t


def _grasp_lift_state(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    jaw1_contact_cfg: SceneEntityCfg,
    jaw2_contact_cfg: SceneEntityCfg,
    force_threshold: float,
    antipodal_cos_threshold: float,
    lift_minimal_height: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shared helper: computes and latches env._grasped/env._lifted from
    live state. Called by every reward/termination/observation function
    below that needs this state, so the latch is always up to date
    regardless of which manager (reward/termination/observation) happens
    to run first in a given step - same idempotent-|=-latch pattern
    Experiment 25 used for env._touched_cube."""
    object: RigidObject = env.scene[object_cfg.name]
    antipodal_now = antipodal_grasp_bonus(
        env, force_threshold, antipodal_cos_threshold, jaw1_contact_cfg, jaw2_contact_cfg,
    ).bool()

    if not hasattr(env, "_grasped"):
        env._grasped = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._grasped |= antipodal_now

    cube_height_above_ground = object.data.root_pos_w[:, 2] - 0.006  # cube half-size, resting height
    if not hasattr(env, "_lifted"):
        env._lifted = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._lifted |= env._grasped & (cube_height_above_ground > lift_minimal_height)

    return env._grasped, env._lifted


def grasp_goal_milestone_bonus(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    jaw1_contact_cfg: SceneEntityCfg,
    jaw2_contact_cfg: SceneEntityCfg,
    reach_dist_norm: float = 0.3,
    lift_minimal_height: float = 0.03,
    lift_target_height: float = 0.10,
    force_threshold: float = 0.05,
    antipodal_cos_threshold: float = -0.7071,
    cube_to_goal_dist: float = 0.4251,
) -> torch.Tensor:
    """Undiscounted running-max milestone bonus over grasp_goal_progress
    - same mechanism as staged_milestone_bonus/touch_goal_milestone_bonus
    (this file): reward = (new best-ever raw progress) - (previous
    best-ever raw progress), never negative."""
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]

    grasped, lifted = _grasp_lift_state(
        env, object_cfg, jaw1_contact_cfg, jaw2_contact_cfg,
        force_threshold, antipodal_cos_threshold, lift_minimal_height,
    )

    reach_dist = torch.norm(ee_pos_w - object.data.root_pos_w, dim=-1)
    cube_height_above_ground = object.data.root_pos_w[:, 2] - 0.006

    if not hasattr(env, "_cube_goal_pos_w"):
        env._cube_goal_pos_w = torch.zeros(env.num_envs, 3, device=env.device)
    goal_dist = torch.norm(object.data.root_pos_w - env._cube_goal_pos_w, dim=-1)

    raw = grasp_goal_progress(
        reach_dist, grasped, lifted, cube_height_above_ground, goal_dist,
        reach_dist_norm, lift_minimal_height, lift_target_height, cube_to_goal_dist,
    )

    if not hasattr(env, "_grasp_goal_milestone_max"):
        env._grasp_goal_milestone_max = torch.zeros(env.num_envs, device=env.device)
    prev = env._grasp_goal_milestone_max.clone()
    env._grasp_goal_milestone_max = torch.maximum(env._grasp_goal_milestone_max, raw)
    return env._grasp_goal_milestone_max - prev


def reset_grasp_goal_milestone(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Event term (mode="reset"): zero the running-max milestone buffer
    and the grasped/lifted latches for resetting envs."""
    if not hasattr(env, "_grasp_goal_milestone_max"):
        env._grasp_goal_milestone_max = torch.zeros(env.num_envs, device=env.device)
    if not hasattr(env, "_grasped"):
        env._grasped = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if not hasattr(env, "_lifted"):
        env._lifted = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._grasp_goal_milestone_max[env_ids] = 0.0
    env._grasped[env_ids] = False
    env._lifted[env_ids] = False


def cube_reached_goal_after_lift(
    env: ManagerBasedRLEnv,
    threshold: float,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    jaw1_contact_cfg: SceneEntityCfg,
    jaw2_contact_cfg: SceneEntityCfg,
    reach_dist_norm: float = 0.3,
    lift_minimal_height: float = 0.03,
    force_threshold: float = 0.05,
    antipodal_cos_threshold: float = -0.7071,
) -> torch.Tensor:
    """Termination: cube within threshold of env._cube_goal_pos_w AND
    env._lifted true for that env (genuine grasp+lift occurred at some
    point this episode, not just incidental cube-goal proximity)."""
    object: RigidObject = env.scene[object_cfg.name]
    _, lifted = _grasp_lift_state(
        env, object_cfg, jaw1_contact_cfg, jaw2_contact_cfg,
        force_threshold, antipodal_cos_threshold, lift_minimal_height,
    )
    if not hasattr(env, "_cube_goal_pos_w"):
        env._cube_goal_pos_w = torch.zeros(env.num_envs, 3, device=env.device)
    goal_dist = torch.norm(object.data.root_pos_w - env._cube_goal_pos_w, dim=-1)
    return (goal_dist < threshold) & lifted


def grasp_state_observation(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    jaw1_contact_cfg: SceneEntityCfg,
    jaw2_contact_cfg: SceneEntityCfg,
    force_threshold: float = 0.05,
    antipodal_cos_threshold: float = -0.7071,
    lift_minimal_height: float = 0.03,
) -> torch.Tensor:
    """Observation: [grasped_float, lifted_float] latched state, shape
    (num_envs, 2) - gives the policy direct access to its own stage
    progress rather than requiring it to infer this from raw
    contact/height signals alone."""
    grasped, lifted = _grasp_lift_state(
        env, object_cfg, jaw1_contact_cfg, jaw2_contact_cfg,
        force_threshold, antipodal_cos_threshold, lift_minimal_height,
    )
    return torch.stack([grasped.float(), lifted.float()], dim=-1)


def cube_goal_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """env._cube_goal_pos_w (set once at reset by set_cube_goal_position)
    expressed in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    if not hasattr(env, "_cube_goal_pos_w"):
        env._cube_goal_pos_w = torch.zeros(env.num_envs, 3, device=env.device)
    goal_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, env._cube_goal_pos_w)
    return goal_pos_b
```

- [ ] **Step 4: Verify Python syntax is valid**

Run: `python3 -c "import ast; ast.parse(open('/home/saps/projects/rl/tasks/ar4/mdp.py').read())"`
Expected: no output, exit code 0.

- [ ] **Step 5: Commit**

```bash
git add tasks/ar4/mdp.py
git commit -m "Add grasp/lift/goal reward, termination, and observation functions for Experiment 26"
```

---

### Task 4: Create `tasks/ar4/pickplace_graspgoal_env_cfg.py`

**Files:**
- Create: `tasks/ar4/pickplace_graspgoal_env_cfg.py`

**Interfaces:**
- Consumes: Task 1's fixed `MirroredGripperActionCfg`
  (`tasks/ar4/actions.py`); Task 3's `set_cube_goal_position`,
  `grasp_goal_milestone_bonus`, `reset_grasp_goal_milestone`,
  `cube_reached_goal_after_lift`, `grasp_state_observation`,
  `cube_goal_position_in_robot_root_frame` (`tasks/ar4/mdp.py`);
  `CUBE_CFG` (`tasks/ar4/objects_cfg.py`); `AR4_MK5_CFG`,
  `ARM_JOINT_NAMES`, `GRIPPER_JOINT_NAMES`, `GRIPPER_OPEN_POS`,
  `GRIPPER_CLOSED_POS` (`tasks/ar4/robot_cfg.py`); `_EE_OFFSET`
  (`tasks/ar4/pickplace_env_cfg.py`).
- Produces: `Ar4PickPlaceGraspGoalEnvCfg` (the class `scripts/train.py`
  will import in Task 6).

- [ ] **Step 1: Write the file**

```python
# tasks/ar4/pickplace_graspgoal_env_cfg.py
"""Grasp/lift/goal variant of the AR4 pick-and-place task (Experiment
26): reintroduces the gripper after Experiment 25 removed it. Composes
Experiment 21's proximity-gated gripper, Experiment 22's mirroring
mechanism (corrected for its own identified reactive-lag bug - see
tasks/ar4/actions.py's MirroredGripperAction), and Experiment 17's
antipodal grasp gate, with a 30s episode and a 4-stage extension of
Experiment 25's validated monotonic reward mechanism. See
docs/superpowers/specs/2026-07-09-ar4-experiment26-gripper-reintroduction-design.md.

Additive/parallel to pickplace_touchgoal_env_cfg.py: deliberately does
NOT modify that file, env_cfg.py, objects_cfg.py, or mdp.py's Experiment
25 functions.

Import this module only after an Isaac Sim/Isaac Lab AppLauncher has
been created.
"""

import math

import isaaclab.envs.mdp as isaaclab_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.manipulation.lift import mdp

from . import mdp as ar4_mdp
from .actions import MirroredGripperActionCfg
from .objects_cfg import CUBE_CFG
from .pickplace_env_cfg import _EE_OFFSET
from .robot_cfg import (
    ARM_JOINT_NAMES,
    AR4_MK5_CFG,
    GRIPPER_CLOSED_POS,
    GRIPPER_JOINT_NAMES,
    GRIPPER_OPEN_POS,
)

# Same fixed cube spawn as Experiment 25; goal is now where the CUBE must
# end up (carried there), not an end-effector waypoint - same offset
# value, different physical meaning.
GOAL_OFFSET = (-0.40, 0.0, 0.144)

CUBE_HALF_SIZE = 0.006
REACH_DIST_NORM = 0.3
LIFT_MINIMAL_HEIGHT = 0.03
LIFT_TARGET_HEIGHT = 0.10
GOAL_TOLERANCE = 0.02
FORCE_THRESHOLD = 0.05
ANTIPODAL_COS_THRESHOLD = -0.7071
PROXIMITY_THRESHOLD = 0.05
# Derived, not hardcoded - matches Experiment 25's final-review lesson on
# geometry constants silently drifting from what they measure.
CUBE_TO_GOAL_DIST = math.sqrt(GOAL_OFFSET[0] ** 2 + GOAL_OFFSET[1] ** 2 + GOAL_OFFSET[2] ** 2)


@configclass
class ActionsCfg:
    """Arm (Experiment 25's proven scale=0.5) + gripper (proximity-gated,
    lag-corrected mirroring - see tasks/ar4/actions.py)."""

    joint_positions = isaaclab_mdp.JointPositionActionCfg(asset_name="robot", joint_names=ARM_JOINT_NAMES, scale=0.5)
    gripper_position = MirroredGripperActionCfg(
        asset_name="robot",
        joint_names=GRIPPER_JOINT_NAMES,
        open_command_expr={name: GRIPPER_OPEN_POS for name in GRIPPER_JOINT_NAMES},
        close_command_expr={name: GRIPPER_CLOSED_POS for name in GRIPPER_JOINT_NAMES},
        object_cfg=SceneEntityCfg("cube"),
        ee_frame_cfg=SceneEntityCfg("ee_frame"),
        proximity_threshold=PROXIMITY_THRESHOLD,
    )


@configclass
class Ar4PickPlaceGraspGoalSceneCfg(InteractiveSceneCfg):
    """AR4 arm + gripper + a single fixed-position cube. Re-adds the
    gripper-jaw contact sensors Experiment 25's touch-goal lineage
    dropped (required for antipodal_grasp_bonus)."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )
    robot: ArticulationCfg = AR4_MK5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cube: RigidObjectCfg = CUBE_CFG.replace(
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.20, 0.28, 0.006)),
        spawn=CUBE_CFG.spawn.replace(
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            )
        ),
    )

    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/root_joint/base_link",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/root_joint/link_6",
                name="end_effector",
                offset=OffsetCfg(pos=_EE_OFFSET),
            ),
        ],
    )
    gripper_jaw1_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/root_joint/gripper_jaw1_link",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
    )
    gripper_jaw2_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/root_joint/gripper_jaw2_link",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
    )


@configclass
class ObservationsCfg:
    """Arm + gripper joint state (unrestricted joint_names - gripper is
    actuated again), cube position, goal position, grasp/lift latch
    state, last action."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        cube_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame, params={"object_cfg": SceneEntityCfg("cube")}
        )
        goal_position = ObsTerm(
            func=ar4_mdp.cube_goal_position_in_robot_root_frame, params={"robot_cfg": SceneEntityCfg("robot")}
        )
        grasp_state = ObsTerm(
            func=ar4_mdp.grasp_state_observation,
            params={
                "object_cfg": SceneEntityCfg("cube"),
                "jaw1_contact_cfg": SceneEntityCfg("gripper_jaw1_contact"),
                "jaw2_contact_cfg": SceneEntityCfg("gripper_jaw2_contact"),
                "force_threshold": FORCE_THRESHOLD,
                "antipodal_cos_threshold": ANTIPODAL_COS_THRESHOLD,
                "lift_minimal_height": LIFT_MINIMAL_HEIGHT,
            },
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset events: whole scene back to default, then snapshot the cube
    goal position, then zero the milestone/latch buffers."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    set_cube_goal_position = EventTerm(
        func=ar4_mdp.set_cube_goal_position,
        mode="reset",
        params={"object_cfg": SceneEntityCfg("cube"), "goal_offset": GOAL_OFFSET},
    )

    reset_grasp_goal_milestone = EventTerm(func=ar4_mdp.reset_grasp_goal_milestone, mode="reset")


@configclass
class TerminationsCfg:
    """Success (genuine grasp+lift occurred, then the cube reached the
    goal) ends the episode early; otherwise a fixed timeout."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    cube_reached_goal = DoneTerm(
        func=ar4_mdp.cube_reached_goal_after_lift,
        params={
            "threshold": GOAL_TOLERANCE,
            "object_cfg": SceneEntityCfg("cube"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "jaw1_contact_cfg": SceneEntityCfg("gripper_jaw1_contact"),
            "jaw2_contact_cfg": SceneEntityCfg("gripper_jaw2_contact"),
            "reach_dist_norm": REACH_DIST_NORM,
            "lift_minimal_height": LIFT_MINIMAL_HEIGHT,
            "force_threshold": FORCE_THRESHOLD,
            "antipodal_cos_threshold": ANTIPODAL_COS_THRESHOLD,
        },
    )


@configclass
class RewardsCfg:
    """Four-stage gated running-max milestone bonus: reach, grasp
    (antipodal gate), lift, goal - no ungated additive sum (Experiment
    16's wedging exploit), no separately-weighted independent terms
    (Experiment 17/18's discoverability gap)."""

    grasp_goal_milestone_bonus = RewTerm(
        func=ar4_mdp.grasp_goal_milestone_bonus,
        weight=25.0,
        params={
            "object_cfg": SceneEntityCfg("cube"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "jaw1_contact_cfg": SceneEntityCfg("gripper_jaw1_contact"),
            "jaw2_contact_cfg": SceneEntityCfg("gripper_jaw2_contact"),
            "reach_dist_norm": REACH_DIST_NORM,
            "lift_minimal_height": LIFT_MINIMAL_HEIGHT,
            "lift_target_height": LIFT_TARGET_HEIGHT,
            "force_threshold": FORCE_THRESHOLD,
            "antipodal_cos_threshold": ANTIPODAL_COS_THRESHOLD,
            "cube_to_goal_dist": CUBE_TO_GOAL_DIST,
        },
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class Ar4PickPlaceGraspGoalEnvCfg(ManagerBasedRLEnvCfg):
    """AR4 grasp/lift/goal task (Experiment 26): reach, grasp, lift, and
    carry the cube to a fixed goal point. num_envs=4096 default -
    scripts/train.py's --num_envs flag overrides this per-run."""

    scene: Ar4PickPlaceGraspGoalSceneCfg = Ar4PickPlaceGraspGoalSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 30.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0)
        self.viewer.eye = (1.5, 1.5, 1.2)
        self.viewer.lookat = (0.0, 0.0, 0.4)
```

- [ ] **Step 2: Verify Python syntax is valid**

Run: `python3 -c "import ast; ast.parse(open('/home/saps/projects/rl/tasks/ar4/pickplace_graspgoal_env_cfg.py').read())"`
Expected: no output, exit code 0.

- [ ] **Step 3: Commit**

```bash
git add tasks/ar4/pickplace_graspgoal_env_cfg.py
git commit -m "Add Ar4PickPlaceGraspGoalEnvCfg (Experiment 26: reintroduce the gripper)"
```

---

### Task 5: Smoke-test the new env cfg in real Isaac Sim

**Files:**
- Create: `scripts/smoke_test_graspgoal_env.py`

**Interfaces:**
- Consumes: `Ar4PickPlaceGraspGoalEnvCfg` (Task 4).

- [ ] **Step 1: Write the smoke-test script**

```python
# scripts/smoke_test_graspgoal_env.py
"""Headless smoke test for Ar4PickPlaceGraspGoalEnvCfg (Experiment 26):
builds the env, steps it a fixed number of times with all-zero actions
across a few envs, and prints observation shapes, reward values, and
grasp/lift latch state - real evidence the env cfg actually builds and
runs, not just that it imports. Same pattern as
scripts/smoke_test_touchgoal_env.py (Experiment 25).

.. code-block:: bash

    /home/saps/IsaacLab/isaaclab.sh -p scripts/smoke_test_graspgoal_env.py
"""

import sys

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser(description="Smoke test for the grasp-goal env cfg.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os  # noqa: E402

import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from tasks.ar4.pickplace_graspgoal_env_cfg import Ar4PickPlaceGraspGoalEnvCfg  # noqa: E402


def main() -> None:
    env_cfg = Ar4PickPlaceGraspGoalEnvCfg()
    env_cfg.scene.num_envs = 4
    env_cfg.sim.device = args_cli.device
    env = ManagerBasedRLEnv(cfg=env_cfg)

    with torch.inference_mode():
        obs, _ = env.reset()
        print(f"[SMOKE] observation shape: {obs['policy'].shape}")
        print(f"[SMOKE] action space shape: {env.action_manager.action.shape} (expect (4, 8) - arm 6 + gripper 2)")

        for step in range(50):
            actions = torch.zeros(4, env.action_manager.total_action_dim, device=env.device)
            obs, rew, terminated, truncated, info = env.step(actions)
            if step % 10 == 0:
                print(
                    f"[SMOKE] step {step}: reward={rew.cpu().tolist()}, "
                    f"terminated={terminated.cpu().tolist()}, truncated={truncated.cpu().tolist()}"
                )

        cube = env.scene["cube"]
        ee_frame = env.scene["ee_frame"]
        print(f"[SMOKE] cube position (env 0): {cube.data.root_pos_w[0].cpu().tolist()}")
        print(f"[SMOKE] ee_frame target position (env 0): {ee_frame.data.target_pos_w[0, 0].cpu().tolist()}")
        print(f"[SMOKE] env._grasped: {env._grasped.cpu().tolist()}")
        print(f"[SMOKE] env._lifted: {env._lifted.cpu().tolist()}")
        print(f"[SMOKE] env._grasp_goal_milestone_max: {env._grasp_goal_milestone_max.cpu().tolist()}")

    env.close()
    print("[SMOKE] PASS: env built, stepped 50 times, no exceptions.")


if __name__ == "__main__":
    main()
    simulation_app.close()
```

- [ ] **Step 2: Run the smoke test**

Run: `PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/smoke_test_graspgoal_env.py`
Expected: `[SMOKE] PASS` as the final line, action space shape `(4, 8)`
(arm 6 + gripper 2 — back to 8 now that the gripper is reintroduced),
`env._grasped`/`env._lifted` both all-`False` (all-zero actions can't
plausibly achieve genuine grasp+lift in 50 steps), no NaNs. If the
observation shape differs from what you expect, recompute by hand from
the actual `ObservationsCfg` terms (`joint_pos`(8) + `joint_vel`(8) +
`cube_position`(3) + `goal_position`(3) + `grasp_state`(2) +
`actions`(8) = 32) and fix whichever term's dimension is wrong before
proceeding - don't just accept a different number silently.

If this fails with an import or attribute error, fix the specific
mismatch and re-run before proceeding - do not move to Task 6 with a
failing smoke test.

- [ ] **Step 3: Commit**

```bash
git add scripts/smoke_test_graspgoal_env.py
git commit -m "Add headless smoke test for the grasp-goal env cfg"
```

---

### Task 6: Wire `--graspgoal` into `scripts/train.py`

**Files:**
- Modify: `scripts/train.py`

**Interfaces:**
- Consumes: `Ar4PickPlaceGraspGoalEnvCfg` (Task 4),
  `Ar4PickPlacePPORunnerCfg` (already imported, no change needed).

- [ ] **Step 1: Add the import**

Immediately after the `from tasks.ar4.pickplace_touchgoal_env_cfg
import Ar4PickPlaceTouchGoalEnvCfg` line (added for Experiment 25 —
find it via `grep -n "pickplace_touchgoal_env_cfg" scripts/train.py`),
add:

```python
from tasks.ar4.pickplace_graspgoal_env_cfg import Ar4PickPlaceGraspGoalEnvCfg  # noqa: E402
```

- [ ] **Step 2: Add the `--graspgoal` flag**

Immediately before `AppLauncher.add_app_launcher_args(parser)`, add:

```python
parser.add_argument(
    "--graspgoal",
    action="store_true",
    default=False,
    help=(
        "Train on the grasp-goal variant (Experiment 26): reintroduces the gripper after Experiment "
        "25 removed it. Composes Experiment 21's proximity-gated gripper, Experiment 22's mirroring "
        "mechanism (corrected for its own identified reactive-lag bug), and Experiment 17's antipodal "
        "grasp gate, with a 30s episode and a 4-stage extension of Experiment 25's validated monotonic "
        "reward mechanism. See "
        "docs/superpowers/specs/2026-07-09-ar4-experiment26-gripper-reintroduction-design.md."
    ),
)
```

- [ ] **Step 3: Add the env cfg selection branch**

In the `if/elif` chain (the current first branch is `if
args_cli.touchgoal:` from Experiment 25 — find it via `grep -n "if
args_cli.touchgoal" scripts/train.py`), add as the new first branch:

```python
    if args_cli.graspgoal:
        env_cfg_cls = Ar4PickPlaceGraspGoalEnvCfg
    elif args_cli.touchgoal:
        env_cfg_cls = Ar4PickPlaceTouchGoalEnvCfg
```

(i.e. insert the new `if` first, change the existing `if
args_cli.touchgoal:` to `elif args_cli.touchgoal:` — do not duplicate
the `if`.)

- [ ] **Step 4: Confirm `--graspgoal` falls through to the default PPO runner cfg**

Read the existing taskspace-family condition (`if args_cli.taskspace or
args_cli.residual or ...`) and confirm `graspgoal` is absent from it, so
it falls through to `else: agent_cfg = Ar4PickPlacePPORunnerCfg()`. No
code change needed, just confirm by reading it.

- [ ] **Step 5: Run a short real training smoke test**

Run: `PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/train.py --graspgoal --num_envs 64 --max_iterations 3`
Expected: training starts, runs 3 iterations, exits cleanly, and
`logs/train/<timestamp>/` contains checkpoint files and
`params/agent.yaml`/`params/env.yaml`. Check `params/agent.yaml` for
`clip_actions: null` (confirms the default, not taskspace, PPO runner
cfg was actually used) and `params/env.yaml` for the grasp-goal-specific
observation/reward term names, same verification method used for
Experiment 25's Task 4. If this errors, debug the actual root cause and
fix it before considering this task done.

- [ ] **Step 6: Commit**

```bash
git add scripts/train.py
git commit -m "Wire --graspgoal flag into scripts/train.py (Experiment 26)"
```
