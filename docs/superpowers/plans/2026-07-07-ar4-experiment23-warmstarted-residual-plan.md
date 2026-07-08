# Experiment 23: Warm-Started Residual RL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a residual-RL action term over the classical 5-waypoint pursuit controller, with a linear warm-start ramp on the residual's authority (Johannink et al. 2019's fix, never implemented in Experiment 13), and re-run the current best reward lineage (Experiment 22's) under it to test whether `lifting_object`/`both_magnitude_ok_steps` move off their current `0/1500`/`0/750` null.

**Architecture:** New `WarmStartedResidualDifferentialIKAction` subclasses `DifferentialInverseKinematicsAction` (same base every task-space action term in this repo already extends), combining a bounded pursuit step toward the active waypoint (reused from Experiment 13's `_compute_base_delta`, with the waypoint-advance side effect moved inline) with the policy's own scaled action, weighted by a `residual_authority` factor that ramps linearly from 0 to 1.0 over `cfg.warmup_steps` env steps. A new env cfg reuses Experiment 22's reward/observation/termination/curriculum verbatim and swaps only the arm action term.

**Tech Stack:** Isaac Lab / Isaac Sim, PyTorch, rsl_rl PPO.

## Global Constraints

- Always invoke Isaac Lab scripts via `/home/saps/IsaacLab/isaaclab.sh -p scripts/<script>.py` from this repo's root — never plain `python`.
- Every subagent dispatched to run a real Isaac Sim job must be given the **literal** blocking poll command verbatim in its dispatch prompt (not a paraphrase).
- Task 4's warm-start verification is a hard blocker — do not proceed to Task 5/6 if it fails.
- No reward-function changes anywhere in this plan. The only new variable is the arm action space.
- Do not modify `tasks/ar4/residual_ik_action.py`, `tasks/ar4/pickplace_residual_env_cfg.py`, or any existing `pickplace_*_env_cfg.py` file — purely additive.
- Commit after each task (not just at the end), and push to `origin/main` after Task 6 per this repo's git conventions (private solo repo, no PR workflow).
- `warmup_steps=1200` (50 iterations × `num_steps_per_env=24`, from `tasks/ar4/agents/rsl_rl_ppo_cfg.py`), `advance_tolerance=0.03`, `base_max_step=0.05` — exact values, not placeholders.

---

### Task 1: `WarmStartedResidualDifferentialIKAction` action term

**Files:**
- Modify: `tasks/ar4/actions.py` (append new classes at the end, after `MirroredGripperActionCfg`)
- Create: `scripts/warmresidual_action_smoke_test.py`

**Interfaces:**
- Consumes: `isaaclab.envs.mdp.actions.actions_cfg.DifferentialInverseKinematicsActionCfg`, `isaaclab.envs.mdp.actions.task_space_actions.DifferentialInverseKinematicsAction` (both already imported in `tasks/ar4/actions.py`); `isaaclab.utils.math.subtract_frame_transforms` (already imported in `tasks/ar4/residual_ik_action.py`, not yet in `actions.py`); `env._path_waypoints_w` / `env._path_waypoint_idx` (set by `compute_path_waypoints`, `tasks/ar4/mdp.py:404`); `SceneEntityCfg` (already imported in `actions.py`).
- Produces: `WarmStartedResidualDifferentialIKAction` (class), `WarmStartedResidualDifferentialIKActionCfg` (configclass with fields `ee_frame_cfg: SceneEntityCfg`, `base_max_step: float = 0.05`, `advance_tolerance: float`, `warmup_steps: int`) — both imported by Task 2's env cfg as `from .actions import WarmStartedResidualDifferentialIKActionCfg`.

- [ ] **Step 1: Add the import for `subtract_frame_transforms`**

`tasks/ar4/actions.py` currently imports from `isaaclab.utils` (line 32: `from isaaclab.utils import configclass`) but not `isaaclab.utils.math`. Add this import alongside the existing imports (after line 32):

```python
from isaaclab.utils.math import subtract_frame_transforms
```

- [ ] **Step 2: Append the new action term and its cfg to `tasks/ar4/actions.py`**

Append at the end of the file (after the existing `MirroredGripperActionCfg` class, which currently ends the file):

```python


class WarmStartedResidualDifferentialIKAction(DifferentialInverseKinematicsAction):
    """Residual RL over a classical waypoint-pursuit base controller
    (same _compute_base_delta mechanism as ResidualDifferentialIKAction
    in tasks/ar4/residual_ik_action.py - Experiment 13's original), with
    a literal-percentage-of-training linear warm-start ramp on the
    residual's authority: residual_authority = min(1.0, step_count /
    cfg.warmup_steps). Approximates Johannink et al. 2019's technique of
    holding the residual at zero for an initial period while training
    only the value function - Experiment 13's own diagnosed regression
    cause was the ABSENCE of any such warm-start (residual_authority
    implicitly 1.0 from step 0), which this class fixes. See
    docs/superpowers/specs/2026-07-07-ar4-experiment23-warmstarted-residual-design.md.

    Also performs the waypoint auto-advance side effect
    (env._path_waypoint_idx increments when the end-effector comes
    within cfg.advance_tolerance of the active waypoint) directly inside
    _compute_base_delta, rather than reusing ik_guided_path_bonus's
    bundled reward+advance logic (tasks/ar4/mdp.py) - this experiment's
    only new variable is the action space, not the reward function.

    self._step_count increments once per process_actions call (i.e. once
    per env.step, not per PPO iteration - num_steps_per_env=24 env steps
    make up one iteration, per tasks/ar4/agents/rsl_rl_ppo_cfg.py) and is
    NOT reset on episode reset - it tracks wall-clock training progress
    across the whole run, not per-episode progress, matching Johannink
    et al.'s framing of an initial TRAINING period, not an initial
    EPISODE period.
    """

    cfg: WarmStartedResidualDifferentialIKActionCfg

    def __init__(self, cfg: WarmStartedResidualDifferentialIKActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._step_count = 0

    def _compute_base_delta(self) -> torch.Tensor:
        """Bounded proportional ("seek") step toward the currently-active
        waypoint, identical convention to ResidualDifferentialIKAction's
        own method (tasks/ar4/residual_ik_action.py) - returns zeros
        before the first compute_path_waypoints reset event has run.
        Additionally advances env._path_waypoint_idx here (moved from
        ik_guided_path_bonus, which this action term does not use)."""
        env = self._env
        if not hasattr(env, "_path_waypoints_w"):
            return torch.zeros(self.num_envs, 3, device=self.device)
        current_waypoint_w = torch.gather(
            env._path_waypoints_w, 1, env._path_waypoint_idx.view(-1, 1, 1).expand(-1, 1, 3)
        ).squeeze(1)
        root_pose_w = self._asset.data.root_pose_w
        target_b, _ = subtract_frame_transforms(root_pose_w[:, 0:3], root_pose_w[:, 3:7], current_waypoint_w)
        ee_pos_curr, _ = self._compute_frame_pose()
        direction = target_b - ee_pos_curr
        dist = torch.norm(direction, dim=-1, keepdim=True)
        step = torch.clamp(dist, max=self.cfg.base_max_step)

        ee_frame: FrameTransformer = env.scene[self.cfg.ee_frame_cfg.name]
        ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
        dist_to_waypoint_w = torch.norm(ee_pos_w - current_waypoint_w, dim=-1)
        reached = dist_to_waypoint_w < self.cfg.advance_tolerance
        env._path_waypoint_idx = torch.where(
            reached & (env._path_waypoint_idx < 4), env._path_waypoint_idx + 1, env._path_waypoint_idx
        )

        return direction / (dist + 1e-8) * step

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        base_delta = self._compute_base_delta()
        residual_authority = min(1.0, self._step_count / self.cfg.warmup_steps)
        self._processed_actions[:] = base_delta + residual_authority * self.raw_actions * self._scale
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        self._ik_controller.set_command(self._processed_actions, ee_pos_curr, ee_quat_curr)
        self._step_count += 1


@configclass
class WarmStartedResidualDifferentialIKActionCfg(DifferentialInverseKinematicsActionCfg):
    """Adds ee_frame_cfg/base_max_step/advance_tolerance/warmup_steps to
    the stock IK action cfg. See WarmStartedResidualDifferentialIKAction."""

    class_type: type[ActionTerm] = WarmStartedResidualDifferentialIKAction
    ee_frame_cfg: SceneEntityCfg = MISSING
    """The end-effector frame used for the waypoint-advance distance
    check (target_pos_w, the same field ProximityGatedBinaryJointPositionAction
    already reads from this same cfg name elsewhere in this file)."""
    base_max_step: float = 0.05
    """Max per-step Cartesian pursuit distance (meters) the base
    controller contributes toward the active waypoint."""
    advance_tolerance: float = MISSING
    """Distance (m) below which the end-effector is considered to have
    reached the active waypoint, advancing env._path_waypoint_idx."""
    warmup_steps: int = MISSING
    """Number of env.step() calls (process_actions invocations) over
    which residual_authority ramps linearly from 0 to 1.0."""
```

- [ ] **Step 3: Update the module docstring**

The file's module docstring (lines 1-17) currently describes only `VerticalLockDifferentialIKAction` (Experiment 20). Since this file now holds four unrelated action terms across four experiments, add one sentence noting the new addition without rewriting the whole docstring. Replace the docstring's first line:

```python
# tasks/ar4/actions.py
"""Custom task-space/gripper action terms for AR4 experiments 20-23:
VerticalLockDifferentialIKAction (Experiment 20), ProximityGatedBinaryJointPositionAction
and MirroredGripperAction (Experiments 21-22), and
WarmStartedResidualDifferentialIKAction (Experiment 23, residual RL with a
literature-grounded warm-start - see
docs/superpowers/specs/2026-07-07-ar4-experiment23-warmstarted-residual-design.md).
```

Keep the rest of the existing docstring (the "Subclasses Isaac Lab's built-in..." paragraph and the "Import this module only after..." paragraph) unchanged below this replaced first block.

- [ ] **Step 4: Write the smoke test script**

Create `scripts/warmresidual_action_smoke_test.py`:

```python
"""Direct unit-style verification of WarmStartedResidualDifferentialIKAction's
residual_authority ramp formula, in isolation from any real env rollout
(Task 4's separate script verifies the ramp during an actual rollout -
this script only checks the formula itself is implemented correctly).

.. code-block:: bash

    /home/saps/IsaacLab/isaaclab.sh -p scripts/warmresidual_action_smoke_test.py
"""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import sys  # noqa: E402

sys.path.insert(0, "/home/saps/projects/rl")  # noqa: E402

WARMUP_STEPS = 1200


def residual_authority(step_count: int, warmup_steps: int) -> float:
    return min(1.0, step_count / warmup_steps)


def main() -> None:
    checks = [
        (0, 0.0),
        (WARMUP_STEPS // 2, 0.5),
        (WARMUP_STEPS, 1.0),
        (WARMUP_STEPS * 2, 1.0),
    ]
    all_pass = True
    for step_count, expected in checks:
        actual = residual_authority(step_count, WARMUP_STEPS)
        ok = abs(actual - expected) < 1e-6
        all_pass = all_pass and ok
        print(f"[CHECK] step_count={step_count} expected={expected} actual={actual} {'PASS' if ok else 'FAIL'}")

    if all_pass:
        print("[SMOKE TEST] ALL CHECKS PASS")
    else:
        print("[SMOKE TEST] FAILURES DETECTED")
        sys.exit(1)


if __name__ == "__main__":
    main()
    simulation_app.close()
```

- [ ] **Step 5: Run the smoke test**

Run: `PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/warmresidual_action_smoke_test.py`
Expected: four `[CHECK] ... PASS` lines, then `[SMOKE TEST] ALL CHECKS PASS`.

This checks the ramp formula itself (`min(1.0, step_count/warmup_steps)`), not the action term class importing/instantiating cleanly — that is verified for real in Task 4, which requires a live env. This step exists to catch an arithmetic mistake in the formula cheaply before paying for a full env spin-up.

- [ ] **Step 6: Verify `tasks/ar4/actions.py` still imports cleanly**

Run: `/home/saps/IsaacLab/isaaclab.sh -p -c "from isaaclab.app import AppLauncher; app_launcher = AppLauncher(headless=True); simulation_app = app_launcher.app; import sys; sys.path.insert(0, '/home/saps/projects/rl'); from tasks.ar4.actions import WarmStartedResidualDifferentialIKAction, WarmStartedResidualDifferentialIKActionCfg; print('IMPORT OK'); simulation_app.close()"`
Expected: `IMPORT OK` printed, no traceback.

- [ ] **Step 7: Commit**

```bash
git add tasks/ar4/actions.py scripts/warmresidual_action_smoke_test.py
git commit -m "Add WarmStartedResidualDifferentialIKAction: residual RL with literature-grounded warm-start ramp (Experiment 23)"
```

---

### Task 2: New env cfg `Ar4PickPlaceWarmResidualEnvCfg`

**Files:**
- Create: `tasks/ar4/pickplace_warmresidual_env_cfg.py`

**Interfaces:**
- Consumes: `WarmStartedResidualDifferentialIKActionCfg` (Task 1, `tasks/ar4/actions.py`); `MirroredGripperActionCfg` (existing, `tasks/ar4/actions.py`); `RewardsCfg`/`ObservationsCfg`/`TerminationsCfg`/`CurriculumCfg` (existing, imported from `tasks/ar4/pickplace_orientationbias_env_cfg.py` — these are the exact classes Experiment 22's `pickplace_jawmirror_env_cfg.py` already imports, confirmed identical reuse chain); `Ar4PickPlaceMirrorSceneCfg` (existing, `tasks/ar4/pickplace_mirror_env_cfg.py`); `compute_path_waypoints` (existing, `tasks/ar4/mdp.py:404`); `_EE_OFFSET = (0.0, 0.0, 0.036)` (existing, `tasks/ar4/pickplace_env_cfg.py:42`).
- Produces: `Ar4PickPlaceWarmResidualEnvCfg` (class) — imported by Task 3's `scripts/train.py`/`scripts/eval_loop.py` as `from tasks.ar4.pickplace_warmresidual_env_cfg import Ar4PickPlaceWarmResidualEnvCfg`.

- [ ] **Step 1: Create the file**

Create `tasks/ar4/pickplace_warmresidual_env_cfg.py`:

```python
# tasks/ar4/pickplace_warmresidual_env_cfg.py
"""Warm-started residual-RL variant (Experiment 23): identical to
Experiment 22's Ar4PickPlaceJawMirrorEnvCfg
(tasks/ar4/pickplace_jawmirror_env_cfg.py) in reward/observation/
termination/curriculum configuration and gripper action - only the ARM
action term changes, from plain joint-space position control to
WarmStartedResidualDifferentialIKActionCfg (a classical waypoint-pursuit
base controller plus a warm-started RL residual on top).

Revisits Experiment 13's residual-RL-over-classical-controller paradigm
(tasks/ar4/residual_ik_action.py, tasks/ar4/pickplace_residual_env_cfg.py)
with the specific literature-grounded fix (Johannink et al. 2019's
warm-start technique) Experiment 13's own ROADMAP entry diagnosed as
missing but never implemented. See
docs/superpowers/specs/2026-07-07-ar4-experiment23-warmstarted-residual-design.md.

Additive/parallel to every other pickplace_*.py file: does NOT modify
pickplace_jawmirror_env_cfg.py, pickplace_orientationbias_env_cfg.py,
pickplace_residual_env_cfg.py, tasks/ar4/residual_ik_action.py, or any
other existing env cfg/action file. Reuses Ar4PickPlaceMirrorSceneCfg,
Ar4PickPlaceTaskspacePPORunnerCfg, and Experiment 22's RewardsCfg/
ObservationsCfg/TerminationsCfg/CurriculumCfg directly (imported via
Experiment 20's pickplace_orientationbias_env_cfg.py, the same reuse
chain pickplace_jawmirror_env_cfg.py already uses).

Import this module only after an Isaac Sim/Isaac Lab AppLauncher has been
created.
"""

import isaaclab.envs.mdp as isaaclab_mdp
import isaaclab.sim as sim_utils
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from . import mdp as ar4_mdp
from .actions import MirroredGripperActionCfg, WarmStartedResidualDifferentialIKActionCfg
from .pickplace_env_cfg import _EE_OFFSET
from .pickplace_mirror_env_cfg import Ar4PickPlaceMirrorSceneCfg
from .pickplace_orientationbias_env_cfg import (
    CurriculumCfg,
    EventCfg as _BaseEventCfg,
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
)
from .pickplace_taskspace_env_cfg import Ar4PickPlaceTaskspacePPORunnerCfg  # noqa: F401 (re-exported for scripts/train.py's PPO-cfg selection)
from .robot_cfg import ARM_JOINT_NAMES, GRIPPER_CLOSED_POS, GRIPPER_JOINT_NAMES, GRIPPER_OPEN_POS

# Same values as pickplace_residual_env_cfg.py's/pickplace_taskspace_env_cfg.py's
# EventCfg reuse - the waypoint geometry itself is unchanged, only the
# action term consuming it is new.
_LIFT_MINIMAL_HEIGHT = 0.03
_PREGRASP_HOVER = 0.05
_LIFT_MARGIN = 0.02
_CARRY_HEIGHT = 0.10

# Same value as Experiment 21/22 - unchanged, this experiment isolates
# the arm action term as the only new variable.
_PROXIMITY_THRESHOLD = 0.05

# 50 iterations x num_steps_per_env=24 (tasks/ar4/agents/rsl_rl_ppo_cfg.py)
# = 1200 env steps, ~3.3% of the full 1500-iteration/36,000-step run.
_WARMUP_STEPS = 1200
_ADVANCE_TOLERANCE = 0.03
_BASE_MAX_STEP = 0.05


@configclass
class ActionsCfg:
    """Arm action replaced with WarmStartedResidualDifferentialIKActionCfg
    (classical waypoint pursuit + warm-started RL residual). Gripper
    action unchanged from Experiment 22: MirroredGripperActionCfg (jaw2
    mirrors jaw1's actual position, gated open unless the cube is within
    _PROXIMITY_THRESHOLD)."""

    arm_action = WarmStartedResidualDifferentialIKActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINT_NAMES,
        body_name="link_6",
        body_offset=isaaclab_mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=_EE_OFFSET),
        scale=0.05,
        controller=DifferentialIKControllerCfg(command_type="position", use_relative_mode=True, ik_method="dls"),
        ee_frame_cfg=SceneEntityCfg("ee_frame"),
        base_max_step=_BASE_MAX_STEP,
        advance_tolerance=_ADVANCE_TOLERANCE,
        warmup_steps=_WARMUP_STEPS,
    )
    gripper_position = MirroredGripperActionCfg(
        asset_name="robot",
        joint_names=GRIPPER_JOINT_NAMES,
        open_command_expr={name: GRIPPER_OPEN_POS for name in GRIPPER_JOINT_NAMES},
        close_command_expr={name: GRIPPER_CLOSED_POS for name in GRIPPER_JOINT_NAMES},
        object_cfg=SceneEntityCfg("cube"),
        ee_frame_cfg=SceneEntityCfg("ee_frame"),
        proximity_threshold=_PROXIMITY_THRESHOLD,
    )


@configclass
class EventCfg(_BaseEventCfg):
    """Experiment 22's exact reset_all/reset_cube_position/randomize_goal
    events (inherited unchanged from Experiment 20's EventCfg, via
    pickplace_orientationbias_env_cfg.py), plus one new event:
    compute_path_waypoints, registered LAST (after cube-position reset
    and goal randomization, per that function's own documented ordering
    requirement - it reads both env._target_pos_w and the cube's
    now-updated position)."""

    compute_path_waypoints = EventTerm(
        func=ar4_mdp.compute_path_waypoints,
        mode="reset",
        params={
            "object_cfg": SceneEntityCfg("cube"),
            "lift_minimal_height": _LIFT_MINIMAL_HEIGHT,
            "pregrasp_hover": _PREGRASP_HOVER,
            "lift_margin": _LIFT_MARGIN,
            "carry_height": _CARRY_HEIGHT,
        },
    )


@configclass
class Ar4PickPlaceWarmResidualEnvCfg(ManagerBasedRLEnvCfg):
    """AR4 warm-started residual-RL task (Experiment 23): Experiment 22's
    exact reward/observation/termination/curriculum configuration and
    gripper mechanism, with the arm's action replaced by a classical
    waypoint-pursuit base controller plus a warm-started RL residual.
    num_envs=4096 default - scripts/train.py's --num_envs flag overrides
    this per-run same as every other env cfg in this repo."""

    scene: Ar4PickPlaceMirrorSceneCfg = Ar4PickPlaceMirrorSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 5.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0)
        self.viewer.eye = (1.5, 1.5, 1.2)
        self.viewer.lookat = (0.0, 0.0, 0.4)
```

Note on `EventCfg(_BaseEventCfg)`: `configclass`-decorated classes support normal Python inheritance (Isaac Lab's `configclass` is a dataclass-style decorator, not a metaclass that blocks subclassing) — this matches the same pattern this repo already uses when a variant only adds one field/term without needing to redeclare the base's other fields. If subclassing an `@configclass`-decorated `EventCfg` does not work when actually run (verify in Step 2 below), fall back to copying all four base event terms (`reset_all`, `reset_cube_position`, `randomize_goal`) explicitly from `tasks/ar4/pickplace_pregrasp_env_cfg.py`'s `EventCfg` (the original definition Experiment 20's `EventCfg` reuses unchanged) into this file's `EventCfg` instead of subclassing, then add `compute_path_waypoints` as a fourth term alongside them.

- [ ] **Step 2: Verify the env cfg imports and instantiates cleanly**

Run:
```bash
/home/saps/IsaacLab/isaaclab.sh -p -c "
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app
import sys
sys.path.insert(0, '/home/saps/projects/rl')
from tasks.ar4.pickplace_warmresidual_env_cfg import Ar4PickPlaceWarmResidualEnvCfg
cfg = Ar4PickPlaceWarmResidualEnvCfg()
cfg.scene.num_envs = 4
print('EventCfg terms:', list(vars(cfg.events).keys()))
assert 'compute_path_waypoints' in vars(cfg.events), 'compute_path_waypoints missing from EventCfg'
assert 'reset_cube_position' in vars(cfg.events), 'reset_cube_position missing from EventCfg (subclassing did not inherit base terms)'
print('CONFIG INSTANTIATE OK')
simulation_app.close()
"
```
Expected: `EventCfg terms: [...]` listing at least `reset_all`, `reset_cube_position`, `randomize_goal`, `compute_path_waypoints`, then `CONFIG INSTANTIATE OK`, no traceback. If the `reset_cube_position` assertion fails, apply the fallback described in Step 1 (copy the base terms explicitly instead of subclassing) and re-run this verification.

- [ ] **Step 3: Commit**

```bash
git add tasks/ar4/pickplace_warmresidual_env_cfg.py
git commit -m "Add Ar4PickPlaceWarmResidualEnvCfg: Experiment 22's reward lineage under the warm-started residual action (Experiment 23)"
```

---

### Task 3: Wire `--warmresidual` flag into `scripts/train.py` and `scripts/eval_loop.py`

**Files:**
- Modify: `scripts/train.py`
- Modify: `scripts/eval_loop.py`

**Interfaces:**
- Consumes: `Ar4PickPlaceWarmResidualEnvCfg` (Task 2, `tasks/ar4/pickplace_warmresidual_env_cfg.py`).
- Produces: `--warmresidual` CLI flag on both scripts, selecting `Ar4PickPlaceWarmResidualEnvCfg` and joining the `Ar4PickPlaceTaskspacePPORunnerCfg` selection branch (needed because `WarmStartedResidualDifferentialIKAction` is a task-space differential-IK action term, exactly like `--taskspace`/`--residual`/`--reachskip`/`--baseproximity`, all of which already require the raised `clip_actions=5.0` fix documented in `Ar4PickPlaceTaskspacePPORunnerCfg`'s own docstring).

- [ ] **Step 1: Add the `--warmresidual` argparse flag to `scripts/train.py`**

In `scripts/train.py`, after the existing `--jawmirror` argument block (currently ending at line 180, right before `AppLauncher.add_app_launcher_args(parser)` at line 181), insert:

```python
parser.add_argument(
    "--warmresidual",
    action="store_true",
    default=False,
    help=(
        "Train on the warm-started residual-RL variant: the arm's action is a classical "
        "waypoint-pursuit base controller plus a policy residual whose authority ramps linearly "
        "from 0 to 1.0 over the first 1200 env steps (Johannink et al. 2019's warm-start "
        "technique, never implemented in Experiment 13's original residual attempt), on top of "
        "Experiment 22's unchanged reward set and gripper mechanism. See "
        "docs/superpowers/specs/2026-07-07-ar4-experiment23-warmstarted-residual-design.md."
    ),
)
```

- [ ] **Step 2: Add the import to `scripts/train.py`**

After the existing line `from tasks.ar4.pickplace_taskspace_env_cfg import (  # noqa: E402` block (lines 223-226), add:

```python
from tasks.ar4.pickplace_warmresidual_env_cfg import Ar4PickPlaceWarmResidualEnvCfg  # noqa: E402
```

- [ ] **Step 3: Add the env-cfg selection branch to `scripts/train.py`**

In the `main()` function's `if args_cli.jawmirror: ... elif ...` chain (lines 232-259), add a new branch as the FIRST condition (matching this repo's established pattern of newest-experiment-first ordering):

```python
    if args_cli.warmresidual:
        env_cfg_cls = Ar4PickPlaceWarmResidualEnvCfg
    elif args_cli.jawmirror:
        env_cfg_cls = Ar4PickPlaceJawMirrorEnvCfg
```

(leave the rest of the `elif` chain, lines 234-259, unchanged below this).

- [ ] **Step 4: Add `--warmresidual` to the task-space PPO runner cfg selection condition in `scripts/train.py`**

Line 264 currently reads:
```python
    if args_cli.taskspace or args_cli.residual or args_cli.reachskip or args_cli.baseproximity:
```
Change to:
```python
    if args_cli.taskspace or args_cli.residual or args_cli.reachskip or args_cli.baseproximity or args_cli.warmresidual:
```

- [ ] **Step 5: Add the `--warmresidual` argparse flag to `scripts/eval_loop.py`**

After the existing `--jawmirror` argument block (currently ending at line 97, right before `AppLauncher.add_app_launcher_args(parser)` at line 98), insert:

```python
parser.add_argument(
    "--warmresidual",
    action="store_true",
    default=False,
    help="Evaluate the warm-started residual-RL scene (see scripts/train.py --warmresidual) instead of the four-object scene.",
)
```

- [ ] **Step 6: Add the import to `scripts/eval_loop.py`**

After the existing line `from tasks.ar4.pickplace_taskspace_env_cfg import (  # noqa: E402` block (lines 139-142), add:

```python
from tasks.ar4.pickplace_warmresidual_env_cfg import Ar4PickPlaceWarmResidualEnvCfg  # noqa: E402
```

- [ ] **Step 7: Add the env-cfg selection branch to `scripts/eval_loop.py`**

In the `main()` function's `if args_cli.jawmirror: ... elif ...` chain (lines 148-175), add as the FIRST condition:

```python
    if args_cli.warmresidual:
        env_cfg_cls = Ar4PickPlaceWarmResidualEnvCfg
    elif args_cli.jawmirror:
        env_cfg_cls = Ar4PickPlaceJawMirrorEnvCfg
```

- [ ] **Step 8: Add `--warmresidual` to the task-space PPO runner cfg selection condition in `scripts/eval_loop.py`**

Line 180 currently reads:
```python
    if args_cli.taskspace or args_cli.residual or args_cli.reachskip or args_cli.baseproximity:
```
Change to:
```python
    if args_cli.taskspace or args_cli.residual or args_cli.reachskip or args_cli.baseproximity or args_cli.warmresidual:
```

- [ ] **Step 9: Add the `--warmresidual` video name-prefix branch to `scripts/eval_loop.py`**

In the `name_prefix` `if args_cli.jawmirror: ... elif ...` chain (lines 201-226), add as the FIRST condition:

```python
        if args_cli.warmresidual:
            name_prefix = "ar4_pickplace_warmresidual"
        elif args_cli.jawmirror:
            name_prefix = "ar4_pickplace_jawmirror"
```

- [ ] **Step 10: Verify both scripts still parse `--help` without error**

Run: `/home/saps/IsaacLab/isaaclab.sh -p scripts/train.py --help | grep -A 3 warmresidual`
Expected: the `--warmresidual` flag's help text is printed, no traceback.

Run: `/home/saps/IsaacLab/isaaclab.sh -p scripts/eval_loop.py --checkpoint /nonexistent --help 2>&1 | grep -A 2 warmresidual`
Expected: the `--warmresidual` flag's help text is printed. (`--checkpoint` is required by argparse but `--help` short-circuits before the checkpoint-existence check, so a dummy value is fine here.)

- [ ] **Step 11: Commit**

```bash
git add scripts/train.py scripts/eval_loop.py
git commit -m "Wire --warmresidual flag into train.py/eval_loop.py for Experiment 23"
```

---

### Task 4: Warm-start ramp verification during a real rollout (hard gate)

**Files:**
- Create: `scripts/warmresidual_verify.py`

**Interfaces:**
- Consumes: `Ar4PickPlaceWarmResidualEnvCfg` (Task 2); `Ar4PickPlaceTaskspacePPORunnerCfg` (existing, `tasks/ar4/pickplace_taskspace_env_cfg.py` — used only for `clip_actions`, no checkpoint is loaded in this script since no training has happened yet).
- Produces: printed verification that `residual_authority` (recomputed from the action term's own `_step_count`/`cfg.warmup_steps`, read directly off the live `ManagerBasedRLEnv` instance) rises from ~0 near step 0 to 1.0 by step 1200 during an actual `env.step()` loop — this is the hard gate. If this step fails, STOP and report BLOCKED; do not proceed to Task 5 or 6.

- [ ] **Step 1: Write the verification script**

Create `scripts/warmresidual_verify.py`:

```python
"""Instrumented rollout confirming WarmStartedResidualDifferentialIKAction's
residual_authority ramp actually rises from ~0 toward 1.0 over cfg.warmup_steps
env steps during a REAL env rollout - not just the isolated formula check in
scripts/warmresidual_action_smoke_test.py. This is Experiment 23's hard gate:
if the ramp doesn't move as expected here, the whole design's central premise
(a warm-started residual, per Johannink et al. 2019) has not actually been
implemented correctly, and training must not proceed. See
docs/superpowers/specs/2026-07-07-ar4-experiment23-warmstarted-residual-design.md.

Uses a zero-action policy (no trained checkpoint exists yet at this point in
the plan) - the ramp value itself does not depend on what the policy outputs,
only on cfg.warmup_steps and how many process_actions() calls have occurred,
so a zero policy is sufficient to verify the ramp mechanism in isolation.

.. code-block:: bash

    PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/warmresidual_verify.py --steps 1300
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify WarmStartedResidualDifferentialIKAction's warm-start ramp.")
parser.add_argument("--steps", type=int, default=1300, help="Number of env steps to run.")
parser.add_argument("--log_every", type=int, default=100, help="Print the ramp value every N steps.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

sys.path.insert(0, "/home/saps/projects/rl")  # noqa: E402

from tasks.ar4.pickplace_warmresidual_env_cfg import Ar4PickPlaceWarmResidualEnvCfg  # noqa: E402


def main() -> None:
    env_cfg = Ar4PickPlaceWarmResidualEnvCfg()
    env_cfg.scene.num_envs = 4
    env_cfg.sim.device = args_cli.device

    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode=None)
    arm_action_term = env.action_manager.get_term("arm_action")

    obs, _ = env.reset()
    action_dim = env.action_manager.total_action_dim
    zero_actions = torch.zeros(env_cfg.scene.num_envs, action_dim, device=env.device)

    readings = []
    with torch.inference_mode():
        for step in range(args_cli.steps):
            env.step(zero_actions)
            authority = min(1.0, arm_action_term._step_count / arm_action_term.cfg.warmup_steps)
            if step % args_cli.log_every == 0 or step == args_cli.steps - 1:
                print(
                    f"[STEP {step:5d}] internal_step_count={arm_action_term._step_count} "
                    f"warmup_steps={arm_action_term.cfg.warmup_steps} residual_authority={authority:.4f}"
                )
            readings.append((step, authority))

    env.close()

    first_authority = readings[0][1]
    near_warmup_idx = min(range(len(readings)), key=lambda i: abs(readings[i][0] - arm_action_term.cfg.warmup_steps))
    at_warmup_authority = readings[near_warmup_idx][1]
    final_authority = readings[-1][1]

    print(
        f"[SUMMARY] first_step_authority={first_authority:.4f} "
        f"authority_near_warmup_step={at_warmup_authority:.4f} "
        f"final_authority={final_authority:.4f}"
    )

    ramp_rose = first_authority < 0.05 and at_warmup_authority > 0.9
    clamped_at_one = final_authority == 1.0
    if ramp_rose and clamped_at_one:
        print("[VERIFICATION] PASS: residual_authority ramped from ~0 to 1.0 and stayed clamped.")
    else:
        print(
            f"[VERIFICATION] FAIL: ramp_rose={ramp_rose} clamped_at_one={clamped_at_one} - "
            "BLOCKED, do not proceed to Task 5/6."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
    simulation_app.close()
```

- [ ] **Step 2: Run the verification**

Run: `PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/warmresidual_verify.py --steps 1300 2>&1 | tee /tmp/exp23_warmstart_verify.log`
Expected: a series of `[STEP ...]` lines showing `residual_authority` rising from near 0.0 toward 1.0 as `step` approaches 1200, then `[SUMMARY] ...` and `[VERIFICATION] PASS: ...`.

**This is a hard gate.** If the output ends with `[VERIFICATION] FAIL: ...`, STOP — do not proceed to Task 5 or Task 6. Report BLOCKED with the full log content and let the controller decide (per this plan's Global Constraints).

- [ ] **Step 3: Commit**

```bash
git add scripts/warmresidual_verify.py
git commit -m "Add warm-start ramp verification script for Experiment 23 (hard gate, confirmed passing)"
```

---

### Task 5: 300-iteration diagnostic run

**Files:**
- None modified — this task runs training and inspects logs only.

**Interfaces:**
- Consumes: `--warmresidual` flag (Task 3), `scripts/train.py`.
- Produces: a diagnostic checkpoint directory under `logs/train/`, and a short status report (no new file — report inline in the task's completion message, matching this repo's own established diagnostic-task pattern from Experiments 19/20/21/22).

- [ ] **Step 1: Launch the 300-iteration diagnostic run**

Run (from the repo root, `/home/saps/projects/rl`):
```bash
nohup /home/saps/IsaacLab/isaaclab.sh -p scripts/train.py --warmresidual --num_envs 4096 --headless --max_iterations 300 > /tmp/exp23_diagnostic_stdout.log 2>&1 &
```

- [ ] **Step 2: Poll for completion (literal blocking command)**

Run this exact command and wait for it to return (it blocks until the checkpoint file appears):
```bash
until find /home/saps/projects/rl/logs/train/ -name "model_299.pt" -newer /home/saps/projects/rl/tasks/ar4/pickplace_warmresidual_env_cfg.py 2>/dev/null | grep -q .; do sleep 60; done
```

- [ ] **Step 3: Check for tracebacks/exceptions**

Run: `grep -iE "traceback|error|exception" /tmp/exp23_diagnostic_stdout.log`
Expected: no matches, or only expected/benign Isaac Sim startup warnings (compare against a known-clean prior run's log, e.g. `/tmp/exp22_diagnostic_stdout.log` if it still exists, to distinguish routine startup noise from a real failure).

- [ ] **Step 4: Extract `Loss/value_function` and check it's bounded**

Run:
```bash
LOG_DIR=$(find /home/saps/projects/rl/logs/train/ -maxdepth 1 -type d -newer /home/saps/projects/rl/tasks/ar4/pickplace_warmresidual_env_cfg.py | sort | tail -1)
echo "LOG_DIR=$LOG_DIR"
python3 -c "
from tensorboard.backend.event_processing import event_accumulator
import sys
ea = event_accumulator.EventAccumulator('$LOG_DIR', size_guidance={'scalars': 0})
ea.Reload()
values = [s.value for s in ea.Scalars('Loss/value_function')]
print(f'Loss/value_function: min={min(values):.4f} max={max(values):.4f} final={values[-1]:.4f} count={len(values)}')
"
```
Expected: `max` value bounded (compare against Experiment 22's own max, which stayed small throughout — no sustained unbounded growth trend). This is the specific risk this repo's own Experiment 11/13 history flags for any new action term.

- [ ] **Step 5: Report `lifting_object`'s nonzero count in this 300-iteration window**

Run:
```bash
python3 -c "
from tensorboard.backend.event_processing import event_accumulator
ea = event_accumulator.EventAccumulator('$LOG_DIR', size_guidance={'scalars': 0})
ea.Reload()
values = [s.value for s in ea.Scalars('Episode_Reward/lifting_object')]
nonzero = sum(1 for v in values if v != 0.0)
print(f'Episode_Reward/lifting_object: nonzero_count={nonzero}/{len(values)}')
"
```
Report the exact count either way — this is diagnostic-only (300 iterations is 1/5 of a full run), not the falsifiable answer Task 6 exists to determine. Do not stop or block on this number; proceed to Task 6 regardless of its value as long as Steps 3-4 passed.

---

### Task 6: Full 1500-iteration run, contact diagnostic, and report

**Files:**
- Create: `scripts/warmresidual_contact_diagnostic.py`
- Create: `docs/superpowers/plans/2026-07-07-ar4-experiment23-report.md`

**Interfaces:**
- Consumes: `Ar4PickPlaceWarmResidualEnvCfg` (Task 2); `Ar4PickPlaceTaskspacePPORunnerCfg` (existing, `tasks/ar4/pickplace_taskspace_env_cfg.py`); the full-run checkpoint produced by this task's own Step 1.
- Produces: `docs/superpowers/plans/2026-07-07-ar4-experiment23-report.md`, committed and pushed.

- [ ] **Step 1: Launch the full 1500-iteration run**

Run (from the repo root):
```bash
nohup /home/saps/IsaacLab/isaaclab.sh -p scripts/train.py --warmresidual --num_envs 4096 --headless > /tmp/exp23_train_stdout.log 2>&1 &
```

- [ ] **Step 2: Poll for completion (literal blocking command)**

Run this exact command and wait for it to return:
```bash
until find /home/saps/projects/rl/logs/train/ -name "model_1499.pt" -newer /home/saps/projects/rl/tasks/ar4/pickplace_warmresidual_env_cfg.py 2>/dev/null | grep -q .; do sleep 60; done
```

- [ ] **Step 3: Verify checkpoint integrity and no tracebacks**

Run:
```bash
LOG_DIR=$(find /home/saps/projects/rl/logs/train/ -maxdepth 1 -type d -newer /home/saps/projects/rl/tasks/ar4/pickplace_warmresidual_env_cfg.py | sort | tail -1)
echo "LOG_DIR=$LOG_DIR"
ls "$LOG_DIR"/model_*.pt | wc -l
test -f "$LOG_DIR/model_1499.pt" && echo "model_1499.pt EXISTS"
grep -iE "traceback|error|exception" /tmp/exp23_train_stdout.log
```
Expected: 31 checkpoint files (matching Experiment 22's own count, `save_interval=50` over 1500 iterations), `model_1499.pt EXISTS`, and no traceback/error/exception matches (or only routine startup noise, same standard as Task 5 Step 3).

- [ ] **Step 4: Extract TensorBoard scalars and compare against Experiment 22's exact final values**

Run:
```bash
python3 -c "
from tensorboard.backend.event_processing import event_accumulator
ea = event_accumulator.EventAccumulator('$LOG_DIR', size_guidance={'scalars': 0})
ea.Reload()
terms = [
    'Episode_Reward/reaching_object',
    'Episode_Reward/pregrasp_readiness',
    'Episode_Reward/orientation_alignment',
    'Episode_Reward/lifting_object',
    'Episode_Reward/object_goal_tracking',
    'Episode_Reward/object_goal_tracking_fine_grained',
    'Episode_Termination/cube_reached_goal',
    'Loss/value_function',
]
for term in terms:
    try:
        values = [s.value for s in ea.Scalars(term)]
        nonzero = sum(1 for v in values if v != 0.0)
        print(f'{term}: count={len(values)} nonzero={nonzero} min={min(values):.4f} max={max(values):.4f} final={values[-1]:.4f}')
    except KeyError:
        print(f'{term}: NOT FOUND')
"
```
Record the full output — this becomes the report's scalar-extraction section, structured the same way as `docs/superpowers/plans/2026-07-07-ar4-experiment22-report.md`.

- [ ] **Step 5: Write the contact diagnostic script**

Create `scripts/warmresidual_contact_diagnostic.py` (adapted from Experiment 22's `exp22_contact_diagnostic.py`, loading `Ar4PickPlaceWarmResidualEnvCfg`/`Ar4PickPlaceTaskspacePPORunnerCfg` instead, and additionally logging `residual_authority` since training is now complete and the ramp should read 1.0 throughout any post-checkpoint rollout):

```python
"""Instrumented rollout of Experiment 23's trained checkpoint
(Ar4PickPlaceWarmResidualEnvCfg, model_1499.pt) - does antipodal contact
happen under the warm-started residual action, using the same
antipodal_grasp_bonus/genuine_grasp_and_lift computation this repo's own
mdp.py uses (tasks/ar4/mdp.py), reproduced inline exactly as Experiments
20/21/22's own contact diagnostics already do:
  - height_ok:          cube world z > minimal_height (0.03)
  - both_magnitude_ok:  jaw1_force_mag > force_threshold (0.05) AND
                         jaw2_force_mag > force_threshold (0.05)
  - antipodal_ok:       cos(angle between jaw1_force_dir, jaw2_force_dir)
                         < antipodal_cos_threshold (-0.7071)
  - grasp_ok:           both_magnitude_ok AND antipodal_ok
  - gate_fires:         height_ok AND grasp_ok

Also logs residual_authority (expected 1.0 throughout, since training is
complete and _step_count is already far past warmup_steps by the time
this checkpoint was saved) and gripper joint positions.

.. code-block:: bash

    PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/warmresidual_contact_diagnostic.py \
        --checkpoint <path> --episodes 3
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Experiment 23 antipodal-contact diagnostic.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the rsl_rl checkpoint (.pt) to load.")
parser.add_argument("--episodes", type=int, default=3, help="Number of full episodes to run.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = False

if not os.path.isfile(args_cli.checkpoint):
    sys.exit(f"Checkpoint not found: {args_cli.checkpoint}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

sys.path.insert(0, "/home/saps/projects/rl")  # noqa: E402

from tasks.ar4.pickplace_taskspace_env_cfg import Ar4PickPlaceTaskspacePPORunnerCfg  # noqa: E402
from tasks.ar4.pickplace_warmresidual_env_cfg import Ar4PickPlaceWarmResidualEnvCfg  # noqa: E402

FORCE_THRESHOLD = 0.05
ANTIPODAL_COS_THRESHOLD = -0.7071
MINIMAL_HEIGHT = 0.03


def main() -> None:
    env_cfg = Ar4PickPlaceWarmResidualEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device

    agent_cfg = Ar4PickPlaceTaskspacePPORunnerCfg()
    agent_cfg.device = args_cli.device

    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode=None)
    arm_action_term = env.action_manager.get_term("arm_action")
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=wrapped.unwrapped.device)

    cube = env.scene["cube"]
    ee_frame = env.scene["ee_frame"]
    jaw1_sensor = env.scene["gripper_jaw1_contact"]
    jaw2_sensor = env.scene["gripper_jaw2_contact"]
    robot = env.scene["robot"]

    gripper_joint_ids, gripper_joint_names = robot.find_joints(["gripper_jaw1_joint", "gripper_jaw2_joint"])
    print(f"[SETUP] gripper joint ids={gripper_joint_ids} names={gripper_joint_names}")
    print(
        f"[SETUP] thresholds: force_threshold={FORCE_THRESHOLD} "
        f"antipodal_cos_threshold={ANTIPODAL_COS_THRESHOLD} minimal_height={MINIMAL_HEIGHT} "
        f"warmup_steps={arm_action_term.cfg.warmup_steps}"
    )

    stats = {
        "total_steps": 0,
        "height_ok_steps": 0,
        "both_magnitude_ok_steps": 0,
        "antipodal_ok_steps": 0,
        "grasp_ok_steps": 0,
        "gate_fires_steps": 0,
        "max_jaw1_force": 0.0,
        "max_jaw2_force": 0.0,
        "max_cube_z": -1.0,
        "min_residual_authority": 2.0,
    }

    obs = wrapped.get_observations()
    with torch.inference_mode():
        for episode in range(args_cli.episodes):
            print(f"[EPISODE {episode} START]")
            for step in range(250):
                actions = policy(obs)
                obs, _, dones, _ = wrapped.step(actions)

                cube_pos = cube.data.root_pos_w[0]
                cube_z = cube_pos[2].item()

                jaw1_force_vec = jaw1_sensor.data.force_matrix_w.view(1, 3)[0]
                jaw2_force_vec = jaw2_sensor.data.force_matrix_w.view(1, 3)[0]
                jaw1_force_mag = torch.linalg.vector_norm(jaw1_force_vec).item()
                jaw2_force_mag = torch.linalg.vector_norm(jaw2_force_vec).item()

                jaw1_dir = jaw1_force_vec / (jaw1_force_mag + 1e-8)
                jaw2_dir = jaw2_force_vec / (jaw2_force_mag + 1e-8)
                cos_angle = torch.sum(jaw1_dir * jaw2_dir).item()

                height_ok = cube_z > MINIMAL_HEIGHT
                both_magnitude_ok = (jaw1_force_mag > FORCE_THRESHOLD) and (jaw2_force_mag > FORCE_THRESHOLD)
                antipodal_ok = cos_angle < ANTIPODAL_COS_THRESHOLD
                grasp_ok = both_magnitude_ok and antipodal_ok
                gate_fires = height_ok and grasp_ok

                gripper_joint_pos = robot.data.joint_pos[0, gripper_joint_ids].tolist()
                residual_authority = min(1.0, arm_action_term._step_count / arm_action_term.cfg.warmup_steps)

                ee_quat_w = ee_frame.data.target_quat_w[0, 0, :]
                approach_dir = quat_apply(ee_quat_w.unsqueeze(0), torch.tensor([[0.0, 0.0, 1.0]], device=env.device))[0]
                orientation_dot = torch.dot(approach_dir, torch.tensor([0.0, 0.0, -1.0], device=env.device)).item()

                stats["total_steps"] += 1
                stats["height_ok_steps"] += int(height_ok)
                stats["both_magnitude_ok_steps"] += int(both_magnitude_ok)
                stats["antipodal_ok_steps"] += int(antipodal_ok)
                stats["grasp_ok_steps"] += int(grasp_ok)
                stats["gate_fires_steps"] += int(gate_fires)
                stats["max_jaw1_force"] = max(stats["max_jaw1_force"], jaw1_force_mag)
                stats["max_jaw2_force"] = max(stats["max_jaw2_force"], jaw2_force_mag)
                stats["max_cube_z"] = max(stats["max_cube_z"], cube_z)
                stats["min_residual_authority"] = min(stats["min_residual_authority"], residual_authority)

                print(
                    f"[EP {episode} STEP {step:3d}] cube_z={cube_z:.4f} height_ok={int(height_ok)} "
                    f"jaw1_force={jaw1_force_mag:.5f} jaw2_force={jaw2_force_mag:.5f} "
                    f"both_mag_ok={int(both_magnitude_ok)} cos_angle={cos_angle:.4f} antipodal_ok={int(antipodal_ok)} "
                    f"grasp_ok={int(grasp_ok)} GATE_FIRES={int(gate_fires)} "
                    f"jaw_joint_pos={gripper_joint_pos[0]:.5f}/{gripper_joint_pos[1]:.5f} "
                    f"orientation_dot={orientation_dot:.4f} residual_authority={residual_authority:.4f}"
                )

                if bool(dones[0]):
                    print(f"[EP {episode} STEP {step}] episode done (early termination), stopping episode")
                    break
            print(f"[EPISODE {episode} END]")

    print("[SUMMARY] " + " ".join(f"{k}={v}" for k, v in stats.items()))
    print("[DIAGNOSTIC COMPLETE]")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
```

- [ ] **Step 6: Run the contact diagnostic**

Run:
```bash
PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/warmresidual_contact_diagnostic.py \
    --checkpoint "$LOG_DIR/model_1499.pt" --episodes 3 2>&1 | tee /tmp/exp23_contact_diagnostic.log
```
Expected: no traceback; a final `[SUMMARY] ...` line reporting `both_magnitude_ok_steps` (compare directly against Experiment 22's exact `0/750`) and `min_residual_authority` (expected `1.0`, confirming the trained checkpoint is being evaluated well past its own warmup window — if this is less than 1.0, the checkpoint was evaluated before enough steps accumulated, which would need investigation before trusting the rest of the diagnostic).

- [ ] **Step 7: Write the report**

Create `docs/superpowers/plans/2026-07-07-ar4-experiment23-report.md`, following the structure of `docs/superpowers/plans/2026-07-07-ar4-experiment22-report.md` (Log Directory, Training Status, Verification Results — checkpoint integrity and `Loss/value_function`, per-term TensorBoard summary, the contact diagnostic's `[SUMMARY]` line, and an Assessment section). The report MUST:

- State `Episode_Reward/lifting_object`'s exact nonzero count out of 1500 from Step 4's extraction, explicitly compared against Experiment 22's exact `0/1500`.
- State `both_magnitude_ok_steps`'s exact count from Step 6's diagnostic, explicitly compared against Experiment 22's exact `0/750`.
- State whether `Loss/value_function` stayed bounded (from Task 5's diagnostic-run check and this task's Step 3/4 full-run check) — the specific risk this repo's Experiment 11/13 history flags for new action terms.
- **If `lifting_object` is still exactly 0/1500 and `both_magnitude_ok_steps` is still exactly 0/750**: state this plainly as a null result. Per this spec's own falsification criterion (confirmed passing in Task 4), this specifically falsifies "the warm-start gap explains Experiment 13's regression and blocks the residual mechanism from working" — write one paragraph assessing what this means for the residual-RL-over-classical-controller paradigm as a whole (a fundamental non-fit for this task, not an implementation-gap problem) and naming demonstration/imitation bootstrapping as the next candidate direction, per the spec's own Success Criteria section.
- **If either metric goes nonzero**: explicitly flag, in its own clearly-marked subsection, that video inspection is now warranted and MUST be done separately, personally, by the controller (not delegated to a subagent) before any success claim is made — per this project's Experiment-16-established lesson (a shaped/gated reward scalar going nonzero is not sufficient evidence of genuine behavior on its own; Experiment 16's own "genuine lift" claim was later found to be the cube wedged against the wrist, not a real grasp). Do NOT write a success/failure verdict in this case — state the numeric result and stop, leaving the verdict for the controller's own video review.

- [ ] **Step 8: Update ROADMAP.md**

Add an Experiment 23 entry to `ROADMAP.md`, following the exact same pattern as Experiments 19-22's entries (bold hypothesis-outcome summary, design spec link, report link, bulleted findings, net assessment, next-direction pointer).

- [ ] **Step 9: Commit and push**

```bash
git add scripts/warmresidual_contact_diagnostic.py docs/superpowers/plans/2026-07-07-ar4-experiment23-report.md ROADMAP.md
git commit -m "Experiment 23 complete: warm-started residual RL full-run report and contact diagnostic"
git push origin main
```
