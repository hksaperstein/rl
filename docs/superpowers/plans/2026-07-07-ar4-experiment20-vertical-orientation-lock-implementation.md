# Experiment 20: Vertical Orientation Lock Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Constrain the AR4 gripper's approach orientation to a fixed, always-vertical (top-down) target via a custom absolute-pose differential-IK action term, removing orientation discovery from the policy's exploration problem entirely, then run this action space through Experiment 18's exact unchanged reward configuration to test whether it moves `Episode_Reward/lifting_object` off its `0/1500` baseline.

**Architecture:** One new file (`tasks/ar4/actions.py`) holds a custom `ActionTerm`/`ActionTermCfg` pair that subclasses Isaac Lab's built-in `DifferentialInverseKinematicsAction`, overriding only `action_dim` (3, position-only) and `process_actions` (builds a 7D absolute pose command every step: policy-controlled position + a hardcoded fixed orientation quaternion). One new env cfg file (`tasks/ar4/pickplace_verticallock_env_cfg.py`) reuses Experiment 18's reward/observation/event/termination/curriculum configs unchanged, swapping in the new action term.

**Tech Stack:** Isaac Lab / Isaac Sim 107.3.26, `DifferentialIKController` (damped least-squares), rsl_rl PPO.

## Global Constraints

- Always invoke Isaac Lab scripts via `/home/saps/IsaacLab/isaaclab.sh -p scripts/<script>.py` from the repo root — never plain `python`.
- Every subagent dispatched to run a real Isaac Sim job must be given the literal blocking poll command verbatim in its dispatch prompt — not told to "poll" in prose.
- Task 4's orientation-lock verification is a hard gate: if the actual end-effector orientation does not stay within 5 degrees of the fixed target throughout a full episode of varied actions, do not proceed to Task 5/6 — report BLOCKED.
- No reward-function changes anywhere in this plan — reuses `tasks/ar4/pickplace_pregrasp_env_cfg.py`'s `RewardsCfg`, `ObservationsCfg`, `EventCfg`, `TerminationsCfg`, `CurriculumCfg` exactly, imported and reused directly (not copy-pasted, to guarantee byte-for-byte equality).
- No change to `tasks/ar4/robot_cfg.py`, `scripts/build_asset.py`, or any other existing env cfg file — purely additive (two new files).
- `assets/` is gitignored — this plan does not touch it at all.
- Standalone Isaac Sim launches in this environment have occasionally taken multiple minutes to initialize, or in a few observed cases crashed silently without a Python traceback (GPU/driver strain from many back-to-back launches, not a code bug) — give scripts their natural runtime before concluding something is wrong, and retry once on a silent crash before reporting BLOCKED.

---

### Task 1: Custom vertical-lock IK action term

**Files:**
- Create: `tasks/ar4/actions.py`

**Interfaces:**
- Produces: `VerticalLockDifferentialIKActionCfg` (an `ActionTermCfg` subclass with one new field, `fixed_orientation: tuple[float, float, float, float]`, quaternion `(w, x, y, z)`) and `VerticalLockDifferentialIKAction` (its `ActionTerm`). Later tasks consume `VerticalLockDifferentialIKActionCfg` by constructing it with `joint_names=ARM_JOINT_NAMES`, `body_name="link_6"`, `body_offset=OffsetCfg(pos=_EE_OFFSET)`, `scale=0.05`, `controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")`, `fixed_orientation=<determined in Task 2>`.

- [ ] **Step 1: Write `tasks/ar4/actions.py`**

```python
# tasks/ar4/actions.py
"""Custom task-space action term for Experiment 20: locks the AR4
gripper's approach orientation to a fixed, always-vertical (top-down)
target every step, exposing only 3D Cartesian position to the policy -
removing orientation discovery from the exploration problem entirely.
See docs/superpowers/specs/2026-07-07-ar4-experiment20-vertical-orientation-lock-design.md.

Subclasses Isaac Lab's built-in DifferentialInverseKinematicsAction to
reuse its jacobian resolution, apply_actions, and reset logic unchanged -
only action_dim and process_actions are overridden. Isolated in its own
file (not tasks/ar4/mdp.py, which holds only reward/observation/event
functions, not ActionTerm/ActionTermCfg classes - a different
responsibility, kept separate).

Import this module only after an Isaac Sim/Isaac Lab AppLauncher has been
created.
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class VerticalLockDifferentialIKAction(DifferentialInverseKinematicsAction):
    """Task-space IK action that re-asserts a fixed end-effector
    orientation every step (not merely leaving it unperturbed), while
    the policy controls only a 3D Cartesian position delta.

    cfg.controller must be configured with command_type="pose",
    use_relative_mode=False - an absolute 7D pose command is required so
    the fixed orientation is actively re-targeted every step, not just
    left alone (which a relative/delta command would do, allowing drift
    to accumulate under contact forces without correction).
    """

    cfg: VerticalLockDifferentialIKActionCfg

    def __init__(self, cfg: VerticalLockDifferentialIKActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._fixed_quat = torch.tensor(cfg.fixed_orientation, device=self.device).repeat(self.num_envs, 1)

    @property
    def action_dim(self) -> int:
        return 3

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        self._processed_actions[:] = self.raw_actions * self._scale
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        desired_pos = ee_pos_curr + self._processed_actions
        command = torch.cat([desired_pos, self._fixed_quat], dim=1)
        self._ik_controller.set_command(command, ee_pos_curr, ee_quat_curr)


@configclass
class VerticalLockDifferentialIKActionCfg(DifferentialInverseKinematicsActionCfg):
    """Adds fixed_orientation to the stock IK action cfg. See
    VerticalLockDifferentialIKAction."""

    class_type: type[ActionTerm] = VerticalLockDifferentialIKAction
    fixed_orientation: tuple[float, float, float, float] = MISSING
    """Quaternion (w, x, y, z) the end-effector orientation is locked to,
    every step, regardless of policy output."""
```

- [ ] **Step 2: Smoke-test the import**

```bash
/home/saps/IsaacLab/isaaclab.sh -p -c "
from isaaclab.app import AppLauncher
app_launcher = AppLauncher({'headless': True})
simulation_app = app_launcher.app
import sys
sys.path.insert(0, '/home/saps/projects/rl')
from tasks.ar4.actions import VerticalLockDifferentialIKAction, VerticalLockDifferentialIKActionCfg
print('IMPORT_OK')
simulation_app.close()
"
```

Expected: `IMPORT_OK` printed, no traceback. (If `isaaclab.sh -p -c` is not a supported invocation in this environment, write this to a temporary `.py` file under the repo's scratchpad convention and run it via `isaaclab.sh -p <file>.py` instead — check `scripts/train.py`'s own `AppLauncher` argparse pattern if a bare dict fails, and copy that exact boilerplate rather than improvising, since ad hoc `AppLauncher` invocations have caused hangs/crashes elsewhere in this repo's history.)

- [ ] **Step 3: Commit**

```bash
git add tasks/ar4/actions.py
git commit -m "Add VerticalLockDifferentialIKAction: fixed-orientation task-space IK action (Experiment 20)"
```

---

### Task 2: Determine the fixed downward quaternion + new env cfg

**Files:**
- Create: `tasks/ar4/pickplace_verticallock_env_cfg.py`

**Interfaces:**
- Consumes: `VerticalLockDifferentialIKActionCfg` (Task 1), `tasks/ar4/pickplace_pregrasp_env_cfg.py`'s `RewardsCfg`, `ObservationsCfg`, `EventCfg`, `TerminationsCfg`, `CurriculumCfg` (imported directly, not copied), `tasks/ar4/pickplace_mirror_env_cfg.py`'s `Ar4PickPlaceMirrorSceneCfg`, `tasks/ar4/pickplace_env_cfg.py`'s `_EE_OFFSET`.
- Produces: `Ar4PickPlaceVerticalLockEnvCfg`, consumed by Task 3's CLI wiring.

- [ ] **Step 1: Determine the fixed downward-facing quaternion empirically**

Write a small standalone script (following `scripts/train.py`'s exact `argparse` + `AppLauncher.add_app_launcher_args` boilerplate, not a bare dict) that loads the AR4 robot alone (reuse `tasks/ar4/robot_cfg.py`'s `AR4_MK5_CFG` directly in a minimal scene, or load any existing env cfg and read the `ee_frame` `FrameTransformer`'s orientation), drives `joint_5` (the wrist pitch joint — verify by reading `tasks/ar4/robot_cfg.py`'s `ARM_JOINT_NAMES` and, if unsure which joint controls pitch, sweep each arm joint individually and observe which one visibly changes the end-effector's Z-axis alignment) to the value that makes the gripper's approach axis point straight down (world -Z), and prints the `ee_frame.data.target_quat_w[:, 0, :]` value at that configuration. Confirm "pointing down" by checking that the gripper's local Z or X axis (whichever this asset's URDF defines as the "forward"/approach direction — check `ar_gripper_macro.xacro`'s gripper mount joint's `<axis>` and `<origin rpy=...>` if uncertain) transformed by this quaternion aligns with world `(0, 0, -1)` within a small tolerance, using `isaaclab.utils.math.quat_apply`.

Record the resulting 4-tuple as `_FIXED_DOWNWARD_QUAT` in the new env cfg file (Step 2) — this is a one-time empirical measurement, not a runtime computation, so it belongs as a module-level constant, not code that re-derives it every environment reset.

- [ ] **Step 2: Write `tasks/ar4/pickplace_verticallock_env_cfg.py`**

```python
# tasks/ar4/pickplace_verticallock_env_cfg.py
"""IK-constrained vertical/top-down approach orientation variant
(Experiment 20): identical reward/observation/event/termination/
curriculum configuration to Experiment 18's Ar4PickPlacePregraspEnvCfg
(tasks/ar4/pickplace_pregrasp_env_cfg.py, imported and reused directly,
not copied), with one changed variable - the arm's action space is
replaced with VerticalLockDifferentialIKActionCfg (tasks/ar4/actions.py),
which locks the end-effector's orientation to a fixed, always-vertical
target every step, leaving only 3D Cartesian position under policy
control. See
docs/superpowers/specs/2026-07-07-ar4-experiment20-vertical-orientation-lock-design.md.

Additive/parallel to every other pickplace_*.py file: does not modify
pickplace_pregrasp_env_cfg.py, pickplace_graspgated_env_cfg.py, or any
other existing env cfg. Reuses Ar4PickPlaceMirrorSceneCfg and
Ar4PickPlaceTaskspacePPORunnerCfg directly (task-space action, same
agent cfg family as Experiment 11's taskspace variant).

Import this module only after an Isaac Sim/Isaac Lab AppLauncher has been
created.
"""

from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.configclass import configclass

from .actions import VerticalLockDifferentialIKActionCfg
from .agents.rsl_rl_ppo_cfg import Ar4PickPlaceTaskspacePPORunnerCfg
from .pickplace_env_cfg import _EE_OFFSET
from .pickplace_mirror_env_cfg import Ar4PickPlaceMirrorSceneCfg
from .pickplace_pregrasp_env_cfg import (
    CurriculumCfg,
    EventCfg,
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
)
from .robot_cfg import ARM_JOINT_NAMES, GRIPPER_CLOSED_POS, GRIPPER_JOINT_NAMES, GRIPPER_OPEN_POS
import isaaclab.envs.mdp as isaaclab_mdp
import isaaclab.sim as sim_utils

# Determined empirically (Task 2, Step 1): the quaternion (w, x, y, z)
# that points the gripper's approach axis straight down (world -Z).
_FIXED_DOWNWARD_QUAT = (<FILL IN FROM STEP 1's MEASURED VALUE>)


@configclass
class ActionsCfg:
    """Task-space action: 3D Cartesian position under policy control,
    orientation locked to _FIXED_DOWNWARD_QUAT every step. Gripper action
    unchanged from every prior experiment."""

    arm_action = VerticalLockDifferentialIKActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINT_NAMES,
        body_name="link_6",
        body_offset=VerticalLockDifferentialIKActionCfg.OffsetCfg(pos=_EE_OFFSET),
        scale=0.05,
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        fixed_orientation=_FIXED_DOWNWARD_QUAT,
    )
    gripper_position = isaaclab_mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=GRIPPER_JOINT_NAMES,
        open_command_expr={name: GRIPPER_OPEN_POS for name in GRIPPER_JOINT_NAMES},
        close_command_expr={name: GRIPPER_CLOSED_POS for name in GRIPPER_JOINT_NAMES},
    )


@configclass
class Ar4PickPlaceVerticalLockEnvCfg(ManagerBasedRLEnvCfg):
    """AR4 vertical-orientation-lock task (Experiment 20): Experiment
    18's exact reward/observation/event/termination/curriculum
    configuration, with the arm's action space replaced by a fixed-
    orientation task-space IK action. num_envs=4096 default -
    scripts/train.py's --num_envs flag overrides this per-run same as
    every other env cfg in this repo."""

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

Replace `<FILL IN FROM STEP 1's MEASURED VALUE>` with the actual 4-tuple measured in Step 1 — this file is not complete until that placeholder is replaced with a real, measured value; do not commit with the placeholder still present.

**Note on `RewardsCfg`'s `reaching_object`/`pregrasp_readiness` params:** these reference `SceneEntityCfg("ee_frame")` internally (already defined in the imported `RewardsCfg`, unchanged) — `Ar4PickPlaceMirrorSceneCfg` already defines the `ee_frame` `FrameTransformer` sensor at the same `_EE_OFFSET`-adjusted pinch point this new action term also targets via `body_offset`, so both the reward's proximity measurement and the action term's controlled point refer to the same physical location — no mismatch to introduce here.

- [ ] **Step 3: Verify the import and construction succeed**

```bash
/home/saps/IsaacLab/isaaclab.sh -p -c "
from isaaclab.app import AppLauncher
app_launcher = AppLauncher({'headless': True})
simulation_app = app_launcher.app
import sys
sys.path.insert(0, '/home/saps/projects/rl')
from tasks.ar4.pickplace_verticallock_env_cfg import Ar4PickPlaceVerticalLockEnvCfg
cfg = Ar4PickPlaceVerticalLockEnvCfg()
print('CFG_OK', cfg.actions.arm_action.fixed_orientation)
simulation_app.close()
"
```

(Same note as Task 1 Step 2: if `-c` inline execution isn't supported, write to a temp file under this repo's scratchpad convention instead.)

- [ ] **Step 4: Commit**

```bash
git add tasks/ar4/pickplace_verticallock_env_cfg.py
git commit -m "Add Ar4PickPlaceVerticalLockEnvCfg: fixed-orientation vertical approach (Experiment 20)"
```

---

### Task 3: Wire `--verticallock` into train.py/eval_loop.py + smoke test

**Files:**
- Modify: `scripts/train.py`
- Modify: `scripts/eval_loop.py`

**Interfaces:**
- Consumes: `Ar4PickPlaceVerticalLockEnvCfg` (Task 2).

- [ ] **Step 1: Add the `--verticallock` flag to `scripts/train.py`**

Read the current exact content of `scripts/train.py` first (it has grown across many prior experiments — the exact line numbers for the flag-parsing block and the `env_cfg_cls`/`agent_cfg` selection chains will have shifted). Add a new `parser.add_argument("--verticallock", ...)` following the exact pattern of the existing `--pregrasp`/`--graspgated` flags (boolean, `action="store_true"`, `default=False`, a help string citing this experiment's design spec path), add the corresponding import (`from tasks.ar4.pickplace_verticallock_env_cfg import Ar4PickPlaceVerticalLockEnvCfg`), and add `elif args_cli.verticallock: env_cfg_cls = Ar4PickPlaceVerticalLockEnvCfg` as a new branch in the `env_cfg_cls` selection chain (checked first, alongside `--pregrasp`/`--graspgated`, before the other branches — order among mutually exclusive flags doesn't functionally matter here since they're each independent booleans, but match this file's existing top-to-bottom ordering convention of newest-experiment-first).

**Critical: this is a task-space IK action, unlike `--pregrasp`/`--graspgated`/`--provenrecipe` (which use plain joint-space action).** Add `args_cli.verticallock` to the `agent_cfg` selection condition, which currently reads (verify the exact current text first):
```python
if args_cli.taskspace or args_cli.residual or args_cli.reachskip or args_cli.baseproximity:
    agent_cfg = Ar4PickPlaceTaskspacePPORunnerCfg()
```
Change to:
```python
if args_cli.taskspace or args_cli.residual or args_cli.reachskip or args_cli.baseproximity or args_cli.verticallock:
    agent_cfg = Ar4PickPlaceTaskspacePPORunnerCfg()
```
This is the exact opposite mistake pattern this repo has repeatedly had to verify against for `--pregrasp`/`--graspgated` (those two correctly stay OUT of this condition since they're joint-space; `--verticallock` must go IN this condition since it's task-space IK). Get this right the first time — grep the file for the current exact condition text before editing, do not guess line numbers.

- [ ] **Step 2: Add the identical flag + branches to `scripts/eval_loop.py`**

Read `scripts/eval_loop.py`'s current exact content and mirror the same two changes (flag definition, `env_cfg_cls` branch, `agent_cfg` selection condition) — this file has historically been kept in lockstep with `train.py`'s flag set.

- [ ] **Step 3: Smoke test**

```bash
/home/saps/IsaacLab/isaaclab.sh -p scripts/train.py --verticallock --num_envs 16 --max_iterations 2 --headless
```

Expected: completes without traceback, writes `model_0.pt`/`model_1.pt` to a new `logs/train/<timestamp>/` directory. Confirm the reward term is present and correctly configured:

```bash
LATEST=$(ls -dt logs/train/*/ | head -1)
grep -A3 "pregrasp_readiness\|arm_action" "${LATEST}params/env.yaml" | head -20
```

Expected: `pregrasp_readiness` weight `2.0` present (confirming the reward config truly matches Experiment 18), `arm_action` shows the `VerticalLockDifferentialIKActionCfg` fields including `fixed_orientation`.

- [ ] **Step 4: Commit**

```bash
git add scripts/train.py scripts/eval_loop.py
git commit -m "Wire --verticallock flag into train.py and eval_loop.py (Experiment 20)"
```

---

### Task 4: Orientation-lock verification (hard gate)

**Files:**
- Create: `scripts/verticallock_verify.py`

**Interfaces:**
- Consumes: `Ar4PickPlaceVerticalLockEnvCfg` (Task 2), the freshly-smoke-tested checkpoint from Task 3 (or a random/untrained policy — this test does not require a trained policy, since it's testing the *mechanism*, not learned behavior).
- Produces: `.superpowers/sdd/task-4-report.md` stating PASS/FAIL. Hard gate — do not proceed to Task 5/6 if FAIL.

- [ ] **Step 1: Write the verification script**

Follow `scripts/train.py`'s exact `argparse` + `AppLauncher.add_app_launcher_args` boilerplate (do not use a bare `SimulationApp({...})` dict — this exact shortcut has caused silent hangs/crashes multiple times in this repo's history). Load `Ar4PickPlaceVerticalLockEnvCfg` with `num_envs=1`, reset, then step the environment for 250 steps feeding **random** 3D actions each step (`torch.rand(1, 3, device=env.device) * 2 - 1`, i.e. uniform in `[-1, 1]`, matching the policy's expected action range) — deliberately exercising varied, aggressive position commands to stress-test whether the orientation lock holds under motion, not just at rest. Each step, read `env.scene["ee_frame"].data.target_quat_w[:, 0, :]` and compute the angular deviation from `_FIXED_DOWNWARD_QUAT` (import it from `tasks.ar4.pickplace_verticallock_env_cfg`) via `isaaclab.utils.math.quat_error_magnitude` (or equivalent — check `isaaclab.utils.math` for the actual available function name and use it; do not hand-roll quaternion angle math when Isaac Lab likely already provides it). Track max and mean deviation in degrees across all 250 steps. Print a `[SUMMARY]`/`[RESULT]` line pair matching this repo's established diagnostic-script convention, with `PASS` if max deviation stays under 5 degrees for the entire rollout, `FAIL` otherwise.

- [ ] **Step 2: Run it**

```bash
/home/saps/IsaacLab/isaaclab.sh -p scripts/verticallock_verify.py 2>&1 | tee /tmp/exp20_orientation_verify_stdout.log
grep -E "^\[SUMMARY\]|^\[RESULT\]" /tmp/exp20_orientation_verify_stdout.log
```

- [ ] **Step 3: Write the gate report**

Write `.superpowers/sdd/task-4-report.md` with the exact `[SUMMARY]`/`[RESULT]` line contents, the max/mean angular deviation, and the PASS/FAIL verdict.

**If FAIL:** stop here, report BLOCKED with the report path — do not proceed to Task 5/6. A training run against a non-functional orientation lock would be uninterpretable.

**If PASS:** commit the script and proceed to Task 5.

```bash
git add scripts/verticallock_verify.py
git commit -m "Add and pass Experiment 20 orientation-lock verification gate"
```

---

### Task 5: 300-iteration diagnostic run

**Files:**
- None (verification-only task, no commit).

- [ ] **Step 1: Launch**

```bash
nohup /home/saps/IsaacLab/isaaclab.sh -p scripts/train.py --verticallock --num_envs 4096 --headless --max_iterations 300 > /tmp/exp20_diagnostic_stdout.log 2>&1 &
echo "PID=$!"
```

- [ ] **Step 2: Poll for completion — BLOCK on this yourself, do not end your turn early**

```bash
until find logs/train/ -name "model_299.pt" -newer tasks/ar4/pickplace_verticallock_env_cfg.py 2>/dev/null | grep -q .; do sleep 60; done
echo "diagnostic run complete"
```

Re-issue this exact blocking command across tool calls if a single call's own timeout is hit before the checkpoint appears.

- [ ] **Step 3: Check the formal gates**

```bash
LATEST=$(ls -dt logs/train/*/ | head -1)
grep -iE "traceback|error|exception" /tmp/exp20_diagnostic_stdout.log
/home/saps/IsaacLab/isaaclab.sh -p -c "
from tensorboard.backend.event_processing import event_accumulator
import glob
path = sorted(glob.glob('${LATEST}events.out.tfevents.*'))[-1]
ea = event_accumulator.EventAccumulator(path)
ea.Reload()
for tag in ['Loss/value_function', 'Episode_Reward/lifting_object', 'Episode_Reward/pregrasp_readiness', 'Episode_Reward/reaching_object']:
    vals = ea.Scalars(tag)
    nonzero = sum(1 for v in vals if v.value != 0.0)
    print(f'{tag}: points={len(vals)} first={vals[0].value} last={vals[-1].value} max={max(v.value for v in vals)} min={min(v.value for v in vals)} nonzero={nonzero}/{len(vals)}')
"
```

Expected: no tracebacks; `Loss/value_function` small and bounded, no sustained growth; `pregrasp_readiness` and `reaching_object` show real nonzero signal (confirming the reward config is genuinely functioning, same as Experiment 18's own diagnostic); `lifting_object`'s nonzero count reported as-is (not necessarily nonzero yet at this scale — report the actual number either way).

If any gate fails: report BLOCKED, do not proceed to Task 6. If clean: proceed to Task 6.

---

### Task 6: Full 1500-iteration run + report

**Files:**
- Create: `docs/superpowers/plans/2026-07-07-ar4-experiment20-report.md`

- [ ] **Step 1: Launch**

```bash
nohup /home/saps/IsaacLab/isaaclab.sh -p scripts/train.py --verticallock --num_envs 4096 --headless > /tmp/exp20_train_stdout.log 2>&1 &
echo "PID=$!"
```

- [ ] **Step 2: Poll for completion — BLOCK on this yourself, do not end your turn early**

```bash
until find logs/train/ -name "model_1499.pt" -newer tasks/ar4/pickplace_verticallock_env_cfg.py 2>/dev/null | grep -q .; do sleep 60; done
echo "full run complete"
```

Expect roughly 15-25 minutes wall-clock (prior 1500-iteration runs at `num_envs=4096` took ~14-20 minutes). Re-issue this exact blocking command across tool calls if a single call's own timeout is hit before the checkpoint appears. Confirm checkpoint integrity (31 checkpoints, `model_1499.pt` present, event file mtime matches completion time) following the exact pattern used in every prior experiment report this session.

- [ ] **Step 3: Extract full scalar trajectories**

Same `EventAccumulator` extraction pattern as every prior experiment report, for tags: `Episode_Reward/reaching_object`, `Episode_Reward/pregrasp_readiness`, `Episode_Reward/lifting_object`, `Episode_Reward/object_goal_tracking`, `Episode_Reward/object_goal_tracking_fine_grained`, `Episode_Termination/cube_reached_goal`, `Loss/value_function`.

- [ ] **Step 4: Write the report**

Write `docs/superpowers/plans/2026-07-07-ar4-experiment20-report.md` following the exact structure of `docs/superpowers/plans/2026-07-07-ar4-experiment18-report.md`. Include a "Orientation Lock Verification Recap" section near the top summarizing Task 4's PASS result. Include a "Key Comparison" section against Experiment 18's exact final `cube_reached_goal` (0.003499) and Experiment 17's (0.002360).

**The report must explicitly and separately answer:**

1. Does `Episode_Reward/pregrasp_readiness` still show real, growing nonzero occurrence (confirming the reward configuration is genuinely unchanged from Experiment 18)?
2. Does `Episode_Reward/lifting_object`'s nonzero count move off exactly `0/1500` — the single specific, falsifiable question this experiment exists to answer. Report the exact nonzero count.

**If `lifting_object` is still exactly `0/1500`:** state this plainly — per this project's established practice, do NOT perform a video-inspection step in this case, the scalar evidence is unambiguous. This would be a clean falsification of the orientation-discovery-bottleneck hypothesis specifically (given Task 4 already confirmed the orientation constraint itself holds).

**If `lifting_object` DOES show any nonzero occurrences:** do NOT draw a final success/failure conclusion from scalars alone — explicitly flag that video inspection is now warranted, to be done separately and personally by the controller (not delegated) before any success claim, per this project's Experiment 16 lesson.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/plans/2026-07-07-ar4-experiment20-report.md
git commit -m "Record Experiment 20 training run: vertical orientation lock scalar trajectories"
```
