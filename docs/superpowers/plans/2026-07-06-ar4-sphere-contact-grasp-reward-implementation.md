# AR4 sphere contact-sensor grasp reward — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the AR4 sphere pick-and-place policy a ground-truth contact-based
grasp reward (via Isaac Lab's `ContactSensor`), replacing the geometric
position/closure proxies that produced four falsified hypotheses (reward
hacking or exploration sparsity — see `ROADMAP.md`'s "grasp/lift never
emerges" entry), and run the full experiment to determine whether this fixes
the arm's inability to grasp and lift the sphere.

**Architecture:** Add a `ContactSensorCfg` covering both gripper jaw links,
filtered to only report contact against the sphere. Add a new, purely
additive `grasp_contact` reward term (`tasks/ar4/mdp.py`) that returns 1.0
only when **both** fingers register real contact force above a calibrated
threshold — a bilateral, physically-grounded signal that can't be satisfied
by closing beside the object (the second experiment's failure mode) and
isn't gated behind a hard-to-discover geometric precision requirement (the
third experiment's failure mode). Calibrate the force threshold against a
real scripted grasp before committing to a full training run. Everything
else in the reward/observation/training setup stays exactly as it is on
`main` — this is a single-variable experiment.

**Tech Stack:** Isaac Lab / Isaac Sim (`ManagerBasedRLEnv`), PyTorch,
`rsl_rl` PPO, existing AR4 mk5 task code in `tasks/ar4/`.

## Global Constraints

- Design spec: `docs/superpowers/specs/2026-07-05-ar4-sphere-contact-sensor-design.md`
  (read it — this plan implements it exactly, including the decided
  parameter values in its "Parameter values" section).
- Always invoke via `/home/saps/IsaacLab/isaaclab.sh -p scripts/<script>.py`
  (or `-p -m pytest ...` / `-p -c "..."`) from the repo root — never plain
  `python` for anything importing `isaaclab`.
- `force_threshold = 0.05` (Newtons) and `grasp_contact` reward `weight =
  20.0` are decided values, not placeholders — use them exactly, and only
  deviate if Task 3's real-data calibration check shows they don't work (in
  which case, change both this file's calibration script constant and the
  `RewardsCfg` param together, and note why in the report).
- Every other reward term (`reaching_sphere`, `lifting_sphere`,
  `sphere_goal_tracking`, `sphere_goal_tracking_fine_grained`,
  `action_rate`, `joint_vel`), every `ObservationsCfg`/`CommandsCfg`/
  `EventCfg` entry, and the training hyperparameters in
  `tasks/ar4/agents/rsl_rl_ppo_cfg.py` must remain byte-for-byte unchanged.
- Verification standard: real evidence over proxies. Don't call a task done
  off exit codes alone — read the actual TensorBoard scalars and look at
  the actual eval video frames.

---

### Task 1: Add the gripper `ContactSensorCfg` to the scene

**Files:**
- Modify: `tasks/ar4/robot_cfg.py:39` (`activate_contact_sensors`)
- Modify: `tasks/ar4/pickplace_env_cfg.py:17` (import), `tasks/ar4/pickplace_env_cfg.py:39-53` (`Ar4PickPlaceSceneCfg`)

**Interfaces:**
- Produces: two scene sensors, `gripper_jaw1_contact` and
  `gripper_jaw2_contact` (`env.scene["gripper_jaw1_contact"]` etc., type
  `ContactSensor`), each reporting `.data.force_matrix_w` with shape
  `(num_envs, 1, 1, 3)` (one body, one filter target), restricted to
  contact against `{ENV_REGEX_NS}/Sphere`. Task 2 consumes these by name.

- [ ] **Step 1: Activate the contact-reporter API on the robot's rigid bodies**

Isaac Lab's `ContactSensor` can only attach to bodies whose spawn config has
opted into the contact-reporter API — without this, the sensor fails at
construction with `RuntimeError: ... could not find any bodies with contact
reporter API`, regardless of whether the prim path pattern is otherwise
correct. In `tasks/ar4/robot_cfg.py`, change:

```python
AR4_MK5_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_resolve_usd_path(),
        activate_contact_sensors=False,
```

to:

```python
AR4_MK5_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_resolve_usd_path(),
        activate_contact_sensors=True,
```

This flips a shared config used by every AR4 script (`grasp_demo.py`,
`interactive_demo.py`, both bare and RL envs) — that's fine: it only
activates the reporter API needed for a `ContactSensor` to attach to the
robot's bodies, it doesn't change physics behavior for anything that isn't
using a `ContactSensor`.

- [ ] **Step 2: Add the `ContactSensorCfg` import**

In `tasks/ar4/pickplace_env_cfg.py`, change:

```python
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
```

to:

```python
from isaaclab.sensors import CameraCfg, ContactSensorCfg, FrameTransformerCfg
```

- [ ] **Step 3: Add the sensor to `Ar4PickPlaceSceneCfg`**

In the same file, change:

```python
@configclass
class Ar4PickPlaceSceneCfg(Ar4SceneCfg):
    """AR4 gripper+objects scene, plus an end-effector FrameTransformer sensor."""

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
```

to:

```python
@configclass
class Ar4PickPlaceSceneCfg(Ar4SceneCfg):
    """AR4 gripper+objects scene, plus an end-effector FrameTransformer sensor
    and a gripper-to-sphere ContactSensor."""

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
    # Ground-truth grasp signal: real contact force between each gripper jaw
    # and the sphere specifically (filter_prim_paths_expr), not just "is
    # anything touching the fingers". Two separate sensors (not one wildcard
    # sensor covering both jaws) because PhysX requires the filter match
    # count to equal the sensor body count - a single sensor matching 2
    # bodies/env can't pair with the sphere's 1-match/env filter. See
    # docs/superpowers/specs/2026-07-05-ar4-sphere-contact-sensor-design.md.
    gripper_jaw1_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/root_joint/gripper_jaw1_link",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Sphere"],
    )
    gripper_jaw2_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/root_joint/gripper_jaw2_link",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Sphere"],
    )
```

- [ ] **Step 4: Smoke test — confirm the prim paths are real**

Run:

```bash
cd /home/saps/projects/rl
/home/saps/IsaacLab/isaaclab.sh -p scripts/train.py --num_envs 16 --max_iterations 2 --headless
```

Expected: exits 0, prints `Training complete. Checkpoints and logs written
to: ...`, no exception mentioning `gripper_jaw_contact`, `gripper_jaw`, or
`does not exist`. If it does raise a prim-path error, the error message
names the actual invalid path — compare it against
`gripper_jaw1_link`/`gripper_jaw2_link` (already confirmed correct against
the AR4 URDF by the grasp-alignment experiment) and correct the pattern in
Step 3, then re-run this smoke test. If it instead raises a filter-count
mismatch error (`Filter pattern ... did not match the correct number of
entries`), that means the two-sensor split above was not applied correctly
— each sensor's `prim_path` must match exactly one body per env.

- [ ] **Step 5: Commit**

```bash
git add tasks/ar4/robot_cfg.py tasks/ar4/pickplace_env_cfg.py
git commit -m "Add gripper-to-sphere ContactSensor to AR4 pick-and-place scene"
```

---

### Task 2: Implement and register the `grasp_contact` reward term

**Files:**
- Create: `tasks/ar4/mdp.py`
- Modify: `tasks/ar4/pickplace_env_cfg.py` (import + `RewardsCfg`)

**Interfaces:**
- Consumes: `env.scene["gripper_jaw1_contact"].data.force_matrix_w` and
  `env.scene["gripper_jaw2_contact"].data.force_matrix_w` (Task 1). Use
  `force_matrix_w`, not `net_forces_w` — `net_forces_w` is not restricted
  by `filter_prim_paths_expr` (it sums *any* contact on the body, per
  Isaac Lab's own `ContactSensorData` docstring), so it would defeat the
  entire point of filtering to the sphere specifically.
- Produces: `contact_grasp_bonus(env, force_threshold, jaw1_contact_cfg,
  jaw2_contact_cfg) -> torch.Tensor` of shape `(num_envs,)`, values in
  `{0.0, 1.0}`. Task 3's calibration script imports this function directly
  by name.

- [ ] **Step 1: Create `tasks/ar4/mdp.py`**

```python
# tasks/ar4/mdp.py
"""Local MDP reward terms for the AR4 pick-and-place task that don't exist
in any of Isaac Lab's built-in `mdp` modules.

Import this module only after an Isaac Sim/Isaac Lab AppLauncher has been
created.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


def contact_grasp_bonus(
    env: ManagerBasedRLEnv,
    force_threshold: float,
    jaw1_contact_cfg: SceneEntityCfg,
    jaw2_contact_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Dense bonus when both gripper fingers register real contact force
    against the sphere specifically - a ground-truth grasp signal
    (ContactSensor, filtered via force_matrix_w), replacing the geometric
    position/closure proxies every prior experiment in this repo's
    grasp-reward history used (see ROADMAP.md's "grasp/lift never emerges"
    entry for why those failed: either reward-hackable via a loose distance
    check, or too sparse to discover via a tight alignment check). Adapted
    from isaaclab_tasks' manipulation/place/agibot task's object_grasped
    pattern (bilateral force-threshold check), using one sensor per jaw and
    the filtered force_matrix_w field rather than the unfiltered
    net_forces_w the reference used - see
    docs/superpowers/specs/2026-07-05-ar4-sphere-contact-sensor-design.md
    for why both corrections were necessary.
    """
    jaw1_sensor: ContactSensor = env.scene[jaw1_contact_cfg.name]
    jaw2_sensor: ContactSensor = env.scene[jaw2_contact_cfg.name]
    # force_matrix_w shape: (num_envs, 1 body, 1 filter, 3) for each sensor.
    jaw1_force = torch.linalg.vector_norm(jaw1_sensor.data.force_matrix_w, dim=-1).view(env.num_envs)
    jaw2_force = torch.linalg.vector_norm(jaw2_sensor.data.force_matrix_w, dim=-1).view(env.num_envs)
    both_fingers_contact = (jaw1_force > force_threshold) & (jaw2_force > force_threshold)
    return both_fingers_contact.float()
```

- [ ] **Step 2: Register the reward term**

In `tasks/ar4/pickplace_env_cfg.py`, add the import (near the top, next to
the existing `isaaclab_tasks` mdp import):

```python
from isaaclab_tasks.manager_based.manipulation.lift import mdp

from . import mdp as ar4_mdp
from .env_cfg import ActionsCfg, Ar4SceneCfg
```

Then in `RewardsCfg`, change:

```python
    lifting_sphere = RewTerm(
        func=mdp.object_is_lifted, params={"minimal_height": 0.03, "object_cfg": SceneEntityCfg("sphere")}, weight=25.0
    )

    sphere_goal_tracking = RewTerm(
```

to:

```python
    lifting_sphere = RewTerm(
        func=mdp.object_is_lifted, params={"minimal_height": 0.03, "object_cfg": SceneEntityCfg("sphere")}, weight=25.0
    )

    grasp_contact = RewTerm(
        func=ar4_mdp.contact_grasp_bonus,
        weight=20.0,
        params={
            "force_threshold": 0.05,
            "jaw1_contact_cfg": SceneEntityCfg("gripper_jaw1_contact"),
            "jaw2_contact_cfg": SceneEntityCfg("gripper_jaw2_contact"),
        },
    )

    sphere_goal_tracking = RewTerm(
```

- [ ] **Step 3: Smoke test — confirm the term runs and is logged**

```bash
cd /home/saps/projects/rl
/home/saps/IsaacLab/isaaclab.sh -p scripts/train.py --num_envs 16 --max_iterations 2 --headless
```

Expected: exits 0, same as Task 1's smoke test (no new errors — this
confirms the reward function itself doesn't crash on real tensors).

- [ ] **Step 4: Confirm the scalar is in TensorBoard**

```bash
cd /home/saps/projects/rl
LATEST=$(ls -dt logs/train/*/ | head -1)
/home/saps/IsaacLab/isaaclab.sh -p -c "
from tensorboard.backend.event_processing import event_accumulator
import glob
path = sorted(glob.glob('${LATEST}events.out.tfevents.*'))[-1]
ea = event_accumulator.EventAccumulator(path)
ea.Reload()
tags = [t for t in ea.Tags()['scalars'] if 'grasp_contact' in t]
print('grasp_contact tags found:', tags)
assert tags, 'grasp_contact reward scalar missing from TensorBoard logs'
"
```

Expected: prints a non-empty list containing something like
`Episode_Reward/grasp_contact`, no assertion error.

- [ ] **Step 5: Commit**

```bash
git add tasks/ar4/mdp.py tasks/ar4/pickplace_env_cfg.py
git commit -m "Add contact-sensor-based grasp_contact reward term for AR4 sphere task"
```

---

### Task 3: Calibrate `force_threshold` against a real scripted grasp

**Files:**
- Create: `scripts/calibrate_gripper_contact.py`
- Modify: `tasks/ar4/pickplace_env_cfg.py:36` (`_EE_OFFSET`) — a real,
  measured bug fix found while validating this task's script (0.09 → the
  measured 0.036), not a new hypothesis. See the design spec's "Major
  finding" section. This one-line numeric correction affects
  `reaching_sphere`'s proximity target for every future run, which is why
  it was flagged to the user rather than folded in silently.

**Interfaces:**
- Consumes: `contact_grasp_bonus` (Task 2), `gripper_jaw1_contact`/
  `gripper_jaw2_contact` sensors (Task 1), `Ar4PickPlaceEnvCfg`
  (`tasks/ar4/pickplace_env_cfg.py`).
- Produces: a printed calibration report (no code interface — this is a
  one-off verification script, not imported elsewhere).

**Revised approach (the plan's original design — reusing `grasp_demo.py`'s
IK waypoints — was tried for real and found broken; see
`docs/superpowers/specs/2026-07-05-ar4-sphere-contact-sensor-design.md`'s
"Calibration method correction" section for the full diagnosis):** skip
IK/reach entirely. Hold the arm motionless at its default zero pose, read
the gripper's real jaw pinch-point position from the `ee_frame` sensor, and
teleport the sphere directly there via `RigidObject.write_root_pose_to_sim`
— a pattern already used in this repo
(`scripts/perception_calibration.py:74`,
`scripts/measure_planarity_residual.py:85`). This gets one genuine
jaws-around-sphere contact event without depending on IK reach accuracy,
which is a separate, unrelated question this experiment doesn't need to
answer.

- [ ] **Step 1: Create `scripts/calibrate_gripper_contact.py`**

```python
"""Calibrate the AR4 gripper's ContactSensor-based grasp reward
(tasks/ar4/mdp.py's contact_grasp_bonus) against a real closed grasp,
before spending a full training run on an untested force_threshold.

Holds the arm motionless at its default (all-zero-joint) pose and
teleports the sphere directly to the gripper's real jaw pinch-point
position (read live from the ee_frame sensor), then closes the gripper on
it - avoiding any dependency on scripts/grasp_demo.py's IK reach waypoints,
which a real run showed do not reliably bring the end-effector near the
object in this scene (see
docs/superpowers/specs/2026-07-05-ar4-sphere-contact-sensor-design.md's
"Calibration method correction" section). Teleport pattern precedented in
scripts/perception_calibration.py and scripts/measure_planarity_residual.py.

.. code-block:: bash

    /home/saps/IsaacLab/isaaclab.sh -p scripts/calibrate_gripper_contact.py --headless
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Calibrate the AR4 gripper ContactSensor against a real sphere grasp.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tasks.ar4.mdp import contact_grasp_bonus  # noqa: E402
from tasks.ar4.pickplace_env_cfg import Ar4PickPlaceEnvCfg  # noqa: E402
from tasks.ar4.robot_cfg import ARM_JOINT_NAMES  # noqa: E402

GRIPPER_OPEN = 1.0
GRIPPER_CLOSE = -1.0

# (duration_steps, gripper_command, label)
PHASES = [
    (150, GRIPPER_OPEN, "settle"),  # let the arm's real pose stabilize before measuring anything
    (60, GRIPPER_OPEN, "open"),
    (60, GRIPPER_CLOSE, "close"),
    (120, GRIPPER_CLOSE, "hold"),
]

# Must match tasks/ar4/pickplace_env_cfg.py's grasp_contact RewTerm params.
FORCE_THRESHOLD = 0.05


def main() -> None:
    env_cfg = Ar4PickPlaceEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device
    total_steps = sum(duration for duration, _, _ in PHASES)
    step_dt = env_cfg.decimation * env_cfg.sim.dt
    env_cfg.episode_length_s = total_steps * step_dt + 2.0

    env = ManagerBasedRLEnv(cfg=env_cfg)
    num_joints = len(ARM_JOINT_NAMES)
    jaw1_cfg = SceneEntityCfg("gripper_jaw1_contact")
    jaw2_cfg = SceneEntityCfg("gripper_jaw2_contact")

    open_forces: list[list[float]] = []
    hold_forces: list[list[float]] = []
    hold_rewards: list[float] = []

    identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)

    with torch.inference_mode():
        env.reset()
        ee_frame = env.scene["ee_frame"]

        for duration, gripper_cmd, label in PHASES:
            for i in range(duration):
                # Re-read the gripper's real jaw pinch-point position every
                # step (not once, frozen) - the arm's commanded pose takes
                # many steps to settle under this robot's actuator gains, so
                # a stale reading leaves the sphere in empty space once the
                # real arm moves away from where it started.
                pinch_point = ee_frame.data.target_pos_w[0, 0].clone()
                sphere_pose = torch.cat([pinch_point, identity_quat]).unsqueeze(0)
                env.scene["sphere"].write_root_pose_to_sim(sphere_pose)

                action = torch.zeros(env.num_envs, num_joints + 1, device=env.device)
                action[:, num_joints] = gripper_cmd
                env.step(action)

                jaw1_sensor = env.scene["gripper_jaw1_contact"]
                jaw2_sensor = env.scene["gripper_jaw2_contact"]
                jaw1_norm = torch.linalg.vector_norm(jaw1_sensor.data.force_matrix_w, dim=-1).view(1)
                jaw2_norm = torch.linalg.vector_norm(jaw2_sensor.data.force_matrix_w, dim=-1).view(1)
                force_norm = [jaw1_norm.item(), jaw2_norm.item()]
                reward = contact_grasp_bonus(env, FORCE_THRESHOLD, jaw1_cfg, jaw2_cfg)

                if label == "hold":
                    hold_forces.append(force_norm)
                    hold_rewards.append(reward.item())
                elif label == "open":
                    open_forces.append(force_norm)

            print(f"[phase done] {label}: last force_norm={force_norm}, reward={reward.item()}, pinch_point={pinch_point.tolist()}")

    print("\n=== Calibration summary ===")
    open_min = min(min(f) for f in open_forces)
    open_max = max(max(f) for f in open_forces)
    hold_min = min(min(f) for f in hold_forces)
    hold_max = max(max(f) for f in hold_forces)
    hold_success = sum(r == 1.0 for r in hold_rewards)
    print(f"open (gripper open, sphere at pinch point) force_norm: min={open_min:.4f}, max={open_max:.4f} N (expect ~0.0)")
    print(f"hold (gripper closed on sphere)             force_norm: min={hold_min:.4f}, max={hold_max:.4f} N")
    print(
        f"hold reward==1.0 fraction: {hold_success}/{len(hold_rewards)} "
        f"(force_threshold={FORCE_THRESHOLD})"
    )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
```

- [ ] **Step 2: Run the calibration script**

```bash
cd /home/saps/projects/rl
/home/saps/IsaacLab/isaaclab.sh -p scripts/calibrate_gripper_contact.py --headless
```

- [ ] **Step 3: Evaluate the calibration output**

Read the printed "Calibration summary". Three possible outcomes:

- **If `hold reward==1.0 fraction` is at or near `120/120`** (every step of
  the hold phase reads as a successful grasp) **and** `open` force_norm max
  is ~0.0: `force_threshold=0.05` is well-calibrated. Proceed to Task 4
  unchanged.
- **If the hold fraction is well below 120/120 but nonzero, or `open` max
  force is non-trivially above 0** (a real force signal exists but the
  threshold is miscalibrated relative to it): adjust `FORCE_THRESHOLD` in
  this script and the matching `force_threshold` param in
  `tasks/ar4/pickplace_env_cfg.py`'s `grasp_contact` `RewTerm` to a value
  between the observed `open` max and `hold` min (e.g., their midpoint),
  re-run this script, and repeat until the hold fraction is at or near
  120/120. Note the final chosen value and why in the Task 3 report — this
  is real-data-driven recalibration, not a guess, so document the numbers
  that motivated it.
- **If `hold` force reads exactly `0.0` throughout (not "below threshold,"
  genuinely zero, as happened on the first real run of the previous
  IK-reach-based script)**: this means the sphere and gripper jaws never
  actually made contact at all — not a threshold problem. Do not pick a
  threshold from this data (there is nothing real to calibrate against).
  Report BLOCKED with the full "Calibration summary" output and the
  `pinch_point` value printed/logged, so the controller can investigate
  the geometry directly (e.g. whether `_EE_OFFSET` or the sphere's radius
  puts it just outside the jaws' closed span) rather than guessing at a
  further script change.

- [ ] **Step 4: Commit**

```bash
git add scripts/calibrate_gripper_contact.py tasks/ar4/pickplace_env_cfg.py
git commit -m "Add ContactSensor calibration script for AR4 sphere grasp reward"
```

(If Step 3 required no threshold change, `tasks/ar4/pickplace_env_cfg.py`
will show no diff here — that's fine, just commit the new script.)

---

### Task 4: Full training run

**Files:** none (no code changes — this task runs the training loop and
inspects its output).

**Interfaces:**
- Consumes: the calibrated env/reward config from Tasks 1-3.
- Produces: `logs/train/<timestamp>/model_1500.pt` and TensorBoard event
  logs, consumed by Task 5.

- [ ] **Step 1: Run the full 1500-iteration training run**

```bash
cd /home/saps/projects/rl
/home/saps/IsaacLab/isaaclab.sh -p scripts/train.py --num_envs 4096 --headless
```

Expected wall-clock time: roughly 15-20 minutes at `num_envs=4096`, based on
this repo's prior full runs at the same scale (not the ~2.7-hour
camera-observed path — this experiment uses privileged state, unaffected by
that separate, still-paused perception experiment).

Note the resulting log directory (`logs/train/<timestamp>/`) for Task 5.

- [ ] **Step 2: Pull the key TensorBoard scalars**

```bash
cd /home/saps/projects/rl
LATEST=$(ls -dt logs/train/*/ | head -1)
/home/saps/IsaacLab/isaaclab.sh -p -c "
from tensorboard.backend.event_processing import event_accumulator
import glob
path = sorted(glob.glob('${LATEST}events.out.tfevents.*'))[-1]
ea = event_accumulator.EventAccumulator(path)
ea.Reload()
for tag in ['Episode_Reward/grasp_contact', 'Episode_Reward/lifting_sphere',
            'Episode_Reward/reaching_sphere', 'Episode_Termination/sphere_reached_goal']:
    if tag in ea.Tags()['scalars']:
        vals = ea.Scalars(tag)
        print(tag, '-> first:', vals[0].value, 'last:', vals[-1].value, 'max:', max(v.value for v in vals))
    else:
        print(tag, '-> NOT FOUND')
"
```

Record these four lines in the Task 4 report verbatim — this is the
quantitative half of the verification (Task 5's eval video is the
qualitative half).

- [ ] **Step 3: Note in the report (no commit — no files changed)**

Write `docs/superpowers/plans/2026-07-06-ar4-sphere-contact-grasp-reward-report.md`
(create it now if it doesn't exist; later tasks append to it) with a
"Task 4" section containing: the log directory path, the four scalar lines
from Step 2, and one sentence on whether `grasp_contact` moved meaningfully
off 0 (a necessary but not sufficient condition — Task 5 confirms whether
it corresponds to a real grasp or another form of reward hacking).

```bash
git add docs/superpowers/plans/2026-07-06-ar4-sphere-contact-grasp-reward-report.md
git commit -m "Record AR4 sphere contact-grasp full training run results"
```

---

### Task 5: Real eval + video inspection (decision gate)

**Files:** none (no code changes — this task runs eval and visually inspects
output).

**Interfaces:**
- Consumes: `logs/train/<timestamp>/model_1500.pt` from Task 4.
- Produces: eval videos in `logs/videos/`, consumed by this task's own
  frame-extraction step, and a final pass/fail verdict consumed by Task 6.

- [ ] **Step 1: Run eval for 10 episodes**

```bash
cd /home/saps/projects/rl
/home/saps/IsaacLab/isaaclab.sh -p scripts/eval_loop.py --checkpoint logs/train/<RUN_DIR>/model_1500.pt --episodes 10
```

(substitute the actual `<RUN_DIR>` from Task 4). Expected: 10 files
`logs/videos/ar4_pickplace-step-0.mp4` through `-step-2250.mp4`.

- [ ] **Step 2: Extract frames from every episode video**

```bash
cd /home/saps/projects/rl
mkdir -p logs/videos/frames
for f in logs/videos/ar4_pickplace-step-*.mp4; do
  name=$(basename "$f" .mp4)
  mkdir -p "logs/videos/frames/$name"
  ffmpeg -y -i "$f" -vf fps=10 "logs/videos/frames/$name/frame_%03d.png"
done
```

This produces ~50 frames per episode (5s episode @ fps=10), dense enough to
see the full reach -> grasp -> lift -> carry -> place arc, matching this
repo's established "densely-sampled frames across a full episode" practice.

- [ ] **Step 3: Visually inspect all 10 episodes**

Use the Read tool to view a representative sample of frames from each of
the 10 episode directories (start, ~25%, ~50%, ~75%, end of each episode is
enough — no need to view all ~50 frames per episode). For each episode,
determine: does the gripper close **around** the sphere (not beside it),
does the sphere visibly leave the ground, and does it travel toward the
target region?

- [ ] **Step 4: Apply the decision gate**

- **If 8/10 or more episodes show a real grasp+lift+carry:** this is a
  successful fix. Proceed to Task 6's success path.
- **If fewer than 8/10 do:** this is either a fifth falsified hypothesis (if
  `grasp_contact` in Task 4's scalars saturated near its max but
  `lifting_sphere` still didn't move — the same "gripper closes near/beside
  it but no lift" signature as before, now with a ground-truth grasp
  signal instead of a proxy) or a partial/ambiguous result worth describing
  precisely rather than forcing into either bucket. Proceed to Task 6's
  non-success path — **do not** attempt a further reward tweak
  unilaterally; the design doc explicitly calls for flagging this back to
  the Principal/user at this point.

- [ ] **Step 5: Commit the report update**

Append a "Task 5" section to
`docs/superpowers/plans/2026-07-06-ar4-sphere-contact-grasp-reward-report.md`
with: the episode-by-episode observations (or a summary if patterns repeat
across most episodes), the X/10 count, and which branch of Step 4 applies.

```bash
git add docs/superpowers/plans/2026-07-06-ar4-sphere-contact-grasp-reward-report.md
git commit -m "Record AR4 sphere contact-grasp eval video inspection results"
```

(Frame PNGs and mp4s under `logs/` are not committed — check
`.gitignore` covers `logs/`; if it doesn't already, that's a pre-existing
condition outside this plan's scope, not something to fix here.)

---

### Task 6: Record the outcome in `ROADMAP.md`

**Files:**
- Modify: `ROADMAP.md`

**Interfaces:** none (documentation-only task, terminal in this plan).

- [ ] **Step 1: Write the outcome into `ROADMAP.md`'s "grasp/lift never
  emerges" follow-up**

Open `ROADMAP.md` and add a new bullet under the existing "grasp/lift never
emerges" entry (after the gripper PD-gain rescale and paused camera-training
sub-bullets), following the exact same level of evidentiary detail as the
existing sub-bullets (quantitative TensorBoard numbers from Task 4 +
qualitative video findings from Task 5 + root-cause reasoning). Use
whichever of the two templates below applies, filling in the actual
measured numbers and observations from Tasks 4-5 (not placeholder text):

**If Task 5's decision gate passed (8/10+):**

```markdown
   - **Follow-up experiment: ContactSensor-based grasp reward (SUCCESS).**
     Per the systematic-debugging Phase 4.5 escalation from four falsified
     reward/control-only hypotheses, added a ground-truth contact-force
     signal (`ContactSensorCfg` on both gripper jaw links, filtered to the
     sphere) instead of a geometric distance/closure proxy, per
     `docs/superpowers/specs/2026-07-05-ar4-sphere-contact-sensor-design.md`.
     `grasp_contact` reward: [Task 4's first/last/max values]. Full run
     data: `docs/superpowers/plans/2026-07-06-ar4-sphere-contact-grasp-reward-report.md`.
     **Result: [X]/10 real eval episodes show the gripper closing around
     (not beside) the sphere, lifting it off the ground, and carrying it
     toward the target.** `lifting_sphere` reward: [Task 4's values].
     This resolves the "grasp/lift never emerges" follow-up.
```

**If Task 5's decision gate did not pass (<8/10):**

```markdown
   - **Follow-up experiment: ContactSensor-based grasp reward (falsified/
     inconclusive — [pick: "fifth falsified hypothesis" if grasp_contact
     saturated without a real lift, matching the second experiment's
     reward-hacking-adjacent pattern but with a ground-truth signal this
     time; or a precise description of whatever else the video actually
     showed]).** Per the systematic-debugging Phase 4.5 escalation from
     four falsified reward/control-only hypotheses, added a ground-truth
     contact-force signal (`ContactSensorCfg` on both gripper jaw links,
     filtered to the sphere) instead of a geometric distance/closure proxy,
     per `docs/superpowers/specs/2026-07-05-ar4-sphere-contact-sensor-design.md`.
     `grasp_contact` reward: [Task 4's first/last/max values]. Full run
     data: `docs/superpowers/plans/2026-07-06-ar4-sphere-contact-grasp-reward-report.md`.
     **Result: [X]/10 real eval episodes show a real grasp+lift** — [one to
     two sentences on what the video actually showed instead, matching the
     specificity of the prior four sub-bullets' failure-signature
     descriptions]. This is the fifth data point on the reward/control axis
     for this failure mode. Per the design doc's own instruction, this is
     flagged back to the Principal/user rather than attempting a sixth
     reward tweak unilaterally — the remaining candidates are the paused
     camera/single-object experiment, a true curriculum with staged reward
     phases, or reconsidering the object/gripper's physical scale
     (friction, mass, or gripper aperture) rather than the reward function.
```

- [ ] **Step 2: Commit**

```bash
git add ROADMAP.md
git commit -m "Record AR4 sphere ContactSensor grasp-reward experiment outcome in ROADMAP"
```

---

## Plan self-review notes

- **Spec coverage:** Task 1 covers the design's "Contact sensor" section,
  Task 2 covers its "Reward term" section (including the exact weight/
  params), Task 3 covers the design's own "verify... don't assume" +
  parameter-values calibration requirement, Tasks 4-5 cover the "Verification
  plan" section (smoke test is folded into Tasks 1-2; full run + eval +
  video inspection are Tasks 4-5), Task 6 covers the "if this also fails...
  flag back to the Principal" instruction.
- **Single-variable discipline:** confirmed no task touches
  `reaching_sphere`, `lifting_sphere`, `sphere_goal_tracking*`,
  `CommandsCfg`, `EventCfg`'s sphere jitter (Task 3's calibration script
  only disables jitter inside its own standalone script's env_cfg instance,
  never touching the committed `pickplace_env_cfg.py` jitter values used by
  real training), or any PPO hyperparameter.
- **Type/name consistency:** `contact_grasp_bonus(env, force_threshold,
  jaw1_contact_cfg, jaw2_contact_cfg)` signature is identical across its
  Task 2 definition, Task 2's `RewardsCfg` registration, and Task 3's
  calibration script's direct call. Sensor names `gripper_jaw1_contact`/
  `gripper_jaw2_contact` are identical across Task 1's scene field names,
  Task 2's `SceneEntityCfg(...)` params, and Task 3's `env.scene[...]`
  lookups — and both sensors read `force_matrix_w`, not `net_forces_w`,
  consistently everywhere.
- **Revision note:** this plan was corrected twice during Task 1's real
  execution (not on paper review) — `activate_contact_sensors` needed to be
  flipped to `True` in `robot_cfg.py`, and the original single wildcard
  sensor + `net_forces_w` design was replaced with two per-jaw sensors +
  `force_matrix_w` after real PhysX errors surfaced both issues. The design
  spec (`docs/superpowers/specs/2026-07-05-ar4-sphere-contact-sensor-design.md`)
  documents both corrections with the actual error text.
