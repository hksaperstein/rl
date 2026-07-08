# Experiment 24 Gate 1: Scripted-Oracle Viability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a pure, non-learned scripted controller that follows this repo's existing 5-waypoint geometric path via live differential IK, with a hand-coded gripper-close schedule, and determine whether it can reliably grasp+lift the cube — establishing (or ruling out) a valid demonstration source for Experiment 24's Gate 2 (BC pretrain + warm-started RL finetune), without any human teleoperation.

**Architecture:** A single standalone script (`scripts/oracle_rollout.py`) reuses the existing, unmodified `Ar4PickPlaceMirrorEnvCfg` (its scene, observation space, and joint-space action space are all already proven and require no new env cfg file). Each step, the script computes a live differential-IK joint target toward the currently-active waypoint (reusing `ik_guided_path_bonus`'s exact IK-controller setup), inverts that target through the existing `JointPositionActionCfg`'s known `scale`/`offset` formula to produce a raw action, and drives the gripper via a hand-coded open/closed schedule tied to waypoint index — then calls `env.step()` normally, so the recorded trajectories are in the exact action/observation space Gate 2's RL policy will later use.

**Tech Stack:** Isaac Lab / Isaac Sim, PyTorch, this repo's existing `tasks/ar4/mdp.py` waypoint/IK/contact-sensor utilities.

## Global Constraints

- Always invoke Isaac Lab scripts via `/home/saps/IsaacLab/isaaclab.sh -p scripts/<script>.py` from repo root, never plain `python`.
- **This is a hard gate.** If fewer than 30/50 episodes pass the grasp+lift gate, OR video review of sampled "successful" episodes reveals false-positive grasps (wedging, not genuine antipodal contact), report FAIL plainly. Do NOT proceed to writing Gate 2's implementation plan — that decision belongs to the controller, not this plan.
- No modification to any existing reward function, env cfg, `compute_path_waypoints`, or existing action term. Purely additive: one new script, one new report, one `.gitignore` entry.
- Every subagent dispatched to run a real Isaac Sim job must be given the literal blocking poll command verbatim.
- Commit after each task; push to `origin/main` after Task 5.
- Exact thresholds (from the design spec, not to be altered): `force_threshold=0.05`, `antipodal_cos_threshold=-0.7071`, `minimal_height=0.03`, `advance_tolerance=0.03`, waypoint params `lift_minimal_height=0.03`, `pregrasp_hover=0.05`, `lift_margin=0.02`, `carry_height=0.10`.

---

### Task 1: Oracle script core — waypoint-following IK controller + hand-coded gripper schedule

**Files:**
- Create: `scripts/oracle_rollout.py`

**Interfaces:**
- Consumes: `Ar4PickPlaceMirrorEnvCfg` (existing, `tasks/ar4/pickplace_mirror_env_cfg.py`, UNMODIFIED — its `ActionsCfg` is `JointPositionActionCfg(joint_names=ARM_JOINT_NAMES, scale=0.5)` + `BinaryJointPositionActionCfg(joint_names=GRIPPER_JOINT_NAMES, open_command_expr={name: GRIPPER_OPEN_POS}, close_command_expr={name: GRIPPER_CLOSED_POS})`; its `ObservationsCfg.policy` concatenates `joint_pos_rel`, `joint_vel_rel`, `cube_position` (object_position_in_robot_root_frame), `target_object_position` (mirrored_target_position_in_robot_root_frame), `actions` (last_action)); `compute_path_waypoints` (existing, `tasks/ar4/mdp.py:404`, NOT currently registered as an event in `Ar4PickPlaceMirrorEnvCfg` — this script registers it itself by calling it directly after each `env.reset()`, not via EventCfg, since env cfg is not being modified); `DifferentialIKController`/`DifferentialIKControllerCfg` (`isaaclab.controllers`); `ARM_JOINT_NAMES`, `GRIPPER_JOINT_NAMES`, `GRIPPER_OPEN_POS=0.014`, `GRIPPER_CLOSED_POS=0.0` (`tasks/ar4/robot_cfg.py`).
- Produces: a runnable per-step rollout loop with `env._path_waypoint_idx` advancing 0→4 over an episode. Later tasks (2, 3) extend this same script's step loop.

- [ ] **Step 1: Write the script's setup and IK-target computation**

Create `scripts/oracle_rollout.py`:

```python
"""Experiment 24 Gate 1: pure, non-learned scripted oracle that follows
this repo's existing 5-waypoint geometric path (compute_path_waypoints,
tasks/ar4/mdp.py) via live differential IK, with a hand-coded
gripper-close schedule tied to waypoint index. No RL, no policy
network - establishes (or rules out) a demonstration source for
Experiment 24 Gate 2. See
docs/superpowers/specs/2026-07-08-ar4-experiment24-scripted-oracle-demonstration-bootstrap-design.md.

Reuses Ar4PickPlaceMirrorEnvCfg UNMODIFIED (tasks/ar4/pickplace_mirror_env_cfg.py) -
its existing JointPositionActionCfg (scale=0.5, offset=default_joint_pos)
action space means recorded (observation, raw_action) trajectories are
already in the format Gate 2's RL policy will use, with no format
conversion needed. compute_path_waypoints is called directly after each
env.reset() (not registered via EventCfg, since this script does not
modify the env cfg) to populate env._path_waypoints_w/env._path_waypoint_idx.

.. code-block:: bash

    PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/oracle_rollout.py --episodes 50
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Experiment 24 Gate 1: scripted-oracle waypoint-following rollout.")
parser.add_argument("--episodes", type=int, default=50, help="Total number of episodes to run across all envs.")
parser.add_argument("--num_envs", type=int, default=10, help="Number of parallel envs.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch  # noqa: E402

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.utils.math import quat_apply, subtract_frame_transforms  # noqa: E402

sys.path.insert(0, "/home/saps/projects/rl")  # noqa: E402

from tasks.ar4 import mdp as ar4_mdp  # noqa: E402
from tasks.ar4.pickplace_env_cfg import _EE_OFFSET  # noqa: E402
from tasks.ar4.pickplace_mirror_env_cfg import Ar4PickPlaceMirrorEnvCfg  # noqa: E402
from tasks.ar4.robot_cfg import ARM_JOINT_NAMES, GRIPPER_JOINT_NAMES  # noqa: E402

ADVANCE_TOLERANCE = 0.03
LIFT_MINIMAL_HEIGHT = 0.03
PREGRASP_HOVER = 0.05
LIFT_MARGIN = 0.02
CARRY_HEIGHT = 0.10
ARM_SCALE = 0.5  # matches Ar4PickPlaceMirrorEnvCfg's ActionsCfg.joint_positions scale


def compute_ik_arm_raw_action(
    env: ManagerBasedRLEnv,
    ik_controller: DifferentialIKController,
    robot_entity_cfg: SceneEntityCfg,
    ik_jacobi_idx: int,
    arm_default_joint_pos: torch.Tensor,
) -> torch.Tensor:
    """Live differential-IK joint target toward the currently-active
    waypoint (env._path_waypoints_w[:, env._path_waypoint_idx]), with the
    pinch-point offset correction - exact reuse of ik_guided_path_bonus's
    own IK setup (tasks/ar4/mdp.py) - then inverted through
    JointPositionActionCfg's known processed_action = raw*scale+offset
    formula (offset=default_joint_pos, scale=ARM_SCALE) to produce the
    raw action env.step() expects."""
    robot = env.scene["robot"]
    ee_frame = env.scene["ee_frame"]

    current_waypoint_w = torch.gather(
        env._path_waypoints_w, 1, env._path_waypoint_idx.view(-1, 1, 1).expand(-1, 1, 3)
    ).squeeze(1)

    jacobian = robot.root_physx_view.get_jacobians()[:, ik_jacobi_idx, :, robot_entity_cfg.joint_ids]
    root_pose_w = robot.data.root_pose_w
    ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )
    joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]

    offset_vec = torch.tensor(_EE_OFFSET, device=env.device).expand(env.num_envs, 3)
    offset_w = quat_apply(ee_pose_w[:, 3:7], offset_vec)
    ik_target_w = current_waypoint_w - offset_w
    waypoint_command_b, _ = subtract_frame_transforms(root_pose_w[:, 0:3], root_pose_w[:, 3:7], ik_target_w)

    ik_controller.set_command(waypoint_command_b, ee_pos=ee_pos_b, ee_quat=ee_quat_b)
    joint_pos_des = ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

    return (joint_pos_des - arm_default_joint_pos) / ARM_SCALE


def gripper_raw_action_for_waypoint(waypoint_idx: torch.Tensor) -> torch.Tensor:
    """+1.0 (open) at waypoint 0 (pregrasp) and 4 (place); -1.0 (closed)
    at waypoints 1-3 (grasp/lift/transit). Matches BinaryJointPositionAction's
    own raw-action convention (action < 0 -> close, action >= 0 -> open),
    isaaclab/envs/mdp/actions/binary_joint_actions.py."""
    closed = (waypoint_idx >= 1) & (waypoint_idx <= 3)
    return torch.where(closed, torch.full_like(waypoint_idx, -1.0, dtype=torch.float32), torch.ones_like(waypoint_idx, dtype=torch.float32))


def advance_waypoint_idx(env: ManagerBasedRLEnv) -> None:
    ee_frame = env.scene["ee_frame"]
    current_waypoint_w = torch.gather(
        env._path_waypoints_w, 1, env._path_waypoint_idx.view(-1, 1, 1).expand(-1, 1, 3)
    ).squeeze(1)
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    dist_to_waypoint = torch.norm(ee_pos_w - current_waypoint_w, dim=-1)
    reached = dist_to_waypoint < ADVANCE_TOLERANCE
    env._path_waypoint_idx = torch.where(
        reached & (env._path_waypoint_idx < 4), env._path_waypoint_idx + 1, env._path_waypoint_idx
    )


def reset_waypoints_for_envs(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Calls compute_path_waypoints directly (not via EventCfg, since
    this script does not modify the env cfg) for the given env_ids,
    immediately after those envs have been reset."""
    ar4_mdp.compute_path_waypoints(
        env,
        env_ids,
        object_cfg=SceneEntityCfg("cube"),
        lift_minimal_height=LIFT_MINIMAL_HEIGHT,
        pregrasp_hover=PREGRASP_HOVER,
        lift_margin=LIFT_MARGIN,
        carry_height=CARRY_HEIGHT,
    )


def main() -> None:
    env_cfg = Ar4PickPlaceMirrorEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode=None)
    robot = env.scene["robot"]

    ik_cfg = DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls")
    ik_controller = DifferentialIKController(ik_cfg, num_envs=env.num_envs, device=env.device)
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES, body_names=["link_6"])
    robot_entity_cfg.resolve(env.scene)
    ik_jacobi_idx = robot_entity_cfg.body_ids[0] - 1 if robot.is_fixed_base else robot_entity_cfg.body_ids[0]

    arm_joint_ids, _ = robot.find_joints(ARM_JOINT_NAMES)
    gripper_joint_ids, _ = robot.find_joints(GRIPPER_JOINT_NAMES)
    arm_default_joint_pos = robot.data.default_joint_pos[:, arm_joint_ids]

    obs, _ = env.reset()
    all_env_ids = torch.arange(env.num_envs, device=env.device)
    reset_waypoints_for_envs(env, all_env_ids)

    print(f"[SETUP] num_envs={env.num_envs} arm_joint_ids={arm_joint_ids} gripper_joint_ids={gripper_joint_ids}")

    with torch.inference_mode():
        for step in range(250):
            arm_raw = compute_ik_arm_raw_action(env, ik_controller, robot_entity_cfg, ik_jacobi_idx, arm_default_joint_pos)
            gripper_raw = gripper_raw_action_for_waypoint(env._path_waypoint_idx).unsqueeze(-1).expand(-1, len(gripper_joint_ids))
            actions = torch.cat([arm_raw, gripper_raw], dim=-1)

            obs, _, terminated, truncated, _ = env.step(actions)
            advance_waypoint_idx(env)

            done = terminated | truncated
            done_ids = done.nonzero(as_tuple=False).squeeze(-1)
            if len(done_ids) > 0:
                reset_waypoints_for_envs(env, done_ids)

            if step % 50 == 0:
                print(f"[STEP {step:3d}] waypoint_idx={env._path_waypoint_idx.tolist()}")

    print("[SMOKE RUN COMPLETE]")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
```

- [ ] **Step 2: Run a short smoke test (250 steps, `--num_envs 4`)**

Run: `PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/oracle_rollout.py --num_envs 4 2>&1 | tee /tmp/exp24_oracle_smoke.log`
(`--episodes` is accepted by argparse at this point but has no effect on the loop yet — Task 1's loop is a fixed 250 steps; Task 2 wires `--episodes` into an early-exit condition. Omit it here to avoid implying it does something yet.)
Expected: no tracebacks; `[STEP ...] waypoint_idx=[...]` lines show the waypoint index genuinely advancing from `[0, 0, 0, 0]` toward higher values (not stuck at 0) across the 250 steps, for at least some envs reaching `4` (place) before the 250-step episode ends. If waypoint_idx never advances past 0 for any env, STOP — this means the IK-target computation or waypoint-advance logic has a bug; do not proceed to Task 2 until this is fixed (re-check `_EE_OFFSET`, the IK controller command-frame convention, and `ADVANCE_TOLERANCE` against `ik_guided_path_bonus`'s exact working pattern).

- [ ] **Step 3: Commit**

```bash
git add scripts/oracle_rollout.py
git commit -m "Add scripted-oracle waypoint-following rollout core (Experiment 24 Gate 1)"
```

---

### Task 2: Success scoring — reproduce the grasp+lift contact-diagnostic criteria inline

**Files:**
- Modify: `scripts/oracle_rollout.py`

**Interfaces:**
- Consumes: `gripper_jaw1_contact`/`gripper_jaw2_contact` `ContactSensor`s (already in `Ar4PickPlaceMirrorEnvCfg`'s scene, `force_matrix_w` field — same read pattern as `scripts/warmresidual_contact_diagnostic.py`); cube's `root_pos_w[:, 2]` (height).
- Produces: per-step `gate_fires: torch.Tensor` (bool, shape `(num_envs,)`), and a per-env running `episode_passed: torch.Tensor` (bool) that latches `True` once `gate_fires` is ever `True` during that env's current episode, reset to `False` alongside each env's waypoint reset. Task 3 reads `episode_passed` to decide which episodes to record as demonstrations.

- [ ] **Step 1: Add the contact-scoring function and per-episode pass tracking**

In `scripts/oracle_rollout.py`, add near the top (after the existing constants):

```python
FORCE_THRESHOLD = 0.05
ANTIPODAL_COS_THRESHOLD = -0.7071
MINIMAL_HEIGHT = 0.03


def compute_gate_fires(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reproduces antipodal_grasp_bonus's exact bilateral force-closure
    check (tasks/ar4/mdp.py) plus the height_ok condition, matching this
    session's established contact-diagnostic pattern
    (scripts/warmresidual_contact_diagnostic.py)."""
    cube = env.scene["cube"]
    jaw1_sensor = env.scene["gripper_jaw1_contact"]
    jaw2_sensor = env.scene["gripper_jaw2_contact"]

    cube_z = cube.data.root_pos_w[:, 2]
    height_ok = cube_z > MINIMAL_HEIGHT

    jaw1_force_vec = jaw1_sensor.data.force_matrix_w.view(env.num_envs, 3)
    jaw2_force_vec = jaw2_sensor.data.force_matrix_w.view(env.num_envs, 3)
    jaw1_force_mag = torch.linalg.vector_norm(jaw1_force_vec, dim=-1)
    jaw2_force_mag = torch.linalg.vector_norm(jaw2_force_vec, dim=-1)
    both_magnitude_ok = (jaw1_force_mag > FORCE_THRESHOLD) & (jaw2_force_mag > FORCE_THRESHOLD)

    jaw1_dir = jaw1_force_vec / (jaw1_force_mag.unsqueeze(-1) + 1e-8)
    jaw2_dir = jaw2_force_vec / (jaw2_force_mag.unsqueeze(-1) + 1e-8)
    cos_angle = torch.sum(jaw1_dir * jaw2_dir, dim=-1)
    antipodal_ok = cos_angle < ANTIPODAL_COS_THRESHOLD

    grasp_ok = both_magnitude_ok & antipodal_ok
    return height_ok & grasp_ok
```

- [ ] **Step 2: Wire per-episode pass tracking into `main()`**

In `main()`, after `obs, _ = env.reset()` and the initial `reset_waypoints_for_envs` call, add:

```python
    episode_passed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    completed_episode_count = 0
    passed_episode_count = 0
```

Inside the step loop, after `advance_waypoint_idx(env)`, add:

```python
            gate_fires = compute_gate_fires(env)
            episode_passed |= gate_fires
```

Inside the `if len(done_ids) > 0:` block (replacing the smoke test's bare `reset_waypoints_for_envs(env, done_ids)` call), change to:

```python
            if len(done_ids) > 0:
                completed_episode_count += len(done_ids)
                passed_episode_count += int(episode_passed[done_ids].sum().item())
                episode_passed[done_ids] = False
                reset_waypoints_for_envs(env, done_ids)
                print(
                    f"[STEP {step:3d}] {len(done_ids)} episode(s) completed "
                    f"(total completed={completed_episode_count}, total passed={passed_episode_count})"
                )
                if completed_episode_count >= args_cli.episodes:
                    break
```

Replace the smoke test's fixed `for step in range(250):` loop bound with a large upper bound instead, since the loop now exits early via the `break` above once enough episodes complete:

```python
        for step in range(args_cli.episodes * 300):
```

(300 steps/episode is a safe upper bound — episodes are capped at 250 steps by `episode_length_s=5.0`/`decimation=2`/`sim.dt=0.01`, so 300 gives headroom without risking an infinite loop if some envs are slow to complete.)

Finally, replace the smoke test's closing prints with:

```python
    print(f"[SUMMARY] completed_episodes={completed_episode_count} passed_episodes={passed_episode_count}")
```

- [ ] **Step 2: Run the smoke test again to confirm scoring runs without error**

Run: `PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/oracle_rollout.py --num_envs 4 --episodes 8 2>&1 | tee /tmp/exp24_oracle_scoring_smoke.log`
Expected: no tracebacks; `[STEP ...] N episode(s) completed (total completed=... total passed=...)` lines appear; the script exits (via the `break`) once `completed_episode_count >= 8`; the final `[SUMMARY] ...` line prints. `passed_episodes` may be 0 at this small scale — that's fine, this step only verifies the scoring mechanism runs correctly, not the actual pass rate (Task 4 measures that at full scale).

- [ ] **Step 3: Commit**

```bash
git add scripts/oracle_rollout.py
git commit -m "Add grasp+lift gate scoring to the oracle rollout script (Experiment 24 Gate 1)"
```

---

### Task 3: Trajectory recording for Gate 2 reuse

**Files:**
- Modify: `scripts/oracle_rollout.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `obs["policy"]` (the concatenated observation tensor `env.step()`/`env.reset()` returns, shape `(num_envs, obs_dim)`); `actions` (the raw action tensor computed each step, shape `(num_envs, action_dim)`); `episode_passed` (Task 2).
- Produces: one `.npz` file per successful episode under `demonstrations/oracle/`, each containing `observations` (shape `(T, obs_dim)`) and `actions` (shape `(T, action_dim)`) arrays for that one episode's full trajectory.

- [ ] **Step 1: Check the `.gitignore` for an existing pattern, add one if needed**

Run: `grep -n "^demonstrations\|^assets" /home/saps/projects/rl/.gitignore`

If `demonstrations/` is not already covered, add it:

```bash
echo "demonstrations/" >> /home/saps/projects/rl/.gitignore
```

- [ ] **Step 2: Add per-env trajectory buffers and recording logic**

In `scripts/oracle_rollout.py`, add near the top:

```python
import numpy as np  # noqa: E402

DEMO_DIR = "/home/saps/projects/rl/demonstrations/oracle"
```

In `main()`, after the `episode_passed`/counters setup, add:

```python
    os.makedirs(DEMO_DIR, exist_ok=True)
    obs_buffers: list[list[torch.Tensor]] = [[] for _ in range(env.num_envs)]
    action_buffers: list[list[torch.Tensor]] = [[] for _ in range(env.num_envs)]
    saved_demo_count = 0
```

Inside the step loop, immediately after computing `actions = torch.cat([arm_raw, gripper_raw], dim=-1)` and BEFORE calling `env.step(actions)` (so the recorded observation/action pair for step `t` is the observation seen at `t` and the action taken from it, not the resulting observation at `t+1`):

```python
            for env_idx in range(env.num_envs):
                obs_buffers[env_idx].append(obs["policy"][env_idx].clone())
                action_buffers[env_idx].append(actions[env_idx].clone())
```

Inside the `if len(done_ids) > 0:` block, after `episode_passed[done_ids] = False` and BEFORE `reset_waypoints_for_envs(env, done_ids)` (so the buffers still hold the just-completed episode's data), add:

```python
                for env_idx in done_ids.tolist():
                    if episode_passed[env_idx]:
                        traj_obs = torch.stack(obs_buffers[env_idx]).cpu().numpy()
                        traj_actions = torch.stack(action_buffers[env_idx]).cpu().numpy()
                        demo_path = os.path.join(DEMO_DIR, f"demo_{saved_demo_count:04d}.npz")
                        np.savez(demo_path, observations=traj_obs, actions=traj_actions)
                        saved_demo_count += 1
                    obs_buffers[env_idx] = []
                    action_buffers[env_idx] = []
```

Note: `episode_passed[env_idx]` here is read from the loop-level `episode_passed` tensor BEFORE it gets zeroed in the same block above it in Task 2's code — since Task 2's `episode_passed[done_ids] = False` line runs first in the block, this recording code must read `episode_passed` into a local before that reset. Adjust Task 2's block ordering so the save-check happens BEFORE the `episode_passed[done_ids] = False` line:

```python
            if len(done_ids) > 0:
                completed_episode_count += len(done_ids)
                passed_episode_count += int(episode_passed[done_ids].sum().item())
                for env_idx in done_ids.tolist():
                    if episode_passed[env_idx]:
                        traj_obs = torch.stack(obs_buffers[env_idx]).cpu().numpy()
                        traj_actions = torch.stack(action_buffers[env_idx]).cpu().numpy()
                        demo_path = os.path.join(DEMO_DIR, f"demo_{saved_demo_count:04d}.npz")
                        np.savez(demo_path, observations=traj_obs, actions=traj_actions)
                        saved_demo_count += 1
                    obs_buffers[env_idx] = []
                    action_buffers[env_idx] = []
                episode_passed[done_ids] = False
                reset_waypoints_for_envs(env, done_ids)
                print(
                    f"[STEP {step:3d}] {len(done_ids)} episode(s) completed "
                    f"(total completed={completed_episode_count}, total passed={passed_episode_count}, "
                    f"demos saved={saved_demo_count})"
                )
                if completed_episode_count >= args_cli.episodes:
                    break
```

(This replaces the block written in Task 2 Step 2 — the final version of this block is the one above, combining both scoring and recording.)

- [ ] **Step 3: Run the smoke test again to confirm recording produces valid files**

Run: `PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/oracle_rollout.py --num_envs 4 --episodes 8 2>&1 | tee /tmp/exp24_oracle_recording_smoke.log`

Then verify at least one `.npz` file was written (if `passed_episodes` was 0 at this small scale in Task 2's smoke test, this may still be 0 here — that's expected; Task 4 runs at full scale where some passes are more likely if the mechanism works at all):

```bash
ls -la /home/saps/projects/rl/demonstrations/oracle/ 2>&1 || echo "No demo files yet (0 passed episodes at this small smoke-test scale - expected, not a failure)"
python3 -c "
import numpy as np
import glob
files = glob.glob('/home/saps/projects/rl/demonstrations/oracle/*.npz')
if files:
    d = np.load(files[0])
    print(f'demo file: {files[0]}, observations shape={d[\"observations\"].shape}, actions shape={d[\"actions\"].shape}')
else:
    print('no files to check yet')
"
```
Expected: no tracebacks; if any `.npz` files exist, they load cleanly and show a `(T, obs_dim)`/`(T, action_dim)` shape pair with `T` in a sane range (up to 250).

- [ ] **Step 4: Commit**

```bash
git add scripts/oracle_rollout.py .gitignore
git commit -m "Record successful-episode trajectories for Experiment 24 Gate 2 reuse"
```

---

### Task 4: Run and score (50 episodes)

**Files:**
- None modified — this task runs the script and inspects its output.

**Interfaces:**
- Consumes: `scripts/oracle_rollout.py` (Tasks 1-3).
- Produces: `demonstrations/oracle/demo_*.npz` files (one per successful episode out of 50 total), and the exact pass count for Task 5/the final report.

- [ ] **Step 1: Clear any smoke-test demo files, then launch the full 50-episode run**

```bash
rm -f /home/saps/projects/rl/demonstrations/oracle/*.npz
nohup /home/saps/IsaacLab/isaaclab.sh -p scripts/oracle_rollout.py --num_envs 10 --episodes 50 --headless < /dev/null > /tmp/exp24_oracle_full_run.log 2>&1 &
```

- [ ] **Step 2: Poll for completion (literal blocking command)**

Run this exact command and wait for it to return:
```bash
until grep -q "\[SUMMARY\]" /tmp/exp24_oracle_full_run.log 2>/dev/null; do sleep 15; done
```

- [ ] **Step 3: Check for tracebacks and extract the final pass count**

```bash
grep -iE "traceback|exception" /tmp/exp24_oracle_full_run.log
grep "\[SUMMARY\]" /tmp/exp24_oracle_full_run.log
ls /home/saps/projects/rl/demonstrations/oracle/*.npz | wc -l
```
Expected: no tracebacks (or only the routine benign USD/Fabric warnings seen throughout this session's other Isaac Sim runs); a `[SUMMARY] completed_episodes=50 passed_episodes=N` line; the `.npz` file count matches `N`.

- [ ] **Step 4: Report the pass rate against the hard-gate threshold**

Compare `N` (passed_episodes) against the spec's threshold: `N >= 30` (60%) is required to proceed. State the exact value of `N` plainly — this is the primary quantitative result Task 5's report will cite.

---

### Task 5: Video verification + final report

**Files:**
- Create: `docs/superpowers/plans/2026-07-08-ar4-experiment24-gate1-report.md`
- Create: `scripts/oracle_video_rollout.py`

**Interfaces:**
- Consumes: `scripts/oracle_rollout.py`'s core logic (Tasks 1-3) — this task creates a second, `num_envs=1` variant wrapped with `gym.wrappers.RecordVideo` for visual review, following `scripts/eval_loop.py`'s established `RecordVideo` pattern (`video_folder`, `step_trigger`, `video_length`, `name_prefix`, `disable_logger=True`).
- Produces: `docs/superpowers/plans/2026-07-08-ar4-experiment24-gate1-report.md` with the explicit PASS/FAIL verdict.

- [ ] **Step 1: Create the video-recording variant of the oracle script**

Create `scripts/oracle_video_rollout.py` (adapted from `scripts/oracle_rollout.py`, `num_envs=1`, wrapped for video, running until 3 successful (`gate_fires`-passing) episodes are captured or a generous episode budget is exhausted):

```python
"""Experiment 24 Gate 1: video-recording variant of oracle_rollout.py,
for visual verification of what the scripted oracle's "successful"
grasp+lift episodes actually look like - per this project's own
Experiment-16-established standard (a scalar/counter claiming success
is not sufficient without visual confirmation the object is genuinely
held between the jaws, not wedged against the wrist). See
docs/superpowers/specs/2026-07-08-ar4-experiment24-scripted-oracle-demonstration-bootstrap-design.md.

.. code-block:: bash

    PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/oracle_video_rollout.py --target_successes 3 --max_episodes 20
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Experiment 24 Gate 1: video-recording oracle rollout.")
parser.add_argument("--target_successes", type=int, default=3, help="Stop after this many gate_fires-passing episodes are recorded.")
parser.add_argument("--max_episodes", type=int, default=20, help="Give up after this many total episodes.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.utils.math import quat_apply, subtract_frame_transforms  # noqa: E402

sys.path.insert(0, "/home/saps/projects/rl")  # noqa: E402

from tasks.ar4 import mdp as ar4_mdp  # noqa: E402
from tasks.ar4.pickplace_env_cfg import _EE_OFFSET  # noqa: E402
from tasks.ar4.pickplace_mirror_env_cfg import Ar4PickPlaceMirrorEnvCfg  # noqa: E402
from tasks.ar4.robot_cfg import ARM_JOINT_NAMES, GRIPPER_JOINT_NAMES  # noqa: E402

ADVANCE_TOLERANCE = 0.03
LIFT_MINIMAL_HEIGHT = 0.03
PREGRASP_HOVER = 0.05
LIFT_MARGIN = 0.02
CARRY_HEIGHT = 0.10
ARM_SCALE = 0.5
FORCE_THRESHOLD = 0.05
ANTIPODAL_COS_THRESHOLD = -0.7071
MINIMAL_HEIGHT = 0.03

VIDEO_DIR = "/home/saps/projects/rl/logs/videos"


def compute_ik_arm_raw_action(env, ik_controller, robot_entity_cfg, ik_jacobi_idx, arm_default_joint_pos):
    robot = env.unwrapped.scene["robot"]
    current_waypoint_w = torch.gather(
        env.unwrapped._path_waypoints_w, 1, env.unwrapped._path_waypoint_idx.view(-1, 1, 1).expand(-1, 1, 3)
    ).squeeze(1)
    jacobian = robot.root_physx_view.get_jacobians()[:, ik_jacobi_idx, :, robot_entity_cfg.joint_ids]
    root_pose_w = robot.data.root_pose_w
    ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )
    joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
    offset_vec = torch.tensor(_EE_OFFSET, device=env.unwrapped.device).expand(env.unwrapped.num_envs, 3)
    offset_w = quat_apply(ee_pose_w[:, 3:7], offset_vec)
    ik_target_w = current_waypoint_w - offset_w
    waypoint_command_b, _ = subtract_frame_transforms(root_pose_w[:, 0:3], root_pose_w[:, 3:7], ik_target_w)
    ik_controller.set_command(waypoint_command_b, ee_pos=ee_pos_b, ee_quat=ee_quat_b)
    joint_pos_des = ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
    return (joint_pos_des - arm_default_joint_pos) / ARM_SCALE


def gripper_raw_action_for_waypoint(waypoint_idx):
    closed = (waypoint_idx >= 1) & (waypoint_idx <= 3)
    return torch.where(closed, torch.full_like(waypoint_idx, -1.0, dtype=torch.float32), torch.ones_like(waypoint_idx, dtype=torch.float32))


def advance_waypoint_idx(env):
    ee_frame = env.unwrapped.scene["ee_frame"]
    current_waypoint_w = torch.gather(
        env.unwrapped._path_waypoints_w, 1, env.unwrapped._path_waypoint_idx.view(-1, 1, 1).expand(-1, 1, 3)
    ).squeeze(1)
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    dist_to_waypoint = torch.norm(ee_pos_w - current_waypoint_w, dim=-1)
    reached = dist_to_waypoint < ADVANCE_TOLERANCE
    env.unwrapped._path_waypoint_idx = torch.where(
        reached & (env.unwrapped._path_waypoint_idx < 4), env.unwrapped._path_waypoint_idx + 1, env.unwrapped._path_waypoint_idx
    )


def reset_waypoints(env, env_ids):
    ar4_mdp.compute_path_waypoints(
        env.unwrapped, env_ids, object_cfg=SceneEntityCfg("cube"),
        lift_minimal_height=LIFT_MINIMAL_HEIGHT, pregrasp_hover=PREGRASP_HOVER,
        lift_margin=LIFT_MARGIN, carry_height=CARRY_HEIGHT,
    )


def compute_gate_fires(env):
    cube = env.unwrapped.scene["cube"]
    jaw1_sensor = env.unwrapped.scene["gripper_jaw1_contact"]
    jaw2_sensor = env.unwrapped.scene["gripper_jaw2_contact"]
    cube_z = cube.data.root_pos_w[:, 2]
    height_ok = cube_z > MINIMAL_HEIGHT
    jaw1_force_vec = jaw1_sensor.data.force_matrix_w.view(env.unwrapped.num_envs, 3)
    jaw2_force_vec = jaw2_sensor.data.force_matrix_w.view(env.unwrapped.num_envs, 3)
    jaw1_force_mag = torch.linalg.vector_norm(jaw1_force_vec, dim=-1)
    jaw2_force_mag = torch.linalg.vector_norm(jaw2_force_vec, dim=-1)
    both_magnitude_ok = (jaw1_force_mag > FORCE_THRESHOLD) & (jaw2_force_mag > FORCE_THRESHOLD)
    jaw1_dir = jaw1_force_vec / (jaw1_force_mag.unsqueeze(-1) + 1e-8)
    jaw2_dir = jaw2_force_vec / (jaw2_force_mag.unsqueeze(-1) + 1e-8)
    cos_angle = torch.sum(jaw1_dir * jaw2_dir, dim=-1)
    antipodal_ok = cos_angle < ANTIPODAL_COS_THRESHOLD
    return height_ok & (both_magnitude_ok & antipodal_ok)


def main() -> None:
    env_cfg = Ar4PickPlaceMirrorEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device

    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=VIDEO_DIR,
        step_trigger=lambda step: step % 250 == 0,
        video_length=250,
        name_prefix="ar4_oracle_gate1",
        disable_logger=True,
    )

    robot = env.unwrapped.scene["robot"]
    ik_cfg = DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls")
    ik_controller = DifferentialIKController(ik_cfg, num_envs=1, device=env.unwrapped.device)
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES, body_names=["link_6"])
    robot_entity_cfg.resolve(env.unwrapped.scene)
    ik_jacobi_idx = robot_entity_cfg.body_ids[0] - 1 if robot.is_fixed_base else robot_entity_cfg.body_ids[0]
    arm_joint_ids, _ = robot.find_joints(ARM_JOINT_NAMES)
    gripper_joint_ids, _ = robot.find_joints(GRIPPER_JOINT_NAMES)
    arm_default_joint_pos = robot.data.default_joint_pos[:, arm_joint_ids]

    env.reset()
    reset_waypoints(env, torch.tensor([0], device=env.unwrapped.device))

    successes = 0
    episodes = 0
    episode_passed = False
    step = 0
    while successes < args_cli.target_successes and episodes < args_cli.max_episodes:
        arm_raw = compute_ik_arm_raw_action(env, ik_controller, robot_entity_cfg, ik_jacobi_idx, arm_default_joint_pos)
        gripper_raw = gripper_raw_action_for_waypoint(env.unwrapped._path_waypoint_idx).unsqueeze(-1).expand(-1, len(gripper_joint_ids))
        actions = torch.cat([arm_raw, gripper_raw], dim=-1)
        _, _, terminated, truncated, _ = env.step(actions)
        advance_waypoint_idx(env)
        episode_passed = episode_passed or bool(compute_gate_fires(env)[0])

        if bool(terminated[0]) or bool(truncated[0]):
            episodes += 1
            if episode_passed:
                successes += 1
            print(f"[EPISODE {episodes}] passed={episode_passed} (successes so far: {successes}/{args_cli.target_successes})")
            episode_passed = False
            reset_waypoints(env, torch.tensor([0], device=env.unwrapped.device))
        step += 1

    print(f"[SUMMARY] episodes={episodes} successes={successes}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
```

- [ ] **Step 2: Run the video-recording script**

Run: `PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/oracle_video_rollout.py --target_successes 3 --max_episodes 20 < /dev/null > /tmp/exp24_oracle_video.log 2>&1`
(Run in foreground or background-and-poll per this session's established pattern — expect a few minutes given `num_envs=1` and up to 20 sequential 250-step episodes.)

Poll if backgrounded:
```bash
until grep -q "\[SUMMARY\]" /tmp/exp24_oracle_video.log 2>/dev/null; do sleep 15; done
```

Expected: no tracebacks; `[SUMMARY] episodes=N successes=M` with `M >= 3` (or `M < 3` if the oracle's pass rate is low even at this small sample — report whatever actually happens). Video files land in `logs/videos/ar4_oracle_gate1-step-*.mp4`.

- [ ] **Step 3: Watch the recorded videos and assess genuineness**

Use the `Read` tool (or equivalent frame-extraction, matching this session's established pattern of pulling specific frames for close inspection) on at least the first 2-3 recorded video files covering successful episodes. Look SPECIFICALLY for the Experiment-16-established false-positive pattern: is the cube visibly held between the two gripper jaw prongs (force-closure grasp), or wedged in the gap between the wrist/gripper housing? State this explicitly, per-video, in the report.

- [ ] **Step 4: Write the final report**

Create `docs/superpowers/plans/2026-07-08-ar4-experiment24-gate1-report.md`, stating:
- The exact 50-episode pass count from Task 4 (`N/50`), compared against the 30/50 (60%) threshold.
- The video-verification findings from Step 3 (genuine grasp vs. false positive, per sampled video).
- The explicit **Gate 1 PASS/FAIL verdict**: PASS requires BOTH `N >= 30` AND video-confirmed genuine grasps in the sampled episodes. State which (if either) condition failed, if it did.
- If PASS: state that Gate 2's implementation plan can now be written (do not write it in this task — that is explicitly out of scope per the spec's Scope note; the controller decides when to proceed).
- If FAIL: state plainly, and note that this falsifies the premise that a demonstrable successful trajectory exists using this repo's current waypoint/gripper mechanism as-is — per the spec's own hypothesis section, this would point to a mechanism-level problem (asset/gripper mechanics/waypoint geometry), not an exploration-only problem, and that the design's own escape valve (one bounded, literature/geometry-grounded refinement attempt — e.g. tightening `ADVANCE_TOLERANCE` specifically at the grasp waypoint) may be worth a single follow-up attempt, but no more.

- [ ] **Step 5: Commit and push**

```bash
git add scripts/oracle_video_rollout.py docs/superpowers/plans/2026-07-08-ar4-experiment24-gate1-report.md
git commit -m "Experiment 24 Gate 1 complete: scripted-oracle viability report"
git push origin main
```
