# AR4 Pick-and-Place RL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a PPO policy (via `rsl_rl`) that picks up the cube from the AR4 gripper+objects scene and places it near the sphere/wedge on the other side, replacing the imprecise scripted IK reach.

**Architecture:** A new `ManagerBasedRLEnvCfg` (`Ar4PickPlaceEnvCfg`) extends the existing gripper+objects scene with a `FrameTransformer` end-effector sensor, a randomized cube start pose, a randomized target-pose command, and reward/termination terms — almost entirely reused from Isaac Lab's own built-in Franka lift-task `mdp` module (`isaaclab_tasks.manager_based.manipulation.lift.mdp`), which is generic enough (parametrized by `SceneEntityCfg`) to point at our own scene entities without writing new reward/observation code. A thin `train.py` directly instantiates the env and wraps it for `rsl_rl` (no `gym.register`/Hydra — this repo's scripts always construct env configs directly, and that pattern is kept here for consistency). A companion `eval_loop.py` loads a checkpoint and records fixed-count episodes to mp4 via `gymnasium`'s `RecordVideo`.

**Tech Stack:** Isaac Lab (`ManagerBasedRLEnv`), `rsl_rl` PPO (`isaaclab_rl.rsl_rl`), `gymnasium.wrappers.RecordVideo`, PyTorch.

## Global Constraints

- Local-only training infra — no cloud compute or experiment tracking (deferred to a later area of the project roadmap).
- No automated pytest test suite for this repo's `rl/` code (established convention — see `rl/scripts/grasp_demo.py`, `drive_joints_demo.py`). Verification here is: headless smoke scripts (ephemeral, not committed) for fast checks, and real training/eval runs for end-to-end checks — consistent with `docs/superpowers/specs/2026-07-04-ar4-pickplace-rl-design.md`'s Testing/Verification section.
- Reuse `isaaclab_tasks.manager_based.manipulation.lift.mdp` reward/observation/termination functions directly rather than reimplementing them — they are generic (parametrized by `SceneEntityCfg`), and this is exactly how Isaac Lab's own robot-specific configs (e.g. Franka) reuse the same task module.
- All new Isaac Lab scripts must run via `./isaaclab.sh -p <path>` from the repo root (`/home/saps/projects/6DoF`), matching every existing script in `rl/scripts/`.
- Every Isaac Sim GUI/headless run in this project takes ~15-45s to start and does not reliably flush trailing `print()` output before shutdown (see `project_ar4_scene_config_lessons` memory) — verify completion via exit code and output-artifact existence (checkpoint files, video files), not trailing prints.

---

### Task 1: Pick-and-place environment config

**Files:**
- Create: `rl/tasks/ar4/pickplace_env_cfg.py`

**Interfaces:**
- Consumes: `Ar4SceneCfg`, `ActionsCfg` from `rl/tasks/ar4/env_cfg.py`; `AR4_MK5_CFG`, `ARM_JOINT_NAMES` from `rl/tasks/ar4/robot_cfg.py` (all already exist from the prior gripper+objects sub-project).
- Produces: `Ar4PickPlaceEnvCfg` (a `ManagerBasedRLEnvCfg` subclass) — consumed by Task 2's `train.py` and Task 3's `eval_loop.py`.

- [ ] **Step 1: Write the env config file**

```python
# rl/tasks/ar4/pickplace_env_cfg.py
"""Pick-and-place RL task for the AR4 mk5 arm: pick up the cube and place it
near the sphere/wedge on the other side of the workspace.

Reward, observation, and termination logic is reused directly from Isaac
Lab's built-in Franka lift-task mdp module (generic, parametrized by
SceneEntityCfg - not Franka-specific despite the import path) rather than
reimplemented here. See docs/superpowers/specs/2026-07-04-ar4-pickplace-rl-design.md.

Import this module only after an Isaac Sim/Isaac Lab AppLauncher has been
created.
"""

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.manipulation.lift import mdp

from .env_cfg import ActionsCfg, Ar4SceneCfg

# Empirically-tuned offset (m) from the ee_link frame to the gripper's jaw
# pinch point along ee_link's local +Z axis - same value used for the
# scripted IK reach in grasp_demo.py.
_EE_OFFSET = (0.0, 0.0, 0.09)


@configclass
class Ar4PickPlaceSceneCfg(Ar4SceneCfg):
    """AR4 gripper+objects scene, plus an end-effector FrameTransformer sensor."""

    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/root_joint/base_link",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/root_joint/ee_link",
                name="end_effector",
                offset=OffsetCfg(pos=_EE_OFFSET),
            ),
        ],
    )


@configclass
class CommandsCfg:
    """The cube's randomized target placement, in the robot's root frame.

    The robot base is rotated 180 deg about world Z (see robot_cfg.py), so
    a world-frame target near the sphere/wedge row (world x=-0.20,
    y in [0.28, 0.34]) becomes (x=+0.20, y in [-0.34, -0.28]) in the
    robot's own root frame (negate x and y - see grasp_demo.py's docstring
    for the same transform).
    """

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="ee_link",
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.18, 0.22),
            pos_y=(-0.34, -0.28),
            pos_z=(0.0, 0.02),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        cube_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame, params={"object_cfg": SceneEntityCfg("cube")}
        )
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset events: put the whole scene back to default, then jitter the cube's start pose."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_cube_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube"),
        },
    )


@configclass
class RewardsCfg:
    """Dense, staged reward - reach, lift, track goal, small action penalties, success bonus."""

    reaching_cube = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.1, "object_cfg": SceneEntityCfg("cube"), "ee_frame_cfg": SceneEntityCfg("ee_frame")},
        weight=1.0,
    )

    lifting_cube = RewTerm(
        func=mdp.object_is_lifted, params={"minimal_height": 0.03, "object_cfg": SceneEntityCfg("cube")}, weight=15.0
    )

    cube_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={
            "std": 0.3,
            "minimal_height": 0.03,
            "command_name": "object_pose",
            "object_cfg": SceneEntityCfg("cube"),
        },
        weight=16.0,
    )

    cube_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={
            "std": 0.05,
            "minimal_height": 0.03,
            "command_name": "object_pose",
            "object_cfg": SceneEntityCfg("cube"),
        },
        weight=5.0,
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class TerminationsCfg:
    """Success (cube at target) ends the episode early; otherwise a fixed timeout."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    cube_reached_goal = DoneTerm(
        func=mdp.object_reached_goal,
        params={"command_name": "object_pose", "threshold": 0.02, "object_cfg": SceneEntityCfg("cube")},
    )


@configclass
class Ar4PickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """AR4 pick-and-place task: pick up the cube, place it near the sphere/wedge."""

    scene: Ar4PickPlaceSceneCfg = Ar4PickPlaceSceneCfg(num_envs=512, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 5.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0)
        self.viewer.eye = (1.5, 1.5, 1.2)
        self.viewer.lookat = (0.0, 0.0, 0.4)
```

- [ ] **Step 2: Smoke-test the env config headlessly**

Requires the USD assets already built (`rl/scripts/build_asset.py` — already done in this repo). Create a throwaway script (not committed) to construct the env with a handful of parallel copies, step it with random actions, and print tensor shapes:

```bash
cat > /tmp/smoke_pickplace_env.py << 'EOF'
from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"headless": True})
simulation_app = app_launcher.app

import os
import sys
import torch

sys.path.insert(0, "rl")
from tasks.ar4.pickplace_env_cfg import Ar4PickPlaceEnvCfg  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

cfg = Ar4PickPlaceEnvCfg()
cfg.scene.num_envs = 4
env = ManagerBasedRLEnv(cfg=cfg)

obs, _ = env.reset()
print("obs shape:", obs["policy"].shape)
with torch.inference_mode():
    for _ in range(20):
        actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
        obs, rew, terminated, truncated, extras = env.step(actions)
print("reward shape:", rew.shape, "sample reward:", rew[0].item())
print("terminated shape:", terminated.shape)
print("SMOKE_TEST_OK")
env.close()
simulation_app.close()
EOF
cd /home/saps/projects/6DoF && /home/saps/IsaacLab/isaaclab.sh -p /tmp/smoke_pickplace_env.py 2>&1 | tail -20
```

Expected: no `Traceback`, and the output includes `SMOKE_TEST_OK` along with `obs shape: torch.Size([4, N])` (some `N` — the exact policy observation width; don't hardcode an expected value, just confirm it's a real 2D tensor with batch size 4) and `reward shape: torch.Size([4])`.

If it fails with an error resolving `{ENV_REGEX_NS}/Robot/root_joint/base_link` or `.../ee_link` (e.g. "did not find a match" from the `FrameTransformer`), open the built asset and confirm the actual prim names:

```bash
grep -c "root_joint/base_link\|root_joint/ee_link" rl/assets/ar4_mk5/ar4_mk5.usd 2>/dev/null || echo "binary file, use pxr to inspect instead"
```

(this is the exact same empirically-confirmed prim path used by `grasp_demo.py`'s logs earlier in the project — see `project_ar4_scene_config_lessons` memory — so a mismatch here would indicate the asset was rebuilt with different settings, not a typo in this file).

- [ ] **Step 3: Delete the throwaway smoke-test script**

```bash
rm /tmp/smoke_pickplace_env.py
```

- [ ] **Step 4: Commit**

```bash
git add rl/tasks/ar4/pickplace_env_cfg.py
git commit -m "Add AR4 pick-and-place RL environment config"
```

---

### Task 2: RL agent config + training script

**Files:**
- Create: `rl/tasks/ar4/agents/__init__.py`
- Create: `rl/tasks/ar4/agents/rsl_rl_ppo_cfg.py`
- Create: `rl/scripts/train.py`

**Interfaces:**
- Consumes: `Ar4PickPlaceEnvCfg` from Task 1 (`rl/tasks/ar4/pickplace_env_cfg.py`).
- Produces: `Ar4PickPlacePPORunnerCfg` (in `rl/tasks/ar4/agents/rsl_rl_ppo_cfg.py`) — consumed by Task 3's `eval_loop.py`. Training writes checkpoints to `rl/logs/train/<timestamp>/model_<iter>.pt`, consumed by Task 3.

- [ ] **Step 1: Write the agent (PPO hyperparameter) config**

```python
# rl/tasks/ar4/agents/__init__.py
```
(empty file — makes `agents` an importable package, matching the pattern Isaac Lab itself uses for `.../lift/config/franka/agents/__init__.py`)

```python
# rl/tasks/ar4/agents/rsl_rl_ppo_cfg.py
"""PPO (rsl_rl) hyperparameters for the AR4 pick-and-place task.

Adapted from Isaac Lab's own Franka cube-lift example
(isaaclab_tasks/manager_based/manipulation/lift/config/franka/agents/rsl_rl_ppo_cfg.py),
which is a proven starting point for a comparable single-arm pick task.
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class Ar4PickPlacePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "ar4_pickplace"
    # Our env has a single "policy" observation group; both the actor and
    # critic read from it (see RslRlBaseRunnerCfg.obs_groups docstring).
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
```

- [ ] **Step 2: Write the training script**

```python
# rl/scripts/train.py
"""Train a PPO policy (rsl_rl) for the AR4 pick-and-place task.

.. code-block:: bash

    ./isaaclab.sh -p rl/scripts/train.py --num_envs 512
    # smoke test (fast, verifies the loop runs end-to-end and writes a checkpoint):
    ./isaaclab.sh -p rl/scripts/train.py --num_envs 16 --max_iterations 2 --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train the AR4 pick-and-place policy with PPO (rsl_rl).")
parser.add_argument("--num_envs", type=int, default=512, help="Number of parallel environments.")
parser.add_argument("--max_iterations", type=int, default=None, help="Override the agent config's max_iterations.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos periodically during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of each recorded video (steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Steps between recorded videos.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import sys
from datetime import datetime

import gymnasium as gym

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tasks.ar4.agents.rsl_rl_ppo_cfg import Ar4PickPlacePPORunnerCfg  # noqa: E402
from tasks.ar4.pickplace_env_cfg import Ar4PickPlaceEnvCfg  # noqa: E402

LOG_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "train")


def main() -> None:
    env_cfg = Ar4PickPlaceEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    agent_cfg = Ar4PickPlacePPORunnerCfg()
    agent_cfg.device = args_cli.device
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations

    log_dir = os.path.join(LOG_ROOT, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)

    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=None)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()
    print(f"Training complete. Checkpoints and logs written to: {log_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
```

- [ ] **Step 3: Run the training smoke test**

```bash
cd /home/saps/projects/6DoF
/home/saps/IsaacLab/isaaclab.sh -p rl/scripts/train.py --num_envs 16 --max_iterations 2 --headless > /tmp/train_smoke.log 2>&1
echo "EXIT:$?"
grep -n "Traceback" /tmp/train_smoke.log
find rl/logs/train -name "model_*.pt" -newer rl/scripts/train.py
```

Expected: `EXIT:0`, no `Traceback` lines, and at least one `model_*.pt` file found under the just-created `rl/logs/train/<timestamp>/` directory (per `rsl_rl`'s `OnPolicyRunner`, a final checkpoint is always written at the end of `learn()` regardless of `save_interval`).

If it fails with an error mentioning `obs_groups` (e.g. a `KeyError` or type error while resolving observation groups), the `obs_groups` dict in `Ar4PickPlacePPORunnerCfg` needs adjusting — print `runner.cfg` or the raised exception's message to see which group name it expected, and match `obs_groups` to the actual group names the env exposes (this env only defines one group, `"policy"`, per `ObservationsCfg` in Task 1).

- [ ] **Step 4: Commit**

```bash
git add rl/tasks/ar4/agents/__init__.py rl/tasks/ar4/agents/rsl_rl_ppo_cfg.py rl/scripts/train.py
git commit -m "Add PPO agent config and training script for AR4 pick-and-place"
```

---

### Task 3: Eval/loop script with video recording

**Files:**
- Create: `rl/scripts/eval_loop.py`

**Interfaces:**
- Consumes: `Ar4PickPlaceEnvCfg`, `Ar4PickPlacePPORunnerCfg` from Tasks 1-2; a trained checkpoint path (any `model_*.pt` under `rl/logs/train/`).
- Produces: `rl/logs/videos/*.mp4` (one per episode).

- [ ] **Step 1: Write the eval/loop script**

```python
# rl/scripts/eval_loop.py
"""Run a trained AR4 pick-and-place PPO policy for a fixed number of episodes,
recording each one as an mp4 to rl/logs/videos/.

.. code-block:: bash

    ./isaaclab.sh -p rl/scripts/eval_loop.py --checkpoint rl/logs/train/<run>/model_1500.pt --episodes 10
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run a trained AR4 pick-and-place policy and record video.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the rsl_rl checkpoint (.pt) to load.")
parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # required for video recording

if not os.path.isfile(args_cli.checkpoint):
    sys.exit(f"Checkpoint not found: {args_cli.checkpoint}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tasks.ar4.agents.rsl_rl_ppo_cfg import Ar4PickPlacePPORunnerCfg  # noqa: E402
from tasks.ar4.pickplace_env_cfg import Ar4PickPlaceEnvCfg  # noqa: E402

VIDEO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "videos")


def main() -> None:
    env_cfg = Ar4PickPlaceEnvCfg()
    env_cfg.scene.num_envs = 1

    agent_cfg = Ar4PickPlacePPORunnerCfg()

    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=VIDEO_DIR,
        episode_trigger=lambda episode_id: episode_id < args_cli.episodes,
        video_length=0,  # 0 = record the full episode, not a fixed step count
        name_prefix="ar4_pickplace",
        disable_logger=True,
    )
    env = RslRlVecEnvWrapper(env, clip_actions=None)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs = env.get_observations()
    completed_episodes = 0
    with torch.inference_mode():
        while completed_episodes < args_cli.episodes and simulation_app.is_running():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if bool(dones[0]):
                completed_episodes += 1
                print(f"[INFO] Completed episode {completed_episodes}/{args_cli.episodes}")

    env.close()
    print(f"Videos written to: {VIDEO_DIR}")


if __name__ == "__main__":
    main()
    simulation_app.close()
```

- [ ] **Step 2: Run the eval smoke test against the Task 2 smoke checkpoint**

```bash
cd /home/saps/projects/6DoF
CKPT=$(find rl/logs/train -name "model_*.pt" | sort | tail -1)
echo "Using checkpoint: $CKPT"
/home/saps/IsaacLab/isaaclab.sh -p rl/scripts/eval_loop.py --checkpoint "$CKPT" --episodes 2 --headless > /tmp/eval_smoke.log 2>&1
echo "EXIT:$?"
grep -n "Traceback" /tmp/eval_smoke.log
find rl/logs/videos -name "*.mp4" -newer rl/scripts/eval_loop.py
```

Expected: `EXIT:0`, no `Traceback`, and at least one `.mp4` file under `rl/logs/videos/`.

If `episode_trigger` produces zero video files or `RecordVideo` raises an error about episode counting (this wrapper is primarily exercised in Isaac Lab's own scripts with `step_trigger`, not `episode_trigger`, for vectorized envs — see Task 3 Step 1's code), replace the `RecordVideo` call with this step-count-based form instead (episode length is fixed: `episode_length_s=5.0` / `step_dt=decimation*sim.dt=2*0.01=0.02s` = 250 steps/episode):

```python
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=VIDEO_DIR,
        step_trigger=lambda step: step % 250 == 0,
        video_length=250,
        name_prefix="ar4_pickplace",
        disable_logger=True,
    )
```

and re-run the same smoke-test command.

- [ ] **Step 3: Commit**

```bash
git add rl/scripts/eval_loop.py
git commit -m "Add eval/loop script recording AR4 pick-and-place episodes to mp4"
```

---

## Self-Review Notes

- **Spec coverage:** Task 1 covers the spec's "Task & scene changes" and "Observations & reward" sections. Task 2 covers "Training & eval" (training half). Task 3 covers "Training & eval" (eval half). The spec's "Out of scope" items (other 3 objects, cloud infra, vision, ROS2, foundation models) are deliberately not touched by any task.
- **Type/interface consistency:** `Ar4PickPlaceEnvCfg` (Task 1) is imported identically in Task 2's `train.py` and Task 3's `eval_loop.py`. `Ar4PickPlacePPORunnerCfg` (Task 2) is imported identically in Task 3. `SceneEntityCfg("cube")` matches the `cube` field name already defined on `Ar4SceneCfg` in the existing `rl/tasks/ar4/objects_cfg.py`/`env_cfg.py` — verified by reading those files, not assumed.
- **No placeholders:** every step has complete, real code; the two points of genuine technical uncertainty (the `obs_groups` dict contents, and `episode_trigger` vs `step_trigger` for `RecordVideo`) are called out with a concrete, ready-to-paste fallback rather than a vague "handle this if it breaks."
