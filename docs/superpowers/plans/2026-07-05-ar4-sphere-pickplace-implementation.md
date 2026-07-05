# AR4 Sphere Pick-and-Place Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retarget the existing AR4 pick-and-place RL task from the Cube to the Sphere, then train and verify a policy that reliably picks up the Sphere and places it at a new (empty) target region.

**Architecture:** `tasks/ar4/pickplace_env_cfg.py`'s `Ar4PickPlaceEnvCfg` is edited in place — every `SceneEntityCfg("cube")` reference and its surrounding symbol names are swapped to the already-registered `sphere` scene entity (`tasks/ar4/env_cfg.py:38`), and the randomized target-placement region is mirrored onto the opposite side of the workspace (the cube/rect_prism row) so a real pick-and-carry is required rather than a near-zero-displacement no-op. Class names (`Ar4PickPlaceEnvCfg`, `Ar4PickPlaceSceneCfg`) are unchanged, so `scripts/train.py` and `scripts/eval_loop.py` need zero code changes. The actual research work is training and evaluating with this retargeted config, with one concrete, pre-identified reward-weight fallback if the baseline (unchanged weights) doesn't converge.

**Tech Stack:** Isaac Lab (`ManagerBasedRLEnv`), `rsl_rl` PPO (already-existing agent config), `ffmpeg` (frame extraction for video review), TensorBoard.

## Global Constraints

- No PR workflow in this repo — commit directly to `main` after each task.
- No automated pytest test suite for this Isaac Lab config code (established convention — see `docs/superpowers/specs/2026-07-04-ar4-pickplace-rl-design.md`'s Testing section). Verification is: headless smoke scripts (ephemeral, not committed) for fast structural checks, and real training/eval runs with video inspection for end-to-end checks.
- All Isaac Lab scripts run via `/home/saps/IsaacLab/isaaclab.sh -p <path>` from this repo's root (`/home/saps/projects/rl`) — never plain `python`.
- Isaac Sim GUI/headless runs do not reliably flush trailing `print()` output before shutdown — verify completion via exit code and output-artifact existence (checkpoint files, video files), not trailing prints.
- GPU is an RTX 5070 Ti (per `CLAUDE.md`) — the existing `num_envs=4096` default is already sized for it; don't reduce it for the full run.
- Reuse `isaaclab_tasks.manager_based.manipulation.lift.mdp` reward/observation/termination functions unchanged — they are already generic (parametrized by `SceneEntityCfg`) and shape-agnostic (pure distance/height math), so retargeting to the Sphere requires renaming and re-pointing existing terms, not writing new ones.
- `tasks/ar4/objects_cfg.py` and `tasks/ar4/env_cfg.py` are not modified — `sphere`, `cube`, `rect_prism`, and `wedge` are already fully-simulated `RigidObjectCfg` scene entities registered on `Ar4SceneCfg`, verified by reading `env_cfg.py:22,36,38` directly.

---

### Task 1: Retarget the environment config from Cube to Sphere

**Files:**
- Modify: `tasks/ar4/pickplace_env_cfg.py:1-194` (module docstring through `Ar4PickPlaceEnvCfg`; lines 196-265 — the perception-camera and interactive-demo configs — are untouched, out of scope per the spec)

**Interfaces:**
- Consumes: the `sphere` scene entity, already registered as `RigidObjectCfg = SPHERE_CFG` on `Ar4SceneCfg` (`tasks/ar4/env_cfg.py:38`) — verified present, no changes needed there.
- Produces: `Ar4PickPlaceEnvCfg`/`Ar4PickPlaceSceneCfg` — same class names as before, so Task 2/3's `scripts/train.py` and `scripts/eval_loop.py` imports are unaffected. Internal reward/observation/event/termination attribute names change (`*_cube` → `*_sphere`).

- [ ] **Step 1: Rewrite lines 1-194 of `tasks/ar4/pickplace_env_cfg.py`**

Replace the file's content from the top through the end of `Ar4PickPlaceEnvCfg` (i.e., everything before the `# World-frame constants shared by perception-consuming entry points` comment on the current line 197) with:

```python
# tasks/ar4/pickplace_env_cfg.py
"""Pick-and-place RL task for the AR4 mk5 arm: pick up the sphere and place it
near the cube/rect_prism on the other side of the workspace.

Reward, observation, and termination logic is reused directly from Isaac
Lab's built-in Franka lift-task mdp module (generic, parametrized by
SceneEntityCfg - not Franka-specific despite the import path) rather than
reimplemented here. See docs/superpowers/specs/2026-07-04-ar4-pickplace-rl-design.md
and docs/superpowers/specs/2026-07-05-ar4-sphere-pickplace-design.md.

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
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.manipulation.lift import mdp

from .env_cfg import ActionsCfg, Ar4SceneCfg

# Empirically-tuned offset (m) from the link_6 frame to the gripper's jaw
# pinch point along link_6's local +Z axis (ee_link sits at this same frame
# with an identity transform, but isn't itself a rigid body, so link_6 is
# used directly) - same value used for the scripted IK reach in grasp_demo.py.
_EE_OFFSET = (0.0, 0.0, 0.09)


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


@configclass
class CommandsCfg:
    """The sphere's randomized target placement, in the robot's root frame.

    The robot base is rotated 180 deg about world Z (see robot_cfg.py), so
    a world-frame target near the cube/rect_prism row (world x=+0.20,
    y in [0.28, 0.34]) becomes (x=-0.22, x=-0.18, y in [-0.34, -0.28]) in
    the robot's own root frame (negate x and y - see grasp_demo.py's
    docstring for the same transform). This mirrors the original cube
    task's target region (which used the sphere/wedge row) onto the
    opposite side: the sphere itself now starts on the sphere/wedge side,
    so reusing that row as the target would only require a near-zero
    displacement to "succeed" rather than a real pick-and-carry.
    """

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="link_6",
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-0.22, -0.18),
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
        sphere_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame, params={"object_cfg": SceneEntityCfg("sphere")}
        )
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset events: put the whole scene back to default, then jitter the sphere's start pose."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_sphere_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("sphere"),
        },
    )


@configclass
class RewardsCfg:
    """Dense, staged reward: reach, lift, coarse + fine-grained goal tracking, and small
    action penalties. There is no separate sparse success-bonus term - success is signaled
    via the `sphere_reached_goal` termination combined with the fine-grained goal-tracking
    reward, which increasingly rewards precise placement as the sphere nears the target."""

    reaching_sphere = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.1, "object_cfg": SceneEntityCfg("sphere"), "ee_frame_cfg": SceneEntityCfg("ee_frame")},
        weight=1.0,
    )

    lifting_sphere = RewTerm(
        func=mdp.object_is_lifted, params={"minimal_height": 0.03, "object_cfg": SceneEntityCfg("sphere")}, weight=15.0
    )

    sphere_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={
            "std": 0.3,
            "minimal_height": 0.03,
            "command_name": "object_pose",
            "object_cfg": SceneEntityCfg("sphere"),
        },
        weight=16.0,
    )

    sphere_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={
            "std": 0.05,
            "minimal_height": 0.03,
            "command_name": "object_pose",
            "object_cfg": SceneEntityCfg("sphere"),
        },
        weight=5.0,
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class TerminationsCfg:
    """Success (sphere at target) ends the episode early; otherwise a fixed timeout."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    sphere_reached_goal = DoneTerm(
        func=mdp.object_reached_goal,
        params={"command_name": "object_pose", "threshold": 0.02, "object_cfg": SceneEntityCfg("sphere")},
    )


@configclass
class Ar4PickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """AR4 pick-and-place task: pick up the sphere, place it near the cube/rect_prism."""

    scene: Ar4PickPlaceSceneCfg = Ar4PickPlaceSceneCfg(num_envs=4096, env_spacing=2.5)
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

Leave everything from the `# World-frame constants shared by perception-consuming entry points` comment onward (the perception camera and interactive-demo configs) exactly as it is — those still reference `"cube"` internally (via `perception/`, `scripts/interactive_demo.py`, `scripts/_perception_adapter.py`) and are explicitly out of scope for this task per the design spec.

- [ ] **Step 2: Smoke-test the retargeted config structurally**

Requires the USD assets already built (already done in this repo). Create a throwaway script (not committed) that constructs the env directly, steps it, and prints both tensor shapes and the actual sampled object/target positions — the position print is the empirical check that the new target region (Step 1's `pos_x` flip) lands where intended, rather than trusting the arithmetic alone:

```bash
cat > /tmp/smoke_sphere_pickplace_env.py << 'EOF'
from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"headless": True})
simulation_app = app_launcher.app

import os
import sys
import torch

sys.path.insert(0, os.getcwd())
from tasks.ar4.pickplace_env_cfg import Ar4PickPlaceEnvCfg  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

cfg = Ar4PickPlaceEnvCfg()
cfg.scene.num_envs = 4
env = ManagerBasedRLEnv(cfg=cfg)

obs, _ = env.reset()
print("obs shape:", obs["policy"].shape)
print("sphere world pos (env 0):", env.scene["sphere"].data.root_pos_w[0].tolist())
print("target command (env 0, robot frame):", env.command_manager.get_command("object_pose")[0].tolist())
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
cd /home/saps/projects/rl && /home/saps/IsaacLab/isaaclab.sh -p /tmp/smoke_sphere_pickplace_env.py 2>&1 | tail -20
```

Expected: no `Traceback`, and output includes `SMOKE_TEST_OK`, a real 2D `obs shape` tensor with batch size 4, a `sphere world pos` roughly near `[-0.20, 0.28, 0.009]` (its configured spawn point, ± the reset jitter), and a `target command` whose first component (position x, robot-root frame) is negative and roughly in `[-0.22, -0.18]` (confirming the mirrored target region from Step 1, not the old positive-x cube-era range).

If the `target command` tensor's layout isn't obviously `[x, y, z, qw, qx, qy, qz]` (e.g. it's a different width), print its full shape and first few values and match against `mdp.UniformPoseCommandCfg`'s documented command format rather than guessing — this determines which index is "x" for the sanity check above.

- [ ] **Step 3: Delete the throwaway smoke-test script**

```bash
rm /tmp/smoke_sphere_pickplace_env.py
```

- [ ] **Step 4: Commit**

```bash
git add tasks/ar4/pickplace_env_cfg.py
git commit -m "Retarget AR4 pick-and-place task from Cube to Sphere"
```

---

### Task 2: Verify the existing train/eval scripts still wire up correctly

**Files:**
- Modify: none — `scripts/train.py`, `scripts/eval_loop.py`, and `tasks/ar4/agents/rsl_rl_ppo_cfg.py` are already generic (they import `Ar4PickPlaceEnvCfg`/`Ar4PickPlacePPORunnerCfg` by name and never reference `"cube"` or `"sphere"` directly — verified by grep), so this task only re-verifies the full RL loop (PPO + `rsl_rl` wrapper + observation/reward wiring) still functions end-to-end against Task 1's retargeted config, before spending GPU time on a full run.

**Interfaces:**
- Consumes: `Ar4PickPlaceEnvCfg` (Task 1), `Ar4PickPlacePPORunnerCfg` (pre-existing, unchanged).
- Produces: a throwaway checkpoint under `logs/train/<timestamp>/`, consumed only by this task's own eval smoke test (not by Task 3, which trains its own fresh run).

- [ ] **Step 1: Training smoke test**

```bash
cd /home/saps/projects/rl
/home/saps/IsaacLab/isaaclab.sh -p scripts/train.py --num_envs 16 --max_iterations 2 --headless > /tmp/sphere_train_smoke.log 2>&1
echo "EXIT:$?"
grep -n "Traceback" /tmp/sphere_train_smoke.log
find logs/train -name "model_*.pt" -newer scripts/train.py
```

Expected: `EXIT:0`, no `Traceback`, at least one `model_*.pt` found under a freshly-created `logs/train/<timestamp>/` directory.

- [ ] **Step 2: Eval smoke test against that checkpoint**

```bash
cd /home/saps/projects/rl
CKPT=$(find logs/train -name "model_*.pt" | sort | tail -1)
echo "Using checkpoint: $CKPT"
/home/saps/IsaacLab/isaaclab.sh -p scripts/eval_loop.py --checkpoint "$CKPT" --episodes 2 --headless > /tmp/sphere_eval_smoke.log 2>&1
echo "EXIT:$?"
grep -n "Traceback" /tmp/sphere_eval_smoke.log
find logs/videos -name "*.mp4" -newer scripts/eval_loop.py
```

Expected: `EXIT:0`, no `Traceback`, at least one `.mp4` under `logs/videos/`. This is a 2-iteration checkpoint, so the policy will not have learned anything yet — this step only confirms the plumbing runs, not that it succeeds at the task.

Nothing to commit — no files changed in this task.

---

### Task 3: Full baseline training run, evaluation, and one bounded reward fallback

**Files:**
- Modify (only if the fallback branch in Step 4 is needed): `tasks/ar4/pickplace_env_cfg.py` (`lifting_sphere`'s `weight` parameter)

**Interfaces:**
- Consumes: `Ar4PickPlaceEnvCfg` (Task 1), verified wiring (Task 2).
- Produces: a real trained checkpoint under `logs/train/<timestamp>/`, real eval videos under `logs/videos/`, and a pass/fail verdict consumed by Task 4's `ROADMAP.md` update.

- [ ] **Step 1: Run the full baseline training**

```bash
cd /home/saps/projects/rl
/home/saps/IsaacLab/isaaclab.sh -p scripts/train.py --num_envs 4096 --headless > /tmp/sphere_train_full.log 2>&1 &
echo "started, PID $!"
```

Run this in the background (per this repo's own documentation, a full run is not a 2-5 minute step) and monitor via TensorBoard rather than waiting on the foreground process:

```bash
tensorboard --logdir logs/train &
```

- [ ] **Step 2: Watch the training curves**

Using the run's log dir (the most recently created directory under `logs/train/`), watch in TensorBoard until `Episode_Termination/sphere_reached_goal` has visibly climbed off zero and plateaued (same qualitative bar this repo's `README.md` already documents for the cube task — there is no fixed iteration count to target). Also sanity-check `Train/mean_reward` is climbing, and `Episode_Reward/lifting_sphere` is off zero (confirms the arm is at least grasping and lifting the sphere, independent of final placement accuracy).

- [ ] **Step 3: Real eval run with video inspection**

```bash
cd /home/saps/projects/rl
CKPT=$(find logs/train -name "model_*.pt" -newer /tmp/sphere_train_full.log 2>/dev/null | sort | tail -1)
if [ -z "$CKPT" ]; then CKPT=$(find logs/train -name "model_*.pt" | sort | tail -1); fi
echo "Using checkpoint: $CKPT"
/home/saps/IsaacLab/isaaclab.sh -p scripts/eval_loop.py --checkpoint "$CKPT" --episodes 10 --headless
find logs/videos -name "ar4_pickplace*.mp4" -newer scripts/eval_loop.py
```

For each of the 10 resulting videos, extract 3 representative frames (start, middle, end) and inspect them directly (per this repo's established verification standard — video files aren't self-verifying from an exit code):

```bash
mkdir -p /tmp/sphere_eval_frames
for f in logs/videos/ar4_pickplace*.mp4; do
  base=$(basename "$f" .mp4)
  ffmpeg -y -i "$f" -vf "select='eq(n,0)+eq(n,60)+eq(n,120)'" -vsync 0 "/tmp/sphere_eval_frames/${base}_%d.png" 2>/dev/null
done
ls /tmp/sphere_eval_frames
```

Read the extracted PNGs directly to determine, per episode, whether the sphere ends up at (or very near) the target region. Count how many of the 10 episodes succeed.

- [ ] **Step 4: Decision gate**

If **at least 8 of the 10** episodes show the sphere reliably placed at the target region (video-verified) **and** the TensorBoard curves from Step 2 climbed and plateaued: this task is done — record the result in Task 4 as success, and skip the fallback below.

Otherwise, apply this one concrete, pre-identified fallback (a stronger lift-reward gradient, since a round object is more prone to slipping out of a lateral pinch grasp than the flat-faced cube the reward weights were originally tuned for) and re-run Steps 1-3 once:

In `tasks/ar4/pickplace_env_cfg.py`, change:

```python
    lifting_sphere = RewTerm(
        func=mdp.object_is_lifted, params={"minimal_height": 0.03, "object_cfg": SceneEntityCfg("sphere")}, weight=15.0
    )
```

to:

```python
    lifting_sphere = RewTerm(
        func=mdp.object_is_lifted, params={"minimal_height": 0.03, "object_cfg": SceneEntityCfg("sphere")}, weight=25.0
    )
```

Re-run Steps 1-3 with this change. Whatever the outcome after this single fallback attempt (success or not), that is the result recorded in Task 4 — further reward/hyperparameter research beyond this one bounded attempt is a follow-up for `ROADMAP.md`, not open-ended iteration within this plan (consistent with how this repo already handled the perception classifier's accuracy issue: root-caused, documented, deferred).

- [ ] **Step 5: Commit (only if the fallback in Step 4 was applied)**

```bash
git add tasks/ar4/pickplace_env_cfg.py
git commit -m "Increase lifting_sphere reward weight to help grasp retention"
```

If the fallback wasn't needed, there is nothing to commit for this task (Task 1 already committed the only code change).

---

### Task 4: Update ROADMAP.md with the outcome

**Files:**
- Modify: `ROADMAP.md`

**Interfaces:**
- Consumes: Task 3's pass/fail verdict and (if the fallback was used) whether it resolved the issue.

- [ ] **Step 1: Record the outcome**

If Task 3 succeeded (with or without the fallback), change the `## Built` section of `ROADMAP.md` from:

```markdown
## Built

- **AR4 pick-and-place** (perception + RL training/eval/interactive demo) —
  working end-to-end.
```

to:

```markdown
## Built

- **AR4 pick-and-place** (perception + RL training/eval/interactive demo) —
  working end-to-end.
- **AR4 sphere pick-and-place** — retargeted the pick-and-place task from
  the Cube to the Sphere (`tasks/ar4/pickplace_env_cfg.py`), reusing the
  existing robot/scene/camera infrastructure. Trained and verified: at
  least 8/10 real eval episodes reach the target region (video-verified).
```

(Add a trailing clause noting the `lifting_sphere` weight increase to 25.0, if Step 4's fallback was the one that made it succeed.)

If Task 3 did **not** succeed even after the one fallback attempt, instead add a new entry to `## Known follow-ups` describing exactly what was observed (e.g., which of the two conditions in Task 3 Step 4 failed, and what the eval videos actually showed — sphere not grasped at all, grasped but dropped, grasped but missing the target, etc.) so the next session doesn't have to re-run the experiment to rediscover the failure mode.

- [ ] **Step 2: Commit**

```bash
git add ROADMAP.md
git commit -m "Record AR4 sphere pick-and-place outcome in ROADMAP"
```

---

## Self-Review Notes

- **Spec coverage:** Task 1 covers the spec's "Scene/objects" and "Sensor selection" (no-op, confirmed unchanged) sections and the target-region redesign. Task 1's reward-term renaming plus Task 3's fallback cover the spec's "Reward function" section. Tasks 2-3 cover "Experiment/iterate loop." Task 4 covers the spec's living-doc maintenance rule. The spec's "Out of scope" items (new robot/sensors, real-camera perception accuracy, RectPrism/Wedge as manipulated objects, robot/gripper/asset changes) are deliberately untouched by every task.
- **Type/interface consistency:** `Ar4PickPlaceEnvCfg`/`Ar4PickPlaceSceneCfg` class names are unchanged from Task 1 through Task 3, so `scripts/train.py`/`scripts/eval_loop.py` imports (Task 2) need no edits — verified against the actual current file contents of both scripts, not assumed. `SceneEntityCfg("sphere")` matches the `sphere` field already defined on `Ar4SceneCfg` in `tasks/ar4/env_cfg.py:38` — verified by reading that file directly, not assumed.
- **No placeholders:** every step has complete, real code or an exact command with expected output. The one point of genuine open-endedness (whether the baseline reward weights converge) is bounded to exactly one concrete, justified fallback change with exact numbers, not open-ended "tune until it works."
