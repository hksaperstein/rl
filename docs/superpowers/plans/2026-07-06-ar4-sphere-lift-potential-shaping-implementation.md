# AR4 sphere lift — potential-based reward shaping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether replacing the current independently-additive reward
terms with a single monotonic potential-based shaping term (Ng, Harada,
Russell, ICML 1999) — which can never produce negative reward for a
transient regression in any sub-signal — lets the AR4 sphere
pick-and-place policy escape its "reach, grip, freeze" local optimum.

**Architecture:** Replace six existing `RewardsCfg` terms
(`reaching_sphere`, `grasp_contact`, `lifting_sphere`,
`sphere_goal_tracking`, `sphere_goal_tracking_fine_grained`,
`lift_height_progress`) with one new term, `staged_potential_progress`,
computed as `γΦ(s') − Φ(s)` where `Φ` is a per-episode running-max of a
staged reach/grasp/lift/goal progress signal. `Φ`'s monotonicity (via a
per-env buffer, reset each episode) is what removes the risky-transition
disincentive — not the potential-based formula alone.

**Tech Stack:** Isaac Lab / Isaac Sim (`ManagerBasedRLEnv`), PyTorch,
`rsl_rl` PPO, existing AR4 mk5 task code in `tasks/ar4/`.

## Global Constraints

- Design spec: `docs/superpowers/specs/2026-07-06-ar4-sphere-lift-potential-shaping-design.md`
  (read it — this plan implements it exactly, including the two bugs
  found and fixed during design: the goal sub-term needs
  `combine_frame_transforms` to convert the command from the robot's root
  frame to world frame before comparing against the object's world-frame
  position, and the grasp sub-term reuses `contact_grasp_bonus` directly
  rather than re-deriving the same jaw-force check).
- Always invoke via `/home/saps/IsaacLab/isaaclab.sh -p scripts/<script>.py`
  from the repo root — never plain `python` for anything importing
  `isaaclab`.
- Decided values, not placeholders: staging weights `reach=0.1,
  grasp=0.2, lift=0.3, goal=0.4` (raw potential max `1.0`);
  `gamma=0.98` (must exactly match
  `Ar4PickPlacePPORunnerCfg.algorithm.gamma`); overall term
  `weight=25.0`; `reach_std=0.1`, `force_threshold=0.05`,
  `lift_minimal_height=0.03`, `goal_std=0.3` (all reused from the
  terms being replaced, not new untested constants).
- `action_rate`, `joint_vel`, `sphere_reached_goal` (a termination, not a
  reward), `_EE_OFFSET`, the `ContactSensorCfg` entries, every
  observation/command, and every PPO hyperparameter besides what's listed
  above must remain byte-for-byte unchanged.
- Verification standard: real evidence over proxies. Read the actual
  TensorBoard scalars and look at the actual eval video frames before
  concluding anything.

---

### Task 1: Implement the potential-based reward term and wire it in

**Files:**
- Modify: `tasks/ar4/mdp.py` (remove `lift_height_progress`, add
  `_raw_lift_progress`, `staged_potential_progress`,
  `reset_lift_potential`, new imports)
- Modify: `tasks/ar4/pickplace_env_cfg.py` (`RewardsCfg`, `EventCfg`)

**Interfaces:**
- Produces: `staged_potential_progress(env, gamma, object_cfg,
  ee_frame_cfg, jaw1_contact_cfg, jaw2_contact_cfg, robot_cfg,
  command_name, reach_std, force_threshold, lift_minimal_height,
  goal_std) -> torch.Tensor`, shape `(num_envs,)`.
- Produces: `reset_lift_potential(env, env_ids) -> None`, an `EventTerm`
  function (`mode="reset"`).
- Consumes: `contact_grasp_bonus` (already exists in `tasks/ar4/mdp.py`,
  unchanged — reused directly, not reimplemented).

- [ ] **Step 1: Update `tasks/ar4/mdp.py`'s imports**

Change:

```python
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor
```

to:

```python
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor, FrameTransformer
```

- [ ] **Step 2: Remove the now-superseded `lift_height_progress` function**

Delete this entire function from `tasks/ar4/mdp.py` (it's fully replaced
by `staged_potential_progress` below — nothing will call it once its
`RewardsCfg` registration is removed in Step 4, and this repo's practice
is to fully remove superseded reward functions rather than leave them as
dead code):

```python
def lift_height_progress(
    env: ManagerBasedRLEnv,
    height_std: float,
    rest_height: float,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Dense reward for upward progress on the object, from its resting
    height - unlike lifting_sphere's binary object_is_lifted threshold,
    this gives a real gradient below the success threshold so ordinary PPO
    exploration has something to climb, rather than needing to stumble
    directly onto minimal_height with no intermediate signal. Curriculum-
    gated (see CurriculumCfg in pickplace_env_cfg.py) rather than
    always-on, so early training (reach + grip) is unaffected until grip
    is already stable. See
    docs/superpowers/specs/2026-07-06-ar4-sphere-lift-curriculum-design.md.
    """
    object: RigidObject = env.scene[object_cfg.name]
    height_above_rest = torch.clamp(object.data.root_pos_w[:, 2] - rest_height, min=0.0)
    return torch.tanh(height_above_rest / height_std)
```

- [ ] **Step 3: Add the new functions to `tasks/ar4/mdp.py`**

Append to the end of the file (after `contact_grasp_bonus`):

```python
def _raw_lift_progress(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    jaw1_contact_cfg: SceneEntityCfg,
    jaw2_contact_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    command_name: str,
    reach_std: float,
    force_threshold: float,
    lift_minimal_height: float,
    goal_std: float,
) -> torch.Tensor:
    """Raw, per-step staged progress signal - NOT itself required to be
    monotonic (the monotonicity comes from the running-max wrapper that
    calls this, staged_potential_progress). Weighted so each higher stage
    dominates once reached: reach (0.1) < grasp (0.2) < lift (0.3) <
    goal-tracking (0.4), max 1.0. See
    docs/superpowers/specs/2026-07-06-ar4-sphere-lift-potential-shaping-design.md.
    """
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    reach_dist = torch.norm(object.data.root_pos_w - ee_frame.data.target_pos_w[:, 0, :], dim=-1)
    reach_term = 1.0 - torch.tanh(reach_dist / reach_std)

    # Reuse the already-tested contact_grasp_bonus directly (same bilateral
    # force-threshold check used by every prior experiment this session)
    # rather than re-deriving the same jaw-force logic inline.
    grasp_term = contact_grasp_bonus(env, force_threshold, jaw1_contact_cfg, jaw2_contact_cfg)

    lift_term = (object.data.root_pos_w[:, 2] > lift_minimal_height).float()

    # The command is generated in the robot's root frame (UniformPoseCommandCfg
    # with asset_name="robot") - must transform to world frame before comparing
    # against the object's world-frame position, exactly matching
    # isaaclab_tasks' own object_goal_distance (the function sphere_goal_tracking
    # used before this experiment replaced it).
    robot: RigidObject = env.scene[robot_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3])
    goal_dist = torch.norm(object.data.root_pos_w - des_pos_w, dim=-1)
    goal_term = 1.0 - torch.tanh(goal_dist / goal_std)

    return 0.1 * reach_term + 0.2 * grasp_term + 0.3 * lift_term + 0.4 * goal_term


def staged_potential_progress(
    env: ManagerBasedRLEnv,
    gamma: float,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    jaw1_contact_cfg: SceneEntityCfg,
    jaw2_contact_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    command_name: str,
    reach_std: float,
    force_threshold: float,
    lift_minimal_height: float,
    goal_std: float,
) -> torch.Tensor:
    """Potential-based reward shaping (Ng, Harada, Russell, ICML 1999):
    F(s,s') = gamma*Phi(s') - Phi(s), where Phi is a per-episode running
    max of _raw_lift_progress. Because Phi never decreases within an
    episode, this reward is always >= 0 - a momentary drop in the raw
    signal (e.g. contact force dipping during a real lift attempt) cannot
    produce negative reward, structurally removing the incentive to avoid
    risky transitions that a plain additive combination of the same
    sub-signals would create. See
    docs/superpowers/specs/2026-07-06-ar4-sphere-lift-potential-shaping-design.md.
    """
    if not hasattr(env, "_lift_potential_max"):
        env._lift_potential_max = torch.zeros(env.num_envs, device=env.device)

    raw = _raw_lift_progress(
        env, object_cfg, ee_frame_cfg, jaw1_contact_cfg, jaw2_contact_cfg, robot_cfg,
        command_name, reach_std, force_threshold, lift_minimal_height, goal_std,
    )
    prev_potential = env._lift_potential_max.clone()
    new_potential = torch.maximum(env._lift_potential_max, raw)
    env._lift_potential_max = new_potential

    return gamma * new_potential - prev_potential


def reset_lift_potential(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Event term (mode="reset"): zero the running-max potential buffer
    for resetting envs, so a new episode starts with no carried-over
    progress. Must be registered in EventCfg alongside reset_scene_to_default.
    """
    if not hasattr(env, "_lift_potential_max"):
        env._lift_potential_max = torch.zeros(env.num_envs, device=env.device)
    env._lift_potential_max[env_ids] = 0.0
```

- [ ] **Step 4: Replace the six reward terms in `RewardsCfg`**

In `tasks/ar4/pickplace_env_cfg.py`, change the entire `RewardsCfg` class
from:

```python
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

    # Active from iteration 0 (no curriculum gate - the prior curriculum-
    # gated version turned on too late, after the static-grip behavior had
    # already entrenched; this term is mechanically ~0 whenever the object
    # hasn't been lifted, which is impossible before grip exists, so there
    # was never a real risk in having it active from the start). Weight
    # matches lifting_sphere's own 25.0. See
    # docs/superpowers/specs/2026-07-06-ar4-sphere-lift-curriculum-design.md's
    # "Revision" section.
    lift_height_progress = RewTerm(
        func=ar4_mdp.lift_height_progress,
        weight=25.0,
        params={
            "height_std": 0.01,
            "rest_height": 0.009,
            "object_cfg": SceneEntityCfg("sphere"),
        },
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
```

to:

```python
@configclass
class RewardsCfg:
    """Single monotonic potential-based staged reward (reach->grasp->lift->
    goal-tracking), replacing six independent additive terms from prior
    experiments. See
    docs/superpowers/specs/2026-07-06-ar4-sphere-lift-potential-shaping-design.md
    for why: the additive combination let a momentary drop in any one
    sub-signal (e.g. grasp_contact dipping during a real lift attempt)
    produce a locally-worse trade than standing still, regardless of
    exploration. The running-max potential this term is built on can
    never do that - it is monotonically non-decreasing per episode, so
    the shaped reward is always >= 0."""

    staged_potential_progress = RewTerm(
        func=ar4_mdp.staged_potential_progress,
        weight=25.0,
        params={
            "gamma": 0.98,  # must match Ar4PickPlacePPORunnerCfg.algorithm.gamma exactly
            "object_cfg": SceneEntityCfg("sphere"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "jaw1_contact_cfg": SceneEntityCfg("gripper_jaw1_contact"),
            "jaw2_contact_cfg": SceneEntityCfg("gripper_jaw2_contact"),
            "robot_cfg": SceneEntityCfg("robot"),
            "command_name": "object_pose",
            "reach_std": 0.1,
            "force_threshold": 0.05,
            "lift_minimal_height": 0.03,
            "goal_std": 0.3,
        },
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")})
```

- [ ] **Step 5: Add the reset event to `EventCfg`**

Change:

```python
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
```

to:

```python
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

    # Zeroes the per-episode running-max potential buffer that
    # staged_potential_progress relies on for its monotonicity guarantee -
    # without this, a new episode would start with the PREVIOUS episode's
    # highest-ever potential still in the buffer, and the shaped reward
    # would read 0 for the entire episode (no new milestone could ever
    # exceed a stale carried-over max).
    reset_lift_potential = EventTerm(func=ar4_mdp.reset_lift_potential, mode="reset")
```

- [ ] **Step 6: Smoke test**

```bash
cd /home/saps/projects/rl
/home/saps/IsaacLab/isaaclab.sh -p scripts/train.py --num_envs 16 --max_iterations 2 --headless
```

Expected: exits 0, prints `Training complete.`. In the startup printout,
`Active Reward Terms` shows exactly three terms
(`staged_potential_progress`, `action_rate`, `joint_vel` — the six prior
terms are gone), and no exception about `_lift_potential_max` (confirming
the lazy-init `hasattr` guards work correctly regardless of whether the
reward function or the reset event runs first on a given step).

- [ ] **Step 7: Commit**

```bash
git add tasks/ar4/mdp.py tasks/ar4/pickplace_env_cfg.py
git commit -m "Replace additive reward terms with monotonic potential-based shaping for AR4 sphere lift"
```

---

### Task 2: Full 1500-iteration training run

**Files:** none (no code changes — this task runs the training loop and
inspects its output).

**Interfaces:**
- Consumes: the reward config from Task 1.
- Produces: `logs/train/<timestamp>/model_1499.pt` and TensorBoard event
  logs, consumed by Task 3.

- [ ] **Step 1: Run the full 1500-iteration training run**

```bash
cd /home/saps/projects/rl
/home/saps/IsaacLab/isaaclab.sh -p scripts/train.py --num_envs 4096 --headless
```

Expected wall-clock time: roughly 15-25 minutes, based on every prior full
run at this scale this session (~16 minutes each).

Note the resulting log directory and confirm `model_1499.pt` exists
(1500 iterations, 0-indexed) for Task 3.

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
for tag in ['Episode_Reward/staged_potential_progress', 'Episode_Termination/sphere_reached_goal',
            'Episode_Reward/action_rate', 'Episode_Reward/joint_vel']:
    if tag in ea.Tags()['scalars']:
        vals = ea.Scalars(tag)
        n = len(vals)
        samples = [vals[i].value for i in range(0, n, max(1, n // 10))]
        print(tag, '-> first:', vals[0].value, 'last:', vals[-1].value, 'max:', max(v.value for v in vals),
              'trajectory (10 samples):', [round(s, 6) for s in samples])
    else:
        print(tag, '-> NOT FOUND')
"
```

Note: `lifting_sphere`, `reaching_sphere`, `grasp_contact`,
`sphere_goal_tracking*` no longer exist as separate logged scalars (their
functionality is now folded into `staged_potential_progress` — this is
expected, not a bug). `sphere_reached_goal` remains available since it's
a termination, unaffected by this reward change. Record all four lines
in the Task 2 report verbatim.

- [ ] **Step 3: Write the report (create the file)**

Create `docs/superpowers/plans/2026-07-06-ar4-sphere-lift-potential-shaping-report.md`
with a "Task 2" section containing: the log directory path, the four
scalar lines from Step 2, and one factual sentence on
`staged_potential_progress`'s trajectory shape (e.g. whether it keeps
growing through training or plateaus early, matching or differing from
prior experiments' `grasp_contact`-plateau pattern) — no success/failure
judgment yet, that's Task 3/4's job after the eval video.

```bash
git add docs/superpowers/plans/2026-07-06-ar4-sphere-lift-potential-shaping-report.md
git commit -m "Record AR4 sphere potential-shaping full training run results"
```

---

### Task 3: Real eval + video inspection (decision gate)

**Files:** none (no code changes — this task runs eval and visually
inspects output).

**Interfaces:**
- Consumes: `logs/train/<timestamp>/model_1499.pt` from Task 2.
- Produces: eval videos in `logs/videos/`, a final pass/fail verdict
  consumed by Task 4.

- [ ] **Step 1: Run eval for 10 episodes**

```bash
cd /home/saps/projects/rl
/home/saps/IsaacLab/isaaclab.sh -p scripts/eval_loop.py --checkpoint logs/train/<RUN_DIR>/model_1499.pt --episodes 10
```

(substitute the actual `<RUN_DIR>` from Task 2). Expected: 10 files
`logs/videos/ar4_pickplace-step-0.mp4` through `-step-2250.mp4` (these
overwrite prior experiments' eval videos of the same name — that's fine).

- [ ] **Step 2: Extract frames from every episode video**

```bash
cd /home/saps/projects/rl
rm -rf logs/videos/frames
mkdir -p logs/videos/frames
for f in logs/videos/ar4_pickplace-step-*.mp4; do
  name=$(basename "$f" .mp4)
  mkdir -p "logs/videos/frames/$name"
  ffmpeg -y -i "$f" -vf fps=10 "logs/videos/frames/$name/frame_%03d.png" -loglevel error
done
```

- [ ] **Step 3: Visually inspect all 10 episodes**

Use the Read tool to view frames from each of the 10 episode directories
(start, ~25%, ~50%, ~75%, end is a good baseline sample; check adjacent
frames before concluding anything if the sphere marker briefly appears
missing — it can be occluded by the gripper body at some camera angles
without having been lifted, as happened in two prior experiments). For
each episode, determine: does the sphere visibly leave the ground at any
point, and if so, does it stay lifted and get carried toward the target?

- [ ] **Step 4: Apply the decision gate**

- **If 8/10 or more episodes show the sphere genuinely lifted and
  carried toward the target:** success. Proceed to Task 4's success
  path.
- **If fewer than 8/10 do, but some episodes show real (even brief)
  lifting that never happened in prior experiments:** partial progress —
  describe precisely what's different.
- **If 0/10 show any lift (same "reach, grip, freeze" signature):** this
  experiment is falsified. This is the fifth real attempt on this
  sub-problem (sparse-only, curriculum-gated dense, always-on dense,
  LR-bump, potential-shaping) — per `superpowers:systematic-debugging`
  Phase 4.5, flag back to the user rather than attempting a sixth
  reward/optimization tweak.

- [ ] **Step 5: Commit the report update**

```bash
git add docs/superpowers/plans/2026-07-06-ar4-sphere-lift-potential-shaping-report.md
git commit -m "Record AR4 sphere potential-shaping eval video inspection results"
```

---

### Task 4: Record the outcome in `ROADMAP.md`

**Files:**
- Modify: `ROADMAP.md`

**Interfaces:** none (documentation-only task, terminal in this plan).

- [ ] **Step 1: Write the outcome into `ROADMAP.md`'s "grasp/lift never
  emerges" follow-up**

Add a new bullet under the existing entry (after the LR-bump sub-bullet),
following the same evidentiary detail as the existing sub-bullets. Use
whichever template applies, filling in real numbers from Tasks 2-3:

**If Task 3's decision gate passed (8/10+):**

```markdown
   - **Follow-up experiment: monotonic potential-based reward shaping
     (SUCCESS).** Replaced six independent additive reward terms with a
     single running-max potential-based term (Ng, Harada, Russell, ICML
     1999), per
     `docs/superpowers/specs/2026-07-06-ar4-sphere-lift-potential-shaping-design.md` -
     structurally removing the possibility that a momentary drop in any
     sub-signal (e.g. grasp_contact dipping during a lift attempt)
     produces negative reward. Full run data:
     `docs/superpowers/plans/2026-07-06-ar4-sphere-lift-potential-shaping-report.md`.
     **Result: [X]/10 real eval episodes show the sphere genuinely
     lifted and carried toward the target.** This resolves the
     "grasp/lift never emerges" follow-up.
```

**If Task 3's decision gate did not pass (0/10 or partial):**

```markdown
   - **Follow-up experiment: monotonic potential-based reward shaping
     ([falsified | partial progress]).** Replaced six independent
     additive reward terms with a single running-max potential-based
     term (Ng, Harada, Russell, ICML 1999), per
     `docs/superpowers/specs/2026-07-06-ar4-sphere-lift-potential-shaping-design.md` -
     the strongest remaining literature-backed candidate after the
     LR-bump experiment's null result argued against pure exploration
     failure as a sufficient explanation. Full run data:
     `docs/superpowers/plans/2026-07-06-ar4-sphere-lift-potential-shaping-report.md`.
     **Result: [X]/10 real eval episodes show any lift** — [one to two
     sentences on what the video actually showed]. This is the fifth
     real attempt on the reward/optimization axis for this sub-problem
     (sparse-only, curriculum-gated dense, always-on dense, LR-bump,
     potential-shaping). Per `superpowers:systematic-debugging` Phase
     4.5, the next step should not be a sixth reward/optimization
     tweak. Flagged back to the Principal/user; remaining candidates are
     the hierarchical reach-then-grasp-policy split, or accepting that
     this repo's specific physical/task setup (gripper geometry, object
     scale) may need reconsidering rather than the reward function.
```

- [ ] **Step 2: Commit**

```bash
git add ROADMAP.md
git commit -m "Record AR4 sphere potential-shaping experiment outcome in ROADMAP"
```

---

## Plan self-review notes

- **Spec coverage:** Task 1 covers the design's full "Design" section
  (imports, dead-code removal, new functions, `RewardsCfg`/`EventCfg`
  registration) exactly, including both bugs found and fixed during
  design (frame transform, `contact_grasp_bonus` reuse). Task 2 covers
  the "Verification plan" section's full-run half. Task 3 covers the
  eval/video half. Task 4 covers the "flag back to the Principal"
  instruction for a fifth falsified attempt.
- **Scope discipline:** confirmed no task touches `action_rate`,
  `joint_vel`, `sphere_reached_goal`, `_EE_OFFSET`, any
  `ContactSensorCfg`, `CommandsCfg`, `ObservationsCfg`, or any PPO
  hyperparameter other than reusing (not changing) `gamma=0.98` as a
  reward param. The six replaced terms and the one new term are the
  entire diff's substance.
- **Type/name consistency:** `staged_potential_progress`'s full parameter
  list (`gamma, object_cfg, ee_frame_cfg, jaw1_contact_cfg,
  jaw2_contact_cfg, robot_cfg, command_name, reach_std, force_threshold,
  lift_minimal_height, goal_std`) is identical between its Task 1
  definition and its `RewardsCfg` registration. `_lift_potential_max` is
  the same buffer name in both `staged_potential_progress` and
  `reset_lift_potential`. `reset_lift_potential`'s registration
  (`ar4_mdp.reset_lift_potential`) matches the function defined in Task 1
  exactly.
