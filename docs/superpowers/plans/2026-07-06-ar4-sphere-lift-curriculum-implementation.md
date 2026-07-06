# AR4 sphere lift-curriculum reward — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get the AR4 sphere pick-and-place policy to actually lift the
sphere, building on the ContactSensor experiment's real progress (grip is
now reliably achieved) by adding a dense, curriculum-gated height-shaping
reward that gives the policy a gradient to climb toward the existing
binary `lifting_sphere` threshold, instead of a sparse cliff with nothing
guiding exploration below it.

**Architecture:** Add one new dense reward term (`lift_height_progress`,
`tanh`-shaped on height above the sphere's resting position) to
`tasks/ar4/mdp.py`, registered with `weight=0.0` so it's inert during
phase-1 (reach + grip) training, then switched to `weight=15.0` at
iteration 700 via Isaac Lab's own `modify_reward_weight` curriculum term
(already shipped, already used by the Franka lift task this repo's reward
functions were adapted from). Every other reward term, and `lifting_sphere`
itself, stay completely untouched.

**Tech Stack:** Isaac Lab / Isaac Sim (`ManagerBasedRLEnv`), PyTorch,
`rsl_rl` PPO, existing AR4 mk5 task code in `tasks/ar4/`.

## Global Constraints

- Design spec: `docs/superpowers/specs/2026-07-06-ar4-sphere-lift-curriculum-design.md`
  (read it — this plan implements it exactly, including the decided
  parameter values in its "Design" section).
- Always invoke via `/home/saps/IsaacLab/isaaclab.sh -p scripts/<script>.py`
  (or `-p -m pytest ...` / `-p -c "..."`) from the repo root — never plain
  `python` for anything importing `isaaclab`.
- Decided values, not placeholders: `height_std=0.01`, `rest_height=0.009`,
  `lift_height_progress` weight `0.0` → `15.0`, curriculum `num_steps=16800`
  (iteration 700 × `num_steps_per_env=24`).
- Every other reward term (`reaching_sphere`, `lifting_sphere`,
  `grasp_contact`, `sphere_goal_tracking`, `sphere_goal_tracking_fine_grained`,
  `action_rate`, `joint_vel`) and every observation/command/event/PPO
  hyperparameter must remain byte-for-byte unchanged — including the
  already-corrected `_EE_OFFSET=0.036` from the prior experiment, which is
  now baseline, not something this plan touches.
- Verification standard: real evidence over proxies. Don't call a task
  done off exit codes alone — read the actual TensorBoard scalars and look
  at the actual eval video frames.

---

### Task 1: Add the dense `lift_height_progress` reward term + curriculum

**Files:**
- Modify: `tasks/ar4/mdp.py` (new function, appended after `contact_grasp_bonus`)
- Modify: `tasks/ar4/pickplace_env_cfg.py` (imports, `RewardsCfg`, new
  `CurriculumCfg`, `Ar4PickPlaceEnvCfg`)

**Interfaces:**
- Produces: `lift_height_progress(env, height_std, rest_height, object_cfg)
  -> torch.Tensor` of shape `(num_envs,)`, values in `[0.0, 1.0)`.
- Produces: a `curriculum: CurriculumCfg` field on `Ar4PickPlaceEnvCfg`,
  consumed automatically by Isaac Lab's `CurriculumManager` (no other task
  reads this directly).

- [ ] **Step 1: Add the reward function to `tasks/ar4/mdp.py`**

Append to the end of the existing file (after `contact_grasp_bonus`):

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

This needs `RigidObject` importable for the type hint. Since the file
starts with `from __future__ import annotations`, annotations are lazy
strings and don't need to be resolvable at runtime — but `contact_grasp_bonus`
already establishes the pattern of putting sim-only types under
`TYPE_CHECKING`. Update the `TYPE_CHECKING` block at the top of the file:

```python
if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor
```

- [ ] **Step 2: Register the reward term in `RewardsCfg`**

In `tasks/ar4/pickplace_env_cfg.py`, change:

```python
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

to:

```python
    grasp_contact = RewTerm(
        func=ar4_mdp.contact_grasp_bonus,
        weight=20.0,
        params={
            "force_threshold": 0.05,
            "jaw1_contact_cfg": SceneEntityCfg("gripper_jaw1_contact"),
            "jaw2_contact_cfg": SceneEntityCfg("gripper_jaw2_contact"),
        },
    )

    # Curriculum-gated: weight starts at 0.0 (inert during phase-1 reach+grip
    # training) and is raised to 15.0 at iteration 700 by CurriculumCfg below.
    lift_height_progress = RewTerm(
        func=ar4_mdp.lift_height_progress,
        weight=0.0,
        params={
            "height_std": 0.01,
            "rest_height": 0.009,
            "object_cfg": SceneEntityCfg("sphere"),
        },
    )

    sphere_goal_tracking = RewTerm(
```

- [ ] **Step 3: Add the `CurriculumCfg` class**

In the same file, add this new class right after `RewardsCfg` (before
`TerminationsCfg`):

```python
@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP. Ramps in the dense lift-height shaping
    term only after grip has converged (see this session's own TensorBoard
    data in docs/superpowers/plans/2026-07-06-ar4-sphere-contact-grasp-reward-report.md,
    where grasp_contact plateaus by iteration ~700), rather than competing
    with grip-learning from iteration 0. Uses Isaac Lab's own
    modify_reward_weight curriculum term - the same mechanism the Franka
    lift task (isaaclab_tasks/manager_based/manipulation/lift/lift_env_cfg.py)
    uses for its own action_rate/joint_vel curriculum - rather than custom
    curriculum code."""

    lift_height_progress = CurrTerm(
        func=modify_reward_weight,
        params={"term_name": "lift_height_progress", "weight": 15.0, "num_steps": 16800},
    )
```

- [ ] **Step 4: Add the required imports and wire the curriculum into the env cfg**

Change:

```python
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import CameraCfg, ContactSensorCfg, FrameTransformerCfg
```

to:

```python
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.curriculums import modify_reward_weight
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import CameraCfg, ContactSensorCfg, FrameTransformerCfg
```

Then change `Ar4PickPlaceEnvCfg` (add the `curriculum` field — this class
currently has no `curriculum` field at all; `ManagerBasedRLEnvCfg`'s own
default is `curriculum: object | None = None`, i.e. no curriculum applied):

```python
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
    curriculum: CurriculumCfg = CurriculumCfg()
```

- [ ] **Step 5: Smoke test — confirm the term and curriculum wire up**

```bash
cd /home/saps/projects/rl
/home/saps/IsaacLab/isaaclab.sh -p scripts/train.py --num_envs 16 --max_iterations 2 --headless
```

Expected: exits 0, prints `Training complete.`. In the startup printout,
the `Active Reward Terms` table now includes `lift_height_progress` with
weight `0.0`, and the `Active Curriculum Terms` table (previously empty)
now lists `lift_height_progress`.

- [ ] **Step 6: Commit**

```bash
git add tasks/ar4/mdp.py tasks/ar4/pickplace_env_cfg.py
git commit -m "Add curriculum-gated dense lift-height reward for AR4 sphere task"
```

---

### Task 2: Full training run

**Files:** none (no code changes — this task runs the training loop and
inspects its output).

**Interfaces:**
- Consumes: the reward/curriculum config from Task 1.
- Produces: `logs/train/<timestamp>/model_1499.pt` and TensorBoard event
  logs, consumed by Task 3.

- [ ] **Step 1: Run the full 1500-iteration training run**

```bash
cd /home/saps/projects/rl
/home/saps/IsaacLab/isaaclab.sh -p scripts/train.py --num_envs 4096 --headless
```

Expected wall-clock time: roughly 15-25 minutes at `num_envs=4096`, based
on the prior ContactSensor experiment's full run at the same scale (~16
minutes).

Note the resulting log directory (`logs/train/<timestamp>/`) and confirm
`model_1499.pt` exists (1500 iterations, 0-indexed) for Task 3.

- [ ] **Step 2: Pull the key TensorBoard scalars, including the pre/post-curriculum split**

```bash
cd /home/saps/projects/rl
LATEST=$(ls -dt logs/train/*/ | head -1)
/home/saps/IsaacLab/isaaclab.sh -p -c "
from tensorboard.backend.event_processing import event_accumulator
import glob
path = sorted(glob.glob('${LATEST}events.out.tfevents.*'))[-1]
ea = event_accumulator.EventAccumulator(path)
ea.Reload()
for tag in ['Episode_Reward/lift_height_progress', 'Episode_Reward/lifting_sphere',
            'Episode_Reward/grasp_contact', 'Episode_Reward/reaching_sphere',
            'Episode_Termination/sphere_reached_goal']:
    if tag in ea.Tags()['scalars']:
        vals = ea.Scalars(tag)
        at_iter_699 = next((v.value for v in vals if v.step == 699), None)
        at_iter_701 = next((v.value for v in vals if v.step == 701), None)
        print(tag, '-> first:', vals[0].value, 'last:', vals[-1].value, 'max:', max(v.value for v in vals),
              'at_iter_699:', at_iter_699, 'at_iter_701:', at_iter_701)
    else:
        print(tag, '-> NOT FOUND')
"
```

Record these five lines in the Task 2 report verbatim. Specifically check:
`lift_height_progress` should read ~0 at iteration 699 (just before the
curriculum switch) since its weight is still 0.0 there, and should be
free to move after iteration 701 (weight now 15.0). This confirms the
curriculum gate actually fired at the intended point, not just that the
run completed.

- [ ] **Step 3: Write the report (create the file)**

Create `docs/superpowers/plans/2026-07-06-ar4-sphere-lift-curriculum-report.md`
with a "Task 2" section containing: the log directory path, the five
scalar lines from Step 2, and one sentence on whether `lifting_sphere`
moved meaningfully off 0 after the curriculum switch (necessary but not
sufficient — Task 3's eval video confirms whether it's a real lift).

```bash
git add docs/superpowers/plans/2026-07-06-ar4-sphere-lift-curriculum-report.md
git commit -m "Record AR4 sphere lift-curriculum full training run results"
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
overwrite the prior experiment's eval videos of the same name — that's
fine, this is the current experiment's eval).

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

This produces ~50 frames per episode (5s episode @ fps=10).

- [ ] **Step 3: Visually inspect all 10 episodes**

Use the Read tool to view a representative sample of frames from each of
the 10 episode directories (start, ~25%, ~50%, ~75%, end is enough — no
need to view all ~50 frames per episode). For each episode, determine:
does the sphere visibly leave the ground at any point, and if so, does it
stay lifted or immediately drop back down?

- [ ] **Step 4: Apply the decision gate**

- **If 8/10 or more episodes show the sphere genuinely lifted (leaves the
  ground and stays up, not a single-frame bounce) and carried toward the
  target:** this is a successful fix. Proceed to Task 4's success path.
- **If fewer than 8/10 do, but at least some episodes show real (even if
  brief or inconsistent) lifting that never happened in the prior
  experiment:** this is partial progress, not a clean success or a flat
  repeat — describe precisely what's now different (e.g., "lifts briefly
  then drops," "lifts in 3/10 episodes," etc.) rather than forcing it into
  either bucket.
- **If 0/10 episodes show any lift at all (same "reach, grip, freeze"
  signature as before):** the dense-shaping-plus-curriculum attempt did
  not work either. Per the design doc's own instruction, do not attempt a
  further reward tweak unilaterally — flag back to the Principal/user.

- [ ] **Step 5: Commit the report update**

Append a "Task 3" section to
`docs/superpowers/plans/2026-07-06-ar4-sphere-lift-curriculum-report.md`
with: the episode-by-episode observations (or a summary if patterns repeat
across most episodes), the X/10 count, and which branch of Step 4 applies.

```bash
git add docs/superpowers/plans/2026-07-06-ar4-sphere-lift-curriculum-report.md
git commit -m "Record AR4 sphere lift-curriculum eval video inspection results"
```

---

### Task 4: Record the outcome in `ROADMAP.md`

**Files:**
- Modify: `ROADMAP.md`

**Interfaces:** none (documentation-only task, terminal in this plan).

- [ ] **Step 1: Write the outcome into `ROADMAP.md`'s "grasp/lift never
  emerges" follow-up**

Open `ROADMAP.md` and add a new bullet under the existing "grasp/lift
never emerges" entry (after the ContactSensor sub-bullet), following the
exact same level of evidentiary detail as the existing sub-bullets
(quantitative TensorBoard numbers from Task 2 + qualitative video findings
from Task 3 + root-cause reasoning). Use whichever of the three templates
below applies, filling in the actual measured numbers and observations
from Tasks 2-3 (not placeholder text):

**If Task 3's decision gate passed (8/10+):**

```markdown
   - **Follow-up experiment: curriculum-gated dense lift-height reward
     (SUCCESS).** Per the ContactSensor experiment's "reach, grip, freeze"
     finding (grip reliably achieved but lift never explored, since
     lifting_sphere's binary threshold gives no gradient below it), added
     a dense tanh-shaped lift_height_progress term curriculum-gated on at
     iteration 700 (once grip converges), per
     `docs/superpowers/specs/2026-07-06-ar4-sphere-lift-curriculum-design.md`.
     `lift_height_progress`/`lifting_sphere` reward: [Task 2's values].
     Full run data: `docs/superpowers/plans/2026-07-06-ar4-sphere-lift-curriculum-report.md`.
     **Result: [X]/10 real eval episodes show the sphere genuinely lifted
     off the ground and carried toward the target.** This resolves the
     "grasp/lift never emerges" follow-up.
```

**If Task 3's decision gate showed partial progress:**

```markdown
   - **Follow-up experiment: curriculum-gated dense lift-height reward —
     partial progress, not yet a full success.** Per the ContactSensor
     experiment's "reach, grip, freeze" finding, added a dense tanh-shaped
     lift_height_progress term curriculum-gated on at iteration 700, per
     `docs/superpowers/specs/2026-07-06-ar4-sphere-lift-curriculum-design.md`.
     `lift_height_progress`/`lifting_sphere` reward: [Task 2's values].
     Full run data: `docs/superpowers/plans/2026-07-06-ar4-sphere-lift-curriculum-report.md`.
     **Result: [X]/10 episodes show real lifting** — [precise description
     of what the video actually showed: partial lifts, drops, inconsistency
     across episodes]. This is real movement beyond the prior experiment's
     flat "reach, grip, freeze" signature, but not yet a reliable success.
     [Your assessment of the most likely next lever, if evident from the
     data - e.g. curriculum timing, height_std tuning, or something else -
     stated as an observation, not a unilateral next action.] Flagged back
     to the Principal/user rather than continuing unilaterally.
```

**If Task 3's decision gate did not pass (0/10, same signature as before):**

```markdown
   - **Follow-up experiment: curriculum-gated dense lift-height reward
     (falsified).** Per the ContactSensor experiment's "reach, grip,
     freeze" finding, added a dense tanh-shaped lift_height_progress term
     curriculum-gated on at iteration 700, per
     `docs/superpowers/specs/2026-07-06-ar4-sphere-lift-curriculum-design.md`.
     `lift_height_progress`/`lifting_sphere` reward: [Task 2's values].
     Full run data: `docs/superpowers/plans/2026-07-06-ar4-sphere-lift-curriculum-report.md`.
     **Result: 0/10 real eval episodes show any lift** — [one to two
     sentences on what the video actually showed, matching the specificity
     of the ContactSensor entry's failure description]. Dense shaping plus
     a curriculum schedule was the literature-backed next step after the
     ContactSensor experiment's finding; with this also not working, per
     the design doc's own instruction this is flagged back to the
     Principal/user rather than attempting a further reward-only tweak —
     remaining candidates are a hierarchical reach-then-grasp-policy split,
     or questioning whether the gripper's real closed-jaw force is
     physically sufficient to support lifting this object at all (a
     physical-plausibility question, not a reward-design one).
```

- [ ] **Step 2: Commit**

```bash
git add ROADMAP.md
git commit -m "Record AR4 sphere lift-curriculum experiment outcome in ROADMAP"
```

---

## Plan self-review notes

- **Spec coverage:** Task 1 covers the design's "Dense reward function",
  "Reward registration", and "Curriculum" sections exactly (function
  signature, weight values, `num_steps=16800` all copied verbatim from the
  spec). Task 2 covers the "Verification plan" section's full-run half
  (including the pre/post-iteration-700 scalar check the spec specifically
  calls for). Task 3 covers the eval/video half. Task 4 covers the design's
  "if this also fails... flag back to the Principal" instruction, with a
  third (partial-progress) template added since this experiment, unlike
  the strictly-binary ContactSensor one, plausibly has a middle outcome
  worth describing precisely rather than forcing into pass/fail.
- **Single-variable discipline:** confirmed no task touches
  `reaching_sphere`, `lifting_sphere`, `grasp_contact`,
  `sphere_goal_tracking*`, `_EE_OFFSET`, any `ContactSensorCfg`, `CommandsCfg`,
  `EventCfg`, or any PPO hyperparameter — Task 1's diff is additive
  (one new reward term + one new curriculum term + the imports/field
  needed to wire them in).
- **Type/name consistency:** `lift_height_progress(env, height_std,
  rest_height, object_cfg)` signature is identical across its Task 1
  definition and its `RewardsCfg`/`CurriculumCfg` registrations (the
  curriculum term references it only by string `term_name="lift_height_progress"`,
  matching the `RewardsCfg` field name exactly, which is how Isaac Lab's
  `modify_reward_weight` looks it up via `env.reward_manager.get_term_cfg`).
