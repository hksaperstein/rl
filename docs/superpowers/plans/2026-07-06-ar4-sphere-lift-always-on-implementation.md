# AR4 sphere lift — always-on dense reward (drop curriculum) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the curriculum-gated lift reward's diagnosed failure (the
incentive turned on too late, after the static-grip behavior had already
entrenched) by making `lift_height_progress` active from iteration 0
instead of gated behind a curriculum switch, and raising its weight to
match `lifting_sphere`.

**Architecture:** Remove `CurriculumCfg` and the `curriculum` field from
`Ar4PickPlaceEnvCfg` entirely (this experiment introduced both, nothing
else uses them). Change `lift_height_progress`'s `RewTerm` weight from
`0.0` directly to `25.0`. No other file changes.

**Tech Stack:** Isaac Lab / Isaac Sim (`ManagerBasedRLEnv`), PyTorch,
`rsl_rl` PPO, existing AR4 mk5 task code in `tasks/ar4/`.

## Global Constraints

- Design spec: `docs/superpowers/specs/2026-07-06-ar4-sphere-lift-curriculum-design.md`'s
  "Revision" section (read it — this plan implements that revision
  exactly).
- Always invoke via `/home/saps/IsaacLab/isaaclab.sh -p scripts/<script>.py`
  (or `-p -m pytest ...` / `-p -c "..."`) from the repo root — never plain
  `python` for anything importing `isaaclab`.
- Decided values, not placeholders: `lift_height_progress` weight `25.0`
  (was `0.0` with a curriculum ramp to `15.0` — both removed).
  `height_std=0.01` and `rest_height=0.009` are unchanged.
- Every other reward term (`reaching_sphere`, `lifting_sphere`,
  `grasp_contact`, `sphere_goal_tracking`, `sphere_goal_tracking_fine_grained`,
  `action_rate`, `joint_vel`), `_EE_OFFSET=0.036`, every
  observation/command/event/`ContactSensorCfg`, and every PPO
  hyperparameter must remain byte-for-byte unchanged.
- Verification standard: real evidence over proxies. Don't call a task
  done off exit codes alone — read the actual TensorBoard scalars and look
  at the actual eval video frames.

---

### Task 1: Remove the curriculum, raise `lift_height_progress`'s weight

**Files:**
- Modify: `tasks/ar4/pickplace_env_cfg.py`

**Interfaces:** none new — `lift_height_progress`'s function signature and
params (`height_std`, `rest_height`, `object_cfg`) are unchanged from the
prior experiment; only its `weight` and the removal of the curriculum
change.

- [ ] **Step 1: Change `lift_height_progress`'s weight to `25.0`**

In `tasks/ar4/pickplace_env_cfg.py`'s `RewardsCfg`, change:

```python
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
```

to:

```python
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
```

- [ ] **Step 2: Remove the `CurriculumCfg` class**

In the same file, delete this entire class (it sits between `RewardsCfg`
and `TerminationsCfg`):

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

(Delete the whole block including the blank lines around it, so
`RewardsCfg` is followed directly by `TerminationsCfg` again.)

- [ ] **Step 3: Remove the now-unused imports**

Change:

```python
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.curriculums import modify_reward_weight
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
```

to:

```python
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
```

- [ ] **Step 4: Remove the `curriculum` field from `Ar4PickPlaceEnvCfg`**

Change:

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

to:

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
```

(No `curriculum` field at all — `ManagerBasedRLEnvCfg`'s own default is
`curriculum: object | None = None`, i.e. no curriculum applied, which is
exactly what's wanted now.)

- [ ] **Step 5: Smoke test**

```bash
cd /home/saps/projects/rl
/home/saps/IsaacLab/isaaclab.sh -p scripts/train.py --num_envs 16 --max_iterations 2 --headless
```

Expected: exits 0, prints `Training complete.`. In the startup printout,
`Active Reward Terms` shows `lift_height_progress` with weight `25.0`
(not `0.0`), and `Active Curriculum Terms` is back to empty (matching how
it looked before this whole lift-curriculum experiment started).

- [ ] **Step 6: Commit**

```bash
git add tasks/ar4/pickplace_env_cfg.py
git commit -m "Make lift_height_progress active from iteration 0, drop curriculum gate"
```

---

### Task 2: Full training run

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

Expected wall-clock time: roughly 15-25 minutes at `num_envs=4096`, based
on both prior experiments' full runs at the same scale (~16 minutes each).

Note the resulting log directory (`logs/train/<timestamp>/`) and confirm
`model_1499.pt` exists (1500 iterations, 0-indexed) for Task 3.

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
for tag in ['Episode_Reward/lift_height_progress', 'Episode_Reward/lifting_sphere',
            'Episode_Reward/grasp_contact', 'Episode_Reward/reaching_sphere',
            'Episode_Termination/sphere_reached_goal']:
    if tag in ea.Tags()['scalars']:
        vals = ea.Scalars(tag)
        # sample roughly every 10% of iterations to see the trajectory, not just endpoints
        n = len(vals)
        samples = [vals[i].value for i in range(0, n, max(1, n // 10))]
        print(tag, '-> first:', vals[0].value, 'last:', vals[-1].value, 'max:', max(v.value for v in vals),
              'trajectory (10 samples):', [round(s, 4) for s in samples])
    else:
        print(tag, '-> NOT FOUND')
"
```

Record these five lines in the Task 2 report verbatim. Unlike the
curriculum experiment (where a pre/post-switch check made sense), this
run has no switch point — instead, the trajectory samples let you see
whether `lift_height_progress` grows meaningfully at any point during
training, not just its final value.

- [ ] **Step 3: Write the report (create the file)**

Create `docs/superpowers/plans/2026-07-06-ar4-sphere-lift-always-on-report.md`
with a "Task 2" section containing: the log directory path, the five
scalar lines from Step 2, and one factual sentence on whether
`lift_height_progress` reached a meaningfully larger value than the
curriculum experiment's `0.0065` max at any point (a necessary but not
sufficient check — Task 3's eval video confirms whether it's a real lift).

```bash
git add docs/superpowers/plans/2026-07-06-ar4-sphere-lift-always-on-report.md
git commit -m "Record AR4 sphere always-on lift reward full training run results"
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

Use the Read tool to view frames from each of the 10 episode directories
(start, ~25%, ~50%, ~75%, end is a good baseline sample; if any frame
shows the sphere marker (blue) missing or in an unexpected position,
check adjacent frames in that same episode before concluding anything —
the sphere can be briefly occluded by the gripper body at some camera
angles without actually having been lifted, as happened in the prior
experiment). For each episode, determine: does the sphere visibly leave
the ground at any point, and if so, does it stay lifted or immediately
drop back down?

- [ ] **Step 4: Apply the decision gate**

- **If 8/10 or more episodes show the sphere genuinely lifted (leaves the
  ground and stays up, not a single-frame bounce or camera-angle
  occlusion) and carried toward the target:** this is a successful fix.
  Proceed to Task 4's success path.
- **If fewer than 8/10 do, but at least some episodes show real (even if
  brief or inconsistent) lifting that never happened in either prior
  experiment:** this is partial progress, not a clean success or a flat
  repeat — describe precisely what's now different rather than forcing it
  into either bucket.
- **If 0/10 episodes show any lift at all (same "reach, grip, freeze"
  signature as both prior experiments):** the always-on dense term did not
  work either. This would be the third real attempt on the reward/
  curriculum axis for this specific sub-problem (sparse-only, curriculum-
  gated dense, always-on dense) — per the design doc's own instruction, do
  not attempt a fourth reward-only tweak unilaterally. Flag back to the
  Principal/user.

- [ ] **Step 5: Commit the report update**

Append a "Task 3" section to
`docs/superpowers/plans/2026-07-06-ar4-sphere-lift-always-on-report.md`
with: the episode-by-episode observations (or a summary if patterns
repeat across most episodes), the X/10 count, and which branch of Step 4
applies.

```bash
git add docs/superpowers/plans/2026-07-06-ar4-sphere-lift-always-on-report.md
git commit -m "Record AR4 sphere always-on lift reward eval video inspection results"
```

---

### Task 4: Record the outcome in `ROADMAP.md`

**Files:**
- Modify: `ROADMAP.md`

**Interfaces:** none (documentation-only task, terminal in this plan).

- [ ] **Step 1: Write the outcome into `ROADMAP.md`'s "grasp/lift never
  emerges" follow-up**

Open `ROADMAP.md` and add a new bullet under the existing "grasp/lift
never emerges" entry (after the curriculum-gated sub-bullet, which ends
with "...a physical-plausibility question, not a reward-design one."),
following the exact same level of evidentiary detail as the existing
sub-bullets. Use whichever of the three templates below applies, filling
in the actual measured numbers and observations from Tasks 2-3 (not
placeholder text):

**If Task 3's decision gate passed (8/10+):**

```markdown
   - **Follow-up experiment: always-on dense lift-height reward
     (SUCCESS).** The curriculum-gated version's diagnosed failure (the
     incentive turned on too late, after the static-grip behavior had
     already entrenched) was fixed by removing the curriculum entirely -
     `lift_height_progress` active from iteration 0 at weight `25.0`
     (matching `lifting_sphere`), per
     `docs/superpowers/specs/2026-07-06-ar4-sphere-lift-curriculum-design.md`'s
     "Revision" section. `lift_height_progress`/`lifting_sphere` reward:
     [Task 2's values]. Full run data:
     `docs/superpowers/plans/2026-07-06-ar4-sphere-lift-always-on-report.md`.
     **Result: [X]/10 real eval episodes show the sphere genuinely lifted
     off the ground and carried toward the target.** This resolves the
     "grasp/lift never emerges" follow-up.
```

**If Task 3's decision gate showed partial progress:**

```markdown
   - **Follow-up experiment: always-on dense lift-height reward — partial
     progress, not yet a full success.** Removed the curriculum-gated
     version's timing gate entirely - `lift_height_progress` active from
     iteration 0 at weight `25.0`, per
     `docs/superpowers/specs/2026-07-06-ar4-sphere-lift-curriculum-design.md`'s
     "Revision" section. `lift_height_progress`/`lifting_sphere` reward:
     [Task 2's values]. Full run data:
     `docs/superpowers/plans/2026-07-06-ar4-sphere-lift-always-on-report.md`.
     **Result: [X]/10 episodes show real lifting** — [precise description
     of what the video actually showed]. Real movement beyond both prior
     experiments' flat signatures, but not yet reliable. Flagged back to
     the Principal/user rather than continuing unilaterally.
```

**If Task 3's decision gate did not pass (0/10, same signature as before):**

```markdown
   - **Follow-up experiment: always-on dense lift-height reward
     (falsified).** Removed the curriculum-gated version's timing gate
     entirely - `lift_height_progress` active from iteration 0 at weight
     `25.0`, per
     `docs/superpowers/specs/2026-07-06-ar4-sphere-lift-curriculum-design.md`'s
     "Revision" section, to rule out "the curriculum turned on too late"
     as the explanation. `lift_height_progress`/`lifting_sphere` reward:
     [Task 2's values]. Full run data:
     `docs/superpowers/plans/2026-07-06-ar4-sphere-lift-always-on-report.md`.
     **Result: 0/10 real eval episodes show any lift** — [one to two
     sentences on what the video actually showed]. This is the third real
     attempt on the reward/curriculum axis for this sub-problem (sparse-
     only, curriculum-gated dense, always-on dense) - per
     `superpowers:systematic-debugging` Phase 4.5, the next step should
     not be a fourth reward-only tweak. Flagged back to the Principal/user;
     remaining candidates are the hierarchical reach-then-grasp-policy
     split, or the physical-plausibility check on whether the gripper's
     real closed-jaw force can support lifting this object at all.
```

- [ ] **Step 2: Commit**

```bash
git add ROADMAP.md
git commit -m "Record AR4 sphere always-on lift reward experiment outcome in ROADMAP"
```

---

## Plan self-review notes

- **Spec coverage:** Task 1 covers the design revision's "Design changes"
  section exactly (weight 0.0→25.0 directly, `CurriculumCfg`/`curriculum`
  field/unused imports all removed). Task 2 covers the "Verification plan"
  section's full-run half (trajectory sampling instead of a pre/post-switch
  check, since there's no switch point anymore). Task 3 covers the eval/
  video half, explicitly carrying forward the occlusion-vs-lift lesson
  from the prior experiment's Task 3. Task 4 covers the "flag back to the
  Principal" instruction with three templates.
- **Single-variable discipline:** confirmed no task touches
  `reaching_sphere`, `lifting_sphere`, `grasp_contact`,
  `sphere_goal_tracking*`, `_EE_OFFSET`, any `ContactSensorCfg`,
  `CommandsCfg`, `EventCfg`, or any PPO hyperparameter — Task 1's diff
  changes exactly one weight value and removes exactly the curriculum
  machinery the prior experiment added (a clean revert of that one piece,
  not a partial rollback of anything else).
- **Type/name consistency:** `lift_height_progress`'s function signature
  and params (`height_std=0.01`, `rest_height=0.009`, `object_cfg`) are
  unchanged from the prior experiment across every reference in this plan.
