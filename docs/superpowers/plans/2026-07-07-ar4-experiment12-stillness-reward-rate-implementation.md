# Experiment 12 Implementation Plan: Antipodal/Stillness Reward-Rate Fix

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Experiment 12 — fix a verified reward-rate bug in the AR4 cube task-space pick-and-place task (`tasks/ar4/pickplace_taskspace_env_cfg.py`): holding a grasp without further progress currently nets **+1.0/step** (`antipodal_grasp_bonus` +3.0 vs. `stillness_penalty` -2.0 once its patience window elapses), which is the direct, quantitative explanation for Experiment 11's observed "reach, grasp, freeze" failure. Raising `stillness_penalty`'s weight to 5.0 flips this to **-2.0/step**, removing the incentive to freeze after grasp.

**Architecture:** Single-parameter change (one `weight=` value in one `RewardsCfg`), followed by the standard verification sequence this repo uses for every reward change: a short diagnostic run to sanity-check scalar trends, then a full 1500-iteration run, then real eval-video inspection (not scalar trends alone), then a ROADMAP record.

**Tech Stack:** Isaac Lab `ManagerBasedRLEnvCfg`, rsl_rl PPO (`Ar4PickPlaceTaskspacePPORunnerCfg`, unchanged), TensorBoard event files for scalar verification.

## Global Constraints

- Do not modify `tasks/ar4/mdp.py`'s `stillness_penalty` or `antipodal_grasp_bonus` function bodies — only the `weight=` parameter passed to `stillness_penalty`'s `RewTerm` in `tasks/ar4/pickplace_taskspace_env_cfg.py` changes.
- Do not modify `pickplace_mirror_env_cfg.py` or `pickplace_ik_guided_env_cfg.py` — this fix is scoped to the task-space experiment only.
- Do not modify `antipodal_grasp_bonus`'s weight (stays 3.0), `patience_steps` (stays 25), or `still_bound` (stays 0.005) — this experiment isolates the single `stillness_penalty` weight variable per the spec's "isolate one verified bug" rationale.
- Decided value (verbatim, not a placeholder): `stillness_penalty` weight `2.0` → `5.0`.
- Always invoke scripts via `/home/saps/IsaacLab/isaaclab.sh -p scripts/<script>.py` (or `/home/saps/IsaacLab/isaaclab.sh -p -c "<snippet>"` for one-off Python) from the repo root, never plain `python`.
- Verify completion via files (checkpoint counts, `model_<N>.pt` existence, event-file mtimes) and actual TensorBoard scalar data, not console text or exit codes alone — this session's established practice against false positives from Isaac Sim's unreliable clean-shutdown/stdout behavior.
- `--taskspace` is already wired into `scripts/train.py` and `scripts/eval_loop.py` (Experiment 11) — no CLI changes needed in this plan.

---

### Task 1: Raise `stillness_penalty`'s weight

**Files:**
- Modify: `tasks/ar4/pickplace_taskspace_env_cfg.py:240-251` (the `stillness_penalty` `RewTerm` inside `RewardsCfg`)

**Interfaces:**
- Consumes: nothing new — same `ar4_mdp.stillness_penalty` function, same `params`.
- Produces: `Ar4PickPlaceTaskspaceEnvCfg` with the updated weight, consumed by Task 2's diagnostic run.

- [ ] **Step 1: Edit the `RewardsCfg` docstring**

In `tasks/ar4/pickplace_taskspace_env_cfg.py`, the `RewardsCfg` class docstring currently reads (starting at line 198):

```python
@configclass
class RewardsCfg:
    """path_proximity_bonus replaces ik_guided_path_bonus (drops the
    now-redundant IK-action-matching sub-signal, since IK is now part of
    the control loop itself - see path_proximity_bonus's docstring).
    antipodal_grasp_bonus (Experiment 10's physics-corrected -0.7071
    threshold), gripper_schedule_bonus, stillness_penalty, action_rate,
    and joint_vel all carry over unchanged from
    pickplace_ik_guided_env_cfg.py - this experiment isolates the
    action-space variable specifically."""
```

Replace the docstring's final sentence to reflect Experiment 12's change:

```python
@configclass
class RewardsCfg:
    """path_proximity_bonus replaces ik_guided_path_bonus (drops the
    now-redundant IK-action-matching sub-signal, since IK is now part of
    the control loop itself - see path_proximity_bonus's docstring).
    antipodal_grasp_bonus (Experiment 10's physics-corrected -0.7071
    threshold), gripper_schedule_bonus, action_rate, and joint_vel carry
    over unchanged from pickplace_ik_guided_env_cfg.py. stillness_penalty's
    weight is raised from 2.0 to 5.0 (Experiment 12): Experiment 11 showed
    grasp-and-freeze nets +1.0/step (antipodal_grasp_bonus's +3.0 minus the
    old stillness_penalty's -2.0 once its patience window elapses) - this
    flips that to -2.0/step. See
    docs/superpowers/specs/2026-07-07-ar4-experiment12-stillness-reward-rate-design.md."""
```

- [ ] **Step 2: Edit the `stillness_penalty` weight and add an explanatory comment**

The current block (lines 240-251):

```python
    stillness_penalty = RewTerm(
        func=ar4_mdp.stillness_penalty,
        weight=2.0,
        params={
            "object_cfg": SceneEntityCfg("cube"),
            "jaw1_contact_cfg": SceneEntityCfg("gripper_jaw1_contact"),
            "jaw2_contact_cfg": SceneEntityCfg("gripper_jaw2_contact"),
            "force_threshold": 0.05,
            "still_bound": 0.005,
            "patience_steps": 25,
        },
    )
```

Replace with:

```python
    # weight raised 2.0 -> 5.0 (Experiment 12): with antipodal_grasp_bonus's
    # weight at 3.0, the old weight=2.0 only closed the reward-rate gap to
    # +3.0 - 2.0 = +1.0/step net POSITIVE for holding a grasp without
    # progress once patience_steps elapses - it never flipped the sign.
    # 5.0 makes it 3.0 - 5.0 = -2.0/step net negative. See
    # docs/superpowers/specs/2026-07-07-ar4-experiment12-stillness-reward-rate-design.md.
    stillness_penalty = RewTerm(
        func=ar4_mdp.stillness_penalty,
        weight=5.0,
        params={
            "object_cfg": SceneEntityCfg("cube"),
            "jaw1_contact_cfg": SceneEntityCfg("gripper_jaw1_contact"),
            "jaw2_contact_cfg": SceneEntityCfg("gripper_jaw2_contact"),
            "force_threshold": 0.05,
            "still_bound": 0.005,
            "patience_steps": 25,
        },
    )
```

- [ ] **Step 3: Syntax-check the file**

Run: `python3 -c "import ast; ast.parse(open('tasks/ar4/pickplace_taskspace_env_cfg.py').read())"`
Expected: no output (parses cleanly).

- [ ] **Step 4: Commit**

```bash
git add tasks/ar4/pickplace_taskspace_env_cfg.py
git commit -m "Raise stillness_penalty weight 2.0->5.0 for Experiment 12 (reward-rate fix)"
```

---

### Task 2: Diagnostic run (300 iterations) — verify the incentive flip before committing to a full run

**Files:**
- None created — this is a verification-only task. If the diagnostic looks wrong, stop and report; do not proceed to Task 3.

**Interfaces:**
- Consumes: `Ar4PickPlaceTaskspaceEnvCfg` with the Task 1 weight change.
- Produces: a confirmed-healthy diagnostic run directory under `logs/train/`, gating Task 3.

- [ ] **Step 1: Launch the diagnostic run**

Run (from repo root, background):
```bash
nohup /home/saps/IsaacLab/isaaclab.sh -p scripts/train.py --taskspace --num_envs 4096 --max_iterations 300 --headless > /tmp/exp12_diagnostic_stdout.log 2>&1 &
echo "PID=$!"
```

- [ ] **Step 2: Poll for completion via files**

```bash
find logs/train/ -name "model_299.pt" -newer tasks/ar4/pickplace_taskspace_env_cfg.py
```

Expected: this appears once the run completes (poll every few minutes rather than assuming a fixed wall-clock time; do not trust stdout alone for completion, per this session's established practice with Isaac Sim's unreliable clean-shutdown behavior).

- [ ] **Step 3: Extract and check the diagnostic scalars**

```bash
LATEST=$(ls -dt logs/train/*/ | head -1)
/home/saps/IsaacLab/isaaclab.sh -p -c "
from tensorboard.backend.event_processing import event_accumulator
import glob
path = sorted(glob.glob('${LATEST}events.out.tfevents.*'))[-1]
ea = event_accumulator.EventAccumulator(path)
ea.Reload()
for tag in ['Episode_Reward/antipodal_grasp_bonus', 'Episode_Reward/stillness_penalty',
            'Episode_Reward/path_proximity_bonus', 'Episode_Reward/gripper_schedule_bonus',
            'Episode_Termination/cube_reached_goal']:
    if tag in ea.Tags()['scalars']:
        vals = ea.Scalars(tag)
        nonzero = sum(1 for v in vals if v.value != 0.0)
        print(tag, '-> first:', vals[0].value, 'last:', vals[-1].value, 'max:', max(v.value for v in vals),
              'min:', min(v.value for v in vals), 'nonzero:', nonzero, '/', len(vals))
    else:
        print(tag, '-> NOT FOUND')
"
```

- [ ] **Step 4: Evaluate the diagnostic against these three checks**

1. **`Episode_Reward/antipodal_grasp_bonus` is not identically zero across the run.** If it stays at 0.0 the entire diagnostic, the stillness_penalty change may have suppressed grasping entirely (over-correction) rather than just freezing — stop and report this as an unexpected regression rather than proceeding to Task 3.
2. **`Episode_Reward/stillness_penalty`'s magnitude (min value, since it's negative) is more negative on average than Experiment 11's full-run values** (Experiment 11's `stillness_penalty` ranged roughly -0.011 to 0.0, see `docs/superpowers/plans/2026-07-06-ar4-experiment11-report.md`). A comparably-sized or smaller magnitude would suggest the weight increase isn't actually being exercised more (e.g., because the policy stopped freezing without needing the penalty, or because grasping got rarer) — note whichever it is in the report either way; this alone doesn't gate Task 3.
3. **No exceptions/tracebacks in `/tmp/exp12_diagnostic_stdout.log`.** Run `grep -i "error\|exception\|traceback" /tmp/exp12_diagnostic_stdout.log` — if this finds anything beyond expected Isaac Sim startup/shutdown noise, read the context and fix before proceeding.

If check 1 and check 3 both pass, proceed to Task 3 regardless of check 2's specific direction (it's informational, not a gate — the real test of whether the incentive flip worked is the full run's `path_proximity_bonus` waypoint-≥2 crediting and, ultimately, eval video). If check 1 fails, stop, do not proceed to Task 3, and report the finding instead — this would be a genuinely new, useful result (over-correction) worth recording before further tuning.

---

### Task 3: Full 1500-iteration training run + TensorBoard verification report

**Files:**
- Create: `docs/superpowers/plans/2026-07-07-ar4-experiment12-report.md`

**Interfaces:**
- Consumes: the Task 2-verified reward change.
- Produces: a complete 1500-iteration training run under `logs/train/`, and a written report with full scalar trajectories.

- [ ] **Step 1: Launch the full training run**

```bash
nohup /home/saps/IsaacLab/isaaclab.sh -p scripts/train.py --taskspace --num_envs 4096 --headless > /tmp/exp12_train_stdout.log 2>&1 &
echo "PID=$!"
```

- [ ] **Step 2: Verify completion via files**

```bash
find logs/train/ -name "model_1499.pt" -newer tasks/ar4/pickplace_taskspace_env_cfg.py
```

Once found, confirm checkpoint integrity:
```bash
LATEST=$(ls -dt logs/train/*/ | head -1)
find "$LATEST" -name "model_*.pt" | wc -l
ls -la "${LATEST}"events.out.tfevents.*
```

Expected: 31 checkpoints (0, 50, 100, ..., 1450, 1499 — `save_interval=50`, matching every prior experiment this session), `model_1499.pt` exists, event file mtime matches the run's actual completion time.

- [ ] **Step 3: Extract full scalar trajectories**

```bash
LATEST=$(ls -dt logs/train/*/ | head -1)
/home/saps/IsaacLab/isaaclab.sh -p -c "
from tensorboard.backend.event_processing import event_accumulator
import glob
path = sorted(glob.glob('${LATEST}events.out.tfevents.*'))[-1]
ea = event_accumulator.EventAccumulator(path)
ea.Reload()
tags = ['Episode_Reward/path_proximity_bonus', 'Episode_Reward/gripper_schedule_bonus',
        'Episode_Reward/antipodal_grasp_bonus', 'Episode_Reward/stillness_penalty',
        'Episode_Termination/cube_reached_goal', 'Loss/value_function']
for tag in tags:
    if tag in ea.Tags()['scalars']:
        vals = ea.Scalars(tag)
        nonzero = sum(1 for v in vals if v.value != 0.0)
        print(f'=== {tag} ===')
        print('  points:', len(vals), 'first:', vals[0].value, 'last:', vals[-1].value,
              'max:', max(v.value for v in vals), 'min:', min(v.value for v in vals),
              'nonzero:', nonzero, '/', len(vals))
        for i in range(0, len(vals), 150):
            print(f'  iteration={vals[i].step:4d}, value={vals[i].value:.6f}')
        print(f'  iteration={vals[-1].step:4d}, value={vals[-1].value:.6f}')
    else:
        print(tag, '-> NOT FOUND')
"
```

- [ ] **Step 4: Write the report**

Write `docs/superpowers/plans/2026-07-07-ar4-experiment12-report.md` following the structure of `docs/superpowers/plans/2026-07-06-ar4-experiment11-report.md` (checkpoint integrity, `Loss/value_function` sanity check — should stay bounded same as Experiment 11 since the action space is unchanged, per-term summary + sampled trajectory for all 5 reward/termination tags above). Include a "Key Comparison" section against **Experiment 11's exact final values** (final-iteration snapshot vs. final-iteration snapshot only, per this project's established correction protocol against cumulative-vs-single-episode comparison errors):

- Experiment 11 final `Episode_Reward/antipodal_grasp_bonus`: 0.018815
- Experiment 11 final `Episode_Reward/stillness_penalty`: -0.002533
- Experiment 11 final `Episode_Termination/cube_reached_goal`: 0.010223

State plainly whether `path_proximity_bonus` shows evidence of the policy reaching waypoint index ≥2 (lift) more than Experiment 11 (this can't be read directly from `path_proximity_bonus`'s scalar alone since it's a running-max delta across all 5 waypoints combined — note in the report that a definitive per-waypoint read requires the video inspection in Task 4, and don't overclaim from this scalar alone). Do not draw a final success/failure conclusion here — that requires Task 4's video inspection.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/plans/2026-07-07-ar4-experiment12-report.md
git commit -m "Record Experiment 12 training run: antipodal/stillness reward-rate fix scalar trajectories"
```

---

### Task 4: Real eval + video inspection, ROADMAP record

**Files:**
- Modify: `ROADMAP.md` (append Experiment 12's outcome, following the existing format for Experiments 9-11)

**Interfaces:**
- Consumes: `model_1499.pt` from Task 3's training run.

- [ ] **Step 1: Run eval with video recording**

```bash
LATEST=$(ls -dt logs/train/*/ | head -1)
/home/saps/IsaacLab/isaaclab.sh -p scripts/eval_loop.py --taskspace --checkpoint "${LATEST}model_1499.pt" --episodes 10
```

Verify the output video file exists under `logs/videos/` (named `ar4_pickplace_taskspace...mp4`) before treating this as complete.

- [ ] **Step 2: Watch the video frame-by-frame**

Extract frames (e.g. at 5fps across a full ~5s episode, matching Experiment 11's inspection density) and inspect them directly (via the Read tool or equivalent), not just file existence. Specifically check for:

1. **Genuine lift**: does the cube visibly leave the ground and stay elevated for more than an instantaneous contact (the specific symptom this experiment targets)?
2. **Carry/place progress**: does the arm move the cube toward the goal region after any lift, or does it lift-and-freeze (a new, different failure mode than Experiment 11's ground-level freeze)?
3. **The in-place-wiggle evasion pattern flagged in the design spec**: does the gripper/cube show small oscillating displacement without net height/position progress, rather than a clean freeze or a clean lift? This is the specific unaddressed loophole noted in `docs/superpowers/specs/2026-07-07-ar4-experiment12-stillness-reward-rate-design.md`'s "What this fix does NOT address" section.

- [ ] **Step 3: Record the outcome in `ROADMAP.md`**

Append a new entry after the Experiment 11 entry, in the same format used for Experiments 9-11 (hypothesis, what changed, quantitative result from Task 3's report, qualitative video finding from Step 2, and an explicit statement of whether "grasp/lift never emerges" is now resolved, improved-but-unresolved, or unchanged/regressed — plus, if the wiggle-evasion pattern from Step 2.3 is observed, flag it explicitly as a new finding requiring its own follow-up rather than folding it into this experiment's read).

- [ ] **Step 4: Commit and push**

```bash
git add ROADMAP.md
git commit -m "Record Experiment 12 outcome (antipodal/stillness reward-rate fix) in ROADMAP"
git push origin main
```
