# AR4 sphere lift — SA-PPO-style LR-bump Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether a late, fixed learning-rate bump (an SA-PPO-style
intervention, per Li et al. *Sensors* 2025) lets the AR4 sphere
pick-and-place policy escape its "reach, grip, freeze" local optimum,
by resuming training from an already-converged checkpoint with the
learning rate bumped and the KL-adaptive schedule disabled.

**Architecture:** A new script, `scripts/train_lr_bump.py`, loads an
existing checkpoint (`model_700.pt` from the just-completed always-on-lift
run) into a fresh `OnPolicyRunner` configured with
`learning_rate=1e-3` and `schedule="fixed"` (vs. the original
`1e-4`/`"adaptive"`), then continues training for 1500 more iterations.
No reward-function changes — this isolates the learning-rate intervention
as the single new variable against an already-fully-characterized reward
config.

**Tech Stack:** Isaac Lab / Isaac Sim (`ManagerBasedRLEnv`), PyTorch,
`rsl_rl` PPO, existing AR4 mk5 task code in `tasks/ar4/`.

## Global Constraints

- Design spec: `docs/superpowers/specs/2026-07-06-ar4-sphere-lift-lr-bump-design.md`
  (read it — this plan implements it exactly).
- Always invoke via `/home/saps/IsaacLab/isaaclab.sh -p scripts/<script>.py`
  from the repo root — never plain `python` for anything importing
  `isaaclab`.
- Decided values, not placeholders: resume from
  `logs/train/2026-07-06_12-24-08/model_700.pt`; `learning_rate=1.0e-3`;
  `schedule="fixed"`; `--max_iterations 1500` (additional, on top of the
  loaded checkpoint's iteration 700, reaching iteration 2200 total).
- Nothing else changes: `entropy_coef` stays `0.006`, the reward function
  (`tasks/ar4/mdp.py`, `RewardsCfg`) is untouched, `scripts/train.py` is
  not modified (this is a new, separate script).
- Verification standard: real evidence over proxies. Confirm the learning
  rate actually held near `1e-3` in the TensorBoard logs (not just assumed
  from the config) before trusting the experiment's conclusion either way.

---

### Task 1: Create `scripts/train_lr_bump.py` and smoke-test the resume mechanics

**Files:**
- Create: `scripts/train_lr_bump.py`

**Interfaces:**
- Produces: a runnable script taking `--checkpoint`, `--num_envs`,
  `--max_iterations`, `--learning_rate` CLI args. No other task consumes
  this as a Python import — it's a standalone entry point, consumed only
  by Task 2 (running it) and Task 3 (running `eval_loop.py` against its
  output checkpoint).

- [ ] **Step 1: Create `scripts/train_lr_bump.py`**

```python
"""Continue AR4 sphere pick-and-place PPO training from an existing
checkpoint with a bumped, fixed learning rate - an SA-PPO-style
intervention (Li et al., Sensors 2025, DOI 10.3390/s25175253) to escape
the "reach, grip, freeze" local optimum every prior experiment this
session has hit. See
docs/superpowers/specs/2026-07-06-ar4-sphere-lift-lr-bump-design.md.

Does NOT change the reward function - reuses whatever checkpoint is
passed in as-is, testing the learning-rate bump in isolation.

.. code-block:: bash

    /home/saps/IsaacLab/isaaclab.sh -p scripts/train_lr_bump.py \\
        --checkpoint logs/train/2026-07-06_12-24-08/model_700.pt \\
        --num_envs 4096 --max_iterations 1500 --headless

    # smoke test:
    /home/saps/IsaacLab/isaaclab.sh -p scripts/train_lr_bump.py \\
        --checkpoint logs/train/2026-07-06_12-24-08/model_700.pt \\
        --num_envs 16 --max_iterations 2 --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Continue AR4 pick-and-place PPO training with a bumped learning rate.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the rsl_rl checkpoint (.pt) to resume from.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of parallel environments.")
parser.add_argument(
    "--max_iterations", type=int, default=1500, help="Additional learning iterations beyond the checkpoint's own iteration count."
)
parser.add_argument("--learning_rate", type=float, default=1.0e-3, help="Bumped, fixed learning rate for this phase.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import sys
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnv
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
    # The bump: fixed (not adaptive) schedule so it isn't corrected back
    # down by KL-divergence feedback from the already-converged policy.
    agent_cfg.algorithm.learning_rate = args_cli.learning_rate
    agent_cfg.algorithm.schedule = "fixed"
    env_cfg.seed = agent_cfg.seed

    log_dir = os.path.join(LOG_ROOT, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)

    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    print(f"Resumed from {args_cli.checkpoint} at iteration {runner.current_learning_iteration}")

    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    runner.learn(num_learning_iterations=args_cli.max_iterations, init_at_random_ep_len=True)

    env.close()
    print(f"Training complete. Checkpoints and logs written to: {log_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
```

- [ ] **Step 2: Smoke test — confirm resume mechanics work**

```bash
cd /home/saps/projects/rl
/home/saps/IsaacLab/isaaclab.sh -p scripts/train_lr_bump.py \
    --checkpoint logs/train/2026-07-06_12-24-08/model_700.pt \
    --num_envs 16 --max_iterations 2 --headless
```

Expected: exits 0, prints `Resumed from
logs/train/2026-07-06_12-24-08/model_700.pt at iteration 700` (confirming
the checkpoint's iteration count was really restored, not reset to 0),
then `Training complete.`. Check the new log directory's
`params/agent.yaml` shows `learning_rate: 0.001` and `schedule: fixed`.

- [ ] **Step 3: Commit**

```bash
git add scripts/train_lr_bump.py
git commit -m "Add SA-PPO-style learning-rate-bump continuation script for AR4 sphere lift"
```

---

### Task 2: Full 1500-iteration continuation run

**Files:** none (no code changes — this task runs the training loop and
inspects its output).

**Interfaces:**
- Consumes: `scripts/train_lr_bump.py` from Task 1,
  `logs/train/2026-07-06_12-24-08/model_700.pt` (pre-existing checkpoint).
- Produces: `logs/train/<timestamp>/model_2199.pt` (0-indexed: resuming
  at iteration 700 and running 1500 more reaches index 2199) and
  TensorBoard event logs, consumed by Task 3.

- [ ] **Step 1: Run the full continuation**

```bash
cd /home/saps/projects/rl
/home/saps/IsaacLab/isaaclab.sh -p scripts/train_lr_bump.py \
    --checkpoint logs/train/2026-07-06_12-24-08/model_700.pt \
    --num_envs 4096 --max_iterations 1500 --headless
```

Expected wall-clock time: roughly 15-25 minutes (same per-iteration cost
as every prior full run at `num_envs=4096`, just resuming from a later
starting point instead of iteration 0).

Note the resulting log directory and confirm `model_2199.pt` exists
(iteration 700 + 1500 additional, 0-indexed final index 2199) for Task 3.

- [ ] **Step 2: Pull the key TensorBoard scalars, including the learning-rate check**

```bash
cd /home/saps/projects/rl
LATEST=$(ls -dt logs/train/*/ | head -1)
/home/saps/IsaacLab/isaaclab.sh -p -c "
from tensorboard.backend.event_processing import event_accumulator
import glob
path = sorted(glob.glob('${LATEST}events.out.tfevents.*'))[-1]
ea = event_accumulator.EventAccumulator(path)
ea.Reload()
for tag in ['Loss/learning_rate', 'Episode_Reward/lifting_sphere', 'Episode_Reward/grasp_contact',
            'Episode_Reward/lift_height_progress', 'Episode_Reward/reaching_sphere',
            'Episode_Termination/sphere_reached_goal']:
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

Record all six lines in the Task 2 report verbatim. **Specifically check
`Loss/learning_rate`'s trajectory**: it must stay near `0.001` throughout
(not decay back toward `0.0001` or lower) — if it drops back down despite
`schedule="fixed"`, that would mean the bump didn't actually hold, and the
experiment's premise (an unconstrained higher LR persists) would be
invalidated regardless of the other results. This is the one check that
determines whether this experiment tested what it claims to.

- [ ] **Step 3: Write the report (create the file)**

Create `docs/superpowers/plans/2026-07-06-ar4-sphere-lift-lr-bump-report.md`
with a "Task 2" section containing: the log directory path, the six
scalar lines from Step 2, and one factual sentence on whether
`Loss/learning_rate` actually held near `0.001` (a prerequisite check,
not a success/failure judgment — that's Task 3/4's job after the eval
video).

```bash
git add docs/superpowers/plans/2026-07-06-ar4-sphere-lift-lr-bump-report.md
git commit -m "Record AR4 sphere LR-bump full continuation run results"
```

---

### Task 3: Real eval + video inspection (decision gate)

**Files:** none (no code changes — this task runs eval and visually
inspects output).

**Interfaces:**
- Consumes: `logs/train/<timestamp>/model_2199.pt` from Task 2.
- Produces: eval videos in `logs/videos/`, a final pass/fail verdict
  consumed by Task 4.

- [ ] **Step 1: Run eval for 10 episodes**

```bash
cd /home/saps/projects/rl
/home/saps/IsaacLab/isaaclab.sh -p scripts/eval_loop.py --checkpoint logs/train/<RUN_DIR>/model_2199.pt --episodes 10
```

(substitute the actual `<RUN_DIR>` from Task 2, and the actual final
checkpoint index if it differs from 2199). Expected: 10 files
`logs/videos/ar4_pickplace-step-0.mp4` through `-step-2250.mp4` (these
overwrite the prior experiment's eval videos of the same name — that's
fine).

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
point, and if so, does it stay lifted?

- [ ] **Step 4: Apply the decision gate**

- **If 8/10 or more episodes show the sphere genuinely lifted and
  carried toward the target:** success. Proceed to Task 4's success
  path.
- **If fewer than 8/10 do, but some episodes show real (even brief)
  lifting that never happened in prior experiments:** partial progress —
  describe precisely what's different.
- **If 0/10 show any lift (same "reach, grip, freeze" signature):** this
  experiment is falsified. Per the user's "try both" instruction, this
  does NOT block moving to the second planned experiment (potential-based
  reward shaping) — note that clearly in the report and proceed to Task 4
  regardless.

- [ ] **Step 5: Commit the report update**

```bash
git add docs/superpowers/plans/2026-07-06-ar4-sphere-lift-lr-bump-report.md
git commit -m "Record AR4 sphere LR-bump eval video inspection results"
```

---

### Task 4: Record the outcome in `ROADMAP.md`

**Files:**
- Modify: `ROADMAP.md`

**Interfaces:** none (documentation-only task, terminal in this plan).

- [ ] **Step 1: Write the outcome into `ROADMAP.md`'s "grasp/lift never
  emerges" follow-up**

Add a new bullet under the existing entry (after the always-on-dense-
reward sub-bullet), following the same evidentiary detail as the existing
sub-bullets. Use whichever template applies, filling in real numbers from
Tasks 2-3:

**If Task 3's decision gate passed (8/10+):**

```markdown
   - **Follow-up experiment: SA-PPO-style learning-rate bump (SUCCESS).**
     Resumed training from the always-on-lift run's `model_700.pt`
     (grip already converged) with `learning_rate` bumped `1e-4`->`1e-3`
     and `schedule` switched `"adaptive"`->`"fixed"` (the adaptive
     schedule would otherwise claw the bump back down given the
     converged policy's low KL divergence), per
     `docs/superpowers/specs/2026-07-06-ar4-sphere-lift-lr-bump-design.md`.
     No reward-function changes - this isolated the learning-rate
     intervention alone. Full run data:
     `docs/superpowers/plans/2026-07-06-ar4-sphere-lift-lr-bump-report.md`.
     Confirmed `Loss/learning_rate` held near `0.001` throughout (the
     fixed schedule worked as intended). **Result: [X]/10 real eval
     episodes show the sphere genuinely lifted and carried toward the
     target.** This resolves the "grasp/lift never emerges" follow-up.
```

**If Task 3's decision gate did not pass (0/10 or partial):**

```markdown
   - **Follow-up experiment: SA-PPO-style learning-rate bump
     ([falsified | partial progress]).** Resumed training from the
     always-on-lift run's `model_700.pt` (grip already converged) with
     `learning_rate` bumped `1e-4`->`1e-3` and `schedule` switched
     `"adaptive"`->`"fixed"`, per
     `docs/superpowers/specs/2026-07-06-ar4-sphere-lift-lr-bump-design.md` -
     the strongest literature-verified candidate from this session's
     research (Li et al., *Sensors* 2025, DOI 10.3390/s25175253).
     Full run data:
     `docs/superpowers/plans/2026-07-06-ar4-sphere-lift-lr-bump-report.md`.
     [State plainly whether `Loss/learning_rate` actually held near
     `0.001` throughout, per Task 2's check - if it didn't, say so
     explicitly, since that would mean this experiment didn't actually
     test what it intended to.] **Result: [X]/10 real eval episodes show
     any lift** — [one to two sentences on what the video actually
     showed]. This is the fourth real attempt on the reward/optimization
     axis for this sub-problem. Per the user's "try both" instruction,
     proceeding directly to the second planned experiment (potential-based
     reward shaping) rather than pausing here.
```

- [ ] **Step 2: Commit**

```bash
git add ROADMAP.md
git commit -m "Record AR4 sphere LR-bump experiment outcome in ROADMAP"
```

---

## Plan self-review notes

- **Spec coverage:** Task 1 covers the design's script + smoke test.
  Task 2 covers the full continuation run + the critical learning-rate-
  held check the spec calls out as load-bearing. Task 3 covers the eval/
  video half. Task 4 covers recording the outcome either way, explicitly
  not gating the next experiment on this one's result (per the design's
  own "proceed directly to the second planned experiment" instruction).
- **Single-variable discipline:** confirmed no task touches
  `tasks/ar4/mdp.py`, `RewardsCfg`, `entropy_coef`, `scripts/train.py`, or
  any other existing script. Only `scripts/train_lr_bump.py` is new.
- **Type/name consistency:** `--checkpoint`, `--num_envs`,
  `--max_iterations`, `--learning_rate` CLI args are used identically in
  Task 1's script and Task 2's/Task 3's invocation commands.
