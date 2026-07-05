# RL Repo Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract `6DoF/rl/` (plus its design docs and subagent-driven-dev ledger) into a new, private, history-preserving GitHub repo `hksaperstein/rl`, flattened to the new repo's root, then remove the migrated content from `6DoF`.

**Architecture:** `git-filter-repo` rewrites a disposable single-branch clone of `6DoF` down to only the commits touching `rl/`, `docs/superpowers/`, and `.superpowers/sdd/`, renaming `rl/*` to the new repo's root. The gitignored `rl/assets/` and `rl/logs/` directories (never in git history) are copied over as plain files afterward. `6DoF` then gets one new commit removing the migrated paths — no history rewrite on the `6DoF` side.

**Tech Stack:** git, git-filter-repo (already installed via `pip install --user --break-system-packages git-filter-repo`), GitHub CLI (`gh`, already authenticated as `hksaperstein`).

## Global Constraints

- New repo name: `hksaperstein/rl`, private visibility.
- History of the ~27 commits touching `rl/`, `docs/superpowers/`, `.superpowers/sdd/` must be preserved (authorship + dates intact) — confirmed clean split, only one commit (`4a9fc78`, the IK node) touches neither path.
- `rl/`'s contents flatten to the new repo's root (no nested `rl/` folder inside a repo already named `rl`).
- `docs/superpowers/{specs,plans}/` and `.superpowers/sdd/` keep their existing relative paths in the new repo.
- No changes to `os.path.dirname(...)` chains anywhere — verified directly (see spec `docs/superpowers/specs/2026-07-05-rl-repo-extraction-design.md`) that these are self-referential within the `rl/` subtree and resolve correctly unchanged after the flatten.
- `6DoF`'s existing commit history is never rewritten; removal happens via one new forward commit on `ar4-mk5-ik-node-2xipt1`.
- Local working copy of the new repo: `~/projects/rl`.

---

### Task 1: Extract filtered history into a disposable clone

**Files:**
- Create (outside both repos): `/tmp/rl-extract/` (disposable working clone, deleted at the end of this task's verification — its useful output is the repo pushed in Task 2)

**Interfaces:**
- Produces: a local git repo at `/tmp/rl-extract` whose `HEAD` (branch `ar4-mk5-ik-node-2xipt1`) contains only the commits touching `rl/`, `docs/superpowers/`, `.superpowers/sdd/` from `6DoF`, with `rl/<X>` renamed to `<X>` (flattened) and `docs/superpowers/`, `.superpowers/sdd/` left at their existing paths. Task 2 pushes this repo's history to GitHub.

- [ ] **Step 1: Single-branch clone of 6DoF into the disposable location**

```bash
rm -rf /tmp/rl-extract
git clone --single-branch --branch ar4-mk5-ik-node-2xipt1 /home/saps/projects/6DoF /tmp/rl-extract
```

Expected: clone succeeds, prints `Branch 'ar4-mk5-ik-node-2xipt1' set up to track...`.

- [ ] **Step 2: Run git-filter-repo with path filters and the rl/ rename**

```bash
cd /tmp/rl-extract
git filter-repo \
  --path rl/ \
  --path docs/superpowers/ \
  --path .superpowers/sdd/ \
  --path-rename rl/:
```

Expected: output ending in `New history written in ...` followed by a commit count. `git filter-repo` also resets the `origin` remote it inherited from the clone (this is expected — it always strips remotes to prevent accidental pushes to the source repo).

- [ ] **Step 3: Verify the resulting history**

```bash
git log --oneline | wc -l
git log --oneline
```

Expected: 27 commits (matches the count of commits touching the three filtered paths in the source repo — verify by comparing against `git -C /home/saps/projects/6DoF log --oneline develop..ar4-mk5-ik-node-2xipt1 -- rl/ docs/superpowers .superpowers | wc -l`, also 27). `4a9fc78 Add AR4 MK5 inverse kinematics ROS 2 node` must NOT appear in this list.

- [ ] **Step 4: Verify the flattened tree structure**

```bash
git ls-tree -d --name-only HEAD
```

Expected output (exact set, order may vary):
```
docs
perception
scripts
tasks
.superpowers
```
(`assets` and `logs` are absent — they were gitignored in the source repo, so `git-filter-repo` correctly has no record of them; Task 3 copies them in as plain files.)

- [ ] **Step 5: Verify no stray rl/ prefix survived**

```bash
git ls-tree -r --name-only HEAD | grep '^rl/' | wc -l
```

Expected: `0`

---

### Task 2: Create the GitHub repo, push, and clone to the working location

**Files:**
- Create (on GitHub): `hksaperstein/rl` (private)
- Create: `~/projects/rl/` (working clone)

**Interfaces:**
- Consumes: the filtered repo at `/tmp/rl-extract` from Task 1.
- Produces: `~/projects/rl`, a normal git working copy with `origin` pointing at `git@github.com:hksaperstein/rl.git`, `main` as the default branch, containing the 27-commit filtered history.

- [ ] **Step 1: Create the GitHub repo**

```bash
gh repo create hksaperstein/rl --private --description "Isaac Lab RL/perception project for the AR4 pick-and-place task"
```

Expected: prints the new repo's URL, `https://github.com/hksaperstein/rl`.

- [ ] **Step 2: Rename the filtered branch to main and push**

```bash
cd /tmp/rl-extract
git branch -m ar4-mk5-ik-node-2xipt1 main
git remote add origin git@github.com:hksaperstein/rl.git
git push -u origin main
```

Expected: push succeeds, prints `branch 'main' set up to track 'origin/main'`.

- [ ] **Step 3: Clone to the real working location**

```bash
rm -rf /tmp/rl-extract
git clone git@github.com:hksaperstein/rl.git ~/projects/rl
```

Expected: clone succeeds; `~/projects/rl/perception`, `~/projects/rl/scripts`, `~/projects/rl/tasks`, `~/projects/rl/docs/superpowers`, `~/projects/rl/.superpowers/sdd` all exist.

- [ ] **Step 4: Verify commit count and log survived the push/clone round-trip**

```bash
cd ~/projects/rl
git log --oneline | wc -l
```

Expected: `27`

---

### Task 3: Copy the gitignored assets/logs content and add a new .gitignore

**Files:**
- Create: `~/projects/rl/assets/` (copied from `6DoF/rl/assets/`)
- Create: `~/projects/rl/logs/` (copied from `6DoF/rl/logs/`)
- Create: `~/projects/rl/.gitignore`

**Interfaces:**
- Consumes: `~/projects/rl` from Task 2.
- Produces: a working repo where `assets/` and `logs/` are present on disk (matching the pre-migration `rl/assets/`, `rl/logs/` byte-for-byte) but untracked by git, ready for Task 4's smoke test to find the built USD assets and Task 5's verification.

- [ ] **Step 1: Copy the directories**

```bash
cp -r /home/saps/projects/6DoF/rl/assets ~/projects/rl/assets
cp -r /home/saps/projects/6DoF/rl/logs ~/projects/rl/logs
```

Expected: no output; exits 0.

- [ ] **Step 2: Verify byte-for-byte match**

```bash
diff -rq /home/saps/projects/6DoF/rl/assets ~/projects/rl/assets
diff -rq /home/saps/projects/6DoF/rl/logs ~/projects/rl/logs
```

Expected: no output from either (identical trees).

- [ ] **Step 3: Write the new .gitignore**

```
assets/
logs/
__pycache__/
*.py[cod]
*$py.class
```

Write this to `~/projects/rl/.gitignore`.

- [ ] **Step 4: Verify git status is clean**

```bash
cd ~/projects/rl
git status
```

Expected: `nothing to commit, working tree clean` (the new `.gitignore` itself is untracked at this point — check next).

Actually re-run:
```bash
git status --porcelain
```
Expected: exactly one line, `?? .gitignore`.

- [ ] **Step 5: Commit the .gitignore**

```bash
cd ~/projects/rl
git add .gitignore
git commit -m "$(cat <<'EOF'
Add .gitignore for generated assets/logs and Python artifacts
EOF
)"
```

Expected: commit succeeds.

- [ ] **Step 6: Push**

```bash
cd ~/projects/rl
git push
```

---

### Task 4: Fix hardcoded 6DoF/rl/ path references in docs and docstrings

**Files:**
- Modify: `~/projects/rl/README.md`
- Modify: `~/projects/rl/scripts/train.py:1-9`
- Modify: `~/projects/rl/scripts/eval_loop.py:1-9`
- Modify: `~/projects/rl/scripts/interactive_demo.py:1-22`
- Modify: `~/projects/rl/scripts/perception_calibration.py:1-16`

**Interfaces:**
- Consumes: `~/projects/rl` from Task 3 (must run after Task 3's commit, not before, to keep history clean).
- Produces: no code-behavior change (docs/docstrings only) — Task 5 verifies runtime behavior is unaffected.

- [ ] **Step 1: Rewrite README.md**

Replace the full contents of `~/projects/rl/README.md` with:

```markdown
# rl - AR4 Pick-and-Place RL

Everything here runs through Isaac Lab's launcher, not plain `python`. `isaaclab.sh`
lives in the separate IsaacLab install, not this repo - always run from this
repo's root and reference it by absolute path:

```bash
cd ~/projects/rl
/home/saps/IsaacLab/isaaclab.sh -p scripts/<script>.py [args]
```

## 1. Build the robot/scene assets (one-time)

```bash
/home/saps/IsaacLab/isaaclab.sh -p scripts/build_asset.py
```

## 2. Sanity-check perception before trusting it anywhere else

Slides the cube across the camera's view for a few seconds and writes a labeled
mp4 - watch it before running anything else that depends on perception:

```bash
/home/saps/IsaacLab/isaaclab.sh -p scripts/perception_calibration.py --headless
```

Check `logs/videos/perception_calibration.mp4`: the sliding cube should be
labeled `"cube"` throughout, and the three static objects (sphere, rectangular
prism, wedge) should be labeled correctly and consistently, without flickering
between shapes frame to frame.

**Known limitation:** The shape classifier currently misclassifies the real cube and rectangular prism as "sphere" in the calibration clip, with only the wedge classifying correctly. This is a threshold-tuning issue where parameters optimized on synthetic test data don't generalize to real sensor noise; it's tracked as a follow-up improvement and does not affect core pick-and-place functionality.

The perception math itself (ground-plane removal, shape classification,
tracking) has its own fast unit test suite, independent of Isaac Sim:

```bash
python3 -m pytest perception/tests/ -v
```

## 3. Train

```bash
# Quick smoke test first (~seconds, confirms the loop runs and writes a checkpoint):
/home/saps/IsaacLab/isaaclab.sh -p scripts/train.py --num_envs 16 --max_iterations 2 --headless

# Full training run:
/home/saps/IsaacLab/isaaclab.sh -p scripts/train.py --num_envs 4096 --headless
```

Checkpoints and TensorBoard logs are written to `logs/train/<timestamp>/`.
Watch training with:

```bash
tensorboard --logdir logs/train
```

What to look at:

- `Train/mean_reward` - overall trend; should climb and plateau.
- `Episode_Reward/lifting_cube` - climbing off zero means the policy is at
  least starting to lift the cube, independent of whether it's placing
  accurately yet.
- `Episode_Reward/cube_goal_tracking_fine_grained` - the sharpest signal that
  placement is getting precise, not just "close enough."
- `Episode_Termination/cube_reached_goal` - the success rate: fraction of
  episodes that ended by actually reaching the goal, rather than timing out.
  This is the clearest single "is it working" number - reward can climb from
  partial credit (reaching, lifting) while this stays at zero, which tells you
  the policy is exploring but not yet succeeding.
- `Episode_Termination/time_out` - the complement of the above (episodes that
  ran out the clock without success).

There's no fixed "enough training" iteration count - stop once
`Episode_Termination/cube_reached_goal` has climbed and plateaued, rather than
running to a predetermined number of iterations.

## 4. Evaluate a checkpoint

```bash
# Privileged simulation state (fast, matches how training worked):
/home/saps/IsaacLab/isaaclab.sh -p scripts/eval_loop.py --checkpoint logs/train/<run>/model_<iter>.pt --episodes 10

# Real camera-based perception instead:
/home/saps/IsaacLab/isaaclab.sh -p scripts/eval_loop.py --checkpoint logs/train/<run>/model_<iter>.pt --episodes 10 --perception
```

Videos are written to `logs/videos/` (`ar4_pickplace-*.mp4` for the default
path, `ar4_pickplace_perception.mp4` for `--perception`, with the detection
overlay burned in for the latter). A healthy result: the cube is reliably
picked up and placed near the target region in most episodes.

## 5. Interactive demo

```bash
/home/saps/IsaacLab/isaaclab.sh -p scripts/interactive_demo.py --checkpoint logs/train/<run>/model_<iter>.pt
```

With the GUI open: drag the cube anywhere in the workspace using the
viewport's drag gizmo, then let go. Once it's settled for about a second, the
arm picks it up and moves it to the target region on the other side, using the
real perception pipeline the whole time (not privileged simulation state) -
including through the brief period where the arm itself blocks the camera's
view of the cube mid-grasp (the tracker holds its last-known position through
that). Drag the cube outside the workspace or camera's view and the arm stays
idle rather than reacting to it.

The session records to `logs/videos/ar4_interactive_demo.mp4` with the
detection overlay burned in, and keeps running (watching for the next drag)
until you close the window.
```

- [ ] **Step 2: Fix train.py's docstring**

In `~/projects/rl/scripts/train.py`, replace lines 1-9:

```python
"""Train a PPO policy (rsl_rl) for the AR4 pick-and-place task.

.. code-block:: bash

    cd /home/saps/projects/6DoF
    /home/saps/IsaacLab/isaaclab.sh -p rl/scripts/train.py --num_envs 4096
    # smoke test (fast, verifies the loop runs end-to-end and writes a checkpoint):
    /home/saps/IsaacLab/isaaclab.sh -p rl/scripts/train.py --num_envs 16 --max_iterations 2 --headless
"""
```

with:

```python
"""Train a PPO policy (rsl_rl) for the AR4 pick-and-place task.

.. code-block:: bash

    cd ~/projects/rl
    /home/saps/IsaacLab/isaaclab.sh -p scripts/train.py --num_envs 4096
    # smoke test (fast, verifies the loop runs end-to-end and writes a checkpoint):
    /home/saps/IsaacLab/isaaclab.sh -p scripts/train.py --num_envs 16 --max_iterations 2 --headless
"""
```

- [ ] **Step 3: Fix eval_loop.py's header comment and docstring**

In `~/projects/rl/scripts/eval_loop.py`, replace lines 1-9:

```python
# rl/scripts/eval_loop.py
"""Run a trained AR4 pick-and-place PPO policy for a fixed number of episodes,
recording each one as an mp4 to rl/logs/videos/.

.. code-block:: bash

    cd /home/saps/projects/6DoF
    /home/saps/IsaacLab/isaaclab.sh -p rl/scripts/eval_loop.py --checkpoint rl/logs/train/<run>/model_1500.pt --episodes 10
"""
```

with:

```python
# scripts/eval_loop.py
"""Run a trained AR4 pick-and-place PPO policy for a fixed number of episodes,
recording each one as an mp4 to logs/videos/.

.. code-block:: bash

    cd ~/projects/rl
    /home/saps/IsaacLab/isaaclab.sh -p scripts/eval_loop.py --checkpoint logs/train/<run>/model_1500.pt --episodes 10
"""
```

- [ ] **Step 4: Fix interactive_demo.py's header comment and docstring**

In `~/projects/rl/scripts/interactive_demo.py`, replace lines 1-22:

```python
# rl/scripts/interactive_demo.py
"""Interactive AR4 pick-and-place demo: drag the cube anywhere in the Isaac Sim
GUI viewport (native drag gizmo), and once it settles the trained policy picks
it up and places it in the fixed target region on the other side - using the
real camera-based perception pipeline the whole time, exactly as
eval_loop.py --perception does at inference time.

An out-of-view or out-of-the-workspace cube position never triggers an
attempt - the arm just keeps watching and waiting.

A single "armed" flag guards each trigger: once a pick-and-place attempt
fires, the demo disarms itself so the cube sitting still at the goal
position right after placement can't immediately re-trigger another
attempt. It only re-arms once the cube is observed to have changed state
(dragged away, gone stale/out of view, or moved past the stability
tolerance) - i.e. real evidence of a fresh human drag.

.. code-block:: bash

    cd /home/saps/projects/6DoF
    /home/saps/IsaacLab/isaaclab.sh -p rl/scripts/interactive_demo.py --checkpoint rl/logs/train/<run>/model_1500.pt
"""
```

with:

```python
# scripts/interactive_demo.py
"""Interactive AR4 pick-and-place demo: drag the cube anywhere in the Isaac Sim
GUI viewport (native drag gizmo), and once it settles the trained policy picks
it up and places it in the fixed target region on the other side - using the
real camera-based perception pipeline the whole time, exactly as
eval_loop.py --perception does at inference time.

An out-of-view or out-of-the-workspace cube position never triggers an
attempt - the arm just keeps watching and waiting.

A single "armed" flag guards each trigger: once a pick-and-place attempt
fires, the demo disarms itself so the cube sitting still at the goal
position right after placement can't immediately re-trigger another
attempt. It only re-arms once the cube is observed to have changed state
(dragged away, gone stale/out of view, or moved past the stability
tolerance) - i.e. real evidence of a fresh human drag.

.. code-block:: bash

    cd ~/projects/rl
    /home/saps/IsaacLab/isaaclab.sh -p scripts/interactive_demo.py --checkpoint logs/train/<run>/model_1500.pt
"""
```

- [ ] **Step 5: Fix perception_calibration.py's header comment and docstring**

In `~/projects/rl/scripts/perception_calibration.py`, replace lines 1-16:

```python
# rl/scripts/perception_calibration.py
"""Sanity-check the perception pipeline before trusting it in eval/demo scripts:
slides the cube across the perception camera's field of view for a few seconds
and writes an mp4 with the detected mask/bbox/shape-label burned into each frame.

Not run during training or as part of any automated flow - a one-time (or
re-run-when-something-changes) manual check. The robot is present but held
motionless throughout (Isaac Lab's manager framework needs at least one
action/observation term, and the existing pick-and-place env config already
provides a well-tested one) - only the cube moves.

.. code-block:: bash

    cd /home/saps/projects/6DoF
    /home/saps/IsaacLab/isaaclab.sh -p rl/scripts/perception_calibration.py --headless
"""
```

with:

```python
# scripts/perception_calibration.py
"""Sanity-check the perception pipeline before trusting it in eval/demo scripts:
slides the cube across the perception camera's field of view for a few seconds
and writes an mp4 with the detected mask/bbox/shape-label burned into each frame.

Not run during training or as part of any automated flow - a one-time (or
re-run-when-something-changes) manual check. The robot is present but held
motionless throughout (Isaac Lab's manager framework needs at least one
action/observation term, and the existing pick-and-place env config already
provides a well-tested one) - only the cube moves.

.. code-block:: bash

    cd ~/projects/rl
    /home/saps/IsaacLab/isaaclab.sh -p scripts/perception_calibration.py --headless
"""
```

- [ ] **Step 6: Verify no stray references remain**

```bash
cd ~/projects/rl
grep -rn "6DoF\|rl/scripts\|rl/logs\|rl/perception" README.md scripts/*.py
```

Expected: no output (empty).

- [ ] **Step 7: Commit**

```bash
cd ~/projects/rl
git add README.md scripts/train.py scripts/eval_loop.py scripts/interactive_demo.py scripts/perception_calibration.py
git commit -m "$(cat <<'EOF'
Fix launch-command paths in README/docstrings for the new repo root

The extraction from 6DoF/rl/ flattened everything up one level; these
were the only hardcoded absolute-path references left behind (the
os.path.dirname() chains used at runtime are self-referential within
the subtree and needed no change - verified directly).
EOF
)"
git push
```

---

### Task 5: Verify the new repo actually works

**Files:** none (verification only)

**Interfaces:**
- Consumes: the fully-assembled `~/projects/rl` from Tasks 1-4.

- [ ] **Step 1: Run the perception unit test suite**

```bash
cd ~/projects/rl
python3 -m pytest perception/tests/ -v
```

Expected: all 25 tests pass (same count as before the move — this suite is pure numpy/scipy/cv2, path-independent via `conftest.py`'s sys.path shim, so it should be unaffected).

- [ ] **Step 2: Smoke-test an Isaac Lab script from the new location**

```bash
cd ~/projects/rl
PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/drive_joints_demo.py --headless --steps 60
```

Expected: script runs to completion without a Python traceback, prints periodic `[Step ...] joint_pos: ...` lines, ends with `Joint data recorded to: .../logs`. This confirms `AR4_MK5_CFG`'s USD asset resolution (`tasks/ar4/robot_cfg.py`'s `_RL_ROOT`-based path) and the `sys.path` imports both resolve correctly from the new, flattened location.

- [ ] **Step 3: If Step 2 fails, do not proceed**

If the smoke test fails with an import error or missing-asset error, stop and diagnose before touching `6DoF` in Task 6 — the source `6DoF/rl/` is still intact at this point and is the fallback if something in the extraction needs redoing.

---

### Task 6: Remove the migrated content from 6DoF

**Files:**
- Delete: `6DoF/rl/` (entire directory)
- Delete: `6DoF/docs/superpowers/` (entire directory)
- Delete: `6DoF/.superpowers/sdd/` (entire directory)
- Modify: `6DoF/.gitignore`

**Interfaces:**
- Consumes: Task 5's passing verification (only proceed once the new repo is confirmed working).

- [ ] **Step 1: Delete the migrated directories**

```bash
cd /home/saps/projects/6DoF
git rm -r rl docs/superpowers .superpowers/sdd
```

Expected: prints a long list of `rm 'rl/...'` / `rm 'docs/superpowers/...'` / `rm '.superpowers/sdd/...'` lines.

- [ ] **Step 2: Remove the now-stale rl/ entries from .gitignore**

Current `6DoF/.gitignore` ends with:

```
# Isaac Lab generated assets/logs (rl/)
rl/assets/
rl/logs/
```

Remove those three lines (the comment and both entries), leaving the file ending at the `.cproject` line of the "Editor/IDE specific" block.

- [ ] **Step 3: Verify what remains**

```bash
cd /home/saps/projects/6DoF
git status --porcelain | grep -v '^D ' | head -20
ls docs/superpowers 2>&1
ls .superpowers 2>&1
```

Expected: first command shows only the modified `.gitignore` as non-deleted; `ls docs/superpowers` and `ls .superpowers` both report `No such file or directory` (docs/superpowers is fully gone; `.superpowers/` itself may still exist if it had other contents — check with `ls .superpowers`).

- [ ] **Step 4: Commit**

```bash
cd /home/saps/projects/6DoF
git add .gitignore
git commit -m "$(cat <<'EOF'
Remove rl/, docs/superpowers/, and .superpowers/sdd/ (migrated to hksaperstein/rl)

This content now lives at github.com/hksaperstein/rl with its git history
preserved via git-filter-repo. This commit does not rewrite 6DoF's own
history - every commit that built this content remains in 6DoF's log.
EOF
)"
```

Expected: commit succeeds, `git log -1 --stat` shows the deletions.

- [ ] **Step 5: Verify 6DoF's remaining structure**

```bash
cd /home/saps/projects/6DoF
find . -maxdepth 1 -not -path . -not -path ./.git
```

Expected: `build`, `docker`, `.github`, `.gitignore`, `install`, `log`, `README.md`, `run_in_docker.sh`, `src`, `.vscode` — no `rl` or leftover `docs/superpowers`.

---

## Self-Review Notes

- **Spec coverage:** All spec sections covered — new repo structure (Task 2-4), extraction method (Task 1-2), post-extraction fixes (Task 4, corrected to docs-only per the spec amendment), 6DoF-side cleanup (Task 6), verification (Task 5).
- **Placeholder scan:** No TBDs; every step has literal commands/content.
- **Type consistency:** N/A (no function signatures span tasks in this plan — it's a migration, not new application code).
