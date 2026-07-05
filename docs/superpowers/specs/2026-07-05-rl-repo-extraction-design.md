# RL repo extraction design

## Goal

Move the Isaac Lab RL/perception project (currently `6DoF/rl/`, plus its
design docs and subagent-driven-dev ledger) out of the `6DoF` repo into its
own dedicated repository, `hksaperstein/rl`, so that `6DoF` can later become
primarily a deployment-of-trained-models repo (a separate, future
sub-project). This is one of four independent pieces of a larger refactor
(the other three: move `src/robot`'s IK node into the `ar4_ros_driver` fork,
slim `6DoF` down to deployment, and a new Blender-based dice-detection
pipeline in the already-separate `Dice-Detection` repo) — this spec covers
only the RL-repo extraction.

## Scope

**Moves, with git history preserved:**

- `rl/` (entire directory: `perception/`, `scripts/`, `tasks/`, `README.md`,
  plus the gitignored `assets/` and `logs/` directories, copied as plain
  files since they were never tracked in git)
- `docs/superpowers/specs/*` and `docs/superpowers/plans/*` (every file in
  both directories is about this RL/perception feature — verified via `git
  log`)
- `.superpowers/sdd/*` (the subagent-driven-development ledger from
  building the perception feature)

**Stays in 6DoF, untouched:** everything else, including `src/robot/`
(the IK node — part of a separate future sub-project), `docker/`,
`.github/`, `README.md`.

## New repo layout

`rl/`'s contents are flattened to the new repo's root rather than nested
under an `rl/` subfolder (avoids the redundant `rl/rl/...` a repo named `rl`
would otherwise produce):

```
hksaperstein/rl (private, new)
├── assets/                  (was rl/assets/, gitignored)
├── logs/                    (was rl/logs/, gitignored)
├── perception/              (was rl/perception/)
├── scripts/                 (was rl/scripts/)
├── tasks/                   (was rl/tasks/)
├── README.md                (was rl/README.md)
├── docs/superpowers/specs/  (unchanged path)
├── docs/superpowers/plans/  (unchanged path)
├── .superpowers/sdd/        (unchanged path)
└── .gitignore               (new: assets/, logs/, __pycache__/, *.pyc)
```

## Extraction method

1. Clone the `ar4-mk5-ik-node-2xipt1` branch of `6DoF` into a disposable
   working directory (single-branch clone, so unrelated branches don't
   enter the filter).
2. Run `git-filter-repo` (installed locally) against that clone with path
   filters for `rl/`, `docs/superpowers/`, `.superpowers/sdd/`, and a path
   rename stripping the `rl/` prefix. This preserves the ~27 commits that
   touch these paths (authorship and dates intact) and naturally drops the
   1 commit that doesn't (the IK node), since filter-repo prunes commits
   that become empty after filtering.
3. Author a new `.gitignore` for the filtered repo (the old root
   `.gitignore` mixed ROS2/RL entries and doesn't carry over as-is).
4. `gh repo create hksaperstein/rl --private`, push the filtered history as
   `main`.
5. Clone the result to `~/projects/rl` as the working copy.
6. Copy `6DoF/rl/assets/` and `6DoF/rl/logs/`'s actual contents (gitignored,
   so absent from git history) into `~/projects/rl/assets/` and
   `~/projects/rl/logs/`.

## Post-extraction fixes (in the new `~/projects/rl` repo)

- **Path-depth fix**: every script/module that derives its repo root via
  chained `os.path.dirname(os.path.dirname(...))` calls needs exactly one
  `dirname()` call removed, since the tree is now one level shallower.
  Affected files: `build_asset.py`, `train.py`, `eval_loop.py`,
  `interactive_demo.py`, `perception_calibration.py`, `drive_joints_demo.py`,
  `grasp_demo.py`, `tasks/ar4/robot_cfg.py`, `tasks/ar4/objects_cfg.py`,
  `perception/tests/conftest.py`.
- **Docstring/README path updates**: launch-command examples currently say
  `cd /home/saps/projects/6DoF` then `isaaclab.sh -p rl/scripts/...`; update
  to `cd ~/projects/rl` then `isaaclab.sh -p scripts/...`. The
  `AR4_DESCRIPTION_PATH` example (pointing at `annin_ws/src/ar4_ros_driver`)
  is unaffected and stays as-is.
- Verify each moved script still runs (smoke test via `isaaclab.sh -p
  <script>.py`, matching how prior work in this project was verified) after
  the path fixes, since a silent path bug here would only surface at
  runtime.

## 6DoF-side cleanup

Once the new repo is verified working, delete `rl/`, `docs/superpowers/`,
and `.superpowers/sdd/` from `6DoF` in a single new commit on the current
branch (`ar4-mk5-ik-node-2xipt1`). This does not rewrite 6DoF's existing
history — every commit that added this content remains in 6DoF's log; the
new commit simply stops carrying the files forward. `6DoF`'s own
`.gitignore` entries for `rl/assets/` and `rl/logs/` are removed as part of
the same commit.

## Out of scope (future sub-projects, not this one)

- Moving `src/robot`'s IK node into the `ar4_ros_driver` fork.
- Slimming `6DoF` down to a deployment-focused repo that consumes trained
  policies from the new `rl` repo.
- The `Dice-Detection` Blender pipeline.

## Verification

- New repo's test suite (`pytest` for `rl/perception/tests/`) passes
  unmodified (path-independent, uses `conftest.py`'s sys.path shim).
- At least one Isaac Lab script (e.g. `drive_joints_demo.py`, the cheapest
  smoke test) runs successfully from `~/projects/rl` after the path-depth
  fix, confirming the flatten didn't break asset/module resolution.
- `6DoF` builds/imports cleanly with `rl/` gone (nothing outside `rl/`
  imports from it — confirmed during design research: `grasp_demo.py`
  references `src/robot/src/ar4_mk5_kinematics.py` only in a docstring
  comment, not a runtime import, and the dependency direction is one-way
  from `rl/` toward `src/robot/`, never the reverse).
