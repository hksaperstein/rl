# Monorepo Merge — Dice-Detection into rl, as `vision/`

**Date:** 2026-07-10
**Status:** Approved (direct user decision: "I think i like the idea of
monorepo. deal with the nuance. fix it"), superseding the federation
recommendation in Dice-Detection's
`docs/superpowers/specs/2026-07-10-training-platform-architecture-design.md`.
That spec's *internal restructure* (already executed) carries over intact;
only its repo-boundary section is superseded.

## Decision

One repo. `rl` hosts (it is the long-running research platform per its
North Star; session memory, kb/, and ROADMAP already live here).
`Dice-Detection` is imported **with full git history** via
`git subtree add --prefix=vision/`, becoming the `vision/` top-level tree:
synthetic data generation (Blender), dataset plumbing, perception-model
training/evaluation, and ONNX+manifest export.

## The nuances, and how each is handled

1. **Two Python runtimes, one repo.** The permanent rule, documented in the
   root CLAUDE.md: *path decides the interpreter.* Everything under
   `vision/` runs `vision/.venv/bin/python` (PyTorch cu128; pytest with
   `-p no:launch_testing`); everything else Isaac-touching runs
   `/home/saps/IsaacLab/isaaclab.sh -p` under the flock lock, unchanged.
   `vision/` code must never import Isaac Lab; `rl`-side code must never
   import `vision/.venv` packages. The GPU is still singular: vision
   training/eval jobs count as GPU jobs and should not run concurrently
   with Isaac Sim work. (Amended 2026-07-10, user decision: vision jobs
   are NOT required to take the `/tmp/rl_isaac_sim.lock` flock — sequence
   by judgment. The lock convention remains for Isaac-Sim-touching work.)
2. **Branch topology.** The subtree lands on `main` (infrastructure, not
   Franka work), then `main` is merged forward into `franka-panda-pivot` so
   the active branch sees `vision/` too. Paths are disjoint except the root
   CLAUDE.md edit; any conflict there is resolved by hand at merge time.
3. **Gitignored artifacts don't travel via git.** Physically moved from the
   old checkout: `data/detection_v1` (10k renders + coco.json),
   `data/real/` (downloads + splits), `data/hdris`, `data/raw`,
   `models/runs/`, `models/eval/`, `models/export/`. NOT moved:
   `.venv` (recreated fresh at `vision/.venv` — venvs don't survive
   relocation), `data/yolo/` (regenerated — its image symlinks embed the
   old absolute path, and its dataset yamls embed absolute `path:` values).
4. **Old repo disposition.** Final commit to Dice-Detection adds a MOVED
   notice at the top of its README (new home: `rl` repo `vision/`); the
   GitHub repo is left un-archived — user's call whether to archive.
5. **Root docs.** Root CLAUDE.md gains a "Monorepo layout & runtimes"
   section and delegates vision-specific conventions to `vision/CLAUDE.md`
   (which keeps governing within its subtree). README gains one line.
   `vision/` keeps its own docs/, specs/, plans/, ledger, and kb-equivalent
   history untouched.

## Verification gates (same evidence standard as the platform refactor)

- (a) `vision/.venv/bin/pytest vision/tests -p no:launch_testing`
  (--ignore=vision/tests/blender): 61 passed.
- (b) Regenerated `vision/data/yolo` label files byte-identical to the
  pre-move originals (diff against a pre-move copy of the label trees);
  9000/1000 split counts preserved; new symlinks resolve.
- (c) 1-epoch smoke train via `vision/scripts/train.py` completes on GPU
  from the new location.
- (d) `vision/scripts/evaluate.py` on the migrated
  `models/runs/s/weights/best.pt` reproduces the committed
  `eval_s.md` real-test table exactly.
- (e) rl side untouched: `git diff --stat main@{pre-merge}..main` shows
  only `vision/*`, CLAUDE.md, README.md, and this spec.

## Out of scope

- Renaming the `rl` GitHub repo (user's namespace; monorepo works under
  the current name).
- Archiving the Dice-Detection GitHub repo (reversible; left to user).
- Any change to Isaac-side code, the Franka pivot work, or vision/
  internals beyond path/venv regeneration.
- Cross-repo "federation contract" docs — moot under a monorepo; the
  deployment manifest (`vision/src/export/`) remains as the interface
  between vision models and the robot stack, now in-repo.
