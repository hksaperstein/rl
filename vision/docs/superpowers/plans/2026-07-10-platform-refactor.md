# Platform Refactor Implementation Plan — Internal Restructure

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the "Internal restructure of this repo" section of
`docs/superpowers/specs/2026-07-10-training-platform-architecture-design.md`
only — `src/{datagen,datasets,training/detection,evaluation,export}/` +
`configs/domains/dice.yaml` + thin `scripts/` CLIs — behavior-preserving,
proven against the dice-detector-v1 regression baseline
(`docs/results/2026-07-dice-detector-v1/summary.md`,
`docs/superpowers/plans/2026-07-10-dice-detector-v1.md`).

Explicitly deferred (per dispatch scope): `docs/platform/` cross-repo
contract docs, anything touching the `rl` repo, repo rename, classification/
segmentation implementations.

## Global constraints

- Repo root: `/home/saps/projects/Dice-Detection`.
- Python: `.venv/bin/python`; pytest with `-p no:launch_testing` (ROS 2
  Jazzy's `launch_testing` plugin otherwise errors on collection).
- GPU: RTX 5070 Ti, confirmed idle before any training/export run.
- Do NOT retrain the real 60-epoch runs (`models/runs/{s,s_plus_r}/`), do
  NOT touch `data/detection_v1`, `data/real/raw`, or committed results docs
  (`docs/results/2026-07-dice-detector-v1/*.md`). Every verification run
  below uses a scratch/`_refactor_*`-suffixed name specifically so it can't
  collide with those.
- Every move is a `git mv` (or Read-then-Write for edited files) — content
  changes are limited to import-path fixes and hardcoded-list ->
  config-driven substitutions; no logic changes to ported code.

## Task 1: Split `src/dice_gen/` into `src/datagen/` (core) + `src/datagen/domains/dice/` (plugin)

**Files:**
- Move: `src/dice_gen/{exporter,materials}.py` -> `src/datagen/`
- Move: `src/dice_gen/{geometry,numbering,glyphs,sampler,orchestrator}.py` -> `src/datagen/domains/dice/`
- Modify: `src/datagen/domains/dice/orchestrator.py` (import split), all
  `tests/blender/test_*.py`, `tests/test_numbering.py`, `tests/test_sampler.py`,
  `scripts/generate_dice_assets.py`, `scripts/validate_dice_assets.py`

**Package-boundary judgment call (documented per dispatch instructions):**
the spec's "domain-agnostic core vs dice-specific plugin" split is not
mechanical for every file in `src/dice_gen/` — most of the module content
(die topology, numbering conventions, glyph-application code coupled to
per-die-type conventions) is inherently dice-specific. The boundary drawn
here: `exporter.py` (generic .blend/USD/STL export + manifest write,
thumbnail render — takes a generic mesh object, no die-type knowledge) and
`materials.py` (generic procedural shader-material library — 6 finish
categories, no die-type knowledge) move to `src/datagen/` as reusable core;
`geometry.py`, `numbering.py`, `glyphs.py`, `sampler.py`, `orchestrator.py`
(all embed die shapes/numbering schemes/per-die-type glyph conventions or
directly orchestrate them) move to `src/datagen/domains/dice/` as the dice
plugin. This is the one architectural judgment call in this refactor beyond
mechanical relocation; flagged here rather than decided silently.

- [x] **Step 1:** `git mv` the two core files and five domain files into
  the new layout; add empty `__init__.py` at `src/datagen/`,
  `src/datagen/domains/`, `src/datagen/domains/dice/`; `git rm` the old
  `src/dice_gen/__init__.py`.
- [x] **Step 2:** Fix `orchestrator.py`'s internal import: `from . import
  exporter, geometry, glyphs, materials, numbering, sampler` ->
  `from . import geometry, glyphs, numbering, sampler` (same-package
  siblings) + `from ... import exporter, materials` (reaching up to the
  `datagen` package for the two core modules).
- [x] **Step 3:** Rewrite every `from dice_gen import ...` /
  `from dice_gen.<mod> import ...` line across `tests/blender/test_*.py`
  (5 files), `tests/test_numbering.py`, `tests/test_sampler.py`,
  `scripts/generate_dice_assets.py`, `scripts/validate_dice_assets.py` to
  the new `datagen`/`datagen.domains.dice` paths (77 lines across 9 files;
  done via a small one-off rewrite script, not by hand, given the volume —
  `test_glyphs.py` alone has ~40 occurrences at function-local scope).
  `sys.path.insert(...)` lines pointing at `src/` are unchanged (still the
  correct root).
- [x] **Step 4:** Verify: `tests/test_numbering.py`, `tests/test_sampler.py`,
  `tests/test_validate_dice_assets.py` (42 tests) green under
  `.venv/bin/pytest -p no:launch_testing`; all 5
  `tests/blender/test_*.py` files green under
  `blender --background --python tests/blender/test_<name>.py`
  (`ALL TESTS PASSED`, matching the pre-refactor baseline run exactly,
  including the same pre-existing d6 engrave-retry warning in
  `test_glyphs.py` — not a regression).

## Task 2: `configs/domains/dice.yaml` + `src/datasets/` (data plumbing library-ification)

**Files:**
- Create: `configs/domains/dice.yaml`
- Create: `src/datasets/{__init__.py,domain_config.py,coco_yolo.py,real_ingest.py}`
- Modify: `scripts/convert_coco_to_yolo.py`, `scripts/prepare_real_data.py` (become thin CLIs)

- [x] **Step 1:** Write `configs/domains/dice.yaml`: `classes` (7-class
  order), `merged_classes` (6-class, d10_pct merged into d10),
  `real_data.sources` (the 2 Roboflow projects), `real_data.source_name_map`,
  `real_data.exclude_stem_prefixes`. This consolidates the `CLASS_NAMES`
  list that was previously duplicated identically 4x across
  `scripts/{convert_coco_to_yolo,evaluate,prepare_real_data,train}.py`
  (confirmed via `grep -rn "CLASS_NAMES\s*=" scripts/*.py` before starting).
- [x] **Step 2:** `src/datasets/domain_config.py`: `load_domain_config(name)`
  — the one place every downstream package reads a domain's config from.
- [x] **Step 3:** `src/datasets/coco_yolo.py` — port of
  `scripts/convert_coco_to_yolo.py`'s logic verbatim
  (`coco_bbox_to_yolo`, `build_split`, `main`), `CLASS_NAMES` now sourced
  from `load_domain_config("dice")` instead of hardcoded.
- [x] **Step 4:** `src/datasets/real_ingest.py` — port of
  `scripts/prepare_real_data.py`'s logic verbatim (`remap_label_line`,
  `group_pairs_by_augmentation`, `split_groups`, `should_exclude_stem`,
  `download_source`, `collect_pairs`, `spotcheck`, `main`),
  `CLASS_NAMES`/`SOURCE_NAME_MAP`/`SOURCES`/`EXCLUDE_STEM_PREFIXES` now
  config-sourced.
- [x] **Step 5:** Rewrite `scripts/convert_coco_to_yolo.py` and
  `scripts/prepare_real_data.py` as thin CLIs: `sys.path.insert` the repo's
  `src/`, import + re-export every name the existing tests import directly
  from the script module (so `tests/test_convert_coco_to_yolo.py` and
  `tests/test_prepare_real_data.py` need **zero** changes), call `main()`.
- [x] **Step 6:** Verify: full pytest suite still 61/61 green; re-run
  `scripts/convert_coco_to_yolo.py --out <scratch>` and diff every label
  file against `data/yolo/labels/` — byte-identical, same 9000/1000 split
  counts.

## Task 3: `src/training/detection/` + `src/evaluation/` (library-ification)

**Files:**
- Create: `src/training/{__init__.py,detection/{__init__.py,yolo_trainer.py}}`
- Create: `src/evaluation/{__init__.py,detection_eval.py}`
- Modify: `scripts/train.py`, `scripts/evaluate.py` (become thin CLIs)

- [x] **Step 1:** `src/training/detection/yolo_trainer.py` — port of
  `scripts/train.py`'s logic verbatim (`build_s_plus_r_yaml`, `main`),
  `CLASS_NAMES` config-sourced. First (and only, for now) `Trainer`-protocol
  implementation the platform spec anticipates.
  **Deviation from strict "same CLI args" (documented, not silent):** added
  one new optional `--name` flag (default: `--variant`'s value) overriding
  the `models/runs/<name>/` output dir, so a regression-check smoke run can
  target `models/runs/s_refactor_smoke/` instead of clobbering
  `models/runs/s/` — which holds the real, un-recreatable-without-60-epochs
  weights backing the committed results docs. Every pre-refactor invocation
  (bare `--variant`) is byte-for-byte unaffected.
- [x] **Step 2:** `src/evaluation/detection_eval.py` — port of
  `scripts/evaluate.py`'s logic verbatim (`merge_d10`, `eval_real`,
  `yolo_label_to_gt`, `main`), `CLASS_NAMES`/`MERGED_CLASS_NAMES`
  config-sourced. First (and only, for now) `Evaluator`-protocol
  implementation.
- [x] **Step 3:** Rewrite `scripts/train.py` and `scripts/evaluate.py` as
  thin CLIs, same re-export pattern as Task 2 Step 5 (`tests/test_evaluate.py`
  needs zero changes).
- [x] **Step 4:** Verify (real GPU runs, not mocked):
  - `scripts/train.py --variant s --epochs 1 --name s_refactor_smoke`
    completes, writes `models/runs/s_refactor_smoke/weights/best.pt`,
    prints a 7-class per-class val table.
  - `scripts/evaluate.py --weights models/runs/s/weights/best.pt --name
    s_refactor_check` writes `eval_s_refactor_check.md`; diffed against the
    committed `eval_s.md` — identical apart from the variant-name label and
    overlay path (every mAP50/mAP50-95 number in both the synthetic-val and
    real-test tables matches exactly, same weights/protocol).
  - Scratch outputs (`models/runs/s_refactor_smoke/`,
    `models/eval/s_refactor_check/`, `eval_s_refactor_check.md`, and a
    stray `runs/detect/val/` ultralytics writes at repo-root cwd when
    `model.val()` is called without an explicit `project=`/`name=` — a
    pre-existing quirk of the unchanged synthetic-val call, not introduced
    here) deleted after capturing the diff; not committed.

## Task 4: `src/export/` — ONNX + manifest CLI (new; the deployment contract)

**Files:**
- Create: `src/export/{__init__.py,onnx_export.py}`
- Create: `scripts/export_model.py`

- [x] **Step 1:** `src/export/onnx_export.py`: `export(weights, name, imgsz,
  dataset_version)` — runs `ultralytics` `model.export(format="onnx")`,
  copies the result to `models/export/<name>/model.onnx`, writes
  `models/export/<name>/manifest.json` with: `classes` (from
  `configs/domains/dice.yaml`), `imgsz`, `git_sha` (`git rev-parse HEAD`),
  `dataset_version` (hand-maintained string, see module docstring — no
  dataset-versioning registry exists yet), `weights_source`,
  `eval_real_test_map50` (parsed from the committed
  `docs/results/2026-07-dice-detector-v1/eval_<name>.md`'s "Real test"
  table — a small markdown-table regex parser, not a re-run of evaluation),
  `export_date_utc`.
- [x] **Step 2:** `scripts/export_model.py` thin CLI:
  `--weights <pt> --name <variant> [--imgsz] [--dataset-version]`.
- [x] **Step 3:** Acceptance test: export `s_plus_r` (the real-data variant,
  per dispatch instruction) —
  `scripts/export_model.py --weights models/runs/s_plus_r/weights/best.pt --name s_plus_r`
  -> `models/export/s_plus_r/{model.onnx,manifest.json}`; manifest's
  `eval_real_test_map50` matches `eval_s_plus_r.md`'s real-test table.

## Task 5: Progress log + report

- [x] Log each task above to `.superpowers/sdd/progress.md` under a new
  "Plan: platform-refactor" section as it completes.
- [x] Write `.superpowers/sdd/platform-refactor-report.md`: what moved
  where, the four verification results (a/b/c/d) with command-output
  excerpts, deviations from spec and why (the Task 1 domain-boundary call,
  the Task 3 `--name` CLI addition, the `/runs/` gitignore addition).
- [x] Commit in coherent chunks (one per task above), push to `origin/main`
  after each.
