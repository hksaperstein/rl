# Training Platform Architecture — Multi-Domain, Multi-Model-Type

**Date:** 2026-07-10
**Status:** Design presented; federation-vs-monorepo recommendation awaiting
user sign-off. Implementation (Sonnet-executed refactor) gated behind
completion of the dice-detector-v1 plan (Tasks 5/6/8) — per direct user
instruction: design now, wait for completion before implementation.

## Goal

Restructure this repo so it can train **more than one type of model across
more than one domain** — where "model types" spans both perception models
(object detection today) and, at the platform level, the manipulation
policies trained in the companion `rl` repo (Isaac Lab / Franka). Dice is
the first domain, not the defining one.

## The repo-boundary decision (recommended: federation, not monorepo)

Two repos, three explicit contracts — **not** a single merged repo.

Reasons against a monorepo now:
1. **Two Python runtimes.** Manipulation training runs on Isaac Lab's
   bundled interpreter; this repo runs a `.venv` with cu128 PyTorch. One
   repo with two interpreters makes every script invocation a footgun.
2. **The `rl` repo is mid-Franka-pivot** with its own conventions, branch
   discipline, and evidence chain; a structural merge would churn it at its
   most sensitive point.
3. **The coupling actually needed is contract-shaped, not shared-code-
   shaped** (see below). YAGNI on a shared package until two consumers
   exist for the same code.

### The three contracts binding the two repos

1. **Deployment contract (perception → robot).** Every perception model
   this platform ships is exported to ONNX (TensorRT-ready) with a
   versioned manifest: class list + order, input size, checkpoint
   provenance (git SHA, dataset version), and eval scores (synthetic +
   real). The `rl`/robot side consumes only these artifacts, never this
   repo's internals. First instance: the dice detector heading for the
   robot camera.
2. **Asset contract (datagen → sim).** The Blender datagen already exports
   USD. Manipulation domains source sim assets *and* matched perception
   training data from the same generator, so the object a policy grasps
   and the object a detector recognizes are the same geometry, materials,
   and scale conventions.
3. **Methodology contract (shared research discipline).** The experiment
   conventions proven across both repos — spec → plan → results with
   falsifiable hypotheses, leak-free splits (group-aware; see
   dice-detector-v1's Roboflow-duplicate and asset-set findings), frozen
   test sets, manifest-tracked known defects — written once as platform
   docs (`docs/platform/`), referenced by both repos' CLAUDE.md.

Policy training itself never moves into this repo; the platform **feeds**
it (assets in, perception models out).

## Internal restructure of this repo

Target layout (executed by a Sonnet-led refactor after dice-detector-v1
completes):

```
src/
├── datagen/              # domain-agnostic Blender pipeline core
│   └── domains/dice/     # dice-specific: numbering conventions, pips,
│                         #   glyph styles (from src/dice_gen)
├── datasets/             # reusable data plumbing (already written in
│                         #   scripts/, now made importable):
│                         #   group-aware splitting, COCO<->YOLO conversion,
│                         #   class remapping, ingest spot-check overlays
├── training/
│   └── detection/        # ultralytics wrapper (guts of scripts/train.py);
│                         #   future model families are sibling packages
│                         #   implementing the same Trainer interface
├── evaluation/           # pycocotools harness, class-merge eval,
│                         #   sim-to-real gap reports, prediction overlays
└── export/               # ONNX/TensorRT export + model manifest
                          #   (deployment-contract implementation)
configs/
└── domains/
    └── dice.yaml         # the domain plugin config: class list/order,
                          #   real-data sources + remap tables, exclusions
scripts/                  # thin CLIs over src/ (entry points keep working)
tests/                    # existing tests follow their code into src/
```

Core principles:
- **Domain = config + datagen plugin.** Adding a new domain (chess pieces,
  screws, produce) means a new `configs/domains/*.yaml` and a
  `src/datagen/domains/<name>/` module — no changes to datasets/training/
  evaluation/export.
- **Model family = interface.** `training/` and `evaluation/` define
  minimal `Trainer`/`Evaluator` protocols; detection (ultralytics) is the
  first implementation. A future classifier or segmenter slots in beside
  it without restructuring. (Deliberately interfaces-first, one concrete
  implementation — full classification/segmentation pipelines are out of
  scope until a real need arrives.)
- **Datasets are versioned artifacts with manifests** — extending the
  datagen manifest discipline (known-imperfect data is recorded, not
  hidden) to ingested real data (source, version, exclusions, split
  policy, group counts).
- **Behavior-preserving refactor.** The dice-detector-v1 results (metrics
  in `docs/results/2026-07-dice-detector-v1/`) are the regression
  baseline: after the refactor, re-running conversion + a smoke train +
  eval through the new CLIs must reproduce the same split counts, the same
  label files (byte-identical), and eval metrics on the same weights.

## Out of scope

- Merging or restructuring the `rl` repo (only the contract docs touch it).
- Concrete classification/segmentation implementations.
- A shared pip-installable package across the two repos (revisit when a
  second consumer of `datasets/` exists).
- Repo rename — user's call; design works either way.

## Open items for user sign-off

1. Federation-vs-monorepo recommendation (this spec assumes federation).
2. Whether to rename the repo once it's no longer only "Dice-Detection".
