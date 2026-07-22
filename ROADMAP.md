# ROADMAP

Forward-looking planning doc: current priorities and what's next, not a
history ledger. Every full experiment result lives in `kb/wiki/` (start at
`kb/wiki/index.md`) or `docs/superpowers/specs|plans/`; this file links out
to it rather than re-narrating it. Update after each completed plan (per
`.superpowers/sdd/progress.md`): move newly-shipped work into "Recently
landed" (one line + kb link) and refresh "Planned / near-term priorities"
below.

## Active workstreams

Nothing is currently mid-execution as of 2026-07-22 — the AR4-vs-Franka
root-cause investigation closed with Task 7, and both the
unified-multi-die-specialist-distillation and target-selection-clutter
experiments reached COMPLETE verdicts. See "Planned / near-term
priorities" below for what's queued next.

## Direction

Isaac-Lab-based robotics RL, expanding beyond the current dice/Franka work
into other tasks/robots, object detection/perception, and mobility. No
committed roadmap items beyond the items below yet — this is a stated
direction, not a scoped backlog.

## Planned / near-term priorities

Roughly in the order they'd likely be picked up:

1. **Target-selection-clutter E2** (3→4 distractors) — the next
   separately-gated stage after Stage E1's clean pass (2026-07-21); not
   auto-started by E1.
2. **Target-selection-clutter S1** (fold d8/d10 back into the clutter
   curriculum) — well-motivated now that d8/d10's grasp-discoverability
   null was closed with a positive resolution (demo-warmstart H2, both
   shapes PASS), after being deferred from the original 4-shape scope.
3. **Revisit unified-multi-die-specialist-distillation's Task 4 scope**
   to include d8/d10 alongside d12/d20, now that both are proven
   grasp-discoverable via demo-warmstart — an open decision for
   Principal (the original scope-narrowing rationale is documented in
   `kb/wiki/experiments/unified-multi-die-specialist-distillation.md`'s
   Task 4 section; not yet revisited).
4. **AR4 gripper mimic-vs-actuator dynamics conflict** — IN PROGRESS, stale
   "not yet started" corrected 2026-07-22. The mimic constraint has since
   been removed entirely (`2576e94`, `scripts/build_asset.py`'s
   `_remove_gripper_jaw2_mimic_constraint` — both jaws now fully
   independent `ImplicitActuatorCfg` PD targets). A live re-test
   (`d16aa76`'s message) found jaw2 still lands at the OPPOSITE end from
   its own commanded target in both phases — the signature of an inverted
   drive sign, not a limit-pinning defect — and a targeted mid-range
   (-0.007) sweep diagnostic was written (`d16aa76`,
   `scripts/_verify_gripper_mirror_fix.py`) to confirm this but never run.
   **2026-07-22 session: blocked before running it** — the desktop
   (`saps@home.local`), the only machine with a GPU, a working Isaac Lab
   install, the already-built AR4 USD asset, and the external
   `annin_ar4_description` ROS package `build_asset.py` depends on
   (`/home/saps/projects/annin_ws/...`, not in this repo, no GCS mirror),
   was confirmed unreachable (DNS/mDNS/SSH all failed identically across
   ~20 min of retries — not a brief reboot blip). Standing GPU-dispatch
   policy (CLAUDE.md) says fall back to cloud on desktop-unreachable, but
   AR4 has never run in cloud before and has no proven recipe (unlike
   Franka's `docs/cloud/franka-cloud-shakedown.md`) — cloud fallback here
   means building an entire new AR4 asset-build+sim pipeline from scratch
   (re-cloning the external ROS description package, verifying xacro/URDF
   import parity with the desktop's already-fixed asset, etc.), a
   cross-cutting infra investment flagged back to Principal rather than
   started unilaterally. See
   `kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md`'s
   2026-07-22 section for full detail. Next step once the desktop is
   reachable again: run the mid-range sweep diagnostic exactly as written.

See `BACKLOG.md` for further-out candidates not yet on this list.

## Recently landed

- **Franka IK dice-line pick-and-place demo** (2026-07-21) — classical
  IK-only pick/line-up/relocate of all 5 dice; 8/10 pick-and-place ops
  succeeded, d4 (this project's well-documented hardest grasp case)
  failed both attempts. `kb/wiki/experiments/franka-ik-dice-line-demo.md`.
- **AR4 grasp-discoverability research arc — CLOSED** (Experiments 1-26,
  the shape-classifier perception-debugging saga, and the AR4-vs-Franka
  root-cause investigation through Task 7; 2026-07-05 → 2026-07-21).
  Reach and antipodal-grasp-contact were solved early and reliably;
  genuine lift+carry+place was never confirmed in eval video across 26
  numbered experiments plus their sphere-era precursors. Mounting
  evidence pointed at AR4-asset-specific defects (an unenforced
  gripper jaw-mimic constraint, a classical-IK positioning miss) rather
  than a fundamental RL/reward-design problem — the direct motivation
  for the Franka platform pivot (see `CLAUDE.md`'s North Star section).
  **Task 7 (2026-07-21) tested the one concrete fix candidate this
  investigation produced** (Franka's own confirmed
  `RelativeJointPositionActionCfg` grasp-discoverability fix) directly
  against AR4 — **FALSIFIED, it does not transfer** — closing this
  investigation without a positive result; the jaw-mimic-vs-actuator
  dynamics conflict and the classical-IK positioning miss remain the
  more likely explanations. Full chronological index (one entry per
  experiment, hypothesis → verdict): `kb/wiki/index.md`. Connecting
  throughline: `kb/wiki/concepts/reach-grasp-lift-gap.md`. AR4
  pick-and-place (perception + RL reach/touch + interactive demo) remains
  working end-to-end for what it does cover; full grasp+lift+carry does
  not.
- **d8-antipodal-grasp-quality — CLOSED** (2026-07-20 → 2026-07-21) — a
  cross-platform replay of the AR4-era joint-space-vs-task-space finding
  on Franka/d8; `RelativeJointPositionActionCfg` (H_relative) confirmed
  as a genuinely joint-space fix for the grasp-discoverability collapse,
  3/3 seeds, no arm-specific IK layer needed — real North Star evidence
  that a task-space layer isn't a hidden prerequisite for a new arm to
  train. `kb/wiki/experiments/d8-antipodal-grasp-quality.md`.
- **Target-selection-clutter — COMPLETE through Stage E1** (2026-07-19 →
  2026-07-21) — 3-die clutter curriculum (distractor-count curriculum +
  a distractor-distance observation term); d12 8/8, d20 8/8 under 3
  simultaneous distractors, no wrong-die grasp observed in any inspected
  video frame. `kb/wiki/experiments/target-selection-clutter.md`.
- **Exploration-bonus grasp discovery — SPLIT** (2026-07-19 →
  2026-07-20) — a potential-based exploration bonus for gripper-closure
  attempts; mechanism-level bar fires in 1/3 seeds, behavioral bar stays
  0/24 — the first result in this project's history to land in the
  explicitly pre-registered third outcome category (not a plain
  pass/fail). `kb/wiki/experiments/exploration-bonus-grasp-discovery.md`.
- **d8/d10 demo-warmstart — CLOSED, positive resolution** (2026-07-19 →
  2026-07-20) — H1 (one-demo BC-pretrain) falsified both shapes; H2
  (checkpoint warm-start from the d12 specialist) PASSED both shapes —
  the original grasp-discoverability null was a cold-start exploration
  problem, not an intrinsic physical or reward-design barrier.
  `kb/wiki/experiments/d8-d10-demo-warmstart.md`.
- **Unified multi-die specialist-distillation — COMPLETE** (2026-07-16 →
  2026-07-19) — per-shape specialist → distill → RL-fine-tune pipeline
  for a single policy that grasps a commanded die; narrowed to d12/d20 on
  real evidence (d8/d10 genuinely null at the time), RL fine-tuning fully
  recovers a real BC/DAgger closed-loop-transfer regression to an exact
  8/8 match with each frozen specialist.
  `kb/wiki/experiments/unified-multi-die-specialist-distillation.md`.
- **d4 edge-grasp rungs 0 and 1 — both FALSIFIED** (2026-07-13 →
  2026-07-15) — stock Franka jaws physically cannot straddle the
  tetrahedron along its edge-pair axis (rung 0); a rigid V-notch
  fingertip fixture sweeps the die aside without ever engaging it
  (rung 1). `kb/wiki/experiments/dice-pick-demo.md`'s "Open follow-ups"
  section.
- **Cloud training pipeline PROVEN** (2026-07-13, re-verified
  2026-07-14/15) — GCP SPOT L4, full create→install→train→GCS-sync→
  teardown cycle exercised twice independently; real per-SKU GCP pricing
  and SPOT-preemption/checkpoint-resume handling both documented.
  `kb/wiki/concepts/cloud-training.md`.
- **RL joint-space die-lift, asset-bisect, size-curriculum**
  (2026-07-12 → 2026-07-13) — isolates *shape* (not action space, mass,
  or bake pipeline) as the reliability gate for d20 grasp discovery;
  yields the project's first confirmed d20 lift+carry at the real
  30.3mm target size. `kb/wiki/experiments/joint-space-die-lift.md`,
  `kb/wiki/experiments/asset-bisect.md`,
  `kb/wiki/experiments/size-curriculum.md`.
- **Dice + Franka + detection convergence milestone** (2026-07-11) —
  commanded die type → trained `vision/` detector identifies/localizes it
  among five dice → staged DiffIK picks the correct one; 4/5 die types
  passing (d4 the sole, pre-declared permitted failure). Scripted
  controller, not RL. `kb/wiki/experiments/dice-pick-demo.md`.
- **Vision platform** (`vision/` monorepo, 2026-07-10 → 2026-07-13) —
  dice-detector-v1's real-photo transfer collapse on d8/d10, fixed by the
  datagen-v2 close-up slice (mAP50 d8 0.090→0.442, d10 0.097→0.534),
  exposing a d6 glyph-confound regression in turn.
  `kb/wiki/concepts/vision-platform.md`.
