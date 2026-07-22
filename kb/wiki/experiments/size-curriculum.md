# Size curriculum — both arms FALSIFIED (mixed-size DR 0/3, staged-anneal 1/3)

**2026-07-13.** Follow-up to [[asset-bisect]] (shape gates d20 grasp
discovery; 0/3-4 at 30.3mm, 1/3 at 48mm). Hypothesis: per-env die-size
variation over {48.0…30.3}mm (5 scales, mass pinned 0.216kg, joint-space
config otherwise per [[joint-space-die-lift]]) lets discovery in large
envs transfer down the size range; ≥2/3 seeds, 3000 iters, verdict =
instrumented all-30.3mm eval, ≥6/8 sustained lifts.

## Primary arm: mixed-size DR — FALSIFIED 0/3

**Result: 0/8 sustained lifts in every seed (42, 7, 123).** Lifting
reward pinned at its 0.1200 spawn-z floor all run in all seeds — no
grasp discovery at ANY size, including 48mm.

**Mechanism lesson (generalizes):** size-DR *dilutes* discovery when
discovery is the bottleneck. The transfer story needs a source signal;
at ~819 envs per size bucket, the already-marginal 48mm discovery event
(1/3 at full 4096) became ~5x rarer and never fired. DR over a
difficulty-correlated parameter helps when the easy end is *reliably*
learnable — otherwise stage it (train easy at full population first).

## Fallback arm: staged anneal (48→39.1→30.3mm, checkpoint-resumed) — FALSIFIED 1/3

Pre-authorized fallback, fired after the primary's falsification: three
`--checkpoint`-resumed stages (1000 iters each, full 4096-env population
every stage — no dilution), seeds 42/123/7. New `FrankaDieLiftJointMidEnvCfg`
(39.1mm, `--variant joint-die-mid`) added to
`tasks/franka/dice_lift_joint_env_cfg.py` for stage 2; stages 1 and 3
reused the existing `joint-die-big` (48mm) and `joint-die-heavy` (30.3mm)
variants verbatim.

**Stage-1 discovery check (48mm, instrumented eval):** 0/8, **8/8**, 0/8
— 1/3 seeds discover, exactly reproducing the asset-bisect's 48mm anchor
(same seed, 123, succeeds both times). **Stage-3 verdict (30.3mm,
instrumented eval):** 0/8, **8/8**, 0/8 — same 1/3, below the ≥2/3 bar.
All 9 training runs healthy (zero NaN, VF loss bounded).

**Mechanism lesson (the interesting result):** the transfer mechanism
itself works — this is NOT a repeat of the primary's dilution failure.
Seed 123's discovered 48mm grasp carried undegraded through both anneal
stages: `position_error` improved monotonically (~0.11 → 0.099 → 0.102
across the three stages), and the eval was 8/8 sustained lift at *both*
the 48mm stage-1 check and the final 30.3mm stage-3 check. Checkpoint-
resumed size-annealing does not destroy an already-discovered grasp.
What it cannot do is manufacture NEW discovery for seeds that never find
one — seeds 42 and 7 stayed pinned at the no-op floor (VF loss collapsing
toward ~0) through every subsequent stage. **The bottleneck is squarely
the ~1/3 base discovery rate at 48mm itself** — neither curriculum
variant (mixed-size DR or staged anneal) introduces new exploration
pressure at the point grasp discovery actually happens; both just
inherit whatever the base 3-seed discovery rate gives them and either
dilute it (primary) or faithfully propagate it without amplifying it
(fallback).

**Decision:** both pre-authorized size-curriculum arms are now
falsified. Shape itself needs a direct attack (grasp-strategy/reward
changes) — new spec/research pass required, not further object-scale
curriculum variants on this asset. Seed 123's 8/8 30.3mm result is the
project's first confirmed d20 lift+carry policy at the real target
size — a reusable existence-proof baseline for that next investigation.

Spec + full verdict (both arms):
`docs/superpowers/specs/2026-07-13-size-curriculum-design.md`. Fallback
task report: `.superpowers/sdd/task-staged-anneal-report.md`.

## Ops findings (both arms)

- New Isaac failure mode — mid-training livelock (~iter 260: log/ckpt
  frozen 2.5h, ~3 CPU cores spinning, GPU 12%, SIGKILL required;
  log-mtime stall detector now standing practice).
- `franka_checkpoint_review.py` fixed output filenames overwrite across
  same-variant evals — rename artifacts between runs (followed
  throughout the fallback arm, no collisions).
- flock waiter handoff is not FIFO; a concurrently-running d4-debug
  thread repeatedly interleaved short jobs ahead of queued launches
  during the fallback arm. Several of those d4-debug jobs matched the
  documented teardown-hang signature (own script prints `[DONE]`, log
  static >5min, GPU pinned ~12-13%) and were killed via `kill -TERM`
  per the standing safe-kill procedure.
- `cmd | tee logfile &` inside a single Bash tool call leaks the
  background child's full stdout into the tool's own output even after
  the call returns (shared pty) — use `> logfile 2>&1 < /dev/null &`
  instead for silent backgrounding.

## Standard-vs-jumbo correction (2026-07-15)

This article's "real target size" of 30.3mm is the baked asset's
*default* spawn size, not a real-world standard d20 — web research
(user-confirmed) established a real standard commercial d20 is ~20-22mm,
and 30.3mm is itself a real, commonly-sold **"jumbo"** d20 size (e.g.
Twenty Sided Store's own "Jumbo Dice 30mm D20" listing), not a mistake or
edge case. This does not change any verdict recorded above — the
30.3mm/48.0mm/etc. results stand exactly as originally reported; it only
corrects what "the real/final target size" means for *future* d20
size-related work.

Added a new forward-facing class, `FrankaDieLiftJointStandardEnvCfg` (+
`_PLAY` variant, `tasks/franka/dice_lift_joint_env_cfg.py`), inheriting
from `FrankaDieLiftJointHeavyEnvCfg` (mass pinned 0.216kg, unchanged) with
`scale=(0.000727, 0.000727, 0.000727)` — derived by fitting this file's
own four existing rung constants (0.001585/48.0mm, 0.001440/43.6mm,
0.001291/39.1mm, 0.001146/34.7mm) to an average scale-per-mm ratio of
3.302305e-5, then 22mm × 3.302305e-5 = 0.000727. Live-verified via a new
diagnostic, `scripts/_diag_d20_standard_scale_check.py` (same no-physics
headless-`SimulationApp` bounding-box-read pattern as
`_diag_die_scale_check.py`/`_diag_dexcube_scale_check.py`): measured bbox
at this scale is **21.993mm** (delta -0.007mm from the 22.0mm target),
well inside a 0.3mm tolerance — no adjustment needed. This class is not
yet used by any training run; it's a target for a future d20
grasp-strategy spec (shape itself, per this article's own verdict above,
is the next thing needing a direct attack).

## Follow-on: unified multi-die specialist-distillation (started 2026-07-16)

This article's own closing verdict ("shape itself needs a new spec/
research pass rather than further object-scale curriculum variants") is
directly picked up by
`docs/superpowers/specs/2026-07-16-unified-multi-die-specialist-distillation-design.md`
— training per-shape specialists (d8/d10/d12/d20) toward one distilled
policy. Its first result (0/9 discovery for d8/d10/d12 at their real
~16-18mm sizes) reopens a version of this article's own scale-vs-shape
question: those three shapes were never tested at this article's own
48mm cube-parity anchor, so the result can't yet be attributed to shape
vs. absolute object scale. See `ROADMAP.md`'s 2026-07-16 entry for the
full finding, including a measurement artifact in
`franka_checkpoint_review.py`'s height computation that Principal caught
by inspecting raw trajectories directly.
