# Size curriculum (mixed-size DR) — FALSIFIED 0/3

**2026-07-13.** Follow-up to [[asset-bisect]] (shape gates d20 grasp
discovery; 0/3-4 at 30.3mm, 1/3 at 48mm). Hypothesis: per-env die-size
variation over {48.0…30.3}mm (5 scales, mass pinned 0.216kg, joint-space
config otherwise per [[joint-space-die-lift]]) lets discovery in large
envs transfer down the size range; ≥2/3 seeds, 3000 iters, verdict =
instrumented all-30.3mm eval, ≥6/8 sustained lifts.

**Result: 0/8 sustained lifts in every seed (42, 7, 123).** Lifting
reward pinned at its 0.1200 spawn-z floor all run in all seeds — no
grasp discovery at ANY size, including 48mm. Spec + full verdict:
`docs/superpowers/specs/2026-07-13-size-curriculum-design.md`.

**Mechanism lesson (generalizes):** size-DR *dilutes* discovery when
discovery is the bottleneck. The transfer story needs a source signal;
at ~819 envs per size bucket, the already-marginal 48mm discovery event
(1/3 at full 4096) became ~5x rarer and never fired. DR over a
difficulty-correlated parameter helps when the easy end is *reliably*
learnable — otherwise stage it (train easy at full population first).
Hence the pre-authorized staged-anneal fallback: 48→39.1→30.3mm,
checkpoint-resumed, full population per stage.

**Ops findings:** new Isaac failure mode — mid-training livelock
(~iter 260: log/ckpt frozen 2.5h, ~3 CPU cores spinning, GPU 12%,
SIGKILL required; log-mtime stall detector now standing practice);
`franka_checkpoint_review.py` fixed output filenames overwrite across
same-variant evals (recover from per-launch tee logs; rename artifacts
between runs); flock waiter handoff is not FIFO.
