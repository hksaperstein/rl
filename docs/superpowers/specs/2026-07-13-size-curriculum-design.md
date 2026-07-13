# Size curriculum: making the 30mm d20 learnable via mixed-size training

**Date:** 2026-07-13 (drafted night of 07-12). **Branch:** `franka-panda-pivot`.
**Prior result:** asset-bisect (`docs/superpowers/plans/2026-07-12-asset-bisect-report.md`)
— grasp discovery on the d20 is 0/4 at 30.3mm (deterministic fail) and
1/3 at 48mm (discoverability boundary); shape is the gate, size sets
severity. **Research grounding:** `.superpowers/sdd/research-asset-bisect.md`
Q1 (curriculum-over-object-scale findings; citation-reviewed 2026-07-12).

## Hypothesis (falsifiable)

> A single policy trained with per-env die-size variation spanning
> 30.3-48.0mm (mass pinned 0.216kg, joint-space config otherwise
> identical to the bisect's) will achieve lift+carry ON THE 30.3mm
> ENVS — `position_error` computed over the smallest-size envs
> decisively below the ~0.216 do-nothing baseline — in ≥2 of 3 seeds
> within 3000 iterations, because grasp discovery in the large-size
> envs supplies the reward signal whose policy gradient generalizes
> down the size range (the mechanism curriculum learning and domain
> randomization share; see research doc Q1).

3000 iterations (2x the bisect's 1500) because the hard sizes must
first benefit from transfer — pre-registered, not tuned post-hoc.

## Design

**Primary arm — mixed-size (size-DR):** envs deterministically assigned
one of 5 die scales {48.0, 43.6, 39.1, 34.7, 30.3}mm (spawn-scale
variants of the same baked d20; ~819 envs per size at 4096). Mass
pinned 0.216kg on all (the bisect's control). Everything else inherits
`FrankaDieLiftJointHeavyEnvCfg` unchanged.

Implementation candidate: Isaac Lab's multi-asset/per-env spawn
variation (`MultiAssetSpawnerCfg` or equivalent in the installed
version — implementer MUST verify against `/home/saps/IsaacLab` source
before the spec's design is finalized in the plan; if per-env scale
variation is unsupported, fire the fallback arm instead and record why).

**Fallback arm — staged anneal (pre-authorized, fires only if the
primary is unimplementable or FAILS):** 48.0 → 39.1 → 30.3mm, one
`--checkpoint`-resumed run per stage (1000 iters/stage, same seed
protocol). Uses only existing mechanisms (env-cfg scale value +
existing resume flag).

## Metrics / verdict protocol

- Authoritative: `Metrics/object_pose/position_error` — but the VERDICT
  metric is the per-size breakdown at the smallest size. The stock env
  logs only the all-env mean, so Task 1 must add a per-size-bucket
  metric or an eval-time per-size readout (instrumented eval on
  30.3mm-only PLAY cfg is acceptable and simplest: eval the trained
  checkpoint in an all-30.3mm env — success = sustained lifts there,
  same 8-env instrumented protocol as the bisect).
- 3 seeds (42/123/7), ≥2/3, full-run VF-boundedness check, eval video
  (full arm + table, ≥2 episodes) on the first passing seed.
- Comparison anchors (no new runs needed): bisect rung 1 (30.3mm-only:
  0/3) and rung 2 (48mm-only: 1/3).

## Success meaning / next

PASS → the platform can train the real die → Phase I (detector-derived
observations) unblocks with this policy as its base. FAIL both arms →
shape itself needs attacking (grasp-strategy/reward changes — new spec,
new research).

## Out of scope

Reward/observation changes; shape curriculum (cube→die morphing);
detector-in-loop anything; d4.
