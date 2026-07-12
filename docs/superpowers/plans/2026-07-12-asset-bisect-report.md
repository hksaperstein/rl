# Asset-bisect ladder — running report

Spec: `docs/superpowers/specs/2026-07-12-asset-bisect-design.md`
Plan: `docs/superpowers/plans/2026-07-12-asset-bisect.md`
Protocol: 3 seeds (42/123/7) per rung, verdict requires 2/3; authoritative
metric `Metrics/object_pose/position_error` last-100 mean vs the ~0.216
do-nothing baseline, corroborated by `lifting_object` vs its 0.12
spawn-artifact floor. All numbers read from full event files.

## Rung 1 — mass (d20 @ 30.3mm, 0.01→0.216kg): FAILED 0/3

| seed | pos_err last-100 | lifting_object | VF bounded |
|---|---:|---|---|
| 42 | 0.355 | 0.12 floor, flat | ✓ (max 0.29) |
| 123 | 0.382 | 0.12 floor, flat | ✓ (max 0.46) |
| 7 | 0.386 | 0.12 floor, flat | ✓ (max 0.15) |

A 21.6x mass increase changed nothing; all three failure curves are
near-identical (deterministic/structural cause, not exploration luck).
**The rung-1 hypothesis (PhysX light-object depenetration impulses as the
gate) is falsified.** Runs: `logs/train_franka_jointdieheavy/`
2026-07-12_12-52-49 / 13-25-30 / 13-58-09.

## Rung 2 — size (d20 @ 48.0mm, mass pinned 0.216kg): FORMAL FAIL (1/3), split

| seed | pos_err last-100 | lifting_object | outcome |
|---|---:|---|---|
| 42 | 0.401 | 0.12 floor | FAIL |
| 123 | **0.0956** | **13.35 / 15** | **FULL SUCCESS** |
| 7 | 0.405 | 0.12 floor | FAIL |

Seed 123's policy grasps, lifts, and carries the 48mm d20 — position
error BETTER than the DexCube reference run's own 0.105. Under the
spec's 2/3 rule this rung formally fails, but the split is the finding:

- **Size is load-bearing**: 30.3mm → deterministic failure (identical
  curves, 6/6 runs across rungs 0-1); 48mm → grasp discovery is possible
  but unreliable (1/3). The task crossed from "unlearnable" to "edge of
  discoverability."
- **The baked asset is NOT physically broken**: a learned policy handles
  it expertly once found. Rules out the bake pipeline producing
  cursed-contact assets (the strongest AR4-era parallel).
- At 48mm the comparison "DexCube 1/1 vs d20 1/3" is confounded by n=1
  on the cube side and conflates shape with provenance → rung 3.

Runs: `logs/train_franka_jointdiebig/` 2026-07-12_14-34-24 (s42) /
15-07-49 (s123) / 15-41-15 (s7). Seed-123 eval + video: see below.

## Rung 3 — shape (bake-pipeline cube @ 48mm, 0.216kg): in progress

Measures whether a flat-faced cube of the same pipeline provenance
recovers RELIABLE training at the size where the rounded d20 is a 1/3
coin flip. Variant `joint-cube-baked`, asset `assets/dice/cube48_physics.usd`.
Provenance caveat recorded: the cube's mesh is authored programmatically
(same schema-bake code path, but not a Blender export like the dice), so
rung 3 isolates shape given the schema-bake half of the pipeline; full
Blender-provenance parity would need a Blender-exported cube (rung 4
territory if it matters).

## Follow-up queued (needs its own spec)

Size curriculum (train at 48mm where discovery works, shrink toward
30.3mm) — recommended by the research pass's Q1 findings and now
supported by this ladder's own rung-2 split.
