# Asset-bisect ladder — shape gates grasp discovery

**Date:** 2026-07-12 (branch `franka-panda-pivot`)
**Spec:** `docs/superpowers/specs/2026-07-12-asset-bisect-design.md`
**Report:** `docs/superpowers/plans/2026-07-12-asset-bisect-report.md`
**Precursor:** [[joint-space-die-lift]] (d20 fails, DexCube succeeds —
this ladder found out why).

## Question

Which object property makes the 30.3mm/0.01kg baked d20 unlearnable for
the validated joint-space Franka lift recipe: mass, size, shape, or
asset-pipeline provenance? One variable per rung; 3 seeds (42/123/7)
per rung; verdict = 2/3; authoritative metric
`Metrics/object_pose/position_error` last-100 vs ~0.216 do-nothing
baseline.

## Results

| rung | change | outcome |
|---|---|---|
| 1 | mass 0.01→0.216kg (21.6x) | **0/3** — identical failure curves |
| 2 | + size 30.3→48mm (mass pinned) | **1/3** — seed 123 FULLY succeeded (0.0956, better than DexCube's 0.105); first learned die lift on the platform |
| 3 | our-pipeline CUBE @ 48mm/0.216kg | **3/3** — 0.116/0.174/0.162, all lifting ~13/15 |
| 4 | (provenance) | unnecessary — rung 3 IS the provenance control |

## Conclusions

- **Shape is the reliability gate**: at identical size/mass/pipeline,
  flat-faced cube 3/3 vs rounded d20 1/3. Mechanism: flat parallel
  faces give random exploration a wide antipodal-grasp basin (clumsy
  contact is occasionally rewarding); a near-sphere rolls away from
  clumsy contact — grasp-affordance scarcity expressed as an RL
  exploration failure.
- **Size modulates severity**: the d20 at 30.3mm is 0/6 across all runs
  (deterministic, near-identical curves); at 48mm it's a 1/3 coin flip.
- **Mass ruled out** (rung 1) — falsifying the PhysX
  light-object-impulse hypothesis for this failure.
- **Bake pipeline exonerated**: its cube trains at DexCube-grade
  reliability. (The AR4-era fear — "our conversions make cursed
  assets" — does not apply to this pipeline.)
- Seed determinism note: failures at 30mm were seed-independent and
  near-identical; at 48mm the outcome is seed-dependent — the task sits
  at the discoverability boundary there.

## Gotchas recorded

- **Bare-`Mesh` default prim silently loses PhysX collision** when the
  asset is referenced into an `Xform` destination — mass/inertia schema
  reads stay correct, the object just falls through the table. Caught
  by a zero-action height trace vs a known-good control. All
  from-scratch assets must use Xform-root/Mesh-child structure
  (`scripts/bake_die_asset.py` cube mode does, post-fix).
- 300-iteration diagnostics mislead at the discoverability boundary —
  full-run seed batches are the unit of evidence here.

## Next

Object curriculum (own spec required): train where discovery is
reliable (48mm / cube-like), anneal toward the 30mm d20 — supported by
the research pass's Q1 curriculum findings and rung 2's own split.
Related standing memory: realistic-noise principle (3-seed minimum is
now precedent).
