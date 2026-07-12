# Asset-bisect: which object property gates grasp discovery (d20 vs DexCube)

**Date:** 2026-07-12. **Branch:** `franka-panda-pivot`.
**Prior result this builds on:** `docs/superpowers/plans/2026-07-11-joint-space-die-lift-report.md`
— identical joint-space lift config trains decisively on DexCube, fails
totally (zero grasp attempts at convergence) on the baked d20.
**Research grounding:** `.superpowers/sdd/research-asset-bisect.md`
(senior research pass + independent senior citation review, 2026-07-12;
all 10 citations verified real/accurate).

## Measured facts (ground truth for rung design, all measured 2026-07-12)

| property | baked d20 | DexCube (recipe) | gap |
|---|---|---|---|
| effective size | 30.3mm | 48.0mm | 1.6x |
| live PhysX mass | 0.0100kg | 0.2160kg | **21.6x** |
| inertia (diag) | ~9.0e-7 | ~8.3e-5 | ~92x |

Context: each Panda finger link is 0.0216kg — the d20 weighs HALF a
finger; the DexCube weighs 10 fingers. Solver params verified identical
between both configs (`_diag_object_mass_check.py`,
`_diag_dexcube_scale_check.py`).

## Hypothesis (rung 1, falsifiable)

> Raising the baked d20's mass from 0.0100kg to the DexCube's measured
> 0.216kg — holding shape, 30.3mm size, friction, and the entire
> joint-space config fixed — will produce sustained lift (authoritative
> metric `Metrics/object_pose/position_error` decisively below the
> ~0.216 do-nothing baseline) in at least 2 of 3 seeds within 1500
> iterations each.

**Mechanism (two-stage, per the citation review's reconciliation):**
(1) At 0.01kg under multi-point finger contact, PhysX produces large
corrective depenetration impulses on light objects (NVIDIA's own
engineering discussion, `isaac-sim/IsaacSim` GitHub Discussion #372
"Object Pops or Flies Away After Being Touched", NVIDIA-staff-attributed,
reproduced as recently as 2026-04; consistent with PhysX SDK docs'
"Advanced Collision Detection" tiny-object warning) — early-training
contacts fling/shove the die instead of yielding graspable outcomes.
(2) Those chaotic, unrewarding contact outcomes teach the policy to stop
touching the object: our own run data shows early die-shoving
(position_error driven ABOVE the do-nothing baseline, object_dropping
spikes ~2-6% mid-run) decaying to zero contact by convergence — the
same converge-away-from-the-rare-rewarding-event signature as this
repo's `kb/wiki/concepts/reward-hacking-and-sparse-discoverability.md`
precedent. A 21.6x mass increase removes stage (1), which removes the
training signal for stage (2).

This is a sim-artifact + learned-avoidance explanation, NOT a claim that
RL exploration is intrinsically harder for small objects — that
distinction is what the ladder tests.

## Ladder (one variable per rung; stop when the gap is explained)

Ordering principle (research doc): rule out the highest-confidence,
first-party-documented sim-artifact confound first, so later rungs can't
be second-guessed against it.

- **Rung 1 — mass:** d20 unchanged except mass 0.01 → 0.216kg (env-cfg
  `mass_props` override; the baked asset has MassAPI so the override
  modifies rather than silently no-ops — dice-demo Task-1 mechanism).
- **Rung 2 — size (only if rung 1 fails):** d20 spawn scale 0.001 →
  0.001585 (30.3 → 48.0mm), mass PINNED at 0.216kg via the same
  override (do NOT let mass scale with volume — that reintroduces
  rung 1's variable).
- **Rung 3 — shape (only if rung 2 fails):** a cube baked through this
  repo's own `bake_die_asset.py` pipeline at 48mm/0.216kg — isolates
  rolling-geometry/antipodal-pair-scarcity (Zhou & Held CoRL 2022,
  arXiv:2211.01500; Danielczuk et al.) from pipeline provenance.
- **Rung 4 — provenance (only if rung 3 fails):** rung-3 bake-pipeline
  cube vs actual DexCube — pure elimination/control rung.

If lift emerges at rung N, optionally run ONE confirmation of rung N-1's
variable in isolation before declaring rung N's property dominant (guards
against interaction effects); this is the only authorized extra run
beyond the ladder.

## Success criteria / verdict protocol (per rung)

Authoritative metric (unchanged from the die-lift experiment):
`Metrics/object_pose/position_error` decisively below the do-nothing
baseline (~0.216) and trending down, corroborated by
`Episode_Reward/lifting_object` rising clearly above its ~0.12
spawn-artifact floor. `Loss/value_function` bounded (auto-reject on
divergence). Verdict per rung requires **N=3 seeds, 2/3 reproduction**
(Henderson et al., AAAI 2018, arXiv:1709.06560 — single-seed RL verdicts
unreliable; repo realistic-noise principle). A single positive seed
never flips a rung to "succeeded."

The FIRST rung that passes also gets: 8-env instrumented eval
(`franka_checkpoint_review.py` height readout) + whole-arm-framed video,
controller-inspected — same evidence standard as the die-lift verdict.
Failing rungs need scalars only (no per-seed videos).

Budget: ~30 min/run → 3 seeds/rung, worst case 4 rungs = 12 runs
(~6h GPU-serial) + 1 confirmation run. Report ladder progress after each
rung; do not silently run the whole ladder if rung 1 passes.

## Confounds to watch (from research doc §"Failure modes")

- PhysX small-object contact artifacts can masquerade as "size matters"
  at rung 2 (CCD/contact-offset effects are size-dependent independent
  of mass).
- Mass override must be verified LIVE (`_diag_object_mass_check.py`
  pattern / printed `get_masses()` at env construction), not assumed
  from the cfg — this repo's rigid-props-silently-no-op history.
- Seed variance: the outcome is closer to a rare-event indicator than a
  smooth scalar; that's what the 2/3 rule is for.

## Out of scope

Reward/action/observation changes of any kind; curriculum over object
properties (a candidate FOLLOW-UP if the ladder localizes the property);
d4 grasp strategy; detector-in-loop observations (Phase I, still gated).
