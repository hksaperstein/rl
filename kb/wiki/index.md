# AR4 Manipulation Research — Wiki

This wiki compiles the research history of this repo's AR4 pick-and-place
project: the effort to get a simulated AR4 arm (Isaac Lab / Isaac Sim) to
reliably grasp an object and move it to a goal location via RL (PPO). Per
this repo's [North Star](../../CLAUDE.md), the long-term goal is a general,
reusable manipulation research platform — multiple tasks, objects, and arms
sharing the same infrastructure and, ideally, the same methodology — but the
work indexed here is the current, deliberately narrow phase: one AR4 arm,
one graspable object (first a sphere, later a cube), pick it up and move it
to a goal. That narrow phase is still unsolved as of Experiment 14: grasp
contact is reliably achieved, but a genuine lift-and-carry has not yet been
observed in evaluation video.

## Contents

- **Experiments** (`experiments/`) — one article per numbered experiment
  (Experiment 1 through Experiment 14), each with hypothesis, design,
  quantitative result, qualitative video finding, and verdict. Linked
  individually below.
- **Concepts** (`concepts/`) — cross-cutting themes that recur across
  multiple experiments, each synthesizing what's been learned about that
  theme across the whole arc rather than repeating it per-experiment.
  Linked individually below.

### Experiments (chronological)

1. [[experiment-01-contact-sensor-grasp-reward]] — ground-truth bilateral
   contact sensing replaces geometric grasp proxies; grip achieved, lift
   still doesn't emerge (sphere).
2. [[experiment-02-curriculum-gated-lift-height]] — dense lift-height term
   gated on at iteration 700; fired as designed, negligible real effect
   (sphere).
3. [[experiment-03-always-on-lift-height]] — same term active from
   iteration 0; still no lift, points at PPO entropy collapse (sphere).
4. [[experiment-04-sa-ppo-lr-bump]] — fixed learning-rate bump at the point
   the literature flagged as critical; no measurable improvement (sphere).
5. [[experiment-05-potential-based-reward-shaping]] — Ng/Harada/Russell
   potential shaping; a discount-handling bug made holding position
   actively costly (sphere).
6. [[experiment-06-mirror-scene-stillness-penalty]] — randomized spawn,
   mirrored goal, grasp-gated stillness penalty; a sign-convention bug
   found and fixed; no genuine lift (sphere).
7. [[experiment-07-sphere-shrink]] — shrink the sphere to test the
   aperture-margin hypothesis; falsified (sphere).
8. [[experiment-08-classical-ik-guided-path]] — live classical-IK
   path-tracking reward; completed on the cube after the sphere→cube
   pivot; its data exposes a 118:1 reward-rate imbalance that motivates
   Experiment 9.
9. [[experiment-09-antipodal-grasp-bonus]] — replaces magnitude-only
   contact reward with a geometric antipodal check, at a much lower
   weight; reward dominance reverses from grasp-favoring to path-favoring.
10. [[experiment-10-antipodal-threshold-action-scale-solver]] — physics-
    derived antipodal threshold, halved action scale, boosted solver
    iterations; antipodal signal regresses to exactly zero.
11. [[experiment-11-taskspace-ik]] — task-space/Cartesian IK-driven action
    replaces joint-space control; first genuine sustained antipodal grasp
    contact this project has seen, after fixing a critic-divergence bug.
12. [[experiment-12-stillness-reward-rate]] — fixes a verified reward-rate
    bug (net +1.0/step for freezing after grasp); result is scalar-mixed
    and video-inconclusive.
13. [[experiment-13-residual-rl]] — residual policy over a classical
    waypoint-seeking base controller; a genuine regression, likely missing
    the literature's warm-start step.
14. [[experiment-14-reach-skip-curriculum]] — one-shot IK reset to a
    pregrasp pose, skipping reach; no improvement on the success criterion,
    plus a new base-collapse failure mode.

*(Experiments 15 through 24 are not yet compiled into their own articles —
see the coverage boundary note below.)*

25. [[experiment-25-touch-goal-reach]] — direct user structural decision to
    drop grasp/lift entirely after six prior mechanism-fix attempts (17-22)
    failed and the task's own reward reintroduced Experiment 16's
    diagnosed wedging-exploit shape; reduces scope to two-stage sequential
    end-effector reaching. A pre-training review caught a running-max
    dead-zone defect before any training run; the actual training run
    itself has not yet been executed as of this pass.

### Concepts

- [[reward-rate-arithmetic]] — the "grasp and freeze" bug class: net
  per-step incentive arithmetic that rewards holding a static state.
- [[action-space-design]] — joint-space vs. task-space/IK vs. residual
  action formulations, and what changed when the action space changed.
- [[ppo-critic-divergence]] — new-action-mechanism instability bugs
  (Experiments 11 and 13) and how they were diagnosed and fixed.
- [[grasp-mechanics-antipodal-vs-magnitude]] — magnitude-only bilateral
  contact vs. geometric antipodal/force-closure grasp checks.
- [[reach-grasp-lift-gap]] — the through-line of this entire research arc:
  reach is solved, grasp is increasingly solved, lift never emerges.
- [[reward-hacking-and-sparse-discoverability]] — the tradeoff between a
  dense reward term being exploitable and a correct term being too sparse
  to ever be found by exploration.
- [[citation-verification-practice]] — the recurring pattern of senior
  review catching fabricated or overstated citations in delegated
  literature research, across nearly every research pass this session.
- [[sim-physics-fidelity]] — dt/decimation control-period-preserving
  changes, PhysX's opaque auto-compute collision offsets, EE-frame
  verification methodology, and the settle-time/dt coupling bug class
  (2026-07-09, post-dates the rest of this first pass — see the coverage
  boundary note below).
- [[staged-reward-co-satisfiability]] — running-max/potential-based staged
  rewards require stages that are co-satisfiable along one trajectory, not
  spatially opposed; the generalized lesson from Experiment 25's
  pre-training dead-zone catch (2026-07-09, also post-dates the rest of
  this first pass).

## Scope of this first pass

This pass (compiled 2026-07-07) covers the numbered AR4 pick-and-place
experiments documented in `ROADMAP.md` through Experiment 14. **Not yet
covered** (per `kb/README.md`'s stated scope): the perception/shape-
classifier debugging saga, the LiDAR investigation, and the
literature-research docs under `docs/superpowers/specs/research/` beyond
what's cited from individual experiment/concept articles. Experiment 15 is
still training as of this pass and has no ROADMAP entry yet — not covered
here, to be added in a later pass. See `kb/README.md` for the wiki's
structure and conventions, and `ROADMAP.md` for the full, unabridged
chronological source record.

## Coverage boundary as of 2026-07-09

`ROADMAP.md`'s "Known follow-ups" section has grown substantially since the
2026-07-07 pass above — it now runs through item 10, covering (at minimum)
Experiment 24 Gate 1's scripted-oracle stall (item 6), the classical
(non-RL) IK reachability investigation (items 7-8), a 2026-07-09
physics-fidelity verification pass (item 9), and Experiment 25's
touch-goal-reach structural pivot (item 10). **Items 6-8, and the numbered
Experiments 15 through 24 that fall between this wiki's first pass and
Experiment 25, are not individually compiled into their own articles yet**
— that backfill is a separate, larger gap left for a future pass, not
attempted here. Two exceptions exist so far: item 9's physics-fidelity
content (dt/decimation, collision offsets, EE-frame verification
methodology, and a settle-time/dt coupling bug class) is covered in
[[sim-physics-fidelity]], with item 9's classical-IK contact-sensor finding
cross-linked from [[reach-grasp-lift-gap]]'s closing sections; and item
10 (Experiment 25) is covered in [[experiment-25-touch-goal-reach]], with
its structural-pivot narrative folded into [[reach-grasp-lift-gap]]'s
newest closing section and its running-max dead-zone finding generalized
in [[staged-reward-co-satisfiability]]. Silence on items 6-8 and
Experiments 15-24 here means "not yet compiled," not "nothing happened" —
see `ROADMAP.md` itself for the full record of those items in the
meantime.
