# The reach → grasp → lift staged-progress gap

## Why this is the through-line of the whole research arc

Every numbered experiment in this project's history (1 through 14, and the
unnumbered sphere-era precursors before Experiment 1) is, at bottom, an
attempt to close one specific gap: the policy reliably learns to **reach**
toward the object almost immediately, increasingly reliably learns to
**grasp** it (as ground-truth contact sensing and antipodal checks matured),
but has **never**, across 14 numbered experiments plus their unnumbered
precursors, been observed on video genuinely lifting, carrying, and placing
the object at a goal.

## The arc, stage by stage

1. **Pre-Experiment-1 (sphere, unnumbered):** reach converges (~0.92-0.93);
   the gripper approaches and then holds a static, open pose — it never
   even attempts to close. Four sequential reward-shaping hypotheses (lift-
   weight bump, dense proximity+closure bonus, multiplicatively-gated
   alignment bonus, gripper PD-gain rescale) were tried and falsified before
   this article's numbered sequence begins — see
   [[reward-hacking-and-sparse-discoverability]] for two of these in
   detail.
2. **[[experiment-01-contact-sensor-grasp-reward]]:** the gap moves. Grasp
   is now real (~92% sustained bilateral contact) — but the arm freezes
   immediately after grasping, in all 10 inspected episodes. "Reach, grip,
   freeze" becomes this project's standing failure signature for the rest
   of the sphere era.
3. **[[experiment-02-curriculum-gated-lift-height]]** through
   **[[experiment-07-sphere-shrink]]:** six further hypotheses targeting
   lift specifically (curriculum timing, always-on dense lift reward,
   learning-rate bump, potential-based shaping, mirror-scene + stillness
   penalty, object-size shrink) — all falsified, none produce a real lift.
4. **[[experiment-08-classical-ik-guided-path]]** (spanning the sphere→cube
   pivot): a denser path-tracking reward completes training but its own
   data reveals *why* freezing is favored — a ~118:1 reward-rate imbalance
   (see [[reward-rate-arithmetic]]).
5. **[[experiment-09-antipodal-grasp-bonus]]** and
   **[[experiment-10-antipodal-threshold-action-scale-solver]]:** fixing the
   grasp-quality signal itself (magnitude-only → antipodal, see
   [[grasp-mechanics-antipodal-vs-magnitude]]) — antipodal contact
   regresses to exactly zero under joint-space control, implicating
   positioning precision, not reward design.
6. **[[experiment-11-taskspace-ik]]:** the single biggest positive result
   in the whole arc — switching to task-space/Cartesian IK-driven action
   (see [[action-space-design]]) produces the first genuine, sustained
   antipodal grasp. But the video signature is unchanged in kind: the arm
   still holds a low, static grasp pose and never lifts to height or
   carries toward the goal.
7. **[[experiment-12-stillness-reward-rate]]:** fixing a verified
   reward-rate bug in the new task-space reward produces a scalar-mixed,
   video-inconclusive result — still no confirmed lift.
8. **[[experiment-13-residual-rl]]:** a structurally different pivot
   (residual policy over a classical base controller) regresses, plausibly
   due to a missing literature warm-start step, not a disproof of the
   approach.
9. **[[experiment-14-reach-skip-curriculum]]:** removing reach from what
   the policy has to rediscover each episode (one-shot IK reset to a
   pregrasp pose) produces no improvement on the lift criterion, plus a new
   base-collapse failure mode in 2/3 inspected episodes.

## Where this stands at the end of this pass

As of Experiment 14, "pick up and move" — the project's actual stated
scope-in goal per `CLAUDE.md` — remains unachieved. Three consecutive
experiments (12, 13, 14) failed to move this specific needle, triggering
this project's own "escalate, don't keep tuning" mandate. The most-cited
still-untried candidate across multiple experiments' own "next steps"
sections is a genuinely different structural lever: **longer episodes
and/or explicit staged sub-objectives** (reach → grasp → lift → carry →
place as separate curriculum phases with their own success criteria, rather
than one flat episode/reward learning the whole sequence at once) — this
matches a standing note in this project's own working memory about future
AR4 iteration direction (longer episodes, staged decomposition, richer
drop-zone placement). Experiment 15 (in progress as of this pass, not yet
covered) continues on the reward-rate axis rather than this structural
axis — worth flagging as a candidate gap for a future pass to assess once
its own ROADMAP entry lands.

## Related experiments

All 14 — this is the connecting narrative across the entire numbered
sequence.
