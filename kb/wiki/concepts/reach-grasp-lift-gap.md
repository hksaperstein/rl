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

## A related but distinct instance: the same gap, on the classical (non-RL) path

ROADMAP.md item 9 (2026-07-09, not yet compiled as its own experiment
article; see [[sim-physics-fidelity]] for what that pass covered) surfaced
a version of this same reach-grasp gap that is worth distinguishing clearly
from everything above, because it did not arise from an RL training run at
all. Every numbered stage in this article so far is about a trained
policy's behavior; this finding is about a purely classical, closed-form-IK
pick-cycle demo (`scripts/interactive_joint_demo.py`, fixed wrist held at
q4=q5=q6=0) that had never previously been run with its own gripper
contact sensors enabled. Once instrumented, it showed exactly 0.0N contact
force on both jaws across every cycle tested - not a partial or near-miss
contact, a total miss: the gripper closes on empty space, with jaw
terminal positions varying cycle-to-cycle rather than showing the
consistent stopping position real contact would produce. A first
hypothesis (settle time silently halving in real duration when `sim.dt`
was halved earlier in the same pass, since the script counts raw substeps
rather than deriving wait time from `env.physics_dt`; see
[[sim-physics-fidelity]]'s writeup of this bug class) was tested, fixed
(commit `e00dd11`), and did not resolve the miss, ruling it out as the
cause.

This connects directly to ROADMAP.md item 8's classical-IK finding
(`scripts/grasp_demo_v2.py`'s DLS-polish-from-grid-search approach getting
the gripper visibly close to the cube but never achieving contact) - this
pass's contact-sensor instrumentation confirms that same "gets close but
doesn't reliably center the object between the open jaws before closing"
signature with ground-truth force data, rather than inferring it from video
frames or residual distance alone as item 8 did. The root cause of the
miss itself is not yet established - a separate investigation into it was
running concurrently with this pass and is intentionally left open here,
not preempted. What's confirmed so far is only that the miss is real,
exactly zero (not partial), and not explained by the settle-time bug.

## The pivot: dropping grasp/lift entirely, not fixing it again (Experiment 25)

This section jumps ahead of the numbered arc above by ten-plus experiments
— Experiments 15 through 24 are not yet compiled into their own articles,
the same acknowledged gap as the classical-IK section above (see
`index.md`'s coverage-boundary note) — to record a genuinely new kind of
stage in this gap's history: for the first time, the response to "grasp/
lift still doesn't reliably work" was not another mechanism fix, but a
direct structural decision to remove the requirement.

By the time Experiment 25 was scoped (ROADMAP.md item 10, 2026-07-09), two
separate findings closed off "train `pickplace_mirror_env_cfg.py` from
scratch" as the next step, each on its own: **(a)** six consecutive prior
experiments (17-22) had each targeted a different angle on the same
underlying mechanical defect — the gripper's two jaws are not actually
mechanically coupled (the source URDF's `mimic` constraint is confirmed
unenforced by Isaac Sim's USD import) — and both a physics-level fix
(Experiment 19) and a software-level fix (Experiment 22) made it worse
rather than better; **(b)** `pickplace_mirror_env_cfg.py`'s own production
reward (`staged_milestone_bonus`, built on `_raw_lift_progress_mirrored`)
turned out to still combine reach/grasp/lift/goal as a plain **ungated**
weighted sum — precisely the exploitable shape Experiment 16 already
diagnosed (the wrist-wedging finding: a policy scoring well on a lift-
shaped reward while the cube was never actually gripped by the fingers at
all, confirmed only after the user directly challenged the controller's
own video read and a fresh instrumented rollout was run — see
[[sim-physics-fidelity]]'s discipline of verifying visual/behavioral claims
with real sensor data rather than eyeballed frames) — without Experiment
17's grasp-gating fix, which lives only in a separate env-cfg lineage
(`pickplace_graspgated_env_cfg.py`) that `pickplace_mirror_env_cfg.py` never
inherited.

Flagged to the user rather than trained blind against those two known
risks. The user's direct decision: stop attempting a seventh fix to the
same jaw-coupling defect, and stop reusing a reward shape already known to
be exploitable — instead, **drop grasp/lift from the task entirely**,
reducing scope to two-stage sequential end-effector reaching (touch the
cube's top, then reach a fixed goal point), leaning on the one sub-behavior
that has converged reliably (~0.92-0.95) across nearly every experiment in
this project's history, independent of reward or action-space design. See
[[experiment-25-touch-goal-reach]] for the full design and a second,
distinct finding this pivot surfaced on its own — a running-max reward
mechanism, sound for the lift task it was built for, turning out to be
unsound for this new pair of spatially-opposed stages (see
[[staged-reward-co-satisfiability]]), caught by review before any training
run.

This is a different kind of stage than 1 through 9 above: those are all
attempts to close the gap by finding the right mechanism; this is the first
point in the project's history where the gap itself was judged not worth
continuing to chase with the current object/gripper hardware and reward
family, at least for now. The North Star's broader manipulation goal is
unchanged, but this specific narrow phase's definition of success was
renegotiated rather than the mechanism retried an eighth time.

## Related experiments

All 14 — this is the connecting narrative across the entire numbered
sequence — plus [[experiment-25-touch-goal-reach]], a later structural
pivot away from this gap rather than a further attempt to close it (see the
section above; Experiments 15-24 in between remain an acknowledged
uncompiled gap).
