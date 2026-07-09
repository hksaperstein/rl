# Staged reward co-satisfiability: running-max/potential shaping requires stages that lie on one trajectory

## The pattern

A running-max (or more generally, potential-based) staged reward tracks the
highest value a combined potential has reached so far in an episode and pays
out only increases above that historical peak. This is a sound, standard
technique — but only under an assumption that is easy to state and easy to
silently violate when the mechanism is reused on a new task: **the stages
being tracked must be co-satisfiable along one continuous trajectory**,
meaning progress toward a later stage must not require passing through
states where the combined potential is *lower* than what an earlier stage
already banked, for any non-trivial fraction of the intervening path. When
two stages are instead spatially or kinematically opposed — satisfying the
later one requires moving away from wherever the earlier one peaked — the
combined potential is not monotonic along the natural trajectory, and
running-max produces a **dead zone**: exactly zero incremental reward from
the moment the first stage is banked until the raw combined value happens to
re-exceed that peak again, however late in the trajectory that occurs.

This is a distinct bug class from [[reward-rate-arithmetic]]. That concept's
instances are all net per-step sign/weight miscalculations (a penalty
registered with the wrong sign, a bonus outweighing a penalty by two orders
of magnitude) — arithmetic errors in an otherwise structurally sound reward.
This one is not an arithmetic error at all; the formula can be implemented
exactly as intended and still produce the same qualitative symptom (a
policy that appears to freeze after partial progress), because the
*mechanism itself* — running-max over a sum of independently-peaked
potentials — doesn't transfer to a new stage geometry the way it did to the
one it was validated on.

## Where this came from: Experiment 25's caught-before-training defect

[[experiment-25-touch-goal-reach]]'s reward reused `staged_milestone_bonus`'s
running-max pattern, previously validated on the lift task (reach → grasp →
lift → carry-to-goal), where all four stages sit on one trajectory: lifting
the cube doesn't require moving away from where "reach" peaked, and carrying
toward the goal doesn't require abandoning the lift. Experiment 25 reduced
the task to two stages 0.42m apart — touch the cube's top, then reach a
fixed goal point — and combined them as two independent `tanh` bumps
(peaking near the touch point and near the goal respectively), summed, then
tracked with running-max.

Because the two peaks are spatially separated, the summed raw potential
actually *dips* — from ~0.3 at the touch point down to ~0.02 partway to the
goal — before climbing back to ~0.7 at the goal itself. Under running-max,
once 0.3 is banked at touch, the tracked reward is pinned at exactly zero
for the entire dip, not recovering until the raw value re-exceeds 0.3, which
this project's review found does not happen until roughly 93% of the way to
the goal. A policy trained against this reward would have no incremental
signal for the great majority of the post-touch trajectory — a structural
setup for "touch-and-freeze," the same qualitative failure signature as the
sphere era's original "reach, grip, freeze," but this time caused by the
reward mechanism itself rather than by anything the policy failed to
discover. This was caught by a final whole-branch review before any
training run, independently re-derived by two separate reviewers rather
than taken on the first reviewer's word.

## The generalizable lesson

Before reusing a running-max/staged-milestone reward mechanism on a new
task, explicitly check whether the new stages' geometry is co-satisfiable
the way the mechanism's originally-validated use case was — not just
whether the mechanism worked previously. A mechanism proven sound for one
staged task does not automatically transfer to a different staging of
different stages; the soundness argument depends on the specific spatial/
kinematic relationship between the stages, not on the running-max
technique in the abstract. Two concrete diagnostics that generalize beyond
this specific case: (1) does achieving a later stage ever require the
combined potential to pass through a value below an earlier stage's own
peak, for a non-trivial stretch of the natural trajectory — if so, running-
max will produce a dead zone there; (2) is each stage's own potential
individually monotonic along the path to the *next* stage (not just at its
own peak) — Experiment 25's fix (a single, genuinely monotonic post-touch
potential, `0.3 + 0.7·clamp(1 - goal_dist/touch_to_goal_dist, 0, 1)`,
verified via a real sim-independent pytest suite rather than assumed
correct by construction) is the general pattern: replace independently-
peaked bumps summed after the fact with one potential function that is
provably non-decreasing along the intended trajectory, and prove it with a
test, not a visual read of the formula.

## Related concepts

[[reward-rate-arithmetic]] — a related but mechanistically distinct bug
class producing the same "freeze after partial progress" symptom: that one
is a sign/weight arithmetic error, this one is a structural non-monotonicity
in a staged running-max potential caused by stage geometry. Worth checking
both when diagnosing a future freeze-after-progress finding, since scalars
alone won't distinguish which is at fault. [[reach-grasp-lift-gap]] —
Experiment 25's broader structural pivot (dropping grasp/lift rather than
fixing it again) is the context this defect was caught within.

## Related experiments

[[experiment-25-touch-goal-reach]] — the sole instance of this pattern so
far in this project's history, caught during pre-training review rather
than after a training run.
