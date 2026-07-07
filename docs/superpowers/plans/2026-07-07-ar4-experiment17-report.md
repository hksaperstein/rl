# Experiment 17 Training Run Report: Grasp-Verification-Gated Lift/Goal-Tracking (Full 1500 Iterations)

**Date:** 2026-07-07
**Log Directory:** `/home/saps/projects/rl/logs/train/2026-07-07_15-41-37`
**Training Status:** COMPLETED (1500 iterations)

## Overview

Experiment 17 adds a genuine bilateral-antipodal-jaw-contact gate in front
of `lifting_object` and `object_goal_tracking` (both were previously
gated only on cube height in Experiment 16, which turned out to permit a
non-grasp "wedge/pin" exploit — see
`docs/superpowers/specs/2026-07-07-ar4-experiment17-grasp-gated-lift-design.md`
for the full root-cause and design rationale). Implemented as
`Ar4PickPlaceGraspGatedEnvCfg` (`tasks/ar4/pickplace_graspgated_env_cfg.py`),
launched via `scripts/train.py --graspgated`.

The 300-iteration diagnostic (Task 4, verification-only,
`logs/train/2026-07-07_15-36-01/`) passed both formal gate checks
(`Loss/value_function` bounded and net-declining across all 300 points, no
tracebacks), clearing this full run. It also flagged, as expected and
anticipated by the design spec's own success criteria, that
`Episode_Reward/lifting_object` and `Episode_Reward/object_goal_tracking`
were exactly 0.0 at all 300/300 logged points — the grasp-gate had not
fired even once in that window, versus Experiment 16 (the ungated
predecessor) already showing `lifting_object` 81.3% nonzero by the
equivalent 150-iteration mark. This report's specific job is to resolve,
using the full 1500-iteration run's actual sampled trajectory (not just
first/last/max), whether that gate ever fires within this training
budget, and if so whether its nonzero rate then grows (delayed but real
discovery) or stays pinned near zero (the gate is too hard to discover in
this budget — itself a valid, anticipated, reportable outcome per the
design spec's own success criteria, not a failure to hide).

## Verification Results

### Checkpoint Integrity
- **Config `save_interval`:** 50
- **Expected checkpoints:** 1500 / 50 + 1 = 31
- **Actual checkpoints:** 31 — confirmed
- **Checkpoint iterations:** 0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1499 — confirmed
- **model_1499.pt exists:** YES (1,199,029 bytes, modified 2026-07-07 15:55:09) — confirmed
- **model_0.pt:** 1,198,359 bytes, modified 2026-07-07 15:41:44 — consistent with run start

### TensorBoard Event File
- **Event file:** `/home/saps/projects/rl/logs/train/2026-07-07_15-41-37/events.out.tfevents.1783453303.home.65271.0`
- **Size:** 2,092,516 bytes (~2.0M)
- **Modification time:** 2026-07-07 15:55:09
- **model_1499.pt modification time:** 2026-07-07 15:55:09
- **Status:** Event file mtime matches checkpoint completion time — confirmed. Run started 15:41:44, completed 15:55:09 (~13.4 minutes wall clock, consistent with Experiment 16's ~13.5-minute prior run at the same `num_envs=4096`, and within the ~15-25 minute guide the launch brief gave).
- **Console log:** `/tmp/exp17_train_stdout.log`. Last printed iteration block (1493/1500 — the log's last full block before process exit, consistent with buffered stdout flush ordering; the checkpoint/event-file evidence above independently confirms all 1500 iterations actually ran): `Mean reward: 1.65`, `Mean episode length: 250.00`, `Episode_Reward/reaching_object: 0.4794`, `Episode_Reward/lifting_object: 0.0000`, `Episode_Reward/object_goal_tracking: 0.0000`, `Episode_Termination/cube_reached_goal: 0.0036`, `Curriculum/action_rate_curr: -0.1000`, `Curriculum/joint_vel_curr: -0.1000` (curriculum already fired by this point, confirmed directly below).
- **No tracebacks/exceptions:** a case-insensitive sweep for `traceback|error|exception` (excluding the same known-benign `[Warning] [omni.usd]` USD-stage asset-loading noise documented in the Task 4 diagnostic report) found zero matches in the full-run log.

## Critical Check: Did `Episode_Reward/lifting_object` Ever Fire Over the Full 1500-Iteration Run?

This is the specific question this task was scoped to resolve, per the
controller's framing: does the diagnostic's all-zero 300-iteration window
eventually give way to real, growing gate discovery somewhere in the
remaining ~1200 iterations, or does it stay pinned at or near zero for the
entire run?

**Full-run scalar extraction (exact command output, `EventAccumulator`
scanning all 1500 logged points, not a sample):**
```
=== Episode_Reward/lifting_object ===
  points: 1500 first: 0.0 last: 0.0 max: 0.0 min: 0.0 nonzero: 0 / 1500
=== Episode_Reward/object_goal_tracking ===
  points: 1500 first: 0.0 last: 0.0 max: 0.0 min: 0.0 nonzero: 0 / 1500
=== Episode_Reward/object_goal_tracking_fine_grained ===
  points: 1500 first: 0.0 last: 0.0 max: 0.0 min: 0.0 nonzero: 0 / 1500
```

**Sampled trajectory, `Episode_Reward/lifting_object`, every 150
iterations across the full run (identical pattern for
`object_goal_tracking` and `object_goal_tracking_fine_grained` — all
three are 0.000000 at every one of these points):**
```
iteration=   0, value=0.000000
iteration= 150, value=0.000000
iteration= 300, value=0.000000
iteration= 450, value=0.000000
iteration= 600, value=0.000000
iteration= 750, value=0.000000
iteration= 900, value=0.000000
iteration=1050, value=0.000000
iteration=1200, value=0.000000
iteration=1350, value=0.000000
iteration=1499, value=0.000000
```

**Windowed nonzero-rate table (10 windows of 150 iterations each), for
direct comparison against Experiment 16's equivalent table:**

| Window | Iterations | lifting_object nonzero | object_goal_tracking nonzero |
|---|---|---|---|
| 0 | 0-149 | 0/150 (0.0%) | 0/150 (0.0%) |
| 1 | 150-299 | 0/150 (0.0%) | 0/150 (0.0%) |
| 2 | 300-449 | 0/150 (0.0%) | 0/150 (0.0%) |
| 3 | 450-599 | 0/150 (0.0%) | 0/150 (0.0%) |
| 4 | 600-749 | 0/150 (0.0%) | 0/150 (0.0%) |
| 5 | 750-899 | 0/150 (0.0%) | 0/150 (0.0%) |
| 6 | 900-1049 | 0/150 (0.0%) | 0/150 (0.0%) |
| 7 | 1050-1199 | 0/150 (0.0%) | 0/150 (0.0%) |
| 8 | 1200-1349 | 0/150 (0.0%) | 0/150 (0.0%) |
| 9 | 1350-1499 | 0/150 (0.0%) | 0/150 (0.0%) |

**Answer to the discovery-curve question, stated directly:** across all
1500 logged iterations — not a sample, the full `EventAccumulator` scan —
`Episode_Reward/lifting_object` and `Episode_Reward/object_goal_tracking`
are exactly `0.0` at every single point (`nonzero: 0/1500` for both). The
gate never fires once in this training run, at any iteration, in any of
the ten 150-iteration windows. This is not "fires late and grows" — there
is no first-firing iteration to report, because none occurred. It is the
"stays at or near zero for the entire run" outcome the controller's task
framing and the design spec's own success criteria both explicitly
anticipated as a real, valid possible result of this training budget: the
bilateral-antipodal-contact gate (`gripper_jaw1_contact` AND
`gripper_jaw2_contact` both registering meaningful opposing contact force,
not just cube height) is strictly harder to satisfy than Experiment 16's
height-only check, and within this run's 1500-iteration/4096-env budget
the policy never discovered a trajectory that satisfies it even once.

**Comparison against Experiment 16's own nonzero rates (per that
experiment's report):** Experiment 16's `lifting_object` was already
81.3% nonzero in its *first* 150 iterations (window 0) and saturated to
100.0% by iteration 150 onward, for the entire rest of its run. Experiment
17's `lifting_object` is 0.0% nonzero in every one of its ten 150-iteration
windows, including its last (iterations 1350-1499). This is a large,
unambiguous drop in discoverability, not a small or ambiguous one — going
from "saturated by iteration 150" to "never observed once in 1500
iterations" is a difference in kind, not just degree.

## Loss/value_function Sanity Check (Full Run)

**Sampled trajectory, every 150 iterations:**
```
iteration=   0, value=0.002371
iteration= 150, value=0.000065
iteration= 300, value=0.000145
iteration= 450, value=0.001006
iteration= 600, value=0.000442
iteration= 750, value=0.000243
iteration= 900, value=0.000169
iteration=1050, value=0.000102
iteration=1200, value=0.000107
iteration=1350, value=0.000114
iteration=1499, value=0.000066
```

**Full-run summary statistics:**
- Total data points: 1500
- First (iteration 0): 0.002371
- Last (iteration 1499): 0.0000659
- Max across all 1500 iterations: 0.0547
- Min: 0.0000509
- Nonzero: 1500/1500

**Finding:** `Loss/value_function` stays small and bounded across the
entire run — the max (0.0547) is roughly two orders of magnitude smaller
than Experiment 16's run-wide peak (4.588, at its curriculum-firing
iteration 417) and roughly comparable to Experiment 16's *initial* ~100-
iteration near-zero baseline (0.002-0.006). There is no sustained climb,
no late spike, and no divergence anywhere in the run. This is directly
consistent with the `lifting_object`/`object_goal_tracking` finding above:
Experiment 16's large, sustained value-loss climb was specifically
attributed (in that experiment's own report) to those two terms flipping
on/off abruptly per-step once the ungated lift condition started firing
frequently — here, since the grasp-gate never fires even once, that
specific abrupt-reward-composition-change source of value-function
instability never occurs, and the loss instead tracks a much smaller,
essentially flat baseline determined only by the always-active
`reaching_object`/`action_rate`/`joint_vel`/`cube_reached_goal` terms.

**Curriculum firing (checked for completeness, same threshold/mechanism
as Experiment 16 — `CurriculumCfg` is explicitly documented in this
experiment's own config as "Identical to Experiment 16's CurriculumCfg"):**
`Curriculum/action_rate_curr` and `Curriculum/joint_vel_curr` both
transition 416: -0.0001 -> -0.0334, 417: -0.0334 -> -0.1000, matching
Experiment 16's iteration-416/417 firing point exactly. The curriculum
mechanism itself is unaffected by the grasp-gate change and fires as
expected; it is not the source of, nor does it interact visibly with, the
lifting_object non-discovery finding above (no corresponding change in
`Loss/value_function`'s already-small trajectory is visible around
iteration 416-417).

## Scalar Trajectories (All 1500 Iterations)

### 1. Episode_Reward/reaching_object
**Summary:**
- Total data points: 1500
- First: iteration=0, value=0.000163
- Last: iteration=1499, value=0.479740
- Min: 0.000163
- Max: 0.694863
- Non-zero occurrences: 1500 / 1500

**Sample trajectory (every ~150 iterations):**
```
iteration=   0, value=0.000163
iteration= 150, value=0.637266
iteration= 300, value=0.666264
iteration= 450, value=0.670842
iteration= 600, value=0.553935
iteration= 750, value=0.519185
iteration= 900, value=0.496607
iteration=1050, value=0.474720
iteration=1200, value=0.482828
iteration=1350, value=0.473922
iteration=1499, value=0.479740
```
Note: this rises quickly then drifts down from its ~iteration-450 peak
(0.67) to settle around 0.47-0.48 for the back third of the run — plausibly
consistent with the policy shifting exploration effort away from pure
reaching once reaching alone stops yielding further progress toward an
undiscovered gate, though this scalar alone does not establish that; it is
reported factually.

### 2. Episode_Reward/lifting_object
**Summary:**
- Total data points: 1500
- First: iteration=0, value=0.000000
- Last: iteration=1499, value=0.000000
- Min: 0.000000
- Max: 0.000000
- Non-zero occurrences: 0 / 1500 (0.0%)

See "Critical Check" section above for the full sampled trajectory and
windowed table — flat zero at every single logged point across the
entire run.

### 3. Episode_Reward/object_goal_tracking
**Summary:**
- Total data points: 1500
- First: iteration=0, value=0.000000
- Last: iteration=1499, value=0.000000
- Min: 0.000000
- Max: 0.000000
- Non-zero occurrences: 0 / 1500 (0.0%)

See "Critical Check" section above — flat zero throughout, tracking
`lifting_object` exactly (both gated on the same grasp-verification
condition, per this experiment's design).

### 4. Episode_Reward/object_goal_tracking_fine_grained
**Summary:**
- Total data points: 1500
- First: iteration=0, value=0.000000
- Last: iteration=1499, value=0.000000
- Min: 0.000000
- Max: 0.000000
- Non-zero occurrences: 0 / 1500 (0.0%)

Flat zero throughout — also gated on the same grasp-verification
condition.

### 5. Episode_Termination/cube_reached_goal
**Summary:**
- Total data points: 1500
- First: iteration=0, value=0.002574
- Last: iteration=1499, value=0.002360
- Min: 0.001099
- Max: 0.011993
- Non-zero occurrences: 1500 / 1500

**Sample trajectory (every ~150 iterations):**
```
iteration=   0, value=0.002574
iteration= 150, value=0.006449
iteration= 300, value=0.006571
iteration= 450, value=0.005391
iteration= 600, value=0.002726
iteration= 750, value=0.002523
iteration= 900, value=0.003326
iteration=1050, value=0.002930
iteration=1200, value=0.003591
iteration=1350, value=0.003906
iteration=1499, value=0.002360
```
Note: since `cube_reached_goal` in this task's termination logic does not
require passing through the grasp-verification gate (it is this repo's own
bespoke "cube within tolerance of goal" check, historically noisy and
non-monotonic across every experiment in this session), its nonzero rate
being unaffected by the gate finding above is expected and not further
diagnostic of the gate itself.

### 6. Loss/value_function
See "Loss/value_function Sanity Check" section above for the full
analysis. Summary: first=0.002371, last=0.0000659, max=0.0547 (well
bounded), min=0.0000509, 1500/1500 nonzero.

## Key Comparison: Experiment 17 vs Experiment 16 and Experiment 12 (Final Values Only)

Per the established correction protocol (final-iteration snapshot vs.
final-iteration snapshot only — not cumulative, mid-run, or diagnostic-
window comparisons).

### Cube Reached Goal (vs. Experiment 12 — original task-space baseline)
- **Experiment 12 final value:** 0.010773
- **Experiment 17 final value:** 0.002360
- **Change:** -0.008413 (-78.1%)

### Cube Reached Goal (vs. Experiment 16 — ungated predecessor)
- **Experiment 16 final value:** 0.008962
- **Experiment 17 final value:** 0.002360
- **Change:** -0.006602 (-73.7%)

**Stated factually, not as a conclusion:** at the final-iteration
snapshot, `cube_reached_goal` is substantially lower for Experiment 17
than both Experiment 12 and Experiment 16. Per this project's established
correction protocol, this single noisy bespoke-termination scalar's
final-snapshot decline is reported as-is and is **not** treated as a
success/failure verdict on its own. It is, however, directionally
consistent with (though does not on its own prove) the much starker
finding above: `lifting_object`/`object_goal_tracking` never fire once in
this run, so if the policy is in fact still relying on some form of
elevated-hold/reach-and-freeze behavior near the goal rather than genuine
lift-carry-place, a lower `cube_reached_goal` rate than either prior
experiment would be an expected downstream consequence of that.

### `lifting_object` Nonzero Rate (vs. Experiment 16 — ungated predecessor)
- **Experiment 16:** first-150/last-150 nonzero rates 81.3%/100.0%
  (per that experiment's report).
- **Experiment 17:** first-150/last-150 nonzero rates 0.0%/0.0% (0/1500
  nonzero across the entire run).
- **Finding, stated factually:** Experiment 17's nonzero rate is much
  lower/sparser than Experiment 16's — not just lower, but zero throughout,
  versus Experiment 16's near-saturation from iteration 150 onward. Per
  the brief's own framing, this is an expected, informative possibility
  given the grasp-gate is strictly harder to satisfy than Experiment 16's
  height-only check — not necessarily a bug in the gate logic itself. No
  data in this report (e.g. an inconsistency between `lifting_object` and
  `object_goal_tracking`, which are both flat zero together, tracking each
  other exactly as designed) suggests a gate-logic bug; the two terms move
  in lockstep at 0.0% throughout, which is the same lockstep-consistency
  pattern Experiment 16's report used as evidence *against* a bug in that
  experiment's ungated version. The most direct reading of the scalar
  evidence alone is that the gate condition itself was not satisfied even
  once in 1500 iterations / ~150M environment steps, not that the gate is
  wired incorrectly — but confirming that reading conclusively (vs. e.g. a
  subtle sign/threshold bug in the gate's contact-force check that makes
  it unsatisfiable in principle) is exactly the kind of question Task 6's
  instrumented contact-force verification is scoped to resolve, not this
  scalar-only report.

## Assessment

**Checkpoint integrity and run health:** confirmed — 31 checkpoints at
the expected intervals, `model_1499.pt` present, event file mtime
consistent with the run's actual completion, zero tracebacks/exceptions
in the full-run log.

**`Loss/value_function` (sanity check):** stays small and bounded
throughout the full run (max 0.0547, roughly two orders of magnitude
below Experiment 16's peak), consistent with the absence of the abrupt
reward-composition discontinuities that drove Experiment 16's much larger
value-loss climb.

**Critical finding (the specific question this task was scoped to
answer):** `Episode_Reward/lifting_object` and `Episode_Reward/
object_goal_tracking` are exactly `0.0` at all 1500/1500 logged points
across the entire full run — the grasp-verification gate never fires even
once, at any iteration, in this training run. This is not a "fires late
and grows" outcome; there is no first-firing iteration. This is the
"stays at or near zero for the entire run" outcome the controller's task
framing, and the design spec's own success criteria, both explicitly
anticipated as a real, valid, and informative possible result of the
gate's added difficulty relative to Experiment 16 — not a failure to
downplay. Whether this reflects (a) the bilateral-antipodal-contact
condition being genuinely too hard for this policy/training budget to
discover from scratch, or (b) a bug in the gate's own contact-force
threshold/sign logic that makes it unsatisfiable regardless of the
policy's behavior, **cannot be distinguished from these scalars alone**
and is not answered by this report.

**Key scalar comparison (`cube_reached_goal`, final-iteration snapshot):**
Experiment 17 is substantially lower than both Experiment 12 (-78.1%) and
Experiment 16 (-73.7%) on this specific bespoke termination metric,
reported factually, consistent with (but not proof of) the gate-never-
fired finding above.

Per this project's established correction protocol (Experiment 12's
original report misread a scalar drop as failure and had to be corrected
by the controller, and Experiment 16's own report/ROADMAP entry initially
misread video evidence as confirming genuine grasp before instrumented
contact-sensor data showed otherwise) — **this report draws no final
success/failure conclusion from scalars alone, and explicitly does not
attempt a video-based conclusion either.** This experiment's entire
purpose is fixing a mechanism-verification failure from Experiment 16
(video looked like a grasp; instrumented contact force proved it wasn't),
so a final judgment on whether the gate is correctly implemented, and on
what the policy is actually doing given the gate never fires, both
require Task 6's instrumented contact-force verification against this
run's own trained checkpoint (`model_1499.pt` /
`logs/train/2026-07-07_15-41-37/`), done next in this same plan.
