# Experiment 16 Training Run Report: Proven-Recipe Replication (Full 1500 Iterations)

**Date:** 2026-07-07
**Log Directory:** `/home/saps/projects/rl/logs/train/2026-07-07_14-40-53`
**Training Status:** COMPLETED (1500 iterations)

## Overview

Experiment 16 replaces this repo's reward-design lineage (Experiments 1-15,
which always included a standalone grasp-quality bonus and an ungated
waypoint-proximity bonus) with a from-scratch replication of the reward/
action/curriculum structure used by two independently-published,
proven Isaac-ecosystem manipulation recipes (Isaac Lab's Franka Cube Lift
task and IsaacGymEnvs' FrankaCubeStack task) — see
`docs/superpowers/specs/2026-07-07-ar4-experiment16-proven-recipe-replication-design.md`
for the full design rationale. Implemented as
`Ar4PickPlaceProvenRecipeEnvCfg` (`tasks/ar4/pickplace_provenrecipe_env_cfg.py`),
launched via `scripts/train.py --provenrecipe`.

The 300-iteration diagnostic (Task 4, verification-only,
`logs/train/2026-07-07_14-34-12/`) passed both formal gate checks
(no divergence signal, no traceback), but flagged something this experiment's
reward shape had never produced before: `Loss/value_function` climbed from
near-zero to a **sustained** ~1.3-2.1 range by iterations ~277-299, not
recovered by the diagnostic window's end (first=0.00237, last=1.511,
max=2.101) — unlike every prior experiment's diagnostic, which showed a
brief isolated spike that fully recovered to near-zero. A plausible but
**unconfirmed** explanation was proposed: this reward design's
`lifting_object` is a binary weight-15.0 term and `object_goal_tracking` is
multiplicatively gated on lift, both of which can flip on/off abruptly
per-step (unlike every prior experiment's smoothly-shaped/running-max
terms), and the diagnostic's own data showed `lifting_object` going from
always-zero to often-firing over that exact window. This report covers the
full 1500-iteration run and specifically resolves whether the loss
stabilizes/declines or continues/accelerates in the ~1200 iterations beyond
the diagnostic's window, using the actual sampled full-run trajectory (not
just first/last/max).

This experiment's `CurriculumCfg` (bumping `action_rate`/`joint_vel`
weights from -1e-4 to -1e-1) activates at `num_steps=10000` (raw per-env
physics steps), which at `num_steps_per_env=24` corresponds to roughly PPO
iteration ~417 — a threshold the 300-iteration diagnostic never reached.
This full run is therefore the first time this repo has exercised the
curriculum manager's actual weight-change firing. This report also verifies
directly whether `Episode_Reward/action_rate` and `Episode_Reward/joint_vel`
show a step-change consistent with the curriculum firing near iteration
~400-450.

## Verification Results

### Checkpoint Integrity
- **Config `save_interval`:** 50
- **Expected checkpoints:** 1500 / 50 + 1 = 31
- **Actual checkpoints:** 31 — confirmed
- **Checkpoint iterations:** 0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1499 — confirmed
- **model_1499.pt exists:** YES (1,199,029 bytes, modified 2026-07-07 14:54) — confirmed
- **model_0.pt:** 1,198,359 bytes, modified 2026-07-07 14:41 — consistent with run start

### TensorBoard Event File
- **Event file:** `/home/saps/projects/rl/logs/train/2026-07-07_14-40-53/events.out.tfevents.1783449658.home.57940.0`
- **Size:** 2,092,514 bytes (~2.0M)
- **Modification time:** 2026-07-07 14:54
- **model_1499.pt modification time:** 2026-07-07 14:54
- **Status:** Event file mtime matches checkpoint completion time — confirmed. Run started 14:40:53, completed 14:54:18 (~13.5 minutes wall clock, consistent with prior 1500-iteration runs at `num_envs=4096`, and within the ~15-20 minute guide the launch brief gave).
- **Final console log (last logged iteration):** `Mean reward: 62.47`, `Mean episode length: 250.00`, `Episode_Termination/time_out: 0.9895`, `Episode_Termination/cube_reached_goal: 0.0105` (console rounding; full-precision TensorBoard value below), `Curriculum/action_rate_curr: -0.1000`, `Curriculum/joint_vel_curr: -0.1000` — confirms the curriculum had already fired by run's end, consistent with the detailed analysis below.

## Critical Check #1: `Loss/value_function` Across the FULL 1500-Iteration Run — Stabilize or Grow?

This is the specific question this task was scoped to resolve: does the
diagnostic's sustained ~1.3-2.1 climb (not yet recovered by iteration 299)
stabilize/decline over the remaining ~1200 iterations, or does it show
continued/accelerating growth (real, unresolved instability)?

**Coarse sampled trajectory (every ~150 iterations, full run):**
```
iteration=   0, value=0.002371
iteration= 150, value=0.396409
iteration= 300, value=1.521805
iteration= 450, value=1.520184
iteration= 600, value=1.065789
iteration= 750, value=0.921776
iteration= 900, value=0.614951
iteration=1050, value=0.406550
iteration=1200, value=0.352437
iteration=1350, value=0.320875
iteration=1499, value=0.297851
```

**Finer sampled trajectory (every 25 iterations, full run) — shows the actual peak and decline shape:**
```
iteration=   0, value=0.002371    iteration= 500, value=1.420327
iteration=  25, value=0.002229    iteration= 525, value=1.199035
iteration=  50, value=0.002404    iteration= 550, value=1.018216
iteration=  75, value=0.002144    iteration= 575, value=1.091573
iteration= 100, value=0.006363    iteration= 600, value=1.065789
iteration= 125, value=0.058899    iteration= 625, value=1.090362
iteration= 150, value=0.396409    iteration= 650, value=1.202547
iteration= 175, value=1.256187    iteration= 675, value=1.104189
iteration= 200, value=0.693953    iteration= 700, value=0.922481
iteration= 225, value=0.840865    iteration= 725, value=0.856562
iteration= 250, value=1.181109    iteration= 750, value=0.921776
iteration= 275, value=1.359904    iteration= 775, value=0.807333
iteration= 300, value=1.521805    iteration= 800, value=0.819056
iteration= 325, value=1.724432    iteration= 825, value=0.624206
iteration= 350, value=2.205614    iteration= 850, value=0.976122
iteration= 375, value=2.699070    iteration= 875, value=0.701912
iteration= 400, value=1.327223    iteration= 900, value=0.614951
iteration= 425, value=2.437778    iteration= 925, value=0.611134
iteration= 450, value=1.520184    iteration= 950, value=0.531887
iteration= 475, value=1.203682    iteration= 975, value=0.524666
                                   iteration=1000, value=0.554447
                                   iteration=1025, value=0.464495
                                   iteration=1050, value=0.406550
                                   iteration=1075, value=0.457839
                                   iteration=1100, value=0.431924
                                   iteration=1125, value=0.392191
                                   iteration=1150, value=0.435714
                                   iteration=1175, value=0.365769
                                   iteration=1200, value=0.352437
                                   iteration=1225, value=0.371056
                                   iteration=1250, value=0.449020
                                   iteration=1275, value=0.334959
                                   iteration=1300, value=0.356493
                                   iteration=1325, value=0.372288
                                   iteration=1350, value=0.320875
                                   iteration=1375, value=0.340930
                                   iteration=1400, value=0.249325
                                   iteration=1425, value=0.275123
                                   iteration=1450, value=0.355667
                                   iteration=1475, value=0.317929
                                   iteration=1499, value=0.297851
```

**Full-run summary statistics:**
- Total data points: 1500
- First (iteration 0): 0.002371
- Last (iteration 1499): 0.297851
- **Max across all 1500 iterations: 4.588496**, at iteration **417**
- Min: 0.0000571

**Answer to the stabilize-vs-grow question:** The loss does **not** show
continued or accelerating growth. It continues to climb somewhat past the
diagnostic's iteration-299 endpoint (1.52) up to a run-wide peak of
**4.588 at iteration 417** — but iteration 417 is not an arbitrary point:
it is the exact iteration at which the curriculum's `action_rate`/
`joint_vel` weight change fires (see Critical Check #2 below), which
introduces a large, sudden reward-scale discontinuity independent of the
lift-gating hypothesis. From that peak, the loss then declines
**steadily and essentially monotonically** for the remaining ~1080
iterations: 2.44 (425) -> 1.52 (450) -> 1.07 (600) -> 0.92 (750) -> 0.61
(900) -> 0.41 (1050) -> 0.35 (1200) -> 0.32 (1350) -> 0.30 (1499). By the
final iteration the loss is down 93.5% from its peak (4.588 -> 0.298).

**Caveat — it does not fully return to the initial near-zero baseline.**
The first 100 iterations sit at 0.002-0.006 (before `lifting_object` starts
firing meaningfully); the final ~300 iterations settle into a 0.25-0.45
band, roughly 50-150x larger than that initial baseline, and the decline
curve is still gently descending rather than flat at iteration 1499 (last
few finer-grained points: 0.355, 0.318, 0.298 — a shallow but real
downward slope, not a plateau). So the honest characterization is: **the
loss is declining, not growing, but has not fully returned to the very
first ~100 iterations' near-zero baseline by the end of training** — a
different, elevated equilibrium consistent with this reward shape's binary/
gated terms (which, per the windowed nonzero-rate data in the "Design Spec
Success Criteria" section below, are saturated at 100% nonzero for
essentially the entire back three-quarters of the run, i.e. produce
per-step reward variance every single logged iteration rather than only
occasionally). This resolves the diagnostic's flagged concern in the
"benign, different equilibrium" direction, not the "unresolved instability"
direction — but flags that the loss's absolute level, while much reduced
from the peak, remains structurally elevated versus this project's
typical near-zero baselines (e.g. Experiment 15's post-transient baseline
of 0.0002-0.002).

## Critical Check #2: Did the Curriculum Actually Fire, and When?

Predicted firing point: `num_steps=10000` / `num_steps_per_env=24` ≈
iteration 417.

**`Curriculum/action_rate_curr` and `Curriculum/joint_vel_curr`, iteration
380-460 (both curriculum terms are identical in value at every step, since
both use the same `num_steps=10000` threshold and target weight -0.1):**
```
iteration 380-415: -0.000100 (unchanged, original weight -1e-4)
iteration 416:      -0.033400  <- transition iteration
iteration 417-1499: -0.100000  <- new weight, holds for rest of run
```
The curriculum fires at **iteration 416-417**, matching the predicted
~417 to within 1 iteration.

**`Episode_Reward/action_rate` around the firing point (iteration
380-470):** flat at ≈-0.0093 to -0.0095 through iteration 415, then jumps
sharply: -0.056 (416) -> -0.764 (417) -> -1.68 (418) -> -2.58 (419) -> ...
peaking at **-7.97 at iteration 427**, then declining back down as the
policy adapts to the new, 1000x-larger penalty weight: -6.65 (433) ->
-5.04 (443) -> -4.53 (450) -> -3.66 (470) -> ... continuing to decline
across the rest of the run to a final value of -0.577 (iteration 1499).

**`Episode_Reward/joint_vel` around the firing point:** flat at
≈-0.00153 to -0.00160 through iteration 415, then jumps: -0.0064 (416) ->
-0.087 (417) -> -0.19 (418) -> ... peaking at **-1.554 at iteration 428**,
then settling into a ~-1.4 to -1.5 range for the next several hundred
iterations before gradually declining to a final value of -0.605
(iteration 1499).

**Finding: YES, the curriculum fired, with a clear, unambiguous
step-change visible directly in the raw per-iteration data at iteration
416-417** — both the `Curriculum/*_curr` scalars (the weight value itself)
and the resulting `Episode_Reward/action_rate` / `Episode_Reward/joint_vel`
magnitudes (the realized penalty, which necessarily also depends on the
policy's actual action-rate/joint-velocity behavior, not just the weight)
show the expected discontinuity at exactly the predicted iteration. This
also directly explains the `Loss/value_function` run-wide peak occurring
at iteration 417 (Critical Check #1 above): the curriculum's weight change
is a sudden ~1000x change in per-step reward composition for two terms
that fire on every single iteration (100% nonzero), which is exactly the
kind of abrupt value-target discontinuity that would spike a value
function's TD-error loss, independent of (and likely compounding with) the
lift-gating hypothesis from the diagnostic.

## Scalar Trajectories (All 1500 Iterations)

### 1. Episode_Reward/reaching_object
**Summary:**
- Total data points: 1500
- First: iteration=0, value=0.000163
- Last: iteration=1499, value=0.493418
- Min: 0.000163
- Max: 0.550015
- Non-zero occurrences: 1500 / 1500

**Sample trajectory (every ~150 iterations):**
```
iteration=   0, value=0.000163
iteration= 150, value=0.421619
iteration= 300, value=0.386255
iteration= 450, value=0.437089
iteration= 600, value=0.434349
iteration= 750, value=0.467544
iteration= 900, value=0.470862
iteration=1050, value=0.472194
iteration=1200, value=0.479762
iteration=1350, value=0.482953
iteration=1499, value=0.493418
```

### 2. Episode_Reward/lifting_object
**Summary:**
- Total data points: 1500
- First: iteration=0, value=0.000000
- Last: iteration=1499, value=12.347164
- Min: 0.000000
- Max: 12.449489
- Non-zero occurrences: 1472 / 1500 (98.1%)

**Sample trajectory (every ~150 iterations):**
```
iteration=   0, value=0.000000
iteration= 150, value=0.817479
iteration= 300, value=3.555429
iteration= 450, value=7.503295
iteration= 600, value=9.396194
iteration= 750, value=10.659444
iteration= 900, value=11.323547
iteration=1050, value=11.737758
iteration=1200, value=11.909633
iteration=1350, value=11.869632
iteration=1499, value=12.347164
```

### 3. Episode_Reward/object_goal_tracking
**Summary:**
- Total data points: 1500
- First: iteration=0, value=0.000000
- Last: iteration=1499, value=0.934243
- Min: 0.000000
- Max: 0.963100
- Non-zero occurrences: 1472 / 1500 (98.1%)

**Sample trajectory (every ~150 iterations):**
```
iteration=   0, value=0.000000
iteration= 150, value=0.142812
iteration= 300, value=0.195970
iteration= 450, value=0.416935
iteration= 600, value=0.484596
iteration= 750, value=0.455459
iteration= 900, value=0.571888
iteration=1050, value=0.626206
iteration=1200, value=0.724164
iteration=1350, value=0.802904
iteration=1499, value=0.934243
```

### 4. Episode_Reward/object_goal_tracking_fine_grained
**Summary:**
- Total data points: 1500
- First: iteration=0, value=0.000000
- Last: iteration=1499, value=0.001617
- Min: 0.000000
- Max: 0.002259
- Non-zero occurrences: 1470 / 1500 (98.0%)

**Sample trajectory (every ~150 iterations):**
```
iteration=   0, value=0.000000
iteration= 150, value=0.000411
iteration= 300, value=0.000416
iteration= 450, value=0.000976
iteration= 600, value=0.000920
iteration= 750, value=0.000686
iteration= 900, value=0.001086
iteration=1050, value=0.000965
iteration=1200, value=0.001200
iteration=1350, value=0.001665
iteration=1499, value=0.001617
```

### 5. Episode_Termination/cube_reached_goal
**Summary:**
- Total data points: 1500
- First: iteration=0, value=0.002574
- Last: iteration=1499, value=0.008962
- Min: 0.001862
- Max: 0.017181
- Non-zero occurrences: 1500 / 1500

**Sample trajectory (every ~150 iterations):**
```
iteration=   0, value=0.002574
iteration= 150, value=0.007640
iteration= 300, value=0.011281
iteration= 450, value=0.011180
iteration= 600, value=0.009318
iteration= 750, value=0.007568
iteration= 900, value=0.011261
iteration=1050, value=0.011403
iteration=1200, value=0.011078
iteration=1350, value=0.012675
iteration=1499, value=0.008962
```
Note: the run-wide max (0.017181) occurs mid-run, not at the final
iteration; the final-iteration value (0.008962) is noisy/lower than
several mid-run samples (e.g. iteration 1350's 0.012675). This is
consistent with this metric's known high run-to-run and iteration-to-
iteration variance in this project (see Key Comparison section for the
established final-snapshot-only comparison protocol).

### 6. Loss/value_function
See Critical Check #1 above for the full analysis. Summary: first=0.002371,
last=0.297851, max=4.588496 (iteration 417, coincident with curriculum
firing), min=0.0000571, 1500/1500 nonzero.

## Design Spec Success Criteria: `lifting_object` and `object_goal_tracking` Nonzero-Rate Analysis

Per the design spec's secondary/scalar success criterion: does
`lifting_object`'s nonzero rate (this experiment's direct per-step lift
indicator) show real growth over training, and does
`object_goal_tracking`'s nonzero rate track it (confirming the lift-gate is
actually unlocking, not stuck at zero because lift never happens)?

**`Episode_Reward/lifting_object` — first 150 vs. last 150 iterations:**
- First 150 (iterations 0-149): nonzero = 122/150 (81.3%)
- Last 150 (iterations 1350-1499): nonzero = 150/150 (100.0%)

**`Episode_Reward/object_goal_tracking` — first 150 vs. last 150 iterations:**
- First 150 (iterations 0-149): nonzero = 122/150 (81.3%)
- Last 150 (iterations 1350-1499): nonzero = 150/150 (100.0%)

**Windowed detail (10 windows of 150 iterations each), both terms:**

| Window | Iterations | lifting_object nonzero | lifting_object avg value | object_goal_tracking nonzero | object_goal_tracking avg value |
|---|---|---|---|---|---|
| 0 | 0-149 | 122/150 (81.3%) | 0.0542 | 122/150 (81.3%) | 0.0107 |
| 1 | 150-299 | 150/150 (100.0%) | 1.7656 | 150/150 (100.0%) | 0.1342 |
| 2 | 300-449 | 150/150 (100.0%) | 5.5845 | 150/150 (100.0%) | 0.3070 |
| 3 | 450-599 | 150/150 (100.0%) | 9.1126 | 150/150 (100.0%) | 0.4650 |
| 4 | 600-749 | 150/150 (100.0%) | 9.9217 | 150/150 (100.0%) | 0.4644 |
| 5 | 750-899 | 150/150 (100.0%) | 10.7773 | 150/150 (100.0%) | 0.5031 |
| 6 | 900-1049 | 150/150 (100.0%) | 11.4251 | 150/150 (100.0%) | 0.5897 |
| 7 | 1050-1199 | 150/150 (100.0%) | 11.7779 | 150/150 (100.0%) | 0.6776 |
| 8 | 1200-1349 | 150/150 (100.0%) | 11.9595 | 150/150 (100.0%) | 0.7463 |
| 9 | 1350-1499 | 150/150 (100.0%) | 12.1071 | 150/150 (100.0%) | 0.8256 |

**Findings, stated factually:**
- (a) `lifting_object`'s nonzero rate rises from 81.3% in the first
  150 iterations to a saturated 100.0% by iteration 150 onward and stays
  saturated for the remaining ~1350 iterations. Its per-window average
  value climbs essentially monotonically across the entire run, from 0.05
  (window 0) to 12.1 (window 9) — a ~220x increase — never regressing
  window-over-window.
- (b) `object_goal_tracking`'s nonzero rate tracks `lifting_object`'s
  exactly, window-for-window (81.3% -> 100.0% at the identical transition
  point, iteration ~150). Its per-window average value also climbs
  essentially monotonically, from 0.011 (window 0) to 0.826 (window 9) — a
  ~75x increase. **The gate does not appear to have a bug**: it is not the
  case that `lifting_object` grows while `object_goal_tracking` stays at
  zero; the two nonzero rates move together in lockstep across every one of
  the ten windows, exactly the pattern the design spec's success criterion
  was checking for.
- This nonzero-rate/magnitude growth is consistent with — though does not
  on its own prove — the policy lifting the cube on a growing and now
  near-universal fraction of per-step observations. Per the brief and this
  project's established correction protocol, this scalar evidence alone
  does not constitute a final success/failure judgment; that requires
  video inspection (see Assessment below).

## Key Comparison: Experiment 16 vs Experiment 15 and Experiment 12 (Final Values Only)

Per the established correction protocol (final-iteration snapshot vs.
final-iteration snapshot only — not cumulative, mid-run, or diagnostic-
window comparisons).

### Cube Reached Goal (vs. Experiment 12 — original task-space baseline)
- **Experiment 12 final value:** 0.010773
- **Experiment 16 final value:** 0.008962
- **Change:** -0.001811 (-16.8%)

### Cube Reached Goal (vs. Experiment 15 — best of session so far)
- **Experiment 15 final value:** 0.017202
- **Experiment 16 final value:** 0.008962
- **Change:** -0.008240 (-47.9%)

**Stated factually, not as a conclusion:** at the final-iteration snapshot,
`cube_reached_goal` is lower for Experiment 16 than both Experiment 12 and
Experiment 15. Note this metric's known noise (Experiment 16's own
run-wide max of 0.017181 occurred mid-run at iteration ~1350's neighborhood,
almost matching Experiment 15's final value, while the specific final-
iteration sample landed lower) — per this project's established correction
protocol (Experiment 12's original report misread a scalar drop as failure
and had to be corrected by the controller), this single scalar's final-
snapshot decline is reported as-is and is **not** treated as a success/
failure verdict on its own, especially since `cube_reached_goal` is this
repo's own bespoke termination condition, not a metric either proven
reference recipe was tuned against — while `lifting_object` and
`object_goal_tracking` (this experiment's own direct, literal success
signals per its design spec) both show strong, consistent, monotonic
growth across the entire run (see previous section).

## Assessment

**Value-function loss (Critical Check #1):** the diagnostic's flagged
sustained climb continued somewhat further after the diagnostic's
iteration-299 window closed, reaching a run-wide peak of 4.588 at
iteration 417 — but that peak coincides exactly with the curriculum
firing (Critical Check #2), not with continued/unbounded growth. From
that peak, the loss declines steadily and essentially monotonically for
the remaining ~1080 iterations, ending at 0.298 (93.5% below the peak).
It does not fully return to the very first ~100 iterations' near-zero
baseline (0.002-0.006) by the end of training, settling instead into an
elevated but bounded and still-declining ~0.25-0.45 range for the last
several hundred iterations. This is evidence against the "unresolved,
continuing instability" reading of the diagnostic's flag and evidence for
a "different, elevated-but-bounded equilibrium, further perturbed by a
real, expected, one-time curriculum-firing discontinuity" reading.

**Curriculum firing (Critical Check #2):** confirmed directly in the raw
per-iteration data — `Curriculum/action_rate_curr`/`joint_vel_curr` step
from -0.0001 to -0.1 at iteration 416-417 (matching the predicted ~417 to
within 1 iteration), and `Episode_Reward/action_rate`/`joint_vel` show a
clear, large-magnitude step-change in the same window (action_rate:
-0.009 -> peak -7.97 at iteration 427; joint_vel: -0.0016 -> peak -1.554
at iteration 428), both subsequently declining as the policy adapts to
the new, larger penalty weights.

**Design spec success criteria (lift-gate nonzero rates):** both
`lifting_object` and `object_goal_tracking` show real, sustained,
monotonic growth in both nonzero rate (81.3% -> 100.0%, saturating by
iteration ~150) and magnitude across the entire run, and the two track
each other in lockstep window-for-window — no sign of a gate-logic bug.

**Key scalar comparison (`cube_reached_goal`, final-iteration snapshot):**
Experiment 16 is lower than both Experiment 12 (-16.8%) and Experiment 15
(-47.9%) on this specific bespoke termination metric, reported factually.

Per this project's established correction protocol (Experiment 12's
original report misread a scalar drop as failure and had to be corrected
by the controller) — **this report does not draw a final success/failure
conclusion from the scalars alone.** The value-function loss and
curriculum-firing questions this task was specifically scoped to answer
are both resolved with direct evidence above (declining/bounded, and
fired-as-expected, respectively). Whether the policy is actually achieving
genuine sustained lift and carry-toward-goal in video — the design spec's
stated primary success criterion — requires video inspection, done
separately by the controller outside this plan (no Task 6 in this plan,
matching this session's established pattern).
