# Experiment 18 Training Run Report: Pre-Grasp-Readiness Shaping (Full 1500 Iterations)

**Date:** 2026-07-07
**Log Directory:** `/home/saps/projects/rl/logs/train/2026-07-07_16-38-01`
**Training Status:** COMPLETED (1500 iterations)

## Overview

Experiment 18 introduces `pregrasp_readiness_bonus` (implemented as a new
shaped reward term, not a gate) with the hypothesis that explicitly rewarding
the policy for maintaining consistent bilateral-jaw contact *during reach*
(a readiness prerequisite for lift) would provide a discoverable intermediate
objective that reduces the exploration burden on lifting itself. Where
Experiment 17's grasp-gate (a hard binary condition, either satisfied or not)
never fired even once across 1500 iterations, this experiment shapes the
reward trajectory during reaching to incentivize configurations in which
grasp is already "ready" before a lift is even attempted — betting that an
explicit gradient toward that readiness state would guide the policy closer
to lift-discovery than Experiment 16's ungated baseline or Experiment 17's
never-firing gate.

Implemented as `Ar4PickPlacePregraspEnvCfg` (`tasks/ar4/pickplace_pregrasp_env_cfg.py`),
launched via `scripts/train.py --pregrasp`. The 300-iteration diagnostic
(Task 4, verification-only, `logs/train/2026-07-07_14-10-41/`) passed formal
gate checks (`Loss/value_function` bounded and net-declining across all 300
points, no tracebacks), and critically showed that the new `Episode_Reward/
pregrasp_readiness` term produced real, nonzero signal across all 300 points
(growing from 0.000166 to 1.199), confirming the term itself is discoverable
and utilized by the policy. This report resolves the final question: does that
discovered readiness signal translate, over 1500 full iterations, into a
*lifting* discovery (the actual success criterion), or does lifting remain
at 0/1500 despite the improved readiness shaping?

## Verification Results

### Checkpoint Integrity
- **Config `save_interval`:** 50
- **Expected checkpoints:** 1500 / 50 + 1 = 31
- **Actual checkpoints:** 31 — confirmed
- **Checkpoint iterations:** 0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1499 — confirmed
- **model_1499.pt exists:** YES (1,199,029 bytes, modified 2026-07-07 16:52:21) — confirmed
- **model_0.pt:** 1,198,359 bytes, modified 2026-07-07 16:38:01 — consistent with run start

### TensorBoard Event File
- **Event file:** `/home/saps/projects/rl/logs/train/2026-07-07_16-38-01/events.out.tfevents.1783456686.home.71291.0`
- **Size:** 2,200,392 bytes (~2.1M)
- **Modification time:** 2026-07-07 16:52:21
- **model_1499.pt modification time:** 2026-07-07 16:52:21
- **Status:** Event file mtime matches checkpoint completion time — confirmed. Run started 16:38:01, completed 16:52:21 (~14.3 minutes wall clock, consistent with prior 1500-iteration runs at `num_envs=4096`).
- **Console log:** `/tmp/exp18_train_stdout.log`. Last printed iteration block (1493/1500): `Mean reward: 8.69`, `Mean episode length: 250.00`, `Episode_Reward/reaching_object: 0.6400`, `Episode_Reward/pregrasp_readiness: 1.2689`, `Episode_Reward/lifting_object: 0.0000`, `Episode_Reward/object_goal_tracking: 0.0000`, `Episode_Termination/cube_reached_goal: 0.0035`, `Curriculum/action_rate_curr: -0.1000`, `Curriculum/joint_vel_curr: -0.1000`.
- **No tracebacks/exceptions:** a case-insensitive sweep for `traceback|error|exception` found zero matches in the full-run log.

## Critical Question 1: Does `Episode_Reward/pregrasp_readiness` Show Real, Growing Nonzero Occurrence Across the Full Run?

This is a prerequisite question: if the new shaped reward term provides no usable gradient at all (stays zero or is immediately saturated), then its failure is interpretable — the mechanism offers no signal. If it does produce real signal, then any failure to discover lifting must be explained differently (the readiness readiness helps, but isn't enough; lift itself remains undiscovered despite readiness being shaped).

**Full-run scalar extraction (exact command output, `EventAccumulator` scanning all 1500 logged points):**
```
=== Episode_Reward/pregrasp_readiness ===
  points: 1500 first: 0.00016637894441373646 last: 1.268934726715088 max: 1.2707538604736328 min: 0.00016637894441373646 nonzero: 1500 / 1500
```

**Sampled trajectory, `Episode_Reward/pregrasp_readiness`, every 150 iterations across the full run:**
```
iteration=   0, value=0.000166
iteration= 150, value=1.146648
iteration= 300, value=1.191666
iteration= 450, value=1.224220
iteration= 600, value=1.224539
iteration= 750, value=1.242727
iteration= 900, value=1.255840
iteration=1050, value=1.255674
iteration=1200, value=1.249108
iteration=1350, value=1.257005
iteration=1499, value=1.268935
```

**Windowed nonzero-rate table (10 windows of 150 iterations each):**

| Window | Iterations | pregrasp_readiness nonzero |
|---|---|---|
| 0 | 0-149 | 150/150 (100.0%) |
| 1 | 150-299 | 150/150 (100.0%) |
| 2 | 300-449 | 150/150 (100.0%) |
| 3 | 450-599 | 150/150 (100.0%) |
| 4 | 600-749 | 150/150 (100.0%) |
| 5 | 750-899 | 150/150 (100.0%) |
| 6 | 900-1049 | 150/150 (100.0%) |
| 7 | 1050-1199 | 150/150 (100.0%) |
| 8 | 1200-1349 | 150/150 (100.0%) |
| 9 | 1350-1499 | 150/150 (100.0%) |

**Answer to Question 1, stated directly:** Yes. `Episode_Reward/pregrasp_readiness`
shows real, growing nonzero signal across the full 1500-iteration run — not just
nonzero, but **consistently 100.0% nonzero in every 150-iteration window**,
growing from initial 0.000166 to final 1.268935. The term is discoverable,
utilized by the policy at every iteration, and shows monotonic-ish upward
trajectory (minor dips late, but settling in the 1.24-1.27 range for the entire
second half of the run). This confirms the shaping mechanism itself is working
— the policy has adopted reaching configurations that register readiness as
defined. The prerequisite for answering Question 2 is met: the readiness signal
exists as a usable gradient throughout training.

## Critical Question 2: Does `Episode_Reward/lifting_object`'s Nonzero Count Move Off Exactly `0/1500`?

This is the single specific, falsifiable question this experiment exists to answer:
given that the new pregrasp_readiness shaping does provide a discovered,
growing intermediate objective, does that help lift discovery, or does lifting
remain undiscovered?

**Full-run scalar extraction (exact command output, `EventAccumulator` scanning all 1500 logged points):**
```
=== Episode_Reward/lifting_object ===
  points: 1500 first: 0.0 last: 0.0 max: 0.0 min: 0.0 nonzero: 0 / 1500
```

**Sampled trajectory, `Episode_Reward/lifting_object`, every 150 iterations across the full run (identical for `object_goal_tracking` and `object_goal_tracking_fine_grained` — all three are 0.000000 at every sampled point):**
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

**Windowed nonzero-rate table (10 windows of 150 iterations each), for direct comparison against Experiment 16 and Experiment 17:**

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

**Answer to Question 2, stated directly:** No. `Episode_Reward/lifting_object`
remains at exactly `0 / 1500` nonzero occurrences across the entire full run.
The lifting term never fires even once at any iteration, in any of the ten
150-iteration windows. This is identical to Experiment 17's outcome.

**The specific finding:** despite `pregrasp_readiness` being a strong, consistently
discovered signal (100% nonzero throughout, growing 0.000166 → 1.268935),
`lifting_object` does not advance beyond the 0/1500 baseline established by
Experiment 17's grasp-gated variant. The pregrasp-readiness shaping does not
enable lift discovery within this training budget.

## Loss/value_function Sanity Check (Full Run)

**Sampled trajectory, every 150 iterations:**
```
iteration=   0, value=0.002434
iteration= 150, value=0.001297
iteration= 300, value=0.000982
iteration= 450, value=0.000801
iteration= 600, value=0.000414
iteration= 750, value=0.000506
iteration= 900, value=0.000316
iteration=1050, value=0.000593
iteration=1200, value=0.000622
iteration=1350, value=0.000421
iteration=1499, value=0.000351
```

**Full-run summary statistics:**
- Total data points: 1500
- First (iteration 0): 0.002434
- Last (iteration 1499): 0.000351
- Max across all 1500 iterations: 0.148235
- Min: 0.000146
- Nonzero: 1500/1500

**Finding:** `Loss/value_function` stays small and bounded across the entire
run — the max (0.148235) is smaller than Experiment 16's peak (4.588) and
only slightly larger than Experiment 17's max (0.0547). The loss declines from
the initial 0.002434 to final 0.000351, with only minor fluctuations and no
sustained climb or divergence anywhere in the run. This is consistent with
both the `pregrasp_readiness` term being continuously active (unlike the
never-firing gate in Experiment 17) and the `lifting_object` term remaining
permanently zero (unlike Experiment 16's frequent on/off transitions). The
loss trajectory suggests training stability throughout, with no instability
sources visible.

## Scalar Trajectories (All 1500 Iterations)

### 1. Episode_Reward/reaching_object
**Summary:**
- Total data points: 1500
- First: iteration=0, value=0.000163
- Last: iteration=1499, value=0.640083
- Min: 0.000163
- Max: 0.640288
- Non-zero occurrences: 1500 / 1500

**Sample trajectory (every ~150 iterations):**
```
iteration=   0, value=0.000163
iteration= 150, value=0.590196
iteration= 300, value=0.612392
iteration= 450, value=0.618668
iteration= 600, value=0.619224
iteration= 750, value=0.626302
iteration= 900, value=0.635187
iteration=1050, value=0.635171
iteration=1200, value=0.628720
iteration=1350, value=0.633514
iteration=1499, value=0.640083
```
Note: reaches quickly to ~0.59 by iteration 150, then continues gradual climb
to final 0.640, suggesting sustained improvement in reaching throughout. Similar
to Experiment 17 in shape, though Experiment 18's final value is slightly
higher (0.640 vs 0.480).

### 2. Episode_Reward/pregrasp_readiness (NEW TERM)
**Summary:**
- Total data points: 1500
- First: iteration=0, value=0.000166
- Last: iteration=1499, value=1.268935
- Min: 0.000166
- Max: 1.270754
- Non-zero occurrences: 1500 / 1500

**Sample trajectory (every ~150 iterations):**
```
iteration=   0, value=0.000166
iteration= 150, value=1.146648
iteration= 300, value=1.191666
iteration= 450, value=1.224220
iteration= 600, value=1.224539
iteration= 750, value=1.242727
iteration= 900, value=1.255840
iteration=1050, value=1.255674
iteration=1200, value=1.249108
iteration=1350, value=1.257005
iteration=1499, value=1.268935
```
See "Critical Question 1" section above for full analysis. This term shows
strong, monotonic-ish growth from near-zero to 1.27 over first 900 iterations,
then stabilizes in the 1.24-1.27 range for the remainder. The term is fully
active and provides consistent signal throughout the run.

### 3. Episode_Reward/lifting_object
**Summary:**
- Total data points: 1500
- First: iteration=0, value=0.000000
- Last: iteration=1499, value=0.000000
- Min: 0.000000
- Max: 0.000000
- Non-zero occurrences: 0 / 1500 (0.0%)

See "Critical Question 2" section above for the full sampled trajectory and
windowed table — flat zero at every single logged point across the entire run.

### 4. Episode_Reward/object_goal_tracking
**Summary:**
- Total data points: 1500
- First: iteration=0, value=0.000000
- Last: iteration=1499, value=0.000000
- Min: 0.000000
- Max: 0.000000
- Non-zero occurrences: 0 / 1500 (0.0%)

Flat zero throughout, tracking `lifting_object` exactly (both gated on the
same lift condition, per this experiment's design). Not directly dependent
on pregrasp_readiness; readiness must be discovered as a side-effect of
pursuing other rewards, not as a prerequisite for goal-tracking to activate.

### 5. Episode_Reward/object_goal_tracking_fine_grained
**Summary:**
- Total data points: 1500
- First: iteration=0, value=0.000000
- Last: iteration=1499, value=0.000000
- Min: 0.000000
- Max: 0.000000
- Non-zero occurrences: 0 / 1500 (0.0%)

Flat zero throughout — also gated on the lift condition.

### 6. Episode_Termination/cube_reached_goal
**Summary:**
- Total data points: 1500
- First: iteration=0, value=0.002574
- Last: iteration=1499, value=0.003499
- Min: 0.001953
- Max: 0.012054
- Non-zero occurrences: 1500 / 1500

**Sample trajectory (every ~150 iterations):**
```
iteration=   0, value=0.002574
iteration= 150, value=0.006317
iteration= 300, value=0.007904
iteration= 450, value=0.006521
iteration= 600, value=0.004710
iteration= 750, value=0.004791
iteration= 900, value=0.005117
iteration=1050, value=0.003947
iteration=1200, value=0.007080
iteration=1350, value=0.005493
iteration=1499, value=0.003499
```
Note: rises to peak around iterations 300-450, then drifts down to ~0.0035
in the final iterations. Similar noisy behavior to prior experiments.

### 7. Loss/value_function
See "Loss/value_function Sanity Check" section above for the full analysis.
Summary: first=0.002434, last=0.000351, max=0.148235 (bounded, no divergence),
min=0.000146, 1500/1500 nonzero.

## Key Comparison: Experiment 18 vs Experiment 17 and Experiment 12 (Final Values Only)

Per the established correction protocol (final-iteration snapshot vs.
final-iteration snapshot only — not cumulative or mid-run comparisons).

### Cube Reached Goal (vs. Experiment 12 — original task-space baseline)
- **Experiment 12 final value:** 0.010773
- **Experiment 18 final value:** 0.003499
- **Change:** -0.007274 (-67.5%)

### Cube Reached Goal (vs. Experiment 17 — grasp-gated predecessor)
- **Experiment 17 final value:** 0.002360
- **Experiment 18 final value:** 0.003499
- **Change:** +0.001139 (+48.3%)

**Stated factually:** at the final-iteration snapshot, `cube_reached_goal`
is lower for Experiment 18 than Experiment 12 (-67.5%), but higher than
Experiment 17 (+48.3%). Per this project's established correction protocol,
this single noisy bespoke-termination scalar's final-snapshot change is
reported as-is and is **not** treated as a success/failure verdict on its own.
The consistent finding that `lifting_object` stays at 0/1500 in both
Experiment 17 and Experiment 18 (despite the readiness shaping in this run)
is the primary evidence for interpreting this scalar's trajectory.

### `lifting_object` Nonzero Rate (vs. Experiment 17 — grasp-gated predecessor)
- **Experiment 17:** 0/1500 nonzero (grasp gate never fires once)
- **Experiment 18:** 0/1500 nonzero (lifting still never fires once)
- **Finding, stated factually:** Experiment 18's nonzero rate is identical to
  Experiment 17's — both remain at zero. The new pregrasp-readiness shaping,
  despite providing strong and consistent signal (100% nonzero, growing
  0.000166 → 1.268935), does not enable lift discovery beyond the 0/1500
  baseline. Readiness and lifting remain decoupled: the policy can learn
  readiness as a side-effect of reaching, but that readiness does not
  bootstrap lift discovery.

## Assessment

**Checkpoint integrity and run health:** confirmed — 31 checkpoints at
the expected intervals, `model_1499.pt` present, event file mtime
consistent with the run's actual completion, zero tracebacks/exceptions
in the full-run log.

**`Loss/value_function` (sanity check):** stays small and bounded
throughout the full run (max 0.148235), with a gentle downward trend from
0.002434 to 0.000351, indicating stable training. No divergence or
instability.

**Critical findings (the two falsifiable questions this task was scoped to answer):**

1. **Does `Episode_Reward/pregrasp_readiness` show real, growing nonzero occurrence?**
   YES — the new term is consistently 100% nonzero across all 1500 iterations,
   growing from 0.000166 to 1.268935. This confirms the shaping mechanism
   itself is discoverable and utilized by the policy. The prerequisite for
   the next question is met: the readiness signal exists as a usable gradient.

2. **Does `Episode_Reward/lifting_object` move off 0/1500?**
   NO — lifting remains at exactly 0/1500 nonzero occurrences across the
   entire run, identical to Experiment 17's outcome. Despite the strong
   pregrasp-readiness signal, lifting is never discovered. Readiness and
   lifting remain decoupled within this training budget.

**Interpretation:** The pregrasp-readiness shaping confirms that intermediate
behavioral guidance (toward a state associated with grasp) is learnable and
provides signal. However, it does not bridge the gap to lift discovery. This
suggests that the core challenge is not reaching a readiness *state* (that
is now learnable), but either:
- The lift action itself (cube height increase) is not being triggered even
  when the policy is in a "ready" configuration, or
- Some other aspect of the lift discovery (e.g., gripper force modulation,
  vertical motion persistence, or object stability during lift) remains
  undiscovered and unrewarded.

This finding informs the next research direction: adding an explicit lift
attempt bonus (independent of success, similar to how reaching is rewarded
even without a successful grasp) may be necessary to close the discovery gap,
since intermediate readiness shaping alone did not suffice.

**Video inspection status:** Per the brief's own guidance, since
`lifting_object` remains exactly 0/1500 with no nonzero occurrences to
investigate, **no video inspection is warranted** at this stage. The scalar
evidence is unambiguous: lifting does not occur, confirming there is nothing
new for video to show beyond what Experiment 17's scalar-only reporting
already established (policy reaches but does not lift, even with readiness
guidance). The report's finding stands on the scalar evidence alone.
