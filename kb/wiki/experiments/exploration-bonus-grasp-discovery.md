# Exploration-bonus grasp discovery experiment (2026-07-19 -> 2026-07-20, H1 COMPLETE — SPLIT)

**Closing verdict: SPLIT, not falsification.** A GRM (Generalized
Reward Matching) `D=1` potential-based exploration bonus for
gripper-closure attempts near the object fires demonstrably in ≥1 of 3
seeds (seed 123: 7/8 envs at `frac_steps_raw_action_negative_near_object
= 1.0`), but real sustained lift never occurs in any seed
(`envs_with_sustained_lift = 0/8` all 3 seeds, 0/24 total). Per the
design spec's own pre-registered rule ("falsified only if BOTH bars fail
across all 3 seeds"), this is **not** a falsification of H1's core
exploration-discoverability claim — it is the spec's own explicitly
anticipated third outcome, and the first result in this project's history
to land there rather than in a plain pass/fail.

**Goal:** test whether a theoretically-principled (policy-invariant,
Ng-Harada-Russell/GRM-style) potential-based exploration bonus for
attempting gripper closure specifically near the object can unlock the
grasp-discoverability null that from-scratch PPO shows for d8 at the
48mm-parity anchor
([[unified-multi-die-specialist-distillation]]'s own established 0/24
d8 baseline, [[d8-d10-demo-warmstart]]'s independent from-scratch
reconfirmation). Spec:
`docs/superpowers/specs/2026-07-19-exploration-bonus-grasp-discovery-design.md`.
Research: `docs/superpowers/specs/research/2026-07-19-exploration-reward-expansion-literature.md`.
Plan: `docs/superpowers/plans/2026-07-19-exploration-bonus-grasp-discovery-implementation.md`.

## Mechanism (Tasks 1-3, committed `10a9588`/`59b0246`/`e9bc14b`)

Two new `RewardsCfg` terms on a new, isolated env-cfg subclass
(`FrankaDieLiftJointD8BigExplorationBonusEnvCfg`, extending
`FrankaDieLiftJointD8BigEnvCfg` untouched):

- **Term 1 (`gripper_closure_attempt_bonus`, `F_t`):** a raw,
  action-dependent bonus — `w_attempt * tanh(k * relu(-raw_gripper_action))
  * (1 - tanh(distance / std_gate))` — rewarding a negative
  (closing-commanded) raw gripper action, gated by proximity to the
  object (`std_gate = 0.05`, matching the design's own near-object
  threshold).
- **Term 2 (`gripper_closure_attempt_bonus_correction`):** the GRM `D=1`
  matching/correction term, implemented as `Correction_t = F'_t - F_t`
  (not `F'_t` verbatim — the implementation plan's own worked derivation,
  see its "Design notes" section, resolves a real double-counting
  ambiguity in the spec's literal formula) so that `Term1 + Term2`
  telescopes to the theorem's own policy-invariance guarantee. A
  `ManagerTermBase`-derived class owning one persistent per-env `F_{t-1}`
  buffer, reset on episode boundary.

TDD-verified (`tests/test_exploration_bonus_reward.py`, 20/20 passing)
including the specific regression test this mechanism was built to
satisfy: **`Correction_t` is explicitly asserted to NOT be constrained to
any particular sign** — a direct, pre-registered defense against
Experiment 5's own failure class (see "Relation to Experiment 5" below).

## Real run (Task 4): 3 seeds, 1500 iterations each, both falsification bars

| seed | mechanism-level (`frac_steps_raw_action_negative_near_object`, per env) | behavioral (`envs_with_sustained_lift`) | `max_height_gain_m` |
|------|--------------------------------------------------------------------------|--------------------------------------------|------------------------|
| 42   | 8/8 envs **null** (final checkpoint never got within 5cm of the object in any env, full 250-step episode) | 0/8 | ≈0.0088 (all envs) |
| 123  | 7/8 envs = **1.0**, 1/8 null (env 4) | 0/8 | ≈0.0088-0.0089 (all envs) |
| 7    | 1/8 env = **0.0** (env 5, confirmed — got near, never attempted), 7/8 null | 0/8 | ≈0.0088 (all envs) |

**Mechanism-level bar:** per the spec, "falsified only if this fraction
is exactly `0.000` in all 3 seeds." Seed 123's `1.0` result — a real,
repeated value across 7 independent parallel envs, not a rounding
artifact — is unambiguously nonzero. **The mechanism-level bar is
confirmed to fire, not falsified.** Note the null/zero distinction
matters here and was reported per-seed rather than collapsed: seed 42's
checkpoint never got close enough to the object at all to test the
mechanism (a training-dynamics observation in its own right — this
particular final checkpoint's reaching behavior had regressed by
iteration 1499, distinct from "reached but never attempted"), while seed
7 shows one env that did get close and genuinely never attempted (a
confirmed `0.000`, not a null).

**Behavioral bar:** a clean, uniform 0/24 — `max_height_gain_m` ≈0.0088m
in literally every env across all 3 seeds (physics-settle noise, not a
real lift; the same magnitude seen in the from-scratch baseline and
[[d8-d10-demo-warmstart]]'s own null), well under the 0.04m sustained-lift
threshold.

**Overall verdict, per the spec's own explicit rule** (falsified only if
*both* bars fail across all 3 seeds): **SPLIT**. The exploration
mechanism demonstrably works in at least one seed — the policy reliably
samples gripper-closure attempts specifically when near the object, not
at random or never — but this does not translate into any completed
lift in any seed.

## Independent verification

Per this project's standing practice, re-derived both bars for seed 123
(the strongest mechanism signal) from raw per-step data, not the summary
JSON:

- **Mechanism bar:** `raw_arrays.npz`'s `raw_gripper_action` and
  `ee_object_distance` arrays, reprocessed with a fresh, independent
  implementation of the near-object-restricted fraction, reproduced the
  exact per-env result (envs 0-3/5-7 = 1.0, env 4 = null) byte-for-byte.
- **Behavioral bar:** a fresh, independent reimplementation of
  `franka_checkpoint_review.py`'s post-`977a748` settle-detection logic
  (resting-z via early-window min, gain, sustained-lift run-length),
  applied to the raw `heights_*.npy`, reproduced `envs_with_sustained_lift
  = 0/8` and the same `max_height_gain_m` values to displayed precision.
- **Frame-by-frame video review** (seed 123, per Experiment 16's
  precedent that a shaped metric or even a summary height number can
  misrepresent what's physically happening): frames 10/15 show the
  gripper descending toward the die; frames 20/30 show the fingers
  visibly closed roughly where the die had been, the object no longer
  visible (occluded or displaced); frames 100/248 show the die still
  resting on the table, ungrasped, with the gripper retracted to a
  resting pose above and away from it. This visually confirms the
  instrumented split result — a real closure attempt near the object
  that does not result in a lift — rather than a video/instrumentation
  mismatch.

## No code bug found

Unlike several prior experiments in this arc, this task's real dispatch
surfaced no defect in the mechanism, instrumentation, or wiring itself —
Tasks 1-3's code behaved exactly as designed. The split result is a
genuine substantive finding about downstream grasp mechanics, not an
artifact of a shaping-formula or instrumentation bug. (Real, non-code
infra friction did occur mid-task: two SPOT preemptions during the
eval/video-review phase, both apparently from contention with the
concurrently-running d8/d10-H2 workstream over this project's shared
`GPUS_ALL_REGIONS=1` project-wide GCP quota — the same constraint
independently flagged by [[target-selection-clutter]] and
[[d8-d10-demo-warmstart]]. Both were resolved by real blocking-poll
until the quota freed, then a plain instance restart —
`--instance-termination-action=STOP` preserved the disk/venv/repo state
across both preemptions, so no re-install or re-training was needed, only
an idempotent re-run of the (short) eval script. All 3 seeds' full
1500-iteration training runs themselves completed in a single shot with
zero preemptions.)

## Relation to Experiment 5

[[experiment-05-potential-based-reward-shaping]] is the direct AR4-era
precedent this mechanism's own design was explicitly built to avoid
repeating: a monotonic running-max potential-based term whose docstring
incorrectly claimed its shaping reward was "always >= 0," which was
actually negative whenever the agent held its best-ever potential without
improving — making "never approach the object at all" the reward-optimal
policy, and producing a total null (0/10, "reach, grip, freeze," the
policy never even attempts closure).

This experiment's own TDD suite explicitly pre-registered and tested
against that exact failure class (`Correction_t` asserted to take both
positive and negative values in the unit tests, not constrained to any
sign) — and the real run confirms the fix generalizes to real behavior,
not just the unit tests: unlike Experiment 5, this checkpoint (seed 123)
does not freeze or refuse to approach — it reliably approaches and
attempts real closure specifically near the object. **This mechanism
avoided repeating Experiment 5's specific formula-sign-bug failure mode**,
a genuinely different and more advanced failure signature than that
precedent, even though the ultimate behavioral outcome (no completed
lift) is superficially the same "0/N" shape.

## Bottom line and forward pointer

This result, together with [[d8-d10-demo-warmstart]]'s independent
from-scratch null, now shows the same shape on two different, independently-
tried fix axes for d8: getting the policy to *attempt* the grasp at the
right place/moment is achievable (this experiment proves it, in ≥1 seed);
turning that attempt into a *completed* lift is not, regardless of
whether the fix targets exploration (this experiment) or warm-starting
from a demonstration ([[d8-d10-demo-warmstart]]). Per the design spec's
own explicit characterization of this exact split outcome, this points
*away* from the pure discoverability question both of those experiments
targeted and *toward* a downstream grasp-mechanics problem — plausible
next candidates are verifying actual finger-object contact/antipodal
alignment at the moment of closure (this project's Experiment 9/10
antipodal-grasp-quality axis — see
[[grasp-mechanics-antipodal-vs-magnitude]] — not revisited since the
Franka pivot), or the d8 die's own geometry/mass/friction parameters at
48mm-parity scale. Neither is decided or started here — a stop-and-report
point back to the controller, since H2 (RND-based intrinsic bonus) and H3
(scheduled entropy/noise) are only pre-authorized as fallbacks for a
plain both-bars falsification, which this is not.

**Checkpoints:** `logs/train_franka_jointdied8bigexplorationbonus/{2026-07-20_09-52-47,2026-07-20_10-25-11,2026-07-20_10-57-38}/model_1499.pt`
(seeds 42/123/7; cloud-local only, not synced to GCS).

**Cost:** ≈$1.2 of the plan's $3 cap (≈3.0hr real SPOT
g2-standard-4+L4 instance time across 3 start/stop windows, duration ×
published-SKU-rate estimate). Full teardown verified.

## Related

[[experiment-05-potential-based-reward-shaping]] (the prior formula-bug
failure this mechanism was built to formally avoid, and did),
[[reach-grasp-lift-gap]] (the project-wide throughline this experiment is
one more data point on), [[reward-hacking-and-sparse-discoverability]]
(the general concept this mechanism targets), [[d8-d10-demo-warmstart]]
(the sibling, independently-run fix attempt on the same null, also
falsified/split-adjacent — together the two now triangulate toward a
grasp-mechanics rather than discoverability explanation),
[[unified-multi-die-specialist-distillation]] (source of the original
0/24 d8 baseline this experiment tested a fix for).
