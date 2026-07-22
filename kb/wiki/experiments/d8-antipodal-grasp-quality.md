# d8 antipodal/force-closure grasp-quality reward (2026-07-20, dual action-space test — H_joint FALSIFIED, H_taskspace CONFIRMED, CLOSED)

**Status: CLOSED. Both conditions complete (H_joint below; H_taskspace in
its own section further down); the closing 5-row-outcome-matrix
classification and combined synthesis are in the "Closing verdict" section
at the bottom of this article. See `ROADMAP.md`'s matching entry for the
same synthesis in the project-status ledger.**

**Goal:** test whether porting AR4's antipodal/force-closure grasp-quality
reward (`tasks/ar4/mdp.py:902-940`), refit to Franka's real `mu=0.5`
friction coefficient (`antipodal_cos_threshold=-0.894427`, vs. AR4's own
`mu=1.0` → `-0.7071`), onto d8's robust, independently-established 0/24
grasp-discoverability null (`FrankaDieLiftJointD8BigEnvCfg`, 48mm-parity)
unlocks sustained-lift discovery — under joint-space control (H_joint,
this article) and, separately, task-space/relative-IK control
(H_taskspace). Spec:
`docs/superpowers/specs/2026-07-20-d8-antipodal-grasp-quality-design.md`.
Plan: `docs/superpowers/plans/2026-07-20-d8-antipodal-grasp-quality-implementation.md`.
Research: `docs/superpowers/specs/research/2026-07-20-d8-grasp-mechanics-literature.md`.

## Mechanism (Tasks 1-2, committed `74bd058`/`de02c5d`)

New, additive-only wiring, base classes untouched:

- `tasks/franka/antipodal_grasp_reward.py` — pure-`torch`
  `antipodal_grasp_bonus_raw`, TDD-verified (magnitude-and-direction check,
  the μ=0.5 vs. μ=1.0 threshold-boundary regression test, zero-force-vector
  epsilon guard, batch correctness).
- `tasks/franka/mdp.py`'s `antipodal_grasp_bonus` — thin wrapper reading
  `panda_leftfinger_contact`/`panda_rightfinger_contact`
  `ContactSensorData.force_matrix_w`.
- `FrankaDieLiftContactSceneCfg` (`dice_lift_joint_env_cfg.py`) — two
  single-body `ContactSensorCfg`s on the Franka fingers, empirically
  confirmed (Task 1's own bounded check) to read genuine nonzero contact
  force under this exact `ManagerBasedRLEnvCfg` wiring, not just assumed
  from the scripted-demo precedent.
- `AntipodalGraspRewardsCfg` (`lift_env_cfg.py`) — `RewardsCfg` +
  `antipodal_grasp_quality` term, `force_threshold=0.05`,
  `antipodal_cos_threshold=-0.894427`, `weight=1.0`, all fixed/non-tunable
  for this experiment.
- `FrankaDieLiftJointD8BigAntipodalEnvCfg` (Condition A) — pure
  field-override leaf on `FrankaDieLiftJointD8BigEnvCfg`, inherited
  `JointPositionActionCfg` unchanged (joint-space).

## H_joint real run (Task 3): 3 seeds, 1500 iterations each, both falsification bars

**Dispatch:** GCP cloud (desktop unreachable this session). SPOT stockout
across all 9 surveyed `us-central1`/`us-east1`/`us-west1`/`us-west4` zones
(matching this project's known failure mode) forced a fallback to
**on-demand** provisioning (`g2-standard-4` + 1x `nvidia-l4`,
`us-central1-a`) — a deliberate, flagged judgment call per the plan's own
"reasonable if the remaining job count is small enough to stay within the
cost cap" guidance, not a general policy change. A concurrent workstream
(`rl-franka-taskspace-antipodal`, evidently the H_taskspace/Task 4 dispatch)
held the project's single `GPUS_ALL_REGIONS=1` quota slot for roughly 3
hours before this task could provision at all; that instance was itself
confirmed independently SPOT-preempted (`compute.instances.preempted`
system event) mid-run — genuine queued waiting, not a false "busy" read,
resolved once quota usage read back to `0.0`.

| seed | mechanism-level (`Episode_Reward/antipodal_grasp_quality`, final-100-iter mean) | behavioral (`envs_with_sustained_lift`) | `max_height_gain_m` (uniform across envs) |
|------|------|------|------|
| 42   | **0.00000000** (nonzero early in training, peak transient 0.0087 around iter 48-60, decays to exact 0.0 by the final 100) | 0/8 | 0.008847 |
| 123  | **0.00000000** (nonzero transient peak 0.0010, decays to exact 0.0) | 0/8 | 0.008847-0.008848 |
| 7    | **0.00000000** (nonzero transient peak 0.0042, decays to exact 0.0) | 0/8 | 0.008847-0.008849 |

**Mechanism-level bar** (spec's own rule: falsified only if `< 1e-4` in
all 3 seeds): **all 3 seeds are exactly `0.00000000`** over the final 100
of 1500 iterations — unambiguously below the `1e-4` bar, and in fact an
exact-zero replay of Experiment 10's own historical AR4-era joint-space
outcome (`0.000000`), not just "small." The reward term is not dead
wiring — direct inspection of the full 1500-iteration tfevents trace shows
real, nonzero, non-uniform values throughout early-to-mid training (up to
775 nonzero data points in seed 42, peak 0.0087) — the mechanism fires,
then the policy converges away from it by the end of training in every
seed. **Mechanism-level bar: FALSIFIED (all 3 seeds).**

**Behavioral bar** (falsified only if 0/8 in all 3 seeds, 0/24 total):
**clean 0/24** — `max_height_gain_m` ≈0.0088m in literally every env
across all 3 seeds, the same physics-settle-noise magnitude seen in this
exact env cfg's already-established baseline null and the immediately-prior
exploration-bonus SPLIT result. **Behavioral bar: FALSIFIED (0/24 total).**

**H_joint verdict, per the spec's own explicit rule (falsified only if
BOTH bars fail): H_joint is FALSIFIED.** This is a clean falsification,
not a SPLIT — both bars fail in every seed, no mixed evidence to preserve.

## Independent verification

- **Mechanism bar** re-derived directly from the raw tfevents scalar trace
  (not a pre-computed summary — `Episode_Reward/antipodal_grasp_quality`
  read via `tensorboard.backend.event_processing.event_accumulator`
  against the actual `events.out.tfevents.*` file) for all 3 seeds:
  confirmed exact `0.0` over the last 100 of 1500 points in each, and
  confirmed the term's full-trajectory nonzero/decay-to-zero shape
  directly (not inferred).
- **Behavioral bar** re-derived for seed 42 from the raw `heights_*.npy`
  via a from-scratch reimplementation of `franka_checkpoint_review.py`'s
  post-`977a748` settle-detection algorithm (early-window
  `min(steps[10:45])` for `resting_z`, gain/lifted-mask/sustained-lift
  computed only over the post-settle portion `steps[10:249]`, excluding
  the pre-settle free-fall). First attempt at this reimplementation
  incorrectly included the pre-settle free-fall frames in the max-gain
  window, producing a spurious `max_gain≈0.0399m` (the object's own spawn
  height before free-fall, not a real ascent) — caught immediately by
  comparing against the production script's own reported `0.008847m`, not
  silently trusted. Re-reading `franka_checkpoint_review.py`'s exact
  slicing logic (`post_settle_start = min(EARLY_SETTLE_START, analysis_end)`,
  `post_settle_history = analysis_history[post_settle_start:]`) and
  correcting the reimplementation reproduced the production script's
  numbers exactly (`resting_z`, `max_z`, `max_gain`, `envs_with_sustained_lift`
  all matched to displayed precision). **This was a bug in the independent
  verification script, not in the project's own `franka_checkpoint_review.py`
  — no production code defect found.**
- **Frame-by-frame video review**, seed 42 (highest transient mechanism
  signal of the 3 seeds, so the most likely to show any positive
  behavioral evidence if one existed): frame 10 shows the gripper
  positioned directly above/near the die immediately after the free-fall
  settles. Frames 60/100/150/248 all show the arm holding a single static,
  reached-down pose near the table for the remainder of the 250-step
  episode — no visible lifting motion, no visible closure around the
  object distinguishable from the reached-down rest pose. This is
  consistent with (not contradicting) the instrumented height data: the
  post-settle max height is reached at step 10 (the tail of the settling
  wobble itself, value 0.0210m vs. resting 0.0121m — the 0.0088m "gain"
  entirely explained by settle overshoot, not any grasp-driven ascent) and
  never rises again for the rest of the episode.

## No code bug found

Task 3's own dispatch surfaced no defect in Tasks 1-2's mechanism,
instrumentation, or wiring — the contact-sensor/reward-term pipeline reads
real, non-degenerate values (confirmed both by the training-time nonzero
transient and the independent tfevents re-derivation). The one bug found
during this task was in the independent-verification reimplementation
itself (wrong window-slicing, see above), caught and corrected within the
same pass per this project's own bug-handling discipline, before it could
produce a wrong conclusion.

## Relation to the AR4-era Experiment 9/10/11 arc

This result is an **exact replay of Experiment 10's own joint-space
antipodal-reward regression-to-zero pattern**
([[experiment-10-antipodal-threshold-action-scale-solver]]: `0.000000`
final, despite the physically-correct threshold) — now reproduced on a
different robot (Franka vs. AR4), a different object (d8 die vs. cube),
and a from-scratch pure-`torch`/`ContactSensorCfg` reimplementation of the
mechanism, not a copy-pasted artifact of AR4's own code. This is exactly
the pattern the design spec's own outcome matrix anticipated as one
possible per-condition result and explicitly why H_taskspace must still be
run to completion regardless: per
[[experiment-11-taskspace-ik]]'s own AR4-era finding, joint-space
positioning precision — not the antipodal mechanism itself — was the
specific thing that regressed this signal to zero there; task-space/IK
control was what first produced a genuine nonzero antipodal signal on
AR4. Whether that same fix transfers to Franka/d8 is H_taskspace's own,
separately-reported question.

## Relation to the exploration-bonus SPLIT result

[[exploration-bonus-grasp-discovery]] is the immediately-prior experiment
on this identical env cfg, whose own forward pointer explicitly named
"verifying actual finger-object contact/antipodal alignment at the moment
of closure" as the next candidate direction after its own SPLIT result
(reliable closure attempts near the object, never a completed lift). This
result **complicates, not confirms, that forward pointer** for the
joint-space condition specifically: the antipodal-quality signal itself
never becomes learnable under joint-space control at all (a hard `0.0`,
not merely "learnable but insufficient" as the exploration-bonus result
was for its own mechanism) — so grasp-quality/antipodal alignment is not
yet demonstrated as *the* missing piece here, at least not reachable via
this action space. H_taskspace's own result will determine whether this
is an action-space-precision gate (matching Experiment 10→11 exactly) or
a genuinely deeper miss.

## Cost

Task 3 (H_joint) alone: ≈2.13hr on-demand `g2-standard-4`+`nvidia-l4`
instance-uptime (SPOT was stocked out project-wide at dispatch time,
on-demand used as a flagged fallback) + ≈3hr of blocking wait for the
concurrent H_taskspace workstream's quota to free (no cost during the
wait itself — no instance was running). Estimated compute cost (duration ×
published on-demand rate, ~2x the documented SPOT rate of $0.361/hr per
`docs/cloud/franka-cloud-shakedown.md`, no BigQuery billing export exists
to get an authoritative figure): **≈$1.6** (≈$1.54 compute + ≈$0.04·2.13/730
disk-month proration). Full teardown verified
(`scripts/check_cloud_state.sh`: zero instances/disks/snapshots belonging
to this task after deletion; the other workstream's own terminated
instance/disk, not this task's, is untouched). Checkpoints (`model_1499.pt`
+ tfevents, all 3 seeds) synced to
`gs://rl-manipulation-hks-runs/d8-antipodal-grasp-quality/joint-die-d8-big-antipodal/seed{42,123,7}/`.

---

## H_taskspace real run (Task 4): 3 seeds, 1500 iterations each, both falsification bars

**Condition B** (`FrankaDieLiftD8BigTaskspaceAntipodalEnvCfg`, `--variant
die-d8-big-taskspace-antipodal`): identical scene/rewards/PPO recipe to
Condition A, but `self.actions.arm_action` re-asserted to task-space/
relative-differential-IK (`DifferentialInverseKinematicsActionCfg`,
`action_dim=6`, confirmed live via the action-manager check noted at Tasks
1-2's own completion) instead of Condition A's inherited
`JointPositionActionCfg`.

**Dispatch:** GCP cloud (desktop unreachable this session, confirmed via
`scripts/check_gpu_availability.sh` → `TARGET=cloud`/exit 2, UNKNOWN).
First instance provisioned SPOT on the first zone attempt
(`us-central1-a`) with no queueing needed (H_joint's own Task 3 dispatch
had not yet claimed the shared `GPUS_ALL_REGIONS=1` quota at that exact
moment). All 3 training seeds completed their first attempt in a single
sitting with no divergence; the friction came later, during post-training
diagnostics: **two further genuine SPOT preemptions** (`gcloud compute
operations list` confirmed `compute.instances.preempted` system events,
one after only ~17s of a freshly-recreated instance's uptime — a cluster
matching `docs/cloud/dispatch-checklist.md`'s documented "repeated
preemptions can occur in bursts" gap), each recovered via the
snapshot-and-recreate-in-a-different-zone technique
(`us-central1-a` → `us-west1-a` → `us-east1-b`), plus one real
checkpoint-corruption hit (seed 7's `model_1400.pt` was a 0-byte
truncated write from the first preemption — caught by a `torch.load`
validity check before trusting it, not assumed from the filename alone;
recovered by resuming from the next-oldest valid checkpoint,
`model_1350.pt`). After the second preemption recurred within a very
short window, switched the instance to on-demand provisioning
(`gcloud compute instances set-scheduling --no-preemptible
--provisioning-model=STANDARD`) per this project's own documented
"reasonable when repeated preemption clusters exceed the earlier norm,
provided remaining work is small" judgment call — no further preemptions
after that. This task's own dispatch also held the shared
`GPUS_ALL_REGIONS=1` quota during part of the window the concurrent
H_joint/Task 3 workstream was itself queued and waiting (see that
section's own "Dispatch" note above) — genuine two-way contention on the
single project-wide slot, not a one-sided block.

| seed | mechanism-level (`Episode_Reward/antipodal_grasp_quality`, final-100-iter mean, re-derived from raw tfevents) | behavioral (`envs_with_sustained_lift`) | `max_height_gain_m` |
|------|------|------|------|
| 42   | **0.00023876** (passes the `1e-4` bar, ~2.4x) | 0/8 | 0.008847-0.008848 (uniform, no-grasp baseline signature) |
| 123  | **0.83944721** (passes overwhelmingly; sustained ~0.84-0.86 for the entire final several hundred iterations, not a transient spike) | **8/8** (clean sweep) | **0.306-0.403** (all 8 envs) |
| 7    | **0.00000012** (fails the bar) | 0/8 | 0.008847-0.008848 (identical no-grasp baseline signature to seed 42) |

**Mechanism-level bar** (falsified only if `< 1e-4` in all 3 seeds): **2
of 3 seeds (42, 123) exceed the bar** — seed 42 only marginally (2.4x),
seed 123 overwhelmingly and durably. **Mechanism-level bar: NOT
FALSIFIED.**

**Behavioral bar** (falsified only if 0/8 in all 3 seeds, 0/24 total):
**seed 123 alone contributes a full 8/8** — a clean, uniform sweep (every
env's `max_height_gain_m` in the 0.31-0.40m range, no partial/spurious
lift), matching this project's own repeated "0 or full-8/8-within-seed"
discovery pattern rather than a fluke single-env result. Seeds 42/7 are
both a clean 0/8 with the exact same ≈0.0088m no-grasp baseline signature
seen throughout this env cfg's history. **Behavioral bar: NOT FALSIFIED
(8/24 total).**

**H_taskspace verdict, per the spec's own explicit rule (falsified only if
BOTH bars fail across all 3 seeds): H_taskspace is CONFIRMED, not
falsified.** Unlike H_joint's clean, uniform falsification, H_taskspace's
own real result is genuinely heterogeneous across seeds — worth
preserving explicitly, not rounding to a single number: seed 123 is a full
clean success on both bars; seed 42 shows a real (re-derived from raw
data, not noise) but marginal mechanism-only signal that never
translates into behavioral lift, a seed-level pattern with the same shape
as this identical env cfg's own immediately-prior exploration-bonus SPLIT
result; seed 7 is a clean, total null on both bars, indistinguishable
from H_joint's own uniform result. Whether this 1-full/1-marginal/1-null
split constitutes a condition-level "SPLIT" or "CONFIRMED" in Task 5's own
5-row outcome matrix is that task's classification to make — this section
records the real, seed-level heterogeneity so that classification isn't
made from a collapsed summary number.

## Critic-divergence contingency (Experiment 11's own AR4-era risk): NOT triggered

Watched `Loss/value_function` live throughout all 3 seeds' full
1500-iteration runs, specifically for Experiment 11's own AR4-era
signature (exploding from ~0 to ~5e23 within a handful of iterations,
~95% of policy updates driven by a diverged critic). **Not observed in any
seed.** Seed 42's max was 0.0600 (bounded throughout). Seed 7's max was
0.0381 (bounded throughout, plus a further bounded 0.0576 max in its
resumed 150-1499 segment after checkpoint-recovery). Seed 123 showed a
real, larger rise (max 5.6484 around iteration ~270, up from a ~0.0006
baseline) — genuinely elevated relative to the other two seeds, and
watched closely as a candidate divergence in progress — but it **plateaued
and then declined** (iterations 260-274 oscillating 2.83-3.25, iterations
386-400 oscillating 2.9-3.7, iterations 594-603 back down to ~3.0-3.7,
iterations 816-820 down to 1.8-2.5, iterations 1033-1247 stable at
1.3-1.5), never approaching Experiment 11's astronomic blowup and fully
recovering to a low, stable value by training's end. **This is ordinary
elevated-but-bounded PPO value-loss variance, not the AR4-era divergence
bug — the plan's documented `clip_actions=5.0` contingency
(`Ar4PickPlaceTaskspacePPORunnerCfg`-style scoped `PPORunnerCfg` fix,
Experiment 11's own recipe) was not needed and was not built.** No change
to `train_franka.py`'s variant dispatch beyond what Task 2 already
shipped.

## Independent verification

- **Mechanism bar** re-derived directly from the raw tfevents scalar trace
  for all 3 seeds (`tensorboard.backend.event_processing.event_accumulator`
  against the actual `events.out.tfevents.*` files, not a pre-computed
  summary) — confirms the table above to full precision, including that
  seed 123's signal is genuinely sustained (not a single-point artifact)
  across its own final 100 logged iterations.
- **Behavioral bar** re-derived for seed 123 (the positive result) and
  seed 42 (a null) from the raw `heights_*.npy` via a from-scratch
  reimplementation of `franka_checkpoint_review.py`'s post-`977a748`
  settle-detection algorithm — matched the production script's own
  `envs_with_sustained_lift`/`max_height_gain_m`/`max_consecutive_lifted_steps`
  numbers exactly for both seeds on the first attempt (no bug found in
  this reimplementation this time).
- **Live physics diagnostic, beyond the plan's own minimum bar**: seed
  123's eval video (env_0, the built-in viewer camera anchored per-env at
  a static `eye=(1.8,1.8,1.1)`/`lookat=(0.4,0,0.35)`) shows the arm's
  reach-down-then-return-to-hover pose subtly enough at a handful of
  sampled frames that a first-pass visual read genuinely could not
  distinguish "held and lifted" from "static/glitched" by eye alone — a
  real ambiguity, not a rhetorical one, and worth recording as a concrete
  instance of this project's own standing "don't trust a proxy, verify
  directly" discipline. Resolved with a dedicated live rollout of this
  checkpoint (`env.scene["object"].data.root_pos_w` vs.
  `env.scene["robot"].data.body_pos_w[panda_hand]`, printed every step):
  the object's position tracks the `panda_hand` frame with a **constant
  ≈0.1001m offset in both X, Y, and Z, sustained across ~120 consecutive
  steps (steps 32-149) while both rise together from ≈0.10m to ≈0.46m** —
  the exact rigid-body signature of a genuine held grasp, not a
  contact-solver launch artifact (which would not maintain a constant
  offset) or an index/tensor-slicing bug (which would not track a
  physically plausible ≈10cm hand-to-object-center offset this
  consistently). A quantitative pixel-diff between the eval video's own
  rest frame and peak-height frame (env_0's own region only, excluding the
  neighboring env visible at the frame edge) confirms real, non-trivial
  visual change (~7% of pixels differ by >15 intensity levels, spanning
  nearly the full vertical extent of the arm's own silhouette) — the
  motion is real and visible, just more subtle in this specific camera
  framing than a first glance suggested, not evidence of a rendering or
  instrumentation bug.
- **Frame-by-frame video review**, seed 123: rest frame shows the gripper
  open, die visible on the table to the gripper's side. By ~0.6-0.9s in
  (steps 30-45) the gripper has closed and the die is no longer visible on
  the table (occluded within the closed fingers, consistent with a
  successful grasp) or anywhere else in frame (ruling out a "die knocked
  away" alternative). The arm holds a materially similar overall silhouette
  for the rest of the episode — confirmed by the physics diagnostic above
  to be the arm returning close to its own pre-reach hover height while
  still holding the object at a fixed offset, not a static freeze.

## No code bug found (beyond Task 2's own already-shipped mechanism)

No new production-code defect surfaced. The one real gap found and worked
around in the same pass was operational, not a code bug in Tasks 1-2's
mechanism: `scripts/franka_checkpoint_review.py` requires `--headless`
when run on a display-less cloud instance despite already setting
`args_cli.enable_cameras = True` internally — omitting it produces
`RuntimeError: Cannot render 'rgb_array' when the simulation render mode
is 'NO_GUI_OR_RENDERING'`. This is a cloud-dispatch operational detail
(the existing `--headless` cloud convention already documented in
`docs/cloud/dispatch-checklist.md` for training, just not yet called out
for the eval/review script specifically), not a defect in this plan's own
antipodal-grasp mechanism — noted here so a future cloud eval dispatch
doesn't have to rediscover it.

## Cost

Task 4 (H_taskspace) alone: ≈2.55hr SPOT `g2-standard-4`+`nvidia-l4`
instance-uptime across 3 segments (2.01hr initial training run,
0.54hr second segment before its own preemption, ~17s negligible third
segment before switching to on-demand) + ≈0.1-0.15hr on-demand uptime for
the post-training diagnostic work. Estimated compute cost (duration ×
published-SKU-rate estimate, this project's standing methodology, no
BigQuery billing export exists): **≈$1.1** (≈$0.92 SPOT compute + ≈$0.1
on-demand compute, on-demand estimated at ~2x the documented SPOT rate
per this project's own stated heuristic, no live billing-catalog lookup
performed this task + ≈$0.1 disk proration across ≈4.5hr elapsed wall
time). Combined with Task 3's own reported ≈$1.6 and Tasks 1-2's smoke
tests (≈$0.5 estimated in the plan), running total ≈$3.2 of the plan's
shared $6 cap — not exceeded. Full teardown verified
(`scripts/check_cloud_state.sh`: zero instances/disks/snapshots after
final deletion). Checkpoints (`model_1499.pt` + tfevents, all 3 seeds) and
eval artifacts (video/`heights_*.npy`/summary JSON, all 3 seeds) exist
only on the now-deleted cloud instance's boot disk — **not synced to
GCS** (this task's own dispatch did not run `sync_run_to_gcs.py`, matching
the immediately-prior exploration-bonus task's own same "cloud-local, not
GCS-synced" precedent); the eval `heights_*.npy`/JSON/video for seed 42
and seed 123, and the raw tfevents-derived numbers for all 3 seeds, were
downloaded locally during this task's own verification pass before
teardown and are the basis for every number reported in this section.

## Closing verdict (Task 5, 2026-07-20): outcome-matrix Row 2 — a real cross-platform confirmation, not a clean win

**Outcome-matrix classification (design spec's own 5-row table): Row 2 — H_joint
falsified, H_taskspace confirmed.** Per the spec's pre-registered rule (each
hypothesis falsified only if *both* its own bars fail across all 3 seeds):
H_joint's mechanism bar is exactly `0.0` in all 3 seeds and its behavioral
bar is a clean 0/24 — both bars fail, H_joint is falsified. H_taskspace's
behavioral bar succeeds in seed 123 (8/8) and its mechanism bar passes in
seeds 42 and 123 — neither bar fails in all 3 seeds, so H_taskspace is
confirmed, not falsified.

**Reading, per the spec's own table for this exact row: "exact replay of
the AR4-era Experiment 10→11 pattern: action-space precision is the real
gate; the antipodal mechanism itself is validated once precision is
available."** This is a real, notable finding, stated directly: the
specific action-space-dependent behavior of a classical antipodal/
force-closure reward — regressing to exact zero under joint-space control,
becoming learnable under task-space/IK control — was previously observed
on exactly one platform (AR4, Experiments 9-11). This experiment
reproduces that same pattern on a structurally different robot (Franka),
a different object (d8 die vs. cube), and a from-scratch pure-`torch`/
`ContactSensorCfg` reimplementation of the mechanism (not a copy of AR4's
own code) — a genuine cross-platform transfer of a mechanistic finding,
which is exactly the kind of methodology-generalizes-across-morphology
result [CLAUDE.md](../../../CLAUDE.md)'s own North Star cares about, found
here on its own empirical merits rather than assumed.

**But Row 2's classification should not be read as "task-space + antipodal
solves d8," and this synthesis does not overstate it as one.** Unlike
Experiment 11's own single-seed AR4 report (a robust nonzero signal in
91.6% of iterations on the one seed run), this experiment's 3-seed design
exposes real heterogeneity the AR4-era arc never had occasion to observe
under this mechanism: seed 123 is a full, clean 8/8 grasp+lift with a
strong, durable mechanism signal and independent physical confirmation (a
constant ≈0.10m hand-to-object offset sustained across ~120 steps while
both rise together — a genuine held grasp, verified beyond video review
alone); seed 42 gets a marginal, barely-above-threshold mechanism signal
with zero behavioral payoff — the same mechanism-fires/behavior-doesn't
shape as [[exploration-bonus-grasp-discovery]]'s own SPLIT result, at the
single-seed level; seed 7 gets nothing at all, indistinguishable from
H_joint's own uniform null. Task-space control is confirmed *necessary*
for this mechanism to ever produce a usable signal on Franka/d8 (H_joint's
hard zero vs. H_taskspace's real positive rate proves that unambiguously)
but is not by itself *sufficient* for reliable discovery — 2 of 3 seeds
under the identical task-space condition still failed behaviorally. The
accurate reading: task-space control is necessary for the antipodal
mechanism to ever become learnable on Franka/d8, and sufficient for
discovery in at least one seed, but not yet a reliable from-scratch recipe
across seeds.

### Relation to the exploration-bonus SPLIT result's own forward pointer

[[exploration-bonus-grasp-discovery]] named "verifying actual
finger-object contact/antipodal alignment at the moment of closure" as the
next candidate direction after its own SPLIT (reliable closure attempts
near the object, 0/24 lift, all under joint-space control — the same
action space as this experiment's H_joint condition). H_joint's result
**complicates that forward pointer taken literally, but confirms its
underlying diagnosis once combined with a second change**: adding the
antipodal-quality signal under the *same* joint-space control the
exploration-bonus experiment used does not merely fail to produce a lift
(as exploration-bonus's own bonus term did) — it fails to even make the
antipodal signal learnable at all, a harder failure than
exploration-bonus's own SPLIT. The forward pointer's implicit framing
(grasp-quality/antipodal alignment as *the* missing ingredient, addressable
on its own) was incomplete — it required an action-space change as a
co-requisite, not identified as necessary until this experiment's
dual-condition design surfaced it. Once that second change (task-space
control) is added, the forward pointer's underlying diagnosis is
vindicated in seed 123 specifically: the exact mechanism the
exploration-bonus result pointed toward is what closes the gap between
"reliable closure attempt" and "completed lift," producing this project's
first-ever sustained-lift discovery on d8 driven by an antipodal/
force-closure reward term. Combined honest reading: the exploration-bonus
forward pointer correctly identified antipodal alignment as the
mechanistically relevant next axis, but the fix that actually worked
needed both that reward term *and* a switch to task-space control together
— neither alone was sufficient here (joint-space + antipodal = hard zero;
task-space alone with no antipodal term is untested by this experiment and
a different question).

### Relation to d8's own H2 success — two independently-valid, non-competing mechanisms

[[d8-d10-demo-warmstart]]'s own H2 (checkpoint warm-start from the
converged d12 specialist) already solved d8 cleanly — **3/3 seeds, a full
24/24 sweep** — using `FrankaDieLiftJointD8BigEnvCfg`'s plain default
joint-space recipe, with no antipodal reward term at all. d8 is now
solvable via at least two structurally different, independently-verified
routes: H2's warm-start (24/24) and this experiment's task-space+antipodal
combination (8/24, seed 123 only). This is worth addressing directly
rather than skated over, because on its face it looks like a confusing
juxtaposition — two "fixes" for the same null, of very different apparent
strength. The honest synthesis is that **both are real, independently
valid findings that don't need to be reconciled into one story**, because
they answer genuinely different questions:

- **H2 answers a practical/engineering question**: given this project's
  existing dice-family curriculum (a converged specialist checkpoint for a
  geometrically similar shape already exists), how do you reliably get PPO
  to discover d8's grasp at all? Answer: seed initialization near an
  already-solved basin, with no reward or action-space change whatsoever.
  This is the more reliable, cheaper, currently-production-ready recipe for
  d8 specifically — but it is a shape-family-specific bootstrapping trick:
  it presupposes another shape's specialist already exists, and says
  nothing about *why* cold-start joint-space training fails, nor anything
  that obviously transfers to a genuinely new shape/arm with no nearby
  specialist to warm-start from.
- **This experiment answers a mechanistic question**: does a real,
  physically-grounded antipodal/force-closure signal matter for grasp
  discovery at all, and if so, under what conditions does it become
  learnable? Answer: yes, mechanistically — but it is gated on
  action-space precision exactly as the AR4-era arc found, and even once
  unlocked, is not yet seed-reliable from scratch. This result is weaker
  in practical seed-reliability than H2's warm-start, but it is the more
  fundamental explanation of *why* from-scratch joint-space training fails
  here (imprecise positioning prevents antipodal contact from ever
  becoming learnable, independent of whether a warm start is available),
  and it is further evidence in favor of task-space/Cartesian action
  formulations — the action-space family the North Star already favors
  for cross-arm/cross-task generalization, on grounds independent of this
  specific result.

Neither mechanism supersedes the other, and this article does not force a
ranking between them. If the practical goal is "get d8 working today,"
H2's warm-start is the stronger, cheaper, more reliable recipe. If the
goal is "understand why the mechanism fails/succeeds in a way likely to
transfer to a new shape or arm with no existing specialist to warm-start
from," this experiment's task-space+antipodal result is the more
diagnostic finding — action-space-family-general rather than
shape-family-specific — even though it is not yet reliable enough on its
own to be a recommended default.

### Honest next candidate direction

This result is not the dispositive "both falsified" row that would close
Direction 1 (contact/antipodal grasp verification) for Franka/d8 — it is a
genuine, if heterogeneous, confirmation that the mechanism matters and
transfers cross-platform once action-space precision is available. The
next well-motivated step, if this axis is revisited, is investigating
*why* 2 of H_taskspace's 3 seeds still fail even with both fixes present
(task-space control + antipodal reward) — whether the surviving failure
mode is itself an exploration problem (the same shape as the
exploration-bonus SPLIT, now one level further in), a residual
positioning-precision gap even under task-space/IK control, or something
else — rather than assuming task-space+antipodal is now a solved recipe
just because it produced this project's first positive antipodal-driven
d8 lift. Not decided or started here, per this plan's own explicit
"do not start any new experiment" constraint on this closing task — a
decision for Principal on whether/how to pursue that residual-seed
question is a separate, later call.

## Root cause investigation (2026-07-21 follow-up): joint-space learns to AVOID contact entirely, not "touch non-antipodally" — and the mechanism is not the one originally hypothesized

**Motivation.** The Closing verdict above establishes THAT H_joint fails and H_taskspace succeeds, but this project's own AR4-era arc (Experiments 9→10→11) never root-caused *why* joint-space regresses the antipodal signal to zero — it only found that switching to task-space worked. This follow-up root-causes the mechanism directly from real rollout data (not just tfevents scalars), using a new instrumented diagnostic script, `scripts/diag_antipodal_root_cause.py` (headless, no video — records per-step contact-force vectors on both jaws, the antipodal geometric sub-conditions, every `AntipodalGraspRewardsCfg` reward term's raw pre-weight value via direct `mdp` function calls — the same direct-call pattern `scripts/smoke_test_graspgoal_ground_penalty.py` established — plus `ee_frame` position/orientation and the policy's own raw actions, all cross-checked against the exact training-time `antipodal_grasp_bonus_raw` function via an in-script assertion).

**Method.** Three data sources, all real rollouts (128 envs × 249 steps ≈ 31,872 samples per checkpoint), not proxies:
1. The three ORIGINAL H_joint seeds' already-existing final checkpoints (`gs://rl-manipulation-hks-runs/d8-antipodal-grasp-quality/joint-die-d8-big-antipodal/seed{42,123,7}/`) — free, no retrain.
2. A fresh joint-space retrain (Condition A, seed 42, identical recipe, GCP SPOT `g2-standard-4`+`nvidia-l4`) with checkpoints kept at iterations {0, 100, 300, 700, 1499} — since the original run's intermediate checkpoints (`save_interval=50`) were never GCS-synced (only the final was), a fresh run was required to get a trajectory, not just an endpoint. Synced throughout to `gs://rl-manipulation-hks-runs/d8-antipodal-root-cause/joint-seed42-retrain/` via a background sync loop (mitigation for the SPOT-preemption/reboot-corruption incident below).
3. A fresh task-space retrain (Condition B, seed 123 — the one seed that succeeded in the original H_taskspace run), same 5 checkpoints, `gs://rl-manipulation-hks-runs/d8-antipodal-root-cause/taskspace-seed123-retrain/`.

**Operational incident (new infra gap, folded into `docs/cloud/dispatch-checklist.md`):** the first SPOT instance was genuinely preempted (`compute.instances.preempted` confirmed via `gcloud compute operations list`) 34 minutes into the joint-space retrain; on `gcloud compute instances start`, the instance came back `RUNNING` but stuck at a GRUB rescue prompt (confirmed via `get-serial-port-output`) and never finished booting — a boot-disk corruption distinct from every previously-documented preemption-recovery case in this project (which all resumed cleanly). Cut losses rather than debug GRUB interactively: deleted the stuck instance (verified zero leftover resources) and re-provisioned fresh (on-demand was fully stocked out project-wide across 10 zones at the time, confirmed empirically before falling back to SPOT again) rather than trying to recover the corrupted disk. The interrupted retrain's own partial checkpoints were unrecoverable (lost with the deleted disk), but this cost no real data since the retrain was restarted from scratch on the new instance and ran to completion cleanly; the three original seeds' free-rollout `.npz` files (not the summary numbers, which were already captured in this task's own transcript) were similarly lost with that same disk and are not re-derivable without re-running those three rollouts.

### Finding 1: contact frequency, not geometric precision, is the discriminating variable — H1 resolved decisively toward "avoidance"

Directly measured fraction of (step, env) samples where BOTH jaws register force above `force_threshold=0.05N` (the antipodal term's own magnitude gate):

| checkpoint | contact_frequency | antipodal_satisfying_frequency | fraction of contact steps that ARE antipodal |
|---|---|---|---|
| joint-space, seed 42 final (original run) | **0.0** | 0.0 | n/a |
| joint-space, seed 123 final (original run) | **0.0** | 0.0 | n/a |
| joint-space, seed 7 final (original run) | **0.0** | 0.0 | n/a |
| joint-space retrain (seed 42), iter 0 | 0.0 | 0.0 | n/a |
| joint-space retrain (seed 42), iter 100 | **0.0** | 0.0 | n/a |
| joint-space retrain (seed 42), iter 300 | **0.0** | 0.0 | n/a |
| joint-space retrain (seed 42), iter 700 | **0.0** | 0.0 | n/a |
| joint-space retrain (seed 42), iter 1499 | **0.0** | 0.0 | n/a |
| task-space retrain (seed 123), iter 0 | 0.0 | 0.0 | n/a |
| task-space retrain (seed 123), iter 100 | 0.00047 | 0.00044 | **93.3%** |
| task-space retrain (seed 123), iter 300 | 0.6297 | 0.6254 | **99.3%** |
| task-space retrain (seed 123), iter 700 | 0.8539 | 0.8518 | **99.7%** |
| task-space retrain (seed 123), iter 1499 | 0.8781 | 0.8780 | **99.996%** |

Joint-space's contact frequency is **exact, literal `0.0` at every single one of 8 checkpoints spanning the full 0→1499 range across 4 different seeds/runs (all 3 original seeds' final checkpoints + the fresh seed-42 retrain's full 5-point trajectory)** — not a low number, an exact zero over ~32k samples each time. This directly answers the question the design spec's own outcome matrix left open: joint-space's policy is not touching the object non-antipodally and failing the geometric condition — **it converges to never touching the object at all.** Task-space, in clean contrast, shows contact frequency **rising monotonically from 0 to 88%** over the identical 1500 iterations, and — critically — whenever contact happens, it is *already* overwhelmingly antipodal from the very first checkpoint where any contact appears at all (93.3% at iter 100, rising to 99.996% by iter 1499). This means task-space's own learning problem was never "achieve contact, then fix its geometry" — geometric correctness essentially comes for free once any contact happens at all under task-space; the entire learning curve is about **achieving contact reliably in the first place**, which is exactly the thing joint-space never manages to do even once in ~256k sampled (step, env) pairs across this investigation.

Independent reproduction: seed 123's re-derived `antipodal_satisfying_frequency=0.8780` at iter 1499 closely matches the original H_taskspace run's own tfevents-derived `Episode_Reward/antipodal_grasp_quality` final value (`0.83944721`) — a genuinely different measurement (single eval rollout of 128 envs at the exact final checkpoint vs. a 4096-env on-policy training average over the final 100 iterations) landing in close agreement, corroborating both the original result and this follow-up's own new instrumentation.

### Finding 2: the reward-structure hypothesis (H2) is ruled out numerically, not just by inference

Read every `RewardsCfg`/`AntipodalGraspRewardsCfg` term's raw (pre-weight) value directly from the same rollouts. The `action_rate`/`joint_vel` penalty terms' *weighted* per-step contributions (`weight=-1e-4` each) are:

| checkpoint | action_rate (raw → weighted) | joint_vel (raw → weighted) |
|---|---|---|
| joint-space retrain, iter 1499 | 0.0644 → **−6.4e-6** | 1.1991 → **−1.2e-4** |
| task-space retrain, iter 1499 | 0.2706 → **−2.7e-5** | 3.6887 → **−3.7e-4** |

Both conditions' weighted penalty contributions are **2-4 orders of magnitude smaller** than the reward terms that actually matter (`reaching_object` realizing 0.10-0.84 at weight 1.0; `antipodal_grasp_quality` realizing up to 0.88 at weight 1.0) — nowhere near large enough to plausibly disincentivize contact-seeking behavior in either condition. More decisively: **the reward structure, including these exact penalty weights, is byte-identical between conditions** (both inherit `AntipodalGraspRewardsCfg` unchanged), and it produces dramatically different outcomes (0% vs. 88% contact frequency) — if the reward structure itself were the cause, it would have to fail identically in both conditions, which it does not. **H2 is ruled out directly, not merely deprioritized.**

### Finding 3: the exploration/action-space-geometry hypothesis (H3) as originally framed is falsified by the data — but a refined, better-evidenced version of it holds

The original H3 hypothesized joint-space's 7-DOF direct actuation produces **noisier** per-step end-effector motion than task-space's direct 6-DOF Cartesian action. Measured directly (`ee_frame` position/orientation step-to-step deltas):

| checkpoint | joint-space EE pos jitter (mean, m) | task-space EE pos jitter (mean, m) | joint-space EE ang jitter (mean, rad) | task-space EE ang jitter (mean, rad) |
|---|---|---|---|---|
| iter 0 | 0.000157 | 0.00701 | 0.000497 | 0.02371 |
| iter 100 | 0.00256 | 0.00596 | 0.00804 | 0.02820 |
| iter 300 | 0.00324 | 0.02036 | 0.01573 | 0.05089 |
| iter 700 | 0.00385 | 0.01460 | 0.01052 | 0.03193 |
| iter 1499 | 0.00320 | 0.00527 | 0.00952 | 0.01271 |

**Task-space's own raw per-step jitter is equal to or LARGER than joint-space's at every single checkpoint, including iteration 0 (pure random-init policy, before any learning)** — the exact opposite of the original hypothesis. "Joint-space produces geometrically noisier motion" is falsified as stated.

The real, data-supported distinguishing signature instead shows up in the **shape of the `reaching_object` reward's own trajectory**:

| checkpoint | joint-space `reaching_object` | task-space `reaching_object` |
|---|---|---|
| iter 0 | 0.00089 | 0.0000682 |
| iter 100 | **0.6015 (peak)** | 0.7899 |
| iter 300 | 0.2337 | 0.6506 |
| iter 700 | 0.1136 | 0.8203 |
| iter 1499 | **0.0957** | **0.8394** |

Joint-space **transiently discovers** a real approach-the-object capability early (peaking at iteration 100, matching the original run's own reported antipodal-signal transient peak window of iter ~48-60), then **regresses away from it** over the remaining 1400 iterations, converging to a policy that stays substantially farther from the object than it once did. Task-space's `reaching_object` reward instead rises and *stays* high throughout. Given the reward structure is identical (Finding 2), this is not a reward-incentive difference — it is a **learnability/credit-assignment difference intrinsic to the action-space mapping itself**: joint-space's 7-DOF direct actuation requires the policy to implicitly learn a configuration-dependent, nonlinearly-coupled mapping from raw joint deltas to precise end-effector motion near the object — a mapping that changes shape depending on the arm's current pose — whereas task-space's action is mediated by a fixed, non-learned differential-IK controller that keeps the action-to-EE-motion relationship consistent regardless of configuration. As PPO's own action-distribution entropy narrows over training (a documented PPO dynamic — see citations below), joint-space's early, marginal, exploration-noise-driven successful approaches are not consistently reinforced or generalized, and the policy abandons them for a lower-variance, lower-ceiling "hover-near-but-never-touch" local optimum; task-space's configuration-independent mapping keeps the same final-approach precision reachable and reinforceable even as entropy shrinks, so it does not fall into the same trap.

**Literature grounding** (existence/accuracy-checked per this project's standing citation practice, not deeply re-litigated once confirmed real):
- Martín-Martín, Lee, Gardner, Savarese, Bohg, Garg, "Variable Impedance Control in End-Effector Space: An Action Space for Reinforcement Learning in Contact-Rich Tasks," IROS 2019 (arXiv:1906.08880) — directly compares joint-torque/joint-PD vs. task-space/impedance action spaces on contact-rich manipulation; task-space parameterizations learn faster and more reliably, attributed to direct alignment with task-relevant Cartesian DOFs vs. a configuration-dependent joint mapping. On-point, not incidental.
- Varin, Grossman, Kuindersma, "A Comparison of Action Spaces for Learning Manipulation Tasks," IROS 2019 (arXiv:1908.08659) — ablates torque/joint-PD/inverse-dynamics/task-space-impedance action spaces across contact-rich tasks with both PPO and SAC; task-space impedance "significantly reduces the number of samples needed" across all tasks/algorithms. Direct sample-efficiency comparison.
- Hsu, Mendler-Dünner, Hardt, "Revisiting Design Choices in Proximal Policy Optimization" (arXiv:2009.10897, 2020 — remains an arXiv preprint, not confirmed peer-reviewed, cited as such) — formally characterizes a PPO failure mode where a continuous Gaussian policy initially converges toward high-reward regions, then diverges into low-reward regions as the policy's variance shrinks (the score function becomes hypersensitive to tail actions, and a single off-policy reward signal can produce an outsized update the policy has little signal to recover from). A close mechanistic match to the observed `reaching_object` rise-then-decay.
- Nikishin, Schwarzer, D'Oro, Bacon, Courville, "The Primacy Bias in Deep Reinforcement Learning," ICML 2022 (arXiv:2205.07802) — thematically adjacent (agents overfit to early experience, fail to incorporate later evidence) but its own experiments are off-policy/value-based (SAC, Rainbow), not on-policy PPO continuous control — cited as a weaker analogy, not a direct mechanistic match, and flagged honestly as such rather than overstated.

### Candidate fixes surveyed, NOT implemented (a design decision, not a scoped bug fix)

Per this task's own explicit instruction not to implement blind: three candidates were considered and none were implemented, because each requires a genuine design decision beyond this investigation's scope (root-causing an already-closed result, not authoring a new experiment):

1. **Warm-start joint-space from the task-space policy's weights.** Blocked cleanly at the output layer (8 total actions — 7 joint + gripper — vs. task-space's 7 — 6 Cartesian + gripper — a shape mismatch), though the shared trunk could in principle transfer with checkpoint-surgery plumbing this repo doesn't currently have. A real candidate, but a nontrivial architecture change.
2. **Reward-shaping adjustment.** Not well-motivated — Finding 2 directly rules out the reward structure as the differentiator, so changing reward weights would not address a credit-assignment/exploration problem.
3. **An action-space parameterization between raw absolute joint targets and full 6-DOF differential-IK.** Isaac Lab v2.3.1's own `isaaclab/envs/mdp/actions/actions_cfg.py` (surveyed directly on the live cloud instance, not merely assumed) has real, concrete options besides the two already tested: `RelativeJointPositionActionCfg` (still fully joint-space, no IK layer at all, but INCREMENTAL/relative deltas rather than absolute-position-with-offset targets — bounds each action's effect to a small, fixed-scale joint delta regardless of current pose) and `OperationalSpaceControllerActionCfg` (a distinct operational-space controller, not the `DifferentialInverseKinematicsActionCfg` already used for H_taskspace). This is the most direct next test this investigation surfaces: `RelativeJointPositionActionCfg` isolates "relative/incremental action semantics" from "joint-space vs. task-space" as two axes this experiment's own 2-condition design changed simultaneously (Condition A vs. B differ in BOTH joint-vs-task AND absolute-vs-relative at once) — a genuinely new Tier 1 structural-experiment candidate (new action term = new hypothesis + spec + plan per `CLAUDE.md`'s workflow), flagged here for Principal's own next-direction call, not decided or spec'd by this investigation.

### Cost

Two SPOT `g2-standard-4`+`nvidia-l4` instances (the first destroyed after the GRUB-corruption incident above, ~1hr instance-uptime before deletion; the second ran the full retrain+diagnostic pipeline end-to-end, ~5.5hr instance-uptime including install/rollouts/two full 1500-iteration training runs). Estimated (duration × published SPOT SKU rate, no BigQuery billing export exists in this project): **≈$2.5** total, well within a reasonable follow-up budget and consistent with this project's own prior per-run cost pattern. Full teardown verified via `scripts/check_cloud_state.sh` (zero instances/disks/snapshots after the final deletion).

## H_relative test (2026-07-21 follow-up): `RelativeJointPositionActionCfg` CONFIRMED — a genuinely joint-space fix, 3/3 seeds

**Motivation.** The root-cause investigation above named `RelativeJointPositionActionCfg` (delta/incremental joint targets, `applied_target = raw_action * scale + current_joint_pos`, recomputed fresh each control step) as the most direct next test: it isolates "relative vs. absolute action semantics" from "joint-space vs. task-space" as two axes the original H_joint/H_taskspace design changed simultaneously. Full design: `docs/superpowers/specs/2026-07-20-d8-relative-joint-action-design.md`. Implementation: `docs/superpowers/plans/2026-07-20-d8-relative-joint-action-implementation.md` (Tasks 1-4). New env cfg `FrankaDieLiftJointD8BigRelativeAntipodalEnvCfg` (`tasks/franka/dice_lift_joint_env_cfg.py`) — Condition A's full inherited chain (d8 48mm-parity, `AntipodalGraspRewardsCfg`, `FrankaDieLiftContactSceneCfg`) untouched, only `self.actions.arm_action` swapped to `RelativeJointPositionActionCfg(scale=0.1, use_zero_offset=True)`, Kuka Allegro dexsuite's own shipped precedent for `scale`. A real critic-divergence bug (`clip_actions` on the wrong cfg class, then on the wrong field) was found and fixed mid-Task-3 (`FrankaLiftRelativeJointPPORunnerCfg`, `clip_actions=5.0`) — a real, if scoped, PPO-runner-cfg exception the plan's own contingency clause anticipated.

**Method.** 3 seeds (42, 123, 7), full 1500-iteration training runs, checkpoints preserved+GCS-synced throughout (iterations 0/100/300/700/1499 — all landing on `save_interval=50`). Measurement: `scripts/diag_antipodal_root_cause.py --variant condition-relative`, 64-env headless rollouts at all 5 checkpoints × 3 seeds (15 rollouts), identical contact-frequency/antipodal-frequency/`reaching_object` instrumentation as the root-cause doc, plus `scripts/franka_checkpoint_review.py`'s sustained-lift behavioral bar (8 envs × 3 seeds) at the final checkpoint. Dispatched to a single GCP SPOT `g2-standard-4`+`nvidia-l4` instance (repo cloned via public HTTPS, no `git archive` needed now that the repo is public); zero preemptions this run.

### Contact-frequency trajectory — all 3 seeds independently CONFIRMED

| checkpoint | seed 42 | seed 123 | seed 7 |
|---|---|---|---|
| iter 0 | 0.0 | 0.0 | 0.0 |
| iter 100 | 0.005208 | 0.0 | 0.068461 |
| iter 300 | 0.503451 | 0.134601 | 0.491403 |
| iter 700 | 0.853853 | 0.011483 | 0.885166 |
| iter 1499 | **0.880146** | **0.825991** | **0.892508** |

Antipodal-satisfying frequency (fraction of ALL samples meeting the full bonus condition) tracks contact frequency almost 1:1 at every checkpoint once contact appears — same "geometry comes for free once contact happens" pattern the root-cause doc found under task-space (fraction-of-contact-that-is-antipodal ≥98% by iter 700 in all 3 seeds, reaching 99.99-100% by iter 1499).

Applying the spec's exact numeric bar per seed:
- **Seed 42: CONFIRMED** — iter 1499 = 0.880146 ≥ 0.05, and ≥ iter 700 (0.853853).
- **Seed 123: CONFIRMED** — iter 1499 = 0.825991 ≥ 0.05, and ≥ iter 700 (0.011483). Note a real, honestly-reported wrinkle: seed 123 dipped to 0.011 at iter 700 (down from 0.135 at iter 300) before recovering sharply to 0.826 by iter 1499 — a transient version of Finding 3's diagnosed "abandon a marginal early gain" pattern, but **self-correcting rather than terminal**, unlike absolute joint-space's identical-shaped dip that never recovered in any of its own 8 measured checkpoints. The spec's bar only checks final-vs-iter-700, not full-trajectory monotonicity, so this seed clears CONFIRMED on the letter of the rule; flagged here rather than smoothed over.
- **Seed 7: CONFIRMED** — iter 1499 = 0.892508 ≥ 0.05, and ≥ iter 700 (0.885166).

**Overall: CONFIRMED, 3/3 seeds** — exceeds the spec's own "at least 2 of 3" bar for confirmation cleanly; no seed meets or approaches the FALSIFIED bar (none show iter-1499 <0.01 and <50% of peak — the exact opposite is true in all 3).

Independent re-derivation (this project's standing practice): seed 42's full 5-checkpoint trajectory was recomputed directly from the raw `.npz` `magnitude_ok`/`antipodal_ok` arrays (not the summary JSON) and matched the reported `diag_antipodal_root_cause.py` summary exactly at all 5 checkpoints, bit for bit.

### Three-way curve-shape comparison — the actual point of this experiment

| checkpoint | absolute joint-space (Condition A, seed 42 retrain) | task-space (Condition B, seed 123 retrain) | relative joint-space seed 42 | relative joint-space seed 123 | relative joint-space seed 7 |
|---|---|---|---|---|---|
| iter 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| iter 100 | 0.0 | 0.00047 | 0.005208 | 0.0 | 0.068461 |
| iter 300 | 0.0 | 0.6297 | 0.503451 | 0.134601 | 0.491403 |
| iter 700 | 0.0 | 0.8539 | 0.853853 | 0.011483 | 0.885166 |
| iter 1499 | 0.0 | 0.8781 | 0.880146 | 0.825991 | 0.892508 |

Absolute joint-space's own contact frequency is exact `0.0` at every checkpoint, every seed, across every measurement this arc has taken (12 checkpoints now, 5 seeds/runs) — never rises at all, the falsified baseline. Task-space's own curve rises monotonically to an 88% asymptote. **Relative joint-space's seeds 42 and 7 reproduce task-space's own curve shape almost exactly** — monotonic rise, near-identical intermediate values (seed 42's iter 700 = 0.8539, literally matching task-space's own iter-700 value to 4 decimal places), and a final value (88.0-89.3%) at or slightly above task-space's own ceiling (87.8%). Seed 123 shows the one real deviation — a genuine mid-training dip (Finding 3's diagnosed signature, in miniature) — but recovers to the same ~83% final band the other two seeds land in, rather than collapsing permanently the way every one of absolute joint-space's own 8 measured checkpoints did. **This directly answers H_relative's own falsifiable question**: the fix is not merely "delayed the same collapse" (which would require final ≪ peak, the opposite of what all 3 seeds show) — it changed the shape of the curve, into something statistically indistinguishable from task-space's own success trajectory in 2/3 seeds, and a transient-not-terminal version of the original failure mode in the third.

### Behavioral bar

`franka_checkpoint_review.py`'s sustained-lift protocol (0.04m threshold, `977a748`'s settle-window fix), final checkpoint, 8 envs/seed: **seed 42: 8/8, seed 123: 8/8, seed 7: 8/8 — 24/24 total**, a full clean sweep exceeding H_taskspace's own aggregate result (8/24, only 1 of 3 seeds a clean 8/8). Video-reviewed per Experiment 16 discipline (a shaped/instrumented metric alone is not trusted at face value): seed 123 (the seed with the mid-training dip, judged the most interesting given the wrinkle above) was reviewed frame-by-frame — the rest frame (step 15) shows the die resting on the table next to an open gripper; the peak frame (step 248, object height 0.453m vs. a 0.012m resting height) shows the gripper closed with no die visible on the table, consistent with a genuine carry; contact-force data for this exact checkpoint/rollout independently shows both jaws registering force with `cos_angle≈-0.99` (deep antipodal geometry) at 82.6% of samples. Three independent signals (physics-buffer height, contact-force geometry, and the video's own object-disappears-from-table framing) triangulate to a real grasp+lift, not a video artifact or a wedge.

### Verdict and next direction

**H_relative is CONFIRMED**, cleanly, 3/3 seeds on both the mechanism bar and the behavioral bar. A genuinely joint-space (no IK, no arm-specific controller) action-term change — isolating "delta vs. absolute joint targeting" as the true variable, exactly as this investigation's own root-cause diagnosis predicted — resolves the exact-zero-contact-forever collapse the original H_joint condition showed in every one of 8 independently-measured checkpoints. This is independently significant for `CLAUDE.md`'s own North Star: unlike task-space/differential-IK (H_taskspace), a relative joint-space action requires no arm-specific IK controller or kinematic-chain configuration at all — it is close to the most morphology-agnostic action space this project has tested, and this result is real evidence that the North Star's "drop in a new arm, training should succeed immediately" bar does not require an IK/task-space layer as a hidden prerequisite.

Honest next candidate direction, given this real (not hoped-for) result: extend this same action-term fix to d10/d12/d20 (the unified-multi-die-specialist work) to check whether the fix generalizes across object scale, since every d8-specific result in this arc has needed a fresh check at other sizes before being trusted as general; and separately, investigate seed 123's own transient dip specifically — is it noise-scale-dependent (a `scale=0.1` hillclimb candidate, per the plan's own Tier-2 note) or a milder, recoverable instance of the same credit-assignment mechanism Finding 3 diagnosed. Neither started here, per this plan's own explicit "do not start any new experiment" constraint — flagged for Principal's next-direction call.

### Cost

One SPOT `g2-standard-4`+`nvidia-l4` instance, ≈40 minutes instance-uptime (install + 15 diagnostic rollouts + 3 behavioral evals, zero preemptions) ≈ **$0.25**. Combined with Tasks 1-3's own recorded spend (≈$1.6-1.75, dominated by the 3-seed/1500-iteration training run), this plan's total cost is **≈$1.9-2.0**, well under its own $5 cap. Full teardown verified via `scripts/check_cloud_state.sh` (zero instances/disks/snapshots after deletion). All raw diagnostic/behavioral-eval artifacts (`.npz`, summary JSONs, videos, height arrays) synced to `gs://rl-manipulation-hks-runs/d8-relative-joint-action/diag_relative_task4/` and `.../behavioral_eval_task4/`.

## Related

[[experiment-09-antipodal-grasp-bonus]],
[[experiment-10-antipodal-threshold-action-scale-solver]],
[[experiment-11-taskspace-ik]] (the AR4-era arc both conditions test
transfer of — H_joint replays Experiment 10's own regression-to-zero
exactly; H_taskspace's seed 123 replays Experiment 11's own "task-space
unlocks the first genuine sustained antipodal signal" finding, though with
seed-level heterogeneity Experiment 11's own single-seed AR4 report did
not have occasion to observe), [[grasp-mechanics-antipodal-vs-magnitude]],
[[action-space-design]] (the joint-space-vs-task-space axis both
conditions were built to test — this experiment's own real result is
direct, if seed-heterogeneous, evidence in that axis's favor),
[[exploration-bonus-grasp-discovery]] (the sibling SPLIT result on this
identical env cfg — H_taskspace's own seed 42 shows the same
mechanism-fires/behavior-doesn't shape at the single-seed level, and whose
own forward pointer this experiment's H_joint condition complicates and
whose H_taskspace condition ultimately vindicates), [[d8-d10-demo-warmstart]]
(H2's own independent, non-antipodal, non-task-space solution to this same
d8 null — see "Relation to d8's own H2 success" above for why the two
don't need reconciling into one story), [[reach-grasp-lift-gap]],
[[ppo-critic-divergence]] (Experiment 11's own AR4-era failure mode,
watched for and not observed here — though a real, if differently-shaped,
`clip_actions`-cfg-plumbing bug surfaced during H_relative's Task 3 and was
fixed, see the H_relative section above), [[action-space-design]] (further
narrowed by H_relative into delta-vs-absolute *within* joint-space — a
genuinely joint-space fix now confirmed, not just task-space vs. joint-space
as a whole), `CLAUDE.md`'s North Star (H_relative's own CONFIRMED result is
direct evidence that closing the North Star's "drop in a new arm, train
immediately" bar does not require an arm-specific IK/task-space controller
layer as a hidden prerequisite).
[[ar4-vs-franka-root-cause-comparison]] (2026-07-21 follow-up: this
H_relative fix was tested for transfer to AR4's own historical
Experiment 26 null — H_ar4_relative is FALSIFIED, 2/3 AR4 seeds reproduce
the identical all-zero collapse this Franka result resolved, and the one
partial exception never reaches real antipodal geometry either; full
per-seed data and the 3-signature jaw-mimic classification live in that
article's own dated section, not duplicated here).
