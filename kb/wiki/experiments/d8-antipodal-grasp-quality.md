# d8 antipodal/force-closure grasp-quality reward (2026-07-20, dual action-space test — H_joint FALSIFIED, H_taskspace CONFIRMED)

**Status: both conditions now complete and reported here (H_joint below;
H_taskspace in its own section further down). The closing 5-row-outcome-
matrix classification and combined verdict are Task 5's own job, not
written here — this article records each condition's own real result and
evidence only.**

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
mechanism-fires/behavior-doesn't shape at the single-seed level),
[[reach-grasp-lift-gap]], [[ppo-critic-divergence]] (Experiment 11's own
AR4-era failure mode, watched for and not observed here).
