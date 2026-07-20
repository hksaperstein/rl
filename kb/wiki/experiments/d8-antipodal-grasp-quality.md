# d8 antipodal/force-closure grasp-quality reward (2026-07-20, dual action-space test — H_joint COMPLETE, FALSIFIED)

**Status: H_joint (Condition A / joint-space) complete and reported here.
H_taskspace (Condition B / task-space-IK) is a separate task, run
independently and not gated on this result — this article's closing
5-row-outcome-matrix verdict is deferred until that lands too.**

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

## Related

[[experiment-09-antipodal-grasp-bonus]],
[[experiment-10-antipodal-threshold-action-scale-solver]],
[[experiment-11-taskspace-ik]] (the AR4-era arc this experiment tests
transfer of), [[grasp-mechanics-antipodal-vs-magnitude]],
[[action-space-design]], [[exploration-bonus-grasp-discovery]] (the
sibling SPLIT result on this identical env cfg), [[reach-grasp-lift-gap]].
