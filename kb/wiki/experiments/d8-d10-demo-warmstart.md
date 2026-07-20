# d8/d10 demonstration-augmented warm-start experiment (2026-07-19 -> 2026-07-20, H1 FALSIFIED both shapes, H2 PASSED both shapes — INVESTIGATION CLOSED)

**Overall closing verdict (both hypotheses resolved):** H1 (DAPG-style
BC-pretrain from a scripted demonstration) falsified for both d8 and d10.
H2 (geometry-ordered checkpoint warm-start from the converged d12
specialist) **passed for both shapes** — d8 3/3 seeds full 8/8 (24/24
envs, a clean sweep matching cube's own perfect record), d10 1/3 seeds
full 8/8 (8/24 envs, seed7 only — the same "0 or full-8/8-within-seed,
never partial" pattern this project's history has shown for d12/d20).
**d8/d10's grasp-discoverability problem is resolved**: it was never an
un-fixable structural barrier — a from-scratch PPO run with this
project's standard recipe cannot discover the grasp for these two shapes,
but PPO fine-tuning from a *different shape's* already-converged
policy weights can, cleanly and repeatably for d8, partially (matching
this project's own established discovery pattern) for d10. See "H2"
section below for full detail; H1's original section is unchanged
beneath it.

**H1 closing verdict:** H1 (DAPG-style behavior-cloning pretrain from a
scripted demonstration trajectory, followed by a full 1500-iteration PPO
fine-tune) is **falsified for both d8 and d10** — 0/8 sustained-lift
discovery in every one of 3 seeds per shape (0/24 total per shape), at
[[unified-multi-die-specialist-distillation]]'s own 48mm-parity anchor.
Independently re-derived from raw per-step height data (not just the
summary JSON) and confirmed by direct frame-by-frame video review for
both shapes — not a measurement artifact, not a marginal miss. No
never-before-observed "partial" (1/8-7/8) result was seen in any of the
6 runs.

**Goal:** test whether seeding PPO with a BC-pretrained student policy
(warm-started from one real scripted-grasp demonstration trajectory per
shape, captured via `dice_pick_demo.py`'s own DiffIK grasp controller at
48mm-parity scale) can unlock the grasp discovery that cold-start PPO
never found for d8/d10 in the unified-multi-die-specialist-distillation
experiment's own robust 0/24-both-shapes null. Spec:
`docs/superpowers/specs/2026-07-19-d8-d10-demo-warmstart-design.md`. Plan:
`docs/superpowers/plans/2026-07-19-d8-d10-demo-warmstart-implementation.md`.
H2 (checkpoint warm-start from the nearest-by-sphericity d12 specialist)
was pre-authorized as a per-shape fallback if H1 falsified — **not run in
this task**; falsification is reported back to the controller for a
decision on whether H2 proceeds.

## Tasks 0-2 (BC-pretrain pipeline build): complete, see plan doc

Re-verified the scripted grasp transfers to 48mm scale for both shapes
(Task 0), built `regress_on_paired_batches` +
`tasks/franka/demo_action_mapping.py`'s closed-form action-space mapping
+ `scripts/extract_demo_trajectory.py` (Task 1), and
`scripts/bc_pretrain_demo_warmstart.py` (Task 2) — BC-pretrain converged
cleanly for both shapes (final loss ≈0.0007-0.0009 after ~20 plateau
rounds) and a bounded PPO-handoff smoke test confirmed the resume
mechanics work. See the plan doc's own task history for the two real bugs
found and fixed along the way (an episode-length/replay-cap mismatch, a
device mismatch on `default_joint_pos`).

## Task 3 (the real H1 run): real infra friction, then a clean result

**Desktop went off-limits mid-session** (direct user decision, after the
Task 0-2 work above had already run there) — the captured demonstration
trajectories and BC-pretrained checkpoints from that earlier work were
unreachable, forcing this task to re-run the whole pipeline fresh on a
cloud instance. This surfaced real, previously-undiscovered gaps in this
project's cloud-dispatch path (dice-shape work had never actually been
cloud-dispatched before — every prior dice/die cloud-adjacent run either
used the asset-free `ik-cube` variant or ran on the desktop where these
assets already exist locally):

- **`vision/data/raw/dice_sets_v1/set_00013_*.usd`** (the demo/capture
  scene's 5-die visual mesh set — note the set NUMBER is `set_00013`, not
  `set_00000`; `set_00000` is a separate, older set only used by
  `dice_pick_demo.py`'s manifest JSON lookup, a real distinction the
  controller's own first attempt at supplying files got wrong and then
  corrected) and the vision detector's model weights
  (`vision/models/runs/s_plus_r/weights/best.pt`) are both gitignored,
  desktop-only, and were never in GCS. Resolved two ways: (1) the
  controller did a narrow, read-only, non-GPU `scp` of the 10 required
  `set_00013_*` USD+JSON files from the desktop to unblock the scene
  itself; (2) `scripts/extract_demo_trajectory.py` was fixed (commit
  `bdb31b2`) to skip `run_detector_subprocess` entirely under
  `--gt-xy-bypass` (previously called unconditionally even when its
  result was discarded) so the capture never needs the vision weights or
  `vision/.venv` on a fresh cloud instance at all — used for BOTH shapes
  on cloud, not just d10 (which already needed the bypass for a different
  reason, per Task 0's own detector-coverage finding).
- **`assets/shapes/notch_fixture.usd`** (the d4 rung-1 fingertip fixture,
  attached to both fingertips for every die type) is built by a small
  offline `pxr`-only script, `scripts/build_notch_fixture_asset.py` — its
  hardcoded `LD_LIBRARY_PATH`/extension-root paths assumed the desktop's
  from-source Isaac Lab install layout. Resolved by symlinking the cloud
  instance's own pip-installed `isaacsim/extscache/...` directories to the
  same absolute path the script expects (`/home/saps/isaacsim`), then
  running the script locally on the cloud instance — no GPU/Isaac Sim
  boot needed, ~1 second.
- **REAL BUG found and fixed in `scripts/franka_checkpoint_review.py`**
  (commit pending in this task): its own output filenames were derived
  from ONLY the checkpoint's basename (`model_1519`), with no path
  context. Since all 3 seeds of a shape in this H1 run trained to the
  identical `max_iterations` target, all 3 seeds' final checkpoints are
  literally named the same (`model_1519.pt` for d8, `model_1517.pt` for
  d10) in different timestamped run directories — the eval script
  silently overwrote each earlier seed's video/heights.npy/summary.json
  with the next seed's, with no warning. Fixed by folding the
  checkpoint's own parent-directory basename (the run's unique timestamp)
  into the output filename; all 6 evals were re-run after the fix and
  produced 6 genuinely distinct sets of artifacts.
- **A project-wide `GPUS_ALL_REGIONS=1` GCP quota** (shared across every
  concurrent Senior cloud workstream, not per-task) blocked this task's
  own instance provisioning for ~40 real minutes while a sibling
  workstream's job (`rl-franka-exploration-bonus`) held the single slot —
  resolved by real blocking-poll (not guessing), same finding
  independently reported by the concurrent target-selection-clutter
  experiment's own Task 4-6 writeup.
- **Severe SPOT preemption clustering**: 6 preemptions across this task's
  real GPU-active time (worse than this project's prior "1-2 per
  multi-hour run" experience, matching the precedent already documented
  in `docs/cloud/dispatch-checklist.md`'s "Known infra gaps" section).
  Recovered each time via checkpoint-resume (`--checkpoint <last saved> `
  without `--policy_only_checkpoint`, same absolute `--max_iterations`
  target) after validating the candidate checkpoint's file size. After
  the 6th preemption, switched the remaining runs to on-demand
  provisioning (following the same project precedent) via a
  snapshot-and-migrate cycle (SPOT capacity was available in
  `us-central1-c` when `us-central1-a` was fully stocked out even for
  SPOT) — zero further preemptions for the remaining d10 seeds/evals.
- A recurring, non-fatal **Isaac Sim Kit-shutdown teardown hang** (this
  project's own documented failure mode, CLAUDE.md) fired on almost every
  single real invocation this task ran (captures, BC-pretrain, PPO
  fine-tunes) — real work always finished and was written to disk first;
  a watchdog wrapper (`run_with_watchdog.sh`, waits for the script's own
  completion-print line, grace period, then `kill -TERM`/`-KILL`s the
  actual Kit python process) was built to stop burning ~10+ minutes per
  invocation waiting out each hang manually.

### Real result: 6/6 runs, 0/8 each

| shape | seed | envs_with_sustained_lift |
|-------|------|---------------------------|
| d8    | 42   | 0/8 |
| d8    | 123  | 0/8 |
| d8    | 7    | 0/8 |
| d10   | 42   | 0/8 |
| d10   | 123  | 0/8 |
| d10   | 7    | 0/8 |

**d8: 0/24 — H1 falsified. d10: 0/24 — H1 falsified.** Both shapes'
per-env `max_height_gain_m` was small (≈0.0088m for every d8 env,
≈0.0043-0.0054m for every d10 env — well under the 0.04m lift threshold)
and, notably, **essentially IDENTICAL across all 8 parallel envs within
each run** (per-step cross-env std ≈1e-9 during the settle window) — the
policy is not producing meaningfully different outcomes across randomized
per-env layouts, consistent with a policy that never learned any real
object-contingent grasp behavior from the single-demonstration BC warm
start, rather than a policy that grasps sometimes and not others.
Independently re-derived (a fresh reimplementation of the settle-window/
gain/sustained-lift logic, not a re-run of the eval script) from the raw
`heights_*.npy` for one seed per shape (d8 seed42, d10 seed42) and got
byte-identical 0/8 both times. Frame-by-frame video review (both of
those same seeds, rest frame + multiple mid/late-episode frames) shows
the gripper approaching and hovering near the die in both shapes but
never closing a real grasp — visually consistent with the null, not a
video/instrumentation mismatch (this project's own Experiment 16
precedent for why that check matters).

**Checkpoints:** `gs://rl-manipulation-hks-runs/d8-d10-demo-warmstart/joint-die-{d8,d10}-big/seed{42,123,7}/<timestamp>/model_{1519,1517}.pt`.
Eval artifacts (6 distinct videos + heights json/npy):
`gs://rl-manipulation-hks-runs/d8-d10-demo-warmstart/eval-artifacts/`.

**Cost:** ≈$4 total across both dispatch sessions (duration × published-
SKU-rate estimate, this project's standing methodology — no exact billing
export exists), well under the plan's $10 cap. Full teardown verified
after every provisioning cycle (`scripts/check_cloud_state.sh` clean).

## Bottom line

A single real demonstration trajectory's worth of BC-pretraining, warm-
starting an otherwise-unchanged full 1500-iteration PPO fine-tune, does
not unlock grasp discovery for either d8 or d10 at the 48mm-parity anchor
— H1 is cleanly, robustly falsified for both shapes, with no partial/
marginal signal in any of the 6 seed×shape runs. This does not by itself
distinguish between "the demonstration-based warm start mechanism doesn't
help this failure mode" and "one demonstration trajectory per shape is
too little signal to matter" (the original Task 2 design pooled 5
trajectories per shape; this run's own captures also produced 5 valid
per-shape trajectories and the BC-pretrain step did pool all 5 — the
single-trajectory caveat does NOT apply to this actual H1 result, unlike
an earlier interim report in this task's own history before the full
5-seed captures completed). H2 (checkpoint warm-start from the d12
specialist) remains pre-authorized as the next rung per the spec's own
fallback design, but was not started here — this is a stop-and-report
point back to the controller per the plan's own Task 3 Step 7 instruction.

See also: [[unified-multi-die-specialist-distillation]] (the null this
experiment was testing a fix for), [[target-selection-clutter]] (the
concurrent workstream that independently hit the same `GPUS_ALL_REGIONS`
quota constraint).

## H2 (pre-authorized fallback, run 2026-07-20 after H1's clean falsification): PASSED both shapes

Per the spec's "one fallback rung, no new spec" trigger condition (H1
falsified for a shape → H2 authorized for that shape, no new spec) — H1
falsified for both d8 and d10, so H2 ran for both.

**Design (per `docs/superpowers/specs/2026-07-19-d8-d10-demo-warmstart-design.md`'s
H2 section, executed exactly as specified, no new code):** direct
`scripts/train_franka.py --checkpoint <d12 checkpoint> --variant
joint-die-{d8,d10}-big --max_iterations 2999` resume — full
optimizer-state resume (no `--policy_only_checkpoint`), from the
already-converged, nearest-by-sphericity d12 specialist checkpoint
(`gs://rl-manipulation-hks-runs/unified-multi-die-specialists/joint-die-d12-big/seed123/2026-07-19_06-37-16/model_1499.pt`,
ψ=0.9286 vs. d8's ψ=0.8896/d10's ψ=0.8959 — closer to both than d20's
ψ=0.9524). `--max_iterations 2999` = checkpoint's own saved `iter=1499` +
a full 1500-iteration budget, matching `train_franka.py`'s documented
absolute-target resume arithmetic. Same 48mm-parity anchor, same reward/
PPO/observation schema as H1 and the whole unified-multi-die-specialist
arc — only the object asset and the warm-start source differ.

**Pre-flight checkpoint verification (done before committing to the real
6-run dispatch):** `gsutil stat` confirmed the checkpoint exists
(1,273,525 bytes, matches this project's own established ~1.27MB
real-checkpoint size). Loaded directly with `torch.load` (no Isaac Sim
needed for a shape check) and inspected `model_state_dict`:
`actor.0.weight` shape `(256, 41)` (41-dim observation input) and
`actor.6.weight` shape `(8, 64)` (8-dim action output) — an exact match
to d8/d10's own 41-dim observation / 8-dim action schema, confirming the
spec's "clean drop-in, no schema-adapter work needed" claim directly
rather than trusting it from the spec text alone. `optimizer_state_dict`
present and non-empty (real PPO optimizer state, as expected for a
non-`--policy_only_checkpoint` full resume) and `iter: 1499`.

**Execution: cloud-only** (desktop off-limits this session per direct
user instruction, and also unreachable). Followed
`docs/cloud/franka-cloud-shakedown.md`'s proven recipe on a fresh
`g2-standard-4`+L4 instance (`rl-d8d10-h2-checkpoint-warmstart`,
`us-central1-a`). All 6 training runs (3 seeds × 2 shapes) plus all 6
evals ran sequentially on one instance per the checklist's convention.

**Real infra friction, same category as H1's Task 3 (not new bugs, but
real operational cost) — reported honestly, not smoothed over:**
- **The single project-wide `GPUS_ALL_REGIONS=1` quota was contended
  twice by a concurrent sibling Senior workstream** (`rl-explbonus-diag`,
  a genuinely quick ~17min smoke test as expected, then
  `rl-exploration-bonus-d8-h1`, a full 3-seed training+eval batch that
  held the quota for ~1h20min the first time and reappeared for a second
  ~23min hold later) — both waits were real blocking polls (not guessed
  or worked around by touching the other workstream's resources),
  confirming this project's "only 1 cloud GPU quota project-wide" limit
  (CLAUDE.md) is a real, recurring cross-workstream friction point, not a
  one-off.
- **Two genuine SPOT preemptions** (confirmed via `gcloud compute
  operations list` `compute.instances.preempted` system events, not
  stockouts or manual stops) during the d8 seed42/seed123 runs. Both
  recovered via checkpoint-resume after validating the candidate
  checkpoint's file size (1,273,525 bytes, not a truncated 0-byte file —
  this project's own documented SPOT-truncation gap,
  `docs/cloud/dispatch-checklist.md`). After the second preemption,
  switched the instance's scheduling from SPOT to on-demand
  (`gcloud compute instances set-scheduling --no-preemptible
  --provisioning-model=STANDARD --clear-instance-termination-action
  --restart-on-failure` — **note: `--maintenance-policy=MIGRATE` fails
  for GPU-attached instances**, `onHostMaintenance` must stay
  `TERMINATE`, a real gotcha not previously documented in this project's
  cloud docs, now folded into `docs/cloud/dispatch-checklist.md`), per
  this project's own established "switch to on-demand after repeated
  preemption clustering" precedent. Zero further preemptions after the
  switch, matching that precedent's prior track record.
- Switching to on-demand did **not** bypass the `GPUS_ALL_REGIONS=1`
  quota (still hit once more, from the sibling workstream) — the quota is
  provisioning-model-independent; on-demand only protects against
  preemption once the instance actually has the GPU, not against
  quota contention for acquiring it in the first place. Worth recording
  explicitly since it would be an easy wrong assumption for a future
  dispatch.

**Bug-check (per dispatch instruction, not found — reported as checked,
not skipped):** confirmed `scripts/franka_checkpoint_review.py`'s
output-filename collision fix (commit `d5b9cd1`, folds the run
directory's own timestamp into output filenames) was already present in
this session's checked-out version before running any eval — verified by
`git log` on that file before dispatch. This task's own 6 evals hit
exactly the collision-prone scenario the fix addresses (all 3 seeds of a
shape landing on the identical `model_2998.pt` basename in different
timestamped run directories) and produced 6 genuinely distinct artifact
sets (`heights_joint-die-d8-big_<timestamp>_model_2998.{npy,json}` etc.),
confirming the fix holds under real re-use, not just the case it was
originally written for.

### Result: d8 clean 3/3 sweep, d10 1/3 (matching this project's established discovery pattern)

| shape | seed | envs_with_sustained_lift | max_gain range (lifted envs) |
|-------|------|---------------------------|-------------------------------|
| d8    | 42   | 8/8 | 0.246–0.253 m |
| d8    | 123  | 8/8 | 0.246–0.271 m |
| d8    | 7    | 8/8 | 0.470–0.530 m |
| d10   | 42   | 0/8 | (null, max ≈0.0043m) |
| d10   | 123  | 0/8 | (null, max ≈0.0043m) |
| d10   | 7    | 8/8 | 0.116–0.256 m |

**d8: 24/24 — H2 PASSED, clean 3-for-3 seed sweep** (a stronger result
than any other shape at this anchor to date, tying cube's own perfect
3/3 record and exceeding d12's 1/3 and d20's 2/3). **d10: 8/24 — H2
PASSED** (falsification bar was 0/24; seed7's full 8/8 is a real positive
per the spec's own "no spurious partial count has ever been observed in
this project's history" reasoning — and indeed seed7's within-seed result
is a clean 8/8, not a partial count, consistent with that history).
Neither shape falsified.

**Independently re-derived from raw `heights_*.npy`** for one seed per
shape plus a null spot-check (d8 seed42, d10 seed7, d10 seed42), via a
from-scratch reimplementation of the resting-z/gain/sustained-lift logic
(not reusing `franka_checkpoint_review.py`'s own code), per this arc's
established rigor discipline. **Self-caught methodology bug in the course
of doing this**, reported rather than silently corrected: the first pass
of the independent reimplementation computed `max_z`/`max_gain` over the
*entire* first-episode window (steps 0–248), which includes the object's
initial spawn-drop transient (steps 0–~20, before it settles onto the
table) — this produced a spuriously elevated `max_gain` for d10 seed42
(≈0.0375m, just under the 0.04m threshold, vs. the tool's own correctly-computed
≈0.0043m) purely from the drop-in motion, not any lift. Fixed by
restricting the lift-analysis window to start at `post_settle_start_step`
(10) as the production tool's own summary JSON already documents it
does, matching `franka_checkpoint_review.py`'s number exactly afterward
(byte-for-byte identical `envs_with_sustained_lift` count for all three
re-derived files: 8/8, 8/8, 0/8). This is not a bug in the production
tool — it already handles this correctly — but a real reminder of how
easy this exact class of measurement error is to reintroduce, consistent
with this project's own repeated settle-detection-bug history.

**Video-verified for one seed per shape plus the d10 null, frame-by-frame
(rest / mid-episode / peak-height frames extracted via `ffmpeg`, viewed
directly):** d8 seed42 and d10 seed7 (the discovering seed) both show the
gripper open and hovering near the die at the rest frame, then visibly
closed and raised well above the table by the mid/peak frames — a real
posture change consistent with a genuine grasp-lift, not a height-number
artifact. d10 seed42 (null) shows the die still resting on the table
surface at the mid-episode frame with the gripper open and hovering,
never engaging — visually consistent with the instrumented 0/8, matching
the same "arm never descends to grasp" pattern H1's own null videos
showed (Experiment 16 precedent for why this check matters).

**Cost:** ≈$3.44 total (duration × published on-demand/SPOT SKU rates —
this project's standing methodology, no exact billing export exists).
Breakdown: ~50min SPOT (~$0.30) + ~2h23min SPOT (~$0.86) + ~3h01min
on-demand after the preemption-driven switch (~$2.13, on-demand L4 rate
$0.560/GPU-hr + G2 core/RAM confirmed via the live Cloud Billing Catalog
API, not assumed) + ~$0.15 boot-disk-hours across the full instance
lifetime. Well under the plan's $10 cap (cumulative with H1's own ≈$4,
total experiment spend ≈$7.44, still under the plan's cap which covers
Tasks 1-4 combined). Full teardown verified
(`scripts/check_cloud_state.sh` clean: zero instances/disks/snapshots).

**Checkpoints:** `gs://rl-manipulation-hks-runs/d8-d10-h2-checkpoint-warmstart/joint-die-{d8,d10}-big/seed{42,123,7}/<timestamp>/model_2998.pt`.
Eval artifacts (6 videos + heights npy/json):
`gs://rl-manipulation-hks-runs/d8-d10-h2-checkpoint-warmstart/eval-artifacts/`.

### Bottom line

**The d8/d10 grasp-discoverability investigation is closed, with a real
positive resolution, not a third null.** From-scratch PPO with this
project's standard recipe cannot discover the d8/d10 grasp (Task 3.5's
original 0/24-both-shapes null, H1's demonstration-BC-warm-start 0/24-both-shapes
null) — but PPO fine-tuned from a *different, geometrically-nearest
shape's* already-converged policy weights can, cleanly for d8 (3/3 seeds)
and partially for d10 (1/3 seeds, matching this project's own established
discovery-rate pattern for harder shapes). This is genuine evidence that
d8/d10's null was a policy-initialization/exploration problem specific to
learning from scratch, not an intrinsic physical or reward-design
barrier — the same underlying reward function, PPO hyperparameters, and
observation schema that never discovered the grasp cold now recovers it
reliably once seeded from nearby-shape weights. `BACKLOG.md`'s "Task 4
scope decision" entry (which deferred d8/d10 from the original
specialist-distillation arc on the original from-scratch null) can now be
revisited with this positive result in hand — a decision for Principal,
not made here, but the evidentiary blocker cited there ("genuinely,
robustly null... more reward-shaping attempts... aren't the next move")
no longer holds unmodified: a working, non-reward-shaping fix exists.
