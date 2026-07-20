# d8/d10 demonstration-augmented warm-start experiment (2026-07-19 -> 2026-07-20, H1 COMPLETE — FALSIFIED both shapes)

**Closing verdict:** H1 (DAPG-style behavior-cloning pretrain from a
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
