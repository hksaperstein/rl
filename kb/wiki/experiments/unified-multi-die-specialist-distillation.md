# Unified multi-die specialist-distillation experiment (started 2026-07-16)

**Goal:** one RL policy that grasps-and-lifts a commanded die among
{d8, d10, d12, d20} (d4 out of scope), by training a per-shape specialist
for each die then distilling them into one policy (UniDexGrasp++'s GiGSL
pattern). Spec:
`docs/superpowers/specs/2026-07-16-unified-multi-die-specialist-distillation-design.md`.
Plan:
`docs/superpowers/plans/2026-07-16-unified-multi-die-specialist-distillation.md`.
Gated on [[reward-hacking-and-sparse-discoverability]]-adjacent research
(DexSinGrasp, arXiv:2504.04516) that uncurriculated multi-object clutter
can collapse RL discovery — distractors/target-selection are explicitly
deferred to a follow-on experiment; every env in this plan spawns exactly
one die.

## Tasks 0-1: assets + observations (complete, reviewed clean)

Baked real-standard-size physics assets for d8/d10/d12 (16.0mm, 16.0mm,
18.0mm face-to-face) alongside the existing d20 (per
[[size-curriculum]]'s standard-vs-jumbo correction). Added a shape-class
one-hot + Wadell-sphericity geometry-descriptor observation term so a
single policy can eventually condition its behavior on which die it's
holding.

## Task 2: train d8/d10/d12 specialists at their real small sizes — 0/9

Trained each of d8/d10/d12 at their own real ~16-18mm size. Result:
**zero discovery, 0/9** (3 shapes x 3 seeds) — worse than the d20's own
1/3-at-48mm [[asset-bisect]] baseline. Independently re-derived the raw
per-step height data (not just the summary JSON) and found the object is
completely motionless for the entire eval in every seed — a *stronger*
null than the initial report suggested. Left an open confound: these
three shapes were never tested at [[asset-bisect]]'s own 48mm-parity
anchor, so this result can't yet be attributed to shape difficulty vs.
these objects simply being too small at the Franka gripper's absolute
scale. (This confound is what Task 3.5 was inserted to resolve — see
below.)

## Task 3: d20 size-DR + geometry-feature retry — 0/120, gate before Task 4

Retried the d20 with `FrankaDieLiftJointRandomSizeEnvCfg`
(`MultiAssetSpawnerCfg(random_choice=True)`, spanning 22-48mm, 5 discrete
sizes) plus Task 1's new geometry-descriptor conditioning — the one
mechanistic difference from the already-falsified
[[size-curriculum]]-era `FrankaDieLiftJointMixedEnvCfg`, which got 0/3.

**Result: 0/120** (3 seeds x 5 scales x 8 envs/eval), confirmed genuinely
motionless (not "attempted but unsustained") by independently re-deriving
raw per-step height data and watching eval video frame-by-frame. This
matches, not corrects, the plan doc's preliminary "0/120" figure.

**Ambiguous verdict, flagged rather than resolved:** `random_choice=True`
still assigns one size per env once at scene-spawn (same mechanism as the
falsified `MixedEnvCfg`, confirmed by direct source read), so Task 3's
48mm arm is itself a diluted ~1/5 sub-population — reproducing the
`MixedEnvCfg`'s 0/3 floor there is *consistent with* population dilution
being the real confound, but doesn't rule out shape/discoverability
remaining a barrier independent of dilution, because Task 3 never paired
an undiluted single-48mm d20 population with the geometry-descriptor
conditioning. Task 3.5's undiluted-48mm design (below) was scoped to
d8/d10/d12 only, so this specific ambiguity was still open for the d20
case as of Task 3.5 — **closed 2026-07-19 by the "d20-big-geom gate"
task below**, once it became decision-relevant (Task 4 needed a
schema-compatible d20 checkpoint regardless). Full grid and reasoning:
`ROADMAP.md`'s Task 3 entry (search "0/120").

## Task 3.5: 48mm-parity check for d8/d10/d12 — complete 2026-07-19, one partial positive

Neither Task 2 nor Task 3 ever tested d8/d10/d12 at a single, undiluted,
48mm population the way the original [[asset-bisect]] ladder did for the
cube (3/3) and d20 (1/3). This task closed that gap: three new env cfg
classes (`FrankaDieLiftJointD8BigEnvCfg`/`D10Big`/`D12Big`), each shape's
own freshly-derived 48.0mm-targeting scale (native mesh bboxes differ per
shape, so the d20's own 0.001585 constant does not transfer), single
undiluted 48mm population, 3 seeds x 3 shapes, 1500 iterations each.
d8-big trained/evaluated by a prior desktop-dispatch agent
([[pi-as-primary-agent-gpu-dispatch]]); d10-big/d12-big (6 seeds) trained
on GCP cloud (SPOT, switched to on-demand after 3 preemptions in ~3hrs —
see `BACKLOG.md`'s "Task 3.5 cloud completion" entry).

**Full grid (envs with sustained lift / 8 per seed) — AS ORIGINALLY
MEASURED, since corrected, see "Task 3.5 re-audit" section below:**

| shape (48mm) | seed 42 | seed 123 | seed 7 | seeds-with-discovery |
|--------------|---------|----------|--------|-----------------------|
| d8-big       | 0/8     | 0/8      | 0/8    | 0/3                   |
| d10-big      | 0/8     | 0/8      | 0/8    | 0/3                   |
| d12-big      | 0/8     | ~~4/8~~ **8/8** | 0/8 | 1/3                  |

Compared to [[asset-bisect]]'s own undiluted-48mm baselines (cube 3/3
seeds full 8/8, d20 1/3 seeds full 8/8): **d8 and d10 remain completely
null even at pure 48mm parity — shape itself is a real barrier for these
two, not population dilution or absolute scale** (matches Task 2's
original ~16-18mm finding). d12-big seed123 was originally read as a
*weaker* echo of d20's own 1/3-seed pattern (4/8, half the envs within
its lucky seed) — the re-audit below found this was a measurement
artifact of the same settle-detection bug fixed in the d20-big-geom gate
task; the corrected reading is a *matching* echo, full 8/8 like d20's
own lucky seeds.

Independently re-verified d12-big seed123's positive result is real, not
a third occurrence of the reset-boundary/settle-window artifacts already
fixed twice in this experiment: re-implemented the settle/gain logic from
scratch against the raw `.npy` and found a smooth continuous rise
(steps ~40-115) to a stable plateau (~0.20-0.23m absolute height, inside
`lift_env_cfg.py`'s own goal-z range `(0.25, 0.5)`) held for the rest of
the episode — physically consistent with a real grasp-lift-carry, not a
contact-explosion glitch (no violent single-step jumps). Could not
directly visually confirm via video this time — `franka_checkpoint_review.py`'s
camera is fixed on env_0, which happened to be one of the *non*-lifting
envs in this run; the verdict rests on the raw-trajectory physics
reasoning, disclosed as a real tooling limitation rather than papered
over. Found (but did not need to fix) a related measurement caveat: the
settle-detector's tolerance is too tight to ever recognize a *held*
object's natural jitter as "settled" — logged to `BACKLOG.md` for a
future pass.

Also found and fixed during this task: d8-big seed42/seed123's synced
eval artifacts actually predated the `977a748` measurement fix (verified
via GCS object timestamps), contradicting the task's own dispatch brief
— re-ran eval-only (no retrain) against the current fixed script,
reconfirmed 0/8 unchanged. Also found a new cloud-infra bug (SPOT
preemption truncating a checkpoint file mid-write to 0 bytes) and fixed
the resume logic to validate checkpoint size before trusting it — see
`docs/cloud/dispatch-checklist.md`'s known-gaps list.

Full grid, reasoning, and cost: `ROADMAP.md`'s Task 3.5 entry (search
"48mm-parity check").

## Task 3.5 re-audit against fixed settle-detection — complete 2026-07-19, d12-big seed123 corrected 4/8 → 8/8

The d20-big-geom gate task below fixed a real settle-detection bug in
`franka_checkpoint_review.py` that predates all of Task 3.5's own
d8-big/d10-big/d12-big runs. This task closed the resulting open risk
(`BACKLOG.md`'s "settle-detection ... may have undercounted true
positives" entry) via pure offline reanalysis — confirmed first that
everything the fixed method needs (the raw `heights_*.npy` per-step
array, plus `episode_length_steps` from the existing summary JSON) was
already downloaded/synced and unaffected by the bug, so no new GPU
rollout was needed.

**Result: 8 of 9 (seed, shape) cells unchanged, all with wide safety
margin (max height gain 0.003-0.009m vs. the 0.04m threshold — not close
calls).** d8-big/d10-big confirmed 0/3 seeds each; d12-big seed42/seed7
confirmed 0/8. **d12-big seed123 corrected from 4/8 to 8/8.** Root
cause, read directly off the old per-env JSON fields: the flatness scan
mistook 3 of the 4 previously-uncredited envs' own held-elevated
plateau for their *resting* state (e.g. one env's reported
`resting_z_m=0.2415` was actually its held height, not the true
~0.0175m table-rest), so their real lift-and-hold registered as
"already resting, zero gain"; the 4th env had an approximately-correct
individually-detected resting_z but the old code's single *shared*
`post_settle_start` (driven by the other envs' bad late detections)
excluded its entire rise-and-fall event from the analysis window.

All 4 newly-corrected envs confirmed as genuine physical lifts, not
noise: all 8 envs in this seed show the same smooth rise starting within
a 5-step window of each other (steps 38-43) with no teleports. 7 hold
their elevated position to end-of-episode (matching the already-
confirmed pattern for the originally-credited 4); the 8th ("env 0") is a
genuine **lift-then-drop** — rises to ~0.111m, holds long enough to clear
the 25-consecutive-step sustained-lift bar (37 steps), then smoothly
descends back to true table-rest by ~step 115. Directly video-confirmed
(env_0 is the one env this script's camera shows): frame at step 65
shows a small elevated white sphere near the gripper matching the
~0.111m reading; frame at step 95 shows it back near table level,
matching the trajectory's descent.

**Does not change which shapes show discovery** — d8/d10 remain fully
null (`BACKLOG.md`'s "Task 4 scope decision" to defer them is
unaffected at the shape-inclusion level), d12 remains the only partial
(still 1/3 seeds). What changes is d12's own specialist-quality
characterization: no longer a "weaker echo" of d20's pattern, now a
full-completeness match — relevant since Task 4 already earmarks this
exact d12 seed123 checkpoint as a frozen teacher. Full mechanism,
corrected grid, and root-cause trace: `ROADMAP.md`'s "Task 3.5 re-audit"
entry.

## d20-big-geom gate task: undiluted-48mm d20 retrain — complete 2026-07-19, result STRONGER than expected

Closed Task 3's dilution ambiguity for d20 specifically (see BACKLOG.md's
"Task 4 scope decision" entry, 2026-07-19): retrained d20 at a single
undiluted 48mm population WITH Task 1's geometry-descriptor conditioning,
3 seeds (42/123/7), 1500 iterations, on GCP cloud (SPOT, zero
preemptions). **No new env cfg class was needed** — direct source read
confirmed the existing `FrankaDieLiftJointBigEnvCfg`/`--variant
joint-die-big` (the asset-bisect rung-2 class) already includes Task 1's
observation terms by inheritance (they were added unconditionally to the
shared base `ObservationsCfg` by commit `ec32bb0`, and `die_shape_class =
"d20"` was already set explicitly in the base `FrankaDieLiftJointEnvCfg`).
The BACKLOG entry's caveat was about the old pre-Task-1 *checkpoint*, not
the env cfg *class*.

**Result (envs with sustained lift / 8 per seed): seed42 0/8, seed123
8/8, seed7 8/8 — 2/3 seeds, both at full within-seed completeness.**
Stronger than the falsifiable expectation going in (~1/3 seeds, likely
seed123 only, matching the asset-bisect ladder's original 1/3-at-8/8
baseline) — reported as observed, not adjusted to fit. Supports "dilution
was Task 3's real confound" more decisively than the weaker form of that
hypothesis would have required.

**A real measurement bug in `franka_checkpoint_review.py` was found and
fixed during this task's own raw-trajectory verification step** (not
optional per this experiment's verification standard for positive
results): the settle-detection flatness-window heuristic from `977a748`
initially reported seed123 at only 1/8 and seed7 at 0/8 — direct
inspection of the raw `.npy` showed this was wrong, every env in both
seeds shows a clean, smooth, continuous grasp-lift-carry-to-goal
trajectory the flatness heuristic was silently mis-measuring (locking
onto a late, fully-static held plateau instead of the true early
table-rest phase, for both the original 5e-5m tolerance and a
first-attempt-loosened 2e-3m tolerance). Fixed by replacing it with a
simpler, more robust MIN-over-a-fixed-early-window approach (physically
grounded: a grasp-driven ascent only moves the object up from its rest
height, so MIN can't be fooled by where in the window the low point
falls). Video-confirmed too (arm reaches down ~step 20, fully extended
and holding by ~step 90-150, matching the raw height data for both
positive seeds).

**Open follow-up, logged to `BACKLOG.md`:** the old flatness-window
approach was used, unquestioned, for Task 3.5's own already-reported
d8-big/d10-big/d12-big grid above — not re-audited here (out of this
task's scope), but plausible that some of those numbers (d8/d10's 0/8s,
d12's 4/8-not-8/8) are themselves undercounts of the same kind, given the
old approach was never observed to work reliably for any env cfg checked
so far. **Closed 2026-07-19 by the "Task 3.5 re-audit" section above:**
d12-big seed123 was indeed undercounted (corrected to 8/8); all other 8
cells confirmed unchanged with wide margin.

**Checkpoints for Task 4** (both fully valid; seed123 nominal default per
this project's recurring "seed123 is the lucky seed" pattern, seed7 an
equally-valid alternate given identical 8/8):
- `gs://rl-manipulation-hks-runs/unified-multi-die-specialists/joint-die-big/seed123/2026-07-19_12-46-42/model_1499.pt`
- `gs://rl-manipulation-hks-runs/unified-multi-die-specialists/joint-die-big/seed7/2026-07-19_13-17-02/model_1499.pt`

Full grid, mechanism, raw-trajectory numbers, and cost (~$0.91):
`ROADMAP.md`'s "d20-big-geom gate task" entry.

## Open, not yet decided

Task 4 (distillation) status per this experiment's own gate discipline:
d8 and d10 remain fully null at 48mm parity (2/4 candidate shapes,
deferred per BACKLOG.md's "Task 4 scope decision", not blocking), d12
shows one partial positive (1/3 seeds, now full 8/8-within-seed after
the re-audit correction above), **d20 is now resolved (2/3 seeds, full
8/8 each)** — the d20-big-geom gate task above closed the last blocker.
Task 4 proceeds with d12 (seed123, 8/8 — corrected from an originally-
reported 4/8) and d20 (seed123 or seed7, both 8/8) as its two frozen
specialists, per BACKLOG.md's "Task 4 scope decision" entry — an explicit
controller-level decision already made there, not re-litigated here.

## Task 4: distillation pipeline built (2026-07-19) — pipeline only, no real training yet

Built `scripts/distill_specialists.py` (thin CLI entry point) +
`tasks/franka/distillation.py` (the actual pure-torch/rsl_rl mechanics,
importable without Isaac Sim) + `tests/test_distillation_data_collection.py`
(28 unit tests, all passing). This is pipeline construction only — no real
distillation training ran; that's Task 5, a separate later cloud dispatch.

**Checkpoint verification (before designing around them):** both frozen
teachers — d20 (`joint-die-big/seed123/2026-07-19_12-46-42/model_1499.pt`)
and d12 (`joint-die-d12-big/seed123/2026-07-19_06-37-16/model_1499.pt`) —
confirmed via `gsutil stat` + a real load through `rsl_rl.modules.ActorCritic`
(not just a shape inspection): both carry an identical 41-dim observation
(joint_pos_rel 9 + joint_vel_rel 9 + object_position 3 +
target_object_position 7 + last_action 8 + shape_class_onehot 4 +
geometry_descriptor 1 = 41) and 8-dim action (7 arm joints + 1 gripper),
matching `FrankaLiftPPORunnerCfg`'s architecture ([256,128,64] hidden dims,
elu) exactly — fully shape-compatible, confirming a single unified-policy
architecture works unchanged for both teachers and the student.

**Imitation-loss choice: multi-teacher DAgger with per-state expert
routing + MSE-on-mean regression** (Ross/Gordon/Bagnell 2011 DAgger,
generalized to 2 frozen teachers routed by the observation's own
shape-onehot feature — full design rationale, including why `rsl_rl`'s own
built-in single-teacher `StudentTeacher`/`DistillationRunner` classes were
NOT reused, is in `tasks/franka/distillation.py`'s own module docstring).
Mechanism: roll a beta-mixture (student/teacher) policy in each teacher's
own single-shape env, relabel every visited state with ITS OWN shape's
teacher (reading the shape-onehot slice already in the observation, so
routing works correctly even on a pooled/shuffled multi-shape batch), pool
+shuffle both shapes' data before every BC regression step.

**"Shape-randomized-per-episode" design note:** rather than building a new
live env cfg that resamples shape at each individual episode reset (Isaac
Lab's `MultiAssetSpawnerCfg` per-episode-resampling semantics are an
unresolved risk flagged elsewhere in this plan, Task 3's own docstring),
the pipeline runs the two teachers' existing single-shape envs
(`FrankaDieLiftJointBigEnvCfg`, `FrankaDieLiftJointD12BigEnvCfg`) side by
side and pools+shuffles their visited-state streams before every gradient
step — the same statistical training distribution (shape varies
episode-to-episode across the pooled stream) without touching that
unresolved spawner question. Flagged explicitly as a scope choice, not an
oversight.

**Verification done (this task's own bar — no training run to video yet,
so mechanical smoke-test + unit tests are the right bar):**
- 28/28 unit tests pass (`tests/test_distillation_data_collection.py`,
  stub envs/policies, no Isaac Sim, no real checkpoints) — TDD discipline
  followed: confirmed failing (`ImportError`) against an emptied
  `distillation.py` before implementing, then passing after.
- `--help` runs clean with no Isaac Sim launch.
- `--dry-run` runs the FULL real pipeline end to end (real checkpoint
  download + load + shape-check, real student `ActorCritic`, only the env
  is a physics-free stub) — BC loss visibly decreases across iterations
  (1.93 → 1.54 → 1.05 over 3 dry-run iterations), and the saved checkpoint
  round-trips (`torch.load` reproduces the expected keys/shapes).
- Full repo test suite (`tests/`, 156 tests) still green — no regressions.

**Two real bugs found and fixed while verifying `--dry-run`** (bug-handling
discipline — fixed and re-verified in this same task, not deferred):
1. Invoking `gcloud storage cp` via `subprocess` from a process launched
   through Isaac Sim's bundled Python inherits that Python's own
   `PYTHONPATH`, which crashes `gcloud`'s own Python invocation
   (`AssertionError: SRE module mismatch`) before it ever runs. Fixed by
   stripping `PYTHONPATH`/`PYTHONHOME` from the child `gcloud` subprocess's
   environment (`tasks/franka/distillation.py`'s
   `_subprocess_env_for_gcloud`).
2. The `--dry-run` synthetic stub env originally sliced `actions` to the
   observation's own non-shape-onehot width, silently assuming
   `num_actions == obs_dim - num_shapes` — true only by coincidence for a
   toy schema, false for the real 41-dim-obs/8-dim-action schema (39 != 8),
   crashing on the first real dry-run. Fixed by broadcasting a per-env
   scalar action summary (`actions.mean(dim=-1)`) across the stub's state
   instead — dimension-agnostic, and a related second issue (the stub's
   own onehot block placement didn't match `REFERENCE_SHAPE_ONEHOT_START`,
   silently misrouting every state to the wrong teacher) was fixed by
   deriving the stub's shape-onehot offset from its own actual layout
   rather than the real env's reference constant.

Files: `scripts/distill_specialists.py`, `tasks/franka/distillation.py`,
`tests/test_distillation_data_collection.py`. Next: Task 5 (separate,
later cloud-GPU dispatch) runs the real distillation training against
these two real checkpoints and reports real discovery-rate numbers —
nothing in this task constitutes a result yet.
