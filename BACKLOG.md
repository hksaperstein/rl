# BACKLOG.md

Alternatives not chosen at a design/technical fork, recorded instead of
silently dropped — per `AUTONOMY.md`'s 2026-07-16 entry: at a fork with a
clear recommendation, take the recommendation and log the rest here
rather than pausing to ask. Not a priority-ordered queue; a durable
memory of roads not taken, in case one becomes relevant later.

Each entry: what was decided, what else was on the table, why it wasn't
picked, and where the decision happened (for context).

---

## Unified multi-die specialist-distillation experiment (2026-07-16)

- **`franka_checkpoint_review.py`'s reset-boundary height-measurement
  artifact** (found in Task 2, re-confirmed and traced exactly in Task
  3): not yet fixed at the source. Two prior tasks worked around it by
  independently re-deriving raw trajectories instead of trusting the
  summary JSON. A real fix (exclude the episode-reset frame from the
  `max()` window, or compute stats within a single episode only) would
  remove the need to re-derive this by hand on every future variant
  whose eval video spans a reset boundary. Deferred rather than fixed
  immediately because it wasn't blocking any decision — Task 3.5's own
  brief asks the implementer to use judgment on whether to fix it now
  that it's recurred a third time. **Correction (2026-07-17): this
  deferral was a mistake, not a considered call** — it cost 3 separate
  tasks independently re-deriving the same diagnosis. Fix directly next
  time this comes up, don't defer again.

## Cloud infrastructure reliability (2026-07-17)

Session hit repeated cloud friction: SPOT preemptions on 3/3 cloud tasks,
a pip-wheel-cache corruption from a preempted mid-install, and a systemd
`Linger=no` default killing detached tmux installs — both infra bugs
found during Task 3.5, neither yet folded into
`docs/cloud/franka-cloud-shakedown.md`. Highest-leverage fix not yet
built: **a pre-baked GCP VM image with Isaac Sim/Isaac Lab already
installed**, so cloud tasks skip the fragile ~15-20min from-scratch
install window entirely — this is also where both new infra bugs hit,
so shrinking that window reduces exposure to future variants of the
same class of failure, not just these two specific ones. Not built this
session because the user said stop; next cloud-heavy task should either
build this image first or explicitly accept the from-scratch-install
risk knowingly rather than by default.

## Task 3.5 execution backend: desktop dispatch instead of cloud (2026-07-18)

Task 3.5 (`docs/superpowers/plans/2026-07-16-unified-multi-die-specialist-distillation.md`)
specifies GCP cloud as its execution backend — written before the
Pi-as-primary-agent desktop-GPU-dispatch infra existed, and precisely
where this experiment already hit real infra friction (SPOT preemption,
pip-cache corruption, `Linger=no` killing a detached install — see the
entry above). Decision: dispatch Task 3.5's actual training runs to the
desktop (`scripts/run_on_desktop_gpu.sh`, per the new
`CLAUDE.md` "Pi-as-primary-agent GPU dispatch" desktop-first-cloud-
fallback routing, headless since this is an unattended dispatched job,
not an interactively-observed local session) instead of re-attempting
cloud provisioning — Isaac Lab is already installed on the desktop (no
15-20min from-scratch install window to fail mid-way), no SPOT
preemption risk since it isn't a cloud SPOT instance at all. Cloud
remains the documented fallback if `check_gpu_availability.sh` reports
the desktop unavailable when the task starts. Not chosen: sticking to
the plan's literal cloud-only instruction, or building the pre-baked
Isaac Lab VM image first per the entry above (still worth doing
eventually for whenever cloud genuinely is the only option, but not a
blocker for this specific task now that desktop dispatch exists).

## Task 3.5 cloud completion (d10-big/d12-big + d8-big re-eval, 2026-07-19)

- **`franka_checkpoint_review.py`'s `_detect_settle_step` tolerance
  (5e-5m range over a 15-step window) is tuned for a motionless
  table-rested object and never matches a *held* object's natural
  grasp-contact jitter.** Found on d12-big seed123's 4 genuinely-lifted
  envs — all 4 report `settle_step: -1` (no stable window ever found in
  249 steps) despite holding a stable plateau (±3-13mm) for 100+ steps,
  falling back to the pre-fix free-fall-window-min baseline instead of a
  directly-detected resting_z. Did not change this run's verdict (the
  fallback happened to be numerically close to the true table-rest
  height, cross-checked against the other envs' own directly-detected
  resting_z), so not fixed now — but not guaranteed harmless in general
  (a held object whose true grasp height is *itself* within the fallback
  window, or whose pre-grasp free-fall min differs meaningfully from the
  true table-rest height, could get a wrong baseline). Widening the
  tolerance (or adding a second, looser pass specifically for
  "held-and-jittering" detection distinct from "resting-and-still")
  would remove the need to reason about this by hand next time a run has
  envs with `settle_step: -1` that also show real gain.
- **Cloud infra gap, folded into `docs/cloud/dispatch-checklist.md`'s
  known-gaps list directly (not just here): a SPOT preemption can
  truncate a checkpoint file mid-write to 0 bytes**, and a naive
  "resume from highest iteration number" strategy will pick and fail to
  load it. Fixed in this task's own one-off orchestration script (skip
  any `model_*.pt` under 100KB when scanning for a resume candidate) but
  that script wasn't a committed, reusable one — worth promoting into a
  shared resume-helper if cloud SPOT training recurs often enough to
  justify it, rather than re-deriving this fix from scratch again.
- **Not chosen: staying on SPOT for the full 6-job batch after the 3rd
  preemption in ~3 hours** (a much higher preemption rate than this
  project's prior single/double-preemption cloud-shakedown history).
  Switched the last 2 jobs to on-demand provisioning instead — same
  instance-type/zone-search logic, ~2x the hourly rate, but eliminates
  further snapshot-recover-resume wall-clock loss for a small remaining
  job count. Not a general policy change (SPOT is still the plan's
  documented default and remains the right choice for larger batches
  where the cost multiplier matters more); a per-task judgment call
  worth reconsidering fresh each time preemption rate is unusually high.

## Task 4 scope decision: narrow to d12+d20, defer d8/d10 (2026-07-19)

Direct controller decision (Principal), made and executed rather than
surfaced as an open question: with Task 3.5's full grid in (d8 0/3, d10
0/3, d12 1/3 at undiluted 48mm), d8 and d10 are dropped from Task 4's
specialist set. Both are genuinely null across two independent size
regimes (real ~16-18mm size in Task 2, and 48mm parity in Task 3.5) —
that's a real shape-discoverability barrier, not a dilution or
absolute-scale artifact, and per this project's own systematic-debugging
precedent (3+ failed fixes on the same mechanism escalates rather than
invites a fourth speculative tweak), more reward-shaping attempts on
these two specifically aren't the next move without new mechanistic
insight. Logged here as an open, explicitly deferred research question
(not silently dropped, not blocking): why d8/d10 fail to ever discover
grasp while d12/d20 partially succeed has no obvious a priori answer yet
(no clean roundness/face-count story — d20 is the roundest shape and
succeeds most decisively) and would need its own dedicated investigation
if picked up later.

**Correction (2026-07-19), after dedicated research:**
`docs/superpowers/specs/research/2026-07-19-d8-d10-grasp-discoverability-literature.md`
found two things this entry got wrong. (1) The "3+ failed fixes on the same
mechanism" precedent cited above does not actually apply — direct source
read of every d8/d10 env cfg class plus the shared PPO/reward config
confirms only ONE recipe (unmodified reward function, unmodified PPO
hyperparameters, no curriculum/demonstration/exploration-tuning) was ever
tried against d8/d10, at two size regimes; that precedent's own origin is a
different task/mechanism (AR4-era sphere-lift) and citing it here to justify
deferral was a category error. (2) "No clean roundness/face-count story"
was itself wrong: this project's own already-computed Wadell sphericity
values (`tasks/franka/shape_observations.py`) are monotonic with discovery
rate across all four shapes at the 48mm-parity anchor (d8 0.8896/0-of-3,
d10 0.8959/0-of-3, d12 0.9286/1-of-3, d20 0.9524/2-of-3) — a real,
previously-unnoticed correlation in this project's own data (n=4, not proof
of a threshold, but a genuinely clean pattern, not "no story"). The research
doc proposes a ranked, falsifiable next step (demonstration-augmented
warm-start from this project's own already-proven scripted d8/d10 grasp in
`scripts/dice_pick_demo.py`, ranked above a geometry-ordered checkpoint
warm-start, ranked above exploration-noise retuning) — not yet spec'd or
executed; this entry's own scope narrowing to d12/d20 for Task 4 stands
unchanged, this is a correction to the *reasoning*, not a reversal of the
Task 4 decision itself.

**Before Task 4 itself: one gap-closing task, not in the original plan.**
Task 3's own d20 size-DR retry (0/120) is confounded by population
dilution (`random_choice=True` still assigns one size per env once,
diluting the 48mm sub-population ~5x) — never resolved, per that task's
own flagged ambiguity. Additionally, and decisively for Task 4 in
particular: asset-bisect's own working d20-at-48mm checkpoint
(`gs://rl-manipulation-hks-runs/asset-bisect/joint-die-big/seed123/2026-07-12_15-07-49/model_1499.pt`)
predates Task 1's shape-onehot/geometry-descriptor observation terms
(added 2026-07-16) — its policy network's observation space doesn't
match the unified schema Task 4's distillation needs, so it cannot be
used directly as a frozen teacher regardless of the dilution question.
Both problems share one fix: retrain d20 at a single undiluted 48mm
population WITH the Task 1 geometry-descriptor conditioning, mirroring
Task 3.5's own d8/d10/d12-big design exactly. This produces a
schema-compatible d20 specialist AND closes Task 3's open ambiguity in
the same run. Not chosen: proceeding to Task 4 with the old-schema
checkpoint anyway (architecturally broken) or retraining d8/d10 first
(deferred per above, not blocking this narrower 2-shape line).

Task 4 proceeds once this closes, using d12 (`seed123`,
`gs://rl-manipulation-hks-runs/unified-multi-die-specialists/joint-die-d12-big/seed123/2026-07-19_06-37-16/model_1499.pt`,
8/8 — corrected from an originally-reported 4/8, see `ROADMAP.md`'s
"Task 3.5 re-audit" entry, 2026-07-19) and the new d20-48mm-with-geometry
checkpoint as its two frozen specialists — a real, if narrower-than-
originally-planned, test of the GiGSL distillation mechanism itself.

## `franka_checkpoint_review.py`'s settle-detection flatness-window
heuristic may have undercounted true positives in already-reported
d8-big/d10-big/d12-big numbers (2026-07-19) — RE-AUDITED, CLOSED (2026-07-19)

Found while executing the d20-big-geom gate task above: the settle-
detection mechanism committed in `977a748` (a forward scan for the first
15-consecutive-step window under a fixed range tolerance, used to find
`resting_z` and separate the pre-grasp free-fall from the post-grasp
analysis window) is fundamentally unsuited to this experiment's fast,
decisive-grasp trajectories - it can silently lock onto a much later,
fully-static held plateau instead of the true early table-rest phase,
with no warning printed (a window *was* found, just the wrong one),
producing false-negative "no sustained lift" verdicts. This was directly
confirmed and fixed for the d20-big-geom gate task's own 3 seeds (see
`ROADMAP.md`'s "d20-big-geom gate task" entry, 2026-07-19, for the full
mechanism and fix - replaced with a MIN-over-a-fixed-early-window
approach in `scripts/franka_checkpoint_review.py`).

**Originally not chosen: re-auditing Task 3.5's own already-reported
d8-big/d10-big/d12-big grid (d8 0/3, d10 0/3, d12 1/3-partial) against the
fixed script.** Out of that task's explicit scope (dispatch brief said do
not attempt d8/d10, and that task's own remit was the d20 retrain, not a
Task 3.5 re-audit). Flagged as a real, non-trivial risk rather than
silently left alone: the old flatness-window approach was never observed
to work reliably for ANY env cfg checked so far in this whole experiment
(it failed for d20-big-geom in two successive ways, at both the original
5e-5m tolerance and a first-attempt-loosened 2e-3m tolerance) - it was
plausible, not just theoretical, that some of d8-big/d10-big's reported
0/8 nulls and/or d12-big's reported 4/8 (not 8/8) partial were themselves
undercounts of the same kind.

**Re-audit done (2026-07-19), closing this risk.** Pure offline
reanalysis (no new GPU rollout — confirmed the fixed settle-detection
logic needs only the already-downloaded raw `.npy` + existing summary
JSON's `episode_length_steps` field, both unaffected by the bug) against
all 9 already-synced (seed, shape) runs. Result: **d8-big/d10-big
confirmed 0/3 seeds each, with wide safety margin (max height gain
0.003-0.009m against the 0.04m threshold — not close calls). d12-big
seed123 corrected from 4/8 to 8/8** — the previously-uncredited 4 envs
were shown, via raw-trajectory shape analysis (matching rise-onset
timing across all 8 envs) plus direct video-frame confirmation, to be
genuine lifts the old algorithm's shared/poisoned `post_settle_start`
window and per-env plateau-mistaken-for-rest bug had hidden, not
artifacts. d12-big seed42/seed7 confirmed 0/8. Full mechanism, corrected
grid, and root-cause trace: `ROADMAP.md`'s "Task 3.5 re-audit" entry.
**Does not change `BACKLOG.md`'s own "Task 4 scope decision" above at
the shape-inclusion level** (d8/d10 still fully null, still deferred;
d12 still the only partial, still in scope) — it does strengthen d12's
own specialist-quality characterization (now a full-completeness echo of
d20's pattern, not a half one), relevant context for Task 4 given that
task already earmarks this exact d12 seed123 checkpoint as a frozen
teacher.

## Task 5 (real distillation run) BLOCKED — Isaac Lab cannot re-create a
`ManagerBasedRLEnv` in-process after `.close()`, in this installation
(2026-07-19)

Dispatched to run `scripts/distill_specialists.py` for real (no
`--dry-run`) against the two real Task 4 frozen teachers (d20
`joint-die-big/seed123`, d12 `joint-die-d12-big/seed123`), on the desktop
GPU (confirmed AVAILABLE via `check_gpu_availability.sh` at dispatch
time). **Two real, sequentially-discovered bugs, the second of which is
a genuine architectural blocker, not a fixable pipeline defect:**

**Bug 1 (fixed, verified): two simultaneous `ManagerBasedRLEnv`s.**
`scripts/distill_specialists.py`'s `main()` (real-run branch) originally
built BOTH teacher envs (`build_real_env("d20", ...)` and
`build_real_env("d12", ...)`) before starting the DAgger loop, exactly as
`tasks/franka/distillation.py`'s own module docstring describes ("Two
ROLLOUT environments run side by side, one per teacher's own single-shape
env"). This crashed immediately: `RuntimeError: Simulation context
already exists. Cannot create a new one.` — confirmed by direct source
read, `isaaclab/envs/manager_based_env.py`'s `ManagerBasedEnv.__init__`:
`SimulationContext` is a process-wide singleton, and Isaac Lab explicitly
refuses to construct a second `ManagerBasedRLEnv` while a first is still
open, outside extension mode. Not caught by Task 4's own `--dry-run`
(its stub envs have no notion of a simulation context at all). **Fixed**
by extracting `run_dagger_iteration`'s regression-step tail into a new
`regress_on_pooled_batches` helper (`tasks/franka/distillation.py`,
identical logic, unit-tested unchanged) and rewriting the real-run driver
to collect each shape's rollout **sequentially** — open shape A's env,
`collect_rollout`, `env.close()`, then shape B, then call
`regress_on_pooled_batches` on both already-collected (env-free) batches
— never holding two envs open at once. `run_dagger_iteration` itself is
untouched and still exercised by `--dry-run`/unit tests (their stub envs
have no simulation-context constraint, so simultaneous stub envs are
still the right test for that code path). Re-verified: 28/28 unit tests
still pass, `--dry-run`'s BC loss curve reproduces unchanged
(1.93→1.54→1.05 over 3 iterations).

**Bug 2 (NOT fixable within this task's scope — a real blocker):**
redispatching with the sequential-reopen fix, the run hung with ZERO log
output for 20+ minutes (log file byte-for-byte unchanged, main thread
pinned at ~104% CPU on a single core, all `carb.tasking*` worker threads
idle at 0%, GPU utilization 1%, no error) while constructing the
**second** `ManagerBasedRLEnv` of the run (the very first shape's env,
`d20`, built and closed in 8.5s with no issue — it's specifically the
env built *after* a prior `.close()` in the same process that never
completes). **Independently confirmed via a minimal, isolated repro**
(`num_envs=16`, no distillation logic at all, just `build → close →
build` on `FrankaDieLiftJointBigEnvCfg`): first env built in 1.44s,
closed in 0.22s; the second `ManagerBasedRLEnv(cfg=...)` call never
returned — still running, single-threaded, 9m11s CPU time, zero progress,
no output — after the diagnostic script's own env builder normally prints
a completion line within ~1-2s. This rules out "just slow at
num_envs=4096"; the hang is present and reproduces at trivial scale, so
it's inherent to constructing a second `ManagerBasedRLEnv` in-process
after a first one's `.close()`, in this specific Isaac Lab installation —
not something a bigger timeout would have ridden out. Both the real run
and the repro were killed (`kill -TERM`); full teardown independently
verified both times via `check_gpu_availability.sh` (AVAILABLE again),
`systemd-inhibit --list` (no `rl-gpu-job`/`rl-gpu-job-auto-detect` guard),
`nvidia-smi --query-compute-apps` (empty), and `lsof
/tmp/rl_isaac_sim.lock` (free) — no leftover desktop state.

**Why this is a blocker, not a bug for this task to fix unilaterally:**
this invalidates Task 4's own foundational design premise — "two rollout
environments run side by side" — under BOTH of its two possible readings
(simultaneous, or sequential-with-reopen) in this actual Isaac Lab
installation. The two remaining ways to get two teachers' rollout data in
one training loop are both genuine new engineering, not bug fixes, and
each has real tradeoffs:
  (a) **Two-process orchestration**: one persistent Isaac Sim process per
      teacher shape, each holding its own env open for the entire run,
      exchanging the current student weights + collected rollout
      observations every DAgger iteration via disk/IPC. Keeps Task 1's
      observation-schema contract (`object_shape_class_onehot` reading a
      single per-cfg `die_shape_class` constant) completely unchanged,
      but is real new distributed-training infrastructure (checkpoint
      I/O every iteration, process lifecycle/failure handling) with its
      own new failure modes.
  (b) **One mixed-population env**: split `num_envs` between d12 and d20
      within a SINGLE `ManagerBasedRLEnv`, via `MultiAssetSpawnerCfg(...,
      random_choice=False)` — an already-validated, already-used
      per-env-fixed-assignment mechanism in this exact codebase
      (`FrankaDieLiftJointMixedEnvCfg`'s own multi-*size* population,
      confirmed by direct source read to NOT have the unresolved
      per-episode-resampling risk that `random_choice=True` has). Simpler
      and reuses proven Isaac Lab mechanics, but **requires extending
      `object_shape_class_onehot`/`object_geometry_descriptor`**
      (`tasks/franka/mdp.py`) from their current "single per-cfg constant
      broadcast to every env" semantics to read each env's own actually-
      spawned asset identity — a real change to an established
      observation-term contract Task 1 built and every downstream task
      (2, 3, 3.5, 4) was told would be "consumed unchanged."

Both are legitimate, but substantively different, architectural choices
(new distributed infra vs. changing a public observation-term contract) —
outside a senior-engineer dispatch's own discretion per this repo's
"Do NOT make cross-cutting architectural decisions... changing a public
interface" boundary. Flagged to the controller rather than picked
unilaterally. **This senior engineer's own read, offered as input, not a
decision:** (b) looks more robust and lower-total-new-surface-area, since
it reuses an already-proven mechanism instead of building new
cross-process orchestration from scratch — but the observation-contract
change needs a real design pass (at minimum: how does a per-env-aware
shape observation read the spawned asset back off the USD stage/spawner
state cleanly, and does `object_geometry_descriptor` need the same
treatment), not a quick patch.

**State at handoff:** no distilled checkpoint exists yet. Task 4's
pipeline code (`scripts/distill_specialists.py`,
`tasks/franka/distillation.py`) is updated with the `regress_on_pooled_
batches` refactor + the (currently blocked, not working) sequential-
reopen real-run driver, both committed; the `franka_checkpoint_review.py`
`load_optimizer=False` fix (needed for the eventual eval step, found
while preparing for it) is committed separately. 28/28 Task 4 unit tests
and `--dry-run` still pass/reproduce unchanged. No cloud spend incurred
(desktop-only); wall-clock cost: two failed desktop dispatches, both
cleanly torn down, no wasted cloud budget.

**Controller decision (2026-07-19): (b), single mixed-population env.**
Reuses `FrankaDieLiftJointMixedEnvCfg`'s already-validated round-robin
mechanism instead of building new distributed multi-process
orchestration — smaller total new surface area, and this codebase
already trusts and understands the mechanism (Task 3's own d20 size-DR
work already confirmed via direct source read that `random_choice=False`
gives an exact, deterministic `index % len(assets_cfg)` per-env
assignment, not a runtime-only fact requiring new introspection). This
determinism is exactly what de-risks the observation-contract change
the senior engineer flagged as the real cost of (b): a per-env-aware
`object_shape_class_onehot`/`object_geometry_descriptor` does **not**
need to read anything back off the live USD stage/spawner state at
runtime — since the assignment is a pure, known-in-advance function of
env index and the assets list order/length, the observation functions
can just replicate that same `index % len(assets)` formula directly
(matching the exact mechanism `FrankaDieLiftJointMixedEnvCfg`'s own
docstring already documents and cites by source location). This turns
what looked like a real observation-contract redesign into a bounded,
mechanical extension: add a new `FrankaDieLiftJointD12D20MixedEnvCfg`
(2-shape round-robin, `assets_cfg=[d12_cfg, d20_cfg]`,
`scene.replicate_physics = False` per the existing class's own
documented gotcha) and make the two observation functions compute
per-env shape class from env index + a shapes-list ordering constant
instead of a single `env.cfg.die_shape_class` broadcast — every other
env cfg in the file keeps the old single-shape broadcast behavior
unchanged (this is additive, not a breaking change to Tasks 2/3/3.5's
already-shipped, already-verified single-shape env cfgs). Not chosen:
(a) two-process IPC orchestration — real distributed-training
infrastructure with its own new failure modes, not justified when (b)
reuses an already-proven mechanism instead.

**Implemented and run, 2026-07-19 — see `ROADMAP.md`'s "Task 5 ... BLOCKER
RESOLVED" entry and `kb/wiki/experiments/unified-multi-die-specialist-
distillation.md`'s matching Task 5 entry for the real result:** the
per-env indexing assumption above was independently re-verified against a
real live env (not just re-trusted from this entry's own source-read
citation) before relying on it, and held exactly.
`object_shape_class_onehot`/`object_geometry_descriptor`'s extension
turned out to be exactly the bounded, mechanical change predicted here —
no live USD/spawner-state introspection needed. Real outcome: distilled
policy 4/8 (d20) / 1/8 (d12) sustained lift vs. each specialist's own 8/8
— a real regression, not the null result this entry's own blocker made
impossible to observe before.

## d8/d10 demo-warmstart Task 0: d10 failed at perception, not grasp (2026-07-19)

d8 passed the 48mm-scale demo re-verification cleanly (242.6mm z-gain).
d10 failed — but at the vision detector (0 detections for class `d10` at
the enlarged 48mm scale, `select_target_detection`'s existing fail-loud
contract raised before the scripted grasp controller ever ran), not at
the grasp mechanism itself, which was never exercised. The implementing
task flagged this as a real per-shape split and deferred the "does d8
proceed alone" call to the controller rather than deciding it.

**Decision: fix d10's demo-capture path with a ground-truth bypass,
proceed with both shapes, not d8 alone.** This project already has
direct precedent for exactly this situation: the d4 rung-1 work hit an
identical detector-miss-at-an-off-distribution-condition failure and
built a ground-truth XY-bypass specifically to test the underlying
mechanism independent of the perception gap (`kb/wiki/experiments/`'s
d4 rung-1 entries, "extending rung 0's own GT-for-orientation isolation
precedent to position"). d10's 48mm-scale detector miss is the same
class of problem — an out-of-distribution scale the detector's training
data doesn't cover, not evidence about grasp discoverability, which is
the actual question this experiment investigates. Bypassing detection
for the demo-capture step only (not touching `dice_pick_demo.py`'s
production fail-loud contract, not attempting to fix the detector)
lets d10 proceed on equal footing with d8 rather than silently narrowing
this experiment's scope a second time without a substantive reason —
the multi-die experiment already narrowed once (d8/d10 dropped from
distillation) on real evidence; narrowing again here on an unrelated
perception gap, when a small precedented fix exists, would not be a
considered call. Not chosen: proceeding with d8 alone and deferring d10
indefinitely (would leave this experiment's own question about d10
specifically unanswered for no good reason) or attempting to actually
fix/retrain the detector for 48mm dice (real scope creep — a vision/
training-data problem, not what this experiment is testing).

**Result (2026-07-19, follow-up task): d10 grasp mechanism PASSES at 48mm
scale — 241.4mm z-gain (threshold 150mm), 0.8mm xy-drift, all 4 other
dice undisturbed (gain ≈0.0mm each)**, via `--gt-xy-bypass` added to
`scripts/_diag_d8d10_48mm_grasp_reverify.py` (commit `bcb0a27`), reusing
`dice_pick_demo.py`'s own `--gt-xy-bypass` mechanism (d4 rung-1
precedent) — full data: `outputs/dice_demo/diag_48mm_grasp_reverify/d10/
verdict_d10_48mm.json` (not committed, gitignored `outputs/`). d10's
grasp mechanism is confirmed sound at 48mm scale, same as d8's prior
242.6mm result; both shapes proceed to Task 1 on equal footing as this
entry's decision intended.

**A real bug was found and fixed in the same pass, and it changes this
entry's own premise**: `run_shape_reverify` called
`run_detector_subprocess` immediately after `override_die_scale`, but
`run_detector_subprocess` only reads whatever `rgb.png` already sits in
`out_dir` — it never re-renders. Those files were last written by
`spawn_scene_and_settle`, BEFORE the die was ever rescaled to 48mm. Every
prior run of this diagnostic's detector step (both the original d8 PASS
and the d10 FAIL that motivated this whole entry) therefore ran object
detection against a STALE REAL-SIZE frame, never a genuine 48mm-scale
image. Fixed via a new `recapture_camera_frame` (re-renders and
overwrites the camera frame from the live post-rescale scene before
detection runs). **With the bug fixed, the detector actually found d10 at
genuine 48mm scale in this same run** — `class=d10 conf=0.101
world_pos=(0.5416, 0.0460, 0.0056)`, only 3.7mm xy-error from ground
truth — contradicting the "0 detections... at the enlarged 48mm scale"
diagnosis this entry opened with. That original diagnosis was measured
against the wrong image; it is not evidence that d10 is
undetectable at 48mm scale, only that this diagnostic's own camera frame
was stale. One low-confidence (10%) single-seed detection is not strong
evidence either way about 48mm-scale detection reliability generally
(this repo's own documented render-nondeterminism/detection-fragility
precedent means one success at low confidence doesn't establish a stable
detection rate) — flagged as an open question, not re-investigated
further in this task (out of scope: this task's job was the grasp
mechanism, not re-characterizing detector reliability at 48mm).
**Follow-up not done in this task, flagged for whoever picks up Task
1**: d8's earlier "PASS" also never exercised true 48mm-scale detection
(same stale-frame bug) — its GRASP-mechanism verdict is unaffected (that
check reads physics state directly, not the camera frame), but a
regression re-run of d8 with the bug fix in place has not been done; low
stakes (Task 1's own capture will exercise 48mm-scale detection for real
regardless) but worth a cheap re-check if convenient.

## Clutter experiment Stage SO gate: confounded, fix is a partial-weight warm start (2026-07-19)

Stage SO (task 4 of the target-selection-clutter plan) failed its ≥7/8
sanity gate at 0/8 both shapes, trained fully from scratch (forced by
the 41→43-dim observation-schema extension having no existing
cross-dimensionality checkpoint-resume mechanism in this codebase). The
implementing task correctly flagged a real confound rather than
accepting the null at face value: no from-scratch (non-distilled) PPO
run of the d12/d20-mixed population has ever succeeded in this project
— every 8/8 result on this population came via distillation+fine-tune
(`unified-multi-die-specialist-distillation.md`'s Task 4-6), not raw PPO
from a random init. Stage SO's "reach but never grasp" failure pattern
may just be reconfirming that pre-existing cold-start difficulty, not
telling us anything about whether the new scene/observation code itself
broke something — which is the only thing the gate is supposed to
isolate.

**Decision: redo Stage SO with a partial-weight-transfer warm start
from the proven 8/8 checkpoint (`model_2998.pt`), not accept the
from-scratch result as final.** This is resolvable cleanly because of a
property already built into Task 2's own design: at Stage SO
(`active_distractor_count=0`), `distractor_distance_summary` is a
hard-zeroed *constant* — carries no information, no variance. A network
whose first layer is extended from 41→43 input columns by copying the
old 41 columns unchanged and randomly initializing only the 2 new
columns will produce **numerically identical output** to the original
checkpoint at Stage SO specifically, since those 2 new weight columns
are always multiplied by zero. This isolates exactly what the gate is
supposed to test (does the wiring/schema extension itself break
anything) from the separate, already-known cold-start question — no new
research needed, this is a mechanical weight-surgery fix. Needs a new
small script (load `model_2998.pt`'s state dict, extend the first
layer's input-weight matrix by 2 zero-or-random columns, save as a new
starting checkpoint), then resume Stage SO training from it via the
already-existing `--policy_only_checkpoint` path (no optimizer-state
compatibility needed, same mechanism already used for the distillation
experiment's own checkpoint-format transitions). Not chosen: accepting
the from-scratch 0/8 as the real Stage SO verdict (would conflate two
different questions and potentially block D1/D2 on a false negative), or
redesigning Stage SO to skip the gate entirely (the gate's intent is
sound, only the from-scratch execution of it was confounded).

**Implemented and run, 2026-07-19 — see
`kb/wiki/experiments/target-selection-clutter.md`'s "Task 4 corrected"
entry for the full result:** the new script
(`scripts/extend_checkpoint_observation_dims.py`) extends the checkpoint
exactly as predicted, and a `--verify` forward-pass check (run twice
against the real `model_2998.pt` — once locally on the Pi before any
cloud spend, once again on the cloud instance right before training)
confirmed byte-for-bit-identical (0.0 max abs diff) output at Stage SO's
zero condition, with a negative control confirming the check isn't
vacuous. Retrained 300 iterations (bounded budget, absolute target 3298 =
2998 preserved + 300 new) — Stage SO's gate now PASSES: d12 8/8, d20 7/8.
Confirms this entry's own predicted explanation was correct: the
scene/observation-schema wiring was never broken, only the original
from-scratch attempt's pre-existing cold-start difficulty made it look
that way. Cost ~$0.39 (cloud), full teardown verified. Stage D1/D2 (plan
Tasks 5/6) are unblocked.
