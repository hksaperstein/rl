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
