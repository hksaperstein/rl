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
