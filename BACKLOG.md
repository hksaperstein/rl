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
