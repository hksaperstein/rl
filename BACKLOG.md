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
  that it's recurred a third time.
