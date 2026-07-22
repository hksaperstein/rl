# BACKLOG.md

A queue of concrete future-work items not already on `ROADMAP.md` —
candidate experiments, infra improvements, and deferred ideas worth
reconsidering. Not priority-ordered (see `ROADMAP.md`'s "Planned / near-term
priorities" for real ordering of what's next). Where an item originated from
a past design fork or investigation, the full "why" lives in `kb/wiki/` or
`docs/superpowers/`; this file states the item plainly with at most a
one-line pointer, not the reasoning inline.

---

## Perception / vision

- **Wrist-mounted camera instead of the current fixed world-frame mount.**
  Better close-up/occlusion handling for fine manipulation, but touches the
  `vision/` detector's training-data assumptions and the perception
  pipeline structurally — closer to a Tier-1 change than a quick
  investigation. (Raised 2026-07-20 as an alternative to the AR4-vs-Franka
  workstream; not started.)
- **"Live video feed" integration — needs scoping with the user before
  starting.** No physical hardware exists in this project (sim-only); this
  most likely means a real-time per-step detection loop instead of the
  current batch render-then-detect pipeline (`dice_pick_demo.py`). Confirm
  that reading is correct before treating it as scoped.
- **Re-run d8's 48mm-scale demo-capture regression check with the
  stale-camera-frame bug fixed.** Never done — d8's original 48mm PASS was
  measured against a stale (pre-rescale) camera frame, same bug d10 hit;
  d8's grasp-mechanism verdict itself is unaffected (reads physics state
  directly), so this is low-stakes, but cheap to verify.
  `kb/wiki/experiments/d8-d10-demo-warmstart.md`.

## Task-scope expansion

- **New dice-specific tasks (rolling, sorting).** Real scope expansion
  beyond the current single-object pick-and-place phase (`CLAUDE.md`'s own
  scope-discipline note: multi-object/multi-task generalization is a real,
  intended future phase, but comes after single-object pick-and-place is
  solved) — a deliberate go/no-go call, not a quick add. (Raised
  2026-07-20.)

## `franka_checkpoint_review.py` measurement gaps

- **Fix the reset-boundary height-measurement artifact at the source.**
  `max()` over a multi-episode recording window catches a spurious height
  spike at an episode-reset boundary, not real policy-driven lift. Worked
  around three separate times (Tasks 2, 3, 3.5) by independently
  re-deriving raw trajectories instead of trusting the summary JSON — a
  real fix (exclude the reset frame from the window, or compute stats
  within a single episode only) would remove the need to re-derive this by
  hand on every future variant whose eval video spans a reset boundary.
  `kb/wiki/experiments/unified-multi-die-specialist-distillation.md`.
- **Widen `_detect_settle_step`'s tolerance for held-and-jittering objects**
  (distinct from the already-fixed flatness-window/resting_z bug). Its
  5e-5m/15-step tolerance is tuned for a motionless table-rested object and
  is too tight to ever match a *held* object's natural grasp-contact
  jitter, so any env with real sustained lift falls back to a free-fall-
  window-min baseline instead of a directly-detected resting_z — harmless
  so far by coincidence, not guaranteed harmless in general (a held
  object's true grasp height could itself fall inside the fallback
  window). `kb/wiki/experiments/unified-multi-die-specialist-distillation.md`.

## Cloud infrastructure

- **Build a pre-baked GCP VM image with Isaac Sim/Isaac Lab already
  installed.** Skips the fragile ~15-20min from-scratch install window
  every cloud task currently pays — also where multiple real infra bugs
  have hit (a pip-wheel-cache corruption from a preempted mid-install, a
  systemd `Linger=no` default killing a detached tmux install). Highest-
  leverage not-yet-built fix for this class of cloud friction.
- **Promote the SPOT-preemption-truncated-checkpoint fix into a shared,
  reusable script.** Currently a one-off orchestration-script fix (skip any
  `model_*.pt` under 100KB when scanning for a resume candidate); worth
  promoting if cloud SPOT training recurs often enough to justify it.
- **Known Isaac Lab installation limit, relevant to any future multi-env
  sequential-training design:** this installation cannot reconstruct a
  `ManagerBasedRLEnv` in-process after a prior one's `.close()` — hangs
  indefinitely (confirmed via a minimal, trivial-scale repro, not just
  slowness at large `num_envs`). Worked around for the distillation
  pipeline via a single mixed-population env instead of sequential
  per-shape envs; keep in mind before designing another multi-teacher/
  multi-env pipeline the same way.
  `kb/wiki/experiments/unified-multi-die-specialist-distillation.md`.

## Perception / interactive-demo loose ends (pre-Franka-pivot, AR4 era)

- **`interactive_demo.py` live GUI drag verification never performed** —
  needs a human running it without `--headless` to confirm the physical
  drag → settle → pick-and-place → idle-again flow.
- **Minor/cosmetic, non-blocking:** `perception/tests/conftest.py`'s
  sys.path-insert comment overstates how many directory levels it climbs;
  `interactive_demo.py` hardcodes `clip_actions=None` instead of reading it
  from agent config; a redundant filter duplicates `find_by_shape`.
- **Final whole-branch review for the perception-integration plan
  (Task 12)** was explicitly skipped per user instruction — still pending
  whenever that work resumes.
