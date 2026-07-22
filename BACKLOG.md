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

## AR4 arm actuator gains (2026-07-22 finding, not yet fixed)

- **Arm `ImplicitActuatorCfg` (stiffness=40, damping=4,
  effort_limit_sim=20.0, `tasks/ar4/robot_cfg.py`) can't hold the arm's own
  pose statically against gravity** — confirmed live: gripper height sagged
  +0.4748m -> +0.1988m over ~1-2s of sim time with a single commanded
  target held (not re-issued every step, unlike RL's own control loop).
  Found while isolating the gripper-jaw2-drive bug (see
  `kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md`'s 2026-07-22
  "later" UPDATE) — not yet known whether this affects RL training itself
  (a policy re-issues targets every control step, which may compensate)
  or is purely a static-diagnostic artifact. Candidate follow-up: bump arm
  stiffness/damping, or confirm via eval video that trained AR4 policies
  don't show visible arm droop mid-episode. **Corroborating evidence
  (same day, `scripts/grasp_demo_v2.py`'s scripted-grasp validation)**:
  same default gains produced a 1.42rad joint-tracking error moving to a
  real multi-joint grasp pose (not just static holding) — a test-local
  stiffness/damping boost (4000/200) dropped this to 0.026rad. This is a
  real dynamic-tracking gap, not only a static-hold one; raises the
  priority of deciding whether to bump these gains for real (a judgment
  call on production `tasks/ar4/robot_cfg.py` values, flagged rather than
  changed unilaterally, since it could affect existing trained checkpoints/
  dynamics fidelity across the whole AR4 task suite).

## AR4 classical-IK positioning precision (Hypothesis 1, longstanding — re-surfaced 2026-07-22)

- With the gripper-jaw2-drive bug and the arm's actuator-gain weakness
  both fixed/worked-around (see above and
  `kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md`'s 2026-07-22
  UPDATE), a scripted grasp attempt (`scripts/grasp_demo_v2.py`) still
  never touched the cube — the remaining ~3.3cm IK positioning residual
  (grid-search + bounded-step DLS polish) is nearly 3x the cube's own
  12mm size. This is this project's own longstanding Hypothesis 1
  (single-Newton-step DLS trapped in a local minimum in standalone
  classical scripts), now cleanly isolated as the sole remaining blocker
  for a scripted (non-RL) grasp on this asset — not a new problem.
  Candidate next step (Tier 1 process required — this is a methodology
  change, not a parameter tweak): a better classical IK solving method
  (finer grid, analytic/closed-form solver, or different global-
  optimization approach) for these standalone demo scripts specifically.
  Note this does NOT necessarily block RL-driven grasping — Experiment 11
  already showed continuous incremental IK driven by an RL policy every
  control tick produces real sustained antipodal contact on this platform,
  suggesting this positioning problem may be specific to single-big-jump
  classical scripts.

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
