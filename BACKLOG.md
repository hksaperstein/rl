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

## AR4 grasp-demo seed selection is not real-robot-deployable (2026-07-23 finding, not yet fixed)

- **`scripts/grasp_demo_v2.py`'s `_find_best_seed` teleports the robot
  (`write_joint_position_to_sim`) through several candidate joint configs
  and scores each before committing to one — no real AR4 can do this (there
  is no "try a config, check, undo, try another" operation on physical
  hardware).** Tested directly (2026-07-23, ar4-grasp-z-envelope task,
  coordinator-directed): a bounded local "wiggle" retry (small PD-driven
  perturbations from HOME_Q, no teleport) FAILED to converge in 7/7 attempts
  (stuck at 59-80° rotation error) — bounded local search cannot substitute
  for the teleport search's much broader candidate pool. **But** a single
  deliberate real move (still no teleport, an ordinary commanded joint move)
  to the already-known-good `KNOWN_GOOD_PREGRASP_Q` reference posture,
  followed by the normal resolve, converged immediately (1.5mm/2.7°, zero
  retries needed) and reproduced the investigation's main Z-height finding
  just as well as the teleport-based version. **Candidate fix, not done
  here**: replace `_find_best_seed`'s live teleport search with either (a)
  this same fixed-posture-move pattern (a single hardcoded good reference
  posture per task/cube-height class, reached via one real commanded move),
  or (b) a small closed-form/geometric heuristic that computes a reasonable
  initial elbow-up/down posture directly from the target position, rather
  than searching at all. Full detail:
  `kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md`'s 2026-07-23
  UPDATE.

## AR4 classical-IK positioning precision (Hypothesis 1 — RESOLVED 2026-07-22, was a measurement/frame-bug artifact, not a solver-mechanics problem)

- **UPDATE 2026-07-22 (later, same day): the "~3.3cm, DLS-trapped-in-a-
  local-minimum" diagnosis directly below was itself wrong.** Full
  investigation in `kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md`'s
  2026-07-22 (later) UPDATE. Four independent bugs found and fixed in
  `scripts/grasp_demo_v2.py`: (1) a world-frame-vs-root-frame Jacobian
  mismatch (the actual cause of "DLS polish diverges"/"joints slam to
  limits"), (2) the original grid search's own "0.033m" reading was a
  transient/unsettled measurement artifact (true settled residual for that
  exact config: `0.42m`), (3) the target was `link_6`'s raw origin, not the
  gripper's actual jaw pinch point 36mm away (`_EE_OFFSET`), (4)
  `CUBE_POS_W` was hardcoded to a position ~20cm from where the cube
  actually spawns in the scene these scripts use. With all four fixed:
  genuine `10.5mm`/`1.8mm` (grasp/pregrasp) precision, real physical cube
  contact confirmed via video for the first time this whole investigation.
  **Still not a full lift** — diagnosed as a grasp-ORIENTATION gap
  (position-only IK has no incentive to select a sensible pinch geometry;
  the found basin's approach is a ~18-degree-tilted side-approach that
  lands ~10mm short of full contact depth, capped by a joint-limit-style
  constraint in that basin, confirmed by testing a re-aimed-lower target
  which made things worse rather than better).

- **Two concrete follow-ups, not done this pass:**
  1. **Apply the same 4 fixes to `grasp_demo.py`/`oracle_rollout.py`.**
     Confirmed (via grep) both share Bug 1's pattern (`get_jacobians()`
     used directly, no `matrix_from_quat`/`quat_inv` anywhere in either
     file); `grasp_demo.py` also has Bug 4's identical wrong `CUBE_POS_W`
     constant. `interactive_joint_demo.py` uses a closed-form 3-DOF IK
     (confirmed via its own docstring/code), not Jacobian/DLS-based at
     all — Bug 1 does not apply there.
  2. **Orientation-aware IK redesign to close the diagnosed remaining
     gap** — switch `DifferentialIKControllerCfg`'s `command_type` from
     `"position"` to `"pose"` with a deliberately-chosen approach
     orientation (e.g. a top-down or a squarely-horizontal pinch,
     rather than the ~18-degree side-tilt the position-only solver
     happened to find), or search for a different elbow/wrist basin with
     a more favorable geometry. Judged a real methodology change (needs a
     justified choice of target orientation, not just a parameter tweak)
     — Tier 1 process required before implementation, flagged rather than
     attempted further in that pass.

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
