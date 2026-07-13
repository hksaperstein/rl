# Plan: d4 edge-grasp rung 0 (scripted opposite-edge antipodal pick)

Spec: `docs/superpowers/specs/2026-07-13-d4-edge-grasp-rung0-design.md`.
Executor: subagent-driven-development (controller = Principal,
implementer = Senior, reviewer = different Senior instance).

## Global constraints

- **GPU embargo until controller handover:** a 3-seed training batch owns
  the flock lock all morning. Tasks 0–1 must not launch Isaac Sim, must
  not touch `/tmp/rl_isaac_sim.lock`, must not run anything on the GPU.
  Task 2 runs only after the controller explicitly hands over the GPU.
- When Task 2 runs: never `--headless` locally, `DISPLAY=:1`, every
  Isaac launch wrapped
  `flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 DISPLAY=:1 /home/saps/IsaacLab/isaaclab.sh -p ..."`;
  teardown-hang reaping procedure per CLAUDE.md (near-idle GPU/CPU +
  [DONE] in log = safe to kill -TERM).
- Commit messages end:
  `Claude-Session: https://claude.ai/code/session_01Cyn9r96MmvaUvHHAFYjvfi`

## Task 0 — desk check (no sim launch)

- Pad width w and contact-material μ: read from the Franka hand asset
  (USD inspection via `/home/saps/IsaacLab/_isaac_sim/python.sh` with
  plain `pxr` imports, NO SimulationApp/AppLauncher so nothing touches
  the GPU). If μ is only resolvable at runtime, record the blocker,
  assume μ ∈ [0.2, 1.0], and pin the runtime readout as a Task 2 step.
- Compute, for a = 30.3mm and a chosen contact-height offset δ (start
  3mm) above the table-resting bottom edge: the contact φ at both jaws,
  the positional window a/2 − w/2, and confirm φ ≤ arctan(μ_min) with
  the demo's 3.4° orientation tolerance added.
- **Gate:** if no δ keeps φ inside the friction cone, STOP — desk-stage
  falsification, report to controller (ladder climbs to rung 1, no sim
  runs burned).

## Task 1 — implement (no sim launch)

- New helper (suggest `tasks/franka/antipodal_edge_grasp.py`):
  `edge_pair_grasp_axes(mesh_vertices, resting_quat) -> list[GraspAxis]`
  — face-down classification (all 4 faces; raise/log on non-settled
  pose), 3 opposite-edge pairs, common-perpendicular axis + tilted
  gripper quat + waypoints per pair. Shape-general signature (any convex
  polyhedron's edge list), d4 is just the first caller.
- `scripts/dice_pick_demo.py` gate G: d4-only branch selecting the
  reachability-best pair (wrist-yaw distance from current config),
  staged approach along the axis normal, tilted descent, close, lift.
  Existing closure-window displacement + z-gain instrumentation extended
  to log lateral ejection explicitly.
- **Non-d4 byte-identity check:** `git diff` must show the
  d8/d10/d12/d20 path untouched (branch guarded strictly on die==d4).
- Unit-test the helper's pure geometry with plain pytest
  (`/home/saps/IsaacLab/_isaac_sim/python.sh -m pytest -p no:launch_testing`):
  known resting quats → expected 35.26° tilt, 21.4mm span, all-4-faces
  coverage, non-settled rejection.
- STOP after Task 1. Report readiness; controller reviews diff and
  triggers Task 2 after GPU handover.

## Task 2 — seeded trials (GPU, gated on controller handover)

- 5 trials, seeds 42/123/7/1000/2026, d4 commanded, per-trial artifacts:
  closure-window lateral ejection (mm), z-gain, video (full arm + table,
  runs past the event).
- 1 regression smoke: d20 pick, seed 42 — must match prior behavior.
- If μ was deferred from Task 0: read contact-material μ at runtime
  first and re-run the Task 0 arithmetic before the trials.

## Task 3 — verdict + docs

- Verdict vs spec's pre-registered criteria (≥4/5, ≤5mm ejection,
  regression clean). Append verdict to the spec; update ROADMAP + kb
  (`dice-pick-demo.md` open-follow-up section) in the same pass;
  send first passing trial's video to user; commit+push.
