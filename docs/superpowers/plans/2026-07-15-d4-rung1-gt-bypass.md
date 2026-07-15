# Plan: d4 rung-1 follow-up — ground-truth XY-bypass + rerun

Spec addendum: `docs/superpowers/specs/2026-07-15-d4-rung1-pad-geometry-design.md`'s
"Addendum: ground-truth XY-bypass" section.
Executor: subagent-driven-development (controller = Principal, implementer
= Senior, reviewer = different Senior instance).

This is a narrow follow-up to the just-closed rung-1 verdict (0/5 d4
trials untested, blocked at perception) — it does not reopen the notch
geometry design, only adds a way to test it.

## Global constraints

- The bypass must be **opt-in, default OFF**. Every other call site
  (all 4 non-d4 die types, any existing script/test invocation) must
  behave byte-identically with the flag unset — this is a regression
  risk on working code, treat it with the same care as Task 1's
  base-asset byte-identity requirement.
- Do not touch the notch fixture geometry, the fixed-joint attachment
  code, or anything in `tasks/franka/notch_fixture.py` — those are
  already reviewed and verified working (all 9 cloud trials passed
  `sim.reset()` cleanly). This plan only changes how the grasp target
  XY is sourced.
- Task 1 (implement + local verify) is no-sim-launch-safe only for the
  code change itself; actually exercising the bypass requires sim, same
  local/no-GPU-available constraint as before — so Task 1 covers the
  code + unit-testable logic only, Task 2 is the GPU-requiring rerun.
- **Execution backend for Task 2: GCP cloud**, same recipe as before
  (`docs/cloud/franka-cloud-shakedown.md`). Cost cap unchanged: notify
  the user if cumulative spend (this task + the already-spent ~$1-2 from
  the prior rung-1 cloud run) exceeds $15 total. Ship the already-known
  asset list this pipeline needs (notch fixture USD, dice USDs +
  per-die manifests, detector weights, vision/.venv setup) — the report
  from the prior cloud task documents exactly what's needed beyond the
  base recipe.
- Commit messages end:
  `Claude-Session: https://claude.ai/code/session_012Yn68ovC4bfZNfctHyX4Sd`

## Task 1 — implement the bypass (no sim launch)

- Add a new CLI flag to `scripts/dice_pick_demo.py`'s argparser (e.g.
  `--gt-xy-bypass`, boolean, default `False`) — read the existing
  `--gate`/`--choice`/`--seed` flag definitions for this repo's argparse
  style before adding.
- In `run_gate_g` and `run_gate_v`, immediately after `gt_pos` is
  computed (both already compute this — currently used only for the
  diagnostic offset print), branch on the new flag: if set, use
  `target_xy = (gt_pos[0], gt_pos[1])` instead of `(det_x, det_y)`. The
  detector subprocess still runs either way (needed for the diagnostic
  comparison print and to keep the code path uniform) — only the value
  actually used for `target_xy` changes.
- Update both gates' print statements so the log output makes it
  unambiguous which source was used for `target_xy` when the flag is
  set (e.g. `[GATE G] target_xy SOURCED FROM GROUND TRUTH (bypass
  active): ...` vs the existing detector-sourced print) — this
  needs to be visually obvious in any future log/video review, not a
  silent internal branch.
- Do NOT change `select_target_detection`'s own behavior or its
  "fails loudly, never falls back to ground truth" contract for the
  non-bypass path — the bypass is a separate, explicit branch, not a
  fallback inside that function.
- Unit-test whatever is unit-testable without sim (e.g. argparse flag
  wiring, if this repo's existing test suite covers CLI parsing at all
  — check `tests/` first; if nothing tests CLI parsing today, a
  pytest addition isn't required, note this rather than inventing a
  test harness for it).
- **Non-bypass regression check**: `git diff` should show the new
  branch is additive — the existing detector-sourced path must remain
  reachable and unchanged when the flag is unset. Grep for every
  existing call site of `run_gate_g`/`run_gate_v` (this repo's own
  scripts/tests) to confirm none of them need updating for a new
  required argument (the flag must have a safe default).
- STOP after Task 1. Report readiness; controller reviews diff before
  Task 2 ships anything to the cloud.

## Task 2 — rerun the 5 d4 seeded trials under the bypass (GCP cloud)

- Provision per the recipe (same asset-shipping list as the prior
  rung-1 cloud task already worked out — do not rediscover this from
  scratch, read `.superpowers/sdd/task-2-report.md`'s "Judgment call"
  section for the exact file list).
- Run the same 5 seeds (42, 123, 7, 1000, 2026) with `--gt-xy-bypass`
  set, d4 commanded, Gate G or V (pick whichever produces the video +
  verdict artifacts this repo's convention expects — Gate V if a video
  is wanted, matching the prior task's own choice).
- Per-trial artifacts: closure-window lateral ejection (mm), z-gain,
  contact-sensor readout from `d4_leftfinger_contact`/
  `d4_rightfinger_contact` (still required — a bypass on perception
  does not relax the "confirmed flush contact, not video-only" success
  criterion from the original spec).
- Do NOT rerun the non-d4 regression smokes — those already passed
  cleanly in the prior cloud task and nothing about this change touches
  their code path.
- Sync all artifacts to GCS, full teardown after, cost-cap check as
  before.

## Task 3 — verdict + docs

- Verdict against the ORIGINAL spec's pre-registered d4 primary
  criterion (≥4/5, ≤5mm ejection, confirmed contact) — this rerun is
  what actually tests that criterion for the first time.
- Explicitly label this verdict as "grasp mechanism only, perception
  bypassed" per the addendum's own scope language — do not conflate
  with "the perception-driven demo can pick a d4."
- Append verdict to the spec (in the Addendum section, not overwriting
  the original 2026-07-15 verdict); update ROADMAP.md and
  `kb/wiki/experiments/dice-pick-demo.md` in the same pass; send video
  if any trial passes; commit + push.
