# Plan: d4 edge-grasp rung 1 (rigid V-notch fingertip geometry)

Spec: `docs/superpowers/specs/2026-07-15-d4-rung1-pad-geometry-design.md`.
Executor: subagent-driven-development (controller = Principal,
implementer = Senior, reviewer = different Senior instance).

**Execution backend for Task 2: GCP cloud (SPOT g2-standard-4 + L4), not
local Isaac Sim** — direct user instruction (2026-07-15), freeing the
local GPU for the concurrent d20 standard-size work. Cloud runs
headless — the confirmed, standing exception to the local
never-headless rule (`kb/wiki/concepts/cloud-training.md`,
`docs/cloud/franka-cloud-shakedown.md`). Recipe of record for every step
below: `docs/cloud/franka-cloud-shakedown.md`.

**Cost cap: notify the user if cumulative spend on this run exceeds
$15** (direct instruction, 2026-07-15). No BigQuery billing export
exists and the Billing console lags real usage by hours, so track by
estimate: combined on-demand rate ≈$0.361/hr (spot usually cheaper, see
per-SKU pricing in `kb/wiki/concepts/cloud-training.md`). Task 2 must
log instance creation time and check elapsed-uptime × rate at least
once before the trials finish; Task 3 must report the final estimate
regardless of outcome.

## Global constraints

- Tasks 0–1 are desk-check/implementation only — no sim launch, no GPU,
  local or cloud. Confirm this with `git diff` scope before starting
  Task 2.
- Task 2 ships to and runs entirely on the cloud instance; do not touch
  `/tmp/rl_isaac_sim.lock` or the local GPU for this task (the local GPU
  is dedicated to the concurrent d20 standard-size work per Principal's
  own sequencing decision this session).
- Per this repo's spec, the fixture is **unconditional** (both fingertips,
  every die type) — Task 1's diff must reflect that, not a d4-only
  branch, and Task 2's trial matrix must include the non-d4 regression
  smokes as a first-class deliverable, not an afterthought.
- Commit messages end:
  `Claude-Session: https://claude.ai/code/session_012Yn68ovC4bfZNfctHyX4Sd`

## Task 0 — desk check (no sim launch, no GPU)

- Measure the Franka fingertip's actual tip geometry directly from the
  asset (pxr USD inspection only, no SimulationApp) — confirm the
  ≈9.3mm table clearance at the ≈10mm grip depth is real given the
  fingertip's real extent, not the research pass's carried-over
  ~14–18mm estimate.
- Confirm a rigid convex-hull notch fixture is attachable via a fixed
  joint to the existing fingertip prim without modifying the base
  Franka asset (needed for the byte-identity regression argument on
  everything except the new attachment prim).
- Re-derive the 110°/~10mm/~4mm/~11mm figures against the actual
  measured d4 mesh (reuse the existing edge-length measurement,
  `a = 23.591mm`, already double-verified in this project's history —
  do not re-run the mesh k-means a third time, just re-check the
  downstream trig).
- **Gate:** if the measured fingertip geometry makes the ≈9.3mm table
  clearance implausible (e.g., real fingertip extent turns out much
  larger than estimated), STOP — report to controller before any
  fixture is built; this would be a desk-stage falsification of the
  chosen grip depth, not the notch concept, and the depth parameter
  should be re-solved, not silently forced.

## Task 1 — implement (no sim launch, no GPU)

- Author the notch fixture as a small mesh (Blender or direct USD
  authoring, whichever this repo's existing asset pipeline supports more
  directly — check `scripts/bake_die_asset.py` / `scripts/build_asset.py`
  for the established mesh-authoring convention before picking a new
  one) sized per the spec's geometry (110° internal angle, ~4mm depth,
  ~11mm opening, ~2mm chamfered lead-in), exported as a rigid convex
  collision mesh.
- Attach it to both Franka fingertip prims via a fixed joint in the
  scene config used by `scripts/dice_pick_demo.py` (`tasks/franka/dice_scene_cfg.py`
  is the scene this demo uses — read it first) — **unconditional**, not
  gated on die type, per the spec's North Star call.
- Extend `scripts/dice_pick_demo.py`'s d4 gate-G branch (or remove the
  branch entirely if the straight-down approach + notch means the d4
  path is no longer geometrically special relative to the other dice —
  check this explicitly, don't assume rung 0's tilted-axis branch
  structure still applies once the approach is untilted again) to use
  the existing straight-down descent, closing on the notch.
- **Base-asset byte-identity check**: `git diff` must show the stock
  Franka fingertip mesh/collision geometry untouched — only a new
  attachment prim added. This is the regression-guard argument Task 2's
  non-d4 smokes will empirically confirm.
- Unit-test whatever pure geometry this introduces (fixture placement
  math, if any) with plain pytest, matching this repo's existing
  convention (`/home/saps/IsaacLab/_isaac_sim/python.sh -m pytest -p no:launch_testing`).
- STOP after Task 1. Report readiness; controller reviews diff before
  Task 2 ships anything to the cloud.

## Task 2 — seeded trials + non-d4 regression (GCP cloud, gated on controller handover)

- Provision per `docs/cloud/franka-cloud-shakedown.md` (SPOT
  g2-standard-4 + L4; fall back through zones on
  `ZONE_RESOURCE_POOL_EXHAUSTED` per the recipe's own precedent). Ship
  repo via `git archive | ssh tar -x`. Record instance creation
  timestamp immediately for the cost-cap tracking above.
- **d4 primary**: 5 seeded trials (42, 123, 7, 1000, 2026), same
  per-trial artifacts as rung 0 (closure-window lateral ejection, z-gain,
  video — full arm + table framing, past the event) plus confirmed
  contact-sensor readout from the existing `d4_leftfinger_contact`/
  `d4_rightfinger_contact` sensors (do not accept a video-only read as
  sufficient for the "confirmed flush notch-facet contact" criterion).
- **Non-d4 regression smokes**: one seeded run per die type
  (d8/d10/d12/d20) under the now-notched gripper, same seed(s) as this
  project's existing passing baseline in `kb/wiki/experiments/dice-pick-demo.md`,
  compared directly against that baseline's z-gain/success figures.
- Sync all artifacts (videos, verdict JSONs, logs) to GCS
  (`scripts/sync_run_to_gcs.py` pattern) before any teardown.
- Check elapsed instance uptime × the ~$0.361/hr estimate at least once
  before finishing; if projected total is approaching $15, flag the
  controller before continuing rather than after.
- Full teardown: verify instances/disks/snapshots all empty, per this
  repo's standing operational discipline.

## Task 3 — verdict + docs

- Verdict vs. the spec's two independent pre-registered criteria (d4
  primary: ≥4/5, ≤5mm ejection, confirmed contact; non-d4 regression
  guard: each die still meets its own baseline) — report both even if
  one passes and the other doesn't; per the spec, a d4 pass does not
  excuse a non-d4 regression.
- Report the final cost estimate (elapsed instance time × rate)
  regardless of the $15 threshold being crossed or not.
- Append verdict to the spec; update ROADMAP.md and the relevant kb
  article(s) (`kb/wiki/experiments/dice-pick-demo.md`'s open-follow-up
  section at minimum) in the same pass, not deferred; send the first
  passing d4 trial's video (not stills) to the user; commit + push.
