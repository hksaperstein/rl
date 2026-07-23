# ROADMAP

Forward-looking planning doc: current priorities and what's next, not a
history ledger. Every full experiment result lives in `kb/wiki/` (start at
`kb/wiki/index.md`) or `docs/superpowers/specs|plans/`; this file links out
to it rather than re-narrating it. Update after each completed plan (per
`.superpowers/sdd/progress.md`): move newly-shipped work into "Recently
landed" (one line + kb link) and refresh "Planned / near-term priorities"
below.

## Active workstreams

Nothing is currently mid-execution as of 2026-07-22 — the AR4-vs-Franka
root-cause investigation closed with Task 7, and both the
unified-multi-die-specialist-distillation and target-selection-clutter
experiments reached COMPLETE verdicts. See "Planned / near-term
priorities" below for what's queued next.

## Direction

Isaac-Lab-based robotics RL, expanding beyond the current dice/Franka work
into other tasks/robots, object detection/perception, and mobility. No
committed roadmap items beyond the items below yet — this is a stated
direction, not a scoped backlog.

## Planned / near-term priorities

Roughly in the order they'd likely be picked up:

1. **Target-selection-clutter E2** (3→4 distractors) — the next
   separately-gated stage after Stage E1's clean pass (2026-07-21); not
   auto-started by E1.
2. **Target-selection-clutter S1** (fold d8/d10 back into the clutter
   curriculum) — well-motivated now that d8/d10's grasp-discoverability
   null was closed with a positive resolution (demo-warmstart H2, both
   shapes PASS), after being deferred from the original 4-shape scope.
3. **Revisit unified-multi-die-specialist-distillation's Task 4 scope**
   to include d8/d10 alongside d12/d20, now that both are proven
   grasp-discoverable via demo-warmstart — an open decision for
   Principal (the original scope-narrowing rationale is documented in
   `kb/wiki/experiments/unified-multi-die-specialist-distillation.md`'s
   Task 4 section; not yet revisited).
4. **AR4 gripper mimic-vs-actuator dynamics conflict — RESOLVED, 2026-07-22.
   Scripted grasp validation attempted next, blocked on a different,
   pre-existing, already-documented problem (Hypothesis 1).**
   The mimic constraint was removed entirely (`2576e94`,
   `scripts/build_asset.py`'s `_remove_gripper_jaw2_mimic_constraint`), but
   this left `gripper_jaw2_joint` with NO PhysX drive at all (it was
   originally a mimic-slave joint; only the mimic's reference joint, jaw1,
   ever got an independent `UsdPhysics.DriveAPI` from the URDF importer) —
   confirmed by direct USD inspection (`prim.GetAppliedSchemas()`: jaw1 has
   `PhysicsDriveAPI:linear`, jaw2 doesn't). Two earlier "opposite end"/
   "pinned at a limit" live-test signatures turned out to be an unrelated
   confound (the arm's own actuator gains are too weak to hold its pose
   statically, so an uncontrolled falling/swinging arm base was injecting
   Coriolis coupling into the gripper joints) — resolved by temporarily
   boosting the arm's stiffness/damping for the diagnostic, which revealed
   jaw2's true behavior (completely inert, no drive at all) for the first
   time. Fix: new `_add_gripper_jaw2_drive` in `scripts/build_asset.py`,
   authoring a `DriveAPI:linear` on jaw2 mirroring jaw1's own. Verified
   live: jaw1/jaw2 now mirror at every step in both CLOSE/OPEN, and an
   isolated mid-range sweep shows jaw2 converging cleanly to its own
   commanded target with a normal PD curve.

   **Scripted (non-RL) grasp validation, continued 2026-07-22 (same day,
   later): the "~3.3cm classical-IK residual" diagnosis above was itself
   wrong — FOUR independent bugs found and fixed, real physical cube
   contact restored for the first time, but a full stable lift is still not
   achieved.** In order of impact: (1) `robot.root_physx_view.get_jacobians()`
   returns the Jacobian in the WORLD frame, but every AR4 classical demo
   script fed it directly into `DifferentialIKController` alongside
   ROOT-frame vectors — harmless in Isaac Lab's own tutorial this pattern
   was copied from (identity-orientation base) but a real sign-mirroring
   bug for AR4's 180-degree-yaw base, and the actual explanation for the
   "DLS polish makes things worse"/"joints slam to limits" signature; (2)
   the original grid search's own "best" reading (`0.033m`, i.e. the
   "~3.3cm" figure) was itself a transient measurement artifact from only
   15 unsettled steps per candidate — the true settled residual for that
   exact reported config was `0.42m`, not `0.033m`; (3) the script's target
   was `link_6`'s own raw origin, not the actual gripper jaw pinch point
   36mm away (`_EE_OFFSET`, already used elsewhere but never applied here);
   (4) `CUBE_POS_W=(0.20,0.28,0.009)`, hardcoded in every classical demo
   script, doesn't match where the cube actually spawns in the scene these
   scripts use (`Ar4PickPlaceMirrorSceneCfg` recenters it to
   `(0.0,0.275,0.006)` for the RL env's own randomization range) — a ~20cm
   targeting error, independent of and dominating the other three. All four
   fixed in `scripts/grasp_demo_v2.py`; genuine, reproducible
   `10.5mm`/`1.8mm` (grasp/pregrasp) precision now verified, and — a first
   for this entire investigation — the cube visibly moves/gets bumped
   during CLOSE/lift (confirmed via video, not just printed metrics). Still
   no full lift: the dominant remaining error is a ~10mm Z-height shortfall
   in the verified-best basin, and directly testing a lower re-aimed target
   made it WORSE (the search re-converged to the same joint config,
   confirming this basin's descent is capped by a joint-limit-style
   constraint, not a simple offset). Diagnosed as a grasp-ORIENTATION gap
   (position-only IK has no incentive to pick a sensible pinch geometry),
   not a positioning-precision gap — a genuinely different, narrower,
   better-characterized problem than the original Hypothesis 1 framing.

   **`command_type="pose"` orientation-aware IK redesign, done 2026-07-22
   (same day, later, ar4-grasp-orientation-fix task): the orientation
   mechanism itself is now confirmed FIXED, but this surfaced a deeper,
   genuine AR4 kinematic limit (joint_3/elbow) that still blocks a full
   lift — not yet a working grasp.** `scripts/grasp_demo_v2.py` switched
   to full pose (position+orientation) DLS with an explicit canonical
   straight-down target (mirroring `demo_franka_ik_dice_line.py`'s own
   `canonical_down_quat_w` precedent, built from AR4's own world-frame
   basis vectors). Verified live via independent axis readout (not just
   the scalar residual) that the solver genuinely reaches vertical when
   not joint-limited (0.2-degree error at a 32cm-reach test position).
   Two real bugs found/fixed en route: an arbitrary jaw-heading choice
   was deadlocking `joint_6` at its own hard limit (fixed by rotating the
   heading 90 degrees), and GRASP's own seed search was picking an
   orientation-incompatible seed because it scored on position alone
   (fixed: combined position+orientation seed scoring, and seeding GRASP
   from PREGRASP's own converged config instead of an independent search).
   **But: AR4's `joint_3` hard limit (`[-1.55, +0.91]` rad) genuinely
   prevents reaching the cube's actual 9mm grasp height while holding a
   vertical wrist, confirmed across 3 different reach distances (20/27.5/
   32cm) via a new `--cube-xy` test override — moving the cube CLOSER
   made it WORSE, not better, and "aim lower to compensate" was retested
   in a non-joint-limited basin and again made it worse, ruling out both
   "just a seed problem" and "just this one basin's limit" as the
   explanation.** A deliberate 30-degree tilt (new `--tilt-deg` option, a
   middle ground between fully-vertical and the original uncontrolled
   result) was tried once and instead hit solver instability (rotation
   error diverged round over round) rather than resolving the conflict —
   flagged as an open follow-up, not debugged further this pass. No real
   cube contact/lift achieved in 11 full runs this session; cube height
   stayed flat at its resting ~6mm in every clean run.

   Full detail: `kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md`'s
   2026-07-22 (later, ar4-grasp-orientation-fix task) UPDATE. Follow-ups
   logged to `BACKLOG.md`: applying the same 4 position-fix bugs to
   `grasp_demo.py`/`oracle_rollout.py` (confirmed to share Bugs 1 and/or
   4; `interactive_joint_demo.py` uses a closed-form 3-DOF IK, unaffected);
   debugging the tilt-induced DLS instability and trying smaller tilt
   angles / a smaller per-round rotation step bound; testing whether a
   different BEARING (not just reach distance) relieves the joint_3
   conflict. Arm-actuator-gain follow-up (unchanged from before) also
   still on `BACKLOG.md`.

   **`joint_3` limit verified against real hardware (Part A) + DLS-tilt
   instability bug fixed (Part B), done 2026-07-22 (same day, later still,
   ar4-tilt-fix task) — but GRASP itself hits a NEW, deeper, tilt-
   independent basin conflict; still no lift.** Part A: the `[-1.553,
   +0.908]` rad `joint_3` limit was checked directly against the vendor's
   own `annin_ar4_description` URDF/YAML source (not secondhand claims) —
   `config/mk5.yaml`'s `j3_limit_min/max: -89/52 degrees` converts to
   `-1.5533/0.9076 rad`, matching the built USD asset to 4 decimal places,
   and all 5 shipped model variants (mk1-mk5) carry the identical limit.
   **Confirmed real hardware, not an import bug — no fix applicable.**
   Part B found and fixed a genuine mechanism bug behind the prior
   UPDATE's "`--tilt-deg 30` diverges" finding: `polish_from_seed` solved
   the DLS Jacobian once per "round" then held that target open-loop for
   30 physics steps before re-checking, unlike
   `demo_franka_ik_dice_line.py`'s own proven every-physics-step re-solve
   pattern; fixed by switching to continuous per-step re-solve, matching
   Franka's own `_MAX_ROT_STEP` bound (0.15rad→0.03rad, a 5x reduction),
   and — the change that actually mattered live — raising DLS damping
   (`lambda_val` 0.02→0.3, new `--lambda-val` CLI override). Validated:
   PREGRASP now converges cleanly and reproducibly to `4.6mm`/`0.4°` at
   multiple tilt angles (15°, 25°) and reach distances. **But GRASP
   itself (the true ~9mm-height waypoint) hits the SAME stable basin
   deadlock (~1.1-1.4rad final rotation error) regardless of tilt angle
   (10/15/25°), reach distance (27.5cm/32cm), damping (0.02/0.1/0.3), or
   seed diversity (6 new wrist-perturbed seed variants tried)** — ruling
   out numerical instability as GRASP's own blocker (already fixed) and
   pointing instead at a genuine, tilt-independent, redundant-wrist
   basin-connectivity property specific to the low grasp height. No
   grasp+lift validated this session at any tested configuration; the
   one phased-execution run that reached that stage showed `cube.z` flat
   at its resting height throughout, consistent with the ~2.6cm final
   residual exceeding the cube's own size. Full detail (including the
   specific failure signature and candidate next steps — per-waypoint
   orientation instead of one shared canonical target, a proper
   null-space secondary objective, or a different bearing):
   `kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md`'s 2026-07-22
   (later still, ar4-tilt-fix task) UPDATE.

   **Incremental PREGRASP->GRASP height descent, done 2026-07-22 (same day,
   later still, ar4-grasp-descent-continuity task): CONFIRMS the
   disconnected-basin/rotation-deadlock hypothesis, but surfaces a
   separate, deeper Z-height reachability floor as the real remaining
   blocker — still no lift.** Instead of solving GRASP as an independent
   one-shot target (which reliably deadlocked at ~1.1-1.4rad rotation
   error, above), interpolated the target height from PREGRASP's converged
   height down to GRASP_AT_HEIGHT in N small steps, re-solving
   `polish_from_seed` at each sub-height without teleporting between steps
   (new `--num-descent-steps` CLI arg, default 30). **Confirmed across 4
   independent configurations (30-step/0° tilt, 60-step/0° tilt, 40-step/
   15° tilt, 30-step at a farther 32cm reach) that the catastrophic
   rotation deadlock is completely avoided** — final rotation error landed
   in the 0.004-0.21rad (0.25-12°) range in every run, nowhere near the
   1.1-1.4rad basin the one-shot method hit. But **all 4 runs instead
   converge to a consistent 17-24mm position residual, and in every case
   the per-axis breakdown shows this is almost entirely a Z-HEIGHT
   shortfall** (X/Y residual near-zero, e.g. run 1's
   `['-0.00028', '-0.00486', '-0.01707']` xyz-axis residual) — reproducing,
   under a materially different continuous-descent methodology, the exact
   Z-shortfall signature the earlier position-only investigation found.
   No cube contact/displacement/lift in any of the 4 runs (`cube.z` flat at
   its ~6mm resting height throughout every CLOSE/lift/hold phase in every
   log; video-confirmed for run 1 — the gripper is visibly not near the
   cube in any CLOSE/lift/hold frame). **Verdict: this task's specific
   hypothesis (small-step continuous descent avoids the one-shot basin
   jump) is CONFIRMED — that problem is now closed with a validated fix —
   but it reveals a second, independent, tilt/reach/step-count-independent
   Z-height reachability limit as the real remaining blocker for an actual
   grasp+lift.** Next diagnostic (not yet run, flagged for a future pass):
   directly sweep the reachable Z-height envelope at this XY position
   (via `--grasp-height`, in fine increments through the descent method
   itself) to map exactly how low this basin can genuinely descend and
   which joint's margin is the actual binding constraint, rather than
   continuing to test only the one target height. Full detail:
   `kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md`'s 2026-07-22
   (later still, ar4-grasp-descent-continuity task) UPDATE.

   **Z-height envelope mapped + bearing sweep + deployability check, done
   2026-07-23 (ar4-grasp-z-envelope task): the Z-height reachability floor
   is now confirmed genuine, direction-independent, and NOT a teleport-
   search artifact — still no lift, and this line of investigation is now
   fairly exhausted for this exact cube position.** New `--z-sweep`/
   `--bearing-sweep` CLI modes in `scripts/grasp_demo_v2.py` mapped the
   envelope directly: the Z-shortfall grows SMOOTHLY (not a cliff) from
   ~0mm at 33mm height to 23mm at the true 9mm height, tracking joint_3's
   own margin shrinking smoothly in lockstep (0.136rad → 0.084rad) —
   joint_3 is unambiguously the binding constraint but never reaches exact
   zero margin (a soft Jacobian-conditioning effect, not a literal hard-stop
   collision). A 7-point bearing sweep (±60°) found the SAME ~19.2mm
   shortfall at every angle to within 0.02mm — ruling out approach
   direction as a fix, on top of reach distance and tilt (already ruled out
   in prior sessions). A scene-setup sanity check found no cube/table
   height calibration mismatch. **Coordinator-directed deployability check**
   (does the finding depend on `_find_best_seed`'s simulation-only
   teleport-based candidate search?): a bounded local "wiggle" retry
   mechanism (no teleport, small PD-driven perturbations from HOME_Q)
   FAILED to converge in 7/7 attempts (stuck at 59-80° rotation error) —
   but one single deliberate real move (still no teleport) to the
   already-known-good reference posture, then the normal resolve, converged
   immediately and reproduced essentially the same ~17mm Z-shortfall. The
   finding is real, not a simulation-only artifact. Verdict: this is a
   genuine, method-independent kinematic property of this arm/cube-height
   combination. Candidate next step (not decided here, flagged for the
   controller per this task's own instruction): adjust the cube's spawn
   height/position closer to this arm's comfortable envelope — a scene-
   design change that could affect other AR4 experiments' randomization
   ranges, not applied unilaterally. `_find_best_seed`'s teleport-based
   search itself is a real deployability gap independent of this specific
   finding, logged to `BACKLOG.md`. Full detail:
   `kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md`'s 2026-07-23
   UPDATE.

   **Reach-distance sweep + a major NEW gripper-jaw bug found, done 2026-07-23
   (later, ar4-grasp-position-search task): repositioning ruled out as a fix
   for the Z-shortfall, AND the gripper's "OPEN" command was found to
   collapse both jaws onto the IDENTICAL point instead of separating them —
   confirmed but NOT YET FIXED. Capstone grasp+lift validation still
   outstanding.** A new `--radius-sweep` swept reach distance 0.30-0.42m at
   bearing=0: `joint_3`'s own margin becomes genuinely healthy at farther
   reach (0.28-0.88rad, vs. ~0.08rad at the 27.5cm default), but **the
   ~18mm Z-shortfall does not shrink at all across this whole range** —
   at some radii a different joint (`joint_4`) becomes newly pinned
   instead, and at others (0.30m, 0.39-0.42m) NO joint is near its limit
   yet the shortfall persists unchanged, confirming this is a genuine
   multi-joint reachability-envelope property of a fully-vertical grasp at
   this height, not fixable by repositioning within the ~0.30-0.42m/±60°
   region tested so far (bearing-independence already established the
   prior session). Separately, direct empirical measurement
   (`scripts/_sweep_jaw2_symmetry.py` — hold jaw1 open, sweep jaw2's
   commanded value, read back both jaws' REAL world positions) found the
   gripper's current "OPEN" command (`gripper_jaw2_joint` commanded to
   `-GRIPPER_OPEN_POS`, mirroring jaw1's `+GRIPPER_OPEN_POS` under the
   2026-07-21 `gearing=-1` convention) makes jaw2 land at the EXACT SAME
   world point as jaw1 (measured separation `0.00001m`) instead of
   spreading apart — the true mirror position is actually
   `+GRIPPER_OPEN_POS` (the SAME signed value as jaw1), confirmed by a
   clean, monotonic 9-point sweep. **This means the gripper has likely
   never actually opened into a pincer shape in any AR4 script/task using
   the shared `tasks/ar4/robot_cfg.py` constants** — a probable
   same-day, independent, additive contributor to every grasp failure
   today, on top of the Z-height reach limit. The fix itself (correcting
   `GRIPPER_OPEN_COMMAND_EXPR` and `gripper_jaw2_joint`'s USD hard limits)
   was root-caused and empirically confirmed but **NOT YET IMPLEMENTED** —
   session stopped here at user request before any fix, and before any
   phased grasp+lift attempt was run at all. Cube-parking (avoids
   interpenetration during seed-search) and gripper joint-position logging
   were added to `scripts/grasp_demo_v2.py` per two live user observations
   but are also unexercised. **Next steps, in order: (1) fix the jaw2
   open-command bug and re-verify via the same sweep script; (2) test a
   moderate tilt at one of the newly-found comfortable-`joint_3`-margin
   positions (0.39-0.42m), untested combination; (3) only then attempt the
   real phased grasp+lift validation.** Full detail:
   `kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md`'s 2026-07-23
   (later, ar4-grasp-position-search task) UPDATE.

See `BACKLOG.md` for further-out candidates not yet on this list.

## Recently landed

- **Franka IK dice-line pick-and-place demo** (2026-07-21) — classical
  IK-only pick/line-up/relocate of all 5 dice; 8/10 pick-and-place ops
  succeeded, d4 (this project's well-documented hardest grasp case)
  failed both attempts. `kb/wiki/experiments/franka-ik-dice-line-demo.md`.
- **AR4 grasp-discoverability research arc — CLOSED** (Experiments 1-26,
  the shape-classifier perception-debugging saga, and the AR4-vs-Franka
  root-cause investigation through Task 7; 2026-07-05 → 2026-07-21).
  Reach and antipodal-grasp-contact were solved early and reliably;
  genuine lift+carry+place was never confirmed in eval video across 26
  numbered experiments plus their sphere-era precursors. Mounting
  evidence pointed at AR4-asset-specific defects (an unenforced
  gripper jaw-mimic constraint, a classical-IK positioning miss) rather
  than a fundamental RL/reward-design problem — the direct motivation
  for the Franka platform pivot (see `CLAUDE.md`'s North Star section).
  **Task 7 (2026-07-21) tested the one concrete fix candidate this
  investigation produced** (Franka's own confirmed
  `RelativeJointPositionActionCfg` grasp-discoverability fix) directly
  against AR4 — **FALSIFIED, it does not transfer** — closing this
  investigation without a positive result; the jaw-mimic-vs-actuator
  dynamics conflict and the classical-IK positioning miss remain the
  more likely explanations. Full chronological index (one entry per
  experiment, hypothesis → verdict): `kb/wiki/index.md`. Connecting
  throughline: `kb/wiki/concepts/reach-grasp-lift-gap.md`. AR4
  pick-and-place (perception + RL reach/touch + interactive demo) remains
  working end-to-end for what it does cover; full grasp+lift+carry does
  not.
- **d8-antipodal-grasp-quality — CLOSED** (2026-07-20 → 2026-07-21) — a
  cross-platform replay of the AR4-era joint-space-vs-task-space finding
  on Franka/d8; `RelativeJointPositionActionCfg` (H_relative) confirmed
  as a genuinely joint-space fix for the grasp-discoverability collapse,
  3/3 seeds, no arm-specific IK layer needed — real North Star evidence
  that a task-space layer isn't a hidden prerequisite for a new arm to
  train. `kb/wiki/experiments/d8-antipodal-grasp-quality.md`.
- **Target-selection-clutter — COMPLETE through Stage E1** (2026-07-19 →
  2026-07-21) — 3-die clutter curriculum (distractor-count curriculum +
  a distractor-distance observation term); d12 8/8, d20 8/8 under 3
  simultaneous distractors, no wrong-die grasp observed in any inspected
  video frame. `kb/wiki/experiments/target-selection-clutter.md`.
- **Exploration-bonus grasp discovery — SPLIT** (2026-07-19 →
  2026-07-20) — a potential-based exploration bonus for gripper-closure
  attempts; mechanism-level bar fires in 1/3 seeds, behavioral bar stays
  0/24 — the first result in this project's history to land in the
  explicitly pre-registered third outcome category (not a plain
  pass/fail). `kb/wiki/experiments/exploration-bonus-grasp-discovery.md`.
- **d8/d10 demo-warmstart — CLOSED, positive resolution** (2026-07-19 →
  2026-07-20) — H1 (one-demo BC-pretrain) falsified both shapes; H2
  (checkpoint warm-start from the d12 specialist) PASSED both shapes —
  the original grasp-discoverability null was a cold-start exploration
  problem, not an intrinsic physical or reward-design barrier.
  `kb/wiki/experiments/d8-d10-demo-warmstart.md`.
- **Unified multi-die specialist-distillation — COMPLETE** (2026-07-16 →
  2026-07-19) — per-shape specialist → distill → RL-fine-tune pipeline
  for a single policy that grasps a commanded die; narrowed to d12/d20 on
  real evidence (d8/d10 genuinely null at the time), RL fine-tuning fully
  recovers a real BC/DAgger closed-loop-transfer regression to an exact
  8/8 match with each frozen specialist.
  `kb/wiki/experiments/unified-multi-die-specialist-distillation.md`.
- **d4 edge-grasp rungs 0 and 1 — both FALSIFIED** (2026-07-13 →
  2026-07-15) — stock Franka jaws physically cannot straddle the
  tetrahedron along its edge-pair axis (rung 0); a rigid V-notch
  fingertip fixture sweeps the die aside without ever engaging it
  (rung 1). `kb/wiki/experiments/dice-pick-demo.md`'s "Open follow-ups"
  section.
- **Cloud training pipeline PROVEN** (2026-07-13, re-verified
  2026-07-14/15) — GCP SPOT L4, full create→install→train→GCS-sync→
  teardown cycle exercised twice independently; real per-SKU GCP pricing
  and SPOT-preemption/checkpoint-resume handling both documented.
  `kb/wiki/concepts/cloud-training.md`.
- **RL joint-space die-lift, asset-bisect, size-curriculum**
  (2026-07-12 → 2026-07-13) — isolates *shape* (not action space, mass,
  or bake pipeline) as the reliability gate for d20 grasp discovery;
  yields the project's first confirmed d20 lift+carry at the real
  30.3mm target size. `kb/wiki/experiments/joint-space-die-lift.md`,
  `kb/wiki/experiments/asset-bisect.md`,
  `kb/wiki/experiments/size-curriculum.md`.
- **Dice + Franka + detection convergence milestone** (2026-07-11) —
  commanded die type → trained `vision/` detector identifies/localizes it
  among five dice → staged DiffIK picks the correct one; 4/5 die types
  passing (d4 the sole, pre-declared permitted failure). Scripted
  controller, not RL. `kb/wiki/experiments/dice-pick-demo.md`.
- **Vision platform** (`vision/` monorepo, 2026-07-10 → 2026-07-13) —
  dice-detector-v1's real-photo transfer collapse on d8/d10, fixed by the
  datagen-v2 close-up slice (mAP50 d8 0.090→0.442, d10 0.097→0.534),
  exposing a d6 glyph-confound regression in turn.
  `kb/wiki/concepts/vision-platform.md`.
