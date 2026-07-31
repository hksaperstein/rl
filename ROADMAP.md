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

0a. **AR4 PURE-FRICTION grasp — reach limit SOLVED, now BLOCKED by a persistent
   vertical-tracking (gravity-droop) gap (2026-07-30, ar4-reachable-friction-grasp
   task; supersedes the earlier reach-limit entry).**
   Fixed the prior blocker: found a genuinely-reachable **near-vertical** grasp
   config (tilt 5.55°, 36°+ margin on every joint) via the pure-FK
   `scripts/ar4_graspable_workspace.py` sweep, re-verified with the corrected
   12.5mm-local−Y pad geometry to put the pad **at the cube center 0.000mm in all
   3 axes (FK)**, and re-ran the pure-friction grasp there. **Reach limit gone —
   but a deeper blocker replaced it, pick still NOT achieved across 4 cloud
   runs.** Live, the empirical-Jacobian servo centers the HORIZONTAL plane well
   (X/Y to 1.8–3.7mm) but the pad **cannot be driven down to the cube's
   mid-height — it floats 11–16mm too high in Z every time**, so the jaws close
   on empty air and pure-friction contact is never established (**0N normal force
   every phase, all 4 runs; cube ground-truth never rises**). The vertical gap
   RESISTED every counter-measure: Cartesian pre-compensation (deeper target just
   droops more — 3.4mm achieved of 16 commanded), raising the cube (taller cube
   props the gripper up), and **60× arm-drive maxForce + 2.5× stiffness (cut droop
   only ~30%)**. That insensitivity to 60× force means it is **NOT simple drive
   saturation nor a stiffness deficit** — mechanism still open. **Escalated to
   Principal:** next step is a **joint-level tracking diagnostic** (commanded-vs-
   achieved per-joint angle + measured effort vs maxForce, gravity-off control) —
   NOT another end-to-end pick run — before choosing a fix (gravity-feedforward/
   integral joint control, a validated stiffer actuator model, or a grasp-assist).
   Cost ~$2.7 (4 runs), all instances torn down clean. Full detail + the 4-run
   measured table: `kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md`'s
   2026-07-30 "ar4-reachable-friction-grasp" UPDATE.

00. **AR4 GENUINE Isaac Sim physics grasp+lift — ACHIEVED 2026-07-30
   (ar4-isaacsim-standalone-pick task).** Closes the whole AR4-pick arc.
   Rebuilt the pick on the lower-level standalone Isaac Sim App API
   (`SimulationApp` + `isaacsim.core.api.World` + `SingleArticulation` +
   `World.step()`, NOT `ManagerBasedRLEnv`) — the layer the prior task
   (2026-07-29) identified as where the grasp mechanism actually works. A
   runtime PhysX fixed joint (link_6<->cube, `jointEnabled` flipped True at
   grasp — the EXACT toggle that failed under `ManagerBasedRLEnv`) engages
   and holds: cube physics z rose 0.0475→0.461 m and was held through the
   full retreat (ground-truth verified, NOT a kinematic weld). SurfaceGripper's
   manager also FIRES on this API (`status=Closed`, `grippedObjects=[Cube]`
   — vs. never under Isaac Lab), though a minimally-authored attachment point
   didn't physically hold (bounded follow-on: port the gantry example's full
   drive/limit schema). Video-confirmed
   (`logs/videos/ar4_isaacsim_standalone_pick/{closeup,elbow}.mp4`, also GCS
   `gs://rl-manipulation-hks-runs/ar4-isaacsim-standalone-pick/`). Script:
   `scripts/ar4_isaacsim_standalone_pick.py`. Cost ~$2.9, instance torn down
   clean. Full detail + the settled 4-stage arc + cloud/standalone-Isaac-Sim
   operational lessons: `kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md`'s
   2026-07-30 UPDATE.

0. **ROS2 + MoveIt pivot for AR4 grasping — RESOLVED POSITIVE 2026-07-28,
   from-scratch cloud path.** After the desktop hosting the vendor
   `ar4_ros_driver`/MoveIt stack was found unreachable (see the prior
   ar4-moveit-pivot entry below), a follow-up task stood up ROS2 Humble +
   MoveIt2 + the vendor AR4 stack from scratch on a fresh, ephemeral GCP
   CPU-only cloud instance (Docker + the prebuilt `moveit/moveit2:humble-release`
   image, per a mid-task user course-correction, not an apt-from-scratch
   install) and **MoveIt successfully planned and executed a full
   collision-aware pick** (approach → descend to grasp → close gripper →
   attach cube → lift/retreat → carry to a goal location) against a real
   cube+pedestal collision object, using the vendor MoveIt config's
   fake/mock hardware in RViz (the task's own pre-authorized Gazebo-physics
   fallback). Verified beyond a log line via `tf2_echo` (exact final-pose
   match) and `get_planning_scene` (cube genuinely attached). This directly
   supports the pivot's central premise — MoveIt's collision-aware planner
   is the right tool for the blockers the hand-rolled Isaac Sim approach
   hit. Honest caveat: the exact same recipe is NOT perfectly repeatable on
   re-run — later re-runs hit real numerical-IK-solver (`kdl_kinematics_plugin`)
   flakiness at the same step, not fully root-caused within this task's
   scope. Two real vendor-package/`ros2_control`-version-skew bugs were
   found and patched along the way (see
   `scripts/ar4_moveit_pick_demo/README.md`). Full detail:
   `kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md`'s 2026-07-28
   (later still, ar4-moveit-cloud-from-scratch task) UPDATE. Natural next
   step, not started here: root-cause the plan-ordering IK sensitivity, or
   try a different IK plugin (`trac_ik`/`pick_ik`) for better repeatability.

   **Direct follow-on, same day: real Gazebo PHYSICS grasp — pure friction
   FAILED, grasp-assist plugin WORKS.** Took the above RViz/fake-hardware
   pick (cube attached as a planning-scene bookkeeping trick, not a real
   grasp) onto the vendor's own `annin_ar4_gazebo` Gazebo Sim bring-up. The
   arm is genuinely physics-driven (verified via `/joint_states`), but pure
   friction (mu=0.8 both surfaces, real gravity) did NOT hold the cube —
   honest negative result, video:
   `logs/videos/ar4_gazebo_pure_friction_attempt_2026-07-28.mp4`. A
   grasp-assist `DetachableJoint` plugin (Gazebo's standard fallback for
   this well-known finicky physics, pre-authorized by the task and reported
   as such, not passed off as pure friction) DOES work, ground-truth
   verified via the cube's real physics pose rising from z=0.0475m to
   z=0.539m during lift+carry: video
   `logs/videos/ar4_gazebo_physics_pick_demo_2026-07-28.mp4`. Four more
   real vendor/upstream bugs found and fixed along the way (a genuine
   `gz_ros2_control` segfault on mimic-jointed robots, a weak-default-gain
   bug recurring in a new simulator via a new mechanism, a
   bullet-featherstone-specific crash trigger, and a URDF-fixed-joint-
   reduction gotcha for the grasp-plugin's own link resolution) — full
   detail: `kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md`'s
   2026-07-28 (later still, ar4-gazebo-physics-pick task) UPDATE. Natural
   next step, not started here: root-cause why pure friction failed
   specifically (mesh-collision jaw geometry vs. contact-solver tuning vs.
   genuinely insufficient jaw travel/force) rather than accepting the
   grasp-assist plugin as the permanent answer.
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

   **Jaw2 open-command sign bug FIXED and video-confirmed (2026-07-23,
   record-jaw-bug-video task) — real 28mm/0mm pincer open/close now
   working.** `tasks/ar4/robot_cfg.py`'s `GRIPPER_OPEN_COMMAND_EXPR`
   now commands `gripper_jaw2_joint` to the SAME signed value as jaw1
   (was negated); `scripts/build_asset.py` rebuilt with matching
   corrected hard limits. Live-recorded OPEN→CLOSE→OPEN cycle:
   0.028m/0.000m/0.028m separation, exactly the intended aperture.
   `kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md`'s
   2026-07-23 (record-jaw-bug-video task) UPDATE. A standing FK
   verification framework (`tasks/ar4/fk_verification.py`,
   `tests/test_ar4_fk_verification.py`) was also built the same day to
   catch this whole bug class automatically going forward.

   **Capstone classical-IK grasp+lift attempt (2026-07-23, this task):
   found the best kinematic configuration this entire investigation has
   ever measured (Z-shortfall 9-10mm, under the cube's own 12mm size,
   for the first time), but STILL NO WORKING GRASP+LIFT — a genuine,
   repeatable negative result, cloud cost cap overrun to ~$3.3 against a
   $2 cap.** Cloud-only (desktop unreachable), full from-scratch AR4
   asset rebuild on GCP (jaw2 fix + jaw2 drive + link_5/6 collision
   fixes all directly USD-verified present, not just trusted from exit
   code — build's own print-based confirmations are lost to stdout
   buffering when `SimulationApp.close()` force-exits). Two genuine SPOT
   preemptions in ~90 minutes forced a switch to on-demand pricing
   (documented judgment call, this project's own precedent) to stop
   losing wall-clock to restart/rerun cycles — this, combined with the
   exploratory search needed, is why the cost cap was exceeded; flagged
   here plainly rather than smoothed over. Added `--tilt-sweep` to
   `scripts/grasp_demo_v2.py` (mirrors the existing sweep flags) and
   swept tilt 0-90° at the reach distance (0.39m) already known to have
   healthy `joint_3` margin: **0-18° reproduces the same flat/negative
   ~20mm shortfall found on 2026-07-22 (no improvement, confirms this
   isn't specific to the old tight-margin position) — but 25-90°
   reveals a genuine, real local minimum at ~65° tilt (Z-shortfall
   drops smoothly from ~20mm at 25° to 9.3mm at 65°, then rises again
   toward 90°)**, the first time this whole investigation's residual has
   ever dropped below the cube's own size. A follow-up reach sweep AT
   this 65° tilt found the improvement holds across 0.30-0.36m reach
   (~9.3-9.5mm, healthy joint margins throughout) before degrading past
   0.39m. **Three full phased grasp+lift attempts at this configuration
   (reach 0.30/0.36/0.39m, all 65° tilt, real video + per-phase jaw
   contact-force logging), all show the identical negative signature:
   cube z stays EXACTLY at its 0.0060m resting height through every
   phase (no ambiguity requiring video review - the ground-truth number
   itself is flat), jaw1 registers a brief light one-sided contact force
   (0.03-0.34N) while jaw2 registers exactly 0.0000N throughout, and the
   cube gets nudged a few mm sideways rather than enclosed.** Videos
   synced to the Pi (`logs/videos/ar4_grasp_demo_v2_pos{1,2,3}_r0*_t65*.mp4`).
   Cube-parking and gripper joint-position logging (claimed committed on
   2026-07-23 but the diff never actually included cube-parking - a
   real doc/code discrepancy found and fixed this session) are now both
   genuinely implemented and exercised live for the first time.
   **Follow-up (2026-07-24, ar4-jaw-bisector-hypothesis task): the
   `_EE_OFFSET`-vs-true-bisector hypothesis this session flagged is
   REFUTED, live-measured at the exact converged 65°-tilt/reach=0.30m
   `grasp_q`** — discrepancy is 0.0002mm (vs. 0.0001mm at a near-vertical
   baseline), utterly negligible against the ~9-13mm real residual; a
   pure offline FK-model check (zero cost) had already predicted this
   (the offset is a rigid, arm-orientation-independent quantity by
   construction). A real but small (0.66mm) jaw-to-cube distance
   asymmetry exists but doesn't cleanly explain jaw1-only contact either
   (jaw2 — the one with zero contact force — is actually the CLOSER
   jaw). Most likely mechanism: `GRASP_Q`'s own real ~4.2° residual
   rotation error, on top of the ~9.5mm position error, at a ~19mm
   jaw-to-cube reach — enough to shift which side of the cube's 12mm
   face each fingertip clears, independent of raw distance-to-center.
   No new asset/command-level bug found (offset math and jaw open/close
   tracking both confirmed correct); this is the same already-
   characterized joint_3/reachability-envelope precision ceiling
   surfacing as a directional/antipodal problem now that the 65° tilt
   fixed the gross-miss half. Two real, separate cloud-infra bugs found
   and fixed along the way (a `set -e`/pipefail trip in the new AR4
   cloud-build script, and a FIFO-reader-lifetime bug in
   `scripts/run_on_cloud_gpu.sh` itself that could silently kill a
   dispatch's log stream mid-run) — both now fixed and benefit every
   future cloud dispatch. Full detail:
   `kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md`'s
   2026-07-24 UPDATE. Next step (not this task's to decide): either a
   genuinely different approach-orientation methodology (Tier 1 gate),
   or treat this as AR4's practical classical-IK precision ceiling and
   revisit priority against the already-working Franka platform.

   **Convergence-tightening + narrow neighborhood sweep (2026-07-24,
   ar4-grasp-ik-convergence-tightening task): CONFIRMED this is a genuine
   local-optimum floor, not a solver-budget limitation, and no nearby
   configuration resolves it either — a clean, direct negative result
   closing off both remaining "just push the solver harder" candidates.**
   New `--grasp-deep-polish-steps`/`--grasp-pos-threshold`/
   `--grasp-rot-threshold` flags (`scripts/grasp_demo_v2.py`, commit
   `bc466a2`) ran one extra polish pass at GRASP's own target with 15x the
   iteration budget and a 3-5x tighter convergence bound than any prior
   pass at reach=0.36m/tilt=65° (the middle of the known 0.30-0.36m
   plateau) — the pass diverged to a WORSE, stable local optimum within
   100 steps (9.4mm/5.7° → 16.4mm/26.9°, frozen there for 1500+ steps) and
   had to be rescued by the existing keep-best guard, landing back at
   9.4mm/6.7° — more solver effort makes things worse here, not better. A
   full phased grasp+lift attempt at the restored config again failed:
   `jaw1_cube_force`/`jaw2_cube_force` both EXACTLY `0.0000N` throughout
   (zero contact on BOTH jaws this run — not even the prior session's
   brief one-sided contact), cube z flat at its `0.0060m` resting height,
   no lift. A follow-up narrow neighborhood sweep (7 tilts 60-70° at
   reach=0.36m fixed, then 4 reaches 0.32-0.38m at tilt=65° fixed — cheap,
   sweep-only launches) found a genuinely shallow, broad plateau: position
   residual flat within ~1.5% (9.38-9.52mm) across the whole 64-68°/
   0.32-0.36m neighborhood, with only sub-1.5° rotation-residual
   improvements at 66° tilt (4.83° vs 65°'s 5.66°) or 0.32-0.34m reach
   (4.26-4.36°) — real but not qualitatively different; 0.38m reach
   clearly falls off the plateau's edge (15.64mm/23.6°), consistent with
   the capstone session's own "degrades past 0.39m" finding. **Honest
   assessment: this specific cube/table/arm-mount geometry has reached a
   genuine, hard-to-avoid classical-scripted-IK precision ceiling at this
   reach/tilt range** — not a solver-tuning or nearby-configuration
   opportunity. Two SPOT preemptions hit mid-session but were recovered
   via `gcloud compute instances start` on the same persisted boot disk
   (no asset rebuild needed either time); cost ≈$0.91 against the $2 cap.
   Full detail: `kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md`'s
   2026-07-24 (later) UPDATE. **Next step flagged to Principal, not
   decided here**: a genuinely different approach-orientation methodology
   (per-waypoint rather than PREGRASP-shared orientation, or an untested
   bearing at this newer tilt/reach range) is real Tier-1-gated
   methodology work, not a bug fix — or treat this as AR4's practical
   classical-IK ceiling and prioritize the already-working Franka
   platform per the North Star.

   **Jaw collision-mesh asymmetry FOUND AND FIXED (2026-07-24 later,
   ar4-jaw-contact-sensor-hypothesis task) — a real, quantified geometric
   defect, but fixing it did NOT resolve the grasp problem; contact-sensor
   hypothesis REFUTED (sensor genuinely working).** Direct extraction of
   both jaws' actual collision-mesh points found jaw2's mesh is missing
   exactly the bottom 2.8mm of its own z-range vs jaw1's (1782 vs 1866
   points, 4.1% smaller hull volume) — a real vendor-STL truncation
   (`gripper_jaw2_link.stl` is its own separate file from
   `gripper_jaw1_link.stl`, not a mirrored reference), not previously
   quantified. Fixed: new `_fix_jaw2_collision_mesh_asymmetry()` in
   `scripts/build_asset.py`, copying jaw1's mesh onto jaw2's, transformed
   into jaw2's local frame (two real USD bugs hit and fixed getting this to
   actually persist: `Gf.Vec3fArray` doesn't exist, and authoring onto an
   instance proxy required permanently disabling `instanceable` on the
   instance-root ancestor — a first attempt that restored it afterward
   silently undid the whole fix). Post-fix, raw point/face counts and hull
   volume/area now match to floating-point noise. Contact-sensor config
   confirmed structurally identical between jaw1/jaw2, and a live post-fix
   run independently reproduced a REAL sustained (60/60 steps) nonzero
   contact force (jaw2=0.027N, jaw1=exactly 0.0000N at reach=0.36m/tilt=65°)
   — proving the sensor mechanism itself works correctly (a broken sensor
   wouldn't sporadically produce sensible, run-appropriate readings). **But
   the fix did not produce a working grasp**: same configuration's own
   grasp_residual (9.65mm/5.01°) is essentially unchanged from every
   pre-fix measurement, and the cube still isn't lifted (nudged ~3mm only)
   — reinforcing, not overturning, the already-established conclusion that
   the ~9-10mm/~4-7° local-optimum-floor residual (immediately above) is
   the actual dominant blocker, with the residual rotation error (not jaw
   geometry) deciding which single jaw happens to register contact. Full
   detail: `kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md`'s
   2026-07-24 (later, ar4-jaw-contact-sensor-hypothesis) UPDATE — includes
   a reproduced, unresolved `run_on_cloud_gpu.sh` blocking-preemption-retry
   bug (worked around via `--detach`, not fixed) and a visual-confirmation
   gap (existing cameras don't resolve the 12mm cube clearly enough;
   numeric contact-force evidence used instead, per this project's own
   Experiment-16 verification standard). Cost ≈$0.6 against the $2 cap.

   **Full arm-chain FK check (2026-07-24 later still, ar4-arm-chain-fk-check
   task): the ARM's own kinematic chain (link_1..link_6, distinct from the
   already-verified gripper) checks out CLEAN against the vendor URDF —
   rules out an asset-import defect as the cause of the standing residual,
   and the canonical antipodal-grasp target definition is independently
   confirmed mathematically sound too.** Extended the standing FK-
   verification framework's Layer 1 check
   (`tasks/ar4/fk_verification.py`, previously only exercised against the
   gripper jaws) across every arm link at HOME_Q, the actual best-known
   converged PREGRASP_Q/GRASP_Q (65°/0.36m), and a synthetic all-joints-
   nontrivial stress config, at a tight 1.0mm tolerance. **Result: PASS at
   every link/config — largest discrepancy 0.0003mm (four orders of
   magnitude below the ~9-10mm residual)**, ruling out a wrong joint
   origin/axis anywhere in the chain as the explanation. Also directly
   verified `scripts/grasp_demo_v2.py`'s canonical target-orientation math
   is a geometrically sound antipodal-grasp target (jaw-slide axis fixed at
   world +Y regardless of tilt, passing exactly through the cube's own
   live-read center; cube spawns world-axis-aligned with no orientation
   randomization in this env cfg) — no defect found there either. **With
   the gripper geometry, contact-sensing pipeline, arm kinematic chain, and
   grasp-target definition all now independently verified correct, this
   investigation has exhausted every asset/target-definition hypothesis it
   has generated** — the ~9-10mm/~4-7° residual is a genuine classical-
   scripted-IK precision/local-optimum limit at this arm's own
   joint_3-constrained low-grasp-height reachability envelope, not a
   findable bug. New `scripts/_verify_arm_chain_fk_integration.py`
   (kept, standing check). Two cloud-infra snags hit and worked around
   (an uncommitted-files shipping gap, a camera-enabled scene hanging on
   RTX warmup — fixed by dropping the unneeded cameras from the check's
   scene). Cost ≈$0.38 against the $2 cap. Full detail:
   `kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md`'s 2026-07-24
   (later still) UPDATE. **Next step flagged to Principal, unchanged**:
   invest in a genuinely different orientation/grasp-planning methodology
   (Tier 1 territory) or treat this as AR4's practical classical-IK
   ceiling and prioritize the already-working Franka platform per the
   North Star.

   **Vertical-approach + fixed-gripper recheck (2026-07-24 later,
   ar4-vertical-fixed-gripper-recheck task, direct user request): reproduced
   `f9bde3e`'s own position-only-IK vertical config against the now-fully-
   fixed gripper — gets real SIMULTANEOUS two-sided jaw contact for the
   first time in this investigation, but still no lift.** Positioning
   precision reproduced in the same ballpark (`pregrasp=1.81mm` vs.
   `f9bde3e`'s 1.8mm; `grasp=15.09mm` vs. 10.5mm, same basin, normal
   run-to-run variance) and the true ~9mm grasp height WAS reached —
   confirmed via real physical contact, not just a low residual: at
   CLOSE, both jaws registered real, sustained, simultaneous nonzero force
   (jaw1≈0.11-0.12N, jaw2≈0.15-0.17N across the full phase), breaking this
   investigation's standing "exactly one jaw always reads 0.0000N"
   asymmetric signature for the first time. But contact drops to exactly
   `0.0000N` on both jaws the moment the arm retreats toward PREGRASP and
   stays there — the cube ends up dragged ~13.5mm across the table, z
   rising only 2.4mm above resting, nowhere near a real lift. This
   directly confirms, rather than overturns, `f9bde3e`'s own original
   diagnosis ("a grasp-ORIENTATION gap — position-only IK has no incentive
   to pick a sensible pinch geometry"). **Resolves the tension the task set
   out to test**: `command_type="pose"` (orientation-locked) provably can't
   reach the true height under a vertical wrist (joint_3 limit, unchanged);
   `command_type="position"` (unlocked) reaches the height with real
   contact but can't hold an orientation through retreat. Neither alone
   produces a grasp+lift — combining both (a position search additionally
   scored to reject non-antipodal orientations, or residual RL on top of
   one of these classical seeds) is Tier 1 methodology work, flagged to
   Principal rather than decided here. Full detail:
   `kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md`'s 2026-07-24
   (later) UPDATE. New `scripts/_verify_vertical_position_ik_fixed_gripper.py`
   kept as a historical record.

   **Cube size bumped 12mm->20mm (2026-07-24 later still,
   ar4-cube-size-increase task, direct user decision): primary
   (visual-confirmation) goal ACHIEVED cleanly; bonus grasp check found a
   better residual but still no contact/lift.** `CUBE_CFG`
   (`tasks/ar4/objects_cfg.py`) and every derived 12mm/0.006m constant
   found via repo-wide grep updated across both the active classical-IK
   path and closed/legacy RL-training env cfgs (full list, live spawn
   verification, and the grasp-check numbers: kb doc's 2026-07-24 (later
   still, ar4-cube-size-increase) UPDATE). Live spawn check: cube rests
   at z=0.010m, 0.000mm error. **Visual goal achieved**: a single grasp
   attempt (reach=0.36m/tilt=65°) with the existing but never-fully-
   exercised `--closeup-camera` mechanism produced a video frame showing
   the cube as a large, sharp, unambiguous square directly under the jaw
   fingertips — closing the standing "neither camera resolves the 12mm
   cube clearly" gap flagged repeatedly above. Bonus finding (one
   attempt only, not investigated further per explicit instruction):
   `grasp_residual=6.12mm/5.30°` — genuinely tighter than the 12mm cube's
   own best-ever capstone residual (~9.4mm/5-6°) at the identical config,
   plausibly because the grasp-height target rose with the cube
   (0.009m->0.013m, more `joint_3` margin) — but `jaw1_cube_force`/
   `jaw2_cube_force` both read exactly `0.0000N` throughout every phase,
   no lift. Not treated as a failed grasp re-investigation; the task's
   own corrected priority was visual confirmation, not grasp precision.

   **Locked-achieved-orientation synthesis test (2026-07-24 later still,
   ar4-locked-achieved-orientation-grasp task, direct instruction): the
   combination the prior two findings above called for — still no
   lift, but the result sharpens the diagnosis rather than reopening it.**
   Built `scripts/_verify_locked_achieved_orientation_grasp.py`: reach the
   true grasp height + real contact via position-only IK exactly as the
   vertical-recheck task above did, then AT THE MOMENT real bilateral
   contact is detected (watched every physics step, not sampled), read the
   arm's LIVE wrist orientation and lock that (not a re-derived canonical
   target) via a genuine closed-loop `command_type="pose"` DLS controller
   (ported from `grasp_demo_v2.py`'s own proven `polish_from_seed`) for
   the rest of CLOSE, retreat, and hold. Run twice, at two different
   retreat speeds, to separate "orientation drifted" from "orientation
   was fine but the grasp was never good." **Fast retreat (0.03m/step, the
   same bound used for reaching a static target elsewhere in this
   codebase)**: orientation held to <0.003rad the whole time (proving the
   lock mechanism itself works), gripper cleanly reached the hover pose to
   2.6mm — but contact still dropped to exactly `0.0000N` on both jaws at
   the very first retreat step and the cube was never moved at all,
   `height_gain=0.0000m`. **Slow retreat (0.001m/step, added specifically
   to rule out "too fast a yank" as the cause)**: contact never dropped to
   zero, but the arm made ZERO net progress toward the hover position for
   the entire 270-step retreat+hold+release window (frozen at the exact
   same 29.5mm residual throughout) while jaw2's force grew to 1.5-1.8N —
   3-4x any single-jaw force ever recorded in this investigation, and
   markedly asymmetric against jaw1's 0.4-0.5N — i.e. the arm got
   mechanically stuck fighting the table rather than lifting. **Diagnosis:
   locking the achieved orientation is necessary but not sufficient** —
   rotation genuinely does not drift under this lock (confirmed
   numerically in both runs), so the failure isn't drift; it's that
   position-only IK's null-space-selected contact orientation produces
   real bilateral NORMAL force (enough to register as "contact") without
   being a force-balanced antipodal pinch, so it has no shear/lift
   resistance regardless of retreat speed. Full detail, both runs' full
   numbers: `kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md`'s
   2026-07-24 (later still, ar4-locked-achieved-orientation-grasp)
   UPDATE. **Next step unchanged, now on firmer evidence**: a mechanism
   that scores/selects the contact orientation itself for force-balance
   (not just position) is Tier 1 methodology work, flagged to Principal;
   otherwise this is AR4's practical classical-IK ceiling and the
   already-working Franka platform remains the North-Star priority.

   **Pinch-point geometry at the contact moment (2026-07-25,
   ar4-pinch-point-geometry-at-contact task, dispatched off a user visual
   read of the video above): confirms and sharpens the diagnosis further
   - the true fingertip midpoint is measurably OUTSIDE the cube's own
   volume at the achieved orientation, not an offset-math bug.** Two
   questions kept separate per the coordinator's explicit framing: (Q1)
   is `_EE_OFFSET` still accurate at this specific achieved orientation -
   yes, to ~1mm (a real but small, secondary, orientation-dependent effect,
   vs. 0.0001-0.0002mm at two previously-tested orientations - not a
   frame-transform bug, the code already rotates by the live orientation
   every step); (Q2) is the achieved orientation itself sane, using the
   TRUE bisector as the pinch point - no: the true bisector sits 14.2mm
   outside the cube's own face along one axis (cube local-frame coords
   `[10.8, 24.2, 9.2]mm`, half-size 10mm), traced to the same
   already-documented `joint_3` height-shortfall (achieved height 21.9mm
   vs. 9mm intended). The user's literal "gripper base resting on the
   cube" hypothesis was directly checked and refuted by measurement (base
   is 63.2mm from the cube, over 2x FARTHER than the 28.0mm fingertip
   bisector) but the qualitative visual read was right in substance - a
   corner/edge catch, confirmed via direct frame extraction (cube nearly
   fully occluded behind one jaw block 1s into CLOSE, not centered between
   two). No fix applied (no offset bug exists to fix); the Tier-1
   recommendation is unchanged and now on firmer, more direct evidence:
   `kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md`'s 2026-07-25
   UPDATE for full numbers.

   **Axis-alignment reduced-DOF IK (2026-07-24/25, ar4-axis-align-ik task):
   a genuinely new IK formulation (position(3) + approach-axis-DIRECTION(2)
   = 5 constraints, 1 redundant DOF, vs `position` mode's 0 orientation
   constraints or `pose` mode's 3/0-redundant-DOF) — orientation control
   CONFIRMED FIXED (genuinely vertical, 0-1° off, at both tested positions,
   vs `position` mode's uncontrolled 18-72°+ tilt), but reachability is
   NOT: position residual still ~15mm at the true grasp height at both
   tested positions (0.275m/0.32m reach), via TWO DIFFERENT mechanisms
   (a real `joint_3` hard-limit wall at 0.275m; a joint-limit-free DLS
   local-optimum plateau at 0.32m) - the extra redundant DOF did not
   reliably deliver the hoped-for routing-around-the-elbow-limit benefit.
   The resulting phased CLOSE+RETREAT produced the STRONGEST, most
   sustained bilateral contact this whole investigation has ever recorded
   (7-40N, vs typically <1N) with a real, stable, 100+-step height-gain
   hold (21.5-21.6mm) - but direct video inspection plus a decisive
   cross-position tell (near-identical final held height, 31.5mm vs
   31.6mm, despite two DIFFERENT targeted reach positions) shows this is a
   NEW wedge-against-the-gripper's-own-base-housing artifact, not a
   genuine antipodal pinch - the gripper's base, not its fingertips, is
   catching the cube because of the still-unresolved ~15mm shortfall. Real
   Jacobian math verified correct first (finite-difference check, 2-5e-5
   error after fixing a dynamics-settle-noise confound in the verification
   method itself), a real side-finding of a small pre-existing (not
   introduced here) ~0.007-0.009 bias in the unrelated base position
   Jacobian flagged but not fixed (out of scope). **Does not close the
   investigation** - orientation-control is a genuine, real fix; the
   reachability half of the hypothesis is falsified as tested. Full
   detail: `kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md`'s
   2026-07-24 (later still, ar4-axis-align-ik task) UPDATE.

   **MoveIt-vs-our-DLS root cause (2026-07-27, ar4-moveit-vs-dls-root-cause
   task): the "wrong solver family" premise is REFUTED — the vendor's own
   MoveIt config uses plain `kdl_kinematics_plugin/KDLKinematicsPlugin`
   (confirmed from the real `github.com/ycheng517/ar4_ros_driver` repo's
   `annin_ar4_moveit_config/config/kinematics.yaml`), not TRAC-IK and not
   an analytic IKFast solver (zero occurrences of either anywhere in the
   repo) — and MoveIt2's actual plugin source shows this KDL solver is
   itself a hand-written Newton loop around an SVD/DLS-style Jacobian
   pseudo-inverse (`ChainIkSolverVelMimicSVD`), the SAME algorithm family
   this project's own `DifferentialIKController` already uses, with
   joint-limit clamping every iteration (our own polish loops already do
   this too) and a random-restart across `kinematics_solver_attempts: 3`
   (broader than our own curated seed list, but budgeted at a mere 0.005s
   timeout). **The best-evidenced real difference is structural, not
   algorithmic**: MoveIt solves IK against a pure kinematic model (no
   physics) and only executes the resulting trajectory afterward, cleanly
   separated from dynamics — and this project's OWN prior sessions already
   independently hit and fixed physics-coupling confounds with nothing to
   do with solver choice (the weak-arm-actuator 1.42rad tracking error,
   the gripper mimic-vs-actuator PhysX conflict, the axis-align-ik task's
   own dynamics-settle-noise bug in Jacobian verification, fixed by
   switching to a pure-FK `env.sim.forward()` refresh — in effect this
   project's own prior discovery of MoveIt's own plan-then-execute
   pattern, just not previously framed that way). Ranked recommendation:
   (1, highest confidence) decouple IK search from live physics using the
   already-validated `sim.forward()` pure-FK pattern, execute only after
   converging; (2, medium) broaden the seed search toward true
   random-restart (cheap, but this investigation's own bearing/reach/tilt
   sweeps suggest it's not the dominant fix); (3) accept the vendor-real
   `joint_3` -89/+52° limit may simply make a fully-vertical 9mm-height
   grasp impractical for this arm regardless of solver, a task-design
   question not a solver bug; (4/5, ruled out) IKFast/TRAC-IK adoption,
   since neither is what makes the vendor's own tooling work. No live
   proof-of-concept run this task (desktop unreachable throughout,
   confirmed via `ssh`/`avahi-resolve`/the GPU status server all timing
   out identically) — flagged as the concrete next step. Full detail:
   `kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md`'s 2026-07-27
   UPDATE.

   **UPDATE (2026-07-28): the follow-on ground-plane-collision finding
   (below, joint2-ground-clearance-fix task) and its own follow-on pedestal
   fix (see "Recently landed" above) ARE that concrete next step, already
   acted on** — the cube was raised onto a 40mm pedestal exactly as that
   task's own "concrete implication" flagged, the workspace re-derived
   (twice, after finding and fixing a real bug in the first re-derivation),
   and a live grasp+lift re-attempted twice. The ground-collision blocker
   is now closed and verified; the remaining blocker is the
   roll/heading-residual-contact question (already independently documented
   below, 2026-07-27 "ar4-graspable-roll-constraint" and 2026-07-24
   "ar4-locked-achieved-orientation-grasp" UPDATEs) — see "Recently landed"
   for the full current-state writeup.

   **UPDATE (2026-07-28, later): a Cartesian fingertip-correction fix was
   tried as an alternative explanation for the residual (see "Recently
   landed" above, ar4-cartesian-fingertip-correction task) and REFUTED as
   the mechanism — even a point where the correction achieved real
   progress (Q2, ~2x residual reduction) still produced zero lift.** This
   strengthens rather than replaces the roll/heading diagnosis above: the
   concrete next step for whoever picks this investigation back up is now
   specifically the grasp-ORIENTATION-search question (tightening
   `ROLL_TOL_DEG` below its current 12°, or redesigning how a grasp
   orientation is selected so it's genuinely antipodal rather than merely
   within-tolerance) — not another position-precision fix, since two
   independent position-precision mechanisms (joint-space tracking,
   Cartesian fingertip correction) have now both been tried and both left
   the identical "contact collapses to exactly 0N the instant LIFT-CLOSE
   begins" failure signature untouched.

   **UPDATE (2026-07-28, later still, ar4-grasp-trivial-friction-check
   task): direct visual inspection (see "Recently landed" above) gives a
   more visually direct mechanism than roll/heading alone — the jaws are
   landing at the cube's TOP EDGE, not its side faces, at all 3
   previously-tested pedestal points.** Not a contradiction of the
   roll/heading evidence (an edge-only, non-antipodal contact is entirely
   consistent with everything found so far) but a sharper, now-visually-
   confirmed candidate for the FIRST thing to fix: lower `GRASP_AT_HEIGHT`
   (currently pedestal-top + 0.0105m, an unchanged carryover from the
   pre-pedestal ground-level convention) so the real fingertip — already
   known to sit ~15-18mm below the abstract pinch point the convention was
   originally tuned against — lands at the cube's actual vertical center
   instead of 0.2mm from its top face. This is the concrete next step,
   flagged rather than applied (outside the trivial-check task's own
   bounded scope) — cheaper and more directly evidenced than jumping
   straight to a `ROLL_TOL_DEG`/orientation-search redesign.

See `BACKLOG.md` for further-out candidates not yet on this list.

## Recently landed

- **Native Isaac Sim AR4 pick — trajectory execution WORKS, native
  grasp-assist BLOCKED in `ManagerBasedRLEnv`** (2026-07-29,
  ar4-isaacsim-curobo-pick task). Replicated the MoveIt/Gazebo recipe
  natively: proper multi-step interpolated trajectory execution through
  the articulation controller drove a collision-free
  `HOME→PREGRASP→GRASP→LIFT→RETREAT` with precise tracking — confirming
  (yet again) that planning/control is NOT the AR4 Isaac Sim blocker. cuRobo
  was a genuine wall (no `nvcc`/CUDA toolkit in the `isaac-lab:2.3.1`
  container) → Isaac-native fallback per authorization. The real wall: the
  grasp-HOLD. **Five distinct native grasp-assist configs all failed under
  Isaac Lab `ManagerBasedRLEnv`** — `SurfaceGripper` manager never registers
  (subscribes to physics-step events `env.sim.step()` doesn't fire); USD
  fixed joints don't attach an external `RigidObject` to an articulation link
  (runtime-create, runtime-toggle, pre-authored-active, GPU and CPU physics).
  Root cause is the Isaac Lab abstraction (physics views built once at reset,
  no runtime joint/topology re-read), NOT Isaac Sim physics. A labeled
  kinematic pose-follow weld demonstrated the full pick visually (cube
  0.0475→0.464 m, held through retreat; closeup video shows the cube gripped
  between the jaws), flagged explicitly as pose-driven, not a physics grasp.
  **Next step:** rebuild on the lower-level Isaac Sim standalone App API
  (`World`+`SingleArticulation`+`World.step()`, where NVIDIA's own
  SurfaceGripper example works) or bridge MoveIt over ROS2; bundled Lula RRT
  is the Isaac-native collision-aware planner. Full detail:
  `kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md`'s 2026-07-29
  UPDATE; artifact `scripts/ar4_isaacsim_surfacegripper_pick.py`; video
  `logs/videos/ar4_isaacsim_surfacegripper_pick/` (cost ≈$3.6, instance torn
  down + verified clean).
- **ROS2+MoveIt2 collision-aware AR4 pick, from scratch on a fresh cloud
  instance: SUCCEEDED** (2026-07-28, ar4-moveit-cloud-from-scratch task).
  MoveIt planned and executed a full pick (approach → descend → grasp →
  attach → lift/retreat → carry to goal), verified via `tf2_echo` and
  `get_planning_scene`, not just log lines — directly supporting the
  platform pivot's premise that MoveIt's collision-aware planner solves
  the hand-rolled approach's descent-collision/grasp-sequencing blockers.
  Two real vendor-package/`ros2_control` version-skew bugs found and
  patched along the way. Honest caveat: not perfectly repeatable on
  re-run (numerical-IK-solver flakiness at the same step, not fully
  root-caused). Full detail:
  `kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md`'s 2026-07-28
  (later still, ar4-moveit-cloud-from-scratch task) UPDATE; artifacts at
  `scripts/ar4_moveit_pick_demo/`; video at
  `logs/videos/ar4_moveit_cloud_pick_demo_2026-07-28.mp4`.
- **AR4 pedestal-grasp-height-fix: the height fix WORKS (real fingertip now
  genuinely lands inside the cube's vertical span, not jammed at the top
  face), but the capstone grasp+lift still fails — root-caused to a NEW,
  distinct blocker: the still-OPEN gripper collides with the cube DURING
  the PREGRASP→GRASP descent itself, pinning the whole arm+gripper
  assembly before CLOSE is ever commanded** (2026-07-28, direct
  continuation of the trivial-friction-check task's own recommended next
  step). Lowered the FK-design grasp-height target
  (`scripts/ar4_graspable_workspace.py`'s `GRASP_AT_HEIGHT` convention) to
  compensate for a measured ~4.3mm physics-vs-kinematics tracking bias,
  re-derived a corrected joint config via a Pi-local pure-FK local search
  (no GPU/Isaac needed), and confirmed via video that the real fingertip
  now lands genuinely inside the cube's vertical span (+6-7mm from center,
  well clear of the old top-face-jamming bug). **But `jaw_separation`
  stays frozen at ~28mm (fully open) through the whole CLOSE phase across
  3 independent live runs** — two hypotheses for this were tested and
  BOTH REFUTED with hard numbers: (1) insufficient gripper actuator force
  — refuted by boosting `effort_limit_sim` 20N→100N (5x the observed
  ~55-60N contact force) with zero change in outcome; (2) a command-wiring
  bug — refuted by a raw `joint_pos`/`joint_pos_target` readout showing
  the CLOSE target correctly reaches the joint (`target=0.0`) while the
  actual position never moves off `0.014` (fully open) even a micron, with
  the calculated actuator spring force at that gap (`stiffness × error` =
  4000 × 0.014 = 56N) closely matching the measured ~58-60N contact force
  — strong quantitative evidence of a genuine physical wedge/obstruction
  already present before CLOSE is even issued (`open_gripper_max_force`
  pre-CLOSE = 60.4N), not a software bug. The arm's own joint-space settle
  also fails to converge at this config (2.3deg residual, ~13mm
  FK-vs-achieved pinch discrepancy — much larger than this investigation's
  previously-documented ~5-7mm residual), consistent with the arm getting
  physically stuck mid-descent rather than cleanly reaching its intended
  pose. Next concrete, bounded step (flagged, not decided unilaterally):
  the corrected joint search only validates the two static PREGRASP/GRASP
  endpoints, not the interpolated path `settle_to_joint_pose` actually
  traverses between them — checking/avoiding collision along that
  intermediate path (or requiring a larger XY margin) is the likely fix,
  but redesigning the search methodology is outside this task's own
  bounded "lower the height and verify" scope. Full detail:
  `kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md`'s 2026-07-28
  "ar4-pedestal-grasp-height-fix task" UPDATE.
- **AR4 grasp-trivial-check: friction DEFINITIVELY ruled out (code +
  live empirical retest with realistic friction applied); direct visual
  inspection instead confirms the real cause is grasp HEIGHT — jaws close
  onto the cube's TOP EDGE, not its vertical middle** (2026-07-28, direct
  user instruction to stop building diagnostic machinery and check the
  trivial causes by direct observation). Cube friction was never near-zero
  (scene-wide `mu=1.0`, already higher than Franka's own working default
  of 0.5) — confirmed by code inspection AND by making it an explicit,
  realistic wood/plastic/resin material (`static=0.8/dynamic=0.7`,
  `tasks/ar4/objects_cfg.py`'s `CUBE_PHYSICS_MATERIAL`, a real committed
  fix per direct user decision) and re-testing live: **the grasp still
  fails identically** (0.00mm height gain at every phase, contact collapses
  to exactly 0.000N the instant LIFT begins). Close-up video/frames
  (`outputs/ar4_pedestal_grasp_trivial_check/`, camera repositioned live
  from the gripper/cube's own post-approach position) directly show the
  jaws descending onto the cube's upper portion with much of the cube
  visible below their reach, `jaw_separation` frozen at ~28mm from OPEN
  through CLOSE (no real closing motion), and — the single clearest frame
  in this whole investigation — the gripper closed on empty air while the
  cube sits completely undisturbed on the pedestal once the arm retreats.
  Height fix itself (lowering `GRASP_AT_HEIGHT` so the real fingertip lands
  at the cube's vertical center instead of its top face) is flagged as the
  concrete next step, not applied here (outside this task's own bounded
  "check the trivial causes" scope). Full detail:
  `kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md`'s 2026-07-28
  "ar4-grasp-trivial-friction-check task" UPDATE.
- **AR4 Cartesian fingertip correction — built and tested as designed, but
  does NOT close the capstone grasp: directly confirms, rather than
  overturns, the roll/heading diagnosis** (2026-07-28, direct continuation
  of the pedestal-fix task below). Built
  `tasks/ar4/joint_tracking.py`'s new `settle_to_cartesian_pose` (a DLS
  outer-loop that nulls the REAL fingertip's Cartesian error, not just
  joint-space error) to test the hypothesis that the pedestal-fix task's
  remaining ~6-7mm pinch-point residual was a joint-space-vs-Cartesian
  lever-arm gap, independent of the already-flagged roll/heading issue.
  Live-tested at the same 3 pedestal-corrected validation points: the
  correction did NOT converge to its own <2mm target at any point within 8
  iterations (Q0: 5.585→5.044mm, Q1: 5.287→5.243mm barely moved, Q2:
  7.376→3.544mm, best case) - but more importantly, even Q2 (the point
  with the largest real improvement, plus a strong-looking two-sided
  PHASE3-CLOSE contact ~64N/~58N) still produced **exactly 0.00mm height
  gain**, with contact collapsing to 0.000N within 20 steps of LIFT-CLOSE
  beginning - identical to every other point and every prior attempt.
  **Honest verdict: this task's own hypothesis (fixable Cartesian position
  gap) is not supported by the evidence; the pedestal-fix task's diagnosis
  (jaw-vs-cube roll/heading misalignment preventing a genuinely antipodal
  grip) stands as the real blocker.** The capstone AR4 grasp+lift remains
  NOT achieved. A secondary, unexplained finding: the new primitive itself
  plateaus rather than converging at these configs, with `joint_5` showing
  the same "correction grows without closing the gap" signature previously
  seen on `joint_2` at a different pose - flagged for whoever extends it,
  not chased further here. Also surfaced a real, reproducible tension
  between this project's "blocking foreground cloud dispatch" convention
  and a job whose own duration (~23min to the teardown hang) exceeds one
  tool call's ~10min timeout ceiling (worked around via `--detach` +
  manual polling/teardown this task, not yet resolved as a standing
  pattern) - see `kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md`'s
  2026-07-28 "ar4-cartesian-fingertip-correction task" UPDATE for full
  detail, including a partial-log-loss infra note (the container `docker
  run` invocation still lacks `PYTHONUNBUFFERED=1`, unlike an already-fixed
  sibling script).
- **AR4 pedestal fix: the ground-collision problem is genuinely SOLVED and
  verified — but a different, already-documented issue (jaw-vs-cube
  roll/heading contact) still blocks a real grasp+lift** (2026-07-28,
  direct continuation of the joint2-ground-clearance-fix task below).
  Added a static 40mm pedestal (`tasks/ar4/objects_cfg.py`'s
  `PEDESTAL_CFG`/`make_pedestal_cfg`, a plain `AssetBaseCfg`+`CuboidCfg`
  static collider) and raised the cube to rest on top of it
  (`Ar4PickPlaceGraspGoalSceneCfg`) — mirrors real pick-and-place (object
  on a raised surface, not the floor at the robot's own base). Re-derived
  the graspable workspace at the pedestal-corrected height: genuinely
  NON-EMPTY (30157 survivors), a dramatic reversal of the prior task's 0.
  **First live grasp attempt at 3 new validation points still failed
  identically to the pre-pedestal case** (sustained 30-60N contact while
  nominally OPEN) — direct FK measurement found the real cause: the FK
  sweep's height filter matched the *abstract* `_EE_OFFSET`-based pinch
  point to the grasp height, not the *real* fingertip (18.475mm further
  down) — a fixed, pedestal-height-independent ~15mm offset that simply
  moved the same collision from the ground onto the pedestal's own new top
  surface. Fixed `ar4_graspable_workspace.py` to target the real fingertip
  height directly; re-swept (still robust, 23560 survivors) and got 3
  freshly-corrected points with **verified positive** real fingertip
  clearance above the pedestal top (9.8-10.5mm, not merely assumed).
  **Second live grasp attempt at the corrected points: the ground/pedestal-
  collision failure signature is GONE** (no more asymmetric force pattern,
  pinch-point discrepancy shrank from ~21mm to ~6mm, closed-loop settle
  converges to <1° in most cases) — **but all 3 points still fail to
  achieve a real lift**: real contact force (51-57N) while nominally OPEN
  persists, and contact drops to exactly 0N the instant LIFT-CLOSE begins
  (the cube was never actually pinned, just nudged). This matches this
  project's own already-documented 2026-07-24 "roll/heading residual
  contact" finding (`ROLL_TOL_DEG=12°` bounds heading but doesn't
  guarantee a genuinely antipodal grip) rather than a new problem — a
  separate, pre-existing blocker this task's own scope (the ground-
  clearance fix) was not tasked to solve. **Honest verdict: the specific
  ground-collision blocker this task targeted is closed and verified; the
  capstone grasp+lift is still not achieved, blocked on the roll/heading-
  residual-contact question instead.** Both cloud runs' teardown hung in
  Isaac Sim's own documented Kit-shutdown-teardown pattern (0% GPU, log
  stale, real work already complete and logged) — recovered via the
  established `kill -TERM` pattern both times, confirmed via direct
  `nvidia-smi`/log inspection before killing. Combined cloud cost ≈$1,
  both runs individually torn down and confirmed clean via
  `scripts/check_cloud_state.sh`. `kb/wiki/concepts/
  ar4-vs-franka-root-cause-comparison.md`'s 2026-07-28
  "ar4-pedestal-ground-clearance-fix task" UPDATE.
- **AR4 joint_2 ~59° wall root-caused: NOT a joint-limit bug — a real
  ground-plane collision the FK workspace tool never modeled, and the
  corrected workspace is genuinely EMPTY at the current grasp height**
  (2026-07-28, direct continuation of the joint-tracking-closed-loop-fix
  task below). Direct pxr/UsdPhysics read of the built USD (after working
  around 3 consecutive cloud cold-start stalls in the full-env-construction
  path by using a lighter `isaacsim.SimulationApp`-only bootstrap, plus a
  buffered-stdout bug that silently ate the first attempt's own output)
  confirms joint_2's authored `physics:lowerLimit/upperLimit` is exactly
  `-42°/+90°` — matching the vendor spec (`config/mk3.yaml`, cross-checked
  against `urdf/ar_macro.xacro`'s own `<limit>` tag) to within float32
  rounding. **Not an asset-import bug.** (Bonus finding from checking all 6
  joints: `joint_1`'s raw USD limit is `±160°`, a real ~10° narrower-than-
  vendor `±170°` mismatch — flagged, not yet fixed, out of this task's own
  scope.) The real mechanism: a Pi-local pure-FK sweep of
  `gripper_jaw1_link`'s real world-frame height (not just the abstract
  pinch-point the tool previously checked) shows it falls monotonically as
  joint_2 approaches GRASP_Q, nearly identically across P0/P1/P2's three
  independently-sampled configs, reaching only ~28-31mm above the z=0
  ground plane right at the empirically-observed ~59° wall — combined with
  the prior task's own no-cube-obstruction control (wall persisted with
  the cube 3m away), this is conclusively **the gripper's real jaw
  geometry hitting the ground plane**, not the cube, not a joint limit,
  not self-collision (already disabled) or an effort ceiling (already
  ruled out). Fixed `scripts/ar4_graspable_workspace.py` by adding a real
  ground-clearance filter (extends the batch FK chain to
  `gripper_jaw1_link`, held at the physically-correct OPEN position) —
  previously totally absent. Re-swept (full 8M-sample sweep): **0
  survivors** — and this is robust, not a tuning artifact: even a
  deliberately permissive 15mm clearance threshold (looser than the known
  ~18.5mm jaw-mesh extent) already finds zero, and the best-case real
  fingertip clearance achievable anywhere in the height/tilt/margin-
  satisfying population is **-3.55mm** (i.e. still below ground even in
  the best case). **AR4 genuinely cannot perform a clean near-vertical
  grasp of the current 15mm cube anywhere in its real reachable
  workspace, at this grasp height** — a clean, well-quantified null result
  per this project's own "report the real discrepancy, don't force a
  positive" standard, not a forced grasp attempt. No live grasp+lift
  re-attempt was made this task since there is no valid corrected point to
  attempt one at. Concrete implication flagged back to whoever picks this
  up (a task-design/architecture decision, not this task's own call): the
  cube likely needs to be raised (a small pedestal) or grasped higher on
  its own body (closer to its top face) to give the gripper's real
  fingertip length room to clear the ground, or the whole vertical-grasp
  strategy needs reconsidering for this arm/gripper/cube-size combination.
  `kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md`'s 2026-07-28
  "ar4-joint2-ground-clearance-fix task" UPDATE.
- **AR4 joint-tracking closed-loop fix — tracking gap DOES close for most
  joints, but joint_2 is pinned at a real hard limit the graspable-
  workspace tooling never modeled** (2026-07-28) — built and tested both
  candidate fixes for the tracking gap: a stiffness sweep (4000→80000, plus
  a 10x effort-limit boost) and a new reusable closed-loop primitive
  (`tasks/ar4/joint_tracking.py`'s `settle_to_joint_pose`, the joint-space
  analog of `oracle_rollout.py`'s already-validated Cartesian integral-
  error fix). Both confirm most joints' droop responds exactly to gain/
  correction as PD theory predicts (e.g. joint_6: 7.47°→0.47° with higher
  stiffness; PREGRASP_Q converges to sub-0.1°/sub-mm at all 3 validation
  points via the closed-loop primitive) — but `joint_2` stays frozen at
  ~3.2-3.3° error at GRASP_Q regardless of a 20x stiffness increase, a 10x
  effort-limit increase, or a diverging +26° closed-loop correction, with
  `applied_torque` staying constant instead of scaling with stiffness as
  real PD droop would. **New diagnosis: joint_2 has a real ~59° ceiling in
  live physics, vs. `ar4_graspable_workspace.py`'s assumed `(-42°, 90°)`
  range** — P0/P1/P2's GRASP_Q all need ~62.3-62.5°, just past this
  unmodeled wall, which explains why every one of that sweep's candidate
  points still collides while nominally open (worse than before: 87-108N
  vs. the roll-constraint task's 45-54N, since the correction increases
  jamming force against a target it can't reach). A direct USD-readback
  confirmation script was written but hit two consecutive cloud infra
  stalls before completing — the ~59° figure is a strong mechanistic
  inference, not yet a directly-read value. Next step: re-run the readback
  script, correct `JOINT_LIMITS_DEG["joint_2"]`, and re-sweep the
  graspable workspace with the corrected range before attempting another
  grasp — the whole P0/P1/P2 candidate set is likely invalidated by this
  finding. `kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md`'s
  2026-07-28 UPDATE.
- **AR4 joint-tracking diagnostic — physics-vs-kinematics confound
  CONFIRMED and quantified** (2026-07-27) — direct commanded-vs-achieved
  measurement, cube parked 3m away (nothing obstructing), testing whether
  the roll-constraint task's live "arm plateaus short, gripper collides
  while OPEN" result was a real joint-tracking failure rather than pure
  kinematics being wrong. **Default arm actuator gains (shipped
  `stiffness=40, damping=4`) catastrophically fail**: max per-joint error
  96° (`joint_5`), arm settles in a qualitatively different pose under
  gravity, not a minor droop. **The boosted gains already used by every
  confirm script in this investigation (`stiffness=4000, damping=200`) are
  much better but NOT clean**: a real, genuinely-settled 3.2°/~20mm
  residual remains with nothing in the way — larger than the 15mm cube
  itself. This reopens the "jaw mesh geometry is the sole remaining
  culprit" conclusion from the roll-constraint task: a real, now-quantified
  actuator-tracking gap is large enough on its own to plausibly explain
  part of that ~45-56N open-gripper collision, independent of jaw mesh.
  Cross-validated methodology (FK-recompute of achieved joint angles
  matches live physics to 0.00024mm) and a real infra finding (the known
  Isaac-Sim-teardown-hang recurred with zero cube contact anywhere in the
  run, weakening "jammed contact" as its sole cause). Next steps (neither
  tried yet): push gains higher still, or add gravity-compensation
  feedforward. `kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md`'s
  2026-07-27 (later, ar4-joint-tracking-diagnostic task) UPDATE.
- **AR4 graspable-workspace-from-FK** (2026-07-27) — inverted the
  multi-week AR4 IK-failure investigation: instead of fighting IK to
  reach a vertical grasp pose at the cube's existing default position,
  forward-sampled the arm's own 6-joint space via pure FK (no IK, no
  physics, 8M+145-step sweep) to find where a genuinely graspable pose
  actually exists. Result directly explains the whole prior investigation
  — the cube's own current default positions sit in a real gap of the
  graspable annulus. Recommended point `(-0.1127, 0.3255)` has excellent
  joint margins (min 27.68°) — reachability is solved. Live confirmation
  (direct joint-target control, no IK) did NOT achieve a grasp+lift
  though: a NEW, root-caused discrepancy found instead — the gripper
  collides with the cube even nominally OPEN, traced to the FK filter
  never constraining gripper roll/heading (only tilt-from-vertical), a
  concrete, fixable gap flagged for the next session.
  `kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md`'s 2026-07-27
  UPDATE.
- **AR4 graspable-roll-constraint — roll/heading hypothesis REFUTED**
  (2026-07-27) — added the missing jaw-slide-axis heading constraint
  (`ROLL_TOL_DEG=12°` from parallel to world X/Y) to the FK sweep above,
  sanity-checked against `grasp_demo_v2.py`'s own known-good heading,
  re-swept (38,521/144,710 survivors, 26.6% — matching the geometric
  prediction almost exactly, a clean correctness signal). Live
  confirmation across 3 independently-chosen points spanning roll offsets
  1.06°-10.83° all show the SAME ~45-56N open-gripper collision force
  regardless of roll quality — a clean, 3-point-replicated negative
  result. **The roll/heading hypothesis is refuted, not just
  unconfirmed.** Next concrete hypothesis (not yet tested): the jaw
  COLLISION GEOMETRY itself (convex-hull approximation, already flagged
  as an unresolved AR4 defect in `CLAUDE.md`'s Franka-pivot rationale) —
  this task's data is a second, independent corroboration of that
  candidate. Also surfaced a real infra finding: the prior session's
  "camera/render-pipeline stall" recurred in a camera-FREE script, mid a
  sustained jammed-contact state — weakens "render-pipeline-specific" as
  that stall's explanation. `kb/wiki/concepts/ar4-vs-franka-root-cause-
  comparison.md`'s 2026-07-27 (later, ar4-graspable-roll-constraint task)
  UPDATE.
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
