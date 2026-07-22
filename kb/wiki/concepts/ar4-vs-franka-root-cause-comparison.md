# AR4 vs. Franka: root-causing the three pivot defects

## What this resolves

CLAUDE.md's "Platform pivot" section (2026-07-09) names three specific,
never-resolved AR4-asset hypotheses as the rationale for moving to Franka:
a 17-27mm classical-IK grasp miss, an unconfirmed jaw-mimic constraint, and
an unverified convex-hull jaw-collision approximation. A dedicated
read-only investigation
(`docs/superpowers/specs/research/2026-07-20-ar4-vs-franka-root-cause-comparison.md`)
went back through this repo's own history and the Franka code
(`tasks/franka/`) to root-cause each one directly, rather than leaving them
as background rationale. Full citations live in that research doc; this
article summarizes the verdicts and why they matter for the North Star's
"drop in a new arm, training should succeed immediately" bar.

## The three verdicts

1. **Classical-IK grasp miss — root cause found, but re-characterized.**
   The literal "17-27mm" figure doesn't appear verbatim anywhere in this
   repo's history (`git log -S` on that string and close variants finds
   nothing); it's a rounded synthesis of several distinct measurements
   (14.6mm-60mm depending on script/method). The actual, independently
   corroborated root cause (echoed by both Experiment 20's damping sweep
   and Experiment 24 Gate 1's diagnostics) is a **single-Newton-step DLS
   differential-IK solver getting trapped in a local-minimum fixed point**
   in a poorly-conditioned kinematic region — a property of the
   *standalone, non-RL, waypoint-jumping demo scripts*
   (`oracle_rollout.py`, `grasp_demo.py`/`v2`, `interactive_joint_demo.py`),
   not a URDF/asset frame-offset bug. The EE-frame offset itself
   (`_EE_OFFSET=0.036` on `link_6`) was independently measured correct to
   <0.001mm. Notably, the same DLS mechanism *did* work when driven
   incrementally by a trained RL policy every control tick (Experiment 11's
   first sustained antipodal grasp contact) — the miss was specific to
   large single-step jumps in a classical script, not to the underlying IK
   method or asset generally. One real structural gap found in comparison:
   Franka's own reference config (`tasks/franka/lift_env_cfg.py:66-82`)
   explicitly separates the FrameTransformer sensing offset (0.1034m) from
   the IK controller's own body offset (0.107m) as two distinct,
   separately-sourced values; AR4's code reused one constant
   (`_EE_OFFSET=0.036`) for both roles. Not shown to be the cause of this
   particular miss, but a real asset-rigor gap worth correcting on
   principle.

2. **Jaw-mimic constraint — UPDATE 2026-07-21: a real `PhysxMimicJointAPI`
   does exist on the currently-built asset; the actual bug is a joint-limit
   mismatch, not absence of the constraint.** Direct USD inspection
   (`docs/superpowers/specs/research/2026-07-21-ar4-usd-asset-debugging.md`,
   built the asset fresh on today's pinned Isaac Lab v2.3.1/Isaac Sim 5.1.0
   stack and opened it via `pxr`/`PhysxSchema`) found `gripper_jaw2_joint`
   genuinely carries a `PhysxMimicJointAPI:rotX` instance —
   `referenceJoint=gripper_jaw1_joint`, `gearing=-1.0`, `offset=0.0` — a
   real, correctly-targeted spring-based constraint. This directly
   contradicts the 2026-07-09-era belief (restated as "confirmed" in the
   2026-07-20 root-cause doc below) that the mimic constraint was never
   enforced by the importer; whether that was an older-version artifact or
   simply never checked at the prim level is not resolved. **The actual
   defect**: jaw2's own hard `physics:lowerLimit`/`upperLimit`, as
   imported, are `[-0.0028, 0.0168]` — but the mimic formula
   (`q2 = -q1`) applied to jaw1's real range `[0, 0.014]` maps to
   `[-0.014, 0]`, which does not fit inside jaw2's own limits. PhysX's hard
   limit clamp overrides the spring constraint, so jaw2 can only track
   jaw1 for the first ~20% of its stroke (`q1` up to `0.0028m`) before
   getting stuck at its own limit — a concrete, direct mechanism for the
   asymmetry Experiments 17/19/22 each independently hit but never
   root-caused this specifically. Fixed in `scripts/build_asset.py`
   (`_fix_gripper_jaw2_mimic_limits`, re-derives jaw2's limits from jaw1's
   own limits under the already-authored gearing/offset) and statically
   re-verified in the rebuilt USD. A live dynamic test (a bare
   `isaacsim.core.api.World` scene, not the full IsaacLab task pipeline)
   showed jaw1 moving normally but jaw2 reading back as exactly `0.0`
   throughout — an unresolved discrepancy (test-rig readback issue, or a
   real remaining engagement problem outside this specific test scene) —
   flagged as the concrete next step, not swept under the static fix.
   **The originally-revealing action-space finding still stands**: AR4's
   default RL gripper action (`BinaryJointPositionActionCfg`, identical
   commanded target to both jaws) remains structurally identical to
   Franka's own validated gripper action — the RL action-space design was
   never the point of difference between the two platforms.
3. **Jaw collision geometry ("unverified convex-hull approximation") —
   UPDATE 2026-07-21: confirmed real on the AR4 side (Franka's own shipped
   asset still unexamined).** Direct instance-proxy stage traversal
   (`docs/superpowers/specs/research/2026-07-21-ar4-usd-asset-debugging.md`)
   found `UsdPhysics.MeshCollisionAPI.approximation = "convexHull"` really
   is authored on `gripper_jaw1_link`, `gripper_jaw2_link`, and
   `gripper_base_link`'s collision meshes — resolving the "unverified"
   status this item carried since the pivot. What remains open: the
   authored attribute only tells PhysX to compute a hull from the
   referenced mesh at simulation start — the hull's own vertex/face count
   isn't stored in the USD, so whether it meaningfully distorts the jaw's
   real (possibly non-convex) fingertip surface still needs an offline
   convex-hull computation against the raw mesh points (e.g.
   `scipy.spatial.ConvexHull`) as a follow-up — not yet done. Franka's own
   shipped asset is still unexamined (not inspectable from the Pi,
   `/home/saps/IsaacLab` unreachable from this machine) — this project's
   own d4 grasp work on Franka still doesn't trust the stock finger
   collision mesh either, building a purpose-authored notch fixture
   instead of reading contact force off it directly.

## Does this support the pivot decision?

**Yes, on balance, but the recorded rationale oversold two of its three
items.** Hypothesis 2's finding (identical action-space mechanism, only
one platform needs no coupling fix) is real, load-bearing evidence for
"AR4 asset defect, not RL/reward-design difficulty," and it's corroborated
by Franka's actual subsequent results (working grasp/lift). But Hypothesis
1's "17-27mm, unresolved root cause" framing overstates both the number's
precision and how unresolved the mechanism was — it was reasonably well
understood (DLS local-minimum trap in specific classical scripts) before
the pivot was even decided. And Hypothesis 3 was never actually verified
on either side, despite reading as settled evidence in the pivot text.
Worth adding to this picture: this project's own last AR4 result before
the pivot, **[[experiment-26-gripper-reintroduction]]**, was itself never
cleanly attributed to the three named asset defects — its own recorded
verdict names a reward-design mechanism (a running-max staged reach
potential with no incentive to hold position) as an equally plausible
contributor, alongside "the antipodal grasp gate is apparently never
satisfied." The pivot was a reasonable, probably-correct call, but the
project's own final AR4 data point was genuinely confounded between an
asset-defect and a reward-design explanation when the decision was made,
not cleanly resolved in favor of the former.

## UPDATE 2026-07-21 (later, ar4-franka-fixes-transfer plan, Task 5): a SECOND, independent gripper-mirror bug found and fixed, PLUS live dynamic confirmation now done — and it surfaces a new, more concrete root-cause candidate than either of the above.

**Bug 1 (fixed): every AR4 env cfg commanded gripper_jaw2_joint to the
IDENTICAL signed value as gripper_jaw1_joint, not jaw2's own mirrored
(negated) value.** Surfaced as a hard crash the moment this task tried to
build `Ar4PickPlaceGraspGoalEnvCfg` on the freshly-rebuilt (post-64ab5cc)
asset: `gripper_jaw2_joint`'s default position (+0.014, from
`GRIPPER_OPEN_POS` applied identically to both jaws in
`tasks/ar4/robot_cfg.py`'s `init_state` and in every env cfg's
`open_command_expr`) fell outside 64ab5cc's own newly-corrected,
mimic-consistent jaw2 hard limits (`[-0.014, 0.000]`). Given jaw2's
`PhysxMimicJointAPI` has `gearing=-1.0, offset=0.0` (confirmed by 64ab5cc's
own direct USD inspection, above), jaw2's physically-correct commanded
position is always `-1.0 * jaw1's`, not the same signed constant — the
OLD, pre-64ab5cc jaw2 hard limits (`[-0.0028, 0.0168]`) happened to
tolerate the wrong `+0.014` without erroring, masking this second,
independent sign bug for as long as this project's own original
(also-wrong) jaw2 limits stood. **This means the gripper's commanded
"open" state has likely been asymmetric since this constant was
introduced, in every AR4 experiment that used it** — a candidate
explanation, on its own, for exactly the kind of asymmetric single-jaw
contact this project's diagnostics have repeatedly found (e.g. Experiment
21's own diagnostic: jaw1 registered zero contact force across 750 rollout
steps while jaw2 registered contact intermittently).

Fixed at the shared source (`tasks/ar4/robot_cfg.py`'s new
`GRIPPER_OPEN_COMMAND_EXPR`/`GRIPPER_CLOSED_COMMAND_EXPR`, mechanically
propagated to all 15 other AR4 env cfg files that had the same pattern),
controller-authorized as a cross-experiment fix outside this task's
original plan scope. Verified empirically, not just asserted: a fresh
`env.reset()` now shows perfect mirroring —
`jaw1=+0.01400  jaw2=-0.01400  jaw1+jaw2=+0.00000`.

**Bug 2 (found, NOT fixed — out of this pass's authorized scope, flagged
for a future pass): the identical symmetric-command bug independently
exists in `tasks/ar4/actions.py`'s `MirroredGripperAction` (its
`process_actions` sets
`self._processed_actions[:, 1] = jaw1_commanded_target` — jaw1's raw
value, not its negation) and in `scripts/interactive_joint_control.py`
(`gripper_target_t = torch.tensor([[gripper_target_val,
gripper_target_val]], ...)` — same value for both sliders' target).**
Neither is used by `Ar4PickPlaceGraspGoalEnvCfg` or its Condition A2/B
variants (which use plain `ProximityGatedBinaryJointPositionActionCfg`),
so out of scope for this fix, but the same bug pattern is confirmed
present in at least two more places.

**Bigger finding: the sign fix is necessary but NOT sufficient — live
dynamic behavior remains broken/asymmetric even with the correct target.**
A direct, real rollout (`scripts/_verify_gripper_mirror_fix.py`, driving
`robot.set_joint_position_target` + `sim.step` directly and reading real
joint positions every 10 steps, not a shaped metric) inside the actual
`Ar4PickPlaceGraspGoalEnvCfg` task env cfg found:

```
[reset, fixed init_state]   jaw1=+0.01400  jaw2=-0.01400  jaw1+jaw2=+0.00000
  [CLOSE step   0] jaw1=+0.01373  jaw2=-0.01388  target=[[0.0, -0.0]]
  [CLOSE step  10] jaw1=+0.00911  jaw2=-0.01400  target=[[0.0, -0.0]]
  [CLOSE step  20] jaw1=+0.00562  jaw2=-0.01393  target=[[0.0, -0.0]]
  [CLOSE step  30] jaw1=+0.00315  jaw2=-0.01397  target=[[0.0, -0.0]]
  [CLOSE step  40] jaw1=+0.00139  jaw2=-0.01399  target=[[0.0, -0.0]]
  [CLOSE step  50] jaw1=+0.00016  jaw2=-0.01394  target=[[0.0, -0.0]]
  [CLOSE step  59] jaw1=+0.00007  jaw2=-0.01305  target=[[0.0, -0.0]]
  [OPEN step   0] jaw1=+0.00024  jaw2=-0.01293  target=[[0.014, -0.014]]
  [OPEN step  10] jaw1=+0.00630  jaw2=-0.00643  target=[[0.014, -0.014]]
  [OPEN step  20] jaw1=+0.01198  jaw2=-0.00000  target=[[0.014, -0.014]]
  [OPEN step  30] jaw1=+0.01159  jaw2=-0.00000  target=[[0.014, -0.014]]
  [OPEN step  40] jaw1=+0.01128  jaw2=+0.00000  target=[[0.014, -0.014]]
  [OPEN step  50] jaw1=+0.01107  jaw2=+0.00000  target=[[0.014, -0.014]]
  [OPEN step  59] jaw1=+0.01094  jaw2=+0.00000  target=[[0.014, -0.014]]
```

jaw1 tracks its own commanded target cleanly in both phases (a normal PD
convergence curve). **jaw2 does not track its target at all in either
phase** — during CLOSE it stays pinned near its *open* extreme (~-0.013 to
-0.014) despite a `0.0` target; during OPEN it moves quickly to and then
sticks exactly at `0.00000` — its own hard *upper* limit, the opposite end
from its `-0.014` target — and stays there. In both phases jaw2 ends up
parked at one of its two hard limits, essentially independent of what it
was actually commanded to do.

**Candidate mechanism (not yet confirmed): the PhysX `MimicJointAPI` spring
constraint (`gearing=-1.0`) and the independent `ImplicitActuatorCfg` PD
actuator are both trying to drive the same joint (`gripper_jaw2_joint`)
simultaneously, and something in that interaction — not either mechanism's
own target in isolation — is winning and driving jaw2 into its own hard
limit.** This is now a more concrete, more directly-measured root-cause
candidate for AR4's long-standing jaw-asymmetry problem than either
Hypothesis 2's original joint-limit-mismatch framing or Hypothesis 3's
still-unverified collision-geometry question above — it would explain
*why* three separate command-level fix attempts (Experiments 19, 22, and
this task's own re-confirmation) all failed to produce symmetric contact:
none of them addressed a physics-solver-level conflict between two
independent constraint mechanisms on the same joint.

**Deliberately not pursued further in this pass** (controller decision,
2026-07-21): tuning the mimic constraint's own damping/naturalFrequency
parameters, or the gripper actuator's stiffness/damping, or dropping the
mimic constraint in favor of pure per-joint actuation, are all real
candidate fixes but constitute a genuine architectural change beyond a
"fix the sign bug" pass — logged to `BACKLOG.md` as a distinct, separate
follow-up rather than attempted here. Training proceeded on the real,
currently-asymmetric dynamics regardless (RL observes real `joint_pos`/
`joint_vel` and rewards off real measured contact forces, not off whether
a target was "correctly" reached), per controller instruction.

## Open follow-up

As of 2026-07-21 (updated later the same day, see UPDATE above): the
jaw-mimic joint-limit bug is fixed and statically verified, and a second,
independent command-sign bug (this section's own finding) is now also
fixed and empirically verified. Full live dynamic confirmation (actually
watching jaw2 track jaw1 through a real simulated grasp) is DONE (see
above) and found a new open question (mimic-vs-actuator conflict) rather
than closing the topic — that new question is the follow-up now, tracked
in `BACKLOG.md`, not this doc. The convex-hull distortion question is
now narrowed to a concrete, cheap, GPU-free follow-up (compute the real
convex hull of the jaw's own mesh points and compare face counts against
the original mesh) rather than a fully open question. Link_5/Link_6's
missing collision (a fourth defect found during the concurrent
`ar4-franka-fixes-transfer` task's build smoke test the same day) is now
also fixed with a substitute box collider — see
`docs/superpowers/specs/research/2026-07-21-ar4-usd-asset-debugging.md`
for the full detail on all of the above. Comparing Franka's own shipped
asset's collision approximation remains unexamined either way (not
inspectable from the Pi).

## H_ar4_relative transfer test (2026-07-21 follow-up): FALSIFIED — Franka's own confirmed relative-joint fix does not transfer to AR4

**What this tests.** The `ar4-franka-fixes-transfer` plan
(`docs/superpowers/plans/2026-07-21-ar4-franka-fixes-transfer-implementation.md`,
spec: `docs/superpowers/specs/2026-07-21-ar4-franka-fixes-transfer-design.md`)
asked whether Franka's own CONFIRMED `RelativeJointPositionActionCfg` fix
(`kb/wiki/experiments/d8-antipodal-grasp-quality.md`'s H_relative section
— a genuinely joint-space, no-IK action-term change that resolved
Franka/d8's exact-zero-contact-forever collapse, 3/3 seeds) transfers to
AR4's own analogous historical null, [[experiment-26-gripper-reintroduction]]
(`cube_reached_goal` exact `0.0000`, "the antipodal grasp gate is
apparently never satisfied"). This is a direct, targeted test of whether
that section's own three named pivot hypotheses (jaw-mimic constraint,
jaw collision geometry, classical-IK positioning miss) — rather than a
joint-space-action-learnability problem of the kind that explained
Franka's — are the real explanation for AR4's problem, now that both real
gripper asset defects this article documents above (the jaw2 mimic-limit
mismatch, `64ab5cc`, and the jaw2 command-sign bug, `928af41`) are fixed.

**Design:** Condition A2 = `Ar4PickPlaceGraspGoalEnvCfg` (Experiment 26)
unmodified, freshly retrained on the now-fixed asset (not assumed from the
historical pre-fix run). Condition B = a new leaf,
`Ar4PickPlaceGraspGoalRelativeEnvCfg`, identical in every other respect
but swapping the arm action term for `RelativeJointPositionActionCfg`
(`scale=0.1`) — Franka's own H_relative recipe, transferred directly. 3
seeds (42, 123, 7) × 2 conditions × 1500 iterations, 5 measured checkpoints
(iterations 0/100/300/700/1499) per the plan's own falsification
protocol.

**Result table (final checkpoint, iter 1499):**

| Condition | Seed | contact_freq | antipodal_freq | ever_lifted | `cube_reached_goal` (across all 1500 iters) |
|---|---|---|---|---|---|
| A2 (absolute) | 42  | 0.0000 | 0.0000 | 0.0000 | 0.0 (exact last/max/min) |
| A2 (absolute) | 123 | 0.0002 | 0.0000 | 0.0000 | 0.0 |
| A2 (absolute) | 7   | 0.0000 | 0.0000 | 0.0000 | 0.0 |
| B (relative)  | 42  | 0.0000 | 0.0000 | 0.0000 | 0.0 |
| B (relative)  | 123 | 0.0000 | 0.0000 | 0.0000 | 0.0 |
| B (relative)  | 7   | **0.9751** (real bilateral jaw contact forces ~0.17-0.27N, confirmed not a sensor artifact) | 0.0000 | 0.0000 | 0.0 |

`cube_reached_goal` is exact `0.0` as the last, max, AND min value across
the full 1500-iteration trajectory in all 6 runs — the behavioral bar is
verified flat-zero for the entire run, not just at the 5 sampled
checkpoints. Critic divergence (the pre-authorized `clip_actions=5.0`
contingency, carried over from Franka's own real H_relative precedent)
did **not** occur in any run (`Loss/value_function` max 0.0055-0.1274
across all 6 runs, nowhere near Franka's own `181→inf` signature).

**Honest gap in this write-up:** the intermediate-checkpoint (iter
0/100/300/700) `contact_freq`/`antipodal_freq` values Task 6 measured are
not reproduced above — only final-checkpoint (iter 1499) values were
carried forward into this closing task's own handoff, and the raw
per-checkpoint diagnostic artifacts exist only on the now-torn-down cloud
instance/GCS sync for this run, not locally. This means this section
cannot assert the exact *shape* of AR4's own curve the way the three-way
comparison in `d8-antipodal-grasp-quality.md`'s H_relative section could
(that comparison showed AR4's Franka-side counterpart reproducing
task-space's own monotonic-rise shape almost point-for-point). The
falsification verdict itself does not depend on this gap — the
pre-registered rule only checks the final-checkpoint bar in ≥2/3 seeds —
but it is flagged rather than silently smoothed over.

**Verdict: H_ar4_relative is FALSIFIED**, per the pre-registered rule
(falsified if ≥2/3 Condition-B seeds hit the exact-zero bar). Seeds 42
and 123 do so cleanly; seed 7 is a genuine, confirmed partial exception
(real, nonzero contact frequency) but still zero antipodal fraction and
zero `cube_reached_goal` — real gripper contact, never a real antipodal/
successful grasp — so it does not prevent falsification at the ≥2/3
threshold.

**3-signature jaw-mimic classification** (all 6 runs): Signature 1
(near-zero contact — an exploration/action-space-level failure, the arm
never seriously approaches/contacts the cube) dominates 5/6 runs: all of
A2, plus condition-B seeds 42 and 123. Signature 2 (nonzero contact, zero
antipodal fraction — consistent with this workstream's own found jaw-mimic-
vs-actuator dynamics conflict: jaw2 stays pinned near its hard limits
regardless of commanded target, so real contact happens but never in an
antipodal/graspable configuration) appears cleanly in condition-B seed 7
only. Signature 3 (contact + antipodal + still no lift) was never
observed in any run — since antipodal contact essentially never occurs at
all except in that one case, "contact+antipodal+no lift" has no
opportunity to apply here.

**Known limitation:** close-up video review was NOT performed for the
condition-B/seed7 exception — `scripts/graspgoal_closeup_video.py` is
hardcoded to the absolute-action demo env cfg and would need modification
to correctly load a relative-action checkpoint. The seed7 finding instead
rests on raw contact-force/antipodal-angle data cross-checked against the
literal training-time `antipodal_grasp_bonus` reward function's own math —
a real limitation of this task's verification depth relative to this
project's usual video-review standard, not glossed over as equivalent.

**What this means for the three hypotheses above.** Condition A2 (fixed
asset, absolute joint-space) reproduces the identical all-zero
`cube_reached_goal` pattern as both the pre-asset-fix Condition A and the
historical [[experiment-26-gripper-reintroduction]] null — **fixing BOTH
real gripper asset defects this article documents (Hypothesis 2's
joint-limit mismatch and the independently-found command-sign bug) did
NOT, by itself, resolve AR4's grasp-discoverability problem.** Condition
B's result rules out "AR4's problem is the same joint-space-action-
learnability issue Franka had" as a sufficient explanation on its own — if
it were, the identical fix should have produced the identical win, and it
did not in 2/3 seeds. This pushes the explanation back toward this
article's own asset-level hypotheses: seed 7's Signature-2 pattern
(contact without antipodal geometry) is directly consistent with the
jaw-mimic-vs-actuator dynamics conflict found earlier this same workstream
(the "Bigger finding" in the UPDATE section above) — jaw2 pinned near its
hard limits regardless of commanded target would produce exactly this
shape: real contact, never correctly-shaped antipodal contact. The
jaw-collision-geometry question (Hypothesis 3, still only "confirmed
present," not shown to distort contact directions) remains equally
consistent with, and unruled-out by, this same seed7 pattern. The
classical-IK positioning miss (Hypothesis 1) is not implicated by this
result at all, since neither condition here uses IK.

**North Star relevance.** Franka's own H_relative result mattered to
[CLAUDE.md](../../../CLAUDE.md)'s North Star specifically because it was
a genuinely joint-space fix (no arm-specific IK/kinematic-chain
controller) that resolved an analogous collapse — real evidence that the
"drop in a new arm, training should succeed immediately" bar does not
hinge on an IK/task-space layer as a hidden prerequisite. Since
H_ar4_relative is FALSIFIED here, **that evidence does not extend to
AR4** — the identical fix, transferred to a second, structurally
different arm, does not reproduce the win. This does not overturn the
North Star finding on Franka/d8 itself, and it does not positively refute
the North Star's cross-arm bar either — it means AR4's own problem is
most likely still explained by asset-specific defects (exactly the
rationale the original platform pivot gave), not by a general property of
joint-space action learnability that a second arm would also need to
overcome. [[experiment-11-taskspace-ik]] — AR4's own only prior positive
result (a genuine, sustained antipodal contact signal, under task-space/
IK control) — remains the one condition on this platform where the
antipodal mechanism has ever fired at all; this plan deliberately did not
retest that condition, since Experiment 26's absolute-joint null was the
more direct analogue of Franka's own falsified H_joint condition, and
retesting IK on AR4 wasn't this plan's question.

**Honest next candidate direction (not started here — AR4 investigation
is not the active priority while the Franka pivot is underway, per
CLAUDE.md's "Platform pivot" section).** The fix does not transfer; AR4's
null is not explained by the same joint-space-action mechanism that
explained Franka's. The asset-level hypotheses this session already
surfaced — the jaw-mimic-vs-actuator dynamics conflict and the
still-unverified jaw collision geometry — and/or the still-unresolved
classical-IK 17-27mm positioning miss remain the more likely
explanations, consistent with the original Franka-pivot rationale.
Concrete next hypothesis, logged to `BACKLOG.md` as flagged-but-deferred,
not executed here: test jaw-mimic vs. independent-actuator by disabling
the mimic constraint entirely and re-running Condition B once.

**Cost:** ≈$2.07 cumulative against the plan's $10 cap. Full teardown
verified. See `ROADMAP.md`'s matching Task 7 entry (2026-07-21) for the
same synthesis in the project-status ledger.

## UPDATE 2026-07-22: mimic constraint removed, sign-inversion signature found, then blocked on desktop unreachability before the confirming diagnostic could run

Continuing directly from the "Bigger finding" above (mimic-vs-actuator
physics conflict candidate). Two more steps landed since:

- `64ab5cc`/`928af41` (already covered above) were confirmed insufficient
  by a live dynamic test: jaw2 didn't track its own commanded target at
  all, staying pinned near a hard limit regardless of target, while jaw1
  (unaffected by any coupling) tracked normally.
- `2576e94` removed the `PhysxMimicJointAPI` mimic constraint from
  `gripper_jaw2_joint` entirely (`scripts/build_asset.py`'s new
  `_remove_gripper_jaw2_mimic_constraint`, replacing the old
  `_fix_gripper_jaw2_mimic_limits`) — both jaws are now driven as fully
  independent `ImplicitActuatorCfg` PD targets, software-mirrored via
  `tasks/ar4/robot_cfg.py`'s `GRIPPER_OPEN_COMMAND_EXPR`/
  `GRIPPER_CLOSED_COMMAND_EXPR`. Jaw2's hard limits are still correctly
  re-derived from jaw1's under the known mirror geometry.
- **First live re-test after the mimic removal (commit `d16aa76`'s
  message) still failed, but with a different, more specific signature**:
  jaw2 now moves substantially in both phases (unlike the pre-removal
  test), but consistently lands at the OPPOSITE end from its own
  commanded target (commanded `0` → stays near `-0.014`; commanded
  `-0.014` → moves to `0`) — the signature of an inverted actuator-drive
  sign specifically for this joint, not a limit-pinning defect this time.
  This is a genuinely new candidate root cause, distinct from both the
  joint-limit-mismatch (Hypothesis 2's original finding) and the
  command-sign bug already fixed in `928af41` (that one was about the
  *commanded value* sent to the joint, not the *joint's own drive-to-
  motion sign* once commanded correctly).
- `d16aa76` added a mid-range (`-0.007`) isolated sweep to
  `scripts/_verify_gripper_mirror_fix.py`, holding jaw1 fixed, specifically
  to distinguish "jaw2 converges toward `-0.007`" (normal tracking,
  something else is wrong) from "jaw2 moves toward the opposite endpoint
  (`0`) regardless" (confirms sign inversion) — **this diagnostic was
  written but never run**; the prior session was stopped right after
  writing it.

**This session (2026-07-22): blocked before running the diagnostic.**
Tasked with running the mid-range sweep live and continuing the debug
loop, but the desktop (`saps@home.local`, ssh alias `desktop`) — the only
machine in this project with a GPU, a working Isaac Lab install, the
already-built AR4 USD asset, and the external `annin_ar4_description` ROS
package `scripts/build_asset.py` requires
(`AR4_DESCRIPTION_PATH=/home/saps/projects/annin_ws/src/ar4_ros_driver/annin_ar4_description`
— not in this git repo, no GCS mirror found) — was confirmed unreachable:
DNS resolution (`ssh`, `getent ahosts`), mDNS resolution (`avahi-resolve`),
and mDNS service browsing (`avahi-browse -a`, which does show ~15 other
LAN devices/services, just nothing matching the desktop) all failed
identically across roughly 20 minutes of retries spread over two bounded
polling windows — consistent with a genuine outage (powered off or
network-disconnected), not a brief reboot blip (a reboot completing
during that window would have re-registered via mDNS at some point).

**Why this didn't get resolved by falling through to the standing
desktop-first/cloud-fallback policy.** That policy (CLAUDE.md's
"Pi-as-primary-agent GPU dispatch" section) is designed for compute that's
agnostic to which machine runs it — e.g. Franka RL training, which has a
proven, repeatable cloud recipe (`docs/cloud/franka-cloud-shakedown.md`).
AR4's diagnostic/grasp-validation work is not that: it depends on
desktop-resident *state*, not just desktop *compute* — the already-built,
already-fixed USD asset and the external ROS description package it's
built from both live only on the desktop's local disk, with no proven
cloud recipe and no cheaper GCS-hosted copy found. Standing up a
from-scratch cloud AR4 pipeline (re-cloning the external ROS package,
running `scripts/build_asset.py` fresh, confirming byte-for-byte or at
least functional parity with the desktop's already-fixed asset before
trusting any diagnostic run against it) is a materially larger, unproven,
real-cost undertaking than "run an already-written diagnostic script" —
judged a cross-cutting infrastructure decision to flag back to the
controller rather than one to take unilaterally as an implementation
detail of this task.

**Net effect: no new empirical data this session.** The mid-range sweep's
actual trajectory is still not observed. The concrete next step, unchanged
from before this session, is to run `scripts/_verify_gripper_mirror_fix.py`
live via `flock -o /tmp/rl_isaac_sim.lock -c "..."` on the desktop exactly
as originally planned, once it's reachable again — or, if the controller
decides the desktop outage is expected to persist, to scope a real
from-scratch cloud AR4 build as its own explicit task rather than folding
it silently into this one.

## Related concepts

[[reach-grasp-lift-gap]] — this comparison is the direct follow-up
investigation to that article's own open question at
[[experiment-26-gripper-reintroduction]] ("not yet root-caused to a
specific fix"); this article's Hypothesis 2 finding is the closest thing
to an answer for *why* AR4's jaw contact was asymmetric at the mechanism
level, even though the deeper physical cause remains open.
[[grasp-mechanics-antipodal-vs-magnitude]] — the antipodal force-closure
check this comparison's Hypothesis 3 examines is the same
`antipodal_grasp_bonus` mechanism that article covers; this article adds
the previously-unexamined question of whether the jaw collision mesh's
approximation quality could have been distorting the contact-force
directions that check reads.
[[action-space-design]] — Hypothesis 1's finding that AR4's classical-IK
miss was a property of single-step DLS in standalone scripts, not the
RL-driven continuous IK action term, is a data point for that article's
broader action-space history.
[[experiment-26-gripper-reintroduction]] — the historical AR4 null the
2026-07-21 H_ar4_relative transfer test above freshly reproduces under
Condition A2, on the newly-fixed asset.
[[experiment-11-taskspace-ik]] — AR4's own only prior positive antipodal
result (task-space/IK control), the reason the H_ar4_relative transfer
test above deliberately did not retest that condition.
[[d8-antipodal-grasp-quality]] — the Franka-side H_relative result the
2026-07-21 transfer test above tests transfer of; that article's own
"Related" section cross-links back here rather than duplicating this
article's table.

## Sources

`docs/superpowers/specs/research/2026-07-20-ar4-vs-franka-root-cause-comparison.md`
(full citations for the original three hypotheses), `docs/superpowers/specs/research/2026-07-21-ar4-usd-asset-debugging.md`
(direct USD-level inspection/fixes for Hypotheses 2 and 3, plus the
Link_5/Link_6 collision fix), `CLAUDE.md` ("Platform pivot" section),
`docs/superpowers/specs/2026-07-21-ar4-franka-fixes-transfer-design.md`
and `docs/superpowers/plans/2026-07-21-ar4-franka-fixes-transfer-implementation.md`
(the 2026-07-21 H_ar4_relative transfer test above).
