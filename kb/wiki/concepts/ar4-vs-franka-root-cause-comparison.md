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

## UPDATE 2026-07-22 (later, same day): desktop reachable again — mid-range sweep run, TWO real confounds found and fixed, jaw2 now tracks its own target CORRECTLY

Desktop confirmed reachable this session (`ssh desktop` succeeds, GPU idle,
no stray tmux). Before running anything: found the on-disk asset at
`~/projects/rl`'s own checkout was untouched since 2026-07-09 — it
predates every fix in this entire investigation (`64ab5cc`, `2576e94`,
etc.), because none of this workstream's actual work ever happened in that
directory. The real, already-rebuilt (2026-07-21 23:00, right after
`2576e94`) asset lives in a separate worktree, `~/projects/rl-ar4-fixes-transfer`
on the desktop — fast-forwarded to current `origin/main` and used for
everything below. Flagging this for future sessions: **check
`stat -c '%y' assets/ar4_mk5/ar4_mk5.usd` against the fix commits' own
dates before trusting any "live re-test" result** — this asset/worktree
mismatch could easily have produced a confidently-wrong verdict if not
caught.

**The mid-range sweep ran, but its first two results were themselves
confounded — not by the sweep design, but by the arm's own physics.**
Chronological order of what actually happened:

1. **First live run (matches the exact prior "opposite end" signature)**:
   jaw2 barely moved during CLOSE (stuck near -0.014, tiny drift to
   -0.0132) and swung all the way to the *opposite* end (0.00000) during
   OPEN — reproducing the earlier finding byte-for-byte. Added
   instrumentation (jaw link world z, cube-filtered contact-sensor forces)
   before concluding anything, per this task's own "don't force-fit a
   theory the data doesn't support" instruction.
2. **Instrumentation revealed the real confound**: `jaw1_cube_force`/
   `jaw2_cube_force` were exact `0.0000N` throughout (rules out cube
   contact), but `gripper_jaw1_link`/`gripper_jaw2_link` world z fell from
   **+0.4748m to +0.1988m over just 120 sim steps** — the entire arm was
   in an uncontrolled free-fall/swing under gravity. This script only ever
   commanded a joint-position target for the *gripper* joints, never the
   *arm* joints, and the arm's own actuator gains
   (`ImplicitActuatorCfg(stiffness=40, damping=4, effort_limit_sim=20.0)`,
   `tasks/ar4/robot_cfg.py`) are too weak to hold this arm's own weight
   statically. An arm swinging/rotating this violently injects real
   Coriolis/base-acceleration coupling into the child gripper joints —
   fully capable of making a joint LOOK like it's tracking "the opposite
   end" when it's actually just being passively dragged by its own moving
   base, unrelated to its own commanded target.
3. **First attempted fix (explicitly commanding the arm to hold its reset
   joint positions every step) was a no-op** — produced byte-for-byte
   identical trajectories to the un-held run. This itself is informative:
   Isaac Lab's `env.reset()` already sets the articulation's joint-position
   targets to the initial state by default, so the explicit hold command
   was redundant, not missing — the arm's actuator gains are genuinely too
   weak to hold this pose statically, even with an active target already
   commanded.
4. **Second fix (test-local only, not touching the shared
   `tasks/ar4/robot_cfg.py`): temporarily boosted the arm actuator's own
   stiffness/damping (40/4 -> 4000/200) inside the diagnostic script
   before constructing the env.** This held the arm genuinely fixed
   (`arm_max_drift` settled at ~0.0127 rad and stayed flat, confirmed via a
   new printed diagnostic field) — and revealed jaw2's TRUE, unconfounded
   behavior for the first time: **completely frozen at -0.014 across ALL
   three commanded targets (0, -0.014, -0.007)** — not sign-inverted, not
   slow, just inert. This single result retroactively explains every prior
   "opposite end"/"pinned at a limit" signature in this workstream as an
   artifact of the arm-swing confound, not evidence about jaw2's own drive
   at all.

**Root cause, found via direct USD inspection (a fast, lock-free,
GPU-free `SimulationApp({"headless": True})` + `pxr` check — no need for
the full task env): `gripper_jaw1_joint` carries a real `PhysicsDriveAPI:linear`
(`prim.GetAppliedSchemas()` includes it, type=acceleration,
stiffness=625.0, damping=0.0, maxForce=3.4e38); `gripper_jaw2_joint`
carries NO DriveAPI schema instance at all** (only
`PhysicsJointStateAPI:linear`, `PhysxJointAPI`, `IsaacJointAPI`). Both
joints are `PhysicsPrismaticJoint`s (axis X) — confirming the 0.014 unit is
a real 14mm linear stroke, not degrees, and that the `PhysxMimicJointAPI`'s
own `:rotX` instance name (removed in `2576e94`) was just PhysX's mimic
schema's generic multi-purpose-axis naming convention, not evidence the
joints were revolute.

This makes sense in hindsight: before `2576e94`, `gripper_jaw2_joint` was a
URDF mimic joint — the importer only authors an independent `DriveAPI` on
a mimic joint's *reference* joint (jaw1), since the mimic's own gearing
constraint was meant to be jaw2's sole actuation mechanism. Removing the
mimic constraint (`2576e94`, itself a real and necessary fix — the
mimic-vs-actuator physics conflict it targeted was genuinely real) stripped
jaw2's only actuation mechanism and never gave it an independent one to
replace it. Confirmed via a targeted research subagent read of Isaac Lab's
own source
(`isaaclab/assets/articulation/articulation.py`'s actuator-processing
path): `ImplicitActuatorCfg` writes `stiffness`/`damping` via
`root_physx_view.set_dof_stiffnesses`/`set_dof_dampings` unconditionally,
with no `DriveAPI`/`HasAPI` check anywhere in that call chain — it silently
"succeeds" (no error, no warning) writing gains to a DOF whose PhysX drive
object was apparently never created in the first place, an apparent silent
no-op at the PhysX level (closed-source PhysX internals beyond what Isaac
Lab's own Python source can confirm further).

**Fix**: new function `_add_gripper_jaw2_drive` in `scripts/build_asset.py`
(called right after `_remove_gripper_jaw2_mimic_constraint` in `main()`),
authoring `UsdPhysics.DriveAPI.Apply(jaw2, "linear")` mirroring jaw1's own
authored type/stiffness/damping/maxForce (these authored values only need
to give PhysX a real drive object to attach to — `ImplicitActuatorCfg`
overwrites the actual runtime gains regardless, per the same source
reading). Applied directly to the already-built asset in the
`rl-ar4-fixes-transfer` worktree via a small standalone script calling the
new function (avoids a full URDF re-import; the committed `build_asset.py`
change means any *future* full rebuild includes this fix automatically).

**Re-ran `scripts/_verify_gripper_mirror_fix.py` after the fix — clean,
complete win, no remaining asymmetry:**

```
[CLOSE step   0] jaw1=+0.01383  jaw2=-0.01383
[CLOSE step  30] jaw1=+0.00000  jaw2=-0.00000
[CLOSE step  59] jaw1=+0.00000  jaw2=+0.00000
[OPEN  step   0] jaw1=+0.00017  jaw2=-0.00017
[OPEN  step  30] jaw1=+0.01400  jaw2=-0.01400
[OPEN  step  59] jaw1=+0.01400  jaw2=-0.01400
MIRROR CHECK (jaw1 ~= -jaw2 in both states, sum ~= 0): PASS
[MID(-0.007) step   0] jaw2=-0.01392
[MID(-0.007) step  30] jaw2=-0.00685
[MID(-0.007) step  59] jaw2=-0.00701
-> jaw2 converged TOWARD -0.007 (correct tracking)
```

jaw1 and jaw2 mirror each other at every single printed step in both
CLOSE and OPEN, not just at the final settled value, and the isolated
mid-range sweep shows jaw2 converging cleanly to its own commanded target
with a normal PD convergence shape matching jaw1's. **This closes the
jaw-mimic-vs-actuator dynamics conflict question (ROADMAP item 4) with a
positive, verified result** — not a partial fix or a new open question.

**Separate, not-yet-fixed finding, flagged but out of this pass's scope**:
the arm's own actuator gains (`stiffness=40, damping=4,
effort_limit_sim=20.0`, `tasks/ar4/robot_cfg.py`'s "arm" actuator) are too
weak to hold the arm's pose statically against gravity — real physical
sag confirmed (+0.4748m -> +0.1988m gripper height over ~1-2 seconds of
sim time with the arm's last commanded target never re-issued). Whether
this matters for RL training itself is unclear and NOT tested here — a
policy issues fresh joint targets every control step (unlike this static
diagnostic, which sets one target and holds it), which may compensate in
practice, and `tasks/ar4/pickplace_graspgoal_env_cfg.py`'s own
`arm_ground_contact_penalty`/"heavily punish it for collision w the
ground" comments suggest the project's reward design already anticipates
arm gravity-sag as a real hazard. Logged to `BACKLOG.md` as a candidate
follow-up (bump arm stiffness/damping, or confirm via video that RL
training doesn't exhibit visible arm droop) rather than fixed here —
out of this pass's scope (gripper jaw fix only).

### Scripted (non-RL) grasp validation, same session: jaw2 fix confirmed sufficient at the gripper level, but a pre-existing classical-IK precision problem (Hypothesis 1) blocks an actual grasp

With jaw2 now tracking correctly, ran `scripts/grasp_demo_v2.py` (grid
search + bounded-step DLS polish, then a phased pick/lift/hold/release
sequence — Experiment 11's incremental-IK precedent, reused as-is) against
the fixed asset, three times, watching the recorded video each time (not
trusting printed metrics alone):

1. **First run**: both waypoints' DLS *polish* step made the IK residual
   WORSE than the grid search's own coarse seed (grasp 0.035m → 0.160m,
   pregrasp 0.005m → 0.041m) — a real bug in the polish loop itself (no
   "keep best across rounds" tracking, so a late divergent round could
   overwrite an earlier, better one). Video confirmed the gripper never
   approached the cube at all.
2. **Fix**: added a regression guard to `grid_search_then_polish` — track
   the best (residual, joint config) seen across every round including
   the grid seed itself, and restore that if the last round wasn't the
   best. Re-ran: residuals correctly stayed at the grid search's own good
   values (0.035m / 0.005m) instead of regressing. But PHASE 2 (moving
   from pregrasp to grasp_q) still showed a 1.42rad max joint tracking
   error after a 90-step settle — the arm never actually reached its
   commanded pose — and the cube still never moved in the video. This is
   the SAME arm-actuator-gain weakness found above (jaw2 diagnostic),
   showing up here as a large tracking error during a real multi-joint
   move, not just static droop.
3. **Applied the same test-local stiffness/damping boost (40/4 → 4000/200,
   arm actuator only, not committed to `tasks/ar4/robot_cfg.py`) used for
   the jaw2 diagnostic.** PHASE 2's max joint error dropped to 0.026rad —
   the arm now genuinely reaches its commanded pose. **But the cube still
   never moved** (`cube z` exact `0.0060m` throughout CLOSE/lift/hold,
   confirmed in the video). The remaining IK residual (0.033m grasp,
   0.007m pregrasp) is the reason: the cube is `0.012m` (12mm) per edge —
   a 33mm positioning miss is nearly 3x the cube's own size, more than
   enough to close the gripper around empty air next to it.

**Verdict: this is Hypothesis 1 (the classical-IK positioning miss),
already documented above as this project's own longest-standing AR4
finding — a single-Newton-step DLS solver trapped in a local minimum in a
poorly-conditioned kinematic region, a property of the standalone
waypoint-jumping demo scripts specifically (not the asset, not the
gripper, not — as of this session — the arm's actuator gains, both now
independently fixed/confirmed-adequate).** This session's two real fixes
(gripper jaw2 drive, arm actuator gains for this validation) were
necessary to even cleanly ISOLATE this as the remaining blocker — before
them, weak/absent actuation on one or both of the arm and gripper would
have made it impossible to tell whether a failed grasp was a positioning
problem or an actuation problem. Now it's unambiguous: positioning is the
sole remaining blocker for a scripted grasp on this asset.

**Deliberately not pursued further in this pass**: improving the classical
IK methodology itself (finer grid search, a proper analytic/closed-form
solver, or a different global-optimization approach) would be a genuine
new mechanism/methodology change, not a parameter tweak or bug fix —
CLAUDE.md's Tier 1 gate (falsifiable hypothesis + literature/precedent
research before implementation) applies to that kind of change, and it's
outside this task's authorized scope (fix the gripper-jaw diagnostic,
validate with the existing scripted-grasp tooling). Flagged to
`BACKLOG.md`/`ROADMAP.md` as the concrete next step for whoever picks this
up: either invest in a better classical-IK solving method for these
standalone scripts (following Tier 1 process), or note that this doesn't
block RL-driven grasping specifically — Experiment 11's own finding
(this same article, Hypothesis 1) is that continuous incremental IK
*driven by an RL policy every control tick* already produces real
sustained antipodal contact on this platform, so this positioning problem
may be specific to single-big-jump classical scripts rather than a
fundamental limit for AR4 grasping generally.

## UPDATE 2026-07-23 (ar4-grasp-z-envelope task): Z-height envelope mapped directly (smooth, not a cliff), joint_3 confirmed as the binding constraint by direct margin data, bearing sweep rules out approach-direction as a fix, and a real-robot-deployability check confirms the shortfall is NOT a teleport-search artifact - still no lift

Tasked with directly answering the prior session's own flagged follow-up:
map the reachable Z-height envelope at the default cube position/bearing via
`--grasp-height` in fine increments through the already-validated
incremental-descent method, cross-reference each joint's own live margin at
each height to identify the actual binding constraint, and test whether a
different approach bearing (not just reach distance, already tested)
relieves the conflict. Two new CLI capabilities were added to
`scripts/grasp_demo_v2.py` to do this in a single Isaac Sim launch each
(avoiding per-point app-startup overhead): `--z-sweep` (a list of target
heights, each re-settled to PREGRASP's own converged config first so sweep
points don't compound) and `--bearing-sweep`/`--bearing-sweep-radius` (a
list of bearing angles at a fixed radius, each running its own full
seed-search + PREGRASP polish + descent). Both exit after logging results,
before the one-shot GRASP solve / phased pick execution.

**Z-height sweep result (default bearing/reach, 9 heights from 9mm to
41mm): a smoothly growing shortfall, NOT a hard cliff.**

| Target height (m) | pos residual (m) | Z-axis residual (m) | joint_3 margin (rad) |
|---|---|---|---|
| 0.041 | 0.00153 | +0.00148 | 0.1314 |
| 0.037 | 0.00190 | +0.00181 | 0.1371 |
| 0.033 | 0.00117 | -0.00070 | 0.1357 |
| 0.029 | 0.00296 | -0.00146 | 0.1366 |
| 0.025 | 0.00569 | -0.00289 | 0.1324 |
| 0.021 | 0.00984 | -0.00696 | 0.1259 |
| 0.017 | 0.01405 | -0.01107 | 0.1174 |
| 0.013 | 0.01877 | -0.01517 | 0.1013 |
| 0.009 (true grasp height) | 0.02331 | -0.01918 | 0.0843 |

Both the Z-residual and joint_3's own margin (the smallest, and by far the
fastest-shrinking, of any joint's margin at every height - joint_1/4/6 stay
essentially flat near their own full range, joint_2/5 shrink only mildly)
degrade smoothly and monotonically as the target height drops - there is no
sudden jump/cliff at any specific height. **Critically, joint_3 never
actually reaches zero margin even at the true 9mm target (0.0843rad, ~4.8
degrees of travel still remaining)** - this rules out "joint_3 physically
hits its hard stop" as the literal mechanism, even though joint_3 is
unambiguously the binding/tightest constraint by a wide margin over every
other joint. The more accurate characterization: this is a **Jacobian-
conditioning/reachability-envelope effect as the arm approaches joint_3's
boundary**, not a literal hard-limit collision - consistent with, and a
direct sharpening of, the "soft multi-joint reachability-envelope boundary"
language earlier sessions used at the farther 32cm reach position.

**Bearing sweep result (7 bearings, -60 to +60 degrees off the default
straight-ahead direction, same 0.275m radius, true 9mm grasp height): the
Z-shortfall is essentially BEARING-INDEPENDENT.**

| Bearing (deg) | pos residual (m) | Z-axis residual (m) | joint_3 margin (rad) |
|---|---|---|---|
| -60 | 0.01990 | -0.01921 | 0.1194 |
| -40 | 0.02057 | -0.01921 | 0.1128 |
| -20 | 0.02232 | -0.01920 | 0.0953 |
| 0 | 0.01727 | -0.01699 | 0.1968 (but rot_err=0.199rad - joint_6 pinned at its own hard limit for this bearing/heading choice specifically, a different, orientation-side deadlock, not the Z-shortfall mechanism) |
| +20 | 0.02257 | -0.01920 | 0.0921 |
| +40 | 0.02068 | -0.01920 | 0.1113 |
| +60 | 0.01987 | -0.01922 | 0.1187 |

Six of the seven bearings converged cleanly in orientation (rot_err
0.008-0.021rad) yet ALL SEVEN land on the same ~19.2mm Z-shortfall to
within 0.02mm - a remarkably tight, direction-independent signature. This
directly answers the standing "does a different approach bearing help"
question from the prior two sessions: **no** - this is not a property of
the default straight-ahead approach direction specifically, it reproduces
identically across a full 120-degree bearing sweep. Combined with the
already-established finding that reach distance (20/27.5/32cm) and tilt
angle (0/10/15/25/30 degrees) also don't resolve it, this now rules out
every "just approach differently" candidate this investigation has tried:
bearing, reach distance, and tilt all leave the same shortfall in place.

**Scene/table-height sanity check: no calibration mismatch found.** Direct
comparison of `tasks/ar4/objects_cfg.py`'s raw `CUBE_CFG` (`pos=(0.20, 0.28,
0.006)`), `tasks/ar4/pickplace_graspgoal_env_cfg.py`'s cube spawn
(`(0.20, 0.28, 0.006)`), and `tasks/ar4/pickplace_mirror_env_cfg.py`'s
recentered spawn (`(0.0, 0.275, 0.006)`, the scene `grasp_demo_v2.py`
actually uses) all agree on a cube resting height of `z=0.006` (half the
cube's own 12mm edge, i.e. resting directly on a table top at `z=0`) -
consistent with this script's own `GRASP_AT_HEIGHT=0.009` (3mm above the
cube's center, a reasonable pinch height for a side-approach grasp of a
12mm cube). No scene-setup/calibration bug found here; this is not a
contributing factor.

**Deployability check (coordinator-directed, addressed before treating any
of the above as settled): does the Z-shortfall finding depend on a
simulation-only teleport-based search?** `_find_best_seed` (used by every
prior session's PREGRASP solve, including this task's own z-sweep/bearing-
sweep above) calls `write_joint_position_to_sim` to instantly snap the
robot through several candidate configs and score each before committing -
a real AR4 can never do this. Two new mechanisms were added and tested
directly against this concern:

1. **`--deployable-seed`'s bounded local "wiggle" retry
   (`_wiggle_and_resolve`): starting from HOME_Q (the robot's actual
   post-reset state, `tasks/ar4/robot_cfg.py`'s own all-zero init_state -
   not a special case, the real starting pose) with NO teleportation
   anywhere, and retrying via small (<=0.3rad, ~17 degree) per-joint
   perturbations commanded through normal PD-driven `env.step` motion (not
   `write_joint_position_to_sim`) if the direct resolve doesn't converge.
   Result: FAILED to converge in 7/7 attempts (1 direct + 6 wiggles) - every
   attempt got stuck at a catastrophic 1.03-1.40rad (59-80 degree) rotation
   error, nowhere near the 0.05rad convergence threshold, and no bounded
   local perturbation ever escaped this basin.** This is a genuinely
   important finding in its own right, independent of the Z-height
   question: PREGRASP's orientation-resolve has a real, severe basin-of-
   attraction problem starting from HOME_Q that small dither motions cannot
   fix - the good basin the teleport search finds is NOT locally reachable
   from HOME_Q via bounded perturbation.
2. **`--fixed-posture-move`: one single, deliberate, real PD-driven move
   (not a teleport - an ordinary commanded joint move, physically identical
   in kind to Phase 0 of the phased execution) from HOME_Q to the
   already-established `KNOWN_GOOD_PREGRASP_Q` reference posture, THEN the
   normal resolve.** This converged immediately: `pos=1.5mm, rot=2.7
   degrees` on the FIRST direct resolve attempt, zero wiggles needed.
   Running the full pipeline from there (real move -> resolve -> the
   already-validated incremental descent) to the true 9mm height gave
   `pos_err=17.1mm, xyz Z-residual=-17.1mm` - **essentially the SAME
   Z-shortfall magnitude as the teleport-search baseline (19.2mm at this
   bearing)**, though this run's final rotation error (0.1885rad, ~10.8
   degrees) was noticeably worse than the teleport baseline's (0.0045rad)
   - the descent drifted into a somewhat different orientation branch
   partway down in this particular run, a real but secondary difference
   from the exact basin the teleport search's broader candidate pool
   happened to find.

**What this means.** The core Z-height-shortfall finding is **not an
artifact of the teleport-based multi-seed search** - it reproduces (17mm,
even marginally better than the 19mm teleport baseline) under a pipeline
that is honestly, fully real-robot-deployable: one deliberate commanded
move to a known good reference posture (itself a completely ordinary,
physically-executable robot action - not a search, not a teleport, not
even an online decision, just "move here first"), followed by the
already-validated continuous-DLS-resolve + interpolated-descent mechanism
(all genuinely real: Jacobian-frame correction, EE-offset correction,
per-physics-step continuous resolve, incremental height descent). This
resolves the coordinator's concern in the direction the coordinator's own
decision tree anticipated for a wiggle-failure outcome: bounded local
perturbation cannot substitute for a good initial guess, but a single
smarter deliberate starting posture can, and does, work as well as (and
without needing) the teleport-assisted search. The `_find_best_seed`
mechanism as currently written remains a real code-cleanliness/deployability
gap (it should be replaced with exactly this fixed-posture-move pattern, or
a small closed-form/geometric heuristic for the initial posture, going
forward) - flagged to `BACKLOG.md` - but it was NOT hiding or fabricating
the Z-height finding itself.

**Gripper open/closed state during measurement (separate, coordinator-
raised concern, addressed directly): confirmed OPEN throughout every
measurement in both the z-sweep and bearing-sweep.** Both `_settle_at` and
`polish_from_seed` hardcode `action[:, num_arm_joints] = GRIPPER_OPEN`
(`=1.0`) on every single `env.step` call in this script, and neither of
this session's new sweep code paths ever calls the `PHASES` loop (the only
place `GRIPPER_CLOSE` is ever commanded). Verified against Isaac Lab's own
`BinaryJointPositionAction.process_actions`
(`isaaclab/envs/mdp/actions/binary_joint_actions.py`): `binary_mask =
actions < 0` selects close, so `GRIPPER_OPEN=1.0` (`>=0`) unambiguously
maps to the open command
(`GRIPPER_OPEN_COMMAND_EXPR`, jaw1/jaw2 at `+-0.014`, a real ~28mm
aperture) at every step this session measured against. Whatever prompted
the "gripper looked closed" visual observation, it was not either of this
session's own sweep runs - most likely a stale/leftover frame from a
different (pre-existing, this-session-unrelated) process, since this
session's own code never issues a close command outside the (never-reached,
in sweep mode) phased pick sequence.

**Verdict on the standing task question.** The Z-height reachability floor
at the cube's true ~9mm grasp point is now confirmed, by four independent
and mutually corroborating lines of evidence (the original descent-
continuity session's 4-configuration test, this session's fine-grained
Z-sweep, this session's 7-bearing sweep, and this session's real-deployable-
pipeline retest), to be a **genuine, method-independent, direction-
independent kinematic property of this arm reaching this specific low
height with a near-vertical wrist** - not a search artifact, not a bearing
artifact, not a scene-calibration bug. It is best characterized as a soft
Jacobian-conditioning/reachability-envelope effect tied most closely to
joint_3 (elbow), not a literal hard-limit collision (margin never reaches
exactly zero). **No grasp+lift was achieved this session** (no run reached
the phased pick-and-place stage - all sweep runs deliberately exit before
it, per their own diagnostic design). Per this task's own instructions, the
concrete next-step candidate this evidence supports - adjusting the cube's
spawn position/height closer to this arm's comfortable envelope - is
flagged for the controller to weigh in on (it could affect other AR4
experiments' cube-randomization ranges) rather than applied unilaterally
here.

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

## UPDATE 2026-07-22 (later, ar4-grasp-ik-precision task): Hypothesis 1 (the classical-IK positioning miss) re-root-caused entirely — THREE independent bugs found and fixed, real physical contact restored, but a full lift is still not achieved

Tasked with closing the "~3.3cm classical-IK residual, nearly 3x the cube's
own 12mm size" gap left by the session above and getting a real, verified
classical-IK grasp+lift working on `scripts/grasp_demo_v2.py`. What
actually happened supersedes the prior "single-Newton-step DLS trapped in a
local minimum" characterization almost entirely: the true story is three
separate, previously-undiagnosed bugs, only one of which is really about
IK solver mechanics at all.

**Bug 1 (dominant): `robot.root_physx_view.get_jacobians()` returns the
Jacobian in the WORLD frame, but every AR4 classical demo script feeds it
directly into `DifferentialIKController` alongside ROOT-frame position/
orientation vectors (via `subtract_frame_transforms`).** Confirmed against
Isaac Lab's own reference implementation
(`source/isaaclab/test/controllers/test_operational_space.py`'s
`_update_states()`), which explicitly rotates the Jacobian into the root
frame first (`jacobian_b = ...; jacobian_b[:, :3] = R_root_inv @
jacobian_w[:, :3]`) before combining it with root-frame quantities. Every
AR4 script (`grasp_demo.py`, `grasp_demo_v2.py`, `oracle_rollout.py`,
`interactive_joint_demo.py`'s closed-form-3DOF path excepted - see below)
copied Isaac Lab's own official tutorial
(`scripts/tutorials/05_controllers/run_diff_ik.py`) verbatim, which skips
this rotation - harmless there because that tutorial's Franka/UR10 scene
uses an identity-orientation base. AR4's base carries a real 180-degree yaw
(`tasks/ar4/robot_cfg.py`'s `init_state` `rot=(0,0,0,1)`), so skipping the
rotation silently mirrors the X/Y correction direction of every DLS step.

A live instrumented diagnostic this session (`scripts/_diag_ik_grasp_convergence.py`)
caught this directly: the polish loop's per-round distance INCREASED
monotonically for 39 straight rounds (`0.42m -> 0.33m` was actually the
*good* direction relative to later rounds, which climbed to `0.61m`), with
joint_2/joint_3 alternating between exact hard-limit values in a stable
3-round limit cycle (`scripts/_diag_ik_grasp_teleport_trace.py` further
isolated this as a real, non-transient physical state, not measurement
noise - see Bug 2 below for why that distinction mattered). Rotating the
Jacobian into the root frame (`scripts/_diag_polish_jacobian_frame_fix.py`)
immediately flipped this to genuine, monotonic convergence
(`0.14m -> 0.03m` in ~15 rounds, then a stable plateau - a real local
optimum, not a divergence). Multiple different starting seeds converged to
different local optima in the 1.9cm-3.3cm range depending on the wrist's
starting orientation - a real property of this redundant 6DOF arm reaching
a 3DOF position target (multiple basins), not a remaining bug. Fixed via a
new `_world_jacobian_to_root_frame()` in `scripts/grasp_demo_v2.py`.

**Bug 2: the ORIGINAL grid search's own reported "best" distance
(`0.033m`, matching the "~3.3cm" figure the prior session's UPDATE
reported) was itself a measurement artifact - the true settled residual for
that exact reported config was `0.42m`, a >10x discrepancy.** The original
`grid_search_then_polish`'s grid loop only allows `GRID_SETTLE_STEPS=15`
unsettled steps per candidate, in a raster (i,k) traversal that produces a
discontinuous ~2.5rad jump in joint_3 every time the outer loop (j2)
advances. With no velocity reset and no teleport between candidates, many
"good" readings were caught mid-swing while the arm was still decelerating
from a wildly different previous candidate - not a real static equilibrium
for the reported joint config at all. Directly confirmed by writing the
exact reported-best config to the sim via `write_joint_position_to_sim` +
an explicit `write_joint_velocity_to_sim` zero + a genuine 100-step hold
(`scripts/_diag_ik_grasp_teleport_trace.py`): joint_2/joint_3 barely moved
from the commanded values (confirming this WAS a real, low-velocity
config), yet the settled distance was `0.42m`, not `0.033m` - meaning the
grid search's own convergence check simply cannot be trusted as written.
Fixed by replacing the 2D raster grid entirely with a small set of diverse,
genuinely-settled `(j2, j3, j5)` candidate seeds
(`_find_best_seed()`/`_settle_at()` in `scripts/grasp_demo_v2.py`, each
evaluated via a clean teleport + explicit zero-velocity write + a real
hold) - a multi-seed search is needed (not a single fixed seed) precisely
because Bug 1's fix still leaves multiple local optima, and picking the
best among several seeds finds a materially better basin than any one seed
alone.

**Bug 3 (found via video review, AFTER fixing 1 and 2): this script's
target was link_6's own raw origin, not the actual gripper jaw pinch
point.** `robot_entity_cfg` controls body `link_6` directly, and every
waypoint's Cartesian target was set to put link_6's own origin at the
computed grasp position - but the real gripper jaw pinch point is offset
`0.036m` along link_6's local +Z axis (`_EE_OFFSET`, the SAME constant
`tasks/ar4/pickplace_env_cfg.py`'s `FrameTransformer` already uses for the
RL env's own observations - measured there, per that module's own comment,
via direct `robot.data.body_pos_w` readings on the gripper jaw links -
never previously applied in any classical demo script). After fixing Bugs 1
and 2, a first live run achieved a clean `<=15mm` link_6-to-target residual
and genuinely excellent joint tracking (`<=13mm` max joint error in every
phase) - yet the cube's z-height never changed even a fraction of a
millimeter across CLOSE/lift/hold. Video review (`ar4_grasp_demo_v2.mp4`,
top-down `perception_camera`) showed the gripper visibly NOT overlapping
the cube in any frame across the whole sequence. Fixed via a new
`_ee_point_pos_and_jacobian()`: computes the true pinch point's position
(`ee_pos + R @ offset_local`) and its own Jacobian
(`J_pos - skew(R @ offset_local) @ J_ang`, the standard rigid-offset-point
velocity relation) and drives THAT toward the target instead of link_6's
raw origin.

**Bug 4 (found investigating Bug 3's video, turned out to be the single
biggest position error of all): `CUBE_POS_W = (0.20, 0.28, 0.009)`,
hardcoded identically in every classical demo script, does not match where
the cube actually spawns in the scene these scripts use.**
`tasks/ar4/objects_cfg.py`'s raw `CUBE_CFG` does default to
`(0.20, 0.28, 0.006)`, but `tasks/ar4/pickplace_mirror_env_cfg.py`'s
`Ar4PickPlaceMirrorSceneCfg` (the scene `Ar4GraspVerifyEnvCfg` - and hence
every classical demo script - actually builds on)
`.replace(init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.275,
0.006)))`s it, "recentered to the workspace midpoint" per that module's own
comment, so `reset_cube_position`'s randomization range in the full RL env
can cover `_WORKSPACE_X`/`_WORKSPACE_Y` symmetrically. `Ar4GraspVerifyEnvCfg`
itself has no `events` field at all (confirmed by reading it directly), so
this bare verification env never randomizes the cube - it just sits,
unmoving, at this recentered `(0.0, 0.275, 0.006)` point every single reset
- but every classical demo script's own `CUBE_POS_W` constant still pointed
at the OLD, pre-recentering default. Confirmed directly: a fresh
`env.reset()`'s actual `env.scene["cube"].data.root_pos_w` reads
`[0.0, 0.275, 0.006]`, not `[0.20, 0.28, 0.006]` - a ~20cm real targeting
error, independent of (and dominating) Bugs 1-3 above. This alone was
sufficient by itself to guarantee the gripper never got near the cube,
regardless of how precise the IK solve was. Fixed by correcting
`CUBE_POS_W` to `(0.0, 0.275, 0.009)` in `scripts/grasp_demo_v2.py`.
**`grasp_demo.py` has the identical wrong constant
(`CUBE_POS_W = (0.20, 0.28, 0.009)`, confirmed via direct grep) - not fixed
there this pass, flagged as a follow-up below.**

**Verified result after all four fixes, multi-seed-retuned for the
corrected target position:** `PREGRASP` converges to `1.8mm` (excellent -
well under the 12mm cube). `GRASP` (the much harder waypoint - 9mm off the
ground, requiring several joints near the edge of their comfortable range)
converges to a genuine, reproducible `10.5mm` - a real, substantial,
independently-verified improvement over the divergent (`0.42-0.6m`,
Bug-1-unfixed), the previously-believed-but-fictional (`3.3cm`,
Bug-2-unfixed), and the link_6-not-fingertip (`15mm` but 36mm+20cm off the
real cube, Bugs 3-4 unfixed) baselines. **Real physical contact and cube
displacement is now confirmed on every run** (cube position/height visibly
perturbed by 1-3cm and briefly bumped in Z during CLOSE/lift, watched via
video and cross-checked against printed `cube.data.root_pos_w` - a first
for this entire investigation; every prior run this session and the one
before it showed the cube's position exactly unchanged to the last decimal
throughout CLOSE/lift/hold). **A full stable pinch+lift was NOT achieved**
- the cube gets pushed/dragged sideways rather than enclosed and lifted;
`cube z` returns to its resting `0.0060m` in every run after a brief bump.

**Diagnosed remaining gap: grasp ORIENTATION, not position, is now the
likely blocker - and it is capped by the same basin's own joint-limit
constraint, not fixable by a simple parameter tweak.** A direct orientation
check (`scripts/_diag_check_orientation.py`) at the verified-best `GRASP_Q`
found link_6's approach axis (local +Z, the `_EE_OFFSET` direction) points
mostly horizontally (dominant `0.943`-magnitude component) with only an
~18-degree downward tilt, while the jaw-slide axis (local +X) is nearly
pure horizontal - a side-approach geometry, not a top-down one. Manually
combining this orientation with `_EE_OFFSET` shows the actual computed
fingertip lands about `10mm` ABOVE the intended contact height (above the
cube's own top face at `z=0.012m`) - the dominant component of the
waypoint's `10.5mm` residual is a Z-height shortfall, not an X/Y bearing
error, consistent with the gripper's bottom edge clipping the cube's top
and shoving it rather than enclosing it. **Tested directly: lowering
`GRASP_AT_HEIGHT` by the diagnosed ~10mm to compensate made the residual
WORSE (20mm), not better** - the multi-seed search converged to
essentially the SAME joint configuration regardless of the lower target,
confirming this specific basin's descent is genuinely capped (most likely
by the same joint-limit-boundary behavior found throughout this
investigation - several of this session's local optima pin one or more
joints at/near their hard limits at this low approach height), not a
simple re-aim-lower fix. Reverted to the better-verified `0.009m` height.

**What this means for Hypothesis 1's status.** The "single-Newton-step DLS
trapped in a local minimum" framing from the 2026-07-22 (earlier) UPDATE
above is now shown to have been almost entirely a MEASUREMENT ARTIFACT
(Bug 2) compounding a genuine sign/frame bug (Bug 1) and two independent,
much larger targeting bugs (Bugs 3-4) that had nothing to do with solver
mechanics at all - "the classical-IK solver gets stuck a few cm short" was
the wrong diagnosis for what was actually happening. With all four bugs
fixed, the position-only DLS solver itself now behaves exactly as the
textbook describes: monotonic convergence to a genuine local optimum,
`10.5mm`/`1.8mm` precision, well past what's needed to CONTACT a 12mm cube
(and contact is now confirmed, repeatably, for the first time this
investigation). What remains - a position-only IK formulation gives the
solver zero incentive to select a sensible pinch ORIENTATION, and the
orientation it does select (a side-approach, ~18-degree-tilted geometry)
happens to fall about 10mm short of full contact depth in a way that's
capped by a joint limit in this specific basin - is a genuinely different,
narrower, and better-characterized problem than Hypothesis 1 ever was. A
proper fix (switching to `command_type="pose"` with a deliberately-chosen
approach orientation, or searching for a DIFFERENT basin with a more
favorable elbow/wrist configuration) is a real next step but was judged
beyond this pass's "fix the classical-IK precision bug" scope - it
would need to select and justify a target orientation, which starts to
resemble a small grasp-planning design choice rather than a bug fix,
and is flagged to `BACKLOG.md` rather than attempted further here given
this pass's own budget.

**Not yet done, flagged as follow-ups:**
- `grasp_demo.py` has the identical Bug 1 (no Jacobian frame rotation) and
  Bug 4 (wrong `CUBE_POS_W`) - not fixed there this pass (this pass's
  actual grasp+lift validation used `grasp_demo_v2.py` only, per the task's
  own instruction to "reuse whatever script you just fixed"). `oracle_rollout.py`
  has Bug 1's pattern too (confirmed via grep: uses `get_jacobians()`
  directly with no `matrix_from_quat`/`quat_inv` anywhere in the file).
  `interactive_joint_demo.py` uses a closed-form 3-DOF IK (not
  Jacobian/DLS-based at all, confirmed via its own docstring/code), so
  Bug 1 does not apply there.
- Multiple different cube positions were NOT tested this pass (the task's
  own instruction to do so was conditioned on "if the first [attempt]
  succeeds" - this one didn't reach a full lift, so that condition wasn't
  met). The core fixes (Bugs 1/3/4) are structural/positional and should
  generalize to any cube position; Bug 2's replacement (multi-seed search)
  is deliberately seed-list-tunable per target and was re-tuned once
  already this session when the target position changed
  (`scripts/_diag_multiseed_corrected_target.py`), demonstrating the
  methodology transfers, even though the specific seed LIST is
  target-position-specific.
- Diagnostic scripts from this investigation kept in `scripts/` as a
  historical record (matching this repo's existing `_diag_*.py`
  convention): `_diag_ik_grasp_convergence.py` (Bug 1's discovery),
  `_diag_ik_grasp_teleport_trace.py` (Bug 2's discovery/confirmation),
  `_diag_fixed_grid_search.py` (Bug 2's fix test), `_diag_polish_jacobian_frame_fix.py`
  (Bug 1's fix verification), `_diag_multiseed_corrected_target.py` (Bug 4's
  seed re-tuning), `_diag_check_orientation.py` (the orientation-gap
  diagnosis). Several more throwaway intermediate iteration scripts from
  this session were deleted rather than kept, to avoid clutter.


## UPDATE 2026-07-22 (later, ar4-grasp-orientation-fix task): pose-IK orientation control confirmed working correctly, but a genuine AR4 joint_3 (elbow) kinematic limit blocks a full-depth vertical grasp of this specific cube - not yet a working lift

Tasked with fixing the diagnosed orientation gap from the session above (a
position-only IK gives the solver zero incentive to pick a sensible pinch
orientation, and the basin it fell into was an ~18-degree tilt that
undershot full pinch depth) and getting a real, verified, repeatable
grasp+lift. Real progress was made on the orientation mechanism itself, but
a new, deeper kinematic constraint was found and confirmed instead of a
working lift.

**The fix: `scripts/grasp_demo_v2.py` switched `DifferentialIKControllerCfg`
from `command_type="position"` to `command_type="pose"` (relative mode),
mirroring `scripts/demo_franka_ik_dice_line.py`'s own established
`canonical_down_quat_w` precedent** - an explicit, deliberately-chosen
target orientation instead of leaving it to the arm's redundant null space.
AR4's own canonical target was built from explicit WORLD-frame basis
vectors (`_CANONICAL_{X,Y,Z}_AXIS_W`, `_build_canonical_target_quat_w()`),
not copied from Franka's own hand-frame quaternion constant (no reason to
transfer to a structurally different arm/gripper), then converted to ROOT
frame via `subtract_frame_transforms` - the same world-to-root conversion
already used for position targets, which correctly and automatically
accounts for AR4's 180-degree base yaw (a 180-degree yaw about Z leaves
world -Z indistinguishable in root frame but flips X/Y - confirmed by
direct calculation matching the printed root-frame quaternion). The
`polish_from_seed` DLS loop now drives a full 6D pose error
(`compute_pose_error`, position + axis-angle rotation) through a combined
6-row Jacobian (the existing offset-corrected position rows, plus link_6's
own unmodified angular rows - the pinch point shares link_6's rotation
exactly, only its *position* needs the rigid-offset correction).

**Verified live, independently of the scalar residual (not just trusting
the math): when NOT joint-limited, the solver genuinely converges the
gripper's approach axis to vertical.** At a 32cm-reach test position, GRASP
converged to `rot_err=0.0037rad` (~0.2 degrees) with the live axis readout
confirming it directly: local +Z (the approach/`_EE_OFFSET` axis) measured
`[0.000, 0.005, -1.000]` in root frame - essentially exactly world -Z. This
is a real, working fix to the originally-diagnosed problem (an
uncontrolled, arbitrary null-space orientation) - the mechanism itself is
correct.

**Two real bugs found and fixed getting there, both through live evidence,
not by inspection:**

1. **Jaw-slide-axis heading choice deadlocked joint_6 at its own hard
   limit.** The jaw-slide axis (local +X) has no principled "correct"
   horizontal heading for a symmetric cube grasp, so it was initially set
   to world +X arbitrarily - but a live run converged GRASP's polish to
   `joint_6 = 3.14159` (exactly pi to float precision), pinned at that
   joint's hard limit (`[-pi, pi]`), and the polish then deadlocked
   (identical residual for 80 straight rounds - a joint-limit wall, not a
   converging solve). PREGRASP's own converged `joint_6` at the same
   heading choice landed at `3.1334`, just under the same wall, confirming
   this was a real, reproducible boundary effect of the *heading choice*,
   not the target itself. Fixed by rotating the heading 90 degrees (world
   +Y instead of +X) - not a claim that +Y is universally correct, just
   that this heading is a genuinely free parameter worth choosing
   deliberately to avoid a known limit rather than leaving to accident.
2. **GRASP's own seed search picked an orientation-incompatible seed because
   it scored candidates on position alone.** The old position-only-tuned
   `KNOWN_GOOD_GRASP_Q` constant always won the (position-only) seed search
   for GRASP - but its orientation turned out to be ~163 degrees (2.85rad)
   from canonical, and the subsequent polish got permanently stuck at that
   same ~163-degree error (identical residual for 80 rounds). Critically,
   PREGRASP's own seed ALSO started from an almost-identical ~171-degree
   (2.98rad) initial rotation error and successfully corrected it (2.98rad
   -> 0.0059rad in 20 rounds) - ruling out "bad seed orientation" as a
   general problem and showing the DLS mechanism itself works fine when not
   joint-limited. Fixed two ways: (a) `_find_best_seed` now scores
   candidates on a combined position+orientation score
   (`ORIENTATION_SCORE_WEIGHT`, a documented judgment-call constant), not
   position alone; (b) GRASP is now solved AFTER PREGRASP and seeded from
   PREGRASP's own converged (already-canonical) config, since it's only 5cm
   away and matches how the phased execution actually moves the arm anyway
   (pregrasp_q -> grasp_q as consecutive nearby waypoints, not independent
   teleports).

**The real remaining blocker: AR4's own joint_3 (elbow) hard limit
(`[-1.5533, +0.9076]` rad, i.e. roughly -89 to +52 degrees - read directly
from `robot.data.joint_pos_limits`, `soft_joint_pos_limit_factor=1.0` so
this is the actual hard limit, not a narrowed soft one) prevents the arm
from simultaneously reaching the cube's true low grasp height (9mm) AND
holding a fully vertical wrist orientation - confirmed as a genuine,
repeatable kinematic property of this arm, not a single-position
coincidence, by testing 3 different reach distances along the same
bearing (via a new `--cube-xy` CLI override, teleporting the cube live
before reading its pose as ground truth):**

| Reach (cube distance from base) | GRASP result | joint_3 pinned at limit? |
|---|---|---|
| 20cm (closer than default) | 4.6cm residual (worse) | Yes - `joint_3=0.90756` vs limit `0.90757` |
| 27.5cm (task's own scene default) | 2.8cm residual | Yes - `joint_3=0.892`, ~0.9 degrees from limit |
| 32cm (farther) | 2.0cm residual | No single joint pinned, but still short - a softer, multi-joint reachability-envelope boundary |

Counter-intuitively, moving the cube CLOSER made the conflict WORSE, not
better (a real, reproducible finding, not noise) - consistent with the
physical picture that reaching a low, close-in point with a vertical wrist
requires MORE elbow flexion (deeper into joint_3's limited positive-travel
direction), not less, much like a human arm needs more elbow bend to reach
straight down close to its own base than at a longer, more extended reach.
At every reach distance, the residual is dominated by a Z-height
shortfall: the achieved pinch point lands well ABOVE the intended grasp
height (e.g. at 27.5cm: target 9mm, achieved ~39mm - a 3cm shortfall,
larger than the cube itself). **"Aim the target lower to compensate" was
retested (via a new `--grasp-height` override) at the 32cm, non-joint-
limited position specifically (to rule out this being just a repeat of the
already-known joint-limited-basin finding from the prior session) and
again made the residual WORSE (2.0cm -> 3.8cm), not better** - the same
qualitative finding as the earlier position-only investigation, now shown
to hold even in a basin where no single joint is pinned at its exact
boundary, meaning this is a genuine multi-joint reachability-envelope
property of AR4's kinematics for a vertical approach, not an artifact
of one specific joint-limit wall.

**One further mitigation attempted and NOT yet working: a deliberate,
controlled tilt (30 degrees from vertical, via a new `--tilt-deg` CLI
option built from a proper rotation-about-the-jaw-axis construction) at
the task's default cube position, as a middle ground between "fully
vertical" (kinematically capped well above the cube) and "uncontrolled
null-space result" (the original problem).** This did NOT resolve the
conflict - instead, the polish became numerically unstable, with rotation
error monotonically INCREASING round over round (0.0995rad at round 0 up
to 1.054rad by round 30, then plateauing there) rather than converging,
ending up worse than the seed's own starting orientation. This is a real,
observed instability in the bounded-step DLS polish when targeting a
non-zero, non-canonical tilt from this particular seed/basin, not
investigated further this session (flagged as a follow-up, not a dead
end) - it's possible a different seed, smaller tilt angle, or smaller
per-round rotation step bound would behave better, but this needs its own
dedicated debugging pass rather than continuing to guess tilt angles.

**No real cube contact, displacement, or lift was achieved in any run this
session** - `cube.data.root_pos_w`'s z-component stayed flat at its resting
~0.006m throughout every clean run's CLOSE/lift/hold phases (11 full runs
total, one Isaac Sim non-deterministic startup hang recovered from mid-session
via `kill -9` + relaunch, matching this project's own documented "known
gap" startup flakiness - confirmed via `ps`/`nvidia-smi` showing genuine
CPU/GPU activity with zero log progress for 22+ minutes before the kill,
not a false read). This is a genuine negative result for THIS specific
cube position/height combination under a canonical-or-near-canonical
vertical approach, not a partial success being overstated.

**What this means.** The orientation-selection MECHANISM this task set out
to fix is now demonstrably correct (verified via independent axis readout,
not just a scalar residual, at multiple positions) - this closes the
originally-diagnosed "uncontrolled null-space orientation" problem cleanly.
But it surfaces a deeper, previously-unconfirmed kinematic property: **AR4's
own joint_3 range does not comfortably support a fully-vertical top-down
grasp low enough to contact a 12mm cube resting on the table, at any of the
3 reach distances tested along this bearing.** This is now a better-
characterized, narrower problem than either the original "position-only
DLS picks an arbitrary orientation" framing or the "single basin capped by
a joint limit" framing from the prior session - it's a property of the
*orientation itself* (vertical) interacting with this *specific arm's*
elbow range, present across multiple positions and seeds, not a single
unlucky configuration. Candidate next steps (not completed here, flagged
for a future pass): (a) debug why non-zero tilt destabilizes the DLS
polish rather than just converging to a worse-but-stable orientation, (b)
try smaller tilt angles (10-15 degrees) with a smaller per-round rotation
step bound, (c) test whether a DIFFERENT bearing (not just reach distance)
relieves the joint_3 conflict, (d) accept a smaller-than-canonical tilt as
this arm's own genuine "canonical" approach angle if a stable, sufficiently
deep option is found. This does NOT reopen Hypothesis 1 (the classical-IK
positioning miss, closed by the prior session) - positioning precision
itself remains excellent (sub-cm to sub-mm when not orientation- or
joint-limit-capped); the open question is now specifically about
orientation-vs-reachability tradeoff, a narrower and better-diagnosed
question than anything this investigation has previously isolated.

**Script changes** (`scripts/grasp_demo_v2.py`): `command_type` switched to
`"pose"`; new `_build_canonical_target_quat_w`/`_build_canonical_target_quat_b`
(with optional `tilt_deg`); `polish_from_seed` now tracks/reports combined
position+rotation residual and per-axis position residual
(`_measure_rot_err`, `_measure_dist_vec`, new); `_find_best_seed` now scores
on combined position+orientation error; PREGRASP solved before GRASP,
seeding GRASP from PREGRASP's converged config; new CLI overrides
`--cube-xy`, `--grasp-height`, `--tilt-deg`, `--video-suffix` for testing
different reach distances/heights/tilts without editing the file between
runs; new `[INFO] Arm joint pos limits` printout for direct joint-limit
diagnosis going forward.

**Sources for this update**: entirely this session's own live runs
(11 full launches of `scripts/grasp_demo_v2.py` against the real Isaac Sim
scene, `logs/videos/ar4_grasp_demo_v2*.mp4`), `robot.data.joint_pos_limits`
read directly at runtime, and Isaac Lab's own `DifferentialIKController`/
`compute_pose_error`/`axis_angle_from_quat` source
(`source/isaaclab/isaaclab/controllers/differential_ik.py`,
`source/isaaclab/isaaclab/utils/math.py`) for the pose-command-mode API
this fix relies on.

## Sources

`docs/superpowers/specs/research/2026-07-20-ar4-vs-franka-root-cause-comparison.md`
(full citations for the original three hypotheses), `docs/superpowers/specs/research/2026-07-21-ar4-usd-asset-debugging.md`
(direct USD-level inspection/fixes for Hypotheses 2 and 3, plus the
Link_5/Link_6 collision fix), `CLAUDE.md` ("Platform pivot" section),
`docs/superpowers/specs/2026-07-21-ar4-franka-fixes-transfer-design.md`
and `docs/superpowers/plans/2026-07-21-ar4-franka-fixes-transfer-implementation.md`
(the 2026-07-21 H_ar4_relative transfer test above). This 2026-07-22
(later) UPDATE's own sources are entirely this session's own live
diagnostics (`scripts/_diag_ik_grasp_convergence.py`,
`scripts/_diag_ik_grasp_teleport_trace.py`, `scripts/_diag_fixed_grid_search.py`,
`scripts/_diag_polish_jacobian_frame_fix.py`, `scripts/_diag_multiseed_corrected_target.py`,
`scripts/_diag_check_orientation.py`) plus Isaac Lab's own source
(`source/isaaclab/test/controllers/test_operational_space.py`,
`scripts/tutorials/05_controllers/run_diff_ik.py`) and this repo's own
`tasks/ar4/pickplace_env_cfg.py`/`pickplace_mirror_env_cfg.py`/
`objects_cfg.py` for Bugs 3/4's ground truth.

## UPDATE 2026-07-22 (later still, ar4-tilt-fix task): Part A confirms joint_3 limit is real hardware (not an import bug); Part B fixes a genuine DLS-divergence bug and gets PREGRASP to sub-5mm at a real tilt, but GRASP itself (the true 9mm-height waypoint) hits a NEW, deeper, tilt-independent basin conflict - still no lift

Tasked with two things: (A) verify the `joint_3` `[-1.553, +0.908]` rad
limit against the real AR4 hardware's own vendor spec (this investigation's
own pattern has repeatedly found "hardware limits" that were actually
asset-import defects), and (B) fix the `--tilt-deg 30` DLS-instability
found in the prior UPDATE and get an actual validated grasp+lift.

**Part A verdict: the limit is REAL hardware, not a bug - confirmed
directly from the vendor's own URDF/config source, not secondhand
claims.** `AR4_DESCRIPTION_PATH`'s own `urdf/ar_macro.xacro` defines
`joint_3`'s limit via `robot_parameters['j3_limit_min'/'j3_limit_max']`,
loaded from `config/mk5.yaml` (the exact model `scripts/build_asset.py`
builds, confirmed via its own `ar_model:=mk5` xacro invocation):
`j3_limit_min: !degrees -89`, `j3_limit_max: !degrees 52`. Converting:
`-89*pi/180 = -1.55334 rad`, `52*pi/180 = 0.90757 rad` - matching the
built USD asset's `[-1.5533, +0.9076]` limit to 4 decimal places. Checked
all 5 shipped model variants (mk1-mk5) - identical `-89/52` limit in
every one, so this isn't even a per-model quirk. No fix applied; this is
confirmed to be the real AR4 elbow's own designed range of motion, and
the earlier sessions' framing of it as a genuine kinematic constraint
(not an asset defect) stands.

**Part B, mechanism bug found and FIXED: the polish loop's own
"solve-once-then-hold-blindly" architecture, combined with an oversized
per-round rotation step and an under-damped DLS lambda, was a real,
independently-reproducible cause of divergence at a deliberate tilt -
distinct from (and in addition to) the deeper basin-conflict finding
below.** `scripts/grasp_demo_v2.py`'s `polish_from_seed` previously solved
the DLS Jacobian ONCE per "round" then held that single computed
`joint_pos_des` open-loop for `POLISH_SETTLE_STEPS=30` physics steps
before ever re-measuring - unlike the proven-stable
`demo_franka_ik_dice_line.py`'s `_step_toward`, which re-solves the
Jacobian and takes one small bounded step EVERY physics step
(closed-loop). Three concrete, live-validated fixes:

1. **Continuous per-step re-solve** (was: one solve, 30-step blind hold).
   `polish_from_seed` now re-measures and re-solves every physics step,
   matching Franka's own proven pattern exactly - any overshoot is caught
   and corrected on the very next step instead of compounding for 29 more
   steps first.
2. **`POLISH_ROT_STEP_MAX`: `0.15rad -> 0.03rad`**, matching
   `demo_franka_ik_dice_line.py`'s own `_MAX_ROT_STEP` EXACTLY - the old
   value was 5x Franka's proven-stable bound with no stated
   justification.
3. **`LAMBDA_VAL` (DLS damping): default kept at `0.02`, but a
   `--lambda-val` CLI override added, and `0.3` found live to be the
   value that actually matters.** Live evidence: at `--tilt-deg 15
   --cube-xy 0.0 0.32` (a farther, less joint-constrained reach than the
   task's own 27.5cm default), `lambda_val=0.02` still produced the exact
   same divergence signature reported in the prior UPDATE (rotation error
   jumping from ~0.05rad to >1.3rad within ~100 physics steps, then
   plateauing - a stable-but-wrong deadlock, not a runaway blowup, once
   the continuous-resolve fix was already in place). Raising
   `lambda_val` to `0.3` (10x higher) completely eliminated this for
   PREGRASP: genuine, monotonic, textbook DLS convergence from a
   `1.56rad` initial rotation error down to a stable **`4.6mm` position /
   `0.0066rad` (`0.4-degree`) rotation residual** - re-confirmed at both
   15-degree and 25-degree tilt, same reach. This is a real, validated
   fix for a real numerical-instability bug: the near-singular-Jacobian
   region this redundant, non-spherical-wristed arm passes through while
   descending from `PREGRASP_HOVER` needs meaningfully more damping than
   Franka's own spherical-wristed kinematics ever required at the same
   nominal `lambda_val`.
4. **`STAGNATION_BREAK_STEPS=500`** added (break out of the polish loop
   early once the combined score hasn't improved for 500 consecutive
   steps) - a pure efficiency/hygiene fix, not a correctness one: without
   it, a genuinely-deadlocked run burns the full `POLISH_MAX_STEPS=3000`
   budget for zero benefit. The existing "restore best round" guard
   already made this safe (a stagnated run was never being reported as
   its own worst state, just wastefully continuing to confirm it).

**Part B, deeper finding NOT fixed: GRASP itself (the low, ~9mm-height
waypoint) hits a qualitatively different, tilt-independent basin
conflict that persists across every mitigation tried - this is now the
real remaining blocker, not the divergence bug above.** Once the
divergence bug (items 1-3 above) was fixed, PREGRASP (the higher,
`+5cm` hover waypoint) converges cleanly and reliably at every tilt/reach
combination tried. GRASP does not, and the SAME failure signature
recurred regardless of:
- **Tilt angle**: 10, 15, and 25 degrees all show it (30 was the
  original prior-UPDATE finding).
- **Reach distance**: both the task's own 27.5cm default and the
  farther, less elbow-constrained 32cm position (already known from the
  position-only investigation to be a "softer" reachability boundary)
  show it.
- **DLS damping**: `lambda_val=0.02` (default), `0.1`, and `0.3` (the
  value that fixed PREGRASP's instability) all show it - ruling out
  "just needs more damping" as the fix for GRASP specifically, even
  though damping WAS the right fix for PREGRASP.
- **Seed diversity**: extending the multi-seed search with 6 additional
  wrist-perturbed variants of PREGRASP's own converged config (nudging
  `joint_4`/`joint_6`, the two DOF `CANDIDATE_SEEDS` never varies) did
  not find a better basin - the unperturbed `pregrasp_q` seed still won
  the combined-score comparison every time.

The failure signature itself is consistent and specific: GRASP's polish
starts from a seed with a genuinely good combined score (position
`~4cm`, rotation `~3-5 degrees` - itself already too imprecise for a
clean pinch, but not divergent), then within roughly 100-200 physics
steps the rotation error jumps to `~1.1-1.4rad` (`63-80 degrees`) and
PLATEAUS there exactly (residual identical to 4 decimal places for
hundreds of consecutive steps) - a genuine, stable local optimum, not
ongoing numerical divergence. `limit_margin` diagnostics (added this
session, printed every 100 steps) confirm no single joint is pinned
exactly at its hard limit when this happens (margins mostly `>0.25rad`
from the nearest wall) - ruling out the simple "hard joint-limit wall"
framing from earlier sessions as the specific mechanism here, even
though it's clearly a *related* reachability-envelope phenomenon.
**Mechanistically, this looks like a genuine disconnected-basin property
of this arm's redundant, non-spherical wrist at the true low grasp
height: closing the last few cm of POSITION error at this height forces
a large joint reconfiguration (a `~1rad` swing was observed in a single
joint between prints) that a position+orientation-weighted DLS descent
cannot avoid without destroying the orientation it had already achieved
at the seed - not a numerical bug, a structural property of the solution
space at this specific height.**

**Net result: no grasp+lift validated this session, at any tilt/reach
combination tried, and no clean 3-4 position sweep was run (correctly,
per the task's own conditioning - the sweep was to happen "once a tilted
approach converges reliably," which never occurred for GRASP itself).**
Phased-execution video/`cube.data.root_pos_w` checks were still performed
for the one run that reached that stage (15-degree tilt, default
lambda) - `cube.z` stayed flat at its resting `~0.006m` throughout
CLOSE/lift/hold, consistent with the `~2.6cm` final residual being
larger than the cube itself. This is a genuine negative result, not an
overstated partial success.

**What this means.** Two genuinely separate problems got conflated in
the prior UPDATE's single "DLS instability at tilt" framing, and this
session split them apart: (1) a real, now-fixed numerical-robustness bug
in the polish loop's architecture and damping, validated by PREGRASP's
clean convergence at multiple tilts/reaches; and (2) a deeper,
NOT-yet-solved kinematic/basin-connectivity property specific to the low
GRASP waypoint, present across every tilt angle and reach distance
tested, that (1)'s fix does not touch. This narrows the open question
usefully: it is no longer "does the solver diverge at a tilt" (answered:
only did because of bug (1), now fixed) but specifically "why can this
arm's redundant wrist not reach a jointly position-AND-orientation-
compatible configuration at the true ~9mm grasp height, at ANY of the
tilt angles 0/10/15/25/30 degrees tried across two sessions" - a
question this session's evidence base narrows but does not close.
Candidate next steps, not attempted this session given its own time
budget: (a) let GRASP's target orientation be genuinely different from
PREGRASP's (per-waypoint orientation, not one shared canonical target) -
i.e., search for whatever orientation IS jointly reachable at 9mm height
first, rather than imposing PREGRASP's own converged orientation as a
starting bias; (b) a proper redundancy-resolution/null-space secondary
objective (explicitly steering the redundant DOF away from this specific
bad branch during the descent, rather than a single bounded-step DLS
correction); (c) accept that a genuinely different BEARING (not just
reach distance, already tested) might avoid this specific conflict,
per the still-untested candidate from the prior UPDATE.

**Script changes** (`scripts/grasp_demo_v2.py`): `polish_from_seed`
restructured from round-based-with-blind-hold to continuous
per-physics-step re-solve; `POLISH_ROT_STEP_MAX` 0.15->0.03;
`POLISH_MAX_STEPS=3000` (physics-step budget, replaces the old
`POLISH_ROUNDS`); `STAGNATION_BREAK_STEPS=500` early-exit; new
`--lambda-val` CLI override; periodic per-step diagnostic print now
includes live joint config + per-joint limit margins; GRASP's seed
search extended with 6 wrist-perturbed (`joint_4`/`joint_6`) variants of
`pregrasp_q`.

**Sources**: entirely this session's own live runs against the real
Isaac Sim scene (non-headless, `DISPLAY=:1`, desktop GPU) - roughly a
dozen full `grasp_demo_v2.py` launches varying `--tilt-deg`
(10/15/25), `--cube-xy` (default 27.5cm reach vs 32cm), and
`--lambda-val` (0.02/0.1/0.3); the vendor's own
`annin_ar4_description` URDF/YAML source
(`urdf/ar_macro.xacro`, `config/mk1-5.yaml`) for Part A;
`demo_franka_ik_dice_line.py`'s own `_step_toward`/`_MAX_POS_STEP`/
`_MAX_ROT_STEP` for the proven-stable reference pattern Part B's fix
mirrors; `robot.data.joint_pos_limits`/live joint-margin printouts for
the basin-conflict diagnosis.

## UPDATE 2026-07-22 (later still, ar4-grasp-descent-continuity task): incremental PREGRASP->GRASP height descent CONFIRMS the rotation-deadlock hypothesis, but surfaces a separate, deeper Z-height reachability floor - still no lift

Tasked with testing a specific hypothesis for the prior session's own
open finding: GRASP solved as an independent one-shot target (multi-seed
search + DLS polish) deadlocks at a stable ~1.1-1.4rad rotation error,
tilt/damping/seed-independent, with no single joint pinned at a hard
limit - "a big jump from PREGRASP's config can't reach GRASP's basin
directly," a disconnected-basin property. The hypothesis: walk the arm
down from PREGRASP's already-converged config to GRASP height in many
small continuous steps, re-solving IK every step, mirroring the pattern
that worked every other time this investigation found something
reliable (Experiment 11's RL-driven incremental IK, `demo_franka_ik_dice_line.py`'s
continuous per-step resolve, this same session's own PREGRASP-tilt fix
above).

**Implementation** (`scripts/grasp_demo_v2.py`): a key property of the
existing `polish_from_seed` (confirmed by reading it, not assumed) makes
this cheap to implement correctly - it NEVER teleports the robot to its
own `seed_q` argument, it always continues the DLS loop from the robot's
actual LIVE physical state. This means calling it repeatedly back-to-back
with no `_settle_at`/teleport in between already behaves as a genuine
continuous resolve from one call's converged end-state to the next call's
starting state - exactly the mechanism the hypothesis needs. New
`--num-descent-steps` CLI arg (default 30): interpolates ONLY the target
height from PREGRASP's converged height down to `GRASP_AT_HEIGHT` in that
many increments (x/y and orientation are already shared between PREGRASP
and GRASP, so only height needs interpolating), each sub-step solved via a
smaller per-substep step/stagnation budget (`DESCENT_SUBSTEP_MAX_STEPS=400`/
`DESCENT_SUBSTEP_STAGNATION_STEPS=150`) than the old one-shot budget.
`--num-descent-steps 1` reproduces the old one-shot independent-target
behavior for direct comparison.

**Result: the disconnected-basin/rotation-deadlock hypothesis is
CONFIRMED across 4 independent live configurations - the catastrophic
1.1-1.4rad deadlock never recurs under descent, in any of them:**

| Run | Config | Final GRASP pos residual | Final GRASP rot residual | Per-axis xyz residual (root frame) |
|---|---|---|---|---|
| 1 | 30 steps, 0° tilt, default 27.5cm reach | 17.7mm | 0.2135rad (12.2°) | `[-0.0003, -0.0049, -0.0171]` |
| 2 | 60 steps, 0° tilt, default reach | 24.2mm | 0.0044rad (0.25°) | `[-0.0000, -0.0155, -0.0192]` |
| 3 | 40 steps, 15° tilt, `--lambda-val 0.3` | 20.5mm | 0.0168rad (1.0°) | `[-0.0011, 0.0003, -0.0205]` |
| 4 | 30 steps, 0° tilt, farther 32cm reach | 19.0mm | 0.0169rad (1.0°) | `[0.0000, -0.0003, -0.0190]` |

Every run's own full per-substep printout (not just the final number) was
inspected: the descent's rotation error rises and falls SMOOTHLY across
sub-steps (e.g. run 1: `0.0427rad` at sub-step 1, dips to a genuine
minimum `~0.0041rad` around sub-step 15, then climbs gradually back up to
`0.2135rad` by sub-step 30) - a bounded, continuous degradation, never the
sudden multi-hundred-percent jump-and-plateau signature that
characterized the one-shot deadlock. `limit_margin` printouts confirm no
joint is pinned exactly at its hard limit in the final converged states
(closest observed: joint_3 margin narrowing to ~0.12-0.19rad, i.e.
7-11 degrees of remaining travel, not zero) - consistent with a
near-limit, not at-limit, regime.

**But: all 4 runs instead converge to a consistent 17-24mm position
residual, and the per-axis breakdown shows this is almost entirely a
Z-HEIGHT shortfall, not an X/Y bearing miss** - X residual is at most
1.1mm and Y at most 15.5mm across all 4 runs, while Z residual is
17-21mm in every single run (see table above). This is the SAME
Z-shortfall signature the earlier position-only investigation found
(the "ar4-grasp-ik-precision task" UPDATE above, before the orientation
fix even existed: "the achieved pinch point lands well ABOVE the
intended grasp height... the dominant component of the residual is a
Z-height shortfall, not an X/Y bearing error") - now independently
reproduced under a materially different solving methodology (continuous
incremental descent instead of one-shot multi-seed search), across 4
different tilt/reach/step-count combinations. **This is strong evidence
the Z-height shortfall is a genuine, method-independent kinematic
reachability limit of this arm at this cube height** (not an artifact of
either the original one-shot solving method or of this session's own
descent method), separate from and deeper than the rotation-deadlock
problem this task specifically targeted.

**No cube contact, displacement, or lift in any of the 4 runs** -
`cube.data.root_pos_w`'s z-component stayed flat at its resting `~0.006m`
throughout every CLOSE/lift/hold phase in all 4 logs. Video-confirmed for
run 1 (not just printed metrics): frames pulled from the demo-camera
video at the CLOSE (step 300), lift (step 400), and hold (step 500)
phases all show the gripper clearly not overlapping the cube (visible as
a small red dot on the ground plane, well clear of the gripper's
fingertip in every frame) - consistent with the printed 17.7mm residual
being larger than the cube's own 12mm size.

**What this means.** This task's own specific hypothesis - that a big
single-jump solve to GRASP (as opposed to PREGRASP, which is only 5cm
away and always converged cleanly) was the cause of the 1.1-1.4rad
deadlock because the solver couldn't cross a disconnected region of
configuration space in one step - is CONFIRMED and now closed with a
validated, reproducible fix: incremental height descent, using the
already-fixed continuous-resolve/damping machinery, avoids that deadlock
in every one of 4 tested configurations. This is a genuine, positive,
validated result for the specific question this task was asked to
answer. However, it does NOT produce a working grasp, because it
surfaces a SEPARATE problem: a ~17-24mm Z-height reachability floor at
the true ~9mm grasp height, robust to tilt angle, descent step count, and
reach distance, that was previously undetected only because the
rotation deadlock was masking it (a one-shot solve stuck at 1.1-1.4rad
rotation error never got far enough into the correct basin to reveal
what its OWN position floor would have been). The joint-limit-margin
data (no joint pinned exactly at zero margin, but joint_3 consistently
narrowing to ~0.12-0.19rad, the smallest margin of any joint in every
run) is consistent with, but does not conclusively prove, this being the
same joint_3-vs-vertical-orientation conflict documented in the prior
UPDATE's reach-distance table - a soft multi-joint reachability-envelope
boundary rather than a hard single-joint wall.

**Next diagnostic, not run this session, flagged for a future pass (per
this task's own instruction to use judgment once the assigned hypothesis
was answered one way or the other): directly sweep the reachable Z-height
envelope at this XY position** - e.g. via `--grasp-height` in fine
increments (rather than jumping straight to the true 9mm target), through
the SAME descent method, to map exactly how low this basin can genuinely
descend before the position residual starts growing, and cross-reference
against each joint's own live margin at that specific height to identify
the actual binding constraint (or confirm it's a genuinely multi-joint,
no-single-culprit envelope boundary, as the reach-distance table in the
prior UPDATE found at the 32cm/farther position). A second candidate,
also not run: since Y-axis residual (not just Z) was non-negligible in
run 2 specifically (15.5mm), it may be worth checking whether
`ORIENTATION_SCORE_WEIGHT`'s combined-score tradeoff is itself
contributing to which failure mode (rotation-dominant vs
position-dominant) a given run lands in, rather than treating tilt/step-
count as the only relevant variables.

**Script changes** (`scripts/grasp_demo_v2.py`): new `--num-descent-steps`
CLI arg (default 30); `polish_from_seed` gained optional `max_steps`/
`stagnation_break_steps` parameters (defaulting to the existing
`POLISH_MAX_STEPS`/`STAGNATION_BREAK_STEPS` module constants) so a
descent sub-step can use a smaller budget than a one-shot solve; new
`DESCENT_SUBSTEP_MAX_STEPS`/`DESCENT_SUBSTEP_STAGNATION_STEPS` constants;
`main()`'s GRASP-solving section now branches on
`args_cli.num_descent_steps <= 1` (old one-shot path, preserved for
comparison) vs. the new incremental-descent loop.

**Sources**: entirely this session's own live runs against the real
Isaac Sim scene on the desktop (non-headless, `DISPLAY=:1`, dispatched via
`scripts/run_on_desktop_gpu.sh` under the `/tmp/rl_isaac_sim.lock` flock) -
4 full `grasp_demo_v2.py` launches (`--video-suffix descent_v1`/
`descent_v2`/`descent_tilt15`/`descent_32cm`), their full logs, the two
recorded videos per run (`perception_camera`+`demo_camera`), and 3
extracted/cropped video frames from run 1's demo-camera video
(`ffmpeg` frame selection + crop, viewed directly) for the video-based
grasp-contact confirmation.

## UPDATE 2026-07-23 (later, ar4-grasp-position-search task): reach-distance sweep RULES OUT repositioning as a fix, cube-parking implemented, and a NEW, serious, unfixed gripper-jaw-asymmetry bug found by direct live measurement — session stopped mid-work at coordinator's request, capstone grasp+lift still NOT achieved

Tasked with finding a cube position within AR4's genuinely comfortable
reach envelope (healthy `joint_3` margin, not just non-zero) and
validating a real classical-IK grasp+lift there, per the immediately-prior
session's own flagged next step. Three separate findings, in the order
they happened; the last one is the most consequential and is the reason
this session stops without a validated grasp+lift.

**1. Reach-distance (radius) sweep: the ~18-19mm Z-shortfall is essentially FLAT from 0.30m to 0.42m reach at bearing=0 — reach distance does NOT resolve it, extending the prior session's bearing-independence finding to reach-independence too.** New `--radius-sweep` CLI flag added to `scripts/grasp_demo_v2.py` (mirrors `--bearing-sweep`'s own structure — full seed-search + PREGRASP polish + incremental descent per point — but varies reach radius at a fixed bearing=0 instead of bearing at a fixed radius). Result, 5 radii tested at the true 9mm grasp height:

| Radius (m) | pos_err (m) | joint_3 margin | joint_4 margin | Notes |
|---|---|---|---|---|
| 0.30 | 0.01838 | 0.2847 | 1.8843 | no joint near its limit at all |
| 0.33 | 0.01831 | 0.4181 | **0.0000** | joint_4 now pinned instead |
| 0.36 | 0.01838 | 0.5569 | **-0.0000** | joint_4 pinned |
| 0.39 | 0.01817 | 0.7091 | 3.1415 | different basin — joint_4≈0, far from any limit |
| 0.42 | 0.01844 | 0.8836 | 3.1415 | same comfortable basin as 0.39 |

`joint_3`'s own margin becomes genuinely healthy at farther reach (0.28-0.88, well above the ~0.08 baseline at the default 27.5cm) exactly as this task's brief hoped — **but the ~18mm Z-shortfall does not shrink at all**, and at 0.33-0.36m a *different* joint (`joint_4`) becomes the new binding constraint instead (margin pinned to ~0). Most importantly, at 0.30m and 0.39-0.42m **no single joint is anywhere near its limit, yet the shortfall is still ~18mm** — this directly confirms (with a much wider sweep than any prior session ran) the "soft multi-joint reachability-envelope/Jacobian-conditioning effect" characterization is the right one: this is not a single-joint-limit artifact that repositioning can dodge, it is a structural property of a fully-vertical top-down approach at this height, stable across at least a 12cm reach range and two qualitatively different joint-configuration basins (one with `joint_4` twisted near ±π, one with `joint_4`≈0). Combined with the prior session's bearing-independence (±60°) and tilt-instability findings, **this closes off "reposition within the reachable workspace" as a viable fix for a literally-vertical grasp** — there does not appear to be a genuinely comfortable envelope for this specific approach-orientation constraint, at least not within the ~0.30-0.42m/±60° region tested so far (untested: reach <0.30m other than the already-known-worse 0.20m point, reach >0.42m, and tilt combined specifically with these NEW comfortable-`joint_3`-margin basins — flagged as the most promising untested combination for whoever resumes this).

**2. Cube-parking implemented (coordinator-directed), replacing capture-then-restore with park-then-place.** Per a live user observation that the seed-search/polish process could interpenetrate-and-shove the cube before the real grasp attempt even starts, `scripts/grasp_demo_v2.py`'s single-position pipeline now teleports the cube to `_CUBE_PARK_POS_W = (5.0, 5.0, -5.0)` (far outside the whole reachable workspace) immediately after capturing `cube_init_pos`/`cube_init_quat`, keeps it there for the ENTIRE seed-search/PREGRASP-polish/incremental-descent/orientation-check process, and only moves it to its real `cube_init_pos` right before Phase 0 of the real phased execution. This is strictly better than the old "capture true pose, restore it after" approach (which could still leave residual velocity from a genuine depenetration event a pure position-restore wouldn't clear) since no interpenetration can occur at all. **This code change has NOT yet been exercised in a live run** (no phased-execution run happened this session after it was added) — verify it works as intended (cube genuinely undisturbed, arrives at the correct final position) the next time a real grasp attempt is run.

**3. Gripper open/close joint-position logging added (coordinator-directed) — also NOT yet exercised live.** Per a live user observation that the gripper looked closed throughout a run despite being commanded open, `_print_gripper_state()` was added, printing `robot.data.joint_pos` for `gripper_jaw1_joint`/`gripper_jaw2_joint` (the ACTUAL physical joint state, not the commanded action tensor) at the start/midpoint/end of every phase in the phased execution, plus a saved demo-camera/perception-camera snapshot PNG at each phase's midpoint for direct visual cross-check. Like item 2, this has not yet been run against a real phased execution this session.

**4. MAJOR FINDING, CONFIRMED BUT NOT YET FIXED: the gripper's "OPEN" command does not actually separate the two jaws — it commands them to the IDENTICAL world position.** A second, separate live user observation ("the two jaws don't look mirrored about a shared center") prompted a direct investigation, in three steps:

- **Static USD inspection** (`scripts/_inspect_jaw_symmetry.py`, requires bootstrapping a headless `SimulationApp` before `pxr` is importable at all under `isaaclab.sh -p` — confirmed live, a bare `from pxr import Usd` with no `SimulationApp` first raises `ModuleNotFoundError`): both `gripper_jaw1_joint` and `gripper_jaw2_joint` share the IDENTICAL `localPos0` origin `(0, -0.036, 0)` in their common parent (`gripper_base_link`) frame, and both jaw links' REST-POSE (joint value 0) world translations are identical to ~1e-7m — ruling out a baked-in asymmetric ORIGIN offset between the two joints. `gripper_jaw2_joint`'s `localRot0` is a genuine ~180° rotation relative to `gripper_jaw1_joint`'s (confirmed non-identity, unlike an initial misread of the raw printed tuple), consistent with the already-documented "180° jaw2 frame flip" from the 2026-07-21 asset debugging doc.
- **Hand-derived axis math from the static USD data** (`scripts/_inspect_jaw_axis_math.py`) predicted jaw1's effective travel axis in the parent frame is `(-1,0,0)` and jaw2's is `(+1,0,0)` — but this prediction did NOT match the live simulation's own numbers when cross-checked directly (a rest-pose offset sign flipped between the static check and a live run at the identical reset pose), most likely a coordinate-convention handling mistake in the static script (e.g. stage up-axis handling), not a real physical discrepancy. **This static/analytical result should be treated as unreliable and is superseded by the live measurement below.**
- **Direct empirical sweep, fully unambiguous** (`scripts/_sweep_jaw2_symmetry.py`): held `gripper_jaw1_joint` fixed at its own "open" target (`+0.014`), widened `gripper_jaw2_joint`'s SIM joint limits at runtime (via `write_joint_position_limit_to_sim`, no USD edit) to `[-0.03, 0.03]` so its commanded target could be swept past its currently-authored `[-0.014, 0]` range, and read back both jaws' ACTUAL world positions at 9 values from `-0.014` to `+0.014`. Result: **jaw2's world-frame X position is (to 5 decimal places) exactly `-1 * (jaw2's own commanded joint value)`**, while jaw1 (fixed at `+0.014`) sits at world-X `+0.014`. This means:
  - At the CURRENTLY-USED "open" command (`jaw1=+0.014, jaw2=-0.014`, i.e. `GRIPPER_OPEN_COMMAND_EXPR`'s existing `gearing=-1` convention): jaw2 lands at world-X `+0.014` — **the exact same point as jaw1** (measured separation: `0.00001m`, i.e. zero). The gripper does not open into a pincer shape at all; both "fingers" collapse onto one point.
  - At `jaw2=+0.014` (the SAME signed value as jaw1, requiring widened limits since this is outside jaw2's current authored range): jaw2 lands at world-X `-0.014` — **the true mirror image of jaw1**, with a clean `0.02800m` (28mm) separation, exactly the intended full-open aperture.
  - The sweep is a clean, monotonic straight line through all 9 points (`q2=-0.014→+0.014` maps linearly to `world_x=+0.014→-0.014`), so this is not noise or a one-off artifact.
  
  **Conclusion: the correct/intended `GRIPPER_OPEN_COMMAND_EXPR` value for `gripper_jaw2_joint` is `+GRIPPER_OPEN_POS` (the SAME signed value as jaw1), not `-GRIPPER_OPEN_POS` as currently authored — the opposite of the `gearing=-1` convention this project adopted on 2026-07-21.** That earlier convention was based on reading the URDF-authored `PhysxMimicJointAPI`'s own `gearing` attribute before it was stripped out (a real, correctly-read value) — but this session's direct empirical measurement of the ACTUAL built asset's geometry shows that value does not produce a physically mirrored gripper once combined with the specific 180°-rotated joint frame this importer/build actually produced. Whatever the historical reason for the mismatch, the live measurement here is unambiguous and internally self-consistent (monotonic sweep, exact expected mirror at one end, exact expected coincidence at the other, matching the user's own visual observation that prompted this check).

**This bug's severity: it is very plausibly a same-day, independent, and additive reason no AR4 grasp has ever succeeded, on top of the separate Z-height reach-limit finding.** A gripper that collapses both jaws onto the same point when "opened" cannot bracket an object regardless of how precisely it's positioned — this affects every AR4 script/task that imports `GRIPPER_OPEN_COMMAND_EXPR`/`GRIPPER_CLOSED_COMMAND_EXPR` from the shared `tasks/ar4/robot_cfg.py` (i.e. essentially all AR4 RL environments and classical demos), not just this task's own standalone demo.

**NOT YET DONE (session stopped here at coordinator's request to wrap up):**
- The actual fix (editing `GRIPPER_OPEN_COMMAND_EXPR` in `tasks/ar4/robot_cfg.py` to command `gripper_jaw2_joint` to `+GRIPPER_OPEN_POS`, AND correcting `gripper_jaw2_joint`'s own hard `physics:lowerLimit`/`upperLimit` in the built USD via `scripts/build_asset.py` from `[-0.014, 0]` to `[0, +0.014]` so the corrected command value is actually reachable, not silently clamped) has NOT been implemented — only root-caused and empirically confirmed.
- No grasp+lift attempt was run this session at all (the reach-distance sweep and the three gripper diagnostics consumed the whole session) — the capstone validation this task was dispatched to produce (a real, repeatable classical-IK grasp+lift, video-confirmed, across 3-4 positions) is **still outstanding**.
- The cube-parking and gripper-joint-logging code additions to `scripts/grasp_demo_v2.py` are unexercised (written but never run against a live phased-execution pick sequence).

**Recommended next steps for whoever resumes this** (in priority order): (1) fix the jaw2 open-command asymmetry bug (both the `robot_cfg.py` constant and the USD hard limits) and re-verify live via `scripts/_sweep_jaw2_symmetry.py`'s same method — this is likely the single highest-leverage fix outstanding for AR4 grasping generally, independent of any position/height work; (2) re-run this task's own reach/tilt combination check (a moderate tilt, e.g. 10-15°, AT one of the newly-found comfortable-`joint_3`-margin positions like 0.39-0.42m — untested combination, distinct from the prior session's tilt tests which were only run at the joint_3-tight-margin 27.5cm/32cm positions); (3) only then attempt the actual phased grasp+lift validation this task was dispatched to produce, with the cube-parking and gripper-logging instrumentation already in place to catch regressions.

**Script changes** (`scripts/grasp_demo_v2.py`): new `--radius-sweep`/reused `--bearing-sweep-radius`-style radius argument; cube-parking (`_CUBE_PARK_POS_W`) replacing the old capture-then-restore logic; `_print_gripper_state()` plus per-phase midpoint snapshot images (`ar4_grasp_gripper_check<suffix>/phase<N>_mid_{demo,perception}.png`). New one-off diagnostic scripts (not part of the normal script set, prefixed `_` per this repo's existing convention for throwaway diagnostics): `scripts/_inspect_jaw_symmetry.py`, `scripts/_inspect_jaw_axis_math.py`, `scripts/_inspect_jaw_symmetry_live.py`, `scripts/_sweep_jaw2_symmetry.py`.

**Sources**: this session's own live runs on the desktop (non-headless where rendering was needed, headless for the two pure-USD-inspection scripts) — one `--radius-sweep` launch (`logs/radius_sweep_v1.log`), one `_inspect_jaw_symmetry.py` static-USD launch, one `_inspect_jaw_symmetry_live.py` live-dynamics launch (`logs/videos/ar4_jaw_symmetry_check_demo_camera.mp4`, reviewed via extracted/cropped frames — inconclusive at this camera's resolution/distance, superseded by the direct numeric telemetry which is unambiguous), one `_inspect_jaw_axis_math.py` static-math launch (found unreliable, see above), one `_sweep_jaw2_symmetry.py` empirical sweep (`logs/jaw2_sweep.log`, the authoritative source for finding 4). Two Isaac Sim processes this session were found hung in post-work shutdown teardown (the documented "known gap" pattern — GPU/CPU still showing activity, log stalled with no progress for several minutes after the run's own final output was already fully written) and killed via `kill -TERM`/`-KILL` after confirming their real output was already captured; desktop confirmed fully torn down at session end (no stray Isaac Sim/kit processes, `nvidia-smi --query-compute-apps` empty, flock lock free, no tmux sessions).

## UPDATE 2026-07-23 (later, record-jaw-bug-video task): jaw2 open-command asymmetry bug FIXED and video-confirmed — real 28mm pincer open/close now working

Dispatched to record a video of the jaw-collapse bug from the immediately-
prior UPDATE (above); scope was widened mid-task by the coordinator to
implement the actual fix first, since the root cause and correct value
were already fully diagnosed and only needed applying.

**Fix applied, two files:**
- `tasks/ar4/robot_cfg.py`: `GRIPPER_OPEN_COMMAND_EXPR`/`GRIPPER_CLOSED_COMMAND_EXPR`
  now command `gripper_jaw2_joint` to the SAME signed value as
  `gripper_jaw1_joint` (previously negated) — per the prior UPDATE's own
  empirical sweep finding that jaw2's local-to-world mapping already
  contains a -1 factor from its 180°-rotated joint frame, so negating the
  command a second time was cancelling out the intended mirror.
- `scripts/build_asset.py`'s `_remove_gripper_jaw2_mimic_constraint`: the
  gearing value used to derive jaw2's hard `physics:lowerLimit`/
  `upperLimit` from jaw1's own limits changed from -1.0 (the URDF-authored
  mimic's own gearing, read off the mimic API before stripping it) to a
  hardcoded +1.0, for the same reason — the URDF-authored gearing
  describes the raw kinematic joint relationship, not the corrected
  command convention. Asset rebuilt via the full URDF→USD pipeline
  (`AR4_DESCRIPTION_PATH` + `PYTHONPATH=/home/saps/_ament_index_shim` env,
  the shim needed to resolve xacro's `$(find ...)` substitution without a
  full ROS install — not on `PYTHONPATH` by default in a plain SSH
  session, unlike an interactive shell that sources it) — jaw2's hard
  limits went from `[-0.0028, 0.0168]` (pre-existing, already-known-wrong
  from the 2026-07-21 doc) to `[0.0000, 0.0140]`, matching jaw1's own
  `[0.0000, 0.0140]` directly.

**Live confirmation, via a new script `scripts/_record_jaw_fix_open_close_cycle.py`**
(same direct `set_joint_position_target` mechanism as
`_sweep_jaw2_symmetry.py`, driving the two jaws through the actual
production `GRIPPER_OPEN_COMMAND_EXPR`/`GRIPPER_CLOSED_COMMAND_EXPR`
constants unmodified, not a widened/swept range) — an OPEN→CLOSE→OPEN
cycle (3s held per phase) with both jaws' world-frame body positions
printed at the end of each phase:

| Phase | separation_dist |
|---|---|
| reset (initial, spawns at OPEN per init_state) | 0.02800m |
| end of Phase 1 (OPEN) | 0.02800m |
| end of Phase 2 (CLOSE) | 0.00000m |
| end of Phase 3 (OPEN again) | 0.02800m |

This is a clean, repeatable 0mm/28mm cycle matching the full intended
open aperture — a large change from the pre-fix measurement in the prior
UPDATE (OPEN command produced 0.00001m separation, i.e. both jaws
collapsed onto the same point). **The jaw2 open-command asymmetry bug is
now fixed and directly video-confirmed, not just root-caused.**

Video recorded with a tight close-up camera (repurposed
`Ar4GraspVerifyEnvCfg.demo_camera`, repositioned/re-zoomed via
`create_rotation_matrix_from_view`/`quat_from_matrix` rather than a new
camera system) framed directly on the gripper jaws:
`logs/videos/ar4_gripper_jaw_open_close_cycle_fixed.mp4` (desktop path;
synced to the Pi at the matching `logs/videos/` path for the controller
to view directly).

**Not yet done / explicitly out of scope for this task:** no grasp+lift
attempt was run with the fix in place — this task was bounded to fixing
and video-confirming the jaw open/close dynamics in isolation, not the
broader grasp-discoverability investigation. The Z-height reachability
shortfall documented earlier in this file is a separate, still-unresolved
issue. Whoever resumes the grasp+lift validation should now do so with a
gripper that actually opens into a real pincer shape, removing one of the
two known-independent confounds this file has been tracking.

**Sources**: this session's own live runs on the desktop (non-headless,
`DISPLAY=:1`, under `/tmp/rl_isaac_sim.lock`) — one full `build_asset.py`
rebuild (log confirms `[mimic-removal]` printed the corrected
`[0.0000, 0.0140]` limits) and one `_record_jaw_fix_open_close_cycle.py`
launch (video + the four separation measurements above, read directly
from its own stdout log). Isaac Sim was killed by something external
mid-teardown once this session (the run's own `[DONE]`-equivalent final
lines — "Video recorded to: ..." — were already written to disk before
the kill, and the video file was confirmed present/valid, `file` reporting
a genuine ISO Media MP4 container, before being synced to the Pi);
desktop confirmed fully torn down afterward (no stray Isaac Sim/kit
processes, `nvidia-smi --query-compute-apps` empty, flock lock free, no
tmux sessions).

## Standing FK verification framework added (2026-07-23) — direct response to this whole file's own pattern of "found by an ad hoc script or by the user eyeballing the sim"

Every defect this article documents (missing gripper physics drive, 4
classical-IK positioning bugs, a wrist-orientation bug, the jaw-mirroring
bug in the section directly above) was found by a one-off diagnostic
script written fresh each time, or by the user directly watching the
simulation and noticing something looked wrong. Tasked with building a
standing, reusable, general-purpose verification framework using forward
kinematics (FK) to catch this whole CLASS of bug automatically as real
test scripts, not agent instructions or another one-off diagnostic.

**Two layers, both implemented in `tasks/ar4/fk_verification.py` (pure
numpy, no isaaclab/torch import — runs on plain `python3`, unlike most of
this project's other torch-based reward-math tests):**

- **Layer 1 (asset-geometry check)**: an independent FK chain
  (`compute_link_pose_from_joint_values`, `assert_link_pose_matches_vendor_fk`)
  built directly from the vendor's raw URDF/xacro source
  (`urdf/ar_macro.xacro`, `urdf/ar_gripper_macro.xacro`, `config/mk5.yaml`
  — read via `ssh desktop` on 2026-07-23, hardcoded with provenance
  comments since that path isn't reachable from the Pi), independently
  re-derived rather than reused from `scripts/build_asset.py`'s own
  import pipeline — the whole point is to catch bugs baked into that
  pipeline, not reproduce them. `pytorch_kinematics` was checked and
  confirmed not installed anywhere in this project's environments; a
  hand-rolled ~10-joint serial-chain FK was simple enough not to need it.
- **Layer 2 (control-intent/task-invariant check)**: `assert_gripper_separation`
  uses Layer 1's FK to check that a COMMANDED joint_values dict produces
  the intended real-world jaw separation, not just "did each joint
  individually reach its own target" — the exact class of check that
  would have caught the jaw-mirroring bug directly above.

**Concrete proof this framework catches the real bug class it was built
for** (`tests/test_ar4_fk_verification.py::TestJawMirroringRegression`,
9/9 tests passing, run both on the Pi's plain `python3 -m pytest` and via
this project's standard desktop convention,
`/home/saps/IsaacLab/_isaac_sim/python.sh -m pytest ... -p no:launch_testing`):
the CURRENT, live-verified-correct SAME-sign convention
(`gripper_jaw1_joint`/`gripper_jaw2_joint` both commanded to `+0.014`,
`tasks/ar4/robot_cfg.py`'s post-2026-07-23 `GRIPPER_OPEN_COMMAND_EXPR`)
predicts `28.000mm` and PASSES `assert_gripper_separation`'s `[20mm,
36mm]` check — matching `tasks/ar4/objects_cfg.py`'s own documented
"~28mm max aperture"; the now-superseded 2026-07-21 OPPOSITE-sign fix
(jaw2 negated) predicts an exact `0.000mm` separation and FAILS. A third
test class deliberately corrupts one arm joint's origin by 50mm in a copy
of the joint table (`with_corrupted_origin`) and confirms Layer 1 catches
that import-style asset-geometry defect too, independent of the
gripper-specific question. Test-suite rigor was verified beyond "tests
pass" twice: mutating the jaw2 axis sign was confirmed to flip exactly
the 3 jaw-mirroring-dependent tests from PASS to FAIL both times (once
during initial TDD, once again after the recalibration below), proving
the tests actually discriminate rather than passing vacuously.

**The framework's own first-draft calibration turned out to be wrong,
and a live integration run is what caught it — arguably the single best
demonstration this whole effort could have produced of why Layer 1
(grounded in the raw vendor source) matters more than calibrating against
"already-empirically-confirmed" institutional history.** A *literal*,
by-the-book application of the raw URDF's own `<origin>`+`<axis>`
semantics to `gripper_jaw2_joint` (rotate-then-translate, matching the raw
URDF's own `<mimic multiplier="1"/>` tag) predicts SAME-sign commanding is
correct. The framework's first draft did not trust that literal reading —
it special-cased `gripper_jaw2_joint` to match this article's 2026-07-21
finding (OPPOSITE-sign commanding, `928af41`), which was the
best-available evidence at the time. A live integration run the same day
(below) directly measured the CURRENT asset producing correct ~28mm
separation from SAME-sign commanding, exactly matching the plain literal
URDF reading and contradicting the framework's own first-draft
calibration. Cross-checking `tasks/ar4/robot_cfg.py`'s own current source
found why: the concurrent gripper-fix task had, that same day, *also*
found the 2026-07-21 opposite-sign fix itself wrong (`scripts/_sweep_jaw2_symmetry.py`,
commit `d59595a` — a direct sweep found jaw2's own local-to-world mapping
already contains the sign flip the 2026-07-21 fix was redundantly
re-applying, "double-negating" it back to a collapse) and reverted to
same-sign, independently arriving at the same literal-URDF answer. The
special-casing (`translate_axis_in_parent_frame`) was removed from
`fk_verification.py` entirely once this was confirmed — the plain literal
joint table, with no jaw2-specific correction, is what ships now.

**Live integration check — cloud, not desktop (plan changed mid-task).**
The original plan was a desktop `flock`-guarded run against the
concurrently-active gripper-fix task's own already-built, already-fixed
asset (read-only-copied from its worktree, `~/projects/rl-ar4-fixes-transfer`,
into a temporary swap of the `~/projects/rl` checkout's own stale
`assets/ar4_mk5/`, restored afterward — the checkout's own asset was
confirmed stale first: its jaw2 hard limits were still the pre-fix
`[-0.003, 0.017]` range, so Isaac Lab's own articulation validator
rejected the scene before this framework's checks even ran). That attempt
was abandoned mid-run on direct controller instruction once the user
signaled the desktop would be shut down once its own concurrent job
finished — moved to a fresh GCP cloud instance instead (`$1` cost cap for
this lightweight check), per `docs/cloud/dispatch-checklist.md`. This
required standing up a *new* AR4-on-cloud capability this project didn't
previously have (only Franka had a proven cloud recipe): the vendor URDF
package (`annin_ar4_description`) has a public GitHub mirror
(`https://github.com/Annin-Robotics/ar4_ros_driver`, confirmed to contain
byte-identical `urdf/ar_macro.xacro`/`urdf/ar_gripper_macro.xacro`/`config/mk5.yaml`
content to the desktop's private-fork checkout via `git diff`), so no
desktop-resident file needed shipping. Two real, previously-undocumented
gaps were found and fixed in the same pass (per this repo's own
bug-handling discipline): pip's `xacro==2.1.1` needs `ament_index_python`
to resolve the URDF's `$(find annin_ar4_description)` substitution, and
`ament-index-python` is not published to PyPI (ROS 2 packages generally
aren't) — fixed with a small, from-scratch reimplementation of its single
needed function (`get_package_share_directory`, a simple, well-documented
resource-index-marker-file lookup) plus a hand-built minimal
`AMENT_PREFIX_PATH` tree, avoiding a full ROS 2/colcon install. The build
also hit `isaacsim.asset.importer.urdf`'s interactive EULA prompt a second
time (separate from the one already accepted during the pip install
step), silently hanging the tmux session on stdin until noticed and
answered — flagged here since it's a real, likely-recurring gap in the
"first AR4 cloud build" recipe, not yet automated away (`OMNI_KIT_ACCEPT_EULA=YES`
apparently doesn't cover every EULA prompt Isaac Sim's own tools can
raise). The instance also hit **two genuine SPOT preemptions** in quick
succession (confirmed via `gcloud compute operations list`, both real
`compute.instances.preempted` events, ~21-60min apart) — resolved per
`docs/cloud/dispatch-checklist.md`'s own documented judgment call
(switching to on-demand is reasonable when the remaining job count is
small and cost allows) via `gcloud compute instances set-scheduling
--no-preemptible --provisioning-model=STANDARD` on the stopped instance,
which preserved the already-built asset/venv on the boot disk with no
rework needed.

**Real result, from a live `env.reset()` inside
`Ar4PickPlaceGraspGoalEnvCfg` on this freshly cloud-built current asset**
(`scripts/_verify_gripper_fk_integration.py`, headless per cloud
convention): commanded/read-back joint state at "open" was
`gripper_jaw1_joint=+0.013996`, `gripper_jaw2_joint=+0.014000` (SAME
sign) with a REAL measured world-frame jaw separation of **27.996mm**.
`fk_verification.py`'s Layer 1 check on `link_1`, `link_6`, and
`gripper_jaw1_link` all PASSED to `0.000mm` discrepancy against this same
live state; `gripper_jaw2_link` FAILED at `28.000mm` discrepancy under
the framework's then-still-uncorrected first-draft calibration — the
exact signal that prompted the recalibration described above. After
removing the special-casing, the plain literal model predicts this same
live jaw2 pose to `0.000mm` and Layer 2's `assert_gripper_separation`
predicts `28.0mm`, matching the real `27.996mm` measurement to within
0.004mm. Full cloud teardown confirmed immediately after
(`scripts/check_cloud_state.sh` clean: zero instances/disks/snapshots).
Approximate cost: ~$0.79 against the $1 cap (two SPOT preemptions each
forced a restart, and the final on-demand segment ran at ~2x rate — no
BigQuery billing export exists for this project, so this is a
duration-times-published-SKU-rate estimate, per this project's standing
practice).

**Where to run it**: `tests/test_ar4_fk_verification.py` (pure numpy —
runs on the Pi directly, no desktop/GPU dependency, unlike most of this
project's other reward-math tests) and, for a live-sim integration check
against a real built asset, `scripts/_verify_gripper_fk_integration.py`
(Isaac-Sim-touching; run non-headless with the desktop `flock` pattern
locally, or `--headless` on a cloud instance per the run above). Pointer
added to `START_HERE.md`'s "Verification standard" section so future AR4
work uses this instead of another one-off script.

## UPDATE 2026-07-23 (ar4-capstone-grasp task): the best kinematic configuration this investigation has ever found (9-10mm residual, under the cube's own size) — but still no working grasp+lift, and a real, honest cost-cap overrun

Dispatched as the explicit capstone of this whole day's AR4 investigation:
every individual blocker (gripper mimic-vs-actuator conflict, jaw2 missing
drive, jaw2 open-command sign, classical-IK Jacobian-frame/grid-search/
EE-offset/cube-position bugs, arm actuator gains) had been found and fixed
by prior sessions except actually running a real grasp+lift end to end. No
RL training involved (classical/scripted IK only, per standing instruction);
desktop unreachable, cloud-only.

**Setup: a full from-scratch AR4 asset rebuild on GCP, directly USD-verified
correct (not just trusted from exit code).** No AR4 asset artifact existed in
GCS or any committed location, so this session redid the 2026-07-23
FK-integration session's own "AR4-on-cloud" recipe from scratch: vendor
`annin_ar4_description` cloned from its public GitHub mirror, a hand-built
`ament_index_python.packages.get_package_share_directory` shim (xacro's
`$(find annin_ar4_description)` resolution has no ROS install to rely on)
plus a symlinked `AMENT_PREFIX_PATH` tree, then `scripts/build_asset.py` via
`isaaclab.sh -p`. **A real, previously-undocumented gotcha found this
session**: `build_asset.py`'s own `print()` confirmations (`[collision-fix]`,
etc.) never appeared anywhere in the captured log despite the build
succeeding (exit code 0, files written) - `SimulationApp.close()` appears to
force-exit the process in a way that skips Python's normal stdout-buffer
flush, so trusting "the log looks quiet, exit 0" would have been trusting an
unverified claim. Caught by writing a small, fast, GPU-free direct-USD
inspection script (`~/verify_asset.py` pattern, not committed - ad hoc for
this session, mirrors the 2026-07-21 asset-debugging sessions' own
methodology) that opens the built `ar4_mk5.usd` via bare `pxr`/`SimulationApp`
and directly checks: `gripper_jaw2_joint` has no `PhysxMimicJointAPI` (PASS),
has its own `DriveAPI:linear` (PASS), hard limits `[0.0, 0.014]` matching
jaw1 exactly (PASS), and `link_5`/`link_6` both have a
`substitute_collision_box` child prim with `CollisionAPI` (PASS/PASS). All 8
checks passed - the asset genuinely carries every fix this investigation has
found, not just "the build didn't crash."

**Tooling added to `scripts/grasp_demo_v2.py`** (all three landed together,
commit `a26a9ea`): `--tilt-sweep` (sweeps multiple tilt angles at a FIXED
cube position in one launch, mirroring the existing `--z-sweep`/
`--bearing-sweep`/`--radius-sweep` structure - the untested "tilt AT a
comfortable-joint-margin position" combination the prior session flagged);
real cube-parking (`_CUBE_PARK_POS_W`, teleporting the cube far outside the
workspace for the whole seed-search/polish/descent duration, un-parking only
right before Phase 0) - **found, while implementing this, that the prior
session's own commit message claimed cube-parking was "implemented in
grasp_demo_v2.py" but the actual committed diff (`4df9de4`) never included
it** - only the gripper-logging half of that claim was real; this is a
concrete instance of exactly the kind of claimed-vs-actual discrepancy this
project's own verification discipline exists to catch, caught by diffing the
commit against its own message rather than trusting the kb doc's prior
narration; and per-phase jaw contact-force logging (`jaw1_cube_force`/
`jaw2_cube_force`, `force_matrix_w` filtered against the Cube prim only) added
alongside the existing cube z/xy printout, directly answering this project's
own Experiment 16 precedent (a video that looked like a lift but was the
object wedged) without needing to trust video review alone.

**Tilt sweep at the reach distance (0.39m) already known to have healthy
`joint_3` margin (2026-07-23, ar4-grasp-position-search task's own flagged
next step): 0-18° reproduces the same flat/negative signature found
everywhere else in this investigation, but 25-90° reveals a genuine, new
local minimum around 65°.**

| Tilt (deg) | pos_err (m) | Z-shortfall (m) | joint_3 margin |
|---|---|---|---|
| 0 | 0.01944 | -0.01988 | 0.672 |
| 5 | 0.02007 | -0.01986 | 0.683 |
| 8 | 0.02078 | -0.02024 | 0.693 |
| 10 | 0.02189 | -0.02047 | 0.706 |
| 12 | 0.02181 | -0.02051 | 0.710 |
| 15 | 0.02186 | -0.02058 | 0.711 |
| 18 | 0.02176 | -0.02054 | 0.712 |
| 25 | 0.02100 | -0.01992 | 0.692 |
| 35 | 0.01888 | -0.01797 | 0.727 |
| 45 | 0.01602 | -0.01524 | 0.738 |
| 60 | 0.01088 | -0.01045 | 0.764 |
| 65 | 0.00937 | -0.00928 | 0.787 |
| 70 | 0.01302 | -0.01356 | 0.804 |
| 75 | 0.01655 | -0.01704 | 0.804 |
| 90 | 0.02086 | -0.02078 | 0.810 |

Two independent runs at tilt=0/5/8 reproduced identically (19.4/20.1/20.8mm)
across a SPOT-preemption-forced restart, confirming this isn't run-to-run
noise. **65° tilt gives a 9.37mm position residual - the first time this
entire multi-week investigation's own residual has dropped below the cube's
own 12mm size**, with `joint_3` margin genuinely healthy (0.79rad, vs. the
~0.08rad baseline at the original 27.5cm/vertical position) and no other
joint anywhere near its limit either. A follow-up reach sweep AT this fixed
65° tilt (0.30-0.45m) found the improvement holds flat across 0.30-0.36m
(9.3-9.5mm, healthy margins throughout) before degrading again past 0.39m -
a genuine, reproducible plateau, not a single lucky point.

**Three full phased grasp+lift attempts at this configuration (reach
0.30m/0.36m/0.39m, all 65° tilt, real recorded video + per-phase jaw
contact-force logging) - all three show the IDENTICAL negative signature,
a genuine repeatable null, not a false positive:**

| Position | Final grasp_residual | Cube z (all phases 2-6) | jaw1_cube_force (CLOSE) | jaw2_cube_force (CLOSE) | Cube xy shift |
|---|---|---|---|---|---|
| reach=0.39m | 9.35mm | flat 0.0060m | 0.23-0.23N (brief) | 0.0000N | ~13mm |
| reach=0.30m | 9.53mm | flat 0.0059-0.0060m | 0.34-0.34N (brief) | 0.0000N | ~1.3mm |
| reach=0.36m | 9.38mm | flat 0.0060m | 0.037-0.038N (brief) | 0.0000N | ~1mm |

In every run: PREGRASP/GRASP converge cleanly (sub-1cm residual, matching
the summary numbers above), the gripper's OPEN/CLOSE joint positions track
correctly (`[0.014,0.014]` open, `[~0.0,~0.0]` closed - the jaw2 fix holds up
under this new geometry too), and the phased sequence runs to completion
with no crash. But `cube.data.root_pos_w`'s z-component is EXACTLY flat
through CLOSE/lift/hold in all three runs - no ambiguity requiring frame-by-
frame video review the way Experiment 16 needed, since the ground-truth
number itself never moves. jaw1 registers a brief (steps 20-40 of Phase 3
only), light, one-sided contact force; jaw2 registers exactly `0.0000N`
throughout every single logged step of every run. The cube gets nudged
sideways by a few mm to ~1.3cm (largest at 0.39m, smallest at 0.36m) rather
than enclosed. Videos: `logs/videos/ar4_grasp_demo_v2_pos1_r039_t65_demo_camera.mp4`,
`..._pos2_r030_t65_demo_camera.mp4`, `..._pos3_r036_t65_demo_camera.mp4` (and
matching `perception_camera` videos + per-phase gripper-check snapshot PNGs),
all synced to the Pi at the matching `logs/videos/` path.

**What this means.** The 65° tilt configuration is a real, substantial,
reproducible improvement over every other position/bearing/reach/tilt
combination this entire investigation has tested (9.3-9.5mm vs. the
~18-24mm floor found everywhere else, including at the same reach distances
under 0-18° tilt) - this closes off "is there ANY reachable configuration
with a sub-cube-size position residual" with a genuine yes, which no prior
session had found. But sub-cube-size position residual alone is NOT
sufficient for a real grasp: the consistent jaw1-only-contact signature
across all three positions suggests the remaining ~9mm gap is now
manifesting as an ANTIPODAL-ALIGNMENT problem (one jaw reaches the cube
before the other, at this specific large-tilt approach geometry) rather than
a clean total miss - plausibly the same class of bug as Bug 3 from the
2026-07-22 ar4-grasp-ik-precision task (`_EE_OFFSET`'s single fixed linear
offset representing the gripper's "pinch point"), which may not correctly
represent the true bisector point between the two jaws once the whole
gripper is oriented at a large, non-near-vertical tilt rather than the
near-vertical geometry that offset was originally measured/validated for.
**Concrete next diagnostic, not done this session**: directly measure both
jaw fingertips' real world positions (mirroring `scripts/_sweep_jaw2_symmetry.py`'s
own direct-measurement methodology) at the converged 65°-tilt `grasp_q`
configuration, to check whether the cube is actually centered between them
or offset toward one side - this would either confirm/refute the
"`_EE_OFFSET` doesn't generalize to large tilts" hypothesis directly, rather
than continuing to guess at more tilt/position combinations.

**Verdict: this is a genuine, well-evidenced NEGATIVE result for the
capstone grasp+lift attempt, not a false positive being smoothed over** -
real height-gain numbers were checked directly (not inferred from video) and
show exactly zero gain in all three attempts, cross-checked against
contact-force data per this project's own Experiment 16 standard. This does
NOT close out the long-running AR4 grasp-discoverability investigation the
way a successful capstone would have - the specific new finding (a much
better, sub-cube-size kinematic configuration exists, but still doesn't
produce a real pinch) is itself a substantial narrowing of the problem,
worth treating as the concrete next step rather than a dead end.

**Cost cap overrun, reported honestly.** The task's cap was $2; actual spend
was approximately **$3.3** (instance-uptime × published-SKU-rate estimate, no
BigQuery billing export exists for this project, per standing practice).
Breakdown: SPOT phase 1 (29m, ended in a genuine `compute.instances.preempted`
event) ≈ $0.18; SPOT phase 2 (59m, second genuine preemption within the same
hour) ≈ $0.36; on-demand phase (3h40m, switched to on-demand per this
project's own documented judgment call once two preemptions in under an hour
made continuing on SPOT a real wall-clock drag, following the exact recipe
`docs/cloud/dispatch-checklist.md` already documents for this situation) ≈
$2.65 at roughly on-demand's ~2x SPOT rate; plus ≈$0.10 boot-disk cost across
the full ~5h uptime. The overrun's root cause: two genuine SPOT preemptions
forced the more expensive on-demand fallback, and the exploratory tilt/
position search needed to find the 65° optimum (19 total sweep points across
4 separate sweep launches, plus 3 full phased-execution attempts, plus the
asset rebuild itself) took longer than the cap anticipated. Flagged plainly
here rather than smoothed over; full teardown confirmed
(`scripts/check_cloud_state.sh`: zero instances/disks/snapshots).

**Sources**: this session's own live cloud runs (`~/tiltsweep039.log` /
`~/tiltsweep039b.log` - first attempt was itself cut short by the first SPOT
preemption mid-sweep, re-run identically after switching to on-demand;
`~/tiltsweep039big.log`, `~/tiltsweep039fine.log`, `~/radiussweep65.log`,
`~/grasp1.log`/`~/grasp2.log`/`~/grasp3.log`), `~/verify_asset.log` (direct
USD-level asset verification), `gcloud compute operations list` for the
preemption/restart timeline used in the cost breakdown above.
