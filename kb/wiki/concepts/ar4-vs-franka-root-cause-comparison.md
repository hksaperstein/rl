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
