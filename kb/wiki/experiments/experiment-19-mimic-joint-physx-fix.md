# Experiment 19: Mimic-joint PhysX fix

**Object:** cube (AR4 era). A falsification attempt on the confirmed
jaw-coupling mechanical defect from [[experiment-17-antipodal-grasp-gate]]
Task 6 — two independently-tested fix configurations both made
synchronization measurably worse than the baseline, a clean diagnostic
that the attempted fix mechanism is actively counterproductive in this
Isaac Sim version, not merely ineffective.

## Hypothesis

Experiment 17 Task 6's own confirmed root cause identified jaw2 drifting
20% past its commanded position under contact load while jaw1 tracked
exactly. The source URDF specifies a `<mimic joint="gripper_jaw1_joint"
multiplier="1" offset="0"/>` constraint on `gripper_jaw2_joint` (confirmed
present, confirmed not enforced by the built USD). Authoring a real
PhysX-level mechanical coupling via `PhysxMimicJointAPI` — connecting the
two jaw joints as a genuinely physically-coupled unit, rather than relying
on the URDF importer's pass-through of an unenforced constraint — should
close enough of the gap to make antipodal grasps physically reachable. This
follows directly from [[sim-physics-fidelity]]'s premise: a faithful asset
should enable the same grasps the real robot is capable of.

## What changed

Two separate fix iterations, each tested in isolation via instrumented
rollout (`scripts/mimic_joint_verify.py`), reusing Experiment 18's trained
checkpoint against the rebuilt asset:

**Fix iteration 1:** Zeroed `gripper_jaw2`'s independent PD drive
(`stiffness=0.0, damping=0.0`), reasoning that the mimic constraint alone
should determine jaw2's position, matching PhysX's own reference
implementation pattern of driving only the reference joint.

**Fix iteration 2:** Restored `gripper_jaw2`'s actuator to match
`gripper_jaw1` exactly (`stiffness=1000.0, damping=50.0`), following the
only confirmed community report of a working mimic-jointed gripper (UR10e
+ Robotiq 2F-85), which keeps all joints independently actuated at full
stiffness.

Technical grounding (read directly from source before implementation): the
installed Isaac Sim 107.3.26 PhysX schema confirms `PhysxMimicJointAPI`
explicitly supports `PhysicsPrismaticJoint` (the gripper jaws' actual joint
type); `build_asset.py` already passed `parse_mimic=True` to the URDF
importer (confirming the intent was present but not working); PhysX's own
official mimic-joint test suite (`omni.physx.tests/.../PhysxMimicJointAPI.py`)
was read directly and used as reference for the exact Python API calls
(`PhysxMimicJointAPI.Apply`, `GetReferenceJointRel().AddTarget`,
`GetGearingAttr().Set(1.0)`, `GetOffsetAttr().Set(0.0)`).

## Quantitative result

Instrumented jaw-tracking diagnostics showed both fix iterations made
synchronization worse than the pre-fix baseline:

**Fix iteration 1:** `max_jaw_pos_diff_during_contact = 0.00548m` — 3.9x
over the 0.0014m pass threshold, and worse than the pre-fix baseline of
0.0028m from Experiment 17 Task 6.

**Fix iteration 2:** `max_jaw_pos_diff_during_contact = 0.00647m` — 18%
worse than fix iteration 1, and more than double the pre-fix baseline.
This is the single most decisive measurement: iteration 2's configuration
is identical to the pre-fix baseline in every respect except the mimic
joint's presence (both jaws independently driven at identical stiffness/
damping in both cases) — a clean, isolated A/B comparison confirming the
mimic constraint itself is actively interfering under real contact load,
not merely failing to help.

## Qualitative diagnostic finding

Mid-investigation research (Google search + direct GitHub API fetches,
cross-verified for accuracy per [[citation-verification-practice]]):
an Isaac Lab maintainer confirms "we currently don't have an example for
this type of joint" — a genuine, admitted maturity gap. However, direct
re-fetching of cited sources revealed one paraphrased claim ("unmerged
`feature/unactuated-joints` branch") could not be reproduced in the actual
comment thread — flagging this as a caught instance of fabricated technical
detail and reinforcing the importance of direct source verification. The
single confirmed working setup (UR10e + Robotiq) keeps every joint,
including mimic-coupled ones, independently actuated at full stiffness,
contradicting fix iteration 1's design premise that the mimic constraint
alone should drive jaw2. Fix iteration 2's failure to improve, despite
following this community-validated pattern, indicates the problem lies
deeper than controller configuration — consistent with PhysX's documented
behavior that a mimic joint applies corrective impulses to both the
reference and mimicking joints, plausibly increasing net system compliance
under load when combined with two already-independently-driven joints.

## Verdict

**Clean falsification — both independently-tested configurations made
synchronization measurably worse, not better. The attempted fix mechanism
is not viable in this Isaac Sim version, not a mere tuning problem.** Two
different configurations both worsened the defect (by 3.9x and 4.6x
respectively) — a pattern systematic-debugging discipline treats as
diagnostic of an architectural mismatch, not a parameter problem, even
absent a formal 3-strikes threshold. The experiment's hard gate correctly
stopped before advancing to training re-runs. Assets and scripts reverted
to pre-Experiment-19 state (commit `255b9b2`), restoring the known-good
baseline.

This experiment does not resolve whether the underlying mechanical defect
(two jaws not coupled, contrary to real robot design) blocks genuine
antipodal grasps — it specifically shows that fixing it via
`PhysxMimicJointAPI` is not a working path forward with this tool/version
combination. The defect remains real and unresolved, but this candidate is
now closed out as tried-and-failed, not deferred. The independently-raised
next direction — constraining the gripper to a fixed vertical/top-down
approach orientation during reach, following classical grasp-planning
assumptions (Dex-Net, GPD) — is not blocked by this negative result and
represents a different lever on [[reach-grasp-lift-gap]]: reduce the
geometric burden of finding an antipodal grasp in the first place, rather
than trying to fix jaw-coupling fidelity.

## Related concepts

[[reach-grasp-lift-gap]] — the core unsolved problem this fix was
attempting to remove as a barrier. [[sim-physics-fidelity]] — this
experiment is fundamentally an asset/physics-fidelity fix attempt using
PhysX's own APIs, directly testing that concept's dependency chain.
[[citation-verification-practice]] — this experiment caught a fabricated
detail in a cited source, reinforcing the importance of direct source
re-fetching for sensitive technical claims.

## Sources

`docs/superpowers/specs/2026-07-07-ar4-experiment19-mimic-joint-fix-design.md`,
`docs/superpowers/plans/2026-07-07-ar4-experiment19-mimic-joint-fix-implementation.md`
