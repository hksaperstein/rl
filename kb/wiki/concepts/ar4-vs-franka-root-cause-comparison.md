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

2. **Jaw-mimic constraint — confirmed, never enforced, root cause of the
   *symptom* found; root cause of the underlying *physical* asymmetry
   still open.** Three architecturally distinct fixes were tried across
   Experiments 17-22 and all failed or were reverted: the native URDF
   `<mimic>` tag (confirmed unenforced by Isaac Sim's USD import despite
   `parse_mimic=True`), a PhysX-level `PhysxMimicJointAPI` constraint
   (Experiment 19 — both tested configs made jaw divergence measurably
   *worse* than the uncoupled baseline, reverted), and a software
   leader-follower action term (Experiment 22 — introduced a new
   "reactive lag" failure mode, then a corrected zero-lag version was found
   to be a structural no-op and retired before ever training, per
   `tasks/ar4/pickplace_graspgoal_env_cfg.py:100-117`'s own docstring).
   **The genuinely revealing finding**: AR4's default RL gripper action
   (`BinaryJointPositionActionCfg`, identical commanded target sent to
   both jaw joints) is *structurally identical* to Franka's own validated
   gripper action (`tasks/franka/lift_env_cfg.py:173-177`,
   `open_command_expr={"panda_finger_.*": 0.04}`) — neither platform uses
   a real mimic joint at the RL-action level. So the RL action-space
   design was never the point of difference; AR4's asymmetric jaw
   behavior happens "at the physics/actuator level under contact," per
   `pickplace_graspgoal_env_cfg.py:108-109`'s own diagnosis — a real,
   asset-specific defect distinct from Franka's asset, but the *specific*
   physical cause (contact-geometry asymmetry between the two jaw links?
   friction mismatch? something else?) was never isolated in three
   attempts. This is the strongest single piece of evidence supporting the
   pivot's "AR4 asset defect, not RL/reward-design difficulty" framing —
   but the defect itself remains only located, not explained.
3. **Jaw collision geometry ("unverified convex-hull approximation") —
   still inconclusive, on both platforms.** The claim traces to exactly
   one place in this repo's history (the 2026-07-11 joint-space-lift
   research doc, itself labeled "unverified"), and was never confirmed by
   directly inspecting the built AR4 USD asset's `MeshCollisionAPI`
   approximation attribute — `scripts/build_asset.py` sets no explicit
   collision-approximation for the AR4 gripper import at all (only the
   unrelated wedge die shape gets an explicit `convexHull`). The same fact
   is equally undocumented for Isaac Lab's own shipped Franka asset, and
   not inspectable from the Pi (`/home/saps/IsaacLab` unreachable from this
   machine). Tellingly, this project's own d4 grasp work on Franka doesn't
   trust the stock finger collision mesh either — it built a
   purpose-authored, geometry-known notch fixture
   (`tasks/franka/notch_fixture.py`) rather than reading contact force off
   the stock finger mesh directly. This item was carried into the pivot
   rationale as if established; it was not, on either platform.

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

## Open follow-up

Resolving Hypothesis 3 (and the still-open physical cause beneath
Hypothesis 2) needs a live Isaac Sim session on both platforms: direct USD
`MeshCollisionAPI` inspection of the built gripper/finger collision shapes,
plus a live instrumented grasp comparing measured contact-force directions
against true geometric surface normals. Not attempted in this pass
(read-only, no GPU used) — a well-defined next step if this project ever
returns to the AR4 asset, or wants to close the loop on whether Franka's
own finger collision mesh has the same latent risk.

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

## Sources

`docs/superpowers/specs/research/2026-07-20-ar4-vs-franka-root-cause-comparison.md`
(full citations), `CLAUDE.md` ("Platform pivot" section).
