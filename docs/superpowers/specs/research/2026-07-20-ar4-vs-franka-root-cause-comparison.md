# Research: root-causing the AR4's three unresolved grasp defects against the working Franka setup

**Date:** 2026-07-20
**Author:** Senior research thread (delegated by Principal)
**Purpose:** CLAUDE.md's "Platform pivot" section names three specific,
never-resolved AR4-specific hypotheses as of the 2026-07-09 pivot decision
(a 17-27mm classical-IK positioning miss, an unconfirmed jaw-mimic
constraint, an unverified convex-hull jaw-collision approximation). This
document root-causes each one directly against the repo's own history and
against the now-working Franka setup (`tasks/franka/`), and states plainly
whether the evidence supports the pivot's own working explanation ("AR4-
asset-specific defects, not a fundamental RL/reward-design difficulty") or
complicates it. Read-only investigation — no code changed.

**Scope note:** no Isaac Sim launch was performed for this pass (GPU is a
Pi-external resource, and this task was scoped as primarily non-GPU
analysis — see §4/§5's "still inconclusive" items for exactly what a live
run would need to add). All findings below come from reading this repo's
own source, specs, plans, `ROADMAP.md`, and `git log` directly.

---

## 1. Hypothesis 1 — the classical-IK "17-27mm" grasp miss

**Verdict: root cause found for the actual investigated failure — but it
is a control-algorithm limitation of the classical demo scripts, not a
URDF/asset-frame defect. The specific "17-27mm" figure does not exist
verbatim anywhere in this repo's history; it is a rounded synthesis of
several distinct measurements from different scripts/methods.**

### 1a. The number itself does not appear verbatim anywhere

`git log -p --all -S"17-27mm"` / `-S"17mm"` / `-S"27mm"` / `-S"1.7-2.7cm"`
returns only `CLAUDE.md` itself and
`docs/superpowers/specs/research/2026-07-11-joint-space-lift-research.md:290,357`,
which quotes/paraphrases CLAUDE.md and post-dates it. The phrase was
introduced by commit `52837e3` ("Adopt Franka Emika Panda as primary
platform," 2026-07-09 21:39), a 34-line commit touching only `CLAUDE.md` —
it does not cite a specific `ROADMAP.md` line for the number, and no other
file changed in that commit.

The closest primary-source numbers, all from the same day (2026-07-09),
pre-dating the 21:39 pivot commit:

| Source | Measurement |
|---|---|
| `ROADMAP.md:2408-2410` (commit `5a78395`, 12:33) | `interactive_joint_demo.py`'s closed-loop IK refinement "diverges (1.46cm → 1.74cm residual over 4 rounds)…leaving the jaws ~2.9cm from the cube when closing is attempted" |
| `ROADMAP.md:2328` | `ik_polish_from_grid.py`'s DLS polish "closed it to 3.648cm before plateauing again" |
| `scripts/interactive_joint_demo.py:84-88` (comment) | "The 25-step approach alone plateaus around 5-6cm residual…more than 3x the cube's 12mm size" |
| `ROADMAP.md:2339-2341` | `grasp_demo_v2.py`: "joint tracking improved to ~0.19-0.32 rad residual" (angular, not directly comparable in mm) |

These range 14.6mm to 60mm depending on script/method — not a tight 17-27mm
band. The 17.4mm and 29mm endpoints of the `interactive_joint_demo.py`
figures are the closest match to "17-27mm," but even that source's own
final number is 29mm, not 27mm. **No single measurement in this repo's
history states "17-27mm."** This is worth flagging on its own: the pivot
rationale's precision ("17-27mm") reads as more exact than the underlying
evidence actually is.

### 1b. What the miss's actual root cause was found to be

Four independent scripts/mechanisms hit the identical "converges partway,
then stalls" signature at/near the cube's pose (`ROADMAP.md:2301-2306`):
the original `classical_pickplace_demo.py` kinematic-singularity stall,
`oracle_rollout.py`'s reactive per-step pursuit, `grasp_demo.py`/
`grasp_demo_v2.py`'s once-per-waypoint solve, and
`interactive_joint_demo.py`'s closed-loop refinement. Root-cause
diagnostics independently ruled out, with direct measurement:

- **Actuator torque saturation**: peak 14.1 N·m vs. a 20 N·m limit, never
  clips (`ROADMAP.md:2262-2264`).
- **Hard joint-limit hit**: ~13° of margin (`ROADMAP.md:2264`).
- **Contact/collision interference**: 0.0N throughout (`ROADMAP.md:2264`).
- **Jacobian rank-collapse**: smallest singular value plateaus ~0.15, not
  ~0 (`ROADMAP.md:2265-2266`).
- **Genuine kinematic unreachability**: refuted directly by forward-
  kinematics measurement (`scripts/measure_reach_envelope.py`, no IK
  solver involved) — 0.538m max reach vs. 0.344m needed
  (`ROADMAP.md:2317-2319`).
- **A wrong EE-frame offset (URDF/asset bug)**: refuted directly.
  `_EE_OFFSET=(0.0, 0.0, 0.036)` on `link_6`
  (`tasks/ar4/pickplace_env_cfg.py:42`) was independently re-verified both
  numerically (`ee_frame.data.target_pos_w` vs. the real jaw-link midpoint,
  <0.001mm residual) and visually (`debug_vis=True`, marker sits between
  the jaw tips) — `ROADMAP.md:2373-2379`.

What the investigation actually converged on: a **single-Newton-step-per-
control-tick DLS differential-IK solver gets trapped in a fixed point of
the receding-horizon control loop in a poorly-conditioned kinematic
region**, where the linearized correction stops reliably pointing toward
the goal (`ROADMAP.md:2266-2268`). This independently echoes Experiment
20's own prior, separately-reached conclusion (a 6-value damping sweep,
`ROADMAP.md:2270-2272`) that single-step differential IK is not a stable
mechanism on this arm — reached via a structurally different investigation
path. The follow-up root-cause pass (`ROADMAP.md:2399-2418`) narrowed this
further: the specific failure reproduced live in `interactive_joint_demo.py`
is "a positioning/dynamics miss intrinsic to this one script's fixed-wrist
(`q4=q5=q6=0`), open-loop-hold approach, not a sim-wiring defect" — and was
explicitly assessed as "clear to proceed with RL training (which drives all
6 joints with continuous closed-loop correction, not bound to this demo's
simplifications)."

**This last point matters for the pivot's own framing**: the "IK miss" was
a property of the *standalone, non-RL, waypoint-jumping demo scripts*
(bounded single large Newton steps toward distant waypoints), not of the
*RL-driven, continuously-corrected* differential-IK action term used in
training. Experiment 11 (`ROADMAP.md:2461-2481` era, cross-referenced via
`docs/superpowers/specs/research/2026-07-11-joint-space-lift-research.md:275-281`)
used the same underlying IK mechanism inside RL and produced this project's
first-ever sustained nonzero antipodal grasp contact (91.6% of iterations
nonzero) — i.e. the mechanism the classical scripts struggled with did
work when driven incrementally, every control tick, by a learned policy
rather than solved for a large jump once per waypoint.

### 1c. Comparison against Franka's asset — a real structural difference found, though not shown to be load-bearing for this specific miss

`tasks/franka/lift_env_cfg.py:66-82` explicitly separates two offsets that
AR4's code conflates into one:

```python
# EE frame measurement point: panda_link0 -> panda_hand, offset 0.1034m ...
_EE_MEASUREMENT_OFFSET = (0.0, 0.0, 0.1034)
# IK control target offset: panda_hand -> the point the differential-IK
# controller drives to the commanded relative pose ... (0.107, not 0.1034
# - two different official reference points for two different roles:
# sensing vs. control - both are stock, not a typo)
_IK_BODY_OFFSET = (0.0, 0.0, 0.107)
```

AR4's code (`tasks/ar4/pickplace_env_cfg.py:42`, reused for both the
`FrameTransformer` sensing offset — `pickplace_env_cfg.py:57` — and the
`DifferentialInverseKinematicsActionCfg.body_offset` — e.g.
`classical_demo_env_cfg.py:94`, `pickplace_baseproximity_env_cfg.py:60`)
uses a single constant, `_EE_OFFSET = (0.0, 0.0, 0.036)`, for both roles.
Franka's own validated reference config treats "the point I measure
reach-distance from" and "the point the IK controller targets" as two
independent, separately-sourced values (differing by 3.6mm) — a
verification-rigor distinction this project's AR4 code never drew. **This
is a real, citable structural difference**, but it is not shown to be the
cause of the classical-IK miss specifically: AR4's single offset value was
independently measured correct to <0.001mm for its own (sensing) role
(§1b above), and the diagnosed failure mode (DLS local-minimum trap) is
unrelated to offset value. Flagging this as a genuine asset/config-rigor
gap worth correcting on principle, not as the resolved root cause of this
particular miss.

### 1d. Net read on Hypothesis 1

**Confirmed root cause found for the classical-script miss (DLS single-
step local-minimum trap in a poorly-conditioned kinematic region,
independently corroborated by two separate investigation paths) — but
this is a control-algorithm limitation, not the URDF/asset frame-offset
defect the pivot's phrasing ("classical closed-form-IK grasp attempt
misses the cube... unresolved root cause") implies.** The EE-offset/frame
question was checked and ruled out directly. Notably, Isaac Lab's own
Franka reference configs (`ik_abs_env_cfg.py`/`ik_rel_env_cfg.py`, per
`docs/superpowers/specs/research/2026-07-11-joint-space-lift-research.md:46-71`)
use the identical `ik_method="dls"` solver — and Isaac Lab's own authors
never ship a trained RL recipe for either IK variant, only for plain
joint-space (§1e of that research doc) — suggesting Isaac Lab's own team
independently avoids relying on single-step DLS IK as a *trained RL*
action space too, consistent with (not contradicting) what this project
found the hard way on AR4.

---

## 2. Hypothesis 2 — the gripper jaw-mimic constraint

**Verdict: confirmed — never correctly enforced, across three
architecturally distinct mechanisms, all failed or reverted. But the
underlying physical cause of the jaw asymmetry itself was never isolated —
this remains a genuine, still-open gap beneath a well-established negative
result.**

### 2a. Three independent fix attempts, three failures

1. **Native URDF `<mimic>` tag, relied on via USD import.** Source URDF
   (external, `annin_ar4_description/urdf/ar_gripper_macro.xacro`, not in
   this repo) carries `<mimic joint="gripper_jaw1_joint" multiplier="1"
   offset="0"/>` on `gripper_jaw2_joint`
   (`docs/superpowers/specs/2026-07-07-ar4-experiment19-mimic-joint-fix-design.md:35-36`,
   `ROADMAP.md:1679-1680`). `scripts/build_asset.py:316` sets
   `import_config.parse_mimic = True`. Despite that, `ROADMAP.md:1440-1444`
   (Experiment 16-era finding) states directly: *"`gripper_jaw1_joint`/
   `gripper_jaw2_joint` do not track each other despite the source URDF's
   explicit `mimic` joint constraint… Isaac Sim's USD import of this asset
   appears not to enforce that constraint."* Intent (`parse_mimic=True`)
   was present; enforcement was not.
2. **Experiment 19, PhysX-level `PhysxMimicJointAPI`.** Two configurations
   tried, both regressions against the 0.0028m pre-fix baseline
   (`ROADMAP.md:1706-1742`): config 1 (drop jaw2's independent drive) →
   0.00548m (3.9x over threshold); config 2 (keep jaw2's actuator alongside
   the constraint) → 0.00647m (18% worse than config 1). Reverted, commit
   `255b9b2` ("Revert Experiment 19's mimic-joint asset fix: falsified,
   made things worse"), removing all mimic-API code from
   `scripts/build_asset.py` and `tasks/ar4/robot_cfg.py`. **Current state
   confirmed**: `tasks/ar4/robot_cfg.py:68-75` still shows both gripper
   joints on a plain symmetric `ImplicitActuatorCfg` with no mimic API —
   the revert is the live state today.
3. **Experiment 22, software leader-follower (`MirroredGripperAction`,
   tracking jaw1's *settled* position).** Result: `both_magnitude_ok_steps`
   stayed 0/750, `lifting_object` stayed 0/1500 (`ROADMAP.md:2022-2032`).
   New failure mode found: jaw2 structurally lags a moving jaw1 target by
   one control step ("reactive lag"), evidenced by
   `max_jaw_pos_diff=0.011m` (79% of the full 0.014m travel range).
4. **A zero-lag correction (tracking jaw1's *commanded* target instead)
   exists in the current codebase** — `MirroredGripperAction` in
   `tasks/ar4/actions.py:160-206` — but was never actually trained.
   `tasks/ar4/pickplace_graspgoal_env_cfg.py:100-117`'s `ActionsCfg`
   docstring states directly why it was dropped before Experiment 26's
   run: *"MirroredGripperActionCfg…is an unconditional no-op here:
   BinaryJointAction already assigns both jaws the IDENTICAL commanded
   value…so there is nothing for jaw2 to diverge from at the command level
   in the first place — the actual jaw asymmetry happens at the physics/
   actuator level under contact, which no command-level mirroring can
   address."* Retired per this project's own "3 failed fixes means
   question the architecture, not patch again" discipline
   (`pickplace_graspgoal_env_cfg.py:114-115`).

### 2b. The ROADMAP's own internal contradiction on this point

`ROADMAP.md:2402-2405` (same-day follow-up, 2026-07-09) states *"a real
`PhysxMimicJointAPI` coupling exists on the currently-built asset"* — but
this directly contradicts `ROADMAP.md:2424-2425` (Experiment 25, also
2026-07-09): *"the gripper's two jaws are not mechanically coupled (the
source URDF's `mimic` constraint confirmed unenforced by Isaac Sim's USD
import)."* Independent verification favors the second claim: Experiment
19's mimic-API code was reverted on 2026-07-07 (commit `255b9b2`), two
days *before* the 2402-2405 claim; `scripts/plot_arm_skeleton.py` (the
script that claim cites as its evidence source) contains no reference to
`PhysxMimicJointAPI` or "mimic" anywhere in it; and the current
`tasks/ar4/robot_cfg.py`/`scripts/build_asset.py` contain zero mimic-API
code. **This reads as an unverified/erroneous claim in the ROADMAP text
itself, uncorrected by any later entry** — flagged here rather than
silently resolved one way.

### 2c. Structural comparison against Franka — the action-space mechanism was never actually the differentiator

Franka's validated gripper action (`tasks/franka/lift_env_cfg.py:173-177`,
`tasks/franka/demo_action_mapping.py:82-85`):
```python
gripper_action = mdp.BinaryJointPositionActionCfg(
    asset_name="robot", joint_names=["panda_finger.*"],
    open_command_expr={"panda_finger_.*": 0.04},
    close_command_expr={"panda_finger_.*": 0.0},
)
```
This commands **both** fingers to the identical target value via a
regex-expanded dict — independent per-joint actuation with a symmetric
action mapping, no PhysX mimic constraint at all. **AR4's own default
gripper action (used across most experiments, including 10/11/16) is
structurally identical**: `tasks/ar4/env_cfg.py:48-52` and
`tasks/ar4/pickplace_mirror_env_cfg.py:71-75` both use
`BinaryJointPositionActionCfg` with `open_command_expr={name:
GRIPPER_OPEN_POS for name in GRIPPER_JOINT_NAMES}` — the same constant
sent to both jaw joint names. **This means the RL action-space mechanism
itself was never the point of difference between AR4 and Franka** — both
platforms command identical target positions to both gripper joints, by
construction, with no mimic joint in the loop at the action level. The
`pickplace_graspgoal_env_cfg.py:104-110` docstring makes this explicit:
the asymmetry AR4 suffers happens "at the physics/actuator level under
contact," a level neither platform's action-space design addresses.

**What this reveals**: Franka's real advantage here isn't a smarter
action-space design (Isaac Lab's own reference platform uses the exact
same symmetric-commanded-target approach this project already used on
AR4) — it's that Franka's underlying asset (mass distribution, joint
friction, collision geometry, calibration) apparently does *not* produce
the same asymmetric-contact behavior AR4's asset did, closely enough that
Isaac Lab's authors never needed a mimic joint or any other coupling
mechanism to get a working two-finger grasp. **This is exactly a physical-
asset-quality difference, not an RL/reward-design or action-space
difference** — consistent with the pivot's working hypothesis, though the
*specific* physical defect responsible on the AR4 side (contact-geometry
asymmetry between the two jaw links? friction coefficient mismatch?
something else?) was never isolated in three attempts, and remains
genuinely open (see §3).

---

## 3. Hypothesis 3 — the jaw collision geometry / "unverified convex-hull approximation"

**Verdict: still inconclusive. The specific claim was never independently
confirmed by inspecting the built AR4 USD asset, and the same fact is
equally undocumented for Franka's own shipped asset — this question was
never actually resolved on either side.**

### 3a. Where the claim comes from

`git log -p --all -S"convex-hull approximation"` returns exactly two hits,
both descending from the same text: the commit adding
`docs/superpowers/specs/research/2026-07-11-joint-space-lift-research.md:300,322`
(*"The jaw collision geometry uses an unverified convex-hull approximation
that may distort contact-force directions read by the antipodal grasp
check"*), and the pivot commit `52837e3` copying the same wording into
`CLAUDE.md`. **The research doc's own text labels this "unverified"
(line 358)** — it was never independently confirmed by inspecting the
built asset's collision schema.

### 3b. What `scripts/build_asset.py` actually does

The only explicit `UsdPhysics.MeshCollisionAPI…CreateApproximationAttr
("convexHull")` call in this file is at line 116, inside
`_generate_wedge_usd` — for the hand-authored triangular-prism die shape,
**not** the AR4 gripper. The AR4 arm/gripper import (lines 300-330) sets
only `fix_base`, `merge_fixed_joints`, `parse_mimic`, `make_default_prim`
on `URDFCreateImportConfig` — no collision-approximation field is set for
the arm/gripper at all, meaning whatever Isaac Sim's URDF importer's
*default* collision-approximation setting is for mesh geometry, that's
what the AR4 jaws got. **This repo does not document what that default
actually is or what it produced**, and no commit in this repo's history
shows anyone opening the built `ar4_mk5.usd` and reading the jaw links'
actual `MeshCollisionAPI` approximation attribute.

### 3c. What is directly confirmed to depend on this

`tasks/ar4/mdp.py:902-941`, `antipodal_grasp_bonus`, reads
`jaw1_sensor.data.force_matrix_w` / `jaw2_sensor.data.force_matrix_w`,
computes each contact force's magnitude and the cosine angle between the
two jaws' force *directions*, and requires `cos_angle <
antipodal_cos_threshold` (line 939) for a grasp to register — i.e. this
reward/verification signal is directly and functionally dependent on
contact-force *direction*, which PhysX derives from the collision shape's
local surface normal at the contact point. If the jaw collision
approximation coarsens or distorts the true surface shape, the force
directions this function reads would be corrupted independent of whether a
real, geometrically-correct grasp occurred. The function's own docstring
cites classical force-closure literature (Nguyen 1988; Ponce & Faverjon
1991/1993) but makes no mention of any collision-geometry caveat — this
dependency was never flagged or checked at the point the reward function
was written.

### 3d. Franka's side is equally undocumented, and this project's own d4 work implicitly treats the risk as real and unresolved even for Franka

No doc in this repo states what collision approximation Isaac Lab's
shipped `FRANKA_PANDA_CFG`/`FRANKA_PANDA_HIGH_PD_CFG` fingertip meshes use.
`/home/saps/IsaacLab` (where the shipped USD lives) is not reachable from
this machine, so this cannot be checked without dispatching to a GPU
machine. `tasks/franka/dice_scene_cfg.py:41-43`'s own docstring states the
stock Franka fingertip collision geometry "is never touched" by this
project's own code — used as-is, unexamined. Tellingly, when this project
needed a *geometrically-known*, trustworthy contact-force read for the d4
grasp task, it did not rely on the stock Franka finger collision mesh at
all — it built a **separate, purpose-authored notch fixture** with an
explicit, simple `convexHull` collision shape
(`tasks/franka/notch_fixture.py:263-265`) and reads contact force off
*that* fixture's own known geometry instead
(`tasks/franka/dice_scene_cfg.py:373-408`). This is a real signal: this
project implicitly does not trust an uninspected stock finger-collision
mesh for force-direction correctness even on the validated Franka
platform — it engineers around the question rather than answering it, on
both platforms.

### 3e. Net read on Hypothesis 3

**Genuinely inconclusive, on both sides of the comparison.** What's
missing to resolve it: (1) direct USD/stage inspection of the built
`ar4_mk5.usd`'s gripper jaw links —
`UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()` — to
determine the actual approximation type Isaac Sim's importer produced by
default; (2) the same inspection against Isaac Lab's shipped Franka asset;
(3) ideally, a live instrumented grasp attempt on each platform comparing
measured `force_matrix_w` contact-normal directions against the true
geometric surface normal at the actual contact point, to determine whether
distortion is real and how large. None of this was ever done in this
repo's history for either platform, and it requires a live Isaac Sim
session (GPU) rather than pure code reading — out of scope for this
read-only pass, and not pursued here given the desktop is currently
off-limits and this was scoped as primarily non-GPU analysis. Flagging
this as a genuine, well-defined open follow-up rather than guessing at an
answer.

---

## 4. Does this support or complicate the pivot's own working hypothesis?

The pivot's working hypothesis (CLAUDE.md, "Platform pivot" section):
*"AR4-asset-specific defects…rather than a fundamental RL/reward-design
difficulty."*

**Partially supported, but with real complications worth recording
plainly rather than smoothing over:**

- **Supports it:** Hypothesis 2's core finding — three architecturally
  different coupling mechanisms (URDF-native mimic, PhysX-level mimic API,
  software command-mirroring) all failed to fix AR4's asymmetric jaw
  contact, while Franka's *identical* action-space mechanism (symmetric
  independently-commanded targets, no mimic joint anywhere) works without
  any such fix needed — this is a real, load-bearing platform/asset
  difference, not an action-space or reward-design difference. The
  Franka pivot's own results (perception-driven dice picking, first
  learned d20 lift+carry, per `kb/wiki/index.md:24-27`) are direct
  positive evidence in the same direction.
- **Complicates it, three ways:**
  1. Hypothesis 1's actual root cause (§1) is a **control-algorithm
     limitation** (single-step DLS local-minimum trap in the *classical,
     non-RL* demo scripts specifically), not a URDF/asset defect as the
     pivot's phrasing implies — and it was already understood well enough
     by 2026-07-09 to be assessed as "clear to proceed with RL," which
     did proceed (Experiments 25/26) without re-litigating this specific
     item. The pivot rationale's "17-27mm, unresolved root cause" framing
     overstates both the precision of the number and how unresolved the
     mechanism actually was.
  2. Hypothesis 3 (§3) was never actually confirmed on the AR4 side, and
     the same fact is equally unverified for Franka's own asset — this
     item was carried into the pivot rationale as established evidence
     when it was, and remains, an open question on both platforms, not a
     confirmed AR4-specific defect.
  3. Even Experiment 26 — the AR4 project's actual final data point before
     the pivot — does **not** cleanly attribute its `cube_reached_goal
     0.0000` result to the three named defects. Its own recorded verdict
     (`ROADMAP.md:2607-2628`, `kb/wiki/experiments/experiment-26-gripper-reintroduction.md:121-150`)
     is "not yet root-caused to a specific fix," with a *reward-design*
     mechanism (the running-max staged-potential reach term providing no
     incentive to hold position once a good reach is banked) named as an
     equally plausible contributor alongside "the antipodal grasp gate is
     apparently never satisfied." This means this project's own last AR4
     result is genuinely confounded between an asset-defect explanation
     and a reward-design explanation — it was never cleanly resolved
     in favor of either before the pivot decision was made.

**Overall:** the pivot decision itself remains well-supported by its
strongest evidence (Hypothesis 2, and Franka's subsequent real results),
but the specific three-item rationale recorded in CLAUDE.md overstates how
resolved/precise two of the three items (1 and 3) actually were at
decision time, and the project's own final pre-pivot experiment was never
cleanly attributed to asset defects rather than reward design. This is a
"the pivot was the right call for good reasons, but the stated reasons
were somewhat oversold" finding, not a "the pivot was wrong" finding — a
distinction worth recording precisely rather than either rubber-stamping
or overturning the original framing.

---

## 5. Summary table

| Hypothesis | Verdict | Key evidence | What's still missing |
|---|---|---|---|
| 1. 17-27mm classical-IK grasp miss | Root cause found, but re-characterized: DLS single-step local-minimum trap in classical demo scripts, not a URDF/frame defect. The literal "17-27mm" figure is unsourced/rounded. | `ROADMAP.md:2261-2272,2317-2319,2373-2379,2399-2418`; `git log -S"17-27mm"` (no hits) | Nothing outstanding — mechanism independently corroborated twice (Exp 20 damping sweep, Exp 24 Gate 1) |
| 2. Jaw-mimic constraint | Confirmed never correctly enforced (3/3 independent fixes failed). Action-space mechanism itself is identical to Franka's — the AR4-specific defect is physical/contact-level, not RL-design-level. | `ROADMAP.md:1440-1444,1706-1742,2022-2032`; `tasks/ar4/pickplace_graspgoal_env_cfg.py:100-117`; commit `255b9b2` | The specific physical cause of AR4's contact-level jaw asymmetry (vs. Franka's asset, which needs no fix) was never isolated |
| 3. Convex-hull jaw collision geometry | Still inconclusive on both platforms — claim was never independently verified for AR4, and the same fact is undocumented for Franka's own shipped asset. | `scripts/build_asset.py:116,300-330` (no explicit setting for AR4 gripper); `tasks/franka/notch_fixture.py:263-265`/`dice_scene_cfg.py:41-43` (Franka's own code avoids relying on the stock mesh too) | Direct USD `MeshCollisionAPI` inspection on both platforms' built assets, plus a live contact-force-vs-surface-normal measurement — requires a GPU/Isaac Sim session, not attempted here |

---

## Sources

`CLAUDE.md` ("Platform pivot" section), `ROADMAP.md:1440-1446,1670-2039,
2246-2628`, `kb/wiki/index.md`,
`kb/wiki/experiments/experiment-26-gripper-reintroduction.md`,
`kb/wiki/concepts/reach-grasp-lift-gap.md`,
`docs/superpowers/specs/research/2026-07-11-joint-space-lift-research.md`,
`docs/superpowers/specs/2026-07-07-ar4-experiment19-mimic-joint-fix-design.md`,
`docs/superpowers/specs/2026-07-07-ar4-experiment22-software-jaw-mirroring-design.md`,
`tasks/ar4/actions.py`, `tasks/ar4/robot_cfg.py`,
`tasks/ar4/pickplace_env_cfg.py`, `tasks/ar4/pickplace_graspgoal_env_cfg.py`,
`tasks/ar4/mdp.py:902-941`, `tasks/franka/lift_env_cfg.py`,
`tasks/franka/demo_action_mapping.py`, `tasks/franka/dice_scene_cfg.py`,
`tasks/franka/notch_fixture.py`, `tasks/franka/antipodal_edge_grasp.py`,
`scripts/build_asset.py`, `scripts/interactive_joint_demo.py`,
`scripts/measure_reach_envelope.py`; commits `52837e3`, `5a78395`,
`255b9b2`.
