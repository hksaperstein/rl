# Research: d8 grasp-mechanics literature grounding — contact/antipodal
verification vs. physical-parameter anomaly

**Date:** 2026-07-20
**Author:** Senior research thread (delegated by Principal)
**Purpose:** Tier 1 hypothesis-gate research following up
[[exploration-bonus-grasp-discovery]]'s SPLIT verdict — the policy reliably
attempts gripper closure at the right place/moment near d8 (seed 123:
`frac_steps_raw_action_negative_near_object = 1.0` in 7/8 envs), but closing
there never produces a lift (0/24 sustained-lift across all 3 seeds). That
result rules out pure exploration-discoverability as the remaining
bottleneck and points at a downstream grasp-mechanics problem. This document
researches the two candidate directions `ROADMAP.md`'s "Bottom line and
forward pointer" named but did not choose between: (1) contact/antipodal
grasp-quality verification at the moment of closure, and (2) d8's own
physical parameters (mass/friction/scale) at 48mm-parity. **This is research
only — no env cfg code, no reward-term design, no Isaac Sim launches.**

---

## 1. Direction 1: contact/antipodal grasp verification

### 1a. What the current Franka reward actually checks (confirmed by direct source read)

`tasks/franka/lift_env_cfg.py:280-304` (`RewardsCfg`), used unmodified by
every d8/d10/d12/d20 env cfg in this arc including the exploration-bonus
variant (`ExplorationBonusRewardsCfg` only *adds* two new terms, per its own
docstring, `lift_env_cfg.py:328-338` — it does not modify or replace any of
`RewardsCfg`'s own terms): the reward is `reaching_object` (dense EE-to-object
distance) + `lifting_object` (binary: object height above 0.04m or not) +
`object_goal_tracking`/`_fine_grained` (gated on the same 0.04m lift flag) +
small action-rate/joint-vel penalties. **`RewardsCfg`'s own docstring states
this explicitly: "Stock reward shape: dense reach + binary lift (NOT
antipodal/contact-force-gated — a deliberate, known difference from this
project's own AR4-era grasp-verification gate...)."** There is no contact
force, contact-normal, or grasp-quality term anywhere in this reward, for any
shape, in this Franka arc — confirmed by direct source read, not taken from
a comment alone. `lifting_object` only reads the object's height; it cannot
distinguish "lifted because of a real antipodal grasp" from any other way
the object's height might transiently rise.

This means the SPLIT result (reliable closure attempts, no lift) is
consistent with a specific, mechanistically plausible explanation: the
reward has no term that discriminates a geometrically-sound (antipodal/
force-closure) hand pose from a degenerate one at the moment of closure —
only "did height go up afterward," which provides no gradient signal at the
moment that actually matters. A policy under this reward has no training
pressure toward antipodal alignment specifically; it only gets rewarded
*after* a lift already (accidentally or not) occurred.

### 1b. This project's own prior, directly relevant precedent (AR4 era)

`kb/wiki/concepts/grasp-mechanics-antipodal-vs-magnitude.md` and the two
experiments it documents are the strongest evidence available for this
exact mechanism, on a closely related task/platform (AR4, not yet Franka,
but the same physical question: parallel-jaw closure near an object):

- **[[experiment-01-contact-sensor-grasp-reward]]**: a `ContactSensorCfg`-
  based bilateral-force-magnitude reward converged to ~92% sustained contact
  — real progress on "does the gripper close on the object" — but this is a
  **magnitude-only** check with no directional requirement.
- **[[experiment-09-antipodal-grasp-bonus]]**: replacing the magnitude-only
  check with a real antipodal check (jaw contact-force directions within
  ~30° of anti-parallel, dot product < −0.85) made the check fire **~1800x
  less often** than the magnitude-only check — direct, quantitative evidence
  that most of what the magnitude-only reward was crediting as "grasped" was
  hard bilateral contact from non-opposing directions, not real force
  closure.
- **[[experiment-10-antipodal-threshold-action-scale-solver]]**: correcting
  the threshold to the scene's own physically-derived value (`mu=1.0` →
  45° friction-cone half-angle → `-0.7071`, not the earlier `-0.85` guess)
  made the antipodal signal **regress to exactly 0.000000** — worse, not
  better. Verdict: "arguing the bottleneck is *precision* of final gripper
  positioning/alignment under direct joint-space control, not reward-
  threshold calibration." This directly motivated abandoning joint-space
  action refinement.
- **[[experiment-11-taskspace-ik]]**: switching the *action space* (not the
  reward) from direct joint-position control to `DifferentialInverseKinematicsActionCfg`
  task-space control produced the **first genuine, sustained antipodal grasp
  contact this project has ever seen** (`antipodal_grasp_bonus` final
  0.018815, nonzero in 91.6% of iterations, vs. exactly 0 under Experiment
  10's joint-space control) — confirming the antipodal check itself is
  learnable once the action space allows precise-enough positioning. Lift
  still did not emerge from that checkpoint (a separate, later-stage gap),
  but the grasp-quality signal itself became real for the first time.

**A load-bearing caveat this history makes unavoidable to flag:** Franka's
die-lift env (`tasks/franka/dice_lift_joint_env_cfg.py:110-113,159-160`)
uses **joint-space** `JointPositionActionCfg(scale=0.5)` — direct joint-angle
control, the same action-space class Experiment 10 diagnosed as the actual
bottleneck once its own antipodal threshold was already physically correct,
on the same shared platform (a Franka Panda arm/hand). If a Franka
antipodal/contact reward is added under the current joint-space action
config and reproduces Experiment 10's own "correctly-thresholded signal
regresses to exactly zero" outcome, that would not be a fresh, unexplained
result — it would be a second confirmation of an already-diagnosed
mechanism, and the tested-and-proven fix already exists in this project's
own history (task-space IK). Notably, this project already has a *working,
verified* task-space controller for exactly d8/d10 at 48mm-parity-adjacent
scale: `scripts/dice_pick_demo.py`'s DiffIK grasp controller achieves a real
grasp-and-lift on d8 (240.9mm variant) and d10 (239.3mm variant) per
[[dice-pick-demo]] and the d8/d10-grasp-discoverability research (§3a,
`docs/superpowers/specs/research/2026-07-19-d8-d10-grasp-discoverability-literature.md`)
— so a task-space fallback, if joint-space turns out to reproduce the
AR4-era precision bottleneck, is not speculative infrastructure but an
already-proven mechanism this project would only need to wire into the RL
action space (as Experiment 11 did for AR4), not invent from scratch.

### 1c. Isaac Lab infrastructure — re-verified against the currently pinned version, not taken on faith

The prior citation (`isaaclab/sensors/contact_sensor/`,
`isaaclab/envs/mdp/rewards.py:281`) was re-checked live against
**v2.3.1** — the exact tag this project's own cloud recipe pins
(`docs/cloud/franka-cloud-shakedown.md:244`: `git clone
https://github.com/isaac-sim/IsaacLab.git --branch v2.3.1 IsaacLab`) —
by fetching the real source directly from GitHub (`raw.githubusercontent.com`,
`api.github.com/repos/isaac-sim/IsaacLab/contents/...?ref=v2.3.1`), not from
memory or the old AR4-era citation. The desktop machine that has this
version physically installed (`/home/saps/IsaacLab`) was off-limits for this
task (unreachable/reserved for another workstream), so this is the closest
available live verification. Confirmed, at the exact pinned tag:

- `source/isaaclab/isaaclab/envs/mdp/rewards.py:281` is still `def
  contact_forces(env, threshold, sensor_cfg)` — the exact line number the
  old citation named, unchanged.
- `source/isaaclab/isaaclab/sensors/contact_sensor/` still contains
  `contact_sensor.py`, `contact_sensor_cfg.py`, `contact_sensor_data.py`.
  `ContactSensorCfg` still has `filter_prim_paths_expr` (per-body-pair
  filtering). `ContactSensorData` still has both `net_forces_w` ("the sum of
  the normal contact forces acting on the sensor bodies," no filtering
  language in its own docstring) and `force_matrix_w` ("filtered between the
  sensor bodies and filtered bodies," `None` if `filter_prim_paths_expr` is
  empty) as **separate fields** — directly confirming this project's own
  AR4-era finding (`net_forces_w` is not actually filtered by
  `filter_prim_paths_expr`; `force_matrix_w` is the field that is) still
  holds at the exact version this project runs today, not just at whatever
  version was installed when that finding was first made.

**Verdict: the infrastructure claim is accurate and current, freshly
re-verified, not stale.** A `ContactSensorCfg`-based, `force_matrix_w`-driven
contact-direction reward is real, available, and has never been built for
any Franka die-lift env cfg in this arc (confirmed by the `RewardsCfg`
docstring in §1a and by grep — no `ContactSensorCfg`/`force_matrix_w`
reference anywhere in `tasks/franka/`).

### 1d. External literature — force-closure/antipodal criteria and RL-specific evidence

Every citation below was checked live this session via the arXiv API
(`export.arxiv.org/api/query`, note: `http://` redirects to `https://` —
use `https://` directly, `http://` with a short timeout and no `-L` returns
an empty body) or the CrossRef API (`api.crossref.org/works`), not taken
from memory, per [[citation-verification-practice]].

**Classical force-closure/antipodal foundations** (re-verified via CrossRef
this session, all real, matching this project's own prior citation of them
in [[experiment-09-antipodal-grasp-bonus]]):
- Nguyen, V-D., "Constructing Force-Closure Grasps," *International Journal
  of Robotics Research* 7(3), 1988. DOI `10.1177/027836498800700301`.
  Confirmed real, title/journal/date match exactly.
- Ferrari, C. & Canny, J., "Planning optimal grasps," *Proc. IEEE ICRA*
  1992. DOI `10.1109/robot.1992.219918`. Confirmed real — this is the
  standard force-closure grasp-quality metric (the "Ferrari-Canny metric")
  still used as the analytic ground truth in modern data-driven grasp
  planners (§1d below).
- Ponce, J. & Faverjon, B., "On computing two-finger force-closure grasps of
  curved 2D objects," *Proc. IEEE ICRA* 1991, DOI `10.1109/robot.1991.131614`;
  the polygonal/three-finger extension has a journal version, "On computing
  three-finger force-closure grasps of polygonal objects," *IEEE
  Transactions on Robotics and Automation*, DOI `10.1109/70.478433`. Both
  confirmed real via CrossRef.
- Miller, A.T. & Allen, P.K., "GraspIt!: A Versatile Simulator for Robotic
  Grasping," *IEEE Robotics & Automation Magazine* 11(4), 2004. DOI
  `10.1109/mra.2004.1371616`. Confirmed real — GraspIt! is the reference
  grasp-analysis tool that formalized checking force closure/contact
  geometry as distinct from simply detecting contact.

**Data-driven grasp planning still treats force closure as the ground-truth
label, not contact alone** (re-verified via arXiv API this session):
- Mahler, J. et al., "Dex-Net 2.0: Deep Learning to Plan Robust Grasps with
  Synthetic Point Clouds and Analytic Grasp Metrics," arXiv:1703.09312.
  Confirmed real (title/authors match exactly). Dex-Net's own training
  labels are analytic force-closure/robust-grasp-quality metrics (Ferrari-
  Canny-family), not raw contact detection.
- ten Pas, A. et al., "Grasp Pose Detection in Point Clouds," arXiv:1706.09911.
  Confirmed real. GPD's candidate generation is explicitly antipodal-sampling-
  based, not "any hard contact."

**RL-specific evidence that a contact-direction/force-aware reward
outperforms a binary or magnitude-only one** (the most directly relevant new
citations found this session, verified via the arXiv API):
- **Koenig, A., Liu, Z., Janson, L., Howe, R., "The Role of Tactile Sensing
  in Learning and Deploying Grasp Refinement Algorithms," arXiv:2109.11234
  (2021).** Confirmed real via the arXiv API (title/authors/abstract match).
  Abstract, read directly: systematically integrates different levels of
  tactile data (contact positions, normals, forces) into RL rewards for
  multi-fingered grasp refinement via analytic grasp-stability metrics, and
  finds "combining information on contact positions, normals, and forces in
  the reward yields the highest average success rates... This contact-based
  reward outperforms a non-tactile binary-reward baseline by 42.9%." This is
  the closest available RL-specific analogue to this project's own
  Experiment 1→9 progression (magnitude-only contact reward → directional/
  antipodal contact reward), independently arriving at the same qualitative
  conclusion (contact *direction/normal* information in the reward beats a
  cruder binary/magnitude signal) on a different manipulator/task.
- **Zhang, B., Andrussow, I., Zell, A., Martius, G., "The Role of Tactile
  Sensing for Learning Reach and Grasp," arXiv:2502.20367 (2025).** Confirmed
  real via the arXiv API. Abstract, read directly: compares tactile/
  environmental setups for model-free RL antipodal grasping, finding tactile
  features improve learning outcomes specifically "under imperfect visual
  perception" — relevant since this project's Franka die-lift observations
  are privileged/ground-truth state (not vision), which is a real, flagged
  gap between this citation's own setting and this project's own (see §1e).

**A related, not-yet-independently-verified-beyond-abstract candidate**
found but not relied on for the hypothesis below (flagged rather than
silently dropped): PONG, "Probabilistic Object Normals for Grasping via
Analytic Bounds on Force Closure Probability," arXiv:2309.16930 — confirmed
to exist via the arXiv API (title matches), but its full text was not read
this session, so its specific claims are not cited here beyond confirming
the paper is real.

### 1e. Open risks / gaps for Direction 1, stated plainly

- **The strongest RL-specific citation (Koenig et al. 2021, §1d) is on a
  multi-fingered anthropomorphic hand with real tactile sensors, not a
  2-finger parallel-jaw Franka gripper with privileged simulated state.**
  The qualitative conclusion (directional/normal contact info in the reward
  beats magnitude-only) is not shown to depend on hand DoF or sensing
  modality in that paper — this project's own Experiment 1→9 result is
  actually the more directly analogous evidence (same 2-finger parallel-jaw
  regime, simulated ground-truth `ContactSensorCfg` state, not real tactile
  sensors) — but this is an extrapolation from a different hand/sensing
  regime, flagged rather than presented as a literal match.
- **This project's own AR4 precedent (§1b) found the antipodal-reward fix
  alone was insufficient under joint-space control** — the real fix required
  changing the action space too (Experiment 11). Franka die-lift currently
  uses joint-space control (§1b). Any spec building on this research should
  explicitly plan for this contingency rather than treating a Franka-side
  reproduction of Experiment 10's "signal regresses to zero" outcome as a
  fresh surprise.
- **No literature source found that studies this exact combination**
  (rigid, non-parallel-face-rich, low-sphericity convex polyhedron + 2-finger
  parallel-jaw gripper + RL-learned antipodal reward) as its own case — every
  citation above is either classical geometric grasp planning (not RL) or RL
  tactile sensing on a different hand morphology. This is a real, disclosed
  gap, not filled by any of the above.

---

## 2. Direction 2: d8's physical parameters at 48mm-parity

### 2a. Mass and friction are pinned identically across ALL FOUR shapes — confirmed by direct source read, not assumed

`tasks/franka/dice_lift_joint_env_cfg.py`'s own docstrings (e.g.
`FrankaDieLiftJointD8BigEnvCfg`, lines 651-679) state this explicitly and
repeatedly: mass is pinned at **0.216kg for every shape at every size in
this file**, inherited from `FrankaDieLiftJointHeavyEnvCfg`'s
`MassPropertiesCfg(mass=0.216)` override — "the same DexCube-measured
carried-over placeholder value used across every shape/size in this file,
**not a real d8-density estimate**... no established method exists yet to
derive a real per-shape mass for d8/d10/d12 analogously to d20 (d20's own
0.216kg was never itself a real-world-density derivation)." This is a
documented, deliberate control variable (the asset-bisect ladder's own
methodology, isolating shape from mass/size), not an oversight, and not
something newly discovered in this task — but its consequences for the
density-consistency question this task was asked to check had not
previously been computed.

Friction: no `RigidBodyMaterialCfg`/friction override exists anywhere in
`tasks/franka/dice_lift_joint_env_cfg.py`, `tasks/franka/lift_env_cfg.py`, or
`scripts/bake_die_asset.py` (confirmed by grep across all three — the only
`friction` reference in the whole `tasks/franka/` tree is
`self.sim.physx.friction_correlation_distance = 0.00625`, a global PhysX
solver setting, not a per-body/per-shape material property). Every die
shape uses whatever Isaac Lab/PhysX's implicit default material provides,
identically. Collision approximation is likewise identical: `bake_die_asset.py`
applies `CreateApproximationAttr("convexHull")` to every shape's mesh with no
per-shape branching, and every env cfg in `dice_lift_joint_env_cfg.py` reuses
the same `_D20_RIGID_PROPS` `RigidBodyPropertiesCfg` (solver iteration
counts, velocity limits — no material/density fields) for d8/d10/d12/d20
alike.

**This directly answers the task's framing question: is d8 anomalously
light/heavy/slippery *relative to the shapes that work* (d12/d20)? No —
mass, friction, and collision-approximation method are bit-for-bit identical
across all four shapes at 48mm-parity.** Since d12/d20 achieve real
grasp-and-lift discovery under this exact same pinned mass and default
friction, the Franka gripper+PPO combination is demonstrably not blocked by
"0.216kg (or whatever the default friction coefficient is) is too much/too
slippery to lift" in any shape-independent sense — ruling out a physical-
parameter miscalibration *relative to the working shapes* as the
explanation for d8's specific null.

### 2b. A real, previously-undocumented density inconsistency — found this session, but argued against as the primary driver

The task asked specifically for a density-consistency check against d12/d20
using real measured geometry. This project's own already-computed convex-
hull volumes (`tasks/franka/shape_observations.py`'s
`SHAPE_GEOMETRY_DESCRIPTORS` derivation comment: d8 V=745.021343, d10
V=1165.970252, d12 V=15330.053595 native stage-unit^3, each measured from the
shape's own native/unscaled baked mesh) combined with each shape's own
freshly-derived 48mm-parity scale constant
(`dice_lift_joint_env_cfg.py`'s own per-class docstrings) let the implied
uniform density at the pinned 0.216kg be computed directly (real geometry ×
real scale × real mass — arithmetic below, not estimated):

| shape | native max-dim (stage units) | 48mm-parity scale | real volume at 48mm (cm³) | implied density at 0.216kg (g/cm³) |
|---|---|---|---|---|
| d8  | 15.1544 | 0.003167 (`FrankaDieLiftJointD8BigEnvCfg`)  | 23.67 | **9.13** |
| d10 | 16.3931 | 0.002928 (`FrankaDieLiftJointD10BigEnvCfg`) | 29.27 | 7.38 |
| d12 | 32.5160 | 0.001476 (derived: `48.0/(32.5160*1000)`, matching `FrankaDieLiftJointD12D20MixedEnvCfg`'s `0.001476` constant) | 49.31 | 4.38 |
| d20 | ≈30.3 (real-standard size at `scale=0.001`, per `_D20_USD`'s own real-size convention) | 0.001585 (`FrankaDieLiftJointBigEnvCfg`) | 58.39 | 3.70 |

(Real volume = native convex-hull volume × scale³, since `real_length_mm =
native_max_dim × scale × 1000` is the file's own documented scale
convention — confirmed against the docstrings' own "effective max dim"
sanity checks, e.g. d8: `15.1544 × 0.003167 × 1000 = 47.994mm`, matching the
docstring's own stated 47.994mm exactly.)

**This is a real, previously-uncomputed finding: at 48mm-parity, d8's
implied material density is ~2.5x d20's and ~2.1x d12's**, because mass is
held constant while d8's actual geometric volume at the same "max dimension"
anchor is much smaller (a low-sphericity octahedron has much less internal
volume per unit bounding-box diagonal than a near-spherical icosahedron).
For context (not independently verified beyond general engineering
knowledge, not requiring a citation): real tabletop dice are typically
resin/ABS plastic (~1.1-1.4 g/cm³) or, for weighted/metal dice, zinc alloy
(~6.6-6.9 g/cm³) or brass (~8.4-8.7 g/cm³). d8's implied 9.13 g/cm³ sits
above real brass dice; d20's implied 3.70 g/cm³ sits well below any common
dice material. **All four shapes are physically unrealistic for a plastic
die at this pinned mass, and d8 is the worst outlier of the four** — a real,
concrete, previously-undocumented anomaly this task surfaced.

**Why this is argued against as the primary explanation for d8's specific
grasp failure, despite being real:** the physics engine (PhysX, via Isaac
Lab's `MassPropertiesCfg`) takes total mass as a direct, specified scalar —
it does not derive dynamics from "density" as an independent physical
quantity beyond normalizing the mesh's own computed inertia tensor to match
that total mass. What actually determines the *force* a two-finger gripper
must exert to support the object against gravity is the specified mass
(0.216kg, IDENTICAL for all four shapes) and the friction coefficient
(default, IDENTICAL for all four shapes) — not the implied density number
above, which is a derived/descriptive ratio, not an input the simulator
uses for grip-force requirements. Since d12/d20 already succeed at lifting
this exact same 0.216kg mass under this exact same default friction, the
Franka gripper/PPO combination is demonstrably *capable* of generating
enough grip force and lift for an object at this pinned mass — ruling out
"d8 is being asked to lift something too heavy for the gripper" as the
mechanism. What plausibly *does* differ geometrically because of d8's small
real volume relative to its bounding-box diagonal is contact-surface
area/local curvature at the actual points the gripper touches (a much
narrower "waist" cross-section for a low-sphericity octahedron than for a
near-spherical d20 of the same nominal bbox size) — but that is a restatement
of the already-known sphericity/parallel-face-pair geometric story
(§2c below and
`docs/superpowers/specs/research/2026-07-19-d8-d10-grasp-discoverability-literature.md`'s
§2a/§2b), not a new, independent mass/density mechanism.

### 2c. Relation to the already-established sphericity/geometry finding

The density computation above is mechanically the *same* underlying fact
(d8's low sphericity → small volume relative to its bounding-box diagonal)
that `docs/superpowers/specs/research/2026-07-19-d8-d10-grasp-discoverability-literature.md`
§2a already identified as the cleanest existing signal in this project's own
data (Wadell sphericity monotonic with discovery rate: d8 0.8896/0-of-3, d10
0.8959/0-of-3, d12 0.9286/1-of-3, d20 0.9524/2-of-3) and §2b already
identified d10's additional anisotropy/no-parallel-face disadvantage. This
task's density computation is a genuinely new number (not previously
computed anywhere in this project's record) but does not point at a
*separate* physical-parameter root cause — it is better read as further,
independent confirmation of the same geometry-affordance story already
established, expressed in mass/density terms rather than sphericity terms.

### 2d. Open risks / gaps for Direction 2, stated plainly

- **The 9.13/7.38/4.38/3.70 g/cm³ figures use each shape's own "native
  max-dim" measurement, one of which (d20's ≈30.3) was not re-measured
  directly in this session** — it is taken from this project's own already-
  documented real-standard-size convention (`scale=0.001` → 30.3mm, per
  `_D20_USD`'s own established real-size derivation, [[size-curriculum]]).
  If that convention is itself imprecise, the d20 density figure shifts
  proportionally (as `native_max_dim³`), though the qualitative d8-vs-d20
  ordering (driven by the volume ratio, not either endpoint's absolute
  precision) is robust to any plausible small correction here.
- **No literature source was found (or searched for) establishing what
  density plastic/resin dice specifically use** — the ~1.1-1.4 g/cm³ figure
  cited above is general engineering knowledge about ABS/resin plastics, not
  a verified dice-specific citation, and is explicitly not presented as one.
- **This document does not test whether the density anomaly, even though
  argued to be mechanically inert for grip-force requirements, has any
  second-order effect via the inertia tensor's *shape*** (not just its
  overall scale) on contact dynamics during the closure moment itself — this
  is a real, unexplored, lower-priority question flagged here rather than
  resolved.

---

## 3. Proposed hypothesis (for a future spec to cite — NOT a spec itself)

**The evidence supports Direction 1 (contact/antipodal grasp verification)
as the primary, better-grounded candidate, not a combination of both
directions.** Direction 2's own evidence argues against a physical-
parameter anomaly being the driver, precisely because d8's mass/friction
values are not anomalous *relative to* the two shapes (d12/d20) that already
succeed under the identical values — the density inconsistency found in
§2b is real but reduces to the same already-known geometry story, not an
independent mechanism.

**H (primary, falsifiable): the missing signal is grasp-quality
(antipodal/force-closure), not exploration frequency or absolute
lift-height, and adding a real contact-force-direction reward term will
measurably increase antipodally-aligned closures on d8 specifically.**

Grounded in: (1) this project's own direct AR4-era precedent
(Experiments 1→9→10→11, §1b) showing a magnitude-only contact signal and a
correctly-thresholded antipodal signal are mechanistically different
things, and that the antipodal signal is learnable once positioning
precision is adequate; (2) the current Franka reward's confirmed complete
absence of any contact-quality term (§1a); (3) Koenig et al. 2021's
independent RL-specific finding that a contact-position/normal/force-aware
reward beats a binary-reward baseline by 42.9% on a different manipulator
(§1d); (4) re-verified, current Isaac Lab infrastructure
(`ContactSensorCfg`/`force_matrix_w`, §1c) that makes this a real, buildable
mechanism, not a hypothetical one.

**Falsification condition:** if a `ContactSensorCfg`-based antipodal/
force-direction reward term (mirroring Experiment 9/10's mechanism —
per-jaw filtered `force_matrix_w`, contact-force-direction dot product
against a friction-cone-derived threshold), added to
`FrankaDieLiftJointD8BigEnvCfg` under its current joint-space
`JointPositionActionCfg(scale=0.5)`, produces **both**: (a) the antipodal
term itself failing to fire above noise level across all 3 seeds
(42/123/7) — mirroring Experiment 10's "regressed to exactly 0.000000"
outcome — **and** (b) 0/8 sustained-lift discovery in every seed at the
48mm-parity anchor, that would falsify "a contact-quality reward alone,
under the current joint-space action space, is sufficient to unlock d8" —
but per §1b's explicit caveat, this specific joint-and-signal combination
already failing on AR4 before task-space IK fixed it means such a result
should be read as a call to test the same fix (task-space action, per
Experiment 11) *before* concluding the whole contact/antipodal direction is
dead for Franka/d8 — not as an independent falsification of the
antipodal-verification hypothesis itself. A cleaner, fully dispositive
falsification of the hypothesis would require testing the contact/antipodal
reward under **both** action spaces (joint-space first per this project's
existing infra, task-space second reusing `dice_pick_demo.py`'s
already-proven DiffIK mechanism if joint-space fails) and finding 0/8 under
both.

**Direction 2's own falsification (already effectively concluded by this
document, not deferred to a future run):** if a future actual d8-density
correction (rescaling mass to a physically realistic ~1.2-1.4 g/cm³ resin
density, i.e. roughly 28-33g rather than 216g at 48mm) were tested and
still produced 0/8 discovery, that would confirm §2b's own reasoning that
the density anomaly is not the operative mechanism (already the
better-supported reading here, given d12/d20 already succeed at a *more*
physically unrealistic mass than d8's own realistic-density figure would
require) — this is not proposed as the recommended next experiment, since
the reasoning above already argues its outcome is predictable.

**Ordering rationale:** Direction 1 is recommended as the next actionable
step because it (a) targets a reward-signal gap that is confirmed to exist
(no contact-quality term at all, for any shape, in this Franka arc) and (b)
has this project's own strongest available evidence — a controlled,
independently-replicated in-repo precedent (Experiments 1/9/10/11) showing
the exact "magnitude-only contact succeeds, antipodal check does not fire
until positioning is precise" pattern this SPLIT result is consistent with.
Direction 2 is not recommended as an independent next experiment: its own
best available evidence (§2a, mass/friction are pinned and shared with the
already-successful shapes) argues against it being the primary driver, and
its one genuinely new finding (§2b's density inconsistency) is best
understood as restating the already-established sphericity story rather
than opening an independent physical-parameter fix.

---

## Related

[[exploration-bonus-grasp-discovery]] (the SPLIT result this document
follows up on), [[grasp-mechanics-antipodal-vs-magnitude]] (source of the
AR4-era antipodal-vs-magnitude precedent this hypothesis is grounded in),
[[experiment-01-contact-sensor-grasp-reward]], [[experiment-09-antipodal-grasp-bonus]],
[[experiment-10-antipodal-threshold-action-scale-solver]],
[[experiment-11-taskspace-ik]] (the four-experiment AR4 arc this hypothesis
directly builds on), [[dice-pick-demo]] (source of the already-proven d8/d10
task-space grasp trajectory this document's falsification condition would
fall back on), `docs/superpowers/specs/research/2026-07-19-d8-d10-grasp-discoverability-literature.md`
(source of the sphericity/geometry finding this document's Direction 2
result is read against),
`kb/wiki/concepts/citation-verification-practice.md` (the standing practice
this document's citation checks follow).
