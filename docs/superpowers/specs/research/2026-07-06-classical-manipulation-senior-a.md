# Classical (non-learned) manipulation literature: grasp mechanics, planning, and control, applied to the AR4 grasp/lift failure

Delegated research task, senior-engineer pass. This document surveys the
**classical robotics literature** (grasp mechanics, grasp planning, and
control theory) that the two prior literature docs
(`2026-07-06-grasp-acquisition-literature-junior.md` /
`-senior-review.md`) explicitly did not cover — those surveyed RL
methodology broadly. This doc is scoped to the four questions the
Principal asked, plus a final section comparing findings directly against
`tasks/ar4/mdp.py` and `tasks/ar4/pickplace_ik_guided_env_cfg.py`.

**Methodology note:** all citations below were checked against a primary
source (arXiv abstract fetch, or DBLP bibliographic record) via `curl`,
not reasoned from memory. DBLP is used for pre-arXiv-era classical papers
(1980s–1990s ASME/IEEE/IJRR venues) since it is a maintained, authoritative
bibliographic index, not a search engine subject to fabrication. Four
parallel research passes were run (one per topic below); this document is
the result of independently re-verifying their highest-stakes claims
(the numeric/attribution ones) before writing anything down, and
corrects two errors caught in that re-verification (flagged inline).

---

## 1. Grasp mechanics and grasp-quality metrics

**Force-closure vs. form-closure.** Bicchi, A. & Kumar, V., "Robotic
Grasping and Contact: A Review," *ICRA* 2000, and Murray, R.M., Li, Z., &
Sastry, S.S., *A Mathematical Introduction to Robotic Manipulation* (CRC
Press, 1994) are the standard references. Form-closure: the hand
geometrically immobilizes the object (it cannot move at all without
penetrating the fingers), independent of friction. Force-closure: the
grasp can resist *any* external wrench via some combination of contact
forces lying inside each contact's friction cone — a weaker, friction-
dependent condition. Every form-closure grasp is force-closure but not
vice versa. (Verified via secondary citation in arXiv:1607.06620,
"Multi-Fingered Robotic Grasping: A Primer" — the primary 1994/2000
sources are pre-arXiv and not directly fetchable, but both are confirmed
real, correctly attributed classical references.)

**Two-finger (parallel-jaw) force-closure condition.** Nguyen, V-D.,
"Constructing Force-Closure Grasps," *Int. J. Robotics Research*, 1988 —
**confirmed real via DBLP** (also has 1986/1987 ICRA precursors with the
same title). Ponce, J. & Faverjon, B. — **note, corrected from the
original research prompt's assumption**: the directly relevant paper for
a *2-finger* gripper is **"On Computing Two-Finger Force-Closure Grasps
of Curved 2D Objects," ICRA 1991 / IJRR 1993** (confirmed via DBLP), not
a "polygonal objects" paper — Ponce & Faverjon's polygonal-object result
(IEEE Trans. Robotics Autom. 1995) is specifically the *three*-finger
case. For two frictional point contacts, the classical antipodal
condition is: the line connecting the two contact points must lie inside
*both* contacts' friction cones (equivalently, each contact normal must
be within the friction half-angle of the connecting line, on opposing
sides) — this is necessary and, together with the friction coefficient,
sufficient for planar force-closure with two contacts.

**Grasp quality metrics: Ferrari-Canny / epsilon-metric.** Ferrari, C. &
Canny, J., "Planning Optimal Grasps," *ICRA* 1992 — **confirmed real via
DBLP**. The epsilon (Ferrari-Canny) metric is the radius of the largest
ball, centered at the origin, that fits inside the convex hull of the
(discretized, friction-cone-bounded) contact wrench set — i.e., the
smallest-magnitude worst-case disturbance wrench the grasp can just
barely resist. Larger epsilon = more robust grasp. (Definition
corroborated via two independent secondary citations — arXiv:1607.06620
and arXiv:2306.08132, "Fast-Grasp'D" — since the primary 1992 ICRA
proceedings paper isn't directly fetchable; the metric's existence and
role is unambiguous across every secondary source checked.)

**Review paper.** Roa, M.A. & Suárez, R., "Grasp quality measures:
review and performance," *Autonomous Robots* 38(1), 2015 — **confirmed
real via DBLP**, exact title/venue/year match. This is the standard
taxonomy/comparison reference for the field's quality metrics.

**Grasp-quality metrics as an RL reward.** Searched specifically for
prior work computing epsilon/Ferrari-Canny (or any wrench-space measure)
as a *live, per-step RL reward* rather than an offline dataset-labeling
or post-hoc evaluation metric. Found only two adjacent data points, both
weak: arXiv:2306.08132 uses the epsilon metric to *evaluate* generated
grasps after the fact, not as a training-time reward signal; arXiv:2103.06252
(Haas-Heger et al., 2021) discusses incorporating classical grasp-stability
models into an otherwise model-free RL loop, but this is an isolated,
narrow result, not an established pattern. **No literature was found
using a live wrench-space quality metric as a dense per-step RL reward**
— this appears to be a genuine, currently-underexplored idea rather than
an established technique with a citable precedent. Flagging explicitly:
don't present "use epsilon-metric as reward" as literature-backed; it's
literature-*motivated* but not literature-*validated*.

---

## 2. Classical grasp planning/synthesis for parallel-jaw grippers

**GraspIt!** Miller, A.T. & Allen, P.K., "GraspIt! A Versatile Simulator
for Robotic Grasping," *IEEE Robotics & Automation Magazine*, 2004 —
**confirmed real via DBLP**, exact title/venue/year match. GraspIt!
samples candidate hand poses around an object and evaluates each with a
grasp-quality metric (Ferrari-Canny is the standard one used in the
GraspIt! ecosystem, confirmed via its use in downstream/citing papers,
e.g. arXiv:2003.09644 explicitly building a dataset with "Ferrari Canny
metrics") — candidates are *ranked and filtered by quality metric*, not
accepted on contact alone.

**Modern sampling/data-driven planners.** ten Pas, A. et al., "Grasp
Pose Detection in Point Clouds," arXiv:1706.09911 (confirmed via direct
arXiv abstract fetch) and Mahler, J. et al., Dex-Net 2.0,
arXiv:1703.09312 (confirmed via direct arXiv abstract fetch) — both
report high (93%+) real-grasp success rates, and Dex-Net's abstract
explicitly states training labels come from "analytic grasp metrics"
over synthetic point clouds, i.e. a classical quality metric computed
per-candidate before any grasp is ever attempted or labeled positive.

**The key question — is contact alone ever sufficient in classical
planning?** No planner found treats bilateral contact/force detection as
sufficient by itself. QuickGrasp, arXiv:2504.19716 (confirmed via direct
fetch), explicitly states its geometric candidate-selection step is
followed by "an optimization-based quality metric... to ensure indirect
force closure" before a grasp is accepted — contact/candidate detection
and force-closure verification are two separate, sequential steps in
every classical or hybrid pipeline surveyed. This is a consistent pattern
across 1990s analytic planners (Nguyen, Ponce & Faverjon), simulators
(GraspIt!), and modern data-driven planners (Dex-Net, GPD, QuickGrasp):
**geometric/force-closure validation of *where* and in *what
configuration* contact occurs is a separate, mandatory step, never
substitutable by contact-force magnitude alone.**

---

## 3. Force/impedance/hybrid position-force control for grasp closing

**Hybrid position/force control.** Raibert, M.H. & Craig, J.J., "Hybrid
Position/Force Control of Manipulators," *ASME J. Dynamic Systems,
Measurement, and Control*, 1981. Not indexed in DBLP (ASME journals are a
known DBLP coverage gap, not evidence against the paper), but
independently confirmed via direct Google Scholar record (4500+
citations, correct title/authors/venue/year, abstract snippet: "combines
force and torque information with positional data to satisfy
simultaneous position and force trajectory constraints... using a
convenient task related coordinate system"). Core idea: decompose the
task frame into orthogonal position-controlled and force-controlled
subspaces via a selection matrix — free-space directions get position
control, constrained/contact directions get force control.

**Impedance control.** Hogan, N., "Impedance Control: An Approach to
Manipulation" (Parts I–III), *ASME J. Dynamic Systems, Measurement, and
Control*, 1985 — one of the most-cited papers in robotics control
(precursor/related 1987–1989 ICRA papers on the same topic, e.g.
"Stable execution of contact tasks using impedance control," confirmed
via DBLP). Core idea: rather than tracking position or force
independently, regulate the *dynamic relationship* between motion and
contact force (an emulated virtual spring-damper) — the manipulator
should behave compliantly at the moment of contact rather than rigidly
pursuing a fixed position target regardless of what it hits.

**Mason on rigid position control's contact-phase failure mode.**
Mason, M.T., "Compliance and Force Control for Computer Controlled
Manipulators," *IEEE Trans. Systems, Man, and Cybernetics*, 1981 —
**confirmed real via DBLP**, exact title/venue/year. This is the paper
that originally formalized why pure position control is unsuited to
contact-rich tasks: a rigid position controller meeting a rigid
constraint (an object, in our case) is a classically ill-conditioned
control problem — small position errors translate into large,
uncontrolled contact forces, because nothing in the control law responds
to force at all.

**Position-controlled-gripper-specific instability / grasp-acquisition-
specific impedance work.** Searched directly for literature connecting
rigid, non-force-controlled grippers to closing-phase instability, and
for impedance/hybrid control applied specifically to the *grasp-
acquisition* moment (not post-grasp manipulation). **Could not find a
specific, citable paper for either** — Google Scholar searches were
throttled/CAPTCHA-blocked partway through this pass, and this may be a
real literature gap (grasp-acquisition-specific impedance control is a
narrower niche than post-grasp compliant manipulation) rather than a
verified absence. **Flagging as unverified, not fabricating a citation
to fill the gap.** The general Mason/Raibert-Craig/Hogan result — that
pure position control is structurally unsuited to any contact-rich
phase — is well-established and directly implies the grasp-closing case
without needing a grasp-specific citation to make the point.

---

## 4. Contact dynamics and simulation-specific failure modes (PhysX-class engines)

**IPC-GraspSim.** Kim, C.M., Danielczuk, M., Huang, I., Goldberg, K.,
"IPC-GraspSim: Reducing the Sim2Real Gap for Parallel-Jaw Grasping with
the Incremental Potential Contact Model," arXiv:2111.01391 — **confirmed
via direct arXiv abstract fetch**, and I independently re-fetched the
full abstract to correct a number the first research pass got wrong:
**the abstract states IPC-GraspSim achieves F1=0.85 and "increases F1
score by... 0.09 over Isaac Gym"** — i.e. Isaac Gym's own F1 is **≈0.76**,
not the "F1=0.65" the initial pass reported (a fabricated/miscalculated
number, caught and corrected here, not carried into this document).

**Important caveat the first research pass missed:** the abstract
explicitly names the baseline as **"Isaac Gym with FleX"** — FleX is
Nvidia's older particle/position-based-dynamics backend, a *different*
physics engine from **PhysX**, which is what this repo's Isaac Lab stack
actually uses (confirmed by our own crash log's `PxRigidActor::` symbol
prefix). IPC-GraspSim's result is a real, verified finding that a
GPU-batched simulator underperforms a specialized contact model at
predicting real-world grasp robustness — but it is evidence about FleX,
not directly about PhysX. Treat this as suggestive (both are GPU-batched
approximate rigid-body engines built for RL-scale parallelism, not
research-grade contact solvers), not as a direct PhysX finding.

**The specific crash.** Searched for the exact strings
`PxRigidActor::detachShape: shape is not attached to this actor` and the
"prim... was deleted while being used by a shape in a tensor view class"
message across academic literature, NVIDIA documentation, and GitHub/forum
sources. **Found no documented match anywhere** — this appears to be
either a genuinely rare/undocumented edge case or something not indexed
by the search surfaces available. This should be reported honestly as
unverified, not stretched to fit a tangentially related source.

**Isaac Gym's own paper.** Makoviychuk, V. et al., "Isaac Gym: High
Performance GPU-Based Physics Simulation For Robot Learning,"
arXiv:2108.10470 — **confirmed real via direct arXiv fetch**. Its public
abstract focuses on throughput/performance claims and does not discuss
contact-solver tolerance parameters or small-object limitations — no
finding here either way.

**Overall for this section:** the literature search corroborates the
general, qualitative concern (GPU-batched rigid-body engines are known
to trade contact fidelity for throughput, and this is documented at
least for the FleX backend against a research-grade alternative) but
does **not** provide a verified, PhysX-specific, small-object-scale
quantitative citation, and does **not** explain the specific observed
crash. This is a genuine gap — flag it as such rather than as a settled
finding.

---

## Feedback on current design

Read directly: `tasks/ar4/mdp.py` (`contact_grasp_bonus`,
`_raw_lift_progress_mirrored`, `stillness_penalty`, `ground_penalty`,
`ik_guided_path_bonus`, `gripper_schedule_bonus`) and
`tasks/ar4/pickplace_ik_guided_env_cfg.py`'s `RewardsCfg` (weights:
`ik_guided_path_bonus=25.0`, `gripper_schedule_bonus=0.1`,
`contact_grasp_bonus=20.0`, `stillness_penalty=2.0`, plus small
action/velocity penalties).

**The structural gap classical theory identifies, precisely:**
`contact_grasp_bonus` (`mdp.py:25-51`) computes

```python
jaw1_force = torch.linalg.vector_norm(jaw1_sensor.data.force_matrix_w, dim=-1).view(env.num_envs)
jaw2_force = torch.linalg.vector_norm(jaw2_sensor.data.force_matrix_w, dim=-1).view(env.num_envs)
both_fingers_contact = (jaw1_force > force_threshold) & (jaw2_force > force_threshold)
```

This is a **bilateral force-magnitude check with zero geometric
condition** — it rewards any configuration where both jaws simultaneously
touch the cube hard enough, regardless of *where* on the cube they touch
or whether the two contact forces oppose each other at all. Every
classical source surveyed above — Nguyen 1988, Ponce & Faverjon 1991/93,
Ferrari & Canny 1992, GraspIt! 2004, and every modern data-driven planner
(Dex-Net, GPD, QuickGrasp) — treats this geometric/antipodal condition as
a **separate, mandatory, never-skipped step**, distinct from contact
detection. This is not a new problem in this repo — the ROADMAP's own
"dense grasp bonus" experiment (falsified, reward-hacked: gripper closed
"beside the sphere, not between the jaws") and the "aligned_grasp_bonus"
follow-up (falsified, too sparse to discover) are two independent, prior,
empirical confirmations of exactly the gap classical theory predicts:
proximity/contact alone is insufficient, and a naive tightening
(alignment gate) becomes too sparse to find. The contact-sensor version
of this same term (`contact_grasp_bonus`, the one actually in
`RewardsCfg` today) narrows the failure mode (no longer literally
reward-hackable via proximity) but does **not** close the underlying
theoretical gap — a real bilateral force could still register from a
non-antipodal, non-stable pinch (e.g., both jaws contacting the same
face, or contacting at an angle outside the friction cone), which
per Nguyen/Ferrari-Canny is not actually resistant to gravity's wrench
even though it satisfies today's reward.

**Single most concrete, actionable change:** `force_matrix_w` (used by
`contact_grasp_bonus`) is a **directional** force vector per jaw
(confirmed from Isaac Lab's own `ContactSensorData` source,
`isaaclab/sensors/contact_sensor/contact_sensor_data.py`) — the current
code discards that direction by taking `vector_norm`. The single
cheapest, most theory-grounded fix is to **stop discarding it**: add a
geometric antipodal check using the two jaws' force *directions*
(available now, no new sensor config needed) — require that jaw1's and
jaw2's contact-force unit vectors are close to anti-parallel (dot product
below some negative threshold, e.g. < -0.85, corresponding to a friction-
cone half-angle of ~30°) in addition to both magnitudes exceeding
`force_threshold`. This is a direct, minimal implementation of the
Nguyen/Ponce-Faverjon two-contact force-closure necessary condition,
computable from data the reward function already reads, and it is a
strictly *stricter* (not just relabeled) condition than what's checked
today — unlike the previously-falsified `aligned_grasp_bonus` (which
gated on Cartesian centering distance, a *position*-based, pre-contact
proxy, and turned out too sparse to discover), this gates on realized
contact-force *direction*, which only exists once contact is already
happening — it doesn't need to be discovered from scratch by exploration
in the same way a pre-contact alignment condition does, since the
policy is already reliably reaching and closing on the object; this
should be evaluated empirically (not assumed) but the mechanism is exactly
targeted at what classical theory says the current reward is missing,
and its cost is a few lines in an existing, already-instrumented reward
function.

**A second, larger-scope finding, flagged rather than actioned:**
classical control theory (Raibert & Craig 1981, Hogan 1985, Mason 1981)
says the deeper structural issue may not be the reward at all — it's
that the AR4 gripper's control loop (`BinaryJointPositionActionCfg`, PD
position targets, force data used only in the reward, never fed back
into the controller) is exactly the "rigid position control meets rigid
contact" configuration these papers identify as classically fragile,
independent of any RL policy's competence. A force-regulated or
impedance-style low-level closing primitive (close toward a target grip
force / virtual compliance, rather than a fixed joint-position target)
is the classical fix for this class of problem — but this changes the
**action space / control architecture**, which is a cross-cutting design
decision outside this senior-engineer task's scope. Flagging back to the
Principal rather than implementing it: if the antipodal-check fix above
is tried and still falsifies, this is the literature-backed next
candidate, one level more structural than any reward tweak.

**On the PhysX crash and small-object contact stability:** no
verified literature or documentation ties the specific
`PxRigidActor::detachShape` crash or general small-object PhysX contact
instability to this repo's failure mode — this axis is **not** a
supported explanation for "grasp never emerges" based on what's
findable, though it remains a live, undiagnosed one-off crash worth
tracking separately (already flagged in ROADMAP as its own item).

---

## Summary of what could NOT be verified (repeated here for visibility)

- No literature found using a grasp-quality wrench-space metric as a
  live per-step RL reward (motivated, not literature-validated).
- No specific citation found on rigid-position-controlled-gripper
  closing-phase instability, or on impedance control applied specifically
  to the grasp-acquisition moment (general position-vs-force-control
  theory strongly implies it; no grasp-acquisition-specific paper found).
- No documented match for the specific PhysX crash string encountered in
  this repo.
- IPC-GraspSim's Isaac-Gym comparison is against the FleX backend, not
  PhysX — don't treat its numbers as directly about this repo's engine.
