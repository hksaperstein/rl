# Research: why d8/d10 show zero grasp discovery while d12/d20 partially succeed — geometry-specific literature grounding

**Date:** 2026-07-19
**Author:** Senior research thread (delegated by Principal)
**Purpose:** Tier 1 hypothesis-gate research for a possible follow-on to the
completed unified-die-specialist experiment
(`kb/wiki/experiments/unified-multi-die-specialist-distillation.md`),
re-examining `BACKLOG.md`'s "Task 4 scope decision" (2026-07-19) deferral of
d8/d10. That entry cited this project's systematic-debugging precedent ("3+
failed fixes on the same mechanism escalates rather than invites a fourth
speculative tweak") as grounds not to keep iterating on d8/d10 — this
document first checks whether that precedent actually applies (it does not:
only one recipe was ever tried), then does the geometry-specific research
that precedent-check was substituting for. **This is research only — no env
cfg code, no reward-term design, no Isaac Sim launches.**

---

## 1. Checking the record: was "only one recipe" actually tried?

Confirmed by direct source read, not by re-reading the prior BACKLOG
paraphrase. Every d8/d10/d12/d20 env cfg class in
`tasks/franka/dice_lift_joint_env_cfg.py` (`FrankaDieLiftJointD8StandardEnvCfg`,
`...D10StandardEnvCfg`, `...D8BigEnvCfg`, `...D10BigEnvCfg`, and their d12/d20
counterparts) overrides exactly three things relative to their shared parent
(`FrankaDieLiftJointHeavyEnvCfg`): `usd_path`, `spawn.scale`, and
`die_shape_class`. None of them touches rewards, PPO hyperparameters,
observations beyond the Task-1 shape/geometry terms (applied identically to
all four shapes), termination conditions, or episode structure.

The PPO/exploration configuration is one shared class,
`FrankaLiftPPORunnerCfg` (`tasks/franka/agents/rsl_rl_ppo_cfg.py:26-54`),
used unchanged by every specialist run in Task 2/Task 3/Task 3.5/the
d20-big-geom gate:

```
policy = RslRlPpoActorCriticCfg(init_noise_std=1.0, actor_hidden_dims=[256,128,64], ...)
algorithm = RslRlPpoAlgorithmCfg(clip_param=0.2, entropy_coef=0.006,
    num_learning_epochs=5, num_mini_batches=4, learning_rate=1.0e-4,
    schedule="adaptive", gamma=0.98, lam=0.95, desired_kl=0.01, ...)
```

The reward function is likewise one shared class, `RewardsCfg`
(`tasks/franka/lift_env_cfg.py:245-269`), whose own docstring states it is
"Stock reward shape: dense reach + binary lift (**NOT** antipodal/
contact-force-gated — a deliberate, known difference from this project's own
AR4-era grasp-verification gate...)". The AR4-era antipodal-grasp-check
mechanism (`grasp_contact`/`antipodal_grasp_bonus`,
[[grasp-mechanics-antipodal-vs-magnitude]]) was never ported to the Franka
task at all — it is a real, available, but wholly untried mechanism-level
lever, not a config knob (see §6).

**Verdict: the reframing in the task brief is accurate.** Task 2 (real
~16-18mm size) and Task 3.5 (48mm parity) both ran the *identical* reward
function and PPO hyperparameters against d8/d10 as they did against the
cube/d12/d20 that worked — object asset (mesh, scale) was the only variable
ever changed. No curriculum specific to shape/geometry difficulty, no
grasp-favorable or expert-informed initialization, no exploration-noise/
entropy retuning, and no demonstration data have ever been attempted for
d8/d10. This is genuinely one recipe applied unmodified to a harder case,
not repeated patching of the same mechanism after multiple distinct fixes —
`BACKLOG.md`'s cited precedent (whose own origin, confirmed by re-reading
[[grasp-mechanics-antipodal-vs-magnitude]] and the AR4-era sphere-lift
sequence it traces to, is a *different* task and mechanism entirely) does
not actually apply here, and citing it to justify deferring d8/d10 was a
category error, not a considered application of that precedent.

---

## 2. Geometric characterization: real numbers, not shape-name guessing

All figures below are measured, not estimated — either already computed and
committed in this repo (`tasks/franka/shape_observations.py`,
`tasks/franka/dice_lift_joint_env_cfg.py`'s per-shape scale-derivation
docstrings, both re-derived from direct `scripts/_diag_shape_sphericity_check.py`/
`scripts/_diag_d8d10d12_standard_scale_check.py` mesh measurements) or a
directly-read engineering constant (`tasks/franka/lift_env_cfg.py`'s gripper
`open_command_expr`).

### 2a. Wadell sphericity — the single cleanest signal found

`tasks/franka/shape_observations.py:77-90` already computes, per shape, the
Wadell sphericity of each baked die's own convex hull (Wadell, H. 1935,
"Volume, shape, and roundness of quartz particles," *Journal of Geology*
43(3):250-280 — a standard, still-routinely-used particle-shape descriptor
in geology/powder-metallurgy/granular-mechanics; not independently
re-verified here beyond confirming this project's own prior citation of it,
since it is a 1935 print journal article predating arXiv, and the formula
itself — psi = pi^(1/3)(6V)^(2/3)/A, computed via `scipy.spatial.ConvexHull`
against the actual bevelled dice mesh, not an idealized Platonic-solid
approximation — is stated and load-bearing in this project's own code
comment, independently checkable there):

| shape | faces | Wadell sphericity psi | seeds w/ discovery (real ~16-18mm, Task 2) | seeds w/ discovery (48mm parity, Task 3.5 + gate, re-audited) |
|-------|-------|------------------------|----------------------------------------------|----------------------------------------------------------------|
| d8    | 8 (triangular, octahedron) | 0.889647 | 0/3 | 0/3 |
| d10   | 10 (kite, pentagonal trapezohedron) | 0.895933 | 0/3 | 0/3 |
| d12   | 12 (pentagonal, dodecahedron) | 0.928597 | 0/3 | 1/3 (full 8/8 within-seed) |
| d20   | 20 (triangular, icosahedron) | 0.952437 | untested at real size in this arc (d20's own real-size rung, `FrankaDieLiftJointStandardEnvCfg` @ 22mm, was not part of Task 2's 3-shape grid) | 2/3 (full 8/8 within-seed each) |

**This is a genuinely clean, monotonic pattern the prior BACKLOG framing
missed.** That entry stated "no clean roundness/face-count story — d20 is
the roundest shape and succeeds most decisively" as if that observation
argued *against* a roundness story; read against the actual sphericity
numbers it is the opposite — discovery rate increases monotonically with
sphericity across all four shapes at the controlled 48mm-parity anchor, with
an especially sharp jump between d10 (psi=0.896, null) and d12 (psi=0.929,
partial). **Caveat, stated plainly rather than overclaimed: this is n=4
shapes, not a controlled sweep across many shapes at fixed sphericity
increments — a real, previously-unnoticed correlation in this project's own
already-collected data, not proof of a causal sphericity threshold.** No
external literature source found that establishes a specific sphericity
threshold for parallel-jaw grasp discoverability (see §5's gap list) — this
finding is grounded entirely in this project's own re-derived measurements,
independently checked against the already-reported discovery grid rather
than taken on faith from either source alone.

### 2b. d10's specific extra disadvantages: anisotropy and no parallel face pairs

d10's native (unscaled) baked mesh bounding box, measured directly
(`FrankaDieLiftJointD10StandardEnvCfg`'s own docstring,
`tasks/franka/dice_lift_joint_env_cfg.py:554-568`): **16.3931 x 15.7156 x
14.9345 stage units — anisotropic**, elongation ratio (longest/shortest
axis) = 16.3931/14.9345 = **1.098 (~9.8% elongation)**. d8, d12, and d20 all
measure isotropic (equal-extent) native bboxes (`...D8StandardEnvCfg`:
15.1544 on all three axes; `...D12StandardEnvCfg`: 32.5160 on all three
axes; d20's own scale-per-mm derivation in `FrankaDieLiftJointStandardEnvCfg`
is likewise fit against a single "max dim," consistent with an isotropic
measured bbox). A pentagonal trapezohedron (d10's real solid-geometry class)
is genuinely not equal-extent the way the other three (octahedron,
dodecahedron, icosahedron — all face-transitive Platonic/Catalan-family
solids) are.

Separately, and not captured by the sphericity/anisotropy numbers above: a
pentagonal trapezohedron has **no pair of exactly parallel opposite faces**
(each of its 10 kite faces is offset by the antiprismatic twist between its
two pentagonal "poles") — unlike d8 (octahedron, dual of the cube, 4 pairs
of exactly parallel triangular faces), d12 (dodecahedron, 6 pairs of
parallel pentagonal faces), and d20 (icosahedron, 10 pairs of parallel
triangular faces), all of which are face-transitive with genuine opposite
parallel faces a parallel-jaw gripper can in principle close flush against.
This is basic, well-known solid geometry (not requiring external citation)
but was not previously stated anywhere in this project's own written
record. **This gives d10 two independent, compounding disadvantages beyond
what its sphericity number alone shows** (anisotropy AND no parallel-face
grasp affordance) — worth noting as a reason d10 may be the hardest of the
four even though its sphericity (0.8959) is marginally *higher* than d8's
(0.8896). d8, despite the lowest sphericity, does at least have the
parallel-face structure d10 lacks; whether that partially offsets its lower
sphericity is not resolvable from this project's own data (both are
currently null at every size tested) and would only become visible if one
of the two starts showing partial discovery under a new intervention while
the other does not.

### 2c. Gripper aperture-to-object-size ratio: measured, does not differentiate d8/d10 from d12/d20

The Franka Hand's per-finger open target is `open_command_expr =
{"panda_finger_.*": 0.04}` (`tasks/franka/lift_env_cfg.py:176`) — 0.04m per
finger from center, i.e. **0.08m (80mm) total maximum aperture**, matching
Franka's own published Hand specification (a standard, non-controversial
engineering fact about this specific end-effector, not requiring a
literature citation). Aperture-to-object ratio at this project's two tested
size regimes:

| size regime | object size | aperture/object ratio | outcome split |
|---|---|---|---|
| real-standard | 16mm (d8/d10) vs 18mm (d12) | 5.0x (d8/d10) vs 4.4x (d12) | all three null (d12 untested at its own real size in this arc, see caveat below) |
| 48mm parity | 48mm (all four shapes) | 1.67x (identical for all four) | d8/d10 null, d12/d20 partial |

At the controlled 48mm-parity anchor, every shape presents the *identical*
aperture-to-object ratio, yet outcomes still split cleanly along the
sphericity axis — this rules out absolute clearance margin/aperture ratio
as the differentiator between d8/d10 and d12/d20 specifically (it may still
matter for the *absolute* discovery rate at any given shape, just not for
explaining the shape-to-shape split). This corroborates, but is not
identical evidence to, this project's own AR4-era finding
([[grasp-mechanics-antipodal-vs-magnitude]], [[experiment-07-sphere-shrink]]):
doubling a sphere's clearance margin (28mm aperture vs 18mm object, shrunk
to 12mm) produced no improvement on that (different arm, different
gripper, different task) setup — flagged explicitly as a different
end-effector, not directly transferable evidence, only a consistent
direction.

---

## 3. Literature survey: geometry-specific sparse-discovery interventions

Every citation below was checked live this session via the arXiv API
(`export.arxiv.org/api/query?id_list=...`) or CrossRef
(`api.crossref.org/works/<doi>`), not taken from memory or a secondary
paraphrase, per this project's citation-verification-practice
([[citation-verification-practice]]).

### 3a. Demonstration-augmented RL — the strongest-grounded candidate, with a project-specific existence witness

**Rajeswaran, Kumar, Gupta, Vezzani, Schulman, Todorov, Levine, "Learning
Complex Dexterous Manipulation with Deep Reinforcement Learning and
Demonstrations," RSS 2018 (arXiv:1709.10087)** — confirmed real via the
arXiv API. Abstract, read directly: augmenting policy-gradient RL with a
small number of human demonstrations "significantly reduces sample
complexity" on high-dimensional dexterous-manipulation tasks that are hard
for RL to solve from scratch, and the resulting policies are "substantially
more robust." This is the standard citation for demonstration-augmented
policy gradient (DAPG) as a fix for exactly this class of problem: a
manipulation task where random exploration essentially never stumbles into
the rare state/action sequence a stable grasp requires.

**What makes this the top candidate for d8/d10 specifically is not the
citation alone but a real, already-existing project asset**: this project's
own scripted DiffIK controller (`scripts/dice_pick_demo.py`) already
achieves a genuine, verified grasp-and-lift on **d8 and d10 specifically**
(`kb/wiki/experiments/dice-pick-demo.md`: "4/5 PASS ... d8 240.9mm / d10
239.3mm ... z-gain, zero non-target drift" — d4 is the sole documented
failure, d8/d10/d12/d20 all pass). A known-feasible grasp trajectory for
exactly the two shapes that show zero RL discovery already exists in this
repo and has already been run in simulation. Unlike every citation in this
document, this is not an analogy from a different setting — it is direct
proof that a grasp *is* physically achievable for these two objects with
this gripper, at this size, in this simulator; the open question is purely
whether the RL policy can discover it, not whether one exists.

**A real caveat, not glossed over**: the scripted demo controls the arm via
task-space DiffIK (`docs/superpowers/specs/2026-07-11-joint-space-die-lift-research.md`
documents this project's Franka RL work uses joint-space `JointPositionActionCfg`
instead, `tasks/franka/dice_lift_joint_env_cfg.py:1-23`'s own module
docstring). A demonstration trajectory for BC/DAPG purposes does not need
the *controller* to match, though — since the scripted run executes inside
the same simulator, the *resultant joint-position time series* the DiffIK
controller actually produces can be logged directly and used as joint-space
demonstration targets, with no inverse-kinematics retargeting needed. This
project's own `tasks/franka/distillation.py` already implements exactly the
DAgger-style rollout-collection/BC-regression plumbing (`collect_rollout`,
`regress_on_pooled_batches`) this would reuse, built for the specialist→
distill pipeline — the mechanism is proven infra, not a new build.

### 3b. Geometry-ordered curriculum

**Wan, Geng, Liu, Shan, Yang, Yi, Wang, "UniDexGrasp++," ICCV 2023
(arXiv:2304.00464)** — confirmed real via the arXiv API this session (title,
authors, abstract match this project's prior citation exactly). Full text
fetched directly (`ar5iv.labs.arxiv.org/html/2304.00464`) and §4.4 read in
full (not previously done in this project's prior citation pass, which only
cited the paper for its GiGSL specialist/distill pattern, explicitly *not*
its GeoCurriculum mechanism — "This spec adopts GiGSL's specialist->distill
->iterate structure, not its own GeoCurriculum diversity-expansion
mechanism," `docs/superpowers/specs/2026-07-16-unified-multi-die-specialist-distillation-design.md:73-75`).

Read directly from §4.4: **GeoCurriculum** is a hierarchical task-space
curriculum that clusters training objects/poses by a *geometric feature*
(not object category/identity), starting training from the single task
nearest the population's own feature-center, then iteratively splitting
each cluster into finer sub-clusters and expanding the training population
outward. The paper's own stated motivation for using a geometric feature
rather than category labels is directly on point for this project's own
open question: object curriculum ordered by *identity/category* (the
mechanism the paper's own predecessor, UniDexGrasp, used) is a weaker
signal than one ordered by *geometric similarity*.

The predecessor paper's own result, quoted directly from UniDexGrasp++'s
own recap (§4.4, read in full): **UniDexGrasp (Xu, Wan, Zhang, et al., CVPR
2023, arXiv:2303.00938 — confirmed real via the arXiv API, abstract
confirms "object curriculum" as one of its own named contributions)
introduced a simpler object curriculum — single object, then several
similar objects from the same category, then the whole category, then all
categories — and reports this "improving the success rate of their
state-based policy from 31% to 74% on training set."** This figure is
UniDexGrasp++'s own characterization of UniDexGrasp's result (a secondary
citation, not independently re-verified against UniDexGrasp's own primary
text beyond confirming UniDexGrasp is real and does claim an "object
curriculum" contribution in its own abstract) — flagged as such rather than
presented as independently re-confirmed to the same standard as the
GeoCurriculum mechanism description above, which was read directly from
its own paper.

**Adaptation for this project, stated explicitly as an adaptation, not a
literal reproduction:** this project's discrete 4-shape setting does not
need GeoCurriculum's continuous hierarchical-clustering machinery (built
for a task space of thousands of objects with continuous pose variation).
The transferable idea is the *ordering principle* — start from the
population's geometric center/easiest member and expand outward by
geometric similarity, not by object identity/category — mapped onto this
project's own already-computed scalar geometry feature (Wadell sphericity,
§2a): warm-start d8/d10 training from the *already-converged, nearest-by-
sphericity* checkpoint (d12, psi=0.9286, the closest existing checkpoint to
d10's 0.8959/d8's 0.8896) rather than training from scratch, then fine-tune
at the real target geometry — the same "already-solved simpler case as a
starting point, then anneal toward the harder target" structure both
UniDexGrasp's object curriculum and UniDexGrasp++'s GeoCurriculum use, just
without the full hierarchical-clustering apparatus this project's small,
discrete shape set doesn't need.

### 3c. Exploration-noise/entropy tuning — weakest-fit candidate, included per the task brief but ranked last

**Li et al., "Improved PPO Optimization for Robotic Arm Grasping Trajectory
Planning and Real-Robot Migration," *Sensors* 2025, 25(17):5253, DOI
10.3390/s25175253** — independently re-verified this session via the
CrossRef API (title, journal, publication date all match this project's
prior citation, `kb/wiki/experiments/experiment-03-always-on-lift-height.md`,
exactly), separate from the fabricated citation that same prior research
pass caught and struck for a different claim
([[citation-verification-practice]]). This paper targets PPO's tendency to
collapse policy entropy around an already-converged, reward-sufficient
"safe" behavior, escaping via a simulated-annealing+PPO hybrid, and is
already this project's own load-bearing citation for the same mechanism in
a *different* task/shape ([[experiment-03-always-on-lift-height]], AR4
sphere-lift).

**Why this is ranked below §3a/§3b for the d8/d10 case specifically:**
`entropy_coef=0.006`/`init_noise_std=1.0` (§1) is the identical setting used
for every shape in this arc, including d12/d20, which *did* discover
grasps at the same 48mm-parity anchor. The same global exploration budget
that was sufficient for d12/d20 not producing discovery for d8/d10 is
corroborating (not dispositive) evidence that this is a geometry-specific
affordance problem rather than a generic PPO-exploration-budget shortfall —
a global entropy/noise increase is a *less* geometry-specific lever than
either §3a or §3b, which is precisely what the task brief's own framing
(distinguishing this question from "reward-shaping" and "hyperparameter
tuning" generically) asks this document to focus on. Not dismissed
entirely — flagged as the fallback if both higher-ranked candidates are
tried and fail (see §4).

### 3d. A real, available, but out-of-scope-for-this-document mechanism-level option

The AR4-era antipodal-grasp-check reward mechanism
([[grasp-mechanics-antipodal-vs-magnitude]], `grasp_contact`/
`antipodal_grasp_bonus`, Experiments 1/9/10) was never ported to the Franka
task — confirmed directly, `RewardsCfg`'s own docstring (§1) states this
explicitly as a deliberate prior difference. This is a real, available
lever (a contact-force-direction-gated grasp reward term, rather than the
current stock dense-reach + binary-lift-height reward every d8/d10/d12/d20
run has used unmodified) that has literally never been tried on any shape
in this Franka arc, d12/d20 included. It is flagged here for completeness
per the task brief's own ask ("this project's own already-proven
mechanisms... could be adapted here") but **not folded into this
document's ranked hypothesis in §4**: adding a new reward *term* is a
structural/mechanism-level change under this repo's own Tier 1 workflow
gate (CLAUDE.md's "Workflow" section: "a genuinely new reward *term*"
requires its own hypothesis and research grounding, not a training-
procedure adjustment this document's scope covers) — worth flagging to
whoever writes the next spec, not resolving here.

---

## 4. Proposed hypothesis and methodology (for a future spec to cite — NOT a spec itself)

Three candidate interventions, ranked by strength of grounding, each with
an explicit falsification condition. The task should retry d8/d10 at the
48mm-parity anchor specifically (`FrankaDieLiftJointD8BigEnvCfg`/
`...D10BigEnvCfg`, already built, Task 3.5's own env cfg classes) — the
already-controlled setting that isolates shape from absolute scale, per
this project's own asset-bisect precedent — not the real ~16-18mm size,
which reintroduces the scale confound Task 3.5 was built to close.

**H1 (primary): demonstration-augmented initialization from this project's
own already-proven scripted grasp.** Re-run `scripts/dice_pick_demo.py`'s
staged DiffIK sequence for d8/d10 (already known to pass, per §3a), log the
resultant joint-position trajectory (not the task-space commands), and use
it either to BC-pretrain the PPO policy's initial weights before training,
or as an auxiliary BC loss term mixed into the PPO objective (DAPG,
Rajeswaran et al. 2018, §3a) — reusing `tasks/franka/distillation.py`'s
existing rollout-collection/regression plumbing rather than building new
infra.

*Falsification:* if BC-pretrained/DAPG-augmented PPO on d8/d10 at the 48mm-
parity anchor (3 seeds, matching Task 3.5's own protocol) still shows 0/8
sustained-lift discovery in every seed, that falsifies "a scripted grasp
demonstration is sufficient to unlock discovery" and implicates either the
joint-space retargeting itself (the logged demonstration trajectory may not
be trackable/consistent under the RL policy's own `JointPositionActionCfg`
scale=0.5 formulation) or a deeper mechanism than demonstration-based
initialization can fix.

**H2 (secondary): geometry-ordered checkpoint warm-start.** Fine-tune d8/d10
training starting from the already-converged, nearest-by-sphericity d12
checkpoint (`joint-die-d12-big/seed123`, psi=0.9286) rather than from
scratch, at the real d8/d10 48mm-parity target geometry — this project's own
adaptation of UniDexGrasp++'s GeoCurriculum ordering principle (§3b),
explicit about being an adaptation rather than the paper's own hierarchical-
clustering mechanism.

*Falsification:* if fine-tuning from the d12 (or d20) checkpoint at the
d8/d10 target geometry still shows 0/8 in every seed, that falsifies
"policy-weight transfer from a nearby-sphericity shape is sufficient" and
would motivate a genuine geometry-interpolation path (an actual synthetic
intermediate-sphericity shape to anneal through) as the next escalation,
not attempted here.

**H3 (tertiary, weakest fit, test only if H1 and H2 both fail): exploration-
noise/entropy retuning**, raising `init_noise_std` above 1.0 and/or
`entropy_coef` above 0.006 specifically for d8/d10 runs, grounded in Li et
al. *Sensors* 2025 (§3c) and this project's own Experiment 3 precedent for
the same mechanism on a different task/shape.

*Falsification:* if raising entropy_coef/init_noise_std materially changes
d8/d10 discovery, that directly falsifies the "geometry-specific affordance,
not exploration budget" reading §3c argues for (corroborating, not proof) —
a real, useful negative result either way.

**Ordering rationale:** H1 is ranked first because it is the only candidate
grounded in a direct existence proof specific to these two shapes (a
verified feasible grasp trajectory already sits in this repo), not an
analogy from a different paper/setting. H2 is ranked second because it
reuses this project's own already-converged checkpoints and a real,
directly-read mechanism from the same paper family this experiment's
distillation phase already validated (GiGSL), at the cost of being an
adaptation rather than a literal reproduction. H3 is ranked last because
it is the least geometry-specific lever and the weakest-supported by this
project's own data (§3c's corroborating-not-dispositive reasoning).

---

## 5. Open risks / gaps, stated plainly

- **The sphericity-discovery correlation (§2a) is n=4, not a controlled
  sweep.** No literature source found (searched directly for this) that
  establishes sphericity, or any single scalar shape descriptor, as a
  validated predictor of parallel-jaw grasp-RL discoverability specifically
  — this project's own re-derived numbers are the only evidence for this
  pattern, and it should be treated as a genuinely new, project-specific
  empirical finding worth testing further (e.g. does a fifth shape at an
  intermediate sphericity also land at an intermediate discovery rate?),
  not as an externally-validated law.
- **d10's compounding disadvantages (anisotropy + no parallel face pairs,
  §2b) are not disentangled from its sphericity.** If H1/H2 succeed for d8
  but not d10 (or vice versa), that would be the first evidence separating
  "low sphericity" from "d10's specific extra geometric disadvantages" as
  the operative mechanism — not resolvable from currently available data.
- **No source found isolates "demonstration-augmented RL for a rigid,
  ungraspable-by-symmetric-antipodal-pair convex polyhedron with a
  parallel-jaw gripper" as its own studied case.** DAPG (§3a) is validated
  on higher-DoF anthropomorphic/dexterous hands (24-DoF in the paper's own
  setting), not a 1-DoF parallel-jaw gripper — the demonstration-
  augmentation *principle* is not shown to depend on hand DoF in the cited
  paper, but this is an extrapolation, flagged rather than assumed away.
- **GeoCurriculum's own quantitative result (§3b) is for a vision-based,
  thousands-of-objects setting**, not this project's ground-truth-state,
  four-shape setting — the adaptation in H2 borrows the *ordering
  principle* only, not a validated quantitative expectation for how much
  it should help at this much smaller scale.
- **This document does not re-derive or question the underlying reward
  function/action space** (Isaac Lab's own stock joint-space lift recipe,
  §1) — per the task brief's own scope, this is training-procedure/
  initialization research, not reward-design research; §3d flags the one
  real reward-mechanism-level option found but explicitly defers it.

---

## Related

[[unified-multi-die-specialist-distillation]] (the experiment this document
re-examines), [[reward-hacking-and-sparse-discoverability]] (the general
sparse-discoverability framing this document specializes to geometry),
[[grasp-mechanics-antipodal-vs-magnitude]] (source of the antipodal-check
infra flagged in §3d and the aperture-ratio precedent in §2c),
[[experiment-03-always-on-lift-height]] (source of the entropy-collapse
citation reused in §3c), [[dice-pick-demo]] (source of the scripted d8/d10
grasp trajectory §3a's H1 depends on),
`kb/wiki/concepts/citation-verification-practice.md` (the standing practice
this document's citation checks follow).
