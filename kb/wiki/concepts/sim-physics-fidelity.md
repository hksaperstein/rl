# Simulation physics fidelity: dt/decimation, collision offsets, EE-frame verification

## Why this theme exists

ROADMAP.md's item 9 (2026-07-09) is a dedicated physics-fidelity
verification pass on the AR4 mk5 cube task, requested directly rather than
arising from a training failure: smaller physics steps, correct collision
behavior, and an end-effector frame anchored at the gripper rather than the
wrist. It's a different kind of entry from most of this wiki's other
concept articles — not a reward or action-space hypothesis test, but a
correctness audit of the simulation substrate underneath every experiment
run so far. Three separable claims were checked, each independently
re-verified (junior-engineer executed, senior-engineer independently
re-checked with its own instrumentation rather than re-reading the junior's
report) — the same "verify with instrumentation, don't trust a self-report"
discipline this project applies to visual/behavioral claims (see
[[reach-grasp-lift-gap]]'s Experiment 16 contact-force finding), applied
here to non-visual, non-behavioral correctness claims instead.

## 1. dt/decimation: substep fidelity can change without touching the MDP interface

Commit `eb5f302` halved `sim.dt` (120Hz→240Hz for `env_cfg.py`/
`grasp_verify_env_cfg.py`; 100Hz→200Hz for `pickplace_mirror_env_cfg.py`)
and doubled `decimation` in lockstep, holding the control period constant
(1/60s and 0.02s respectively, unchanged).

The reason this is safe for an *already-trained* RL policy, and more
generally a reusable pattern for any future dt retune: Isaac Lab's control
period (`sim.dt * decimation`) is what defines the RL MDP's actual
interface — how often the policy observes and acts, and at what physical
time granularity its reward is computed. `sim.dt` alone (PhysX's internal
substep size) is strictly an implementation-fidelity knob for the physics
integration *inside* one control step: smaller substeps produce a more
accurate resolution of what happens physically inside that time slice
(contact resolution, integration error), but do not change how often the
policy is queried or how many control decisions it gets to make. Holding
`sim.dt * decimation` fixed while adjusting `sim.dt` is therefore a pure
fidelity improvement, orthogonal to anything a trained policy's weights
encode — no retraining is implied or required by this change alone. This
generalizes beyond AR4/cube: any manipulation task on a small/light object
where PhysX's per-substep contact resolution is coarse relative to object
scale is a candidate for the same fix, and the same safety argument (hold
the product constant) applies regardless of arm or task.

## 2. Collision contact/rest offset: PhysX's `-inf` auto-compute sentinel needs empirical, not assumed, verification

`objects_cfg.py`'s cube/rect_prism/sphere/wedge `_COLLISION_PROPS` leave
`contact_offset`/`rest_offset` at PhysX's auto-compute default. In the USD
schema this is literally the sentinel value `-inf` — meaning "let PhysX
resolve this at runtime based on object scale," with no public API exposing
what value it actually resolves to at runtime. For objects at ordinary
scale this default is usually fine and rarely worth checking. But this
project's props are all sub-cm scale (the cube is 12mm), and PhysX's
default offset heuristics are tuned around more ordinary object sizes — so
"probably fine" is not the same claim as "verified fine," and for objects
this small the two can diverge without any error or warning surfacing.

The verification method: a 2400Hz free-fall drop test, letting the object
fall under pure -9.81 m/s² acceleration (no collision engagement at all
until the very end) down to a 0.42mm gap above the resting surface, then
observing a clean single-substep arrest at a final rest height of exactly
0.006000m (the cube's own known half-extent — i.e. it rests flush, not
floating on a phantom offset gap or interpenetrating). This empirically
bounds the resolved offset to well under 0.5mm, negligible relative to the
cube's 6mm half-extent, and confirms no override is needed. The general
lesson: **PhysX's `-inf`/auto-compute pattern is opaque by design (no
readback API) — the only way to verify it behaves correctly for a small
object is a targeted physical test isolating the parameter in question**
(here, free-fall with no other forces present), not just reasoning about
what the default "should" do, and not just trusting that training runs
without visible interpenetration means the auto-computed value is small. If
future work introduces objects at a different scale again (smaller, or
much larger), this same drop-test method — not just re-reading this
finding — is the right way to re-check it, since the resolved value is a
function of the specific object's own geometry.

## 3. EE-frame verification: numeric cross-check *and* visual confirmation, as a reusable two-step pattern

`_EE_OFFSET=(0,0,0.036)` on `link_6` had already been corrected once before
this pass, from a wrong `0.09` value that had silently fed every grasp
experiment's `reaching_sphere`/proximity reward with a target point 5.4cm
from where the jaws actually meet (see
[[experiment-01-contact-sensor-grasp-reward]]'s "major finding" — the
retroactive recontextualization of four prior falsified grasp experiments).
This pass re-confirmed `0.036` is still correct, using two independent
checks rather than either alone:

- **Numeric**: comparing `ee_frame.data.target_pos_w` directly against the
  real jaw-link midpoint computed from `robot.data.body_pos_w`, finding
  <0.001mm residual.
- **Visual** (new this pass): enabling `debug_vis=True` in a live GUI run
  and confirming the rendered marker sits visibly between the jaw tips, not
  back at `link_6`/the wrist.

Neither check alone is as strong as both together. The numeric check alone
already caught the original 5.4cm bug, so it's not merely redundant — but a
numeric cross-check against the same kinematic chain that produced the
error in the first place cannot catch a class of bug where the *reference*
computation itself is wrong (e.g. a frame-orientation mixup that happens to
produce a numerically-plausible-looking but semantically wrong point).
Visual confirmation via `debug_vis` is a cheap, qualitatively different
check that catches that class of error by inspection instead. **This
two-step pattern — numeric cross-check plus live visual confirmation — is
the reusable methodology for verifying any future frame-offset claim** in
this project (a new arm's EE offset, a new gripper's jaw-pinch-point
offset, etc.), not just a one-off validation of this specific `0.036`
value.

## 4. Reachability claims need direct forward-kinematics verification, not just a stalled solver

A related methodological pattern, surfaced by ROADMAP.md items 7-8
(2026-07-09, the classical non-RL IK-reachability investigation that
directly precedes item 9's own pass — see [[reach-grasp-lift-gap]]'s
extended writeup of items 7-8 for the full narrative): **when an iterative
solver plateaus short of a target, that alone is evidence about the
solver, not evidence that the target is unreachable.** Item 7's rebuilt
`scripts/grasp_demo.py` (fixing a stale-joint-state bug and the same
unbounded-Cartesian-jump bug `oracle_rollout.py` had already found)
consistently plateaued well short of the cube's grasp pose even after
bounding the per-round IK step — a result that, taken alone, reads as
"this position may not be reachable," the fourth independent script/
mechanism to hit the same stall signature.

Item 8 refuted that reading with a direct, non-iterative measurement
instead of a fifth iterative attempt: `scripts/measure_reach_envelope.py`
computes the arm's reach envelope from forward kinematics alone (no IK
solver involved, so it cannot get stuck in a local minimum) and found the
cube target comfortably within reach (0.538m max vs. 0.344m needed).
The actual blocker was then isolated to the DLS Newton-step iteration
getting trapped in a local minimum independent of starting direction (a
geometrically-aimed seeded start barely helped) — resolved with a direct
forward-kinematics grid search (625 points, `scripts/ik_grid_search.py`)
that found a configuration within 3.5-6cm of the target, which a DLS
polish then closed to 3.648cm before plateauing again at the same
bit-exact fixed-point signature every prior attempt had shown.

The general, reusable lesson: **a local iterative solver's failure to
converge and a target's genuine unreachability are two different claims,
and only a direct, non-iterative measurement (forward kinematics here) can
tell them apart.** Treating a stalled solver as proof of unreachability —
without first checking via direct measurement — risks concluding a task is
impossible when it is actually just poorly solved. This is the same
"verify with instrumentation/direct measurement, don't trust a plausible-
looking failure" discipline as the other verification patterns in this
article (the collision-offset drop test, the EE-frame numeric+visual
check), applied here to a solver-convergence claim instead of a physics or
frame-offset claim.

## The settle-time-must-scale-with-dt bug class

While instrumenting `scripts/interactive_joint_demo.py` with gripper
contact sensors for the first time (see
[[reach-grasp-lift-gap]] for the finding itself — a contact-sensor-
confirmed zero-force grasp miss on the classical IK path), a plausible
first hypothesis for the miss was that the script's settle-time logic,
which counts raw physics substeps rather than going through
`env.step()`/`decimation`, had its real-world duration silently halved when
`sim.dt` was halved in this same pass (item 1 above) — i.e., a constant
"settle for N substeps" became half as much real settle time once each
substep got twice as short. This was tested and fixed (commit `e00dd11`,
settle time now derived from `env.physics_dt` rather than a raw substep
count) — it did **not** turn out to be the cause of the zero-contact-force
miss (see [[reach-grasp-lift-gap]]), but the bug class itself is real and
worth generalizing regardless of whether it explained this particular
symptom: **any script or component that counts "how long to wait" in raw
physics substeps, rather than deriving it from `sim.dt`/`physics_dt`, will
silently change its real-world wall-clock behavior whenever `sim.dt` is
retuned** — with no error, no crash, and no obviously wrong-looking
symptom, just a quietly shorter (or longer) settle/wait period than
intended. This is exactly the kind of latent coupling that a control-
period-preserving dt change (item 1) is supposed to be invisible to at the
RL-policy level, but is *not* automatically invisible to at the level of
raw-substep-counting utility code that sits outside the formal
`env.step()`/decimation loop — a distinction worth remembering the next
time `sim.dt` changes for any reason. Any future script that counts
substeps directly, not just `interactive_joint_demo.py`, should be
considered a candidate for this same bug until checked.

## A related gotcha, not physics but the same "verify, don't assume" discipline: camera frames can be silently flipped

Not a physics-fidelity bug itself, but the same category of silent,
easy-to-miss correctness defect this article otherwise documents (opaque
PhysX offsets, a raw-substep settle-time coupling). While building
Experiment 26's close-up camera (see
[[experiment-26-gripper-reintroduction]]), both
`scripts/graspgoal_closeup_video.py` and the existing
`scripts/touchgoal_closeup_video.py` were found saving every frame
vertically flipped — an OpenGL framebuffer row-order convention
(row-0-at-bottom) never corrected before writing to PNG/mp4, confirmed
empirically only once an unflipped render showed the ground grid at the top
of frame and sky at the bottom. Any future script that reads a camera
sensor's raw buffer directly (`camera.data.output["rgb"]` or equivalent)
rather than going through a library that already handles row order is a
candidate for this same bug until checked with exactly this kind of direct
visual sanity check, not assumed correct because the render "looks like an
image."

## Related concepts

[[reach-grasp-lift-gap]] — the zero-contact-force classical-IK grasp miss
this pass's contact-sensor instrumentation surfaced (item 9), and the full
items 7-8 stall/reachability narrative this article's own section 4
summarizes the reusable methodology from.

## Related follow-ups

ROADMAP.md item 9 is the source record for this whole pass; items 7-8
(preceding it) are the classical-IK reachability investigation this pass's
contact-sensor finding connects to, and the source of section 4's
reachability-verification methodology above.
