# AR4 sphere grasp: contact-sensor-based reward design

## Problem

Four independent reward/control hypotheses have failed to get the AR4 arm
to grasp+lift the sphere: lift-weight bump (no-op), additive grasp-closure
bonus (reward-hacked — closes beside not around the sphere), multiplicative
alignment gate (structurally sound but too sparse to discover), gripper
PD-gain rescale (no change). Full history in `ROADMAP.md`'s "grasp/lift
never emerges" entry. Every prior grasp-related reward term was a
*geometric proxy* for "is the object grasped" (distance + closure state) —
none of them can distinguish a real, physically-supported grasp from a
gripper that merely happens to be closed near the object. Per the user's
go-ahead, this experiment replaces the proxy with a **ground-truth contact
signal**: does the sphere actually register contact force against both
gripper fingers.

## Reference implementation (real, found in the installed Isaac Lab source)

`isaaclab_tasks/manager_based/manipulation/place/config/agibot/place_toy2box_rmp_rel_env_cfg.py`
adds, verbatim:

```python
self.scene.contact_grasp = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/right_.*_Pad_Link",
    update_period=0.05,
    history_length=6,
    debug_vis=True,
    filter_prim_paths_expr=["{ENV_REGEX_NS}/ToyTruck"],
)
```

and `isaaclab_tasks/manager_based/manipulation/place/mdp/observations.py`'s
`object_grasped()` consumes it:

```python
contact_force_grasp = env.scene["contact_grasp"].data.net_forces_w  # (N, 2, 3), one row per finger
contact_force_norm = torch.linalg.vector_norm(contact_force_grasp, dim=2)  # (N, 2)
both_fingers_force_ok = torch.all(contact_force_norm > force_threshold, dim=1)  # BOTH fingers must exceed threshold
grasped = torch.logical_and(pose_diff < diff_threshold, both_fingers_force_ok)
```

This is a real, shipped, working pattern for exactly our problem —
port/adapt it rather than designing from scratch (existing-research-first
principle).

## Design

### 1. Contact sensors

**Correction (found empirically during implementation, not anticipated by
the original design below): one wildcard sensor covering both jaw links
does not work.** A single `ContactSensorCfg` whose `prim_path` matches 2
bodies per env (`gripper_jaw[12]_link`, 32 bodies total across 16 envs)
cannot pair with `filter_prim_paths_expr=["{ENV_REGEX_NS}/Sphere"]`, which
only expands to 1 match per env (16 total) — PhysX's rigid contact view
requires the filter match count to equal the sensor body count, and fails
at construction with `Filter pattern ... did not match the correct number
of entries (expected 32, found 16)`. The reference this design cited
(`place_toy2box_rmp_rel_env_cfg.py`'s single wildcard sensor) apparently
gets away with this only because that robot has exactly one pad link per
side matched by its wildcard, not two — so the 1:1 body-to-filter ratio
happened to hold there by coincidence, not because a multi-body wildcard
sensor is generally supported against a single-instance filter target.

**Fix: two separate `ContactSensorCfg` entries, one per jaw link** (mirrors
the *other* real reference this repo's research already found —
`dexsuite/mdp/rewards.py`'s `contacts()`, which reads one
`ContactSensor` per finger link, e.g. `thumb_link_3_object_s`,
`index_link_3_object_s` — a working precedent for exactly this "one sensor
per body" structure):

```python
gripper_jaw1_contact = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/root_joint/gripper_jaw1_link",
    update_period=0.0,  # every physics step, matching this env's other sensors
    history_length=6,
    debug_vis=False,
    filter_prim_paths_expr=["{ENV_REGEX_NS}/Sphere"],
)
gripper_jaw2_contact = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/root_joint/gripper_jaw2_link",
    update_period=0.0,
    history_length=6,
    debug_vis=False,
    filter_prim_paths_expr=["{ENV_REGEX_NS}/Sphere"],
)
```

Each sensor now matches exactly 1 body per env against exactly 1 filtered
prim per env (16 == 16) — no count mismatch.

Also confirmed empirically during implementation: the robot's spawn config
(`tasks/ar4/robot_cfg.py`'s `AR4_MK5_CFG`) had
`activate_contact_sensors=False`, which must be `True` for any body on the
robot to support a `ContactSensor` at all (fails fast with a clear
"could not find any bodies with contact reporter API" error otherwise).

Prim path pattern confirmed correct against the real scene (the flat-
sibling-under-`root_joint` convention already confirmed correct for
`gripper_jaw1_link`/`gripper_jaw2_link` by the alignment-gate experiment's
smoke test) and the sphere's real prim path
(`SPHERE_CFG.prim_path = "{ENV_REGEX_NS}/Sphere"` in `objects_cfg.py`).

`filter_prim_paths_expr` restricts what counts as a "contact" to the
sphere specifically (not the table or other props) — this is the
mechanism that makes the sensor a genuine "is this object being gripped"
signal rather than "is anything touching the fingers."

### 2. Reward term

**Correction (also found empirically): `net_forces_w` is not filtered.**
Per Isaac Lab's own `ContactSensorData` docstring, `net_forces_w` is "the
sum of the normal contact forces acting on the sensor bodies" — from *any*
contact, not restricted to `filter_prim_paths_expr`'s target. Using it
would silently readmit exactly the failure mode this whole experiment
exists to avoid: a jaw touching the ground plane or another prop would
count as "grasping the sphere." The reference snippet's use of
`net_forces_w` despite specifying a filter is either a bug in that
reference or relies on that scene having no other possible contact source
for the sensor body — not safe to copy blindly here, where the gripper
jaws can plausibly touch the ground or other static props during
exploration. **The correctly-filtered field is `force_matrix_w`**, shape
`(num_envs, num_bodies, num_filters, 3)` — with the two-sensor fix above,
each sensor's `force_matrix_w` has `num_bodies=1, num_filters=1`.

Add a reward function (new, in `tasks/ar4/mdp.py` — this file doesn't
currently exist since prior experiments' versions were reverted; create
fresh) adapting `object_grasped`'s bilateral-force-threshold logic, reading
both jaw sensors:

```python
def contact_grasp_bonus(
    env: ManagerBasedRLEnv,
    force_threshold: float,
    jaw1_contact_cfg: SceneEntityCfg,
    jaw2_contact_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Dense bonus when both gripper fingers register real contact force
    against the sphere specifically - a ground-truth grasp signal
    (ContactSensor, filtered via force_matrix_w), replacing the geometric
    position/closure proxies every prior experiment this session used (see
    ROADMAP.md's grasp/lift history for why those failed: either
    reward-hackable via a loose distance check, or too sparse to discover
    via a tight alignment check). Adapted from isaaclab_tasks'
    manipulation/place/agibot task's object_grasped pattern (bilateral
    force-threshold check), corrected to use the filtered force_matrix_w
    field and one sensor per jaw (see this file's Design section for why).
    """
    jaw1_sensor: ContactSensor = env.scene[jaw1_contact_cfg.name]
    jaw2_sensor: ContactSensor = env.scene[jaw2_contact_cfg.name]
    jaw1_force = torch.linalg.vector_norm(jaw1_sensor.data.force_matrix_w, dim=-1).view(env.num_envs)
    jaw2_force = torch.linalg.vector_norm(jaw2_sensor.data.force_matrix_w, dim=-1).view(env.num_envs)
    both_fingers_contact = (jaw1_force > force_threshold) & (jaw2_force > force_threshold)
    return both_fingers_contact.float()
```

Register as a new `RewardsCfg` term, e.g. `grasp_contact` — pick a weight
using the same reasoning as prior experiments (this term is a clean
0/1 indicator per step, unlike the previous continuous dense terms, so
its scale is directly comparable to `lifting_sphere`'s weight=25.0 binary
indicator — consider a similar order of magnitude, e.g. 15-25, so it's a
meaningful signal without dwarfing `reaching_sphere`). Document the weight
choice and `force_threshold` choice (this needs a real number in Newtons —
the sphere is 0.01kg per `objects_cfg.py`'s `_MASS`, so its own weight is
~0.098N; a force_threshold meaningfully above sensor/contact noise but
well below anything that would crush/damage a real gripper is appropriate
— use judgment, but ground it in this object's actual mass rather than
copying the reference's threshold blindly, since that reference task's
object is presumably much heavier).

### 3. Single-variable experiment, standard scene

Use the **standard 4-object scene** (not the paused single-object camera
variant — that's a separate, still-open experiment) and **privileged-state
training** (not camera-based) — this experiment is testing whether a
better *grasp signal* fixes the problem, independent of scene complexity
or observation modality, so don't combine multiple untested variables.
Add ONLY the `grasp_contact` reward term; leave `reaching_sphere`,
`lifting_sphere`, `sphere_goal_tracking*` exactly as they are on `main`
right now.

## Verification plan

Same rigor as every prior experiment: smoke test
(`--num_envs 16 --max_iterations 2`), full run (`--num_envs 4096`, 1500
iterations, monitor `Episode_Reward/grasp_contact`,
`Episode_Reward/lifting_sphere`, `Episode_Termination/sphere_reached_goal`
via TensorBoard scalars pulled at the end), then real eval (`--episodes
10`) with frame-extracted video inspection of all 10 episodes — this time
specifically checking whether contact-sensor-confirmed "grasped" moments
in the reward log actually correspond to the sphere leaving the ground in
the video (the whole point of this signal is that it shouldn't be
hackable the way the geometric proxies were, but verify this empirically
rather than assuming the sensor is infallible).

If this also fails to move `lifting_sphere` off 0.0000, this is a fifth
falsified hypothesis for the reward/control axis — at that point, per
`superpowers:systematic-debugging` Phase 4.5, a fundamentally different
approach (the paused camera/single-object experiment, or a true curriculum
with staged reward phases, or accepting this repo's own physical setup
may need per-object mass/friction retuning) is likely warranted over a
sixth single-shot reward tweak — flag this back to the Principal/user
rather than continuing unilaterally.

## Parameter values (decided, not left to implementer judgment)

- `force_threshold = 0.05` N. The sphere's own weight is ~0.098N (0.01kg,
  `objects_cfg.py`'s `_MASS`, at g=9.81); 0.05N is about half that — enough
  above the simulator's contact noise floor (unlike the real-camera depth
  noise this repo also deals with, a contact sensor reports exactly 0N with
  no contact, so there's no equivalent "real noise floor" to measure
  empirically first) to require genuine load-bearing contact on each
  finger, without demanding a firmer grip than this light an object
  plausibly needs. Verify in the smoke test by printing the raw
  `net_forces_w` norms during a few random steps — if real contact events
  during exploration read far above or below this value, adjust before the
  full run rather than after.
- `grasp_contact` reward weight = `20.0`. Same order of magnitude as
  `lifting_sphere`'s `weight=25.0` (both are binary 0/1-per-step
  indicators, so directly comparable), placed slightly below it so lifting
  the sphere still matters more than merely gripping it.

## Calibration method correction (found empirically, Task 3)

The original plan for calibrating `force_threshold` reused
`scripts/grasp_demo.py`'s already-solved IK joint waypoints (computed for
the cube's fixed world position), relocating the sphere onto that same
position on the premise that "the arm's IK solution doesn't care which
object sits there." **This premise held for the joint angles themselves,
but the real run showed the end-effector never gets near either object at
all**: a diagnostic instrumented with the `ee_frame` sensor showed the
end-effector settling around world `(-0.28, -0.10, 0.0)` during
close/lift/hold — roughly the sphere task's own mirrored *target* region
(`CommandsCfg.ranges.pos_x=(-0.22,-0.18)`,
`pos_y=(-0.34,-0.28)`), nowhere near the `(0.20, 0.28, 0.009)` grasp
position 0.7m away. Whether this is a real, previously-unverified bug in
`grasp_demo.py`'s own IK-to-world-frame conversion for the 180°-rotated
base (that script's own docstring already flags its ~0.09m TCP offset as
"a rough estimate ... if the gripper doesn't land on the cube, this is the
first number to adjust" — i.e. it was never itself confirmed via a contact
sensor, only presumably by eye) or something specific to reusing those
joint angles inside `Ar4PickPlaceEnvCfg` was not chased down — re-deriving
IK is a separate, materially bigger problem than calibrating a reward
threshold, and orthogonal to what this experiment needs.

**Fix: skip IK/reach entirely.** Task 3 does not need a realistic
reach-grasp motion — it only needs one genuine physics-resolved contact
event between the gripper jaws and the sphere, to read real force numbers
off `force_matrix_w`. This repo already has a working, precedented pattern
for exactly this (`scripts/perception_calibration.py:74`,
`scripts/measure_planarity_residual.py:85`): hold the arm motionless and
teleport an object directly via `RigidObject.write_root_pose_to_sim(pose)`
(`pose` shape `(num_envs, 7)`, `[x, y, z, qw, qx, qy, qz]`, world frame).

Revised calibration approach:
1. Construct `Ar4PickPlaceEnvCfg`, `num_envs=1`, `env.reset()`. The arm's
   default reset pose is all-zero joint angles (same as `grasp_demo.py`'s
   `HOME_Q`), which `ActionsCfg`'s `JointPositionActionCfg` (absolute
   targets, `scale=1.0`) will hold exactly as long as the action commands
   `0.0` for all six arm joints every step — no arm motion needed at all.
2. Read the gripper's real jaw pinch-point world position once, straight
   from the sensor: `env.scene["ee_frame"].data.target_pos_w[0, 0]` (the
   `end_effector` target frame, already offset to the jaw pinch point by
   `_EE_OFFSET` — the same frame `reaching_sphere`'s reward already
   trusts in real training).
3. Every step, teleport the sphere to that exact position via
   `env.scene["sphere"].write_root_pose_to_sim(...)`, holding the arm at
   zero and driving the gripper joint action from open to closed over a
   short "close" phase, then holding closed for a "hold" phase.
4. Read both jaw sensors' `force_matrix_w` and call `contact_grasp_bonus`
   each step, same as originally planned — "open" phase is the negative
   control (expect ~0N), "hold" phase is the positive control (expect
   force above `force_threshold` on both jaws).

This isolates exactly what Task 3 needs to measure (does the sensor +
threshold correctly detect a real jaws-around-sphere contact) without
depending on the separate, unverified question of whether `grasp_demo.py`'s
IK actually reaches the cube — that question is out of scope for this
experiment and is not resolved by this fix.

## Major finding: `_EE_OFFSET` was wrong by 5.4cm (found while validating the fix above)

Running the teleport-based calibration script above still read **exactly
0.0N** contact force throughout, even with the sphere pinned to the
`ee_frame` sensor's own live position every step. Two further real issues,
found by direct measurement (not guessing):

1. **The arm does not hold its commanded pose rigidly.** Commanding all six
   arm joints to target `0.0` every step (matching the default reset pose)
   does not keep the arm static — under this robot's actuator gains
   (`stiffness=40`, `robot_cfg.py`), the arm visibly sags/settles for well
   over 100 steps after `env.reset()` before its pose stabilizes (measured:
   the gripper jaw midpoint's world Z dropped from `0.475` at the instant
   after reset to a stable `~0.14-0.19` by step ~100-150). Reading the
   pinch point once, immediately after `reset()`, captures a transient
   pre-settling value; the calibration script now re-reads
   `ee_frame.data.target_pos_w` **every step** instead of freezing one
   reading, and adds a 150-step settle phase (gripper open, no
   measurements taken) before the measured open/close/hold sequence.
2. **`_EE_OFFSET = (0.0, 0.0, 0.09)` (`tasks/ar4/pickplace_env_cfg.py`) is
   itself wrong.** Measured directly via
   `robot.data.body_pos_w` for `gripper_jaw1_link`/`gripper_jaw2_link`
   against `link_6`'s position at the arm's real settled pose: the true
   distance from `link_6` to the real jaw midpoint is **0.036m**, not
   0.09m — a 5.4cm error. `_EE_OFFSET`'s own code comment already flagged
   it as "empirically-tuned... a rough estimate" and `grasp_demo.py`'s
   docstring independently flagged its TCP offset as unmeasured; both
   turned out to be the same wrong number, now corrected to `0.036`.

   **This is a bigger deal than this experiment.** `_EE_OFFSET` feeds the
   `ee_frame` sensor's `end_effector` target frame, which `reaching_sphere`
   (`object_ee_distance`) has used as its proximity signal in **every**
   grasp experiment this session (lift-weight bump, dense grasp bonus,
   alignment gate, PD-gain rescale). A reward that maximizes proximity to
   a point 5.4cm away from where the jaws actually meet would let the
   policy converge to "high `reaching_sphere` reward" while the real
   gripper sits systematically offset from the sphere — a plausible
   deeper explanation for why grasping never emerged across all four
   prior falsified hypotheses, independent of whatever reward-shaping term
   was layered on top each time. This was corrected directly (`_EE_OFFSET`
   → `0.036`) as a prerequisite bug fix, the same way `activate_contact_sensors`
   was — not treated as a new hypothesis to test in isolation, since it is
   an objectively wrong measured constant, not a design choice. Flagged
   explicitly to the user as a significant, unplanned finding rather than
   silently folded in.

**Final calibration result (committed script, `scripts/calibrate_gripper_contact.py`,
after one more round of review-driven fixes — see below):**

```
[info] measured link_6->jaw-midpoint distance: 0.0360 m (ee_frame's _EE_OFFSET-based estimate: 0.0360 m)
far (sphere untouched, nowhere near gripper) force_norm: min=0.0000, max=0.0000 N (real negative control)
hold (gripper closed on sphere)               force_norm: min=27.3466, max=30.1718 N
hold reward==1.0 fraction: 120/120 (force_threshold=0.05)
```

An independent review of the first version of this script (which only had
`open`/`close`/`hold` phases, all with the sphere pinned to the pinch
point) correctly flagged two gaps: the `0.036` offset value rested on an
uncommitted, unreproducible diagnostic, and the `open` phase's own
`(expect ~0.0)` label was false — pinning the sphere to the geometric
center every step (needed to survive the arm's settling dynamics) puts it
inside the jaws' collision volume regardless of commanded aperture, so
`open` read large force too, not a real negative control. Both fixed: the
script now measures and prints the real `link_6`-to-jaw-midpoint distance
directly (confirming `_EE_OFFSET=0.036` against live geometry, not just a
throwaway script), and adds a genuine `far` phase where the sphere is left
untouched at its normal spawn position, comfortably outside contact range,
giving a real `0.0N` baseline in the same committed, re-runnable output.
`force_threshold=0.05` sits safely between that unambiguous true zero and
the unambiguous real contact reading (tens of Newtons), so it is kept
unchanged.
