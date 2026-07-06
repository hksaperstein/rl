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
