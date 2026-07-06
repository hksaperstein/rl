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

### 1. Contact sensor

Add a `ContactSensorCfg` to `Ar4PickPlaceSceneCfg`
(`tasks/ar4/pickplace_env_cfg.py`) covering both gripper jaw links in one
sensor (mirroring the reference's `right_.*_Pad_Link` wildcard pattern):

```python
gripper_contact = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/root_joint/gripper_jaw[12]_link",
    update_period=0.0,  # every physics step, matching this env's other sensors
    history_length=6,
    debug_vis=False,
    filter_prim_paths_expr=["{ENV_REGEX_NS}/Sphere"],
)
```

Confirm the exact prim_path pattern against the real scene (the flat-
sibling-under-`root_joint` convention already confirmed correct for
`gripper_jaw1_link`/`gripper_jaw2_link` by the alignment-gate experiment's
smoke test) and the sphere's real prim path
(`SPHERE_CFG.prim_path = "{ENV_REGEX_NS}/Sphere"` in `objects_cfg.py`) —
don't assume, verify via the smoke test the same way the alignment
experiment did (a wrong path raises a clear, fast-failing error).

`filter_prim_paths_expr` restricts what counts as a "contact" to the
sphere specifically (not the table or other props) — this is the
mechanism that makes the sensor a genuine "is this object being gripped"
signal rather than "is anything touching the fingers."

### 2. Reward term

Add a reward function (new, in `tasks/ar4/mdp.py` — this file doesn't
currently exist since prior experiments' versions were reverted; create
fresh) adapting `object_grasped`'s bilateral-force-threshold logic:

```python
def contact_grasp_bonus(
    env: ManagerBasedRLEnv,
    force_threshold: float,
    contact_sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Dense bonus when both gripper fingers register real contact force
    against the sphere - a ground-truth grasp signal (ContactSensor),
    replacing the geometric position/closure proxies every prior
    experiment this session used (see ROADMAP.md's grasp/lift history for
    why those failed: either reward-hackable via a loose distance check,
    or too sparse to discover via a tight alignment check). Adapted from
    isaaclab_tasks' manipulation/place/agibot task's object_grasped
    pattern (net_forces_w bilateral force-threshold check).
    """
    contact_sensor: ContactSensor = env.scene[contact_sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w  # shape depends on body_ids matched by the prim_path wildcard
    force_norm = torch.linalg.vector_norm(net_forces, dim=-1)
    both_fingers_contact = torch.all(force_norm > force_threshold, dim=-1)
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
