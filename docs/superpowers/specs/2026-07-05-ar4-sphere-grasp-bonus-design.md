# AR4 sphere grasp-bonus reward design

## Problem

Per `ROADMAP.md`'s "AR4 sphere pick-and-place: grasp/lift never emerges"
follow-up: two full 1500-iteration training runs (baseline, and a
lift-weight bump 15.0->25.0) both converge `reaching_sphere` to ~0.92-0.93
while `lifting_sphere`/`sphere_goal_tracking`/`sphere_reached_goal` stay at
0.0000. Eval video confirms the gripper reaches the sphere and holds a
static, open pose for the rest of the episode - it never attempts to
close. Literature review (delegated, citation-verified) concluded this is
an exploration problem, not a reward-scale problem: nothing in the current
reward (`reaching_sphere`, `lifting_sphere`, `sphere_goal_tracking*`) gives
any gradient for ever closing the gripper near the object, so random
exploration essentially never stumbles into a full accidental grasp+lift
across a 4096-env x 1500-iteration budget.

Recommended next steps, in priority order: (a) contact-based reward
(needs a new `ContactSensorCfg`, not yet in this config), (b) curriculum
(reach-only -> reach+close-gripper bonus -> full lift), (c) hierarchical
frozen-reach-then-grasp.

## Decision

Implement a dense "grasp bonus" reward term - a static, always-on version
of (b)'s middle stage, rather than building full curriculum
phase-scheduling infrastructure. This is the same pattern Isaac Lab's own
`manipulation/cabinet` task already uses in production
(`mdp.grasp_handle`, `cabinet_env_cfg.py:213`): a dense term that rewards
closing the gripper only when the end-effector is within a threshold
distance of the target, weight 0.5 in that task. It gives the policy
partial credit for the *precursor* to lifting (closing near the object)
that's currently entirely unrewarded, without needing new sensors or
multi-phase training infrastructure. This is the minimal, lowest-risk
change that directly targets the diagnosed exploration gap, and reuses an
existing, working Isaac Lab pattern rather than inventing one from
scratch (per this repo's existing-research-first practice).

Not chosen: (a) requires new `ContactSensorCfg` infra this config doesn't
have yet - larger surface area for a first attempt. (c) is explicitly the
literature's last resort. Both remain fallbacks if this doesn't work.

## Design

New reward function (no existing Isaac Lab mdp module has an
object-agnostic version of `grasp_handle` - it hardcodes the cabinet's
`"cabinet_frame"` scene key), added to a new local
`tasks/ar4/mdp.py`:

```python
def grasp_object_bonus(
    env: ManagerBasedRLEnv,
    threshold: float,
    open_joint_pos: float,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    gripper_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Dense bonus for closing the gripper while near an object.

    Bootstraps grasp-attempt exploration: adapted from isaaclab_tasks'
    manipulation/cabinet task's grasp_handle reward (identical
    is_close * sum(open - current) pattern), generalized from a fixed
    cabinet-handle frame to any object_cfg/ee_frame_cfg pair.
    """
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    gripper_joint_pos = env.scene[gripper_asset_cfg.name].data.joint_pos[:, gripper_asset_cfg.joint_ids]

    distance = torch.norm(object.data.root_pos_w - ee_pos_w, dim=-1)
    is_close = distance <= threshold

    return is_close * torch.sum(open_joint_pos - gripper_joint_pos, dim=-1)
```

Registered in `RewardsCfg` (`tasks/ar4/pickplace_env_cfg.py`) as:

```python
grasp_sphere = RewTerm(
    func=mdp_local.grasp_object_bonus,
    weight=10.0,
    params={
        "threshold": 0.04,
        "open_joint_pos": GRIPPER_OPEN_POS,
        "object_cfg": SceneEntityCfg("sphere"),
        "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        "gripper_asset_cfg": SceneEntityCfg("robot", joint_names=GRIPPER_JOINT_NAMES),
    },
)
```

Parameter reasoning:
- `threshold=0.04`: sphere radius is 0.009m (`objects_cfg.py`); cabinet
  uses 0.03 for its handle. A slightly larger threshold gives margin for
  approach imprecision while still requiring the EE to be genuinely near
  the sphere, not just in the same general area.
- `weight=10.0`: the raw term's max value is tiny by construction (gripper
  stroke is only `GRIPPER_OPEN_POS - GRIPPER_CLOSED_POS = 0.014` per
  joint, 2 joints -> max raw value 0.028, versus Franka's much larger
  finger stroke in the cabinet task). Weight chosen so the term's max
  per-step contribution (~0.28) is a real but secondary signal relative to
  `reaching_sphere` (max 1.0/step) and well below `lifting_sphere` (25.0
  on success) - a nudge toward closing, not a reward strong enough to
  create its own local optimum (e.g. hovering and repeatedly
  opening/closing for free reward without ever lifting).
- This is the only reward change in this experiment - `lifting_sphere`,
  `sphere_goal_tracking*`, `reaching_sphere` are left exactly as the prior
  bounded-fallback run left them (weight 25.0), per this repo's
  established practice of changing one variable per experiment.

## Verification plan

Same procedure as the prior sphere-retargeting plan's Task 2/3 (see
`docs/superpowers/plans/2026-07-05-ar4-sphere-pickplace-implementation.md`):
smoke test (`--num_envs 16 --max_iterations 2`), then full run
(`--num_envs 4096`, headless, monitored via TensorBoard until
`Episode_Termination/sphere_reached_goal` climbs off zero or clearly
plateaus at 0), then real eval (`--episodes 10`) with frame-extracted
video inspection of all 10 episodes - not exit codes or reward curves
alone. Decision gate: at least 8/10 episodes show the sphere reliably
placed at the target region.

If this doesn't move `lifting_sphere` off 0.0000 either, that's a second
falsified hypothesis after the lift-weight-bump fallback - per
`superpowers:systematic-debugging`'s Phase 4.5, don't attempt a third
tweak; escalate to (a) contact sensor or (c) hierarchical instead, and
say so explicitly rather than continuing open-ended tuning.
