# AR4 sphere lift: monotonic potential-based reward shaping

## Problem

Four real attempts on the AR4 sphere lift sub-problem this session
(sparse-only, curriculum-gated dense, always-on dense, SA-PPO-style
learning-rate bump) all produce the identical outcome: the policy
reliably reaches and grips the sphere, then holds a completely static
pose for the rest of every episode. `lifting_sphere` never fires.

The most recent experiment (LR-bump,
`docs/superpowers/plans/2026-07-06-ar4-sphere-lift-lr-bump-report.md`)
is the most informative negative result yet: a substantial (10x),
sustained, confirmed-held learning-rate bump, applied at exactly the
point of behavioral entrenchment, produced **zero** measurable effect —
`lifting_sphere` read exactly `0.0` across the *entire* trajectory, with
not even the small noise blips two prior runs showed. If the failure
mode were pure PPO entropy collapse (the policy has simply stopped
exploring), a 10x optimizer perturbation should have produced at least
*some* visible behavioral variance. Getting nothing instead suggests a
different, more structural explanation: **the policy's value function may
have correctly learned that any movement away from the static grip is
genuinely worse expected return**, given how the current reward is
structured — not that it has stopped exploring toward a better option.

## Why this points at reward structure, not just optimization

The current reward (`tasks/ar4/pickplace_env_cfg.py`'s `RewardsCfg`) sums
several **independent** dense terms every step:
`reaching_sphere + lifting_sphere + grasp_contact + sphere_goal_tracking* +
lift_height_progress + action/joint penalties`. `grasp_contact` (weight
20.0) requires *sustained, bilateral* contact force above a threshold —
if a genuine lift attempt causes even a momentary reduction in one jaw's
contact force (plausible: repositioning the arm while holding an object
naturally perturbs grip pressure), `grasp_contact` drops to `0` for that
step, an instantaneous loss of up to 20 reward. Against that, `lifting_sphere`
and `lift_height_progress`'s partial credit for the same step is tiny
(this session's own measurements: real height gains on the order of
0.001-0.01mm per attempt). A policy that has already learned an accurate
value function would correctly avoid this trade — not because it isn't
exploring, but because the trade is genuinely bad under the current
reward's accounting. This is exactly the "reward conflict" hypothesis
this session's literature research flagged as plausible but unverified in
any specific citation (`docs/superpowers/specs/research/2026-07-06-lift-reward-literature-junior.md`,
Question 1) — the LR-bump's null result is real empirical evidence
*for* this hypothesis (it rules out pure exploration failure as a
sufficient explanation), even without a supporting citation.

## Decision: potential-based shaping with a monotonic (running-max) potential

Per the verified citation (Ng, Harada, Russell, ICML 1999, "Policy
Invariance Under Reward Transformations"): a shaping term of the form
`F(s, s') = γΦ(s') − Φ(s)` added to any base reward preserves the set of
optimal policies, for *any* potential function `Φ` over states. This
session's design uses a **monotonic, running-max** potential specifically
because it structurally eliminates the reward-conflict mechanism
identified above:

```
Φ(s_t) = max(Φ(s_0), Φ(s_1), ..., Φ(s_t))   where each Φ(s_i) is a raw,
                                              possibly-non-monotonic
                                              per-step progress signal
```

Because `Φ` is defined as a running maximum, it **never decreases** within
an episode. The shaped reward `γΦ(s') − Φ(s)` is therefore always `≥ 0`:
exactly `0` on any step that doesn't set a new best (including a step
where the raw signal *dropped*, e.g. a momentary contact-force dip during
a lift attempt), and strictly positive the first time a new milestone is
reached. This directly removes the incentive to avoid risky transitions —
there is no way for attempting a lift to ever score worse than standing
still, regardless of what happens to the raw sub-signals during the
attempt.

This is a legitimate application of the cited theorem: the theorem holds
for any potential function of the environment's state; treating the
running-max buffer as part of an augmented per-episode state (reset at
episode boundaries) keeps this a valid instance, not an ad-hoc
approximation. Reward ratcheting on a running-max potential is also a
documented practical technique in sparse-reward RL, not a novel invention
for this repo.

## Design

### 1. Replace, not add

This experiment **replaces** the current independently-additive terms —
`reaching_sphere`, `grasp_contact`, `lifting_sphere`,
`sphere_goal_tracking`, `sphere_goal_tracking_fine_grained`,
`lift_height_progress` — with a single new term,
`staged_potential_progress`. This is a bigger diff than any prior
single-term addition this session, but it is testing one coherent
hypothesis (potential-based combination vs. independent-additive
combination) — a partial version (adding this as a seventh term alongside
the existing six) would not test the hypothesis at all, since the old
terms' conflict-prone dynamics would still dominate.

**Not touched:** `action_rate`, `joint_vel` (small fixed penalties,
unrelated to this hypothesis), `sphere_reached_goal` (a *termination*, not
a reward — unaffected), `_EE_OFFSET`, `ContactSensorCfg` entries, every
observation/command, every PPO hyperparameter.

### Required import changes to `tasks/ar4/mdp.py`

The file's existing `TYPE_CHECKING` block (`RigidObject`, `ManagerBasedRLEnv`,
`ContactSensor`) needs `FrameTransformer` added (used as a type hint in the
new function below, same lazy-import pattern already established for the
other sim-only types):

```python
if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor, FrameTransformer
```

And a new **runtime** (not `TYPE_CHECKING`) import for the frame-transform
helper used in the goal-tracking sub-term:

```python
from isaaclab.utils.math import combine_frame_transforms
```

### 2. Raw (non-monotonic) per-step progress signal

`tasks/ar4/mdp.py`, new helper (not itself a `RewTerm` — called by the
stateful wrapper below):

```python
def _raw_lift_progress(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    jaw1_contact_cfg: SceneEntityCfg,
    jaw2_contact_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    command_name: str,
    reach_std: float,
    force_threshold: float,
    lift_minimal_height: float,
    goal_std: float,
) -> torch.Tensor:
    """Raw, per-step staged progress signal - NOT itself required to be
    monotonic (the monotonicity comes from the running-max wrapper that
    calls this). Weighted so each higher stage dominates once reached:
    reach (0.1) < grasp (0.2) < lift (0.3) < goal-tracking (0.4), max 1.0.
    """
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    reach_dist = torch.norm(object.data.root_pos_w - ee_frame.data.target_pos_w[:, 0, :], dim=-1)
    reach_term = 1.0 - torch.tanh(reach_dist / reach_std)

    jaw1_sensor: ContactSensor = env.scene[jaw1_contact_cfg.name]
    jaw2_sensor: ContactSensor = env.scene[jaw2_contact_cfg.name]
    jaw1_force = torch.linalg.vector_norm(jaw1_sensor.data.force_matrix_w, dim=-1).view(env.num_envs)
    jaw2_force = torch.linalg.vector_norm(jaw2_sensor.data.force_matrix_w, dim=-1).view(env.num_envs)
    grasp_term = ((jaw1_force > force_threshold) & (jaw2_force > force_threshold)).float()

    lift_term = (object.data.root_pos_w[:, 2] > lift_minimal_height).float()

    # The command is generated in the robot's root frame (UniformPoseCommandCfg
    # with asset_name="robot") - must transform to world frame before comparing
    # against the object's world-frame position, exactly matching
    # isaaclab_tasks' own object_goal_distance (the function sphere_goal_tracking
    # already used, before this experiment replaces it).
    robot: RigidObject = env.scene[robot_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3])
    goal_dist = torch.norm(object.data.root_pos_w - des_pos_w, dim=-1)
    goal_term = 1.0 - torch.tanh(goal_dist / goal_std)

    return 0.1 * reach_term + 0.2 * grasp_term + 0.3 * lift_term + 0.4 * goal_term
```

Requires a new import in `tasks/ar4/mdp.py`:
`from isaaclab.utils.math import combine_frame_transforms` (a runtime
import, not `TYPE_CHECKING` — it's a plain function, not a type
annotation, matching how `isaaclab_tasks`' own `object_goal_distance`
imports it in `isaaclab_tasks/manager_based/manipulation/lift/mdp/rewards.py`).

### 3. Stateful, monotonic potential-shaping wrapper

Also `tasks/ar4/mdp.py` — this is the actual `RewTerm` function. It needs
per-env state (the running max), attached directly to the `env` object
(a lazy-init pattern, since Isaac Lab's reward functions don't otherwise
carry state across calls):

```python
def staged_potential_progress(
    env: ManagerBasedRLEnv,
    gamma: float,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    jaw1_contact_cfg: SceneEntityCfg,
    jaw2_contact_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    command_name: str,
    reach_std: float,
    force_threshold: float,
    lift_minimal_height: float,
    goal_std: float,
) -> torch.Tensor:
    """Potential-based reward shaping (Ng, Harada, Russell, ICML 1999):
    F(s,s') = gamma*Phi(s') - Phi(s), where Phi is a per-episode running
    max of _raw_lift_progress. Because Phi never decreases within an
    episode, this reward is always >= 0 - a momentary drop in the raw
    signal (e.g. contact force dipping during a real lift attempt) cannot
    produce negative reward, structurally removing the incentive to avoid
    risky transitions that a plain additive combination of the same
    sub-signals would create. See
    docs/superpowers/specs/2026-07-06-ar4-sphere-lift-potential-shaping-design.md.
    """
    if not hasattr(env, "_lift_potential_max"):
        env._lift_potential_max = torch.zeros(env.num_envs, device=env.device)

    raw = _raw_lift_progress(
        env, object_cfg, ee_frame_cfg, jaw1_contact_cfg, jaw2_contact_cfg, robot_cfg,
        command_name, reach_std, force_threshold, lift_minimal_height, goal_std,
    )
    prev_potential = env._lift_potential_max.clone()
    new_potential = torch.maximum(env._lift_potential_max, raw)
    env._lift_potential_max = new_potential

    return gamma * new_potential - prev_potential


def reset_lift_potential(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Event term (mode="reset"): zero the running-max potential buffer
    for resetting envs, so a new episode starts with no carried-over
    progress. Must be registered in EventCfg alongside reset_scene_to_default.
    """
    if not hasattr(env, "_lift_potential_max"):
        env._lift_potential_max = torch.zeros(env.num_envs, device=env.device)
    env._lift_potential_max[env_ids] = 0.0
```

### 4. Registration

`RewardsCfg` (replacing the six terms named in section 1):

```python
staged_potential_progress = RewTerm(
    func=ar4_mdp.staged_potential_progress,
    weight=25.0,
    params={
        "gamma": 0.98,  # must match Ar4PickPlacePPORunnerCfg.algorithm.gamma exactly
        "object_cfg": SceneEntityCfg("sphere"),
        "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        "jaw1_contact_cfg": SceneEntityCfg("gripper_jaw1_contact"),
        "jaw2_contact_cfg": SceneEntityCfg("gripper_jaw2_contact"),
        "robot_cfg": SceneEntityCfg("robot"),
        "command_name": "object_pose",
        "reach_std": 0.1,
        "force_threshold": 0.05,
        "lift_minimal_height": 0.03,
        "goal_std": 0.3,
    },
)
```

`EventCfg` (new event, alongside the existing `reset_all`/
`reset_sphere_position`):

```python
reset_lift_potential = EventTerm(func=ar4_mdp.reset_lift_potential, mode="reset")
```

All parameter values reuse this repo's own already-established constants
(`reach_std=0.1` matches the old `reaching_sphere`'s `std`;
`force_threshold=0.05` matches `grasp_contact`'s calibrated value;
`lift_minimal_height=0.03` matches `lifting_sphere`'s threshold;
`goal_std=0.3` matches `sphere_goal_tracking`'s coarse `std`) — no new
untested constants introduced beyond the staging weights (`0.1/0.2/0.3/0.4`)
and the overall term weight (`25.0`, matching the largest prior single
term's scale).

### 5. `gamma` must match the PPO config exactly

`Ar4PickPlacePPORunnerCfg.algorithm.gamma = 0.98`
(`tasks/ar4/agents/rsl_rl_ppo_cfg.py`). The theorem's invariance guarantee
specifically requires the shaping function's discount to match the base
MDP's discount — passing a different value would still produce *a* valid
reward, but not one with the policy-invariance property motivating this
whole experiment.

## Verification plan

Smoke test first (`--num_envs 16 --max_iterations 2`) to confirm the new
term, the stateful buffer, and the reset event all wire up without error
— specifically watch for any error about `env._lift_potential_max` not
existing on first access from a *reward* call before any *event* call has
run (the lazy-init `hasattr` guard in both functions should prevent this,
but confirm no exception either way). Then the full 1500-iteration run
(`--num_envs 4096`), monitoring `Episode_Reward/staged_potential_progress`,
`Episode_Reward/lifting_sphere` (removed from `RewardsCfg` but still
computable/loggable if useful — otherwise infer lift success from the
termination), `Episode_Termination/sphere_reached_goal`. Then real eval
(10 episodes) with frame-extracted video inspection, same rigor as every
prior experiment.

If this also fails to produce real lifting, this is the fifth real
attempt on this sub-problem (sparse-only, curriculum-gated dense,
always-on dense, LR-bump, potential-shaping) — per
`superpowers:systematic-debugging` Phase 4.5, flag back to the user rather
than attempting a sixth reward/optimization tweak. At that point the
remaining candidates are genuinely architectural (hierarchical
reach-then-grasp-policy split) or physical (though grip force has already
been independently ruled out as the bottleneck).
