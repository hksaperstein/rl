# tasks/franka/mdp.py
"""Fresh MDP glue for the Franka Panda stock lift-recipe reproduction
(franka-panda-pivot, see CLAUDE.md's "Platform pivot (2026-07-09)" section).

Reads live simulated state (cube pose, end-effector FrameTransformer pose,
the randomized goal command) and delegates the actual reward arithmetic to
tasks/franka/lift_reward.py's pure-tensor functions - the same
math/observation/termination behavior as Isaac Lab's own official
isaaclab_tasks/manager_based/manipulation/lift/mdp/{rewards,observations,
terminations}.py, reimplemented from scratch here rather than imported,
per the pivot's "everything new, no reuse" instruction. This module
deliberately does NOT import `isaaclab_tasks.manager_based.manipulation.
lift.mdp` (Isaac Lab's own task-specific reference module) even though the
math it reproduces came from reading that module directly - only the fully
generic, robot/task-agnostic `isaaclab.envs.mdp` library (joint_pos_rel,
action_rate_l2, UniformPoseCommandCfg, BinaryJointPositionActionCfg,
reset_scene_to_default, root_height_below_minimum, etc.) is re-exported
below, the same kind of "official Isaac Lab library, not this repo's own
code" exception CLAUDE.md's pivot section explicitly grants for
`FRANKA_PANDA_CFG`.

Import this module only after an Isaac Sim/Isaac Lab AppLauncher has been
created.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor, FrameTransformer
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms

# Generic, robot/task-agnostic manager-term library (joint_pos_rel, last_action,
# action_rate_l2, joint_vel_l2, time_out, reset_scene_to_default,
# reset_root_state_uniform, root_height_below_minimum, modify_reward_weight,
# UniformPoseCommandCfg, JointPositionActionCfg, BinaryJointPositionActionCfg,
# DifferentialInverseKinematicsActionCfg, generated_commands, ...) - this is
# Isaac Lab's own installed-package library, not this repo's AR4-era code.
from isaaclab.envs.mdp import *  # noqa: F401, F403

from .antipodal_grasp_reward import antipodal_grasp_bonus_raw as _antipodal_grasp_bonus_raw_pure
from .distractor_observations import distractor_distance_summary as _distractor_distance_summary_pure
from .exploration_bonus_reward import (
    gripper_closure_attempt_bonus_correction as _gripper_closure_attempt_bonus_correction_pure,
)
from .exploration_bonus_reward import gripper_closure_attempt_bonus_raw as _gripper_closure_attempt_bonus_raw_pure
from .lift_reward import lifting_object_reward, object_goal_distance_reward, reaching_object_reward
from .shape_observations import (
    geometry_descriptor_broadcast,
    geometry_descriptor_per_env,
    shape_class_onehot,
    shape_class_onehot_per_env,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Default shape class for env cfgs that never set `die_shape_class` (the
# plain DexCube recipe, FrankaLiftEnvCfg itself, has no shape in
# {d8,d10,d12,d20} at all - this observation term is only meaningful for
# the die-specialist subclasses; d20 is used as an arbitrary but
# historically-dominant fallback so the base class still produces a
# well-formed (num_envs, 4) one-hot rather than erroring). Task 1 of
# docs/superpowers/plans/2026-07-16-unified-multi-die-specialist-
# distillation.md - see tasks/franka/shape_observations.py for the pure
# math and docs/superpowers/specs/2026-07-16-unified-multi-die-specialist-
# distillation-design.md for the spec.
_DEFAULT_SHAPE_CLASS = "d20"


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Cube position expressed in the robot's own root frame (observation term)."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w)
    return object_pos_b


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward term: reaching_object. 1 - tanh(dist/std) kernel on cube-to-EE distance."""
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cube_pos_w = object.data.root_pos_w
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    return reaching_object_reward(cube_pos_w, ee_pos_w, std)


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward term: lifting_object. Binary reward once the cube clears minimal_height."""
    object: RigidObject = env.scene[object_cfg.name]
    return lifting_object_reward(object.data.root_pos_w[:, 2], minimal_height)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward term: object_goal_tracking / object_goal_tracking_fine_grained.
    Gated 1 - tanh(dist/std) kernel on cube-to-goal distance, gated on lift height."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    return object_goal_distance_reward(
        object.data.root_pos_w, des_pos_w, object.data.root_pos_w[:, 2], minimal_height, std
    )


def object_shape_class_onehot(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Observation term: one-hot (num_envs, 4) over {d8, d10, d12, d20}.

    This is a per-env-cfg static property (which shape THIS training run's
    object is - a config-time choice, e.g. FrankaDieLiftJointD8StandardEnvCfg
    sets `self.die_shape_class = "d8"` in its own __post_init__), broadcast
    identically to every parallel env - NOT read off live simulated object
    state (the live sim only has raw mesh/pose, no semantic shape label).
    See tasks/franka/shape_observations.py's module docstring for the full
    scope rationale (every single-shape-per-env-cfg consumer of Task 1 uses
    this path unchanged).

    Task 5 extension (BACKLOG.md's 2026-07-19 controller decision "(b)
    single mixed-population env"): if the env cfg sets `die_shape_classes_
    per_env` (non-None - only FrankaDieLiftJointD12D20MixedEnvCfg does),
    each env's OWN shape class is computed as
    `env_index % len(die_shape_classes_per_env)` instead of broadcasting a
    single constant - still a config-time-known, deterministic function of
    env index, never read off live USD/spawner state (see
    tasks/franka/shape_observations.py's shape_class_onehot_per_env).
    """
    per_env_classes = getattr(env.cfg, "die_shape_classes_per_env", None)
    if per_env_classes is not None:
        return shape_class_onehot_per_env(per_env_classes, env.num_envs, device=env.device)
    shape_class = getattr(env.cfg, "die_shape_class", _DEFAULT_SHAPE_CLASS)
    return shape_class_onehot(shape_class, env.num_envs, device=env.device)


def object_geometry_descriptor(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Observation term: (num_envs, K) continuous geometry-descriptor
    feature (K=1, Wadell sphericity - see tasks/franka/shape_observations.py's
    module docstring for the exact formula/derivation), same per-env-cfg
    static-property/broadcast treatment (and same Task 5
    `die_shape_classes_per_env` branch) as object_shape_class_onehot above.
    """
    per_env_classes = getattr(env.cfg, "die_shape_classes_per_env", None)
    if per_env_classes is not None:
        return geometry_descriptor_per_env(per_env_classes, env.num_envs, device=env.device)
    shape_class = getattr(env.cfg, "die_shape_class", _DEFAULT_SHAPE_CLASS)
    return geometry_descriptor_broadcast(shape_class, env.num_envs, device=env.device)


def distractor_distance_summary(
    env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Observation term: (num_envs, 2) fixed-size, hard-zero-padded
    target-to-distractor distance summary (DexSinGrasp's own `d_t^S`
    mechanism, arXiv:2504.04516 §III-A Eq. 1 - see
    tasks/franka/distractor_observations.py's module docstring for the full
    design rationale and the exact zero-padding semantics).

    Reads THREE live scene entities - `object_cfg.name` (the target,
    default "object"), and the two always-present clutter-scene slots
    `env.scene["distractor_1"]`/`env.scene["distractor_2"]` (added by
    dice_lift_joint_env_cfg.py's FrankaDieLiftTargetSelectionSceneCfg, Task
    1) - plus `env.cfg.active_distractor_count` (0/1/2, a per-env-cfg
    constant set by each curriculum-stage env cfg's own __post_init__, see
    lift_env_cfg.py's FrankaLiftEnvCfg.active_distractor_count docstring),
    and delegates the actual distance/zero-padding computation to the pure
    function in tasks/franka/distractor_observations.py. Only meaningful
    for env cfgs that actually have distractor_1/distractor_2 scene
    entities (the 3 new target-selection curriculum-stage classes) - NOT
    wired into the shared base ObservationsCfg.PolicyCfg for exactly that
    reason (see TargetSelectionObservationsCfg in lift_env_cfg.py)."""
    target: RigidObject = env.scene[object_cfg.name]
    distractor_1: RigidObject = env.scene["distractor_1"]
    distractor_2: RigidObject = env.scene["distractor_2"]
    active_distractor_count = getattr(env.cfg, "active_distractor_count", 0)
    return _distractor_distance_summary_pure(
        target.data.root_pos_w[:, :3],
        distractor_1.data.root_pos_w[:, :3],
        distractor_2.data.root_pos_w[:, :3],
        active_distractor_count,
    )


# Object-dropping termination threshold, reused BY REFERENCE (not re-typed)
# from lift_env_cfg.py's own TerminationsCfg.object_dropping
# (`params={"minimum_height": -0.05, ...}`, lift_env_cfg.py:315) - both
# gripper_closure_attempt_bonus_correction's own independent is_last_step
# recomputation below and the real TerminationsCfg entry must use the exact
# same constant, per Task 2's own design note (avoid depending on
# env.termination_manager's own buffers having run yet, by recomputing the
# same predicate self-contained instead of importing lift_env_cfg's own
# TerminationsCfg, which would be a circular import - lift_env_cfg.py
# imports this module).
_OBJECT_DROPPING_MINIMUM_HEIGHT = -0.05


def gripper_closure_attempt_bonus(
    env: ManagerBasedRLEnv,
    w_attempt: float,
    k: float,
    std_gate: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    action_term_name: str = "gripper_action",
) -> torch.Tensor:
    """Reward term 1 (`gripper_closure_attempt_bonus` in ExplorationBonusRewardsCfg,
    lift_env_cfg.py): the raw, action-dependent F_t (GRM D=1 exploration
    bonus, Task 2 of docs/superpowers/plans/2026-07-19-exploration-bonus-
    grasp-discovery-implementation.md). Plain stateless function - this term
    has zero history dependence (see exploration_bonus_reward.py's own module
    docstring); the ONE new stateful mechanism this plan introduces is term
    2, GripperClosureAttemptBonusCorrection below.

    `env.action_manager.get_term(action_term_name).raw_actions` genuinely
    exists as a real (num_envs, 1)-shape tensor property (confirmed by direct
    source read, isaaclab/envs/mdp/actions/binary_joint_actions.py:63,108-109
    - `self._raw_actions = torch.zeros(self.num_envs, 1, ...)`, `raw_actions`
    property returns it), set via `self._raw_actions[:] = actions` inside
    `process_actions()` (binary_joint_actions.py:130), which
    ManagerBasedRLEnv.step() calls at the very start of step() via
    `action_manager.process_action()` (manager_based_rl_env.py:174) - i.e.
    this step's own action, unchanged for the rest of step(), so its value at
    reward-computation time is bit-identical to
    scripts/_diag_gripper_lowpass_check.py's own pre-step() `actions[:, -1]`
    capture (direct empirical cross-check, not just an attribute-existence
    check: `FrankaLiftPPORunnerCfg` sets no `clip_actions` field, defaulting
    to `None` per `RslRlOnPolicyRunnerCfg`'s own default, and
    `RslRlVecEnvWrapper.step()` only clamps actions when
    `self.clip_actions is not None` (isaaclab_rl/rsl_rl/vecenv_wrapper.py:
    152-155) - so no clipping occurs between the diagnostic script's captured
    value and what `.raw_actions` stores for this project's actual runner
    config, confirmed by direct source read of both files on the desktop
    Isaac Lab install, 2026-07-19).
    """
    gripper_term = env.action_manager.get_term(action_term_name)
    raw_gripper_action = gripper_term.raw_actions[:, 0]
    object_: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cube_pos_w = object_.data.root_pos_w
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    return _gripper_closure_attempt_bonus_raw_pure(raw_gripper_action, cube_pos_w, ee_pos_w, w_attempt, k, std_gate)


class GripperClosureAttemptBonusCorrection(ManagerTermBase):
    """Reward term 2 (`gripper_closure_attempt_bonus_correction` in
    ExplorationBonusRewardsCfg, lift_env_cfg.py): the GRM D=1 correction
    term, Correction_t = F'_t - F_t (see exploration_bonus_reward.py's own
    module docstring for the full derivation). Task 2's ONE new stateful
    mechanism (docs/superpowers/plans/2026-07-19-exploration-bonus-grasp-
    discovery-implementation.md) - owns one persistent per-env scalar buffer
    (`self._prev_raw`, the previous control step's raw bonus F_{t-1}),
    something no other reward term in this codebase needs.

    **Task 2 Step 1's mandatory empirical/source-read confirmation, recorded
    here per the plan's own requirement (not assumed from the plan's own
    draft, which turned out to have gotten one of the two comparisons wrong
    - see below):**

    (a) Class-based (`ManagerTermBase`-derived) reward terms are supported
    and reset automatically. Direct source read,
    isaaclab/managers/manager_base.py:412-414 (`_prepare_terms`): a
    class-valued `term_cfg.func` is instantiated once as
    `term_cfg.func = term_cfg.func(cfg=term_cfg, env=self._env)` - i.e.
    `__init__(self, cfg, env)` below receives the term's own `RewardTermCfg`
    as `cfg` (not a bare `ManagerTermBaseCfg`) and the live env as `env`.
    isaaclab/managers/reward_manager.py:244-246 (`_prepare_terms`) appends
    every such term to `self._class_term_cfgs`; reward_manager.py:122-124
    (`RewardManager.reset`) then calls `term_cfg.func.reset(env_ids=env_ids)`
    for every one of those terms whenever `RewardManager.reset(env_ids)` is
    invoked - which `ManagerBasedRLEnv._reset_idx` triggers for exactly the
    envs that terminated this step (per (b) below), matching this class's own
    "reset self._prev_raw to 0 on episode reset" requirement automatically,
    with no extra wiring needed beyond inheriting from ManagerTermBase.

    (b) `episode_length_buf`'s value at reward-computation time - direct
    source read, isaaclab/envs/manager_based_rl_env.py:154-236 (`step`):
    the literal order of operations is
    `episode_length_buf += 1` (line 202) -> `termination_manager.compute()`
    (line 205) -> `reward_manager.compute()` (line 209) -> only THEN,
    `_reset_idx()` for terminated envs (line 222), which sets
    `episode_length_buf[env_ids] = 0` (manager_based_rl_env.py:394). So at
    reward-computation time, `episode_length_buf` has ALREADY been
    incremented for this step but has NOT yet been reset - i.e. it holds
    `t + 1` (1-indexed step count), not `t` (0-indexed), for both boundaries:
    a fresh episode's first control step (`t=0` in the spec's own indexing)
    reads `episode_length_buf == 1`, not `0`; the episode's last control step
    (`t=N-1`) reads `episode_length_buf == N == max_episode_length`, not
    `N-1`. This project's own existing `time_out` termination term
    independently confirms the same post-increment convention already:
    `env.episode_length_buf >= env.max_episode_length`
    (isaaclab/envs/mdp/terminations.py:30-32), read by
    `termination_manager.compute()` one line before reward computation, using
    a plain `>=` with no `+1` - if `episode_length_buf` were pre-increment at
    that point, this project's OWN time_out term would already be off by one
    every single episode, which it is not (the asset-bisect/multi-die arc's
    episodes have always ended at exactly the configured length). **This
    directly contradicts the implementation plan's own draft formula**
    (`is_last_step = (episode_length_buf + 1 >= max_episode_length) | ...`,
    the plan's "Design notes" section) - that draft assumed a pre-increment
    read; the actual convention is post-increment, so the `+1` is dropped
    below (`is_last_step` uses `episode_length_buf >= max_episode_length`
    directly, no `+1`). **Cross-checked live in Isaac Sim** (not source-read
    alone, per the plan's own "empirical, not just source-read" requirement
    for exactly this kind of timing question): a throwaway diagnostic
    (`_empirical_episode_boundary_check.py`, random actions,
    `FrankaDieLiftJointD8BigExplorationBonusEnvCfg_PLAY`, 4 envs, run on a
    real cloud GPU instance 2026-07-20) monkeypatched
    `GripperClosureAttemptBonusCorrection.__call__` (via
    `functools.wraps`, to keep the wrapper's own introspectable signature
    intact - the manager's own `_resolve_common_term_cfg` signature check
    would otherwise reject an unwrapped `*args, **kwargs` wrapper, a real
    false-positive this diagnostic itself hit once before being fixed) to
    record `env.episode_length_buf[0]` at the exact moment reward
    computation reads it, across a full 250-step episode. Directly
    observed: **`episode_length_buf[0] == 1` at the episode's first control
    step (t=0)**; **`episode_length_buf[0] == 250 == max_episode_length` at
    the episode's last control step (t=N-1, step index 249)**, immediately
    followed by a reset (`truncated[0]=True`, `episode_length_buf[0]` reads
    back as `0` once `step()` returns, matching `977a748`'s own "post-step
    reads are already post-reset" finding for observations); and
    `correction_term._prev_raw[0] == 0.0` immediately after that boundary,
    confirming `reset()` fired automatically for the terminated env, per
    (a) above. (Incidental finding, not a bug: `raw_actions` is ALSO reset
    to `0.0` for a terminated env within the same `step()` call, before
    `step()` returns - matching `BinaryJointAction.reset()`'s own
    `self._raw_actions[env_ids] = 0.0`, isaaclab/envs/mdp/actions/
    binary_joint_actions.py:146 - which only affects a POST-step read, not
    reward computation itself, since reward computation reads
    `raw_actions` before this same-step reset happens.) All three numbers
    match this docstring's own conclusions exactly - no further correction
    needed.

    (c) `object_dropping`'s own `minimum_height=-0.05` constant
    (lift_env_cfg.py:315) is reused by reference via the module-level
    `_OBJECT_DROPPING_MINIMUM_HEIGHT` constant above, not re-typed, so this
    class's own self-contained `is_last_step` recomputation (deliberately
    NOT reading `env.termination_manager`'s own already-computed buffers,
    even though (b) shows those buffers ARE in fact already populated by
    reward-computation time - the plan's own "Design notes" section chooses
    the self-contained recomputation anyway, to stay correct regardless of
    manager-registration order in any future env cfg) can never silently
    diverge from the real termination term's own value.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._prev_raw = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._prev_raw[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        w_attempt: float,
        k: float,
        std_gate: float,
        gamma: float,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
        action_term_name: str = "gripper_action",
    ) -> torch.Tensor:
        gripper_term = env.action_manager.get_term(action_term_name)
        raw_gripper_action = gripper_term.raw_actions[:, 0]
        object_: RigidObject = env.scene[object_cfg.name]
        ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
        cube_pos_w = object_.data.root_pos_w
        ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
        F_t = _gripper_closure_attempt_bonus_raw_pure(raw_gripper_action, cube_pos_w, ee_pos_w, w_attempt, k, std_gate)

        # is_first_step / is_last_step per this class's own docstring finding
        # (b) above - episode_length_buf is POST-increment, PRE-reset at this
        # point in step(), so t=0 reads as 1 and t=N-1 reads as N (no `+1`).
        is_first_step = env.episode_length_buf == 1
        object_dropping = object_.data.root_pos_w[:, 2] < _OBJECT_DROPPING_MINIMUM_HEIGHT
        is_last_step = (env.episode_length_buf >= env.max_episode_length) | object_dropping

        correction = _gripper_closure_attempt_bonus_correction_pure(F_t, self._prev_raw, is_first_step, is_last_step, gamma)
        self._prev_raw = F_t
        return correction


def antipodal_grasp_bonus(
    env: ManagerBasedRLEnv,
    force_threshold: float,
    antipodal_cos_threshold: float,
    jaw1_contact_cfg: SceneEntityCfg,
    jaw2_contact_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """`antipodal_grasp_quality` reward term (AntipodalGraspRewardsCfg,
    lift_env_cfg.py) - Task 1 of docs/superpowers/plans/2026-07-20-d8-
    antipodal-grasp-quality-implementation.md; spec: docs/superpowers/specs/
    2026-07-20-d8-antipodal-grasp-quality-design.md. Thin wrapper: reads live
    `ContactSensor` state and delegates the actual force-closure/antipodal
    computation to antipodal_grasp_reward.antipodal_grasp_bonus_raw (pure
    torch, no isaaclab import, unit-tested in isolation via
    tests/test_franka_antipodal_grasp_reward.py). Ported (not imported, per
    the spec - tasks/franka/ never imports tasks/ar4/) from tasks/ar4/mdp.py:
    902-940's own antipodal_grasp_bonus - same math/signature, refit to this
    Franka scene's real friction coefficient (mu=0.5 ->
    antipodal_cos_threshold=-0.894427, NOT AR4's own mu=1.0 -> -0.7071 - see
    antipodal_grasp_reward.py's own module docstring for the full
    derivation).

    force_matrix_w reshape (`.view(env.num_envs, 3)`, identical to the AR4
    source's own reshape): empirically confirmed correct for this scene's
    own single-body/single-filter ContactSensorCfg wiring
    (FrankaDieLiftContactSceneCfg, dice_lift_joint_env_cfg.py) by Task 1's
    own required empirical check - see that class's own docstring for the
    exact observed shape/values, not assumed to transfer byte-for-byte from
    the AR4 source's own scene topology without re-confirming here.
    """
    jaw1_sensor: ContactSensor = env.scene[jaw1_contact_cfg.name]
    jaw2_sensor: ContactSensor = env.scene[jaw2_contact_cfg.name]
    jaw1_force_vec = jaw1_sensor.data.force_matrix_w.view(env.num_envs, 3)
    jaw2_force_vec = jaw2_sensor.data.force_matrix_w.view(env.num_envs, 3)
    return _antipodal_grasp_bonus_raw_pure(jaw1_force_vec, jaw2_force_vec, force_threshold, antipodal_cos_threshold)
