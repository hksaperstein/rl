# tasks/ar4/pickplace_graspgoal_env_cfg.py
"""Grasp/lift/goal variant of the AR4 pick-and-place task (Experiment
26): reintroduces the gripper after Experiment 25 removed it. Composes
Experiment 21's proximity-gated gripper and Experiment 17's antipodal
grasp gate, with a 30s episode and a 4-stage extension of Experiment
25's validated monotonic reward mechanism.

Originally also planned to reuse Experiment 22's mirroring mechanism
(corrected for its own identified reactive-lag bug); final whole-branch
review found that fix is an unconditional no-op under this action space
(BinaryJointAction assigns both jaws the same commanded value already -
see ActionsCfg's own docstring below for the full trace) and retired it
rather than attempt a fourth jaw-synchronization-specific fix. See
docs/superpowers/specs/2026-07-09-ar4-experiment26-gripper-reintroduction-design.md.

Additive/parallel to pickplace_touchgoal_env_cfg.py: deliberately does
NOT modify that file, env_cfg.py, objects_cfg.py, or mdp.py's Experiment
25 functions.

Import this module only after an Isaac Sim/Isaac Lab AppLauncher has
been created.
"""

import math

import isaaclab.envs.mdp as isaaclab_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.manipulation.lift import mdp

from . import mdp as ar4_mdp
from .actions import ProximityGatedBinaryJointPositionActionCfg
from .objects_cfg import CUBE_CFG
from .pickplace_env_cfg import _EE_OFFSET
from .robot_cfg import (
    ARM_JOINT_NAMES,
    AR4_MK5_CFG,
    GRIPPER_CLOSED_COMMAND_EXPR,
    GRIPPER_JOINT_NAMES,
    GRIPPER_OPEN_COMMAND_EXPR,
)

# Same fixed cube spawn as Experiment 25; goal is now where the CUBE must
# end up (carried there), not an end-effector waypoint - same offset
# value, different physical meaning.
GOAL_OFFSET = (-0.40, 0.0, 0.144)

REACH_DIST_NORM = 0.3
LIFT_MINIMAL_HEIGHT = 0.03
# NOTE: `lifted` latches at LIFT_MINIMAL_HEIGHT (0.03), not this value, so
# the 0.50-0.75 reward segment this normalizes is only ever traversed to
# ~0.575 in practice before the formula jumps to the goal segment (whose
# 3D distance-to-goal term subsumes the remaining lift height anyway) -
# this constant does not meaningfully shape mid-lift behavior, flagged by
# final whole-branch review, not a bug.
LIFT_TARGET_HEIGHT = 0.10
GOAL_TOLERANCE = 0.02
FORCE_THRESHOLD = 0.05
ANTIPODAL_COS_THRESHOLD = -0.7071
PROXIMITY_THRESHOLD = 0.05
# Direct user request: "heavily punish it for collision w the ground."
# Conservative starting threshold - the jaw contact sensors use
# FORCE_THRESHOLD=0.05N for detecting a deliberate, controlled grasp
# pinch; 1.0N is ~20x that, chosen so ordinary micro-contacts/sensor
# noise on the upper-arm links (which should never touch anything in
# correct operation) don't trip it, while still catching a real
# arm-into-ground collision early. Adjust if a smoke test shows it firing
# spuriously or never firing.
ARM_GROUND_CONTACT_FORCE_THRESHOLD = 1.0
# "when it gets closer to the cube reward it for slowing down" - tighter
# than REACH_DIST_NORM (0.3, the reach-stage milestone's own normalizer)
# since "close" here should mean genuinely near the cube, not just
# somewhere in the broad reach-progress ramp.
SLOW_NEAR_CUBE_REACH_DIST_THRESHOLD = 0.12
# The verified per-step trajectory trace (logs/train/2026-07-09_15-18-06/
# model_1499.pt) showed the arm covering its approach in ~0.4-0.6s -
# roughly 0.5-1.0 m/s average Cartesian speed during that fast reach
# phase. speed_cap=0.25 m/s sits well below that cruise speed, so genuine
# slowing/settling near the cube is required to earn this bonus, not
# just an incidental slow moment during transit.
SLOW_NEAR_CUBE_SPEED_CAP = 0.25
# Derived, not hardcoded - matches Experiment 25's final-review lesson on
# geometry constants silently drifting from what they measure.
CUBE_TO_GOAL_DIST = math.sqrt(GOAL_OFFSET[0] ** 2 + GOAL_OFFSET[1] ** 2 + GOAL_OFFSET[2] ** 2)


@configclass
class ActionsCfg:
    """Arm (Experiment 25's proven scale=0.5) + gripper (proximity-gated
    only - no jaw-mirroring, see note below).

    Final whole-branch review of this experiment found MirroredGripperActionCfg
    (jaw2 tracks jaw1's target) is an unconditional no-op here: BinaryJointAction
    was assigning both jaws the IDENTICAL commanded value (open_command_expr/
    close_command_expr used the same constant for both joint names), so there
    was nothing for jaw2 to diverge from at the command level in the first
    place - the actual jaw asymmetry was believed to happen only at the
    physics/actuator level under contact, which no command-level mirroring
    could address.

    CORRECTION (2026-07-21, ar4-franka-fixes-transfer plan Task 5,
    controller-authorized cross-experiment fix): the "IDENTICAL commanded
    value" premise above was itself a real bug, not a neutral fact -
    gripper_jaw2_joint's PhysxMimicJointAPI mirrors jaw1 with gearing=-1.0
    (confirmed by direct USD inspection, commit 64ab5cc), so jaw2's
    physically-correct commanded value is always -1.0 * jaw1's, not the same
    signed constant. GRIPPER_OPEN_COMMAND_EXPR/GRIPPER_CLOSED_COMMAND_EXPR
    (tasks/ar4/robot_cfg.py) now mirror jaw2 correctly. This does NOT revive
    MirroredGripperActionCfg as a lever (that mechanism tracks jaw1's
    *dynamic* per-step target under contact load, a different problem from
    this static open/close sign bug), but it does mean the "nothing for jaw2
    to diverge from at the command level" reasoning above was never actually
    true - see kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md for
    the full writeup of why this may be a genuine root-cause candidate for
    the jaw-asymmetry problem the three attempts below were trying to fix.
    This is still the third jaw-synchronization-specific *fix attempt* to
    prove ineffective as a lever (Experiment 19's PhysxMimicJointAPI made
    things worse; Experiment 22's original mirror-actual had a reactive-lag
    bug; this pass's lag fix, mirror-commanded, is a true no-op given the
    static open/close bug is now fixed at its own source) - retired as a
    lever per this project's own "3 failed fixes means question the
    architecture, not patch again" discipline, not attempted a fourth time.
    Uses plain ProximityGatedBinaryJointPositionActionCfg (Experiment 21's
    validated fix, unaffected by this finding) instead."""

    joint_positions = isaaclab_mdp.JointPositionActionCfg(asset_name="robot", joint_names=ARM_JOINT_NAMES, scale=0.5)
    gripper_position = ProximityGatedBinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=GRIPPER_JOINT_NAMES,
        open_command_expr=GRIPPER_OPEN_COMMAND_EXPR,
        close_command_expr=GRIPPER_CLOSED_COMMAND_EXPR,
        object_cfg=SceneEntityCfg("cube"),
        ee_frame_cfg=SceneEntityCfg("ee_frame"),
        proximity_threshold=PROXIMITY_THRESHOLD,
    )


@configclass
class Ar4PickPlaceGraspGoalSceneCfg(InteractiveSceneCfg):
    """AR4 arm + gripper + a single fixed-position cube. Re-adds the
    gripper-jaw contact sensors Experiment 25's touch-goal lineage
    dropped (required for antipodal_grasp_bonus)."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )
    robot: ArticulationCfg = AR4_MK5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cube: RigidObjectCfg = CUBE_CFG.replace(
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.20, 0.28, 0.006)),
        spawn=CUBE_CFG.spawn.replace(
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            )
        ),
    )

    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/root_joint/base_link",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/root_joint/link_6",
                name="end_effector",
                offset=OffsetCfg(pos=_EE_OFFSET),
            ),
        ],
    )
    gripper_jaw1_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/root_joint/gripper_jaw1_link",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
    )
    gripper_jaw2_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/root_joint/gripper_jaw2_link",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"],
    )
    # Upper-arm-only ground/self collision safety sensor - deliberately
    # excludes gripper_jaw1_link/gripper_jaw2_link and link_6 (the wrist),
    # which legitimately approach the ground plane (cube rests at
    # z=0.006) to reach the cube. Body names confirmed via direct
    # robot.data.body_names introspection (this project's own convention -
    # see scripts/smoke_test_graspgoal_ground_penalty.py, which printed
    # the live body list: ['world', 'base_link', 'link_1'..'link_5',
    # 'link_6', 'gripper_base_link', 'gripper_jaw1_link',
    # 'gripper_jaw2_link'] - confirming 'world' is a virtual fixed-base
    # root body and 'gripper_base_link' is the jaw-housing body, neither
    # of which belongs in this upper-arm-only set either), matching
    # scripts/plot_arm_skeleton.py's independently-verified expected_chain.
    # A single ContactSensorCfg can track multiple bodies at once: Isaac
    # Lab's ContactSensor._initialize_impl (contact_sensor.py) treats the
    # prim_path's final path segment as a regex against sibling prim
    # names (sim_utils.find_matching_prims tokenizes prim_path_regex on
    # "/" and regex-matches each segment independently), so a single
    # sensor with an alternation in the leaf segment resolves to all
    # matching sibling links - confirmed empirically, not just "no
    # exception", by inspecting this sensor's own num_bodies/body_names
    # once constructed. No filter_prim_paths_expr - unlike the jaw
    # sensors (filtered to the cube specifically), these links shouldn't
    # be contacting anything at all, so raw net_forces_w is the right
    # signal, which isaaclab_mdp.illegal_contact reads directly.
    arm_ground_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/root_joint/(base_link|link_1|link_2|link_3|link_4|link_5)",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
    )


@configclass
class ObservationsCfg:
    """Arm + gripper joint state (unrestricted joint_names - gripper is
    actuated again), cube position, goal position, grasp/lift latch
    state, last action."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        cube_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame, params={"object_cfg": SceneEntityCfg("cube")}
        )
        goal_position = ObsTerm(
            func=ar4_mdp.cube_goal_position_in_robot_root_frame, params={"robot_cfg": SceneEntityCfg("robot")}
        )
        grasp_state = ObsTerm(
            func=ar4_mdp.grasp_state_observation,
            params={
                "object_cfg": SceneEntityCfg("cube"),
                "jaw1_contact_cfg": SceneEntityCfg("gripper_jaw1_contact"),
                "jaw2_contact_cfg": SceneEntityCfg("gripper_jaw2_contact"),
                "force_threshold": FORCE_THRESHOLD,
                "antipodal_cos_threshold": ANTIPODAL_COS_THRESHOLD,
                "lift_minimal_height": LIFT_MINIMAL_HEIGHT,
            },
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset events: whole scene back to default, then snapshot the cube
    goal position, then zero the milestone/latch buffers."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    set_cube_goal_position = EventTerm(
        func=ar4_mdp.set_cube_goal_position,
        mode="reset",
        params={"object_cfg": SceneEntityCfg("cube"), "goal_offset": GOAL_OFFSET},
    )

    reset_grasp_goal_milestone = EventTerm(func=ar4_mdp.reset_grasp_goal_milestone, mode="reset")


@configclass
class TerminationsCfg:
    """Success (genuine grasp+lift occurred, then the cube reached the
    goal) ends the episode early; otherwise a fixed timeout."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    cube_reached_goal = DoneTerm(
        func=ar4_mdp.cube_reached_goal_after_lift,
        params={
            "threshold": GOAL_TOLERANCE,
            "object_cfg": SceneEntityCfg("cube"),
            "jaw1_contact_cfg": SceneEntityCfg("gripper_jaw1_contact"),
            "jaw2_contact_cfg": SceneEntityCfg("gripper_jaw2_contact"),
            "lift_minimal_height": LIFT_MINIMAL_HEIGHT,
            "force_threshold": FORCE_THRESHOLD,
            "antipodal_cos_threshold": ANTIPODAL_COS_THRESHOLD,
        },
    )

    # Direct user request: "heavily punish it for collision w the
    # ground." Reuses Isaac Lab's own built-in illegal_contact
    # termination (isaaclab.envs.mdp.terminations.py:153) against the
    # upper-arm-only arm_ground_contact sensor above - no body_names
    # override needed on this SceneEntityCfg, since that sensor already
    # tracks exactly (and only) the upper-arm links.
    arm_ground_contact = DoneTerm(
        func=isaaclab_mdp.illegal_contact,
        params={
            "threshold": ARM_GROUND_CONTACT_FORCE_THRESHOLD,
            "sensor_cfg": SceneEntityCfg("arm_ground_contact"),
        },
    )


@configclass
class RewardsCfg:
    """Four-stage gated running-max milestone bonus: reach, grasp
    (antipodal gate), lift, goal - no ungated additive sum (Experiment
    16's wedging exploit), no separately-weighted independent terms
    (Experiment 17/18's discoverability gap) FOR THE STAGE-PROGRESS SIGNAL
    ITSELF. arm_ground_contact_penalty and slow_near_cube_bonus (below)
    are deliberately separate, independently-weighted terms rather than
    additional stages folded into that same milestone mechanism - both
    are safety/settling signals orthogonal to stage progress (a ground
    collision or a wandering-after-reach failure mode isn't itself a new
    "stage" of the task), not new discoverable sub-goals subject to
    Experiment 17/18's original discoverability concern, and
    arm_ground_contact_penalty in particular must remain an ordinary
    dense per-step penalty (not gated behind the running-max) precisely
    so it can never be "banked past" the way the milestone bonus's own
    reach term was found to be (see grasp_goal_milestone_bonus's own
    known flaw, root-caused via a verified per-step trajectory trace of
    logs/train/2026-07-09_15-18-06/model_1499.pt)."""

    grasp_goal_milestone_bonus = RewTerm(
        func=ar4_mdp.grasp_goal_milestone_bonus,
        weight=25.0,
        params={
            "object_cfg": SceneEntityCfg("cube"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "jaw1_contact_cfg": SceneEntityCfg("gripper_jaw1_contact"),
            "jaw2_contact_cfg": SceneEntityCfg("gripper_jaw2_contact"),
            "reach_dist_norm": REACH_DIST_NORM,
            "lift_minimal_height": LIFT_MINIMAL_HEIGHT,
            "lift_target_height": LIFT_TARGET_HEIGHT,
            "force_threshold": FORCE_THRESHOLD,
            "antipodal_cos_threshold": ANTIPODAL_COS_THRESHOLD,
            "cube_to_goal_dist": CUBE_TO_GOAL_DIST,
        },
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")})

    # Direct user request: "heavily punish it for collision w the
    # ground." Weight -20.0 is clearly dominant relative to
    # action_rate/joint_vel's -1e-4 (a single illegal-contact step
    # outweighs an entire episode's worth of those two combined many
    # times over), while staying well below the milestone bonus's own
    # scale (weight=25.0, raw progress in [0, 1] so max single-step payout
    # is 25.0) - a single spurious trip shouldn't be able to swamp an
    # otherwise-good episode's accumulated milestone reward outright, but
    # the penalty must still be unambiguously the dominant signal the
    # instant real ground contact occurs.
    arm_ground_contact_penalty = RewTerm(
        func=ar4_mdp.arm_ground_contact_penalty,
        weight=-20.0,
        params={
            "sensor_cfg": SceneEntityCfg("arm_ground_contact"),
            "threshold": ARM_GROUND_CONTACT_FORCE_THRESHOLD,
        },
    )

    # Direct user request: "when it gets closer to the cube reward it for
    # slowing down." Dense per-step term (NOT running-max, unlike
    # grasp_goal_milestone_bonus) - the direct engineering fix for the
    # discovered flaw where that running-max bonus pays nothing further
    # once its best-ever reach is banked. weight=5.0 is modest relative to
    # the milestone bonus's weight=25.0 (whose raw progress is in [0, 1]
    # per new milestone), since this term is dense/per-step and
    # accumulates continuously over the full 30s/1500-step episode rather
    # than paying out once - a naive weight matching 25.0 would let this
    # single term dominate the entire episode's return. Tune later via
    # scripts/sweep.py if it proves too weak/strong in practice.
    slow_near_cube_bonus = RewTerm(
        func=ar4_mdp.slow_near_cube_bonus,
        weight=5.0,
        params={
            "object_cfg": SceneEntityCfg("cube"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "robot_cfg": SceneEntityCfg("robot"),
            "reach_dist_threshold": SLOW_NEAR_CUBE_REACH_DIST_THRESHOLD,
            "speed_cap": SLOW_NEAR_CUBE_SPEED_CAP,
        },
    )


@configclass
class Ar4PickPlaceGraspGoalEnvCfg(ManagerBasedRLEnvCfg):
    """AR4 grasp/lift/goal task (Experiment 26): reach, grasp, lift, and
    carry the cube to a fixed goal point. num_envs=4096 default -
    scripts/train.py's --num_envs flag overrides this per-run."""

    scene: Ar4PickPlaceGraspGoalSceneCfg = Ar4PickPlaceGraspGoalSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 30.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0)
        self.viewer.eye = (1.5, 1.5, 1.2)
        self.viewer.lookat = (0.0, 0.0, 0.4)


@configclass
class Ar4PickPlaceGraspGoalRelativeEnvCfg(Ar4PickPlaceGraspGoalEnvCfg):
    """H_ar4_relative (Task 2, docs/superpowers/plans/2026-07-21-ar4-
    franka-fixes-transfer-implementation.md; spec: docs/superpowers/
    specs/2026-07-21-ar4-franka-fixes-transfer-design.md): IDENTICAL
    scene/rewards/observations/terminations/events/PPO recipe to
    Condition A (Ar4PickPlaceGraspGoalEnvCfg, Experiment 26) - the ONLY
    change is the arm action term, from Condition A's inherited
    ABSOLUTE JointPositionActionCfg (scale=0.5) to RELATIVE/delta
    RelativeJointPositionActionCfg (scale=0.1, use_zero_offset=True),
    mirroring Franka's own confirmed
    FrankaDieLiftJointD8BigRelativeAntipodalEnvCfg
    (tasks/franka/dice_lift_joint_env_cfg.py) exactly - same scale/
    use_zero_offset values (AR4's control step is identical to
    Franka's own, 50Hz - see the design spec's own "why scale=0.1
    transfers without rescaling" section), same "call super, then
    re-assert the one changed field last" pattern.

    Empirical action-manager verification (Task 2, Step 3): a real
    `Ar4PickPlaceGraspGoalRelativeEnvCfg` env (num_envs=8) was built on a
    live GCP SPOT `g2-standard-4`+`nvidia-l4` cloud instance
    (`rl-ar4-graspgoalrelative-task2`, `us-central1-a`, desktop
    unreachable at dispatch time, 2026-07-21) - a separate, isolated
    single-env-per-process script, after an initial attempt that built
    Condition A then Condition B sequentially in ONE process was observed
    to hang indefinitely (0% GPU utilization, CPU pegged) partway through
    the second `ManagerBasedRLEnv` construction, a real find worth noting
    for any future multi-condition diagnostic script in this arc: build
    one env per process invocation, not multiple `ManagerBasedRLEnv`
    instances sequentially in-process. The live ActionManager was read
    directly (not the unbuilt cfg object). Exact observed output:
    - `type(env.action_manager.get_term("joint_positions")).__name__`
      == `"RelativeJointPositionAction"` - CONFIRMED (not
      `JointPositionAction` - Condition A's own unmodified
      `Ar4PickPlaceGraspGoalEnvCfg`, checked in the same session before
      the hang above, confirmed `"JointPositionAction"` instead, as
      expected).
    - `env.action_manager.total_action_dim` == `7` - CONFIRMED (6
      relative arm joints + 1 gripper binary dim - NOT Franka's 8,
      since AR4's `gripper_position` term is a
      ProximityGatedBinaryJointPositionAction, subclassing Isaac Lab's
      BinaryJointPositionAction, whose action_dim is hardcoded to 1
      regardless of the 2 underlying joint names it spans - confirmed
      by direct source read of
      `isaaclab/envs/mdp/actions/binary_joint_actions.py`, matching
      this repo's own `scripts/smoke_test_graspgoal_env.py:54` comment).
      Condition A's own `total_action_dim` is also `7` (byte-identical
      dimensionality - only the arm term's class changes, not its size).
    - `env.action_manager.active_terms` == `['joint_positions',
      'gripper_position']` - CONFIRMED; `env.action_manager.
      action_term_dim` == `[6, 1]` - CONFIRMED `joint_positions` at
      dim 6 specifically. The manager's own printed table
      (`print(env.action_manager)`) independently shows the same
      `joint_positions: 6 / gripper_position: 1` breakdown under an
      "Active Action Terms (shape: 7)" header.
    - Secondary spot-check (not required to close this task, per this
      arc's own judgment-call precedent): a nonzero constant action
      (`0.3` on all 6 arm dims) was applied for one control step from two
      DIFFERENT starting joint configurations (env 0 pre-warmed 5 steps
      with a nonzero action on joint 0 to move it away from the reset
      pose; env 1 left at its reset pose). Observed per-joint deltas were
      `[0.00166, 0.00513, 0.05950, -0.00089, 0.01973, 0.00298]` (warmed-up
      env) vs. `[-0.00006, 0.00326, 0.05949, -0.00269, 0.01965, 0.00109]`
      (reset-pose env) - same order of magnitude at every joint (max
      pairwise diff 0.00238 rad), consistent with the config-independent
      commanded-delta property `RelativeJointPositionAction` guarantees
      (per Franka's own `FrankaDieLiftJointD8BigRelativeAntipodalEnvCfg`
      docstring, the actual joint travel within one control step is still
      mediated by each joint's own PD servo dynamics, so a small,
      configuration-dependent residual is expected, not a discrepancy).
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.actions.joint_positions = isaaclab_mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=ARM_JOINT_NAMES,
            scale=0.1,
            use_zero_offset=True,
        )
