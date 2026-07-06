# AR4 sphere mirror-goal scene + stillness penalty Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new, parallel AR4 pick-and-place task — sphere only (no
cube/rect_prism/wedge), spawn randomized across the full workspace, goal
always on the opposite side of the robot from the spawn — with a
corrected (undiscounted) staged milestone reward and a new grasp-gated
stillness penalty, then run it and record whether the sphere finally
lifts.

**Architecture:** New files `tasks/ar4/pickplace_mirror_env_cfg.py`
(scene/observations/events/rewards/terminations/env-cfg) and new
functions appended to `tasks/ar4/mdp.py` (mirrored-goal event +
observation + termination, corrected milestone-bonus reward, stillness
penalty). `scripts/train.py`/`scripts/eval_loop.py` get a new `--mirror`
flag to select the new env cfg. No existing file's existing symbols are
modified.

**Tech Stack:** Isaac Lab / Isaac Sim (`ManagerBasedRLEnv`), PyTorch,
`rsl_rl` PPO, existing AR4 mk5 task code in `tasks/ar4/`.

## Global Constraints

- Design spec: `docs/superpowers/specs/2026-07-06-ar4-sphere-mirror-scene-design.md`
  — read it. In particular, do not reintroduce a `gamma`-discounted delta
  for the new milestone-bonus reward; that discount is a real bug found
  in the existing `staged_potential_progress` (see the design's "Why
  now" section for the derivation) — `staged_milestone_bonus` must have
  no `gamma` param at all.
- Always invoke via `/home/saps/IsaacLab/isaaclab.sh -p scripts/<script>.py`
  from the repo root — never plain `python` for anything importing
  `isaaclab`.
- Do **not** modify `tasks/ar4/env_cfg.py` (`Ar4SceneCfg`),
  `tasks/ar4/objects_cfg.py`, `tasks/ar4/pickplace_env_cfg.py`, or any
  existing function in `tasks/ar4/mdp.py` (`contact_grasp_bonus`,
  `_raw_lift_progress`, `staged_potential_progress`,
  `reset_lift_potential`) — those stay exactly as they are; this plan
  only adds new files/functions alongside them.
- `_EE_OFFSET = (0.0, 0.0, 0.036)` and the `ContactSensorCfg` fields
  (`update_period=0.0`, `history_length=6`, `debug_vis=False`,
  `filter_prim_paths_expr=["{ENV_REGEX_NS}/Sphere"]`) are copied verbatim
  from `Ar4PickPlaceSceneCfg` in `pickplace_env_cfg.py`.
- Workspace bounds (env-local/robot-relative, confirmed against
  `robot_cfg.py`'s `AR4_MK5_CFG` having no explicit `init_state.pos` —
  defaults to `(0,0,0)`, i.e. the robot base sits at each env's own local
  origin): x in `(-0.30, 0.30)`, y in `(0.10, 0.45)`. Goal z range:
  `(0.0, 0.02)` (matches the existing command's `pos_z` range).
  `still_bound=0.005`, `patience_steps=25`, stillness-penalty
  `weight=-2.0`, milestone-bonus `weight=25.0` — decided values, not
  placeholders.
- `num_envs=4096` for the real training run (user-specified explicitly,
  overriding the smaller `num_envs=16` default used by the existing
  single-object-scene precedent, which is for a different, camera-based
  experiment).
- Verification standard: real evidence over proxies. Read the actual
  TensorBoard scalars and look at the actual eval video frames before
  concluding anything. Isaac Sim's clean-shutdown path sometimes hides
  final console prints ("Training complete.") due to a benign
  stdout-buffering quirk seen repeatedly this session — verify success
  via file artifacts (checkpoints, dumped `env.yaml`, TensorBoard event
  files) rather than assuming failure from missing console text alone.

---

### Task 1: Mirrored-goal mechanism — event, observation, termination

**Files:**
- Modify: `tasks/ar4/mdp.py` (add imports, add `set_mirrored_goal`,
  `mirrored_target_position_in_robot_root_frame`,
  `object_reached_mirrored_goal`)
- Create: `tasks/ar4/pickplace_mirror_env_cfg.py` (scene,
  `ObservationsCfg`, `EventCfg`'s first three terms, `TerminationsCfg` —
  `RewardsCfg` and the full env cfg class come in Task 2)

**Interfaces:**
- Produces: `set_mirrored_goal(env, env_ids, sphere_cfg, goal_y_range,
  goal_z_range) -> None` (EventTerm function, `mode="reset"`), writes
  `env._target_pos_w` (shape `(num_envs, 3)`, world frame).
- Produces: `mirrored_target_position_in_robot_root_frame(env,
  robot_cfg) -> torch.Tensor`, shape `(num_envs, 3)`.
- Produces: `object_reached_mirrored_goal(env, threshold, object_cfg) ->
  torch.Tensor`, shape `(num_envs,)`, dtype bool.
- Produces: `Ar4PickPlaceMirrorSceneCfg(InteractiveSceneCfg)` with fields
  `ground`, `light`, `robot`, `sphere`, `ee_frame`,
  `gripper_jaw1_contact`, `gripper_jaw2_contact`.
- Consumes: nothing from other tasks (this is the first task).

- [ ] **Step 1: Add imports to `tasks/ar4/mdp.py`**

Change the top of `tasks/ar4/mdp.py` from:

```python
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms
```

to:

```python
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, sample_uniform, subtract_frame_transforms
```

- [ ] **Step 2: Append `set_mirrored_goal` to `tasks/ar4/mdp.py`**

Add at the end of the file:

```python
def set_mirrored_goal(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    sphere_cfg: SceneEntityCfg,
    goal_y_range: tuple[float, float],
    goal_z_range: tuple[float, float],
) -> None:
    """Event term (mode="reset"): must be registered AFTER the sphere's
    own reset_root_state_uniform event in the same EventCfg (Isaac Lab's
    EventManager runs same-mode terms in registration order - confirmed
    against event_manager.py's apply(), which iterates
    self._mode_term_cfgs[mode] in a plain for loop over registration
    order) so this reads the sphere's freshly-randomized position, not
    the previous episode's. Computes the goal as the mirror image of the
    sphere's spawn across the robot's local x=0 plane (robot_cfg.py's
    AR4_MK5_CFG has no explicit init_state.pos, defaulting to (0,0,0) -
    the robot base sits at each env's own local origin, so negating
    local x is exactly "the other side of the robot"). goal_y is
    independently resampled (not mirrored) for a second degree of
    freedom. Stores the result in env._target_pos_w (world frame,
    per-env, shape (num_envs, 3)) - this stateful buffer replaces
    CommandsCfg/UniformPoseCommandCfg for this scene, since the command
    manager has no way to make one term's target a function of another
    term's own random draw within the same reset. See
    docs/superpowers/specs/2026-07-06-ar4-sphere-mirror-scene-design.md.
    """
    sphere: RigidObject = env.scene[sphere_cfg.name]
    if not hasattr(env, "_target_pos_w"):
        env._target_pos_w = torch.zeros(env.num_envs, 3, device=env.device)

    origins = env.scene.env_origins[env_ids]
    sphere_local_x = sphere.data.root_pos_w[env_ids, 0] - origins[:, 0]

    num = len(env_ids)
    goal_local_x = -sphere_local_x
    goal_local_y = sample_uniform(goal_y_range[0], goal_y_range[1], (num,), env.device)
    goal_local_z = sample_uniform(goal_z_range[0], goal_z_range[1], (num,), env.device)

    env._target_pos_w[env_ids, 0] = origins[:, 0] + goal_local_x
    env._target_pos_w[env_ids, 1] = origins[:, 1] + goal_local_y
    env._target_pos_w[env_ids, 2] = origins[:, 2] + goal_local_z
```

- [ ] **Step 3: Append `mirrored_target_position_in_robot_root_frame` to `tasks/ar4/mdp.py`**

```python
def mirrored_target_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """The mirrored goal position (env._target_pos_w, set by
    set_mirrored_goal) expressed in the robot's root frame - mirrors
    isaaclab_tasks' object_position_in_robot_root_frame pattern exactly,
    reading the stateful buffer instead of an object's own root_pos_w."""
    robot: RigidObject = env.scene[robot_cfg.name]
    if not hasattr(env, "_target_pos_w"):
        env._target_pos_w = torch.zeros(env.num_envs, 3, device=env.device)
    target_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, env._target_pos_w)
    return target_pos_b
```

- [ ] **Step 4: Append `object_reached_mirrored_goal` to `tasks/ar4/mdp.py`**

```python
def object_reached_mirrored_goal(
    env: ManagerBasedRLEnv,
    threshold: float,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Termination: object within threshold of env._target_pos_w - same
    shape as isaaclab_tasks' object_reached_goal, but compares against
    the stateful mirrored-goal buffer instead of the command manager."""
    object: RigidObject = env.scene[object_cfg.name]
    if not hasattr(env, "_target_pos_w"):
        env._target_pos_w = torch.zeros(env.num_envs, 3, device=env.device)
    distance = torch.norm(object.data.root_pos_w - env._target_pos_w, dim=-1)
    return distance < threshold
```

- [ ] **Step 5: Create `tasks/ar4/pickplace_mirror_env_cfg.py`**

```python
# tasks/ar4/pickplace_mirror_env_cfg.py
"""Mirror-goal variant of the AR4 pick-and-place task: only the sphere is
present in the scene (no cube/rect_prism/wedge), its spawn is randomized
across the full workspace, and the goal is always on the opposite side of
the robot from wherever it spawned. See
docs/superpowers/specs/2026-07-06-ar4-sphere-mirror-scene-design.md.

Additive/parallel to pickplace_env_cfg.py: deliberately does NOT touch
`Ar4SceneCfg` (env_cfg.py) or `objects_cfg.py`, since interactive_demo.py,
perception/tests, and other scripts depend on all four objects existing
there - same convention as pickplace_single_object_env_cfg.py. Also does
NOT reuse pickplace_env_cfg.py's CommandsCfg/RewardsCfg/ObservationsCfg/
TerminationsCfg - this task replaces the CommandManager-based goal with a
stateful per-env buffer (env._target_pos_w) that can express "goal is a
function of the sphere's own random spawn", which UniformPoseCommandCfg
cannot.

Import this module only after an Isaac Sim/Isaac Lab AppLauncher has been
created.
"""

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
from .env_cfg import ActionsCfg
from .objects_cfg import SPHERE_CFG
from .pickplace_env_cfg import _EE_OFFSET
from .robot_cfg import AR4_MK5_CFG

# Env-local (robot-relative) workspace bounds this task randomizes the
# sphere's spawn and the goal's y-coordinate within. Defined independently
# of pickplace_env_cfg.py's WORKSPACE_BOUNDS (that constant is documented
# as being for the interactive demo/perception entry points specifically),
# even though the values currently match.
_WORKSPACE_X = (-0.30, 0.30)
_WORKSPACE_Y = (0.10, 0.45)
_GOAL_Z = (0.0, 0.02)


@configclass
class Ar4PickPlaceMirrorSceneCfg(InteractiveSceneCfg):
    """AR4 gripper + a single sphere (no cube/rect_prism/wedge), plus the
    same end-effector FrameTransformer and gripper-to-sphere ContactSensors
    as Ar4PickPlaceSceneCfg (pickplace_env_cfg.py) - copied, not imported,
    since the base scene class differs (no cube/rect_prism/wedge fields)."""

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )
    robot: ArticulationCfg = AR4_MK5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # Recentered to the workspace midpoint (local x=0.0, y=0.275) so
    # reset_sphere_position's pose_range in EventCfg below can cover the
    # full _WORKSPACE_X/_WORKSPACE_Y range symmetrically - SPHERE_CFG
    # itself (objects_cfg.py) is unchanged; .replace() returns a new cfg.
    sphere: RigidObjectCfg = SPHERE_CFG.replace(
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.275, 0.009))
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
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Sphere"],
    )
    gripper_jaw2_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/root_joint/gripper_jaw2_link",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Sphere"],
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        sphere_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame, params={"object_cfg": SceneEntityCfg("sphere")}
        )
        target_object_position = ObsTerm(
            func=ar4_mdp.mirrored_target_position_in_robot_root_frame,
            params={"robot_cfg": SceneEntityCfg("robot")},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset events, in registration order (Isaac Lab's EventManager runs
    same-mode terms in registration order - later terms may depend on
    earlier ones' output within the same reset):
    1. reset_all - whole scene back to default.
    2. reset_sphere_position - randomize the sphere across the full
       workspace (reuses the existing, proven reset_root_state_uniform,
       just with a wider pose_range than pickplace_env_cfg.py's ±2cm
       jitter).
    3. randomize_goal - reads the sphere's now-updated position, sets the
       mirrored goal into env._target_pos_w."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_sphere_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": _WORKSPACE_X, "y": (-0.175, 0.175), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("sphere"),
        },
    )

    randomize_goal = EventTerm(
        func=ar4_mdp.set_mirrored_goal,
        mode="reset",
        params={
            "sphere_cfg": SceneEntityCfg("sphere"),
            "goal_y_range": _WORKSPACE_Y,
            "goal_z_range": _GOAL_Z,
        },
    )


@configclass
class TerminationsCfg:
    """Success (sphere at the mirrored goal) ends the episode early;
    otherwise a fixed timeout."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    sphere_reached_goal = DoneTerm(
        func=ar4_mdp.object_reached_mirrored_goal,
        params={"threshold": 0.02, "object_cfg": SceneEntityCfg("sphere")},
    )
```

Note: `RewardsCfg` and `Ar4PickPlaceMirrorEnvCfg` are added in Task 2 —
this file is incomplete (no assembled env cfg yet) until then, so Task
1's verification below only checks the pieces built so far via a
standalone script, not a full training smoke test.

- [ ] **Step 6: Verify the mirroring math directly**

Create a throwaway verification script (do not commit it — delete after
use) at `/tmp/verify_mirror.py`:

```python
"""Throwaway check: confirm set_mirrored_goal produces a target whose
local x is the negation of the sphere's local x, for every env, right
after a reset. Not part of the test suite - delete after running."""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args(["--headless"])
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

from tasks.ar4.pickplace_mirror_env_cfg import Ar4PickPlaceMirrorSceneCfg, EventCfg, ObservationsCfg  # noqa: E402
from tasks.ar4.env_cfg import ActionsCfg  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.utils.configclass import configclass  # noqa: E402
from isaaclab.managers import TerminationTermCfg as DoneTerm  # noqa: E402
from isaaclab_tasks.manager_based.manipulation.lift import mdp  # noqa: E402


@configclass
class _MinimalTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class _CheckCfg(ManagerBasedRLEnvCfg):
    scene: Ar4PickPlaceMirrorSceneCfg = Ar4PickPlaceMirrorSceneCfg(num_envs=8, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: _MinimalTerminationsCfg = _MinimalTerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 5.0
        self.sim.dt = 0.01


from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

env = ManagerBasedRLEnv(cfg=_CheckCfg())
env.reset()

sphere = env.scene["sphere"]
origins = env.scene.env_origins
sphere_local_x = sphere.data.root_pos_w[:, 0] - origins[:, 0]
target_local_x = env._target_pos_w[:, 0] - origins[:, 0]

print("sphere_local_x:", sphere_local_x.tolist())
print("target_local_x:", target_local_x.tolist())
signs_opposite = torch.all((sphere_local_x * target_local_x) < 0)
print("ALL SIGNS OPPOSITE:", signs_opposite.item())
assert signs_opposite, "Mirroring bug: goal is not on the opposite side for every env"
print("MIRROR CHECK PASSED")

simulation_app.close()
```

Run: `/home/saps/IsaacLab/isaaclab.sh -p /tmp/verify_mirror.py`

Expected output: `MIRROR CHECK PASSED`, with 8 printed
`sphere_local_x`/`target_local_x` pairs of opposite sign. If it fails,
the bug is almost certainly in `set_mirrored_goal`'s use of
`env.scene.env_origins` or the EventCfg registration order — re-check
against `reset_root_state_uniform`'s own `positions = root_states[:, 0:3]
+ env.scene.env_origins[env_ids] + rand_samples[:, 0:3]` line
(`isaaclab/envs/mdp/events.py:1082`) for the exact origin-handling
convention to match.

Delete `/tmp/verify_mirror.py` after this passes (throwaway, not part of
the repo).

- [ ] **Step 7: Commit**

```bash
cd /home/saps/projects/rl
git add tasks/ar4/mdp.py tasks/ar4/pickplace_mirror_env_cfg.py
git commit -m "Add mirrored-goal scene, event, observation, and termination for AR4 sphere mirror task"
```

---

### Task 2: Corrected milestone-bonus reward + stillness penalty + full env cfg

**Files:**
- Modify: `tasks/ar4/mdp.py` (add `_raw_lift_progress_mirrored`,
  `staged_milestone_bonus`, `reset_lift_milestone`, `stillness_penalty`,
  `reset_stillness_buffers`)
- Modify: `tasks/ar4/pickplace_mirror_env_cfg.py` (add `RewardsCfg`,
  extend `EventCfg` with the two new reset events, add
  `Ar4PickPlaceMirrorEnvCfg`)

**Interfaces:**
- Consumes: `contact_grasp_bonus` (existing, `tasks/ar4/mdp.py`,
  unchanged). `Ar4PickPlaceMirrorSceneCfg`, `ObservationsCfg`,
  `TerminationsCfg`, `EventCfg`'s first three terms (Task 1).
- Produces: `staged_milestone_bonus(env, object_cfg, ee_frame_cfg,
  jaw1_contact_cfg, jaw2_contact_cfg, reach_std, force_threshold,
  lift_minimal_height, goal_std) -> torch.Tensor`, shape `(num_envs,)`.
- Produces: `stillness_penalty(env, object_cfg, jaw1_contact_cfg,
  jaw2_contact_cfg, force_threshold, still_bound, patience_steps) ->
  torch.Tensor`, shape `(num_envs,)`.
- Produces: `Ar4PickPlaceMirrorEnvCfg(ManagerBasedRLEnvCfg)`, the
  complete assembled env config, `num_envs=4096` default.

- [ ] **Step 1: Append the corrected milestone-bonus reward functions to `tasks/ar4/mdp.py`**

```python
def _raw_lift_progress_mirrored(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    jaw1_contact_cfg: SceneEntityCfg,
    jaw2_contact_cfg: SceneEntityCfg,
    reach_std: float,
    force_threshold: float,
    lift_minimal_height: float,
    goal_std: float,
) -> torch.Tensor:
    """Same staged reach/grasp/lift/goal signal as _raw_lift_progress,
    but the goal sub-term compares against env._target_pos_w directly
    (already world-frame, set by set_mirrored_goal) instead of
    transforming a CommandsCfg-generated command - this scene has no
    CommandsCfg. See
    docs/superpowers/specs/2026-07-06-ar4-sphere-mirror-scene-design.md.
    """
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    reach_dist = torch.norm(object.data.root_pos_w - ee_frame.data.target_pos_w[:, 0, :], dim=-1)
    reach_term = 1.0 - torch.tanh(reach_dist / reach_std)

    grasp_term = contact_grasp_bonus(env, force_threshold, jaw1_contact_cfg, jaw2_contact_cfg)

    lift_term = (object.data.root_pos_w[:, 2] > lift_minimal_height).float()

    if not hasattr(env, "_target_pos_w"):
        env._target_pos_w = torch.zeros(env.num_envs, 3, device=env.device)
    goal_dist = torch.norm(object.data.root_pos_w - env._target_pos_w, dim=-1)
    goal_term = 1.0 - torch.tanh(goal_dist / goal_std)

    return 0.1 * reach_term + 0.2 * grasp_term + 0.3 * lift_term + 0.4 * goal_term


def staged_milestone_bonus(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    jaw1_contact_cfg: SceneEntityCfg,
    jaw2_contact_cfg: SceneEntityCfg,
    reach_std: float,
    force_threshold: float,
    lift_minimal_height: float,
    goal_std: float,
) -> torch.Tensor:
    """Undiscounted running-max milestone bonus: reward = (new
    best-ever raw progress) - (previous best-ever raw progress) - 0 at a
    plateau, positive on any new milestone, never negative. Corrects a
    bug found in staged_potential_progress (tasks/ar4/mdp.py, used by
    the four-object scene in pickplace_env_cfg.py): that function's
    `gamma * new_potential - prev_potential` formula goes NEGATIVE
    whenever the agent holds a plateaued potential (since gamma < 1),
    making "never approach the object" the reward-minimizing policy -
    see
    docs/superpowers/specs/2026-07-06-ar4-sphere-mirror-scene-design.md's
    "Why now" section for the full derivation. This version has no gamma
    at all - do not add one back.
    """
    if not hasattr(env, "_lift_milestone_max"):
        env._lift_milestone_max = torch.zeros(env.num_envs, device=env.device)

    raw = _raw_lift_progress_mirrored(
        env, object_cfg, ee_frame_cfg, jaw1_contact_cfg, jaw2_contact_cfg,
        reach_std, force_threshold, lift_minimal_height, goal_std,
    )
    prev = env._lift_milestone_max.clone()
    env._lift_milestone_max = torch.maximum(env._lift_milestone_max, raw)
    return env._lift_milestone_max - prev


def reset_lift_milestone(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Event term (mode="reset"): zero the running-max milestone buffer
    for resetting envs, so a new episode starts with no carried-over
    progress. Must be registered in EventCfg alongside
    reset_scene_to_default. Uses a different buffer name
    (_lift_milestone_max) than reset_lift_potential's
    (_lift_potential_max) so the two scenes' state can never collide if
    both were ever imported in the same process.
    """
    if not hasattr(env, "_lift_milestone_max"):
        env._lift_milestone_max = torch.zeros(env.num_envs, device=env.device)
    env._lift_milestone_max[env_ids] = 0.0
```

- [ ] **Step 2: Append the stillness-penalty functions to `tasks/ar4/mdp.py`**

```python
def stillness_penalty(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    jaw1_contact_cfg: SceneEntityCfg,
    jaw2_contact_cfg: SceneEntityCfg,
    force_threshold: float,
    still_bound: float,
    patience_steps: int,
) -> torch.Tensor:
    """Grasp-gated penalty for the object failing to move beyond
    still_bound within patience_steps of its last significant movement.
    Targets the 'reach, grip, freeze' failure mode directly: 0 whenever
    grasp hasn't been achieved yet (pre-grasp settling isn't penalized),
    -1.0 once the object has been essentially stationary for too long
    while gripped. See
    docs/superpowers/specs/2026-07-06-ar4-sphere-mirror-scene-design.md.
    """
    object: RigidObject = env.scene[object_cfg.name]
    if not hasattr(env, "_still_ref_pos"):
        env._still_ref_pos = object.data.root_pos_w.clone()
        env._still_steps = torch.zeros(env.num_envs, device=env.device)

    pos = object.data.root_pos_w
    moved = torch.norm(pos - env._still_ref_pos, dim=-1) > still_bound
    env._still_ref_pos = torch.where(moved.unsqueeze(-1), pos, env._still_ref_pos)
    env._still_steps = torch.where(moved, torch.zeros_like(env._still_steps), env._still_steps + 1)

    grasped = contact_grasp_bonus(env, force_threshold, jaw1_contact_cfg, jaw2_contact_cfg) > 0.5
    stagnant = env._still_steps > patience_steps
    return -(grasped & stagnant).float()


def reset_stillness_buffers(env: ManagerBasedRLEnv, env_ids: torch.Tensor, object_cfg: SceneEntityCfg) -> None:
    """Event term (mode="reset"): must be registered after randomize_goal
    (EventCfg in pickplace_mirror_env_cfg.py) so the reference position
    reflects the new episode's spawn, not the prior episode's end state.
    """
    object: RigidObject = env.scene[object_cfg.name]
    if not hasattr(env, "_still_ref_pos"):
        env._still_ref_pos = torch.zeros(env.num_envs, 3, device=env.device)
        env._still_steps = torch.zeros(env.num_envs, device=env.device)
    env._still_ref_pos[env_ids] = object.data.root_pos_w[env_ids]
    env._still_steps[env_ids] = 0.0
```

- [ ] **Step 3: Add `RewardsCfg`, extend `EventCfg`, add `Ar4PickPlaceMirrorEnvCfg` in `tasks/ar4/pickplace_mirror_env_cfg.py`**

Change the `EventCfg` class from (Task 1's version):

```python
@configclass
class EventCfg:
    """Reset events, in registration order (Isaac Lab's EventManager runs
    same-mode terms in registration order - later terms may depend on
    earlier ones' output within the same reset):
    1. reset_all - whole scene back to default.
    2. reset_sphere_position - randomize the sphere across the full
       workspace (reuses the existing, proven reset_root_state_uniform,
       just with a wider pose_range than pickplace_env_cfg.py's ±2cm
       jitter).
    3. randomize_goal - reads the sphere's now-updated position, sets the
       mirrored goal into env._target_pos_w."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_sphere_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": _WORKSPACE_X, "y": (-0.175, 0.175), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("sphere"),
        },
    )

    randomize_goal = EventTerm(
        func=ar4_mdp.set_mirrored_goal,
        mode="reset",
        params={
            "sphere_cfg": SceneEntityCfg("sphere"),
            "goal_y_range": _WORKSPACE_Y,
            "goal_z_range": _GOAL_Z,
        },
    )
```

to (adds two new reset events at the end, same docstring plus the two
new steps):

```python
@configclass
class EventCfg:
    """Reset events, in registration order (Isaac Lab's EventManager runs
    same-mode terms in registration order - later terms may depend on
    earlier ones' output within the same reset):
    1. reset_all - whole scene back to default.
    2. reset_sphere_position - randomize the sphere across the full
       workspace (reuses the existing, proven reset_root_state_uniform,
       just with a wider pose_range than pickplace_env_cfg.py's ±2cm
       jitter).
    3. randomize_goal - reads the sphere's now-updated position, sets the
       mirrored goal into env._target_pos_w.
    4. reset_lift_milestone / reset_stillness_buffers - zero the new
       reward terms' stateful buffers, so a new episode starts with no
       carried-over progress or stale stillness reference."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_sphere_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": _WORKSPACE_X, "y": (-0.175, 0.175), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("sphere"),
        },
    )

    randomize_goal = EventTerm(
        func=ar4_mdp.set_mirrored_goal,
        mode="reset",
        params={
            "sphere_cfg": SceneEntityCfg("sphere"),
            "goal_y_range": _WORKSPACE_Y,
            "goal_z_range": _GOAL_Z,
        },
    )

    reset_lift_milestone = EventTerm(func=ar4_mdp.reset_lift_milestone, mode="reset")

    reset_stillness_buffers = EventTerm(
        func=ar4_mdp.reset_stillness_buffers,
        mode="reset",
        params={"object_cfg": SceneEntityCfg("sphere")},
    )
```

Then append `RewardsCfg` and `Ar4PickPlaceMirrorEnvCfg` at the end of the
file (after `TerminationsCfg`):

```python
@configclass
class RewardsCfg:
    """Corrected undiscounted staged milestone bonus (see
    staged_milestone_bonus's docstring for the bug this fixes in
    staged_potential_progress) plus a grasp-gated stillness penalty."""

    staged_milestone_bonus = RewTerm(
        func=ar4_mdp.staged_milestone_bonus,
        weight=25.0,
        params={
            "object_cfg": SceneEntityCfg("sphere"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "jaw1_contact_cfg": SceneEntityCfg("gripper_jaw1_contact"),
            "jaw2_contact_cfg": SceneEntityCfg("gripper_jaw2_contact"),
            "reach_std": 0.1,
            "force_threshold": 0.05,
            "lift_minimal_height": 0.03,
            "goal_std": 0.3,
        },
    )

    stillness_penalty = RewTerm(
        func=ar4_mdp.stillness_penalty,
        weight=-2.0,
        params={
            "object_cfg": SceneEntityCfg("sphere"),
            "jaw1_contact_cfg": SceneEntityCfg("gripper_jaw1_contact"),
            "jaw2_contact_cfg": SceneEntityCfg("gripper_jaw2_contact"),
            "force_threshold": 0.05,
            "still_bound": 0.005,
            "patience_steps": 25,
        },
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class Ar4PickPlaceMirrorEnvCfg(ManagerBasedRLEnvCfg):
    """AR4 mirror-goal task: pick up the sphere (randomized spawn across
    the full workspace) and place it on the opposite side of the robot.
    num_envs=4096 default (a real training-scale run, not the smaller
    num_envs=16 used by the single-object camera-training precedent) -
    scripts/train.py's --num_envs flag overrides this per-run same as
    every other env cfg in this repo."""

    scene: Ar4PickPlaceMirrorSceneCfg = Ar4PickPlaceMirrorSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 5.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0)
        self.viewer.eye = (1.5, 1.5, 1.2)
        self.viewer.lookat = (0.0, 0.0, 0.4)
```

Note: `Ar4PickPlaceMirrorEnvCfg` does not set a `commands` field.
`ManagerBasedRLEnvCfg.commands` (`isaaclab/envs/manager_based_rl_env_cfg.py:76`)
defaults to `None` ("no commands are generated") - confirmed from source,
this is intentional, not an oversight, since this scene's goal comes from
`env._target_pos_w` instead of the command manager.

- [ ] **Step 4: Smoke test**

```bash
cd /home/saps/projects/rl
/home/saps/IsaacLab/isaaclab.sh -p -c "
from isaaclab.app import AppLauncher
import argparse
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args(['--headless'])
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv
from tasks.ar4.pickplace_mirror_env_cfg import Ar4PickPlaceMirrorEnvCfg

import torch

cfg = Ar4PickPlaceMirrorEnvCfg()
cfg.scene.num_envs = 16
env = ManagerBasedRLEnv(cfg=cfg)
env.reset()
zeros = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
for _ in range(5):
    env.step(zeros)
print('SMOKE TEST PASSED: 4 reward terms =', list(env.reward_manager.active_terms))
simulation_app.close()
"
```

Expected: prints `SMOKE TEST PASSED: 4 reward terms =
['staged_milestone_bonus', 'stillness_penalty', 'action_rate',
'joint_vel']`, exits 0, no exceptions (in particular no
`AttributeError` on `_target_pos_w`/`_lift_milestone_max`/
`_still_ref_pos`/`_still_steps` regardless of reward-fn/reset-event
ordering).

- [ ] **Step 5: Commit**

```bash
cd /home/saps/projects/rl
git add tasks/ar4/mdp.py tasks/ar4/pickplace_mirror_env_cfg.py
git commit -m "Add corrected undiscounted milestone-bonus reward and grasp-gated stillness penalty for AR4 sphere mirror task"
```

---

### Task 3: Wire `--mirror` flag into `scripts/train.py` and `scripts/eval_loop.py`

**Files:**
- Modify: `scripts/train.py`
- Modify: `scripts/eval_loop.py`

**Interfaces:**
- Consumes: `Ar4PickPlaceMirrorEnvCfg` (Task 2).
- Produces: `--mirror` CLI flag on both scripts.

- [ ] **Step 1: Add the `--mirror` flag and env-selection branch to `scripts/train.py`**

Change:

```python
parser.add_argument(
    "--perception",
    action="store_true",
    default=False,
    help=(
        "Train on the single-object (sphere-only) scene, observing the sphere's position via the real "
        "RGB-D perception_camera + perception pipeline instead of privileged simulation state. The reward "
        "function is unchanged (stays privileged). Implies --enable_cameras."
    ),
)
AppLauncher.add_app_launcher_args(parser)
```

to:

```python
parser.add_argument(
    "--perception",
    action="store_true",
    default=False,
    help=(
        "Train on the single-object (sphere-only) scene, observing the sphere's position via the real "
        "RGB-D perception_camera + perception pipeline instead of privileged simulation state. The reward "
        "function is unchanged (stays privileged). Implies --enable_cameras."
    ),
)
parser.add_argument(
    "--mirror",
    action="store_true",
    default=False,
    help=(
        "Train on the mirror-goal scene (sphere only, spawn randomized across the full workspace, goal "
        "always on the opposite side of the robot from the spawn), with the corrected undiscounted "
        "milestone-bonus reward and a grasp-gated stillness penalty. See "
        "docs/superpowers/specs/2026-07-06-ar4-sphere-mirror-scene-design.md."
    ),
)
AppLauncher.add_app_launcher_args(parser)
```

Change:

```python
from tasks.ar4.pickplace_env_cfg import GROUND_Z, Ar4PickPlaceEnvCfg  # noqa: E402
from tasks.ar4.pickplace_single_object_env_cfg import Ar4PickPlaceSingleObjectEnvCfg  # noqa: E402
```

to:

```python
from tasks.ar4.pickplace_env_cfg import GROUND_Z, Ar4PickPlaceEnvCfg  # noqa: E402
from tasks.ar4.pickplace_mirror_env_cfg import Ar4PickPlaceMirrorEnvCfg  # noqa: E402
from tasks.ar4.pickplace_single_object_env_cfg import Ar4PickPlaceSingleObjectEnvCfg  # noqa: E402
```

Change:

```python
def main() -> None:
    env_cfg_cls = Ar4PickPlaceSingleObjectEnvCfg if args_cli.perception else Ar4PickPlaceEnvCfg
    env_cfg = env_cfg_cls()
```

to:

```python
def main() -> None:
    if args_cli.mirror:
        env_cfg_cls = Ar4PickPlaceMirrorEnvCfg
    elif args_cli.perception:
        env_cfg_cls = Ar4PickPlaceSingleObjectEnvCfg
    else:
        env_cfg_cls = Ar4PickPlaceEnvCfg
    env_cfg = env_cfg_cls()
```

- [ ] **Step 2: Smoke-test `scripts/train.py --mirror`**

```bash
cd /home/saps/projects/rl
/home/saps/IsaacLab/isaaclab.sh -p scripts/train.py --mirror --num_envs 16 --max_iterations 2 --headless
```

Expected: exits 0. Verify via file evidence (per the Global Constraints'
stdout-buffering note): the newest `logs/train/<timestamp>/` directory
has `model_0.pt` and `model_1.pt`, and its `params/env.yaml` shows
exactly the 4 reward terms from Task 2 and no `commands` section (or an
empty one).

- [ ] **Step 3: Add the `--mirror` flag and env-selection branch to `scripts/eval_loop.py`**

Change:

```python
parser.add_argument(
    "--perception",
    action="store_true",
    default=False,
    help="Use the real camera-based perception pipeline instead of privileged simulation state for the sphere's observed position.",
)
AppLauncher.add_app_launcher_args(parser)
```

to:

```python
parser.add_argument(
    "--perception",
    action="store_true",
    default=False,
    help="Use the real camera-based perception pipeline instead of privileged simulation state for the sphere's observed position.",
)
parser.add_argument(
    "--mirror",
    action="store_true",
    default=False,
    help="Evaluate the mirror-goal scene (see scripts/train.py --mirror) instead of the four-object scene.",
)
AppLauncher.add_app_launcher_args(parser)
```

Change:

```python
from tasks.ar4.pickplace_env_cfg import GROUND_Z, Ar4PickPlaceEnvCfg, Ar4PickPlacePerceptionEnvCfg  # noqa: E402
```

to:

```python
from tasks.ar4.pickplace_env_cfg import GROUND_Z, Ar4PickPlaceEnvCfg, Ar4PickPlacePerceptionEnvCfg  # noqa: E402
from tasks.ar4.pickplace_mirror_env_cfg import Ar4PickPlaceMirrorEnvCfg  # noqa: E402
```

Change:

```python
def main() -> None:
    env_cfg_cls = Ar4PickPlacePerceptionEnvCfg if args_cli.perception else Ar4PickPlaceEnvCfg
    env_cfg = env_cfg_cls()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device
```

to:

```python
def main() -> None:
    if args_cli.mirror:
        env_cfg_cls = Ar4PickPlaceMirrorEnvCfg
    elif args_cli.perception:
        env_cfg_cls = Ar4PickPlacePerceptionEnvCfg
    else:
        env_cfg_cls = Ar4PickPlaceEnvCfg
    env_cfg = env_cfg_cls()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device
```

Change (the `else` branch's `RecordVideo` call — gives the mirror scene's
eval videos a distinct filename prefix so they can't be confused with or
overwrite the four-object scene's `ar4_pickplace-step-*.mp4` files):

```python
    else:
        # step_trigger, not episode_trigger: RslRlVecEnvWrapper resets the env exactly
        # once at construction and never again, so episode_trigger's episode_id never
        # advances and silently merges every episode into one video. 250 = one episode's
        # worth of steps (episode_length_s=5.0 / step_dt=decimation*sim.dt=2*0.01=0.02s),
        # from Ar4PickPlaceEnvCfg - update this if that config changes.
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=VIDEO_DIR,
            step_trigger=lambda step: step % 250 == 0,
            video_length=250,
            name_prefix="ar4_pickplace",
            disable_logger=True,
        )
```

to:

```python
    else:
        # step_trigger, not episode_trigger: RslRlVecEnvWrapper resets the env exactly
        # once at construction and never again, so episode_trigger's episode_id never
        # advances and silently merges every episode into one video. 250 = one episode's
        # worth of steps (episode_length_s=5.0 / step_dt=decimation*sim.dt=2*0.01=0.02s),
        # from Ar4PickPlaceEnvCfg - update this if that config changes.
        name_prefix = "ar4_pickplace_mirror" if args_cli.mirror else "ar4_pickplace"
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=VIDEO_DIR,
            step_trigger=lambda step: step % 250 == 0,
            video_length=250,
            name_prefix=name_prefix,
            disable_logger=True,
        )
```

- [ ] **Step 4: Smoke-test `scripts/eval_loop.py --mirror`**

First get a checkpoint from Step 2's smoke test:

```bash
cd /home/saps/projects/rl
LATEST=$(ls -dt logs/train/*/ | head -1)
/home/saps/IsaacLab/isaaclab.sh -p scripts/eval_loop.py --mirror --checkpoint "${LATEST}model_1.pt" --episodes 1
```

Expected: exits 0, produces `logs/videos/ar4_pickplace_mirror-step-0.mp4`.

- [ ] **Step 5: Commit**

```bash
cd /home/saps/projects/rl
git add scripts/train.py scripts/eval_loop.py
git commit -m "Wire --mirror flag into train.py and eval_loop.py for the AR4 sphere mirror task"
```

---

### Task 4: Full 1500-iteration training run (4096 envs)

**Files:** none (no code changes — this task runs the training loop and
inspects its output).

**Interfaces:**
- Consumes: `Ar4PickPlaceMirrorEnvCfg` (Task 2), `--mirror` flag (Task 3).
- Produces: `logs/train/<timestamp>/model_1499.pt` and TensorBoard event
  logs, consumed by Task 5.

- [ ] **Step 1: Run the full 1500-iteration training run**

```bash
cd /home/saps/projects/rl
/home/saps/IsaacLab/isaaclab.sh -p scripts/train.py --mirror --num_envs 4096 --headless
```

Expected wall-clock time: roughly 15-25 minutes, based on every prior
full run at this scale this session (~16 minutes each).

Note the resulting log directory and confirm `model_1499.pt` exists
(1500 iterations, 0-indexed) for Task 5.

- [ ] **Step 2: Pull the key TensorBoard scalars**

```bash
cd /home/saps/projects/rl
LATEST=$(ls -dt logs/train/*/ | head -1)
/home/saps/IsaacLab/isaaclab.sh -p -c "
from tensorboard.backend.event_processing import event_accumulator
import glob
path = sorted(glob.glob('${LATEST}events.out.tfevents.*'))[-1]
ea = event_accumulator.EventAccumulator(path)
ea.Reload()
for tag in ['Episode_Reward/staged_milestone_bonus', 'Episode_Reward/stillness_penalty',
            'Episode_Termination/sphere_reached_goal', 'Episode_Reward/action_rate', 'Episode_Reward/joint_vel']:
    if tag in ea.Tags()['scalars']:
        vals = ea.Scalars(tag)
        n = len(vals)
        samples = [vals[i].value for i in range(0, n, max(1, n // 10))]
        print(tag, '-> first:', vals[0].value, 'last:', vals[-1].value, 'max:', max(v.value for v in vals),
              'min:', min(v.value for v in vals),
              'trajectory (10 samples):', [round(s, 6) for s in samples])
    else:
        print(tag, '-> NOT FOUND')
"
```

Record all five lines verbatim in the Task 4 report. In particular check
`stillness_penalty`'s trajectory: if it's growing more negative over
training (rather than shrinking toward 0), that's a sign the policy is
still choosing to freeze despite the penalty.

- [ ] **Step 3: Write the report (create the file)**

Create `docs/superpowers/plans/2026-07-06-ar4-sphere-mirror-scene-report.md`
with a "Task 4" section containing: the log directory path, the five
scalar lines from Step 2, and one factual sentence on whether
`staged_milestone_bonus` grows through training (it should be
non-negative at every point, per its corrected formula — flag
immediately if any value is negative, that would mean the fix itself has
a bug) — no success/failure judgment yet, that's Task 5's job after the
eval video.

```bash
git add docs/superpowers/plans/2026-07-06-ar4-sphere-mirror-scene-report.md
git commit -m "Record AR4 sphere mirror-scene full training run results"
```

---

### Task 5: Real eval + video inspection (decision gate)

**Files:** none (no code changes — this task runs eval and visually
inspects output).

**Interfaces:**
- Consumes: `logs/train/<timestamp>/model_1499.pt` from Task 4.
- Produces: eval videos in `logs/videos/`, a final pass/fail verdict
  consumed by Task 6.

- [ ] **Step 1: Run eval for 10 episodes**

```bash
cd /home/saps/projects/rl
/home/saps/IsaacLab/isaaclab.sh -p scripts/eval_loop.py --mirror --checkpoint logs/train/<RUN_DIR>/model_1499.pt --episodes 10
```

(substitute the actual `<RUN_DIR>` from Task 4). Expected: 10 files
`logs/videos/ar4_pickplace_mirror-step-0.mp4` through `-step-2250.mp4`.

- [ ] **Step 2: Extract frames from every episode video**

```bash
cd /home/saps/projects/rl
rm -rf logs/videos/frames_mirror
mkdir -p logs/videos/frames_mirror
for f in logs/videos/ar4_pickplace_mirror-step-*.mp4; do
  name=$(basename "$f" .mp4)
  mkdir -p "logs/videos/frames_mirror/$name"
  ffmpeg -y -i "$f" -vf fps=10 "logs/videos/frames_mirror/$name/frame_%03d.png" -loglevel error
done
```

- [ ] **Step 3: Visually inspect all 10 episodes**

Use the Read tool to view frames from each of the 10 episode directories
(start, ~25%, ~50%, ~75%, end is a good baseline sample; check adjacent
frames before concluding anything if the sphere marker briefly appears
missing — it can be occluded by the gripper body at some camera angles
without having been lifted, as happened in two prior experiments this
session). For each episode, determine: does the sphere spawn at
different (visibly varied) positions across episodes, does the sphere
visibly leave the ground at any point, and if so, does it stay lifted
and get carried toward the opposite side of the robot from its spawn?

- [ ] **Step 4: Apply the decision gate**

- **If 8/10 or more episodes show the sphere genuinely lifted and
  carried toward the target:** success. Proceed to Task 6's success
  path.
- **If fewer than 8/10 do, but some episodes show real (even brief)
  lifting that never happened in prior experiments:** partial progress —
  describe precisely what's different.
- **If 0/10 show any lift (same "reach, grip, freeze" signature):** this
  experiment is falsified. This is the sixth real attempt on this
  sub-problem — flag back to the user (per
  `superpowers:systematic-debugging` Phase 4.5) rather than attempting a
  seventh reward/optimization tweak unilaterally; candidates worth
  raising: a hierarchical reach-then-grasp policy split, or
  reconsidering the physical/task setup (gripper geometry, object scale)
  rather than the reward function.

- [ ] **Step 5: Commit the report update**

```bash
git add docs/superpowers/plans/2026-07-06-ar4-sphere-mirror-scene-report.md
git commit -m "Record AR4 sphere mirror-scene eval video inspection results"
```

---

### Task 6: Record the outcome in `ROADMAP.md`

**Files:**
- Modify: `ROADMAP.md`

**Interfaces:** none (documentation-only task, terminal in this plan).

- [ ] **Step 1: Write the outcome into `ROADMAP.md`'s "grasp/lift never
  emerges" follow-up**

Add a new bullet after the potential-shaping sub-bullet already recorded
there, following the same evidentiary detail as the existing
sub-bullets. Use whichever template applies, filling in real numbers
from Tasks 4-5:

**If Task 5's decision gate passed (8/10+):**

```markdown
   - **Follow-up experiment: mirror-goal scene + corrected milestone
     bonus + stillness penalty (SUCCESS).** Simplified the scene to the
     sphere only (randomized spawn across the full workspace, goal
     always on the opposite side of the robot), fixed the discounted
     potential-shaping decay bug (undiscounted `staged_milestone_bonus`),
     and added a grasp-gated stillness penalty, per
     `docs/superpowers/specs/2026-07-06-ar4-sphere-mirror-scene-design.md`.
     Full run data:
     `docs/superpowers/plans/2026-07-06-ar4-sphere-mirror-scene-report.md`.
     **Result: [X]/10 real eval episodes show the sphere genuinely
     lifted and carried to the opposite side.** This resolves the
     "grasp/lift never emerges" follow-up.
```

**If Task 5's decision gate did not pass (0/10 or partial):**

```markdown
   - **Follow-up experiment: mirror-goal scene + corrected milestone
     bonus + stillness penalty ([falsified | partial progress]).**
     Simplified the scene to the sphere only (randomized spawn across
     the full workspace, goal always on the opposite side of the
     robot), fixed the discounted potential-shaping decay bug
     (undiscounted `staged_milestone_bonus`), and added a grasp-gated
     stillness penalty, per
     `docs/superpowers/specs/2026-07-06-ar4-sphere-mirror-scene-design.md` -
     this was user-directed rather than a unilateral sixth
     reward/optimization retry. Full run data:
     `docs/superpowers/plans/2026-07-06-ar4-sphere-mirror-scene-report.md`.
     **Result: [X]/10 real eval episodes show any lift** — [one to two
     sentences on what the video actually showed]. This is the sixth
     real attempt on the reward/optimization axis for this sub-problem.
     Flagged back to the Principal/user; remaining candidates are a
     hierarchical reach-then-grasp-policy split, or reconsidering this
     repo's specific physical/task setup (gripper geometry, object
     scale) rather than the reward function.
```

- [ ] **Step 2: Commit**

```bash
git add ROADMAP.md
git commit -m "Record AR4 sphere mirror-scene experiment outcome in ROADMAP"
```

---

## Plan self-review notes

- **Spec coverage:** Task 1 covers the design's scene/event/observation/
  termination pieces. Task 2 covers the corrected milestone-bonus reward
  and the stillness penalty, plus the full assembled env cfg. Task 3
  covers script wiring (not explicitly in the design doc, but required
  for Tasks 4-5 to be runnable at all). Task 4 covers the full-run half
  of the design's "Verification plan". Task 5 covers the eval/video half.
  Task 6 covers ROADMAP recording.
- **Scope discipline:** confirmed no task modifies `env_cfg.py`,
  `objects_cfg.py`, `pickplace_env_cfg.py`, or any existing `mdp.py`
  function. `staged_milestone_bonus` has no `gamma` param anywhere in the
  plan (checked every appearance).
- **Type/name consistency:** `staged_milestone_bonus`'s params
  (`object_cfg, ee_frame_cfg, jaw1_contact_cfg, jaw2_contact_cfg,
  reach_std, force_threshold, lift_minimal_height, goal_std`) match
  between Task 2's function definition and `RewardsCfg`'s registration.
  `stillness_penalty`'s params (`object_cfg, jaw1_contact_cfg,
  jaw2_contact_cfg, force_threshold, still_bound, patience_steps`) match
  likewise. `_target_pos_w`, `_lift_milestone_max`, `_still_ref_pos`,
  `_still_steps` are the same buffer names everywhere they're used
  across Tasks 1-2. `EventCfg`'s registration order
  (`reset_all` → `reset_sphere_position` → `randomize_goal` →
  `reset_lift_milestone` → `reset_stillness_buffers`) matches every
  task's stated dependency (`randomize_goal` needs the sphere's new
  position; `reset_stillness_buffers` needs the new goal already set,
  though it only actually reads the sphere's position, not the goal —
  registered after `randomize_goal` regardless, for a single consistent
  "final state" ordering that's easy to reason about).
