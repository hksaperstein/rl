# Experiment 18 Implementation Plan: Dense Pre-Grasp-Readiness Shaping

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Experiment 18 — add one dense shaping term (proximity × gripper-closedness) on top of Experiment 17's unchanged binary grasp gate, testing whether it gives the policy a discoverable gradient toward combining "get close" and "close the gripper" — the two halves Task 6's instrumented evidence showed being explored independently but never together.

**Architecture:** One new function `pregrasp_readiness_bonus` in `tasks/ar4/mdp.py`, wired into a new `Ar4PickPlacePregraspEnvCfg` (`tasks/ar4/pickplace_pregrasp_env_cfg.py`) that is otherwise byte-identical to Experiment 17's `Ar4PickPlaceGraspGatedEnvCfg` — the grasp gate itself (`lifting_object_grasp_gated`, `mirrored_goal_distance_grasp_gated`) is reused completely unchanged, isolating the new dense term as the only new variable.

**Tech Stack:** Isaac Lab `ManagerBasedRLEnvCfg`, `FrameTransformer.data.target_pos_w` (already-proven pattern from `reaching_object`), `Articulation.find_joints`/`data.joint_pos` (already-proven pattern from `gripper_schedule_bonus`), rsl_rl PPO (`Ar4PickPlacePPORunnerCfg`, unchanged).

## Global Constraints

- Do not modify `tasks/ar4/pickplace_graspgated_env_cfg.py` or any existing function in `tasks/ar4/mdp.py` (including `genuine_grasp_and_lift`, `lifting_object_grasp_gated`, `mirrored_goal_distance_grasp_gated`, all reused unchanged) — purely additive (one new function, one new env cfg file).
- Reward weights, exact values (identical to Experiment 17 except one new term added): `reaching_object` (weight 1.0), `lifting_object` (`ar4_mdp.lifting_object_grasp_gated`, weight 15.0, unchanged params), `object_goal_tracking` (`ar4_mdp.mirrored_goal_distance_grasp_gated`, weight 16.0, unchanged params), `object_goal_tracking_fine_grained` (same function, weight 5.0, unchanged params), `action_rate` (weight -1e-4), `joint_vel` (weight -1e-4), plus new `pregrasp_readiness` (`ar4_mdp.pregrasp_readiness_bonus`, weight 2.0).
- Action space: plain `isaaclab_mdp.JointPositionActionCfg(asset_name="robot", joint_names=ARM_JOINT_NAMES, scale=0.5)` — identical to Experiments 16-17. PPO runner cfg: plain `Ar4PickPlacePPORunnerCfg` — **not** `Ar4PickPlaceTaskspacePPORunnerCfg`.
- `ActionsCfg`, `ObservationsCfg`, `EventCfg`, `TerminationsCfg`, `CurriculumCfg` are all identical to Experiment 17's — copy verbatim, no changes.
- Always invoke scripts via `/home/saps/IsaacLab/isaaclab.sh -p scripts/<script>.py` (or `/home/saps/IsaacLab/isaaclab.sh -p -c "<snippet>"` for one-off Python) from the repo root, never plain `python`.
- Verify via file evidence (checkpoints, TensorBoard event files, `params/env.yaml`) — this repo has no pytest-based unit tests for Isaac-Sim-dependent code.
- **Any subagent dispatched to launch or wait on a real Isaac Sim training job must be given the literal blocking poll command in its dispatch prompt** (not just told to "poll" in prose) — this exact mistake has recurred multiple times this session even when warned only in prose.
- This experiment does **not** need Experiment 17's Task-6-style instrumented contact-force verification — it isn't making a mechanism-verification claim about grasp quality; it's testing whether `Episode_Reward/lifting_object`'s nonzero rate moves off exactly `0/1500`, a question the standard TensorBoard scalar report answers directly. Video inspection is conditional (see Task 5) — only performed if `lifting_object` shows real nonzero occurrences worth visually characterizing.

---

### Task 1: `pregrasp_readiness_bonus` reward function

**Files:**
- Modify: `tasks/ar4/mdp.py` (append new function at end of file — current file ends at line 892, after `mirrored_goal_distance_grasp_gated`)

**Interfaces:**
- Consumes: `FrameTransformer`, `Articulation`, `SceneEntityCfg`, `torch` (all already imported/type-checked at the top of `mdp.py` — `Articulation`/`FrameTransformer` are under the existing `TYPE_CHECKING` block, confirmed by reading the file's current header), `Articulation.find_joints()` (already-proven pattern, used identically by `gripper_schedule_bonus` earlier in the same file: `gripper_joint_ids, _ = robot.find_joints(gripper_joint_names)` then `robot.data.joint_pos[:, gripper_joint_ids].mean(dim=-1)`).
- Produces: `pregrasp_readiness_bonus(env, std, object_cfg, ee_frame_cfg, robot_cfg, gripper_joint_names, open_pos, closed_pos) -> torch.Tensor` — a `RewardTermCfg` function, consumed by Task 2's `RewardsCfg`.

- [ ] **Step 1: Append the new function to `tasks/ar4/mdp.py`**

Add this function at the end of the file:

```python
def pregrasp_readiness_bonus(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    gripper_joint_names: list[str],
    open_pos: float,
    closed_pos: float,
) -> torch.Tensor:
    """Dense reward for combining proximity AND gripper closure - the two
    halves Task 6's instrumented rollout showed being explored
    independently but never together (Experiment 17: one event showed
    the gripper fully closed nowhere near the cube; another showed the
    arm within 2.6cm of the cube with the gripper pinned open). Reward is
    the product of a proximity term (same tanh-kernel shape as
    reaching_object) and a normalized "closedness" term (1.0 when the
    gripper is fully closed, 0.0 when fully open) - maximized only when
    both are true simultaneously, giving zero credit for closing far
    from the object or approaching without closing. Does NOT reward
    antipodal alignment or contact force - purely a positional/
    configuration signal, kept deliberately weaker/less specific than
    antipodal_grasp_bonus's own force-closure check, which remains the
    only gate for lifting_object/object_goal_tracking, unchanged from
    Experiment 17. See
    docs/superpowers/specs/2026-07-07-ar4-experiment18-pregrasp-readiness-shaping-design.md.
    """
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    dist = torch.norm(object.data.root_pos_w - ee_pos_w, dim=-1)
    proximity_term = 1.0 - torch.tanh(dist / std)

    gripper_joint_ids, _ = robot.find_joints(gripper_joint_names)
    gripper_pos = robot.data.joint_pos[:, gripper_joint_ids].mean(dim=-1)
    closedness_term = torch.clamp((open_pos - gripper_pos) / (open_pos - closed_pos), 0.0, 1.0)

    return proximity_term * closedness_term
```

No new imports needed: `SceneEntityCfg`, `torch`, `RigidObject`, `Articulation`, `FrameTransformer`, `ManagerBasedRLEnv` are all already imported/type-checked at the top of `tasks/ar4/mdp.py`.

- [ ] **Step 2: Syntax-check the file**

Run: `python3 -c "import ast; ast.parse(open('tasks/ar4/mdp.py').read())"`
Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add tasks/ar4/mdp.py
git commit -m "Add pregrasp_readiness_bonus reward function for Experiment 18"
```

---

### Task 2: New pre-grasp-readiness env cfg file

**Files:**
- Create: `tasks/ar4/pickplace_pregrasp_env_cfg.py`

**Interfaces:**
- Consumes: `Ar4PickPlaceMirrorSceneCfg` (from `tasks/ar4/pickplace_mirror_env_cfg.py`, unmodified), `ar4_mdp.pregrasp_readiness_bonus` (Task 1), `ar4_mdp.lifting_object_grasp_gated`/`mirrored_goal_distance_grasp_gated`/`set_mirrored_goal`/`mirrored_target_position_in_robot_root_frame`/`object_reached_mirrored_goal` (all pre-existing, unmodified), `mdp.object_ee_distance`/`modify_reward_weight` (from `isaaclab_tasks.manager_based.manipulation.lift.mdp`, reused directly), `ARM_JOINT_NAMES`/`GRIPPER_JOINT_NAMES`/`GRIPPER_OPEN_POS`/`GRIPPER_CLOSED_POS` (from `robot_cfg.py`), `Ar4PickPlacePPORunnerCfg` (from `tasks/ar4/agents/rsl_rl_ppo_cfg.py`, unmodified, reused directly).
- Produces: `Ar4PickPlacePregraspEnvCfg` class — consumed by Task 3 (script wiring).

- [ ] **Step 1: Write the new file**

```python
# tasks/ar4/pickplace_pregrasp_env_cfg.py
"""Dense pre-grasp-readiness shaping variant (Experiment 18): identical to
Experiment 17's Ar4PickPlaceGraspGatedEnvCfg
(tasks/ar4/pickplace_graspgated_env_cfg.py) in every respect except one
new reward term, pregrasp_readiness (proximity x gripper-closedness).
Experiment 17's own instrumented investigation (Task 6,
.superpowers/sdd/task-6-report.md) found the trained policy exploring
"get close to the object" and "close the gripper" independently but
never combining them - the binary antipodal grasp gate
(lifting_object_grasp_gated/mirrored_goal_distance_grasp_gated, both
REUSED UNCHANGED here) never fired even once across 1500 iterations
because the compound behavior it requires was never discovered. This
experiment adds a dense stepping-stone signal toward that compound
behavior without touching the gate itself. See
docs/superpowers/specs/2026-07-07-ar4-experiment18-pregrasp-readiness-shaping-design.md.

Additive/parallel to every other pickplace_*.py file: does NOT modify
env_cfg.py, objects_cfg.py, pickplace_mirror_env_cfg.py,
pickplace_taskspace_env_cfg.py, pickplace_residual_env_cfg.py,
pickplace_reachskip_env_cfg.py, pickplace_baseproximity_env_cfg.py,
pickplace_provenrecipe_env_cfg.py, or pickplace_graspgated_env_cfg.py.
Reuses Ar4PickPlaceMirrorSceneCfg and Ar4PickPlacePPORunnerCfg directly.

Import this module only after an Isaac Sim/Isaac Lab AppLauncher has been
created.
"""

import isaaclab.envs.mdp as isaaclab_mdp
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.manipulation.lift import mdp

from . import mdp as ar4_mdp
from .pickplace_mirror_env_cfg import Ar4PickPlaceMirrorSceneCfg
from .agents.rsl_rl_ppo_cfg import Ar4PickPlacePPORunnerCfg
from .robot_cfg import ARM_JOINT_NAMES, GRIPPER_CLOSED_POS, GRIPPER_JOINT_NAMES, GRIPPER_OPEN_POS


@configclass
class ActionsCfg:
    """Plain joint-space action, identical to Experiment 17's
    ActionsCfg."""

    joint_positions = isaaclab_mdp.JointPositionActionCfg(asset_name="robot", joint_names=ARM_JOINT_NAMES, scale=0.5)
    gripper_position = isaaclab_mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=GRIPPER_JOINT_NAMES,
        open_command_expr={name: GRIPPER_OPEN_POS for name in GRIPPER_JOINT_NAMES},
        close_command_expr={name: GRIPPER_CLOSED_POS for name in GRIPPER_JOINT_NAMES},
    )


@configclass
class ObservationsCfg:
    """Identical to Experiment 17's ObservationsCfg."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        cube_position = ObsTerm(
            func=mdp.object_position_in_robot_root_frame, params={"object_cfg": SceneEntityCfg("cube")}
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
    """Identical to Experiment 17's EventCfg."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_cube_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.30, 0.30), "y": (-0.175, 0.175), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube"),
        },
    )

    randomize_goal = EventTerm(
        func=ar4_mdp.set_mirrored_goal,
        mode="reset",
        params={
            "object_cfg": SceneEntityCfg("cube"),
            "goal_y_range": (0.10, 0.45),
            "goal_z_range": (0.0, 0.02),
        },
    )


@configclass
class TerminationsCfg:
    """Identical to Experiment 17's TerminationsCfg."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    cube_reached_goal = DoneTerm(
        func=ar4_mdp.object_reached_mirrored_goal,
        params={"threshold": 0.02, "object_cfg": SceneEntityCfg("cube")},
    )


@configclass
class RewardsCfg:
    """Seven terms: the six from Experiment 17, unchanged (including the
    binary grasp gate on lifting_object/object_goal_tracking - reused
    completely unmodified), plus one new dense term, pregrasp_readiness
    (Task 1). See
    docs/superpowers/specs/2026-07-07-ar4-experiment18-pregrasp-readiness-shaping-design.md."""

    reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        weight=1.0,
        params={"std": 0.1, "object_cfg": SceneEntityCfg("cube"), "ee_frame_cfg": SceneEntityCfg("ee_frame")},
    )

    pregrasp_readiness = RewTerm(
        func=ar4_mdp.pregrasp_readiness_bonus,
        weight=2.0,
        params={
            "std": 0.1,
            "object_cfg": SceneEntityCfg("cube"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "robot_cfg": SceneEntityCfg("robot"),
            "gripper_joint_names": GRIPPER_JOINT_NAMES,
            "open_pos": GRIPPER_OPEN_POS,
            "closed_pos": GRIPPER_CLOSED_POS,
        },
    )

    lifting_object = RewTerm(
        func=ar4_mdp.lifting_object_grasp_gated,
        weight=15.0,
        params={
            "object_cfg": SceneEntityCfg("cube"),
            "jaw1_contact_cfg": SceneEntityCfg("gripper_jaw1_contact"),
            "jaw2_contact_cfg": SceneEntityCfg("gripper_jaw2_contact"),
            "force_threshold": 0.05,
            "antipodal_cos_threshold": -0.7071,
            "minimal_height": 0.03,
        },
    )

    object_goal_tracking = RewTerm(
        func=ar4_mdp.mirrored_goal_distance_grasp_gated,
        weight=16.0,
        params={
            "std": 0.3,
            "minimal_height": 0.03,
            "object_cfg": SceneEntityCfg("cube"),
            "jaw1_contact_cfg": SceneEntityCfg("gripper_jaw1_contact"),
            "jaw2_contact_cfg": SceneEntityCfg("gripper_jaw2_contact"),
            "force_threshold": 0.05,
            "antipodal_cos_threshold": -0.7071,
        },
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=ar4_mdp.mirrored_goal_distance_grasp_gated,
        weight=5.0,
        params={
            "std": 0.05,
            "minimal_height": 0.03,
            "object_cfg": SceneEntityCfg("cube"),
            "jaw1_contact_cfg": SceneEntityCfg("gripper_jaw1_contact"),
            "jaw2_contact_cfg": SceneEntityCfg("gripper_jaw2_contact"),
            "force_threshold": 0.05,
            "antipodal_cos_threshold": -0.7071,
        },
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class CurriculumCfg:
    """Identical to Experiment 17's CurriculumCfg."""

    action_rate_curr = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    )

    joint_vel_curr = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    )


@configclass
class Ar4PickPlacePregraspEnvCfg(ManagerBasedRLEnvCfg):
    """AR4 pre-grasp-readiness task (Experiment 18): identical to
    Experiment 17 plus one new dense shaping term rewarding the
    combination of proximity and gripper closure. num_envs=4096 default -
    scripts/train.py's --num_envs flag overrides this per-run same as
    every other env cfg in this repo."""

    scene: Ar4PickPlaceMirrorSceneCfg = Ar4PickPlaceMirrorSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 5.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0)
        self.viewer.eye = (1.5, 1.5, 1.2)
        self.viewer.lookat = (0.0, 0.0, 0.4)
```

- [ ] **Step 2: Syntax-check the file**

Run: `python3 -c "import ast; ast.parse(open('tasks/ar4/pickplace_pregrasp_env_cfg.py').read())"`
Expected: no output.

- [ ] **Step 3: Commit**

```bash
git add tasks/ar4/pickplace_pregrasp_env_cfg.py
git commit -m "Add Ar4PickPlacePregraspEnvCfg: dense pre-grasp-readiness shaping (Experiment 18)"
```

---

### Task 3: Wire `--pregrasp` into `scripts/train.py` and `scripts/eval_loop.py`, then smoke-test

**Files:**
- Modify: `scripts/train.py`
- Modify: `scripts/eval_loop.py`

**Interfaces:**
- Consumes: `Ar4PickPlacePregraspEnvCfg` (Task 2), `Ar4PickPlacePPORunnerCfg` (pre-existing, already imported by both scripts).
- Produces: `--pregrasp` CLI flag on both scripts, verified via a headless 2-iteration smoke test.

- [ ] **Step 1: Add the flag and import to `scripts/train.py`**

Immediately after the existing `--graspgated` `parser.add_argument(...)` block (ending right before `AppLauncher.add_app_launcher_args(parser)`), insert:

```python
parser.add_argument(
    "--pregrasp",
    action="store_true",
    default=False,
    help=(
        "Train on the dense pre-grasp-readiness shaping variant: adds one new reward term (proximity "
        "x gripper-closedness) on top of --graspgated's unchanged binary antipodal grasp gate, giving "
        "the policy a discoverable gradient toward combining 'get close' and 'close the gripper' - the "
        "two halves Experiment 17's own instrumented investigation found being explored independently "
        "but never together. See "
        "docs/superpowers/specs/2026-07-07-ar4-experiment18-pregrasp-readiness-shaping-design.md."
    ),
)
```

Add the import next to the existing `pickplace_graspgated_env_cfg` import:

```python
from tasks.ar4.pickplace_pregrasp_env_cfg import Ar4PickPlacePregraspEnvCfg  # noqa: E402
```

Change the `env_cfg_cls` selection to add `--pregrasp` as the first branch:

```python
    if args_cli.pregrasp:
        env_cfg_cls = Ar4PickPlacePregraspEnvCfg
    elif args_cli.graspgated:
        env_cfg_cls = Ar4PickPlaceGraspGatedEnvCfg
    elif args_cli.provenrecipe:
        env_cfg_cls = Ar4PickPlaceProvenRecipeEnvCfg
    elif args_cli.baseproximity:
        env_cfg_cls = Ar4PickPlaceBaseProximityEnvCfg
    elif args_cli.reachskip:
        env_cfg_cls = Ar4PickPlaceReachskipEnvCfg
    elif args_cli.residual:
        env_cfg_cls = Ar4PickPlaceResidualEnvCfg
    elif args_cli.taskspace:
        env_cfg_cls = Ar4PickPlaceTaskspaceEnvCfg
    elif args_cli.ik_guided:
        env_cfg_cls = Ar4PickPlaceIkGuidedEnvCfg
    elif args_cli.mirror:
        env_cfg_cls = Ar4PickPlaceMirrorEnvCfg
    elif args_cli.perception:
        env_cfg_cls = Ar4PickPlaceSingleObjectEnvCfg
    else:
        env_cfg_cls = Ar4PickPlaceEnvCfg
```

**Do NOT add `--pregrasp` to the `agent_cfg` selection** — the current condition is exactly `if args_cli.taskspace or args_cli.residual or args_cli.reachskip or args_cli.baseproximity:` (4 flags — `--provenrecipe` and `--graspgated` are both already correctly excluded, and `--pregrasp` must be excluded the same way, since this experiment uses plain joint-space action). Leave this code exactly as it is:

```python
    if args_cli.taskspace or args_cli.residual or args_cli.reachskip or args_cli.baseproximity:
        agent_cfg = Ar4PickPlaceTaskspacePPORunnerCfg()
    else:
        agent_cfg = Ar4PickPlacePPORunnerCfg()
```

- [ ] **Step 2: Add the flag and import to `scripts/eval_loop.py`**

Immediately after the existing `--graspgated` `parser.add_argument(...)` block (ending right before `AppLauncher.add_app_launcher_args(parser)`), insert:

```python
parser.add_argument(
    "--pregrasp",
    action="store_true",
    default=False,
    help="Evaluate the dense pre-grasp-readiness shaping scene (see scripts/train.py --pregrasp) instead of the four-object scene.",
)
```

Add the import next to the existing `pickplace_graspgated_env_cfg` import:

```python
from tasks.ar4.pickplace_pregrasp_env_cfg import Ar4PickPlacePregraspEnvCfg  # noqa: E402
```

Change the `env_cfg_cls` selection to add `--pregrasp` as the first branch:

```python
    if args_cli.pregrasp:
        env_cfg_cls = Ar4PickPlacePregraspEnvCfg
    elif args_cli.graspgated:
        env_cfg_cls = Ar4PickPlaceGraspGatedEnvCfg
    elif args_cli.provenrecipe:
        env_cfg_cls = Ar4PickPlaceProvenRecipeEnvCfg
    elif args_cli.baseproximity:
        env_cfg_cls = Ar4PickPlaceBaseProximityEnvCfg
    elif args_cli.reachskip:
        env_cfg_cls = Ar4PickPlaceReachskipEnvCfg
    elif args_cli.residual:
        env_cfg_cls = Ar4PickPlaceResidualEnvCfg
    elif args_cli.taskspace:
        env_cfg_cls = Ar4PickPlaceTaskspaceEnvCfg
    elif args_cli.ik_guided:
        env_cfg_cls = Ar4PickPlaceIkGuidedEnvCfg
    elif args_cli.mirror:
        env_cfg_cls = Ar4PickPlaceMirrorEnvCfg
    elif args_cli.perception:
        env_cfg_cls = Ar4PickPlacePerceptionEnvCfg
    else:
        env_cfg_cls = Ar4PickPlaceEnvCfg
```

**Do NOT add `--pregrasp` to the `agent_cfg` selection** — same reasoning, leave exactly as-is:

```python
    if args_cli.taskspace or args_cli.residual or args_cli.reachskip or args_cli.baseproximity:
        agent_cfg = Ar4PickPlaceTaskspacePPORunnerCfg()
    else:
        agent_cfg = Ar4PickPlacePPORunnerCfg()
```

And change the `name_prefix` selection to add `--pregrasp` as the first branch:

```python
        if args_cli.pregrasp:
            name_prefix = "ar4_pickplace_pregrasp"
        elif args_cli.graspgated:
            name_prefix = "ar4_pickplace_graspgated"
        elif args_cli.provenrecipe:
            name_prefix = "ar4_pickplace_provenrecipe"
        elif args_cli.baseproximity:
            name_prefix = "ar4_pickplace_baseproximity"
        elif args_cli.reachskip:
            name_prefix = "ar4_pickplace_reachskip"
        elif args_cli.residual:
            name_prefix = "ar4_pickplace_residual"
        elif args_cli.taskspace:
            name_prefix = "ar4_pickplace_taskspace"
        elif args_cli.ik_guided:
            name_prefix = "ar4_pickplace_ik_guided"
        elif args_cli.mirror:
            name_prefix = "ar4_pickplace_mirror"
        else:
            name_prefix = "ar4_pickplace"
```

- [ ] **Step 3: Syntax-check both files**

Run: `python3 -c "import ast; ast.parse(open('scripts/train.py').read()); ast.parse(open('scripts/eval_loop.py').read())"`
Expected: no output.

- [ ] **Step 4: Smoke test — 2-iteration headless training run**

This is the FIRST time `Ar4PickPlacePregraspEnvCfg` will actually run inside Isaac Sim — Tasks 1-2 only had syntax checks.

Run (from repo root, foreground, allow up to 5 minutes):
```bash
rm -f /tmp/pregrasp_smoke_stdout.log
timeout 280 /home/saps/IsaacLab/isaaclab.sh -p scripts/train.py --pregrasp --num_envs 16 --max_iterations 2 --headless > /tmp/pregrasp_smoke_stdout.log 2>&1
echo "exit_code=$?"
```

A `timeout`/nonzero exit code alone is NOT proof of failure (Isaac Sim's shutdown sequence sometimes hangs after genuinely finishing) — verify via files:

```bash
grep -i "error\|exception\|traceback" /tmp/pregrasp_smoke_stdout.log
ls -la logs/train/ | tail -5
```

Find the newest `logs/train/<timestamp>/` directory and check:
```bash
find logs/train/<newest_timestamp_dir> -name "model_*.pt"
cat logs/train/<newest_timestamp_dir>/params/env.yaml | grep -A8 "pregrasp_readiness"
```

Expected: `model_0.pt` and `model_1.pt` both exist, `env.yaml` confirms `pregrasp_readiness`'s function is `pregrasp_readiness_bonus` with `std`/`ee_frame_cfg`/`gripper_joint_names`/`open_pos`/`closed_pos` params present, weight 2.0, and no traceback in the stdout log. If an exception appears, the most likely culprit is a params mismatch between `RewardsCfg`'s `pregrasp_readiness` params dict and `pregrasp_readiness_bonus`'s actual signature from Task 1 — re-check both directly.

- [ ] **Step 5: Commit**

```bash
git add scripts/train.py scripts/eval_loop.py
git commit -m "Wire --pregrasp flag into train.py and eval_loop.py for Experiment 18"
```

---

### Task 4: Diagnostic run (300 iterations) — verify the new dense term is stable and provides real signal

**Files:**
- None created — this is a verification-only task. If the diagnostic looks wrong, stop and report; do not proceed to Task 5.

**Interfaces:**
- Consumes: `Ar4PickPlacePregraspEnvCfg` (Task 2) via the `--pregrasp` flag (Task 3).
- Produces: a confirmed-healthy diagnostic run directory under `logs/train/`, gating Task 5.

- [ ] **Step 1: Launch the diagnostic run**

```bash
nohup /home/saps/IsaacLab/isaaclab.sh -p scripts/train.py --pregrasp --num_envs 4096 --max_iterations 300 --headless > /tmp/exp18_diagnostic_stdout.log 2>&1 &
echo "PID=$!"
```

- [ ] **Step 2: Poll for completion — BLOCK on this yourself, do not end your turn early**

Run this exact command (a real blocking loop — if one call's timeout is hit before the run finishes, re-issue the same command again):
```bash
until find logs/train/ -name "model_299.pt" -newer tasks/ar4/pickplace_pregrasp_env_cfg.py 2>/dev/null | grep -q .; do sleep 30; done
echo "diagnostic complete"
```
Based on this repo's prior 300-iteration diagnostic runs, expect roughly 3-5 minutes of real wall-clock time — do not assume failure before at least 10-15 minutes have elapsed.

- [ ] **Step 3: Extract and check the diagnostic scalars**

```bash
LATEST=$(ls -dt logs/train/*/ | head -1)
/home/saps/IsaacLab/isaaclab.sh -p -c "
from tensorboard.backend.event_processing import event_accumulator
import glob
path = sorted(glob.glob('${LATEST}events.out.tfevents.*'))[-1]
ea = event_accumulator.EventAccumulator(path)
ea.Reload()
for tag in ['Loss/value_function', 'Episode_Reward/pregrasp_readiness', 'Episode_Reward/lifting_object',
            'Episode_Reward/reaching_object', 'Episode_Termination/cube_reached_goal']:
    if tag in ea.Tags()['scalars']:
        vals = ea.Scalars(tag)
        nonzero = sum(1 for v in vals if v.value != 0.0)
        print(tag, '-> first:', vals[0].value, 'last:', vals[-1].value, 'max:', max(v.value for v in vals),
              'min:', min(v.value for v in vals), 'nonzero:', nonzero, '/', len(vals))
    else:
        print(tag, '-> NOT FOUND')
"
```

- [ ] **Step 4: Evaluate the diagnostic against these three checks**

1. **`Loss/value_function` stays bounded (no sustained exponential growth).** This experiment reuses already-proven action space (joint-space, validated in Experiments 16-17) and already-proven `ee_frame`/gripper-joint-position reading patterns (`reaching_object`, `gripper_schedule_bonus`) — lower novel-mechanism risk, but the diagnostic gate is this project's uniform standing practice regardless of perceived risk. A small transient spike with immediate recovery, or a larger spike that clearly declines by the run's end (matching Experiments 16/17's own curriculum-firing precedent), is acceptable; a sustained, non-recovering climb is not.
2. **No exceptions/tracebacks in `/tmp/exp18_diagnostic_stdout.log`.**
3. **`Episode_Reward/pregrasp_readiness` shows a real nonzero rate early in the diagnostic window** (it's a dense term, unlike `lifting_object`'s sparse gate — it should NOT be flat zero like `lifting_object` was in every prior diagnostic in this repo's history). If `pregrasp_readiness` is also flat zero for the full 300 iterations, that is a real concern worth flagging explicitly before proceeding — it would suggest a bug in the new term itself (e.g. the proximity/closedness product never becoming meaningfully positive) rather than the expected, already-seen slow-discovery pattern of the binary-gated terms, and is worth a closer look before committing to the full run rather than assuming it will resolve itself.

If checks 1 and 2 pass, proceed to Task 5. If check 3 raises a real concern (flat zero), still proceed to Task 5 (this is not a hard blocking gate the way 1-2 are — `pregrasp_readiness` could plausibly need a few dozen more iterations than the 300-iteration window to become visible even if working correctly, since exploration takes time even for dense-adjacent terms in a fresh policy) but flag it explicitly and instruct Task 5 to check this specifically and early in its own full-run trajectory.

---

### Task 5: Full 1500-iteration training run + TensorBoard verification report

**Files:**
- Create: `docs/superpowers/plans/2026-07-07-ar4-experiment18-report.md`

**Interfaces:**
- Consumes: the Task 4-verified reward configuration.
- Produces: a complete 1500-iteration training run under `logs/train/`, and a written report with full scalar trajectories.

- [ ] **Step 1: Launch the full training run**

```bash
nohup /home/saps/IsaacLab/isaaclab.sh -p scripts/train.py --pregrasp --num_envs 4096 --headless > /tmp/exp18_train_stdout.log 2>&1 &
echo "PID=$!"
```

- [ ] **Step 2: Poll for completion — BLOCK on this yourself, do not end your turn early**

```bash
until find logs/train/ -name "model_1499.pt" -newer tasks/ar4/pickplace_pregrasp_env_cfg.py 2>/dev/null | grep -q .; do sleep 60; done
echo "full run complete"
```
Expect roughly 15-25 minutes of real wall-clock time based on this repo's prior 1500-iteration runs at `num_envs=4096` — treat that as a rough guide, not a hard cutoff; keep re-issuing the blocking command if a single tool call's own timeout is hit before the checkpoint appears.

Once found, confirm checkpoint integrity:
```bash
LATEST=$(ls -dt logs/train/*/ | head -1)
find "$LATEST" -name "model_*.pt" | wc -l
ls -la "${LATEST}"events.out.tfevents.*
```
Expected: 31 checkpoints (0, 50, 100, ..., 1450, 1499 — `save_interval=50`), `model_1499.pt` exists, event file mtime matches the run's actual completion time.

- [ ] **Step 3: Extract full scalar trajectories**

```bash
LATEST=$(ls -dt logs/train/*/ | head -1)
/home/saps/IsaacLab/isaaclab.sh -p -c "
from tensorboard.backend.event_processing import event_accumulator
import glob
path = sorted(glob.glob('${LATEST}events.out.tfevents.*'))[-1]
ea = event_accumulator.EventAccumulator(path)
ea.Reload()
tags = ['Episode_Reward/reaching_object', 'Episode_Reward/pregrasp_readiness', 'Episode_Reward/lifting_object',
        'Episode_Reward/object_goal_tracking', 'Episode_Reward/object_goal_tracking_fine_grained',
        'Episode_Termination/cube_reached_goal', 'Loss/value_function']
for tag in tags:
    if tag in ea.Tags()['scalars']:
        vals = ea.Scalars(tag)
        nonzero = sum(1 for v in vals if v.value != 0.0)
        print(f'=== {tag} ===')
        print('  points:', len(vals), 'first:', vals[0].value, 'last:', vals[-1].value,
              'max:', max(v.value for v in vals), 'min:', min(v.value for v in vals),
              'nonzero:', nonzero, '/', len(vals))
        for i in range(0, len(vals), 150):
            print(f'  iteration={vals[i].step:4d}, value={vals[i].value:.6f}')
        print(f'  iteration={vals[-1].step:4d}, value={vals[-1].value:.6f}')
    else:
        print(tag, '-> NOT FOUND')
"
```

- [ ] **Step 4: Write the report**

Write `docs/superpowers/plans/2026-07-07-ar4-experiment18-report.md` following the structure of `docs/superpowers/plans/2026-07-07-ar4-experiment17-report.md` (checkpoint integrity, `Loss/value_function` sanity check, per-term summary + sampled trajectory for all 7 tags above). Include a "Key Comparison" section against **Experiment 17's exact final value** and **Experiment 12's exact final value**, final-iteration snapshot vs. final-iteration snapshot only, per this project's established correction protocol:

- Experiment 12 final `Episode_Termination/cube_reached_goal`: 0.010773
- Experiment 17 final `Episode_Termination/cube_reached_goal`: 0.002360

**The report must explicitly and separately answer two questions, each with the actual sampled trajectory as evidence (not just first/last/max):**

1. Does `Episode_Reward/pregrasp_readiness` show real, growing nonzero occurrence across the run (confirming the new term provides a usable, discovered gradient at all — a prerequisite for the next question to have a chance)?
2. Does `Episode_Reward/lifting_object`'s nonzero count move off exactly `0/1500` at any point in the run — this is the single specific, falsifiable question this experiment exists to answer, per the design spec's own success criteria. Report the exact nonzero count (e.g. "3/1500" or "0/1500"), not just a qualitative description.

**If `lifting_object` is still exactly `0/1500`:** state this plainly as the finding. Per the design spec's own verification plan, do NOT perform a video-inspection step in this case — there is nothing new for video to show beyond what Experiment 17's video-free scalar-only reporting already established, and this project's practice is to use video for genuine ambiguity, not to fill out a template when the scalar evidence is already unambiguous. The report itself may state the "still never happens" outcome directly.

**If `lifting_object` DOES show any nonzero occurrences:** do NOT draw a final success/failure conclusion from scalars alone in that case — flag explicitly in the report that video inspection is now warranted (to be done separately, personally, by the controller, matching this session's established pattern for decisive evidence) before any success claim, per this project's own established lesson from Experiment 16.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/plans/2026-07-07-ar4-experiment18-report.md
git commit -m "Record Experiment 18 training run: pre-grasp-readiness shaping scalar trajectories"
```
