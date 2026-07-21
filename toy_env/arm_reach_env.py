"""Gymnasium-compatible, physics-free proxy env for reach/grip/lift prototyping.

**Scope note (read this before trusting anything here for a real decision):**
Part of `toy_env/` — a CPU-only, physics-free (pure kinematics, no contact/
collision/friction/mass) proxy environment for fast algorithm/action-space
prototyping. It exists to let questions like "PPO vs SAC" or "does absolute
joint control cause the same training pathology here that it did in the real
Isaac Lab experiment" be iterated in seconds/minutes on CPU, before spending
real Isaac Sim GPU time on anything that looks promising. It is NOT a
substitute for Isaac Sim: there is no real grasp mechanics (a "grasp" here is
a kinematic attachment triggered by a distance threshold + a closed-gripper
action, not a contact-force/antipodal condition), no dynamics, and no
collision. Any interesting result found here is a hypothesis generator, not a
conclusion, and needs re-verification in the real simulator before being
trusted. See `kb/wiki/concepts/toy-kinematic-proxy-env.md` for the full
writeup, and `kb/wiki/experiments/d8-antipodal-grasp-quality.md`'s "Root cause
investigation" section for the real finding this environment tries to make
cheaply reproducible.

Action modes (the actual point of this environment)
-----------------------------------------------------
Three modes, selected via `ArmReachEnv(action_mode=...)`, chosen specifically
to reproduce the configuration-dependent-vs-independent action semantics
implicated in the real finding above — not just "an arm reaching a target":

- ``"absolute"``: each of the 7 arm-action components maps directly (via a
  fixed linear scaling, independent of the arm's current state) to an
  *absolute target joint angle* in `[-JOINT_LIMIT, JOINT_LIMIT]`. Each step,
  the arm moves toward that target at a bounded max joint velocity
  (`clip(target - current, -MAX_JOINT_VEL, MAX_JOINT_VEL)`). This is the
  toy-scale analogue of Isaac Lab's `JointPositionActionCfg` w/
  `use_default_offset` semantics: the same action value always maps to the
  same *target*, but the resulting *motion from wherever the arm currently
  is* is heavily configuration-dependent — a given action can mean "barely
  move" or "move fast toward the far side of the joint's range" depending on
  where the arm already is. This is the mode expected (per the real Isaac Lab
  finding) to show the transient-discovery-then-abandonment pathology.
- ``"relative"``: each arm-action component maps to a fixed-scale *joint-angle
  delta* (`MAX_JOINT_DELTA` per step) added directly to the current joint
  angles. The same action always produces the same joint-space motion
  regardless of current configuration — this is the toy-scale analogue of
  Isaac Lab's `RelativeJointPositionActionCfg`. Still joint-space, but
  configuration-independent in joint-space (not fully configuration-
  independent in Cartesian/end-effector space, since the same joint-space
  delta still produces different end-effector motion at different arm poses
  via the pose-dependent Jacobian — a real, deliberately-preserved
  distinction between this mode and full task-space control below).
- ``"task_space"``: the 3 non-gripper action components are interpreted as a
  desired end-effector Cartesian velocity (`MAX_EE_DELTA` per step), converted
  to a joint-space delta via `kinematic_arm.jacobian_position`'s
  Moore-Penrose pseudo-inverse (`dq = pinv(J) @ v_desired`). This is the
  toy-scale analogue of Isaac Lab's `DifferentialInverseKinematicsActionCfg`:
  the action's effect on the end effector is ~configuration-independent by
  construction (that's what the IK solve is for), at the cost of only
  controlling end-effector *position* here (no orientation control — a
  deliberate simplification per the design brief's "even a basic Jacobian
  pseudo-inverse solve is fine, doesn't need to be sophisticated").

All three modes share the same observation space, reward structure, and
episode/termination logic — only the action interpretation differs — so a
training-curve comparison across modes isolates the action-space effect.

Staged reward (reach -> grip -> lift)
--------------------------------------
Follows this project's own staged/potential-based precedent (AR4-era
Experiment 25, `kb/wiki/experiments/experiment-25-touch-goal-reach.md`,
and the non-decreasing-staged-reward principle in
`kb/wiki/concepts/staged-reward-co-satisfiability.md`): later stages are never
worth less than fully achieving an earlier stage, because each stage's reward
is *added on top of* the previous stage's, not a replacement for it.

- **Reach** (dense, every step): ``REACH_WEIGHT * (1 - tanh(dist / REACH_SCALE))``,
  in ``(0, REACH_WEIGHT)``, maximized as end-effector-to-target distance ``dist``
  shrinks.
- **Grip** (bonus, every step while satisfied): a flat ``GRIP_BONUS`` awarded
  whenever the arm is currently "carrying" the object (see below). Reaching
  grip range at all requires `dist < GRIP_DIST_THRESHOLD`, i.e. the reach
  term is already near its own maximum — so achieving grip is strictly
  additive on top of a near-maxed reach reward, never a trade-off against it.
- **Lift** (bonus, every step while carrying): ``LIFT_WEIGHT *
  clip(height_gain / LIFT_HEIGHT_TARGET, 0, 1)``, where `height_gain` is the
  object's current height above its spawn height. Only nonzero while
  `carrying` is true (i.e. grip's own bonus is already being earned), so lift
  reward is additive on top of grip, never a substitute for it.

"Grasping" here is a physics-free kinematic proxy, not a contact simulation:
the object becomes rigidly attached to the end-effector tip position exactly
when ``dist < GRIP_DIST_THRESHOLD`` AND the gripper action is "closed"; it
stays attached (and its position tracks the end-effector 1:1) for as long as
the gripper remains closed; opening the gripper drops it (freezes its
position, ends the `carrying` bonus stream). There is no contact force, no
slip, no friction, and no possibility of a partial/misaligned grasp — this is
the sharpest way in which this proxy is *not* the real d8/Franka grasp
mechanics (see `kb/wiki/concepts/grasp-mechanics-antipodal-vs-magnitude.md`
for what the real mechanism actually requires) and should not be mistaken
for one.

Episode ends in success (`terminated=True`, `info["success"]=True`) once
`height_gain >= LIFT_HEIGHT_TARGET` has been continuously true for
`SUSTAINED_LIFT_STEPS` consecutive steps (mirroring this project's own
"sustained lift" concept, e.g. `envs_with_sustained_lift` in
`scripts/franka_checkpoint_review.py`), or truncates at `max_episode_steps`.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from toy_env import kinematic_arm as ka

# --- Tunable constants (toy-scale, not fit to any real robot) ---------------

MAX_JOINT_VEL = 0.15  # rad/step, absolute-mode target-chase speed limit
MAX_JOINT_DELTA = 0.08  # rad/step, relative-mode per-step joint delta scale
MAX_EE_DELTA = 0.03  # m/step, task-space-mode per-step Cartesian delta scale

REACH_SCALE = 0.25  # m, distance scale for the tanh reach-reward shaping
GRIP_DIST_THRESHOLD = 0.05  # m
LIFT_HEIGHT_TARGET = 0.15  # m
SUSTAINED_LIFT_STEPS = 10

REACH_WEIGHT = 1.0
GRIP_BONUS = 2.0
LIFT_WEIGHT = 3.0

DEFAULT_MAX_EPISODE_STEPS = 150

VALID_ACTION_MODES = ("absolute", "relative", "task_space")


class ArmReachEnv(gym.Env):
    """7-joint kinematic-arm reach/grip/lift proxy environment.

    See module docstring for the full design rationale, action-mode
    semantics, and reward structure. This class implements the standard
    Gymnasium `reset()`/`step()`/`observation_space`/`action_space` API so it
    plugs directly into Stable-Baselines3 (or any other Gymnasium-compatible
    RL library) with no additional integration work.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        action_mode: str = "relative",
        max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if action_mode not in VALID_ACTION_MODES:
            raise ValueError(f"action_mode must be one of {VALID_ACTION_MODES}, got {action_mode!r}")
        self.action_mode = action_mode
        self.max_episode_steps = max_episode_steps

        n_arm_dims = 3 if action_mode == "task_space" else ka.N_JOINTS
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_arm_dims + 1,), dtype=np.float32
        )
        # obs: joint_angles(7)/limit, ee_pos(3), target_pos(3),
        # target_pos-ee_pos(3), gripper_closed(1), carrying(1) = 18
        # Finite bounds (rather than +/-inf) based on this chain's own
        # max_reach(), with generous margin - real values should never
        # approach these bounds given the reachable-workspace target
        # sampling in reset(), but a finite Box is better-behaved for
        # RL libraries/wrappers that assume bounded observations.
        reach_bound = ka.max_reach() * 1.5
        obs_low = np.concatenate(
            [
                np.full(ka.N_JOINTS, -1.2),
                np.full(3, -reach_bound),
                np.full(3, -reach_bound),
                np.full(3, -2 * reach_bound),
                [0.0],
                [0.0],
            ]
        ).astype(np.float32)
        obs_high = -obs_low
        obs_high[-2:] = 1.0
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self._rng = np.random.default_rng(seed)
        self.theta = np.zeros(ka.N_JOINTS)
        self.target_pos = np.zeros(3)
        self.object_pos = np.zeros(3)
        self.object_spawn_z = 0.0
        self.carrying = False
        self.gripper_closed = False
        self.step_count = 0
        self.sustained_lift_count = 0
        self.trajectory: list[np.ndarray] = []

    # -- Gymnasium API --------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Start from a modest, non-singular home pose (not all-zeros, which
        # is a fully-vertical/degenerate configuration for this chain).
        self.theta = self._rng.uniform(-0.3, 0.3, size=ka.N_JOINTS)

        # Sample a target within a reachable-ish forward workspace shell.
        radius = self._rng.uniform(0.35, 0.65)
        az = self._rng.uniform(-np.pi / 3, np.pi / 3)  # forward-facing cone
        height = self._rng.uniform(0.05, 0.45)
        self.target_pos = np.array(
            [radius * np.cos(az), radius * np.sin(az), height]
        )

        self.object_pos = self.target_pos.copy()
        self.object_spawn_z = self.object_pos[2]
        self.carrying = False
        self.gripper_closed = False
        self.step_count = 0
        self.sustained_lift_count = 0

        ee_pos = ka.forward_kinematics(self.theta).ee_pos
        self.trajectory = [ee_pos.copy()]

        return self._get_obs(), self._get_info(dist=float(np.linalg.norm(ee_pos - self.target_pos)))

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        gripper_action = action[-1]
        arm_action = action[:-1]

        self.theta = self._apply_action(arm_action)
        self.gripper_closed = bool(gripper_action > 0.0)

        ee_pos = ka.forward_kinematics(self.theta).ee_pos
        self.trajectory.append(ee_pos.copy())
        dist = float(np.linalg.norm(ee_pos - self.target_pos))

        grip_now = dist < GRIP_DIST_THRESHOLD and self.gripper_closed
        if grip_now:
            self.carrying = True
        elif not self.gripper_closed:
            self.carrying = False

        if self.carrying:
            self.object_pos = ee_pos.copy()

        height_gain = max(0.0, float(self.object_pos[2] - self.object_spawn_z))

        reach_reward = REACH_WEIGHT * (1.0 - np.tanh(dist / REACH_SCALE))
        grip_reward = GRIP_BONUS if self.carrying else 0.0
        lift_reward = (
            LIFT_WEIGHT * float(np.clip(height_gain / LIFT_HEIGHT_TARGET, 0.0, 1.0))
            if self.carrying
            else 0.0
        )
        reward = float(reach_reward + grip_reward + lift_reward)

        sustained_lift = self.carrying and height_gain >= LIFT_HEIGHT_TARGET
        self.sustained_lift_count = self.sustained_lift_count + 1 if sustained_lift else 0

        self.step_count += 1
        success = self.sustained_lift_count >= SUSTAINED_LIFT_STEPS
        terminated = success
        truncated = self.step_count >= self.max_episode_steps

        info = self._get_info(
            dist=dist,
            reach_reward=reach_reward,
            grip_reward=grip_reward,
            lift_reward=lift_reward,
            height_gain=height_gain,
            success=success,
        )
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):  # pragma: no cover - thin delegation, exercised via visualize.py
        from toy_env import visualize

        return visualize.plot_arm_pose_3d(self)

    # -- internals --------------------------------------------------------

    def _apply_action(self, arm_action: np.ndarray) -> np.ndarray:
        if self.action_mode == "absolute":
            target = arm_action * ka.JOINT_LIMIT
            delta = np.clip(target - self.theta, -MAX_JOINT_VEL, MAX_JOINT_VEL)
            new_theta = self.theta + delta
        elif self.action_mode == "relative":
            delta = arm_action * MAX_JOINT_DELTA
            new_theta = self.theta + delta
        elif self.action_mode == "task_space":
            v_desired = arm_action * MAX_EE_DELTA
            J = ka.jacobian_position(self.theta)
            dq = np.linalg.pinv(J) @ v_desired
            # Bound the resulting joint-space step for stability, same order
            # of magnitude as the other modes' own per-step joint bound.
            dq = np.clip(dq, -MAX_JOINT_VEL, MAX_JOINT_VEL)
            new_theta = self.theta + dq
        else:  # pragma: no cover - guarded in __init__
            raise ValueError(self.action_mode)
        return np.clip(new_theta, -ka.JOINT_LIMIT, ka.JOINT_LIMIT)

    def _get_obs(self) -> np.ndarray:
        ee_pos = ka.forward_kinematics(self.theta).ee_pos
        obs = np.concatenate(
            [
                self.theta / ka.JOINT_LIMIT,
                ee_pos,
                self.target_pos,
                self.target_pos - ee_pos,
                [1.0 if self.gripper_closed else 0.0],
                [1.0 if self.carrying else 0.0],
            ]
        ).astype(np.float32)
        return obs

    def _get_info(self, **kwargs: Any) -> dict[str, Any]:
        info = {
            "action_mode": self.action_mode,
            "carrying": self.carrying,
            "step_count": self.step_count,
        }
        info.update(kwargs)
        return info
