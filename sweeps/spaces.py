"""Declarative parameter-space definitions, one per trainable task.

A ``TaskSpace`` says, for one task: how to launch it (``train_flag``), which
TensorBoard scalar is its REAL success metric (never a shaped reward term -
see kb/wiki/concepts/reward-hacking-and-sparse-discoverability.md), which
scalar is its stability guard, and which parameters are tunable (each a
``ParameterSpec`` giving the config field(s) to patch, baseline value,
bounds, and a hillclimb step).

Adding a new task = add one ``TaskSpace`` entry to ``TASK_SPACES`` below.
No sweep/store/strategy code changes are needed - that is the whole point of
keeping this declarative and separate from the execution machinery.

A parameter maps to a *list* of override keys, not just one, because a
single logical parameter is sometimes duplicated across several config
fields (e.g. a tolerance that appears in both a reward-term's params and a
termination-term's params); a sweep must patch all copies together or the
env becomes internally inconsistent.

Override-key syntax matches scripts/train.py's --overrides_file resolver:
a dotted path prefixed with 'env.' (applied to the env cfg) or 'agent.'
(applied to the agent/PPO cfg). Segments resolve as attributes, or as dict
keys when the current object is a dict (so reward-term ``params`` entries
like '...params.touch_std' are reachable).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    """Logical parameter name (unique within a TaskSpace); what the store,
    strategies, and reports key on."""

    keys: tuple[str, ...]
    """One or more train.py override keys this parameter drives. All are set
    to the same value in a trial."""

    baseline: float
    """The current in-source value; a trial that omits this parameter uses
    this (nothing is overridden, so train.py's own default applies)."""

    bounds: tuple[float, float]
    """(low, high) inclusive sampling/clamp range for this parameter."""

    step: float
    """Hillclimb step size. Interpreted by ``step_mode``."""

    step_mode: str = "mult"
    """'mult' (multiply/divide by ``step``) or 'add' (add/subtract ``step``)."""

    scale: str = "linear"
    """'linear' or 'log' - how random/grid sampling spaces values across
    ``bounds`` (log for things like learning rate / entropy that span
    orders of magnitude)."""

    integer: bool = False
    """True if this field must stay integral (e.g. num_steps_per_env); values
    are rounded before use."""

    def clamp(self, value: float) -> float:
        lo, hi = self.bounds
        return min(max(value, lo), hi)

    def quantize(self, value: float) -> float:
        value = self.clamp(value)
        return float(round(value)) if self.integer else value


@dataclass(frozen=True)
class TaskSpace:
    name: str
    """Short task key used on the sweep CLI and stored per trial."""

    train_flag: str
    """The scripts/train.py flag selecting this task (e.g. '--touchgoal').
    Empty string means the default task (no flag)."""

    success_metric: str
    """TensorBoard tag of the REAL success-termination rate. This is the
    only thing a trial's outcome is judged on - never a shaped reward
    scalar."""

    stability_metric: str
    """TensorBoard tag used as the automatic instability reject gate
    (value-function loss divergence)."""

    parameters: tuple[ParameterSpec, ...]

    params_by_name: dict[str, ParameterSpec] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "params_by_name", {p.name: p for p in self.parameters})

    def param(self, name: str) -> ParameterSpec:
        return self.params_by_name[name]


# Shared PPO / training parameters. Live fields on the rsl_rl runner cfg
# (tasks/ar4/agents/rsl_rl_ppo_cfg.py); patchable on whichever runner cfg
# train.py selects for the task. Reused across every task's space.
_PPO_PARAMS: tuple[ParameterSpec, ...] = (
    ParameterSpec(
        name="learning_rate",
        keys=("agent.algorithm.learning_rate",),
        baseline=1.0e-4,
        bounds=(3.0e-5, 3.0e-4),
        step=2.0,
        step_mode="mult",
        scale="log",
    ),
    ParameterSpec(
        name="entropy_coef",
        keys=("agent.algorithm.entropy_coef",),
        baseline=0.006,
        bounds=(0.001, 0.02),
        step=1.5,
        step_mode="mult",
        scale="log",
    ),
    ParameterSpec(
        name="num_steps_per_env",
        keys=("agent.num_steps_per_env",),
        baseline=24,
        bounds=(16, 48),
        step=8,
        step_mode="add",
        scale="linear",
        integer=True,
    ),
)


TASK_SPACES: dict[str, TaskSpace] = {}


def _register(space: TaskSpace) -> None:
    TASK_SPACES[space.name] = space


# --------------------------------------------------------------------------
# Experiment 25 - touch-then-goal (tasks/ar4/pickplace_touchgoal_env_cfg.py)
# --------------------------------------------------------------------------
_register(
    TaskSpace(
        name="touchgoal",
        train_flag="--touchgoal",
        success_metric="Episode_Termination/goal_reached",
        stability_metric="Loss/value_function",
        parameters=(
            ParameterSpec(
                name="milestone_bonus_weight",
                keys=("env.rewards.touch_goal_milestone_bonus.weight",),
                baseline=25.0,
                bounds=(5.0, 50.0),
                step=1.5,
                step_mode="mult",
            ),
            ParameterSpec(
                name="touch_std",
                keys=("env.rewards.touch_goal_milestone_bonus.params.touch_std",),
                baseline=0.05,
                bounds=(0.02, 0.15),
                step=0.02,
                step_mode="add",
            ),
            ParameterSpec(
                name="action_scale",
                keys=("env.actions.joint_positions.scale",),
                baseline=0.5,
                bounds=(0.1, 1.0),
                step=1.5,
                step_mode="mult",
            ),
            ParameterSpec(
                name="episode_length_s",
                keys=("env.episode_length_s",),
                baseline=20.0,
                bounds=(10.0, 40.0),
                step=5.0,
                step_mode="add",
            ),
            *_PPO_PARAMS,
        ),
    )
)


# --------------------------------------------------------------------------
# Experiment 26 - grasp/lift/goal (tasks/ar4/pickplace_graspgoal_env_cfg.py)
# --------------------------------------------------------------------------
_register(
    TaskSpace(
        name="graspgoal",
        train_flag="--graspgoal",
        success_metric="Episode_Termination/cube_reached_goal",
        stability_metric="Loss/value_function",
        parameters=(
            ParameterSpec(
                name="milestone_bonus_weight",
                keys=("env.rewards.grasp_goal_milestone_bonus.weight",),
                baseline=25.0,
                bounds=(5.0, 50.0),
                step=1.5,
                step_mode="mult",
            ),
            ParameterSpec(
                name="action_scale",
                keys=("env.actions.joint_positions.scale",),
                baseline=0.5,
                bounds=(0.1, 1.0),
                step=1.5,
                step_mode="mult",
            ),
            ParameterSpec(
                name="proximity_threshold",
                # Duplicated across the gripper action gate and both the
                # reward and termination antipodal checks would be separate
                # params; here only the action-gate copy is a live cfg field
                # (the reward/termination copies come from a module constant
                # baked into their params dicts, patched independently if
                # ever needed). Sweeping the action-gate copy alone is the
                # behaviourally dominant one (it decides when the gripper is
                # even allowed to close).
                keys=("env.actions.gripper_position.proximity_threshold",),
                baseline=0.05,
                bounds=(0.02, 0.12),
                step=0.02,
                step_mode="add",
            ),
            ParameterSpec(
                name="episode_length_s",
                keys=("env.episode_length_s",),
                baseline=30.0,
                bounds=(15.0, 45.0),
                step=5.0,
                step_mode="add",
            ),
            *_PPO_PARAMS,
        ),
    )
)


# --------------------------------------------------------------------------
# Experiment 15 - base-proximity reward shaping
# (tasks/ar4/pickplace_baseproximity_env_cfg.py). This is the task the
# original scripts/hillclimb_rewards.py hardcoded; its 6 reward params are
# reproduced here as override keys (no source edit), plus shared PPO params.
# --------------------------------------------------------------------------
_register(
    TaskSpace(
        name="baseproximity",
        train_flag="--baseproximity",
        success_metric="Episode_Termination/cube_reached_goal",
        stability_metric="Loss/value_function",
        parameters=(
            ParameterSpec(
                name="ground_penalty_weight",
                keys=("env.rewards.ground_penalty.weight",),
                baseline=0.1,
                bounds=(0.01, 2.0),
                step=1.5,
                step_mode="mult",
            ),
            ParameterSpec(
                name="ground_height_threshold",
                keys=("env.rewards.ground_penalty.params.ground_height_threshold",),
                baseline=0.015,
                bounds=(0.005, 0.05),
                step=0.005,
                step_mode="add",
            ),
            ParameterSpec(
                name="base_proximity_penalty_weight",
                keys=("env.rewards.base_proximity_penalty.weight",),
                baseline=0.1,
                bounds=(0.01, 2.0),
                step=1.5,
                step_mode="mult",
            ),
            ParameterSpec(
                name="base_xy_threshold",
                keys=("env.rewards.base_proximity_penalty.params.base_xy_threshold",),
                baseline=0.08,
                bounds=(0.02, 0.15),
                step=0.02,
                step_mode="add",
            ),
            ParameterSpec(
                name="antipodal_grasp_bonus_weight",
                keys=("env.rewards.antipodal_grasp_bonus.weight",),
                baseline=4.0,
                bounds=(1.0, 10.0),
                step=1.0,
                step_mode="add",
            ),
            ParameterSpec(
                name="stillness_penalty_weight",
                keys=("env.rewards.stillness_penalty.weight",),
                baseline=6.0,
                bounds=(1.0, 12.0),
                step=1.0,
                step_mode="add",
            ),
            *_PPO_PARAMS,
        ),
    )
)


def resolve_overrides(space: TaskSpace, param_values: dict[str, float]) -> dict[str, float]:
    """Expand a {param_name: value} vector into a flat {override_key: value}
    dict for scripts/train.py --overrides_file. A parameter with multiple
    keys sets all of them to the same value."""
    overrides: dict[str, float] = {}
    for pname, value in param_values.items():
        spec = space.param(pname)
        for key in spec.keys:
            overrides[key] = value
    return overrides


def assert_no_nan(values: dict[str, float]) -> None:
    for k, v in values.items():
        if isinstance(v, float) and math.isnan(v):
            raise ValueError(f"Parameter {k!r} resolved to NaN.")
