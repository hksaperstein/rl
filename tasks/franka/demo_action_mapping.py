# tasks/franka/demo_action_mapping.py
"""Pure-torch closed-form conversion from `scripts/dice_pick_demo.py`'s own
scripted DiffIK grasp controller's ABSOLUTE joint-position/gripper targets
to the RL env's own action space (`FrankaDieLiftJointD8BigEnvCfg`/
`...D10BigEnvCfg`'s `ActionsCfg`: `JointPositionActionCfg(scale=0.5,
use_default_offset=True)` for the 7 arm joints,
`BinaryJointPositionActionCfg` for the 2 gripper finger joints).

Task 1 of docs/superpowers/plans/2026-07-19-d8-d10-demo-warmstart-
implementation.md. NO isaaclab/pxr imports anywhere in this module - same
importable-without-Isaac-Sim split `tasks/franka/shape_observations.py`/
`tasks/franka/lift_reward.py`/`tasks/franka/distillation.py` already
establish (see `distillation.py`'s own module docstring), so
`tests/test_demo_action_mapping.py` can exercise this logic directly, no
Isaac Sim launch needed. `scripts/extract_demo_trajectory.py` (capture) and
Task 2's BC-pretrain replay driver (consumer) are the two real call sites.

=====================================================================
ARM JOINT-POSITION MAPPING - confirmed by direct source read (2026-07-19)
=====================================================================

`isaaclab.envs.mdp.actions.joint_actions.JointAction.process_actions`
(`/home/saps/IsaacLab/source/isaaclab/isaaclab/envs/mdp/actions/joint_actions.py`,
read directly on the desktop where Isaac Lab is installed, lines 164-173):

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # apply the affine transformations
        self._processed_actions = self._raw_actions * self._scale + self._offset
        ...

and `JointPositionAction.__init__` (same file, lines 187-192):

    if cfg.use_default_offset:
        self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

So for `FrankaDieLiftJointD8BigEnvCfg`/`...D10BigEnvCfg`'s `arm_action`
(`tasks/franka/dice_lift_joint_env_cfg.py`'s
`FrankaDieLiftJointEnvCfg.__post_init__`: `mdp.JointPositionActionCfg(...,
scale=0.5, use_default_offset=True)`, no `clip`):

    target_joint_pos = default_joint_pos + scale * raw_action

`default_joint_pos` is captured ONCE at env-construction time (not re-read
per step) when `use_default_offset=True` - a fixed offset, so the inverse
below is a closed-form, per-step, no-live-feedback computation directly
from a logged reference trajectory's absolute joint-position targets:

    raw_action = (target_joint_pos - default_joint_pos) / scale

=====================================================================
GRIPPER BINARY-ACTION CONVENTION - confirmed by direct source read
(2026-07-19), NOT assumed from memory
=====================================================================

`isaaclab.envs.mdp.actions.binary_joint_actions.BinaryJointAction.process_actions`
(`/home/saps/IsaacLab/source/isaaclab/isaaclab/envs/mdp/actions/binary_joint_actions.py`,
read directly on the desktop, lines 128-139):

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        if actions.dtype == torch.bool:
            # true: close, false: open
            binary_mask = actions == 0
        else:
            # true: close, false: open
            binary_mask = actions < 0
        self._processed_actions = torch.where(binary_mask, self._close_command, self._open_command)

For float raw actions (the RL env's own action dtype - never bool for this
task), the confirmed convention is: raw_action < 0 selects
`close_command_expr`; raw_action >= 0 (including exactly 0.0) selects
`open_command_expr`. `BinaryJointPositionAction` (same file, lines 149-156)
only overrides `apply_actions` (`set_joint_position_target` with whichever
command was selected) - it inherits `process_actions` (the convention
above) UNCHANGED from `BinaryJointAction`.

`FrankaDieLiftJointD8BigEnvCfg`/`...D10BigEnvCfg`'s `gripper_action` is
inherited unchanged from `tasks/franka/lift_env_cfg.py`'s `ActionsCfg`:

    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot", joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )

- the EXACT same numeric joint-position values (0.04 open, 0.0 close)
`scripts/dice_pick_demo.py`'s own `open_target`/`close_target` constants
already use for its scripted grasp's gripper joint-position targets (see
that file's `run_pick_sequence`: `open_target = torch.full(..., 0.04, ...)`,
`close_target = torch.full(..., 0.0, ...)`). A demo-logged gripper_target
is therefore directly classifiable against the midpoint of those two known
values (0.02) - no additional scale/offset math needed, and no risk of the
demo's own gripper convention silently drifting from the RL env's.
"""

from __future__ import annotations

import torch

# tasks/franka/lift_env_cfg.py's ActionsCfg.gripper_action open_command_expr/
# close_command_expr values, reused unchanged (not re-derived) - see module
# docstring. dice_pick_demo.py's own open_target/close_target constants
# already match these exactly.
_GRIPPER_OPEN_JOINT_POS = 0.04
_GRIPPER_CLOSE_JOINT_POS = 0.0
_GRIPPER_MIDPOINT = (_GRIPPER_OPEN_JOINT_POS + _GRIPPER_CLOSE_JOINT_POS) / 2.0

# Canonical raw-action magnitudes this module emits for the gripper's binary
# action term - any positive/negative value satisfies BinaryJointAction's own
# confirmed convention (see module docstring); +-1.0 chosen only for a clean,
# readable round-trip, not itself part of the confirmed convention.
_GRIPPER_OPEN_RAW = 1.0
_GRIPPER_CLOSE_RAW = -1.0


def joint_pos_to_raw_action(
    target_joint_pos: torch.Tensor, default_joint_pos: torch.Tensor, scale: float = 0.5
) -> torch.Tensor:
    """Closed-form inverse of isaaclab's `JointPositionAction` affine
    transform (`target_joint_pos = default_joint_pos + scale * raw_action`,
    confirmed by direct source read - see module docstring's "ARM
    JOINT-POSITION MAPPING" section): `raw_action = (target_joint_pos -
    default_joint_pos) / scale`. `scale` defaults to 0.5,
    `FrankaDieLiftJointD8BigEnvCfg`/`...D10BigEnvCfg`'s own `arm_action`
    cfg value, so real call sites don't have to repeat it.

    `target_joint_pos`/`default_joint_pos` may be any matching shape
    (a single `(num_joints,)` row or a batched `(num_steps, num_joints)`
    trajectory) - this is a pure elementwise affine inverse, no reduction.

    Raises `ValueError` on `scale == 0` (degenerate - the forward map
    `raw_action -> target_joint_pos` collapses every raw_action to the same
    `default_joint_pos` and is not invertible)."""
    if scale == 0:
        raise ValueError(
            "joint_pos_to_raw_action: scale=0 is degenerate (division by zero) - "
            "the forward JointPositionAction map isn't invertible at scale=0"
        )
    return (target_joint_pos - default_joint_pos) / scale


def gripper_target_to_raw_action(gripper_target: torch.Tensor) -> torch.Tensor:
    """Maps a demo-logged gripper joint-position target
    (`dice_pick_demo.py`'s own `open_target`=0.04 / `close_target`=0.0,
    per-finger-joint tensor or scalar) to the RL env's
    `BinaryJointPositionActionCfg` raw-action convention confirmed by
    direct source read (module docstring's "GRIPPER BINARY-ACTION
    CONVENTION" section): raw_action >= 0 selects `open_command_expr`,
    raw_action < 0 selects `close_command_expr`. Returns +/-1.0 per row
    (see `_GRIPPER_OPEN_RAW`/`_GRIPPER_CLOSE_RAW`'s own docstring note on
    why the exact magnitude is not itself part of the confirmed
    convention).

    `gripper_target`'s LAST dim is always treated as the per-finger-joint
    dim and reduced by mean before classifying (e.g. shape `(num_steps, 2)`
    for a batched trajectory, or `(2,)` for a single step - both finger
    joints are always commanded to the identical value by
    `dice_pick_demo.py`'s own `open_target`/`close_target`, so this mean is
    a no-op on real demo data, not a lossy approximation). Output has
    action_dim=1 (`BinaryJointAction.action_dim` is always 1, regardless of
    how many physical joints it drives), added as a new trailing dim -
    shape `(num_steps, 1)` / `(1,)` respectively for the two examples
    above."""
    reduced = gripper_target.mean(dim=-1)
    is_open = reduced >= _GRIPPER_MIDPOINT
    raw = torch.where(
        is_open,
        torch.full_like(reduced, _GRIPPER_OPEN_RAW),
        torch.full_like(reduced, _GRIPPER_CLOSE_RAW),
    )
    return raw.unsqueeze(-1)
