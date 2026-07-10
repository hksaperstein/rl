"""Pure-tensor reward math for Franka cube-lift task — reproduces Isaac Lab's
own official stock Franka-lift reward functions, reimplemented fresh with no
isaaclab imports (just torch), so testable via plain pytest+torch. Follows the
pure-function pattern established by tasks/ar4/grasp_goal_reward.py.
tasks/franka/mdp.py reads live simulated state and delegates the actual reward
computation to the functions below."""

import torch


def reaching_object_reward(cube_pos: torch.Tensor, ee_pos: torch.Tensor, std: float) -> torch.Tensor:
    """End-effector-to-cube proximity reward: 1 - tanh(distance / std).

    Rewards reducing the Euclidean distance from end-effector to cube center
    using a smooth tanh-based kernel. Maximum reward 1.0 at distance 0;
    reward decays smoothly as distance increases.

    Args:
        cube_pos: Cube position in world frame, shape (N, 3) where N is batch size.
        ee_pos: End-effector position in world frame, shape (N, 3).
        std: Length scale for the tanh kernel (distance at which reward ~= 0.5).

    Returns:
        Reward tensor, shape (N,), values in [0, 1].
    """
    distance = torch.norm(cube_pos - ee_pos, dim=1)
    return 1.0 - torch.tanh(distance / std)


def lifting_object_reward(cube_height: torch.Tensor, minimal_height: float) -> torch.Tensor:
    """Binary reward for lifting cube above a minimum height threshold.

    Returns 1.0 once the cube's z-coordinate exceeds minimal_height, else 0.0.
    Provides a discrete, non-shaped gate for the lifting stage of the task.

    Args:
        cube_height: Cube z-position (height) in world frame, shape (N,).
        minimal_height: Threshold height above which cube is considered "lifted".

    Returns:
        Reward tensor, shape (N,), values either 0.0 or 1.0. Uses strict >
        comparison (not >=), matching Isaac Lab's reference implementation.
    """
    return torch.where(cube_height > minimal_height, 1.0, 0.0)


def object_goal_distance_reward(
    cube_pos: torch.Tensor,
    goal_pos: torch.Tensor,
    cube_height: torch.Tensor,
    minimal_height: float,
    std: float,
) -> torch.Tensor:
    """Cube-to-goal proximity reward, gated by cube lifting state.

    Reward is 0.0 when cube_height <= minimal_height (cube not yet lifted);
    when cube_height > minimal_height, reward follows the 1 - tanh(distance / std)
    kernel applied to cube-to-goal Euclidean distance. The gate is applied
    per-batch-element (row-wise), enabling mixed batches of lifted and
    not-yet-lifted cubes.

    Args:
        cube_pos: Cube position in world frame, shape (N, 3).
        goal_pos: Goal position in world frame, shape (N, 3).
        cube_height: Cube z-position (height) in world frame, shape (N,).
        minimal_height: Height threshold below which reward is gated to 0.0.
        std: Length scale for the tanh kernel in the goal distance term.

    Returns:
        Reward tensor, shape (N,), values in [0, 1]. Exactly 0.0 where
        cube_height <= minimal_height, else in [0, 1] based on tanh kernel.
    """
    distance = torch.norm(goal_pos - cube_pos, dim=1)
    is_lifted = (cube_height > minimal_height).float()
    return is_lifted * (1.0 - torch.tanh(distance / std))
