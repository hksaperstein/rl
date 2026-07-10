#!/usr/bin/env python3
"""
HER Feasibility Check: Toy environment to test whether HER's goal-relabeling
actually helps when the target event is rare-but-occasionally-reachable (vs
essentially never reached by chance).

Exports results as a plain-text summary table to stdout.
This script uses ONLY gymnasium + stable_baselines3 + torch, no Isaac Sim.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import SAC
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer


class SparseReachGraspToy(gym.Env):
    """
    Custom toy environment for HER feasibility testing.

    State: 2D position pos in [-1, 1]^2, plus 1D jaw value in [-1, 1].
    Action: 3D continuous delta (position_dx, position_dy, jaw_direct).
    Observation: Dict with 'observation', 'achieved_goal', 'desired_goal' keys.

    Reward: sparse binary via compute_reward().
    Episode: 50 steps, truncate after that.
    """

    metadata = {"render_modes": []}

    def __init__(self, grasp_lo=-0.3, grasp_hi=0.3, episode_length=50, seed=None):
        """
        Args:
            grasp_lo, grasp_hi: jaw bounds for valid grasp (configurable)
            episode_length: steps before truncation
            seed: random seed
        """
        super().__init__()

        self.grasp_lo = grasp_lo
        self.grasp_hi = grasp_hi
        self.episode_length = episode_length
        self.max_pos_delta = 0.05
        self.target_pos = np.array([0.6, 0.6], dtype=np.float32)
        self.reach_threshold = 0.05

        # State
        self.pos = None
        self.jaw = None
        self._grasped = False
        self.step_count = 0

        # Action space: 3D, position delta + direct jaw
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # Observation space: Dict with observation, achieved_goal, desired_goal
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low=-2.0, high=2.0, shape=(4,), dtype=np.float32),
            "achieved_goal": spaces.Box(low=-2.0, high=2.0, shape=(3,), dtype=np.float32),
            "desired_goal": spaces.Box(low=-2.0, high=2.0, shape=(3,), dtype=np.float32),
        })

        self.rng = np.random.RandomState(seed)

        # Tracking for analysis
        self.training_grasps_achieved = 0

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        if seed is not None:
            self.rng.seed(seed)

        self.pos = self.rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
        self.jaw = np.float32(self.rng.uniform(-1.0, 1.0))
        self._grasped = False
        self.step_count = 0

        return self._get_obs(), {}

    def _get_obs(self):
        """Return the observation dict."""
        achieved_goal = np.array([self.pos[0], self.pos[1], float(self._grasped)], dtype=np.float32)
        desired_goal = np.array([self.target_pos[0], self.target_pos[1], 1.0], dtype=np.float32)
        observation = np.concatenate([self.pos, [self.jaw], [float(self._grasped)]]).astype(np.float32)

        return {
            "observation": observation,
            "achieved_goal": achieved_goal,
            "desired_goal": desired_goal,
        }

    def _check_reach(self):
        """Check if position is close to target."""
        return np.linalg.norm(self.pos - self.target_pos) < self.reach_threshold

    def _check_grasp(self):
        """Check if grasp condition is satisfied (must be at reach + jaw in range)."""
        if not self._check_reach():
            return False
        return self.grasp_lo <= self.jaw <= self.grasp_hi

    def step(self, action):
        """
        Execute one step.

        Returns:
            obs, reward, terminated, truncated, info
        """
        # Clip action
        action = np.clip(action, -1.0, 1.0)

        # Update position: first 2 components of action are position delta
        pos_delta = action[:2] * self.max_pos_delta
        self.pos = np.clip(self.pos + pos_delta, -1.0, 1.0).astype(np.float32)

        # Update jaw: 3rd component directly sets jaw value
        self.jaw = np.clip(action[2], -1.0, 1.0).astype(np.float32)

        # Check for grasp achievement (latch it)
        was_grasped = self._grasped
        if self._check_grasp():
            self._grasped = True

        # Track training-time grasp achievements
        if self._grasped and not was_grasped:
            self.training_grasps_achieved += 1

        # Compute reward via the method
        obs = self._get_obs()
        reward = self.compute_reward(
            obs["achieved_goal"],
            obs["desired_goal"],
            {}
        )

        self.step_count += 1

        # Check termination: newly grasped (optional), or episode length exceeded
        terminated = (self._grasped and not was_grasped)
        truncated = (self.step_count >= self.episode_length)

        return obs, float(reward), terminated, truncated, {}

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Compute reward for HER.

        Args:
            achieved_goal: shape (3,) or (N, 3)
            desired_goal: shape (3,) or (N, 3)
            info: dict (unused)

        Returns:
            reward: scalar or (N,) array
        """
        # Handle both scalar and batched calls
        if achieved_goal.ndim == 1:
            distance = np.linalg.norm(achieved_goal - desired_goal)
            return 0.0 if distance < 0.05 else -1.0
        else:
            # Batched: shape (N, 3)
            distances = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
            return np.where(distances < 0.05, 0.0, -1.0)


def run_experiment(condition_name, grasp_lo, grasp_hi, use_her, seed=0, timesteps=30000):
    """
    Run a single experiment: train a policy and evaluate it.

    Returns:
        dict with results
    """
    print(f"\n=== Running: {condition_name} (use_HER={use_her}) ===")

    # Create environment
    env = SparseReachGraspToy(grasp_lo=grasp_lo, grasp_hi=grasp_hi, seed=seed)
    env = TimeLimit(env, max_episode_steps=50)

    # Create policy
    if use_her:
        policy_kwargs = {"net_arch": [256, 256]}
        replay_buffer_class = HerReplayBuffer
        replay_buffer_kwargs = {
            "goal_selection_strategy": "future",
            "n_sampled_goal": 4,
        }
        model = SAC(
            "MultiInputPolicy",
            env,
            policy_kwargs=policy_kwargs,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            verbose=0,
            seed=seed,
            learning_starts=1000,
        )
    else:
        policy_kwargs = {"net_arch": [256, 256]}
        model = SAC(
            "MultiInputPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=seed,
            learning_starts=1000,
        )

    # Train
    print(f"Training for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps)

    # Count training-time grasp achievements
    train_grasps = env.unwrapped.training_grasps_achieved
    train_grasp_rate = train_grasps / (timesteps / 50)  # Approximate by assuming ~50 steps per episode

    print(f"Training-time grasp achievements: {train_grasps} (estimated ~{train_grasp_rate:.2%})")

    # Evaluate on 100 episodes
    print("Evaluating on 100 deterministic episodes...")
    eval_successes = 0

    for episode_idx in range(100):
        obs, info = env.reset()
        done = False
        truncated = False

        for step in range(50):
            # Deterministic action (mean of policy)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

            if done or truncated:
                break

        # Check if grasped by end
        if obs["achieved_goal"][2] >= 0.5:  # Grasp latch is True
            eval_successes += 1

    eval_rate = eval_successes / 100.0
    print(f"Eval success rate: {eval_successes}/100 ({eval_rate:.2%})")

    env.close()

    return {
        "condition": condition_name,
        "use_her": use_her,
        "train_grasps_achieved": train_grasps,
        "train_grasp_rate": train_grasp_rate,
        "eval_success_rate": eval_rate,
        "eval_successes": eval_successes,
    }


def main():
    """Run all 4 experiments and print summary."""

    print("=" * 80)
    print("HER Feasibility Check: Rare Event Learning")
    print("=" * 80)

    results = []

    # Condition A: Easy window (60% of jaw range)
    print("\n--- CONDITION A: Easy Grasp Window (grasp_lo=-0.3, grasp_hi=0.3) ---")
    results.append(run_experiment("A_Vanilla", grasp_lo=-0.3, grasp_hi=0.3, use_her=False, seed=0))
    results.append(run_experiment("A_HER", grasp_lo=-0.3, grasp_hi=0.3, use_her=True, seed=0))

    # Condition B: Hard window (4% of jaw range)
    print("\n--- CONDITION B: Hard Grasp Window (grasp_lo=-0.02, grasp_hi=0.02) ---")
    results.append(run_experiment("B_Vanilla", grasp_lo=-0.02, grasp_hi=0.02, use_her=False, seed=0))
    results.append(run_experiment("B_HER", grasp_lo=-0.02, grasp_hi=0.02, use_her=True, seed=0))

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Condition':<15} {'Method':<10} {'Train Grasps':<15} {'Train Rate':<12} {'Eval Success':<15}")
    print("-" * 80)

    for result in results:
        cond = result["condition"]
        method = "HER" if result["use_her"] else "Vanilla"
        train_count = result["train_grasps_achieved"]
        train_rate = result["train_grasp_rate"]
        eval_success = f"{result['eval_successes']}/100"

        print(f"{cond:<15} {method:<10} {train_count:<15} {train_rate:<12.2%} {eval_success:<15}")

    print("\n" + "=" * 80)
    print("End of report.")
    print("=" * 80)


if __name__ == "__main__":
    main()
