# tasks/franka/agents/rsl_rl_ppo_cfg.py
"""PPO (rsl_rl) hyperparameters for the from-scratch Franka Panda cube-lift
task, written fresh for the franka-panda-pivot.

Matches Isaac Lab's own official Franka lift-task PPO config exactly
(isaaclab_tasks/manager_based/manipulation/lift/config/franka/agents/
rsl_rl_ppo_cfg.py, read directly - not imported, per the pivot's
"everything new" instruction), so the bounded convergence probe is a clean
test of "does Franka + the stock recipe converge as published" rather than
a confound introduced by different hyperparameters. One addition beyond
the official file: `obs_groups` is set explicitly, because this repo's
installed isaaclab_rl version (isaaclab_rl/rsl_rl/rl_cfg.py) declares it as
a MANDATORY field (`MISSING`, no default) on RslRlBaseRunnerCfg, unlike
whatever isaaclab_rl version Isaac Lab's own reference file was written
against - confirmed by reading rl_cfg.py directly rather than guessing.
Follows tasks/ar4/agents/rsl_rl_ppo_cfg.py's own precedent for setting
this same field (style reference only, not imported).
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class FrankaLiftPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "franka_lift"
    # A single "policy" observation group; both actor and critic read from it.
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
