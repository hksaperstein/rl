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


@configclass
class FrankaLiftRelativeJointPPORunnerCfg(FrankaLiftPPORunnerCfg):
    """Variant-specific PPO runner cfg for
    `FrankaDieLiftJointD8BigRelativeAntipodalEnvCfg` only (2026-07-21,
    d8-relative-joint-action plan Task 3 real-run divergence) - the shared
    `FrankaLiftPPORunnerCfg` default is NOT modified, per the plan's Global
    Constraints ("do not build a contingency PPORunnerCfg subclass
    preemptively; only if Task 3's real run actually shows divergence,
    scoped to this variant only").

    Real critic divergence was observed empirically, not hypothesized: a
    real 3-seed/1500-iteration cloud run (seed 42 succeeded cleanly at
    scale=0.1; seed 123 diverged) showed `Mean value_function loss` collapse
    from a stable ~0.001-0.003 straight to `inf` within 6 PPO update
    iterations (iteration ~298/1500 - `181.0 -> 2.2e8 -> 3.8e14 -> 6.0e20 ->
    9.7e26 -> 1.5e33 -> inf`), immediately followed by a
    `RuntimeError: normal expects all elements of std >= 0.0` crash (a NaN/Inf-
    corrupted policy std) - the exact `[[ppo-critic-divergence]]` signature
    already catalogued from Experiment 11's task-space/IK swap
    (`kb/wiki/concepts/ppo-critic-divergence.md`): a new action-term
    mechanism (here, `RelativeJointPositionAction`'s
    `applied_target = raw_action * scale + current_joint_pos`, read fresh
    every step) lets one outlier raw policy action - more likely once the
    policy's own exploration std grows, observed pinned at 1.50 immediately
    before this divergence - produce a large-enough single-step joint delta
    to destabilize the critic's value estimate for that transition.

    Fix, applied by exact analogy to Experiment 11's own established fix
    (not a fresh guess): `clip_actions=5.0` (~3.3x the 1.50 action-noise std
    observed immediately pre-divergence, same margin-above-observed-std
    Experiment 11 itself used - 5.0 vs its own 1.46 std), clamping the raw
    policy action before `RelativeJointPositionAction`'s scale/offset math
    ever sees it (confirmed by direct read of
    `isaaclab_rl/rsl_rl/vecenv_wrapper.py:154-155` in this project's pinned
    Isaac Lab checkout - `RslRlVecEnvWrapper.__init__`'s own `clip_actions`
    param, applied via `torch.clamp` on the raw action tensor). `clip_actions`
    is a field on `RslRlBaseRunnerCfg` itself (the top-level runner cfg,
    read as `agent_cfg.clip_actions` by `train_franka.py`/
    `franka_checkpoint_review.py`/`diag_antipodal_root_cause.py`), NOT on
    the nested `RslRlPpoAlgorithmCfg` (confirmed by direct read of
    `isaaclab_rl/rsl_rl/rl_cfg.py:135-178` class boundaries) - an initial
    version of this fix mis-added it as an `RslRlPpoAlgorithmCfg(...)`
    kwarg and failed fast with `TypeError: RslRlPpoAlgorithmCfg.__init__()
    got an unexpected keyword argument 'clip_actions'` on the very next
    real-run attempt; caught and corrected in this same pass before further
    training time was spent.
    `scale=0.1` itself is untouched (per the plan's own "not tunable
    mid-experiment" - the fix here is a PPO-runner-level safety clip, not an
    action-scale change) - the env cfg
    (`FrankaDieLiftJointD8BigRelativeAntipodalEnvCfg`) is not modified by
    this fix at all.

    All other `algorithm`/`policy` fields are copied byte-identical from
    `FrankaLiftPPORunnerCfg` - only the top-level `clip_actions` is added.
    Applied to ALL THREE seeds (42/123/7), including a re-run of seed 42
    (which had already completed cleanly under the unclipped config) -
    re-run for a clean, directly-comparable 3-seed set all trained under
    the identical corrected runner cfg, since `clip_actions=5.0` is a
    generous bound expected to be a no-op for any already-well-behaved
    trajectory (never triggers unless an action actually exceeds +-5.0),
    not a behavior change for seed 42's own already-clean run.
    """

    clip_actions = 5.0
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
