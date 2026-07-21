"""Train a PPO policy (rsl_rl) for the Franka Panda cube-lift task.

This script trains on the stock-recipe Franka Panda environment with relative-IK
action space.

.. code-block:: bash

    cd ~/projects/rl
    /home/saps/IsaacLab/isaaclab.sh -p scripts/train_franka.py --num_envs 4096
    # bounded probe (real GUI window expected per current instruction, not --headless):
    /home/saps/IsaacLab/isaaclab.sh -p scripts/train_franka.py --num_envs 64 --max_iterations 300
    # resume a previously-interrupted run from its last checkpoint, continuing on to the SAME
    # absolute --max_iterations target (not +max_iterations more iterations on top of it):
    /home/saps/IsaacLab/isaaclab.sh -p scripts/train_franka.py --num_envs 4096 --max_iterations 5000 \
        --checkpoint logs/train_franka/2026-07-09_22-05-51/model_800.pt
    # RL-fine-tune a distilled (BC) checkpoint with no meaningful optimizer state - --policy_only_checkpoint
    # skips restoring PPO optimizer state (Task 6, docs/superpowers/plans/2026-07-16-unified-multi-die-
    # specialist-distillation.md):
    /home/saps/IsaacLab/isaaclab.sh -p scripts/train_franka.py --variant joint-die-d12-d20-mixed \
        --num_envs 4096 --checkpoint distilled-d12-d20/model_1499.pt --policy_only_checkpoint \
        --max_iterations 2999
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train the Franka Panda cube-lift policy with PPO (rsl_rl).")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of parallel environments.")
parser.add_argument("--max_iterations", type=int, default=None, help="Override the agent config's max_iterations.")
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help=(
        "Path to an rsl_rl checkpoint (.pt) to resume from - restores model + optimizer state and the "
        "checkpoint's own recorded iteration count via rsl_rl.OnPolicyRunner.load(), so training continues "
        "instead of restarting from scratch. --max_iterations is the ABSOLUTE target iteration count when "
        "resuming (e.g. resuming a checkpoint saved at iteration 800 with --max_iterations 5000 runs "
        "iterations 800->5000, not 800->5800 - OnPolicyRunner.learn()'s own num_learning_iterations argument "
        "is iterations-from-here, not an absolute target, so this script converts between the two). Writes "
        "to a NEW timestamped log_dir regardless (the checkpoint's weights/optimizer/iteration are what "
        "carry over - its previous TensorBoard event file is not appended to)."
    ),
)
parser.add_argument(
    "--policy_only_checkpoint",
    action="store_true",
    default=False,
    help=(
        "Pass load_optimizer=False to rsl_rl.OnPolicyRunner.load() instead of the default True. Needed for "
        "--checkpoint paths that carry no meaningful PPO optimizer state - e.g. "
        "scripts/distill_specialists.py's save_student_checkpoint() output (Task 6 of "
        "docs/superpowers/plans/2026-07-16-unified-multi-die-specialist-distillation.md fine-tunes a "
        "distilled/BC checkpoint via fresh PPO), which intentionally writes an EMPTY optimizer_state_dict "
        "since the distillation run's own BC optimizer's Adam moments have nothing to do with PPO's. Without "
        "this flag, runner.load()'s default load_optimizer=True unconditionally calls "
        "self.alg.optimizer.load_state_dict({}) and crashes with a KeyError - the same failure mode already "
        "fixed for the eval-only scripts/franka_checkpoint_review.py (see its own load_optimizer=False "
        "comment). Leave unset (default False->load_optimizer=True) for a genuine same-run PPO resume (e.g. "
        "SPOT-preemption recovery), where restoring Adam's own momentum/variance state is the whole point of "
        "resuming instead of restarting."
    ),
)
parser.add_argument("--video", action="store_true", default=False, help="Record videos periodically during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of each recorded video (steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Steps between recorded videos.")
parser.add_argument(
    "--variant",
    choices=[
        "ik-cube",
        "joint-die",
        "joint-cube",
        "joint-die-heavy",
        "joint-die-big",
        "joint-cube-baked",
        "joint-die-mixed",
        "joint-die-mid",
        "joint-die-d8-std",
        "joint-die-d10-std",
        "joint-die-d12-std",
        "joint-die-random-size",
        "joint-die-d8-big",
        "joint-die-d8-big-exploration-bonus",
        "joint-die-d8-big-antipodal",
        "die-d8-big-taskspace-antipodal",
        "joint-die-d8-big-relative-antipodal",
        "joint-die-d10-big",
        "joint-die-d12-big",
        "joint-die-d12-d20-mixed",
        "joint-die-target-selection-so",
        "joint-die-target-selection-d1",
        "joint-die-target-selection-d2",
    ],
    default="ik-cube",
    help=(
        "ik-cube: the existing stock-recipe cube-lift with relative-IK actions (default, unchanged). "
        "joint-die: d20-die lift with direct joint-position actions (no IK) - see "
        "docs/superpowers/specs/2026-07-11-joint-space-die-lift-design.md. "
        "joint-cube: the spec's fallback rung - joint-position actions with the recipe's own DexCube "
        "(asset-vs-recipe isolation). "
        "joint-die-heavy: asset-bisect rung 1 - the d20 at DexCube's measured 0.216kg mass "
        "(docs/superpowers/specs/2026-07-12-asset-bisect-design.md). "
        "joint-die-big: asset-bisect rung 2 - the d20 scaled to DexCube's measured 48.0mm size, "
        "mass pinned at 0.216kg (docs/superpowers/specs/2026-07-12-asset-bisect-design.md). "
        "joint-cube-baked: asset-bisect rung 3 - a flat-faced cube baked through this repo's own "
        "bake_die_asset.py pipeline at 48.0mm/0.216kg, isolating shape from pipeline provenance "
        "(docs/superpowers/specs/2026-07-12-asset-bisect-design.md). "
        "joint-die-mixed: size-curriculum primary arm - per-env d20 size varied across "
        "{48.0,43.6,39.1,34.7,30.3}mm (deterministic round-robin), mass pinned 0.216kg "
        "(docs/superpowers/specs/2026-07-13-size-curriculum-design.md). "
        "joint-die-mid: size-curriculum staged-anneal fallback, stage 2 - the d20 scaled to 39.1mm, "
        "mass pinned at 0.216kg, meant to be --checkpoint-resumed from a joint-die-big run and itself "
        "resumed onward into joint-die-heavy (docs/superpowers/specs/2026-07-13-size-curriculum-design.md "
        "Verdict section). "
        "joint-die-d8-std: multi-die specialist (Task 2) - physics-baked d8 die at its real standard "
        "~16mm size, mass pinned at 0.216kg (docs/superpowers/plans/2026-07-16-unified-multi-die-"
        "specialist-distillation.md). "
        "joint-die-d10-std: multi-die specialist (Task 2) - physics-baked d10 die at its real standard "
        "~16mm face-to-face size, mass pinned at 0.216kg (docs/superpowers/plans/2026-07-16-unified-multi-"
        "die-specialist-distillation.md). "
        "joint-die-d12-std: multi-die specialist (Task 2) - physics-baked d12 die at its real standard "
        "~18mm face-to-face size, mass pinned at 0.216kg (docs/superpowers/plans/2026-07-16-unified-multi-"
        "die-specialist-distillation.md). "
        "joint-die-random-size: d20 size-DR + geometry-feature retry (Task 3) - per-env d20 size fixed at "
        "scene-spawn time via MultiAssetSpawnerCfg(random_choice=True) across {22.0,28.5,35.0,41.5,48.0}mm, "
        "mass pinned at 0.216kg; NOT per-episode resampling - see FrankaDieLiftJointRandomSizeEnvCfg's own "
        "docstring (docs/superpowers/plans/2026-07-16-unified-multi-die-specialist-distillation.md). "
        "joint-die-d8-big/d10-big/d12-big: 48mm-parity gate (Task 3.5) - physics-baked d8/d10/d12 die each "
        "scaled to its OWN freshly-derived 48.0mm-targeting scale (NOT the d20 Big rung's 0.001585 constant, "
        "which does not transfer across shapes), single undiluted 48mm population per shape/seed, mass pinned "
        "at 0.216kg - directly comparable to the asset-bisect's own cube (3/3) and d20 (1/3) 48mm baselines "
        "(docs/superpowers/plans/2026-07-16-unified-multi-die-specialist-distillation.md, "
        ".superpowers/sdd/task-3.5-brief.md). "
        "joint-die-d8-big-exploration-bonus: H1 (GRM D=1 action-dependent gripper-closure-attempt exploration "
        "bonus) grasp-discoverability test (Task 2) - IDENTICAL to joint-die-d8-big (same scene/object/"
        "observations/actions/events/terminations/PPO recipe) except its RewardsCfg adds two new terms, "
        "gripper_closure_attempt_bonus + gripper_closure_attempt_bonus_correction "
        "(FrankaDieLiftJointD8BigExplorationBonusEnvCfg, docs/superpowers/plans/2026-07-19-exploration-bonus-"
        "grasp-discovery-implementation.md; spec: docs/superpowers/specs/2026-07-19-exploration-bonus-grasp-"
        "discovery-design.md). "
        "joint-die-d8-big-antipodal: H_joint / Condition A (Task 2) - IDENTICAL to joint-die-d8-big (same "
        "48mm-parity d8 object/scale/mass, 41-dim observations, JOINT-SPACE actions, events, terminations, "
        "PPO recipe) except its scene adds two new panda_leftfinger/panda_rightfinger ContactSensorCfg fields "
        "and its RewardsCfg adds one new term, antipodal_grasp_quality (a bilateral force-closure/antipodal "
        "grasp-quality bonus ported from AR4's Experiments 9-11, refit to this scene's real mu=0.5). Arm "
        "control stays joint-space, unchanged from joint-die-d8-big - this is the JOINT-SPACE half of a "
        "two-condition test (FrankaDieLiftJointD8BigAntipodalEnvCfg, docs/superpowers/plans/2026-07-20-d8-"
        "antipodal-grasp-quality-implementation.md; spec: docs/superpowers/specs/2026-07-20-d8-antipodal-"
        "grasp-quality-design.md). "
        "die-d8-big-taskspace-antipodal: H_taskspace / Condition B (Task 2) - IDENTICAL scene/object/rewards "
        "to joint-die-d8-big-antipodal (same new ContactSensorCfg wiring + antipodal_grasp_quality reward "
        "term) but under TASK-SPACE/relative-differential-IK arm control instead of joint-space - the exact "
        "stock DifferentialInverseKinematicsActionCfg recipe (FrankaLiftEnvCfg.ActionsCfg.arm_action's own "
        "values) re-asserted after the joint-space-defaulting __post_init__ chain runs. Tests whether this "
        "project's own AR4-era finding (antipodal signal requires task-space control, Experiments 9->10->11) "
        "transfers onto Franka/d8 (FrankaDieLiftD8BigTaskspaceAntipodalEnvCfg, same plan/spec as "
        "joint-die-d8-big-antipodal above - NOT gated on that condition's own result, both run to completion "
        "unconditionally). "
        "joint-die-d8-big-relative-antipodal: H_relative (Task 1) - IDENTICAL scene/object/rewards/observations/"
        "events/terminations/PPO recipe to joint-die-d8-big-antipodal (same new ContactSensorCfg wiring + "
        "antipodal_grasp_quality reward term) but under RELATIVE/delta joint-space arm control "
        "(RelativeJointPositionActionCfg, scale=0.1, use_zero_offset=True) instead of Condition A's inherited "
        "ABSOLUTE JointPositionActionCfg - stays genuinely joint-space, isolating delta-vs-absolute action "
        "semantics from the joint-space-vs-task-space axis joint-die-d8-big-antipodal/die-d8-big-taskspace-"
        "antipodal already tested. Tests whether this fixes the root-cause doc's own diagnosed "
        "configuration-dependent absolute-target collapse mechanism (FrankaDieLiftJointD8BigRelativeAntipodalEnvCfg, "
        "docs/superpowers/plans/2026-07-20-d8-relative-joint-action-implementation.md; spec: "
        "docs/superpowers/specs/2026-07-20-d8-relative-joint-action-design.md). "
        "joint-die-d12-d20-mixed: Task 6 RL fine-tune env - the same ONE-env, deterministic-round-robin "
        "d12/d20 mixed-population env Task 5's distillation training ran against "
        "(FrankaDieLiftJointD12D20MixedEnvCfg, tasks/franka/dice_lift_joint_env_cfg.py), meant to be "
        "--checkpoint-resumed from Task 5's distilled student weights via --policy_only_checkpoint "
        "(docs/superpowers/plans/2026-07-16-unified-multi-die-specialist-distillation.md Task 6). "
        "joint-die-target-selection-so: target-selection-in-clutter curriculum Stage SO (0 active "
        "distractors) - the full 3-die scene topology (target + 2 always-present-but-PARKED distractor "
        "bodies), 43-dim observation schema (Task 2's distractor_distance_summary term added), trained "
        "FROM SCRATCH (not resumed from model_2998.pt - dimensionality mismatch, 41 vs 43 dims). Internal "
        "sanity gate before Stage D1/D2 (FrankaDieLiftJointD12D20TargetSelectionSOEnvCfg, "
        "docs/superpowers/plans/2026-07-19-target-selection-clutter-implementation.md Task 4, "
        "docs/superpowers/specs/2026-07-19-target-selection-clutter-design.md). "
        "joint-die-target-selection-d1: Stage D1 (1 active distractor, drawn from {d12,d20} via "
        "MultiAssetSpawnerCfg(random_choice=True)) - meant to be --checkpoint-resumed from Stage SO's own "
        "checkpoint (same 43-dim schema, a normal same-dimensionality PPO resume, no "
        "--policy_only_checkpoint needed) (FrankaDieLiftJointD12D20TargetSelectionD1EnvCfg, "
        "docs/superpowers/plans/2026-07-19-target-selection-clutter-implementation.md Task 5). "
        "joint-die-target-selection-d2: Stage D2 (2 active distractors, both independently drawn from "
        "{d12,d20}) - the experiment's target configuration and primary falsification check, meant to be "
        "--checkpoint-resumed from Stage D1's own checkpoint "
        "(FrankaDieLiftJointD12D20TargetSelectionD2EnvCfg, "
        "docs/superpowers/plans/2026-07-19-target-selection-clutter-implementation.md Task 6)."
    ),
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Override the PPO runner cfg's seed (asset-bisect 3-seed protocol; default: keep agent cfg's own).",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import sys
from datetime import datetime

import gymnasium as gym

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tasks.franka.agents.rsl_rl_ppo_cfg import FrankaLiftPPORunnerCfg  # noqa: E402
from tasks.franka.lift_env_cfg import FrankaLiftEnvCfg  # noqa: E402

LOG_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "train_franka")


def main() -> None:
    if args_cli.variant == "joint-die":
        from tasks.franka.dice_lift_joint_env_cfg import FrankaDieLiftJointEnvCfg

        env_cfg = FrankaDieLiftJointEnvCfg()
    elif args_cli.variant == "joint-cube":
        from tasks.franka.dice_lift_joint_env_cfg import FrankaCubeLiftJointEnvCfg

        env_cfg = FrankaCubeLiftJointEnvCfg()
    elif args_cli.variant == "joint-die-heavy":
        from tasks.franka.dice_lift_joint_env_cfg import FrankaDieLiftJointHeavyEnvCfg

        env_cfg = FrankaDieLiftJointHeavyEnvCfg()
    elif args_cli.variant == "joint-die-big":
        from tasks.franka.dice_lift_joint_env_cfg import FrankaDieLiftJointBigEnvCfg

        env_cfg = FrankaDieLiftJointBigEnvCfg()
    elif args_cli.variant == "joint-cube-baked":
        from tasks.franka.dice_lift_joint_env_cfg import FrankaCubeBakedLiftJointEnvCfg

        env_cfg = FrankaCubeBakedLiftJointEnvCfg()
    elif args_cli.variant == "joint-die-mixed":
        from tasks.franka.dice_lift_joint_env_cfg import FrankaDieLiftJointMixedEnvCfg

        env_cfg = FrankaDieLiftJointMixedEnvCfg()
    elif args_cli.variant == "joint-die-mid":
        from tasks.franka.dice_lift_joint_env_cfg import FrankaDieLiftJointMidEnvCfg

        env_cfg = FrankaDieLiftJointMidEnvCfg()
    elif args_cli.variant == "joint-die-d8-std":
        from tasks.franka.dice_lift_joint_env_cfg import FrankaDieLiftJointD8StandardEnvCfg

        env_cfg = FrankaDieLiftJointD8StandardEnvCfg()
    elif args_cli.variant == "joint-die-d10-std":
        from tasks.franka.dice_lift_joint_env_cfg import FrankaDieLiftJointD10StandardEnvCfg

        env_cfg = FrankaDieLiftJointD10StandardEnvCfg()
    elif args_cli.variant == "joint-die-d12-std":
        from tasks.franka.dice_lift_joint_env_cfg import FrankaDieLiftJointD12StandardEnvCfg

        env_cfg = FrankaDieLiftJointD12StandardEnvCfg()
    elif args_cli.variant == "joint-die-random-size":
        from tasks.franka.dice_lift_joint_env_cfg import FrankaDieLiftJointRandomSizeEnvCfg

        env_cfg = FrankaDieLiftJointRandomSizeEnvCfg()
    elif args_cli.variant == "joint-die-d8-big":
        from tasks.franka.dice_lift_joint_env_cfg import FrankaDieLiftJointD8BigEnvCfg

        env_cfg = FrankaDieLiftJointD8BigEnvCfg()
    elif args_cli.variant == "joint-die-d8-big-exploration-bonus":
        from tasks.franka.dice_lift_joint_env_cfg import FrankaDieLiftJointD8BigExplorationBonusEnvCfg

        env_cfg = FrankaDieLiftJointD8BigExplorationBonusEnvCfg()
    elif args_cli.variant == "joint-die-d8-big-antipodal":
        from tasks.franka.dice_lift_joint_env_cfg import FrankaDieLiftJointD8BigAntipodalEnvCfg

        env_cfg = FrankaDieLiftJointD8BigAntipodalEnvCfg()
    elif args_cli.variant == "die-d8-big-taskspace-antipodal":
        from tasks.franka.dice_lift_joint_env_cfg import FrankaDieLiftD8BigTaskspaceAntipodalEnvCfg

        env_cfg = FrankaDieLiftD8BigTaskspaceAntipodalEnvCfg()
    elif args_cli.variant == "joint-die-d8-big-relative-antipodal":
        from tasks.franka.dice_lift_joint_env_cfg import FrankaDieLiftJointD8BigRelativeAntipodalEnvCfg

        env_cfg = FrankaDieLiftJointD8BigRelativeAntipodalEnvCfg()
    elif args_cli.variant == "joint-die-d10-big":
        from tasks.franka.dice_lift_joint_env_cfg import FrankaDieLiftJointD10BigEnvCfg

        env_cfg = FrankaDieLiftJointD10BigEnvCfg()
    elif args_cli.variant == "joint-die-d12-big":
        from tasks.franka.dice_lift_joint_env_cfg import FrankaDieLiftJointD12BigEnvCfg

        env_cfg = FrankaDieLiftJointD12BigEnvCfg()
    elif args_cli.variant == "joint-die-d12-d20-mixed":
        from tasks.franka.dice_lift_joint_env_cfg import FrankaDieLiftJointD12D20MixedEnvCfg

        env_cfg = FrankaDieLiftJointD12D20MixedEnvCfg()
    elif args_cli.variant == "joint-die-target-selection-so":
        from tasks.franka.dice_lift_joint_env_cfg import FrankaDieLiftJointD12D20TargetSelectionSOEnvCfg

        env_cfg = FrankaDieLiftJointD12D20TargetSelectionSOEnvCfg()
    elif args_cli.variant == "joint-die-target-selection-d1":
        from tasks.franka.dice_lift_joint_env_cfg import FrankaDieLiftJointD12D20TargetSelectionD1EnvCfg

        env_cfg = FrankaDieLiftJointD12D20TargetSelectionD1EnvCfg()
    elif args_cli.variant == "joint-die-target-selection-d2":
        from tasks.franka.dice_lift_joint_env_cfg import FrankaDieLiftJointD12D20TargetSelectionD2EnvCfg

        env_cfg = FrankaDieLiftJointD12D20TargetSelectionD2EnvCfg()
    else:
        env_cfg = FrankaLiftEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    agent_cfg = FrankaLiftPPORunnerCfg()
    agent_cfg.device = args_cli.device
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations

    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed

    env_cfg.seed = agent_cfg.seed

    _log_suffix = {
        "ik-cube": "",
        "joint-die": "_jointdie",
        "joint-cube": "_jointcube",
        "joint-die-heavy": "_jointdieheavy",
        "joint-die-big": "_jointdiebig",
        "joint-cube-baked": "_jointcubebaked",
        "joint-die-mixed": "_jointdiemixed",
        "joint-die-mid": "_jointdiemid",
        "joint-die-d8-std": "_jointdied8std",
        "joint-die-d10-std": "_jointdied10std",
        "joint-die-d12-std": "_jointdied12std",
        "joint-die-random-size": "_jointdierandomsize",
        "joint-die-d8-big": "_jointdied8big",
        "joint-die-d8-big-exploration-bonus": "_jointdied8bigexplorationbonus",
        "joint-die-d8-big-antipodal": "_jointdied8bigantipodal",
        "die-d8-big-taskspace-antipodal": "_died8bigtaskspaceantipodal",
        "joint-die-d8-big-relative-antipodal": "_jointdied8bigrelativeantipodal",
        "joint-die-d10-big": "_jointdied10big",
        "joint-die-d12-big": "_jointdied12big",
        "joint-die-d12-d20-mixed": "_jointdied12d20mixed",
        "joint-die-target-selection-so": "_jointdietargetselectionso",
        "joint-die-target-selection-d1": "_jointdietargetselectiond1",
        "joint-die-target-selection-d2": "_jointdietargetselectiond2",
    }[args_cli.variant]
    log_dir = os.path.join(
        LOG_ROOT + _log_suffix,
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )
    os.makedirs(log_dir, exist_ok=True)

    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "name_prefix": "franka_lift_train",
            "disable_logger": True,
        }
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    num_learning_iterations = agent_cfg.max_iterations
    if args_cli.checkpoint is not None:
        runner.load(args_cli.checkpoint, load_optimizer=not args_cli.policy_only_checkpoint)
        resumed_at = runner.current_learning_iteration
        num_learning_iterations = max(agent_cfg.max_iterations - resumed_at, 0)
        print(
            f"Resumed from {args_cli.checkpoint} at iteration {resumed_at}; running "
            f"{num_learning_iterations} more iteration(s) to reach the absolute target "
            f"{agent_cfg.max_iterations}."
        )

    runner.learn(num_learning_iterations=num_learning_iterations, init_at_random_ep_len=True)

    env.close()
    print(f"Training complete. Checkpoints and logs written to: {log_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
