"""Train a PPO policy (rsl_rl) for the AR4 pick-and-place task.

.. code-block:: bash

    cd ~/projects/rl
    /home/saps/IsaacLab/isaaclab.sh -p scripts/train.py --num_envs 4096
    # smoke test (fast, verifies the loop runs end-to-end and writes a checkpoint):
    /home/saps/IsaacLab/isaaclab.sh -p scripts/train.py --num_envs 16 --max_iterations 2 --headless

    # single-object scene (sphere only), observing the sphere's position via the
    # real RGB-D camera + perception pipeline instead of privileged sim state
    # (reward stays privileged) - see
    # docs/superpowers/specs/2026-07-05-ar4-single-object-camera-training-design.md.
    # Keep --num_envs small: the perception pipeline is plain numpy/CPU, not
    # GPU-batched, and camera rendering itself isn't free at scale.
    /home/saps/IsaacLab/isaaclab.sh -p scripts/train.py --perception --num_envs 16 --max_iterations 2 --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train the AR4 pick-and-place policy with PPO (rsl_rl).")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of parallel environments.")
parser.add_argument("--max_iterations", type=int, default=None, help="Override the agent config's max_iterations.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos periodically during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of each recorded video (steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Steps between recorded videos.")
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
parser.add_argument(
    "--ik_guided",
    action="store_true",
    default=False,
    help=(
        "Train on the classical-IK-guided variant of the mirror-goal scene: reach/grasp/lift/carry is "
        "shaped by a live classical-IK path-tracking reward instead of ad hoc end-state distances. See "
        "docs/superpowers/specs/2026-07-06-ar4-ik-guided-path-design.md."
    ),
)
parser.add_argument(
    "--taskspace",
    action="store_true",
    default=False,
    help=(
        "Train on the task-space IK-driven-action variant of the mirror-goal scene: the arm's action "
        "is a Cartesian end-effector delta converted to joint targets by a live differential-IK "
        "controller in the control loop, instead of direct joint-position deltas. See "
        "docs/superpowers/specs/2026-07-06-ar4-taskspace-ik-action-design.md."
    ),
)
parser.add_argument(
    "--residual",
    action="store_true",
    default=False,
    help=(
        "Train on the residual-action variant of the task-space scene: the arm's action is a bounded "
        "pursuit step toward the active waypoint (classical base controller) plus the policy's own "
        "scaled action (RL residual) on top, instead of the policy's action alone. See "
        "docs/superpowers/specs/2026-07-07-ar4-residual-ik-action-design.md."
    ),
)
parser.add_argument(
    "--reachskip",
    action="store_true",
    default=False,
    help=(
        "Train on the reach-skip curriculum variant of the task-space scene: the arm starts each "
        "episode already at the pregrasp pose for that episode's randomized cube position (via a "
        "one-shot IK reset), instead of a fixed home pose - removing the reach sub-problem so the "
        "full step budget goes toward grasp/lift/carry/place. See "
        "docs/superpowers/specs/2026-07-07-ar4-reachskip-curriculum-design.md."
    ),
)
parser.add_argument(
    "--baseproximity",
    action="store_true",
    default=False,
    help=(
        "Train on the reward-shaping variant of the task-space scene: adds a ground-contact penalty "
        "and a new cube-to-robot-base proximity penalty, and raises the antipodal grasp bonus's weight "
        "(with a matched stillness-penalty raise preserving the anti-freeze reward-rate margin), on top "
        "of Experiment 12's clean baseline reward. See "
        "docs/superpowers/specs/2026-07-07-ar4-experiment15-reward-shaping-design.md."
    ),
)
parser.add_argument(
    "--provenrecipe",
    action="store_true",
    default=False,
    help=(
        "Train on the from-scratch proven-recipe replication: no standalone grasp reward, a plain "
        "binary lift reward, goal-tracking reward gated on lift, and plain joint-space action - "
        "replicating Isaac Lab's own Franka Cube Lift task and IsaacGymEnvs' FrankaCubeStack task, "
        "both read directly from source. See "
        "docs/superpowers/specs/2026-07-07-ar4-experiment16-proven-recipe-replication-design.md."
    ),
)
parser.add_argument(
    "--graspgated",
    action="store_true",
    default=False,
    help=(
        "Train on the grasp-verification-gated variant of the proven-recipe scene: identical to "
        "--provenrecipe except the lift and goal-tracking reward terms now require genuine bilateral "
        "antipodal jaw contact, not just object height, fixing a confirmed 'stage leakage' exploit "
        "where the policy wedged the cube against its own wrist geometry instead of grasping it. See "
        "docs/superpowers/specs/2026-07-07-ar4-experiment17-grasp-gated-lift-design.md."
    ),
)
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
parser.add_argument(
    "--orientationbias",
    action="store_true",
    default=False,
    help=(
        "Train on the soft orientation-alignment-bias variant: adds one new reward term rewarding "
        "the policy for keeping the gripper's approach axis close to vertical (top-down), on top of "
        "--pregrasp's unchanged reward set. Revised from Experiment 20's original hard IK-lock action "
        "term, which independent instrumented verification found structurally unstable. See "
        "docs/superpowers/specs/2026-07-07-ar4-experiment20-vertical-orientation-lock-design.md."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video or args_cli.perception:
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

from _perception_adapter import PerceptionObservationWrapper  # noqa: E402
from tasks.ar4.agents.rsl_rl_ppo_cfg import Ar4PickPlacePPORunnerCfg  # noqa: E402
from tasks.ar4.pickplace_env_cfg import GROUND_Z, Ar4PickPlaceEnvCfg  # noqa: E402
from tasks.ar4.pickplace_ik_guided_env_cfg import Ar4PickPlaceIkGuidedEnvCfg  # noqa: E402
from tasks.ar4.pickplace_mirror_env_cfg import Ar4PickPlaceMirrorEnvCfg  # noqa: E402
from tasks.ar4.pickplace_single_object_env_cfg import Ar4PickPlaceSingleObjectEnvCfg  # noqa: E402
from tasks.ar4.pickplace_baseproximity_env_cfg import Ar4PickPlaceBaseProximityEnvCfg  # noqa: E402
from tasks.ar4.pickplace_graspgated_env_cfg import Ar4PickPlaceGraspGatedEnvCfg  # noqa: E402
from tasks.ar4.pickplace_orientationbias_env_cfg import Ar4PickPlaceOrientationBiasEnvCfg  # noqa: E402
from tasks.ar4.pickplace_pregrasp_env_cfg import Ar4PickPlacePregraspEnvCfg  # noqa: E402
from tasks.ar4.pickplace_provenrecipe_env_cfg import Ar4PickPlaceProvenRecipeEnvCfg  # noqa: E402
from tasks.ar4.pickplace_reachskip_env_cfg import Ar4PickPlaceReachskipEnvCfg  # noqa: E402
from tasks.ar4.pickplace_residual_env_cfg import Ar4PickPlaceResidualEnvCfg  # noqa: E402
from tasks.ar4.pickplace_taskspace_env_cfg import (  # noqa: E402
    Ar4PickPlaceTaskspaceEnvCfg,
    Ar4PickPlaceTaskspacePPORunnerCfg,
)

LOG_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "train")


def main() -> None:
    if args_cli.orientationbias:
        env_cfg_cls = Ar4PickPlaceOrientationBiasEnvCfg
    elif args_cli.pregrasp:
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
    env_cfg = env_cfg_cls()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    if args_cli.taskspace or args_cli.residual or args_cli.reachskip or args_cli.baseproximity:
        agent_cfg = Ar4PickPlaceTaskspacePPORunnerCfg()
    else:
        agent_cfg = Ar4PickPlacePPORunnerCfg()
    agent_cfg.device = args_cli.device
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    env_cfg.seed = agent_cfg.seed

    log_dir = os.path.join(LOG_ROOT, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)

    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "name_prefix": "ar4_pickplace_train",
            "disable_logger": True,
        }
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    if args_cli.perception:
        env = PerceptionObservationWrapper(env, ground_z=GROUND_Z)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()
    print(f"Training complete. Checkpoints and logs written to: {log_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
