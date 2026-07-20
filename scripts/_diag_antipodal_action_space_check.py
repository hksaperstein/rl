"""Diagnostic (not a training/eval-suite script): Task 2's own required
"confirm Condition B's action space really is task-space" check
(docs/superpowers/plans/2026-07-20-d8-antipodal-grasp-quality-
implementation.md Task 2's "verify it empirically (build the env, confirm
its actual action space is task-space/IK, not joint-space, before assuming
your class composition worked)").

Builds exactly ONE of the two new leaf env cfgs
(tasks/franka/dice_lift_joint_env_cfg.py), selected via --variant, and
prints env.action_manager's own per-term action_dim/class breakdown plus
the total action dimensionality:
  - condition-a -> FrankaDieLiftJointD8BigAntipodalEnvCfg: expected
    arm_action term class JointPositionAction, action_dim=7 (panda_joint.*),
    total_action_dim=8 (7 arm + 1 gripper) - joint-space, inherited
    unchanged from FrankaDieLiftJointD8BigEnvCfg's own __post_init__ chain.
  - condition-b -> FrankaDieLiftD8BigTaskspaceAntipodalEnvCfg: expected
    arm_action term class DifferentialInverseKinematicsAction, action_dim=6
    (command_type="pose": 3 position + 3 rotation), total_action_dim=7
    (6 arm + 1 gripper) - task-space/relative-IK, re-asserted by this
    class's own __post_init__ AFTER the inherited joint-space chain runs
    (see that class's own docstring for the full MRO-ordering argument this
    check empirically confirms, not just asserts from the class hierarchy
    alone).

ONE env cfg per process, not both in one script: this project's own
already-documented Isaac Lab limitation - a second ManagerBasedRLEnv
cannot be constructed in-process after a first one is built, either
simultaneously (RuntimeError: Simulation context already exists) or
sequentially after .close() (hangs indefinitely) - see
tasks/franka/dice_lift_joint_env_cfg.py's own
FrankaDieLiftJointD12D20MixedEnvCfg docstring for the confirmed repro.
Run this script twice (--variant condition-a, then --variant condition-b)
as two separate isaaclab.sh -p invocations under the shared flock.

.. code-block:: bash

    flock -o /tmp/rl_isaac_sim.lock -c \
      "PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/_diag_antipodal_action_space_check.py \
        --variant condition-a --headless"
    flock -o /tmp/rl_isaac_sim.lock -c \
      "PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/_diag_antipodal_action_space_check.py \
        --variant condition-b --headless"

(Cloud dispatch runs --headless per this project's own standing cloud
exception; drop --headless for local desktop dispatch, per CLAUDE.md's
"Run non-headless for the time being" instruction.)
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Empirical action-space check for the antipodal Condition A/B env cfgs.")
parser.add_argument("--variant", choices=["condition-a", "condition-b"], required=True)
parser.add_argument("--num_envs", type=int, default=8)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if args_cli.variant == "condition-a":
    from tasks.franka.dice_lift_joint_env_cfg import FrankaDieLiftJointD8BigAntipodalEnvCfg as _EnvCfg  # noqa: E402
else:
    from tasks.franka.dice_lift_joint_env_cfg import FrankaDieLiftD8BigTaskspaceAntipodalEnvCfg as _EnvCfg  # noqa: E402


def main() -> None:
    env_cfg = _EnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    env = ManagerBasedRLEnv(cfg=env_cfg)
    action_manager = env.action_manager

    print(f"\n=== action-space check: --variant {args_cli.variant} ({type(env_cfg).__name__}) ===")
    print(f"total_action_dim = {action_manager.total_action_dim}")
    for term_name in action_manager.active_terms:
        term = action_manager.get_term(term_name)
        print(f"  term '{term_name}': class={type(term).__name__}, action_dim={term.action_dim}")

    arm_term = action_manager.get_term("arm_action")
    arm_class_name = type(arm_term).__name__
    is_taskspace = "DifferentialInverseKinematics" in arm_class_name
    is_jointspace = "JointPosition" in arm_class_name
    print(f"\narm_action term class: {arm_class_name}")
    print(f"is_taskspace (DifferentialInverseKinematicsAction): {is_taskspace}")
    print(f"is_jointspace (JointPositionAction): {is_jointspace}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
