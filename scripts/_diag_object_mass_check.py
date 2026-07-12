"""One-off diagnostic (asset-bisect spec prep, 2026-07-12): constructs the
requested lift-env variant with 2 envs and prints the object's LIVE
PhysX-computed mass/inertia (root_physx_view.get_masses()), plus the
robot's finger-link masses for contact-impulse context. The research pass
(.superpowers/sdd/research-asset-bisect.md) estimated DexCube ~0.11kg by
default-density inference and flagged direct measurement as a pre-rung-1
to-do — this script is that measurement. One variant per launch (two env
constructions in one process is a known Isaac teardown hang, see
progress.md Experiment 10 note).

.. code-block:: bash

    flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 DISPLAY=:1 \\
        /home/saps/IsaacLab/isaaclab.sh -p scripts/_diag_object_mass_check.py --variant joint-cube"
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Print live PhysX mass of the lift object.")
parser.add_argument("--variant", choices=["joint-die", "joint-cube", "joint-die-heavy"], required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows AppLauncher."""

import os  # noqa: E402
import sys  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402


def main() -> None:
    if args_cli.variant == "joint-die":
        from tasks.franka.dice_lift_joint_env_cfg import FrankaDieLiftJointEnvCfg as Cfg
    elif args_cli.variant == "joint-die-heavy":
        from tasks.franka.dice_lift_joint_env_cfg import FrankaDieLiftJointHeavyEnvCfg as Cfg
    else:
        from tasks.franka.dice_lift_joint_env_cfg import FrankaCubeLiftJointEnvCfg as Cfg

    env_cfg = Cfg()
    env_cfg.scene.num_envs = 2
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env.reset()

    obj = env.scene["object"]
    masses = obj.root_physx_view.get_masses()
    inertias = obj.root_physx_view.get_inertias()
    print(f"\n[MASS CHECK] variant={args_cli.variant}")
    print(f"  object masses (kg, per env): {masses.cpu().numpy().tolist()}")
    print(f"  object inertia diag (env 0): {inertias.cpu().numpy()[0].reshape(3, 3).diagonal().tolist()}")

    robot = env.scene["robot"]
    body_names = robot.body_names
    robot_masses = robot.root_physx_view.get_masses()[0].cpu().numpy()
    for i, name in enumerate(body_names):
        if "finger" in name.lower() or "hand" in name.lower():
            print(f"  robot body '{name}': {robot_masses[i]:.4f} kg")

    print("[DONE]")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
