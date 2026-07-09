# scripts/_check_ground_termination_fires.py
"""Follow-up to scripts/_check_arm_ground_collision_pose.py: that script
found joint_2=1.571 (upper limit), joint_3=0.087 drives link_5 to
z=-0.0523 (below the ground plane) via pure FK, with the robot otherwise
at its default joint pose. This script actually teleports the robot into
that configuration (both write_joint_position_to_sim AND
set_joint_position_target, so the PD drive holds there instead of
immediately fighting back toward its old target - same convention
tasks/ar4/mdp.py's reset_arm_to_pregrasp_pose already uses), then steps
physics forward and confirms:

1. The arm_ground_contact ContactSensor actually registers nonzero
   contact force.
2. isaaclab_mdp.illegal_contact / the arm_ground_contact termination
   actually fires.
3. arm_ground_contact_penalty actually returns -1.0 for the colliding
   env(s).

This is the direct, non-guessed collision test that
scripts/smoke_test_graspgoal_ground_penalty.py's Phase 2 (a guessed
RL-style action) failed to reproduce over 200 steps.

.. code-block:: bash

    /home/saps/IsaacLab/isaaclab.sh -p scripts/_check_ground_termination_fires.py
"""

import os
import sys

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from tasks.ar4 import mdp as ar4_mdp  # noqa: E402
from tasks.ar4.pickplace_graspgoal_env_cfg import ARM_GROUND_CONTACT_FORCE_THRESHOLD, Ar4PickPlaceGraspGoalEnvCfg  # noqa: E402


def main() -> None:
    env_cfg = Ar4PickPlaceGraspGoalEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device
    env = ManagerBasedRLEnv(cfg=env_cfg)

    with torch.inference_mode():
        env.reset()
        robot = env.scene["robot"]
        arm_ground_sensor = env.scene["arm_ground_contact"]

        joint_idx = {name: robot.data.joint_names.index(name) for name in ["joint_2", "joint_3"]}
        joint_pos = robot.data.default_joint_pos.clone()
        joint_pos[0, joint_idx["joint_2"]] = 1.571
        joint_pos[0, joint_idx["joint_3"]] = 0.087

        robot.write_joint_position_to_sim(joint_pos)
        robot.write_joint_velocity_to_sim(torch.zeros_like(joint_pos))
        robot.set_joint_position_target(joint_pos)

        arm_ground_sensor_cfg = SceneEntityCfg("arm_ground_contact")
        arm_ground_sensor_cfg.resolve(env.scene)

        fired_penalty = False
        fired_termination = False
        max_force_seen = 0.0
        for step in range(60):
            # Step physics directly (not env.step, which would run the
            # action manager and could move the arm away from the forced
            # pose via the arm action term) - same low-level pattern
            # reset_arm_to_pregrasp_pose's callers rely on for a one-shot
            # teleport, extended here to hold across several physics
            # ticks so contact resolution has time to develop.
            robot.set_joint_position_target(joint_pos)
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)

            net_forces = arm_ground_sensor.data.net_forces_w_history
            max_force = torch.max(torch.norm(net_forces, dim=-1)).item()
            max_force_seen = max(max_force_seen, max_force)

            penalty = ar4_mdp.arm_ground_contact_penalty(
                env, arm_ground_sensor_cfg, ARM_GROUND_CONTACT_FORCE_THRESHOLD
            )
            terminated = ar4_mdp.isaaclab_mdp.illegal_contact(
                env, threshold=ARM_GROUND_CONTACT_FORCE_THRESHOLD, sensor_cfg=arm_ground_sensor_cfg
            )
            if torch.any(penalty != 0.0):
                fired_penalty = True
            if torch.any(terminated):
                fired_termination = True

            if step % 10 == 0 or fired_termination:
                print(
                    f"[STEP {step}] max_contact_force={max_force:.4f} penalty={penalty.cpu().tolist()} "
                    f"terminated={terminated.cpu().tolist()}"
                )
            if fired_termination:
                break

        print("=" * 70)
        print(f"[RESULT] max contact force observed over rollout: {max_force_seen:.4f} N")
        print(f"[RESULT] arm_ground_contact_penalty fired: {fired_penalty}")
        print(f"[RESULT] arm_ground_contact (illegal_contact) termination fired: {fired_termination}")
        print("=" * 70)

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
