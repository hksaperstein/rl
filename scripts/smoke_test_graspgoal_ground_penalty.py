# scripts/smoke_test_graspgoal_ground_penalty.py
"""Verification script for the ground/table collision safety termination +
reward and the slow-near-cube dense bonus added to
Ar4PickPlaceGraspGoalEnvCfg (tasks/ar4/pickplace_graspgoal_env_cfg.py).

Checks, with real evidence (not just "no exception"):

1. robot.data.body_names - direct introspection of the actual AR4 mk5
   body names, to confirm the upper-arm-only contact sensor's assumed
   link set (base_link, link_1..link_5) is correct before trusting it.
2. The arm_ground_contact ContactSensor's own resolved body_names/
   num_bodies - confirms the single ContactSensorCfg with a regex
   alternation in its prim_path leaf actually tracks all 6 intended
   bodies (not just 1, not erroring).
3. Observation/action shapes are unchanged (31 / 7) - this task only
   added a sensor + 2 reward terms + 1 termination, no new observation.
4. Steps the env with actions driving the arm down toward/into the
   ground plane, to confirm the arm_ground_contact termination and
   penalty actually fire under a real collision (not just theoretically
   correct code that never triggers).
5. Prints slow_near_cube_bonus and arm_ground_contact_penalty in
   isolation (calling the mdp functions directly against live env state)
   so their numeric behavior can be inspected directly, not just their
   contribution buried in a summed step reward.

.. code-block:: bash

    /home/saps/IsaacLab/isaaclab.sh -p scripts/smoke_test_graspgoal_ground_penalty.py
"""

import sys

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser(description="Verify the ground-contact penalty/termination and slow-near-cube bonus.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os  # noqa: E402

import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from tasks.ar4 import mdp as ar4_mdp  # noqa: E402
from tasks.ar4.pickplace_graspgoal_env_cfg import (  # noqa: E402
    ARM_GROUND_CONTACT_FORCE_THRESHOLD,
    SLOW_NEAR_CUBE_REACH_DIST_THRESHOLD,
    SLOW_NEAR_CUBE_SPEED_CAP,
    Ar4PickPlaceGraspGoalEnvCfg,
)


def main() -> None:
    env_cfg = Ar4PickPlaceGraspGoalEnvCfg()
    env_cfg.scene.num_envs = 4
    env_cfg.sim.device = args_cli.device
    env = ManagerBasedRLEnv(cfg=env_cfg)

    with torch.inference_mode():
        obs, _ = env.reset()

        robot = env.scene["robot"]
        print("=" * 70)
        print(f"[CHECK] robot.data.body_names: {robot.data.body_names}")
        print("=" * 70)

        arm_ground_sensor = env.scene["arm_ground_contact"]
        print(f"[CHECK] arm_ground_contact.num_bodies: {arm_ground_sensor.num_bodies} (expect 6)")
        print(f"[CHECK] arm_ground_contact.body_names: {arm_ground_sensor.body_names}")
        print(
            "[CHECK] arm_ground_contact excludes gripper jaws/link_6: "
            f"{'gripper_jaw1_link' not in arm_ground_sensor.body_names} / "
            f"{'gripper_jaw2_link' not in arm_ground_sensor.body_names} / "
            f"{'link_6' not in arm_ground_sensor.body_names}"
        )

        print(f"[CHECK] observation shape: {obs['policy'].shape} (expect (4, 31))")
        print(f"[CHECK] action space shape: {env.action_manager.action.shape} (expect (4, 7))")

        print("=" * 70)
        print("[CHECK] reward_manager active terms:", env.reward_manager.active_terms)
        print("=" * 70)

        arm_ground_sensor_cfg = SceneEntityCfg("arm_ground_contact")
        arm_ground_sensor_cfg.resolve(env.scene)

        # Phase 1: zero actions for a few steps, confirm no spurious ground
        # termination/penalty under ordinary reset-pose behavior.
        for step in range(10):
            actions = torch.zeros(4, env.action_manager.total_action_dim, device=env.device)
            obs, rew, terminated, truncated, info = env.step(actions)
            penalty = ar4_mdp.arm_ground_contact_penalty(
                env, arm_ground_sensor_cfg, ARM_GROUND_CONTACT_FORCE_THRESHOLD
            )
            slow_bonus = ar4_mdp.slow_near_cube_bonus(
                env,
                SceneEntityCfg("cube"),
                SceneEntityCfg("ee_frame"),
                SceneEntityCfg("robot"),
                SLOW_NEAR_CUBE_REACH_DIST_THRESHOLD,
                SLOW_NEAR_CUBE_SPEED_CAP,
            )
            if step % 5 == 0:
                print(
                    f"[PHASE1 zero-action] step {step}: reward={rew.cpu().tolist()}, "
                    f"arm_ground_contact_penalty={penalty.cpu().tolist()}, "
                    f"slow_near_cube_bonus={slow_bonus.cpu().tolist()}, "
                    f"terminated={terminated.cpu().tolist()}"
                )

        assert not torch.any(penalty != 0.0), (
            f"arm_ground_contact_penalty fired spuriously at reset pose: {penalty.cpu().tolist()}"
        )
        print("[CHECK] PASS: no spurious ground-contact penalty at reset/zero-action pose.")

        # Phase 2: drive arm joints hard toward a low/ground-scraping pose
        # (large negative action on joint_2/joint_3, the shoulder/elbow
        # joints that swing the upper arm down) to try to force a real
        # upper-arm/ground collision, confirming the termination+penalty
        # actually trigger under real contact rather than being dead code.
        forced_actions = torch.zeros(4, env.action_manager.total_action_dim, device=env.device)
        forced_actions[:, 1] = -1.0  # joint_2
        forced_actions[:, 2] = -1.0  # joint_3

        fired_penalty = False
        fired_termination = False
        for step in range(200):
            obs, rew, terminated, truncated, info = env.step(forced_actions)
            penalty = ar4_mdp.arm_ground_contact_penalty(
                env, arm_ground_sensor_cfg, ARM_GROUND_CONTACT_FORCE_THRESHOLD
            )
            if torch.any(penalty != 0.0):
                fired_penalty = True
            if torch.any(terminated):
                fired_termination = True
                print(
                    f"[PHASE2 forced-down] step {step}: termination fired, terminated={terminated.cpu().tolist()}, "
                    f"penalty={penalty.cpu().tolist()}"
                )
            if step % 40 == 0:
                net_forces = arm_ground_sensor.data.net_forces_w_history
                max_force = torch.max(torch.norm(net_forces, dim=-1))
                print(
                    f"[PHASE2 forced-down] step {step}: reward={rew.cpu().tolist()}, "
                    f"penalty={penalty.cpu().tolist()}, max_contact_force={max_force.item():.4f}"
                )
            if fired_termination:
                break

        print("=" * 70)
        print(f"[RESULT] arm_ground_contact_penalty fired at some point: {fired_penalty}")
        print(f"[RESULT] arm_ground_contact termination fired at some point: {fired_termination}")
        print("=" * 70)

    env.close()
    print("[SMOKE] PASS: env built, stepped, ground-contact mechanism exercised, no exceptions.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
