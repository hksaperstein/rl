# scripts/_sweep_jaw2_symmetry.py
"""Empirical follow-up to _inspect_jaw_symmetry_live.py's finding that
commanding the current "open" convention (jaw1=+0.014, jaw2=-0.014) makes
both jaw links converge to the IDENTICAL world point instead of spreading
symmetrically apart. A static-USD hand-derivation of the joint axis math
(_inspect_jaw_axis_math.py) produced a prediction that didn't match the
live simulation's own numbers when cross-checked (a rest-pose offset sign
flipped between the static check and the live run - almost certainly a
stage coordinate-convention handling mistake in the static script, not a
real physical discrepancy) - so this abandons the analytical/static approach
entirely and determines the answer PURELY empirically and directly from the
running simulation, per this project's own verification standard (real
evidence over proxies, and independent re-derivation rather than trusting a
single analysis method).

Holds jaw1 fixed at its own OPEN target (+0.014) and SWEEPS jaw2 across
several values spanning (and slightly beyond, via direct joint-position
writes rather than the clamped actuator target, to probe past the current
hard limit if needed) its currently-authored range, reading back jaw2's
actual world position at each value and comparing against jaw1's fixed
world position and the shared-centerline midpoint - directly identifying
which jaw2 value (if any, within or outside the current limit) produces a
genuinely mirrored (equal-and-opposite offset from the centerline) result.

Run: flock -o /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 DISPLAY=:1 /home/saps/IsaacLab/isaaclab.sh -p scripts/_sweep_jaw2_symmetry.py"
"""
import os
import sys

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser(description="Empirically sweep jaw2 to find the value that mirrors jaw1.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch  # noqa: E402

from isaaclab.envs import ManagerBasedEnv  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from tasks.ar4.grasp_verify_env_cfg import Ar4GraspVerifyEnvCfg  # noqa: E402
from tasks.ar4.robot_cfg import ARM_JOINT_NAMES, GRIPPER_JOINT_NAMES  # noqa: E402


def main() -> None:
    env_cfg = Ar4GraspVerifyEnvCfg()
    env_cfg.sim.device = args_cli.device
    env_cfg.scene.robot.actuators["arm"].stiffness = 4000.0
    env_cfg.scene.robot.actuators["arm"].damping = 200.0

    env = ManagerBasedEnv(cfg=env_cfg)

    robot = env.scene["robot"]
    gripper_cfg = SceneEntityCfg("robot", joint_names=GRIPPER_JOINT_NAMES)
    gripper_cfg.resolve(env.scene)
    arm_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)
    arm_cfg.resolve(env.scene)
    jaw_body_ids = [robot.data.body_names.index(n) for n in ["gripper_jaw1_link", "gripper_jaw2_link"]]
    base_body_id = robot.data.body_names.index("gripper_base_link")
    num_arm_joints = len(ARM_JOINT_NAMES)

    with torch.inference_mode():
        env.reset()
        arm_hold_target = robot.data.joint_pos[0, arm_cfg.joint_ids].clone().tolist()

        # Widen the SIM joint limits (not just actuator clamp) for jaw2 so a
        # direct write_joint_position_to_sim can hold values outside the
        # originally-authored [-0.014, 0] range. This is a live, runtime-only
        # override (robot.data / articulation view), NOT a USD file edit -
        # safe to try values outside the current authored range without
        # touching the asset yet.
        jaw2_joint_id = gripper_cfg.joint_ids[1]
        wide_limits = robot.data.joint_pos_limits.clone()
        wide_limits[:, jaw2_joint_id, 0] = -0.03
        wide_limits[:, jaw2_joint_id, 1] = 0.03
        robot.write_joint_position_limit_to_sim(wide_limits[:, [jaw2_joint_id]], joint_ids=[jaw2_joint_id])

        def _settle_gripper(jaw1_target: float, jaw2_target: float, steps: int = 90):
            for _ in range(steps):
                action = torch.zeros(env.num_envs, num_arm_joints + 1, device=env.device)
                for j in range(num_arm_joints):
                    action[:, j] = arm_hold_target[j]
                # Use direct joint position target writes for the gripper
                # (bypassing the BinaryJointPositionAction's single-scalar
                # open/close abstraction, which can't express arbitrary
                # per-joint sweep values) - same underlying
                # set_joint_position_target + write_data_to_sim mechanism
                # the action manager itself calls, just addressed directly.
                robot.set_joint_position_target(
                    torch.tensor([[jaw1_target, jaw2_target]], device=env.device), joint_ids=gripper_cfg.joint_ids
                )
                robot.set_joint_position_target(
                    torch.tensor([arm_hold_target], device=env.device), joint_ids=arm_cfg.joint_ids
                )
                robot.write_data_to_sim()
                env.sim.step(render=False)
                robot.update(env.physics_dt)

        jaw1_open = 0.014
        sweep_values = [-0.014, -0.010, -0.007, -0.003, 0.0, 0.003, 0.007, 0.010, 0.014]
        print("=" * 100)
        print(f"jaw1 held fixed at {jaw1_open}. Sweeping jaw2 across {sweep_values} (limits temporarily widened to [-0.03, 0.03])")
        print("=" * 100)
        for q2 in sweep_values:
            _settle_gripper(jaw1_open, q2)
            jaw1_pos = robot.data.body_pos_w[0, jaw_body_ids[0]].cpu().tolist()
            jaw2_pos = robot.data.body_pos_w[0, jaw_body_ids[1]].cpu().tolist()
            base_pos = robot.data.body_pos_w[0, base_body_id].cpu().tolist()
            actual_q = robot.data.joint_pos[0, gripper_cfg.joint_ids].cpu().tolist()
            midpoint = [(a + b) / 2 for a, b in zip(jaw1_pos, jaw2_pos)]
            sep = [a - b for a, b in zip(jaw1_pos, jaw2_pos)]
            sep_dist = sum(v * v for v in sep) ** 0.5
            offset_from_base_1 = [a - b for a, b in zip(jaw1_pos, base_pos)]
            offset_from_base_2 = [a - b for a, b in zip(jaw2_pos, base_pos)]
            print(
                f"[q2_target={q2:+.4f}] actual_joint_pos={['%.5f' % v for v in actual_q]} "
                f"jaw1_world={['%.5f' % v for v in jaw1_pos]} jaw2_world={['%.5f' % v for v in jaw2_pos]} "
                f"separation_dist={sep_dist:.5f}m "
                f"offset1_from_base={['%.5f' % v for v in offset_from_base_1]} "
                f"offset2_from_base={['%.5f' % v for v in offset_from_base_2]}"
            )
        print("=" * 100)
        print("Looking for the q2 whose offset2_from_base is the NEGATIVE (mirror) of offset1_from_base.")

    env.close()


if __name__ == "__main__":
    main()
