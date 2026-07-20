"""Diagnostic (not a training/eval-suite script): Task 1's own required
empirical check (docs/superpowers/plans/2026-07-20-d8-antipodal-grasp-
quality-implementation.md, "Design notes" #3) that `ContactSensorCfg` /
`ContactSensorData.force_matrix_w` actually produces real, sensible
contact-force data for `FrankaDieLiftContactSceneCfg`
(tasks/franka/dice_lift_joint_env_cfg.py) inside a genuine
`ManagerBasedRLEnvCfg` - the scripted-demo precedent
(tasks/franka/dice_scene_cfg.py / scripts/dice_pick_demo.py) proves the
`activate_contact_sensors=True` + two-single-body-`ContactSensorCfg`
wiring works for a plain `InteractiveScene`, but never inside a real
`RewardManager`-driven training loop, so this is checked directly rather
than assumed to transfer.

Adapts scripts/calibrate_gripper_contact.py's own AR4-era
teleport-object-to-the-live-pinch-point technique (never modifies that
file - a new, Franka-specific sibling): teleports the d8 die to the live
`ee_frame` sensor position every step (tracking the arm's own real,
possibly-still-settling pose, not a stale one-time read) and drives the
gripper open/closed via `BinaryJointPositionActionCfg`'s own sign
convention (`action < 0` -> close, per
tasks/franka/exploration_bonus_reward.py's own confirmed source read of
`isaaclab/envs/mdp/actions/binary_joint_actions.py`). Arm action stays
zero every step (`JointPositionActionCfg` with `use_default_offset=True`:
a zero action holds the arm at its own default reset joint pose - same
"hold still" convention as the AR4 calibration script).

Uses a throwaway leaf env cfg (`_DiagContactCheckEnvCfg`, defined in this
script only, never added to tasks/franka/dice_lift_joint_env_cfg.py)
subclassing `FrankaDieLiftJointD8BigEnvCfg` with only `scene` overridden to
`FrankaDieLiftContactSceneCfg()` - per the implementation plan's own
"implementer's judgment on the cleanest bounded way to exercise this
without yet building Task 2's real leaf classes."

.. code-block:: bash

    flock -o /tmp/rl_isaac_sim.lock -c \
      "PYTHONUNBUFFERED=1 /home/saps/IsaacLab/isaaclab.sh -p scripts/_diag_contact_sensor_check.py --headless"

(Cloud dispatch runs `--headless` per this project's own standing cloud
exception; drop `--headless` for local desktop dispatch, per CLAUDE.md's
"Run non-headless for the time being" instruction.)
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Empirical ContactSensorCfg check for FrankaDieLiftContactSceneCfg.")
parser.add_argument("--num_envs", type=int, default=8)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tasks.franka.dice_lift_joint_env_cfg import (  # noqa: E402
    FrankaDieLiftContactSceneCfg,
    FrankaDieLiftJointD8BigEnvCfg,
)

GRIPPER_OPEN = 1.0
GRIPPER_CLOSE = -1.0

# (duration_steps, gripper_command, label, teleport_die_to_pinch_point)
# "far" is a real negative control: the die is left at its own spawn/reset
# position (untouched, not teleported) so this script's own output includes
# a genuine no-contact baseline, not just an assumed one - same discipline
# as calibrate_gripper_contact.py's own PHASES table.
PHASES = [
    (30, GRIPPER_OPEN, "far", False),
    (60, GRIPPER_OPEN, "open", True),
    (60, GRIPPER_CLOSE, "close", True),
    (120, GRIPPER_CLOSE, "hold", True),
]


@configclass
class _DiagContactCheckEnvCfg(FrankaDieLiftJointD8BigEnvCfg):
    """Throwaway diagnostic-only env cfg - only `scene` is overridden
    (to `FrankaDieLiftContactSceneCfg`), everything else (actions, rewards,
    d8-48mm object, PPO recipe) inherits unchanged from
    `FrankaDieLiftJointD8BigEnvCfg`. Never added to
    tasks/franka/dice_lift_joint_env_cfg.py - this script's own throwaway
    harness only, per Task 1's own "implementer's judgment" note."""

    scene: FrankaDieLiftContactSceneCfg = FrankaDieLiftContactSceneCfg()


def main() -> None:
    env_cfg = _DiagContactCheckEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.env_spacing = 2.5
    env_cfg.sim.device = args_cli.device
    settle_steps = 30
    total_steps = settle_steps + sum(duration for duration, _, _, _ in PHASES)
    step_dt = env_cfg.decimation * env_cfg.sim.dt
    env_cfg.episode_length_s = total_steps * step_dt + 2.0

    env = ManagerBasedRLEnv(cfg=env_cfg)
    num_arm_joints = 7  # panda_joint.* - matches JointPositionActionCfg's own joint_names count

    identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1)

    far_forces: list[float] = []
    hold_forces: list[float] = []

    with torch.inference_mode():
        env.reset()
        ee_frame = env.scene["ee_frame"]
        object_entity = env.scene["object"]

        print(f"[diag] num_envs={env.num_envs}")
        print("[diag] settling arm at default reset pose...")
        for _ in range(settle_steps):
            action = torch.zeros(env.num_envs, num_arm_joints + 1, device=env.device)
            action[:, num_arm_joints] = GRIPPER_OPEN
            env.step(action)

        jaw1_shape_reported = False
        for duration, gripper_cmd, label, track_pinch_point in PHASES:
            for _ in range(duration):
                if track_pinch_point:
                    pinch_point = ee_frame.data.target_pos_w[:, 0, :].clone()
                    die_pose = torch.cat([pinch_point, identity_quat], dim=-1)
                    object_entity.write_root_pose_to_sim(die_pose)
                    object_entity.write_root_velocity_to_sim(torch.zeros(env.num_envs, 6, device=env.device))

                action = torch.zeros(env.num_envs, num_arm_joints + 1, device=env.device)
                action[:, num_arm_joints] = gripper_cmd
                env.step(action)

                jaw1_sensor = env.scene["panda_leftfinger_contact"]
                jaw2_sensor = env.scene["panda_rightfinger_contact"]

                if not jaw1_shape_reported:
                    print(f"[diag] panda_leftfinger_contact.data.force_matrix_w.shape = {tuple(jaw1_sensor.data.force_matrix_w.shape)}")
                    print(f"[diag] panda_rightfinger_contact.data.force_matrix_w.shape = {tuple(jaw2_sensor.data.force_matrix_w.shape)}")
                    reshaped = jaw1_sensor.data.force_matrix_w.view(env.num_envs, 3)
                    print(f"[diag] view(num_envs, 3) reshape OK, shape = {tuple(reshaped.shape)}")
                    jaw1_shape_reported = True

                jaw1_force_vec = jaw1_sensor.data.force_matrix_w.view(env.num_envs, 3)
                jaw2_force_vec = jaw2_sensor.data.force_matrix_w.view(env.num_envs, 3)
                jaw1_norm = torch.linalg.vector_norm(jaw1_force_vec, dim=-1)
                jaw2_norm = torch.linalg.vector_norm(jaw2_force_vec, dim=-1)
                force_norm = [jaw1_norm.max().item(), jaw2_norm.max().item()]

                if label == "far":
                    far_forces.append(max(force_norm))
                elif label == "hold":
                    hold_forces.append(max(force_norm))

            print(f"[phase done] {label}: last per-jaw max-over-envs force_norm={force_norm}")

    print("\n=== ContactSensorCfg empirical check summary ===")
    far_min, far_max = min(far_forces), max(far_forces)
    hold_min, hold_max = min(hold_forces), max(hold_forces)
    print(f"far  (die untouched, not near gripper) max-per-step force_norm: min={far_min:.6f}, max={far_max:.6f} N")
    print(f"hold (gripper closed on teleported die) max-per-step force_norm: min={hold_min:.6f}, max={hold_max:.6f} N")
    print(f"far all (near-)zero: {far_max < 1e-6}")
    print(f"hold has genuinely nonzero contact: {hold_max > 0.01}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
