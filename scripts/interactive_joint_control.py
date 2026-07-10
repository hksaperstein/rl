"""Open the AR4 in the Isaac Sim GUI with a live, interactive joint-control
panel: one slider per arm joint (bounded to that joint's real physical
limits) plus one gripper slider, each driving robot.set_joint_position_target
live every frame, plus a live readout of each joint's actual measured
position. Unlike scripts/interactive_joint_demo.py (which autonomously runs
a scripted pick-and-place cycle and would fight manual intervention), this
script commands nothing on its own - every joint target comes directly from
the current slider positions, read fresh every physics step, so you can
freely drag any slider at any time and watch the arm respond and the
readout update live.

This script never calls the manager-based env's own step/action pipeline -
it drives PhysX directly (write_data_to_sim + sim.step) and explicitly
refreshes robot.data.* after every step, since Isaac Lab doesn't do that
automatically outside of env.step() - same pattern interactive_joint_demo.py
uses.

.. code-block:: bash

    DISPLAY=:1 flock /tmp/rl_isaac_sim.lock -c "/home/saps/IsaacLab/isaaclab.sh -p scripts/interactive_joint_control.py"
"""

import os
import sys

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser(description="Interactive AR4 joint-control GUI (manual sliders, not scripted).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
if args_cli.headless:
    sys.exit("This tool is for live GUI interaction - run without --headless.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math  # noqa: E402

import omni.ui as ui  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import ManagerBasedEnv  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from tasks.ar4.grasp_verify_env_cfg import Ar4GraspVerifyEnvCfg  # noqa: E402
from tasks.ar4.robot_cfg import ARM_JOINT_NAMES, GRIPPER_CLOSED_POS, GRIPPER_JOINT_NAMES, GRIPPER_OPEN_POS  # noqa: E402

# Same reasoning as interactive_joint_demo.py: the default arm PD gains
# (stiffness=40, damping=4) let the arm sag noticeably under gravity at a
# held target - raised here (this env instance only) so dragging a slider
# actually holds the arm where you put it instead of drooping back down.
ARM_STIFFNESS = 2500.0
ARM_DAMPING = 45.0


class JointControlPanel:
    """Floating on-screen window (part of the Isaac Sim app, not a separate
    OS window): one slider per arm joint (bounded to that joint's real
    physical limits, read from the live robot at construction time) plus a
    gripper slider, each paired with a live readout label. Call
    get_arm_targets()/get_gripper_target() every physics step to read the
    current commanded values, and update_readout() every step to refresh
    the displayed actual joint positions."""

    def __init__(self, arm_joint_names, arm_limits, gripper_open_pos, gripper_closed_pos):
        self.arm_joint_names = arm_joint_names
        self.gripper_open_pos = gripper_open_pos
        self.gripper_closed_pos = gripper_closed_pos
        self.sliders = {}
        self.readout_labels = {}

        row_height = 28
        n_rows = len(arm_joint_names) + 2  # + gripper row + reset-button row
        self._window = ui.Window("AR4 Interactive Joint Control", width=460, height=60 + row_height * n_rows)
        with self._window.frame:
            with ui.VStack(spacing=6, style={"font_size": 14}):
                ui.Label("Drag sliders to command joint targets live (real physical limits).")
                for name, (lo, hi) in zip(arm_joint_names, arm_limits):
                    with ui.HStack(height=row_height):
                        ui.Label(name, width=80)
                        slider = ui.FloatSlider(min=lo, max=hi, step=0.001)
                        slider.model.set_value(0.0)
                        self.sliders[name] = slider
                        label = ui.Label("", width=170)
                        self.readout_labels[name] = label
                with ui.HStack(height=row_height):
                    ui.Label("gripper", width=80)
                    slider = ui.FloatSlider(min=gripper_closed_pos, max=gripper_open_pos, step=0.0005)
                    slider.model.set_value(gripper_open_pos)
                    self.sliders["gripper"] = slider
                    label = ui.Label("", width=170)
                    self.readout_labels["gripper"] = label
                with ui.HStack(height=row_height):
                    reset_btn = ui.Button("Reset to home (arm=0, gripper open)")
                    reset_btn.set_clicked_fn(self._on_reset)

    def _on_reset(self):
        for name in self.arm_joint_names:
            self.sliders[name].model.set_value(0.0)
        self.sliders["gripper"].model.set_value(self.gripper_open_pos)

    def get_arm_targets(self):
        return [self.sliders[name].model.get_value_as_float() for name in self.arm_joint_names]

    def get_gripper_target(self):
        return self.sliders["gripper"].model.get_value_as_float()

    def update_readout(self, arm_positions, gripper_positions):
        for name, val in zip(self.arm_joint_names, arm_positions):
            self.readout_labels[name].text = f"actual: {val:+.4f} rad ({math.degrees(val):+6.1f} deg)"
        self.readout_labels["gripper"].text = f"actual: jaw1={gripper_positions[0]:+.4f}  jaw2={gripper_positions[1]:+.4f}"


def main() -> None:
    env_cfg = Ar4GraspVerifyEnvCfg()
    env_cfg.sim.device = args_cli.device
    # Contact sensors are read-only and irrelevant to manual joint control -
    # disabled here, scoped to just this tool's own env instance, same as
    # interactive_joint_demo.py does.
    env_cfg.scene.gripper_jaw1_contact = None
    env_cfg.scene.gripper_jaw2_contact = None
    env = ManagerBasedEnv(cfg=env_cfg)

    robot = env.scene["robot"]
    arm_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)
    arm_cfg.resolve(env.scene)
    gripper_cfg = SceneEntityCfg("robot", joint_names=GRIPPER_JOINT_NAMES)
    gripper_cfg.resolve(env.scene)

    stiff_t = torch.full((1, len(arm_cfg.joint_ids)), ARM_STIFFNESS, device=env.device)
    damp_t = torch.full((1, len(arm_cfg.joint_ids)), ARM_DAMPING, device=env.device)
    robot.write_joint_stiffness_to_sim(stiff_t, joint_ids=arm_cfg.joint_ids)
    robot.write_joint_damping_to_sim(damp_t, joint_ids=arm_cfg.joint_ids)

    with torch.inference_mode():
        env.reset()

    joint_pos_limits = robot.data.joint_pos_limits[:, arm_cfg.joint_ids]
    arm_limits = [
        (joint_pos_limits[0, i, 0].item(), joint_pos_limits[0, i, 1].item()) for i in range(len(ARM_JOINT_NAMES))
    ]

    panel = JointControlPanel(ARM_JOINT_NAMES, arm_limits, GRIPPER_OPEN_POS, GRIPPER_CLOSED_POS)

    print("=" * 70)
    print("[READY] Interactive joint control window open.")
    print("Drag sliders in the 'AR4 Interactive Joint Control' window to command")
    print("joints live. Close the window / stop the simulation app to exit.")
    print("=" * 70)

    while simulation_app.is_running():
        arm_targets = panel.get_arm_targets()
        gripper_target_val = panel.get_gripper_target()
        arm_target_t = torch.tensor([arm_targets], device=env.device)
        gripper_target_t = torch.tensor([[gripper_target_val, gripper_target_val]], device=env.device)

        robot.set_joint_position_target(arm_target_t, joint_ids=arm_cfg.joint_ids)
        robot.set_joint_position_target(gripper_target_t, joint_ids=gripper_cfg.joint_ids)
        robot.write_data_to_sim()
        env.sim.step(render=True)
        robot.update(env.physics_dt)

        arm_positions = robot.data.joint_pos[0, arm_cfg.joint_ids].cpu().tolist()
        gripper_positions = robot.data.joint_pos[0, gripper_cfg.joint_ids].cpu().tolist()
        panel.update_readout(arm_positions, gripper_positions)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
