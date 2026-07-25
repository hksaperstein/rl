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

Each arm joint (not the gripper - its slider is a millimeter/aperture
value, not an angle) also gets a numeric DEGREES entry field
(`omni.ui.FloatDrag`, introspected live from the installed omni.ui module
rather than assumed - see `_ui.pyi` in the omni.ui extension package,
confirmed to support both click-drag AND click-to-type-a-value-then-Enter,
same as a plain field) next to its slider, bounded to that joint's own
real limits (converted to degrees) exactly like the slider is. The slider
is still the source of truth read every physics step
(JointControlPanel.get_arm_targets() only reads slider.model, never the
degree field directly) - the degree field's `end_edit` callback writes
into the slider's own model on commit (Enter / focus-loss), which is
sufficient to drive the arm, since the per-frame loop always re-reads the
slider. A `value_changed` callback on the slider's model keeps the degree
field's displayed text in sync any time the slider moves for ANY reason
(a real drag, a typed-degree commit, or the reset button), so both
widgets always agree. `self._suppress_sync` is a simple reentrancy guard
so the two callbacks don't ping-pong each other.

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
parser.add_argument(
    "--self-test",
    action="store_true",
    help=(
        "Before entering the normal interactive loop, run an automated check of the slider<->degree-field "
        "sync and physical joint movement for the first 3 arm joints, printing PASS/FAIL. Drives the exact "
        "same model callbacks a real drag or typed-Enter commit would trigger (model.set_value / "
        "model.end_edit are the model-layer notification hooks the omni.ui widgets themselves call - see "
        "JointControlPanel._wire_sync's docstring), then steps real physics and reads robot.data.joint_pos "
        "to confirm the arm actually moved, not just that the UI models agree. The window stays open "
        "afterward (normal interactive loop continues) so results can also be confirmed visually."
    ),
)
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
        self.deg_fields = {}
        self.readout_labels = {}
        # Reentrancy guard: prevents the slider's value_changed callback and
        # the degree field's end_edit callback from ping-ponging off each
        # other when one programmatically updates the other.
        self._suppress_sync = False

        row_height = 28
        n_rows = len(arm_joint_names) + 2  # + gripper row + reset-button row
        self._window = ui.Window("AR4 Interactive Joint Control", width=600, height=60 + row_height * n_rows)
        with self._window.frame:
            with ui.VStack(spacing=6, style={"font_size": 14}):
                ui.Label("Drag sliders OR type a value in deg (Enter to commit) to command joint targets live.")
                for name, (lo, hi) in zip(arm_joint_names, arm_limits):
                    with ui.HStack(height=row_height):
                        ui.Label(name, width=80)
                        slider = ui.FloatSlider(min=lo, max=hi, step=0.001, width=150)
                        slider.model.set_value(0.0)
                        self.sliders[name] = slider
                        ui.Label("deg:", width=28)
                        deg_field = ui.FloatDrag(min=math.degrees(lo), max=math.degrees(hi), step=0.5, width=70)
                        deg_field.model.set_value(0.0)
                        self.deg_fields[name] = deg_field
                        label = ui.Label("", width=190)
                        self.readout_labels[name] = label

                        self._wire_sync(slider, deg_field, lo, hi)
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

    def _wire_sync(self, slider, deg_field, lo_rad, hi_rad):
        """Bidirectional sync between one arm joint's radian slider and its
        degree field, guarded by self._suppress_sync so neither callback
        re-triggers the other. lo_rad/hi_rad are that joint's real physical
        limits (radians) - the same bounds the slider already enforces,
        used here to clamp typed-degree input rather than trusting it."""

        def _on_slider_changed(model):
            if self._suppress_sync:
                return
            self._suppress_sync = True
            try:
                deg_field.model.set_value(math.degrees(model.get_value_as_float()))
            finally:
                self._suppress_sync = False

        def _on_deg_commit(model):
            if self._suppress_sync:
                return
            requested_rad = math.radians(model.get_value_as_float())
            clamped_rad = max(lo_rad, min(hi_rad, requested_rad))
            self._suppress_sync = True
            try:
                slider.model.set_value(clamped_rad)
                # Reflect the clamped value back into the field itself, so
                # an out-of-range typed value snaps visibly to the real
                # limit instead of silently keeping the rejected number
                # on screen.
                model.set_value(math.degrees(clamped_rad))
            finally:
                self._suppress_sync = False

        slider.model.add_value_changed_fn(_on_slider_changed)
        deg_field.model.add_end_edit_fn(_on_deg_commit)

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


def _step_physics_once(env, robot, panel, arm_cfg, gripper_cfg):
    """One physics step, reading current commanded targets fresh from the
    panel's sliders (the same body the main interactive loop runs) -
    factored out so run_self_test can drive real physics too, not just
    poke UI models."""
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
    return arm_positions


def run_self_test(env, robot, panel, arm_cfg, gripper_cfg, arm_joint_names, arm_limits, n_joints=3):
    """Automated live verification of Change 2 (degree text input), run
    under --self-test. Exercises the SAME model-layer calls a real drag or
    typed-Enter commit makes (see JointControlPanel._wire_sync), then
    steps real physics and reads robot.data.joint_pos to confirm the arm
    actually moved - not a UI-only check. Prints one PASS/FAIL line per
    check and a final summary; does not raise, so a failure doesn't kill
    the whole tool."""
    failures = []

    def check(label, cond, detail=""):
        status = "PASS" if cond else "FAIL"
        print(f"[SELFTEST] {status}: {label}{(' - ' + detail) if detail else ''}")
        if not cond:
            failures.append(label)

    for i in range(min(n_joints, len(arm_joint_names))):
        name = arm_joint_names[i]
        lo, hi = arm_limits[i]
        slider = panel.sliders[name]
        deg_field = panel.deg_fields[name]

        # 1) Simulate a slider drag (model.set_value is exactly what the
        # FloatSlider widget calls internally while being dragged) and
        # confirm the degree field's displayed value updates to match.
        drag_target_rad = max(lo, min(hi, math.radians(15.0)))
        slider.model.set_value(drag_target_rad)
        shown_deg = deg_field.model.get_value_as_float()
        check(
            f"{name}: drag->degree-field sync",
            abs(shown_deg - math.degrees(drag_target_rad)) < 0.05,
            f"slider set to {drag_target_rad:.4f} rad, degree field shows {shown_deg:.2f} deg "
            f"(expected {math.degrees(drag_target_rad):.2f})",
        )

        # 2) Simulate typing a degree value and pressing Enter. A real
        # edit session is begin_edit() (click into the field) -> set_value
        # (type) -> end_edit() (Enter/focus-loss) - empirically confirmed
        # (first --self-test run, 2026-07-24) that end_edit() alone,
        # without a preceding begin_edit(), does NOT fire the registered
        # add_end_edit_fn callback (_on_deg_commit never ran, so the
        # slider silently stayed at its previous value) - so all three
        # commits below open an edit session first, matching what the
        # widget itself does on a real click-then-type-then-Enter.
        typed_deg = max(math.degrees(lo), min(math.degrees(hi), 20.0))
        deg_field.model.begin_edit()
        deg_field.model.set_value(typed_deg)
        deg_field.model.end_edit()
        new_slider_rad = slider.model.get_value_as_float()
        expected_rad = max(lo, min(hi, math.radians(typed_deg)))
        check(
            f"{name}: typed-degree commit -> slider moves",
            abs(new_slider_rad - expected_rad) < 1e-6,
            f"typed {typed_deg:.2f} deg, slider now {new_slider_rad:.4f} rad (expected {expected_rad:.4f})",
        )

        # 3) Out-of-range input clamps to the real physical limit instead
        # of being silently accepted.
        beyond_deg = math.degrees(hi) + 15.0
        deg_field.model.begin_edit()
        deg_field.model.set_value(beyond_deg)
        deg_field.model.end_edit()
        clamped_rad = slider.model.get_value_as_float()
        clamped_deg_shown = deg_field.model.get_value_as_float()
        check(
            f"{name}: out-of-range degree input clamps to joint limit",
            abs(clamped_rad - hi) < 1e-6 and abs(clamped_deg_shown - math.degrees(hi)) < 0.05,
            f"requested {beyond_deg:.2f} deg (limit {math.degrees(hi):.2f}), "
            f"slider clamped to {clamped_rad:.4f} rad, field shows {clamped_deg_shown:.2f} deg",
        )

        # 4) The arm actually physically moves: command a real target via
        # a typed-degree commit, step real physics, and confirm
        # robot.data.joint_pos converges toward it.
        move_deg = max(math.degrees(lo), min(math.degrees(hi), -10.0))
        deg_field.model.begin_edit()
        deg_field.model.set_value(move_deg)
        deg_field.model.end_edit()
        move_target_rad = slider.model.get_value_as_float()
        arm_positions = None
        for _ in range(90):
            arm_positions = _step_physics_once(env, robot, panel, arm_cfg, gripper_cfg)
        actual_rad = arm_positions[i]
        check(
            f"{name}: arm physically reaches typed target",
            abs(actual_rad - move_target_rad) < 0.05,
            f"commanded {move_target_rad:.4f} rad via typed {move_deg:.2f} deg, "
            f"robot.data.joint_pos after 90 steps = {actual_rad:.4f} rad "
            f"(residual {abs(actual_rad - move_target_rad) * 1000:.1f} mrad)",
        )

        # Reset this joint back to 0 before moving to the next.
        slider.model.set_value(0.0)
        for _ in range(30):
            _step_physics_once(env, robot, panel, arm_cfg, gripper_cfg)

    # Gripper diagnostic (2026-07-24, added after a live user report that
    # the gripper slider doesn't move the gripper). Checks two candidate
    # root causes directly rather than guessing: (a) whether the live
    # robot.data.joint_pos_limits for the gripper joints still match the
    # GRIPPER_OPEN_POS/GRIPPER_CLOSED_POS constants the slider's own
    # min/max range was built from (a stale-range mismatch would mean the
    # slider lets you drag somewhere the real joint can't reach, or
    # commands something PhysX silently clamps/rejects before it can
    # move), and (b) whether commanding the gripper slider's own target
    # via _step_physics_once (the exact same per-frame path a real drag
    # drives) actually moves robot.data.joint_pos for both jaw joints.
    live_limits = robot.data.joint_pos_limits[0, gripper_cfg.joint_ids].cpu().tolist()
    print(
        f"[SELFTEST] gripper live joint_pos_limits = {live_limits} vs. assumed "
        f"[GRIPPER_CLOSED_POS={GRIPPER_CLOSED_POS}, GRIPPER_OPEN_POS={GRIPPER_OPEN_POS}] "
        f"(slider's own min/max range)"
    )
    limits_match = all(
        abs(lo_hi[0] - min(GRIPPER_CLOSED_POS, GRIPPER_OPEN_POS)) < 1e-4
        and abs(lo_hi[1] - max(GRIPPER_CLOSED_POS, GRIPPER_OPEN_POS)) < 1e-4
        for lo_hi in live_limits
    )
    check(
        "gripper: slider range matches live joint_pos_limits (no stale-range mismatch)",
        limits_match,
        f"live={live_limits}, assumed=[{min(GRIPPER_CLOSED_POS, GRIPPER_OPEN_POS)}, "
        f"{max(GRIPPER_CLOSED_POS, GRIPPER_OPEN_POS)}]",
    )

    gripper_slider = panel.sliders["gripper"]
    for target_name, target_val in (("closed", GRIPPER_CLOSED_POS), ("open", GRIPPER_OPEN_POS)):
        gripper_slider.model.set_value(target_val)
        gripper_positions = None
        for _ in range(90):
            _step_physics_once(env, robot, panel, arm_cfg, gripper_cfg)
            gripper_positions = robot.data.joint_pos[0, gripper_cfg.joint_ids].cpu().tolist()
        check(
            f"gripper: slider->{target_name} physically reaches target on both jaws",
            all(abs(p - target_val) < 0.002 for p in gripper_positions),
            f"commanded {target_val:.4f}, robot.data.joint_pos after 90 steps = {gripper_positions} "
            f"(residuals mm: {[round(abs(p - target_val) * 1000, 2) for p in gripper_positions]})",
        )
    gripper_slider.model.set_value(GRIPPER_OPEN_POS)  # leave open for the interactive session that follows
    for _ in range(30):
        _step_physics_once(env, robot, panel, arm_cfg, gripper_cfg)

    print("=" * 70)
    if failures:
        print(f"[SELFTEST] SUMMARY: {len(failures)} FAILURE(S): {failures}")
    else:
        print(f"[SELFTEST] SUMMARY: all checks passed for {min(n_joints, len(arm_joint_names))} joints.")
    print("=" * 70)


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
    print("Drag sliders (or type a value in deg and press Enter) in the 'AR4 Interactive")
    print("Joint Control' window to command joints live. Close the window / stop the")
    print("simulation app to exit.")
    print("=" * 70)

    if args_cli.self_test:
        run_self_test(env, robot, panel, arm_cfg, gripper_cfg, ARM_JOINT_NAMES, arm_limits)

    while simulation_app.is_running():
        _step_physics_once(env, robot, panel, arm_cfg, gripper_cfg)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
