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
import omni.usd  # noqa: E402
import torch  # noqa: E402
from pxr import Gf, Usd, UsdGeom  # noqa: E402

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: E402
from isaaclab.envs import ManagerBasedEnv  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.utils.math import compute_pose_error, matrix_from_quat, quat_inv, subtract_frame_transforms  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from tasks.ar4.grasp_verify_env_cfg import Ar4GraspVerifyEnvCfg  # noqa: E402
from tasks.ar4.robot_cfg import ARM_JOINT_NAMES, GRIPPER_CLOSED_POS, GRIPPER_JOINT_NAMES, GRIPPER_OPEN_POS  # noqa: E402

# Same reasoning as interactive_joint_demo.py: the default arm PD gains
# (stiffness=40, damping=4) let the arm sag noticeably under gravity at a
# held target - raised here (this env instance only) so dragging a slider
# actually holds the arm where you put it instead of drooping back down.
ARM_STIFFNESS = 2500.0
ARM_DAMPING = 45.0

# --- End-effector drag-via-IK gizmo -----------------------------------------
# New feature (2026-07-25, direct user request, added ON TOP of the existing
# per-joint sliders/degree fields - NOT a replacement for them): lets you
# freely position the gripper by dragging a small visible marker prim
# (EE_MARKER_PATH) around in the viewport with Isaac Sim's own NATIVE
# prim-move gizmo (Kit shows this automatically for any selected Xformable
# prim - no custom omni.ui.scene manipulator widget needed, see
# _spawn_ee_marker below), while a continuous per-physics-step damped-least-
# squares (DLS) IK solve drives the real arm joints toward wherever the
# marker currently is.
#
# Investigated first, per this task's own brief, whether Isaac Sim's native
# viewport drag (the same mechanism that now moves the cube, via PhysX's
# built-in mouse-grab-a-rigid-body interaction) could be used DIRECTLY on one
# of the arm's own articulation links instead of a separate marker -
# judged impractical within reasonable scope and NOT implemented: (a) each
# arm link is not a free 6DOF rigid body like the cube, it's constrained by
# its joint to the rest of the chain, so PhysX's generic body-grab
# interaction (even if it engages on an articulation link at all, which is
# not confirmed/documented anywhere this session found) would have to fight
# both the joint constraint itself AND this script's own ARM_STIFFNESS=2500
# position-hold PD gain (raised specifically, see above, so a commanded
# target actually holds instead of sagging) - most likely resulting in
# either no visible motion (successfully resisted) or an unpredictable
# fight between the mouse-drag spring force and the joint drive, not a clean
# "grab and place" interaction; (b) even if it did move, translating an
# arbitrary 6DOF nudge on a single link back into one joint-limit-respecting
# angle is the same single-DOF-projection problem the EE IK solve below
# already exists to solve, just applied to the wrong body. The EE-drag path
# below is also the more directly useful interaction for "let me position
# the gripper where I want it" (this whole tool's grasp-orientation-
# investigation motivation), so it's the prioritized/only deliverable here -
# reported explicitly rather than silently dropped, per this task's own
# instruction.
#
# IK math below (the world->root Jacobian-frame rotation, the _EE_OFFSET
# rigid-offset correction from link_6's own origin to the true gripper
# pinch point, and the DLS solve/step-clamp structure itself) is PORTED
# VERBATIM from scripts/grasp_demo_v2.py's own already-validated
# implementation (_world_jacobian_to_root_frame, _ee_point_pos_and_jacobian,
# and the continuous per-step re-solve loop structure of polish_from_seed) -
# not re-derived - per this task's own instruction to reuse that reference
# implementation rather than re-deriving the math. grasp_demo_v2.py itself
# can't be imported as a library (it's a standalone script that calls
# argparse.parse_args() at import time, which would collide with this
# script's own argparse parser), so the specific pure-math pieces are
# copied here instead, with this comment as the paper trail back to the
# original.
#
# Target ORIENTATION is held FIXED at whatever the gripper's actual live
# orientation is at the moment EE-drag mode is switched on (see the main
# loop's ik_active/prev_ik_active handling) - NOT snapped to
# grasp_demo_v2.py's canonical straight-down orientation
# (_build_canonical_target_quat_b). Judgment call: this tool is for general
# manual positioning (this task's own framing - "let me position the
# gripper where I want it"), not specifically for grasping, so forcing a
# sudden reorientation to straight-down the instant the checkbox is ticked
# (a potentially large, surprising jump, and one that might not even be
# reachable from wherever the arm currently is) seemed like the wrong
# default; freezing the CURRENT orientation instead means enabling drag
# mode never causes a surprise jump. grasp_demo_v2.py's
# _build_canonical_target_quat_b remains available there if a fixed-
# orientation variant is wanted later.
_EE_OFFSET = (0.0, 0.0, 0.036)  # matches grasp_demo_v2.py's _EE_OFFSET exactly - see that file for how it was measured
EE_MARKER_PATH = "/World/ee_target_marker"
EE_MARKER_RADIUS = 0.01
IK_LAMBDA_VAL = 0.02  # matches grasp_demo_v2.py's own default LAMBDA_VAL
IK_POS_STEP_MAX = 0.03  # matches grasp_demo_v2.py's POLISH_STEP_MAX
IK_ROT_STEP_MAX = 0.03  # matches grasp_demo_v2.py's POLISH_ROT_STEP_MAX


def _world_jacobian_to_root_frame(jacobian_w: torch.Tensor, root_quat_w: torch.Tensor) -> torch.Tensor:
    """Ported verbatim from scripts/grasp_demo_v2.py - see that module's own
    docstring for the full derivation/validation. Required because AR4's
    base carries a real 180-degree yaw (tasks/ar4/robot_cfg.py's
    init_state), so root_physx_view.get_jacobians()'s WORLD-frame Jacobian
    must be rotated into the root frame before use against root-frame
    position/orientation vectors."""
    root_rot_matrix = matrix_from_quat(quat_inv(root_quat_w))
    jacobian_b = jacobian_w.clone()
    jacobian_b[:, 0:3, :] = torch.bmm(root_rot_matrix, jacobian_b[:, 0:3, :])
    if jacobian_b.shape[1] > 3:
        jacobian_b[:, 3:6, :] = torch.bmm(root_rot_matrix, jacobian_b[:, 3:6, :])
    return jacobian_b


def _ee_point_pos_and_jacobian(ee_pos_b: torch.Tensor, ee_quat_b: torch.Tensor, jacobian_b: torch.Tensor):
    """Ported verbatim from scripts/grasp_demo_v2.py. Returns the ACTUAL
    gripper jaw pinch point's position (root frame) and Jacobian, given
    link_6's own pose/Jacobian and the constant local offset _EE_OFFSET -
    link_6's own raw origin is NOT the pinch point (a real ~36mm targeting
    bug when this distinction was first missed, see grasp_demo_v2.py's own
    docstring)."""
    offset_local = torch.tensor([_EE_OFFSET], device=ee_pos_b.device).expand(ee_pos_b.shape[0], 3)
    rot = matrix_from_quat(ee_quat_b)
    world_offset = torch.bmm(rot, offset_local.unsqueeze(-1)).squeeze(-1)
    point_pos_b = ee_pos_b + world_offset

    skew = torch.zeros(ee_pos_b.shape[0], 3, 3, device=ee_pos_b.device)
    skew[:, 0, 1], skew[:, 0, 2] = -world_offset[:, 2], world_offset[:, 1]
    skew[:, 1, 0], skew[:, 1, 2] = world_offset[:, 2], -world_offset[:, 0]
    skew[:, 2, 0], skew[:, 2, 1] = -world_offset[:, 1], world_offset[:, 0]
    jac_ang = jacobian_b[:, 3:6, :]
    point_jac_pos = jacobian_b[:, 0:3, :] - torch.bmm(skew, jac_ang)
    return point_pos_b, point_jac_pos


def _spawn_ee_marker(stage, initial_pos_w):
    """Creates a small, bright, purely-visual (no physics/collision APIs at
    all, so it can never trigger the cube-drag-style physics-tensor-
    invalidation crash the other in-flight task hit - this prim is never a
    PhysX rigid body in the first place) sphere prim at EE_MARKER_PATH, and
    selects it so Isaac Sim's own native viewport Move gizmo appears on it
    immediately - no custom omni.ui.scene manipulator needed, Kit shows
    this gizmo automatically for any selected Xformable prim. Returns the
    created prim."""
    sphere = UsdGeom.Sphere.Define(stage, EE_MARKER_PATH)
    sphere.GetRadiusAttr().Set(EE_MARKER_RADIUS)
    sphere.GetDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.15, 0.85)])
    xformable = UsdGeom.Xformable(sphere)
    xformable.ClearXformOpOrder()
    translate_op = xformable.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(*[float(v) for v in initial_pos_w]))
    omni.usd.get_context().get_selection().set_selected_prim_paths([EE_MARKER_PATH], False)
    return sphere.GetPrim()


def _set_marker_world_pos(marker_prim, pos_w):
    """Rebuilds the marker's xform ops to a single translate op at pos_w -
    used to snap the marker back onto the live EE pinch point every frame
    EE-drag mode is OFF (see the main loop), so enabling drag mode never
    starts with a jump between the marker and the arm's actual position.
    Unconditionally clears+rebuilds the op stack (rather than writing into
    whatever op happens to already be first) because Kit's own Move-gizmo
    drag interaction is free to rewrite the prim's xformOpOrder to a
    different op type (e.g. a single combined transform matrix op instead
    of the plain translate op this script created it with) - rebuilding
    avoids a type mismatch against whatever the gizmo last left behind. Only
    ever called with the marker's own current pinch-point position, so
    losing any incidental rotation the gizmo might have introduced is fine -
    the marker's orientation is never read (see _read_marker_world_pos)."""
    xformable = UsdGeom.Xformable(marker_prim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in pos_w]))


def _read_marker_world_pos(marker_prim):
    """Reads the marker's CURRENT world position via its full local-to-world
    transform (not just the translate op directly) - robust to however
    Kit's own Move-gizmo drag interaction chooses to author the prim's
    xformOps while dragging, rather than assuming it only ever touches the
    single translate op this script itself created it with."""
    matrix = UsdGeom.Xformable(marker_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    translation = matrix.ExtractTranslation()
    return [translation[0], translation[1], translation[2]]


def _ee_pinch_point_world(robot, arm_cfg):
    """Live world-frame position of the actual gripper pinch point (link_6's
    own body position + the world-rotated _EE_OFFSET) - used both to seed
    the marker's initial position and to keep it glued to the arm whenever
    EE-drag mode is off."""
    ee_pose_w = robot.data.body_pose_w[0, arm_cfg.body_ids[0]]
    ee_pos_w = ee_pose_w[0:3]
    ee_quat_w = ee_pose_w[3:7]
    rot = matrix_from_quat(ee_quat_w.unsqueeze(0))
    offset = torch.tensor([_EE_OFFSET], device=ee_pos_w.device)
    world_offset = torch.bmm(rot, offset.unsqueeze(-1)).squeeze(-1)[0]
    return (ee_pos_w + world_offset).cpu().tolist()


def _ik_step_joint_targets(env, robot, arm_cfg, ik_controller, ik_jacobi_idx, target_pos_w, target_quat_b, joint_pos_limits):
    """ONE continuous DLS IK step toward target_pos_w (world frame - the
    marker's own live position) / target_quat_b (root frame, frozen at
    whatever it was when drag mode was switched on) - the same per-step
    math grasp_demo_v2.py's polish_from_seed runs every physics step
    (bounded-magnitude position/rotation step, offset-corrected Jacobian,
    DLS solve, explicit joint-limit clamp on the result) but called once
    per frame from the main loop here instead of inside its own bounded
    convergence loop, since the marker can keep moving indefinitely while
    being dragged - the main loop itself IS the convergence loop here, just
    spread across live frames instead of a bounded batch. Returns the
    clamped joint_pos_des tensor; does NOT write it to the sim (the caller
    does that, same as the slider path in _step_physics_once)."""
    root_pos_w = robot.data.root_pose_w[:, 0:3]
    root_quat_w = robot.data.root_pose_w[:, 3:7]
    target_pos_b, _ = subtract_frame_transforms(
        root_pos_w, root_quat_w, torch.tensor([target_pos_w], device=env.device), root_quat_w
    )

    current_joint_pos = robot.data.joint_pos[:, arm_cfg.joint_ids].clone()
    ee_pose_w = robot.data.body_pose_w[:, arm_cfg.body_ids[0]]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])

    jacobian_w = robot.root_physx_view.get_jacobians()[:, ik_jacobi_idx, :, arm_cfg.joint_ids]
    jacobian_b = _world_jacobian_to_root_frame(jacobian_w, root_quat_w)

    point_pos_b, point_jac_pos = _ee_point_pos_and_jacobian(ee_pos_b, ee_quat_b, jacobian_b)
    jac_ang = jacobian_b[:, 3:6, :]
    full_jac = torch.cat([point_jac_pos, jac_ang], dim=1)

    pos_error, rot_error = compute_pose_error(point_pos_b, ee_quat_b, target_pos_b, target_quat_b, rot_error_type="axis_angle")
    pos_norm = torch.norm(pos_error, dim=-1, keepdim=True)
    pos_step = pos_error / (pos_norm + 1e-8) * torch.clamp(pos_norm, max=IK_POS_STEP_MAX)
    rot_norm = torch.norm(rot_error, dim=-1, keepdim=True)
    rot_step = rot_error / (rot_norm + 1e-8) * torch.clamp(rot_norm, max=IK_ROT_STEP_MAX)
    delta_command = torch.cat([pos_step, rot_step], dim=-1)

    ik_controller.set_command(delta_command, ee_pos=point_pos_b, ee_quat=ee_quat_b)
    joint_pos_des = ik_controller.compute(point_pos_b, ee_quat_b, full_jac, current_joint_pos)

    lo = joint_pos_limits[:, :, 0]
    hi = joint_pos_limits[:, :, 1]
    return torch.clamp(joint_pos_des, min=lo, max=hi)


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
        n_rows = len(arm_joint_names) + 3  # + gripper row + EE-drag-toggle row + reset-button row
        self._window = ui.Window("AR4 Interactive Joint Control", width=640, height=60 + row_height * n_rows)
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
                    ui.Label("EE drag (IK)", width=110)
                    self.ik_drag_checkbox = ui.CheckBox(width=20)
                    self.ik_drag_checkbox.model.set_value(False)
                    ui.Label(
                        "checked = arm follows the pink marker sphere in the viewport (drag it with the "
                        "viewport's own Move gizmo) via live IK, instead of the sliders above",
                        width=460,
                    )
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

    def get_ik_drag_active(self):
        return self.ik_drag_checkbox.model.get_value_as_bool()

    def sync_sliders_to_arm(self, arm_positions):
        """Writes the arm's ACTUAL live joint positions into the sliders
        (and, via the existing _wire_sync callback, their paired degree
        fields) - called once, the moment EE-drag mode is switched back OFF
        (see the main loop), so control handoff back to the sliders doesn't
        fight the arm's current pose: the sliders would otherwise still be
        holding whatever stale target they had before drag mode was
        switched on, snapping the arm back there the instant the checkbox
        is unticked."""
        for name, val in zip(self.arm_joint_names, arm_positions):
            self.sliders[name].model.set_value(val)

    def update_readout(self, arm_positions, gripper_positions):
        for name, val in zip(self.arm_joint_names, arm_positions):
            self.readout_labels[name].text = f"actual: {val:+.4f} rad ({math.degrees(val):+6.1f} deg)"
        self.readout_labels["gripper"].text = f"actual: jaw1={gripper_positions[0]:+.4f}  jaw2={gripper_positions[1]:+.4f}"


def _step_physics_once(env, robot, panel, arm_cfg, gripper_cfg, ik_override_target=None):
    """One physics step. Arm targets normally come fresh from the panel's
    sliders every call (the same body the main interactive loop runs) -
    factored out so run_self_test can drive real physics too, not just
    poke UI models. When ik_override_target is given (a (1, n_arm_joints)
    tensor - see _ik_step_joint_targets), it's used as the arm target
    INSTEAD of the sliders for this one step, which is how EE-drag mode
    commands the arm without touching the slider-reading path at all."""
    if ik_override_target is not None:
        arm_target_t = ik_override_target
    else:
        arm_targets = panel.get_arm_targets()
        arm_target_t = torch.tensor([arm_targets], device=env.device)
    gripper_target_val = panel.get_gripper_target()
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


def run_ee_drag_self_test(env, robot, panel, arm_cfg, gripper_cfg, ik_controller, ik_jacobi_idx, joint_pos_limits, marker_prim):
    """Automated live verification of the EE-drag IK feature, run under
    --self-test (same convention as run_self_test above: exercise the SAME
    underlying calls a real mouse drag would trigger). Writing the marker
    prim's own translate op via _set_marker_world_pos IS exactly what Kit's
    native Move gizmo does while you drag it in the viewport - this isn't a
    simulation of a mouse event (Kit's viewport doesn't expose a scriptable
    hook to synthesize one), it's driving the actual same downstream state
    (the prim's transform) a real drag would leave behind, then running the
    exact same per-frame IK step the main loop runs off of it.

    Checks: (1) enabling EE-drag mode and moving the marker to a REACHABLE
    nearby point makes the arm's real measured pinch point converge close
    to it; (2) dragging the marker to a target far outside the arm's reach
    degrades gracefully - no exception, no NaN, every joint stays within
    its own robot.data.joint_pos_limits throughout - instead of erroring or
    commanding something unsafe."""
    failures = []

    def check(label, cond, detail=""):
        status = "PASS" if cond else "FAIL"
        print(f"[SELFTEST-EE] {status}: {label}{(' - ' + detail) if detail else ''}")
        if not cond:
            failures.append(label)

    panel.ik_drag_checkbox.model.set_value(True)
    current_pos_w = _ee_pinch_point_world(robot, arm_cfg)
    ee_pose_w = robot.data.body_pose_w[:, arm_cfg.body_ids[0]]
    _, ik_target_quat_b = subtract_frame_transforms(
        robot.data.root_pose_w[:, 0:3], robot.data.root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )
    _set_marker_world_pos(marker_prim, current_pos_w)

    # 1) Reachable target: nudge 4cm in +X/+Y/+Z from the current live
    # pinch point - well within a settled arm's local reach.
    reachable_target = [current_pos_w[0] + 0.04, current_pos_w[1] + 0.04, current_pos_w[2] + 0.04]
    _set_marker_world_pos(marker_prim, reachable_target)
    for _ in range(200):
        target_pos_w = _read_marker_world_pos(marker_prim)
        ik_target = _ik_step_joint_targets(
            env, robot, arm_cfg, ik_controller, ik_jacobi_idx, target_pos_w, ik_target_quat_b, joint_pos_limits
        )
        _step_physics_once(env, robot, panel, arm_cfg, gripper_cfg, ik_override_target=ik_target)
    achieved_pos_w = _ee_pinch_point_world(robot, arm_cfg)
    residual = math.dist(achieved_pos_w, reachable_target)
    check(
        "reachable EE-drag target converges (arm actually follows the marker)",
        residual < 0.02,
        f"target={['%.4f' % v for v in reachable_target]} achieved={['%.4f' % v for v in achieved_pos_w]} "
        f"residual={residual * 1000:.1f}mm",
    )

    # 2) Unreachable target (far outside the arm's workspace): must degrade
    # gracefully - no crash/NaN, every joint stays within its own real
    # physical limit the whole time (the explicit clamp inside
    # _ik_step_joint_targets is what's being verified here).
    unreachable_target = [2.0, 2.0, 2.0]
    _set_marker_world_pos(marker_prim, unreachable_target)
    lo_list = joint_pos_limits[0, :, 0].tolist()
    hi_list = joint_pos_limits[0, :, 1].tolist()
    within_limits_throughout = True
    saw_nan = False
    for _ in range(200):
        target_pos_w = _read_marker_world_pos(marker_prim)
        ik_target = _ik_step_joint_targets(
            env, robot, arm_cfg, ik_controller, ik_jacobi_idx, target_pos_w, ik_target_quat_b, joint_pos_limits
        )
        if torch.isnan(ik_target).any():
            saw_nan = True
            break
        arm_positions = _step_physics_once(env, robot, panel, arm_cfg, gripper_cfg, ik_override_target=ik_target)
        if any(math.isnan(v) for v in arm_positions):
            saw_nan = True
            break
        if not all(lo - 1e-3 <= v <= hi + 1e-3 for v, lo, hi in zip(arm_positions, lo_list, hi_list)):
            within_limits_throughout = False
    check("unreachable EE-drag target does not crash / produce NaN", not saw_nan)
    check(
        "unreachable EE-drag target stays within real joint limits throughout "
        "(graceful degradation, no unsafe overshoot)",
        within_limits_throughout,
    )

    # Reset: turn drag mode off, sync sliders to wherever the arm ended up,
    # and glue the marker back onto the (now-current) pinch point - exactly
    # what the main loop does on a real drag-mode-off transition, leaving a
    # clean, non-surprising state for the interactive session that follows.
    panel.ik_drag_checkbox.model.set_value(False)
    panel.sync_sliders_to_arm(robot.data.joint_pos[0, arm_cfg.joint_ids].cpu().tolist())
    _set_marker_world_pos(marker_prim, _ee_pinch_point_world(robot, arm_cfg))
    for _ in range(30):
        _step_physics_once(env, robot, panel, arm_cfg, gripper_cfg)

    print("=" * 70)
    if failures:
        print(f"[SELFTEST-EE] SUMMARY: {len(failures)} FAILURE(S): {failures}")
    else:
        print("[SELFTEST-EE] SUMMARY: all EE-drag checks passed.")
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
    # body_names=["link_6"] added (2026-07-25) alongside the existing
    # joint_names, so this same SceneEntityCfg's body_ids can also be used
    # for the EE-drag IK path below (link_6 is the body the gripper's own
    # _EE_OFFSET pinch point is rigidly attached to - see grasp_demo_v2.py).
    arm_cfg = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES, body_names=["link_6"])
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

    # EE-drag-via-IK setup (2026-07-25) - see the big comment block above
    # JointControlPanel for the full design rationale. ik_jacobi_idx mirrors
    # grasp_demo_v2.py's own indexing exactly (root_physx_view.get_jacobians()
    # is indexed one behind body_ids for a fixed-base articulation, since it
    # excludes the fixed root body itself).
    ik_jacobi_idx = arm_cfg.body_ids[0] - 1 if robot.is_fixed_base else arm_cfg.body_ids[0]
    ik_cfg = DifferentialIKControllerCfg(
        command_type="pose", use_relative_mode=True, ik_method="dls", ik_params={"lambda_val": IK_LAMBDA_VAL}
    )
    ik_controller = DifferentialIKController(ik_cfg, num_envs=env.num_envs, device=env.device)
    stage = omni.usd.get_context().get_stage()
    marker_prim = _spawn_ee_marker(stage, _ee_pinch_point_world(robot, arm_cfg))

    panel = JointControlPanel(ARM_JOINT_NAMES, arm_limits, GRIPPER_OPEN_POS, GRIPPER_CLOSED_POS)

    print("=" * 70)
    print("[READY] Interactive joint control window open.")
    print("Drag sliders (or type a value in deg and press Enter) in the 'AR4 Interactive")
    print("Joint Control' window to command joints live. Tick 'EE drag (IK)' in that same")
    print("window and drag the pink marker sphere in the viewport (Isaac Sim's own native")
    print("Move gizmo - click the sphere to select it if the gizmo isn't already showing)")
    print("to position the gripper directly instead; untick it to hand control back to the")
    print("sliders. Close the window / stop the simulation app to exit.")
    print("=" * 70)

    if args_cli.self_test:
        run_self_test(env, robot, panel, arm_cfg, gripper_cfg, ARM_JOINT_NAMES, arm_limits)
        run_ee_drag_self_test(env, robot, panel, arm_cfg, gripper_cfg, ik_controller, ik_jacobi_idx, joint_pos_limits, marker_prim)

    prev_ik_active = False
    ik_target_quat_b = None
    while simulation_app.is_running():
        ik_active = panel.get_ik_drag_active()

        if ik_active and not prev_ik_active:
            # Just switched ON: freeze the current live orientation as the
            # IK target orientation (see the module-level comment for why),
            # and snap the marker onto the arm's own current pinch point so
            # there's no jump the instant drag mode engages.
            ee_pose_w = robot.data.body_pose_w[:, arm_cfg.body_ids[0]]
            _, ik_target_quat_b = subtract_frame_transforms(
                robot.data.root_pose_w[:, 0:3], robot.data.root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            _set_marker_world_pos(marker_prim, _ee_pinch_point_world(robot, arm_cfg))
        elif not ik_active and prev_ik_active:
            # Just switched OFF: sync the sliders to the arm's actual
            # current pose so handing control back doesn't snap the arm to
            # a stale slider target.
            panel.sync_sliders_to_arm(robot.data.joint_pos[0, arm_cfg.joint_ids].cpu().tolist())

        if ik_active:
            target_pos_w = _read_marker_world_pos(marker_prim)
            ik_target = _ik_step_joint_targets(
                env, robot, arm_cfg, ik_controller, ik_jacobi_idx, target_pos_w, ik_target_quat_b, joint_pos_limits
            )
            _step_physics_once(env, robot, panel, arm_cfg, gripper_cfg, ik_override_target=ik_target)
        else:
            # Keep the marker glued to the live pinch point while inactive,
            # so it's always right where you'd expect when you next enable
            # drag mode instead of wherever it was last left mid-drag.
            _set_marker_world_pos(marker_prim, _ee_pinch_point_world(robot, arm_cfg))
            _step_physics_once(env, robot, panel, arm_cfg, gripper_cfg)

        prev_ik_active = ik_active

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
