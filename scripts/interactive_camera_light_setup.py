# scripts/interactive_camera_light_setup.py
"""Opens the dice-pick scene (Franka + table + 5 dice) in the Isaac Sim GUI
with the arm held in a representative "just closed on the die" pose (the
real joint configuration measured from an actual successful d8 pick,
scripts/dice_pick_demo.py's own seed=42 Gate V run), and provides two
buttons for interactively tuning and capturing this repo's ArmCamera pose
and light rig settings, instead of deriving/guessing them via coordinate
math and re-rendering to check (see kb/wiki/experiments/dice-pick-demo.md
and this session's own camera-angle back-and-forth for why that loop is
slow and error-prone).

"Save ArmCamera Pose" reads the viewport's OWN live camera (the one you
navigate with the mouse - orbit/pan/zoom as normal), converts its native
USD camera convention (local -Z forward, +Y up) into this repo's own
established CameraCfg "world" convention (local +X forward, +Z up - see
tasks/franka/dice_scene_cfg.py's DICE_CAMERA_POS/QUAT_WORLD comment for
where that convention was first established/verified), and prints ready-
to-paste ARM_CAMERA_POS/ARM_CAMERA_QUAT_WORLD constants.

"Save Light Settings" reads whatever Intensity/Exposure/Color Temperature
you've set (via Isaac Sim's own built-in Property panel - select the
/World/light or /World/sun prim in the Stage window, edit its attributes
directly, no custom UI needed for this part) and prints ready-to-paste
DomeLightCfg/DistantLightCfg constants.

Nothing here is auto-applied to the scene config file - copy the printed
values into tasks/franka/dice_scene_cfg.py yourself once you're happy.

.. code-block:: bash

    DISPLAY=:1 flock -o /tmp/rl_isaac_sim.lock -c "/home/saps/IsaacLab/isaaclab.sh -p scripts/interactive_camera_light_setup.py"
"""

import os
import sys

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser(description="Interactive ArmCamera/light setup GUI for the dice-pick scene.")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--seed", type=int, default=42, help="Dice layout seed (cosmetic only for this tool).")
args_cli = parser.parse_args()
args_cli.enable_cameras = True
if args_cli.headless:
    sys.exit("This tool is for live GUI interaction - run without --headless.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np  # noqa: E402
import omni.ui as ui  # noqa: E402
import omni.usd  # noqa: E402
import torch  # noqa: E402
from omni.kit.viewport.utility import get_active_viewport  # noqa: E402
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.sim import schemas  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

# dice_pick_demo.py is a SCRIPT, not an importable module - it calls
# parser.parse_args() at import time (requires --gate) and would crash this
# tool's own argparse. DIE_TYPES/the dice rigid-body-property constants
# live in dice_scene_cfg.py (a plain module, safe to import); the small
# amount of collision-schema-application logic normally in dice_pick_demo.py's
# apply_convex_hull_collision is inlined directly below instead of imported.
from tasks.franka.dice_scene_cfg import (  # noqa: E402
    _DICE_COLLISION_PROPS,
    _DICE_MASS,
    _DICE_RIGID_PROPS,
    DIE_TYPES,
    DiceSceneCfg,
)


def _apply_convex_hull_collision(stage, die_prim_path: str) -> int:
    """Inlined copy of scripts/dice_pick_demo.py's apply_convex_hull_collision
    (see that function's own docstring for the full rationale) - not
    imported because dice_pick_demo.py is a script, not a module."""
    root_prim = stage.GetPrimAtPath(die_prim_path)
    if not root_prim.IsValid():
        raise RuntimeError(f"Die prim path not found on stage: {die_prim_path}")
    UsdPhysics.RigidBodyAPI.Apply(root_prim)
    PhysxSchema.PhysxRigidBodyAPI.Apply(root_prim)
    UsdPhysics.MassAPI.Apply(root_prim)
    schemas.modify_rigid_body_properties(die_prim_path, _DICE_RIGID_PROPS, stage)
    schemas.modify_mass_properties(die_prim_path, _DICE_MASS, stage)
    schemas.modify_collision_properties(die_prim_path, _DICE_COLLISION_PROPS, stage)
    mesh_count = 0
    for prim in Usd.PrimRange(root_prim):
        if prim.IsA(UsdGeom.Mesh):
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
            mesh_count += 1
    return mesh_count

# Real joint configuration measured from an actual successful d8 pick
# (scripts/dice_pick_demo.py Gate V, seed=42, printed at the first step of
# stage4_lift, i.e. immediately after gripper closure - see this repo's own
# outputs/dice_demo/gate_v/dice_pick_demo_stdout.log-equivalent console
# output from this session's own runs). Held here as a static target so the
# camera/light setup happens against a representative "grip just closed"
# pose, not the arm's idle rest configuration.
_AT_CLOSURE_ARM_JOINT_POS = [-0.104, 0.469, -0.069, -2.451, 0.143, 2.918, 0.481]
_AT_CLOSURE_GRIPPER_POS = 0.0  # closed


def _rotate_by_quat(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = q_wxyz
    qv = np.array([x, y, z])
    t = 2 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def _matrix_to_quat_wxyz(rot_cols: np.ndarray) -> np.ndarray:
    """rot_cols: 3x3 matrix whose COLUMNS are the local X/Y/Z axes expressed
    in world frame (i.e. a proper rotation matrix). Returns (w,x,y,z)."""
    R = rot_cols
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = np.argmax([R[0, 0], R[1, 1], R[2, 2]])
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    return np.array([w, x, y, z])


def _capture_viewport_camera_as_arm_camera_cfg() -> str:
    """Reads the live viewport camera's world transform, converts from raw
    USD camera convention (local -Z forward, +Y up) to this repo's own
    CameraCfg "world" convention (local +X forward, +Z up - see
    tasks/franka/dice_scene_cfg.py's own established/verified convention),
    and returns ready-to-paste Python constants."""
    viewport = get_active_viewport()
    cam_path = viewport.camera_path
    stage = omni.usd.get_context().get_stage()
    cam_prim = stage.GetPrimAtPath(cam_path)
    xform = UsdGeom.Xformable(cam_prim)
    world_mat: Gf.Matrix4d = xform.ComputeLocalToWorldTransform(0)

    pos = np.array([world_mat[3][0], world_mat[3][1], world_mat[3][2]])
    # Matrix4d rows 0-2 are the local X/Y/Z axes expressed in world frame
    # (Gf.Matrix4d row-vector convention) - native USD camera: row0=local+X
    # (right), row1=local+Y (up), row2=local+Z (backward, since camera looks
    # down -Z).
    local_x_world = np.array([world_mat[0][0], world_mat[0][1], world_mat[0][2]])
    local_y_world = np.array([world_mat[1][0], world_mat[1][1], world_mat[1][2]])
    local_z_world = np.array([world_mat[2][0], world_mat[2][1], world_mat[2][2]])
    native_forward = -local_z_world / np.linalg.norm(local_z_world)  # camera looks down -Z
    native_up = local_y_world / np.linalg.norm(local_y_world)

    # Rebuild in THIS REPO'S convention: local +X -> native_forward,
    # local +Z -> up hint, local +Y = right (same construction already
    # validated against this repo's own existing DICE_CAMERA_POS/QUAT_WORLD
    # and ARM_CAMERA_POS/QUAT_WORLD constants earlier this session).
    f = native_forward
    up_hint = native_up if abs(np.dot(native_up, f)) < 0.99 else np.array([0.0, 0.0, 1.0])
    right = np.cross(up_hint, f)
    right /= np.linalg.norm(right)
    true_up = np.cross(f, right)
    true_up /= np.linalg.norm(true_up)
    R = np.column_stack([f, right, true_up])
    q = _matrix_to_quat_wxyz(R)

    return (
        f"ARM_CAMERA_POS = ({pos[0]:.6f}, {pos[1]:.6f}, {pos[2]:.6f})\n"
        f"ARM_CAMERA_QUAT_WORLD = ({q[0]:.8f}, {q[1]:.8f}, {q[2]:.8f}, {q[3]:.8f})"
    )


def _capture_light_settings() -> str:
    stage = omni.usd.get_context().get_stage()
    lines = []
    for prim_path, label in [("/World/light", "DomeLightCfg (light)"), ("/World/sun", "DistantLightCfg (sun)")]:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            lines.append(f"# {label}: prim not found at {prim_path}")
            continue
        intensity = prim.GetAttribute("inputs:intensity").Get()
        exposure = prim.GetAttribute("inputs:exposure").Get()
        color_temp = prim.GetAttribute("inputs:colorTemperature").Get()
        enable_ct = prim.GetAttribute("inputs:enableColorTemperature").Get()
        lines.append(
            f"# {label}: intensity={intensity}, exposure={exposure}, "
            f"enable_color_temperature={enable_ct}, color_temperature={color_temp}"
        )
    return "\n".join(lines)


class SetupPanel:
    def __init__(self):
        self._window = ui.Window("ArmCamera / Light Interactive Setup", width=560, height=260)
        with self._window.frame:
            with ui.VStack(spacing=8, style={"font_size": 14}):
                ui.Label(
                    "Navigate the viewport (mouse orbit/pan/zoom) to frame the grip.\n"
                    "Select /World/light or /World/sun in the Stage window to edit\n"
                    "Intensity/Exposure/Color Temperature in the Property panel.\n"
                    "Click a button below when happy - values print to the console."
                )
                cam_btn = ui.Button("Save current viewport view as ArmCamera pose", height=32)
                cam_btn.set_clicked_fn(self._on_save_camera)
                self._cam_output = ui.Label("", word_wrap=True)
                light_btn = ui.Button("Save current light settings", height=32)
                light_btn.set_clicked_fn(self._on_save_lights)
                self._light_output = ui.Label("", word_wrap=True)

    def _on_save_camera(self):
        try:
            result = _capture_viewport_camera_as_arm_camera_cfg()
        except Exception as e:  # noqa: BLE001 - surface any capture failure directly to the console/UI
            result = f"# CAPTURE FAILED: {e!r}"
        print("=" * 70)
        print("[CAPTURED ArmCamera pose]")
        print(result)
        print("=" * 70)
        self._cam_output.text = result

    def _on_save_lights(self):
        result = _capture_light_settings()
        print("=" * 70)
        print("[CAPTURED light settings]")
        print(result)
        print("=" * 70)
        self._light_output.text = result


def main() -> None:
    # Uses DiceSceneCfg's own already-authored placeholder die positions
    # (die_d4/d8/d10/d12/d20's class-level init_state.pos) rather than a
    # randomized seeded layout - this tool is for camera/light framing, not
    # reproducing a specific run's exact dice arrangement.
    scene_cfg = DiceSceneCfg(num_envs=1, env_spacing=4.0)

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([1.6, -1.0, 1.2], [0.5, 0.0, 0.1])

    scene = InteractiveScene(scene_cfg)

    stage = omni.usd.get_context().get_stage()
    env_root = scene.env_prim_paths[0]
    for die_type in DIE_TYPES:
        _apply_convex_hull_collision(stage, f"{env_root}/Die_{die_type}")

    sim.reset()
    scene.reset()

    # Settle the dice under gravity before holding the arm pose (same
    # 180-step/3s settle window dice_pick_demo.py's own spawn_scene_and_settle
    # uses).
    sim_dt = sim.get_physics_dt()
    for _ in range(180):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

    robot = scene["robot"]
    from isaaclab.managers import SceneEntityCfg  # noqa: E402

    arm_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"])
    arm_cfg.resolve(scene)
    gripper_cfg = SceneEntityCfg("robot", joint_names=["panda_finger.*"])
    gripper_cfg.resolve(scene)

    arm_target = torch.tensor([_AT_CLOSURE_ARM_JOINT_POS], device=scene.device)
    gripper_target = torch.tensor(
        [[_AT_CLOSURE_GRIPPER_POS, _AT_CLOSURE_GRIPPER_POS]], device=scene.device
    )

    panel = SetupPanel()

    print("=" * 70)
    print("[READY] Interactive ArmCamera/light setup window open.")
    print("Arm is held in a representative 'just closed the gripper' pose.")
    print("Navigate the viewport and use the panel buttons to capture values.")
    print("Close the window / stop the simulation app to exit.")
    print("=" * 70)

    while simulation_app.is_running():
        robot.set_joint_position_target(arm_target, joint_ids=arm_cfg.joint_ids)
        robot.set_joint_position_target(gripper_target, joint_ids=gripper_cfg.joint_ids)
        robot.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim_dt)


if __name__ == "__main__":
    main()
    simulation_app.close()
