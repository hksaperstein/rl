# scripts/record_viewport_camera_position.py
"""Interactive tool (2026-07-24, isaac-sim-video-capture-skill task):
lets a human navigate the Isaac Sim viewport (mouse orbit/pan/zoom, as
normal) and, when a spot looks good, capture it under a NAME into
`tasks/common/camera_positions.py`'s registry — so a future capture task
can reference it by name instead of hand-deriving/hand-tuning coordinates
fresh, the exact problem this whole skill/registry exists to fix (see
`~/.claude/skills/isaac-sim-video-capture/reference.md`).

Same architecture as the already-committed
`scripts/interactive_camera_light_setup.py` (a Kit `omni.ui` panel with a
button, not blocking console input) and for the same reason: this script's
own main loop must keep calling `sim.step(render=True)` continuously for
the viewport to actually render/respond to mouse navigation, so the
"capture" action has to be a non-blocking UI callback (an `omni.ui.Button`
click), not a blocking `input()` — blocking on console input would freeze
the render loop and the user couldn't see or move the camera at all.

The read mechanism itself (`tasks/common/viewport_camera_io.py`'s
`read_active_viewport_camera()`) is verified independently (without needing
a live human/GUI session) by `scripts/_verify_viewport_camera_roundtrip.py`
— see that script's own docstring.

Usage (per the standing "run non-headless, the user wants to watch"
convention — this tool REFUSES --headless, it needs a real GUI to navigate):
    DISPLAY=:1 flock -o /tmp/rl_isaac_sim.lock -c \\
        "/home/saps/IsaacLab/isaaclab.sh -p scripts/record_viewport_camera_position.py <name> \\
         [--scene franka-dice|empty] [--description TEXT] [--verified-subject TEXT] \\
         [--target-distance 0.5]"

`<name>` pre-fills the editable name field in the tool's own panel — the
panel supports capturing MULTIPLE named positions in one running session
(edit the name field, click Record again) so Isaac Sim doesn't need to be
relaunched between spots; you don't have to pass a new CLI invocation per
spot, though re-running the whole script per spot (as literally described
in the skill's reference.md) also works fine, just slower.

`--scene` picks what's loaded so there's something to frame:
  - "franka-dice" (default): the current primary task's scene
    (`tasks.franka.dice_scene_cfg.DiceSceneCfg`) with the arm held in a
    representative "just closed on the die" pose — reuses
    `scripts/interactive_camera_light_setup.py`'s own scene-setup code
    directly rather than re-deriving it.
  - "empty": a bare ground plane + default lighting, no robot — useful for
    a pure background/establishing-shot position, or when the scene itself
    doesn't matter for the specific spot being captured.
  Add more scenes by registering a new zero-argument scene-builder function
  in `_SCENE_BUILDERS` below (each must return an already-`reset()`
  `isaaclab.scene.InteractiveScene` or `None` for the empty case) — this is
  a narrow, mechanical extension point, not a redesign.

Every capture is written as `"recorded_by": "user"` (this script's own
provenance, distinct from the seed registry's `"derived"` entries) with
today's date and, until independently re-verified by extracting real
frames per the skill's own checklist, a `"verified_subject"` value that
says so explicitly rather than falsely claiming verification happened.
"""
import datetime
import os
import sys

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser(
    description="Interactively capture the active viewport camera into tasks/common/camera_positions.py."
)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument(
    "name",
    help="Registry name for the first capture (pre-fills the panel's editable name field; capture more names in the same session by editing that field and clicking Record again).",
)
parser.add_argument(
    "--scene",
    choices=["franka-dice", "empty"],
    default="franka-dice",
    help="What to load so there's something to frame (default: franka-dice).",
)
parser.add_argument("--description", default="", help="Initial description text (editable in the panel).")
parser.add_argument(
    "--verified-subject",
    dest="verified_subject",
    default="NOT YET VERIFIED — run the isaac-sim-video-capture skill's verification checklist "
    "(extract real frames, confirm non-black, confirm subject clearly visible) before trusting this entry.",
    help="Initial verified_subject text (editable in the panel) — see tasks/common/camera_positions.py's own docstring for what this field means.",
)
parser.add_argument(
    "--target-distance",
    type=float,
    default=0.5,
    help="Synthetic distance (m) along the camera's look direction used to compute the stored 'target' point (direction is exact/measured; this magnitude is cosmetic — see tasks/common/viewport_camera_io.py's docstring).",
)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
if args_cli.headless:
    sys.exit("This tool is for live GUI interaction — run without --headless (per this repo's standing convention).")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import omni.ui as ui  # noqa: E402
import omni.usd  # noqa: E402
from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.sim import schemas  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from tasks.common.camera_positions import save_camera_position  # noqa: E402
from tasks.common.viewport_camera_io import read_active_viewport_camera  # noqa: E402


def _apply_convex_hull_collision(stage, die_prim_path: str) -> int:
    """Inlined from scripts/interactive_camera_light_setup.py (itself
    inlined from scripts/dice_pick_demo.py, which is a script, not an
    importable module — see that file's own comment for why this can't
    just be imported)."""
    from tasks.franka.dice_scene_cfg import _DICE_COLLISION_PROPS, _DICE_MASS, _DICE_RIGID_PROPS

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
# (see scripts/interactive_camera_light_setup.py's identical constant and
# comment for full provenance) - held here as a static target so a
# franka-dice capture session frames a representative "grip just closed"
# pose, not the arm's idle rest configuration.
_AT_CLOSURE_ARM_JOINT_POS = [-0.104, 0.469, -0.069, -2.451, 0.143, 2.918, 0.481]
_AT_CLOSURE_GRIPPER_POS = 0.0  # closed


def _build_franka_dice_scene(sim: sim_utils.SimulationContext):
    from tasks.franka.dice_scene_cfg import DIE_TYPES, DiceSceneCfg

    scene_cfg = DiceSceneCfg(num_envs=1, env_spacing=4.0)
    scene = InteractiveScene(scene_cfg)

    stage = omni.usd.get_context().get_stage()
    env_root = scene.env_prim_paths[0]
    for die_type in DIE_TYPES:
        _apply_convex_hull_collision(stage, f"{env_root}/Die_{die_type}")

    sim.reset()
    scene.reset()

    sim_dt = sim.get_physics_dt()
    for _ in range(180):  # settle the dice under gravity, same window dice_pick_demo.py uses
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

    return scene


def _build_empty_scene(sim: sim_utils.SimulationContext):
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9))
    light_cfg.func("/World/light", light_cfg)
    sim.reset()
    return None


_SCENE_BUILDERS = {
    "franka-dice": _build_franka_dice_scene,
    "empty": _build_empty_scene,
}


class RecordPanel:
    def __init__(self, initial_name: str, initial_description: str, initial_verified_subject: str, target_distance: float):
        self._target_distance = target_distance
        self._window = ui.Window("Record Viewport Camera Position", width=620, height=340)
        with self._window.frame:
            with ui.VStack(spacing=8, style={"font_size": 14}):
                ui.Label(
                    "Navigate the viewport (mouse orbit/pan/zoom) to a spot you like.\n"
                    "Edit the fields below, then click Record — this writes to\n"
                    "tasks/common/camera_positions.py immediately. Repeat for more spots\n"
                    "in this same session (edit Name, click Record again)."
                )
                ui.Label("Name:")
                self._name_field = ui.StringField(height=24)
                self._name_field.model.set_value(initial_name)
                ui.Label("Description:")
                self._description_field = ui.StringField(height=24)
                self._description_field.model.set_value(initial_description)
                ui.Label("Verified subject (leave as-is unless you've actually run the verification checklist):")
                self._verified_field = ui.StringField(height=24)
                self._verified_field.model.set_value(initial_verified_subject)
                record_btn = ui.Button("Record current viewport view", height=32)
                record_btn.set_clicked_fn(self._on_record)
                self._output = ui.Label("", word_wrap=True)

    def _on_record(self):
        name = self._name_field.model.get_value_as_string().strip()
        if not name:
            self._output.text = "CAPTURE FAILED: name field is empty."
            print(self._output.text)
            return
        try:
            eye, target, focal_length, cam_path = read_active_viewport_camera(self._target_distance)
            entry = {
                "eye": eye,
                "target": target,
                "focal_length": focal_length if focal_length is not None else 24.0,
                "description": self._description_field.model.get_value_as_string(),
                "recorded_by": "user",
                "recorded_date": datetime.date.today().isoformat(),
                "verified_subject": self._verified_field.model.get_value_as_string(),
            }
            save_camera_position(name, entry)
            result = f"Saved '{name}': eye={eye} target={target} focal_length={focal_length} (camera_path={cam_path})"
        except Exception as e:  # noqa: BLE001 - surface any capture failure directly, don't silently swallow
            result = f"CAPTURE FAILED: {e!r}"
        print("=" * 70)
        print(f"[record_viewport_camera_position] {result}")
        print("=" * 70)
        self._output.text = result


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([1.6, -1.0, 1.2], [0.5, 0.0, 0.1])  # a reasonable starting view, not final

    build_fn = _SCENE_BUILDERS[args_cli.scene]
    scene = build_fn(sim)

    robot = None
    arm_cfg = None
    gripper_cfg = None
    arm_target = None
    gripper_target = None
    if args_cli.scene == "franka-dice":
        from isaaclab.managers import SceneEntityCfg
        import torch

        robot = scene["robot"]
        arm_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"])
        arm_cfg.resolve(scene)
        gripper_cfg = SceneEntityCfg("robot", joint_names=["panda_finger.*"])
        gripper_cfg.resolve(scene)
        arm_target = torch.tensor([_AT_CLOSURE_ARM_JOINT_POS], device=scene.device)
        gripper_target = torch.tensor([[_AT_CLOSURE_GRIPPER_POS, _AT_CLOSURE_GRIPPER_POS]], device=scene.device)

    panel = RecordPanel(args_cli.name, args_cli.description, args_cli.verified_subject, args_cli.target_distance)

    print("=" * 70)
    print(f"[READY] scene={args_cli.scene!r}. Navigate the viewport, edit the panel fields, click Record.")
    print("Close the window / stop the simulation app to exit.")
    print("=" * 70)

    sim_dt = sim.get_physics_dt()
    while simulation_app.is_running():
        if robot is not None:
            robot.set_joint_position_target(arm_target, joint_ids=arm_cfg.joint_ids)
            robot.set_joint_position_target(gripper_target, joint_ids=gripper_cfg.joint_ids)
            robot.write_data_to_sim()
        sim.step(render=True)
        if scene is not None:
            scene.update(sim_dt)


if __name__ == "__main__":
    main()
    simulation_app.close()
