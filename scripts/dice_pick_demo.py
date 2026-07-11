# scripts/dice_pick_demo.py
"""Dice-pick commanded-grasp demo (see .superpowers/sdd/dice-demo-report.md):
Franka Panda + table + five dice (d4/d8/d10/d12/d20) + an angled RGB-D
perception camera (tasks/franka/dice_scene_cfg.py's DiceSceneCfg). Structured
around four gates, only Gate A implemented so far:

  A - dice settle: spawn the scene with a randomized, minimum-spacing dice
      layout, apply runtime rigid-body + convex-hull-collision schemas to
      each die (the USDs ship with no baked physics schemas at all - see
      dice_scene_cfg.py's module docstring and apply_convex_hull_collision's
      own docstring here for what that requires beyond just collision), let
      physics settle, verify every die's root height/position is sane, and
      save an RGB-D camera frame.
  P/G/V - perception bridge / scripted pick / full demo - not implemented
      yet, see the report for the resume plan.

.. code-block:: bash

    flock /tmp/rl_isaac_sim.lock -c "PYTHONUNBUFFERED=1 \\
        /home/saps/IsaacLab/isaaclab.sh -p scripts/dice_pick_demo.py --gate a --seed 42"

Never pass --headless - a display exists (DISPLAY=:1) and the user wants to
watch (see CLAUDE.md's Environment conventions).
"""

import argparse
import json
import os
import random
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Dice-pick commanded-grasp demo (gated).")
parser.add_argument(
    "--gate", type=str, choices=["a", "p", "g", "v", "full"], required=True, help="Which gate to run."
)
parser.add_argument(
    "--choice",
    type=str,
    default="d20",
    choices=["d4", "d8", "d10", "d12", "d20"],
    help="Commanded die type (used by gates G/V, not Gate A).",
)
parser.add_argument("--seed", type=int, default=42, help="Seed for the randomized dice layout.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
# NEVER set args_cli.headless - non-headless is environment law here (a
# display exists, DISPLAY=:1, the user wants to watch). Leave it at
# AppLauncher's own default (False / unset).

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows - isaaclab/pxr imports must come after AppLauncher."""

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.sim import schemas  # noqa: E402
from isaacsim.core.utils.stage import get_current_stage  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from tasks.franka.dice_scene_cfg import (  # noqa: E402
    DICE_CAMERA_POS,
    DICE_CAMERA_QUAT_WORLD,
    DIE_TYPES,
    DiceSceneCfg,
    _DICE_COLLISION_PROPS,
    _DICE_MASS,
    _DICE_RIGID_PROPS,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "outputs", "dice_demo", "gate_a")

# Table region the camera looks at (dice_scene_cfg.py's DICE_CAMERA_POS/QUAT
# is aimed at center (0.5, 0, 0.03)) - keep well inside Franka reach for
# later gates.
_REGION_X = (0.35, 0.65)
_REGION_Y = (-0.25, 0.25)
_MIN_SPACING = 0.09  # m, minimum pairwise center distance between dice
_DROP_Z = 0.10  # m, initial drop height before settling
_REGION_SLOP = 0.15  # m, allowed x/y drift from the sampled region after settling
_Z_FLOOR = 0.0  # m, below this -> fell through the table
_Z_CEIL = 0.10  # m, above this -> exploded/launched
_SETTLE_SECONDS = 3.0  # sim-time seconds to step before reading final state

# Camera intrinsics (computed from camera config)
_FOCAL_LENGTH = 24.0  # mm
_HORIZONTAL_APERTURE = 20.955  # mm
_IMAGE_WIDTH = 640
_IMAGE_HEIGHT = 480
_FX = _FY = _IMAGE_WIDTH * _FOCAL_LENGTH / _HORIZONTAL_APERTURE  # ≈ 733.0
_CX = _IMAGE_WIDTH / 2.0  # 320
_CY = _IMAGE_HEIGHT / 2.0  # 240
_PROJECTION_MARGIN = 50  # pixels, margin to keep dice away from frame edges


def _world_to_camera_frame(world_pos: np.ndarray, cam_pos: np.ndarray, cam_quat: np.ndarray) -> np.ndarray:
    """Transform world position to camera frame using camera pose.

    Args:
        world_pos: [x, y, z] in world frame
        cam_pos: camera position in world frame
        cam_quat: camera orientation quaternion (w, x, y, z) in world frame

    Returns:
        Position in camera frame [x_cam, y_cam, z_cam] where +X is forward, +Z is up.
    """
    # Compute world-to-camera rotation: inverse of camera's world rotation
    # Quat inverse for unit quat is (w, -x, -y, -z)
    quat_inv = np.array([cam_quat[0], -cam_quat[1], -cam_quat[2], -cam_quat[3]])

    # Translate to camera origin
    p_rel = world_pos - cam_pos

    # Rotate using quaternion: quat * point * quat_inv
    # For a vector, quat * v = (w^2 - ||xyz||^2) * v + 2(xyz · v) * xyz + 2w (xyz × v)
    w, x, y, z = quat_inv
    xyz = np.array([x, y, z])

    scalar = w * w - np.dot(xyz, xyz)
    cross = np.cross(xyz, p_rel)
    dot = np.dot(xyz, p_rel)

    p_cam = scalar * p_rel + 2 * dot * xyz + 2 * w * cross
    return p_cam


def _project_to_image(cam_pos: np.ndarray, z_world: float) -> tuple[float, float] | None:
    """Project a world position to image coordinates.

    Args:
        cam_pos: [x, y, z] in world frame
        z_world: z coordinate in world frame (die height, ~0.01m)

    Returns:
        (u, v) pixel coordinates, or None if behind camera or outside image.
    """
    p_cam = _world_to_camera_frame(np.array(cam_pos + (z_world,)), np.array(DICE_CAMERA_POS), np.array(DICE_CAMERA_QUAT_WORLD))

    # Check if point is in front of camera (z_cam > 0)
    if p_cam[2] <= 0:
        return None

    # Project to image: u = fx * x_cam / z_cam + cx, v = fy * y_cam / z_cam + cy
    u = _FX * p_cam[0] / p_cam[2] + _CX
    v = _FY * p_cam[1] / p_cam[2] + _CY

    return (u, v)


def sample_dice_layout(seed: int, num_dice: int) -> tuple[list[tuple[float, float, float]], dict[str, tuple[float, float]]]:
    """Rejection-samples `num_dice` (x, y, _DROP_Z) positions over the table
    region with minimum pairwise spacing `_MIN_SPACING`, and PROJECTION-AWARE
    framing (all dice must project into image with 50px margin), seeded by
    `seed` for reproducibility.

    Returns: (positions, projected_uv_dict) where projected_uv_dict maps
    die index to (u, v) pixel coordinates for auditability."""
    rng = random.Random(seed)
    positions: list[tuple[float, float]] = []
    projected_uv: dict[int, tuple[float, float]] = {}
    max_attempts = 50000  # Higher max since projection adds rejection
    attempts = 0
    while len(positions) < num_dice and attempts < max_attempts:
        attempts += 1
        x = rng.uniform(*_REGION_X)
        y = rng.uniform(*_REGION_Y)

        # Check spacing first (cheaper than projection)
        if not all((x - px) ** 2 + (y - py) ** 2 >= _MIN_SPACING**2 for px, py in positions):
            continue

        # Project to image and check bounds (with margin)
        proj = _project_to_image((x, y), _DROP_Z)
        if proj is None:
            continue
        u, v = proj

        if not (_PROJECTION_MARGIN <= u < _IMAGE_WIDTH - _PROJECTION_MARGIN and
                _PROJECTION_MARGIN <= v < _IMAGE_HEIGHT - _PROJECTION_MARGIN):
            continue

        # Accept this position
        positions.append((x, y))
        projected_uv[len(positions) - 1] = (u, v)

    if len(positions) < num_dice:
        raise RuntimeError(
            f"Rejection sampling failed to place {num_dice} dice with min spacing {_MIN_SPACING}m "
            f"and projection within image bounds after {max_attempts} attempts (only placed {len(positions)})."
        )
    return ([(x, y, _DROP_Z) for x, y in positions], projected_uv)


def apply_convex_hull_collision(stage, die_prim_path: str) -> int:
    """Makes the die prim at `die_prim_path` a dynamic rigid body with
    convex-hull collision, entirely at runtime - the dice USDs ship with NO
    physics schemas baked in at all (see dice_scene_cfg.py's module
    docstring), so `RigidObjectCfg`'s `rigid_props`/`collision_props`/
    `mass_props` are silently no-op'd (they only *modify* existing schemas).

    Applies schemas directly via pxr then configures them via isaaclab's
    schema helpers (pattern from scripts/build_asset.py):
      - UsdPhysics.RigidBodyAPI + PhysxSchema.PhysxRigidBodyAPI on the root
        prim (makes it a dynamic rigid body).
      - Tuned rigid/mass/collision properties via modify_*_properties helpers
        (which now work because the schemas exist).
      - UsdPhysics.CollisionAPI + UsdPhysics.MeshCollisionAPI
        (approximation="convexHull") on every UsdGeom.Mesh prim.
    Returns the number of mesh prims found/patched."""
    root_prim = stage.GetPrimAtPath(die_prim_path)
    if not root_prim.IsValid():
        raise RuntimeError(f"Die prim path not found on stage: {die_prim_path}")

    # Apply bare schemas first (they don't exist on the USD).
    UsdPhysics.RigidBodyAPI.Apply(root_prim)
    PhysxSchema.PhysxRigidBodyAPI.Apply(root_prim)
    UsdPhysics.MassAPI.Apply(root_prim)

    # Now that schemas exist, apply the tuned properties via isaaclab helpers.
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


def run_gate_a() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    positions, projected_uv = sample_dice_layout(args_cli.seed, len(DIE_TYPES))
    print(f"[GATE A] sampled dice layout (seed={args_cli.seed}):")
    for idx, (die_type, pos) in enumerate(zip(DIE_TYPES, positions)):
        u, v = projected_uv[idx]
        print(f"  {die_type}: x={pos[0]:.4f} y={pos[1]:.4f} z={pos[2]:.4f} (projected: u={u:.0f} v={v:.0f})")

    scene_cfg = DiceSceneCfg(num_envs=1, env_spacing=4.0)
    for die_type, pos in zip(DIE_TYPES, positions):
        die_field = f"die_{die_type}"
        getattr(scene_cfg, die_field).init_state.pos = pos

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([1.6, -1.0, 1.2], [0.5, 0.0, 0.1])

    scene = InteractiveScene(scene_cfg)

    stage = get_current_stage()
    env_root = scene.env_prim_paths[0]
    for die_type in DIE_TYPES:
        die_prim_path = f"{env_root}/Die_{die_type}"
        mesh_count = apply_convex_hull_collision(stage, die_prim_path)
        print(
            f"[GATE A] applied RigidBodyAPI + convex-hull collision to {die_type} "
            f"({mesh_count} mesh prim(s) at {die_prim_path})"
        )
        if mesh_count == 0:
            raise RuntimeError(f"No UsdGeom.Mesh prims found under {die_prim_path} - collision plan failed.")

    sim.reset()
    print("[GATE A] sim.reset() complete. Settling physics...")

    sim_dt = sim.get_physics_dt()
    settle_steps = int(_SETTLE_SECONDS / sim_dt)
    for _ in range(settle_steps):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

    # Let RTX path tracer converge by rendering extra frames with physics frozen
    # (pattern from render_color_check.py). Without this, the camera captures
    # an unconverged/black first sample.
    print("[GATE A] rendering RTX convergence frames...")
    for _ in range(40):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

    print(f"[GATE A] settled after {settle_steps} steps ({_SETTLE_SECONDS}s sim time). Final die states:")
    print(f"{'die':<6} {'x':>10} {'y':>10} {'z':>10}")
    results = {}
    all_ok = True
    for die_type, sampled_pos in zip(DIE_TYPES, positions):
        die = scene[f"die_{die_type}"]
        pos_w = die.data.root_pos_w[0].cpu().numpy()
        # env_origins offset is (0,0,0) for a single env at the default
        # origin, but subtract it anyway so this is correct if num_envs>1
        # is ever used here.
        pos = pos_w - scene.env_origins[0].cpu().numpy()
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        print(f"{die_type:<6} {x:>10.4f} {y:>10.4f} {z:>10.4f}")
        results[die_type] = {"x": x, "y": y, "z": z, "sampled_x": sampled_pos[0], "sampled_y": sampled_pos[1]}

        z_ok = _Z_FLOOR <= z <= _Z_CEIL
        xy_ok = (abs(x - sampled_pos[0]) <= _REGION_SLOP) and (abs(y - sampled_pos[1]) <= _REGION_SLOP)
        if not z_ok:
            print(f"[GATE A] FAIL: {die_type} z={z:.4f} outside [{_Z_FLOOR}, {_Z_CEIL}] "
                  f"({'fell through' if z < _Z_FLOOR else 'exploded/launched'})")
            all_ok = False
        if not xy_ok:
            print(f"[GATE A] FAIL: {die_type} drifted outside sampled region +/- {_REGION_SLOP}m "
                  f"(x={x:.4f} vs sampled {sampled_pos[0]:.4f}, y={y:.4f} vs sampled {sampled_pos[1]:.4f})")
            all_ok = False

    if not all_ok:
        raise AssertionError("Gate A FAILED: see per-die diagnostics above (do not paper over - report actual numbers).")
    print("[GATE A] PASS: all five dice within z/xy bounds after settling.")

    # Ground truth for Gate P: settled world-frame root positions of each die.
    gt_dice = {die_type: [results[die_type]["x"], results[die_type]["y"], results[die_type]["z"]]
               for die_type in DIE_TYPES}
    gt_dice_path = os.path.join(OUT_DIR, "gt_dice.json")
    with open(gt_dice_path, "w") as f:
        json.dump(gt_dice, f, indent=2)
    print(f"[GATE A] saved ground truth: {gt_dice_path}")

    # Camera capture - extraction pattern from scripts/_perception_adapter.py.
    camera = scene["camera"]
    rgb = camera.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
    depth = camera.data.output["distance_to_image_plane"][0, ..., 0].cpu().numpy()
    intrinsics = camera.data.intrinsic_matrices[0].cpu().numpy()
    cam_pos_w = camera.data.pos_w[0].cpu().numpy()
    cam_quat_w_ros = camera.data.quat_w_ros[0].cpu().numpy()

    rgb_path = os.path.join(OUT_DIR, "rgb.png")
    depth_path = os.path.join(OUT_DIR, "depth.npy")
    params_path = os.path.join(OUT_DIR, "camera_params.json")

    Image.fromarray(rgb).save(rgb_path)
    np.save(depth_path, depth)
    with open(params_path, "w") as f:
        json.dump(
            {
                "intrinsic_matrix": intrinsics.tolist(),
                "pos_w": cam_pos_w.tolist(),
                "quat_w_ros": cam_quat_w_ros.tolist(),
                "width": int(camera.data.output["rgb"].shape[2]),
                "height": int(camera.data.output["rgb"].shape[1]),
            },
            f,
            indent=2,
        )
    print(f"[GATE A] saved camera frame: {rgb_path}, {depth_path}, {params_path}")
    print(f"[GATE A] rgb shape={rgb.shape} depth shape={depth.shape} intrinsics=\n{intrinsics}")
    print("[GATE A] DONE")


def main() -> None:
    if args_cli.gate == "a":
        run_gate_a()
    else:
        sys.exit(f"--gate {args_cli.gate} not implemented yet (only 'a' is implemented in this task).")


if __name__ == "__main__":
    main()
    print("[GATE A] holding window briefly before close...")
    time.sleep(3.0)
    simulation_app.close()
