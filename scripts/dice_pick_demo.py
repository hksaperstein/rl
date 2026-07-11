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
from isaacsim.core.utils.stage import get_current_stage  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from tasks.franka.dice_scene_cfg import DIE_TYPES, DiceSceneCfg  # noqa: E402

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
# Matches dice_scene_cfg.py's _DICE_MASS (mass=0.01) - applied here directly since
# RigidObjectCfg's mass_props only *modifies* an already-present UsdPhysics.MassAPI
# (isaaclab/sim/schemas/schemas.py's modify_mass_properties silently no-ops if the
# schema isn't already on the prim), and these die USDs ship with none.
_DIE_MASS_KG = 0.01


def sample_dice_layout(seed: int, num_dice: int) -> list[tuple[float, float, float]]:
    """Rejection-samples `num_dice` (x, y, _DROP_Z) positions over the table
    region with minimum pairwise spacing `_MIN_SPACING`, seeded by `seed` for
    reproducibility."""
    rng = random.Random(seed)
    positions: list[tuple[float, float]] = []
    max_attempts = 20000
    attempts = 0
    while len(positions) < num_dice and attempts < max_attempts:
        attempts += 1
        x = rng.uniform(*_REGION_X)
        y = rng.uniform(*_REGION_Y)
        if all((x - px) ** 2 + (y - py) ** 2 >= _MIN_SPACING**2 for px, py in positions):
            positions.append((x, y))
    if len(positions) < num_dice:
        raise RuntimeError(
            f"Rejection sampling failed to place {num_dice} dice with min spacing {_MIN_SPACING}m "
            f"after {max_attempts} attempts (only placed {len(positions)})."
        )
    return [(x, y, _DROP_Z) for x, y in positions]


def apply_convex_hull_collision(stage, die_prim_path: str) -> int:
    """Makes the die prim at `die_prim_path` a dynamic rigid body with
    convex-hull collision, entirely at runtime - the dice USDs ship with NO
    physics schemas baked in at all (see dice_scene_cfg.py's module
    docstring), so `RigidObjectCfg`'s `rigid_props`/`collision_props`/
    `mass_props` (which only *modify* an already-present schema -
    `isaaclab/sim/schemas/schemas.py`'s `modify_rigid_body_properties`/
    `modify_collision_properties` both silently `return False` and do
    nothing if the API isn't already applied - confirmed by reading that
    source directly after Gate A's first run hit
    'Failed to find a rigid body ... ensure the prim has USD RigidBodyAPI
    applied') leave the die with no RigidBodyAPI, no CollisionAPI, and no
    mass at all. Applies, directly via pxr (same technique
    scripts/build_asset.py used for the AR4-era wedge asset):
      - UsdPhysics.RigidBodyAPI + PhysxSchema.PhysxRigidBodyAPI on the root
        prim (makes it a dynamic rigid body PhysX will actually simulate).
      - UsdPhysics.MassAPI on the root prim, mass=_DIE_MASS_KG (skips
        density-based mass computation, which would depend on the mesh's
        actual scale/units - not yet independently verified).
      - UsdPhysics.CollisionAPI + UsdPhysics.MeshCollisionAPI
        (approximation="convexHull") on every UsdGeom.Mesh prim in the
        subtree (PhysX rejects an exact-triangle-mesh collider on a
        dynamic body).
    Returns the number of mesh prims found/patched (0 is a bug - report it,
    don't silently proceed)."""
    root_prim = stage.GetPrimAtPath(die_prim_path)
    if not root_prim.IsValid():
        raise RuntimeError(f"Die prim path not found on stage: {die_prim_path}")

    UsdPhysics.RigidBodyAPI.Apply(root_prim)
    PhysxSchema.PhysxRigidBodyAPI.Apply(root_prim)
    mass_api = UsdPhysics.MassAPI.Apply(root_prim)
    mass_api.CreateMassAttr(_DIE_MASS_KG)

    mesh_count = 0
    for prim in Usd.PrimRange(root_prim):
        if prim.IsA(UsdGeom.Mesh):
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
            mesh_count += 1
    return mesh_count


def run_gate_a() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    positions = sample_dice_layout(args_cli.seed, len(DIE_TYPES))
    print(f"[GATE A] sampled dice layout (seed={args_cli.seed}):")
    for die_type, pos in zip(DIE_TYPES, positions):
        print(f"  {die_type}: x={pos[0]:.4f} y={pos[1]:.4f} z={pos[2]:.4f}")

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
