"""Apply scripts/build_asset.py's new _fix_jaw2_collision_mesh_asymmetry()
directly to the ALREADY-BUILT asset (avoids a full URDF re-import, matching
the precedent scripts/build_asset.py's own _add_gripper_jaw2_drive commit
message describes for applying a single targeted USD fix without rebuilding
the whole pipeline).

Run via: /path/to/isaaclab.sh -p scripts/_apply_jaw2_collision_fix_standalone.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

usd_manifest = os.path.join(REPO_ROOT, "assets", "ar4_mk5", "usd_path.txt")
with open(usd_manifest) as f:
    usd_path = f.read().strip()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": True})

from build_asset import _fix_jaw2_collision_mesh_asymmetry  # noqa: E402

_fix_jaw2_collision_mesh_asymmetry(usd_path)

simulation_app.close()
