# scripts/_verify_viewport_camera_roundtrip.py
"""One-off diagnostic (2026-07-24, isaac-sim-video-capture-skill task):
verifies tasks/common/viewport_camera_io.py's read_active_viewport_camera()
mechanism — the same mechanism scripts/record_viewport_camera_position.py
uses to capture a user-positioned camera in a live GUI session — actually
reads back a KNOWN camera pose correctly.

Context: the desktop was unreachable this session, so the real interactive
part (a human moving the camera in the GUI) could not be live-tested here
and is left for the user's own upcoming session (see
~/.claude/skills/isaac-sim-video-capture/reference.md). This script instead
proves the READ side of the mechanism independently, without a human or a
GUI: it sets the active viewport's default camera to a known (eye, target)
via isaaclab.sim.SimulationContext.set_camera_view (the exact primitive
scripts/interactive_camera_light_setup.py already uses, and the same
underlying USD camera prim transform a human's own GUI navigation
ultimately writes to), then reads that pose back via
read_active_viewport_camera() and checks:
  - the readback eye matches the known eye (tight tolerance).
  - the readback DIRECTION (target - eye, normalized) matches the known
    direction (tight tolerance). The synthetic 'target' distance itself is
    NOT expected to match the known target's distance — see
    viewport_camera_io.py's own docstring for why only direction is
    real/measured.

No robot/task asset needed (deliberately lighter than a full scene launch)
— just a bare SimulationContext and the default viewport camera, so this
can run on a fresh cloud instance without shipping/building any
gitignored asset.

Run (headless, cloud - desktop unreachable; --enable_cameras is required
even headless so the active viewport actually has an offscreen render
product to query):
    flock -o /tmp/rl_isaac_sim.lock -c \\
        "PYTHONUNBUFFERED=1 python scripts/_verify_viewport_camera_roundtrip.py --headless --enable_cameras"
(inside the cloud instance's isaac-venv, cwd ~/rl, per
docs/cloud/franka-cloud-shakedown.md's install recipe)

Run (local, non-headless, if ever run on a machine with Isaac Lab
installed and a display):
    DISPLAY=:1 flock -o /tmp/rl_isaac_sim.lock -c \\
        "/home/saps/IsaacLab/isaaclab.sh -p scripts/_verify_viewport_camera_roundtrip.py"

Exit code 0 = PASS, 1 = FAIL (also printed explicitly either way).
"""
import os
import sys

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser(description="Round-trip verify the active-viewport-camera read mechanism.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # needed for offscreen render / an active viewport to exist, even headless

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from tasks.common.viewport_camera_io import read_active_viewport_camera  # noqa: E402

# Arbitrary but deliberately non-degenerate ground truth (not axis-aligned,
# not colinear through the world origin) so the test can't accidentally
# pass via some degenerate direction.
_KNOWN_EYE = (1.234, -0.567, 0.890)
_KNOWN_TARGET = (0.111, 0.222, 0.05)

_EYE_TOL_M = 1e-3
_ANGLE_TOL_DEG = 0.5


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(list(_KNOWN_EYE), list(_KNOWN_TARGET))
    # Render a few frames so the viewport/render product actually updates
    # before reading it back (matches this repo's own render-then-read
    # pattern elsewhere, e.g. scripts/_record_jaw_fix_open_close_cycle.py's
    # per-step demo_camera.update()+read).
    for _ in range(5):
        sim.render()

    eye_rb, target_rb, focal_length, cam_path = read_active_viewport_camera(target_distance=0.5)

    known_dir = np.array(_KNOWN_TARGET) - np.array(_KNOWN_EYE)
    known_dir /= np.linalg.norm(known_dir)
    rb_dir = np.array(target_rb) - np.array(eye_rb)
    rb_dir /= np.linalg.norm(rb_dir)

    eye_err_m = float(np.linalg.norm(np.array(eye_rb) - np.array(_KNOWN_EYE)))
    angle_err_deg = float(np.degrees(np.arccos(np.clip(np.dot(known_dir, rb_dir), -1.0, 1.0))))

    print("=" * 100)
    print(f"[camera_path]  {cam_path}")
    print(f"[known]        eye={_KNOWN_EYE} target={_KNOWN_TARGET}")
    print(f"[readback]     eye={eye_rb} target={target_rb} focal_length={focal_length}")
    print(f"[error]        eye_err_m={eye_err_m:.6f} (tol {_EYE_TOL_M}) direction_angle_err_deg={angle_err_deg:.6f} (tol {_ANGLE_TOL_DEG})")
    ok = eye_err_m < _EYE_TOL_M and angle_err_deg < _ANGLE_TOL_DEG
    print(f"[RESULT] {'PASS' if ok else 'FAIL'}")
    print("=" * 100)

    simulation_app.close()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
