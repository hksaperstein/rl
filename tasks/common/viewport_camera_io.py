# tasks/common/viewport_camera_io.py
"""Isaac-Sim-dependent read mechanism for the ACTIVE VIEWPORT's current
camera pose (2026-07-24, isaac-sim-video-capture-skill task).

Unlike `tasks/common/camera_positions.py` (deliberately import-light, no
Isaac Sim deps), this module imports `omni`/`pxr` and is only importable
from inside an already-bootstrapped `SimulationApp` (i.e. after
`AppLauncher` has run) — exactly like every other Isaac-Sim-touching module
in this repo.

API confirmed by introspecting the actually-installed Isaac Lab source
(`isaaclab.sim.SimulationContext`, v2.3.1, the version this project is
pinned to) rather than assumed from memory: IsaacLab's OWN
`SimulationContext.set_camera_view`/`.render` methods internally import
`from omni.kit.viewport.utility import get_active_viewport` — the identical
import this module uses below. This module's read-side transform-decoding
logic (position = row 3, forward = -row 2 of the camera prim's
`ComputeLocalToWorldTransform`) also matches the convention already
committed and used in `scripts/interactive_camera_light_setup.py`'s own
`_capture_viewport_camera_as_arm_camera_cfg()` (native USD/OpenGL camera
convention: local -Z is forward, local +Y is up, translation is row 3 of a
row-vector `Gf.Matrix4d`) — not re-derived from scratch.

Verified via `scripts/_verify_viewport_camera_roundtrip.py` (sets a known
eye/target via `SimulationContext.set_camera_view`, reads it back via this
module's `read_active_viewport_camera()`, checks the readback matches).
"""
from __future__ import annotations

import numpy as np
import omni.usd
from omni.kit.viewport.utility import get_active_viewport
from pxr import Gf, UsdGeom


def read_active_viewport_camera(
    target_distance: float = 0.5,
) -> tuple[tuple[float, float, float], tuple[float, float, float], float | None, str]:
    """Read the active viewport's current camera as (eye, target,
    focal_length, camera_prim_path).

    `eye` is the camera's real world-frame position.

    `target` is a SYNTHETIC point placed `target_distance` meters along the
    camera's actual (real, measured) look direction from `eye` — only the
    DIRECTION (target - eye) is guaranteed to match the live camera; the
    distance is an arbitrary placeholder, not a measured subject distance.
    This is deliberate and harmless for this registry's purposes: every
    consumer of a stored (eye, target) pair in this repo reconstructs the
    camera's orientation via
    `isaaclab.utils.math.create_rotation_matrix_from_view(eye, target, ...)`,
    which depends only on the (target - eye) *direction*, not its
    magnitude — any point on the correct ray reproduces the identical
    camera orientation. Verified directly by
    `scripts/_verify_viewport_camera_roundtrip.py`.

    `focal_length` is `None` if the active viewport's camera prim has no
    authored `focalLength` attribute (shouldn't normally happen for a real
    `UsdGeom.Camera` prim, but read defensively rather than assuming).

    Raises `RuntimeError` if the active viewport's camera prim can't be
    resolved (e.g. called before a stage/viewport actually exists).
    """
    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("get_active_viewport() returned None — no active viewport (no GUI/offscreen render?).")
    cam_path = viewport.camera_path
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No active USD stage — call this after the SimulationContext/scene is constructed.")
    cam_prim = stage.GetPrimAtPath(cam_path)
    if not cam_prim.IsValid():
        raise RuntimeError(f"Active viewport camera prim not found at {cam_path!r}.")

    xform = UsdGeom.Xformable(cam_prim)
    world_mat: Gf.Matrix4d = xform.ComputeLocalToWorldTransform(0)

    # Gf.Matrix4d row-vector convention (identical to
    # interactive_camera_light_setup.py's own already-committed derivation):
    # row 3 = world translation; row 2 = local +Z expressed in world frame.
    # Native USD camera convention: the camera looks down its own local -Z,
    # so the real look direction is the NEGATIVE of row 2.
    eye = np.array([world_mat[3][0], world_mat[3][1], world_mat[3][2]])
    local_z_world = np.array([world_mat[2][0], world_mat[2][1], world_mat[2][2]])
    norm = np.linalg.norm(local_z_world)
    if norm < 1e-8:
        raise RuntimeError(f"Degenerate camera transform at {cam_path!r} (zero-length local Z axis).")
    forward = -local_z_world / norm

    target = eye + forward * target_distance

    focal_length = None
    cam_schema = UsdGeom.Camera(cam_prim)
    if cam_schema:
        attr = cam_schema.GetFocalLengthAttr()
        if attr and attr.HasAuthoredValue():
            focal_length = float(attr.Get())

    return tuple(eye.tolist()), tuple(target.tolist()), focal_length, str(cam_path)
