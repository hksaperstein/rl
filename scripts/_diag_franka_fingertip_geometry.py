# scripts/_diag_franka_fingertip_geometry.py
"""Desk-check diagnostic (d4 rung-1 pad-geometry Task 0, 2026-07-15, see
docs/superpowers/specs/2026-07-15-d4-rung1-pad-geometry-design.md and
.superpowers/sdd/task-0-brief.md): measures the Franka Panda fingertip's
REAL tip geometry directly from the stock asset mesh via plain pxr USD
inspection, to replace the research pass's carried-over ~14-18mm tip-extent
/ 17.5mm tip-width estimate with an actual measurement before any notch
fixture is designed/built.

Deliberately does NOT boot `isaacsim.SimulationApp`/`AppLauncher` (unlike
scripts/_diag_dexcube_scale_check.py and scripts/_diag_die_scale_check.py,
which both launch a headless Kit process - a real, if lightweight, sim
launch that touches the GPU). Task 0's own brief is explicit: no sim
launch, no GPU, local or cloud. Kit's `pxr` bindings are ordinary compiled
Python packages living under the `omni.usd.libs` extension's own `pxr/`
directory (Usd, UsdGeom, Sdf, Gf, Tf, ...) - importable directly by adding
that directory to `sys.path`, with no Kit/renderer boot and no GPU context
at all. Confirmed by hand before writing this script (plain `python3
-c "from pxr import Usd, UsdGeom"` after the sys.path insert below
succeeds standalone).

Measures the fingertip prim's mesh geometry in its OWN LOCAL frame (not
world), specifically to avoid the world-frame measurement's tilt/cosine
confound that a past run's diagnostic print already flagged (Gate G's
"measured hand->finger-BODY-ORIGIN z offset" was taken at the default
(tilted, non-straight-down) reset orientation - see
outputs/d4_rung0/trial_seed42.log:201-202 - so any world-frame z gap
between panda_hand and panda_leftfinger/rightfinger there conflates the
finger's real geometric pad extent with a directional-cosine term from the
tilt). Local-frame bounds sidestep that: the finger's own USD prim origin
coincides with its joint attachment frame, so the local bbox gives the
pad's real geometric extent below/around that frame regardless of how the
arm happens to be posed.

.. code-block:: bash

    LD_LIBRARY_PATH="/home/saps/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311/bin" \\
        /home/saps/IsaacLab/_isaac_sim/kit/python/bin/python3 scripts/_diag_franka_fingertip_geometry.py

LD_LIBRARY_PATH must be set in the process environment *before* launch (not
via os.environ inside the script) - the dynamic linker resolves
libusd_tf.so and friends via dlopen at import time, which is too early for
an in-script os.environ.setdefault to take effect.
"""

import os
import sys

# Only the pxr namespace package (Usd, UsdGeom, Sdf, Gf, Tf, ...) is needed;
# adding this single extension dir's root to sys.path is enough - no Kit
# boot, no renderer, no GPU touch. Path confirmed present on this machine
# before writing this script. libusd_tf.so and friends live in this same
# extension's bin/ dir - must be on LD_LIBRARY_PATH *before* the process
# starts (the dynamic linker resolves this at dlopen time; setting
# os.environ here is too late), see the invocation command below.
_OMNI_USD_LIBS_ROOT = (
    "/home/saps/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311"
)
if _OMNI_USD_LIBS_ROOT not in sys.path:
    sys.path.insert(0, _OMNI_USD_LIBS_ROOT)

from pxr import Gf, Usd, UsdGeom  # noqa: E402

# A bare (no-SimulationApp) pxr process has no https/Nucleus asset resolver
# plugin loaded (those are supplied by additional Kit extensions that
# SimulationApp would normally load) - Usd.Stage.Open() on the raw
# omniverse-content-production URL fails with "Failed to open layer"
# (confirmed by hand). Fix: download the instanceable USD *and* its
# relative-referenced payload layers (Props/panda_leftfinger.usd,
# Props/panda_rightfinger.usd, Props/instanceable_collision_meshes.usd -
# the only three this script's own prims need) into a local directory
# mirroring the same relative layout, then open the LOCAL copy - the
# default filesystem resolver (no special extension needed) handles plain
# relative-path references on disk. URL taken from a past training run's
# own saved config (logs/train_franka/2026-07-09_22-05-51/params/env.yaml:130),
# same reason scripts/_diag_dexcube_scale_check.py hardcodes its resolved
# URL rather than resolving ISAACLAB_NUCLEUS_DIR at runtime (that carb
# setting is only populated once an isaaclab/AppLauncher environment has
# loaded its experience config, which this script deliberately does not
# do).
FRANKA_USD_BASE_URL = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
    "/Assets/Isaac/5.1/Isaac/IsaacLab/Robots/FrankaEmika"
)
# Cache dir deliberately OUTSIDE the repo (this repo is public per
# CLAUDE.md's git conventions, and asset caches like this don't belong
# tracked) - overridable via env var for a different sandbox/session.
_LOCAL_CACHE_DIR = os.environ.get(
    "FRANKA_ASSET_CACHE_DIR",
    "/tmp/_diag_franka_fingertip_geometry_asset_cache",
)
_NEEDED_RELATIVE_FILES = [
    "panda_instanceable.usd",
    "Props/panda_leftfinger.usd",
    "Props/panda_rightfinger.usd",
    "Props/instanceable_collision_meshes.usd",
]


def _ensure_local_cache() -> str:
    """Download the instanceable USD + its 3 needed referenced payload
    layers into a local dir mirroring the Nucleus relative layout, if not
    already cached. Returns the local root USD path to open. Uses urllib
    (stdlib only) rather than requests/curl subprocess, to keep this
    script dependency-free."""
    import urllib.request

    for rel in _NEEDED_RELATIVE_FILES:
        local_path = os.path.join(_LOCAL_CACHE_DIR, rel)
        if os.path.isfile(local_path) and os.path.getsize(local_path) > 0:
            continue
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        url = f"{FRANKA_USD_BASE_URL}/{rel}"
        print(f"  fetching {url} -> {local_path}")
        urllib.request.urlretrieve(url, local_path)
    return os.path.join(_LOCAL_CACHE_DIR, "panda_instanceable.usd")


FINGER_PRIM_NAMES = ["panda_leftfinger", "panda_rightfinger"]


def _local_mesh_bbox(finger_prim: Usd.Prim) -> tuple[Gf.Vec3d, Gf.Vec3d, int]:
    """Bounding box of every UsdGeom.Mesh under `finger_prim`, expressed in
    `finger_prim`'s OWN local frame (i.e. relative to its own origin/joint
    attachment frame, not world or stage root) - computed by taking each
    mesh's world bound and transforming by the INVERSE of the finger prim's
    own local-to-world transform, so the result is independent of the
    articulation's current pose (default reset joint config, any tilt,
    etc.)."""
    stage = finger_prim.GetStage()
    xform_cache = UsdGeom.XformCache()
    finger_to_world = xform_cache.GetLocalToWorldTransform(finger_prim)
    world_to_finger = finger_to_world.GetInverse()

    lo = Gf.Vec3d(float("inf"), float("inf"), float("inf"))
    hi = Gf.Vec3d(float("-inf"), float("-inf"), float("-inf"))
    mesh_count = 0
    # TraverseInstanceProxies: this Franka USD is instanceable (like the
    # DexCube asset scripts/_diag_dexcube_scale_check.py already had to
    # special-case) - a plain Traverse() from finger_prim would silently
    # see zero meshes if the finger's own geometry sits inside a
    # referenced/instanced payload.
    for prim in Usd.PrimRange(finger_prim, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh_count += 1
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        if points is None:
            continue
        mesh_to_world = xform_cache.GetLocalToWorldTransform(prim)
        mesh_to_finger = mesh_to_world * world_to_finger
        for p in points:
            wp = mesh_to_finger.Transform(Gf.Vec3d(p))
            for i in range(3):
                lo[i] = min(lo[i], wp[i])
                hi[i] = max(hi[i], wp[i])
    return lo, hi, mesh_count


def main() -> None:
    local_usd_path = _ensure_local_cache()
    stage = Usd.Stage.Open(local_usd_path)
    if stage is None:
        print(f"FAILED to open {local_usd_path}")
        sys.exit(1)

    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    print(f"[FRANKA FINGERTIP GEOMETRY] stage={local_usd_path} (source: {FRANKA_USD_BASE_URL}/panda_instanceable.usd)")
    print(f"  metersPerUnit={mpu}")

    # Find the two finger prims by name, wherever they sit in the hierarchy
    # (instanceable assets nest actual geometry under a referenced payload,
    # so search instance proxies rather than assuming a fixed depth/path).
    found = {}
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        name = prim.GetName()
        if name in FINGER_PRIM_NAMES and name not in found:
            found[name] = prim

    if not found:
        print("  FAILED: no panda_leftfinger/panda_rightfinger prim found")
        sys.exit(1)

    for name in FINGER_PRIM_NAMES:
        prim = found.get(name)
        if prim is None:
            print(f"  [{name}] NOT FOUND")
            continue
        print(f"  [{name}] prim path: {prim.GetPath()}")
        lo, hi, mesh_count = _local_mesh_bbox(prim)
        dims = [hi[i] - lo[i] for i in range(3)]
        print(f"    mesh_count={mesh_count}")
        print(
            f"    local-frame bbox (stage units): min={tuple(round(v, 6) for v in lo)} "
            f"max={tuple(round(v, 6) for v in hi)}"
        )
        print(
            f"    local-frame bbox (mm, assuming metersPerUnit={mpu} -> mm scale {mpu * 1000}): "
            f"dims={tuple(round(d * mpu * 1000, 3) for d in dims)} "
            f"min_mm={tuple(round(v * mpu * 1000, 3) for v in lo)} "
            f"max_mm={tuple(round(v * mpu * 1000, 3) for v in hi)}"
        )

    print("\n[DONE]")


if __name__ == "__main__":
    main()
