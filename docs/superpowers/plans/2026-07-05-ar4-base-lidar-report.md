# AR4 base-mounted LiDAR: exploratory addition and empirical check

Junior report. Task: add an experimental LiDAR sensor mounted at the AR4
robot's base (Isaac Lab's `RayCasterCfg` + `LidarPatternCfg`), alongside (not
replacing) the existing top-down RGB-D `perception_camera`, and honestly
assess whether it resolves anything useful about the tabletop objects at this
scale — a direct empirical test of the literature conclusion in
`docs/superpowers/specs/research/2026-07-05-perception-sensing-literature-junior.md`
/ `-senior-review.md` (RGB-D over LiDAR for objects this small at this range,
due to LiDAR's coarser angular resolution).

Does not touch `perception/shape_classifier.py` or its threshold (separate
work by another subagent this session).

## What was added

`tasks/ar4/pickplace_env_cfg.py`:

- New import: `RayCasterCfg, patterns` from `isaaclab.sensors`.
- New `Ar4PickPlaceLidarSceneCfg(Ar4PickPlaceSceneCfg)` and
  `Ar4PickPlaceLidarEnvCfg(Ar4PickPlaceEnvCfg)` (num_envs=1), added as a
  sibling of the existing `Ar4PickPlacePerceptionSceneCfg` /
  `Ar4PickPlacePerceptionEnvCfg` pair rather than modifying the training scene
  or the perception scene directly — same "opt-in sensor subclass" convention
  those already establish, so this stays fully out of `train.py`'s path.
- `base_lidar: RayCasterCfg`, mounted at
  `"{ENV_REGEX_NS}/Robot/root_joint/base_link"` (the same parent prim the
  `ee_frame` FrameTransformer already uses), with:
  - `offset.pos = (0.0, 0.0, 0.05)` — 5cm above the base mount point.
  - `pattern_cfg = LidarPatternCfg(channels=16, vertical_fov_range=(-20, 0),
    horizontal_fov_range=(-140, -40), horizontal_res=0.5)`
  - `ray_alignment="base"`, `max_distance=5.0`, `debug_vis=True`,
    `mesh_prim_paths=["/World/ground"]`.

FOV reasoning (documented in-code as a comment block above the config): the
robot base is yawed 180deg about world Z, so a world-frame angle `theta`
(measured from world +X) sits at local angle `theta - 180` in the base's own
frame. The four objects span world angle ~54.5deg (cube, near corner) to
~120.5deg (wedge, far corner) as seen from the base origin, i.e. local
-125.5deg to -59.5deg — `horizontal_fov_range=(-140, -40)` covers that with
margin. Vertically, a sensor 0.05m above a workspace at z~0 needs a
shallow-downward look (not straight down, which is what `perception_camera`
already does): depression angle to hit the ground is `atan(h/r)`, ~14deg at
the near edge of the workspace (r=0.20m) and ~6deg at the far edge (r=0.45m),
so `vertical_fov_range=(-20, 0)` brackets that. 16 channels / 0.5deg azimuth
resolution mirrors a real 16-channel rotating LiDAR (Velodyne Puck-class:
coarse vertical, fine azimuth), rather than an artificially fine pattern, so
the experiment reflects realistic base-LiDAR resolution.

**Load-bearing caveat discovered while implementing, not assumed up front:**
Isaac Lab's `RayCaster` (`ray_caster.py`, `_initialize_warp_meshes`) currently
ray-casts against **exactly one static mesh** —
`if len(self.cfg.mesh_prim_paths) != 1: raise ValueError(...)` — and its own
docstring says "Currently, only a single static mesh is supported... dynamic
meshes [are not]." The cube/sphere/rect_prism/wedge are all dynamic
`RigidObjectCfg` prims, not part of any static mesh, and cannot be added to
`mesh_prim_paths` even in principle with this Isaac Lab version. So
`mesh_prim_paths=["/World/ground"]` is not just my choice of the simpler
option — it's the *only* option; the objects are structurally invisible to
this sensor type as currently implemented, independent of pattern resolution.

## Script: `scripts/lidar_calibration.py`

Modeled on `scripts/perception_calibration.py`'s pattern (AppLauncher, env
construction, a few motionless-robot steps, output to `logs/videos/`). Builds
`Ar4PickPlaceLidarEnvCfg`, steps 10 frames, reads `lidar.data.ray_hits_w`
(confirmed field name from `RayCasterData` — shape `(N, B, 3)`, world-frame
hit points, `inf` for non-hit rays), and produces:

- `logs/videos/lidar_calibration.png` — top-down XY scatter of hit points
  (colored by hit z), with the sensor origin and each object's true (x, y)
  overlaid as markers plus a 30mm reference circle.
- `logs/videos/lidar_calibration_stats.txt` — hit-count/z-range/per-object
  near-hit stats. (Added this file because the Kit app's stdout capture is
  unreliable right at shutdown — `perception_calibration.py`'s own final
  `print()` is silently dropped from the captured terminal output in this
  environment too, even though its file output does get written. Mirroring
  stats to a file sidesteps that rather than trusting a possibly-truncated
  terminal log.)

Run: `/home/saps/IsaacLab/isaaclab.sh -p scripts/lidar_calibration.py --headless`

## Results

```
Total rays cast: 3200
Total finite hits: 3000 (93.8% of rays)
Hit z range: [-0.0314, 0.0309] m (mean -0.0000 m)

Per-object near-hit check (hits within 30mm of true (x, y)):
  cube         true=(+0.200, +0.280)  hits_within_30mm=26, z=[-0.0302, 0.0124] (object top expected ~0.0260m)
  rect_prism   true=(+0.200, +0.340)  hits_within_30mm=5,  z=[-0.0056, 0.0033] (object top expected ~0.0220m)
  sphere       true=(-0.188, +0.296)  hits_within_30mm=15, z=[-0.0080, 0.0117] (object top expected ~0.0180m)
  wedge        true=(-0.200, +0.340)  hits_within_30mm=11, z=[-0.0128, 0.0047] (object top expected ~0.0300m)
```

The scatter plot (`logs/videos/lidar_calibration.png`) shows a set of nested
arcs sweeping from ~40deg to ~140deg in world azimuth (matching the FOV
derivation above) out to a max radius of ~2.3m, plus a denser cluster of
points in the 0-0.5m radius band where the object markers sit. The 200
missing rays out of 3200 (6.3%) are exactly the one channel at the
`vertical_fov_range` boundary closest to 0deg — a beam parallel to the ground
that geometrically never converges, correctly returned as non-finite.

## Verdict: confirms the literature conclusion, and more strongly than expected

Two independent problems, not one:

1. **Architectural: the objects are invisible to this sensor, period.**
   Because `RayCaster` only traces the static ground mesh, hits "near" an
   object's true (x, y) are pure ground-plane geometry samples — nothing in
   the ray-casting pipeline knows the object is there. This is directly
   visible in the z-values: if these hits were real object-surface
   detections, z should cluster tightly near each object's known top height
   (0.018-0.030m). Instead every object's near-hit z range spans through
   values well below its expected top (down to -0.030m for the cube, whose
   top is at +0.026m) — i.e., these are ground noise, not object returns. A
   14-30% of that 3cm ground noise is real: the sensor origin's reported
   `pos_w` is the raw `base_link` pose (~(0,0,0) as expected — the robot root
   sits at the world origin per `robot_cfg.py`), and I traced the ±3cm z
   spread to the ray-cast geometry itself at the achieved radii (up to 2.3m
   from grazing-incidence beams near the 0deg channel), not to object
   presence — either way it carries zero object-position information.

2. **Even if RayCaster supported dynamic meshes, angular resolution at this
   range is still coarse relative to 9-30mm objects**, consistent with the
   literature's core claim. The channels landing within the workspace's
   0.20-0.45m radius band (roughly -6 to -14deg of the 16 evenly-spaced
   channels) number about 6 of 16, each spaced ~1.3deg apart — at a 0.3m
   range that's ~7mm of vertical footprint per channel step, i.e. comparable
   to or larger than the sphere's 18mm diameter. Horizontally, 0.5deg
   resolution at 0.3m range is ~2.6mm between adjacent azimuth samples,
   which is finer and could resolve azimuthal extent reasonably well *if*
   the objects were traceable at all — but per point 1, they aren't.

**Bottom line:** this experiment empirically confirms the RGB-D-over-LiDAR
literature conclusion, and for a more fundamental reason than "coarse
resolution" alone — in this Isaac Lab version, `RayCaster` cannot ray-cast
against dynamic rigid-body objects at all (single static mesh only, by
explicit code-level restriction, not a config oversight), so a base-mounted
LiDAR of this kind provides *zero* information about the cube/sphere/
rect_prism/wedge regardless of channel count or azimuth resolution. The
angular-resolution argument from the literature review is a secondary,
independently valid concern that would still apply even if Isaac Lab added
dynamic-mesh support later. This is a clean negative result, not a partial
success — recommend not pursuing this sensor further for shape/pose sensing
on this task; `perception_camera` (RGB-D) remains the right modality here.

## debug_vis

`RayCasterCfg.debug_vis=True` was tried and runs cleanly headless (produces a
benign `[Warning] FabricManager::initializePointInstancer mismatched
prototypes on point instancer: /Visuals/RayCaster` line, no crash, no
Traceback — confirmed by rerunning with `debug_vis=True` twice and observing
fresh output files both times). It would be a useful additional check in an
interactive GUI session (its `RAY_CASTER_MARKER_CFG` draws the actual hit
points as markers in the viewport, letting you eyeball the fan pattern
directly against the robot/objects in 3D) — but no GUI/display-capturing
setup was available in this run, so it wasn't used to produce the delivered
artifact; the headless run's saved PNG is what backs this report. Left
`debug_vis=True` in the delivered config since it's free-ish and harmless
headless, and useful if someone opens this scene in the GUI later.

## Files changed

- `tasks/ar4/pickplace_env_cfg.py` — added `Ar4PickPlaceLidarSceneCfg` /
  `Ar4PickPlaceLidarEnvCfg` and the `base_lidar` RayCaster config (new code
  only; no existing classes modified).
- `scripts/lidar_calibration.py` — new script.
- `logs/videos/lidar_calibration.png`, `logs/videos/lidar_calibration_stats.txt`
  — generated artifacts.

Nothing committed; left for review as instructed.
