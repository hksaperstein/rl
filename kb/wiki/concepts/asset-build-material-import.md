# Asset build: the URDF importer discards per-visual colors

## Why this theme exists

Every rendered view of the AR4 arm in this repo (training videos, close-up
demo cameras, the skeleton plotter) showed the robot as a flat,
undifferentiated white/light-gray silhouette, even though the source URDF
(`annin_ar4_description`'s `ar_macro.xacro` / `ar_gripper_macro.xacro`)
authors real per-part `<material><color rgba>` values — a light aluminum
`0.863`, a gray enclosure `0.627`, near-black motors `0.078`, and blue
covers `0 0.35 1`. This is a correctness/legibility issue in the one
shared robot asset every task cfg depends on (`AR4_MK5_CFG` in
`tasks/ar4/robot_cfg.py`), fixed 2026-07-09.

## Root cause (introspected, not assumed)

The installed URDF importer is
`isaacsim.asset.importer.urdf._urdf.ImportConfig`. Dumping its live
attribute list (`dir()` inside a running Isaac Sim launch — not from
recalled docs, per this repo's "introspect the live module" discipline)
shows **no material/color import flag at all**: the only visual-related
attributes are `collision_from_visuals` / `set_collision_from_visuals`.
There is no `import_materials`, `convert_visuals`, `import_visuals`, etc.

Traversing the built USD (instance-proxy-aware) confirmed every visual
mesh *is* bound to a material — but the importer authors a white
`DefaultMaterial` whose MDL shader has `diffuse_color_constant = (1,1,1)`
for essentially every mesh, discarding the URDF's authored colors. (It
does deduplicate two colors into shared `/mk5/Looks/material_A0A0A0`
[=0.627] and `material_131313` [=0.078] materials, but leaves most arm
meshes on white per-mesh `DefaultMaterial`s.) So the colors are lost **at
import time** — not stripped by a downstream Isaac Lab spawn override, and
not merely hidden from a naive traversal.

## The fix: post-import color authoring in `build_asset.py`

`scripts/build_asset.py` gained a post-import pass (runs right after
`URDFImportRobot`, before the manifest is written):

- `_build_urdf_color_map()` parses the just-generated URDF for each
  `<visual>`'s `<geometry><mesh filename>` basename → `<material><color>`
  (resolving URDF's define-once/reference-by-name materials globally).
- `_apply_visual_colors()` opens the **base configuration layer**
  (`configuration/ar4_mk5_<...>_base.usd` — the layer that actually
  *defines* the geometry+materials, so editing it composes into the
  referenced asset and sidesteps instanceable-proxy edit restrictions),
  and for every visual mesh sets both the bound shader's
  `diffuse_color_constant` (what RTX uses) and the mesh's `displayColor`
  primvar (a Hydra-Storm/preview fallback).

### The layer-namespace gotcha that bit the first attempt

In the composed asset the visual prim path is
`/mk5/root_joint/<link>/visuals/<STLName>/node_STL_BINARY_/...`, so the
mesh name looks like the component right after `visuals`. But inside the
`_base.usd` layer's own namespace the geometry lives under a separate
`/visuals/<link>/<STLName>/node_STL_BINARY_/...` prototype scope — there
the component after `visuals` is the **link** name, not the STL name. The
robust key is the component immediately **before** the `node_STL_*`
marker. (The first attempt keyed off "after visuals" and silently matched
only the three gripper meshes, whose link and STL basenames happen to
coincide.)

## Verification that it stayed cosmetic-only

Rebuilding this shared asset has silently changed unrelated things before
(Experiment 19's mimic-joint saga — see
[[experiment-26-gripper-reintroduction]] lineage / ROADMAP item on the
gripper). This change was confirmed purely cosmetic:

- `configuration/*_physics.usd` and `*_robot.usd` are **byte-for-byte
  identical** to the pre-fix asset (`cmp`); only `*_base.usd` (geometry +
  materials) changed.
- Collision geometry still present on all expected links (7 collision
  meshes: link_1–4 + the 3 gripper links), introspected instance-proxy-
  aware.
- `_EE_OFFSET=(0,0,0.036)` re-verified numerically
  (`scripts/_check_ee_vs_gripper_fk.py`): gripper-FK midpoint matches the
  `ee_frame` target to 0.000000 m; link_6→ee_frame = 0.036000 m exactly.
- A close-up render (`scripts/render_color_check.py`, no policy needed)
  shows the real scheme: silver aluminum body, near-black motor housings,
  blue covers — not a white silhouette.

Reusable tooling left in the repo: `scripts/verify_asset_colors.py`
(color + collision audit of the composed asset) and
`scripts/render_color_check.py` (still-frame render). Re-run these after
any future `build_asset.py` rebuild.

## Generalization note

This is arm-agnostic: any URDF imported through this Isaac Sim version's
URDF importer loses its `<material><color>` values the same way, so the
post-import color pass is the general fix a future arm swap should keep,
not an AR4-specific patch.
