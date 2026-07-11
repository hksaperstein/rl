# Fillet Edges and Full-Detail .blend Export — Design

## Context

The dice pipeline's `.blend`/`.stl` export (added earlier this session) revealed two issues when a user opened generated `.blend` files directly in Blender:

1. Dice appeared as blank grey polygons. Root cause (confirmed by loading a generated `.blend` headlessly and comparing properties): every material's `diffuse_color` — the legacy property Blender's default "Solid" viewport shading reads — was never set. Only the Principled BSDF node's Base Color was set, which Solid shading doesn't evaluate. The material/color data was always correct (confirmed identical to the thumbnail render); it just wasn't visible without switching to Material Preview/Rendered shading.
2. Edges/corners showed a flat faceted look rather than meeting cleanly. Root cause (confirmed by comparing a pristine d6 cube, 8 verts/6 faces, against the same cube after `exporter.py`'s existing Bevel modifier, 24 verts/26 faces): this is the intentional single-segment chamfer bevel already documented in `exporter.py`, not corruption. The user wants this changed from a flat chamfer to a smooth rounded fillet instead.

## Decisions

- **Fillet, not chamfer, uniformly on every die.** No per-asset randomization, no hard-edge option. `exporter.py`'s existing Bevel modifier (`limit_method='ANGLE'`, `angle_limit=35°`, width = `size_mm * bevel_fraction` sampled 0.02-0.06 in `sampler.py`, unchanged) gains `segments = 8` for a smooth rounded look.
- **`materials.py` sets `diffuse_color`** on every material it builds (`build_material` and `build_fill_material`), mirroring the same representative HSV-derived base color already computed for the Principled BSDF's Base Color, so Solid-shading viewport mode shows the correct color even without switching modes.
- **Every `.blend`'s viewport shading defaults to Material Preview.** `_save_blend_copy` (in `exporter.py`) sets `shading.type = 'MATERIAL'` on every `VIEW_3D` space across every workspace screen before saving, confirmed feasible in `--background` mode (all 10 default workspaces' `VIEW_3D` areas are present and settable even headlessly). This makes color, texture, and material immediately visible on open, with no manual shading-mode switch — covering engraved numeral recesses (visible via Material Preview's built-in studio lighting) and decal numeral textures (visible since Material Preview evaluates the full shader node graph, including image textures) alike.
- **`export_asset`'s order changes so the `.blend` is saved right after the model is finished, before any export.** New order: apply Bevel modifier (with fillet segments) → save `.blend` → export USD → export STL → render thumbnail. This makes the `.blend` the definitive, complete source state that USD/STL/thumbnail are all derived from, per explicit instruction ("all the work should be done on the blender model, before anything else is exported"). It also sidesteps a contamination risk: the thumbnail render creates its own temporary camera/light in the scene; saving `.blend` before that step means those temp objects never exist at purge/save time, so they can't leak into the `.blend` (mirroring the existing default-Cube/Light/Camera concern `_save_blend_copy` already handles).

## Non-goals

- No change to `bevel_fraction`'s sampled range (still 0.02-0.06 of `size_mm`) — only the modifier's `segments` value changes.
- No hard-edge variant, no per-asset edge-treatment randomization.
- No change to USD/STL export logic itself, or to the engrave/decal glyph pipelines — this is purely about the Bevel modifier's segment count, material viewport-display properties, and export ordering.
- Materials' actual shader node graphs (procedural noise/ramp for marbled/speckled, voronoi for glitter, etc.) are unchanged — `diffuse_color` is a supplementary flat-color property for Solid-mode display only, not a replacement for the node-based appearance.

## Files touched

- `src/dice_gen/exporter.py` — `export_asset`'s call order; Bevel modifier gains `segments = 8`; `_save_blend_copy` gains the viewport-shading-to-Material-Preview step.
- `src/dice_gen/materials.py` — `build_material` and `build_fill_material` each set `mat.diffuse_color`.
