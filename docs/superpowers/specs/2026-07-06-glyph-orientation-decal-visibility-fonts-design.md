# Glyph Orientation, Decal Visibility, and Font Wiring — Design

## Context

After the fillet/full-detail-`.blend` work, closer visual inspection of generated dice (thumbnails, and a manual render probe built during this investigation) surfaced four issues, each root-caused with direct empirical evidence before any fix was designed:

1. **Painted/decal numerals often invisible.** `apply_decal_glyphs` (`src/dice_gen/glyphs.py:628`) calls `bpy.ops.uv.smart_project` once across the whole die, which packs every face's UV island into one shared `0-1` atlas. Measured on a d8: each face's island only covers roughly a `0.27 x 0.31` patch (e.g. face 0 at `u=[0.013,0.279]`, `v=[0.013,0.320]`), never the full square. But each face has its own **dedicated** texture (the composited swatch+glyph PNG, glyph centered at `(0.5, 0.5)`), so most faces end up sampling only a background-colored corner of their own image, missing the centered glyph.
2. **Inconsistent numeral rotation between faces.** `_face_orientation_matrix` (`src/dice_gen/glyphs.py:60`) picks an "up" hint per face (global Z, or global Y if `abs(normal.z) >= 0.9`). Dumped every face's computed orientation on a d20: the fallback threshold lands exactly on the near-pole ring (`normal.z ≈ ±0.934`), creating an abrupt jump between that ring and its neighbor — reproduced visually as one face reading "20" upright while the adjacent face's numeral appeared at a distinctly different angle.
3. **`font_or_style_id` is sampled but never applied.** `sampler.py` already samples one of `FONT_IDS = ["font_sans_bold", "font_serif_regular", "font_display_condensed"]` per die and stores it in the manifest, but neither `apply_engraved_glyphs` nor `_render_label_to_image` (`glyphs.py:398`, `:697`) ever reads it — every die uses Blender's single bundled default font regardless.
4. **Engravings are a bit deep.** `ENGRAVE_DEPTH_FRACTION = 0.04` (`glyphs.py:14`) — a user-facing preference for a shallower cut, not a bug.

## Decisions

### 1. Decal UV: manual per-face unwrap, not `smart_project`

Replace the single `bpy.ops.uv.smart_project(island_margin=0.05)` call in `apply_decal_glyphs` with a manual per-face projection: for each polygon, project its vertices into a local 2D frame (tangent, bitangent — see below) relative to the face center, then scale so the larger of the two axis spans fits `1.0 - 2*margin` (margin `0.1`), centered at `(0.5, 0.5)`. This guarantees every face's UV island independently fills its own full `0-1` square, matching what its dedicated texture assumes. Verified empirically: rebuilt a d8 with this approach end-to-end (material, per-face decal texture, composite, render) and the numeral appeared correctly, centered, and legible.

### 2. Shared, corrected tangent/bitangent basis

Extract the "given a face normal, compute (tangent, bitangent)" logic out of `_face_orientation_matrix` into a standalone helper, used by both:
- `_face_orientation_matrix` (engrave cutter placement, world-space normal), and
- the new decal UV unwrap (local-space normal, no world matrix needed since UVs are computed directly from the mesh's own local vertex/polygon data).

The helper also fixes the discontinuity: the up-hint fallback threshold moves from `abs(normal.z) < 0.9` to `abs(normal.z) < 0.999`, so the Y-fallback only fires for the genuinely-degenerate case (a face normal exactly axis-aligned with global Z, e.g. d6/d8's top/bottom faces, where the Z-based cross product would be undefined) rather than merely-steep faces. Verified empirically across all 20 of a d20's faces: the old threshold produced a hard jump at the `normal.z ≈ ±0.934` ring; the new one varies smoothly with no discontinuity anywhere, while d6/d8's exactly-axis-aligned faces still correctly trigger the fallback.

Using one shared, corrected helper for both paths means engraved and decal numerals on the same die follow one consistent orientation convention, instead of engrave using one (buggy) heuristic and decal using none at all.

### 3. Font wiring

Map the existing `FONT_IDS` to real installed font files, loaded via `bpy.data.fonts.load(path)` and assigned to `txt_obj.data.font` in both `apply_engraved_glyphs` and `_render_label_to_image`:

| `font_or_style_id` | Font file |
|---|---|
| `font_sans_bold` | `/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf` |
| `font_serif_regular` | `/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf` |
| `font_display_condensed` | `/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Bold.ttf` |

**Exception, confirmed by direct render test:** when `glyph_style == "cjk_numerals"`, do NOT apply this mapping — always leave `txt_obj.data.font` at Blender's default. Rendered "七" and "五" with Blender's current default font: both show the correct character. Rendered the same characters with Liberation Sans Bold: shows an empty rectangle (missing-glyph placeholder) — Liberation fonts have no CJK coverage. Arabic numerals, Roman numerals, and Greek numerals (tested with "Δ") all render correctly with the Liberation fonts, so the exception is scoped to `cjk_numerals` only.

Each font file should be loaded once and reused (check `bpy.data.fonts` for an already-loaded font with the matching filepath before calling `bpy.data.fonts.load` again) rather than reloading per die within a batch.

### 4. Shallower engraving

Reduce `ENGRAVE_DEPTH_FRACTION` from `0.04` to `0.025` (size_mm scaling and everywhere it's used stay the same — this is a single constant change).

### 5. Next regeneration uses a new seed

Not a code change: when the batch at `data/raw/dice_assets/` is regenerated to pick up these fixes, use a different `--seed` value (e.g. `7`) instead of reusing `42`, per your request for variety. Still fully deterministic and reproducible from whatever seed is chosen.

## Non-goals

- The earlier resting-pose request (face 1 flush on the ground plane) is explicitly retracted — the die's object origin stays at its geometric center, unchanged.
- No change to `smart_project`'s use anywhere else (it's only used in `apply_decal_glyphs`).
- No attempt to make numeral orientation "look upright from any arbitrary viewing angle" — a 20-sided (or 12-, 10-, 8-sided) die has no single natural "up" when viewed from an arbitrary angle. The achievable, well-defined goal is **consistency**: no abrupt jumps between neighboring faces, and engrave/decal sharing one convention. This is what the threshold fix delivers, verified face-by-face.
- No CJK-capable font is being added to the font roster; `cjk_numerals` continues to rely on Blender's bundled default font, which already covers it correctly.
- No change to `_composite_alpha_over`, `_render_material_swatch`, or the USD/STL/`.blend` export pipeline from the prior plan.

## Files touched

- `src/dice_gen/glyphs.py` — new shared tangent/bitangent helper; `_face_orientation_matrix` uses it; `apply_decal_glyphs` replaces `smart_project` with the manual per-face unwrap; `apply_engraved_glyphs` and `_render_label_to_image` apply the `font_or_style_id` → font-file mapping (with the `cjk_numerals` exception); `ENGRAVE_DEPTH_FRACTION` changes to `0.025`.
- No changes to `sampler.py` (already samples `font_or_style_id`), `orchestrator.py`, `exporter.py`, or `materials.py`.
