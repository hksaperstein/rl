# D4 Vertex-Read Numbering — Design

## Context

Researched how real, commercially-manufactured TTRPG dice number their faces, focusing on numeral orientation (per user request: "go research dice and look at indice orientation"). Finding, cross-checked by a second research pass: for a **standard tetrahedron d4** (the shape our `geometry.py` builds — `base_vertices: [(1,1,1),(1,-1,-1),(-1,1,-1),(-1,-1,1)]`), the dominant real-world convention is **vertex-read numbering**: each face shows the SAME digit three times, once near each of its three corners, oriented so that whichever corner is pointing up (die resting on the opposite face) reads correctly. This displaced an older "bottom-read" edge-numbered style in modern production. Single-centered-number d4s exist in the market, but are tied to a *different* physical shape (truncated/Archimedean or "single-number pyramid" d4 variants) — not the standard tetrahedron this pipeline generates. Since our d4 mesh IS a standard tetrahedron, vertex-read is the more faithful convention, and our current single-centered-numeral placement is the less accurate choice for this shape.

No comparable documented rotation-direction convention exists for d6/d8/d10/d12/d20 beyond the already-implemented opposite-faces-sum-to-N rule — those dice are unaffected by this change.

## Decision

Change d4's **numeral styles only** (`arabic_numerals`, `roman_numerals`, `greek_numerals`, `cjk_numerals` — NOT `pips`, which has no researched real-world vertex convention and stays as-is) to place the same value 3 times per face, once near each corner, radially oriented (each copy's "up" direction points from the face center toward that corner). Verified empirically before committing to this: built a real d4, computed per-corner placement via a `(tangent, bitangent, normal)` frame where `bitangent` is the corner-ward radial direction (projected into the face plane) instead of the existing global-up-hint convention, and confirmed via bmesh volume/face-count deltas that all 12 cuts (4 faces x 3 corners) land correctly and produce real, distinct geometry (a "1" cutter removes ~0.94 volume units per copy consistently; "2"/"3"/"4" remove more, consistently, matching their different ink areas) — not a silent no-op. Rendered with painted fill and confirmed all three corner copies of a value are visible, at 120°-apart rotations, exactly the pattern real vertex-read d4s show.

## Engraved path

New helper `_face_vertex_orientations(face, obj_matrix, inset=0.55)` returns one orientation matrix per vertex of `face`: for each vertex, `bitangent` is the (face-plane-projected, normalized) direction from `face.center` to that vertex — replacing the global up-hint convention only for this per-corner case — and the cut position is inset 55% of the way from center to vertex (matches the empirically-tested value; keeps the numeral clear of both the face center and the beveled edge). `apply_engraved_glyphs`'s Phase-1 planning loop calls this (instead of the single `_face_orientation_matrix` call) when `die_type == "d4" and glyph_style != "pips"`, producing 3 planned cuts per face instead of 1, all using the same `value`/label. Font size for this case is reduced (tested `size_mm * 0.13`, vs. the existing `size_mm * 0.18` for single-centered numerals) so three copies fit near the three corners of a d4's small triangular faces without overlapping.

Must preserve the existing Phase-1/Phase-2 split (compute ALL orientations against the pristine mesh before any cut is applied) — this was directly hit and fixed during verification: an early draft recomputed `face.vertices`/`face.normal` mid-loop after earlier cuts had already rebuilt the mesh topology, causing a hard Blender crash (segfault), exactly the class of bug `apply_engraved_glyphs`'s own docstring already warns about.

## Decal path

`_render_label_to_image` gains a `die_type` parameter (threaded through from `apply_decal_glyphs`'s existing call site). When `die_type == "d4" and glyph_style != "pips"`, render 3 copies of the label instead of 1, at fixed canonical equilateral-triangle corner offsets in the flat 2D render scene (e.g. corners at angle 90°/210°/330° from center at some radius within the camera's `ortho_scale=1.4` frame), each rotated so its own "up" points radially outward from the image center toward that corner. Since every d4 face is a congruent equilateral triangle and `_unwrap_faces_to_full_square` centers/scales every face's projection consistently regardless of which specific 3D vertex maps to which image position, a fixed canonical corner layout (not tied to real per-vertex 3D coordinates) is sufficient — all three copies show the identical value, so exact vertex correspondence doesn't matter, only that each of the image's 3 corners gets one correctly-outward-rotated copy.

## Non-goals

- No change to pips layout/placement (any die type) — not part of the researched convention.
- No change to d6/d8/d10/d12/d20 numeral placement or the opposite-sum numbering rule.
- No change to the shared `_tangent_bitangent`/`_face_orientation_matrix` engrave-orientation convention used by every other die type — this is a d4-specific addition, not a replacement.
- Not attempting pixel-perfect alignment between the decal image's fixed canonical triangle and each specific face's real 3D vertex positions (unnecessary, since all 3 copies are identical).

## Files touched

- `src/dice_gen/glyphs.py` — new `_face_vertex_orientations` helper; `apply_engraved_glyphs`'s Phase 1 planning branches for `d4` numeral styles; `_render_label_to_image` gains `die_type` and a d4-specific 3-corner render branch; `apply_decal_glyphs`'s call site passes `die_type` through.
- No changes to `geometry.py`, `numbering.py`, `sampler.py`, `orchestrator.py`, `materials.py`, `exporter.py`.
