# Percentile d10 (`d10_pct`) — Design

## Context

The pipeline currently generates 6 die types (`sampler.DIE_TYPES = ["d4", "d6", "d8", "d10", "d12", "d20"]`), and `orchestrator.generate_set_batch` already produces matched sets (one of each type, shared material/font/glyph_method per `set_id`) by looping that list. A real D&D dice set has a 7th die: the percentile die (`d%`), paired with the units d10 to read a 2-digit percentile roll. Goal: add `d10_pct` as a 7th entry in `DIE_TYPES` so every future set is a true 7-piece set, as a step toward generating a training dataset of thousands of dice.

Real-world facts below were obtained via delegated web research, then independently reviewed by a second agent (per `delegating-technical-research`). One claim from the first pass — a "Gamescience 1990 origin" story — could not be verified against any real source (Wikipedia has no such content, no organic search hits) and was dropped as likely fabricated. The surviving facts:

- Face values are **00, 10, 20, ..., 90** — the one universally-confirmed convention (Wikipedia's dice article; no credible alternative labeling found).
- The die uses the **same physical mesh** as a units d10 (pentagonal trapezohedron) — a pure face-label difference, not a different shape.
- "Opposite faces sum to 90" (00↔90, 10↔80, ...) is the commonly-cited convention, mirroring the units d10's sum-to-9 rule, but is **not verified as a universal manufacturing law** — real d10 face layouts are documented as inconsistent across molds/manufacturers in general.

This project has previously shipped an invented-but-wrong numbering scheme (the `GREEK_NUMERALS` table, flagged in `docs/ROADMAP.md`) by not researching real conventions first. This design deliberately does not repeat that: where a real convention is verified (face values, geometry), we match it; where it isn't (opposite-sum as physical law), we adopt it as our own generation-time design choice rather than asserting it as fact, and don't invent conventions with no real basis at all (see numeral-rendering decision below).

## Decisions

### 1. Geometry: reuse d10's mesh unchanged

Add `"d10_pct"` to `DIE_SPECS` (`geometry.py`) with the same `base_vertices: _d10_base_vertices()`, `expected_faces: 10`, `expected_verts: 12`, `expected_edges: 20` as `"d10"`. Add `"d10_pct"` alongside `"d8"`/`"d10"` in the two-pole check at `geometry.py:185` (`compute_face_poles`), since it has the identical pole structure. No new geometry-building code.

### 2. Numbering: scale the existing d10 assignment, don't parallel it

`numbering.py`'s hemisphere-consistency assignment (`assign_values_to_opposite_pairs`) depends on every scheme's `opposite_sum` being **odd**, so each antipodal pair has exactly one odd and one even value, routed to top/bottom pole by parity (`numbering.py:43-46,77`). `d10_pct`'s values (0,10,...,90) are **all even** — a naive `{"opposite_sum": 90}` entry fed through the generic machinery would break the odd/even split entirely (every pair would hit the same branch).

Fix: a percentile die is physically the same mold as a units d10 with different digits printed on it. So `d10_pct`'s face assignment reuses the **existing d10 scheme** (`values=range(0,10)`, `opposite_sum=9`, full hemisphere logic unchanged) to compute the assignment, then scales every assigned value ×10 for display/manifest/engraving purposes. This is a value-scale transform on top of the proven algorithm, not a new parallel one. `verify_opposite_sum` for `d10_pct` checks the *scaled* values sum to 90 — mathematically guaranteed to hold, since it's the same combinatorial solution scaled uniformly. Enforced as a hard invariant (`raise ValueError` on failure) exactly like every other die type — no new soft/manifest-flag machinery, since satisfiability isn't actually in question here.

### 3. Glyph rendering: arabic-only, zero-padded

`d10_pct` is restricted to `glyph_style = "arabic_numerals"`, always — even within a matched set where `sample_set` picks one shared non-arabic `glyph_style` (roman/greek/CJK) for the other 6 dice. Rationale: real percentile dice are only ever sold with arabic digits (verified above); no real roman/greek/CJK percentile-die convention exists to draw from, and inventing one would repeat the exact mistake `GREEK_NUMERALS` already made. The percentile die's label overrides the set's sampled style for its own face labels; the rest of the set (material, font, glyph_method, glyph_fill, bevel) still matches normally.

Zero-face label: real percentile dice show "**00**", not "0". In `glyphs.py`'s arabic path, `d10_pct` values are formatted `f"{value:02d}"` — zero-pads the 0 face to "00" and leaves 10-90 unchanged ("10" already 2 digits).

### 4. Sampler wiring

Add `"d10_pct"` to `DIE_TYPES` (after `"d10"`) and to `SIZE_RANGES_MM` (reuse the existing `"d10"` range, `14.0-20.0mm` — same physical die size class). In `sample_variant` and `sample_set`, force `glyph_style = "arabic_numerals"` whenever `die_type == "d10_pct"`, overriding the otherwise-shared/sampled style per decision 3.

### 5. Verification

- Implement via subagent-driven-development (junior implementer + senior reviewer per task).
- Generate a real batch (`--sets 5` → 35 assets) and run `scripts/validate_dice_assets.py`.
- Fresh-reload spot check (per this project's standing practice — verify exported files fresh, not in-memory state) specifically on the `d10_pct` assets: labels read 00-90 correctly with no collapse/corruption, opposite-face sum-to-90 actually holds post-scaling, and 2-character labels don't break proportional font sizing (d20/roman-numeral dice already produce 2-char labels today, so this should already work, but verify rather than assume).

Once verified, running `scripts/generate_dice_assets.py --sets N` for a larger batch (toward the "thousands of dice" dataset target) is a follow-up execution step with its own batch-size/outdir decision — not part of this design.

## Out of scope

- Fixing the pre-existing, separately-tracked `GREEK_NUMERALS` correctness issue (roadmap item 5) — untouched by this work.
- Any non-dice asset-generation work ("learn how to use Claude to create assets in Blender beyond dice") — a separate future session/spec per the project's own guidance to decompose large goals into their own brainstorm cycles.
