# Blender Dice Asset Generator — Design

## Context

This project (Dice-Detection) aims to train an RGB-D object detector for the six standard
TTRPG dice (d4, d6, d8, d10, d12, d20). Training data will ultimately come from a
physics-based synthetic capture pipeline in Isaac Lab (scene composition, dice tumbling,
sensor simulation, ground-truth image/annotation export) — but that is a separate,
downstream subsystem and out of scope for this spec.

This spec covers only the **first subsystem**: a Blender-based generator that produces a
library of individual dice 3D assets (geometry + numerals + materials), exported in a
format Isaac Lab can later import. Blender decides what a die *looks like*; Isaac Lab (future
work) decides what a *scene* looks like.

## Goals

- Procedurally generate varied, realistic dice assets for all 6 die types, with randomized
  size, edge bevel, material/finish, and numeral/pip style — enough visual diversity that a
  detector trained on renders of these assets generalizes to real dice.
- Export each asset in a format + with metadata that a downstream scene-composition tool
  (Isaac Lab) can consume without needing to inspect the Blender file.
- Keep the pipeline scriptable/headless and reproducible (seeded).

## Non-goals

- Scene composition, physics simulation, camera/sensor simulation, lighting/background
  randomization, or annotation (bounding box/segmentation) export — all Isaac Lab's job in a
  future spec.
- Reading/labeling the rolled face value — only shape/type classification matters downstream,
  though standard face-numbering is still enforced (see below) since it costs little extra and
  keeps assets realistic / future-proofs for a possible later rolled-value task.

## Architecture

A Python package invoked headlessly via Blender's background mode
(`blender --background --python ...`), targeting Blender 5.x's `bpy` API and built-in USD
exporter (confirmed available: Blender 5.1.2 installed, headless `bpy` verified working).

```
src/dice_gen/
  geometry.py      # parametric mesh builders for each of the 6 polyhedra
  numbering.py      # standard face-numbering schemes per die type
  glyphs.py         # numeral/pip/Roman-numeral/script glyph generation
  materials.py      # procedural shader graphs per material category
  sampler.py        # draws one random parameter set for a variant
  exporter.py        # USD export + per-asset JSON manifest + thumbnail render
  orchestrator.py    # drives sampling -> build -> validate -> export for N variants,
                      # aggregates the master manifest, handles per-asset failures

scripts/
  generate_dice_assets.py   # CLI entry point (parses --count/--seed/--outdir, invokes orchestrator)
  validate_dice_assets.py    # standalone validation pass over an existing output directory
```

Each module has one clear responsibility and can be reasoned about / tested independently
(e.g. `geometry.py` can be checked for correct face/vertex counts without touching materials
or export).

### Die geometry (`geometry.py`)

All 6 shapes (tetrahedron/d4, cube/d6, octahedron/d8, pentagonal trapezohedron/d10,
dodecahedron/d12, icosahedron/d20) are built parametrically via Blender's Python API
(bmesh), with `size_mm` and `bevel` (edge rounding amount) as continuous parameters.

### Face numbering (`numbering.py`)

Each die type has a fixed, correct real-world numbering convention baked in (not randomized):
- d6: opposite faces sum to 7
- d20: opposite faces sum to 21
- d10: numbered 0-9 (the base single d10; percentile "tens" d10 variants are out of scope —
  the six requested types are single dice, not paired sets)
- d4: numbers per face, using either vertex-labeled or edge-labeled convention (both are real
  manufacturing conventions — this placement style **is** randomized per-asset)
- d8, d12: standard sequential/opposite-sum conventions

### Numerals & pips (`glyphs.py`)

Supports, randomized per asset:
- **Glyph style**: Arabic numerals, Roman numerals, pip/dot arrangements (primarily for d6,
  optionally d4), other scripts (e.g. Greek, Chinese numerals) — pulling stylistic variety
  inspiration from real-world TTRPG dice sets
- **Glyph method**: engraved geometry (boolean cut/emboss into the mesh — real depth detail,
  matters for RGB-D) vs. printed texture/decal (UV-mapped image), both supported and randomized
- **Glyph fill**: for engraved style, painted-fill vs. same-color blank engraving
- Font/script asset selection is a categorical parameter drawn from a small local library

### Materials (`materials.py`)

Procedural shader-node graphs for: opaque solid, translucent/gemstone, marbled/swirled,
glitter/sparkle, metallic, speckled. Each category has its own randomized sub-parameters
(hue/saturation, roughness, IOR, noise/voronoi scale for marbling, sparkle density, etc.).
Fine engraved detail is baked to a normal map against a lighter-weight mesh where useful, so
downstream rendering (Isaac Lab, at scene-generation scale) stays cheap.

### Sampling (`sampler.py`)

Randomized sampling with an explicit seed, not a fixed combinatorial grid: continuous
parameters (size, bevel, hue, roughness, ...) are drawn from ranges centered on realistic
values; categorical parameters (die type, material category, glyph style/method/fill, d4
placement convention) are drawn from weighted lists. A run of N variants with seed S is
fully reproducible, and extending a batch (generating more later) is just running again
with a new seed.

### Export (`exporter.py`)

Per asset:
- `<asset_id>.usd` — the die mesh + material + UV, exported via Blender's native USD exporter
- `<asset_id>.json` — sidecar manifest with all sampled ground-truth parameters (schema below)
- `<asset_id>_thumb.png` — a fast low-sample preview render, for visual spot-checking without
  opening Blender

Plus one aggregate `manifest.json` in the output directory indexing every asset record, so
downstream tooling (Isaac Lab's importer) can query the library without touching individual
sidecar files.

### Manifest schema

```json
{
  "asset_id": "d20_00047",
  "usd_path": "d20_00047.usd",
  "thumbnail_path": "d20_00047_thumb.png",
  "die_type": "d20",
  "num_sides": 20,
  "size_mm": 18.3,
  "bevel": 0.08,
  "numbering_scheme": "standard_d20_opposite_sum_21",
  "glyph_style": "arabic_numerals",
  "glyph_method": "engraved",
  "glyph_fill": "painted",
  "font_or_style_id": "font_futura_bold",
  "material_category": "translucent",
  "material_params": { "hue": 0.58, "sat": 0.7, "roughness": 0.15, "ior": 1.45 },
  "seed": 1042042
}
```

### Orchestration & CLI

```
blender --background --python scripts/generate_dice_assets.py -- \
    --count 100 --seed 42 --outdir data/raw/dice_assets
```

`orchestrator.py` loops: sample params -> build geometry -> apply glyphs -> apply material ->
export -> validate. Output goes to `data/raw/dice_assets/` per the repo's existing convention
(README already designates `data/raw/` for raw generated data, kept out of VCS).

Initial batch size: ~100 assets, to validate the pipeline end-to-end before scaling to the
full hundreds/thousands-size library (count is just a CLI parameter — no design difference to
scale up later).

## Error handling

Per-asset failures (e.g. an extreme bevel value causing a self-intersecting mesh, or a
boolean engrave operation failing on a very small/thin face) are caught by the orchestrator,
logged to `failures.json` (asset index + the sampled parameters that caused the failure), and
skipped — the batch continues rather than aborting. At the end, the run reports
`N generated / M failed` so failure rate is visible and reproducible (since params + seed are
logged, a failing case can be re-run in isolation for debugging).

## Validation ("testing")

Since output is generated 3D content rather than conventional unit-testable logic, validation
is a dedicated script (`scripts/validate_dice_assets.py`) run over a generated batch, checking:

- **Mesh sanity**: correct face count for the die type, manifold/watertight
- **Numbering invariant**: programmatically verifies the numbering scheme (e.g. opposite
  faces sum to 7 for d6, 21 for d20)
- **Export integrity**: each USD file opens/parses, each manifest JSON matches the schema and
  correctly cross-references an existing USD + thumbnail file
- **Visual spot-check**: the thumbnail contact sheet lets a human flip through a batch quickly

Additionally, the individual geometry/numbering/sampler modules are structured so their pure
logic (face/vertex counts, numbering invariants, parameter range sampling) can be exercised
with standard Python unit tests without needing a full Blender render, where practical.

## Open items for the future (explicitly out of scope here)

- Isaac Lab import/scene-composition spec (physics tumbling, sensor simulation, environment/
  lighting randomization, annotation export) — separate subsystem, separate spec.
- Model training and real-sensor inference — later subsystems.
