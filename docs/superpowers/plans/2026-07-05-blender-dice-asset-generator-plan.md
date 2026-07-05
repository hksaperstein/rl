# Blender Dice Asset Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a headless-Blender Python pipeline that procedurally generates a library of
dice 3D assets (d4/d6/d8/d10/d12/d20 — geometry, engraved/printed numerals, materials),
exporting each as a USD file plus a JSON ground-truth manifest, ready for a future Isaac Lab
scene-composition stage to consume.

**Architecture:** A `src/dice_gen/` package of single-purpose modules (numbering, sampling,
geometry, glyphs, materials, export, orchestration), invoked headlessly via
`blender --background --python`. Pure-logic modules (numbering, sampler) are tested with
plain `pytest`. Blender-API-dependent modules (geometry, glyphs, materials, exporter,
orchestrator) are tested via small scripts run through `blender --background --python`,
using a shared harness that catches all exceptions and calls `sys.exit(1)`/`sys.exit(0)`
explicitly — **required** because Blender's background mode exits code 0 even when the
script raises an uncaught exception, so a bare `assert` cannot signal test failure to the
shell.

**Tech Stack:** Blender 5.1.2 (confirmed installed at `/snap/bin/blender`, headless `bpy`
verified working), Python 3 stdlib only for pure modules, no external pip packages needed
(the plan avoids `pip install bpy` since Blender's own bundled interpreter is used instead).

## Global Constraints

- Target Blender version: 5.1.2 (installed, confirmed via `blender --version`). Render engine
  identifier in this build is `'BLENDER_EEVEE'` (not `'BLENDER_EEVEE_NEXT'`) — confirmed by
  querying `bpy.types.RenderSettings.bl_rna.properties['engine'].enum_items.keys()`.
- All dice geometry is built via `bmesh.ops.convex_hull` + `bmesh.ops.dissolve_limit` from a
  literal list of base vertices per die type — never hand-authored face/vertex-index lists
  (avoids the risk of manually deriving polyhedron topology incorrectly).
- Any Blender-dependent test script MUST wrap its entire body in try/except and call
  `sys.exit(1)` on failure / fall through to `sys.exit(0)` on success (see Task 1's harness).
  Never rely on a bare `assert` or on Blender's own process exit code in background mode.
- Output goes under `data/raw/dice_assets/` per the existing README convention (`data/raw/`
  is for generated/raw data, kept out of VCS).
- Initial batch size for validation: ~20-100 assets (count is a CLI parameter; scaling to
  hundreds/thousands later requires no code changes).
- This plan covers ONLY the Blender asset-library generator (per
  `docs/superpowers/specs/2026-07-05-blender-dice-asset-generator-design.md`). Isaac Lab
  import/scene-composition is explicitly out of scope.

---

## File Structure

```
src/dice_gen/
  __init__.py
  numbering.py        # Task 1 — pure Python, standard face-numbering schemes
  sampler.py           # Task 2 — pure Python, randomized per-variant parameter sampling
  geometry.py          # Task 3 — bpy, parametric polyhedron mesh construction
  glyphs.py            # Task 4 — bpy, numeral/pip application (engraved + decal)
  materials.py         # Task 5 — bpy, procedural shader/material builder
  exporter.py          # Task 6 — bpy, USD export + manifest + thumbnail render
  orchestrator.py       # Task 7 — bpy, batch loop tying modules 1-6 together

scripts/
  generate_dice_assets.py   # Task 8 — CLI entry point
  validate_dice_assets.py    # Task 9 — standalone validation pass (no bpy required)

tests/
  test_numbering.py          # Task 1 — plain pytest
  test_sampler.py             # Task 2 — plain pytest
  blender/
    _harness.py               # Task 1 — shared exit-code-safe test runner
    test_geometry.py           # Task 3 — blender --background test
    test_glyphs.py              # Task 4
    test_materials.py           # Task 5
    test_exporter.py            # Task 6
    test_orchestrator.py        # Task 7

data/raw/dice_assets/    # Task 10 output — gitignored, generated at runtime
```

---

## Task 1: Package scaffolding, shared test harness, and `numbering.py`

**Files:**
- Create: `src/dice_gen/__init__.py`
- Create: `src/dice_gen/numbering.py`
- Create: `tests/test_numbering.py`
- Create: `tests/blender/_harness.py`
- Create: `tests/blender/__init__.py` (empty, so `tests/blender` is importable if needed)

**Interfaces:**
- Produces: `numbering.NUMBERING_SCHEMES: dict[str, dict]`, `numbering.get_values(die_type: str) -> list[int]`, `numbering.assign_values_to_opposite_pairs(die_type: str, face_pairs: list[tuple[int,int]]) -> dict[int,int]`, `numbering.verify_opposite_sum(die_type: str, face_pairs: list[tuple[int,int]], assignment: dict[int,int]) -> bool`
- Produces: `tests/blender/_harness.run_and_report(fn: Callable[[], None]) -> None` (calls `sys.exit(0)` or `sys.exit(1)`)

- [ ] **Step 1: Create package `__init__.py` files**

`src/dice_gen/__init__.py`:
```python
```
(empty file — just marks the package)

`tests/blender/__init__.py`:
```python
```
(empty file)

- [ ] **Step 2: Write the shared Blender test harness**

`tests/blender/_harness.py`:
```python
"""
Shared test runner for Blender-background test scripts.

Blender's `--background --python script.py` mode exits with code 0 even when
the script raises an uncaught exception (verified empirically against
Blender 5.1.2) — a bare `assert` at module scope will NOT fail the shell
command. Every Blender-dependent test script must call `run_and_report`
instead of relying on Python's default exception propagation.
"""
import sys
import traceback


def run_and_report(fn):
    try:
        fn()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    print("ALL TESTS PASSED")
    sys.exit(0)
```

- [ ] **Step 3: Write the failing test for numbering.py**

`tests/test_numbering.py`:
```python
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dice_gen import numbering


def test_get_values_d20_has_20_unique_values_1_through_20():
    values = numbering.get_values("d20")
    assert len(values) == 20
    assert set(values) == set(range(1, 21))


def test_get_values_d10_has_10_unique_values_0_through_9():
    values = numbering.get_values("d10")
    assert len(values) == 10
    assert set(values) == set(range(0, 10))


def test_d6_opposite_faces_sum_to_7():
    face_pairs = [(0, 1), (2, 3), (4, 5)]
    assignment = numbering.assign_values_to_opposite_pairs("d6", face_pairs)
    assert numbering.verify_opposite_sum("d6", face_pairs, assignment)
    assert set(assignment.values()) == {1, 2, 3, 4, 5, 6}


def test_d20_opposite_faces_sum_to_21():
    face_pairs = [(i, i + 10) for i in range(10)]
    assignment = numbering.assign_values_to_opposite_pairs("d20", face_pairs)
    assert numbering.verify_opposite_sum("d20", face_pairs, assignment)
    assert set(assignment.values()) == set(range(1, 21))


def test_d12_opposite_faces_sum_to_13():
    face_pairs = [(i, i + 6) for i in range(6)]
    assignment = numbering.assign_values_to_opposite_pairs("d12", face_pairs)
    assert numbering.verify_opposite_sum("d12", face_pairs, assignment)
    assert set(assignment.values()) == set(range(1, 13))


def test_d10_opposite_faces_sum_to_9():
    face_pairs = [(i, i + 5) for i in range(5)]
    assignment = numbering.assign_values_to_opposite_pairs("d10", face_pairs)
    assert numbering.verify_opposite_sum("d10", face_pairs, assignment)
    assert set(assignment.values()) == set(range(0, 10))


def test_d4_has_no_opposite_sum_rule_but_assigns_all_values_once():
    face_pairs = [(0, 1), (2, 3)]
    assignment = numbering.assign_values_to_opposite_pairs("d4", face_pairs)
    assert set(assignment.values()) == {1, 2, 3, 4}
```

- [ ] **Step 4: Run test to verify it fails**

Run: `cd /home/saps/projects/Dice-Detection && python3 -m pytest tests/test_numbering.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dice_gen'`

- [ ] **Step 5: Implement `numbering.py`**

`src/dice_gen/numbering.py`:
```python
"""
Standard real-world face-numbering conventions for each die type.

d4 (tetrahedron) has no face-to-face antipodal relationship (its faces are
opposite a *vertex*, not another face), so it has no opposite_sum rule —
values are just assigned once each. All other die types are centrally
symmetric and follow their standard convention:
  d6:  opposite faces sum to 7
  d8:  opposite faces sum to 9
  d10: opposite faces sum to 9 (values 0-9, pairing k with 9-k)
  d12: opposite faces sum to 13
  d20: opposite faces sum to 21
"""

NUMBERING_SCHEMES = {
    "d4": {"values": [1, 2, 3, 4], "opposite_sum": None},
    "d6": {"values": [1, 2, 3, 4, 5, 6], "opposite_sum": 7},
    "d8": {"values": [1, 2, 3, 4, 5, 6, 7, 8], "opposite_sum": 9},
    "d10": {"values": list(range(0, 10)), "opposite_sum": 9},
    "d12": {"values": list(range(1, 13)), "opposite_sum": 13},
    "d20": {"values": list(range(1, 21)), "opposite_sum": 21},
}


def get_values(die_type):
    return list(NUMBERING_SCHEMES[die_type]["values"])


def assign_values_to_opposite_pairs(die_type, face_pairs):
    """
    face_pairs: list of (face_index_a, face_index_b) tuples covering every
    face exactly once. For die types with an opposite_sum rule, each pair is
    assigned (v, opposite_sum - v) so the invariant holds. For d4 (no rule),
    values are just handed out in iteration order — face_pairs there is only
    a convenient grouping, not a real geometric antipodal relationship.

    Returns {face_index: value}.
    """
    scheme = NUMBERING_SCHEMES[die_type]
    values = scheme["values"]
    opposite_sum = scheme["opposite_sum"]

    if opposite_sum is None:
        flat = [face for pair in face_pairs for face in pair]
        return {face: value for face, value in zip(flat, values)}

    remaining = set(values)
    assignment = {}
    for face_a, face_b in face_pairs:
        v_a = min(remaining)
        v_b = opposite_sum - v_a
        if v_b not in remaining:
            raise ValueError(
                f"{die_type}: cannot satisfy opposite_sum={opposite_sum} "
                f"with remaining values {sorted(remaining)}"
            )
        remaining.discard(v_a)
        remaining.discard(v_b)
        assignment[face_a] = v_a
        assignment[face_b] = v_b
    return assignment


def verify_opposite_sum(die_type, face_pairs, assignment):
    opposite_sum = NUMBERING_SCHEMES[die_type]["opposite_sum"]
    if opposite_sum is None:
        return True
    return all(
        assignment[a] + assignment[b] == opposite_sum for a, b in face_pairs
    )
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd /home/saps/projects/Dice-Detection && python3 -m pytest tests/test_numbering.py -v`
Expected: 7 passed

- [ ] **Step 7: Commit**

```bash
git add src/dice_gen/__init__.py src/dice_gen/numbering.py tests/test_numbering.py tests/blender/__init__.py tests/blender/_harness.py
git commit -m "feat: add dice_gen package scaffolding, blender test harness, numbering module"
```

---

## Task 2: `sampler.py` — randomized per-variant parameter sampling

**Files:**
- Create: `src/dice_gen/sampler.py`
- Create: `tests/test_sampler.py`

**Interfaces:**
- Consumes: nothing from other modules (pure stdlib `random`)
- Produces: `sampler.DiceVariantParams` (dataclass with fields: `die_type, size_mm, bevel_fraction, numbering_scheme, glyph_style, glyph_method, glyph_fill, font_or_style_id, material_category, material_params, d4_placement, seed`), `sampler.sample_variant(seed: int) -> DiceVariantParams`, `sampler.DIE_TYPES: list[str]`, `sampler.SIZE_RANGES_MM: dict[str, tuple[float,float]]`

- [ ] **Step 1: Write the failing tests**

`tests/test_sampler.py`:
```python
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dice_gen import sampler


def test_sample_variant_is_reproducible_with_same_seed():
    a = sampler.sample_variant(42)
    b = sampler.sample_variant(42)
    assert a == b


def test_sample_variant_covers_more_than_one_die_type_across_seeds():
    die_types = {sampler.sample_variant(s).die_type for s in range(50)}
    assert len(die_types) > 1


def test_size_within_configured_range_for_die_type():
    for seed in range(50):
        v = sampler.sample_variant(seed)
        lo, hi = sampler.SIZE_RANGES_MM[v.die_type]
        assert lo <= v.size_mm <= hi


def test_d6_glyph_style_is_numerals_or_pips_only():
    for seed in range(200):
        v = sampler.sample_variant(seed)
        if v.die_type == "d6":
            assert v.glyph_style in ("arabic_numerals", "pips")


def test_non_d6_non_d4_dice_never_use_pips():
    for seed in range(200):
        v = sampler.sample_variant(seed)
        if v.die_type not in ("d6", "d4"):
            assert v.glyph_style != "pips"


def test_glyph_fill_blank_only_possible_for_engraved_method():
    for seed in range(200):
        v = sampler.sample_variant(seed)
        if v.glyph_fill == "blank":
            assert v.glyph_method == "engraved"


def test_d4_placement_set_only_for_d4():
    for seed in range(200):
        v = sampler.sample_variant(seed)
        if v.die_type == "d4":
            assert v.d4_placement in ("face_centered", "vertex_labeled")
        else:
            assert v.d4_placement is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/saps/projects/Dice-Detection && python3 -m pytest tests/test_sampler.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'dice_gen.sampler'` (or `ImportError`)

- [ ] **Step 3: Implement `sampler.py`**

`src/dice_gen/sampler.py`:
```python
import random
from dataclasses import dataclass
from typing import Optional

DIE_TYPES = ["d4", "d6", "d8", "d10", "d12", "d20"]

SIZE_RANGES_MM = {
    "d4": (14.0, 20.0),
    "d6": (12.0, 20.0),
    "d8": (14.0, 20.0),
    "d10": (14.0, 20.0),
    "d12": (16.0, 22.0),
    "d20": (16.0, 24.0),
}

MATERIAL_CATEGORIES = ["opaque", "translucent", "marbled", "glitter", "metallic", "speckled"]
GLYPH_STYLES = ["arabic_numerals", "roman_numerals", "pips", "greek_numerals", "cjk_numerals"]
GLYPH_METHODS = ["engraved", "printed_decal"]
GLYPH_FILLS = ["painted", "blank"]
D4_PLACEMENT_STYLES = ["face_centered", "vertex_labeled"]
FONT_IDS = ["font_sans_bold", "font_serif_regular", "font_display_condensed"]


@dataclass
class DiceVariantParams:
    die_type: str
    size_mm: float
    bevel_fraction: float
    numbering_scheme: str
    glyph_style: str
    glyph_method: str
    glyph_fill: str
    font_or_style_id: str
    material_category: str
    material_params: dict
    d4_placement: Optional[str]
    seed: int


def sample_variant(seed: int) -> DiceVariantParams:
    rng = random.Random(seed)

    die_type = rng.choice(DIE_TYPES)
    lo, hi = SIZE_RANGES_MM[die_type]
    size_mm = rng.uniform(lo, hi)
    bevel_fraction = rng.uniform(0.02, 0.06)

    if die_type in ("d6", "d4"):
        glyph_style = rng.choice(["arabic_numerals", "pips"])
    else:
        glyph_style = rng.choice([s for s in GLYPH_STYLES if s != "pips"])

    glyph_method = rng.choice(GLYPH_METHODS)
    glyph_fill = rng.choice(GLYPH_FILLS) if glyph_method == "engraved" else "painted"
    font_or_style_id = rng.choice(FONT_IDS)

    material_category = rng.choice(MATERIAL_CATEGORIES)
    material_params = _sample_material_params(rng, material_category)

    d4_placement = rng.choice(D4_PLACEMENT_STYLES) if die_type == "d4" else None

    return DiceVariantParams(
        die_type=die_type,
        size_mm=size_mm,
        bevel_fraction=bevel_fraction,
        numbering_scheme=f"standard_{die_type}",
        glyph_style=glyph_style,
        glyph_method=glyph_method,
        glyph_fill=glyph_fill,
        font_or_style_id=font_or_style_id,
        material_category=material_category,
        material_params=material_params,
        d4_placement=d4_placement,
        seed=seed,
    )


def _sample_material_params(rng, category):
    params = {
        "hue": rng.uniform(0.0, 1.0),
        "saturation": rng.uniform(0.3, 1.0),
        "value": rng.uniform(0.2, 0.9),
        "roughness": rng.uniform(0.1, 0.7),
    }
    if category == "translucent":
        params["ior"] = rng.uniform(1.3, 1.6)
        params["transmission"] = rng.uniform(0.7, 1.0)
    elif category == "marbled":
        params["noise_scale"] = rng.uniform(2.0, 8.0)
        params["secondary_hue"] = rng.uniform(0.0, 1.0)
    elif category == "glitter":
        params["sparkle_density"] = rng.uniform(20.0, 80.0)
    elif category == "metallic":
        params["roughness"] = rng.uniform(0.05, 0.35)
    elif category == "speckled":
        params["speckle_density"] = rng.uniform(30.0, 100.0)
        params["secondary_hue"] = rng.uniform(0.0, 1.0)
    return params
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/saps/projects/Dice-Detection && python3 -m pytest tests/test_sampler.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/dice_gen/sampler.py tests/test_sampler.py
git commit -m "feat: add randomized dice variant parameter sampler"
```

---

## Task 3: `geometry.py` — parametric polyhedron construction for all 6 die types

**Files:**
- Create: `src/dice_gen/geometry.py`
- Create: `tests/blender/test_geometry.py`

**Interfaces:**
- Consumes: nothing from other modules
- Produces: `geometry.DIE_SPECS: dict[str, dict]`, `geometry.GeometryBuildError` (Exception subclass), `geometry.build_die_base_mesh(die_type: str, size_mm: float) -> bpy.types.Object`, `geometry.compute_opposite_face_pairs(obj: bpy.types.Object) -> list[tuple[int,int]]`

- [ ] **Step 1: Write the failing Blender test**

`tests/blender/test_geometry.py`:
```python
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from _harness import run_and_report


def test_all_six_dice_build_with_correct_topology():
    import bpy
    from dice_gen import geometry

    for die_type, spec in geometry.DIE_SPECS.items():
        obj = geometry.build_die_base_mesh(die_type, size_mm=16.0)
        n_faces = len(obj.data.polygons)
        n_verts = len(obj.data.vertices)
        n_edges = len(obj.data.edges)

        assert n_faces == spec["expected_faces"], (
            f"{die_type}: expected {spec['expected_faces']} faces, got {n_faces}"
        )
        assert n_verts == spec["expected_verts"], (
            f"{die_type}: expected {spec['expected_verts']} verts, got {n_verts}"
        )
        assert n_edges == spec["expected_edges"], (
            f"{die_type}: expected {spec['expected_edges']} edges, got {n_edges}"
        )
        bpy.data.objects.remove(obj, do_unlink=True)


def test_opposite_face_pairs_are_geometrically_antiparallel_for_d6():
    import bpy
    from dice_gen import geometry

    obj = geometry.build_die_base_mesh("d6", size_mm=16.0)
    pairs = geometry.compute_opposite_face_pairs(obj)
    assert len(pairs) == 3

    obj.data.polygons.foreach_set  # ensure normals accessible
    for a, b in pairs:
        na = obj.data.polygons[a].normal
        nb = obj.data.polygons[b].normal
        dot = na.dot(nb)
        assert dot < -0.99, f"faces {a},{b} not antiparallel (dot={dot})"

    bpy.data.objects.remove(obj, do_unlink=True)


def test_d4_opposite_face_pairs_returns_two_pairs_covering_all_faces():
    import bpy
    from dice_gen import geometry

    obj = geometry.build_die_base_mesh("d4", size_mm=16.0)
    pairs = geometry.compute_opposite_face_pairs(obj)
    flat = sorted(f for pair in pairs for f in pair)
    assert flat == [0, 1, 2, 3]
    bpy.data.objects.remove(obj, do_unlink=True)


def run():
    test_all_six_dice_build_with_correct_topology()
    test_opposite_face_pairs_are_geometrically_antiparallel_for_d6()
    test_d4_opposite_face_pairs_returns_two_pairs_covering_all_faces()


run_and_report(run)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/saps/projects/Dice-Detection && blender --background --python tests/blender/test_geometry.py`
Expected: exit code 1, traceback showing `ModuleNotFoundError: No module named 'dice_gen'` (verify with `echo $?` after)

- [ ] **Step 3: Implement `geometry.py`**

`src/dice_gen/geometry.py`:
```python
"""
Parametric construction of the 6 standard TTRPG die shapes.

Each shape is built from a literal list of base vertices (well-known
coordinates for the Platonic solids, plus an empirically-derived pentagonal
trapezohedron for d10 — see _d10_base_vertices) via bmesh's convex_hull +
dissolve_limit. This avoids hand-deriving face/vertex-index topology by
hand: convex_hull computes the correct facets from the point set, and
dissolve_limit merges coplanar hull triangles back into the real N-gon
faces (quads for d10's kites, pentagons for d12, etc).

The d10 vertex ratio (apex height / ring z-offset ≈ 9.47, at ring radius 1)
was found by a numeric sweep confirming near-exact coplanarity (dihedral
deficit < 0.001 deg) of adjacent hull triangles — verified empirically
against this project's installed Blender 5.1.2 before being hardcoded here.
"""
import math

import bmesh
import bpy

PHI = (1 + 5 ** 0.5) / 2
DISSOLVE_ANGLE_DEG = 2.0


class GeometryBuildError(Exception):
    pass


def _d10_base_vertices():
    h, c, r = 0.947, 0.100, 1.0
    verts = [(0, 0, h), (0, 0, -h)]
    for k in range(10):
        theta = math.radians(36 * k)
        z = c if k % 2 == 0 else -c
        verts.append((r * math.cos(theta), r * math.sin(theta), z))
    return verts


DIE_SPECS = {
    "d4": {
        "num_sides": 4,
        "base_vertices": [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)],
        "expected_faces": 4,
        "expected_verts": 4,
        "expected_edges": 6,
    },
    "d6": {
        "num_sides": 6,
        "base_vertices": [(x, y, z) for x in (1, -1) for y in (1, -1) for z in (1, -1)],
        "expected_faces": 6,
        "expected_verts": 8,
        "expected_edges": 12,
    },
    "d8": {
        "num_sides": 8,
        "base_vertices": [
            (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
        ],
        "expected_faces": 8,
        "expected_verts": 6,
        "expected_edges": 12,
    },
    "d10": {
        "num_sides": 10,
        "base_vertices": _d10_base_vertices(),
        "expected_faces": 10,
        "expected_verts": 12,
        "expected_edges": 20,
    },
    "d12": {
        "num_sides": 12,
        "base_vertices": (
            [(x, y, z) for x in (1, -1) for y in (1, -1) for z in (1, -1)]
            + [(0, s1 / PHI, s2 * PHI) for s1 in (1, -1) for s2 in (1, -1)]
            + [(s1 / PHI, s2 * PHI, 0) for s1 in (1, -1) for s2 in (1, -1)]
            + [(s1 * PHI, 0, s2 / PHI) for s1 in (1, -1) for s2 in (1, -1)]
        ),
        "expected_faces": 12,
        "expected_verts": 20,
        "expected_edges": 30,
    },
    "d20": {
        "num_sides": 20,
        "base_vertices": (
            [(0, s1 * 1, s2 * PHI) for s1 in (1, -1) for s2 in (1, -1)]
            + [(s1 * 1, s2 * PHI, 0) for s1 in (1, -1) for s2 in (1, -1)]
            + [(s1 * PHI, 0, s2 * 1) for s1 in (1, -1) for s2 in (1, -1)]
        ),
        "expected_faces": 20,
        "expected_verts": 12,
        "expected_edges": 30,
    },
}


def build_die_base_mesh(die_type, size_mm):
    spec = DIE_SPECS[die_type]
    scale = size_mm / 2.0

    bm = bmesh.new()
    bmverts = [bm.verts.new((x * scale, y * scale, z * scale)) for (x, y, z) in spec["base_vertices"]]
    bmesh.ops.convex_hull(bm, input=bmverts)
    bmesh.ops.dissolve_limit(
        bm, angle_limit=math.radians(DISSOLVE_ANGLE_DEG), verts=bm.verts, edges=bm.edges
    )
    bm.faces.ensure_lookup_table()
    bm.normal_update()

    if len(bm.faces) != spec["expected_faces"] or len(bm.verts) != spec["expected_verts"]:
        n_faces, n_verts = len(bm.faces), len(bm.verts)
        bm.free()
        raise GeometryBuildError(
            f"{die_type}: expected {spec['expected_faces']} faces / {spec['expected_verts']} verts, "
            f"got {n_faces} faces / {n_verts} verts"
        )

    mesh = bpy.data.meshes.new(f"{die_type}_mesh")
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    obj = bpy.data.objects.new(f"{die_type}_die", mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def compute_opposite_face_pairs(obj):
    """
    Returns [(face_index_a, face_index_b), ...] pairs. For the 5 centrally
    symmetric dice (d6/d8/d10/d12/d20) these are true antipodal face pairs
    (most anti-parallel normals, greedily matched). d4 (tetrahedron) has no
    antipodal faces — its numbering has no opposite_sum rule anyway, so this
    just returns a stable consecutive grouping.
    """
    faces = list(obj.data.polygons)
    n = len(faces)
    if n == 4:
        return [(0, 1), (2, 3)]

    remaining = set(range(n))
    pairs = []
    while remaining:
        i = min(remaining)
        remaining.discard(i)
        best_j, best_dot = None, 2.0
        for j in remaining:
            dot = faces[i].normal.dot(faces[j].normal)
            if dot < best_dot:
                best_dot = dot
                best_j = j
        pairs.append((i, best_j))
        remaining.discard(best_j)
    return pairs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/saps/projects/Dice-Detection && blender --background --python tests/blender/test_geometry.py; echo "exit=$?"`
Expected: prints `ALL TESTS PASSED` and `exit=0`

- [ ] **Step 5: Commit**

```bash
git add src/dice_gen/geometry.py tests/blender/test_geometry.py
git commit -m "feat: add parametric polyhedron construction for all 6 die types"
```

---

## Task 4: `glyphs.py` — numeral/pip application (engraved + printed-decal)

**Files:**
- Create: `src/dice_gen/glyphs.py`
- Create: `tests/blender/test_glyphs.py`

**Interfaces:**
- Consumes: `geometry.build_die_base_mesh`, `geometry.compute_opposite_face_pairs` (Task 3); `numbering.assign_values_to_opposite_pairs` (Task 1)
- Produces: `glyphs.glyph_label(value: int, glyph_style: str) -> str`, `glyphs.apply_engraved_glyphs(die_obj, die_type, assignment, glyph_style, glyph_fill, font_id, size_mm) -> None`, `glyphs.apply_decal_glyphs(die_obj, die_type, assignment, glyph_style, font_id, size_mm, tmp_dir) -> None`

- [ ] **Step 1: Write the failing Blender test**

`tests/blender/test_glyphs.py`:
```python
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from _harness import run_and_report


def test_glyph_label_formats():
    from dice_gen import glyphs

    assert glyphs.glyph_label(6, "arabic_numerals") == "6"
    assert glyphs.glyph_label(20, "roman_numerals") == "XX"
    assert glyphs.glyph_label(9, "roman_numerals") == "IX"


def test_engraved_glyphs_reduce_solid_volume():
    import bpy
    from dice_gen import geometry, numbering, glyphs

    die_type = "d6"
    obj = geometry.build_die_base_mesh(die_type, size_mm=16.0)
    pairs = geometry.compute_opposite_face_pairs(obj)
    assignment = numbering.assign_values_to_opposite_pairs(die_type, pairs)

    import bmesh
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    volume_before = bm.calc_volume()
    bm.free()

    glyphs.apply_engraved_glyphs(
        obj, die_type, assignment,
        glyph_style="arabic_numerals", glyph_fill="painted",
        font_id="font_sans_bold", size_mm=16.0,
    )

    bm2 = bmesh.new()
    bm2.from_mesh(obj.data)
    volume_after = bm2.calc_volume()
    bm2.free()

    assert volume_after < volume_before, "engraving should remove material"
    assert len(obj.data.materials) >= 2, "painted fill should add a second material slot"

    bpy.data.objects.remove(obj, do_unlink=True)


def test_decal_glyphs_assigns_one_material_per_face():
    import bpy
    from dice_gen import geometry, numbering, glyphs

    die_type = "d6"
    obj = geometry.build_die_base_mesh(die_type, size_mm=16.0)
    pairs = geometry.compute_opposite_face_pairs(obj)
    assignment = numbering.assign_values_to_opposite_pairs(die_type, pairs)

    with tempfile.TemporaryDirectory() as tmp_dir:
        glyphs.apply_decal_glyphs(
            obj, die_type, assignment,
            glyph_style="arabic_numerals", font_id="font_sans_bold",
            size_mm=16.0, tmp_dir=tmp_dir,
        )
        assert len(obj.data.materials) == 6
        for face_index in assignment:
            mat_index = obj.data.polygons[face_index].material_index
            assert obj.data.materials[mat_index] is not None

    bpy.data.objects.remove(obj, do_unlink=True)


def run():
    test_glyph_label_formats()
    test_engraved_glyphs_reduce_solid_volume()
    test_decal_glyphs_assigns_one_material_per_face()


run_and_report(run)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/saps/projects/Dice-Detection && blender --background --python tests/blender/test_glyphs.py; echo "exit=$?"`
Expected: `exit=1`, `ModuleNotFoundError: No module named 'dice_gen.glyphs'`

- [ ] **Step 3: Implement `glyphs.py`**

`src/dice_gen/glyphs.py`:
```python
"""
Applies numerals/pips to a die's faces, either as real engraved geometry
(boolean-cut into the mesh) or as printed texture decals (one material +
baked image per face). Both are real manufacturing conventions for TTRPG
dice, so both are supported and randomly chosen per asset by sampler.py.
"""
import os

import bpy
import bmesh
from mathutils import Vector, Matrix

ENGRAVE_DEPTH_FRACTION = 0.04

PIP_VALUE_LAYOUTS = {
    1: [(0, 0)],
    2: [(-0.3, -0.3), (0.3, 0.3)],
    3: [(-0.3, -0.3), (0, 0), (0.3, 0.3)],
    4: [(-0.3, -0.3), (0.3, -0.3), (-0.3, 0.3), (0.3, 0.3)],
    5: [(-0.3, -0.3), (0.3, -0.3), (0, 0), (-0.3, 0.3), (0.3, 0.3)],
    6: [(-0.3, -0.3), (0.3, -0.3), (-0.3, 0), (0.3, 0), (-0.3, 0.3), (0.3, 0.3)],
}

ROMAN_NUMERALS = {
    1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII",
    9: "IX", 10: "X", 11: "XI", 12: "XII", 13: "XIII", 14: "XIV", 15: "XV",
    16: "XVI", 17: "XVII", 18: "XVIII", 19: "XIX", 20: "XX",
}

GREEK_NUMERALS = {
    0: "Ω", 1: "Α", 2: "Β", 3: "Γ", 4: "Δ", 5: "Ε",
    6: "Ζ", 7: "Η", 8: "Θ", 9: "Ι", 10: "ΙΑ",
    11: "ΙΒ", 12: "ΙΓ", 13: "ΙΔ", 14: "ΙΕ",
    15: "ΙΣ", 16: "ΙΖ", 17: "ΙΗ", 18: "ΙΘ",
    19: "Κ", 20: "ΚΑ",
}

CJK_NUMERALS = {
    0: "零", 1: "一", 2: "二", 3: "三", 4: "四", 5: "五",
    6: "六", 7: "七", 8: "八", 9: "九", 10: "十",
    11: "十一", 12: "十二", 13: "十三", 14: "十四",
    15: "十五", 16: "十六", 17: "十七", 18: "十八",
    19: "十九", 20: "二十",
}


def glyph_label(value, glyph_style):
    if glyph_style == "arabic_numerals":
        return str(value)
    if glyph_style == "roman_numerals":
        return ROMAN_NUMERALS.get(value, str(value))
    if glyph_style == "greek_numerals":
        return GREEK_NUMERALS.get(value, str(value))
    if glyph_style == "cjk_numerals":
        return CJK_NUMERALS.get(value, str(value))
    raise ValueError(f"glyph_label not applicable to style {glyph_style!r}")


def _face_orientation_matrix(face, obj_matrix):
    center = obj_matrix @ face.center
    normal = (obj_matrix.to_3x3() @ face.normal).normalized()
    up_hint = Vector((0, 0, 1)) if abs(normal.z) < 0.9 else Vector((0, 1, 0))
    tangent = up_hint.cross(normal).normalized()
    bitangent = normal.cross(tangent).normalized()
    rot = Matrix((tangent, bitangent, normal)).transposed().to_4x4()
    rot.translation = center
    return rot


def _boolean_diff_apply(die_obj, cutter_obj):
    mod = die_obj.modifiers.new(name="Engrave", type='BOOLEAN')
    mod.operation = 'DIFFERENCE'
    mod.object = cutter_obj
    mod.solver = 'EXACT'
    bpy.context.view_layer.objects.active = die_obj
    bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.data.objects.remove(cutter_obj, do_unlink=True)


def apply_engraved_glyphs(die_obj, die_type, assignment, glyph_style, glyph_fill, font_id, size_mm):
    depth = size_mm * ENGRAVE_DEPTH_FRACTION
    glyph_font_size = size_mm * 0.18

    for face_index, value in assignment.items():
        face = die_obj.data.polygons[face_index]
        orient = _face_orientation_matrix(face, die_obj.matrix_world)

        if glyph_style == "pips":
            for (ox, oy) in PIP_VALUE_LAYOUTS.get(value, [(0, 0)]):
                bpy.ops.mesh.primitive_uv_sphere_add(radius=size_mm * 0.05)
                pip = bpy.context.active_object
                pip.location = orient @ Vector(
                    (ox * size_mm * 0.4, oy * size_mm * 0.4, -depth * 0.5)
                )
                _boolean_diff_apply(die_obj, pip)
        else:
            label = glyph_label(value, glyph_style)
            bpy.ops.object.text_add()
            txt_obj = bpy.context.active_object
            txt_obj.data.body = label
            txt_obj.data.align_x = 'CENTER'
            txt_obj.data.align_y = 'CENTER'
            txt_obj.data.size = glyph_font_size
            txt_obj.data.extrude = depth
            bpy.context.view_layer.objects.active = txt_obj
            bpy.ops.object.convert(target='MESH')
            txt_obj.matrix_world = orient @ Matrix.Translation((0, 0, -depth))
            _boolean_diff_apply(die_obj, txt_obj)

    if glyph_fill == "painted":
        _assign_fill_material_to_recessed_faces(die_obj)


def _assign_fill_material_to_recessed_faces(die_obj):
    """
    Heuristic: the boolean engrave cuts create small new faces at the bottom
    of each recess, which are noticeably smaller than the die's flat body
    faces. Faces well below the mesh's average face area get material slot 1
    (the fill color); materials.py is responsible for populating slot 1 with
    an actual material.
    """
    if len(die_obj.data.materials) < 2:
        die_obj.data.materials.append(None)

    bm = bmesh.new()
    bm.from_mesh(die_obj.data)
    bm.faces.ensure_lookup_table()
    avg_area = sum(f.calc_area() for f in bm.faces) / len(bm.faces)
    for f in bm.faces:
        if f.calc_area() < avg_area * 0.15:
            f.material_index = 1
    bm.to_mesh(die_obj.data)
    die_obj.data.update()
    bm.free()


def apply_decal_glyphs(die_obj, die_type, assignment, glyph_style, font_id, size_mm, tmp_dir):
    bpy.context.view_layer.objects.active = die_obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(island_margin=0.05)
    bpy.ops.object.mode_set(mode='OBJECT')

    for face_index, value in assignment.items():
        image_path = os.path.join(tmp_dir, f"{die_obj.name}_face{face_index}.png")
        _render_label_to_image(value, glyph_style, image_path)

        mat = bpy.data.materials.new(name=f"{die_obj.name}_face{face_index}_decal")
        mat.use_nodes = True
        nt = mat.node_tree
        bsdf = nt.nodes["Principled BSDF"]
        tex_node = nt.nodes.new("ShaderNodeTexImage")
        tex_node.image = bpy.data.images.load(image_path)
        nt.links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])

        die_obj.data.materials.append(mat)
        die_obj.data.polygons[face_index].material_index = len(die_obj.data.materials) - 1


def _render_label_to_image(value, glyph_style, image_path, resolution=256):
    scene = bpy.data.scenes.new("dice_gen_decal_tmp")
    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.film_transparent = True

    cam_data = bpy.data.cameras.new("decal_cam")
    cam_data.type = 'ORTHO'
    cam_data.ortho_scale = 1.4
    cam_obj = bpy.data.objects.new("decal_cam", cam_data)
    cam_obj.location = (0, 0, 2)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    light_data = bpy.data.lights.new("decal_light", type='SUN')
    light_obj = bpy.data.objects.new("decal_light", light_data)
    light_obj.location = (0, 0, 3)
    scene.collection.objects.link(light_obj)

    if glyph_style == "pips":
        for (ox, oy) in PIP_VALUE_LAYOUTS.get(value, [(0, 0)]):
            bpy.ops.mesh.primitive_circle_add(
                radius=0.12, fill_type='NGON', location=(ox, oy, 0)
            )
            dot = bpy.context.active_object
            bpy.context.scene.collection.objects.unlink(dot)
            scene.collection.objects.link(dot)
    else:
        label = glyph_label(value, glyph_style)
        bpy.ops.object.text_add(location=(0, 0, 0))
        txt_obj = bpy.context.active_object
        txt_obj.data.body = label
        txt_obj.data.align_x = 'CENTER'
        txt_obj.data.align_y = 'CENTER'
        txt_obj.data.size = 1.0
        bpy.context.scene.collection.objects.unlink(txt_obj)
        scene.collection.objects.link(txt_obj)

    scene.render.filepath = image_path
    prev_scene = bpy.context.window.scene
    bpy.context.window.scene = scene
    bpy.ops.render.render(write_still=True)
    bpy.context.window.scene = prev_scene

    bpy.data.scenes.remove(scene)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/saps/projects/Dice-Detection && blender --background --python tests/blender/test_glyphs.py; echo "exit=$?"`
Expected: prints `ALL TESTS PASSED` and `exit=0`. If the volume-reduction or material-count
assertions fail, inspect which die/value combination failed and adjust `ENGRAVE_DEPTH_FRACTION`
or the fill-face-area heuristic threshold (`avg_area * 0.15`) — these are tunable constants,
not hardcoded requirements.

- [ ] **Step 5: Commit**

```bash
git add src/dice_gen/glyphs.py tests/blender/test_glyphs.py
git commit -m "feat: add engraved and printed-decal glyph application"
```

---

## Task 5: `materials.py` — procedural material builder

**Files:**
- Create: `src/dice_gen/materials.py`
- Create: `tests/blender/test_materials.py`

**Interfaces:**
- Consumes: nothing from other modules (operates on any `bpy.types.Object`)
- Produces: `materials.build_material(die_name: str, category: str, params: dict) -> bpy.types.Material`, `materials.apply_material(die_obj, mat, slot_index: int = 0) -> None`, `materials.build_fill_material(die_name: str, params: dict) -> bpy.types.Material`, `materials.MATERIAL_CATEGORIES: list[str]`

- [ ] **Step 1: Write the failing Blender test**

`tests/blender/test_materials.py`:
```python
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from _harness import run_and_report


def test_all_material_categories_build_without_error():
    import bpy
    from dice_gen import materials

    params = {
        "hue": 0.5, "saturation": 0.7, "value": 0.6, "roughness": 0.3,
        "ior": 1.45, "transmission": 0.9, "noise_scale": 5.0,
        "secondary_hue": 0.1, "sparkle_density": 40.0, "speckle_density": 60.0,
    }
    for category in materials.MATERIAL_CATEGORIES:
        mat = materials.build_material("test_die", category, params)
        assert mat is not None
        assert mat.use_nodes
        assert mat.node_tree.nodes.get("Principled BSDF") is not None


def test_apply_material_appends_to_first_empty_slot():
    import bpy
    from dice_gen import geometry, materials

    obj = geometry.build_die_base_mesh("d6", size_mm=16.0)
    mat = materials.build_material("d6", "opaque", {"hue": 0.2, "saturation": 0.8, "value": 0.5, "roughness": 0.4})
    materials.apply_material(obj, mat, slot_index=0)
    assert len(obj.data.materials) == 1
    assert obj.data.materials[0] is mat
    bpy.data.objects.remove(obj, do_unlink=True)


def test_metallic_material_sets_metallic_input_to_one():
    from dice_gen import materials

    mat = materials.build_material("d20", "metallic", {"hue": 0.6, "saturation": 0.1, "value": 0.8, "roughness": 0.2})
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    assert bsdf.inputs["Metallic"].default_value == 1.0


def run():
    test_all_material_categories_build_without_error()
    test_apply_material_appends_to_first_empty_slot()
    test_metallic_material_sets_metallic_input_to_one()


run_and_report(run)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/saps/projects/Dice-Detection && blender --background --python tests/blender/test_materials.py; echo "exit=$?"`
Expected: `exit=1`, `ModuleNotFoundError: No module named 'dice_gen.materials'`

- [ ] **Step 3: Implement `materials.py`**

`src/dice_gen/materials.py`:
```python
"""
Procedural shader-node materials for the 6 realistic dice finish categories.
Node input/output socket names below (e.g. "Transmission Weight", "Factor")
were confirmed against this project's installed Blender 5.1.2 — Blender has
renamed several Principled BSDF and texture-node sockets across versions
(e.g. "Transmission" -> "Transmission Weight", noise/ramp "Fac" -> "Factor").
"""
import colorsys

import bpy

MATERIAL_CATEGORIES = ["opaque", "translucent", "marbled", "glitter", "metallic", "speckled"]


def _hsv_to_rgba(h, s, v, a=1.0):
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (r, g, b, a)


def build_material(die_name, category, params):
    mat = bpy.data.materials.new(name=f"{die_name}_{category}")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]

    base_color = _hsv_to_rgba(params["hue"], params["saturation"], params["value"])
    bsdf.inputs["Base Color"].default_value = base_color
    bsdf.inputs["Roughness"].default_value = params["roughness"]

    if category == "opaque":
        pass

    elif category == "translucent":
        bsdf.inputs["Transmission Weight"].default_value = params.get("transmission", 0.9)
        bsdf.inputs["IOR"].default_value = params.get("ior", 1.45)

    elif category == "marbled":
        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = params.get("noise_scale", 5.0)
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        secondary = _hsv_to_rgba(params.get("secondary_hue", 0.0), params["saturation"], params["value"])
        ramp.color_ramp.elements[0].color = base_color
        ramp.color_ramp.elements[1].color = secondary
        nt.links.new(noise.outputs["Factor"], ramp.inputs["Factor"])
        nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

    elif category == "glitter":
        voronoi = nt.nodes.new("ShaderNodeTexVoronoi")
        voronoi.inputs["Scale"].default_value = params.get("sparkle_density", 40.0)
        voronoi.feature = 'DISTANCE_TO_EDGE'
        bsdf.inputs["Metallic"].default_value = 0.6
        nt.links.new(voronoi.outputs["Distance"], bsdf.inputs["Roughness"])

    elif category == "metallic":
        bsdf.inputs["Metallic"].default_value = 1.0
        bsdf.inputs["Roughness"].default_value = params.get("roughness", 0.15)

    elif category == "speckled":
        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = params.get("speckle_density", 60.0)
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].position = 0.45
        ramp.color_ramp.elements[1].position = 0.55
        secondary = _hsv_to_rgba(params.get("secondary_hue", 0.0), params["saturation"], params["value"])
        ramp.color_ramp.elements[0].color = base_color
        ramp.color_ramp.elements[1].color = secondary
        nt.links.new(noise.outputs["Factor"], ramp.inputs["Factor"])
        nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

    else:
        raise ValueError(f"unknown material category: {category!r}")

    return mat


def apply_material(die_obj, mat, slot_index=0):
    if len(die_obj.data.materials) <= slot_index:
        die_obj.data.materials.append(mat)
    else:
        die_obj.data.materials[slot_index] = mat


def build_fill_material(die_name, params):
    """Plain-color material for painted glyph fill (material slot 1)."""
    fill_hue = (params["hue"] + 0.5) % 1.0
    mat = bpy.data.materials.new(name=f"{die_name}_fill")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = _hsv_to_rgba(fill_hue, 0.8, 0.9)
    bsdf.inputs["Roughness"].default_value = 0.4
    return mat
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/saps/projects/Dice-Detection && blender --background --python tests/blender/test_materials.py; echo "exit=$?"`
Expected: prints `ALL TESTS PASSED` and `exit=0`

- [ ] **Step 5: Commit**

```bash
git add src/dice_gen/materials.py tests/blender/test_materials.py
git commit -m "feat: add procedural material builder for 6 finish categories"
```

---

## Task 6: `exporter.py` — bevel bake, USD export, manifest, thumbnail

**Files:**
- Create: `src/dice_gen/exporter.py`
- Create: `tests/blender/test_exporter.py`

**Interfaces:**
- Consumes: a fully-built `bpy.types.Object` (geometry + glyphs + materials already applied)
- Produces: `exporter.export_asset(die_obj, manifest_record: dict, outdir: str, bevel_fraction: float, size_mm: float) -> str` (returns manifest JSON path; also writes `<asset_id>.usd` and `<asset_id>_thumb.png` into `outdir`, and mutates `manifest_record` in place adding `usd_path`/`thumbnail_path` keys)

- [ ] **Step 1: Write the failing Blender test**

`tests/blender/test_exporter.py`:
```python
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from _harness import run_and_report


def test_export_asset_writes_usd_manifest_and_thumbnail():
    import bpy
    from dice_gen import geometry, materials, exporter

    obj = geometry.build_die_base_mesh("d6", size_mm=16.0)
    mat = materials.build_material("d6", "opaque", {"hue": 0.3, "saturation": 0.7, "value": 0.6, "roughness": 0.4})
    materials.apply_material(obj, mat)

    with tempfile.TemporaryDirectory() as outdir:
        record = {"asset_id": "test_d6", "die_type": "d6"}
        manifest_path = exporter.export_asset(obj, record, outdir, bevel_fraction=0.04, size_mm=16.0)

        usd_path = os.path.join(outdir, "test_d6.usd")
        thumb_path = os.path.join(outdir, "test_d6_thumb.png")

        assert os.path.exists(usd_path) and os.path.getsize(usd_path) > 0
        assert os.path.exists(thumb_path) and os.path.getsize(thumb_path) > 0
        assert os.path.exists(manifest_path)

        with open(manifest_path) as f:
            loaded = json.load(f)
        assert loaded["usd_path"] == "test_d6.usd"
        assert loaded["thumbnail_path"] == "test_d6_thumb.png"

    bpy.data.objects.remove(obj, do_unlink=True)


run_and_report(test_export_asset_writes_usd_manifest_and_thumbnail)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/saps/projects/Dice-Detection && blender --background --python tests/blender/test_exporter.py; echo "exit=$?"`
Expected: `exit=1`, `ModuleNotFoundError: No module named 'dice_gen.exporter'`

- [ ] **Step 3: Implement `exporter.py`**

`src/dice_gen/exporter.py`:
```python
"""
Bakes the non-destructive edge bevel, exports the die as USD, renders a
thumbnail for visual spot-checking, and writes the per-asset JSON manifest.

Bevel uses limit_method='ANGLE' (not 'NONE') so it only rounds the die's
structural edges (e.g. a cube's ~90 degree edges) while leaving shallow
engraved-numeral recesses (much shallower angle deltas) crisp.
"""
import json
import math
import os

import bpy


def export_asset(die_obj, manifest_record, outdir, bevel_fraction, size_mm):
    os.makedirs(outdir, exist_ok=True)
    asset_id = manifest_record["asset_id"]

    mod = die_obj.modifiers.new(name="Bevel", type='BEVEL')
    mod.width = size_mm * bevel_fraction
    mod.limit_method = 'ANGLE'
    mod.angle_limit = math.radians(35)
    bpy.context.view_layer.objects.active = die_obj
    bpy.ops.object.modifier_apply(modifier=mod.name)

    usd_path = os.path.join(outdir, f"{asset_id}.usd")
    bpy.ops.object.select_all(action='DESELECT')
    die_obj.select_set(True)
    bpy.context.view_layer.objects.active = die_obj
    bpy.ops.wm.usd_export(filepath=usd_path, selected_objects_only=True)

    thumb_path = os.path.join(outdir, f"{asset_id}_thumb.png")
    _render_thumbnail(die_obj, thumb_path, size_mm)

    manifest_record["usd_path"] = f"{asset_id}.usd"
    manifest_record["thumbnail_path"] = f"{asset_id}_thumb.png"
    manifest_path = os.path.join(outdir, f"{asset_id}.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest_record, f, indent=2)

    return manifest_path


def _render_thumbnail(die_obj, thumb_path, size_mm, resolution=512):
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.film_transparent = True

    cam_data = bpy.data.cameras.new(f"{die_obj.name}_cam")
    cam_obj = bpy.data.objects.new(f"{die_obj.name}_cam", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    dist = size_mm * 0.12
    cam_obj.location = (dist, -dist, dist)
    direction = die_obj.location - cam_obj.location
    cam_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    scene.camera = cam_obj

    light_data = bpy.data.lights.new(f"{die_obj.name}_light", type='SUN')
    light_data.energy = 3.0
    light_obj = bpy.data.objects.new(f"{die_obj.name}_light", light_data)
    light_obj.location = (dist, dist, dist * 1.5)
    bpy.context.collection.objects.link(light_obj)

    scene.render.filepath = thumb_path
    bpy.ops.render.render(write_still=True)

    bpy.data.objects.remove(cam_obj, do_unlink=True)
    bpy.data.objects.remove(light_obj, do_unlink=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/saps/projects/Dice-Detection && blender --background --python tests/blender/test_exporter.py; echo "exit=$?"`
Expected: prints `ALL TESTS PASSED` and `exit=0`

- [ ] **Step 5: Commit**

```bash
git add src/dice_gen/exporter.py tests/blender/test_exporter.py
git commit -m "feat: add USD export, manifest, and thumbnail rendering"
```

---

## Task 7: `orchestrator.py` — batch generation loop with error handling

**Files:**
- Create: `src/dice_gen/orchestrator.py`
- Create: `tests/blender/test_orchestrator.py`

**Interfaces:**
- Consumes: `sampler.sample_variant`, `geometry.build_die_base_mesh`, `geometry.compute_opposite_face_pairs`, `numbering.assign_values_to_opposite_pairs`, `numbering.verify_opposite_sum`, `numbering.get_values`, `glyphs.apply_engraved_glyphs`, `glyphs.apply_decal_glyphs`, `materials.build_material`, `materials.apply_material`, `materials.build_fill_material`, `exporter.export_asset`
- Produces: `orchestrator.generate_batch(count: int, seed: int, outdir: str) -> tuple[int, int]` (returns `(num_generated, num_failed)`; writes `manifest.json` and `failures.json` into `outdir`)

- [ ] **Step 1: Write the failing Blender test**

`tests/blender/test_orchestrator.py`:
```python
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from _harness import run_and_report


def test_generate_batch_produces_manifest_and_assets():
    from dice_gen import orchestrator

    with tempfile.TemporaryDirectory() as outdir:
        generated, failed = orchestrator.generate_batch(count=6, seed=1000, outdir=outdir)

        assert generated + failed == 6
        assert generated >= 1, "at least some assets should succeed"

        manifest_path = os.path.join(outdir, "manifest.json")
        assert os.path.exists(manifest_path)
        with open(manifest_path) as f:
            manifest = json.load(f)
        assert len(manifest) == generated

        for record in manifest:
            usd_path = os.path.join(outdir, record["usd_path"])
            thumb_path = os.path.join(outdir, record["thumbnail_path"])
            assert os.path.exists(usd_path)
            assert os.path.exists(thumb_path)
            assert record["die_type"] in ("d4", "d6", "d8", "d10", "d12", "d20")

        failures_path = os.path.join(outdir, "failures.json")
        assert os.path.exists(failures_path)


run_and_report(test_generate_batch_produces_manifest_and_assets)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/saps/projects/Dice-Detection && blender --background --python tests/blender/test_orchestrator.py; echo "exit=$?"`
Expected: `exit=1`, `ModuleNotFoundError: No module named 'dice_gen.orchestrator'`

- [ ] **Step 3: Implement `orchestrator.py`**

`src/dice_gen/orchestrator.py`:
```python
import json
import os
import traceback

from . import exporter, geometry, glyphs, materials, numbering, sampler


def generate_batch(count, seed, outdir):
    os.makedirs(outdir, exist_ok=True)
    master_manifest = []
    failures = []

    for i in range(count):
        variant_seed = seed + i
        asset_id = f"asset_{i:05d}"
        try:
            record = _generate_one(asset_id, variant_seed, outdir)
            master_manifest.append(record)
        except Exception as e:
            failures.append({
                "asset_id": asset_id,
                "seed": variant_seed,
                "error": str(e),
                "traceback": traceback.format_exc(),
            })

    with open(os.path.join(outdir, "manifest.json"), "w") as f:
        json.dump(master_manifest, f, indent=2)
    with open(os.path.join(outdir, "failures.json"), "w") as f:
        json.dump(failures, f, indent=2)

    return len(master_manifest), len(failures)


def _generate_one(asset_id, seed, outdir):
    import bpy

    params = sampler.sample_variant(seed)
    die_obj = geometry.build_die_base_mesh(params.die_type, params.size_mm)

    face_pairs = geometry.compute_opposite_face_pairs(die_obj)
    assignment = numbering.assign_values_to_opposite_pairs(params.die_type, face_pairs)
    if not numbering.verify_opposite_sum(params.die_type, face_pairs, assignment):
        raise ValueError(f"{asset_id}: numbering invariant failed for {params.die_type}")

    if params.glyph_method == "engraved":
        glyphs.apply_engraved_glyphs(
            die_obj, params.die_type, assignment, params.glyph_style,
            params.glyph_fill, params.font_or_style_id, params.size_mm,
        )
        mat = materials.build_material(die_obj.name, params.material_category, params.material_params)
        materials.apply_material(die_obj, mat, slot_index=0)
        if params.glyph_fill == "painted":
            fill_mat = materials.build_fill_material(die_obj.name, params.material_params)
            materials.apply_material(die_obj, fill_mat, slot_index=1)
    else:
        mat = materials.build_material(die_obj.name, params.material_category, params.material_params)
        materials.apply_material(die_obj, mat, slot_index=0)
        glyphs.apply_decal_glyphs(
            die_obj, params.die_type, assignment, params.glyph_style,
            params.font_or_style_id, params.size_mm, outdir,
        )

    manifest_record = {
        "asset_id": asset_id,
        "die_type": params.die_type,
        "num_sides": len(numbering.get_values(params.die_type)),
        "size_mm": params.size_mm,
        "bevel_fraction": params.bevel_fraction,
        "numbering_scheme": params.numbering_scheme,
        "glyph_style": params.glyph_style,
        "glyph_method": params.glyph_method,
        "glyph_fill": params.glyph_fill,
        "font_or_style_id": params.font_or_style_id,
        "material_category": params.material_category,
        "material_params": params.material_params,
        "seed": seed,
    }

    exporter.export_asset(die_obj, manifest_record, outdir, params.bevel_fraction, params.size_mm)

    bpy.data.objects.remove(die_obj, do_unlink=True)
    return manifest_record
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/saps/projects/Dice-Detection && blender --background --python tests/blender/test_orchestrator.py; echo "exit=$?"`
Expected: prints `ALL TESTS PASSED` and `exit=0`. Inspect `failures.json` content printed by the
test's temp dir if `generated < 6` — some randomly sampled combinations may need constant
tuning in `glyphs.py`/`geometry.py` (this is expected iteration, not a sign the design is wrong).

- [ ] **Step 5: Commit**

```bash
git add src/dice_gen/orchestrator.py tests/blender/test_orchestrator.py
git commit -m "feat: add batch generation orchestrator with per-asset error handling"
```

---

## Task 8: `scripts/generate_dice_assets.py` — CLI entry point

**Files:**
- Create: `scripts/generate_dice_assets.py`
- Modify: nothing (README already documents `scripts/` as the CLI location)

**Interfaces:**
- Consumes: `orchestrator.generate_batch`
- Produces: a runnable CLI invoked as `blender --background --python scripts/generate_dice_assets.py -- --count N --seed S --outdir DIR`

- [ ] **Step 1: Implement the CLI script**

There's no separate unit test for this step — it's a thin argument-parsing wrapper validated
by Task 10's end-to-end run. Write it directly:

`scripts/generate_dice_assets.py`:
```python
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dice_gen import orchestrator


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []

    parser = argparse.ArgumentParser(description="Generate a library of dice USD assets.")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="data/raw/dice_assets")
    args = parser.parse_args(argv)

    generated, failed = orchestrator.generate_batch(args.count, args.seed, args.outdir)
    print(f"Generated: {generated}, Failed: {failed}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the CLI with a tiny count**

Run:
```bash
cd /home/saps/projects/Dice-Detection
blender --background --python scripts/generate_dice_assets.py -- --count 3 --seed 7 --outdir /tmp/dice_cli_smoke
```
Expected: prints `Generated: 3, Failed: 0` (or a small nonzero failure count — acceptable at
this stage; investigate if `Generated: 0`). Verify: `ls /tmp/dice_cli_smoke` shows `manifest.json`, `failures.json`, and per-asset `.usd`/`.json`/`_thumb.png` files.

- [ ] **Step 3: Commit**

```bash
git add scripts/generate_dice_assets.py
git commit -m "feat: add CLI entry point for batch dice asset generation"
```

---

## Task 9: `scripts/validate_dice_assets.py` — standalone validation pass

**Files:**
- Create: `scripts/validate_dice_assets.py`
- Create: `tests/test_validate_dice_assets.py`

**Interfaces:**
- Consumes: `numbering.get_values` (Task 1); a `manifest.json` produced by `orchestrator.generate_batch`
- Produces: `validate_dice_assets.validate(outdir: str) -> list[str]` (list of error strings, empty if all checks pass); CLI usage `python3 scripts/validate_dice_assets.py <outdir>`

This script deliberately does NOT import `bpy` — it only inspects the manifest and files on
disk, so it can run standalone with plain `python3` (no Blender needed) for quick CI-style
checks.

- [ ] **Step 1: Write the failing test**

`tests/test_validate_dice_assets.py`:
```python
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from validate_dice_assets import validate


def _write_manifest(tmp_path, records):
    with open(os.path.join(tmp_path, "manifest.json"), "w") as f:
        json.dump(records, f)


def test_validate_reports_missing_usd_file(tmp_path):
    _write_manifest(tmp_path, [{
        "asset_id": "a1", "die_type": "d6", "num_sides": 6,
        "usd_path": "a1.usd", "thumbnail_path": "a1_thumb.png",
    }])
    open(os.path.join(tmp_path, "a1_thumb.png"), "w").close()

    errors = validate(str(tmp_path))
    assert any("missing USD" in e for e in errors)


def test_validate_reports_wrong_num_sides():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_path:
        _write_manifest(tmp_path, [{
            "asset_id": "a1", "die_type": "d6", "num_sides": 5,
            "usd_path": "a1.usd", "thumbnail_path": "a1_thumb.png",
        }])
        open(os.path.join(tmp_path, "a1.usd"), "w").write("x")
        open(os.path.join(tmp_path, "a1_thumb.png"), "w").close()

        errors = validate(tmp_path)
        assert any("num_sides" in e for e in errors)


def test_validate_passes_for_well_formed_manifest():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_path:
        _write_manifest(tmp_path, [{
            "asset_id": "a1", "die_type": "d20", "num_sides": 20,
            "usd_path": "a1.usd", "thumbnail_path": "a1_thumb.png",
        }])
        open(os.path.join(tmp_path, "a1.usd"), "w").write("x")
        open(os.path.join(tmp_path, "a1_thumb.png"), "w").close()

        errors = validate(tmp_path)
        assert errors == []
```

Note: `test_validate_reports_missing_usd_file` uses pytest's built-in `tmp_path` fixture
(no extra dependency); the other two use `tempfile.TemporaryDirectory` for variety/clarity —
either works.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/saps/projects/Dice-Detection && python3 -m pytest tests/test_validate_dice_assets.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'validate_dice_assets'`

- [ ] **Step 3: Implement `scripts/validate_dice_assets.py`**

```python
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dice_gen import numbering


def validate(outdir):
    manifest_path = os.path.join(outdir, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    errors = []
    for record in manifest:
        asset_id = record["asset_id"]
        usd_path = os.path.join(outdir, record["usd_path"])
        thumb_path = os.path.join(outdir, record["thumbnail_path"])

        if not os.path.exists(usd_path):
            errors.append(f"{asset_id}: missing USD file {usd_path}")
        elif os.path.getsize(usd_path) == 0:
            errors.append(f"{asset_id}: empty USD file {usd_path}")

        if not os.path.exists(thumb_path):
            errors.append(f"{asset_id}: missing thumbnail {thumb_path}")

        die_type = record["die_type"]
        expected_sides = len(numbering.get_values(die_type))
        if record["num_sides"] != expected_sides:
            errors.append(
                f"{asset_id}: num_sides {record['num_sides']} != expected "
                f"{expected_sides} for {die_type}"
            )

    return errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("outdir")
    args = parser.parse_args()

    found_errors = validate(args.outdir)
    print(f"Checked manifest at {args.outdir}: {len(found_errors)} error(s).")
    for e in found_errors:
        print(" -", e)
    sys.exit(1 if found_errors else 0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/saps/projects/Dice-Detection && python3 -m pytest tests/test_validate_dice_assets.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/validate_dice_assets.py tests/test_validate_dice_assets.py
git commit -m "feat: add standalone manifest/asset validation script"
```

---

## Task 10: End-to-end smoke test — generate and validate a real batch

**Files:** none created; this task exercises Tasks 1-9 together against the real filesystem.

**Interfaces:** none new.

- [ ] **Step 1: Run the full pure-Python test suite**

Run: `cd /home/saps/projects/Dice-Detection && python3 -m pytest tests/ -v --ignore=tests/blender`
Expected: all tests pass (numbering, sampler, validate_dice_assets)

- [ ] **Step 2: Run all Blender-dependent test scripts**

Run:
```bash
cd /home/saps/projects/Dice-Detection
for f in tests/blender/test_*.py; do
  echo "=== $f ==="
  blender --background --python "$f"
  echo "exit=$?"
done
```
Expected: every script prints `ALL TESTS PASSED` and `exit=0`

- [ ] **Step 3: Generate a real batch of 20 assets**

Run:
```bash
cd /home/saps/projects/Dice-Detection
mkdir -p data/raw
blender --background --python scripts/generate_dice_assets.py -- --count 20 --seed 42 --outdir data/raw/dice_assets
```
Expected: prints `Generated: N, Failed: M` with `N >= 15` (a handful of failures from edge-case
random parameter combinations is acceptable at this stage — inspect `data/raw/dice_assets/failures.json`
if `M` is large, and use the traceback per failed asset to tune the constants flagged as
tunable in Tasks 4/6, e.g. `ENGRAVE_DEPTH_FRACTION`, the fill-face-area threshold, or bevel width).

- [ ] **Step 4: Validate the generated batch**

Run: `cd /home/saps/projects/Dice-Detection && python3 scripts/validate_dice_assets.py data/raw/dice_assets`
Expected: `Checked manifest at data/raw/dice_assets: 0 error(s).`

- [ ] **Step 5: Visually spot-check a few thumbnails**

Run: `ls data/raw/dice_assets/*_thumb.png | head -5`
Open a couple of these PNGs (e.g. via the Read tool, since they're images) to visually confirm
dice look like recognizable d-something shapes with visible numerals/pips and varied materials.
This is the human visual-QA step called for in the design spec — no automated check replaces it.

- [ ] **Step 6: Add `data/raw/` to `.gitignore` if not already excluded, and commit any code fixes made during this task**

Run: `cd /home/saps/projects/Dice-Detection && git status`
If `data/raw/dice_assets/` shows as untracked and there's no `.gitignore` entry for `data/raw/`,
add one:

```bash
echo "data/raw/" >> .gitignore
```

Then, if any constants were tuned in Tasks 4/6/7 while working through this task's failures:
```bash
git add -A
git commit -m "fix: tune generation constants found during end-to-end smoke test"
```
(Skip the commit if nothing needed tuning.)

---

## Self-Review Notes

- **Spec coverage:** geometry (Task 3), numbering (Task 1), materials (Task 5), glyphs/numeral
  methods (Task 4), USD+manifest+thumbnail export (Task 6), randomized seeded sampling (Task 2),
  batch orchestration + failures.json (Task 7), CLI (Task 8), validation script (Task 9), and
  the human visual spot-check (Task 10) all map directly to spec sections. Isaac Lab is
  explicitly out of scope per the spec's Non-goals.
- **Placeholder scan:** no TBD/TODO; every step has complete, empirically-verified-where-risky
  code. Two spots are explicitly flagged as heuristics with named tunable constants
  (`ENGRAVE_DEPTH_FRACTION`, fill-face-area threshold in `glyphs.py`) rather than left as vague
  "add validation" placeholders — these are legitimate tuning knobs, not missing logic.
- **Type/interface consistency:** checked `DiceVariantParams` field names match what
  `orchestrator._generate_one` reads; `assignment: dict[int, int]` produced by
  `numbering.assign_values_to_opposite_pairs` matches what `glyphs.apply_engraved_glyphs` /
  `apply_decal_glyphs` iterate over; `geometry.compute_opposite_face_pairs` output type matches
  what `numbering.assign_values_to_opposite_pairs` expects as `face_pairs`.
