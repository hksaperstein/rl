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
    "d10_pct": {
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

    # Mark every structural edge of the pristine polyhedron before any
    # engraving cut ever runs -- see exporter.export_asset's Bevel modifier
    # (limit_method='WEIGHT') for why this must happen here rather than
    # right before bevel: boolean DIFFERENCE cuts don't rebuild untouched
    # edges away from the cut, so this weight survives every cut intact,
    # letting the eventual bevel round only the die's real structural
    # edges and never the many similarly-steep-angled edges an engraved
    # numeral's recess introduces.
    bevel_layer = bm.edges.layers.float.new('bevel_weight_edge')
    for e in bm.edges:
        e[bevel_layer] = 1.0

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


def compute_face_poles(obj, die_type):
    """
    d8, d10, and d10_pct are all built (see DIE_SPECS / _d10_base_vertices)
    as bipyramids: exactly two pole vertices at the extremal local-Z
    positions, plus a ring of equatorial vertices. d10_pct shares d10's
    exact mesh (same base_vertices) -- only the face labels differ, so it
    has the identical pole structure. Every face touches exactly one pole
    (confirmed empirically this session via direct vertex-index inspection
    on both die types). Real dice orient each face's numeral relative to
    its OWN pole, not one global up-vector -- see glyphs.py's
    _tangent_bitangent for the orientation fix this enables, and
    numbering.py's assign_values_to_opposite_pairs for the matching
    hemisphere-consistent value assignment.

    Returns None for die types without this two-pole structure (d4, d6,
    d12, d20) -- those keep their existing single-global-up-vector
    orientation convention unchanged.
    """
    if die_type not in ("d8", "d10", "d10_pct"):
        return None

    mesh = obj.data
    top_idx = max(range(len(mesh.vertices)), key=lambda i: mesh.vertices[i].co.z)
    bottom_idx = min(range(len(mesh.vertices)), key=lambda i: mesh.vertices[i].co.z)
    top_co = obj.matrix_world @ mesh.vertices[top_idx].co
    bottom_co = obj.matrix_world @ mesh.vertices[bottom_idx].co

    poles = {}
    for face in mesh.polygons:
        verts = set(face.vertices)
        if top_idx in verts:
            poles[face.index] = top_co
        elif bottom_idx in verts:
            poles[face.index] = bottom_co
        else:
            raise GeometryBuildError(
                f"{die_type} face {face.index} touches neither pole vertex "
                f"-- the two-pole bipyramid assumption compute_face_poles "
                f"relies on doesn't hold for this mesh"
            )
    return poles


def compute_face_inradius(mesh, face, obj_matrix):
    """
    World-space distance from face's centroid to the nearest of its
    edges (treated as infinite lines -- for a point at a convex polygon's
    centroid, this equals the true minimum distance to the polygon
    boundary). Used by glyphs.py to size engraved/decal numerals
    proportionally to each die type's actual face size, instead of one
    fixed fraction of the die's overall size_mm -- confirmed this
    session that face inradius varies nearly 2.5x across die types at
    the same size_mm (3.674mm for d8 vs 9.0mm for d6, at size_mm=18.0),
    so a single fixed fraction cannot be well-proportioned for every die
    type at once.
    """
    center = obj_matrix @ face.center
    verts_world = [obj_matrix @ mesh.vertices[i].co for i in face.vertices]
    n = len(verts_world)
    min_dist = None
    for i in range(n):
        a = verts_world[i]
        b = verts_world[(i + 1) % n]
        edge_dir = (b - a).normalized()
        proj = (center - a).dot(edge_dir)
        closest = a + edge_dir * proj
        dist = (center - closest).length
        if min_dist is None or dist < min_dist:
            min_dist = dist
    return min_dist
