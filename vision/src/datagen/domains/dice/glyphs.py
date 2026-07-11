"""
Applies numerals/pips to a die's faces, either as real engraved geometry
(boolean-cut into the mesh) or as printed texture decals (one material +
baked image per face). Both are real manufacturing conventions for TTRPG
dice, so both are supported and randomly chosen per asset by sampler.py.
"""
import math
import os

import bpy
import bmesh
import numpy as np
from mathutils import Vector, Matrix

from .geometry import compute_face_poles, compute_face_inradius
from .numbering import d4_vertex_values

# TRUE penetration depth of the engraving below the die surface, as a
# fraction of die size: 0.003 = 0.04-0.07mm across the sampled size range
# ("a fraction of a fraction of a fraction of a fingernail" -- direct
# user spec; a fingernail is ~0.4mm). Note the semantic fix: this
# previously set the cutter's extrude amount with the slab translated a
# full depth down, which (a) put the cutter's top face exactly COPLANAR
# with the die face -- a documented Blender boolean failure trigger
# (developer.blender.org T51389/T82736; the "hotlining" failure in
# community practice) that fragmented or silently dropped micro cuts --
# and (b) made the real recess depth 2x the named value. The cutter is
# now built thick (see ENGRAVE_CUTTER_HALF_THICKNESS_FRACTION) and
# overshoots well above the surface, penetrating exactly this deep:
# cut depth and cutter conditioning are decoupled, which is what makes
# micro depths reliable.
ENGRAVE_DEPTH_FRACTION = 0.003
# Half-thickness (Blender curve `extrude` is symmetric: E gives a 2E
# slab) of the engraving text cutter, as a fraction of die size. Chosen
# for solver robustness, NOT cut depth -- 0.01 matches the cutter scale
# of this pipeline's historically-reliable cuts. The cutter protrudes
# (2 * this - depth) above the face and only ENGRAVE_DEPTH_FRACTION
# below it.
ENGRAVE_CUTTER_HALF_THICKNESS_FRACTION = 0.01
# Em size of every d4 corner numeral, as a fraction of the face
# inradius -- CONSTANT per die, deliberately not routed through
# _proportional_font_size's character-count shrink: that shrink exists
# to fit long labels on small faces, but on a d4's huge face (inradius
# ~0.41 of die size) it made I, II, III and IV render at visibly
# different heights on the same die (user feedback, third round of d4
# sizing). Real dice keep numeral cap height constant and let width
# vary; wide labels are protected by the corner clearance clamp below
# instead. 0.85r em = ~0.6r cap height = ~17% of the edge length,
# in real vertex-read d4 territory.
D4_CORNER_EM_INRADIUS_FRACTION = 0.85
# How far along its median (face centroid -> vertex) each d4 corner copy
# sits; shared by the engraved and decal paths and by the corner
# clearance clamps, which depend on it geometrically.
D4_CORNER_INSET = 0.42
# 0.9 (raised from 0.5): at 0.5 an engraved single-character numeral's em
# size was half the face inradius (cap height ~0.35r) -- visually ~2x
# smaller relative to its face than the same label on the decal path,
# whose canvas calibration lands closer to real dice (numerals filling
# most of the inscribed circle). DECAL_FONT_CANVAS_SCALE below is
# rebalanced in lockstep so the decal path's output is bit-identical to
# before this change; only engraved glyphs grow.
FONT_INRADIUS_FRACTION = 0.9
FONT_EXTRA_CHAR_SHRINK = 0.35

# _render_label_to_image renders into a fixed, dimensionless orthographic
# camera setup (ortho_scale=1.4) shared by every die type -- it never sees
# real mm dimensions, since _unwrap_faces_to_full_square already normalizes
# each face's own UV island to fill the same square regardless of the
# face's real size. So face-shape proportionality here comes from
# inradius/size_mm (a dimensionless per-die-type ratio -- 0.5 for d6, ~0.2
# for d8/d10, matching the same shape variation _proportional_font_size
# corrects for in the engraved path), not from inradius alone. This
# constant rescales that ratio into this function's local text-size units.
# Calibrated so a single-character label on a d6 face (inradius/size_mm ==
# 0.5, the largest ratio of any die type) reproduces this path's old fixed
# default of 1.0 exactly (0.5 * FONT_INRADIUS_FRACTION * SCALE == 1.0),
# anchoring the new proportional sizing to the one old value most likely to
# have already been visually reasonable, while every smaller-faced die type
# and every longer label now correctly renders smaller than that anchor
# instead of at the same fixed size. Verified by rendering
# test_render_label_to_image_renders_three_corner_copies_for_d4 with real
# d4 geometry and inspecting the resulting ink regions (see that test).
# Rebalanced 4.0 -> 20/9 when FONT_INRADIUS_FRACTION rose 0.5 -> 0.9 (an
# engraved-path-only size correction), keeping this path's effective size
# exactly where it was: 0.5 * 0.9 * (20/9) == 1.0, same anchor as before.
DECAL_FONT_CANVAS_SCALE = 20.0 / 9.0

FONT_FILES = {
    "font_sans_bold": "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "font_serif_regular": "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "font_display_condensed": "/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Bold.ttf",
}


def _load_font(font_id, glyph_style):
    """
    Maps a sampled font_or_style_id to a real, distinct installed font,
    loaded once and reused (bpy.data.fonts.load creates a new VectorFont
    datablock each call unless one with a matching filepath is already
    loaded, so this checks first to avoid redundant loads across a
    batch).

    Returns None for glyph_style == "cjk_numerals" regardless of
    font_id -- confirmed empirically that none of FONT_FILES' fonts have
    CJK glyph coverage (rendering a CJK character with Liberation Sans
    Bold produces an empty placeholder rectangle, not the correct
    character), while Blender's own default bundled font already renders
    CJK correctly. Returning None means the caller leaves
    txt_obj.data.font unset, i.e. Blender's default font.
    """
    if glyph_style == "cjk_numerals":
        return None
    font_path = FONT_FILES.get(font_id)
    if font_path is None:
        return None
    for font in bpy.data.fonts:
        if font.filepath == font_path:
            return font
    return bpy.data.fonts.load(font_path)


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

# Real Milesian (Ionic) Greek numerals -- the actual historical system, not
# an invented approximation (the previous dict here was shifted by one from
# 6 upward because it skipped digamma/stigma, and used Omega for a zero the
# Greek system never had). 6 is properly the archaic letter stigma (a
# ligature of sigma-tau); the ΣΤ digraph used below is the standard modern
# typographic substitute, chosen over the literal stigma codepoint (U+03DA)
# because LiberationSansNarrow-Bold has no stigma glyph (renders a
# placeholder rectangle -- verified empirically in this project's Blender,
# 8-vert placeholder vs a real 172-vert outline in LiberationSans-Bold) and
# fonts are sampled independently of glyph style. There is NO zero: the
# system predates it, which is why d10 (values 0-9) must never sample
# greek_numerals (enforced in sampler.py, same mechanism as d10_pct's
# arabic-only restriction).
GREEK_NUMERALS = {
    1: "Α", 2: "Β", 3: "Γ", 4: "Δ", 5: "Ε",
    6: "ΣΤ", 7: "Ζ", 8: "Η", 9: "Θ", 10: "Ι",
    11: "ΙΑ", 12: "ΙΒ", 13: "ΙΓ", 14: "ΙΔ", 15: "ΙΕ",
    16: "ΙΣΤ", 17: "ΙΖ", 18: "ΙΗ", 19: "ΙΘ", 20: "Κ",
}

CJK_NUMERALS = {
    0: "零", 1: "一", 2: "二", 3: "三", 4: "四", 5: "五",
    6: "六", 7: "七", 8: "八", 9: "九", 10: "十",
    11: "十一", 12: "十二", 13: "十三", 14: "十四",
    15: "十五", 16: "十六", 17: "十七", 18: "十八",
    19: "十九", 20: "二十",
}


def glyph_label(value, glyph_style, die_type=None):
    if glyph_style == "arabic_numerals":
        if die_type == "d10_pct":
            return f"{value:02d}"
        return str(value)
    if glyph_style == "roman_numerals":
        return ROMAN_NUMERALS.get(value, str(value))
    if glyph_style == "greek_numerals":
        return GREEK_NUMERALS.get(value, str(value))
    if glyph_style == "cjk_numerals":
        return CJK_NUMERALS.get(value, str(value))
    raise ValueError(f"glyph_label not applicable to style {glyph_style!r}")


# Hard cap on a label's total rendered width as a multiple of its face's
# inradius. The per-character shrink in _proportional_font_size assumes
# roughly digit-width characters; full-width CJK labels ("十九") are ~3x
# wider per character, and at the shrink formula's size their strokes
# reached and crossed face edges on d20s (confirmed visually on a real
# batch). 1.4 keeps even wide labels inside the face's visual center
# region on the tightest face shape (triangle) while leaving ordinary
# numerals untouched -- the cap only binds when measured width demands it.
MAX_LABEL_WIDTH_INRADIUS_FRACTION = 1.4

_LABEL_WIDTH_CACHE = {}


def _label_width_per_em(label, font_id, glyph_style):
    """
    Measures the label's real rendered width, in em units (width at
    txt.data.size == 1.0), for the actual font that will draw it --
    including the CJK default-font exception in _load_font. Cached per
    (label, font, style): a batch re-renders the same labels constantly.
    """
    key = (label, font_id, glyph_style)
    if key in _LABEL_WIDTH_CACHE:
        return _LABEL_WIDTH_CACHE[key]

    bpy.ops.object.text_add()
    txt_obj = bpy.context.active_object
    txt_obj.data.body = label
    font = _load_font(font_id, glyph_style)
    if font is not None:
        txt_obj.data.font = font
    txt_obj.data.size = 1.0
    bpy.context.view_layer.update()
    width = float(txt_obj.dimensions.x)
    txt_data = txt_obj.data
    bpy.data.objects.remove(txt_obj, do_unlink=True)
    bpy.data.curves.remove(txt_data)

    _LABEL_WIDTH_CACHE[key] = width
    return width


def _proportional_font_size(inradius, label, width_per_em=None):
    """
    Calibrated against the real worst cases across every die type/glyph
    style combination -- see
    test_proportional_font_size_shrinks_for_longer_labels. Longer labels
    (e.g. d20's 2-digit arabic numerals, or "XVIII" for roman numeral 18)
    need a smaller per-character size to occupy roughly the same total
    footprint as a single-character label at the same font size would.

    width_per_em (from _label_width_per_em) additionally clamps the size
    so the label's real measured width never exceeds
    MAX_LABEL_WIDTH_INRADIUS_FRACTION of the face inradius -- the
    character-count shrink alone underestimates full-width CJK labels
    badly enough for strokes to cross face edges.
    """
    n = len(label)
    size = inradius * FONT_INRADIUS_FRACTION / (1 + (n - 1) * FONT_EXTRA_CHAR_SHRINK)
    if width_per_em is not None and width_per_em > 0:
        size = min(size, MAX_LABEL_WIDTH_INRADIUS_FRACTION * inradius / width_per_em)
    return size


def _tangent_bitangent(normal, up_reference=None, threshold=0.999):
    """
    Given a (normalized) face normal, returns a consistent (tangent,
    bitangent) in-plane basis by projecting an "up" reference direction
    onto the face's plane.

    When up_reference is given (d8/d10's hemisphere-aware orientation --
    see _face_orientation_matrix), THAT vector is projected instead of
    the global hint below -- this is what lets each face orient toward
    its OWN pole vertex rather than one direction shared by the whole
    die, which is what real d8/d10 dice do (confirmed empirically: the
    single-global-vector convention is structurally incapable of
    producing the mirrored-hemisphere pattern real dice show, since it's
    a smooth function of the normal alone with no reflection anywhere).

    Global +Z is used as the up reference for every OTHER face (d4, d6,
    d12, d20, and any d8/d10 caller that doesn't pass up_reference)
    EXCEPT when normal is itself (near-)parallel to +/-Z, where the
    projection is undefined (up_hint.cross(normal) would be the zero
    vector) -- global +Y is used instead for that narrow case only.

    The threshold (normal.z's absolute value) for switching to the Y
    fallback must stay very close to 1.0. An earlier version used 0.9,
    which also caught merely-steep-but-not-vertical faces (e.g. a d20's
    near-pole ring, normal.z ~= +/-0.9342): those faces got the flat
    (0, 1, 0) fallback while their immediate neighbors (normal.z ~= +/-0.577)
    got the smoothly-varying Z-projection, producing an abrupt rotation
    jump between adjacent faces -- confirmed both numerically (dumping
    every d20 face's computed bitangent) and visually (one face's engraved
    numeral reading upright, the adjacent face's numeral at a distinctly
    different angle). 0.999 only catches genuinely axis-aligned normals
    (e.g. d6/d8's exactly-vertical top/bottom faces), where the fallback
    is actually required to avoid a degenerate zero-length tangent.

    Shared by _face_orientation_matrix (engraved cutter placement,
    world-space normal) and _unwrap_faces_to_full_square (decal UV
    unwrap, local-space normal) so both glyph methods use one consistent
    orientation convention instead of two independently-behaving ones.
    """
    if up_reference is None:
        up_reference = Vector((0, 0, 1)) if abs(normal.z) < threshold else Vector((0, 1, 0))
    tangent = up_reference.cross(normal).normalized()
    bitangent = normal.cross(tangent).normalized()
    return tangent, bitangent


def _snap_up_to_face_vertex(up_reference, normal, vertex_dirs, num_verts):
    """
    Real dice align each numeral's "up" with one of its face's own
    vertices (a d20/d8 triangle's numeral points at the apex, a d10
    kite's at its pole corner, a d12 pentagon's at a corner) -- EXCEPT
    quad faces (d6), whose numerals read edge-aligned; snapping a d6
    numeral to a corner would rotate it 45 degrees off every real die.

    Takes the smoothly-varying projected up hint (global-Z projection or
    pole direction) and snaps it to whichever in-plane center-to-vertex
    direction it is most aligned with, making the convention exact
    instead of approximate. For d8/d10 the pole IS a vertex of every
    face, so this is a no-op-to-exactifying change there; for d4/d12/d20
    it turns "roughly toward global up" into "exactly at a vertex," the
    real-world convention.

    vertex_dirs: normalized in-plane (already normal-projected) unit
    vectors from the face center to each of its vertices, in the same
    space (world or local) as up_reference/normal.
    """
    if num_verts == 4 or up_reference is None:
        return up_reference
    return max(vertex_dirs, key=lambda d: d.dot(up_reference))


def _face_orientation_matrix(face, obj_matrix, pole_world_co=None):
    center = obj_matrix @ face.center
    normal = (obj_matrix.to_3x3() @ face.normal).normalized()
    if pole_world_co is not None:
        to_pole = pole_world_co - center
        up_reference = (to_pole - to_pole.dot(normal) * normal).normalized()
    else:
        # Reproduce _tangent_bitangent's default up hint (global Z, or Y
        # for genuinely-vertical faces) explicitly, so it too can be
        # vertex-snapped below instead of only pole-based references.
        up_hint = Vector((0, 0, 1)) if abs(normal.z) < 0.999 else Vector((0, 1, 0))
        up_reference = (up_hint - up_hint.dot(normal) * normal).normalized()

    mesh = face.id_data
    vertex_dirs = []
    for vi in face.vertices:
        d = (obj_matrix @ mesh.vertices[vi].co) - center
        d = (d - d.dot(normal) * normal).normalized()
        vertex_dirs.append(d)
    up_reference = _snap_up_to_face_vertex(
        up_reference, normal, vertex_dirs, len(face.vertices)
    )

    tangent, bitangent = _tangent_bitangent(normal, up_reference=up_reference)
    rot = Matrix((tangent, bitangent, normal)).transposed().to_4x4()
    rot.translation = center
    return rot


def _weld_cutter_mesh(obj):
    """
    bpy.ops.object.convert(target='MESH') on an extruded text curve produces
    a mesh with duplicate, unwelded vertices at every seam between the front
    cap, back cap, and extrusion walls (confirmed empirically across every
    glyph style: Latin, Roman, Greek, digits). Feeding this un-welded,
    non-watertight cutter into _boolean_diff_apply's EXACT boolean solver
    usually gets away with it, but can occasionally corrupt the target mesh
    catastrophically (e.g. causing polygon count to drop instead of grow
    after a cut). Welding duplicate vertices and recomputing normals before
    the boolean fixes this. The pip-sphere cutter branch doesn't need this:
    UV spheres from primitive_uv_sphere_add are already watertight/manifold.
    """
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(obj.data)
    obj.data.update()
    bm.free()


def _connected_components(bm):
    """
    Splits a bmesh into its connected shells. Each shell is reported as a
    dict with:
      - "faces": the BMFace list making up the shell
      - "has_boundary": True if any edge in the shell is non-manifold (i.e.
        this is an open surface, not a closed solid) -- e.g. a recessed cut
        surface that got stitched from an unwelded-but-coincident seam
        (confirmed harmless on asset_00091, see _boolean_diff_apply)
      - "bbox_diag_sq": squared world-space bounding-box diagonal of the
        shell's vertices -- large for the die's own body (spans close to the
        die's full envelope), tiny for un-subtracted glyph-cutter debris
        (a few mm, matching individual glyph size) regardless of face count

    Shared by _non_body_closed_component_count (detects un-subtracted cutter
    debris after a boolean cut) and the end-of-loop debris-discarding
    backstop in apply_engraved_glyphs (picks which closed shell is the real
    body to keep).
    """
    bm.faces.ensure_lookup_table()
    visited = set()
    components = []
    for f in bm.faces:
        if f.index in visited:
            continue
        stack = [f]
        comp_faces = []
        has_boundary = False
        while stack:
            cur = stack.pop()
            if cur.index in visited:
                continue
            visited.add(cur.index)
            comp_faces.append(cur)
            for e in cur.edges:
                if not e.is_manifold:
                    has_boundary = True
                for lf in e.link_faces:
                    if lf.index not in visited:
                        stack.append(lf)
        verts = {v for fc in comp_faces for v in fc.verts}
        xs = [v.co.x for v in verts]
        ys = [v.co.y for v in verts]
        zs = [v.co.z for v in verts]
        diag_sq = (max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2 + (max(zs) - min(zs)) ** 2
        components.append({
            "faces": comp_faces,
            "has_boundary": has_boundary,
            "bbox_diag_sq": diag_sq,
        })
    return components


def _non_body_closed_component_count(bm):
    """
    Number of closed (zero-boundary-edge) shells in the mesh, EXCLUDING
    whichever single component has the largest bounding-box diagonal (assumed
    to be the die's own body). This exclusion matters because, empirically,
    across every asset inspected during this investigation, the die's own
    body is essentially never itself fully closed after a cut -- even
    accepted-correct results like asset_00091 have boundary edges on the
    body/recessed-surface component. So counting ALL closed shells and
    expecting the body to be one of them (an earlier version of this check)
    doesn't work: a single un-subtracted debris blob is itself closed, and
    with the (open) body excluded from consideration entirely, that's
    "1 closed shell" whether zero or one debris blobs are present --
    indistinguishable, and the check can never fire. Excluding the
    largest-bbox-diagonal component (whatever its own manifoldness) up front
    fixes this: any closed shell that remains after that exclusion is, by
    definition, not the body, so it can only be un-subtracted glyph-cutter
    debris.
    """
    components = _connected_components(bm)
    if not components:
        return 0
    body = max(components, key=lambda c: c["bbox_diag_sq"])
    return sum(1 for c in components if c is not body and not c["has_boundary"])


def _discard_non_body_closed_debris(die_obj):
    """
    Final backstop, run ONCE after the entire cut loop in apply_engraved_glyphs
    finishes (not per-cut -- no need to re-scan/delete mid-loop when every cut
    already got its own EXACT->FLOAT retry chance via _boolean_diff_apply).
    Even that per-cut retry is not guaranteed to fully merge every glyph
    cutter into the die on every degenerate input -- four rounds of
    progressively subtler EXACT-solver pathologies have been found
    empirically on this codebase (afb1af5's Alpha-glyph volume collapse,
    cd7b268's total silent no-op, the debris-outweighs-body face-count
    masking, and the closed-component-count/absolute-threshold blind spot --
    see _boolean_diff_apply and _non_body_closed_component_count for the
    full history). Rather than assume this is finally the last one,
    guarantee the SHIPPED asset is always clean: delete any remaining closed
    shell other than the largest-bbox-diagonal one (the real body; any other
    OPEN pieces, e.g. asset_00091-style harmless seam splits, are left
    alone), so no exported asset ever contains stray un-subtracted cutter
    geometry. This can leave a single numeral missing from one face in the
    rare worst case, which is a far smaller defect than shipping floating
    garbage polygons in training data.

    Returns the warning message (also printed, as before) if debris was
    found and discarded, or None if the die was already clean -- callers
    (apply_engraved_glyphs) collect this into the asset's manifest record so
    a batch-level validation pass can flag it without depending on anyone
    reading the console output of the Blender generation run.
    """
    bm = bmesh.new()
    bm.from_mesh(die_obj.data)
    components = _connected_components(bm)

    warning = None
    if components:
        body = max(components, key=lambda c: c["bbox_diag_sq"])
        debris = [c for c in components if c is not body and not c["has_boundary"]]

        if debris:
            debris_face_count = sum(len(c["faces"]) for c in debris)
            for extra in debris:
                bmesh.ops.delete(bm, geom=extra["faces"], context='FACES')
            bm.to_mesh(die_obj.data)
            die_obj.data.update()
            warning = (
                f"{die_obj.name}: {len(debris)} un-subtracted closed debris "
                f"shell(s) ({debris_face_count} faces total) survived every "
                f"per-cut EXACT->FLOAT retry and were discarded at the end "
                f"of the cut loop to keep the shipped asset clean; this "
                f"likely means at least one numeral/pip cut failed to "
                f"engrave on this die."
            )
            print(f"WARNING: {warning}")

    bm.free()
    return warning


def _boolean_diff_apply(die_obj, cutter_obj):
    """
    Even after _weld_cutter_mesh, some glyphs still tessellate to a cutter
    that is slightly non-manifold -- confirmed for the Greek capital Alpha
    ("Α") glyph, which is a genuine self-overlap baked into Blender's built-in
    Bfont's own curve fill for that outline, not just an unwelded seam. On
    asset_00091 (d10, greek_numerals, size_mm=16.12691595326456) this residual
    non-manifoldness is harmless for 9 of 10 cuts but makes the EXACT boolean
    solver collapse the die outright on the "Α" cut: a single small glyph
    incision dropped the die's volume from 970.03 to 2.984 in one
    modifier_apply. The FLOAT solver is more tolerant of this class of
    degenerate cutter and produces the expected tiny volume change (970.03 ->
    970.13) on the identical cut, but EXACT remains preferred generally (it's
    what the other 9 well-behaved cuts on this same die use without issue),
    so we only fall back to FLOAT when EXACT's result looks like a collapse.

    A second, distinct failure mode was found on asset_00026 (d20,
    arabic_numerals, size_mm=19.73093050365471): EXACT silently no-op'd on
    every single one of its 20 numeral cuts -- the die's own body came out of
    each modifier_apply byte-for-byte untouched, with the cutter's mesh
    merely appended alongside it as an inert, un-subtracted floating solid.
    Because nothing was actually subtracted, the die's volume barely changed
    per cut, so this never tripped the volume-collapse check above; the die
    just silently ended up completely unengraved.

    A third failure mode was found on 4 assets in a 100-asset batch
    (asset_00079, asset_00095, asset_00098, asset_00004; all d4/
    arabic_numerals): an earlier fix for the asset_00026 no-op tracked
    whichever connected shell had the most faces, on the theory that a real
    cut always grows the body it touches. This is fooled on low-poly dice (a
    d4 starts at only 4 faces, a d6 at 6, etc.) early in a multi-cut
    sequence: if EXACT silently no-ops a cut, the untouched cutter mesh --
    which can easily have hundreds of faces for a single glyph -- gets
    appended as a new component with MORE faces than the real body has
    accumulated so far, so "the largest component grew" looked true even
    though it was debris masquerading as growth, not real engraving. Worse,
    even after correctly identifying the body by world-space bounding-box
    diagonal instead of raw face count (debris is always physically tiny --
    a few mm, matching individual glyph size -- while the real body's
    bounding box always spans close to the die's full envelope), a bare
    body-face-count-growth check still isn't sufficient: on asset_00079,
    EXACT left 3 of 4 numeral cuts as fully disconnected debris blobs (263,
    437, and 45 faces) while incidentally grazing the real body for a single
    stray face each time (13->14->15->16), so "the body grew" remained
    trivially true even though the numerals were never actually engraved.

    A fourth iteration replaced the growth heuristic with a structural check
    instead: a genuinely successful cut merges the cutter into the die,
    leaving at most one closed (fully watertight, zero-boundary-edge) shell
    in the result, so a count of closed shells greater than one meant
    un-subtracted debris. This was ALSO insufficient, for a subtler reason:
    empirically, the die's own body is essentially never itself fully closed
    after a cut (confirmed even in the accepted-correct asset_00091 result),
    so a lone debris blob (which IS closed) coexisting with the (open) body
    always presented as exactly "1 closed shell" -- bitwise indistinguishable
    from the "0 debris" case, since the body was never being counted as the
    permitted one anyway. The absolute-count check could therefore only ever
    fire when 2+ debris blobs coexisted simultaneously, and even then it only
    retried the *current* cut, letting older debris (like asset_00079's
    263-face "2" blob) ride through every subsequent snapshot/retry cycle
    untouched.

    The fix that actually closes this gap asks a per-cut DELTA question
    instead of an absolute count: did *this specific cut* create a new
    closed shell that wasn't there before it ran, excluding whichever single
    component has the largest bounding-box diagonal (assumed to be the die's
    own body, regardless of whether that component itself is open or
    closed -- see _non_body_closed_component_count)? This is well-defined
    regardless of how much old debris already exists, correctly fires on the
    exact cut that produced new debris, and doesn't get confused by (or
    endlessly re-retry because of) debris already carried over from an
    earlier cut. It also doesn't false-trigger on asset_00091-style harmless
    open-boundary splits, since those never count as closed on either side of
    the comparison.

    A fifth failure mode was found via direct interactive/visual inspection
    (MatCap+Cavity viewport shading, which reveals recess geometry regardless
    of material/lighting) on asset_00006 (d12, greek_numerals) and
    asset_00042 (d20, cjk_numerals): the Greek "Δ"/"Ζ" and CJK "五"/"九"
    glyphs left their faces completely unengraved, yet neither existing
    check fired -- no collapse, no debris, volume barely changed (e.g. "Δ":
    10365.645047 -> 10365.638003, a -0.007 change against a 2.42-volume
    cutter). A natural-seeming fix -- retry with FLOAT whenever a cut removes
    under ~10% of its own cutter's volume -- does correctly catch both of
    these specific cases (FLOAT removes a normal amount, -4.16/-2.09, on
    retry). It was NOT shipped, though: broadly testing it (100-asset real
    batch regen) showed a 67% per-asset trigger rate, vastly higher than the
    ~1.8% baseline this file's other checks produce, meaning it was flagging
    (and needlessly retrying/skipping) a huge number of perfectly good cuts,
    not just the rare true no-ops. Root cause: "cutter volume" and "post-cut
    die volume" are only meaningful with *consistently outward-facing*
    normals, but recalculating normals on an isolated, often still slightly
    non-manifold single-glyph cutter mesh (see _weld_cutter_mesh -- welding
    doesn't fully fix every glyph) has no reliable way to know which overall
    direction is "outward" for an isolated shape with no surrounding
    reference, so the sign of a bare calc_volume() reading after
    bmesh.ops.recalc_face_normals is frequently flipped for cutters, making
    "did this cut remove a plausible fraction of its cutter's volume" a much
    noisier signal across the general population than it first appeared from
    the two known-bad examples alone. This class of check is NOT currently
    implemented here as a result -- the "Δ"/"Ζ" silent-near-zero-no-op
    failure mode (and a related one found alongside it, where a cut on the
    same die produces a volume *increase* instead of a decrease -- "Γ",
    value=3, delta -1.10 against a 1.42-volume cutter under EXACT, and WORSE
    under FLOAT, -2.66) remain known, unresolved gaps: neither is caught by
    any check below, and a future fix needs a way to validate a cut's effect
    that doesn't depend on trusting an isolated cutter mesh's own volume
    sign.

    Returns None if the cut succeeded (on EXACT or after a FLOAT retry), or
    the warning message (also printed, as before) if both solvers failed and
    the cut was skipped -- collected by apply_engraved_glyphs into the
    asset's manifest record.
    """
    bm_before = bmesh.new()
    bm_before.from_mesh(die_obj.data)
    volume_before = bm_before.calc_volume()
    debris_before = _non_body_closed_component_count(bm_before)

    def _apply_and_measure(solver):
        mod = die_obj.modifiers.new(name="Engrave", type='BOOLEAN')
        mod.operation = 'DIFFERENCE'
        mod.object = cutter_obj
        mod.solver = solver
        bpy.context.view_layer.objects.active = die_obj
        bpy.ops.object.modifier_apply(modifier=mod.name)

        bm = bmesh.new()
        bm.from_mesh(die_obj.data)
        volume_result = bm.calc_volume()
        debris_result = _non_body_closed_component_count(bm)
        bm.free()

        bad = (
            (volume_before > 0 and volume_result < volume_before * 0.5)
            or (debris_result > debris_before)
        )
        return bad

    bad = _apply_and_measure('EXACT')

    warning = None

    if bad:
        bm_before.to_mesh(die_obj.data)
        die_obj.data.update()
        bad = _apply_and_measure('FLOAT')

        if bad:
            # Both solvers produced a collapsed or still-debris-laden result
            # -- give up on this specific cut rather than ship a broken
            # mesh. The die ends up exactly as it was before this glyph was
            # ever attempted; the missing numeral is the same class of rare,
            # tracked exception apply_engraved_glyphs' end-of-loop backstop
            # already accepts for the debris case.
            bm_before.to_mesh(die_obj.data)
            die_obj.data.update()
            warning = (
                f"{die_obj.name}: a glyph cut was skipped entirely -- both "
                f"EXACT and FLOAT solvers produced a collapsed or "
                f"debris-laden result for this cutter; this die is missing "
                f"one numeral/pip as a result."
            )
            print(f"WARNING: {warning}")

    bm_before.free()
    bpy.data.objects.remove(cutter_obj, do_unlink=True)
    return warning


def _face_vertex_orientations(mesh, face, obj_matrix, inset=D4_CORNER_INSET):
    """
    Returns one (vertex_index, orientation matrix) pair per vertex of
    `face`, for d4's vertex-read numeral convention: real vertex-read d4
    dice key values to VERTICES, not faces -- each face shows the values
    of its own 3 corners (3 DIFFERENT numbers per face), and all 3 faces
    meeting at a vertex show that vertex's value at their shared corner,
    so the rolled result reads at the apex. (An earlier version of this
    pipeline engraved the face's single assigned value at all 3 corners,
    which is not how real vertex-read d4s work -- corrected per direct
    user feedback.) The vertex_index in each returned pair is what lets
    the caller look up the right value per corner via
    numbering.d4_vertex_values.

    For each vertex, "up" (bitangent) is the direction from the face
    center toward that vertex, projected into the face plane -- i.e.
    each copy points radially outward toward its own corner, matching
    the 120-degree-apart rotational pattern real vertex-read d4s show.
    `inset` places each copy 42% of the way from the face center to the
    vertex, tuned together with D4_CORNER_EM_INRADIUS_FRACTION against
    real renders (0.55 clipped rotated 2-character labels against the
    face edge; smaller insets with small fonts read as corner-hugging
    around a dead center).
    """
    center = obj_matrix @ face.center
    normal = (obj_matrix.to_3x3() @ face.normal).normalized()

    orientations = []
    for vertex_index in face.vertices:
        vertex_world = obj_matrix @ mesh.vertices[vertex_index].co
        radial = vertex_world - center
        radial = (radial - radial.dot(normal) * normal).normalized()
        tangent = radial.cross(normal).normalized()
        bitangent = normal.cross(tangent).normalized()
        rot = Matrix((tangent, bitangent, normal)).transposed().to_4x4()
        rot.translation = center + (vertex_world - center) * inset
        orientations.append((vertex_index, rot))
    return orientations


def apply_engraved_glyphs(die_obj, die_type, assignment, glyph_style, glyph_fill, font_id, size_mm):
    depth = size_mm * ENGRAVE_DEPTH_FRACTION
    is_d4_vertex_numerals = die_type == "d4" and glyph_style != "pips"
    warnings = []

    # Phase 1: compute every cut's (value, orientation, font_size) against
    # the PRISTINE mesh, entirely before any boolean modifier is applied.
    # Each bpy.ops.object.modifier_apply call below rebuilds die_obj.data's
    # topology (reindexing/reordering polygons), so face_index values from
    # `assignment` (captured once upfront by geometry.compute_opposite_face_pairs)
    # must never be re-resolved against die_obj.data.polygons after a cut.
    # Real vertex-read d4 dice show 3 DIFFERENT numerals per face -- one
    # per corner, each corner showing its own VERTEX's value (see
    # numbering.d4_vertex_values / _face_vertex_orientations) -- rather
    # than the single centered numeral every other die type uses. This
    # branch must stay inside Phase 1 (computed entirely against the
    # pristine mesh): recomputing face.vertices/face.normal mid-loop,
    # after an earlier cut has already rebuilt the mesh topology, causes
    # a Blender crash.
    #
    # font_size is also computed here, per cut, rather than once for the
    # whole die: it depends on this face's own inradius (which varies
    # nearly 2.5x across die types at the same size_mm -- see
    # geometry.compute_face_inradius) AND this cut's own label length (see
    # _proportional_font_size), so one fixed size for the whole die can
    # never be well-proportioned for every face/label combination at once.
    # Computing it here, against the pristine mesh, and carrying it
    # through in the planned_cuts tuple (rather than carrying face_index
    # and recomputing in Phase 2) avoids re-indexing die_obj.data.polygons
    # after a cut has already rebuilt the mesh topology.
    face_poles = compute_face_poles(die_obj, die_type)
    vertex_values = d4_vertex_values(len(die_obj.data.vertices)) if is_d4_vertex_numerals else None
    planned_cuts = []
    for face_index, value in assignment.items():
        face = die_obj.data.polygons[face_index]
        if is_d4_vertex_numerals:
            # Values come from each corner's VERTEX, not from the face's
            # own assigned value; sized down so 3 numerals fit one face.
            inradius = compute_face_inradius(die_obj.data, face, die_obj.matrix_world)
            for vertex_index, orient in _face_vertex_orientations(
                die_obj.data, face, die_obj.matrix_world
            ):
                v_value = vertex_values[vertex_index]
                v_label = glyph_label(v_value, glyph_style, die_type)
                width_per_em = _label_width_per_em(v_label, font_id, glyph_style)
                v_size = inradius * D4_CORNER_EM_INRADIUS_FRACTION
                # Corner clearance clamp: a corner copy sits inset of
                # the way up its median, where clearance to the two
                # adjacent edges is (1 - inset) * r -- and keeps
                # shrinking toward the vertex at half the median rate,
                # so the binding constraint is at the glyph's TOP
                # corners (half a cap-height, ~0.35 em, farther up the
                # median): width/2 <= r(1-t) - 0.175*size. Solved for
                # size with a 0.9 safety margin. A center-only clearance
                # model let wide rotated labels ("IV") poke their top
                # corners across the face edge (seen in a real render).
                if width_per_em > 0:
                    clearance = inradius * (1.0 - D4_CORNER_INSET)
                    v_size = min(
                        v_size,
                        0.9 * clearance / (width_per_em / 2.0 + 0.175),
                    )
                planned_cuts.append((v_value, orient, v_size))
            continue
        if glyph_style == "pips":
            font_size = None
        else:
            inradius = compute_face_inradius(die_obj.data, face, die_obj.matrix_world)
            label = glyph_label(value, glyph_style, die_type)
            font_size = _proportional_font_size(
                inradius, label,
                width_per_em=_label_width_per_em(label, font_id, glyph_style),
            )
        pole_co = face_poles[face_index] if face_poles is not None else None
        orient = _face_orientation_matrix(face, die_obj.matrix_world, pole_world_co=pole_co)
        planned_cuts.append((value, orient, font_size))

    # Captured against the pristine convex polyhedron, before any cut:
    # the original face planes, used after all cuts to identify recessed
    # faces exactly (by depth below the original surface) for fill
    # painting -- see _assign_fill_material_to_recessed_faces.
    pristine_planes = [
        (p.normal.copy(), p.normal.dot(p.center)) for p in die_obj.data.polygons
    ]

    # Phase 2: build and apply each cutter using only the precomputed
    # orientation matrices and font sizes — no further indexing into
    # die_obj.data.polygons.
    for value, orient, font_size in planned_cuts:
        if glyph_style == "pips":
            for (ox, oy) in PIP_VALUE_LAYOUTS.get(value, [(0, 0)]):
                bpy.ops.mesh.primitive_uv_sphere_add(radius=size_mm * 0.05)
                pip = bpy.context.active_object
                pip.location = orient @ Vector(
                    (ox * size_mm * 0.4, oy * size_mm * 0.4, -depth * 0.5)
                )
                cut_warning = _boolean_diff_apply(die_obj, pip)
                if cut_warning is not None:
                    warnings.append(cut_warning)
        else:
            label = glyph_label(value, glyph_style, die_type)
            bpy.ops.object.text_add()
            txt_obj = bpy.context.active_object
            txt_obj.data.body = label
            font = _load_font(font_id, glyph_style)
            if font is not None:
                txt_obj.data.font = font
            txt_obj.data.align_x = 'CENTER'
            txt_obj.data.align_y = 'CENTER'
            txt_obj.data.size = font_size
            # Thick cutter, micro penetration: curve extrude is symmetric
            # (E gives a slab spanning [-E, +E]), so translating by
            # (E - depth) puts the slab's bottom exactly `depth` below
            # the face plane and its top well ABOVE the surface -- never
            # coplanar with it, which is the documented boolean failure
            # trigger the old construction (extrude=depth, translate
            # -depth, top face flush at z=0) kept hitting at micro
            # depths. See ENGRAVE_DEPTH_FRACTION's comment.
            cutter_half_t = size_mm * ENGRAVE_CUTTER_HALF_THICKNESS_FRACTION
            txt_obj.data.extrude = cutter_half_t
            bpy.context.view_layer.objects.active = txt_obj
            bpy.ops.object.convert(target='MESH')
            _weld_cutter_mesh(txt_obj)
            txt_obj.matrix_world = orient @ Matrix.Translation((0, 0, cutter_half_t - depth))
            cut_warning = _boolean_diff_apply(die_obj, txt_obj)
            if cut_warning is not None:
                warnings.append(cut_warning)

    # Final backstop, run once after every cut has had its own per-cut retry
    # chance: discard any un-subtracted closed debris shell still left over
    # (see _discard_non_body_closed_debris) so the shipped die is guaranteed
    # free of stray cutter geometry.
    debris_warning = _discard_non_body_closed_debris(die_obj)
    if debris_warning is not None:
        warnings.append(debris_warning)

    if glyph_fill == "painted":
        _assign_fill_material_to_recessed_faces(
            die_obj, pristine_planes, min_depth=depth * 0.3,
        )

    return warnings


def _assign_fill_material_to_recessed_faces(die_obj, pristine_planes, min_depth):
    """
    A face belongs to an engraving recess iff its center sits measurably
    BELOW the pristine convex polyhedron's surface. For a convex solid,
    a point's signed distance to the surface is max_i(n_i . p - d_i)
    over the original face planes (pristine_planes, captured in Phase 1
    before any cut): ~0 for genuine surface faces, about -depth for
    recess floors, roughly -depth/2 for recess walls. min_depth (a
    fraction of the cut depth) cleanly separates recess from surface at
    ANY engraving depth.

    This replaces an area heuristic ("faces much smaller than average
    are recess faces"), which failed both ways at micro depths --
    missing large connected recess-floor faces entirely (numerals with
    unpainted floors) while painting unrelated small surface slivers the
    boolean tessellation leaves around each glyph outline (both
    confirmed visually on a real micro-depth d20).
    """
    if len(die_obj.data.materials) < 2:
        die_obj.data.materials.append(None)

    bm = bmesh.new()
    bm.from_mesh(die_obj.data)
    for f in bm.faces:
        center = f.calc_center_median()
        dist = max(n.dot(center) - d for n, d in pristine_planes)
        if dist < -min_depth:
            f.material_index = 1
    bm.to_mesh(die_obj.data)
    die_obj.data.update()
    bm.free()


def _render_material_swatch(material, resolution, tmp_dir, asset_id):
    """
    Renders what `material` actually looks like -- its real color/procedural
    pattern, fully lit and shaded -- as a flat, opaque PNG, by applying it to
    a simple plane that fills the camera's frame and rendering with EEVEE.
    Reuses the same new-temp-scene/camera/light/render/cleanup technique as
    _render_label_to_image, just with a plane standing in for glyph geometry
    and no film_transparent (the swatch needs to be fully opaque everywhere,
    since it stands in for the die's actual material as the background layer
    in _composite_alpha_over).

    The output filename is keyed on `asset_id`, not `material.name`: within a
    single batch, every same-die-type asset used to derive this (and every
    other decal-related) filename from die_obj.name, which is always just
    f"{die_type}_die" -- identical across every asset of that die type,
    because orchestrator._generate_one frees the name by removing the die
    object at the end of each iteration. That let later same-die-type assets
    in a batch silently overwrite earlier ones' texture files on disk (and
    leave earlier assets' still-referencing USD/materials pointing at a
    different asset's texture). asset_id is unique per asset in a batch, so
    it can't collide this way.
    """
    image_path = os.path.join(tmp_dir, f"{asset_id}_swatch.png")

    scene = bpy.data.scenes.new("dice_gen_swatch_tmp")
    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.film_transparent = False

    cam_data = bpy.data.cameras.new("swatch_cam")
    cam_data.type = 'ORTHO'
    cam_data.ortho_scale = 1.4
    cam_obj = bpy.data.objects.new("swatch_cam", cam_data)
    cam_obj.location = (0, 0, 2)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    light_data = bpy.data.lights.new("swatch_light", type='SUN')
    light_data.energy = 3.0
    light_obj = bpy.data.objects.new("swatch_light", light_data)
    light_obj.location = (0, 0, 3)
    # Sun lights are directional (location irrelevant to lighting, only
    # rotation is) -- a Sun light left at the default identity rotation
    # shines straight down -Z, exactly coaxial with the camera at (0, 0, 2)
    # looking down -Z at the same plane. For any material with noticeable
    # specularity (glitter, metallic -- low roughness), that alignment
    # points the light's specular reflection straight back into the camera,
    # blowing the swatch out to near-white regardless of the material's true
    # color (confirmed: a moderate-value green glitter material rendered as
    # pure white (1,1,1)). Tilting the light off-axis avoids this.
    light_obj.rotation_euler = (0.6, 0.35, 0)
    scene.collection.objects.link(light_obj)

    # size=2.0 (spanning -1..1) is comfortably larger than the camera's
    # ortho_scale=1.4 view width, so the plane fills the entire frame with no
    # background bleed -- the whole rendered image is the material's surface.
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0, 0, 0))
    plane_obj = bpy.context.active_object
    bpy.context.collection.objects.unlink(plane_obj)
    scene.collection.objects.link(plane_obj)
    plane_obj.data.materials.append(material)

    scene.render.filepath = image_path
    prev_scene = bpy.context.window.scene
    bpy.context.window.scene = scene
    bpy.ops.render.render(write_still=True)
    bpy.context.window.scene = prev_scene

    plane_data = plane_obj.data
    bpy.data.objects.remove(plane_obj, do_unlink=True)
    if plane_data is not None and plane_data.users == 0:
        bpy.data.meshes.remove(plane_data)

    bpy.data.objects.remove(cam_obj, do_unlink=True)
    bpy.data.cameras.remove(cam_data)

    bpy.data.objects.remove(light_obj, do_unlink=True)
    bpy.data.lights.remove(light_data)

    bpy.data.scenes.remove(scene)

    return image_path


def _composite_alpha_over(background_path, foreground_path, output_path, resolution):
    """
    Alpha-composites `foreground_path` (the rendered glyph decal -- opaque
    ink on a transparent background, see _render_label_to_image) over
    `background_path` (the die's real material appearance, see
    _render_material_swatch), writing the flattened result to
    `output_path`.

    This exists because the earlier fix (see apply_decal_glyphs) originally
    did this same alpha compositing *inside Blender's shader graph* -- a
    ShaderNodeMix wired between the decal's Image Texture node and the
    Principled BSDF's Base Color input, using the decal's alpha as the mix
    factor. That looked correct in Blender's own viewport/render, but
    bpy.ops.wm.usd_export cannot represent a ShaderNodeMix in either its
    default UsdPreviewSurface mode or with generate_materialx_network=True:
    every exported asset silently reverted to Base Color being fed directly
    by the raw (uncomposited) decal texture, reproducing the exact
    solid-black-face defect the shader-graph fix was meant to solve, just
    invisibly, only in the exported USD. Doing the compositing here, in
    Python/numpy, on the actual pixels, means the resulting PNG already
    contains the final composited appearance -- so it can be wired to Base
    Color with a plain, single Image Texture node (which *is* confirmed to
    survive USD export faithfully), and there is no shader-graph construct
    left for the exporter to silently drop. Do not reintroduce a shader
    Mix/compositing node for this purpose; it will not survive USD export.

    PIL/Pillow is not available in Blender's bundled Python, so this uses
    numpy directly on bpy.data.images pixel buffers instead.
    """
    bg_image = bpy.data.images.load(background_path)
    fg_image = bpy.data.images.load(foreground_path)

    bg_pixels = np.empty(resolution * resolution * 4, dtype=np.float32)
    bg_image.pixels.foreach_get(bg_pixels)
    bg = bg_pixels.reshape(resolution, resolution, 4)

    fg_pixels = np.empty(resolution * resolution * 4, dtype=np.float32)
    fg_image.pixels.foreach_get(fg_pixels)
    fg = fg_pixels.reshape(resolution, resolution, 4)

    fg_alpha = fg[:, :, 3:4]
    result = np.empty_like(bg)
    result[:, :, :3] = fg[:, :, :3] * fg_alpha + bg[:, :, :3] * (1.0 - fg_alpha)
    # The background swatch is a fully opaque render covering the whole
    # frame (see _render_material_swatch), so the composited result is
    # opaque everywhere too.
    result[:, :, 3] = 1.0

    out_image = bpy.data.images.new(
        "dice_gen_composited_tmp", width=resolution, height=resolution, alpha=True
    )
    out_image.pixels.foreach_set(result.reshape(-1))
    out_image.filepath_raw = output_path
    out_image.file_format = 'PNG'
    out_image.save()

    bpy.data.images.remove(bg_image)
    bpy.data.images.remove(fg_image)
    bpy.data.images.remove(out_image)


def _unwrap_faces_to_full_square(die_obj, die_type, margin=0.1):
    """
    Gives every face its OWN UV island filling the full 0-1 square,
    instead of bpy.ops.uv.smart_project's shared-atlas packing (which
    only gives each face a small fraction of the 0-1 space -- confirmed
    empirically on a d8, each face's island only covered roughly a
    0.27x0.31 patch). apply_decal_glyphs gives each face its own
    DEDICATED texture image (the glyph centered at (0.5, 0.5)), so an
    atlas-style shared unwrap is the wrong tool: it leaves most faces
    sampling only a background-colored corner of their own image,
    missing the centered glyph entirely.

    Projects each face's vertices into its own (tangent, bitangent) frame
    (see _tangent_bitangent) relative to the face center, then scales so
    the larger of the two axis spans fits into 1.0 - 2*margin, centered
    at (0.5, 0.5). Verified empirically to produce full per-face coverage
    across d6/d8/d10/d12/d20's differently-shaped faces (triangle, quad,
    kite, pentagon).

    For d8/d10, each face's tangent/bitangent frame uses that face's own
    pole (see geometry.compute_face_poles) as the up_reference passed to
    _tangent_bitangent, mirroring _face_orientation_matrix's engraved-path
    fix -- otherwise the same die type would show the correct mirrored
    numeral pattern when engraved but the old smoothly-rotating (wrong)
    pattern when printed as a decal. compute_face_poles returns WORLD-space
    pole positions, but this function operates in LOCAL space (see below),
    so each pole is transformed into local space via
    die_obj.matrix_world.inverted() before use. For all other die types,
    compute_face_poles returns None and behavior is unchanged.

    For triangular faces (d4 and d8/d20's faces), the raw tangent/bitangent
    projection does NOT guarantee the same triangle orientation face to
    face: confirmed empirically on a d4 that faces alternate between
    "apex-up" (one vertex at high v, two sharing a low v) and "apex-down"
    (the reverse) depending on each face's normal direction relative to
    the shared global-up-hint convention. This broke
    _render_label_to_image's fixed 3-corner d4 layout (which assumes every
    face is apex-up): with alternating orientation, the top-corner copy
    landed on a real vertex for half the faces and on the middle of a flat
    edge for the other half, clipping it, while the two "bottom" copies
    fell partly or fully outside the actual triangular UV footprint on
    apex-down faces -- confirmed via manual batch regeneration (thumbnails
    showed only a clipped top numeral and no bottom-corner numerals on
    several d4 assets). Every triangular face's single distinctive vertex
    (the one whose local v differs most from the other two) is normalized
    to be the HIGH-v one (apex-up) if it came out low-v instead -- by
    rotating the face 180 degrees about its center (negating BOTH u and
    v), NOT by flipping v alone. An earlier version of this fix negated
    only v, which is a mirror reflection (determinant -1): it corrected
    apex-up-ness but reversed winding/handedness on exactly the faces it
    "fixed", which would render any text on those faces backwards.
    Confirmed via signed-area (shoelace formula) comparison: the raw
    pre-fix projection was already winding-consistent across every face
    (all the same sign); the v-only-negate version produced a 50/50 mix
    of signs. Negating both u and v (determinant +1) corrects the same
    apex-up/down orientation while preserving winding on every face.
    """
    mesh = die_obj.data
    if mesh.uv_layers.active is None:
        mesh.uv_layers.new(name="decal_uv")
    uv_layer = mesh.uv_layers.active.data

    face_poles = compute_face_poles(die_obj, die_type)
    matrix_world_inv = die_obj.matrix_world.inverted()

    glyph_anchors = {}
    for poly in mesh.polygons:
        normal = poly.normal
        center = poly.center

        if face_poles is not None:
            pole_local = matrix_world_inv @ face_poles[poly.index]
            to_pole = pole_local - center
            up_reference = (to_pole - to_pole.dot(normal) * normal).normalized()
        else:
            # Same explicit default-up-hint + vertex-snap treatment as
            # _face_orientation_matrix, so engraved and decal numerals on
            # the same die type share one exact orientation convention.
            up_hint = Vector((0, 0, 1)) if abs(normal.z) < 0.999 else Vector((0, 1, 0))
            up_reference = (up_hint - up_hint.dot(normal) * normal).normalized()

        vertex_dirs = []
        for loop_index in poly.loop_indices:
            vertex_index = mesh.loops[loop_index].vertex_index
            d = mesh.vertices[vertex_index].co - center
            d = (d - d.dot(normal) * normal).normalized()
            vertex_dirs.append(d)
        up_reference = _snap_up_to_face_vertex(
            up_reference, normal, vertex_dirs, len(poly.loop_indices)
        )

        tangent, bitangent = _tangent_bitangent(normal, up_reference=up_reference)

        local_coords = []
        for loop_index in poly.loop_indices:
            vertex_index = mesh.loops[loop_index].vertex_index
            rel = mesh.vertices[vertex_index].co - center
            local_coords.append([rel.dot(tangent), rel.dot(bitangent)])

        if len(local_coords) == 3:
            vs_raw = [c[1] for c in local_coords]
            _, apex_index = min(
                (abs(vs_raw[0] - vs_raw[1]), 2),
                (abs(vs_raw[1] - vs_raw[2]), 0),
                (abs(vs_raw[0] - vs_raw[2]), 1),
            )
            other_v = sum(v for i, v in enumerate(vs_raw) if i != apex_index) / 2.0
            if vs_raw[apex_index] < other_v:
                # Negate BOTH u and v (a 180-degree rotation about the
                # face center) -- NOT v alone. Negating only v is a
                # mirror reflection (determinant -1), which reverses
                # winding/handedness and renders any text on that face
                # backwards. Confirmed empirically: comparing the signed
                # area of the raw (pre-normalization) UV triangle against
                # the final one, the raw projection was ALREADY
                # winding-consistent across every face of a d8 (all
                # +86.603) -- a v-only flip made exactly the faces that
                # needed apex correction flip sign too (mixed +/-0.277),
                # proving it mirrors text on those faces alone. A 180
                # degree rotation (negating both axes) corrects the
                # apex-up/down orientation the exact same way (v still
                # ends up negated) while preserving winding, since
                # negating both coordinates has determinant +1.
                for c in local_coords:
                    c[0] = -c[0]
                    c[1] = -c[1]

        us = [c[0] for c in local_coords]
        vs = [c[1] for c in local_coords]
        span = max(max(us) - min(us), max(vs) - min(vs))
        scale = (1.0 - 2 * margin) / span if span > 0 else 1.0

        # Recenter on the bounding-box midpoint (not poly.center's projection)
        # to guarantee the final UV range is symmetric around 0.5 and bounded
        # within [margin, 1-margin] for asymmetric face shapes like d10's kites.
        u_mid = (max(us) + min(us)) / 2.0
        v_mid = (max(vs) + min(vs)) / 2.0

        for loop_index, (u, v) in zip(poly.loop_indices, local_coords):
            uv_layer[loop_index].uv = (0.5 + (u - u_mid) * scale, 0.5 + (v - v_mid) * scale)

        # Where the face CENTER (poly.center -- the same anchor the
        # engraved path cuts at) lands in this face's UV square. The
        # island is bbox-centered (see u_mid/v_mid above), so for
        # non-centrally-symmetric faces the center is NOT at (0.5, 0.5):
        # an apex-up triangle's center sits half an inradius below its
        # bbox midpoint, which is exactly how far off-center (toward the
        # apex, where there's the LEAST room) a glyph rendered at canvas
        # center used to land on every d4/d8/d20 decal face. The
        # 180-degree apex flip above negates both axes, so the center's
        # projection is (-0, -0) == (0, 0) relative coords either way and
        # this expression is correct whether or not the face was flipped.
        # "span" is the face's larger in-plane bbox extent in mesh units
        # (mm) -- the length that gets scaled to fill 1.0 - 2*margin of
        # UV space -- letting canvas-space consumers convert real mm
        # lengths (e.g. the face inradius, for label-width clamping in
        # _render_label_to_image) into canvas units exactly.
        glyph_anchors[poly.index] = {
            "anchor": (0.5 - u_mid * scale, 0.5 - v_mid * scale),
            "span": span,
        }

    return glyph_anchors


def apply_decal_glyphs(die_obj, die_type, assignment, glyph_style, font_id, size_mm, asset_id, tmp_dir):
    """
    `asset_id` is used as the filename prefix for every image file this
    function (and its helpers _render_label_to_image, _render_material_swatch,
    _composite_alpha_over) writes under `tmp_dir`. It must NOT be derived from
    die_obj.name: die_obj.name is always just f"{die_type}_die" for every
    asset of a given die type, and orchestrator._generate_one frees that name
    (by removing the die object) at the end of every batch iteration, so
    Blender never auto-suffixes it -- every same-die-type printed_decal asset
    in a batch used to derive colliding filenames (e.g. "d8_die_face0.png"),
    silently overwriting earlier assets' texture files on disk. asset_id is
    unique per asset within a batch, so it cannot collide this way. In-memory
    Blender datablock names (e.g. the decal materials created below) are
    unaffected by this and intentionally left keyed on die_obj.name, since
    Blender auto-uniquifies datablock names within a session and this is not
    a filename-collision concern.
    """
    glyph_anchors = _unwrap_faces_to_full_square(die_obj, die_type)

    base_mat = die_obj.data.materials[0] if len(die_obj.data.materials) > 0 else None

    # 512 (was 256): at 256 a d4's corner-placed numerals occupy so few
    # texels that strokes render visibly blurry even in modest close-ups
    # (user-reported); 512 quadruples texel density for every decal face
    # at modest per-face render cost.
    resolution = 512

    # The base material is identical across every face of a die, so its
    # appearance only needs to be rendered once, not per-face.
    swatch_path = None
    ink_rgba = (0.02, 0.02, 0.02, 1.0)
    if base_mat is not None:
        swatch_path = _render_material_swatch(base_mat, resolution, tmp_dir, asset_id)
        # Pick the ink color from the swatch's real rendered luminance, not
        # from the material's HSV params: marbled/speckled/glitter/metallic
        # appearance comes from the whole node graph, and the swatch is the
        # one place that appearance already exists as pixels. Dark die ->
        # near-white ink, light die -> near-black ink. (Previously the
        # glyph objects had NO material at all and rendered default-shader
        # gray -- invisible on dark dice, washed out on light ones.)
        luminance = _swatch_luminance(swatch_path)
        ink_rgba = (0.02, 0.02, 0.02, 1.0) if luminance > 0.5 else (0.95, 0.95, 0.95, 1.0)

    for face_index, value in assignment.items():
        image_path = os.path.join(tmp_dir, f"{asset_id}_face{face_index}.png")
        # apply_decal_glyphs never applies a boolean modifier or otherwise
        # mutates die_obj.data's topology (unlike apply_engraved_glyphs), so
        # -- unlike that function's Phase 1/Phase 2 split -- indexing
        # die_obj.data.polygons directly inside this per-face loop is safe.
        face = die_obj.data.polygons[face_index]
        inradius = compute_face_inradius(die_obj.data, face, die_obj.matrix_world)
        face_info = glyph_anchors.get(face_index, {})
        _render_label_to_image(
            value, glyph_style, font_id, die_type, image_path,
            resolution=resolution, inradius=inradius, size_mm=size_mm,
            anchor_uv=face_info.get("anchor"), face_span=face_info.get("span"),
            ink_rgba=ink_rgba,
            corner_labels=_d4_corner_labels(die_obj, die_type, face_index, glyph_style),
        )

        if base_mat is not None:
            mat = base_mat.copy()
            mat.name = f"{die_obj.name}_face{face_index}_decal"

            # Composite the glyph decal onto the die's real material at the
            # pixel level (see _composite_alpha_over's docstring for why this
            # replaced the earlier shader-graph Mix-node approach), so a
            # plain, direct Image-Texture-to-Base-Color wire below already
            # carries the correct final appearance.
            composited_path = os.path.join(
                tmp_dir, f"{asset_id}_face{face_index}_composited.png"
            )
            _composite_alpha_over(swatch_path, image_path, composited_path, resolution)
            texture_path = composited_path
        else:
            # No pre-assigned base material to composite onto (e.g. tests
            # that call apply_decal_glyphs standalone) -- fall back to
            # wiring the raw glyph decal straight to Base Color, exactly as
            # before.
            mat = bpy.data.materials.new(name=f"{die_obj.name}_face{face_index}_decal")
            mat.use_nodes = True
            texture_path = image_path

        nt = mat.node_tree
        tex_node = nt.nodes.new("ShaderNodeTexImage")
        tex_node.image = bpy.data.images.load(texture_path)
        nt.links.new(tex_node.outputs["Color"], nt.nodes["Principled BSDF"].inputs["Base Color"])

        die_obj.data.materials.append(mat)
        die_obj.data.polygons[face_index].material_index = len(die_obj.data.materials) - 1


def _swatch_luminance(swatch_path):
    """Mean Rec.709 luminance of a rendered material swatch PNG."""
    swatch_img = bpy.data.images.load(swatch_path)
    px = np.array(swatch_img.pixels[:]).reshape(-1, 4)
    luminance = float((0.2126 * px[:, 0] + 0.7152 * px[:, 1] + 0.0722 * px[:, 2]).mean())
    bpy.data.images.remove(swatch_img)
    return luminance


def material_rendered_luminance(material, tmp_dir, asset_id, resolution=64):
    """
    Renders a small swatch of `material` and returns its mean luminance.
    This is the trustworthy signal for light-vs-dark ink/fill decisions:
    a material's HSV `value` param is only an input to its node graph,
    and can badly misstate what the material actually looks like
    (confirmed on a real translucent die: value=0.55 -- nominally light
    -- rendered dark olive, so its param-chosen dark engraving fill was
    invisible). Rendering is what the training data consumer sees, so
    luminance is measured at that level.
    """
    swatch_path = _render_material_swatch(
        material, resolution, tmp_dir, f"{asset_id}_lum"
    )
    return _swatch_luminance(swatch_path)


def _d4_corner_labels(die_obj, die_type, face_index, glyph_style):
    """
    For a d4 face, pairs each of the face's REAL UV corner positions
    with the label of the mesh vertex that landed there, using the UV
    layout _unwrap_faces_to_full_square already wrote. This is what
    makes the vertex-read convention hold on the decal path: every face
    shows its 3 corners' own vertex values, and the 3 faces meeting at
    any vertex agree on the value shown there.

    Returning the real per-face UV positions (not just labels for an
    idealized fixed layout) also fixes the numerals' centering:
    _render_label_to_image places each copy on the MEDIAN from the face
    center toward its own vertex. An earlier fixed canonical layout
    inset the copies about the CANVAS center instead -- but the canvas
    center is the UV island's bbox midpoint, which for a triangle sits
    half an inradius toward the apex from the centroid, so all three
    numerals were visibly displaced off their medians (user-reported
    twice).

    Returns [(label, (u, v)), ...] for the face's 3 corners, or None for
    anything that isn't a d4 numeral face (including pips).
    """
    if die_type != "d4" or glyph_style == "pips":
        return None
    mesh = die_obj.data
    uv_layer = mesh.uv_layers.active.data
    poly = mesh.polygons[face_index]
    vertex_values = d4_vertex_values(len(mesh.vertices))

    entries = []
    for loop_index in poly.loop_indices:
        u, v = uv_layer[loop_index].uv
        entries.append((u, v, mesh.loops[loop_index].vertex_index))
    return [
        (glyph_label(vertex_values[vi], glyph_style, die_type), (u, v))
        for (u, v, vi) in entries
    ]


def _ink_material(rgba):
    """
    Flat unlit (emission) material for decal glyph objects, so the ink
    renders as exactly `rgba` regardless of the canvas scene's lighting.
    """
    mat = bpy.data.materials.new("dice_gen_ink_tmp")
    mat.use_nodes = True
    nt = mat.node_tree
    for node in list(nt.nodes):
        nt.nodes.remove(node)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    emission = nt.nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = rgba
    nt.links.new(emission.outputs["Emission"], out.inputs["Surface"])
    return mat


def _render_label_to_image(value, glyph_style, font_id, die_type, image_path, resolution=256,
                            inradius=None, size_mm=None, anchor_uv=None, face_span=None,
                            ink_rgba=None, corner_labels=None):
    """
    `inradius` (this face's world-space inradius, mm -- see
    geometry.compute_face_inradius) and `size_mm` (the die's overall
    size) are only required for the non-"pips" branches below (pips are
    circle primitives with a fixed local radius, unaffected by font
    sizing). Both are combined into a dimensionless inradius/size_mm
    ratio before being handed to the shared `_proportional_font_size`
    helper, then rescaled into this function's own local text-size units
    by DECAL_FONT_CANVAS_SCALE -- see that constant's docstring for why
    a ratio (not the raw mm inradius) is the right input here: this
    function's camera/canvas setup is fixed and dimensionless (the same
    for every die type), with per-face real-world scale already
    normalized away by apply_decal_glyphs's prior call to
    _unwrap_faces_to_full_square, so only the face's SHAPE (how its
    inradius compares to its own die's overall size) remains meaningful
    here, not its absolute mm size.
    """
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

    # Canvas-space offset of the face's real center (where the engraved
    # path puts its glyphs) from the canvas midpoint -- see
    # _unwrap_faces_to_full_square's returned glyph_anchors. The island is
    # bbox-centered in UV, so on non-centrally-symmetric faces (triangles,
    # kites, pentagons) rendering at (0,0) would land the glyph up to half
    # an inradius away from the face's visual center. ortho_scale=1.4 maps
    # canvas x in [-0.7, 0.7] to u in [0, 1], hence the 1.4 factor.
    anchor_x, anchor_y = 0.0, 0.0
    if anchor_uv is not None:
        anchor_x = (anchor_uv[0] - 0.5) * 1.4
        anchor_y = (anchor_uv[1] - 0.5) * 1.4

    ink_mat = _ink_material(ink_rgba if ink_rgba is not None else (0.02, 0.02, 0.02, 1.0))

    glyph_objs = []
    if glyph_style == "pips":
        for (ox, oy) in PIP_VALUE_LAYOUTS.get(value, [(0, 0)]):
            bpy.ops.mesh.primitive_circle_add(
                radius=0.12, fill_type='NGON',
                location=(ox + anchor_x, oy + anchor_y, 0)
            )
            dot = bpy.context.active_object
            dot.data.materials.append(ink_mat)
            bpy.context.collection.objects.unlink(dot)
            scene.collection.objects.link(dot)
            glyph_objs.append(dot)
    elif die_type == "d4":
        # Real vertex-read d4 dice key values to VERTICES: each face
        # shows its 3 corners' own vertex values (3 DIFFERENT numbers per
        # face), and the 3 faces meeting at a vertex agree on the value
        # shown at that shared corner -- see numbering.d4_vertex_values
        # and _d4_corner_labels, which maps this face's UV corners back
        # to the real mesh vertices that landed there. (An earlier
        # version rendered the face's single assigned value at all 3
        # corners; corrected per direct user feedback.)
        #
        # Corner geometry: each copy sits on its own MEDIAN -- the line
        # from the face center (anchor_uv, the same centroid the engrave
        # path cuts at) toward its own real UV vertex position (carried
        # in corner_labels by _d4_corner_labels), `inset` of the way out.
        # Two earlier layouts were both visibly off: a fixed-radius
        # circle (an equilateral triangle's vertices are not equidistant
        # from its bbox center), then a fixed bbox-relative corner list
        # inset about the CANVAS center -- the canvas center is the UV
        # island's bbox midpoint, half an inradius toward the apex from
        # the centroid, so all three numerals sat displaced off their
        # medians (user-reported). Real per-face positions replace all
        # idealized layouts.
        ortho_scale = 1.4
        inset = D4_CORNER_INSET
        if corner_labels is not None:
            placements = []
            cx0 = anchor_uv[0] if anchor_uv is not None else 0.5
            cy0 = anchor_uv[1] if anchor_uv is not None else 0.5
            for label, (vu, vv) in corner_labels:
                pu = cx0 + (vu - cx0) * inset
                pv = cy0 + (vv - cy0) * inset
                placements.append((
                    label,
                    ((pu - 0.5) * ortho_scale, (pv - 0.5) * ortho_scale),
                    math.atan2(vv - cy0, vu - cx0),
                ))
        else:
            # Fallback for direct callers without a real unwrapped face
            # (tests exercising canvas mechanics): idealized apex-up
            # equilateral layout about the canvas center.
            label = glyph_label(value, glyph_style, die_type)
            half_width = 0.4
            half_height = half_width * math.sqrt(3) / 2
            placements = [
                (label, (cx * inset * ortho_scale, cy * inset * ortho_scale),
                 math.atan2(cy, cx))
                for cx, cy in [(0.0, half_height), (-half_width, -half_height),
                               (half_width, -half_height)]
            ]
        for label, (ox, oy), angle in placements:
            width_per_em = _label_width_per_em(label, font_id, glyph_style)
            # Constant em height for every value on the die (see
            # D4_CORNER_EM_INRADIUS_FRACTION), in canvas units via the
            # face_span mapping; falls back to the span-free approximation
            # (inradius/span for an equilateral triangle) if span is absent.
            if face_span:
                inradius_canvas_em = (inradius / face_span) * 1.12
            else:
                inradius_canvas_em = 0.2887 * 1.12
            font_size = inradius_canvas_em * D4_CORNER_EM_INRADIUS_FRACTION
            # Corner clearance clamp, mirroring the engraved path (see
            # its comment for the geometry: the binding constraint is at
            # the glyph's TOP corners, which sit closer to the
            # converging adjacent edges than its center does). Canvas
            # units via the face_span mapping.
            if face_span and width_per_em > 0:
                inradius_canvas = (inradius / face_span) * 1.12
                clearance_canvas = inradius_canvas * (1.0 - inset)
                font_size = min(
                    font_size,
                    0.9 * clearance_canvas / (width_per_em / 2.0 + 0.175),
                )
            bpy.ops.object.text_add(location=(ox, oy, 0))
            txt_obj = bpy.context.active_object
            txt_obj.data.body = label
            font = _load_font(font_id, glyph_style)
            if font is not None:
                txt_obj.data.font = font
            txt_obj.data.align_x = 'CENTER'
            txt_obj.data.align_y = 'CENTER'
            txt_obj.data.size = font_size
            # Rotate so this copy's "up" points radially outward toward
            # its own corner (the apex, straight up, needs zero rotation
            # since text already reads "up" by default).
            txt_obj.rotation_euler = (0, 0, angle - math.pi / 2)
            txt_obj.data.materials.append(ink_mat)
            bpy.context.collection.objects.unlink(txt_obj)
            scene.collection.objects.link(txt_obj)
            glyph_objs.append(txt_obj)
    else:
        label = glyph_label(value, glyph_style, die_type)
        font_size = _proportional_font_size(inradius / size_mm, label) * DECAL_FONT_CANVAS_SCALE
        # Exact canvas-space width clamp: the face's larger bbox extent
        # (face_span, mm) maps to (1 - 2*margin) * ortho_scale = 1.12
        # canvas units (see _unwrap_faces_to_full_square), so the face
        # inradius in canvas units is (inradius / face_span) * 1.12. The
        # character-count shrink alone underestimates full-width CJK
        # labels badly enough for strokes to visibly cross face edges
        # (confirmed on a real d20 batch).
        width_per_em = _label_width_per_em(label, font_id, glyph_style)
        if face_span and width_per_em > 0:
            inradius_canvas = (inradius / face_span) * 1.12
            font_size = min(
                font_size,
                MAX_LABEL_WIDTH_INRADIUS_FRACTION * inradius_canvas / width_per_em,
            )
        bpy.ops.object.text_add(location=(anchor_x, anchor_y, 0))
        txt_obj = bpy.context.active_object
        txt_obj.data.body = label
        font = _load_font(font_id, glyph_style)
        if font is not None:
            txt_obj.data.font = font
        txt_obj.data.align_x = 'CENTER'
        txt_obj.data.align_y = 'CENTER'
        txt_obj.data.size = font_size
        txt_obj.data.materials.append(ink_mat)
        bpy.context.collection.objects.unlink(txt_obj)
        scene.collection.objects.link(txt_obj)
        glyph_objs.append(txt_obj)

    scene.render.filepath = image_path
    prev_scene = bpy.context.window.scene
    bpy.context.window.scene = scene
    bpy.ops.render.render(write_still=True)
    bpy.context.window.scene = prev_scene

    for glyph_obj in glyph_objs:
        glyph_data = glyph_obj.data
        bpy.data.objects.remove(glyph_obj, do_unlink=True)
        if glyph_data is None or glyph_data.users > 0:
            continue
        if isinstance(glyph_data, bpy.types.Mesh):
            bpy.data.meshes.remove(glyph_data)
        elif isinstance(glyph_data, bpy.types.Curve):
            bpy.data.curves.remove(glyph_data)

    bpy.data.objects.remove(cam_obj, do_unlink=True)
    bpy.data.cameras.remove(cam_data)

    bpy.data.objects.remove(light_obj, do_unlink=True)
    bpy.data.lights.remove(light_data)

    bpy.data.materials.remove(ink_mat)

    bpy.data.scenes.remove(scene)
