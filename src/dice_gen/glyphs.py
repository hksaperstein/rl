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

ENGRAVE_DEPTH_FRACTION = 0.02
FONT_INRADIUS_FRACTION = 0.5
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
# default of 1.0 exactly (0.5 * FONT_INRADIUS_FRACTION * 4.0 == 1.0),
# anchoring the new proportional sizing to the one old value most likely to
# have already been visually reasonable, while every smaller-faced die type
# and every longer label now correctly renders smaller than that anchor
# instead of at the same fixed size. Verified by rendering
# test_render_label_to_image_renders_three_corner_copies_for_d4 with real
# d4 geometry and inspecting the resulting ink regions (see that test).
DECAL_FONT_CANVAS_SCALE = 4.0

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


def _proportional_font_size(inradius, label):
    """
    Calibrated this session (FONT_INRADIUS_FRACTION=0.5,
    FONT_EXTRA_CHAR_SHRINK=0.35) against the real worst cases across
    every die type/glyph style combination -- see
    test_proportional_font_size_shrinks_for_longer_labels. Longer labels
    (e.g. d20's 2-digit arabic numerals, or "XVIII" for roman numeral 18)
    need a smaller per-character size to occupy roughly the same total
    footprint as a single-character label at the same font size would.
    """
    n = len(label)
    return inradius * FONT_INRADIUS_FRACTION / (1 + (n - 1) * FONT_EXTRA_CHAR_SHRINK)


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


def _face_orientation_matrix(face, obj_matrix, pole_world_co=None):
    center = obj_matrix @ face.center
    normal = (obj_matrix.to_3x3() @ face.normal).normalized()
    up_reference = None
    if pole_world_co is not None:
        to_pole = pole_world_co - center
        up_reference = (to_pole - to_pole.dot(normal) * normal).normalized()
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


def _soften_cutter_edges(cutter_obj, depth):
    """
    Explicit user request: engraved recesses should have softened
    (rounded) edges, not a sharp 90-degree transition where the cut wall
    meets the die's flat face. Bevels every edge of the cutter mesh
    itself (limit_method='NONE') before it's used in the boolean cut --
    the cutter is one small glyph feature, unlike the die body's own
    bevel (see exporter.py), which is deliberately scoped to structural
    edges only via a bevel-weight attribute. width/segments calibrated
    this session: large enough to visibly round the recess, small enough
    relative to `depth` to keep the added boolean-cut complexity (and
    its small, measured non-manifold-junction cost -- 10-22 edges on a
    representative case, tracked via mesh_quality_warnings like every
    other known residual in this pipeline) modest.
    """
    mod = cutter_obj.modifiers.new(name="SoftenEdges", type='BEVEL')
    mod.width = depth * 0.25
    mod.segments = 2
    mod.limit_method = 'NONE'
    bpy.ops.object.select_all(action='DESELECT')
    cutter_obj.select_set(True)
    bpy.context.view_layer.objects.active = cutter_obj
    bpy.ops.object.modifier_apply(modifier=mod.name)

    bm = bmesh.new()
    bm.from_mesh(cutter_obj.data)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(cutter_obj.data)
    cutter_obj.data.update()
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


def _face_vertex_orientations(mesh, face, obj_matrix, inset=0.55):
    """
    Returns one orientation matrix per vertex of `face`, for d4's
    vertex-read numeral convention: real commercial d4 dice (standard
    tetrahedra -- confirmed this is the shape geometry.py builds) show
    the same digit three times per face, once near each corner, oriented
    so whichever corner points up when the die rests on the opposite
    face reads correctly -- unlike every other die type, which uses a
    single centered numeral via _face_orientation_matrix's global
    up-hint convention.

    For each vertex, "up" (bitangent) is the direction from the face
    center toward that vertex, projected into the face plane -- i.e.
    each copy points radially outward toward its own corner, matching
    the 120-degree-apart rotational pattern real vertex-read d4s show.
    `inset` places each copy 55% of the way from the face center to the
    vertex (tested empirically: keeps the numeral clear of both the
    face center and the beveled edge).
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
        orientations.append(rot)
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
    # Real commercial d4 dice show the same numeral 3 times per face (once
    # per corner, vertex-read) rather than the single centered numeral every
    # other die type uses -- see _face_vertex_orientations. This branch must
    # stay inside Phase 1 (computed entirely against the pristine mesh):
    # recomputing face.vertices/face.normal mid-loop, after an earlier cut
    # has already rebuilt the mesh topology, causes a Blender crash.
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
    planned_cuts = []
    for face_index, value in assignment.items():
        face = die_obj.data.polygons[face_index]
        if glyph_style == "pips":
            font_size = None
        else:
            inradius = compute_face_inradius(die_obj.data, face, die_obj.matrix_world)
            label = glyph_label(value, glyph_style)
            font_size = _proportional_font_size(inradius, label)
        if is_d4_vertex_numerals:
            for orient in _face_vertex_orientations(die_obj.data, face, die_obj.matrix_world):
                planned_cuts.append((value, orient, font_size))
        else:
            pole_co = face_poles[face_index] if face_poles is not None else None
            orient = _face_orientation_matrix(face, die_obj.matrix_world, pole_world_co=pole_co)
            planned_cuts.append((value, orient, font_size))

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
            label = glyph_label(value, glyph_style)
            bpy.ops.object.text_add()
            txt_obj = bpy.context.active_object
            txt_obj.data.body = label
            font = _load_font(font_id, glyph_style)
            if font is not None:
                txt_obj.data.font = font
            txt_obj.data.align_x = 'CENTER'
            txt_obj.data.align_y = 'CENTER'
            txt_obj.data.size = font_size
            txt_obj.data.extrude = depth
            bpy.context.view_layer.objects.active = txt_obj
            bpy.ops.object.convert(target='MESH')
            _weld_cutter_mesh(txt_obj)
            _soften_cutter_edges(txt_obj, depth)
            txt_obj.matrix_world = orient @ Matrix.Translation((0, 0, -depth))
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
        _assign_fill_material_to_recessed_faces(die_obj)

    return warnings


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

    for poly in mesh.polygons:
        normal = poly.normal
        center = poly.center

        up_reference = None
        if face_poles is not None:
            pole_local = matrix_world_inv @ face_poles[poly.index]
            to_pole = pole_local - center
            up_reference = (to_pole - to_pole.dot(normal) * normal).normalized()
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
    _unwrap_faces_to_full_square(die_obj, die_type)

    base_mat = die_obj.data.materials[0] if len(die_obj.data.materials) > 0 else None

    resolution = 256

    # The base material is identical across every face of a die, so its
    # appearance only needs to be rendered once, not per-face.
    swatch_path = None
    if base_mat is not None:
        swatch_path = _render_material_swatch(base_mat, resolution, tmp_dir, asset_id)

    for face_index, value in assignment.items():
        image_path = os.path.join(tmp_dir, f"{asset_id}_face{face_index}.png")
        # apply_decal_glyphs never applies a boolean modifier or otherwise
        # mutates die_obj.data's topology (unlike apply_engraved_glyphs), so
        # -- unlike that function's Phase 1/Phase 2 split -- indexing
        # die_obj.data.polygons directly inside this per-face loop is safe.
        face = die_obj.data.polygons[face_index]
        inradius = compute_face_inradius(die_obj.data, face, die_obj.matrix_world)
        _render_label_to_image(
            value, glyph_style, font_id, die_type, image_path,
            resolution=resolution, inradius=inradius, size_mm=size_mm,
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


def _render_label_to_image(value, glyph_style, font_id, die_type, image_path, resolution=256,
                            inradius=None, size_mm=None):
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

    glyph_objs = []
    if glyph_style == "pips":
        for (ox, oy) in PIP_VALUE_LAYOUTS.get(value, [(0, 0)]):
            bpy.ops.mesh.primitive_circle_add(
                radius=0.12, fill_type='NGON', location=(ox, oy, 0)
            )
            dot = bpy.context.active_object
            bpy.context.collection.objects.unlink(dot)
            scene.collection.objects.link(dot)
            glyph_objs.append(dot)
    elif die_type == "d4":
        # Real commercial d4 dice (standard tetrahedra) show the same
        # digit at all three corners of each face, oriented so whichever
        # corner is "up" reads correctly -- see the vertex-read design
        # doc. Every d4 face is a congruent equilateral triangle, so a
        # fixed canonical 3-corner layout (not tied to this face's real
        # 3D vertex positions) is sufficient: all three copies show the
        # identical value, so exact per-vertex correspondence doesn't
        # matter, only that each corner gets one correctly-outward-
        # rotated copy.
        #
        # An earlier version placed the 3 copies on a fixed-radius circle
        # (equal distance from center for all 3). That's the WRONG shape:
        # confirmed via manual batch regeneration that only the "top"
        # copy landed near a real corner (and was still clipped), while
        # both "bottom" copies rendered well inside the face, nowhere
        # near the actual bottom-left/bottom-right vertices -- visible as
        # faint marks near the middle of the face rather than at its
        # corners. Root cause: a real equilateral triangle's vertices are
        # NOT equidistant from its own bounding-box center (which is what
        # _unwrap_faces_to_full_square centers UV coordinates on) -- the
        # two base vertices sit farther from that center than the apex
        # does. The circle assumption ignored this, so the two "base"
        # copies ended up positioned much closer to center (in UV terms)
        # than the real base vertices.
        #
        # Fixed by computing the actual bounding-box-relative vertex
        # offsets of an equilateral triangle whose half-width matches
        # _unwrap_faces_to_full_square's default margin (0.1), i.e.
        # half_width = 0.5 - 0.1 = 0.4 (UV-delta units), half_height =
        # half_width * sqrt(3)/2 (an equilateral triangle's height/width
        # ratio) -- then converting UV-delta units to this function's
        # world-space scene via world = uv_delta * ortho_scale (this
        # camera's ortho_scale=1.4 means world spans [-0.7,0.7] map to
        # UV [0,1]), and applying `inset` (0.55, matching the engrave
        # path's own corner inset) to bring each copy in from the true
        # vertex position for clearance from the real edge. Verified via
        # full-pipeline render (real UV unwrap + composite + 3D render)
        # during this fix: all three copies now appear at the face's
        # actual three corners.
        label = glyph_label(value, glyph_style)
        font_size = _proportional_font_size(inradius / size_mm, label) * DECAL_FONT_CANVAS_SCALE
        ortho_scale = 1.4
        inset = 0.55
        half_width = 0.4
        half_height = half_width * math.sqrt(3) / 2
        corners = [(0.0, half_height), (-half_width, -half_height), (half_width, -half_height)]
        for cx, cy in corners:
            angle = math.atan2(cy, cx)
            ox, oy = cx * inset * ortho_scale, cy * inset * ortho_scale
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
            bpy.context.collection.objects.unlink(txt_obj)
            scene.collection.objects.link(txt_obj)
            glyph_objs.append(txt_obj)
    else:
        label = glyph_label(value, glyph_style)
        font_size = _proportional_font_size(inradius / size_mm, label) * DECAL_FONT_CANVAS_SCALE
        bpy.ops.object.text_add(location=(0, 0, 0))
        txt_obj = bpy.context.active_object
        txt_obj.data.body = label
        font = _load_font(font_id, glyph_style)
        if font is not None:
            txt_obj.data.font = font
        txt_obj.data.align_x = 'CENTER'
        txt_obj.data.align_y = 'CENTER'
        txt_obj.data.size = font_size
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

    bpy.data.scenes.remove(scene)
