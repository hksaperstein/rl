"""
Applies numerals/pips to a die's faces, either as real engraved geometry
(boolean-cut into the mesh) or as printed texture decals (one material +
baked image per face). Both are real manufacturing conventions for TTRPG
dice, so both are supported and randomly chosen per asset by sampler.py.
"""
import os

import bpy
import bmesh
import numpy as np
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


def apply_engraved_glyphs(die_obj, die_type, assignment, glyph_style, glyph_fill, font_id, size_mm):
    depth = size_mm * ENGRAVE_DEPTH_FRACTION
    glyph_font_size = size_mm * 0.18
    warnings = []

    # Phase 1: compute every cut's (value, orientation) against the PRISTINE
    # mesh, entirely before any boolean modifier is applied. Each
    # bpy.ops.object.modifier_apply call below rebuilds die_obj.data's
    # topology (reindexing/reordering polygons), so face_index values from
    # `assignment` (captured once upfront by geometry.compute_opposite_face_pairs)
    # must never be re-resolved against die_obj.data.polygons after a cut.
    planned_cuts = []
    for face_index, value in assignment.items():
        face = die_obj.data.polygons[face_index]
        orient = _face_orientation_matrix(face, die_obj.matrix_world)
        planned_cuts.append((value, orient))

    # Phase 2: build and apply each cutter using only the precomputed
    # orientation matrices — no further indexing into die_obj.data.polygons.
    for value, orient in planned_cuts:
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
            txt_obj.data.align_x = 'CENTER'
            txt_obj.data.align_y = 'CENTER'
            txt_obj.data.size = glyph_font_size
            txt_obj.data.extrude = depth
            bpy.context.view_layer.objects.active = txt_obj
            bpy.ops.object.convert(target='MESH')
            _weld_cutter_mesh(txt_obj)
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
    bpy.context.view_layer.objects.active = die_obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(island_margin=0.05)
    bpy.ops.object.mode_set(mode='OBJECT')

    base_mat = die_obj.data.materials[0] if len(die_obj.data.materials) > 0 else None

    resolution = 256

    # The base material is identical across every face of a die, so its
    # appearance only needs to be rendered once, not per-face.
    swatch_path = None
    if base_mat is not None:
        swatch_path = _render_material_swatch(base_mat, resolution, tmp_dir, asset_id)

    for face_index, value in assignment.items():
        image_path = os.path.join(tmp_dir, f"{asset_id}_face{face_index}.png")
        _render_label_to_image(value, glyph_style, image_path, resolution=resolution)

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
    else:
        label = glyph_label(value, glyph_style)
        bpy.ops.object.text_add(location=(0, 0, 0))
        txt_obj = bpy.context.active_object
        txt_obj.data.body = label
        txt_obj.data.align_x = 'CENTER'
        txt_obj.data.align_y = 'CENTER'
        txt_obj.data.size = 1.0
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
