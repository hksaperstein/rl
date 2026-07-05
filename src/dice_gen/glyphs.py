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


def _largest_component_face_count(bm):
    """
    Size (in faces) of the largest connected shell in a bmesh. Used to detect
    when a boolean cut silently failed to engage the die's body at all (see
    _boolean_diff_apply): a genuine engrave cut always adds at least a few
    wall/floor faces to whatever it touches, so the largest connected shell's
    face count must strictly grow after a successful cut.
    """
    bm.faces.ensure_lookup_table()
    visited = set()
    best = 0
    for f in bm.faces:
        if f.index in visited:
            continue
        count = 0
        stack = [f]
        while stack:
            cur = stack.pop()
            if cur.index in visited:
                continue
            visited.add(cur.index)
            count += 1
            for e in cur.edges:
                for lf in e.link_faces:
                    if lf.index not in visited:
                        stack.append(lf)
        best = max(best, count)
    return best


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
    just silently ended up completely unengraved. This is instead caught by
    tracking the largest connected shell's face count: a real cut always
    grows the body it touches by at least a few wall/floor faces, so if the
    largest shell's face count fails to increase at all, EXACT has produced
    a no-op and we fall back to FLOAT exactly as in the collapse case.
    """
    bm_before = bmesh.new()
    bm_before.from_mesh(die_obj.data)
    volume_before = bm_before.calc_volume()
    largest_component_before = _largest_component_face_count(bm_before)

    mod = die_obj.modifiers.new(name="Engrave", type='BOOLEAN')
    mod.operation = 'DIFFERENCE'
    mod.object = cutter_obj
    mod.solver = 'EXACT'
    bpy.context.view_layer.objects.active = die_obj
    bpy.ops.object.modifier_apply(modifier=mod.name)

    bm_after = bmesh.new()
    bm_after.from_mesh(die_obj.data)
    volume_after = bm_after.calc_volume()
    largest_component_after = _largest_component_face_count(bm_after)
    bm_after.free()

    needs_retry = (
        (volume_before > 0 and volume_after < volume_before * 0.5)
        or (largest_component_after <= largest_component_before)
    )

    if needs_retry:
        # EXACT produced a degenerate/collapsed result, or silently no-op'd
        # without engaging the die's body at all -- restore the die's
        # pre-cut mesh from the snapshot and retry the identical cut with the
        # more tolerant FLOAT solver instead.
        bm_before.to_mesh(die_obj.data)
        die_obj.data.update()

        mod = die_obj.modifiers.new(name="Engrave", type='BOOLEAN')
        mod.operation = 'DIFFERENCE'
        mod.object = cutter_obj
        mod.solver = 'FLOAT'
        bpy.context.view_layer.objects.active = die_obj
        bpy.ops.object.modifier_apply(modifier=mod.name)

    bm_before.free()
    bpy.data.objects.remove(cutter_obj, do_unlink=True)


def apply_engraved_glyphs(die_obj, die_type, assignment, glyph_style, glyph_fill, font_id, size_mm):
    depth = size_mm * ENGRAVE_DEPTH_FRACTION
    glyph_font_size = size_mm * 0.18

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
            _weld_cutter_mesh(txt_obj)
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


def _socket_by_identifier(sockets, identifier):
    """
    ShaderNodeMix (data_type='RGBA') exposes several same-named sockets
    (Factor/A/B repeated per data type: float/vector/color/rotation), so
    looking them up by .name is ambiguous. Only the identifier is unique.
    """
    for socket in sockets:
        if socket.identifier == identifier:
            return socket
    raise KeyError(identifier)


def _wire_decal_texture_onto_material(mat, tex_node):
    """
    Composites the rendered glyph decal onto a face material's Base Color
    using the decal image's alpha channel, instead of overwriting Base Color
    outright. The decal images are rendered with a transparent background
    (see _render_label_to_image), so their non-glyph pixels are RGB (0,0,0)
    with alpha 0 -- wiring the texture's Color output straight into Base
    Color (the previous approach) made every decal-numbered face solid
    black except for the tiny glyph mark, hiding the die's actual material
    entirely. Mixing by alpha preserves the underlying material's
    color/pattern wherever there is no glyph ink, and stamps the glyph
    where alpha is 1.
    """
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    base_color_input = bsdf.inputs["Base Color"]

    existing_link = None
    for link in nt.links:
        if link.to_socket == base_color_input:
            existing_link = link
            break

    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = 'RGBA'

    a_color = _socket_by_identifier(mix.inputs, "A_Color")
    b_color = _socket_by_identifier(mix.inputs, "B_Color")
    factor = _socket_by_identifier(mix.inputs, "Factor_Float")
    result_color = _socket_by_identifier(mix.outputs, "Result_Color")

    if existing_link is not None:
        nt.links.new(existing_link.from_socket, a_color)
    else:
        a_color.default_value = tuple(base_color_input.default_value)

    nt.links.new(tex_node.outputs["Color"], b_color)
    nt.links.new(tex_node.outputs["Alpha"], factor)
    nt.links.new(result_color, base_color_input)


def apply_decal_glyphs(die_obj, die_type, assignment, glyph_style, font_id, size_mm, tmp_dir):
    bpy.context.view_layer.objects.active = die_obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(island_margin=0.05)
    bpy.ops.object.mode_set(mode='OBJECT')

    base_mat = die_obj.data.materials[0] if len(die_obj.data.materials) > 0 else None

    for face_index, value in assignment.items():
        image_path = os.path.join(tmp_dir, f"{die_obj.name}_face{face_index}.png")
        _render_label_to_image(value, glyph_style, image_path)

        if base_mat is not None:
            mat = base_mat.copy()
            mat.name = f"{die_obj.name}_face{face_index}_decal"
        else:
            mat = bpy.data.materials.new(name=f"{die_obj.name}_face{face_index}_decal")
            mat.use_nodes = True

        nt = mat.node_tree
        tex_node = nt.nodes.new("ShaderNodeTexImage")
        tex_node.image = bpy.data.images.load(image_path)
        _wire_decal_texture_onto_material(mat, tex_node)

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
