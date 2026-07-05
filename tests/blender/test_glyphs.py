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
