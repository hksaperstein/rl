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
