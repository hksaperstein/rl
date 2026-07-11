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


def test_validate_flags_non_empty_engraving_warnings(tmp_path):
    _write_manifest(tmp_path, [{
        "asset_id": "a1", "die_type": "d6", "num_sides": 6,
        "usd_path": "a1.usd", "thumbnail_path": "a1_thumb.png",
        "engraving_warnings": [
            "a1_d6_die: a glyph cut was skipped entirely -- both EXACT and "
            "FLOAT solvers produced a collapsed or debris-laden result for "
            "this cutter; this die is missing one numeral/pip as a result."
        ],
    }])
    open(os.path.join(tmp_path, "a1.usd"), "w").write("x")
    open(os.path.join(tmp_path, "a1_thumb.png"), "w").close()

    errors = validate(str(tmp_path))
    assert any("a1" in e and "glyph cut was skipped" in e for e in errors), errors


def test_validate_does_not_flag_empty_engraving_warnings(tmp_path):
    _write_manifest(tmp_path, [{
        "asset_id": "a1", "die_type": "d6", "num_sides": 6,
        "usd_path": "a1.usd", "thumbnail_path": "a1_thumb.png",
        "engraving_warnings": [],
    }])
    open(os.path.join(tmp_path, "a1.usd"), "w").write("x")
    open(os.path.join(tmp_path, "a1_thumb.png"), "w").close()

    errors = validate(str(tmp_path))
    assert errors == []


def test_validate_flags_missing_blend_file(tmp_path):
    _write_manifest(tmp_path, [{
        "asset_id": "a1", "die_type": "d6", "num_sides": 6,
        "usd_path": "a1.usd", "thumbnail_path": "a1_thumb.png",
        "blend_path": "a1.blend", "stl_path": "a1.stl",
    }])
    open(os.path.join(tmp_path, "a1.usd"), "w").write("x")
    open(os.path.join(tmp_path, "a1_thumb.png"), "w").close()
    open(os.path.join(tmp_path, "a1.stl"), "w").write("x")
    # a1.blend deliberately not created

    errors = validate(str(tmp_path))
    assert any("missing .blend file" in e for e in errors), errors


def test_validate_flags_missing_stl_file(tmp_path):
    _write_manifest(tmp_path, [{
        "asset_id": "a1", "die_type": "d6", "num_sides": 6,
        "usd_path": "a1.usd", "thumbnail_path": "a1_thumb.png",
        "blend_path": "a1.blend", "stl_path": "a1.stl",
    }])
    open(os.path.join(tmp_path, "a1.usd"), "w").write("x")
    open(os.path.join(tmp_path, "a1_thumb.png"), "w").close()
    open(os.path.join(tmp_path, "a1.blend"), "w").write("x")
    # a1.stl deliberately not created

    errors = validate(str(tmp_path))
    assert any("missing STL file" in e for e in errors), errors


def test_validate_passes_when_blend_and_stl_present(tmp_path):
    _write_manifest(tmp_path, [{
        "asset_id": "a1", "die_type": "d6", "num_sides": 6,
        "usd_path": "a1.usd", "thumbnail_path": "a1_thumb.png",
        "blend_path": "a1.blend", "stl_path": "a1.stl",
    }])
    open(os.path.join(tmp_path, "a1.usd"), "w").write("x")
    open(os.path.join(tmp_path, "a1_thumb.png"), "w").close()
    open(os.path.join(tmp_path, "a1.blend"), "w").write("x")
    open(os.path.join(tmp_path, "a1.stl"), "w").write("x")

    errors = validate(str(tmp_path))
    assert errors == []


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


def _set_record(tmp_path, asset_id, die_type, set_id):
    usd_name = f"{asset_id}.usd"
    thumb_name = f"{asset_id}_thumb.png"
    open(os.path.join(tmp_path, usd_name), "w").write("x")
    open(os.path.join(tmp_path, thumb_name), "w").close()
    num_sides = {"d4": 4, "d6": 6, "d8": 8, "d10": 10, "d10_pct": 10, "d12": 12, "d20": 20}[die_type]
    return {
        "asset_id": asset_id, "die_type": die_type, "num_sides": num_sides,
        "usd_path": usd_name, "thumbnail_path": thumb_name, "set_id": set_id,
    }


def test_validate_passes_for_complete_set():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_path:
        die_types = ["d4", "d6", "d8", "d10", "d10_pct", "d12", "d20"]
        records = [
            _set_record(tmp_path, f"set_00000_{dt}", dt, "set_00000")
            for dt in die_types
        ]
        _write_manifest(tmp_path, records)

        errors = validate(tmp_path)
        assert errors == []


def test_validate_reports_missing_die_type_in_set():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_path:
        die_types = ["d4", "d6", "d8", "d10", "d12"]  # missing d20
        records = [
            _set_record(tmp_path, f"set_00000_{dt}", dt, "set_00000")
            for dt in die_types
        ]
        _write_manifest(tmp_path, records)

        errors = validate(tmp_path)
        set_errors = [e for e in errors if "set_00000" in e]
        assert len(set_errors) == 1
        assert "missing die types" in set_errors[0]
        assert "d20" in set_errors[0]


def test_validate_reports_missing_percentile_die_in_set():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_path:
        die_types = ["d4", "d6", "d8", "d10", "d12", "d20"]  # missing d10_pct
        records = [
            _set_record(tmp_path, f"set_00000_{dt}", dt, "set_00000")
            for dt in die_types
        ]
        _write_manifest(tmp_path, records)

        errors = validate(tmp_path)
        set_errors = [e for e in errors if "set_00000" in e]
        assert len(set_errors) == 1
        assert "missing die types" in set_errors[0]
        assert "d10_pct" in set_errors[0]
