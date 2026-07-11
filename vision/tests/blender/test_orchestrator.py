import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from _harness import run_and_report


def test_generate_batch_produces_manifest_and_assets():
    from datagen.domains.dice import orchestrator

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
            assert record["die_type"] in ("d4", "d6", "d8", "d10", "d10_pct", "d12", "d20")
            assert isinstance(record.get("engraving_warnings"), list), (
                f"{record['asset_id']}: expected an engraving_warnings list "
                f"in every manifest record (empty for decal-method or "
                f"cleanly-engraved dice), got {record.get('engraving_warnings')!r}"
            )

        failures_path = os.path.join(outdir, "failures.json")
        assert os.path.exists(failures_path)


def test_generate_set_batch_produces_matching_set():
    from datagen.domains.dice import orchestrator

    with tempfile.TemporaryDirectory() as outdir:
        generated, failed = orchestrator.generate_set_batch(num_sets=1, seed=2000, outdir=outdir)

        assert generated + failed == 7

        manifest_path = os.path.join(outdir, "manifest.json")
        with open(manifest_path) as f:
            manifest = json.load(f)
        assert len(manifest) == generated

        set_ids = {record["set_id"] for record in manifest}
        assert len(set_ids) == 1

        die_types = {record["die_type"] for record in manifest}
        assert die_types == {"d4", "d6", "d8", "d10", "d10_pct", "d12", "d20"}

        material_categories = {record["material_category"] for record in manifest}
        font_ids = {record["font_or_style_id"] for record in manifest}
        assert len(material_categories) == 1
        assert len(font_ids) == 1

        # d10_pct is always arabic; d10 falls back to arabic when the set's
        # shared style is roman/greek (neither numeral system has a zero,
        # and d10's values include 0 -- see sampler.sample_set). The
        # remaining 5 dice always share exactly one sampled style.
        shared_styles = {
            record["glyph_style"] for record in manifest
            if record["die_type"] not in ("d10", "d10_pct")
        }
        assert len(shared_styles) == 1, (
            f"expected d4/d6/d8/d12/d20 to share one glyph_style, got {shared_styles}"
        )
        shared = next(iter(shared_styles))
        pct_record = next(r for r in manifest if r["die_type"] == "d10_pct")
        assert pct_record["glyph_style"] == "arabic_numerals"
        d10_record = next(r for r in manifest if r["die_type"] == "d10")
        if shared in ("roman_numerals", "greek_numerals"):
            assert d10_record["glyph_style"] == "arabic_numerals"
        else:
            assert d10_record["glyph_style"] == shared


def run():
    test_generate_batch_produces_manifest_and_assets()
    test_generate_set_batch_produces_matching_set()


run_and_report(run)
