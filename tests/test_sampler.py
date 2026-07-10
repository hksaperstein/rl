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


def test_sample_set_is_reproducible_with_same_seed():
    a = sampler.sample_set(7)
    b = sampler.sample_set(7)
    assert a == b


def test_sample_set_has_exactly_the_expected_die_type_keys():
    for seed in range(10):
        variants = sampler.sample_set(seed)
        assert set(variants.keys()) == set(sampler.DIE_TYPES)
        assert set(variants.keys()) == {"d4", "d6", "d8", "d10", "d10_pct", "d12", "d20"}


def test_sample_set_glyph_style_is_never_pips():
    for seed in range(50):
        variants = sampler.sample_set(seed)
        for v in variants.values():
            assert v.glyph_style != "pips"


def test_sample_set_shares_attributes_across_all_die_types():
    shared_fields = [
        "material_category", "material_params", "glyph_method",
        "glyph_fill", "font_or_style_id", "bevel_fraction",
    ]
    for seed in range(20):
        variants = sampler.sample_set(seed)
        values = list(variants.values())
        first = values[0]
        for v in values[1:]:
            for field in shared_fields:
                assert getattr(v, field) == getattr(first, field)


def test_sample_set_size_mm_within_each_die_types_own_range():
    for seed in range(20):
        variants = sampler.sample_set(seed)
        for die_type, v in variants.items():
            lo, hi = sampler.SIZE_RANGES_MM[die_type]
            assert lo <= v.size_mm <= hi


def test_sample_set_d4_placement_only_set_for_d4():
    for seed in range(20):
        variants = sampler.sample_set(seed)
        for die_type, v in variants.items():
            if die_type == "d4":
                assert v.d4_placement in ("face_centered", "vertex_labeled")
            else:
                assert v.d4_placement is None


def test_sample_set_seed_field_matches_input_seed_for_all_dice():
    for seed in range(10):
        variants = sampler.sample_set(seed)
        for v in variants.values():
            assert v.seed == seed


def test_sample_variant_d10_pct_glyph_style_is_always_arabic_numerals():
    seen_d10_pct = False
    for seed in range(300):
        v = sampler.sample_variant(seed)
        if v.die_type == "d10_pct":
            seen_d10_pct = True
            assert v.glyph_style == "arabic_numerals"
    assert seen_d10_pct, "expected at least one d10_pct sample across 300 seeds"


def test_sample_set_d10_pct_glyph_style_is_always_arabic_numerals():
    for seed in range(50):
        variants = sampler.sample_set(seed)
        assert variants["d10_pct"].glyph_style == "arabic_numerals"
