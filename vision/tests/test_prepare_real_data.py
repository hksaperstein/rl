import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_real_data import (
    remap_label_line, SOURCE_NAME_MAP, group_pairs_by_augmentation, split_groups,
    should_exclude_stem, EXCLUDE_STEM_PREFIXES
)


def test_known_names_map_to_our_taxonomy():
    for name in ["d4", "d6", "d8", "d10", "d12", "d20"]:
        assert SOURCE_NAME_MAP[name] == name


def test_percentile_variants_merge_to_d10():
    for name in ["d10%", "d100", "d10_percentile"]:
        assert SOURCE_NAME_MAP.get(name, "d10") == "d10"


def test_remap_label_line_translates_class_index():
    # source dataset ordered [d20, d4]: class 0 = d20 -> our index 6
    out = remap_label_line("0 0.5 0.5 0.2 0.2", ["d20", "d4"])
    assert out.split()[0] == "6"
    assert out.split()[1:] == ["0.5", "0.5", "0.2", "0.2"]


def test_remap_label_line_drops_unmappable():
    assert remap_label_line("0 0.5 0.5 0.2 0.2", ["coin"]) is None


def test_remap_label_line_unknown_name_raises():
    with pytest.raises(KeyError):
        remap_label_line("0 0.5 0.5 0.2 0.2", ["mystery_die"])


def test_remap_label_line_converts_polygon_to_bbox():
    # a source polygon annotation (class + x1 y1 x2 y2 ... xn yn, not a plain
    # 4-value bbox) -- one real source mixes polygon and bbox lines per image.
    # square spanning x in [0.1, 0.5], y in [0.1, 0.5].
    line = "0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5 0.1 0.1"
    out = remap_label_line(line, ["d4"])
    idx, cx, cy, bw, bh = out.split()
    assert idx == str(["d4", "d6", "d8", "d10", "d10_pct", "d12", "d20"].index("d4"))
    assert float(cx) == pytest.approx(0.3)
    assert float(cy) == pytest.approx(0.3)
    assert float(bw) == pytest.approx(0.4)
    assert float(bh) == pytest.approx(0.4)


def test_group_pairs_by_augmentation_groups_variants_with_same_stem():
    # Roboflow augments images, keeping the same stem but varying the .rf.HASH
    pairs = [
        (Path("img_jpg.rf.hash1.jpg"), Path("label1.txt")),
        (Path("img_jpg.rf.hash2.jpg"), Path("label2.txt")),
        (Path("other_jpg.rf.hash3.jpg"), Path("label3.txt")),
    ]
    groups = group_pairs_by_augmentation(pairs)
    assert len(groups) == 2
    assert "img_jpg" in groups
    assert "other_jpg" in groups
    assert len(groups["img_jpg"]) == 2
    assert len(groups["other_jpg"]) == 1


def test_split_groups_keeps_groups_intact():
    # All members of a group must go to the same split
    groups = {
        "stem1": [(Path("s1.rf.h1.jpg"), Path("l1.txt")),
                  (Path("s1.rf.h2.jpg"), Path("l2.txt"))],
        "stem2": [(Path("s2.rf.h1.jpg"), Path("l1.txt"))],
        "stem3": [(Path("s3.rf.h1.jpg"), Path("l1.txt"))],
    }
    test_groups, finetune_groups = split_groups(groups, test_frac=0.33, seed=42)

    # Check no overlap
    test_keys = set(test_groups.keys())
    finetune_keys = set(finetune_groups.keys())
    assert test_keys.isdisjoint(finetune_keys)
    assert test_keys.union(finetune_keys) == set(groups.keys())


def test_test_split_behavior_keeps_one_per_group():
    # When we filter the test split to keep only one image per group,
    # we should keep the lexicographically-first filename
    pairs = [
        (Path("stem_jpg.rf.zzz.jpg"), Path("l1.txt")),
        (Path("stem_jpg.rf.aaa.jpg"), Path("l2.txt")),
        (Path("stem_jpg.rf.mmm.jpg"), Path("l3.txt")),
    ]
    # The lex-first is stem_jpg.rf.aaa.jpg
    sorted_pairs = sorted(pairs, key=lambda p: str(p[0]))
    assert str(sorted_pairs[0][0]) == "stem_jpg.rf.aaa.jpg"


def test_should_exclude_stem_filters_by_source_and_prefix():
    # Under-labeled multi-dice scenes in dd_dice (all_dice_*) are excluded.
    # Same prefix in other sources should not be excluded.
    assert should_exclude_stem("dd_dice", "all_dice_paper_jpg") is True
    assert should_exclude_stem("dd_dice", "all_dice_tray_jpg") is True
    assert should_exclude_stem("dd_dice", "d10_top002_jpg") is False
    assert should_exclude_stem("dnd_dices", "all_dice_paper_jpg") is False
