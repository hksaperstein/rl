import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate import merge_d10, eval_real, MERGED_CLASS_NAMES


def test_merge_maps_d10_pct_onto_d10():
    # CLASS_NAMES: d4=0 d6=1 d8=2 d10=3 d10_pct=4 d12=5 d20=6
    assert merge_d10(4) == 3
    assert [merge_d10(i) for i in [0, 1, 2, 3, 5, 6]] == [0, 1, 2, 3, 4, 5]


def test_merged_class_names():
    assert MERGED_CLASS_NAMES == ["d4", "d6", "d8", "d10", "d12", "d20"]


def test_eval_real_perfect_predictions_score_1():
    # one image, one d20 GT box (merged-space index 5, MERGED_CLASS_NAMES[5]
    # == "d20"); prediction identical with conf 0.9
    gt = [{"image_id": 1, "class_idx": 5, "bbox": [10, 10, 50, 50],
           "width": 640, "height": 480}]
    preds = [{"image_id": 1, "class_idx": 5, "bbox": [10, 10, 50, 50],
              "score": 0.9}]
    per_class = eval_real(preds, gt, MERGED_CLASS_NAMES)
    map50, map5095 = per_class["d20"]
    assert map50 == pytest.approx(1.0)
    assert map5095 == pytest.approx(1.0)


def test_eval_real_merges_pct_prediction_onto_d10_gt():
    # GT says d10 (merged idx 3); model predicted d10_pct (raw idx 4).
    # After merge they must count as a match.
    gt = [{"image_id": 1, "class_idx": 3, "bbox": [10, 10, 50, 50],
           "width": 640, "height": 480}]
    preds = [{"image_id": 1, "class_idx": merge_d10(4),
              "bbox": [10, 10, 50, 50], "score": 0.9}]
    per_class = eval_real(preds, gt, MERGED_CLASS_NAMES)
    assert per_class["d10"][0] == pytest.approx(1.0)


def test_eval_real_empty_dets_scores_zero_without_crashing():
    # No predictions at all (e.g. model detects nothing on the whole test
    # set). COCOeval can't be constructed with a None detections object, so
    # eval_real must special-case this and return 0.0 for every class
    # instead of crashing.
    gt = [{"image_id": 1, "class_idx": 5, "bbox": [10, 10, 50, 50],
           "width": 640, "height": 480}]
    per_class = eval_real([], gt, MERGED_CLASS_NAMES)
    assert per_class == {name: (0.0, 0.0) for name in MERGED_CLASS_NAMES}
