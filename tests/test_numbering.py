import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dice_gen import numbering


def test_get_values_d20_has_20_unique_values_1_through_20():
    values = numbering.get_values("d20")
    assert len(values) == 20
    assert set(values) == set(range(1, 21))


def test_get_values_d10_has_10_unique_values_0_through_9():
    values = numbering.get_values("d10")
    assert len(values) == 10
    assert set(values) == set(range(0, 10))


def test_d6_opposite_faces_sum_to_7():
    face_pairs = [(0, 1), (2, 3), (4, 5)]
    assignment = numbering.assign_values_to_opposite_pairs("d6", face_pairs)
    assert numbering.verify_opposite_sum("d6", face_pairs, assignment)
    assert set(assignment.values()) == {1, 2, 3, 4, 5, 6}


def test_d20_opposite_faces_sum_to_21():
    face_pairs = [(i, i + 10) for i in range(10)]
    assignment = numbering.assign_values_to_opposite_pairs("d20", face_pairs)
    assert numbering.verify_opposite_sum("d20", face_pairs, assignment)
    assert set(assignment.values()) == set(range(1, 21))


def test_d12_opposite_faces_sum_to_13():
    face_pairs = [(i, i + 6) for i in range(6)]
    assignment = numbering.assign_values_to_opposite_pairs("d12", face_pairs)
    assert numbering.verify_opposite_sum("d12", face_pairs, assignment)
    assert set(assignment.values()) == set(range(1, 13))


def test_d10_opposite_faces_sum_to_9():
    face_pairs = [(i, i + 5) for i in range(5)]
    assignment = numbering.assign_values_to_opposite_pairs("d10", face_pairs)
    assert numbering.verify_opposite_sum("d10", face_pairs, assignment)
    assert set(assignment.values()) == set(range(0, 10))


def test_d4_has_no_opposite_sum_rule_but_assigns_all_values_once():
    face_pairs = [(0, 1), (2, 3)]
    assignment = numbering.assign_values_to_opposite_pairs("d4", face_pairs)
    assert set(assignment.values()) == {1, 2, 3, 4}


def test_get_values_d10_pct_has_10_unique_values_multiples_of_ten():
    values = numbering.get_values("d10_pct")
    assert len(values) == 10
    assert set(values) == {0, 10, 20, 30, 40, 50, 60, 70, 80, 90}


def test_d10_pct_opposite_faces_sum_to_90():
    face_pairs = [(i, i + 5) for i in range(5)]
    assignment = numbering.assign_values_to_opposite_pairs("d10_pct", face_pairs)
    assert numbering.verify_opposite_sum("d10_pct", face_pairs, assignment)
    assert set(assignment.values()) == {0, 10, 20, 30, 40, 50, 60, 70, 80, 90}


def test_d10_pct_assignment_is_d10_assignment_scaled_by_ten():
    face_pairs = [(i, i + 5) for i in range(5)]
    d10_assignment = numbering.assign_values_to_opposite_pairs("d10", face_pairs)
    pct_assignment = numbering.assign_values_to_opposite_pairs("d10_pct", face_pairs)
    assert pct_assignment == {face: value * 10 for face, value in d10_assignment.items()}


def test_d10_pct_assignment_respects_hemisphere_parity_split():
    face_pairs = [(i, i + 5) for i in range(5)]
    hemisphere_of_face = {
        0: "top", 1: "top", 2: "top", 3: "top", 4: "top",
        5: "bottom", 6: "bottom", 7: "bottom", 8: "bottom", 9: "bottom",
    }
    d10_assignment = numbering.assign_values_to_opposite_pairs(
        "d10", face_pairs, hemisphere_of_face=hemisphere_of_face,
    )
    pct_assignment = numbering.assign_values_to_opposite_pairs(
        "d10_pct", face_pairs, hemisphere_of_face=hemisphere_of_face,
    )
    assert pct_assignment == {face: value * 10 for face, value in d10_assignment.items()}
    assert numbering.verify_opposite_sum("d10_pct", face_pairs, pct_assignment)
    assert len(set(pct_assignment.values())) == 10, "expected 10 unique values, got duplicates"
    assert set(pct_assignment.values()) == {0, 10, 20, 30, 40, 50, 60, 70, 80, 90}
