"""
Standard real-world face-numbering conventions for each die type.

d4 (tetrahedron) has no face-to-face antipodal relationship (its faces are
opposite a *vertex*, not another face), so it has no opposite_sum rule —
values are just assigned once each. All other die types are centrally
symmetric and follow their standard convention:
  d6:      opposite faces sum to 7
  d8:      opposite faces sum to 9
  d10:     opposite faces sum to 9 (values 0-9, pairing k with 9-k)
  d10_pct: opposite faces sum to 90 (percentile/tens die, values
           0,10,...,90 -- same physical mold as d10, different digits
           printed on each face; see assign_values_to_opposite_pairs'
           d10_pct special case for why this scales d10's own assignment
           rather than running through the generic scheme below)
  d12:     opposite faces sum to 13
  d20:     opposite faces sum to 21
"""

NUMBERING_SCHEMES = {
    "d4": {"values": [1, 2, 3, 4], "opposite_sum": None},
    "d6": {"values": [1, 2, 3, 4, 5, 6], "opposite_sum": 7},
    "d8": {"values": [1, 2, 3, 4, 5, 6, 7, 8], "opposite_sum": 9},
    "d10": {"values": list(range(0, 10)), "opposite_sum": 9},
    "d10_pct": {"values": [v * 10 for v in range(0, 10)], "opposite_sum": 90},
    "d12": {"values": list(range(1, 13)), "opposite_sum": 13},
    "d20": {"values": list(range(1, 21)), "opposite_sum": 21},
}


def get_values(die_type):
    return list(NUMBERING_SCHEMES[die_type]["values"])


def d4_vertex_values(num_vertices=4):
    """
    Real vertex-read d4 convention: values belong to VERTICES, not faces.
    Each face displays the values of its own 3 corners (3 different
    numbers per face), and all 3 faces meeting at a vertex show that
    vertex's value at their shared corner -- the rolled result is read at
    the apex vertex. Which value lands on which physical vertex is not
    standardized across manufacturers, so vertex-index order is used
    deterministically.

    Returns {vertex_index: value}.
    """
    return {vi: vi + 1 for vi in range(num_vertices)}


def assign_values_to_opposite_pairs(die_type, face_pairs, hemisphere_of_face=None):
    """
    face_pairs: list of (face_index_a, face_index_b) tuples covering every
    face exactly once. For die types with an opposite_sum rule, each pair is
    assigned (v, opposite_sum - v) so the invariant holds. For d4 (no rule),
    values are just handed out in iteration order — face_pairs there is only
    a convenient grouping, not a real geometric antipodal relationship.

    hemisphere_of_face: optional {face_index: "top"|"bottom"}, for d8/d10.
    Real d8/d10 dice show a consistent odd/even split by hemisphere (every
    face touching one pole shows one parity, every face touching the other
    pole shows the other) -- confirmed this session to require an explicit
    fix, since the plain min(remaining)-to-face_a assignment below has no
    hemisphere awareness and produces an arbitrary, inconsistent split.
    Every scheme's opposite_sum here is odd, so every antipodal pair always
    has exactly one odd and one even value -- this makes "assign the
    even one to whichever face is 'top', the odd one to whichever is
    'bottom'" always well-defined, for every pair, with no exceptions.
    When None (default), behavior is exactly as before this parameter
    existed.

    Returns {face_index: value}.
    """
    if die_type == "d10_pct":
        # Same physical mold as d10, different digits printed on each
        # face. d10_pct's own values (0,10,...,90) are all even, so
        # running them directly through the generic even/odd hemisphere
        # split below would find no parity variance to split on. Instead,
        # reuse d10's own assignment (which already solves the real
        # hemisphere-parity problem on the underlying 0-9 digit) and
        # scale every value x10 for display.
        base_assignment = assign_values_to_opposite_pairs(
            "d10", face_pairs, hemisphere_of_face=hemisphere_of_face,
        )
        return {face: value * 10 for face, value in base_assignment.items()}

    scheme = NUMBERING_SCHEMES[die_type]
    values = scheme["values"]
    opposite_sum = scheme["opposite_sum"]

    if opposite_sum is None:
        flat = [face for pair in face_pairs for face in pair]
        return {face: value for face, value in zip(flat, values)}

    remaining = set(values)
    assignment = {}
    for face_a, face_b in face_pairs:
        v_a = min(remaining)
        v_b = opposite_sum - v_a
        if v_b not in remaining:
            raise ValueError(
                f"{die_type}: cannot satisfy opposite_sum={opposite_sum} "
                f"with remaining values {sorted(remaining)}"
            )
        remaining.discard(v_a)
        remaining.discard(v_b)

        if hemisphere_of_face is None:
            assignment[face_a] = v_a
            assignment[face_b] = v_b
        else:
            even_value, odd_value = (v_a, v_b) if v_a % 2 == 0 else (v_b, v_a)
            for face in (face_a, face_b):
                assignment[face] = (
                    even_value if hemisphere_of_face[face] == "top" else odd_value
                )
    return assignment


def verify_opposite_sum(die_type, face_pairs, assignment):
    opposite_sum = NUMBERING_SCHEMES[die_type]["opposite_sum"]
    if opposite_sum is None:
        return True
    return all(
        assignment[a] + assignment[b] == opposite_sum for a, b in face_pairs
    )
