"""Sim-independent unit tests for tasks/ar4/fk_verification.py's standing
FK-based verification framework (Layer 1: asset-geometry check; Layer 2:
control-intent/task-invariant check). Pure numpy, no isaaclab/torch import -
run via plain `python3 -m pytest tests/test_ar4_fk_verification.py -v` (no
Isaac Sim/desktop dependency needed for this module specifically, unlike
tests/test_franka_antipodal_grasp_reward.py's torch-only tests - confirmed
by running these directly on the Pi during TDD).

TDD history (written before implementation, per this repo's convention):
these tests were written first, confirmed failing against a stub, then
tasks/ar4/fk_verification.py was implemented until all green. The jaw-sign
convention below was corrected once more the same day (2026-07-23) after a
live Isaac Sim integration run (scripts/_verify_gripper_fk_integration.py,
on a fresh GCP cloud build of the current asset) showed the framework's
first-draft calibration itself no longer matched reality - see
tasks/ar4/fk_verification.py's own "IMPORTANT" docstring section for the
full history of that correction.

The single most important test here is TestJawMirroringRegression - a
direct, concrete demonstration that this framework catches an incorrect
gripper jaw sign convention: this project's history has used BOTH
same-sign and opposite-sign jaw1/jaw2 commands at different points
(kb/wiki/concepts/ar4-vs-franka-root-cause-comparison.md's 2026-07-21 and
2026-07-23 UPDATEs), and a live measurement on the CURRENT asset
(2026-07-23) confirmed SAME-SIGN commanding is the one that produces the
correct ~28mm real-world separation, while OPPOSITE-SIGN commanding (the
now-superseded 2026-07-21 fix) collapses both jaws onto the same point.
"""

import numpy as np
import pytest

from tasks.ar4.fk_verification import (
    DEFAULT_JOINT_TABLE,
    assert_gripper_separation,
    assert_link_pose_matches_vendor_fk,
    compute_link_pose_from_joint_values,
    with_corrupted_origin,
)

# Mirrors tasks/ar4/robot_cfg.py's own GRIPPER_OPEN_POS/GRIPPER_CLOSED_POS
# constants (not imported directly - that module requires an Isaac
# Sim/Isaac Lab AppLauncher to already exist before it can be imported,
# per its own module docstring, which this pure-numpy test file must not
# require).
GRIPPER_OPEN_POS = 0.014
GRIPPER_CLOSED_POS = 0.0


class TestHandComputableReferenceConfiguration:
    """(a) FK chain vs. a known/hand-computable reference configuration."""

    def test_link_1_pose_at_all_zero_joint_values(self):
        # Single hop from base_link: joint_1's <origin xyz="0 0 0.092"
        # rpy="pi 0 0"/> with q=0 (revolute, so the joint's own motion
        # contributes no rotation) - the resulting pose IS the origin
        # transform itself. rpy=(pi,0,0) is a pure 180-degree rotation
        # about X, whose quaternion (w,x,y,z) is hand-computable directly:
        # (cos(pi/2), sin(pi/2), 0, 0) = (0, 1, 0, 0).
        pos, quat = compute_link_pose_from_joint_values({}, "link_1")
        assert pos == pytest.approx([0.0, 0.0, 0.092], abs=1e-9)
        assert quat == pytest.approx([0.0, 1.0, 0.0, 0.0], abs=1e-9)

    def test_gripper_jaws_coincide_when_fully_closed(self):
        # Both gripper_jaw1_joint and gripper_jaw2_joint share the IDENTICAL
        # origin_xyz=(0, -0.036, 0) in the raw URDF (urdf/ar_gripper_macro.xacro).
        # At q=0 for both (GRIPPER_CLOSED_POS), each jaw's translation
        # contribution from its own axis is exactly zero regardless of
        # axis-sign convention - so the two jaw links MUST land at the
        # exact same 3D position. This is a convention-agnostic,
        # hand-computable sanity check of the whole chain (arm joints left
        # at their own default 0.0 too - separation is invariant to the
        # common upstream arm transform, see fk_verification.py's own
        # assert_gripper_separation docstring).
        joint_values = {
            "gripper_jaw1_joint": GRIPPER_CLOSED_POS,
            "gripper_jaw2_joint": GRIPPER_CLOSED_POS,
        }
        p1, _ = compute_link_pose_from_joint_values(joint_values, "gripper_jaw1_link")
        p2, _ = compute_link_pose_from_joint_values(joint_values, "gripper_jaw2_link")
        assert p1 == pytest.approx(p2, abs=1e-9)


class TestJawMirroringRegression:
    """(b) The concrete demonstration this framework catches an incorrect
    gripper jaw sign convention: the CURRENT, live-verified-correct
    SAME-sign commanding PASSES (~28mm, matching
    tasks/ar4/objects_cfg.py's own documented "~28mm max aperture"), and
    the now-superseded 2026-07-21 OPPOSITE-sign convention FAILS
    (near-zero separation - both jaws collapse onto the same point)."""

    def test_same_sign_convention_passes_separation_check(self):
        # tasks/ar4/robot_cfg.py's CURRENT convention (as of the 2026-07-23
        # correction, commit d59595a): both jaws commanded to the IDENTICAL
        # +0.014 value. Live-verified 2026-07-23 via
        # scripts/_verify_gripper_fk_integration.py on a fresh cloud build
        # of the current asset: real measured world-frame separation
        # 27.996mm.
        same_sign_open = {
            "gripper_jaw1_joint": GRIPPER_OPEN_POS,
            "gripper_jaw2_joint": GRIPPER_OPEN_POS,
        }
        separation_mm = assert_gripper_separation(same_sign_open, min_mm=20.0, max_mm=36.0)
        assert separation_mm == pytest.approx(28.0, abs=1.0)

    def test_same_sign_convention_closed_state_still_coincides(self):
        same_sign_closed = {
            "gripper_jaw1_joint": GRIPPER_CLOSED_POS,
            "gripper_jaw2_joint": GRIPPER_CLOSED_POS,
        }
        p1, _ = compute_link_pose_from_joint_values(same_sign_closed, "gripper_jaw1_link")
        p2, _ = compute_link_pose_from_joint_values(same_sign_closed, "gripper_jaw2_link")
        assert p1 == pytest.approx(p2, abs=1e-9)

    def test_opposite_sign_convention_fails_separation_check(self):
        # The now-superseded 2026-07-21 fix (commit 928af41): jaw2
        # commanded to jaw1's negation. Found wrong the same day
        # (2026-07-23) it was superseded by a direct sweep
        # (scripts/_sweep_jaw2_symmetry.py) - this double-negates jaw2's
        # own already-flipped local-to-world mapping and collapses both
        # jaws onto the same point.
        opposite_sign_open = {
            "gripper_jaw1_joint": GRIPPER_OPEN_POS,
            "gripper_jaw2_joint": -GRIPPER_OPEN_POS,
        }
        with pytest.raises(AssertionError, match="separation"):
            assert_gripper_separation(opposite_sign_open, min_mm=20.0, max_mm=36.0)

    def test_opposite_sign_convention_separation_is_near_zero(self):
        # Direct measurement (not just "the assertion raised"): confirm the
        # actual predicted separation really is near-zero, not just some
        # other out-of-range value.
        opposite_sign_open = {
            "gripper_jaw1_joint": GRIPPER_OPEN_POS,
            "gripper_jaw2_joint": -GRIPPER_OPEN_POS,
        }
        p1, _ = compute_link_pose_from_joint_values(opposite_sign_open, "gripper_jaw1_link")
        p2, _ = compute_link_pose_from_joint_values(opposite_sign_open, "gripper_jaw2_link")
        separation_mm = float(np.linalg.norm(p1 - p2) * 1000.0)
        assert separation_mm == pytest.approx(0.0, abs=1e-6)


class TestCorruptedOriginIsCaught:
    """(c) A deliberately-corrupted joint origin (an import-style asset
    defect, distinct from the control-intent jaw-mirroring bug above) is
    caught by Layer 1's assert_link_pose_matches_vendor_fk."""

    def test_matching_table_passes(self):
        joint_values = {}
        live_pos, live_quat = compute_link_pose_from_joint_values(joint_values, "link_2")
        result = assert_link_pose_matches_vendor_fk(live_pos, live_quat, joint_values, "link_2", tolerance_mm=1.0)
        assert result.passed
        assert result.pos_discrepancy_mm == pytest.approx(0.0, abs=1e-6)

    def test_corrupted_joint_2_origin_is_caught(self):
        joint_values = {}
        # "Live" ground truth: computed from the correct, uncorrupted table
        # (standing in for a real Isaac-Sim-reported link pose).
        live_pos, live_quat = compute_link_pose_from_joint_values(joint_values, "link_2", DEFAULT_JOINT_TABLE)

        # An independently-corrupted copy of the vendor table - a 50mm
        # import-style defect in joint_2's own origin (e.g. a bad
        # URDF-to-USD unit conversion or a transcription error), NOT
        # touching the correct DEFAULT_JOINT_TABLE used to produce live_pos
        # above.
        corrupted_table = with_corrupted_origin(DEFAULT_JOINT_TABLE, "joint_2", delta_xyz=(0.05, 0.0, 0.0))

        with pytest.raises(AssertionError, match="FK mismatch"):
            assert_link_pose_matches_vendor_fk(
                live_pos, live_quat, joint_values, "link_2", tolerance_mm=1.0, joint_table=corrupted_table
            )

    def test_corrupted_joint_2_origin_reports_expected_discrepancy(self):
        joint_values = {}
        live_pos, live_quat = compute_link_pose_from_joint_values(joint_values, "link_2", DEFAULT_JOINT_TABLE)
        corrupted_table = with_corrupted_origin(DEFAULT_JOINT_TABLE, "joint_2", delta_xyz=(0.05, 0.0, 0.0))

        try:
            assert_link_pose_matches_vendor_fk(
                live_pos, live_quat, joint_values, "link_2", tolerance_mm=1.0, joint_table=corrupted_table
            )
            pytest.fail("expected AssertionError")
        except AssertionError as exc:
            assert "50.0" in str(exc) or "50.00" in str(exc) or "49." in str(exc) or "50" in str(exc)
