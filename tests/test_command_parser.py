"""Pure-stdlib unit tests for tasks/franka/command_parser.py. Run via:
python3 -m pytest tests/test_command_parser.py -v
(no Isaac Sim python needed — this module is pure stdlib with zero external dependencies,
unlike other tests/ files in this repo that require /home/saps/IsaacLab/isaaclab.sh)."""

import pytest

from tasks.franka.command_parser import parse_dice_command


class TestValidCommands:
    """Valid, unambiguous commands that should parse successfully."""

    def test_docstring_example_1_pick_up_and_roll_d20(self):
        result = parse_dice_command("pick up and roll the d20")
        assert result == {"action": "roll", "target_shape": "d20"}

    def test_docstring_example_2_pick_up_and_move_d12(self):
        result = parse_dice_command("pick up and move the d12")
        assert result == {"action": "move", "target_shape": "d12"}

    def test_docstring_example_3_move_d8_to_goal(self):
        result = parse_dice_command("move the d8 to the goal")
        assert result == {"action": "move", "target_shape": "d8"}

    def test_docstring_example_4_roll_d10(self):
        result = parse_dice_command("roll the d10")
        assert result == {"action": "roll", "target_shape": "d10"}

    def test_docstring_example_5_please_roll_d4(self):
        result = parse_dice_command("please pick up and roll die d4")
        assert result == {"action": "roll", "target_shape": "d4"}

    def test_docstring_example_6_move_d100_uppercase(self):
        result = parse_dice_command("Move D100 over there")
        assert result == {"action": "move", "target_shape": "d10"}

    def test_case_insensitive_ROLL_uppercase(self):
        result = parse_dice_command("ROLL the D20")
        assert result == {"action": "roll", "target_shape": "d20"}

    def test_case_insensitive_mixed_case(self):
        result = parse_dice_command("MoVe ThE d8")
        assert result == {"action": "move", "target_shape": "d8"}

    def test_extra_whitespace(self):
        result = parse_dice_command("  roll   the   d4   ")
        assert result == {"action": "roll", "target_shape": "d4"}

    def test_extra_punctuation_periods(self):
        result = parse_dice_command("roll the d10.")
        assert result == {"action": "roll", "target_shape": "d10"}

    def test_extra_punctuation_commas(self):
        result = parse_dice_command("move, the, d12,")
        assert result == {"action": "move", "target_shape": "d12"}

    def test_extra_punctuation_mixed(self):
        result = parse_dice_command("Please!!! Roll the D20??")
        assert result == {"action": "roll", "target_shape": "d20"}

    def test_rolling_variant(self):
        result = parse_dice_command("I want to roll the d4, rolling it around")
        assert result == {"action": "roll", "target_shape": "d4"}

    def test_tumble_synonym(self):
        result = parse_dice_command("tumble the d8")
        assert result == {"action": "roll", "target_shape": "d8"}

    def test_tumbling_variant(self):
        result = parse_dice_command("tumbling the d12")
        assert result == {"action": "roll", "target_shape": "d12"}

    def test_relocate_synonym(self):
        result = parse_dice_command("relocate the d10")
        assert result == {"action": "move", "target_shape": "d10"}

    def test_relocating_variant(self):
        result = parse_dice_command("relocating the d20")
        assert result == {"action": "move", "target_shape": "d20"}

    def test_place_synonym(self):
        result = parse_dice_command("place the d4")
        assert result == {"action": "move", "target_shape": "d4"}

    def test_placing_variant(self):
        result = parse_dice_command("placing the d8")
        assert result == {"action": "move", "target_shape": "d8"}

    def test_put_synonym(self):
        result = parse_dice_command("put the d12")
        assert result == {"action": "move", "target_shape": "d12"}

    def test_putting_variant(self):
        result = parse_dice_command("putting the d20")
        assert result == {"action": "move", "target_shape": "d20"}


class TestAliasNormalization:
    """d100 and d10_pct aliases should normalize to d10 per dice_pick_demo.py."""

    def test_d100_alias_normalizes_to_d10(self):
        result = parse_dice_command("roll the d100")
        assert result["target_shape"] == "d10"

    def test_d10_pct_alias_normalizes_to_d10(self):
        result = parse_dice_command("roll the d10_pct")
        assert result["target_shape"] == "d10"

    def test_d10_pct_case_insensitive(self):
        result = parse_dice_command("roll the D10_PCT")
        assert result["target_shape"] == "d10"


class TestD100VsD10Disambiguation:
    """Edge case: d100 must NOT accidentally match as d10 inside another token."""

    def test_d100_matches_not_d10(self):
        """Parsing 'd100' should yield d10 via alias, not somehow match d10 twice."""
        result = parse_dice_command("roll the d100")
        assert result["target_shape"] == "d10"
        # The key check: a single d100 token should be detected as one shape, not two.
        # This is implicitly verified by the fact that we get a result at all
        # (if d100 and d10 were both detected as separate shapes, we'd raise ValueError).

    def test_d100_in_phrase_with_other_text(self):
        """d100 with surrounding text should still match correctly."""
        result = parse_dice_command("please roll my d100 die")
        assert result["target_shape"] == "d10"

    def test_word_boundary_prevents_false_match(self):
        """A malformed token like 'd101' should NOT match 'd10' or 'd100'."""
        # This command has no valid shape, so should raise ValueError
        with pytest.raises(ValueError, match="no recognized die shape"):
            parse_dice_command("roll the d101")


class TestErrorConditions:

    def test_unknown_shape_raises_valueerror(self):
        with pytest.raises(ValueError, match="no recognized die shape"):
            parse_dice_command("roll the d7")

    def test_unknown_shape_typo_d20_as_d2(self):
        with pytest.raises(ValueError, match="no recognized die shape"):
            parse_dice_command("roll the d2")

    def test_no_shape_mentioned_raises_valueerror(self):
        with pytest.raises(ValueError, match="no recognized die shape"):
            parse_dice_command("roll it")

    def test_unknown_action_raises_valueerror(self):
        with pytest.raises(ValueError, match="no recognized action"):
            parse_dice_command("the d20")

    def test_unknown_action_typo_roll_as_rool(self):
        with pytest.raises(ValueError, match="no recognized action"):
            parse_dice_command("rool the d20")

    def test_neither_action_nor_shape_raises_valueerror(self):
        # ValueError priority: action check happens first, so this should complain about action
        with pytest.raises(ValueError, match="no recognized action"):
            parse_dice_command("foo bar baz")

    def test_ambiguous_both_move_and_roll_keywords(self):
        with pytest.raises(ValueError, match="both 'move' family keywords.*and 'roll' family"):
            parse_dice_command("move and roll the d20")

    def test_ambiguous_both_move_and_roll_with_different_words(self):
        with pytest.raises(ValueError, match="both 'move' family keywords.*and 'roll' family"):
            parse_dice_command("place the d10 and then tumble it")

    def test_ambiguous_multiple_distinct_shapes(self):
        with pytest.raises(ValueError, match="multiple die shapes mentioned"):
            parse_dice_command("roll the d20 and the d12")

    def test_ambiguous_multiple_distinct_shapes_d10_and_d100_are_same(self):
        """d10 and d100 are aliases (both map to d10), so they should NOT be
        treated as multiple distinct shapes — just one shape detected twice."""
        # This should parse successfully, not raise "multiple shapes" error
        result = parse_dice_command("roll the d100 and d10")
        # Both normalize to d10, which is one distinct shape, so this is fine
        assert result["target_shape"] == "d10"

    def test_ambiguous_three_different_shapes(self):
        with pytest.raises(ValueError, match="multiple die shapes mentioned"):
            parse_dice_command("roll d4, d8, and d12")

    def test_error_message_clarity_no_action(self):
        try:
            parse_dice_command("the d20")
            assert False, "should have raised ValueError"
        except ValueError as e:
            # Message should mention action keywords
            assert "move" in str(e).lower() or "roll" in str(e).lower()

    def test_error_message_clarity_no_shape(self):
        try:
            parse_dice_command("roll it")
            assert False, "should have raised ValueError"
        except ValueError as e:
            # Message should mention dice shape names
            assert "d4" in str(e) or "die" in str(e).lower()

    def test_error_message_clarity_both_actions(self):
        try:
            parse_dice_command("move and roll the d20")
            assert False, "should have raised ValueError"
        except ValueError as e:
            # Message should mention ambiguity
            assert "ambiguous" in str(e).lower()


class TestReturnStructure:
    """The return value must be exactly {action, target_shape}, nothing more."""

    def test_return_type_is_dict(self):
        result = parse_dice_command("roll the d20")
        assert isinstance(result, dict)

    def test_return_has_exactly_two_keys(self):
        result = parse_dice_command("roll the d20")
        assert set(result.keys()) == {"action", "target_shape"}

    def test_action_value_is_string(self):
        result = parse_dice_command("roll the d20")
        assert isinstance(result["action"], str)

    def test_target_shape_value_is_string(self):
        result = parse_dice_command("roll the d20")
        assert isinstance(result["target_shape"], str)

    def test_action_is_exactly_roll_or_move(self):
        roll_result = parse_dice_command("roll the d20")
        assert roll_result["action"] in ("roll", "move")
        move_result = parse_dice_command("move the d20")
        assert move_result["action"] in ("roll", "move")

    def test_target_shape_is_exactly_one_of_canonical_five(self):
        for die_type in ["d4", "d8", "d10", "d12", "d20"]:
            result = parse_dice_command(f"roll the {die_type}")
            assert result["target_shape"] in ("d4", "d8", "d10", "d12", "d20")
