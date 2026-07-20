"""Pure-Python, rule-based dice command parser (stdlib only, no external dependencies).

This is a rule-based parser for natural-language dice commands, not an LLM/VLM — no LLM
API access is available in this environment. It provides a legitimate first version in its
own right, turning phrases like "pick up and roll the d20" into structured
{action, target_shape} dicts the demo pipeline can dispatch on.

Examples of supported phrasings:
  - "pick up and roll the d20" -> {"action": "roll", "target_shape": "d20"}
  - "pick up and move the d12" -> {"action": "move", "target_shape": "d12"}
  - "move the d8 to the goal" -> {"action": "move", "target_shape": "d8"}
  - "roll the d10" -> {"action": "roll", "target_shape": "d10"}
  - "please pick up and roll die d4" -> {"action": "roll", "target_shape": "d4"}
  - "Move D100 over there" -> {"action": "move", "target_shape": "d10"}  (d100 alias -> d10)

Raises ValueError with clear, specific messages if action or shape cannot be confidently
determined (e.g., ambiguous dual action families, unrecognized shape, multiple shapes).
"""

import re

# Dice shape normalizations - must match dice_pick_demo.py's exact aliases
_CHOICE_ALIASES = {"d100": "d10", "d10_pct": "d10"}

# Roll-family action keywords -> "roll"
_ROLL_KEYWORDS = {"roll", "rolling", "tumble", "tumbling"}

# Move-family action keywords -> "move"
_MOVE_KEYWORDS = {"move", "moving", "relocate", "relocating", "place", "placing", "put", "putting"}


def parse_dice_command(command: str) -> dict:
    """Parses a natural-language dice command into a structured dict.

    Args:
        command: Natural-language command string (e.g. "pick up and roll the d20").

    Returns:
        dict with exactly two keys:
          - "action": str, one of "roll" or "move"
          - "target_shape": str, one of "d4", "d8", "d10", "d12", "d20"
        (d100 and d10_pct inputs are normalized to "d10")

    Raises:
        ValueError: if action or shape cannot be confidently determined. Message
        specifies what was ambiguous or missing (e.g., "both 'move' and 'roll'
        keywords found" or "no recognized die shape mentioned").
    """
    # Normalize: lowercase and strip whitespace
    normalized = command.lower().strip()

    # Step 1: Detect action keywords
    roll_found = any(f"\\b{kw}\\b" in re.sub(r"[^a-z0-9\s]", " ", normalized) for kw in _ROLL_KEYWORDS)
    # More robust: tokenize by splitting on non-alphanumeric, then match whole tokens
    tokens = re.findall(r"\b\w+\b", normalized)

    roll_found = any(token in _ROLL_KEYWORDS for token in tokens)
    move_found = any(token in _MOVE_KEYWORDS for token in tokens)

    if roll_found and move_found:
        raise ValueError(
            "ambiguous command: both 'move' family keywords (move, moving, relocate, "
            "relocating, place, placing, put, putting) and 'roll' family keywords "
            "(roll, rolling, tumble, tumbling) found — please specify one action only"
        )
    if not roll_found and not move_found:
        raise ValueError(
            "no recognized action found: use 'move'/'moving'/'relocate'/'relocating'/'place'/"
            "'placing'/'put'/'putting' (for move) or 'roll'/'rolling'/'tumble'/'tumbling' "
            "(for roll)"
        )

    action = "roll" if roll_found else "move"

    # Step 2: Detect dice shape
    # Pattern: d100/d10_pct/d10 must be checked before d4-d8-d12-d20 to avoid d10
    # matching inside d100. Use word boundaries + longest-first alternation.
    shape_pattern = r"\b(d100|d10_pct|d10|d4|d8|d12|d20)\b"
    shape_matches = re.findall(shape_pattern, normalized)

    if not shape_matches:
        raise ValueError(
            "no recognized die shape mentioned: use d4, d8, d10, d12, or d20 "
            "(d100 and d10_pct are also valid aliases for d10)"
        )

    # Normalize all detected shapes (d100 -> d10, d10_pct -> d10) before checking uniqueness
    normalized_shapes = [_CHOICE_ALIASES.get(shape, shape) for shape in shape_matches]

    # Check for multiple distinct shapes (after normalization, so d10 and d100 count as one)
    unique_shapes = set(normalized_shapes)
    if len(unique_shapes) > 1:
        raise ValueError(
            f"ambiguous command: multiple die shapes mentioned ({', '.join(sorted(unique_shapes))}) "
            "— please specify exactly one die type"
        )

    # The target shape is already normalized
    target_shape = normalized_shapes[0]

    return {
        "action": action,
        "target_shape": target_shape,
    }
