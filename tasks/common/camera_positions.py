# tasks/common/camera_positions.py
"""Named camera-position registry, shared across tasks/scripts (2026-07-24,
isaac-sim-video-capture-skill task).

Built after a session of ad hoc Isaac Sim video captures (AR4 gripper open/
close cycle) where camera eye/target coordinates were hand-derived and
hand-tuned fresh for every single clip, several requiring user correction
after the fact (one "zoomed in weird," one producing solid black frames
from a bad eye position, a reversed wrist-to-elbow view, a zoomed-out
revision). The point of this module is to let a GOOD camera spot, once
found, be looked up by NAME instead of re-derived from scratch — see the
`isaac-sim-video-capture` skill (`~/.claude/skills/isaac-sim-video-capture/`)
for the full capture procedure this registry feeds into.

This module is intentionally plain Python with NO Isaac Sim/omni/pxr
imports — it must stay importable from a bare `python3` (e.g. for tooling,
docs generation, or a quick lookup) without requiring `isaaclab.sh -p` or a
running SimulationApp. The separate, Isaac-Sim-dependent read mechanism
that populates this registry from a live viewport camera lives in
`tasks/common/viewport_camera_io.py`, not here.

Entry format (all keys required except where noted):
    "eye":              (x, y, z) world-frame camera position, meters.
    "target":           (x, y, z) world-frame look-at point, meters. Only
                         the DIRECTION (target - eye) is guaranteed
                         meaningful for entries captured via
                         `scripts/record_viewport_camera_position.py` — see
                         that script's own docstring and
                         `viewport_camera_io.py`'s for why the *distance*
                         is a synthetic placeholder, not a measured
                         subject distance, for those entries specifically.
                         For entries derived by hand from a measured
                         subject position (the `"recorded_by": "derived"`
                         entries below), `target` IS the measured subject
                         position and both direction and distance are
                         meaningful.
    "focal_length":     float, mm. Isaac Lab's `CameraCfg`/
                         `PinholeCameraCfg` `spawn.focal_length` units.
    "clipping_range":   (near, far) meters, optional (defaults to whatever
                         the consuming CameraCfg already has if omitted —
                         only include it here if the position was tuned
                         together with a specific clipping range, e.g. a
                         close-up shot that needed a smaller `near` than a
                         task's default to avoid clipping).
    "description":      free-text human-readable description of the shot
                         (what it frames, why the eye/target were placed
                         there).
    "recorded_by":      "user" (captured via `record_viewport_camera_position.py`
                         from a real interactive GUI session) or "derived"
                         (hand-computed from a live-measured subject
                         position, not from an interactive session — see
                         each entry's own description for provenance).
    "recorded_date":    "YYYY-MM-DD".
    "verified_subject":  free-text description of what was confirmed
                         visible/in-frame — non-black frames extracted and
                         inspected, subject clearly visible and reasonably
                         sized, not cropped/tiny (see the skill's
                         verification checklist). Use "NOT YET VERIFIED —
                         <reason>" if an entry hasn't been through that
                         checklist yet (should be rare; don't seed
                         unverified entries without a clear label).

Usage:
    from tasks.common.camera_positions import get_camera_position
    pos = get_camera_position("ar4_gripper_closeup_v1")
    # pos["eye"], pos["target"], pos["focal_length"] etc.

Consuming a registry entry in a CameraCfg (this repo's established AR4-side
"opengl" lookat pattern — see scripts/_record_jaw_fix_open_close_cycle.py):
    from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix
    import torch

    pos = get_camera_position("ar4_gripper_zoomed_out")
    eye, target = pos["eye"], pos["target"]
    R = create_rotation_matrix_from_view(torch.tensor([eye]), torch.tensor([target]), up_axis="Z")
    quat = tuple(quat_from_matrix(R)[0].tolist())
    camera_cfg.spawn.focal_length = pos["focal_length"]
    camera_cfg.offset = CameraCfg.OffsetCfg(pos=eye, rot=quat, convention="opengl")

Note: this is a DIFFERENT convention from `tasks/franka/dice_scene_cfg.py`'s
own `ARM_CAMERA_POS`/`ARM_CAMERA_QUAT_WORLD` constants (pos+quat directly,
`convention="world"`, x-forward/z-up) — both conventions are legitimate and
already used elsewhere in this repo; this registry standardizes on
eye/target + "opengl" specifically because that is the convention
`scripts/record_viewport_camera_position.py`'s own read mechanism and this
module's `save_camera_position()` both produce. If you need a "world"-
convention pos/quat pair instead, convert eye/target -> rotation matrix ->
quat exactly as `scripts/interactive_camera_light_setup.py`'s own
`_capture_viewport_camera_as_arm_camera_cfg()` does (rebuilds a local
+X-forward/+Z-up frame from the native forward/up vectors) rather than
re-deriving that conversion from scratch.

Adding an entry:
  - From a live interactive GUI session (the intended normal path once the
    user has Isaac Sim open and has navigated to a spot they like):
    `scripts/record_viewport_camera_position.py <name>` — see that script
    and the isaac-sim-video-capture skill's reference.md for the exact
    remaining interactive step.
  - By hand (e.g. seeding a value seen in a git-history comment, as the two
    entries below were): add a dict literal inside the
    "BEGIN/END GENERATED REGISTRY" markers below, following the format
    above. Keep every entry inside those markers a plain literal (no
    function calls, no imports needed to evaluate it) — `save_camera_position()`
    parses/replaces exactly that region programmatically and will corrupt
    anything cleverer placed there.
"""
from __future__ import annotations

import os
from typing import Any

_REGISTRY_PATH = os.path.abspath(__file__)

# Built via string concatenation (not a single literal) so these constants'
# OWN definition lines don't themselves textually match what they search
# for below (a real bug hit while testing this module: a single-literal
# definition matches text.index() against its own assignment line first,
# silently corrupting the file on save). Do not "simplify" this back to one
# literal each.
_MARKER_TAG = "=== "
_BEGIN_MARKER = "# " + _MARKER_TAG + "BEGIN GENERATED REGISTRY ==="
_END_MARKER = "# " + _MARKER_TAG + "END GENERATED REGISTRY ==="

# === BEGIN GENERATED REGISTRY ===
# Do not hand-format this block unusually — save_camera_position() replaces
# everything between the BEGIN/END markers verbatim, using its own fixed
# rendering, on every write. Manual edits to VALUES are fine and preserved
# until the next programmatic write; manual edits to the surrounding
# markers/comment lines themselves are not.
CAMERA_POSITIONS: dict[str, dict[str, Any]] = {
    'ar4_gripper_closeup_v1': {
        'eye': (0.0, 0.15, 0.52),
        'target': (0.0, 0.364, 0.47),
        'focal_length': 50.0,
        'clipping_range': (0.05, 2.0),
        'description': 'Tight close-up on the AR4 gripper jaws (gripper_jaw1_link/gripper_jaw2_link), eye in front of and slightly below the gripper looking up/back at the jaws so world-X jaw separation reads as left-right motion in frame. Original working camera for the jaw open/close-cycle recording, before the zoomed-out/reversed-view revisions below/elsewhere.',
        'recorded_by': 'derived',
        'recorded_date': '2026-07-23',
        'verified_subject': 'AR4 gripper jaws during OPEN->CLOSE->OPEN cycle (logs/videos/ar4_gripper_jaw_open_close_cycle_fixed.mp4) — confirmed via extracted non-black frames. Source: scripts/_record_jaw_fix_open_close_cycle.py as committed at d59595a (2026-07-23), before the same-session black-frame attempt and later zoomed-out/reversed-view revisions.',
    },
    'ar4_gripper_zoomed_out': {
        'eye': (0.0, 0.05, 0.55),
        'target': (0.0, 0.364, 0.47),
        'focal_length': 30.0,
        'clipping_range': (0.05, 2.0),
        'description': "Same line-of-sight as ar4_gripper_closeup_v1, eye pulled back further along that line (larger Y/Z offset from target) and FOV widened (focal_length 50->30) so the full gripper plus some surrounding wrist/arm context is visible with margin, not cropped — the user's own 'too close/zoomed-in, weird-looking' feedback on the close-up prompted this revision. 'Zoom out' done by moving the eye back along the SAME look-direction, not just widening focal_length alone — see the isaac-sim-video-capture skill's reference.md for why that distinction mattered.",
        'recorded_by': 'derived',
        'recorded_date': '2026-07-23',
        'verified_subject': "AR4 gripper + wrist/forearm context during OPEN->CLOSE->OPEN cycle (logs/videos/ar4_gripper_jaw_open_close_cycle_fixed_zoomedout.mp4) — confirmed via extracted non-black frames, user-reviewed and judged reasonable framing (though the user separately described its viewing DIRECTION as 'oriented from the elbow down to the wrist,' which prompted the reversed-view revision — see this module's own docstring / the skill's reference.md for why that revision's exact eye/target are NOT seeded here as a fixed entry: they are computed dynamically at runtime from a live elbow/gripper measurement each run, not a persisted constant). Source: scripts/_record_jaw_fix_open_close_cycle.py's own commented-out 'superseded by REVERSED-VIEW REVISION' block, committed at 63cd0cc (2026-07-23).",
    },
    'ar4_gripper_front_view': {
        'eye': (0.0, 0.56409, 0.53478),
        'target': (0.0, 0.36409, 0.47478),
        'focal_length': 60.0,
        'clipping_range': (0.02, 3.0),
        'description': "Genuine FRONT view of the AR4 gripper jaws: eye on the +Y open-workspace side of the gripper (0.20m in front of the jaw midpoint), at the jaws' own height plus a 0.06m lift, looking back along -Y at the live-measured jaw midpoint. The jaws slide open/closed along world-X, so from this -Y sight-line the open/close GAP is seen at full lateral width with the two jaws left/right and nothing occluding them — the head-on front view the user asked for after rejecting every prior along-the-forearm (world-Y axis) view as 'the wrong view'. Eye sits in open space beyond the fingertips so it cannot land inside geometry (no black frames — contrast the base-side level eye that rendered solid black). Derived analytically (vendor-URDF FK, tasks/ar4/fk_verification.py) + confirmed against the live jaw body_pos_w; recorder: scripts/_record_gripper_front_view_open_close.py (computes eye/target from the live jaw midpoint at runtime).",
        'recorded_by': 'derived',
        'recorded_date': '2026-08-04',
        'verified_subject': "AR4 gripper OPEN->CLOSE->OPEN cycle (logs/videos/ar4_gripper_open_close_front_view.mp4), rendered on a GCP L4 via the container+GCS-cache path. Verified by extracting frames at t=0.3/2.8/4.5/5.8/7.0/8.8s: all non-black and well-lit (mean L~178), the gripper framed front-on and legibly sized, and the two jaws clearly separated left/right in the OPEN frames (serrated inner faces + visible gap) vs. met at a central seam in the CLOSED frame — gap widens/narrows symmetrically. Physics logs confirm 28.0mm (open) -> 0.0mm (closed) -> 28.0mm (reopen).",
    },
}
# === END GENERATED REGISTRY ===


def get_camera_position(name: str) -> dict[str, Any]:
    """Look up a named camera position. Raises KeyError with the list of
    available names if `name` isn't registered — check that list before
    hand-deriving a new position, per the isaac-sim-video-capture skill."""
    try:
        return CAMERA_POSITIONS[name]
    except KeyError:
        available = ", ".join(sorted(CAMERA_POSITIONS)) or "(none yet)"
        raise KeyError(
            f"No camera position named {name!r} in tasks/common/camera_positions.py. "
            f"Available: {available}. If none fit, derive a new one from a live "
            "measured subject position (see the isaac-sim-video-capture skill) rather "
            "than guessing coordinates, and consider recording it here via "
            "scripts/record_viewport_camera_position.py for reuse."
        ) from None


def _render_dict_literal(entries: dict[str, dict[str, Any]]) -> str:
    """Deterministic, human-readable re-serialization of the registry dict
    used by save_camera_position() to regenerate the GENERATED REGISTRY
    block. Fixed key order for entries that have the standard fields;
    any extra/custom keys on an entry are appended after, in insertion
    order, so nothing is silently dropped on a re-save."""
    standard_keys = (
        "eye",
        "target",
        "focal_length",
        "clipping_range",
        "description",
        "recorded_by",
        "recorded_date",
        "verified_subject",
    )
    lines = ["CAMERA_POSITIONS: dict[str, dict[str, Any]] = {"]
    for name in sorted(entries):
        entry = entries[name]
        lines.append(f"    {name!r}: {{")
        for key in standard_keys:
            if key in entry:
                lines.append(f"        {key!r}: {entry[key]!r},")
        for key, val in entry.items():
            if key not in standard_keys:
                lines.append(f"        {key!r}: {val!r},")
        lines.append("    },")
    lines.append("}")
    return "\n".join(lines)


def save_camera_position(
    name: str, entry: dict[str, Any], registry_path: str = _REGISTRY_PATH
) -> None:
    """Merge `entry` into the registry under `name` and rewrite this
    module's own source file on disk, replacing only the text between the
    BEGIN/END GENERATED REGISTRY markers (leaves this docstring/these
    functions untouched). Also updates the in-process CAMERA_POSITIONS dict
    so a caller that just wrote an entry can immediately `get_camera_position`
    it back without re-importing.

    Used by scripts/record_viewport_camera_position.py — not Isaac-Sim-
    dependent itself (pure file I/O), so it's safe to call from a plain
    Python context too (e.g. a test)."""
    with open(registry_path) as f:
        text = f.read()

    begin = text.index(_BEGIN_MARKER)
    begin_line_end = text.index("\n", begin) + 1
    # Skip the fixed "do not hand-format" comment lines directly under
    # BEGIN before finding END, so re-running this doesn't need to
    # regenerate the comment itself.
    end = text.index(_END_MARKER, begin_line_end)

    merged = dict(CAMERA_POSITIONS)
    merged[name] = entry
    new_dict_src = _render_dict_literal(merged)

    comment_block = (
        "# Do not hand-format this block unusually — save_camera_position() replaces\n"
        "# everything between the BEGIN/END markers verbatim, using its own fixed\n"
        "# rendering, on every write. Manual edits to VALUES are fine and preserved\n"
        "# until the next programmatic write; manual edits to the surrounding\n"
        "# markers/comment lines themselves are not.\n"
    )
    new_text = text[:begin_line_end] + comment_block + new_dict_src + "\n" + text[end:]

    with open(registry_path, "w") as f:
        f.write(new_text)

    CAMERA_POSITIONS[name] = entry
