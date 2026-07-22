# Franka IK dice-line demo (2026-07-21) — pick, line up, and relocate five dice

**Result: real, mostly-successful classical (non-RL) pick-and-place demo —
8/10 pick-and-place operations succeeded; d4 the sole failure, in both
attempts.** A single Franka Panda, driven by classical differential IK only
(no learned policy, no detector — ground-truth poses), picks up all 5
canonical dice (d4/d8/d10/d12/d20) from `DiceSceneCfg`'s own scattered
default layout and lines them up (**Act 1**), then re-picks each die from
the line and relocates the whole line to a rotated (column→row) and shifted
position (**Act 2**). Script: `scripts/demo_franka_ik_dice_line.py`. Fun/demo
deliverable, not a Tier-1/2 experiment — no hypothesis/spec/plan gate
applies.

This reuses [[dice-pick-demo]]'s already-validated staged-IK mechanism
(joint-space ready-pose prep, canonical straight-down orientation, per-die
measured grasp height, d4 V-notch fingertip fixture) via a generalized
`pick_and_place()`, rather than re-deriving any of it.

## What ran

Dispatched to GCP rather than the desktop (the desktop was busy with the
concurrent AR4-transfer workstream at dispatch time — see
[[pi-as-primary-agent-gpu-dispatch]]). A real cloud infra gap was hit and
fixed along the way: buffering all captured video frames in host RAM before
encoding OOM-killed the first attempt on a 16GB instance; fixed by streaming
frames straight to the imageio/ffmpeg writer instead of buffering the whole
video in memory.

## Result

Across 3 cloud runs, the most recent shipped result: **8/10 pick-and-place
operations (4 dice × 2 passes) landed within a few mm to ~11cm of their
target.** **d4 — this project's own well-documented hardest grasp case (see
[[dice-pick-demo]]'s open follow-ups on d4's edge-grasp and V-notch
attempts) — failed to be physically grasped in both attempts**: every IK
waypoint converged, but the die simply never left the table. d4 was
deliberately sequenced last in the final video (not first, its original
ascending-size position in the layout) once this was observed, purely so
the video doesn't open on a stall — a sequencing fix, not a mechanism fix;
d4's underlying grasp reliability is unchanged and remains this project's
open d4 problem.

Video: `outputs/dice_demo/ik_dice_line/franka_ik_dice_line_demo.mp4`
(gitignored, durable local copy) and
`site/assets/videos/projects/franka-dice-pick/dice-line-pick-and-place.mp4`
(committed, portfolio-site content-pack convention).

## Related concepts

[[dice-pick-demo]] — the staged-IK pick mechanism this demo generalizes and
reuses wholesale, including its still-open d4 grasp-reliability problem.
[[cloud-training]] — the GCP cloud pipeline this demo ran on; this demo's
own OOM-by-buffering-video-frames finding is a new entry in that pipeline's
running list of gaps found beyond NVIDIA's own docs. [[pi-as-primary-agent-gpu-dispatch]]
— why this ran on GCP rather than the desktop.

## Sources

`scripts/demo_franka_ik_dice_line.py`, `ROADMAP.md`'s 2026-07-21 "Built"
entry (no separate spec/plan doc exists — fun/demo deliverable, no Tier-1/2
gate applies).
