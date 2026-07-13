# d4 grasp, rung 1: V-groove fingertip pads (straight-down approach)

**Date:** 2026-07-13 (drafted evening; execution next GPU window).
**Branch:** `franka-panda-pivot`.
**Prior result:** rung 0 (edge-grasp reorientation) FALSIFIED at the
reachability level — the tilted opposite-edge pose requires the lower
jaw below the table (spec:
`2026-07-13-d4-edge-grasp-rung0-design.md`, Rung-0 closure). The
original flat-pad failure stands: sub-mm convergence, then closure
ejects the d4 16.3mm sideways.
**Research grounding:** `.superpowers/sdd/research-d4-grasp.md` Axis 2 +
rung-1 entry (citation-reviewed 2026-07-12): Guo et al., ICRA 2017 —
1377 physical trials, stock rigid pads 28.7% vs modified tip surfaces up
to 93.7% on hard-to-grasp objects under 5mm/1cm position uncertainty
(verified exact against the primary source); MDPI *Robotics* 14(7):87 —
V-profile geometry must be sized to the target's characteristic
dimension (a mis-sized V is actively unstable); Pekel's Blender→USD
end-effector swap pipeline as the asset pattern.

## Why V-groove and not compliant pads (design decision, owned here)

Guo et al.'s best result (93.7%) came from a compliant micro-void
silicone surface — but rigid-body PhysX cannot faithfully simulate pad
compliance; a "compliant" pad in Isaac would test contact-parameter
tuning, not the mechanism. The V-groove is rigid geometry: PhysX
simulates it directly, and it targets the d4's actual failure mode
(lateral ejection under closure) with a lateral-centering constraint.
Compliance remains the likely winner on real hardware; that distinction
is recorded for the eventual sim-to-real phase.

## Hypothesis (falsifiable)

> Replacing the flat fingertip pads with V-grooved pads (groove sized to
> the measured d4, straight-down approach unchanged) will yield pick
> success in ≥4/5 seeded trials with lateral ejection ≤5mm at closure —
> because the groove converts the single-plane pad/face contact (whose
> tangential force component ejects the die) into multi-line contact
> that kinematically constrains lateral slide, the mechanism Guo et al.
> measure (modified tip surfaces, 28.7%→80-94%) and the MDPI V-profile
> paper demonstrates for cylindrical stock. The 2D-cross-section force
> argument is this project's own extrapolation to a tetrahedron, not a
> claim of either source.

**Falsification condition:** ≥2/5 trials ejecting >5mm despite converged
straight-down closure with the V-pads (convergence verified from logged
EE pose; contact location verified from the rung-0 contact sensors, now
committed and unexercised). On failure: rung 2 (suction) is next on the
pre-registered ladder; no V-angle iteration beyond one bounded pass.

## Desk check (gate before any sim run — includes the new standing rule)

1. **Swept-volume vs support surface** (the rung-0 lesson, now
   mandatory): straight-down approach with open jaws — verify no jaw
   structure passes below the table plane at any commanded waypoint.
   (Expected trivially clean for straight-down, but computed, not
   assumed.)
2. Cross-section analysis at closure height for the face-resting d4
   (measured a = 23.591mm): which edge/face pair the jaws actually meet,
   groove angle/depth such that the die's near-vertical edge seats in
   the groove with both groove faces in contact, and the opposing flat
   face contacts the other pad's groove shoulders. Friction-cone check
   at all contact lines (μ = 0.5 verified).
3. **Regression geometry for the other four dice**: groove
   depth/opening must leave the shoulders' flat contact area sufficient
   for d8/d10/d12/d20 face contacts (MDPI caution — a V sized wrong is
   actively unstable). Compute the largest die-face inscribed-circle
   contact patch vs shoulder width.

## Design

- Bake a custom fingertip USD (visual + collision mesh, simple
  extruded-V prism cut into the stock pad envelope) in Blender; swap
  onto the existing gripper rig following the repo's asset-baking
  conventions (Xform-root/Mesh-child — the bisect's PhysX gotcha).
  Joints, drives, rig topology untouched.
- Demo integration: a `--fingertip vgroove` flag (or env-cfg variant)
  selecting the pad asset; d4 uses the ORIGINAL straight-down gate-G
  path — the rung-0 tilted branch stays dormant. All five dice runnable
  with either pad set.
- Instrumentation: rung-0's contact sensors (committed b588ff7) filter
  to Die_d4 — extend the filter to the commanded die per trial.

## Success criteria (pre-registered)

- **Primary:** ≥4/5 seeded d4 trials (seeds 42/123/7/1000/2026) with
  pick success (z-gain ≥200mm sustained) and closure lateral ejection
  ≤5mm, V-pads, straight-down.
- **Regression guard:** all four previously-passing dice re-run (seed
  42) with the V-pads — each must still pick successfully. A V that
  wins the d4 but loses any other die FAILS the guard (North Star: no
  per-shape pad swapping in the demo loop).
- **Climb rule:** primary fail or guard fail → rung 2 (suction,
  SurfaceGripper manager-based path per PR #3174), CPU-cost scoping
  first.

## Out of scope

- Compliant/soft-body pad simulation; real-hardware pad fabrication.
- Perception-side pose estimation (unchanged from rung 0's scoping).
- Any change to the straight-down controller logic beyond the pad asset
  and per-die contact-sensor filter.

## Verdict

_(appended after trials)_
