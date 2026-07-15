# d4 edge-grasp, rung 1: rigid V-notch fingertip geometry

**Date:** 2026-07-15.
**Prior result:** rung 0 (`docs/superpowers/specs/2026-07-13-d4-edge-grasp-rung0-design.md`)
FALSIFIED at the reachability level — the tilted opposite-edge-pair
approach requires the lower jaw to occupy space below the table for
every plausible opening/finger-geometry combination on a face-resting
d4; PhysX blocks the descent before the grasp mechanism (line contact
resisting the ejection twist) is ever tested. The ladder's own
pre-registered climb rule moves to rung 1: a fingertip/pad geometry
modification that works with the existing STRAIGHT-DOWN approach already
validated (reachably) for the other 4 die types — no tilt, no table
interaction, so rung 0's table-collision constraint doesn't apply here
by construction.
**Research grounding:** `.superpowers/sdd/research-d4-rung1-pad-geometry.md`
(senior research pass, 2026-07-15) + `.superpowers/sdd/review-d4-rung1-pad-geometry.md`
(independent senior review, 2026-07-15 — every citation and the core
geometric derivation independently re-verified from primary sources, not
re-read from the original doc). One citation-attribution error was
caught and is corrected below; see "Corrections from independent review."

## Hypothesis (falsifiable)

> Replacing the Franka's stock flat parallel-jaw fingertips with a rigid
> V-notch fixture (110° internal notch angle, ~10mm grip depth below the
> die's apex, ~4mm notch depth, fixed-joint-attached to both fingertips
> identically) — using the existing straight-down, untilted approach
> already reachable for every die type — will yield pick success (z-gain
> ≥200mm sustained, matching the other dice's measured 237–241mm) in ≥4/5
> seeded d4 trials with confirmed flush notch-facet contact, **and**
> produce no regression (still meeting each die's own currently-passing
> baseline) on d8/d10/d12/d20 under the same, unconditionally-modified
> gripper — because the notch converts a flat pad's line/near-point
> contact against the die's sloped facets into two-facet flush area
> contact plus a self-centering funnel effect (the V-block principle,
> standard machine-design practice), addressing the twist-and-eject
> mechanism Wang et al. (CASE 2019) identify for pinches on angled
> surfaces and the convex-vertex fragility Montana's result (cited via
> Smith et al., ICRA 1999) predicts for the die's current near-vertex
> contact — both already the grounding for rung 0's own hypothesis, now
> applied to a notched rather than flat pad.

**Falsification condition:** if ≥2/5 d4 trials still fail (lateral
ejection >5mm at closure, or no sustained z-gain ≥200mm) **despite
confirmed flush notch-facet contact** (verified via the existing
`d4_leftfinger_contact`/`d4_rightfinger_contact` sensors, not inferred
from video — same discipline rung 0 applied), the V-notch geometry is
falsified as a fix for the d4 specifically. Independently: if ≥1 of
d8/d10/d12/d20 drops below its own currently-passing baseline (per
`kb/wiki/experiments/dice-pick-demo.md`) under the notched gripper, the
notch is falsified as a *general*, non-regressing modification — the
project would then face a real choice (iterate the notch design, or
accept a swappable per-task fixture as a documented real-world tradeoff)
rather than a free win. Both conditions are evaluated independently; d4
success does not excuse a non-d4 regression, and vice versa.

## Corrections from independent review (applied here, not in the research doc itself)

The research doc's core recommendation was found sound; three factual
corrections from the independent review are load-bearing for this spec
and are applied directly:

1. **Citation correction**: the V-groove precedent paper is **Habibi,
   Sutera, Guastella & Muscato, "Design of a Novel Parallel-Jaw Gripper
   for Cylindrical Object Manipulation," *Robotics* 14(7):87 (2025-06-25)**
   — the research doc's "Zhang et al. ... 2026" attribution was invented
   (a fresh instance of this project's known citation-fabrication-detail
   risk, `kb/wiki/concepts/citation-verification-practice.md` — the paper
   and every numeric claim drawn from it are real and independently
   re-confirmed; only the author/year label was wrong). This paper's role
   here is narrow and explicitly qualified: it establishes that V-angle
   must be sized to the target object (its own 45° profile was
   "fundamentally unsuitable," 135° worked) — a general caution, not a
   tetrahedron-specific result (it's cylindrical-stock-oriented
   throughout; no paper anywhere analyzes V/notch geometry against a
   polyhedral object). The 110° notch angle below is this project's own
   derivation from the tetrahedron's dihedral geometry, not from this
   paper.
2. **Isaac Lab deformable-attachment issue #4291** is closed as answered
   (maintainer confirms direct deformable-to-articulation attachment is
   still effectively unsupported in this project's version line,
   5.1.0/2.3.0), not "open, unresolved" as the research doc stated. The
   substantive conclusion — a true compliant/deformable pad is not
   currently buildable in this project's sim, so the rigid V-notch is a
   geometric *proxy* for Guo et al.'s compliant-surface mechanism, not a
   reproduction of it — is unaffected and still the basis for choosing a
   rigid attachment below.
3. **The centroid-height "cross-check"** the research doc uses (4.82mm
   vs. rung 0's own 4.8mm figure) is an arithmetic-consistency check, not
   independent corroboration — both apply the identical `h/4` formula to
   the identical measured edge length. Treated here only as "no
   transcription error," not as two independently-derived measurements
   agreeing.

**Extrapolation ownership**, matching rung 0's own convention: the 110°
notch angle, ~10mm grip depth, ~4mm notch depth, and the whole notch
design are this project's own derivation from solid-tetrahedron geometry
(dihedral angle `arccos(1/3) ≈ 70.53°`, supplement `109.47°`; both
independently re-derived twice — once by the original researcher, once
by the reviewer from scratch coordinates — and confirmed exact). No
paper analyzes V-notch geometry against a tetrahedron specifically; Guo
et al. 2017 supports only "compliant/textured surface beats rigid flat
pad" (its actual result), not a wedge/notch shape (its full text was
grepped directly and contains none), and the Habibi et al. paper supports
only "V-angle must be sized to the target" (general caution). Also
unremarked by either research pass, worth stating plainly here: this spec
does not compare "straight-down + notch" head-to-head against "revisit
the tilted approach with a shorter/recessed custom jaw" — the
straight-down premise is inherited directly from ROADMAP's existing rung
framing (a prior Principal-level decision, made when rung 1 was first
named at rung 0's closure), not re-derived from scratch here.

## North Star call — the fixture is unconditional, not a per-shape branch (explicit, not defaulted)

Rung 0 was scoped as a pure software/control-path branch: a `d4`-only
code path in the scripted demo, with the other 4 die types' path
required to stay byte-identical. A fingertip notch is different in kind
— it is physical gripper hardware. Modeling it as "only present for d4
sim runs" would be a sim-only fiction with no real-robot analogue (a
real Franka can't have simultaneously-notched-for-d4 and flat-for-others
fingertips without a physical tooling change between tasks). Per this
repo's North Star bar (an approach that would actually generalize, not
one that only works because it was hand-fit to this sim), **this rung's
fixture is modeled as unconditionally attached to both fingertips for
every die type**, and this rung's success criteria (above) require it to
still pass the existing d8/d10/d12/d20 regression baseline, not just the
d4 target. If it regresses any of the other four, that is real
information, not a scoping failure to route around — the project would
then be choosing between iterating the notch (asymmetric edge/face
design, see Future work) or accepting a swappable fixture as a genuine
per-task tooling change (a normal real-robotics practice, e.g. quick-change
end-effector adapters), which would itself need to be modeled explicitly
rather than assumed free.

## Design

### Geometry (derived from the d4's own measured mesh, a = 23.591mm edge length)

- Apex height above base: `h = a·√(2/3) ≈ 19.26mm`.
- Dihedral angle along any edge: `arccos(1/3) ≈ 70.53°`; notch opens to
  its supplement, `arccos(-1/3) ≈ 109.47°`, rounded to a buildable
  **110° internal notch angle**.
- **Target grip height ≈10mm below the apex** (≈9.3mm above the table) —
  shallow enough that the local triangular cross-section (`a·10/h ≈
  12.2mm` edge length, vs. 23.6mm at the base) is meaningfully larger
  than rung 0's near-point contact target; deep enough to keep clear of
  the table (Franka fingertip's own measured ≈14–18mm tip extent is
  comparable in scale to this 9.3mm margin — verify directly in the desk
  check below before any sim run, do not assume).
- **Notch depth ≈4mm, opening width ≈11mm** (`2·depth·tan(55°)` for a
  110° notch), leaving ≈3mm of flanking material on each side of the
  Franka fingertip's measured 17.5mm tip width.
- **Chamfered lead-in**: ~2mm outward flare at the notch mouth before the
  true 110° notch begins, to absorb this project's own measured IK
  position error (1–5mm, per rung 0's desk check) during closing rather
  than requiring an already-centered approach.
- **Both fingertips get an identical notch** (not an asymmetric
  edge/face-ramp pair — see Future work). Because the die's 3 lateral
  edges sit 120° apart in yaw and the gripper's two jaws sit 180° apart,
  and `180 mod 120 = 60`, whenever one jaw lands edge-aligned the
  opposite jaw is *exactly* face-midpoint-aligned (an exact consequence
  of 3-fold symmetry, re-derived independently by the reviewer, not an
  approximation) — the identical symmetric notch gets flush 2-facet
  contact on the edge-aligned jaw and 2-line contact against a single
  face on the opposite jaw, still wider-based than rung 0's point/edge
  contact on both sides, with one part geometry.

### Implementation path

**Rigid fixed-joint attachment onto the existing fingertip prim**, not a
full fingertip mesh replacement — lower risk (the stock Franka asset used
by the other 4 die types stays untouched at the base-mesh level; the
attachment is the only new geometry), faster to iterate notch dimensions
without re-authoring the whole finger collision mesh, and keeps the
existing `git diff -w` byte-identical regression-guard technique
available for everything except the new attachment prim itself. A true
compliant/deformable pad (Guo et al.'s actual winning mechanism) is
confirmed not currently buildable in this project's Isaac Lab version
(deformables can't attach to an articulation at all, and rigid-deformable
contact force reporting is unsupported — both independently re-verified
against the live GitHub issues, see "Corrections" above) — a rigid convex
collision mesh is used instead, correctly understood as a geometric proxy
for the self-centering effect compliance gives, not equivalent to it, and
likely recovering less of Guo et al.'s 93.7%-vs-28.7% gap than a true
compliant pad would.

## Desk check (must pass before any sim run, matching rung 0's own convention)

1. Measure the Franka fingertip's actual tip geometry directly from the
   asset (not the ~14–18mm figure carried over from the research pass) —
   confirm the ≈9.3mm table clearance at the ≈10mm grip depth is real,
   not assumed, the same "swept-volume vs. support surface" check rung
   0's closure flagged as a gap for future grasp-geometry specs.
2. Confirm the notch fixture's collision mesh is buildable as a rigid
   convex hull attachable via a fixed joint to the existing fingertip
   prim without altering the base Franka asset used by non-d4 runs.
3. Re-derive (or re-read, not re-trust) the 110°/~10mm/~4mm/~11mm figures
   against the actual measured d4 mesh at plan-writing time, the same way
   rung 0's own desk check caught a 30.3mm-vs-23.591mm transcription
   error before it reached sim.

## Success criteria (pre-registered)

- **d4 primary**: ≥4/5 seeded trials (seeds 42, 123, 7, 1000, 2026) end
  in pick success — z-gain ≥200mm sustained — with lateral ejection at
  closure ≤5mm, **and** confirmed flush notch-facet contact via the
  existing d4 contact sensors (not inferred from video alone).
- **Non-d4 regression guard**: one smoke re-run per die type
  (d8/d10/d12/d20) under the notched gripper, each still meeting its own
  currently-passing baseline from `kb/wiki/experiments/dice-pick-demo.md`.
  This is a first-class success criterion per the North Star call above,
  not a secondary check.
- **Climb/iterate rule**: if the d4 primary fails despite confirmed flush
  contact, treat the notch mechanism (not just its exact dimensions) as
  falsified and consider the scooping/underactuated-fingertip strategy
  (Odhner & Dollar 2012; Babin & Gosselin 2018 — a genuinely different,
  motion-based axis, deferred from this rung, existence-confirmed but not
  detail-verified) as the next rung. If the d4 primary passes but a
  non-d4 regression appears, do not silently ship the fixture — bring the
  regression back to Principal as an explicit tradeoff decision (iterate
  the notch vs. accept swappable per-task tooling), per the North Star
  call above.

## Out of scope (this rung)

- Any RL training — this is scripted-controller (`scripts/dice_pick_demo.py`)
  work only, same scope boundary as rung 0.
- The asymmetric edge/face-ramp refinement (two different fingertip
  shapes) — kept on record as a second-iteration option if the symmetric
  notch's d4 numbers are only partially improved, not the first build.
  Note for whoever picks this up: the research doc's own worked offset
  example (`≈1.8mm` waypoint shift for the asymmetric case) has the
  right magnitude but the wrong stated direction (independent review
  re-derivation: the shift is toward the edge/vertex side, which needed
  more travel, not the face side) — fix before implementing, not
  urgent for this rung since the primary design doesn't use it.
- A true compliant/deformable pad (blocked by current Isaac Lab
  version's articulation-attachment and contact-force-reporting gaps,
  see Corrections above) — revisit if/when those Isaac Lab limitations
  are resolved upstream.
- A head-to-head comparison against a tilted-approach-with-recessed-jaw
  alternative (see "Extrapolation ownership" above) — not ruled out,
  just not evaluated in this pass.

## Future work

- If the symmetric notch partially succeeds on d4 but leaves headroom,
  the asymmetric edge/face-ramp design (documented in the research doc,
  offset-direction-corrected per above) is the recorded next iteration.
- If Isaac Lab's deformable-articulation-attachment gap (#4291) is
  resolved upstream, revisit whether a true compliant pad recovers more
  of Guo et al.'s gap than this rigid proxy.

## Verdict (2026-07-15)

**UNTESTED — 0/5 d4 seeded trials reached the grasp mechanism.** Not
falsified: every trial failed at the perception step (`select_target_detection`
raised, no `d4`-class detection returned by the detector), strictly
upstream of any grasp attempt — the same category of outcome as rung 0's
own closure, and for the same underlying reason (a pre-mechanism failure
gates the trial before the hypothesis is ever exercised). Full per-trial
data: `.superpowers/sdd/task-2-report.md`.

**What did get tested and passed cleanly**: the implementation itself.
`sim.reset()` succeeded in all 5 d4 trials and all 4 non-d4 smokes — the
notch fixture's `activate_contact_sensors` fix (Task 1's Critical review
finding) holds under real physics, not just the desk-check/unit-test
level it was verified at before this run. The `xformOp`/`SetTranslate`
warning flagged as an unverified risk in Task 1 fired consistently and
non-fatally in every trial, with the fixture's live-measured position
always correct — confirmed harmless, not silently degrading.

**Non-d4 regression guard: 3/4 clean PASS (d8/d10/d12), 1/4 attributable
FAIL (d20).** d8/d10/d12 all matched or exceeded the kb baseline z-gain
range (237-241mm) with zero cross-die drift — the unconditional fixture
attachment (this spec's own North Star call) does not regress the three
die types it was tested against. d20's failure reproduces, byte-for-byte
across two reruns, `kb/wiki/experiments/dice-pick-demo.md`'s already-
closed "Fragility attribution" finding (cross-session RTX render
nondeterminism at this exact seed+die combination, predating this rung
entirely) — not new information about the notch fixture.

**New finding, distinct from the pre-registered hypothesis**: all 5 d4
trials failed identically (0/5, not the kb's previously-documented
occasional per-seed noise pattern), and the one trial where a `d4`-class
candidate appeared at all (seed 123) was low-confidence (0.27-0.36) and
displaced by a same-location, higher-confidence `d10` candidate. A 5/5
identical-shaped failure rate across 5 different seeds/scene layouts is
a stronger, more systematic-looking signal than a single seed's
occasional offset — this reads as "d4 is a weak/marginal detection class
for this detector on this scene region" rather than noise, though this
was not investigated further here (out of this task's scope, per the
report) and is recorded as an open hypothesis, not a conclusion.

**Disposition**: rung 1's actual question (does the V-notch geometry
survive closure without ejecting the die) cannot be fairly tested while
perception gates every trial before the grasp attempt. This is a
different kind of blocker than rung 0 hit (an IK-reachability gap in the
*grasp* controller) — here the blocker is entirely upstream, in
detection, and orthogonal to anything this rung's own diff touches. Two
live options for Principal, not decided here: (a) scope a ground-truth
XY-bypass path for grasp-mechanism trials specifically (mirroring rung
0's own "isolates the grasp-mechanism variable" precedent, which used
GT for orientation — this would extend that same isolation principle to
position), or (b) treat the d4 detection weakness as its own
research question before returning to rung 1. See ROADMAP.md's
2026-07-15 entry and `kb/wiki/experiments/dice-pick-demo.md`'s "Open
follow-ups" for the recorded state.
