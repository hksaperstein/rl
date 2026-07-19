# Unified multi-die specialist-distillation experiment (started 2026-07-16)

**Goal:** one RL policy that grasps-and-lifts a commanded die among
{d8, d10, d12, d20} (d4 out of scope), by training a per-shape specialist
for each die then distilling them into one policy (UniDexGrasp++'s GiGSL
pattern). Spec:
`docs/superpowers/specs/2026-07-16-unified-multi-die-specialist-distillation-design.md`.
Plan:
`docs/superpowers/plans/2026-07-16-unified-multi-die-specialist-distillation.md`.
Gated on [[reward-hacking-and-sparse-discoverability]]-adjacent research
(DexSinGrasp, arXiv:2504.04516) that uncurriculated multi-object clutter
can collapse RL discovery — distractors/target-selection are explicitly
deferred to a follow-on experiment; every env in this plan spawns exactly
one die.

## Tasks 0-1: assets + observations (complete, reviewed clean)

Baked real-standard-size physics assets for d8/d10/d12 (16.0mm, 16.0mm,
18.0mm face-to-face) alongside the existing d20 (per
[[size-curriculum]]'s standard-vs-jumbo correction). Added a shape-class
one-hot + Wadell-sphericity geometry-descriptor observation term so a
single policy can eventually condition its behavior on which die it's
holding.

## Task 2: train d8/d10/d12 specialists at their real small sizes — 0/9

Trained each of d8/d10/d12 at their own real ~16-18mm size. Result:
**zero discovery, 0/9** (3 shapes x 3 seeds) — worse than the d20's own
1/3-at-48mm [[asset-bisect]] baseline. Independently re-derived the raw
per-step height data (not just the summary JSON) and found the object is
completely motionless for the entire eval in every seed — a *stronger*
null than the initial report suggested. Left an open confound: these
three shapes were never tested at [[asset-bisect]]'s own 48mm-parity
anchor, so this result can't yet be attributed to shape difficulty vs.
these objects simply being too small at the Franka gripper's absolute
scale. (This confound is what Task 3.5 was inserted to resolve — see
below.)

## Task 3: d20 size-DR + geometry-feature retry — 0/120, gate before Task 4

Retried the d20 with `FrankaDieLiftJointRandomSizeEnvCfg`
(`MultiAssetSpawnerCfg(random_choice=True)`, spanning 22-48mm, 5 discrete
sizes) plus Task 1's new geometry-descriptor conditioning — the one
mechanistic difference from the already-falsified
[[size-curriculum]]-era `FrankaDieLiftJointMixedEnvCfg`, which got 0/3.

**Result: 0/120** (3 seeds x 5 scales x 8 envs/eval), confirmed genuinely
motionless (not "attempted but unsustained") by independently re-deriving
raw per-step height data and watching eval video frame-by-frame. This
matches, not corrects, the plan doc's preliminary "0/120" figure.

**Ambiguous verdict, flagged rather than resolved:** `random_choice=True`
still assigns one size per env once at scene-spawn (same mechanism as the
falsified `MixedEnvCfg`, confirmed by direct source read), so Task 3's
48mm arm is itself a diluted ~1/5 sub-population — reproducing the
`MixedEnvCfg`'s 0/3 floor there is *consistent with* population dilution
being the real confound, but doesn't rule out shape/discoverability
remaining a barrier independent of dilution, because Task 3 never paired
an undiluted single-48mm d20 population with the geometry-descriptor
conditioning. Task 3.5's undiluted-48mm design (below) was scoped to
d8/d10/d12 only, so this specific ambiguity is still open for the d20
case if it becomes decision-relevant. Full grid and reasoning:
`ROADMAP.md`'s Task 3 entry (search "0/120").

## Task 3.5: 48mm-parity check for d8/d10/d12 — inserted 2026-07-16, gate before Task 4

Neither Task 2 nor Task 3 ever tested d8/d10/d12 at a single, undiluted,
48mm population the way the original [[asset-bisect]] ladder did for the
cube (3/3) and d20 (1/3). This task closes that gap: three new env cfg
classes (`FrankaDieLiftJointD8BigEnvCfg`/`D10Big`/`D12Big`), each shape's
own freshly-derived 48.0mm-targeting scale (native mesh bboxes differ per
shape, so the d20's own 0.001585 constant does not transfer), single
undiluted 48mm population, 3 seeds x 3 shapes. Code committed
(`1ce90a4`); training execution is in progress as of 2026-07-18, dispatched
via the new [[pi-as-primary-agent-gpu-dispatch]] desktop-first routing
rather than cloud (this exact task previously hit real cloud infra
friction — SPOT preemption, pip-cache corruption, a `Linger=no` systemd
default killing a detached install — before the desktop dispatch system
existed; see `BACKLOG.md`'s "Task 3.5 execution backend" entry). Result
not yet known — this article will be updated once it reports.

## Open, not yet decided

Task 4 (distillation) is currently unsatisfiable — zero working
specialists exist among the four shapes attempted so far (d20: 0/120
across two attempts; d8/d10/d12: 0/9 at real size, 3.5's 48mm-parity
result pending). Per the plan's own discipline, this is an explicit
controller-level gate, not something either training task decided
unilaterally.
