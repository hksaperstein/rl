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

## Task 3.5: 48mm-parity check for d8/d10/d12 — complete 2026-07-19, one partial positive

Neither Task 2 nor Task 3 ever tested d8/d10/d12 at a single, undiluted,
48mm population the way the original [[asset-bisect]] ladder did for the
cube (3/3) and d20 (1/3). This task closed that gap: three new env cfg
classes (`FrankaDieLiftJointD8BigEnvCfg`/`D10Big`/`D12Big`), each shape's
own freshly-derived 48.0mm-targeting scale (native mesh bboxes differ per
shape, so the d20's own 0.001585 constant does not transfer), single
undiluted 48mm population, 3 seeds x 3 shapes, 1500 iterations each.
d8-big trained/evaluated by a prior desktop-dispatch agent
([[pi-as-primary-agent-gpu-dispatch]]); d10-big/d12-big (6 seeds) trained
on GCP cloud (SPOT, switched to on-demand after 3 preemptions in ~3hrs —
see `BACKLOG.md`'s "Task 3.5 cloud completion" entry).

**Full grid (envs with sustained lift / 8 per seed):**

| shape (48mm) | seed 42 | seed 123 | seed 7 | seeds-with-discovery |
|--------------|---------|----------|--------|-----------------------|
| d8-big       | 0/8     | 0/8      | 0/8    | 0/3                   |
| d10-big      | 0/8     | 0/8      | 0/8    | 0/3                   |
| d12-big      | 0/8     | **4/8**  | 0/8    | 1/3                   |

Compared to [[asset-bisect]]'s own undiluted-48mm baselines (cube 3/3
seeds full 8/8, d20 1/3 seeds full 8/8): **d8 and d10 remain completely
null even at pure 48mm parity — shape itself is a real barrier for these
two, not population dilution or absolute scale** (matches Task 2's
original ~16-18mm finding). **d12 shows a genuine but weaker echo of
d20's own 1/3-seed pattern** — same lucky seed (123) as d20's original
bisect discovery, but only half the envs within that seed (4/8, not
d20's full 8/8).

Independently re-verified d12-big seed123's positive result is real, not
a third occurrence of the reset-boundary/settle-window artifacts already
fixed twice in this experiment: re-implemented the settle/gain logic from
scratch against the raw `.npy` and found a smooth continuous rise
(steps ~40-115) to a stable plateau (~0.20-0.23m absolute height, inside
`lift_env_cfg.py`'s own goal-z range `(0.25, 0.5)`) held for the rest of
the episode — physically consistent with a real grasp-lift-carry, not a
contact-explosion glitch (no violent single-step jumps). Could not
directly visually confirm via video this time — `franka_checkpoint_review.py`'s
camera is fixed on env_0, which happened to be one of the *non*-lifting
envs in this run; the verdict rests on the raw-trajectory physics
reasoning, disclosed as a real tooling limitation rather than papered
over. Found (but did not need to fix) a related measurement caveat: the
settle-detector's tolerance is too tight to ever recognize a *held*
object's natural jitter as "settled" — logged to `BACKLOG.md` for a
future pass.

Also found and fixed during this task: d8-big seed42/seed123's synced
eval artifacts actually predated the `977a748` measurement fix (verified
via GCS object timestamps), contradicting the task's own dispatch brief
— re-ran eval-only (no retrain) against the current fixed script,
reconfirmed 0/8 unchanged. Also found a new cloud-infra bug (SPOT
preemption truncating a checkpoint file mid-write to 0 bytes) and fixed
the resume logic to validate checkpoint size before trusting it — see
`docs/cloud/dispatch-checklist.md`'s known-gaps list.

Full grid, reasoning, and cost: `ROADMAP.md`'s Task 3.5 entry (search
"48mm-parity check").

## Open, not yet decided

Task 4 (distillation) status per this task's own gate discipline: d8 and
d10 remain fully null at 48mm parity (2/4 candidate shapes), d12 shows
one partial positive (1/3 seeds, half-envs-within-seed), d20 itself is
still gated on Task 3's own open dilution-vs-shape ambiguity (never
retested at undiluted 48mm together with geometry conditioning). Whether/
how Task 4 proceeds given this mixed grid is an explicit controller-level
decision, not made by either training task unilaterally.
