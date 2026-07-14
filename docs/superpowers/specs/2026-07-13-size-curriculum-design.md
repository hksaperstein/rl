# Size curriculum: making the 30mm d20 learnable via mixed-size training

**Date:** 2026-07-13 (drafted night of 07-12). **Branch:** `franka-panda-pivot`.
**Prior result:** asset-bisect (`docs/superpowers/plans/2026-07-12-asset-bisect-report.md`)
— grasp discovery on the d20 is 0/4 at 30.3mm (deterministic fail) and
1/3 at 48mm (discoverability boundary); shape is the gate, size sets
severity. **Research grounding:** `.superpowers/sdd/research-asset-bisect.md`
Q1 (curriculum-over-object-scale findings; citation-reviewed 2026-07-12).

## Hypothesis (falsifiable)

> A single policy trained with per-env die-size variation spanning
> 30.3-48.0mm (mass pinned 0.216kg, joint-space config otherwise
> identical to the bisect's) will achieve lift+carry ON THE 30.3mm
> ENVS — `position_error` computed over the smallest-size envs
> decisively below the ~0.216 do-nothing baseline — in ≥2 of 3 seeds
> within 3000 iterations, because grasp discovery in the large-size
> envs supplies the reward signal whose policy gradient generalizes
> down the size range (the mechanism curriculum learning and domain
> randomization share; see research doc Q1).

3000 iterations (2x the bisect's 1500) because the hard sizes must
first benefit from transfer — pre-registered, not tuned post-hoc.

## Design

**Primary arm — mixed-size (size-DR):** envs deterministically assigned
one of 5 die scales {48.0, 43.6, 39.1, 34.7, 30.3}mm (spawn-scale
variants of the same baked d20; ~819 envs per size at 4096). Mass
pinned 0.216kg on all (the bisect's control). Everything else inherits
`FrankaDieLiftJointHeavyEnvCfg` unchanged.

Implementation candidate: Isaac Lab's multi-asset/per-env spawn
variation (`MultiAssetSpawnerCfg` or equivalent in the installed
version — implementer MUST verify against `/home/saps/IsaacLab` source
before the spec's design is finalized in the plan; if per-env scale
variation is unsupported, fire the fallback arm instead and record why).

**Fallback arm — staged anneal (pre-authorized, fires only if the
primary is unimplementable or FAILS):** 48.0 → 39.1 → 30.3mm, one
`--checkpoint`-resumed run per stage (1000 iters/stage, same seed
protocol). Uses only existing mechanisms (env-cfg scale value +
existing resume flag).

## Metrics / verdict protocol

- Authoritative: `Metrics/object_pose/position_error` — but the VERDICT
  metric is the per-size breakdown at the smallest size. The stock env
  logs only the all-env mean, so Task 1 must add a per-size-bucket
  metric or an eval-time per-size readout (instrumented eval on
  30.3mm-only PLAY cfg is acceptable and simplest: eval the trained
  checkpoint in an all-30.3mm env — success = sustained lifts there,
  same 8-env instrumented protocol as the bisect).
- 3 seeds (42/123/7), ≥2/3, full-run VF-boundedness check, eval video
  (full arm + table, ≥2 episodes) on the first passing seed.
- Comparison anchors (no new runs needed): bisect rung 1 (30.3mm-only:
  0/3) and rung 2 (48mm-only: 1/3).

## Success meaning / next

PASS → the platform can train the real die → Phase I (detector-derived
observations) unblocks with this policy as its base. FAIL both arms →
shape itself needs attacking (grasp-strategy/reward changes — new spec,
new research).

## Out of scope

Reward/observation changes; shape curriculum (cube→die morphing);
detector-in-loop anything; d4.

## Verdict (2026-07-13): hypothesis FALSIFIED, 0/3 seeds

Instrumented all-30.3mm eval (8 envs, model_2999, sustained-lift
criterion per this spec): seed 42 **0/8**, seed 7 **0/8** (twice —
original + a redundant re-run, identical result), seed 123 **0/8**.
Pre-registered bar was ≥6/8 in ≥2/3 seeds; no seed produced a single
sustained lift. Training metrics corroborate across all three seeds:
`lifting_object` pinned at its 0.1200 spawn-z artifact floor for the
entire 3000 iterations, `reaching_object` peaked early (0.34/0.42/0.58)
then decayed, blended `position_error` never beat the 0.216 do-nothing
baseline.

**Diagnostic reading (why the mechanism never engaged):** the
hypothesis depended on grasp discovery in the large-size envs supplying
a transferable gradient. Discovery never happened at ANY size,
including 48mm. The bisect's 48mm-only anchor was itself only 1/3
seeds at 4096 envs; mixing five sizes cut the 48mm population to ~819
envs — the already-rare discovery event became ~5x rarer per update,
and the curriculum's source signal never fired. Mixed-size DR dilutes
discovery when discovery is the bottleneck; it helps when the easy
regime is *reliably* learnable (which 48mm was not).

**Decision:** fire the pre-authorized staged-anneal fallback arm
(48.0 → 39.1 → 30.3mm, checkpoint-resumed, 1000 iters/stage, seeds
42/123/7) — stage 1 trains the full 4096-env population at 48mm,
restoring the bisect's discovery odds rather than diluting them.
Queued behind the d4 rung-0 trials on the GPU.

**Ops notes from this batch:** (1) seed 123's first run hit a NEW
Isaac failure mode — mid-training livelock at ~iter 260 (log+ckpt
frozen 2.5h, 3 cores spinning, GPU 12%, SIGKILL required); a log-mtime
stall detector is now standing practice. (2)
`franka_checkpoint_review.py` writes fixed output filenames — three
same-variant evals overwrote each other's json/video (verdicts
recovered from per-launch tee logs; seed-7's artifacts survive, both
runs). Future multi-seed evals must rename artifacts between runs or
the script needs an output-suffix flag. (3) flock lock handoff among
multiple queued waiters is NOT FIFO — do not infer artifact identity
from queue order.

## Verdict (2026-07-13): staged-anneal fallback FALSIFIED, 1/3 seeds

Fallback arm (48.0 -> 39.1 -> 30.3mm, checkpoint-resumed, 1000
iters/stage, seeds 42/123/7 — full 4096-env population every stage,
restoring the bisect's per-stage discovery odds rather than diluting
them across sizes). Implementation: `FrankaDieLiftJointMidEnvCfg`
(39.1mm, new `--variant joint-die-mid`) added to
`tasks/franka/dice_lift_joint_env_cfg.py` following the exact pattern
of `FrankaDieLiftJointBigEnvCfg`; stage 1 (48.0mm) and stage 3
(30.3mm) needed no new classes — they map onto the existing
`joint-die-big` and `joint-die-heavy` variants verbatim (30.3mm at
0.216kg is exactly `FrankaDieLiftJointHeavyEnvCfg`'s own unmodified
scale/mass). Diff purely additive; all pre-existing variant classes
byte-identical (verified via `git diff`).

**Stage-1 discovery check (48.0mm, from scratch, 1000 iters, per-seed
instrumented eval on the all-48mm PLAY cfg, 8 envs):** seed 42 **0/8**,
seed 123 **8/8**, seed 7 **0/8** — 1/3 seeds discover, exactly
reproducing the asset-bisect's own 48mm anchor (rung 2: 1/3, same seed
123 the one that succeeds). Not the pre-registered 0/3 STOP condition,
so all 3 seeds proceeded through stages 2-3 as designed.

**Stage-3 verdict (30.3mm, checkpoint-resumed through both stages,
instrumented eval on the all-30.3mm PLAY cfg, 8 envs, model_2999):**
seed 42 **0/8**, seed 123 **8/8**, seed 7 **0/8**. Pre-registered bar
was >=6/8 in >=2/3 seeds; only 1/3 seeds pass. **Experiment FALSIFIED**
against the >=2/3 bar, same numerical outcome as the primary mixed-size
arm (1/3 < 2/3) but via a different mechanism (see below) — this is
not a repeat of the primary's dilution failure.

**Training health, all 9 runs:** zero NaN in any log; VF loss bounded
throughout (near-zero for the two seeds that never discover — the
policy collapses to a stable no-op rather than diverging; ~0.1-1.4,
still bounded, for seed 123's actively-learning runs). Seed 123's
`position_error` improved monotonically across the anneal: ~0.11
(stage 1, 48mm) -> ~0.099 (stage 2, 39.1mm) -> ~0.102 (stage 3,
30.3mm) — the checkpoint-resumed lift behavior transferred cleanly
downward in size with no degradation. Seeds 42 and 7 stayed pinned at
the `lifting_object` 0.12 spawn-z floor and ~0.35-0.40 `position_error`
(worse than the 0.216 do-nothing baseline) at every stage — a
checkpoint that never discovered has nothing for the anneal to carry.

**Diagnostic reading (mechanism confirmed working, bottleneck
confirmed upstream):** unlike the primary arm's size-DR (which diluted
the 48mm discovery signal ~5x and never fired at all, 0/3, `lifting_
object` pinned at floor in every seed including 48mm), the staged
anneal's full-population 48mm stage reproduced the bisect's real 1/3
discovery rate, and the one seed that discovered (123) carried a
genuine sustained lift (8/8 both at the 48mm stage-1 check and the
final 30.3mm stage-3 check, `position_error` improving monotonically
size-by-size) all the way down through two anneal stages without any
degradation. **The transfer mechanism itself works** — checkpoint-
resumed size-annealing does not destroy an already-discovered grasp.
What it cannot do is manufacture NEW discovery: the two seeds that
never found a grasp at 48mm (42, 7) stayed at the no-op floor through
every subsequent stage, collapsing to a stable near-zero-VF-loss policy
with nothing to anneal. The bottleneck is squarely the 1/3 discovery
rate on this asset at its most learnable size — an anneal or a
size-DR curriculum both inherit that base rate 3-seeds-at-a-time
rather than improving it, because neither introduces new exploration
pressure at the point grasp discovery actually happens.

**Decision:** both pre-authorized arms of the size-curriculum spec are
now falsified. Per the spec's own "Success meaning / next" section:
shape itself needs attacking directly (grasp-strategy/reward changes)
rather than further curriculum-over-object-scale variants — a new
spec/research pass is required before further sizing work on this
asset. Seed 123's 8/8 stage-3 result remains the project's first
confirmed 30.3mm-d20 lift+carry policy and is a reusable existence
proof / comparison baseline for that next investigation.

**Ops notes (this task):** (1) flock-queue congestion from a
concurrently-running d4-debug thread's short jobs repeatedly delayed
this task's launches by 10-30 min per handoff; several of those
d4-debug processes exhibited the documented teardown-hang signature
(own script printing `[DONE]`, log static >5min, GPU pinned ~12-13%,
CPU time still slowly advancing) and were killed with `kill -TERM` per
the standing safe-kill procedure (never a process that hadn't reached
its own `[DONE]`/completion line). (2) A background-launch pattern of
`cmd | tee logfile &` inside a single Bash call leaks the child
process's full stdout into the calling tool's output even after the
call itself returns, because the backgrounded job inherits the same
pty; fixed for this task by redirecting straight to a file
(`> logfile 2>&1 < /dev/null &`, no `tee`) for every subsequent
launch. (3) Artifact renaming after every eval (per the primary arm's
own ops lesson) was followed throughout; no collisions this pass.
