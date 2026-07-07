# Reward-rate arithmetic and the "grasp and freeze" bug class

## The pattern

Across this project's entire arc, the single most recurring, concrete bug
class is not "the reward doesn't encode the right idea" but **the net
per-step arithmetic of two or more competing reward terms silently makes a
static/frozen state the locally-optimal one to hold.** This is distinct
from reward hacking (a term rewarding the wrong proxy) and from sparse-
signal discoverability (a correct term too weak to ever be found) — it's a
straightforward sign/weight/rate miscalculation that happens to have a
large, real behavioral consequence.

## Instances

- **[[experiment-06-mirror-scene-stillness-penalty]]**: `stillness_penalty`'s
  function body already returned a signed value (`-1.0` when triggered),
  but its `RewardsCfg` registration used `weight=-2.0`. Since
  `RewardManager.compute()` multiplies `func(...) * weight * dt`, two
  negatives turned an intended penalty into a **+2.0*dt reward** for the
  exact stay-still-after-grasp behavior the term existed to punish. Caught
  by reading the raw TensorBoard scalar (`stillness_penalty` growing to
  +1.3 over training — impossible for a true penalty), not by trusting the
  design doc's stated intent.
- **[[experiment-08-classical-ik-guided-path]]**: the clearest quantitative
  confirmation of the whole bug class. `contact_grasp_bonus` (16.80 final
  episode-cumulative) outweighed `ik_guided_path_bonus` (0.14) by **~118:1**
  in the actual trained policy's behavior — externally cross-checked
  against Isaac Lab's own shipped lift task, IsaacGymEnvs `FrankaCubeStack`,
  and ManiSkill3 `PickCube-v1` source, all three of which gate downstream
  reward behind grasp/lift state; this repo's reward was the structural
  outlier.
- **[[experiment-09-antipodal-grasp-bonus]]**: the direct fix — weight
  reduced 20.0 → 3.0 alongside the antipodal geometric gate — reversed
  dominance to **~107:1 path-favoring**, the opposite direction, in one
  change.
- **[[experiment-12-stillness-reward-rate]]**: a subtler instance found by
  direct arithmetic rather than by inspecting a scalar anomaly — holding a
  grasp without further progress netted **+1.0/step**
  (`antipodal_grasp_bonus`'s continuous +3.0/step, only partly offset by
  `stillness_penalty`'s -2.0/step once a 25-step patience window elapsed).
  Fixing the weight (2.0 → 5.0, restoring a net -2.0/step) produced a
  scalar-mixed, video-inconclusive result — a reminder that fixing a real,
  verified reward-rate bug does not guarantee an observable behavior
  change, especially when it interacts with an already-entrenched policy.
- **[[experiment-14-reach-skip-curriculum]]**: the same antipodal-drop-vs-
  stillness-penalty pattern recurs a third time, now confounded by a
  changed episode/reset structure (episodes starting mid-task) that makes
  direct comparison to full-reach-included episodes harder to interpret.

## The recurring interpretive trap

Nearly every one of these instances comes with the same caveat: **a drop in
a grasp-quality metric (like `antipodal_grasp_bonus`) after a reward-rate
fix is not distinguishable, from scalars alone, between "the policy grasps
less" and "the policy grasps the same amount but holds the grasp for less
static time."** [[experiment-12-stillness-reward-rate]]'s report explicitly
documents the implementer's own report misreading this ambiguity as a
clean failure, and the controller rejecting that verdict as premature. This
is why every reward-rate fix in this project's later history is paired with
direct video inspection rather than trusted on scalars alone — and even
then, small (3/10 episode) samples are often not powered to resolve the
ambiguity definitively.

## Status as of this pass

Experiment 15 (in progress, not yet covered by this wiki pass) continues
directly on this axis per its own design spec
(`docs/superpowers/specs/2026-07-07-ar4-experiment15-reward-shaping-design.md`) —
wiring in the existing `ground_penalty` function and raising
`antipodal_grasp_bonus`'s weight, motivated by Experiment 14's base-collapse
finding.

## Related experiments

[[experiment-06-mirror-scene-stillness-penalty]], [[experiment-08-classical-ik-guided-path]],
[[experiment-09-antipodal-grasp-bonus]], [[experiment-11-taskspace-ik]],
[[experiment-12-stillness-reward-rate]], [[experiment-14-reach-skip-curriculum]]
