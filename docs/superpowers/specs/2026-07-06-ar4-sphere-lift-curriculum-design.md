# AR4 sphere lift: dense curriculum-gated height reward

## Problem

Per `ROADMAP.md`'s "grasp/lift never emerges" follow-up, the ContactSensor
experiment (`docs/superpowers/specs/2026-07-05-ar4-sphere-contact-sensor-design.md`)
achieved a first for this session: the policy reliably closes the gripper
on the sphere with genuine, sustained bilateral contact (`grasp_contact`
converges to ~92% per-step, `Episode_Reward/grasp_contact` ~18.4/20). But
`lifting_sphere` (the existing binary `object_is_lifted` term, `weight=25.0`,
fires only once `root_pos_w[:, 2] > 0.03`) stayed at essentially exactly
`0.0000` for the entire 1500-iteration run — not a near-miss, a flat zero.

Pulling the actual TensorBoard curves (not just first/last/max) shows why
this isn't simply "needs more training": `grasp_contact` plateaus by
iteration ~600-750 and stays flat for the remaining ~750 iterations, while
`lifting_sphere` never once rises above noise (one blip of `0.0016` at
iteration 300, otherwise exactly `0.0000` throughout, including the entire
back half of training where grip was already stable). The policy converges
to a stable "hold a safe grip on the ground" optimum and never explores
past it — because `lifting_sphere` is a hard threshold with zero gradient
below `0.03`, there is no reward signal anywhere in the 0-to-threshold
range to guide exploration toward it. This matches the Dense2Sparse
literature already cited in this ROADMAP thread (Luo et al.,
arXiv:2003.02740): a sparse, threshold-only success signal is a known
failure mode for exactly this kind of RL exploration problem, and the
established fix is a dense shaping term active early, tightened/replaced
later — not simply raising the sparse term's weight (which doesn't add any
gradient in the region that matters).

## Decision

Add a **dense, curriculum-gated height-progress reward** rather than
modifying `lifting_sphere` itself:

1. A new reward term, `lift_height_progress` — continuous (`tanh`-shaped),
   rewarding *any* upward progress from the sphere's resting height, not
   just crossing the 0.03m threshold. Unlike `lifting_sphere`'s binary
   cliff, this gives the policy a real gradient to climb during ordinary
   PPO exploration.
2. **Curriculum-gated, not always-on**, using Isaac Lab's own
   `modify_reward_weight` curriculum term (`isaaclab.envs.mdp.curriculums`,
   already shipped and used by the Franka lift task this repo's own reward
   functions were adapted from — reused directly, not reinvented). The
   term's weight starts at `0.0` and switches to a real value once
   `env.common_step_counter` crosses a threshold calibrated to this run's
   own data: grip converges by iteration ~700, so the switch is set at
   iteration 700 × `num_steps_per_env` (24) = **16800**. This keeps phase 1
   (reach + grip) training identical to what already worked, then opens a
   dedicated ~800-iteration window for the policy to discover lifting once
   grip is already stable — rather than the two problems competing for
   exploration budget from iteration 0, or a fully-open-ended dense reward
   changing phase-1 dynamics from the start.
3. `lifting_sphere` itself is **not modified** — it remains the true binary
   success signal (still `weight=25.0`, still `minimal_height=0.03`). This
   is a single new additive term plus a curriculum schedule on that new
   term only.

### Why not the alternatives

- **Just raise `lifting_sphere`'s weight further** (already tried once
  this session, 15.0→25.0, no-op): doesn't add any gradient in the
  zero-to-threshold region — a bigger reward for an event that never
  triggers is still zero expected reward. Ruled out by this run's own
  data (weight is already 25.0, the largest term in the reward, and still
  exactly 0.0000).
- **Always-on dense term from iteration 0, no curriculum:** risks
  destabilizing phase-1 grip learning by changing the reward landscape
  the policy is already known to converge well under (this run's own
  `grasp_contact`/`reaching_sphere` convergence curves are the only
  positive data point this whole session has produced — not worth risking
  without a specific reason to believe it would help, and the curriculum
  gate is a one-line, precedented mechanism that avoids the risk for free).
- **A full hierarchical reach-then-grasp-policy split** (the "last resort"
  option flagged back in the original literature research): much larger
  architectural change; not warranted before a much cheaper, literature-
  backed dense-shaping-plus-curriculum attempt has been tried.

## Design

### 1. Dense reward function (`tasks/ar4/mdp.py`, new function alongside `contact_grasp_bonus`)

```python
def lift_height_progress(
    env: ManagerBasedRLEnv,
    height_std: float,
    rest_height: float,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Dense reward for upward progress on the object, from its resting
    height - unlike lifting_sphere's binary object_is_lifted threshold,
    this gives a real gradient below the success threshold so ordinary PPO
    exploration has something to climb, rather than needing to stumble
    directly onto minimal_height with no intermediate signal. Curriculum-
    gated (see CurriculumCfg) rather than always-on, so early training
    (reach + grip) is unaffected until grip is already stable. See
    docs/superpowers/specs/2026-07-06-ar4-sphere-lift-curriculum-design.md.
    """
    object: RigidObject = env.scene[object_cfg.name]
    height_above_rest = torch.clamp(object.data.root_pos_w[:, 2] - rest_height, min=0.0)
    return torch.tanh(height_above_rest / height_std)
```

Parameters (decided, not left to implementer judgment):
- `rest_height = 0.009` — the sphere's resting world-Z (its own radius,
  `objects_cfg.py`'s `SPHERE_CFG`, resting directly on the ground plane).
  Subtracting this makes the term read exactly `0.0` at rest, not a
  nonzero baseline just for sitting still.
- `height_std = 0.01` — chosen so the term is already near-saturated
  (`tanh(0.021/0.01) ≈ 0.97`) by the time the object reaches
  `lifting_sphere`'s own `0.03` threshold (`0.03 - 0.009 = 0.021` real
  height gained), so this term smoothly leads into, rather than competes
  with or overshoots, the existing success criterion.

### 2. Reward registration (`tasks/ar4/pickplace_env_cfg.py`'s `RewardsCfg`)

```python
lift_height_progress = RewTerm(
    func=ar4_mdp.lift_height_progress,
    weight=0.0,  # curriculum-gated: raised to 15.0 once grip has converged (see CurriculumCfg)
    params={
        "height_std": 0.01,
        "rest_height": 0.009,
        "object_cfg": SceneEntityCfg("sphere"),
    },
)
```

Weight `15.0` after the curriculum switch: comparable in scale to
`sphere_goal_tracking`'s `16.0` (both are dense, saturating terms), and
smaller than `lifting_sphere`'s `25.0` — this term is scaffolding toward
the real success signal, not a replacement for it, so it should matter
less than the true binary success once both are active.

### 3. Curriculum (`tasks/ar4/pickplace_env_cfg.py`, new `CurriculumCfg` — this
class doesn't exist yet in this file; `Ar4PickPlaceEnvCfg` currently has no
`curriculum` field at all)

```python
from isaaclab.envs.mdp.curriculums import modify_reward_weight
from isaaclab.managers import CurriculumTermCfg as CurrTerm

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP. Ramps in the dense lift-height shaping
    term only after grip has converged (see this run's own TensorBoard data
    in docs/superpowers/plans/2026-07-06-ar4-sphere-contact-grasp-reward-report.md),
    rather than competing with grip-learning from iteration 0."""

    lift_height_progress = CurrTerm(
        func=modify_reward_weight,
        params={"term_name": "lift_height_progress", "weight": 15.0, "num_steps": 16800},
    )
```

Add `curriculum: CurriculumCfg = CurriculumCfg()` to `Ar4PickPlaceEnvCfg`.
`num_steps=16800` = iteration 700 × `num_steps_per_env` (24, from
`tasks/ar4/agents/rsl_rl_ppo_cfg.py`) — `env.common_step_counter` increments
once per `env.step()` call regardless of `num_envs` (it's a global,
per-rollout-step counter, not per-parallel-environment), so this
corresponds to iteration 700 of the planned 1500-iteration run, the same
schedule this repo's own Franka-lift-derived `CurriculumCfg` reference
uses `num_steps` for.

### 4. Single-variable-ish experiment, but building on the already-changed baseline

This experiment adds ONLY `lift_height_progress` + its curriculum. Every
other term (`reaching_sphere`, `lifting_sphere`, `grasp_contact`,
`sphere_goal_tracking`, `sphere_goal_tracking_fine_grained`, `action_rate`,
`joint_vel`) and their weights stay exactly as they are on `main` right
now — including the corrected `_EE_OFFSET=0.036` and the `grasp_contact`
reward from the prior experiment, both of which are now the baseline, not
a variable being tested here.

## Verification plan

Same rigor as the ContactSensor experiment: smoke test
(`--num_envs 16 --max_iterations 2`, confirms the new term + curriculum
wire up without error and the `Active Curriculum Terms` table in the
startup printout now shows `lift_height_progress` instead of being empty),
full run (`--num_envs 4096`, 1500 iterations, monitor
`Episode_Reward/lift_height_progress`, `Episode_Reward/lifting_sphere`,
`Episode_Reward/grasp_contact`, `Episode_Termination/sphere_reached_goal`
via TensorBoard scalars — specifically checking that `lift_height_progress`
stays at exactly `0.0` before iteration 700 (confirming the curriculum
gate genuinely didn't affect phase 1) and that `lifting_sphere` moves off
`0.0000` at some point after the switch), then real eval (`--episodes 10`)
with frame-extracted video inspection of all 10 episodes, specifically
checking whether the sphere visibly leaves the ground this time.

If `lifting_sphere` still doesn't move off `0.0000` after the curriculum
switch, this is a genuinely new data point (dense shaping + curriculum
tried, not just another flat reward-weight tweak) — per
`superpowers:systematic-debugging` Phase 4.5, that would be grounds to
consider the hierarchical reach-then-grasp-policy option, or to question
whether the gripper's physical grip (closed-jaw force against this
0.01kg/9mm sphere) is even strong enough to support a lift at all
(a physical-plausibility question, not a reward-design one) — flag back
to the user rather than attempting a further reward-only tweak.

## Revision: remove the curriculum gate, raise the weight (per user request to keep iterating)

**Result of the experiment above (full data in
`docs/superpowers/plans/2026-07-06-ar4-sphere-lift-curriculum-report.md`):
the curriculum mechanism fired exactly as designed at iteration 700, but
its real-world effect was negligible** — the logged `Episode_Reward`
max of `0.0065` is `weight(15.0) ×` the mean per-step `tanh` value, so the
real per-step `tanh` was only ~`0.00043`, corresponding to ~0.0043mm of
real height gain (via `tanh`'s small-angle behavior) — many orders of
magnitude short of the 21mm `lifting_sphere` requires. `lifting_sphere`
never rose above noise. Real eval: 0/10 episodes showed any lift, same
"reach, grip, freeze" signature as before. Diagnosis: `grasp_contact` was
already at ~17.8/20 (essentially its plateau) by iteration 700 — the
static-grip behavior had already converged too deeply for a newly-
introduced incentive to perturb it in the remaining ~800 iterations.

**Fix: drop the curriculum, make `lift_height_progress` active from
iteration 0.** The curriculum was originally added out of caution — "risks
destabilizing phase-1 grip learning by changing the reward landscape the
policy is already known to converge well under" (see this spec's original
"Why not the alternatives" section) — but that caution is no longer
warranted, for a structural reason visible in the reward function itself:
`lift_height_progress = tanh(clamp(height - rest_height, min=0) /
height_std)` is mechanically `~0` whenever the object hasn't actually
been lifted, which is impossible before grip exists. The term literally
cannot pay out during phase 1 (reach + grip), regardless of its weight —
so there was never a real risk to protect against by delaying its
activation, only an assumed one. Delaying it instead cost the experiment
its entire runway: by the time it turned on, the alternative (static-hold)
behavior was already entrenched. Making it active from iteration 0 means
the first accidental upward jostle of the gripped sphere — however early,
however small — gets reinforced immediately, rather than only after
iteration 700.

**Also raise the weight, 15.0 → 25.0** (matching `lifting_sphere`'s own
weight), so that once real height gain becomes achievable, this dense
shaping term is at least as influential as the binary success signal it's
meant to lead into, rather than being subordinate to it.

### Design changes

1. Remove the `CurriculumCfg` class and the `curriculum` field from
   `Ar4PickPlaceEnvCfg` entirely (this plan introduced both; nothing else
   in this task's history uses a curriculum, so removing them is a clean
   revert of just this one piece, not a partial rollback).
2. Change `lift_height_progress`'s `RewTerm` `weight` from `0.0` to
   `25.0` directly (no curriculum-driven change over time).
3. `height_std=0.01` and `rest_height=0.009` are unchanged — these
   parameters were never the suspected problem (the math already gives
   meaningful reward for sub-millimeter progress: `tanh(0.001/0.01) ≈
   0.0997`, ~10% of max for just 1mm of lift), so there's no evidence
   basis to retune them yet. If this revision also fails to produce real
   lifting, retuning these would be a more targeted next hypothesis than
   this revision's curriculum-timing fix.
4. Every other reward term (`reaching_sphere`, `lifting_sphere`,
   `grasp_contact`, `sphere_goal_tracking`, `sphere_goal_tracking_fine_grained`,
   `action_rate`, `joint_vel`) and `_EE_OFFSET` remain untouched — same
   single-variable discipline as every prior experiment.

### Verification plan

Same as before: smoke test, full 1500-iteration run (monitor
`Episode_Reward/lift_height_progress` from iteration 0 this time — expect
it to start contributing meaningfully as soon as `grasp_contact` starts
converging, not stay at a fixed `0.0` through iteration 700), then real
eval with frame-extracted video inspection of all 10 episodes.

If `lifting_sphere` still doesn't move off `0.0000` even with the term
active from the start at a higher weight, this rules out "curriculum
timing" as the explanation too — per `superpowers:systematic-debugging`
Phase 4.5 (now three real attempts on the reward/curriculum axis for this
specific sub-problem: sparse-only, curriculum-gated dense, always-on
dense), the next step should not be a fourth reward-only tweak. Flag back
to the user; the remaining candidates are the hierarchical policy split or
the physical-plausibility check on the gripper's actual lifting force,
both already named in ROADMAP.md.
