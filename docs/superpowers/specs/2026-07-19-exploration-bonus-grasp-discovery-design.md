# Exploration-bonus grasp discovery (H1: GRM-grounded, action-dependent attempt bonus) — design spec

## Context

This project's standing "reach solved, grasp never discovered" pattern
(`kb/wiki/concepts/reach-grasp-lift-gap.md`) has a new, direct confirmation:
the target-selection-clutter experiment's Stage SO checkpoint
(`kb/wiki/experiments/target-selection-clutter.md`, 0/8 both shapes) was
instrumented (`scripts/_diag_gripper_lowpass_check.py`) and shown to output a
raw gripper action that is **positive ("open"), confidently (+0.48 to
+7.77), in 100% of sampled steps, across all 8 envs, for a full episode** —
`frac_steps_raw_action_negative = 0.000` everywhere. The addendum to
`docs/superpowers/specs/research/2026-07-19-exploration-reward-expansion-literature.md`
(hereafter "the research doc") also ruled out an actuator/action-space
explanation for this directly: this project's gripper action
(`mdp.BinaryJointPositionActionCfg`) is a hard sign-threshold mapping, not
the continuous-velocity low-pass-filtered mapping Neunert et al.
(arXiv:2001.00449) describe, so there is no small-noise-attenuation
mechanism that could explain the policy never once sampling a
closure-attempt. **This is a reward/exploration-discovery problem: the
converged policy has confidently learned to keep the gripper open, and
nothing is filtering that decision out.**

The research doc surveyed and ranked three candidate interventions,
recommending **H1 — a non-Markovian-aware, action-dependent potential-based
exploration bonus for gripper-closure attempts near the object** — as
primary, built on two 2024/2025 extensions to Ng, Harada & Russell's
potential-based reward shaping (PBRS, ICML 1999): **PBIM/GRM** (Forbes,
Villalobos-Arias, Wang, Jhala, Roberts, "Potential-Based Intrinsic
Motivation: Preserving Optimality With Complex, Non-Markovian Shaping
Rewards," arXiv:2410.12197) and **ADOPS** (Forbes, Wang, Villalobos-Arias,
Jhala, Roberts, "Action-Dependent Optimality-Preserving Reward Shaping,"
arXiv:2505.12611). This spec designs that H1 experiment. It is a Tier 1
structural experiment (a new reward *term* and a new stateful reward-manager
mechanism, not a weight/threshold tweak) per CLAUDE.md's Workflow section,
gated on the research doc above. **This spec is design only — no
implementation plan, no code changes, no Isaac Sim launches.**

This project has direct, painful prior experience with exactly this class of
mechanism going wrong:
[[experiment-05-potential-based-reward-shaping]] replaced six additive
reward terms with a running-max potential (`gamma * new_potential -
prev_potential`), whose docstring claimed the result was "always >= 0" —
false whenever the agent merely *holds* its best-ever potential without
improving (`Φ*(gamma-1)`, negative for any `gamma<1`), and this drove the
policy to a 0/10 "never approach" optimum, worse than doing nothing.
`kb/wiki/concepts/reward-hacking-and-sparse-discoverability.md` frames the
general tradeoff this project keeps re-hitting: a term loose enough to be
*discoverable* is usually loose enough to be *hacked*; a term tight enough
to be *correct* is usually too tight to be *discovered*. H1's whole appeal
is a formal, not folk, escape from that tradeoff — but only if this spec
gets the actual theorem's conditions right, which is where Experiment 5
failed. This spec's design was independently re-derived directly from both
papers' equations (not from the research doc's title/abstract-level
citations alone) specifically to avoid repeating that failure — see
"Research grounding" and "A refinement on the research doc" below.

## Research grounding — verified directly against both papers' actual text

Both papers were fetched (arXiv PDF) and their equations read directly for
this spec, not taken from the research doc's citation-level summary alone,
per this project's citation-verification practice.

**PBIM/GRM (arXiv:2410.12197):**

- **Theorem 1** ("Sufficient Condition For Optimality," p.9): a shaping
  reward `Ft = γΦt+1 − Φt` leaves the set of optimal policies unchanged if
  the boundary condition (Eq. 25) holds: `E[γ^(N−t)ΦN − Φt] = Φ'_t ∀t`,
  where `Φ'_t` is any function **constant with respect to the current
  action `a_t`** (note: not "constant with respect to time" — the
  requirement is action-independence of this specific expectation, at each
  `t`). This is a direct, proved generalization of Ng/Harada/Russell (1999),
  whose own theorem assumed a Markovian `Φ(s)`; Theorem 1's proof (Eq.
  20–23) is a pure telescoping-sum identity that holds for a
  **history-dependent (non-Markovian) `Φt`** — this is the exact class of
  potential Experiment 5's running-max `Φ` belongs to, and which
  Ng/Harada/Russell's own proof never covered.
- **Assumption 1** ("future-agnostic," p.12): `Ft` must be constant with
  respect to *future* actions `a_{t'>t}` — it may depend on the state and on
  the action taken *at or before* `t`, just never on an action not yet
  taken. **PBIM** (Eq. 32–33) is one concrete, "plug-and-play" way to
  satisfy Theorem 1 for any `Ft` meeting Assumption 1: choose
  `Φt = -U^π_t` (the policy's own future return from `t`), which — by pure
  algebra, not a hand-derived formula — collapses to "give `Ft` unchanged at
  every non-final step, then subtract the full discounted sum of all prior
  `Ft`'s at the very last step" (Eq. 33). **GRM** (Sec. 5, Eq. 44–49)
  generalizes this further: a "matching function" `m_{t,t'} ∈ [0,1]`
  specifies *when* each `Ft'` gets subtracted back (not necessarily only at
  the last step), subject to being "fully-matching" (Eq. 46, every `Ft'` is
  subtracted exactly once in total) and "future-agnostic" (Eq. 47, never
  subtracted before it was earned). PBIM is proved to be one specific GRM
  instance (Eq. 49); GRM's own delay-parameterized family (Eq. 8 in the
  ADOPS paper, restated below) lets the subtraction happen a bounded `D`
  steps later instead of waiting for the episode's end.
- **No running-max discussion, checked directly**: neither paper's text
  discusses a running-max potential as a worked example. The paper's
  "always ≥ 0"-*adjacent* mechanism (Eq. 37/38) is a running-**mean**
  subtraction for variance reduction, a different thing entirely. This
  confirms Experiment 5's bug class (a hand-derived non-Markovian potential
  with an incorrect informal non-negativity claim) is not something either
  paper directly patches — what they provide instead is a **formally proved
  recipe** (Theorem 1 + Assumption 1 + the GRM matching-function conditions)
  that removes the need to hand-derive (and get wrong, as Experiment 5 did)
  a bespoke correctness argument in the first place.

**ADOPS (arXiv:2505.12611):**

- Confirms directly (Sec. 4.1, Eq. 12–14) that *classical* PBRS requires
  `Q*_I(s,a)` to be action-independent (`= V*_I ∀a`) to guarantee optimality
  — this is the restriction ADOPS is built to remove.
- Its mechanism (Eq. 16, Theorem 5.2) is a **real-time, per-step correction
  term** `F2`, computed from four separately-learned critics
  (`V̂^π_E, V̂^π_I, Q̂^π_E, Q̂^π_I` — extrinsic and intrinsic value/Q
  estimates), that pushes the intrinsic reward up or down *exactly enough*,
  every step, to keep the ordering of extrinsically-optimal vs.
  extrinsically-suboptimal actions from ever flipping. Its guarantee
  (Theorem 5.2) additionally requires **Assumption 5.1** ("policy
  stability" — the training algorithm never converges to a policy for which
  a single-action perturbation is strictly preferred).
- ADOPS's own Sec. 6 empirical section (Montezuma's Revenge, RND) is
  directly informative for a design choice below: **plain end-of-episode
  PBIM catastrophically fails** in that environment — its correction term's
  denominator (`1/γ^(N-1)`) explodes for long episodes/low γ
  (`N=4500, γ=0.99` → `~1/0.99^4500 ≈ 10^19`), saturating the policy's
  action probabilities and preventing any real learning (Appendix A.2).
  **GRM with a short delay outperforms full-episode PBIM even without that
  explosion risk** — their own hyperparameter sweep over GRM's delay `D`
  (Eq. 8) found **`D=1` was the best-performing value tested** (Sec. 6.2,
  Appendix A.3), better than `D=N-1` (equivalent to PBIM).

## A refinement on the research doc: GRM alone (not full ADOPS) already supports a same-step action-dependent attempt bonus

The research doc's ranking (§4a) characterized ADOPS as necessary for "a
small proximity-gated exploration bonus for attempting gripper closure near
the object," on the grounds that "vanilla state-only PBRS structurally
cannot express [this] without either losing the policy-invariance guarantee
or requiring an awkward state-augmentation workaround." Having now read
both papers' actual theorems directly (not just their abstracts), **this is
not quite right, and this spec corrects it rather than propagating it**:
Assumption 1's "future-agnostic" restriction only forbids `Ft` from
depending on actions taken *after* `t` — it does **not** forbid `Ft` from
depending on the action taken *at* `t`. Theorem 2's own proof (in the PBIM
paper, `docs`/scratch-verified directly, see "Research grounding" above)
shows the boundary/correction term at any time `t` sums only over `F_0
... F_{t-1}` — strictly *past* rewards — so `Ft`'s own dependence on `a_t`
never has to cancel algebraically at the same step; it gets matched later,
by construction, regardless of whether it depended on `a_t` or not. **A
same-step "reward attempting gripper closure right now" bonus is exactly
the kind of `Ft` PBIM/GRM's Assumption 1 was built to allow** (it is, in
fact, the *simplest* case: a function of only the current `(s_t, a_t)`, no
history at all) — full ADOPS's dual-critic, real-time-corrected machinery
is not required to get this specific pattern's policy-invariance guarantee.

**Consequence for this spec:** H1's primary proposed mechanism is **GRM
with a short delay (`D=1`, following ADOPS's own empirical finding above)**,
not full ADOPS. Full ADOPS is retained as a documented, flagged escalation
path (see "Global constraints"), not this spec's primary proposal, because
it requires a materially larger new piece of infrastructure (four separate
learned critic heads, patched into `rsl_rl`'s own PPO update — the same
"larger and riskier than reusing already-validated machinery" judgment this
project already made once, in the d8/d10 demo-warmstart spec, when it chose
DAPG's BC-pretrain half over DAPG's own live augmented-loss half for an
analogous reason) that this simpler, equally rigorously-grounded mechanism
does not need.

**A second, project-specific reason to prefer a short delay over
end-of-episode PBIM, beyond ADOPS's own Montezuma finding:** this project's
PPO recipe (`tasks/franka/agents/rsl_rl_ppo_cfg.py`) uses
`num_steps_per_env=24` — GAE/advantage estimates are computed over rolling
24-step segments, not full 250-step episodes
(`episode_length_s=5.0`/`decimation=2`/`dt=0.01` → 250 steps,
`lift_env_cfg.py:390-391`). A correction applied at the very end of a
250-step episode would almost always land in a *different* 24-step PPO
rollout segment than the original bonus it is supposed to offset — the
theorem's guarantee is about the environment's true optimal policy under
exact expectations, not about a single finite on-policy gradient batch, so
an end-of-episode correction's cancellation would be true only in the
limit, not observably present in any one PPO update. `D=1` keeps the
original bonus and its correction in the *same or adjacent* rollout
segment, which is both what ADOPS's own sweep found empirically best and
the more defensible choice given this project's own rollout-window size —
this project's own number (`24`), not just the paper's own Montevideo/
Montezuma-scale finding, argues for `D` small.
(For completeness: this project's own episode length/discount would *not*
have produced Montezuma-scale blowup even with full end-of-episode PBIM —
`1/γ^(N-1) = 1/0.98^249 ≈ 153`, nowhere near Montezuma's `~10^19` — so the
magnitude-explosion risk specifically is not why `D=1` is chosen here; the
PPO-rollout-window argument is the decisive one.)

## Exact mechanism proposed

**New reward term 1 — `gripper_closure_attempt_bonus` (the raw, action-dependent `Ft`):**

```
F_t = w_attempt * tanh(k * relu(-raw_gripper_action_t)) * (1 - tanh(d_t / std_gate))
```

- `raw_gripper_action_t`: `env.action_manager.get_term("gripper_action").raw_actions`
  (confirmed exact API, reused verbatim from
  `scripts/_diag_gripper_lowpass_check.py`'s own already-working access
  pattern) — the policy's own raw output for the gripper action dimension,
  pre-threshold. Negative values command "close" (`BinaryJointAction`'s
  `binary_mask = actions < 0`, per the addendum's own confirmed source
  read). `relu(-raw_gripper_action_t)` is therefore positive exactly when
  the policy is attempting closure, zero otherwise.
- `d_t`: end-effector-to-object distance, reusing
  `tasks/franka/lift_reward.py`'s existing `1 - tanh(distance/std)` kernel
  convention (same shape as `reaching_object_reward`, `object_goal_distance_reward`)
  rather than inventing a new kernel family. `std_gate` should be tight
  (e.g. `0.05`, matching `object_goal_tracking_fine_grained`'s existing
  "fine-grained" std, not `reaching_object`'s loose `0.1`) — the bonus
  should fire only when genuinely near the object, not across the whole
  workspace; a loose gate would just be a generic entropy nudge, not a
  targeted fix for the specific "never attempts closure *near the object*"
  failure the diagnostic found.
- `tanh(k * ...)`: bounds the bonus's magnitude regardless of how large the
  policy's raw action gets (the diagnostic found raw actions up to `7.77` in
  magnitude — an unbounded linear term would let a policy chase reward via
  action-magnitude alone rather than via a genuine attempt). `k` and
  `w_attempt` are implementer-tunable constants, not load-bearing for this
  spec's own falsification bar (a Tier 2 hillclim candidate once/if this
  mechanism is validated) — an initial magnitude comparable to or smaller
  than `reaching_object`'s own weight (`1.0`) is a reasonable starting
  point, since this term's job is a small nudge to the *sampling*
  distribution, not a competing objective.
- This `Ft` depends only on the *current* state and action — it is, in the
  papers' own terms, the simplest possible case of "future-agnostic" (no
  history dependence at all in the raw term; the history-dependence is
  confined entirely to the correction term below, which is the part the
  papers formally prove correct).

**New reward term 2 — `gripper_closure_attempt_bonus_correction` (the GRM `D=1` matching term):**

```
F'_t = F_t                          if t = 0
     = F_t - (1/γ) * F_{t-1}        if 1 <= t < N-1
     = -(1/γ) * F_{N-2}             if t = N-1
```

(GRM's own delay-`D` family, ADOPS paper Eq. 8, instantiated at `D=1`.)
Requires **one new piece of persistent per-env state**: a scalar buffer
holding the previous step's raw `F_{t-1}` value, reset to `0` on episode
reset. This is a genuinely new mechanism for this codebase — every existing
reward term in `tasks/franka/mdp.py`/`lift_reward.py` is a pure function of
the *current* step's live sim state, with no persistent memory across steps
— but it is a small, bounded one (one float per env, one reset hook), not
the four-critic infrastructure full ADOPS would need. The correction is
**not** claimed to be non-negative, or to have any particular sign — unlike
Experiment 5's informal (and false) "always ≥ 0" claim, this design's
correctness rests entirely on Theorem 1 (boundary condition, Eq. 25) +
Assumption 1 (future-agnostic `Ft`) + GRM's own proved delay-`D` family
(Eqs. 46–49, ADOPS Eq. 8) holding by construction, not on any claim about
the shaped reward's sign at any individual step.

**Both terms are added to (not replacing) the existing `RewardsCfg`** — the
existing `reaching_object`/`lifting_object`/`object_goal_tracking`/
`object_goal_tracking_fine_grained`/`action_rate`/`joint_vel` terms are
untouched, matching the "structural experiment must not silently change
what's already validated" convention this project's specs follow
throughout.

## Controlled test bed: `FrankaDieLiftJointD8BigEnvCfg` (d8, 48mm-parity, single shape)

**Chosen over the harder d12/d20-mixed and clutter populations, per the
task's own explicit instruction to isolate the mechanism first:**

- **Single shape, not mixed or clutter.** `FrankaDieLiftJointD8BigEnvCfg`
  (`tasks/franka/dice_lift_joint_env_cfg.py:645`) has exactly one object,
  no distractor scene entities, and the plain 41-dim `ObservationsCfg` (not
  `TargetSelectionObservationsCfg`'s 43-dim schema) — none of Stage SO's own
  confounds (its own 43-dim schema being trained from scratch for the first
  time ever, its own not-yet-resolved warm-start question) leak into this
  experiment.
- **A robust, already-fully-characterized 0/8-type failure to fix.** Per
  `docs/superpowers/specs/2026-07-19-d8-d10-demo-warmstart-design.md`'s own
  "Context" section, d8 at this exact 48mm-parity anchor is
  **genuinely, robustly null: 0/24 envs (0/8 in all 3 seeds 42/123/7)**,
  same reward function, same PPO recipe, same observation schema as the
  shapes (d12/d20) that *do* succeed from scratch — only the object asset
  differs. This is the strongest, most cleanly-isolated failure baseline in
  this project's own history to test a discovery-unlocking mechanism
  against (matching this project's own "same-recipe-different-shape
  isolates one variable" reasoning, `research/...literature.md` §1).
- **d8 over d10, specifically.** The research doc's own d8/d10 comparison
  (echoed in the demo-warmstart spec) notes d10 carries two *additional*
  confounds beyond sphericity (bbox anisotropy, no parallel opposite-face
  pairs) that d8 does not — d8 is the cleaner single-variable-different
  shape for testing whether an *exploration* mechanism specifically (not a
  grasp-*geometry* mechanism) is the fix.
- **48mm parity, not real ~16mm size.** Matches this project's own
  already-established isolation anchor (Task 3.5, and the concurrently-running
  demo-warmstart spec) — at 48mm every shape presents an identical
  1.67x aperture-to-object ratio, removing a real absolute-scale confound
  that's present at real size. Reusing the same anchor as prior isolation
  work is a "why this is the cleanest bed" argument in its own right, not
  just convenient.
- **Grasp is independently known to be physically achievable for this
  shape/gripper**, not just RL-undiscovered: `scripts/dice_pick_demo.py`'s
  scripted DiffIK controller already achieves a verified grasp+lift for d8
  (`kb/wiki/experiments/dice-pick-demo.md`) at real size. **Caveat, stated
  explicitly, not assumed away:** that verification was at real size, not
  48mm-parity — the demo-warmstart spec's own "Task 0" (re-verify the
  scripted grasp at 48mm scale) is the relevant prerequisite check here too.
  This spec does not require re-running that check itself (if the
  concurrently-running demo-warmstart spec's own Task 0 has already run,
  its result is directly reusable evidence for this spec as well, since it
  is a property of the shape/scale/gripper, not of the warm-start
  mechanism); if it has not yet run when this spec's own implementation
  starts, running it once is a shared, one-time prerequisite, not
  duplicated work.

## H1 (primary and only hypothesis this spec authorizes)

### Falsifiable hypothesis

> Adding the `gripper_closure_attempt_bonus` + `gripper_closure_attempt_bonus_correction`
> reward terms (GRM, `D=1`) to the existing, otherwise-unmodified
> `FrankaDieLiftJointD8BigEnvCfg` reward function will cause a from-scratch
> PPO run (identical PPO recipe, identical 1500-iteration budget, identical
> seeds 42/123/7, no warm start) to produce a policy whose raw gripper
> action goes negative (attempts closure) near the object at some point
> during training or in the final checkpoint's own rollout — where the
> unmodified reward function, per the just-completed diagnostic on a
> structurally analogous checkpoint, confidently never does (0.000 fraction
> of steps, 100% "open," across 8 envs and a full episode, at iteration
> 1499) — and, further, that this newly-discovered attempt behavior
> produces genuine sustained-lift grasp discovery in at least one of the 3
> seeds, where the unmodified recipe on this exact env cfg is separately
> and independently established at 0/24 (Task 3.5, re-audited in the
> demo-warmstart spec's own "Context").

### Falsification bar

Two tiers, reported and evaluated **both**, per shape (d8 only, this spec's
scope):

1. **Mechanism-level bar (necessary, not sufficient):** for each of the 3
   seeds' final (iteration-1499) checkpoints, run the same instrumented
   rollout methodology as `scripts/_diag_gripper_lowpass_check.py`
   (deterministic inference policy, 8 envs, one full 250-step episode),
   restricted to steps where end-effector-to-object distance `< 5cm`
   ("near the object," matching this design's own `std_gate`), and compute
   `frac_steps_raw_action_negative_near_object`. **H1's mechanism-level
   claim is falsified if this fraction is exactly `0.000` in all 3
   seeds** — an outcome indistinguishable from the unmodified baseline's
   already-confirmed absolute-zero result.
2. **Behavioral bar (this project's standard success metric):** via
   `franka_checkpoint_review.py`, same protocol/thresholds used throughout
   this project's die-lift arc (`num_envs=8`, sustained lift-height gain
   over the established threshold). **H1 is falsified for d8 if all 3
   seeds show 0/8 sustained-lift discovery (0/24 total)** — identical to
   the already-established from-scratch null this spec is trying to beat.
   Per this project's own repeatedly-confirmed precedent (every prior
   discovering seed in this project's history reached full 8/8-within-seed
   completeness, never a partial count — d8/d10 spec's own "Context"), any
   nonzero count in even one seed (`>=1/8`) is treated as a real positive
   signal, not noise.

**Overall verdict rule:** H1 is **falsified** only if *both* bars fail
across all 3 seeds (mechanism-level `0.000` everywhere **and** 0/24
behavioral). **A split result — mechanism-level bar passes (nonzero
attempt rate appears) but the behavioral bar still fails (0/24, no
completed lift)** is explicitly *not* treated as falsification of H1's
core exploration-discovery claim; it is a genuinely novel, reportable
finding in its own right (the exploration mechanism worked — attempts are
now being sampled — but some other downstream grasp-mechanics problem
remains, distinct from and pointing away from the pure discoverability
question this spec targets). This mirrors this project's own established
practice of flagging genuinely novel partial outcomes rather than silently
rounding them to a binary pass/fail (see the demo-warmstart spec's
identical treatment of a hypothetical partial per-seed count).

### Iteration budget and seeds

1500 iterations (matching `FrankaLiftPPORunnerCfg.max_iterations` and this
project's own CLAUDE.md Tier 1 mandate: full run + video review before any
verdict, no early stopping on a promising-looking curve), seeds 42/123/7
(matching every other experiment on this same shape/anchor in this
project's history, for direct comparability against the existing 0/24
baseline).

## Global constraints — what is deliberately NOT combined into this first test

- **No clutter/distractor mechanism.** No `distractor_1`/`distractor_2`
  scene entities, no `distractor_distance_summary` observation term, no
  `TargetSelectionObservationsCfg`/`FrankaDieLiftTargetSelectionSceneCfg`.
  This spec's env cfg has exactly one object.
- **No d12/d20-mixed population, no target-selection curriculum stages.**
  Single shape (d8) only, in this spec's own scope.
- **No demonstration-warm-start (the d8/d10 demo-warmstart spec's own H1/H2)
  combined in.** These are two independent, complementary candidate fixes
  for the same underlying null result, deliberately run as *separate*
  mechanism tests so a positive or negative result for either is
  attributable to that mechanism alone, not a combination. If this spec's
  own implementation plan (not authorized by this spec) is written and
  executed, it must add the two new reward terms via a **new env-cfg
  subclass** (e.g. `FrankaDieLiftJointD8BigExplorationBonusEnvCfg(FrankaDieLiftJointD8BigEnvCfg)`),
  mirroring `TargetSelectionObservationsCfg`'s own established precedent of
  extending via a new subclass rather than editing the shared base
  `RewardsCfg`/`FrankaDieLiftJointD8BigEnvCfg` in place — this keeps the
  concurrently-running demo-warmstart spec's own use of the plain,
  unmodified `FrankaDieLiftJointD8BigEnvCfg` completely unaffected.
- **No H2 (RND-based intrinsic bonus) or H3 (scheduled entropy/noise) from
  the research doc.** Not pre-authorized here; per this project's "one
  fallback rung, no new spec" convention (established in the
  demo-warmstart spec, itself following `2026-07-11-joint-space-die-lift-design.md`'s
  precedent), if H1 falsifies on *both* bars above, that is a stop-and-report
  point back to Principal — not an automatic trigger for H2/H3 without a
  new spec.
- **No full ADOPS (dual-critic, real-time-corrected) mechanism.** Flagged
  explicitly as a documented escalation path, not this spec's primary
  proposal (see "A refinement on the research doc" above) — pre-authorized
  *only* as a fallback if GRM `D=1` is judged, after a real run, to have
  failed for a reason specific to its own construction (e.g. the `D=1`
  correction's interaction with PPO's finite rollout batches turns out to
  meaningfully undermine the intended cancellation in practice, a risk
  named explicitly above but not yet empirically checked) — not
  automatically triggered by a plain 0/24 falsification, which per the
  "one fallback rung" convention would instead be a stop-and-report point.
- **No tuning of `w_attempt`, `k`, or `std_gate` as part of this
  experiment's own falsification.** Initial values are implementer-set
  (per "Exact mechanism proposed" above); if H1's mechanism-level bar shows
  a genuine but small effect, tuning these three scalars is exactly the
  kind of bounded, single-parameter search this project's Tier 2 hillclimb
  loop (`scripts/hillclimb_rewards.py`) exists for — not a reason to write
  a new Tier 1 spec.

## Reused vs. new infrastructure

**Reused, unchanged:**
- `tasks/franka/lift_reward.py`'s `1 - tanh(distance/std)` kernel
  convention (for the proximity gate).
- `env.action_manager.get_term("gripper_action").raw_actions` access
  pattern (already proven working in `scripts/_diag_gripper_lowpass_check.py`).
- `franka_checkpoint_review.py`'s existing eval protocol/thresholds (behavioral bar).
- `scripts/_diag_gripper_lowpass_check.py`'s own rollout/instrumentation
  methodology (mechanism-level bar), generalized to compute a
  near-object-restricted fraction instead of an unrestricted one.
- `FrankaDieLiftJointD8BigEnvCfg`'s existing scene/observation/action/PPO
  config, entirely unmodified.

**Genuinely new, flagged explicitly:**
- Two new reward-manager terms (`gripper_closure_attempt_bonus`,
  `gripper_closure_attempt_bonus_correction`) in `tasks/franka/mdp.py`/`lift_reward.py`.
- **One new stateful mechanism**: a persistent per-env scalar buffer (the
  previous step's raw `F_{t-1}`), reset on episode reset — no existing
  reward term in this codebase carries state across steps; the
  implementing task will need to determine the correct integration point
  (an event-manager reset hook is the natural fit, mirroring how
  `EventCfg`'s existing reset terms already run on episode reset, but the
  exact mechanics are implementation-plan-level detail, not resolved by
  this design spec).
- A new env-cfg subclass (`FrankaDieLiftJointD8BigExplorationBonusEnvCfg`)
  to keep this addition isolated from the shared base class per "Global
  constraints" above.
- A near-object-restricted variant of the diagnostic script's fraction
  metric (mechanism-level bar).

## Success/failure reporting

Full 1500-iteration training run per seed (no early verdicts), video-review
of any positive result (a rest frame vs. a peak-height frame showing a
visibly different, genuinely gripped arm pose — not just a height number
crossing a threshold), raw per-step `.npy`/instrumented-rollout inspection
for both bars (not summary-JSON-only trust, per this project's own
repeated settle-detection-bug discipline). Report the mechanism-level and
behavioral bars **both**, explicitly, per seed — including an explicit
call-out if the split "mechanism fired, grasp didn't complete" outcome
occurs, since (per "Falsification bar" above) that is itself a new,
worth-recording empirical fact about this project's own discovery dynamics
regardless of the experiment's overall verdict.

## Related

[[experiment-05-potential-based-reward-shaping]] (the prior-failure this
spec's mechanism is built to formally avoid repeating),
[[reward-hacking-and-sparse-discoverability]] (the general
loose-vs-tight tradeoff this mechanism is designed to formally escape, not
just empirically balance), [[reach-grasp-lift-gap]],
[[target-selection-clutter]] (source of the diagnostic evidence this spec's
hypothesis is grounded in),
`docs/superpowers/specs/research/2026-07-19-exploration-reward-expansion-literature.md`
(the Tier 1 research-gate document this spec executes on, including the
refinement in "A refinement on the research doc" above),
`docs/superpowers/specs/2026-07-19-d8-d10-demo-warmstart-design.md` (the
concurrently-running, complementary demo-warmstart spec on the same env
cfg/anchor, deliberately not combined with this spec per "Global
constraints").
