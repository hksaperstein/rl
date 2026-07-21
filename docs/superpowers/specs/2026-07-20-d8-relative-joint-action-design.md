# d8 relative/delta joint-position action — design spec (targeting the diagnosed joint-space collapse mechanism directly)

## Context

`kb/wiki/experiments/d8-antipodal-grasp-quality.md`'s "Root cause
investigation (2026-07-21 follow-up)" section (hereafter "the root-cause
doc") is the Tier 1 hypothesis-gate research this spec executes on. It
closed the question left open by that same article's own H_joint/H_taskspace
result (H_joint FALSIFIED — 0/24 sustained lift, exact `0.0` mechanism signal
in all 3 seeds; H_taskspace CONFIRMED — 8/24, seed 123 a clean sweep) by
root-causing **why** joint-space collapses, using a from-scratch instrumented
diagnostic (`scripts/diag_antipodal_root_cause.py`) reading real per-step
contact-force vectors, not just TensorBoard scalars. This spec is design
only — no implementation plan, no code changes, no Isaac Sim launches.

**This is a Tier 1 structural experiment** (a new action-term configuration
— a change to what the policy can act on/how its actions map to motion) per
`CLAUDE.md`'s Workflow section.

## Research grounding (restated from the root-cause doc; not re-derived)

- **Finding 1 (the discriminating variable):** joint-space's contact
  frequency (fraction of (step, env) samples where both jaws register force
  `>0.05N`) is **exact, literal `0.0` at all 8 checkpoints measured**
  spanning iterations 0→1499 across 4 different seeds/runs. Task-space's own
  contact frequency **rises monotonically from 0 to 88%** over the identical
  1500 iterations, and whenever contact occurs at all it is already
  93.3-99.996% antipodal-satisfying from the first checkpoint onward. The
  learning problem under joint-space is not "achieve contact, then fix its
  geometry" (task-space's own problem, solved almost for free) — it is
  **achieving contact at all**, which joint-space never does, even once, in
  ~256k sampled (step, env) pairs.
- **Finding 2 (reward structure ruled out numerically):** the
  `action_rate`/`joint_vel` penalty terms' weighted per-step contributions
  are 2-4 orders of magnitude smaller than the terms that matter
  (`reaching_object`, `antipodal_grasp_quality`) in both conditions, and the
  reward structure is byte-identical between conditions yet produces
  dramatically different outcomes (0% vs. 88% contact) — the reward
  structure cannot be the differentiator.
- **Finding 3 (the real mechanism — transient discovery, then abandonment):**
  joint-space's own `reaching_object` reward **transiently rises to a peak
  around iteration 100** (0.6015, close to task-space's own 0.7899 at that
  same checkpoint) and then **regresses to 0.0957 by iteration 1499** —
  actively abandoning a real, already-discovered approach capability, not
  merely failing to improve on it. Task-space's `reaching_object` instead
  rises and *stays* high (0.8394 final). The original noisier-motion
  hypothesis (H3 as first framed) was **falsified directly**: task-space's
  own raw per-step EE jitter is equal-or-larger than joint-space's at every
  single checkpoint including iteration 0 (pure random policy, before any
  learning) — the opposite of what H3 originally predicted.
- **The refined, evidence-supported mechanism**, per the root-cause doc's
  own synthesis: joint-space's absolute-target action (`JointPositionAction`,
  `applied_target = raw_action * scale + default_joint_pos`, verified by
  direct source read below) makes the actual physical motion produced by a
  *given* raw action depend on the arm's current configuration (how far the
  current position already is from that fixed target) — a mapping that
  reshapes itself as the arm moves. Task-space's differential-IK action is
  instead mediated by a fixed, non-learned controller that keeps the
  action-to-EE-motion relationship consistent regardless of configuration.
  As PPO's action-distribution entropy narrows over training — a documented
  dynamic (Hsu, Mendler-Dünner, Hardt, arXiv:2009.10897, 2020; existence/
  citation-accuracy already checked per this project's standing practice) —
  joint-space's early, marginal, exploration-noise-driven approach successes
  are not consistently reinforced or generalized under this
  configuration-dependent mapping, and the policy collapses to a
  lower-variance "hover-near-but-never-touch" local optimum. Task-space's
  configuration-independent mapping keeps final-approach precision
  reinforceable even as entropy shrinks. Cross-validated against two direct,
  on-point comparisons of joint vs. task-space/impedance action spaces on
  contact-rich manipulation (Martín-Martín et al., IROS 2019, arXiv:1906.08880;
  Varin, Grossman, Kuindersma, IROS 2019, arXiv:1908.08659), both already
  cited and existence/accuracy-checked in the root-cause doc.
- **Candidate fixes surveyed but not implemented, in the same root-cause
  doc:** a checkpoint warm-start (blocked by an action-dimensionality
  mismatch vs. task-space's 7-dim action), reward-shaping (ruled unmotivated
  by Finding 2), and — the candidate this spec pursues — an action-term
  parameterization that stays joint-space but changes the action's semantics
  from absolute-target to relative/incremental, isolating "delta vs. absolute
  action semantics" from "joint-space vs. task-space" as two axes the
  original H_joint-vs-H_taskspace design changed simultaneously.

## Precise falsifiable hypothesis

**H_relative:** Replacing `FrankaDieLiftJointD8BigAntipodalEnvCfg`'s
inherited `JointPositionActionCfg` (absolute joint-space) with
`RelativeJointPositionActionCfg` (delta/incremental joint-space) — with
`AntipodalGraspRewardsCfg`/`FrankaDieLiftContactSceneCfg` and every other
field otherwise byte-identical — makes the action-to-motion mapping locally
consistent (each raw action always produces approximately the same joint
delta, regardless of current configuration) rather than globally
configuration-dependent (Finding 3's diagnosed mechanism). This predicts
that the early, transiently-discovered `reaching_object`/contact-approach
behavior (peaking near iteration 100 under absolute joint-space, then
abandoned) will instead **persist and grow through the remainder of
training**, measured the same way the root-cause doc measured it — **contact
frequency across multiple checkpoints spanning the full training run, not a
final-iteration snapshot alone** — because a persisting, non-collapsing
signal (not merely a nonzero final number, which a transient late-training
bounce could also produce) is what would actually distinguish "the diagnosed
mechanism was fixed" from "the same collapse, just reshaped or delayed."

This hypothesis is stated as genuinely falsifiable in the direction the
project cares about: it does **not** claim relative joint-space matches or
exceeds task-space's own 88% asymptote — only that it breaks the
zero-contact-forever pattern absolute joint-space showed in every one of 8
independently-measured checkpoints. A weak but real, non-collapsing signal
below task-space's own ceiling is a positive result for this hypothesis, not
a null.

## Exact mechanism — confirmed by direct source read, Isaac Lab v2.3.1 (this project's pinned tag)

Read directly from `isaaclab/envs/mdp/actions/actions_cfg.py` and
`isaaclab/envs/mdp/actions/joint_actions.py` at the `v2.3.1` tag (matching
`docs/cloud/franka-cloud-shakedown.md`'s pinned version, the same tag the
root-cause doc's own action-space survey used) — not assumed from the class
name alone:

```python
@configclass
class RelativeJointPositionActionCfg(JointActionCfg):
    class_type: type[ActionTerm] = joint_actions.RelativeJointPositionAction
    use_zero_offset: bool = True
    """Whether to ignore the offset defined in articulation asset. Defaults to True.
    If True, this flag results in overwriting the values of offset to zero."""
```

```python
class RelativeJointPositionAction(JointAction):
    r"""...the processed actions are added to the current joint positions
    of the articulation's joints before being sent as position commands.
    applied action = current joint positions + processed actions
    """
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if cfg.use_zero_offset:
            self._offset = 0.0

    def apply_actions(self):
        current_actions = self.processed_actions + self._asset.data.joint_pos[:, self._joint_ids]
        self._asset.set_joint_position_target(current_actions, joint_ids=self._joint_ids)
```

where `processed_actions = raw_action * scale + offset` (the shared
`JointAction.process_actions`, identical base class to `JointPositionAction`).

**This is the mechanism-level contrast that directly targets Finding 3's
diagnosis:**

- **Absolute (`JointPositionActionCfg`, Condition A's current action):**
  `applied_target = raw_action * scale + default_joint_pos` — a *fixed*
  offset (the robot's own rest pose), computed once at action-term
  construction. The actual joint *motion* a given `raw_action` produces on
  any control step is `applied_target - current_joint_pos`, which depends
  entirely on how far `current_joint_pos` already is from that fixed target
  — i.e., on the arm's current configuration. Two identical raw actions
  issued from two different arm poses move the arm by two different amounts.
- **Relative (`RelativeJointPositionActionCfg`):** `applied_target =
  raw_action * scale + current_joint_pos`, with `current_joint_pos` read
  fresh via `self._asset.data.joint_pos` at the moment `apply_actions` runs
  each control step (this project's env cfg calls this once per 50Hz control
  step, `decimation=2`/`sim.dt=0.01`, confirmed in `tasks/franka/
  lift_env_cfg.py:483-486`). A given `raw_action` therefore always produces
  approximately the same joint delta, **independent of the arm's current
  configuration** — this is the "locally consistent, not globally
  configuration-dependent" property Finding 3 identifies as what task-space's
  own fixed differential-IK controller provides and absolute joint-space
  does not.

**Composability with this project's existing PPO architecture — verified,
not assumed:** `RelativeJointPositionActionCfg` inherits `JointActionCfg`
exactly like `JointPositionActionCfg`, resolves the same `joint_names=
["panda_joint.*"]` (the 7 arm joints), and produces the same `action_dim`
(7). Combined with the unchanged `BinaryJointPositionActionCfg` gripper
action (1 dim), the total action space is **8 dims — byte-identical to
`FrankaDieLiftJointD8BigAntipodalEnvCfg`'s own current absolute joint-space
action space**. No change to `FrankaLiftPPORunnerCfg`
(`max_iterations=1500`, `gamma=0.98`, `save_interval=50`), no change to the
41-dim observation schema (joint_pos/joint_vel observations are already
present regardless of which action term drives them), no actor/critic
network architecture change. This is a strictly easier composition question
than task-space's own 6-Cartesian-dim action (7 total with gripper) — which
is exactly why the root-cause doc's own warm-start candidate was blocked by
a dimensionality mismatch there but would not be blocked here. (Whether to
actually attempt a warm-start from Condition A's own converged joint-space
checkpoint is a real, interesting option this dimensional match opens up —
flagged here as a possible implementation-plan-level convenience, not
decided or required by this spec; H_relative's own falsification does not
depend on it.)

## Scale — grounded in real Isaac Lab precedent, not an arbitrary guess

Per this project's Tier 1 discipline (a load-bearing parameter needs real
grounding, not "seems reasonable"), Isaac Lab's own shipped task configs
using `RelativeJointPositionActionCfg` were located by a direct grep of the
full `v2.3.1` source tree (not assumed to not exist):

- `isaaclab_tasks/manager_based/manipulation/dexsuite/config/kuka_allegro/
  dexsuite_kuka_allegro_env_cfg.py:20`: `scale=0.1`, all joints
  (`joint_names=[".*"]`) — a contact-rich dexterous *manipulation* task (an
  arm + Allegro hand reorienting an object), the closest precedent in spirit
  to this project's own contact-rich lift task. Its own env cfg
  (`dexsuite_env_cfg.py:409,425`) runs `decimation=2`, `sim.dt=1/120` — a
  60Hz control rate, close to this project's own 50Hz
  (`decimation=2`, `sim.dt=0.01`).
- `isaaclab_tasks/manager_based/manipulation/deploy/reach/config/ur_10e/
  joint_pos_env_cfg.py:66-68`: `scale=0.0625`, all joints — a pure
  reach-to-pose task (no contact/grasp component), a less directly analogous
  precedent but a second independent real data point in the same
  order-of-magnitude range.

**Proposed starting scale: 0.1** (the Kuka Allegro precedent), on the
strength of (a) being the closer analog — contact-rich manipulation, not
pure reaching — and (b) running at a control frequency close enough to this
project's own 50Hz that no rescaling is needed to make the comparison
meaningful. This follows the same "implementer-set starting constant,
grounded in real precedent, not blindly guessed, Tier 2 hillclimb candidate
later if the mechanism validates but needs retuning" treatment the
antipodal spec already gave `force_threshold`/reward weight — **not**
load-bearing for H_relative's own falsification bar below. If 0.1 produces
visibly excessive per-step jitter or an unstable/oscillating gripper
approach on a short smoke-test rollout (an implementation-plan-level check,
not a design decision), 0.0625 (the UR10e precedent) is the documented
fallback value, not a fresh guess.

## `OperationalSpaceControllerActionCfg` — surveyed, deliberately deferred, not a second condition

Also confirmed present in `v2.3.1`'s `actions_cfg.py` (`class_type:
task_space_actions.OperationalSpaceControllerAction`) and read alongside its
one real Isaac Lab reference usage,
`isaaclab_tasks/manager_based/manipulation/reach/config/franka/
osc_env_cfg.py`. Judgment: **do not include as a second condition in this
spec**, for two independent reasons, either alone sufficient:

1. **It is not genuinely joint-space** — the same reason the user explicitly
   flagged for this experiment. `OperationalSpaceControllerAction` operates
   via `body_name`/`controller_cfg` on Cartesian pose/wrench targets
   (`target_types=["pose_abs"]` in Isaac Lab's own reference config) mediated
   by an operational-space controller — mechanistically a task-space/
   Cartesian action family, like `DifferentialInverseKinematicsActionCfg`
   (already tested as H_taskspace), not a variant of joint-space control.
   Including it here would reintroduce exactly the "is task-space secretly
   doing all the work again" ambiguity this spec is designed to avoid by
   staying strictly within the joint-space family.
2. **It requires cross-cutting changes unrelated to the diagnosed
   mechanism.** Isaac Lab's own reference OSC config disables gravity on the
   robot spawn, zeroes stiffness/damping on two actuator groups (switching
   those joints to effort/torque control), and uses `FRANKA_PANDA_CFG`
   rather than this project's `FRANKA_PANDA_HIGH_PD_CFG`-derived robot cfg
   (`_FRANKA_ROBOT_CFG_WITH_CONTACT` in `dice_lift_joint_env_cfg.py`, needed
   for the antipodal reward's own contact sensors) — a real actuator/physics
   configuration change, not a drop-in action-term swap, and one that adds a
   large new controller-tuning surface (impedance stiffness/damping ratio,
   nullspace control mode, wrench scale) that has nothing to do with
   Finding 3's diagnosed absolute-vs-relative target mechanism. Testing it
   would answer a different, broader question ("does a wholesale
   controller-family change help") than this spec's specific, narrower one.

If `RelativeJointPositionActionCfg` does not resolve the collapse, OSC
remains a legitimate future candidate — but as its own separately-motivated
experiment (closer in kind to H_taskspace than to this one), not folded into
this spec.

## Scope: d8, `FrankaDieLiftJointD8BigAntipodalEnvCfg`'s existing reward/scene wiring, action term only

The only change from `FrankaDieLiftJointD8BigAntipodalEnvCfg` (Condition A,
the already-closed, falsified H_joint run) is the arm action term. Everything
else — the d8 48mm-parity object, `AntipodalGraspRewardsCfg`,
`FrankaDieLiftContactSceneCfg`'s two `ContactSensorCfg`s, the 41-dim
observation schema, events/terminations/episode length, `FrankaLiftPPORunnerCfg`
— is reused byte-identical. This is deliberate, for reasons directly
continuing this arc's own established scope discipline:

- **This is the cleanest possible isolation of the one variable this spec is
  actually about.** The root-cause doc's own Finding 2 already ruled out the
  reward structure as a differentiator; re-testing with any reward or scene
  change at the same time would reopen a question this arc's own prior work
  already closed, for no isolation benefit.
- **d8 at 48mm-parity is this project's most robustly characterized
  joint-space null** — independently reproduced across 3 original H_joint
  seeds plus a fresh retrain (4 independent runs, 8 checkpoints, all exact
  `0.0` contact frequency) — the strongest baseline to test a targeted fix
  against, for the same reason the antipodal spec gave for choosing d8 over
  d10/d12/d20.
- **Combining this with d10/d12/d20 or any other open thread (exploration
  bonus, demo warm-start, size curriculum) now would multiply cost before
  this specific, narrow question — does a delta-based action term fix the
  diagnosed mechanism — has an answer.** Per `CLAUDE.md`'s own "one thing at
  a time, in sequence" scope discipline: if this succeeds, extending to
  other shapes/mechanisms is the natural next spec, not a decision to make
  now.

**New env cfg (implementation-plan-level naming, not decided here):** a new
leaf subclassing `FrankaDieLiftJointD8BigAntipodalEnvCfg`, whose own
`__post_init__` calls `super().__post_init__()` (so the full inherited chain
— die swap, mass/scale, `AntipodalGraspRewardsCfg`, `FrankaDieLiftContactSceneCfg`
— runs first) and then re-asserts `self.actions.arm_action` to
`RelativeJointPositionActionCfg(asset_name="robot",
joint_names=["panda_joint.*"], scale=0.1, use_zero_offset=True)` — the exact
same "call super, then overwrite the one changed field last" pattern
`FrankaDieLiftD8BigTaskspaceAntipodalEnvCfg` already established for
Condition B.

## Measurement plan — per the root-cause doc's own established method, not a final-iteration snapshot

The entire point of this hypothesis is to distinguish "the mechanism is
fixed" from "the same collapse, just reshaped or delayed" — a single
final-iteration number cannot do this (Finding 3 showed absolute joint-space
itself produces a *nonzero*, even substantial, `reaching_object` value at an
intermediate checkpoint before collapsing). This spec reuses the root-cause
doc's own two-tier measurement design exactly:

1. **Full-trajectory mechanism diagnostic** (`scripts/diag_antipodal_root_cause.py`'s
   existing per-step instrumentation, extended with a new `--variant` choice
   for this condition — exact wiring is implementation-plan-level, the
   measured quantities are decided here): contact frequency (fraction of
   (step, env) samples with both jaws' `ContactSensorData.force_matrix_w`
   magnitude `>0.05N`, identical definition to Finding 1), antipodal-satisfying
   frequency, and the `reaching_object` reward's own raw trajectory — each
   measured at the **same 5-checkpoint cadence the root-cause doc already
   used**: iterations **{0, 100, 300, 700, 1499}**. Checkpoints must be
   preserved and GCS-synced throughout training (`save_interval=50`, synced
   incrementally, not only at the end) — the root-cause doc's own dispatch
   hit a real operational gap here (a GRUB-corruption incident lost
   un-synced intermediate checkpoints), and this spec's measurement plan
   depends on having all 5 checkpoints survive.
2. **Final-checkpoint mechanism/behavioral bars**, identical structure and
   thresholds to the closed antipodal spec's own H_joint/H_taskspace bars,
   evaluated at iteration 1499 across all 3 seeds (42/123/7): the
   `antipodal_grasp_quality` TensorBoard mean over the final 100 iterations,
   and `franka_checkpoint_review.py`'s existing sustained-lift protocol
   (0.04m threshold, `977a748`'s settle-window fix).

Running the full 5-checkpoint trajectory diagnostic on all 3 seeds (not just
one, unlike the root-cause doc's own single-seed-per-condition trajectory)
is this spec's own upgrade to the method — the root-cause doc used one seed
per condition because it was root-causing an already-closed aggregate
result; this spec's own falsification bar (below) is defined per-seed and
needs the trajectory shape for each seed individually to apply correctly.

## Falsification bar — numeric, explicit

**H_relative is FALSIFIED** ("just delayed the same collapse", not a genuine
fix) if, in **at least 2 of 3 seeds**:
- contact frequency at iteration 1499 is **< 0.01** (an order of magnitude
  below any meaningful sustained-contact signal — the same floor Finding 1's
  own data makes an unambiguous "no real contact" reading at, since
  task-space's own weakest positive seed, seed 42, still reached 0.00047 by
  iteration 100 and kept climbing), **AND**
- that same seed's contact frequency at iteration 1499 is **less than 50% of
  its own peak value** across the 5 measured checkpoints {0, 100, 300, 700,
  1499} — the "rose, then substantially decayed" shape that is this
  hypothesis's own specific signature of "delayed, not fixed," distinguishing
  it from a policy that simply never discovered contact in the first place
  (peak ≈ final ≈ 0, which also satisfies falsification under the first
  bullet alone).

**H_relative is CONFIRMED** if, in **at least 2 of 3 seeds**:
- contact frequency at iteration 1499 is **≥ 0.05** (5% of samples — an
  order of magnitude above absolute joint-space's exact `0.0` at every one of
  its own 8 measured checkpoints, and inside the real, non-noise range
  task-space's own seed 42 first crossed by iteration 300-700 en route to a
  much higher final value), **AND**
- contact frequency at iteration 1499 is **not less than** its own value at
  iteration 700 (ruling out a late-training collapse the 5-checkpoint
  cadence would otherwise miss if only checked at 1499 against 0/100/300).

**Anything else — including a signal that grows but plateaus below 0.05, or
a genuine per-seed split between the FALSIFIED and CONFIRMED shapes above —
is reported as a SPLIT**, per this project's own standing precedent
(`exploration-bonus-grasp-discovery`, H_taskspace's own seed-level
heterogeneity) of not forcing an ambiguous result into a binary verdict.

**Behavioral confirmation, reported alongside but not substituting for the
above:** `franka_checkpoint_review.py`'s existing 0.04m sustained-lift
protocol, all 3 seeds. A real positive lift on even one seed is independently
reportable regardless of where that seed's mechanism-level shape lands
above (mirroring H_taskspace's own seed 123, which combined a strong
mechanism signal with a clean 8/8 behavioral sweep) — video-reviewed per this
project's standing "a shaped metric can misrepresent what's physically
happening" discipline (Experiment 16 precedent) before being reported as a
genuine grasp.

## Iteration budget and seeds

1500 iterations (matching `FrankaLiftPPORunnerCfg.max_iterations` and this
project's Tier 1 "full run + video review before any verdict" mandate),
seeds 42/123/7 (matching every other experiment on this exact env-cfg
family/anchor for direct comparability) — **1 condition × 3 seeds = 3 full
runs**, plus the 5-checkpoint trajectory diagnostic (`diag_antipodal_root_cause.py`,
extended) applied to all 3 seeds' own checkpoints (no additional training
runs — the same checkpoints already being saved for the bars above).

## Global constraints — what is deliberately NOT combined into this test

- **No reward or scene changes.** `AntipodalGraspRewardsCfg`/
  `FrankaDieLiftContactSceneCfg` stay exactly as `FrankaDieLiftJointD8BigAntipodalEnvCfg`
  already has them — the action term is this experiment's only variable.
- **No task-space/IK anywhere in this condition.** This spec exists
  specifically to test a genuinely joint-space fix, per direct user
  instruction not to treat task-space as "the answer" by default.
  `OperationalSpaceControllerActionCfg` is deferred for the reasons stated
  above, not included as a second condition.
- **No d10/d12/d20, no mixed population, no distractors/clutter, no
  demonstration warm-start, no exploration-bonus reward terms** — same
  "isolate one variable per experiment" reasoning as every prior spec in
  this arc.
- **No PPO-runner-cfg change pre-authorized.** `FrankaLiftPPORunnerCfg`
  reused unmodified. If training shows a critic-divergence signature
  (`Loss/value_function` exploding, `[[ppo-critic-divergence]]`'s own
  Experiment 11 signature) — a lower-risk scenario here than for Condition B
  since this stays within the same joint-space PD-servo actuation family
  Condition A already trains stably under, but not zero-risk given the
  action-magnitude-per-step semantics genuinely change — applying a scoped
  fix is implementation-plan-level judgment, not something this spec
  pre-resolves.
- **`scale=0.1` is an implementer-set, precedent-grounded starting value,
  not load-bearing for H_relative's own falsification bar** — a Tier 2
  hillclimb candidate later if the mechanism validates but needs retuning,
  same treatment as `force_threshold`/reward weight in the antipodal spec.

## Reused vs. new infrastructure

**Reused, unchanged:**
- `FrankaDieLiftJointD8BigAntipodalEnvCfg`'s full inherited chain (d8
  48mm-parity object, `AntipodalGraspRewardsCfg`, `FrankaDieLiftContactSceneCfg`,
  41-dim observations, events/terminations/episode length).
- `FrankaLiftPPORunnerCfg` (`max_iterations=1500`, `gamma=0.98`,
  `save_interval=50`).
- `scripts/diag_antipodal_root_cause.py`'s existing per-step instrumentation
  (contact-force reads, antipodal sub-condition computation, `reaching_object`
  raw-value read) and `scripts/franka_checkpoint_review.py`'s existing
  sustained-lift eval protocol.
- Isaac Lab's own `RelativeJointPositionActionCfg`/`RelativeJointPositionAction`
  — zero new action-space code, an already-shipped, already-precedented
  (Kuka Allegro dexsuite, UR10e reach) Isaac Lab class.

**Genuinely new:**
- One new leaf env cfg (subclassing `FrankaDieLiftJointD8BigAntipodalEnvCfg`,
  overriding only `self.actions.arm_action`) plus its `_PLAY` variant, exact
  naming left to the implementation plan.
- A new `--variant` choice for `scripts/train_franka.py`/
  `scripts/franka_checkpoint_review.py`/`scripts/diag_antipodal_root_cause.py`
  (exact names/wiring left to the implementation plan).

## Success/failure reporting

Full 1500-iteration training run per seed — no early verdicts, per
`CLAUDE.md`'s Tier 1 mandate. Report the 5-checkpoint contact-frequency
trajectory per seed explicitly (not collapsed into a single final number),
the mechanism/behavioral bars per seed, and the resulting classification
(FALSIFIED / CONFIRMED / SPLIT) per the rules above — plus, regardless of
that classification, an explicit comparison against the root-cause doc's own
already-measured absolute-joint-space trajectory (Finding 3's `reaching_object`
values / Finding 1's exact-zero contact frequency at the same 5 checkpoints)
so the "did this actually change the shape of the curve, not just delay it"
question is answered from real matched data, not inference.

## Related

`kb/wiki/experiments/d8-antipodal-grasp-quality.md` (the root-cause doc this
spec executes on — Findings 1-3, the candidate-fixes survey, and the closed
H_joint/H_taskspace result this spec's own condition is a targeted follow-up
to), `docs/superpowers/specs/2026-07-20-d8-antipodal-grasp-quality-design.md`
(this spec's own template/precedent for structure, falsification-bar style,
and the `AntipodalGraspRewardsCfg`/`FrankaDieLiftContactSceneCfg` machinery
reused unchanged here), [[action-space-design]] (the joint-space-vs-task-space
axis this spec narrows further, into delta-vs-absolute *within* joint-space),
[[ppo-critic-divergence]] (the risk this spec explicitly does not
pre-authorize a fix for), [[reach-grasp-lift-gap]], `CLAUDE.md`'s North Star
(a genuinely joint-space fix, if confirmed, is independently interesting
there: unlike task-space/IK, it requires no arm-specific IK-controller
configuration at all — a delta joint-space action is close to the most
morphology-agnostic action space this project has tested).
