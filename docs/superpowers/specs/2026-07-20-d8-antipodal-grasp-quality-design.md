# d8 antipodal/force-closure grasp-quality reward — design spec (dual action-space test)

## Context

`docs/superpowers/specs/research/2026-07-20-d8-grasp-mechanics-literature.md`
(hereafter "the research doc") is the Tier 1 hypothesis-gate research this
spec executes on, following up
[[exploration-bonus-grasp-discovery]]'s SPLIT verdict: the from-scratch PPO
policy on `FrankaDieLiftJointD8BigEnvCfg` (d8, 48mm-parity) reliably attempts
gripper closure near the object (seed 123: `frac_steps_raw_action_negative_near_object
= 1.0` in 7/8 envs) but that closure never produces a lift (0/24
sustained-lift, all 3 seeds). The research doc confirms, by direct source
read, that this Franka reward (`tasks/franka/lift_env_cfg.py:280-304`,
inherited unmodified by every d8/d10/d12/d20 env cfg in this arc) has **no
contact-force, contact-direction, or grasp-quality term at all** — only
`lifting_object`, a binary height check that fires *after* a lift already
happened, with zero gradient signal at the moment of closure itself. This is
consistent with this project's own AR4-era precedent
([[grasp-mechanics-antipodal-vs-magnitude]], Experiments 1→9→10→11): a
magnitude-only bilateral-contact reward converges easily but rewards
non-antipodal, physically-unstable pinches; a real antipodal/force-closure
check is a mechanistically different, much sparser signal.

**The load-bearing complication, which this spec is structured around
rather than working around:** Experiment 10 found that even a
*physically-correctly-thresholded* antipodal check regressed to **exactly
0.000000** under direct joint-space (`JointPositionActionCfg`) control —
not a reward-calibration problem but a gripper-positioning-precision
problem. Only Experiment 11's switch to task-space/Cartesian
(`DifferentialInverseKinematicsActionCfg`) control produced the project's
first genuine, sustained antipodal signal (0.018815 final, nonzero in 91.6%
of iterations). Franka's current d8 env cfg
(`FrankaDieLiftJointD8BigEnvCfg`, `tasks/franka/dice_lift_joint_env_cfg.py`)
uses joint-space control — the same action-space class Experiment 10
diagnosed as the bottleneck, on the same underlying platform (a Franka Panda
arm/hand, both AR4-era and now). Testing the antipodal mechanism only under
joint-space would reproduce, not test, an already-known result; the research
doc's own falsification section states this explicitly and calls for testing
**both** action spaces before treating any negative result as dispositive.
This spec's design is built around that requirement.

**This is a Tier 1 structural experiment** (a new reward term/mechanism and,
for one of its two conditions, a new action-space configuration) per
CLAUDE.md's Workflow section, gated on the research doc above. **This spec
is design only — no implementation plan, no code changes, no Isaac Sim
launches.**

## Research grounding (see the research doc for full detail; load-bearing points restated here)

- **Classical force-closure/antipodal theory** (Nguyen 1988; Ferrari & Canny,
  ICRA 1992 — the standard "Ferrari-Canny" grasp-quality metric; Ponce &
  Faverjon 1991/93; GraspIt!, Miller & Allen 2004; Dex-Net 2.0, Mahler et al.
  arXiv:1703.09312; GPD, ten Pas et al. arXiv:1706.09911): a geometric
  antipodal/force-closure check is treated as necessary for a real grasp,
  never substitutable by contact-force magnitude alone — hard bilateral
  contact can register from a non-opposing, unstable pinch.
- **RL-specific evidence a contact-direction-aware reward beats a
  magnitude/binary one**: Koenig, Liu, Janson, Howe, arXiv:2109.11234 (2021)
  — a contact-position/normal/force-aware reward outperforms a non-tactile
  binary baseline by 42.9% (different hand morphology — multi-fingered, real
  tactile sensors — flagged by the research doc as an extrapolation, not a
  literal match; this project's own Experiment 1→9 result, same 2-finger
  parallel-jaw regime and simulated ground-truth `ContactSensorCfg` state, is
  the more directly analogous evidence).
- **This project's own AR4-era arc is the strongest, most directly relevant
  precedent** (Experiments 1, 9, 10, 11 — see
  [[grasp-mechanics-antipodal-vs-magnitude]] and [[action-space-design]]):
  magnitude-only contact converges to ~92% sustained contact but is not a
  real grasp signal; a real antipodal check (Experiment 9) fires ~1800x less
  often; correcting the threshold to the scene's own physically-derived value
  (Experiment 10) makes the signal regress to exactly zero under joint-space
  control; switching to task-space/IK control (Experiment 11) is what
  actually unlocks a real, sustained antipodal signal for the first time.
  **Switching action space, not any reward-threshold change, was the single
  highest-leverage lever in that entire ten-plus-experiment arc**
  ([[action-space-design]]) — and it is also the choice CLAUDE.md's North
  Star explicitly favors for cross-arm generalization, independent of this
  project's own empirical result agreeing with it.
- **Isaac Lab infrastructure re-verified live at v2.3.1** (this project's own
  pinned cloud tag): `ContactSensorCfg.filter_prim_paths_expr` and
  `ContactSensorData.force_matrix_w` (the filtered field — `net_forces_w` is
  NOT filtered by `filter_prim_paths_expr`, a distinction this project's own
  AR4-era work already discovered and re-confirmed current) are both present
  and unchanged from the AR4-era citation.

## Exact mechanism proposed

### Reward function — ported from `tasks/ar4/mdp.py`'s `antipodal_grasp_bonus`, not designed from scratch

`tasks/ar4/mdp.py:902-940` already has a working, previously-validated
implementation of exactly this mechanism (Experiments 9/10/11 all used it,
under three different threshold/action-space configurations). `tasks/franka/`
never imports from `tasks/ar4/` (confirmed by grep — the two arm packages are
independent per this repo's own convention), so this spec proposes **porting
the same math into a new function in `tasks/franka/mdp.py`**, not a
literal import:

```python
def antipodal_grasp_bonus(
    env: ManagerBasedRLEnv,
    force_threshold: float,
    antipodal_cos_threshold: float,
    jaw1_contact_cfg: SceneEntityCfg,
    jaw2_contact_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Bilateral force-closure grasp bonus: requires both jaw contact-force
    magnitudes to exceed force_threshold AND their force directions to be
    nearly anti-parallel (cosine below antipodal_cos_threshold) - the
    classical two-contact force-closure necessary condition (Nguyen 1988;
    Ferrari & Canny 1992). Ported verbatim from tasks/ar4/mdp.py's own
    antipodal_grasp_bonus (Experiments 9-11) - same math, same signature,
    new module because tasks/franka/ does not import tasks/ar4/."""
    jaw1_sensor: ContactSensor = env.scene[jaw1_contact_cfg.name]
    jaw2_sensor: ContactSensor = env.scene[jaw2_contact_cfg.name]
    jaw1_force_vec = jaw1_sensor.data.force_matrix_w.view(env.num_envs, 3)
    jaw2_force_vec = jaw2_sensor.data.force_matrix_w.view(env.num_envs, 3)
    jaw1_force_mag = torch.linalg.vector_norm(jaw1_force_vec, dim=-1)
    jaw2_force_mag = torch.linalg.vector_norm(jaw2_force_vec, dim=-1)
    both_magnitude_ok = (jaw1_force_mag > force_threshold) & (jaw2_force_mag > force_threshold)
    jaw1_dir = jaw1_force_vec / (jaw1_force_mag.unsqueeze(-1) + 1e-8)
    jaw2_dir = jaw2_force_vec / (jaw2_force_mag.unsqueeze(-1) + 1e-8)
    cos_angle = torch.sum(jaw1_dir * jaw2_dir, dim=-1)
    antipodal_ok = cos_angle < antipodal_cos_threshold
    return (both_magnitude_ok & antipodal_ok).float()
```

### Threshold — physically derived from day 1, not an initial guess corrected later

Experiment 9's own history is an explicit lesson here: guessing
`antipodal_cos_threshold=-0.85` first, then correcting it in Experiment 10
once the scene's real friction coefficient was known, is not repeated. The
Franka die-lift scene has **no `RigidBodyMaterialCfg` override anywhere**
(confirmed by the research doc's own grep across `lift_env_cfg.py`,
`dice_lift_joint_env_cfg.py`, `bake_die_asset.py`) — Isaac Lab's
`RigidBodyMaterialCfg()` default, **μ = 0.5/0.5**, applies, independently
verified for this exact `tasks/franka/` asset stack by
`docs/superpowers/specs/2026-07-13-d4-edge-grasp-rung0-design.md`'s own
friction check. Using Experiment 10's own derivation
(`threshold = -cos(arctan(μ))`, the 45°-friction-cone case giving -0.7071 at
μ=1.0): at **μ = 0.5, half-angle = arctan(0.5) = 26.565°, threshold =
-cos(26.565°) = -0.894427** (recomputed and verified this session, not
carried over from the μ=1.0 case). This is the value this spec proposes
using from the first run — not a value to discover is wrong after a null
result, as happened in Experiment 9.

`force_threshold`: **0.05** (N), reusing Experiment 9/10's own value
unchanged — an implementer-set starting constant, not load-bearing for this
spec's falsification bars (a Tier 2 hillclimb candidate once/if this
mechanism is validated, same treatment as the exploration-bonus spec's
`w_attempt`/`k`/`std_gate`). **`antipodal_cos_threshold` is explicitly NOT
in this "free to tune" category** — unlike `force_threshold`/reward weight,
it has one physically-correct value for this scene's real friction
coefficient, and this spec's whole point (per Experiment 9's own lesson) is
not to treat it as an arbitrary dial.

Reward weight: **1.0**, an implementer-set starting value in the same "small
nudge to the *sampling* distribution, not a competing objective" register
this project already uses for exploration-bonus-style terms — deliberately
far below `lifting_object` (15.0) / `object_goal_tracking` (16.0) so this
term cannot itself reproduce Experiment 8/9's own diagnosed reward-rate-
dominance failure mode. Not load-bearing for falsification; Tier 2 candidate
later.

### Scene wiring — `ContactSensorCfg` on `panda_leftfinger`/`panda_rightfinger`, adapting an already-proven-on-Franka pattern

`tasks/franka/dice_scene_cfg.py` (the scripted dice-pick demo's scene) already
proves the exact wiring this needs works on this Franka asset:

- `_FRANKA_ROBOT_CFG_WITH_CONTACT = FRANKA_PANDA_HIGH_PD_CFG.copy();
  .spawn.activate_contact_sensors = True` — PhysX activates
  `PhysxContactReportAPI` per-body for the **whole** robot at spawn time, not
  selectively, so this one line is sufficient to instrument every Franka
  body including both fingers.
- A `ContactSensorCfg` can point directly at an existing body prim
  (`{ENV_REGEX_NS}/Robot/panda_leftfinger` /
  `.../panda_rightfinger`) with `filter_prim_paths_expr` targeting the
  object — this is exactly the AR4 pattern already validated in
  `tasks/ar4/pickplace_env_cfg.py`'s `gripper_jaw1_contact`/
  `gripper_jaw2_contact` (`prim_path=".../gripper_jaw1_link"`,
  `filter_prim_paths_expr=[".../Sphere"]`). **This is a different situation
  from `dice_scene_cfg.py`'s own d4 rung-1 retargeting to a notch fixture**:
  that retargeting was needed only because rung-1 added *new collision
  geometry* (a fixture mesh) as a child of the finger, which instanceable
  prims don't straightforwardly allow — a bare `ContactSensorCfg` reading an
  *existing* body's already-activated contact API needs no new child prim
  at all, so no fixture is needed here.

Proposed new scene subclass (mirrors `FrankaDieLiftTargetSelectionSceneCfg`'s
own already-established "extend `FrankaLiftSceneCfg` with new sibling
fields" precedent in the same file, `dice_lift_joint_env_cfg.py:1024`):

```python
class FrankaDieLiftContactSceneCfg(FrankaLiftSceneCfg):
    robot: ArticulationCfg = _FRANKA_ROBOT_CFG_WITH_CONTACT.replace(prim_path="{ENV_REGEX_NS}/Robot")
    panda_leftfinger_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
        update_period=0.0, history_length=0, track_contact_points=True, debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )
    panda_rightfinger_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
        update_period=0.0, history_length=0, track_contact_points=True, debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )
```

New `RewardsCfg` subclass (mirrors `ExplorationBonusRewardsCfg`'s own
"new subclass, base `RewardsCfg` untouched" precedent immediately above it in
`lift_env_cfg.py`):

```python
class AntipodalGraspRewardsCfg(RewardsCfg):
    antipodal_grasp_quality = RewTerm(
        func=mdp.antipodal_grasp_bonus,
        params={
            "force_threshold": 0.05,
            "antipodal_cos_threshold": -0.894427,
            "jaw1_contact_cfg": SceneEntityCfg("panda_leftfinger_contact"),
            "jaw2_contact_cfg": SceneEntityCfg("panda_rightfinger_contact"),
        },
        weight=1.0,
    )
```

## The two action-space conditions

### Condition A — joint-space (current default, no new action-space code)

`FrankaDieLiftJointD8BigAntipodalEnvCfg(FrankaDieLiftJointD8BigEnvCfg)`:
overrides only `scene` (→ `FrankaDieLiftContactSceneCfg`) and `rewards`
(→ `AntipodalGraspRewardsCfg`). The arm action stays the inherited
`JointPositionActionCfg(scale=0.5, use_default_offset=True)` — identical to
every other env cfg in this file's joint-space lineage. This is the
cheapest, lowest-new-infra-risk condition, and the one the research doc
flags as *not* a clean standalone test given Experiment 10's precedent — but
it must still be run, not skipped, both because a positive result here would
be a stronger and cheaper win than expected, and because the task's own
4-way outcome matrix (below) needs both conditions' real data, not an
assumption.

### Condition B — task-space/IK (reuses this repo's own already-existing, already-trained Franka IK action config — no new IK code)

**Concretely answering the question of what this condition needs:** Isaac
Lab's `DifferentialInverseKinematicsActionCfg` + `DifferentialIKControllerCfg`
is not new infrastructure to build — it is already the base class's own
default. `tasks/franka/lift_env_cfg.py`'s `FrankaLiftEnvCfg.ActionsCfg`
(lines 161-178) already defines exactly this (`command_type="pose",
use_relative_mode=True, ik_method="dls"`, `scale=0.5`,
`body_offset=(0,0,0.107)`) — this is the "ik-cube" recipe this project has
already trained at 4096 envs to completion multiple times
(`docs/cloud/franka-cloud-shakedown.md`, `ROADMAP.md`'s "ik-cube ... trained
uninterrupted to 1500/1500" entries). `FrankaDieLiftJointEnvCfg`'s own
`__post_init__` is what *replaces* this default with joint-space control;
the task-space condition for this spec is simply **not applying that
override** — re-asserting the base class's own original `arm_action` in a
new subclass's `__post_init__`, after `super().__post_init__()` has already
run the joint-space override:

```python
class FrankaDieLiftD8BigTaskspaceAntipodalEnvCfg(FrankaDieLiftJointD8BigAntipodalEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.actions.arm_action = mdp.DifferentialInverseKinematicsActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5,
            body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.107)),
        )
```

**`scripts/dice_pick_demo.py`'s DiffIK controller is a different mechanism
and is explicitly NOT what this condition reuses**, a distinction worth
stating plainly since the task raised the question directly: `dice_pick_demo.py`
is a scripted, non-RL waypoint-follower — it computes a fixed sequence of
Cartesian targets from classical geometry (see
`tasks/franka/antipodal_edge_grasp.py`) and drives Isaac Lab's
`DifferentialInverseKinematicsAction` open-loop, with no learned policy in
the loop at all. What an RL-trained task-space policy needs is the
**ActionTerm** variant — the policy's own output *becomes* the per-step
Cartesian delta command, and IK converts it inside the control loop every
step — which is exactly what `FrankaLiftEnvCfg.ActionsCfg.arm_action` already
is. Both usages share the same underlying Isaac Lab controller class, but
`dice_pick_demo.py`'s own trajectory-planning logic is not reused here; only
the ActionTerm configuration is, and it already exists.

**A real, known risk to watch, not to resolve at this spec's stage:**
Experiment 11 (AR4, first-ever IK-action RL run on that platform) hit a PPO
critic-divergence bug (`Loss/value_function` exploding from ~0 to ~5e23),
fixed with `clip_actions=5.0` on that experiment's PPO runner cfg
([[ppo-critic-divergence]]). `FrankaLiftPPORunnerCfg`
(`tasks/franka/agents/rsl_rl_ppo_cfg.py`) has no `clip_actions` override
today. This project's own Franka "ik-cube" history already trains stably at
4096 envs without one, which lowers but does not eliminate this risk here —
a new, potentially discontinuous contact-force-based reward term is a new
element even on an already-stable action space. If Condition B's training
run shows the same divergence signature, applying Experiment 11's own fix
(scoped only to this condition's own PPO runner cfg, never the shared
default) is implementation-plan-level detail, not a design decision this
spec needs to resolve now.

## Scope: d8 only, not d10/d12/d20

This spec deliberately stays on d8, for the same reasons the exploration-
bonus spec gave for the identical scope choice on this identical env cfg,
restated because they still hold:

- **The cleanest, most robustly-characterized null in this project's
  history to test a fix against.** d8 at 48mm-parity is 0/24 across 3 seeds
  from scratch, with the SPLIT result (reliable closure attempt, no lift)
  already independently instrumented, verified frame-by-frame, and not
  attributable to any known code bug.
- **d8 over d10/d12/d20 as the single-shape isolation target.** d10 carries
  two additional confounds beyond sphericity (bbox anisotropy, no
  parallel-face pairs, per the d8/d10-grasp-discoverability research); d12
  and d20 already show partial from-scratch discovery (1/3, 2/3 seeds) at
  this anchor, so a null-vs-fix comparison there is less clean than d8's
  robust 0/24 baseline.
- **This experiment already doubles its own scope along the one axis the
  task requires (2 action-space conditions × 3 seeds = 6 full 1500-iteration
  runs).** Adding more shapes on top of that, before either condition's
  result is known, would multiply cost with no isolation benefit — if
  Condition B (task-space) succeeds on d8, extending to d10/d12/d20 is the
  natural, well-motivated next spec; deciding that now would be scope creep
  against CLAUDE.md's own "one thing at a time, in sequence" discipline.

## Falsifiable hypotheses (two, distinct, per action-space condition)

Per the research doc's own explicit call: a joint-space-only test cannot
dispositively falsify the underlying grasp-quality hypothesis (Experiment 10
already showed a correctly-thresholded joint-space signal can regress to
zero for reasons unrelated to whether antipodal grasp quality matters), so
this spec states two separate, individually falsifiable claims rather than
one combined claim.

### H_joint — antipodal grasp-quality reward under joint-space control (Condition A)

**Claim:** adding `AntipodalGraspRewardsCfg`'s `antipodal_grasp_quality` term
to `FrankaDieLiftJointD8BigEnvCfg`'s otherwise-unmodified reward, under its
current joint-space `JointPositionActionCfg`, will produce a measurable,
above-noise antipodal signal during training, and that signal will translate
into sustained-lift grasp discovery in at least one of 3 seeds.

**Falsification bar (both required, all 3 seeds):**
1. **Mechanism-level:** the `antipodal_grasp_quality` reward term's own
   TensorBoard-logged mean value, averaged over the final 100 of 1500
   iterations, is **< 1e-4**, in **all 3 seeds** (42/123/7) — calibrated so
   Experiment 10's own real historical outcome (exactly `0.000000`) and
   Experiment 11's own real historical outcome (`0.018815` final) fall
   unambiguously on opposite sides of this bar.
2. **Behavioral:** `franka_checkpoint_review.py`'s existing sustained-lift
   protocol (0.04m threshold, `977a748`'s settle-window fix) shows **0/24**
   sustained-lift discovery (0/8 in every one of the 3 seeds).

**H_joint is falsified only if both bars fail.** A SPLIT (mechanism fires,
no lift) is reported as its own distinct outcome, not falsification —
identical treatment to the exploration-bonus spec's own precedent for this
exact env cfg.

### H_taskspace — antipodal grasp-quality reward under task-space/IK control (Condition B)

**Claim:** the identical reward mechanism, under `FrankaLiftEnvCfg`'s own
existing relative-IK `DifferentialInverseKinematicsActionCfg` (re-applied per
Condition B's `__post_init__` above) instead of joint-space control, will
produce a measurable, above-noise antipodal signal, and that signal will
translate into sustained-lift grasp discovery in at least one of 3 seeds —
mirroring Experiment 11's own AR4-era result under the identical "task-space
control fixes the positioning-precision bottleneck" mechanism.

**Falsification bar:** identical structure and identical numeric thresholds
to H_joint's two bars above, evaluated on the Condition B checkpoints
(seeds 42/123/7) instead of Condition A's.

**H_taskspace is falsified only if both bars fail**, with the same
SPLIT-is-not-falsification treatment.

### The combined outcome matrix — why both hypotheses must be run to completion regardless of either one's own result

Per the research doc's own framing, only one combination is a **dispositive
negative result for Direction 1 on Franka/d8 as a whole**: both H_joint and
H_taskspace independently falsified (both bars fail, both conditions, all 3
seeds — 0/48 total sustained-lift across the whole experiment). Every other
combination is a genuinely different, reportable finding, and is exactly the
kind of "might work under one action space and not the other" outcome the
task asked this spec to distinguish clearly rather than collapse:

| H_joint | H_taskspace | Reading |
|---|---|---|
| confirmed | confirmed | antipodal grasp-quality was the missing ingredient regardless of action space — action-space precision was not actually gating this mechanism on Franka the way it did on AR4. |
| falsified | confirmed | exact replay of the AR4-era Experiment 10→11 pattern: action-space precision is the real gate; the antipodal mechanism itself is validated once precision is available. |
| falsified | SPLIT (or confirmed→SPLIT swapped) | mixed evidence — report the specific per-condition pattern plainly, do not force it into a binary verdict. |
| both SPLIT | — | grasp-quality signal is learnable in at least one condition but insufficient alone for lift in either — points downstream, toward lift-execution itself ([[reach-grasp-lift-gap]]), not away from grasp-quality. |
| both falsified | — | **dispositive**: Direction 1 (contact/antipodal grasp verification) is closed for Franka/d8 specifically; report back to Principal per Direction 2 (physical-parameter) or a genuinely new direction, not an automatic escalation. |

Both conditions are run to completion (full 1500 iterations, all 3 seeds)
**unconditionally** — Condition B is not gated on Condition A falsifying
first. Running them sequentially-but-unconditionally (A then B, both always
executed) is preferred over parallel dispatch only for this project's own
practical GPU-scheduling reasons (one Isaac Sim process at a time via the
`flock` convention), not because B is contingent on A's result.

## Iteration budget and seeds

1500 iterations per run (matching `FrankaLiftPPORunnerCfg.max_iterations`
and this project's Tier 1 "full run + video review before any verdict"
mandate), seeds 42/123/7 (matching every other experiment on this exact env
cfg/anchor for direct comparability), **2 conditions × 3 seeds = 6 full
runs** total for this spec.

## Global constraints — what is deliberately NOT combined into this test

- **No exploration-bonus reward terms combined in.** This spec's
  `AntipodalGraspRewardsCfg` extends the plain `RewardsCfg`, not
  `ExplorationBonusRewardsCfg` — isolates the antipodal mechanism as its own
  variable, independent of the (already-tested, SPLIT-verdict) exploration
  bonus.
- **No demonstration warm-start** ([[d8-d10-demo-warmstart]]'s own
  mechanism) combined in — same "isolate one variable per experiment"
  reasoning as every prior spec in this arc.
- **No d10/d12/d20, no mixed population, no distractors/clutter** — see
  "Scope" above.
- **No physical-parameter changes (Direction 2)** — mass, friction, and
  collision approximation for d8 stay exactly as `FrankaDieLiftJointD8BigEnvCfg`
  already has them (0.216kg, default μ=0.5, convex-hull). Direction 2 is a
  separate, already-argued-against candidate in the research doc, not
  combined with Direction 1 here.
- **No tuning of `force_threshold` or the reward weight as part of this
  experiment's own falsification** — implementer-set starting values, Tier 2
  hillclimb candidates later if this mechanism validates.
- **`antipodal_cos_threshold` is not a tunable at all for this experiment** —
  it is fixed at the scene's own physically-derived value (-0.894427) for
  both conditions, precisely to avoid repeating Experiment 9's
  guess-then-correct mistake.
- **No `clip_actions` or other PPO-runner-cfg change pre-authorized for
  Condition A.** If Condition B's training run shows critic divergence,
  applying Experiment 11's own fix (scoped to Condition B's own runner cfg
  only) is implementation-plan-level judgment, not something this spec
  pre-resolves or extends to Condition A.

## Reused vs. new infrastructure

**Reused, unchanged or lightly adapted:**
- `tasks/ar4/mdp.py`'s `antipodal_grasp_bonus` math/signature (ported to a
  new function of the same name in `tasks/franka/mdp.py` — `tasks/franka/`
  never imports `tasks/ar4/`, so this is a port, not a shared import).
- `FrankaLiftEnvCfg.ActionsCfg.arm_action`'s existing
  `DifferentialInverseKinematicsActionCfg`/`DifferentialIKControllerCfg`
  configuration (Condition B) — zero new IK code.
- `tasks/franka/dice_scene_cfg.py`'s `activate_contact_sensors=True`
  copy-then-mutate pattern and its `ContactSensorCfg`-on-existing-finger-body
  wiring (proven on this exact Franka asset, for the scripted demo; this
  spec is the first time it is wired into an RL `ManagerBasedRLEnvCfg`).
- `tasks/ar4/pickplace_env_cfg.py`'s `gripper_jaw1_contact`/
  `gripper_jaw2_contact` pattern (direct `ContactSensorCfg` on a jaw body
  prim, `filter_prim_paths_expr` on the object) as the two-single-body-
  sensors-not-one-two-body-sensor convention.
- `FrankaDieLiftTargetSelectionSceneCfg`'s own "extend `FrankaLiftSceneCfg`
  with new sibling fields" precedent, same file, for the new
  `FrankaDieLiftContactSceneCfg`.
- `ExplorationBonusRewardsCfg`'s own "new `RewardsCfg` subclass, base
  untouched" precedent, for the new `AntipodalGraspRewardsCfg`.
- `franka_checkpoint_review.py`'s existing sustained-lift eval protocol
  (behavioral bar, both conditions).
- The friction-cone-threshold derivation formula itself (Experiment 10),
  re-applied with this scene's own real μ=0.5 (not carried over from AR4's
  μ=1.0).
- `FrankaDieLiftJointD8BigEnvCfg`'s own 48mm-parity die asset/scale/mass,
  PPO recipe (`FrankaLiftPPORunnerCfg`, `gamma=0.98`, `max_iterations=1500`),
  and 41-dim observation schema — all otherwise unmodified.

**Genuinely new:**
- `antipodal_grasp_bonus` function in `tasks/franka/mdp.py` (ported math,
  new module).
- `FrankaDieLiftContactSceneCfg` (new scene subclass: contact-activated
  robot spawn + two new `ContactSensorCfg` fields).
- `AntipodalGraspRewardsCfg` (new `RewardsCfg` subclass, one new additive
  term).
- `FrankaDieLiftJointD8BigAntipodalEnvCfg` (Condition A leaf env cfg) and
  `FrankaDieLiftD8BigTaskspaceAntipodalEnvCfg` (Condition B leaf env cfg),
  plus their `_PLAY` variants, mirroring every other env cfg in
  `dice_lift_joint_env_cfg.py`.
- New `--variant` choices for `scripts/train_franka.py`/
  `scripts/franka_checkpoint_review.py` (exact names/wiring left to the
  implementation plan).

## Success/failure reporting

Full 1500-iteration training run per (condition, seed) — no early verdicts,
per CLAUDE.md's Tier 1 mandate. Report both bars, per condition, per seed,
explicitly — not collapsed into a single pass/fail — and report the combined
outcome-matrix row this run's real result lands in (see table above), not
just "H confirmed/falsified." Video-review any positive result (rest frame
vs. peak-height frame showing a visibly different, genuinely gripped pose),
per this project's own standing "a shaped metric can misrepresent what's
physically happening" discipline (Experiment 16 precedent, reused explicitly
by the exploration-bonus spec's own reporting).

## Related

`docs/superpowers/specs/research/2026-07-20-d8-grasp-mechanics-literature.md`
(the Tier 1 research-gate document this spec executes on),
[[exploration-bonus-grasp-discovery]] (source of the SPLIT result motivating
this spec), [[grasp-mechanics-antipodal-vs-magnitude]] (the AR4-era
Experiment 1/9/10/11 arc this spec's mechanism and threshold derivation are
ported from), [[action-space-design]] (the joint-space-vs-task-space finding
this spec's dual-condition structure is built to test on Franka), [[reach-
grasp-lift-gap]], [[ppo-critic-divergence]] (the Condition B risk to watch),
`docs/superpowers/specs/2026-07-19-exploration-bonus-grasp-discovery-design.md`
(template/precedent for this spec's SPLIT-aware falsification structure, same
env cfg family), `docs/superpowers/specs/2026-07-13-d4-edge-grasp-rung0-design.md`
(source of the verified μ=0.5 default-friction fact this spec's threshold
depends on), `scripts/dice_pick_demo.py` (the scripted DiffIK mechanism this
spec explicitly does NOT reuse, distinguished from the ActionTerm it does
reuse).
