# Research: joint-space (direct `JointPositionActionCfg`) vs task-space/IK actions for a Franka tabletop lift policy

**Date:** 2026-07-11
**Author:** Senior research thread (delegated by Principal, `franka-panda-pivot`)
**Purpose:** Tier 1 hypothesis-gate research for an upcoming spec building a
joint-action-space variant of the Franka lift task for a d20 die. Per
CLAUDE.md's scientific-method gate, this document must exist and be cited
*before* that spec is written.

**Scope note:** research only. No implementation, no Isaac Sim launches (GPU
was held by another thread at research time). All Isaac Lab facts below come
from reading the installed package source directly at
`/home/saps/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/`
and `/home/saps/IsaacLab/docs/`.

---

## 1. Isaac Lab's own shipped Franka lift configs — exact facts

Three action-space variants of the same `LiftEnvCfg` base class exist under
`config/franka/`, each a `configclass` override of the arm's `ActionsCfg`:

### 1a. `joint_pos_env_cfg.py` — direct joint-space (the base variant)

`/home/saps/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/franka/joint_pos_env_cfg.py:34-42`

```python
self.actions.arm_action = mdp.JointPositionActionCfg(
    asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
)
self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
    asset_name="robot",
    joint_names=["panda_finger.*"],
    open_command_expr={"panda_finger_.*": 0.04},
    close_command_expr={"panda_finger_.*": 0.0},
)
```

Robot: plain `FRANKA_PANDA_CFG` (default PD gains, not the stiffened variant
— see 1b/1c). No inverse kinematics anywhere in the action path; the policy's
7 continuous action dimensions are joint-angle deltas around the current
default pose (`use_default_offset=True`), scaled by 0.5, sent straight to
each joint's position controller. Gripper is a separate 8th binary action
(open/close to fixed finger positions), identical across all three variants.

### 1b. `ik_abs_env_cfg.py` — absolute-pose differential IK

`/home/saps/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/franka/ik_abs_env_cfg.py:31-47`

Inherits from `joint_pos_env_cfg.FrankaCubeLiftEnvCfg` (i.e. built as a
diff on top of the joint-space config, not the other way around), then
overrides:
```python
self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(...)   # stiffer PD gains
self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
    asset_name="robot", joint_names=["panda_joint.*"], body_name="panda_hand",
    controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
    body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
)
```
Comment in-source: *"We switch here to a stiffer PD controller for IK
tracking to be better"* — i.e. Isaac Lab's own authors found plain
`FRANKA_PANDA_CFG` gains insufficient for good IK tracking and needed a
dedicated high-PD robot variant for the IK configs specifically. The
joint-space config does **not** need this stiffened variant.

### 1c. `ik_rel_env_cfg.py` — relative-pose differential IK

Same pattern, `use_relative_mode=True`, `scale=0.5`, also on
`FRANKA_PANDA_HIGH_PD_CFG`.

### 1d. Which variant is "the" trained-RL default — registration facts

`config/franka/__init__.py` registers all three as separate gym IDs. The
`kwargs` passed to `gym.register` differ sharply:

| Env ID | RL agent cfg entry points registered |
|---|---|
| `Isaac-Lift-Cube-Franka-v0` (joint_pos) | `rsl_rl_cfg_entry_point`, `skrl_cfg_entry_point`, `rl_games_cfg_entry_point`, `sb3_cfg_entry_point` — **all four** RL frameworks |
| `Isaac-Lift-Cube-Franka-IK-Abs-v0` | none |
| `Isaac-Lift-Cube-Franka-IK-Rel-v0` | `robomimic_bc_cfg_entry_point` only (behavior-cloning, not RL) |
| `Isaac-Lift-Teddy-Bear-Franka-IK-Abs-v0` | none |

This matches Isaac Lab's own published environment table exactly
(`/home/saps/IsaacLab/docs/source/overview/environments.rst:873-884`):
IK-Abs and IK-Rel Franka-lift rows have a **blank** "RL Library" column;
only the joint_pos row (`Isaac-Lift-Cube-Franka-v0`) lists
`rsl_rl (PPO), skrl (PPO), rl_games (PPO), sb3 (PPO)`. The IK-Rel variant's
only registered downstream consumer is `robomimic`'s BC config — Isaac
Lab's own `release_notes.rst:1466,1472` shows the IK-Rel env is used with
`teleop_se3_agent.py` (keyboard/SpaceMouse teleoperation for collecting
demonstrations), not standalone RL training.

**This is a direct, verifiable fact, not an inference from silence: Isaac
Lab's own shipped Franka-cube-lift task ships PPO hyperparameters for
exactly one action space — plain joint-space — across all four supported RL
libraries. The IK variants exist for teleoperation / imitation-learning
data collection, not as alternative RL-trained baselines.**

### 1e. PPO hyperparameters (joint-space variant, `rsl_rl`)

`agents/rsl_rl_ppo_cfg.py`:
```python
num_steps_per_env = 24
max_iterations = 1500
save_interval = 50
policy: actor_hidden_dims=[256,128,64], critic_hidden_dims=[256,128,64], activation="elu", init_noise_std=1.0
algorithm: value_loss_coef=1.0, clip_param=0.2, entropy_coef=0.006,
           num_learning_epochs=5, num_mini_batches=4, learning_rate=1.0e-4,
           schedule="adaptive", gamma=0.98, lam=0.95, desired_kl=0.01, max_grad_norm=1.0
```
No hyperparameter file exists for IK-Abs/IK-Rel in `rsl_rl` — they were
never given one because they're not registered for `rsl_rl` at all (see
1d).

### 1f. Episode length / scene / reward facts (`lift_env_cfg.py`, shared base)

- `episode_length_s = 5.0`, `sim.dt = 0.01` (100 Hz physics), `decimation = 2`
  → 50 Hz control, 250 policy-observable control ticks per episode.
- `scene.num_envs = 4096` (default; `_PLAY` variant drops to 50).
- Object: Nucleus `Props/Blocks/DexCube/dex_cube_instanceable.usd`, spawned
  with `scale=(0.8, 0.8, 0.8)` and `solver_position_iteration_count=16,
  solver_velocity_iteration_count=1, max_depenetration_velocity=5.0`. I did
  **not** independently verify the DexCube's native/absolute edge length
  (asset lives on Omniverse Nucleus, not inspected locally as USD) — flag
  this as an open item for whoever sizes the d20 die spawn (see §5).
- Reward terms: `object_ee_distance` (std=0.1, weight=1.0),
  `object_is_lifted` (minimal_height=0.04, weight=15.0),
  `object_goal_distance` coarse (std=0.3, weight=16.0) and fine-grained
  (std=0.05, weight=5.0), `action_rate_l2` (weight=-1e-4, curriculum-ramped
  to -1e-1 at 10000 steps), `joint_vel_l2` (same ramp). Termination:
  `time_out`, plus `object_dropping` at `root_height_below_minimum=-0.05`.
- This entire reward/termination/curriculum set is **shared unmodified**
  across all three action-space variants (defined once in the base
  `LiftEnvCfg`, only `ActionsCfg` and `scene.robot` differ per variant) —
  i.e. Isaac Lab's own design treats action space as an orthogonal axis from
  reward design, consistent with this repo's North Star preference for
  reward designs that don't hardcode action-space assumptions.

---

## 2. Verified external literature

Searched Google Scholar and arXiv directly (`export.arxiv.org/api/query`,
fetched abstracts/full text via `ar5iv.labs.arxiv.org`) per this repo's
"research means web literature" practice. All five below were fetched and
their claims checked against the actual abstract/body text — not taken on
citation-string faith. I deliberately include the two that argue *against*
plain joint-space to avoid one-sided cherry-picking (this repo's own
citation-verification history flags fabricated/overstated citations as a
recurring risk).

1. **Varin, Grossman, Kuindersma, "A Comparison of Action Spaces for
   Learning Manipulation Tasks," IROS 2019** (arXiv:1908.08659, verified via
   arXiv API — title/authors/date/abstract all confirmed).
   **What it actually claims:** compares four action spaces (torque, joint
   PD, inverse dynamics, task-space impedance) on three tasks (peg
   insertion, hammering, pushing) with PPO and SAC. Verbatim: *"Our results
   lend support to the hypothesis that learning references for a task-space
   impedance controller significantly reduces the number of samples needed
   to achieve good performance across all tasks and algorithms."*
   **Read against joint-space, not for it** — this is the strongest
   external evidence *against* defaulting to plain joint-space PD for
   contact-rich precision tasks. Important caveat for the spec: this
   compares low-level torque/impedance controllers on force-sensitive
   insertion/hammering tasks, not massively-parallel GPU-batched PPO with a
   simple binary gripper and a pick-and-lift (not insertion) success
   criterion — the regime differs from Isaac Lab's actual lift-task setup
   in both scale and task class.

2. **Martín-Martín, Lee, Gardner, Savarese, Bohg, Garg, "Variable Impedance
   Control in End-Effector Space: An Action Space for Reinforcement
   Learning in Contact-Rich Tasks," IROS 2019** (arXiv:1906.08880, verified).
   **What it actually claims:** proposes VICES (task-space variable
   impedance) and shows it *"improves sample efficiency, maintains low
   energy consumption, and ensures safety"* versus other action spaces on
   path-following, door-opening, and surface-wiping tasks, and transfers
   sim-to-real. Also **against** plain joint-space by design — same caveat
   as #1: contact-continuous/constrained tasks, not GPU-parallel pick-lift.

3. **Makoviychuk et al., "Isaac Gym: High Performance GPU-Based Physics
   Simulation For Robot Learning," 2021** (arXiv:2108.10470, verified).
   Isaac Gym is the direct GPU-parallel-simulation precedent for Isaac
   Lab's own manager-based RL stack; its shipped manipulation tasks
   (including the Franka-cabinet/Franka-cube family this Isaac Lab lift
   task's PPO recipe descends from) use direct joint-space PD-target
   actuation trained at thousands-of-envs scale. This is precedent for
   *this specific infrastructure lineage* treating joint-space as
   sufficient at the massively-parallel-PPO scale Isaac Lab now runs, not
   an independent academic comparison — cited as infrastructure precedent,
   not as a controlled action-space study.

4. **OpenAI et al. (Andrychowicz et al.), "Learning Dexterous In-Hand
   Manipulation," 2018** (arXiv:1808.00177, verified — fetched full text via
   ar5iv, not just abstract). Verbatim, §4.2: *"Policy actions correspond to
   desired joints angles relative to the current ones."* Confirms: pure
   joint-space (relative joint-angle) actions, trained with PPO plus large-
   scale domain randomization, successfully sim-to-real transferred a
   24-DOF Shadow Hand dexterous in-hand reorientation policy. This is
   evidence that plain joint-space action is not fundamentally limited even
   for a substantially higher-DOF, more contact-rich manipulation problem
   than a 7-DOF arm + binary parallel gripper picking up a die — the
   Varin/Martín-Martín caveat about "contact-rich tasks favor task-space"
   is not universal; scale of training and randomization matters as an
   alternative lever to action-space choice.

5. **Mittal et al., "Orbit: A Unified Simulation Framework for Interactive
   Robot Learning Environments," 2023** (arXiv:2301.04195, verified). Orbit
   is Isaac Lab's direct predecessor/rename (same NVIDIA team, same
   manager-based task API this repo now runs on) — the paper that
   introduced the "fixed-arm and mobile manipulators with different...
   action spaces" design Isaac Lab's `config/franka/{joint_pos,ik_abs,
   ik_rel}_env_cfg.py` pattern is a direct continuation of. Cited as the
   platform's own primary source, establishing that the joint_pos/IK-abs/
   IK-rel three-way split (and joint_pos's status as the only one shipped
   with a full RL recipe) is a first-party design decision by the
   simulator's own authors, not an artifact this repo introduced.

**Not included / explicitly not cited:** I did not find, and am not
citing, any paper making a direct, controlled claim of the form "joint-space
PPO succeeds at X% on Franka-cube-lift specifically" — no such number is
published as far as I found in this search pass. Any success-rate target in
the eventual spec should be framed against this repo's own upcoming
empirical run, or against Isaac Lab community-reported informal numbers if
those are separately found and verified, not against a fabricated academic
figure.

---

## 3. Small objects, binary gripper + joint-space arm, episode/iteration budgets

- **Small-object grasping with joint-space arm actions specifically:** the
  literature search above did not surface a paper isolating "joint-space
  arm action + small (~2-3cm) rigid object" as its own variable — Varin/
  Martín-Martín's tasks are peg-insertion/hammering/door/wiping (not
  free rigid-body pick-lift of a small object), and Dactyl's object is a
  block manipulated in-hand, not table-picked. **This is a real gap**: the
  strongest available grounding for the *small-object* case specifically is
  Isaac Lab's own shipped task itself (§1), whose object (`dex_cube_
  instanceable.usd` at `scale=0.8`) is already a small tabletop cube of
  comparable order of magnitude to a d20 die, trained with exactly this
  action space. That is repo-adjacent empirical precedent, not literature,
  and should be named as such in the spec rather than dressed up as a
  citation.
- **Binary gripper action combined with joint-space arm action:** this
  combination is exactly what `joint_pos_env_cfg.py` already does
  (`JointPositionActionCfg` for the arm + `BinaryJointPositionActionCfg` for
  the gripper, §1a) — first-party precedent, not academic literature. No
  external paper found isolates the interaction effect between a discrete/
  binary gripper action and a continuous joint-space arm action as its own
  variable.
- **Episode length / iteration budget:** Isaac Lab's own shipped recipe
  (§1e/1f) is the only concretely-sourced number: `episode_length_s=5.0`
  (250 control ticks at 50 Hz), `num_steps_per_env=24`, `max_iterations=
  1500`, `num_envs=4096`. This repo's own AR4-era Tier-1 runs also
  standardized on 1500-iteration full runs (per CLAUDE.md's Workflow
  section) — coincidentally matching Isaac Lab's own `max_iterations=1500`,
  which is worth noting as alignment rather than treating as independent
  confirmation of anything (this repo picked 1500 iterations for AR4 before
  the Franka pivot, for its own reasons, and Isaac Lab's Franka default
  happens to match).

---

## 4. Repo-internal precedent — does AR4 Experiment 10/11 predict joint-space failure on Franka?

`ROADMAP.md:794-813` (Experiment 10) and `:814-879` (Experiment 11) are the
directly relevant AR4-era results:

- **Experiment 10** (direct joint-space action, AR4, 10th attempt on the
  grasp-discoverability sub-problem): `antipodal_grasp_bonus` regressed to
  **exactly 0.000000** by end of training. Conclusion recorded at the time:
  *"the bottleneck is precision of final gripper positioning/alignment
  under direct joint-space control, not reward-threshold calibration."*
- **Experiment 11** (switched to task-space differential-IK action, same
  reward set carried over, same PPO config apart from a diverged-critic fix
  specific to the new IK action term): first nonzero antipodal grasp
  contact this project had ever recorded (`antipodal_grasp_bonus` final
  0.018815, nonzero 91.6% of iterations) — read at the time as "task-space
  IK-driven action produced the first genuine sustained antipodal grasp
  contact."

**Why this does not predict joint-space failure on Franka:** the pivot
decision itself (CLAUDE.md, "Platform pivot" section) already identifies
the confound this research task was asked to re-examine: Experiment 10's
zero-grasp result was never isolated from AR4's own asset defects, three of
which are independently documented and unresolved as of the pivot:

1. A classical closed-form-IK pick attempt (not RL, not confounded by
   reward shaping) still missed the cube by 17-27mm — i.e. even a
   *non-learned* controller with perfect IK math could not reliably reach
   the cube on this AR4 asset, which caps what *any* action space could
   have achieved in Experiment 10, independent of joint-space vs IK.
2. The gripper's jaw-mimic constraint was never confirmed correctly
   enforced in Experiments 17-22 (`ROADMAP.md:1670-1991` — Experiments 19
   and 22 both attempted fixes and Experiment 22's own re-run showed
   `max_jaw1_force` at exactly 0.0N in one direction, a symmetric-contact
   defect that would degrade *any* action space's ability to close a
   stable grasp).
3. The jaw collision geometry uses an unverified convex-hull approximation
   that may distort contact-force directions read by the antipodal grasp
   check — meaning Experiment 10's *reward signal itself* (not just the
   controller) may have been reading distorted contact geometry, which
   would produce a zero/near-zero antipodal bonus regardless of how good
   the underlying grasp actually was.

Under this reading, Experiment 10 vs 11's contrast is confounded three
ways: (a) an unresolved IK-positioning-accuracy asset defect capping reach
precision for the *object* Experiment 10's policy needed to align to
regardless of action space, (b) a gripper mechanism defect that would
independently suppress grasp closure under joint-space specifically (the
policy has to *learn* to command precise finger closure timing without a
classical controller's help, so a broken jaw-mimic would hurt joint-space
more than IK, since IK's stiffer `FRANKA_PANDA_HIGH_PD_CFG`-style precision
gains were never in play on AR4 to begin with), and (c) a contact-force
reading defect that could suppress the *reward signal* Experiment 10 relied
on to detect a real grasp had happened, independent of whether one had.

Franka removes all three: it is Isaac Lab's own officially-supported,
validated reference platform (`isaaclab_tasks...manipulation.lift.config.
franka`, exactly what this research read directly in §1), with an asset,
gripper mimic constraint, and collision geometry that is the community's
most-replicated manipulation benchmark rather than a from-scratch URDF this
project built and calibrated itself. Combined with §1's finding that Isaac
Lab's own authors ship a full validated PPO recipe for joint-space
specifically (and *not* for either IK variant), Experiment 10's AR4 result
is not strong evidence against joint-space generically — it is better
explained as AR4-asset-defect-driven, consistent with the pivot's own
stated rationale, and the Franka joint-space case is a genuinely
independent test that hasn't been run yet.

**What Experiment 10/11 *does* still legitimately transfer:** the general
finding that "the same reward/PPO scaffold can look structurally different
depending on action space" (e.g. Experiment 11's critic-divergence fix,
`clip_actions=5.0`, needed specifically because of an IK action term's
occasional discontinuous joint-space jump) is a real methodological
lesson — action-space swaps can introduce their own training-stability
failure modes unrelated to the grasp-discoverability question, and the
upcoming joint-space Franka spec's plan should include an explicit
`Loss/value_function` stability check from iteration 1, not just the final
antipodal/lift metrics, regardless of which direction it turns out to be
unnecessary in practice.

---

## 5. Proposed falsifiable hypothesis + success criteria for the spec to adopt

**Hypothesis:** Direct joint-space PPO training (`JointPositionActionCfg`
mirroring Isaac Lab's own validated `Isaac-Lift-Cube-Franka-v0` recipe:
`scale=0.5, use_default_offset=True`, plain `FRANKA_PANDA_CFG` PD gains, no
IK anywhere in the action path) will produce a genuine, sustained grasp-and-
lift of a d20 die on the Franka platform within a comparable iteration
budget to Isaac Lab's own shipped recipe (`max_iterations=1500`,
`num_steps_per_env=24`), at a rate that is not explainable as an artifact of
AR4-specific asset defects — because this project's one prior joint-space
failure (AR4 Experiment 10) is confounded by three independently-documented,
still-unresolved AR4 asset defects (17-27mm classical-IK positioning miss,
unconfirmed jaw-mimic enforcement, unverified convex-hull contact-force
geometry) that cap what *any* action space could achieve on that platform,
none of which apply to Franka's officially-supported, community-validated
asset.

**Falsification condition:** if the joint-space Franka run reproduces
Experiment 10's exact failure signature (antipodal/grasp-contact metric
pinned at/near 0.0 for the full run, with `Loss/value_function` confirmed
stable — i.e. not a training-stability artifact) on a platform with none of
AR4's three documented defects, that would falsify the "AR4-asset-defect"
explanation and constitute real evidence of a more fundamental joint-space
RL limitation for this task class, worth escalating back to Principal
rather than re-attempted as a parameter tweak.

**Success criteria the spec should adopt (not a citation, a proxy analogous
to this repo's Tier-1 practice):** nonzero, non-collapsing grasp-contact
metric sustained across a meaningful fraction of iterations (comparable in
kind to Experiment 11's 91.6% nonzero rate, though the exact number isn't
independently benchmarked in literature — see §3's gap), *plus* independent
video/contact-force verification of an actual lift-to-height-and-carry
behavior, not just a held low-pose grasp (this repo's own Experiment 11/12
history shows a nonzero antipodal metric alone is consistent with "grasp
and freeze," not proof of task completion — see `ROADMAP.md:860-879`).

**Open item for the spec, not resolved by this research pass:** the d20
die's actual dimensions vs the DexCube's native size were not independently
verified here (§1f) — the spec/implementer should measure both directly
(USD inspection or a quick sim spawn check) before assuming the die is
"small enough" to behave like the validated DexCube recipe; if the die is
meaningfully smaller (a d20's inscribed sphere/circumsphere geometry is
different from a cube's uniform grasp width) or requires a much wider/
narrower gripper aperture than the DexCube's `scale=0.8` trains for, that's
a design-time parameter to size explicitly, not an assumption to inherit.

---

## 6. Open risks

- **Small-object literature gap (§3):** no external paper isolates
  "joint-space arm + small rigid tabletop object" as its own comparison —
  the spec's grounding for the small-object case rests on Isaac Lab's own
  shipped DexCube recipe (repo-adjacent precedent), not academic literature.
  State this plainly rather than overstating citation coverage.
- **Varin/Martín-Martín counter-evidence (§2, items 1-2)** is real and
  should be cited in the spec, not omitted — the honest reading is "joint-
  space is the platform's own validated default and removes an entire class
  of AR4-specific confounds, not that joint-space is unconditionally
  optimal for contact-rich manipulation in general."
- **Die geometry vs cube geometry (§5):** unresolved sizing/aperture
  question, flagged as a design-time task, not answered by this research.
- **No independent literature number for "expected success rate"** — the
  spec should not assert a numeric target as literature-derived; any
  numeric bar should be framed as this project's own proxy metric choice
  (per Tier-1 practice) or left qualitative (sustained grasp + lift +
  carry, verified via contact force + video, not scalar-only).
