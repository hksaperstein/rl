# d8/d10 demonstration-augmented warm-start — design spec

## Context

`kb/wiki/experiments/unified-multi-die-specialist-distillation.md`'s FINAL
VERDICT closed the unified-multi-die-specialist experiment with d8/d10
genuinely, robustly null (0/3 seeds each, both at real ~16-18mm size and at
the controlled 48mm-parity anchor, wide safety margins, independently
re-derived from raw trajectories twice) while d12/d20 fully succeeded
end-to-end (specialist → distill → RL-fine-tune, both recovered to their
frozen specialists' own 8/8 exactly). d8/d10 were narrowed out of that
experiment's scope on real evidence, not by default, leaving them as "open,
unsolved shapes for a future experiment, with a documented, evidence-backed
reason to start from."

That documented reason is
`docs/superpowers/specs/research/2026-07-19-d8-d10-grasp-discoverability-literature.md`
(hereafter "the research doc"), the Tier 1 hypothesis-gate research this
spec is gated on. Its headline finding: this project's own already-computed
Wadell sphericity values are monotonic with discovery rate across all four
shapes at the controlled 48mm-parity anchor (d8 ψ=0.8896/0-of-3, d10
ψ=0.8959/0-of-3, d12 ψ=0.9286/1-of-3, d20 ψ=0.9524/2-of-3, all "of 3 seeds
with discovery" — every discovering seed reached full 8/8-within-seed
completeness, never a partial count), and confirmed only one training
recipe was ever tried against d8/d10 (identical reward function, PPO
hyperparameters, and observation schema as the shapes that worked — only
the object asset differs). It ranked three candidate interventions by
strength of grounding; this spec adopts the top-ranked as its primary
hypothesis and pre-authorizes the second-ranked as a bounded fallback,
per this repo's established "one fallback rung, no new spec" pattern (see
`docs/superpowers/specs/2026-07-11-joint-space-die-lift-design.md`'s
"Success criteria / verdict protocol" section, mirrored here).

This is a Tier 1 structural experiment (a new training-initialization
mechanism, not a reward/observation/PPO-hyperparameter tweak) per CLAUDE.md's
Workflow section, gated on the research doc above. **This spec is design
only — no implementation plan, no code changes, no Isaac Sim launches.**

## Research grounding (summary — full detail in the research doc)

- **H1's citation**: Rajeswaran, Kumar, Gupta, Vezzani, Schulman, Todorov,
  Levine, "Learning Complex Dexterous Manipulation with Deep Reinforcement
  Learning and Demonstrations" (DAPG), RSS 2018, arXiv:1709.10087 —
  confirmed real via the arXiv API by the research doc. DAPG's actual
  mechanism has two parts: (a) behavior-cloning pretraining of the policy
  on demonstrations before RL begins, and (b) an augmented policy-gradient
  objective during RL itself — an additional imitation term computed on the
  demonstration data, weighted by a coefficient that decays geometrically
  over training iterations, so the demonstration's pull on the policy
  persists (softly) through RL rather than being forgotten after a one-time
  pretrain. **This spec adopts only part (a)**, not part (b) — stated
  explicitly as a partial adoption, not a full reproduction, for reasons
  given in "Design choice: BC-pretrain-then-fine-tune, not a live augmented
  loss" below.
- **What makes H1 the top-ranked candidate specifically for d8/d10** (not
  just the citation): this project's own scripted DiffIK controller
  (`scripts/dice_pick_demo.py`) already achieves a verified grasp-and-lift
  on d8 and d10 specifically (`kb/wiki/experiments/dice-pick-demo.md`:
  "d8 240.9mm / d10 239.3mm z-gain," both PASS) — direct, in-this-simulator
  proof that a grasp is physically achievable for exactly these two
  objects, at real commercial size, with this gripper. The open question is
  whether the RL policy can *discover* it, not whether it exists.
- **H2's citation**: Wan, Geng, Liu, Shan, Yang, Yi, Wang, "UniDexGrasp++,"
  ICCV 2023, arXiv:2304.00464, §4.4's GeoCurriculum (geometry-feature-
  ordered curriculum, not category/identity-ordered) and its predecessor's
  own reported result (Xu, Wan, Zhang, et al., "UniDexGrasp," CVPR 2023,
  arXiv:2303.00938 — object curriculum improved success rate 31%→74% on
  their training set, a secondary citation via UniDexGrasp++'s own recap,
  not independently re-verified against UniDexGrasp's primary text beyond
  confirming the paper is real). This spec's H2 adopts only the *ordering
  principle* (start from the population's easiest/nearest member, not the
  paper's full hierarchical-clustering machinery, which targets a
  thousands-of-objects continuous task space this project's discrete
  4-shape setting doesn't have).
- **H3 (exploration-noise/entropy retuning) is explicitly out of scope for
  this spec** — ranked weakest-fit by the research doc (the same PPO
  exploration settings already worked for d12/d20, so this is less likely
  to be the operative mechanism for d8/d10 specifically) and not
  pre-authorized here. If both H1 and H2 falsify, that is a stop-and-report
  point back to Principal, not an automatic trigger for H3.

## Scope

**In scope:**
- Shapes: d8 and d10, evaluated **separately** — the research doc's own
  §2b flags real, non-overlapping reasons the two might behave differently
  under an intervention (d10 has two compounding disadvantages beyond
  sphericity: ~9.8% bbox anisotropy and no parallel opposite-face pairs,
  neither of which d8 has), so this spec does not assume both shapes must
  pass or fail together. H1/H2 verdicts are reported per-shape.
- Size: 48mm parity, matching the already-built `FrankaDieLiftJointD8BigEnvCfg`/
  `...D10BigEnvCfg` — **not** real ~16-18mm size. This is the same
  controlled anchor Task 3.5 used precisely to isolate shape difficulty
  from absolute-scale confounds (at 48mm every shape presents an identical
  1.67x aperture-to-object ratio; at real size the ratios differ per shape,
  5.0x for d8/d10 vs 4.4x for d12, and Task 2's real-size results are
  strictly worse than Task 3.5's 48mm results for every shape tested).
  Re-introducing that confound here would make any positive OR negative
  result for this experiment unattributable to the demonstration/warm-start
  mechanism specifically.
- One training-initialization mechanism per hypothesis (BC-pretrain warm
  start for H1, checkpoint warm start for H2) — no reward-term changes, no
  PPO-hyperparameter changes, no observation-schema changes. Everything
  Task 3.5/the d20-big-geom gate task already validated (reward function,
  `FrankaLiftPPORunnerCfg`, 41-dim observation schema, 1500-iteration
  from-scratch budget as the comparison baseline) stays pinned.

**Explicitly out of scope:**
- H3 (exploration-noise/entropy retuning) — see above.
- The AR4-era antipodal-grasp-check reward mechanism (research doc §3d) —
  a genuine available lever, but a reward-*term* addition is its own Tier 1
  structural change requiring its own hypothesis/research grounding, not
  folded into this spec.
- d4 (no working grasp mechanism exists for it at all — a different,
  unsolved problem, not a demonstration-warm-start candidate).
- Vision-detector-in-the-loop observations, multi-object/distractor scenes,
  full pick-and-place-to-goal — all already out of scope for the whole
  multi-die arc this spec continues.
- Modifying `dice_pick_demo.py`'s own tested Gate A/G/V contracts in place
  (see "Demonstration extraction," below — a new sibling script is used
  instead, to avoid touching an already-verified script's pass/fail
  behavior).

## H1 (primary hypothesis)

### Falsifiable hypothesis

> Warm-starting PPO from a demonstration-derived initialization — behavior-
> cloning pretraining of the policy's weights on a logged joint-position
> trajectory from this project's own already-verified scripted DiffIK grasp
> (`scripts/dice_pick_demo.py`), re-captured at 48mm scale, followed by a
> full 1500-iteration PPO fine-tune resumed from that pretrained checkpoint
> — will produce nonzero sustained-lift grasp discovery for d8 and/or d10
> at the 48mm-parity anchor, where the vanilla from-scratch recipe got 0/24
> (0/8 in all 3 seeds) for both shapes (Task 3.5, re-audited).

### Falsification bar

For a given shape (d8 or d10 evaluated independently): **H1 is falsified
for that shape if all 3 seeds (42, 123, 7 — matching Task 3.5's own seed
set) show 0/8 sustained-lift discovery after the full BC-pretrain + 1500-
iteration PPO fine-tune (0/24 envs total for that shape)** — an outcome
identical to that shape's already-established from-scratch null.

**Why any nonzero count in even one seed counts as a real positive signal,
not noise** — this project's own complete experimental record for this
metric (cube 3/3 seeds, d20 1/3→2/3 seeds, d12 1/3 seed) contains **zero
instances of a spurious partial per-seed count**: every seed that ever
showed any discovery at all showed full 8/8-within-seed completeness (a
consequence of `num_envs` sharing one policy per seed — once a seed's
policy converges on the grasp mechanism, it applies to that whole
population, not a fraction of it). No shape in this project's history has
ever produced a lone 1/8 or 3/8 result uncorrelated with subsequent
full-seed completion. Given that precedent, this spec sets the bar at
**≥1/8 in at least one of the 3 seeds** as sufficient to call H1 successful
for that shape (not requiring all 3 seeds, matching this project's existing
"partial success" reporting convention for d12/d20 at this same anchor) —
and treats a genuinely novel *partial* per-seed count (something between
1/8 and 7/8, which has never been observed in this project before) as
itself a reportable finding worth flagging explicitly, not silently rounded
to "success" or "failure."

### Design choice: BC-pretrain-then-fine-tune, not a live augmented loss

DAPG's own mechanism (see "Research grounding" above) has two parts; this
spec implements only the behavior-cloning-pretrain half, for a concrete,
stated reason: this project already has a **proven, debugged, exactly this
pattern** — Task 5/Task 6 of the unified-multi-die-specialist experiment
built and ran precisely "BC-pretrained checkpoint → PPO-fine-tune resumed
from it" (`scripts/train_franka.py --checkpoint ... --policy_only_checkpoint`,
added specifically because a BC-only checkpoint's `optimizer_state_dict` is
intentionally empty and would otherwise crash `rsl_rl`'s default
`load_optimizer=True`), and it fully recovered both d12 and d20 from a
real BC-distillation regression (4/8 d20, 1/8 d12) back to each
specialist's own 8/8 exactly. Building DAPG's second half — a decaying
augmented-loss term mixed directly into `rsl_rl`'s own PPO gradient step —
would require patching `rsl_rl`'s algorithm class itself, a materially
larger and riskier change than reusing an already-validated checkpoint-
resume path. This is flagged as a genuine, partial (not full) adoption of
DAPG's mechanism, not smoothed over: if BC-pretrain-then-fine-tune
falsifies, that does not by itself rule out DAPG's full mechanism (the
decaying augmented loss might rescue a case where a one-time pretrain gets
washed out by 1500 iterations of on-policy PPO) — but building that is a
separate, larger scope decision this spec does not pre-authorize (would
need its own spec if H1-as-designed here falsifies and this specific
gap is judged worth pursuing).

### Demonstration extraction: what gets logged and how

**A real scale mismatch, addressed as an explicit prerequisite sub-step,
not assumed away:** `dice_pick_demo.py`'s existing verified d8/d10 PASS
(240.9mm/239.3mm z-gain) runs at real commercial die size (~16mm), spawned
via `tasks/franka/dice_scene_cfg.py`'s default scale — not the 48mm-parity
size this spec's target env (`FrankaDieLiftJointD8BigEnvCfg`/
`...D10BigEnvCfg`) trains at. The demo's own per-die grasp-height table
(`_DIE_REST_HEIGHT_M`) and tolerances (`_GRASP_POS_TOL` etc.) were measured
at real size and are not assumed to transfer unchanged to a ~3x larger die.
**Task 0 (prerequisite, bounded, mechanical — not a new hypothesis test):**
re-run the scripted grasp against a 48mm-scaled d8/d10 die and re-verify
PASS (sim-ground-truth lift check, `dice_pick_demo.py`'s own existing
verdict mechanism) before trusting the trajectory as a demonstration
source. If this re-verification itself fails (the scripted grasp doesn't
transfer to 48mm scale), that is worth reporting on its own — it would mean
H1's premise ("a known-feasible grasp trajectory already exists for these
two shapes") does not actually hold at the size this spec trains at, which
would undercut H1 before any PPO training even starts.

**Capture script**: a **new sibling script**
(e.g. `scripts/extract_demo_trajectory.py`), not an in-place modification
of `dice_pick_demo.py` — reuses that file's existing functions
(`spawn_scene_and_settle`, `run_detector_subprocess`, `select_target_detection`,
`run_pick_sequence`) by import, the same way `dice_pick_demo.py`'s own Gate
V reuses Gate G's flow via the `on_step` hook, rather than touching Gate
G/V's own already-tested pass/fail contract. The new script:
1. Overrides the target die's spawn scale to the 48mm-parity constant
   (0.003167 for d8, 0.002928 for d10 — the same values
   `FrankaDieLiftJointD8BigEnvCfg`/`...D10BigEnvCfg`'s own docstrings
   derive) instead of `dice_scene_cfg.py`'s real-size default.
2. Re-measures `_DIE_REST_HEIGHT_M` for the scaled die (Task 0 above) and
   passes the corrected value into `_die_grasp_height_m`'s call site rather
   than reusing the real-size constant.
3. Passes a logging `on_step` callback into `run_pick_sequence` (the exact
   extension point Gate V's video capture already uses) that, on every
   physics step, reads and appends: (a) the desired joint-position target
   just issued to `panda_joint.*` (the `joint_pos_des` tensor
   `_step_toward`/`_joint_space_prep` compute internally before calling
   `robot.set_joint_position_target`), and (b) the gripper target tensor
   (`open_target`/`close_target`, whichever is currently commanded).
4. Runs this capture for **5 distinct seeds per shape** (d8: 5 runs, d10: 5
   runs — varying `--seed`, which resamples the whole table layout and
   therefore the commanded die's own detected XY position each time). Five
   is chosen as a concrete instance of DAPG's own "small number of
   demonstrations" framing, made cheap here because the demonstration
   source is a fully automated scripted controller rather than a human
   teleoperator — bounded and stated explicitly as an implementer-tunable
   count within DAPG's framing, not a number the cited paper itself
   specifies.
5. **Open risk, stated plainly, not resolved here**: `dice_pick_demo.py`'s
   own table-placement region (`_REGION_X=(0.40,0.60)`, `_REGION_Y=(-0.15,0.15)`,
   absolute world coordinates for its 5-die scene) does not necessarily
   correspond to the RL target env's own reset-randomization range
   (`reset_object_position`'s `pose_range={"x": (-0.1, 0.1), "y": (-0.25,
   0.25)}`, offsets around a different single-die nominal spawn pose in a
   different scene). The 5 captured seeds are not guaranteed to
   representatively cover the actual state distribution the RL policy must
   handle at every reset. This is an accepted limitation of a first
   attempt, not swept under the rug — if H1 succeeds despite this gap, that
   is itself informative (demonstration-shaped exploration/weight-init
   value transfers even without close reset-distribution matching); if H1
   falsifies, this gap is one of (not the only) plausible explanations
   worth naming in the report, alongside the joint-space-retargeting risk
   the research doc's own falsification note already flags.

### Integration into `tasks/franka/distillation.py`'s existing plumbing

**Reusable as-is (no changes needed):**
- `collect_rollout(env, action_fn, num_steps, device)` — built for a live
  teacher/student policy's `obs → action` callable, but its contract only
  requires `action_fn` to accept the current observation and return an
  action; nothing in its implementation requires that action to actually
  *depend* on the passed observation. A scripted-replay `action_fn` — a
  closure over a step counter that ignores its `obs` argument and returns
  the logged reference trajectory's precomputed action at the current
  index, incrementing on each call — satisfies this contract exactly.
  Reusing this function unchanged means the replay pass gets `collect_rollout`'s
  existing `torch.no_grad()` discipline and return-shape convention for
  free.
- `pool_and_shuffle(batches, generator)` — pools multiple demo trajectories'
  logged observation batches together; no changes needed (it already
  handles an arbitrary list of batches).
- `build_student_actor_critic(obs_dim, num_actions, device)` — the fresh
  network this pretrain targets; architecture-identical to every existing
  specialist/distilled checkpoint by construction, unchanged.
- `behavior_cloning_loss(student_action_mean, teacher_action_mean)` — plain
  MSE; the "teacher_action_mean" argument is just whatever action tensor is
  being regressed against, which works identically whether that tensor
  came from a live teacher network's inference call or a replayed demo's
  logged action.
- `save_student_checkpoint(student, output_path, iteration, extra_infos)` —
  exact same `rsl_rl`-compatible format Task 5/6 already proved loads
  cleanly via `train_franka.py --checkpoint --policy_only_checkpoint`.

**New, not reusable as-is — and precisely why:** `regress_on_pooled_batches`
and `MultiShapeTeacherRouter` were both built around a live, queryable
**teacher policy network** (`router.relabel(pooled_obs)` calls a frozen
`ActorCritic.act_inference` per row, routed by shape). H1 has no such
network for either shape — a scripted DiffIK controller's replayed
trajectory produces *fixed, already-paired* (observation, action) pairs
directly from `collect_rollout`'s output, with no relabeling step possible
or needed (there is exactly one "teacher" per shape, and it is not a
callable model, just recorded data). A new, small function is needed —
e.g. `regress_on_paired_batches(obs, actions, student, optimizer,
batch_size, num_epochs, generator)` — mirroring `regress_on_pooled_batches`'s
shuffle/minibatch/epoch loop and its call to `behavior_cloning_loss`
verbatim, but taking pre-paired `(obs, actions)` tensors directly instead
of calling a router. This is a mechanical, well-specified addition (same
loop shape as existing, tested code), not a new mechanism.

**Also new**: the closed-form conversion from the replayed reference
trajectory's absolute joint-position targets to the RL env's own action
space. `FrankaDieLiftJointD8BigEnvCfg`/`...D10BigEnvCfg` use
`JointPositionActionCfg(scale=0.5, use_default_offset=True)` for the 7 arm
joints (`tasks/franka/dice_lift_joint_env_cfg.py`), whose documented
semantics (Isaac Lab's own `JointPositionAction`) are `target_joint_pos =
default_joint_pos + scale * raw_action`, with `default_joint_pos` fixed at
env-construction time (not re-read per step) when `use_default_offset=True`
— so the inverse, `raw_action = (target_joint_pos - default_joint_pos) /
scale`, is a closed-form, per-step, no-live-feedback computation directly
from the logged reference trajectory, requiring no IK or replay-time
control-loop logic. The gripper term (`BinaryJointPositionActionCfg`) needs
one explicit verification step before implementation: **the implementing
task must confirm, by direct read of Isaac Lab's own
`isaaclab.envs.mdp.actions` binary-action source (on the desktop, where
Isaac Lab is installed — not assumed from memory here), the exact
raw-action-value convention that selects `open_command_expr` vs.
`close_command_expr`**, then map the logged open/close target at each
replayed step to that raw value. This is flagged as an unresolved
mechanical detail requiring direct source confirmation, per this project's
own citation/fact-verification discipline — not asserted as already known.

**Replay pass**: for each of the 5 logged reference trajectories per shape,
construct the real `FrankaDieLiftJointD8BigEnvCfg`/`...D10BigEnvCfg`
`ManagerBasedRLEnv` (`num_envs=1` is sufficient — replay is deterministic
open-loop tracking, not exploration, so no benefit to more envs per
replay), reset it, and call `collect_rollout` with the closed-form replay
`action_fn` described above for `num_steps = len(reference_trajectory)`.
This produces exactly the (`obs["policy"]`, `action`) pairs
`regress_on_paired_batches` needs, in the RL env's own real 41-dim
observation / 8-dim action schema, by construction — no separate schema-
matching step required. Pool all 5 shapes' replayed trajectories (~5 ×
1000-1900 steps ≈ 5,000-9,500 pairs per shape, exact count depends on how
many steps each of the 5 captured runs actually took to converge) as that
shape's BC pretraining dataset.

**Training procedure, end to end, per shape:**
1. Task 0: re-verify scripted grasp PASS at 48mm scale.
2. Capture 5 reference trajectories (varying seed) via the new sibling
   script.
3. Replay each through the real target env, logging paired (obs, action).
4. BC-pretrain a fresh student (`build_student_actor_critic`) against the
   pooled paired data via `regress_on_paired_batches`, save via
   `save_student_checkpoint`.
5. PPO fine-tune: `scripts/train_franka.py --checkpoint <BC checkpoint>
   --policy_only_checkpoint --variant <joint-die-d8-big|joint-die-d10-big>
   --max_iterations 1500` (mirroring Task 6's exact resume mechanics and
   iteration-arithmetic caveat — confirm via the same bounded smoke test
   Task 6 used before trusting the real dispatch).
6. Eval via `franka_checkpoint_review.py`, identical mechanism/undiluted-
   48mm/`num_envs=8` protocol as every existing specialist baseline in this
   arc, 3 seeds (42, 123, 7).

## H2 (pre-authorized fallback)

### Trigger condition

For a given shape: **H2 is authorized to run, without a new spec, if and
only if H1 falsifies for that shape** (0/24 per the bar above). H2 is
evaluated per-shape, independently — if H1 succeeds for d8 but falsifies
for d10 (or vice versa), only the falsifying shape proceeds to H2.

### Design

Geometry-ordered checkpoint warm-start: PPO-fine-tune the target shape's
env starting from the **already-converged, nearest-by-sphericity d12
checkpoint**
(`gs://rl-manipulation-hks-runs/unified-multi-die-specialists/joint-die-d12-big/seed123/2026-07-19_06-37-16/model_1499.pt`,
ψ=0.9286, the closest existing checkpoint to d8's ψ=0.8896 / d10's
ψ=0.8959 — d20's checkpoint, ψ=0.9524, is farther from both and is not
used here) instead of training from scratch. Schema compatibility is
already confirmed (Task 4's own checkpoint verification: both d12/d20
checkpoints carry an identical 41-dim observation / 8-dim action, matching
`FrankaLiftPPORunnerCfg`'s architecture exactly, and `FrankaDieLiftJointD8BigEnvCfg`/
`...D10BigEnvCfg` inherit the same observation terms unconditionally via
the shared base `ObservationsCfg` — no schema-adapter work is needed).
Mechanically simpler than H1: a direct `scripts/train_franka.py
--checkpoint <d12 checkpoint path> --variant <joint-die-d8-big|joint-die-d10-big>
--max_iterations 1500` resume — full optimizer-state resume (not
`--policy_only_checkpoint`), since d12's checkpoint carries real PPO
optimizer state and the only thing changing across the resume is the
object asset (same reward function, same PPO recipe, same 48mm-parity
size), a materially smaller distribution shift than H1's BC-checkpoint
resume. No new code in `tasks/franka/distillation.py` is needed for H2 at
all — it is a checkpoint-path substitution against `train_franka.py`'s
already-existing `--checkpoint` mechanism.

### Falsification / stop point

**H2 is falsified for a shape if all 3 seeds show 0/8 sustained-lift
discovery after the 1500-iteration fine-tune from the d12 checkpoint** —
same bar and same "any nonzero count in any seed is a real signal"
reasoning as H1. If H2 also falsifies for a shape, **that is a
stop-and-report point for that shape** — report both null results back to
Principal; do not proceed to H3 or any further intervention without a new
spec, per this repo's "one fallback rung, no new spec" convention.

## Global constraints

- **Shapes**: d8 and d10, evaluated and reported independently (see
  "Scope" above for why they are not assumed to behave identically).
- **Size**: 48mm parity (`FrankaDieLiftJointD8BigEnvCfg`/`...D10BigEnvCfg`),
  not real ~16-18mm size — preserves the scale-confound isolation Task 3.5
  was built to establish; comparing against a from-scratch baseline that
  used a different size would make any result unattributable to the
  warm-start mechanism specifically.
- **Execution backend**: desktop-first, cloud-fallback per current standing
  policy (`scripts/check_gpu_availability.sh` →
  `scripts/run_on_desktop_gpu.sh`, falling back to the GCP cloud path
  per `docs/cloud/dispatch-checklist.md` only if the desktop is BUSY or
  UNKNOWN — never assume availability). Non-headless (`DISPLAY=:1`,
  standing environment law) for any Isaac-Sim-touching step, including the
  demo-trajectory capture. Every Isaac-Sim-touching invocation wrapped in
  `flock -o /tmp/rl_isaac_sim.lock`.
- **Cost expectations**: this experiment reuses entirely existing infra
  (env cfgs, PPO recipe, distillation.py's core mechanics, train_franka.py's
  checkpoint-resume path) — no new asset baking, no new Isaac Sim scene
  construction beyond a lightweight scale-override on the existing dice
  scene. Per-shape cost is expected to be comparable to a single Task
  3.5/Task 6-style specialist run (~$0-1 on desktop, well under $1 if
  cloud-dispatched given this arc's own SPOT-preemption-adjusted precedent
  of ~$0.91 for a 3-seed 1500-iteration retrain) plus a cheap CPU/short-GPU
  demo-capture-and-replay step. Total expected spend for both shapes'
  H1 attempts plus any triggered H2 fallback: well under this project's
  established $15 cloud-spend-cap convention for an experiment of this
  size (the just-finished unified-multi-die-specialist experiment's own
  total was ≈$5.87 of that cap).

## Success/failure reporting

Per this project's verification standard: full 1500-iteration training run
(no early verdicts) for every PPO phase (BC-pretrain has its own, shorter,
loss-plateau-based stopping criterion — the implementing task should log
BC loss and confirm it has plateaued, mirroring Task 4's `--dry-run`
loss-decrease check, before starting the PPO fine-tune phase); real eval
via `franka_checkpoint_review.py` with raw per-step `.npy` height-trajectory
inspection (not summary-JSON-only trust, per this project's own repeated
settle-detection-bug discipline — three separate measurement bugs in this
exact tool were found and fixed across the unified-multi-die-specialist
experiment); video-frame spot-check of any positive result (a rest frame
vs. a peak-height frame showing a visibly different arm pose with the die
gripped, not just a height-number crossing a threshold). Report per-shape,
per-hypothesis (H1 result, H2 result if triggered) verdicts explicitly
against the falsification bars above — including an explicit call-out if
any seed lands in the previously-never-observed "partial" range (1/8-7/8),
since that would itself be a new empirical fact about this project's own
discovery dynamics worth recording in the kb regardless of the experiment's
overall verdict.
