# Research: alternative RL algorithms (SAC/auto-entropy) and open-source pretrained manipulation policies

**Date:** 2026-07-21
**Author:** Senior research thread (delegated by Principal)
**Purpose:** Tier 1 hypothesis-gate research, requested directly by the user
following today's root-cause investigation in
[[d8-antipodal-grasp-quality]] (`kb/wiki/experiments/d8-antipodal-grasp-quality.md`,
"Root cause investigation" section), which found joint-space training
transiently discovers a real approach-the-object capability (peaking at
iteration ~100) then abandons it over the remaining ~1400 iterations,
converging to a "hover-near-but-never-touch" policy with **exact zero
contact frequency across 8 checkpoints spanning 4 seeds/runs** — a pattern
the same investigation ties to PPO's own entropy-narrowing dynamics
(Hsu et al. 2020) compounding with joint-space's configuration-dependent
action-to-motion mapping. This document surveys two candidate responses
the user asked about directly: (1) switching to a different RL algorithm,
specifically SAC, and (2) starting from an open-source pretrained
manipulation policy instead of training from scratch. **This is research
only — no env cfg / agent cfg code, no reward-term design, no Isaac Sim
launches, no spec.**

---

## Direction 1: alternative RL algorithms

### 1a. Does this project's actual stack (`rsl_rl`) support SAC or any other off-policy/auto-entropy algorithm?

**No — confirmed by reading the installed version's own source directly,
not assumed from general RL knowledge.** This project's cloud dispatch
recipe pins `rsl-rl-lib==3.0.1` (`docs/cloud/franka-cloud-shakedown.md:299`).
Reading `rsl_rl/algorithms/` at the `v3.0.1` git tag on the library's own
GitHub repo (`leggedrobotics/rsl_rl`) directly shows exactly two files:
`ppo.py` and `distillation.py` — no SAC, DDPG, or any off-policy algorithm
exists in the installed version at all. The current `main` branch (now at
`v5.4.2`) has the identical two-file `algorithms/` directory — this has not
changed since `v3.0.1`.

The library's own README (`v3.0.1` tag, and unchanged through several later
tags) states explicitly: *"The main branch supports PPO and Student-Teacher
Distillation... the `algorithms` branch supports additional algorithms
(SAC, DDPG, DSAC, and more). However, it isn't currently actively
maintained."* This is a real, load-bearing finding worth flagging directly:
**that `algorithms` branch no longer exists.** `git ls-remote --heads
https://github.com/leggedrobotics/rsl_rl.git` and the GitHub branches API
both show only a single remote branch (`main`) as of this research. The
current `main` README no longer even mentions an `algorithms` branch —
the note has been fully removed, not merely left stale. **`rsl_rl`'s own
maintainers built an experimental SAC branch, described it as unmaintained
even at the time, and have since deleted it entirely.** There is no
existing SAC implementation anywhere in this project's actual training
library, current or historical-but-recoverable.

One directly relevant thing `rsl_rl` v3.0.1 *does* already support, found
while checking the algorithm-support question: **Random Network
Distillation (RND)**, a curiosity-driven intrinsic-reward exploration
bonus, is real, present, and wired all the way through `v3.0.1`'s own
`ppo.py` (`rnd_cfg: dict | None = None` constructor parameter,
`RandomNetworkDistillation` module, intrinsic reward added to the PPO
transition's extrinsic reward every step) — and Isaac Lab's own
`isaaclab_rl/rsl_rl/rl_cfg.py` wrapper already exposes it as a first-class
field, `RslRlPpoAlgorithmCfg.rnd_cfg: RslRlRndCfg | None = None`, `None` by
default. Enabling RND in this project's own `FrankaLiftPPORunnerCfg`
(`tasks/franka/agents/rsl_rl_ppo_cfg.py`) would require setting one
existing config field — **zero new library integration, zero new training
infra.** See §1d for its literature grounding and §4 for why this is worth
naming as a distinct, much lower-risk option than switching algorithms.

### 1b. What would integrating SAC into this Isaac Lab + Franka pipeline actually require?

Not a from-scratch implementation — Isaac Lab already ships wrapper
adapters for four separate RL libraries, confirmed by directly listing
`isaaclab_rl/isaaclab_rl/` on Isaac Lab's own GitHub repo: `rl_games/`,
`rsl_rl/`, `sb3.py`, `skrl.py`. Two of those four already-integrated
libraries have real, maintained SAC implementations, confirmed by directly
listing their own algorithm/agent directories:

- **Stable-Baselines3** — SAC is one of its core, actively-maintained
  algorithms (well-established, not separately re-verified here beyond
  confirming Isaac Lab's own `sb3.py` wrapper exists).
- **`rl_games`** (`Denys88/rl_games`) — `rl_games/algos_torch/sac_agent.py`
  confirmed present directly.
- **`skrl`** (`Toni-SM/skrl`) — `skrl/agents/torch/sac/` confirmed present
  directly (alongside `ddpg`, `td3`, `trpo`, and others — skrl is a
  broad-algorithm library built specifically with Isaac Gym/Isaac Sim/Isaac
  Lab environments as a first-class target).

Further, **Isaac Lab's own officially-shipped Franka cube-lift task already
registers gym entry points for all four wrapper libraries simultaneously**
(confirmed by reading
`isaaclab_tasks/manager_based/manipulation/lift/config/franka/__init__.py`
directly): `Isaac-Lift-Cube-Franka-v0` carries `rsl_rl_cfg_entry_point`,
`skrl_cfg_entry_point`, `rl_games_cfg_entry_point`, and `sb3_cfg_entry_point`
side by side, each pointing at its own PPO agent config file
(`rsl_rl_ppo_cfg.py`, `skrl_ppo_cfg.yaml`, `rl_games_ppo_cfg.yaml`,
`sb3_ppo_cfg.yaml`). This means the env-side plumbing (the
`ManagerBasedRLEnv`, the `VecEnv`-style adapter each wrapper needs) is
already proven to work uniformly across libraries for this exact task
family — this project would not be building a new environment bridge.

**Real, concrete cost, then:** the env-cfg side is a non-issue (this
project's own `FrankaDieLiftJointD8BigEnvCfg`-family env cfgs are already
`ManagerBasedRLEnvCfg`s, the same base class Isaac Lab's stock Franka
task uses, and gym registration already supports attaching multiple
`*_cfg_entry_point`s to one env cfg). The real cost is:

1. Adding a dependency on `stable-baselines3` or `skrl` (a new pinned
   package, new install step in the cloud dispatch recipe) — this project
   currently only installs `rsl-rl-lib`.
2. Writing and tuning a new SAC agent config from scratch for this
   project's specific task (replay buffer size, target-entropy schedule,
   twin-Q update ratio, learning rates) — this project has **zero existing
   tuning history for any off-policy algorithm**, unlike PPO where 20+
   experiments already inform what works. SAC's own well-known sensitivity
   to replay-buffer-size and update-to-data ratio in a GPU-parallelized,
   thousands-of-envs Isaac Lab setting (very different from typical
   single-env robotics SAC papers) is itself an open question this project
   has no data on.
3. Confirming this project's own `RewardsCfg`/episode-length/reset
   conventions transfer correctly to an off-policy training loop (PPO's
   on-policy rollout-per-iteration bookkeeping and SAC's replay-buffer
   bookkeeping are structurally different in `rsl_rl`-style massively
   parallel training — Isaac Lab's own `skrl`/`sb3` wrappers handle this,
   but this project has never exercised that path).
4. A full validation run (this project's own Tier 1 standard: multi-seed,
   1500 iterations or equivalent wall-clock, video review) before any
   verdict — the same cost any Tier 1 experiment already pays.

**Honest sizing: a multi-day, non-trivial integration effort — not "SAC
exists so it's free," but meaningfully cheaper than "build an RL library
from scratch" because Isaac Lab's own wrapper infrastructure for at least
one SAC-capable library (`skrl`, arguably the more manipulation-oriented of
the two given its Isaac-ecosystem-first design) is already proven against
this project's exact task family.**

### 1c. Real, verified precedent for SAC (or another off-policy/auto-entropy algorithm) outperforming PPO on this class of problem

- **Haarnoja, Zhou, Abbeel, Levine, "Soft Actor-Critic: Off-Policy Maximum
  Entropy Deep Reinforcement Learning with a Stochastic Actor," ICML 2018
  (arXiv:1801.01290)** — existence/accuracy-verified directly via the arXiv
  API (title, author list, and publication date confirmed to match). The
  original SAC paper; establishes the maximum-entropy off-policy mechanism
  the user described (automatic entropy-driven exploration + replay-buffer
  reuse). Real precedent for SAC's general sample-efficiency/stability
  advantage over on-policy methods, but this paper's own benchmarks are
  locomotion + a small number of manipulation tasks, not the specific
  "discovers behavior, then abandons it" failure mode this project
  diagnosed.
- **Haarnoja et al., "Soft Actor-Critic Algorithms and Applications,"
  2018 (arXiv:1812.05905)** — verified real via arXiv API. This is the
  follow-up that introduces SAC's automatic temperature/entropy-coefficient
  tuning specifically (the mechanism the user referenced as "automatically
  tunes its own exploration/entropy level" — that property is this paper's
  contribution, not the original 2018 ICML paper's fixed-temperature
  version). Includes real manipulation results (a Sawyer robot
  valve-turning task, among others).
- **Varin, Grossman, Kuindersma, "A Comparison of Action Spaces for
  Learning Manipulation Tasks," IROS 2019 (arXiv:1908.08659)** — verified
  real via arXiv API (already independently verified in this project's
  own `docs/superpowers/specs/research/2026-07-20-d8-grasp-mechanics-literature.md`
  and re-confirmed here). **This is the most directly relevant real paper
  found for this exact question** — it evaluates three contact-rich
  manipulation tasks (peg insertion, hammering, pushing) across four action
  spaces **and both PPO and SAC**. Its own conclusion, read from the
  abstract directly: task-space/impedance-control action spaces reduce
  sample complexity "across all tasks and algorithms" — i.e., **action
  space, not algorithm choice, is this paper's own headline finding.** This
  is worth stating honestly rather than stretched to fit: this is real,
  on-point evidence that SAC and PPO are both evaluated on this exact
  problem class, but it does not establish "SAC specifically fixes the
  found-it-then-lost-it pattern" — its own emphasis is that switching
  action space helps both algorithms, which is closer to what this
  project's own root-cause finding (§ below) already independently
  concluded.
- **Andrychowicz et al., "Hindsight Experience Replay," NeurIPS 2017
  (arXiv:1707.01495)** — verified real via arXiv API. Not a SAC paper
  itself (built on DDPG), but the single most on-point real precedent for
  the *mechanistic* argument the user raised: an off-policy, replay-buffer
  algorithm can relabel and reuse a rare success from any point in
  training, rather than discarding it after one on-policy update the way
  PPO does. This is real, credible support for the *plausibility* of the
  user's stated reasoning, honestly caveated: HER's own mechanism is
  goal-relabeling for sparse-reward *goal-reaching* tasks, and its
  original result is on DDPG, not SAC — it does not itself demonstrate SAC
  recovers from PPO's specific entropy-narrowing-driven abandonment
  pattern, only that off-policy replay of rare good experience is a
  real, well-established, mechanistically-motivated idea.

**No paper was found (searched directly, not merely absent from memory)
that specifically demonstrates SAC or another off-policy method recovering
a PPO-abandoned behavior in a joint-space contact-rich manipulation setting
matching this project's own diagnosed mechanism.** The precedent that
exists (Varin et al. 2019) supports the plausibility of trying SAC but its
own strongest finding argues action space matters more than algorithm —
which matters directly for the recommendation in §4.

### 1d. Lower-risk PPO-compatible alternatives, not requiring abandoning `rsl_rl`

- **RND (Random Network Distillation)**, already covered in §1a: real,
  present in the exact installed `rsl_rl` version, zero new
  infrastructure. Literature grounding: **Schwarke, Klemm, van der Boon,
  Bjelonic, Hutter, "Curiosity-Driven Learning of Joint Locomotion and
  Manipulation Tasks," CoRL 2023 (PMLR v229, cited directly in `rsl_rl`'s
  own README as the citation for this exact feature)** — existence
  verified directly via the PMLR proceedings URL
  (`proceedings.mlr.press/v229/schwarke23a.html`) `rsl_rl`'s own README
  links to; this is real, peer-reviewed CoRL 2023 work, and it is a
  manipulation-relevant citation (not merely locomotion), matching this
  project's own task class. **Honest caveat:** RND encourages broader
  state-space exploration via a novelty bonus; it is not literally
  "automatic entropy tuning" the way SAC is, and it does not directly
  address "the policy found and then actively abandoned a good behavior"
  — it is a genuinely different, related mechanism (encouraging visiting
  novel states) that could plausibly help re-discovery but is not a
  demonstrated fix for the specific abandonment dynamic this project
  diagnosed.
- **Entropy-coefficient scheduling/annealing** (as opposed to this
  project's current fixed `entropy_coef=0.006` for the entire run) is a
  common technique in PPO implementations generally, but no evidence was
  found that `rsl_rl` v3.0.1 supports scheduling `entropy_coef` at all
  (only its `desired_kl`-driven adaptive *learning-rate* schedule is
  built in, itself a real, separately-cited technique — the same paper
  `rsl_rl`'s own README cites for its core algorithm, Rudin et al., "Learning
  to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning,"
  CoRL 2022). Implementing entropy-coefficient scheduling in `rsl_rl`
  would require a local patch/subclass of the installed library's own
  `PPO.update()` — a small, scoped, in-repo code change, not a new
  dependency, but a real code change nonetheless (not evaluated further
  here — that would be a design decision, out of scope for this research
  step).

---

## Direction 2: open-source pretrained policies

### 2a. Octo and OpenVLA — what they actually are, verified directly

- **Octo Model Team et al., "Octo: An Open-Source Generalist Robot
  Policy," RSS 2024 (arXiv:2405.12213)** — verified real via arXiv API.
  A transformer-based policy trained on 800k trajectories from the Open
  X-Embodiment dataset, spanning **9 real robot platforms**. Conditioned on
  **language commands or goal images**, with **RGB image observations**
  (primary + optional wrist camera) as the core input modality;
  proprioception can be added as an auxiliary token but images are the
  model's central pretrained representation. Action output: normalized,
  per-dataset action space (commonly a short-horizon chunk of 7-DoF
  end-effector deltas — position, orientation, gripper — decoded via a
  diffusion action head), explicitly designed to be finetuned to "new
  sensory inputs and action spaces... within a few hours on standard
  consumer GPUs." Available in small (~27M) and base (~93M) parameter
  variants — comparatively lightweight next to VLA-style models.
- **Kim et al., "OpenVLA: An Open-Source Vision-Language-Action Model,"
  2024 (arXiv:2406.09246)** — verified real via arXiv API. A **7B-parameter**
  vision-language-action model (Llama-2 7B backbone + a fused
  DINOv2/SigLIP visual encoder), trained on **970k real-world robot
  demonstrations** from Open X-Embodiment. Input: **RGB image(s) + a
  natural-language task instruction** — no state/proprioception input path
  in the base architecture. Output: discretized 7-DoF robot action tokens,
  autoregressively decoded through the language-model head. Reports
  outperforming the 55B-parameter closed RT-2-X by 16.5% absolute success
  rate across 29 real tasks/multiple embodiments.

### 2b. Mapping onto this project's actual setup — a real, not superficial, mismatch

This project's Franka observation space (`tasks/franka/lift_env_cfg.py`'s
`ObservationsCfg.PolicyCfg`, confirmed by direct source read) is a flat,
concatenated low-dimensional state vector: joint positions/velocities,
object position (from the physics scene directly, not vision-derived, in
this specific env cfg), a commanded goal position, last action, a 4-dim
shape one-hot + 1-dim Wadell-sphericity geometry descriptor, and (in the
target-selection variant) a 2-dim distractor-distance summary — on the
order of 41-43 scalars total, no images anywhere in the policy's own input.
This is not merely "a different data format" from what Octo/OpenVLA expect
— it is the **opposite representational choice**: both pretrained policies'
entire pretraining signal is built around processing raw camera pixels
(Octo's transformer tokenizes image patches; OpenVLA's visual encoder is
the majority of its parameter count and its primary source of
generalization). Feeding this project's own compact state vector into
either model instead would not be "fine-tuning with a new sensor" (the
kind of adaptation Octo's own paper claims support for) — it would mean
discarding the pretrained vision backbone entirely and keeping, at best, a
transformer trunk initialized on an input modality it was never trained to
receive. This is not a proven or even commonly-attempted use of either
model; no evidence was found of anyone doing this successfully.

The other real direction — feeding actual rendered camera images from
Isaac Lab into Octo/OpenVLA instead of this project's compact features —
is architecturally the "intended" use, but is a genuinely different
project decision: it would mean **replacing this project's own
detector-derived (`vision/` subtree) compact-feature pipeline with raw
image conditioning inside the policy itself**, which does not obviously
complement the existing vision pipeline so much as compete with it as a
second, redundant perception path — the `vision/` detector's whole value
proposition (a small, fast, purpose-built ONNX model producing a compact
manifest the RL policy consumes cheaply) is a different design philosophy
than "let a 7B-parameter VLA look at the raw pixels itself."

**Compute cost is also a real, concrete constraint, not a vague concern:**
OpenVLA is 7B parameters — at bf16 that is ~14GB of weights alone, before
optimizer state, LoRA adapters, or activations, on a single RTX 5070 Ti
(16GB) that this project's own training already runs Isaac Sim on
simultaneously. Even LoRA-based fine-tuning (OpenVLA's own paper's
documented lower-cost fine-tuning path) would be tight-to-infeasible
alongside a live Isaac Sim process on this project's actual hardware,
per this project's own environment conventions (`CLAUDE.md`'s "Environment
conventions" section, single-GPU, Isaac Sim + vision jobs already
contending for the one GPU). Octo, at 27M-93M parameters, is far cheaper
and would not have this specific problem — but still carries the
observation-space mismatch above.

### 2c. Real precedent for fine-tuning on a narrow simulated single-task benchmark

- **Zhu et al., "LIBERO: Benchmarking Knowledge Transfer for Lifelong
  Robot Learning," NeurIPS 2023 Datasets & Benchmarks (arXiv:2306.03310)**
  — verified real via arXiv API. A simulated (robosuite/MuJoCo-based)
  benchmark of task suites purpose-built for evaluating knowledge transfer/
  fine-tuning of pretrained manipulation policies — establishes that
  fine-tuning generalist policies on bounded simulated task suites is a
  real, standard evaluation practice in this literature.
- **Kim et al. (a different, later paper), "Fine-Tuning Vision-Language-Action
  Models: Optimizing Speed and Success," 2025 (arXiv:2502.19645,
  "OpenVLA-OFT")** — verified real via arXiv API. Directly fine-tunes
  OpenVLA on the LIBERO simulation benchmark, raising its average success
  rate from 76.5% to 97.1% across four LIBERO task suites, and separately
  demonstrates real-robot fine-tuning (bimanual ALOHA) beating both other
  VLAs and from-scratch imitation-learning baselines (Diffusion Policy,
  ACT) by up to 15% absolute. **This is real, strong, on-point precedent
  that fine-tuning OpenVLA specifically on simulated manipulation tasks
  works well** — but the important caveat for this project: LIBERO's own
  task suites are each a *bundle* of ~10 related tasks (varying objects/
  layouts/language), not a single narrow task, and LIBERO's own simulator
  (robosuite/MuJoCo) is a different physics/rendering stack from Isaac Lab
  — meaning even a successful LIBERO fine-tune would not transfer directly
  to this project's own Isaac Lab renderer/physics without its own
  from-scratch fine-tuning pass here.

**The core tension worth stating plainly:** both real fine-tuning
precedents found (LIBERO broadly, OpenVLA-OFT specifically) demonstrate
the payoff logic these models are designed for — amortizing a large
pretraining investment across *many* tasks/objects/language variations,
where a small amount of task-specific data buys broad generalization. This
project's current phase is explicitly one arm, one object class, one task
(`CLAUDE.md`'s own "Scope discipline" section) — fine-tuning a
multi-task-generalist policy for a single narrow die-lift task pays the
full integration/compute cost of adopting a VLA while capturing almost
none of its actual value proposition (broad task/language generalization).
That value proposition would become real once this project's own
North-Star-intended multi-object/multi-task phase actually begins — not
before.

---

## Recommendation

**Do not pursue either direction as the immediate next step. Pursue the
already-scoped, cheaper, better-evidenced alternative this project's own
same-day root-cause finding already surfaced: test
`RelativeJointPositionActionCfg` (incremental/relative joint actions,
confirmed present in the currently-installed Isaac Lab v2.3.1's
`isaaclab/envs/mdp/actions/actions_cfg.py`) as a new Tier 1 action-space
experiment, still entirely within the existing `rsl_rl`/PPO stack.**

Grounded concretely in what this research found, not a hedge:

1. **This project's own most recent, most rigorous evidence (today's
   root-cause investigation in [[d8-antipodal-grasp-quality]]) already
   identifies the mechanism as action-space credit assignment, not an
   algorithm-level limitation of PPO itself.** Joint-space's
   `reaching_object` reward rises to a real peak (0.60 at iteration 100)
   then decays to 0.10 by the end of training, while task-space's
   identical reward structure rises and *stays* high (0.79→0.84) under
   the exact same PPO/`entropy_coef`/reward setup — the only thing that
   differs is the action space's mapping from action to end-effector
   motion. Switching to task-space IK alone (no algorithm change) already
   took contact frequency from an exact `0.0` (8 checkpoints, 4 runs) to
   88% by the end of training. The literature this same investigation
   found (Martín-Martín et al. IROS 2019, Varin et al. IROS 2019 — both
   already verified real) independently corroborates that action space,
   not algorithm, is the load-bearing variable for exactly this failure
   shape.
2. **SAC would address a plausible but now less load-bearing hypothesis
   than a cheaper, already-identified fix, on a target this project has
   already demonstrated a cheaper fix works for.** SAC integration is a
   real, multi-day-plus effort (new dependency, new agent config with zero
   in-repo tuning precedent, off-policy bookkeeping never exercised in
   this pipeline) to test a mechanism (auto-entropy, replay reuse) whose
   own best real precedent (Varin et al. 2019) argues action space matters
   more anyway. The `RelativeJointPositionActionCfg` test is a same-stack,
   same-infra, same-day-actionable experiment that isolates the one
   variable (relative vs. absolute joint action semantics) this project's
   own H_joint/H_taskspace design conflated, at a fraction of the
   integration cost.
3. **RND (§1a/§1d) is a legitimate, essentially-free complementary
   thing worth trying separately if exploration/rediscovery remains a
   concern after the action-space question is settled** — real,
   peer-reviewed manipulation-relevant citation, already present in the
   exact installed library version, a single config field to enable. Not
   proposed as a substitute for the action-space test, but a much lower-risk
   fallback than SAC if the action-space fix alone proves insufficient.
4. **Pretrained policies (Octo/OpenVLA) are not a good fit right now** —
   not because they're not real or not impressive (both are legitimate,
   verified, state-of-the-art work), but because this project's current
   single-task, single-object, state-based-observation phase is
   structurally the wrong point to adopt them: the integration cost
   (observation-pipeline restructuring from compact-state to raw-image
   conditioning, a 7B-parameter model that barely fits this project's
   single GPU alongside Isaac Sim for OpenVLA specifically) is real and
   substantial, while the actual payoff these models are built to deliver
   — broad multi-task/multi-object/language generalization — doesn't
   apply yet to a single narrow task. **Revisit this when the project
   actually reaches its own intended multi-object/multi-task phase
   (`CLAUDE.md`'s "North Star"/"Scope discipline" sections) — that is
   the point at which a pretrained generalist policy's value proposition
   would actually be realized, not before.**

---

## Related

[[d8-antipodal-grasp-quality]] (the same-day root-cause finding this
research was directly requested in response to — its own "Root cause
investigation" section already names `RelativeJointPositionActionCfg` as
the next candidate Tier 1 experiment, which this document's recommendation
endorses over both surveyed alternatives),
`docs/superpowers/specs/research/2026-07-20-d8-grasp-mechanics-literature.md`
(source of the Varin et al. 2019 citation, independently re-verified here),
`kb/wiki/concepts/citation-verification-practice.md` (the standing
citation-verification practice this document follows — every citation
above was checked for real existence via the arXiv API, or via the
publisher's own proceedings page for the one non-arXiv citation),
`CLAUDE.md` (North Star / Scope discipline sections, the basis for this
document's Direction 2 "revisit later, not now" recommendation).
